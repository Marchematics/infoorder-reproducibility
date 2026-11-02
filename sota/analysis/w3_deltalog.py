"""Score gold answers with a fixed reader using arrived evidence and saved memory."""
import json, os, statistics, sys
BASE = str(__import__("pathlib").Path(__file__).resolve().parents[2])
for p in (BASE + "/deps", BASE + "/deps_fla", BASE + "/sota/harness"):
    if p not in sys.path:
        sys.path.append(p)
import torch, math
from transformers import AutoModelForCausalLM, AutoTokenizer
import scoring

MODEL = BASE + "/models/Qwen3.5-4B"
MEM = BASE + "/sota/results/recursive_retention_memories.jsonl"
MAN = BASE + "/sota/data/manifest_recursive_100.jsonl"
DEC = BASE + "/sota/data/manifest_500.jsonl"
OUT = BASE + "/sota/results/w3_deltalog.jsonl"
T = int(os.environ.get("W3_T", "5"))
LIM = int(os.environ.get("W3_LIMIT", "40"))
READER = "You answer the question using ONLY the context provided. Answer:"


def render(paras):
    return "\n\n".join("Passage %d (%s): %s" % (p["idx"], p["title"], p["paragraph_text"]) for p in paras)


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map={"": 0}).eval()
    dec = {json.loads(l)["id"]: json.loads(l) for l in open(DEC)}
    rows = {json.loads(l)["id"]: json.loads(l) for l in open(MAN)}
    mems = [json.loads(l) for l in open(MEM)]

    prefixes = {}
    for qid, r in rows.items():
        paras = sorted(r["paragraphs"], key=lambda p: p["idx"])
        size = max(1, (len(paras) + T - 1) // T)
        chunks = [paras[i:i + size] for i in range(0, len(paras), size)][:T]
        seen = []
        for t in range(1, len(chunks) + 1):
            seen = paras[: min(len(paras), t * size)]
            prefixes[(qid, t)] = render(seen)

    def logprob(ctx, q, ans):
        text = tok.apply_chat_template([{"role": "system", "content": READER},
                                        {"role": "user", "content": "Context:\n%s\n\nQuestion: %s" % (ctx, q)}],
                                       tokenize=False, add_generation_prompt=True, enable_thinking=False)
        ids = tok(text, add_special_tokens=False)["input_ids"][-3000:]
        a_ids = tok(" " + ans, add_special_tokens=False)["input_ids"]
        x = torch.tensor([ids + a_ids], device="cuda")
        with torch.no_grad():
            out = model(input_ids=x)
        logits = out.logits[0, len(ids) - 1: len(ids) - 1 + len(a_ids)]
        lp = torch.log_softmax(logits.float(), dim=-1)
        return float(sum(lp[i, a_ids[i]].item() for i in range(len(a_ids))) / max(1, len(a_ids)))

    out = []
    n = 0
    for m in mems:
        qid, t, cond = m["id"], m["t"], m["cond"]
        if t > T or (qid, t) not in prefixes:
            continue
        q = rows[qid]["question"]; ans = str(rows[qid].get("answer", "")).strip()
        if not ans:
            continue
        prefix = prefixes[(qid, t)]
        lp_full = logprob(prefix, q, ans)
        lp_mem = logprob(m["memory"], q, ans)
        hops = [s for s in (dec.get(qid, {}).get("question_decomposition") or []) if str(s.get("answer", "")).strip()]
        cov = [float(bool(scoring.answer_in_context(h.get("answer"), m["memory"], None))) for h in hops] if hops else [0.0]
        out.append({"id": qid, "t": t, "cond": cond, "dlog": lp_full - lp_mem,
                    "lp_full": lp_full, "lp_mem": lp_mem, "hop_cov": statistics.mean(cov),
                    "tokens": len(tok(m["memory"], add_special_tokens=False)["input_ids"])})
        n += 1
        if n % 100 == 0:
            print("[w3] %d rows" % n, flush=True)
    with open(OUT, "w") as f:
        for x in out:
            f.write(json.dumps(x) + "\n")
    xs = [x["hop_cov"] for x in out]; ys = [x["dlog"] for x in out]
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((a - mx) * (b - my) for a, b in zip(xs, ys))
    den = math.sqrt(sum((a - mx) ** 2 for a in xs) * sum((b - my) ** 2 for b in ys)) or 1.0
    print("rows", len(out), "corr(hop_cov, dlog) = %.3f" % (num / den))
    for t in (1, 3, 5):
        for c in ("direct_cap", "rec_append"):
            v = [x for x in out if x["t"] == t and x["cond"] == c]
            if v:
                print("| t=%d | %s | dlog %.3f | hop_cov %.3f | tok %.1f |" % (
                    t, c, statistics.mean(x["dlog"] for x in v),
                    statistics.mean(x["hop_cov"] for x in v), statistics.mean(x["tokens"] for x in v)))


if __name__ == "__main__":
    main()
