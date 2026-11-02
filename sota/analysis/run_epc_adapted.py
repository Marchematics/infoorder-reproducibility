"""Generate anticipated queries and pack their evidence in two writer calls."""
import json, os, re, statistics, sys, time
BASE = str(__import__("pathlib").Path(__file__).resolve().parents[2])
for p in (BASE + "/deps", BASE + "/deps_fla", BASE + "/sota/harness"):
    if p not in sys.path:
        sys.path.append(p)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import scoring

MODEL = os.environ.get("SOTA_WRITER_MODEL", BASE + "/models/Qwen3.5-4B")
ADAPTER = os.environ.get("SOTA_ADAPTER", BASE + "/sota/infoorder/chain_writer_qwen_adapter")
MAN = os.environ.get("SOTA_MANIFEST", BASE + "/sota/data/manifest_recursive_100.jsonl")
OUT = os.environ.get("SOTA_OUT", BASE + "/sota/results/epc_adapted_100.jsonl")
BUDGET = int(os.environ.get("SOTA_BUDGET", "128"))
K = int(os.environ.get("SOTA_K", "5"))
PROBE_SYS = ("You list likely future questions about the evidence. Output exactly %d short questions, "
             "one per line, no numbering, no commentary." % K)
PACK_SYS = ("You compress evidence for future questions. For each listed question, copy the minimal set of "
            "evidence sentences needed to answer it, verbatim. Output one line per question in the form "
            "'Q: <question> | E: <sentence>'. No commentary.")
READER_SYS = ("You answer questions using ONLY the context provided. Reply with the shortest possible "
              "answer phrase: a few words at most, no explanation, no full sentence, no quotation marks, "
              "no citations. If the context does not contain enough information to determine the answer, "
              "reply with exactly UNANSWERABLE.")


def render(paras):
    return "\n\n".join("Passage %d (%s): %s" % (p["idx"], p["title"], p["paragraph_text"]) for p in paras)


def count(tok, s):
    return len(tok(s, add_special_tokens=False)["input_ids"])


def call(tok, model, sys_p, user, maxnew):
    text = tok.apply_chat_template([{"role": "system", "content": sys_p}, {"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc = tok(text, return_tensors="pt", truncation=True, max_length=3072).to("cuda")
    with torch.no_grad():
        o = model.generate(**enc, max_new_tokens=maxnew, do_sample=False, pad_token_id=tok.eos_token_id)
    out = tok.decode(o[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    return out, int(enc["input_ids"].shape[1]), len(tok(out, add_special_tokens=False)["input_ids"])


def sentences(paras):
    out = []
    for p in paras:
        for s in re.split(r"(?<=[.!?])\s+", p["paragraph_text"]):
            s = s.strip()
            if len(s) >= 30:
                out.append((p["title"], s))
    return out


def pack(tok, probes, quoted, sents):
    """Merge/dedupe/rank small supporting pieces, pack to the token budget."""
    pool = []
    for q, e in zip(probes, quoted):
        pool.append((q, e))
    if not pool:
        pool = [(probes[0] if probes else "", s) for _t, s in sents[:6]]

    def spec(s):
        return float(bool(re.search(r"\d", s))) + 0.5 * float(len(re.findall(r"\b[A-Z][a-z]+", s)) >= 2)
    scored = []
    for q, e in pool:
        s = e.strip().strip('"')
        if not s:
            continue
        cov = sum(1 for _, e2 in pool if s[:60] in e2 or e2[:60] in s)
        scored.append((cov + spec(s) - 0.2 * len(scored) / max(1, len(pool)), s))
    scored.sort(key=lambda x: -x[0])
    mem, used = [], 0
    for _, s in scored:
        n = count(tok, s)
        if used + n > BUDGET:
            continue
        mem.append(s)
        used += n
        if used >= BUDGET * 0.95:
            break
    return " ".join(mem)


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map={"": 0})
    writer = PeftModel.from_pretrained(base, ADAPTER).eval()
    reader = writer
    rows = [json.loads(l) for l in open(MAN)]
    LIMIT = int(os.environ.get("SOTA_LIMIT", "0"))
    if LIMIT > 0:
        rows = rows[:LIMIT]
    dev = {}
    for line in open(BASE + "/sota/data/data/musique_ans_v1.0_dev.jsonl"):
        r = json.loads(line)
        dev[r["id"]] = r
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            done.add(json.loads(l)["id"])
    t0 = time.time()
    with open(OUT, "a") as f:
        for i, r in enumerate(rows):
            if r["id"] in done:
                continue
            ev = render(r["paragraphs"])

            probes_txt, ptok1, ctok1 = call(tok, writer, PROBE_SYS,
                                            "Question: %s\n\nEvidence:\n%s\n\nQuestions:" % (r["question"], ev), 160)
            probes = [x.strip(" -0123456789.") for x in probes_txt.splitlines() if len(x.strip()) > 8][:K]

            pack_user = "Evidence:\n%s\n\nQuestions:\n%s\n\nCompressed evidence:" % (ev, "\n".join(probes))
            packed_txt, ptok2, ctok2 = call(tok, writer, PACK_SYS, pack_user, 400)
            quoted = []
            for line in packed_txt.splitlines():
                if "|" in line:
                    quoted.append(line.split("|", 1)[1].strip())
            mem = pack(tok, probes, quoted, sentences(r["paragraphs"]))
            with writer.disable_adapter():
                pred, _, _ = call(tok, reader, READER_SYS,
                                  "Context:\n%s\n\nQuestion: %s\n\nAnswer:" % (mem, r["question"]), 24)
            golds = [r["answer"]] + list((dev.get(r["id"], {}) or {}).get("answer_aliases") or [])
            res = scoring.score_answer(pred, golds)
            if isinstance(res, dict):
                em, f1 = float(res.get("em", 0)), float(res.get("f1", 0.0))
            elif isinstance(res, tuple):
                em, f1 = float(res[0]), float(res[1]) if len(res) > 1 else 0.0
            else:
                em, f1 = float(res), 0.0
            f.write(json.dumps({"id": r["id"], "hop": r["hop"], "memory": mem, "tokens": count(tok, mem),
                                "probes": probes, "calls": 2, "prompt_tokens": ptok1 + ptok2,
                                "completion_tokens": ctok1 + ctok2, "pred": pred, "em": em, "f1": f1,
                                "aic": float(scoring.answer_in_context(r["answer"], mem, (dev.get(r["id"], {}) or {}).get("answer_aliases") or []))}, ensure_ascii=False) + "\n")
            f.flush()
            if i % 10 == 0:
                print("[epc] %d/%d elapsed %.0fs" % (i + 1, len(rows), time.time() - t0), flush=True)
    recs = [json.loads(l) for l in open(OUT)]
    print("EPC-adapted n %d EM %.3f F1 %.3f AIC %.3f mem_tok %.1f write_tok %.0f calls %.1f" % (
        len(recs), statistics.mean(r["em"] for r in recs), statistics.mean(r["f1"] for r in recs),
        statistics.mean(r["aic"] for r in recs), statistics.mean(r["tokens"] for r in recs),
        statistics.mean(r["prompt_tokens"] + r["completion_tokens"] for r in recs),
        statistics.mean(r["calls"] for r in recs)))


if __name__ == "__main__":
    main()
