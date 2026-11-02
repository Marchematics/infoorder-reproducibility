"""Vary reader-memory caps and direct-input allowances over five rounds."""
import json, os, re, statistics, sys, time
BASE = str(__import__("pathlib").Path(__file__).resolve().parents[2])
for p in (BASE + "/deps", BASE + "/deps_fla", BASE + "/sota/harness"):
    if p not in sys.path:
        sys.path.append(p)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import scoring

MODEL = BASE + "/models/Qwen3.5-4B"
CHAIN_AD = BASE + "/sota/infoorder/chain_writer_qwen_adapter"
UPD_AD = BASE + "/sota/infoorder/updater_adapter"
APP_AD = BASE + "/sota/infoorder/append_updater_adapter"
MAN = BASE + "/sota/data/manifest_recursive_100.jsonl"
B = int(os.environ.get("M4_B", "128"))
OUT = os.environ.get("M4_OUT", BASE + "/sota/results/m4_cap_%d.jsonl" % B)
LIM = int(os.environ.get("M4_LIMIT", "40"))
T = int(os.environ.get("M4_T", "5"))
CHAIN_SYS = ("You write compact memory for a question. Read the evidence and write the fact chain, "
             "as numbered hops. Use only entities and values present in the evidence.")
UPD_SYS = ("You update a compact memory when new evidence arrives for a fixed question. Merge the "
           "previous memory with the new evidence into a new compact memory of numbered hops.")
APP_SYS = ("You maintain an append-only fact chain for a fixed question. Copy the previous memory "
           "verbatim, then append ONLY the newly supported hops as 'N) sub-question => answer'.")
READER = ("You answer questions using ONLY the context provided. Reply with the shortest possible "
          "answer phrase. If the context does not contain enough information, reply with exactly UNANSWERABLE.")
STOP = {"Which", "Who", "What", "When", "Where", "How", "Many", "The", "In", "Of", "And", "A", "An"}


def count(tok, s):
    return len(tok(s, add_special_tokens=False)["input_ids"])


def cap(tok, s, b):
    ids = tok(s, add_special_tokens=False)["input_ids"]
    return tok.decode(ids[:b], skip_special_tokens=True) if len(ids) > b else s


def render(paras):
    return "\n\n".join("Passage %d (%s): %s" % (p["idx"], p["title"], p["paragraph_text"]) for p in paras)


def gen(tok, model, sys_p, user, maxnew=160):
    text = tok.apply_chat_template([{"role": "system", "content": sys_p}, {"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc = tok(text, return_tensors="pt", truncation=True, max_length=3072).to("cuda")
    with torch.no_grad():
        o = model.generate(**enc, max_new_tokens=maxnew, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(o[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def entity_of(q):
    c = re.findall(r"[A-Z][a-zA-Z\-]+(?:\s+[A-Z][a-zA-Z\-]+)*", q or "")
    c = [x for x in c if x.split()[0] not in STOP]
    return max(c, key=len) if c else None


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map={"": 0})
    writer = PeftModel.from_pretrained(base, CHAIN_AD, adapter_name="chain")
    for n, p in (("upd", UPD_AD), ("app", APP_AD)):
        if os.path.isdir(p):
            writer.load_adapter(p, adapter_name=n)
    writer.eval()
    dec = {json.loads(l)["id"]: json.loads(l) for l in open(BASE + "/sota/data/manifest_500.jsonl")}
    rows = [json.loads(l) for l in open(MAN)][:LIM]
    out = []
    t0 = time.time()
    for idx, r in enumerate(rows):
        src = dec.get(r["id"], {})
        hops = [s for s in (src.get("question_decomposition") or []) if str(s.get("answer", "")).strip()]
        aliases = src.get("answer_aliases")
        paras = sorted(r["paragraphs"], key=lambda p: p["idx"])
        size = max(1, (len(paras) + T - 1) // T)
        chunks = [paras[i:i + size] for i in range(0, len(paras), size)][:T]
        seen, prev = [], {"chain": "", "append": ""}
        for t in range(1, len(chunks) + 1):
            seen.extend(chunks[t - 1])
            prefix = render(seen)
            direct_in = cap(tok, prefix, B + count(tok, render(chunks[t - 1])))
            recs = {"direct_cap": cap(tok, gen(tok, writer, CHAIN_SYS,
                                              "Question: %s\n\nEvidence:\n%s\n\nMemory:" % (r["question"], direct_in)), B)}
            writer.set_adapter("upd")
            m = gen(tok, writer, UPD_SYS, "Question: %s\n\nPrevious memory:\n%s\n\nNew evidence:\n%s\n\nUpdated memory:"
                    % (r["question"], prev["chain"] or "(empty)", render(chunks[t - 1])))
            writer.set_adapter("chain"); prev["chain"] = m; recs["rec_chain"] = cap(tok, m, B)
            if "app" in getattr(writer, "peft_config", {}):
                writer.set_adapter("app")
                m = gen(tok, writer, APP_SYS, "Question: %s\n\nPrevious memory:\n%s\n\nNew evidence:\n%s\n\nUpdated memory:"
                        % (r["question"], prev["append"] or "(empty)", render(chunks[t - 1])))
                writer.set_adapter("chain"); prev["append"] = m; recs["rec_append"] = cap(tok, m, B)
            for cond, mem in recs.items():
                with writer.disable_adapter():
                    pred = gen(tok, writer, READER, "Context:\n%s\n\nQuestion: %s\n\nAnswer:" % (mem, r["question"]), 24)
                sc = scoring.score_answer(pred, [r.get("answer")])
                hop = [float(bool(scoring.answer_in_context(h.get("answer"), mem, aliases))) for h in hops]
                bind = []
                for h, ok in zip(hops, hop):
                    ent = entity_of(h.get("question"))
                    if ent:
                        bind.append(float(ok and (ent.lower() in mem.lower())))
                out.append({"id": r["id"], "t": t, "cond": cond, "B": B,
                            "em": float(sc.get("em", 0)),
                            "hop_cov": statistics.mean(hop) if hop else 0.0,
                            "bind_cov": statistics.mean(bind) if bind else 0.0,
                            "tokens": count(tok, mem)})
        if (idx + 1) % 10 == 0:
            print("[m4] %d/%d elapsed %.0fs" % (idx + 1, len(rows), time.time() - t0), flush=True)
    with open(OUT, "w") as f:
        for x in out:
            f.write(json.dumps(x) + "\n")
    print("| B | t | cond | EM | hop-cov | bind-cov | tokens |")
    print("|---|---|---|---|---|---|---|")
    for t in (1, 3, 5):
        for cond in ("direct_cap", "rec_chain", "rec_append"):
            v = [x for x in out if x["t"] == t and x["cond"] == cond]
            if v:
                print("| %d | %d | %s | %.3f | %.3f | %.3f | %.1f |" % (
                    B, t, cond, statistics.mean(x["em"] for x in v),
                    statistics.mean(x["hop_cov"] for x in v), statistics.mean(x["bind_cov"] for x in v),
                    statistics.mean(x["tokens"] for x in v)))


if __name__ == "__main__":
    main()
