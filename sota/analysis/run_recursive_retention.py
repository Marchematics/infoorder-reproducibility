"""Measure gold-hop and entity-answer co-coverage without reader calls."""
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
OUT = os.environ.get("RT_OUT", BASE + "/sota/results/recursive_retention.jsonl")
MEMOUT = os.environ.get("RT_MEM", BASE + "/sota/results/recursive_retention_memories.jsonl")
LIM = int(os.environ.get("RT_LIMIT", "40"))
T = int(os.environ.get("RT_T", "5"))
B = int(os.environ.get("RT_BUDGET", "128"))
CHAIN_SYS = ("You write compact memory for a question. Read the evidence and write the fact chain, "
             "as numbered hops. Use only entities and values present in the evidence.")
UPD_SYS = ("You update a compact memory when new evidence arrives for a fixed question. Merge the "
           "previous memory with the new evidence into a new compact memory of numbered hops.")
APP_SYS = ("You maintain an append-only fact chain for a fixed question. Copy the previous memory "
           "verbatim, then append ONLY the newly supported hops as 'N) sub-question => answer'.")
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


def entity_of(question):
    cands = re.findall(r"[A-Z][a-zA-Z\-]+(?:\s+[A-Z][a-zA-Z\-]+)*", question or "")
    cands = [c for c in cands if c.split()[0] not in STOP]
    return max(cands, key=len) if cands else None


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map={"": 0})
    writer = PeftModel.from_pretrained(base, CHAIN_AD, adapter_name="chain")
    for n, p in (("upd", UPD_AD), ("app", APP_AD)):
        if os.path.isdir(p):
            writer.load_adapter(p, adapter_name=n)
    writer.eval()
    rows = [json.loads(l) for l in open(MAN)][:LIM]
                                                                                              
    dec = {}
    for line in open(BASE + "/sota/data/manifest_500.jsonl"):
        d = json.loads(line)
        dec[d["id"]] = d
    print("[rt] decomp lookup", len(dec), flush=True)
    out, mems = [], []
    t0 = time.time()
    for idx, r in enumerate(rows):
        src = dec.get(r["id"], {})
        hops = [s for s in (src.get("question_decomposition") or []) if str(s.get("answer", "")).strip()]
        aliases = src.get("answer_aliases")
        if not hops:
            print("[rt] no decomposition for", r["id"], flush=True)
        paras = sorted(r["paragraphs"], key=lambda p: p["idx"])
        size = max(1, (len(paras) + T - 1) // T)
        chunks = [paras[i:i + size] for i in range(0, len(paras), size)][:T]
        seen, prev = [], {"chain": "", "append": ""}
        for t in range(1, len(chunks) + 1):
            seen.extend(chunks[t - 1])
            prefix = render(seen)
            chunk_tokens = count(tok, render(chunks[t - 1]))
            direct_in = cap(tok, prefix, B + chunk_tokens)
            recs = {}
            recs["direct_cap"] = cap(tok, gen(tok, writer, CHAIN_SYS,
                                              "Question: %s\n\nEvidence:\n%s\n\nMemory:" % (r["question"], direct_in)), B)
            writer.set_adapter("upd")
            mem = gen(tok, writer, UPD_SYS, "Question: %s\n\nPrevious memory:\n%s\n\nNew evidence:\n%s\n\nUpdated memory:"
                      % (r["question"], prev["chain"] or "(empty)", render(chunks[t - 1])))
            writer.set_adapter("chain")
            prev["chain"] = mem
            recs["rec_chain"] = cap(tok, mem, B)
            if "app" in getattr(writer, "peft_config", {}):
                writer.set_adapter("app")
                mem = gen(tok, writer, APP_SYS, "Question: %s\n\nPrevious memory:\n%s\n\nNew evidence:\n%s\n\nUpdated memory:"
                          % (r["question"], prev["append"] or "(empty)", render(chunks[t - 1])))
                writer.set_adapter("chain")
                prev["append"] = mem
                recs["rec_append"] = cap(tok, mem, B)
            for cond, m in recs.items():
                hop_ok, bind_ok = [], []
                for h in hops:
                    ok = bool(scoring.answer_in_context(h.get("answer"), m, aliases))
                    hop_ok.append(float(ok))
                    ent = entity_of(h.get("question"))
                    if ent:
                        bind_ok.append(float(ok and (ent.lower() in m.lower())))
                final_ok = float(bool(scoring.answer_in_context(r.get("answer"), m, aliases)))
                out.append({"id": r["id"], "t": t, "cond": cond, "tokens": count(tok, m),
                            "final_aic": final_ok,
                            "hop_coverage": statistics.mean(hop_ok) if hop_ok else 0.0,
                            "binding_coverage": statistics.mean(bind_ok) if bind_ok else 0.0,
                            "n_hops": len(hop_ok)})
                mems.append({"id": r["id"], "t": t, "cond": cond, "memory": m})
        if (idx + 1) % 10 == 0:
            print("[rt] %d/%d elapsed %.0fs" % (idx + 1, len(rows), time.time() - t0), flush=True)
    with open(OUT, "w") as f:
        for x in out:
            f.write(json.dumps(x) + "\n")
    with open(MEMOUT, "w") as f:
        for x in mems:
            f.write(json.dumps(x, ensure_ascii=False) + "\n")
    print("| t | cond | hop-cov | binding-cov | final AIC | tokens |")
    print("|---|---|---|---|---|---|")
    for t in range(1, T + 1):
        for cond in ("direct_cap", "rec_chain", "rec_append"):
            v = [x for x in out if x["t"] == t and x["cond"] == cond]
            if not v:
                continue
            print("| %d | %s | %.3f | %.3f | %.3f | %.1f |" % (
                t, cond, statistics.mean(x["hop_coverage"] for x in v),
                statistics.mean(x["binding_coverage"] for x in v),
                statistics.mean(x["final_aic"] for x in v),
                statistics.mean(x["tokens"] for x in v)))


if __name__ == "__main__":
    main()
