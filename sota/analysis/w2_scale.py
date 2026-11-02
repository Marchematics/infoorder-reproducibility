"""Measure future-query answer coverage in a fixed question-free memory."""
import json, os, statistics, sys, time
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
NOQ_AD = BASE + "/sota/infoorder/noq_writer_adapter"
MAN = BASE + "/sota/data/manifest_500.jsonl"
OUT = os.environ.get("W2_OUT", BASE + "/sota/results/w2_scale.jsonl")
P = int(os.environ.get("W2_PREFIX", "16"))
B = int(os.environ.get("W2_BUDGET", "128"))
N = int(os.environ.get("W2_N", "500"))
NOQ2 = ("You write a compact memory from evidence alone, for a task whose question is not yet known. "
        "Output numbered factual hops and then a table of entity=value bindings.")
CHAIN = ("Write a compact fact chain from the evidence: numbered hops 'N) entity relation value'. "
         "Use only entities and values present in the evidence.")
SUMMARY = ("Compress the evidence into a compact plain-text memory. Keep concrete facts; copy values verbatim.")


def count(tok, s):
    return len(tok(s, add_special_tokens=False)["input_ids"])


def cap(tok, s, b):
    ids = tok(s, add_special_tokens=False)["input_ids"]
    return tok.decode(ids[:b], skip_special_tokens=True) if len(ids) > b else s


def gen(tok, model, sys_p, user, maxnew=160):
    text = tok.apply_chat_template([{"role": "system", "content": sys_p}, {"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc = tok(text, return_tensors="pt", truncation=True, max_length=3072).to("cuda")
    with torch.no_grad():
        o = model.generate(**enc, max_new_tokens=maxnew, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(o[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map={"": 0})
    writer = PeftModel.from_pretrained(base, CHAIN_AD, adapter_name="chain")
    if os.path.isdir(NOQ_AD):
        writer.load_adapter(NOQ_AD, adapter_name="noq")
    writer.eval()
    rows = [json.loads(l) for l in open(MAN)][:N]
    out = []
    t0 = time.time()
    for i, r in enumerate(rows):
        prefix = [p for p in sorted(r["paragraphs"], key=lambda x: x["idx"]) if p["idx"] < P]
        idxs = {p["idx"] for p in prefix}
        sels = [s for s in (r.get("question_decomposition") or [])
                if isinstance(s.get("paragraph_support_idx"), int) and s["paragraph_support_idx"] in idxs
                and str(s.get("answer", "")).strip()]
        if len(sels) < 2:
            continue
        ev = "\n\n".join("Passage %d (%s): %s" % (p["idx"], p["title"], p["paragraph_text"]) for p in prefix)
        conds = {}
        writer.set_adapter("chain")
        conds["chain_noQ"] = cap(tok, gen(tok, writer, CHAIN, "Evidence:\n%s\n\nMemory:" % ev), B)
        if "noq" in getattr(writer, "peft_config", {}):
            writer.set_adapter("noq")
            conds["chain_noQ_trained"] = cap(tok, gen(tok, writer, NOQ2, "Evidence:\n%s\n\nMemory:" % ev), B)
            writer.set_adapter("chain")
        conds["full_prefix_uncapped"] = ev
        rec = {"id": r["id"], "n_selectors": len(sels)}
        for c, mem in conds.items():
            ok = [float(bool(scoring.answer_in_context(s.get("answer"), mem, s.get("answer_aliases")))) for s in sels]
            rec[c + "_cov"] = statistics.mean(ok)
            rec[c + "_all"] = float(all(x > 0 for x in ok))
            rec[c + "_tokens"] = count(tok, mem)
        out.append(rec)
        if (i + 1) % 50 == 0:
            print("[w2] %d/%d kept=%d elapsed %.0fs" % (i + 1, len(rows), len(out), time.time() - t0), flush=True)
    with open(OUT, "w") as f:
        for x in out:
            f.write(json.dumps(x) + "\n")
    print("histories", len(out), "mean selectors %.2f" % statistics.mean(x["n_selectors"] for x in out))
    print("| condition | selector coverage | all-selectors-correct | memory tokens |")
    print("|---|---|---|---|")
    for c in ("full_prefix_uncapped", "chain_noQ", "chain_noQ_trained"):
        if c + "_cov" in out[0]:
            print("| %s | %.3f | %.3f | %.1f |" % (c,
                  statistics.mean(x[c + "_cov"] for x in out),
                  statistics.mean(x[c + "_all"] for x in out),
                  statistics.mean(x[c + "_tokens"] for x in out)))


if __name__ == "__main__":
    main()
