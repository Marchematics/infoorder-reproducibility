import json, os, re, sys, collections
BASE = str(__import__("pathlib").Path(__file__).resolve().parents[2])
for p in (BASE + "/deps", BASE + "/deps_fla", BASE + "/sota/harness"):
    if p not in sys.path:
        sys.path.append(p)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import scoring

MODEL = os.environ.get("SOTA_WRITER_MODEL", BASE + "/models/Qwen3.5-4B")
ADAPTERS = {
    "chain": BASE + "/sota/infoorder/chain_writer_qwen_adapter",
    "summary": BASE + "/sota/infoorder/summary_writer_qwen_adapter",
    "facts": BASE + "/sota/infoorder/facts_format_adapter",
}
MAN = os.environ.get("CF_MANIFEST", BASE + "/sota/data/controlled_bothcorrect_200.jsonl")
OUT = os.environ.get("CF_OUT", BASE + "/sota/results/both_correct_controlled.jsonl")
BUDGET = int(os.environ.get("SOTA_BUDGET", "128"))
CHAIN_SYS = ("You write compact memory for a question. Read the evidence and write the fact chain, "
             "as numbered hops. Use only entities and values present in the evidence.")
SUMMARY_SYS = ("You compress evidence into a compact plain-text memory that another model will use to "
               "answer a question. Keep only relevant facts; copy values verbatim; output plain text.")
FACT_SYS = ("You write compact memory. Output the answer facts as a short list of values separated by periods.")
TABLE_SYS = ("List every entity mentioned in the evidence together with its stored value. "
             "Output one line per entity in the form 'name = value'. No commentary.")
UPD_SYS = ("You update a compact memory when new evidence arrives for a fixed question. Merge the previous "
           "memory with the new evidence. Keep only evidence-supported values.")
READER_SYS = ("You answer questions using ONLY the context provided. Reply with a single token: 0 or 1. "
              "If the context is insufficient, reply UNANSWERABLE.")


def bit_of(text):
    """Extract a 0/1 answer from a free-form reply (first standalone 0 or 1)."""
    if text is None:
        return None
    t = text.strip()
    m = re.search(r"\b([01])\b", t)
    if m:
        return m.group(1)
    m = re.search(r"([01])", t)
    return m.group(1) if m else None


def count(tok, s):
    return len(tok(s, add_special_tokens=False)["input_ids"])


def cap(tok, s):
    ids = tok(s, add_special_tokens=False)["input_ids"]
    return tok.decode(ids[:BUDGET], skip_special_tokens=True) if len(ids) > BUDGET else s


def gen(tok, model, sys_p, user, maxnew=96):
    text = tok.apply_chat_template([{"role": "system", "content": sys_p}, {"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc = tok(text, return_tensors="pt", truncation=True, max_length=3072).to("cuda")
    with torch.no_grad():
        o = model.generate(**enc, max_new_tokens=maxnew, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(o[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map={"": 0})
    writer = PeftModel.from_pretrained(base, ADAPTERS["chain"], adapter_name="chain")
    for name in ("summary", "facts"):
        if os.path.isdir(ADAPTERS[name]):
            writer.load_adapter(ADAPTERS[name], adapter_name=name)
    writer.eval()
    reader = writer                                                 
    pairs = [json.loads(l) for l in open(MAN)]
    per_fam = int(os.environ.get("CF_PER_FAMILY", "0"))
    if per_fam > 0:
        seen, capped = {}, []
        for it in pairs:
            f = it["family"]
            if seen.get(f, 0) < per_fam:
                capped.append(it)
                seen[f] = seen.get(f, 0) + 1
        pairs = capped
    print("[bc] pairs", len(pairs), flush=True)
    records = []
    for it in pairs:
        rec = {"id": it["id"], "family": it["family"]}
        for policy in ("full_history", "trunc_128", "chain", "summary", "facts", "chain_table"):
            preds, oks = {}, {}
            for br, hkey, akey in (("A", "history", "answer_A"), ("B", "paired_history", "answer_B")):
                h = it[hkey]
                if policy == "full_history":
                    mem = h + "\n" + it["continuation"]
                elif policy == "trunc_128":
                    mem = cap(tok, h + "\n" + it["continuation"])
                else:
                    sys_p = {"chain": CHAIN_SYS, "summary": SUMMARY_SYS, "facts": FACT_SYS,
                             "chain_table": CHAIN_SYS}[policy]
                    mem = cap(tok, gen(tok, writer, sys_p,
                                       "Question: %s\n\nEvidence:\n%s\n\nMemory:" % (it["question"], h)))
                    if policy == "chain_table":
                        tbl = gen(tok, writer, TABLE_SYS, "Evidence:\n%s\n\nTable:" % h, 128)
                        mem = cap(tok, mem + "\n" + tbl.strip())
                        policy_for_adapter = "chain"
                    else:
                        policy_for_adapter = policy
                    writer.set_adapter(policy_for_adapter)
                    mem = cap(tok, gen(tok, writer, UPD_SYS,
                                       "Question: %s\n\nPrevious memory:\n%s\n\nNew evidence:\n%s\n\nUpdated memory:"
                                       % (it["question"], mem, it["continuation"])))
                    writer.set_adapter("chain")
                with writer.disable_adapter():
                    pred = gen(tok, reader, READER_SYS,
                               "Context:\n%s\n\nQuestion: %s\n\nAnswer:" % (mem, it["question"]), 8)
                preds[br] = pred
                gold = str(it[akey])
                pb = bit_of(pred)
                oks[br] = float(pb is not None and pb == str(gold).strip())
                if os.environ.get("CF_DEBUG") and len(records) < 2:
                    print("[dbg]", rec["id"], policy, br, "pred=", repr(pred[:60]), "gold=", gold, "ok=", oks[br], flush=True)
            rec[policy + "_both_correct"] = float(oks["A"] and oks["B"])
            ba, bb = bit_of(preds["A"]), bit_of(preds["B"])
            rec[policy + "_separated"] = float(ba is not None and bb is not None and ba != bb)
        records.append(rec)
        if len(records) % 10 == 0:
            print("[bc] %d/%d" % (len(records), len(pairs)), flush=True)
    with open(OUT, "w") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")
    print("| policy | both-correct | branch-separation |")
    print("|---|---|---|")
    for policy in ("full_history", "trunc_128", "chain", "summary", "facts", "chain_table"):
        bc = sum(r[policy + "_both_correct"] for r in records) / len(records)
        sep = sum(r[policy + "_separated"] for r in records) / len(records)
        print("| %s | %.3f | %.3f |" % (policy, bc, sep))


if __name__ == "__main__":
    main()
