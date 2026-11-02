"""Measure six policy lengths and record the first-digit readout."""
import json, os, re, statistics, sys, time
BASE = str(__import__("pathlib").Path(__file__).resolve().parents[2])
for p in (BASE + "/deps", BASE + "/deps_fla"):
    if p not in sys.path:
        sys.path.append(p)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL = BASE + "/models/Qwen3.5-4B"
AD = {"chain": BASE + "/sota/infoorder/chain_writer_qwen_adapter",
      "summary": BASE + "/sota/infoorder/summary_writer_qwen_adapter",
      "facts": BASE + "/sota/infoorder/facts_format_adapter"}
MAN = os.environ.get("CF_MANIFEST", BASE + "/sota/data/controlled_bothcorrect_200.jsonl")
OUT = os.environ.get("M5_OUT", BASE + "/sota/results/m5_lengths_rule.jsonl")
PER_FAM = int(os.environ.get("CF_PER_FAMILY", "15"))
BUDGET = int(os.environ.get("CF_BUDGET", "128"))
CHAIN_SYS = ("You write compact memory for a question. Read the evidence and write the fact chain, "
             "as numbered hops. Use only entities and values present in the evidence.")
SUMMARY_SYS = ("You compress evidence into a compact plain-text memory that another model will use to "
               "answer a question. Keep only relevant facts; copy values verbatim; output plain text.")
FACT_SYS = ("You write compact memory. Output the answer facts as a short list of values separated by periods.")
TABLE_SYS = ("List every entity mentioned in the evidence together with its stored value. "
             "Output one line per entity in the form 'name = value'. No commentary.")
UPD_SYS = ("You update a compact memory when new evidence arrives for a fixed question. Merge the previous "
           "memory with the new evidence. Keep only evidence-supported values.")


def gen(tok, model, sys_p, user, maxnew=96):
    text = tok.apply_chat_template([{"role": "system", "content": sys_p}, {"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc = tok(text, return_tensors="pt", truncation=True, max_length=3072).to("cuda")
    with torch.no_grad():
        o = model.generate(**enc, max_new_tokens=maxnew, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(o[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def count(tok, s):
    return len(tok(s, add_special_tokens=False)["input_ids"])


def cap(tok, s, b):
    ids = tok(s, add_special_tokens=False)["input_ids"]
    return tok.decode(ids[:b], skip_special_tokens=True) if len(ids) > b else s


def rule_read(mem):
    m = re.search(r"\b([01])\b", mem or "")
    return m.group(1) if m else None


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map={"": 0})
    writer = PeftModel.from_pretrained(base, AD["chain"], adapter_name="chain")
    for n in ("summary", "facts"):
        if os.path.isdir(AD[n]):
            writer.load_adapter(AD[n], adapter_name=n)
    writer.eval()
    pairs = [json.loads(l) for l in open(MAN)]
    seen, capped = {}, []
    for it in pairs:
        f = it["family"]
        if seen.get(f, 0) < PER_FAM:
            capped.append(it); seen[f] = seen.get(f, 0) + 1
    print("[m5] pairs", len(capped), flush=True)
    rows = []
    for it in capped:
        rec = {"id": it["id"], "family": it["family"]}
        for policy in ("full_history", "trunc_128", "chain", "summary", "facts", "chain_table"):
            ok_rule, toks = {}, {}
            for br, hkey, akey in (("A", "history", "answer_A"), ("B", "paired_history", "answer_B")):
                h = it[hkey]
                if policy == "full_history":
                    mem = h + "\n" + it["continuation"]
                elif policy == "trunc_128":
                    mem = cap(tok, h + "\n" + it["continuation"], BUDGET)
                else:
                    sys_p = {"chain": CHAIN_SYS, "summary": SUMMARY_SYS, "facts": FACT_SYS, "chain_table": CHAIN_SYS}[policy]
                    writer.set_adapter("chain" if policy != "summary" and policy != "facts" else policy)
                    mem = cap(tok, gen(tok, writer, sys_p, "Question: %s\n\nEvidence:\n%s\n\nMemory:" % (it["question"], h)), BUDGET)
                    if policy == "chain_table":
                        tbl = gen(tok, writer, TABLE_SYS, "Evidence:\n%s\n\nTable:" % h, 128)
                        mem = cap(tok, mem + "\n" + tbl.strip(), BUDGET)
                    writer.set_adapter("chain")
                    mem = cap(tok, gen(tok, writer, UPD_SYS,
                                       "Question: %s\n\nPrevious memory:\n%s\n\nNew evidence:\n%s\n\nUpdated memory:"
                                       % (it["question"], mem, it["continuation"])), BUDGET)
                toks[br] = count(tok, mem)
                rb = rule_read(mem)
                ok_rule[br] = float(rb is not None and rb == str(it[akey]).strip())
            rec[policy + "_both_rule"] = float(ok_rule["A"] and ok_rule["B"])
            rec[policy + "_tokens"] = statistics.mean(toks.values())
        rows.append(rec)
        if len(rows) % 10 == 0:
            print("[m5] %d/%d" % (len(rows), len(capped)), flush=True)
    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print("| policy | both-correct (rule reader) | memory tok |")
    print("|---|---|---|")
    for p in ("full_history", "trunc_128", "chain", "summary", "facts", "chain_table"):
        print("| %s | %.3f | %.1f |" % (p,
              statistics.mean(r[p + "_both_rule"] for r in rows),
              statistics.mean(r[p + "_tokens"] for r in rows)))


if __name__ == "__main__":
    main()
