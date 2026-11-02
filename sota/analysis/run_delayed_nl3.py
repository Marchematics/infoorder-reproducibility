"""Write one question-free memory and answer subsequent subquestions."""
import json, os, sys, statistics, time
BASE = str(__import__("pathlib").Path(__file__).resolve().parents[2])
for p in (BASE + "/deps", BASE + "/deps_fla", BASE + "/sota/harness"):
    if p not in sys.path:
        sys.path.append(p)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import scoring

MODEL = BASE + "/models/Qwen3.5-4B"
AD = {"chain": BASE + "/sota/infoorder/chain_writer_qwen_adapter",
      "summary": BASE + "/sota/infoorder/summary_writer_qwen_adapter",
      "noq": BASE + "/sota/infoorder/noq_writer_adapter"}
NOQ_SYS = ("You write a compact memory from evidence alone, for a task whose question is not yet known. "
           "Output numbered factual hops and then a table of entity=value bindings.")
MAN = BASE + "/sota/data/manifest_500.jsonl"
OUT = os.environ.get("DN_OUT", BASE + "/sota/results/delayed_nl.jsonl")
N = int(os.environ.get("DN_N", "100"))
PREFIX = int(os.environ.get("DN_PREFIX", "8"))
BUDGET = int(os.environ.get("DN_BUDGET", "128"))
CHAIN_NOQ = ("Write a compact fact chain from the evidence: numbered hops "
             "'N) entity relation value'. Use only entities and values present in the evidence.")
CHAIN_Q = ("You write compact memory for a question. Read the evidence and write the fact chain, "
           "as numbered hops. Use only entities and values present in the evidence.")
SUMMARY_NOQ = ("Compress the evidence into a compact plain-text memory. Keep only concrete facts; "
               "copy values verbatim; output plain text.")
READER = ("You answer questions using ONLY the context provided. Reply with the shortest possible "
          "answer phrase. If the context does not contain enough information, reply with exactly UNANSWERABLE.")


def count(tok, s):
    return len(tok(s, add_special_tokens=False)["input_ids"])


def cap(tok, s, b):
    ids = tok(s, add_special_tokens=False)["input_ids"]
    return tok.decode(ids[:b], skip_special_tokens=True) if len(ids) > b else s


def gen(tok, model, sys_p, user, maxnew=128):
    text = tok.apply_chat_template([{"role": "system", "content": sys_p}, {"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc = tok(text, return_tensors="pt", truncation=True, max_length=3072).to("cuda")
    with torch.no_grad():
        o = model.generate(**enc, max_new_tokens=maxnew, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(o[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map={"": 0})
    writer = PeftModel.from_pretrained(base, AD["chain"], adapter_name="chain")
    writer.load_adapter(AD["noq"], adapter_name="noq")
    if os.path.isdir(AD["summary"]):
        writer.load_adapter(AD["summary"], adapter_name="summary")
    writer.eval()
    rows = [json.loads(l) for l in open(MAN)][:N]
    out = []
    t0 = time.time()
    for i, r in enumerate(rows):
        prefix = [p for p in sorted(r["paragraphs"], key=lambda x: x["idx"]) if p["idx"] < PREFIX]
        idxs = {p["idx"] for p in prefix}
        sels = [s for s in (r.get("question_decomposition") or [])
                if isinstance(s.get("paragraph_support_idx"), int) and s["paragraph_support_idx"] in idxs
                and str(s.get("answer", "")).strip()]
        if not prefix or len(sels) < 2:
            continue
        ev = "\n\n".join("Passage %d (%s): %s" % (p["idx"], p["title"], p["paragraph_text"]) for p in prefix)
        mems = {}
        mems["full_prefix_uncapped"] = ev
        mems["full_prefix_128"] = cap(tok, ev, BUDGET)
        mems["chain_noQ"] = cap(tok, gen(tok, writer, CHAIN_NOQ,
                                         "Evidence:\n%s\n\nMemory:" % ev), BUDGET)
        writer.set_adapter("chain")
        writer.set_adapter("noq")
        mems["chain_noQ_trained"] = cap(tok, gen(tok, writer, NOQ_SYS,
                                                 "Evidence:\n%s\n\nMemory:" % ev), BUDGET)
        writer.set_adapter("chain")
        mems["chain_with_parentQ"] = cap(tok, gen(tok, writer, CHAIN_Q,
                                                  "Question: %s\n\nEvidence:\n%s\n\nMemory:" % (r["question"], ev)), BUDGET)

        tbl = gen(tok, writer, "List every entity mentioned in the evidence with its value, "
                               "one line per entity as 'name = value'. No commentary.",
                  "Evidence:\n%s\n\nTable:" % ev, 128)
        mems["chain_table_noQ"] = cap(tok, mems["chain_noQ"] + "\n" + tbl.strip(), BUDGET)
        if "summary" in getattr(writer, "peft_config", {}):
            writer.set_adapter("summary")
            mems["summary_noQ"] = cap(tok, gen(tok, writer, SUMMARY_NOQ,
                                               "Evidence:\n%s\n\nMemory:" % ev), BUDGET)
            writer.set_adapter("chain")
        rec = {"id": r["id"], "n_selectors": len(sels)}
        for cond, mem in mems.items():
            acc = []
            for s in sels:
                with writer.disable_adapter():
                    pred = gen(tok, writer, READER,
                               "Context:\n%s\n\nQuestion: %s\n\nAnswer:" % (mem, s["question"]), 24)
                sc = scoring.score_answer(pred, [s["answer"]])
                acc.append(float(sc.get("em", 0)))
            rec[cond + "_acc"] = statistics.mean(acc)
            rec[cond + "_all"] = float(all(a > 0 for a in acc))
            rec[cond + "_tokens"] = count(tok, mem)
        out.append(rec)
        if (i + 1) % 10 == 0:
            print("[dn] %d/%d elapsed %.0fs" % (i + 1, len(rows), time.time() - t0), flush=True)
    with open(OUT, "w") as f:
        for x in out:
            f.write(json.dumps(x) + "\n")
    conds = ["full_prefix_uncapped", "full_prefix_128", "chain_noQ", "chain_table_noQ",
             "chain_with_parentQ", "summary_noQ"]
    print("| condition | selector acc | all-selectors-correct | memory tok |")
    print("|---|---|---|---|")
    for c in conds:
        if not out or c + "_acc" not in out[0]:
            continue
        print("| %s | %.3f | %.3f | %.1f |" % (
            c, statistics.mean(x[c + "_acc"] for x in out),
            statistics.mean(x[c + "_all"] for x in out),
            statistics.mean(x[c + "_tokens"] for x in out)))


if __name__ == "__main__":
    main()
