"""Generate atomic-proposition summaries with the base writer and cap 128."""
import json, os, sys, time
BASE = str(__import__("pathlib").Path(__file__).resolve().parents[2])
for p in (BASE + "/deps", BASE + "/deps_fla"):
    if p not in sys.path:
        sys.path.append(p)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = BASE + "/models/Qwen3.5-4B"
MAN = BASE + "/sota/data/manifest_500.jsonl"
OUT = os.environ.get("BS_OUT", BASE + "/sota/data/brief_style_memories.jsonl")
BUDGET = int(os.environ.get("BS_BUDGET", "128"))
SYS = ("You are a query-aware compressor. From the documents, write a dense multi-hop summary made of "
       "self-contained atomic propositions that will let a separate reader answer the question. "
       "Resolve pronouns, keep entity names and values verbatim, omit irrelevant passages, no commentary.")


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map={"": 0}).eval()
    rows = [json.loads(l) for l in open(MAN)]
    t0 = time.time()
    with open(OUT, "w") as f:
        for i, r in enumerate(rows):
            ev = "\n\n".join("Passage %d (%s): %s" % (p["idx"], p["title"], p["paragraph_text"]) for p in r["paragraphs"])
            user = "Question: %s\n\nDocuments:\n%s\n\nCompressed summary:" % (r["question"], ev)
            text = tok.apply_chat_template([{"role": "system", "content": SYS}, {"role": "user", "content": user}],
                                           tokenize=False, add_generation_prompt=True, enable_thinking=False)
            enc = tok(text, return_tensors="pt", truncation=True, max_length=3072).to("cuda")
            with torch.no_grad():
                o = model.generate(**enc, max_new_tokens=200, do_sample=False, pad_token_id=tok.eos_token_id)
            mem = tok.decode(o[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()
            ids = tok(mem, add_special_tokens=False)["input_ids"]
            if len(ids) > BUDGET:
                mem = tok.decode(ids[:BUDGET], skip_special_tokens=True)
            f.write(json.dumps({"id": r["id"], "memory": mem, "memory_tokens": len(tok(mem, add_special_tokens=False)["input_ids"])}, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 50 == 0:
                print("[bs] %d/%d elapsed %.0fs" % (i + 1, len(rows), time.time() - t0), flush=True)
    print("[bs] done ->", OUT)


if __name__ == "__main__":
    main()
