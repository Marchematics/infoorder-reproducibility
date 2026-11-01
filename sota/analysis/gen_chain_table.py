import json, os, sys
BASE = str(__import__("pathlib").Path(__file__).resolve().parents[2])
for p in (BASE + "/deps", BASE + "/deps_fla"):
    if p not in sys.path:
        sys.path.append(p)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL = BASE + "/models/Qwen3.5-4B"
ADAPTER = BASE + "/sota/infoorder/chain_writer_qwen_adapter"
MAN = BASE + "/sota/data/manifest_500.jsonl"
OUT = BASE + "/sota/data/chain_table_memories_b128.jsonl"
BUDGET = int(os.environ.get("SOTA_GEN_BUDGET", "128"))
CHAIN_SYS = ("You write compact memory for a multi-hop question. Read the evidence and write the fact chain "
             "that answers the question, as numbered hops: '1) sub-question => answer; 2) sub-question => answer'. "
             "Use only entities and relations present in the evidence. Be terse.")
TABLE_SYS = ("List every entity mentioned in the evidence together with its stored value. "
             "Output one line per entity in the form 'name = value'. No commentary.")


def render(paras):
    ps = sorted(paras, key=lambda p: p["idx"])
    return "\n\n".join("Passage %d (%s): %s" % (p["idx"], p["title"], p["paragraph_text"]) for p in ps)


def count(tok, s):
    return len(tok(s, add_special_tokens=False)["input_ids"])


def gen(tok, model, sys_p, user, maxnew):
    text = tok.apply_chat_template([{"role": "system", "content": sys_p}, {"role": "user", "content": user}],
                                   tokenize=False, add_generation_prompt=True, enable_thinking=False)
    enc = tok(text, return_tensors="pt", truncation=True, max_length=3072).to("cuda")
    with torch.no_grad():
        o = model.generate(**enc, max_new_tokens=maxnew, do_sample=False, pad_token_id=tok.eos_token_id)
    return tok.decode(o[0][enc["input_ids"].shape[1]:], skip_special_tokens=True).strip()


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    base = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map={"": 0})
    model = PeftModel.from_pretrained(base, ADAPTER).eval()
    rows = [json.loads(l) for l in open(MAN)]
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            done.add(json.loads(l)["id"])
    todo = [r for r in rows if r["id"] not in done]
    print("[ct] todo", len(todo), flush=True)
    with open(OUT, "a") as f:
        for i, r in enumerate(todo):
            ev = render(r["paragraphs"])
            chain = gen(tok, model, CHAIN_SYS, "Question: %s\n\nEvidence:\n%s\n\nMemory:" % (r["question"], ev), BUDGET + 32)
            tbl = gen(tok, model, TABLE_SYS, "Evidence:\n%s\n\nTable:" % ev, 160)
            mem = chain.strip() + "\n" + tbl.strip()
            ids = tok(mem, add_special_tokens=False)["input_ids"]
            if len(ids) > BUDGET:
                                                                
                chain_ids = tok(chain, add_special_tokens=False)["input_ids"]
                if len(chain_ids) >= BUDGET:
                    mem = tok.decode(chain_ids[:BUDGET], skip_special_tokens=True)
                else:
                    rest = tok(tbl, add_special_tokens=False)["input_ids"][: BUDGET - len(chain_ids)]
                    mem = chain.strip() + "\n" + tok.decode(rest, skip_special_tokens=True)
            f.write(json.dumps({"id": r["id"], "hop": r["hop"], "memory": mem,
                                "memory_tokens": count(tok, mem)}, ensure_ascii=False) + "\n")
            f.flush()
            if (i + 1) % 50 == 0:
                print("[ct] %d/%d" % (i + 1, len(todo)), flush=True)
    print("[ct] done", flush=True)


if __name__ == "__main__":
    main()
