"""Generate capped hop-chain memories."""
import json, os, sys, time
BASE = str(__import__("pathlib").Path(__file__).resolve().parents[2])
for p in (BASE + "/deps", BASE + "/deps_fla"):
    sys.path.append(p)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

MODEL = os.environ.get("SOTA_WRITER_MODEL", BASE + "/models/Qwen3.5-4B")
ADAPTER = os.environ.get("SOTA_ADAPTER", BASE + "/sota/infoorder/chain_writer_qwen_adapter")
OUT = os.environ.get("SOTA_GEN_OUT", BASE + "/sota/data/chain_writer_memories_b128.jsonl")
MAXNEW = int(os.environ.get("SOTA_GEN_MAXNEW", "128"))
BUDGET = int(os.environ.get("SOTA_GEN_BUDGET", "128"))
MANIFEST_MAXLEN = int(os.environ.get("SOTA_GEN_MAXLEN", "3072"))
SYSTEM = ("You write compact memory for a multi-hop question. Read the evidence and "
          "write the fact chain that answers the question, as numbered hops: "
          "'1) sub-question => answer; 2) sub-question => answer'. "
          "Use only entities and relations present in the evidence. Be terse.")


def render(paragraphs):
    ps = sorted(paragraphs, key=lambda p: p["idx"])
    return "\n\n".join("Passage %d (%s): %s" % (p["idx"], p["title"], p["paragraph_text"]) for p in ps)


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map={"": 0})
    model = PeftModel.from_pretrained(model, ADAPTER)
    model.eval()
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    MANIFEST = os.environ.get("SOTA_GEN_MANIFEST", BASE + "/sota/data/manifest_500.jsonl")
    rows = [json.loads(l) for l in open(MANIFEST)]
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            done.add(json.loads(l)["id"])
    todo = [r for r in rows if r["id"] not in done]
    print("[gen] todo", len(todo), flush=True)
    bs = 4
    with open(OUT, "a") as f:
        for i in range(0, len(todo), bs):
            chunk = todo[i:i + bs]
            prompts = [tok.apply_chat_template(
                [{"role": "system", "content": SYSTEM},
                 {"role": "user", "content": "Question: %s\n\nEvidence:\n%s\n\nMemory:" % (r["question"], render(r["paragraphs"]))}],
                tokenize=False, add_generation_prompt=True) for r in chunk]
            enc = tok(prompts, return_tensors="pt", padding=True, padding_side="left", truncation=True, max_length=MANIFEST_MAXLEN).to("cuda")
            t0 = time.time()
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=MAXNEW, do_sample=False, pad_token_id=tok.pad_token_id)
            gen = out[:, enc["input_ids"].shape[1]:]
            for r, g in zip(chunk, gen):
                mem = tok.decode(g, skip_special_tokens=True).strip()
                n = len(tok(mem, add_special_tokens=False)["input_ids"])
                if n > BUDGET:
                    mem = tok.decode(tok(mem, add_special_tokens=False)["input_ids"][:BUDGET], skip_special_tokens=True)
                    n = len(tok(mem, add_special_tokens=False)["input_ids"])
                f.write(json.dumps({"id": r["id"], "hop": r["hop"], "memory": mem,
                                    "memory_tokens": n, "seconds": round((time.time() - t0) / len(chunk), 3)},
                                   ensure_ascii=False) + "\n")
            f.flush()
            if i % 40 == 0:
                print("[gen]", i + len(chunk), "/", len(todo), flush=True)
    print("[gen] done", flush=True)


if __name__ == "__main__":
    main()
