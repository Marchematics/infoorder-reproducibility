import json, os, sys, time
BASE = str(__import__("pathlib").Path(__file__).resolve().parents[2])
for p in (BASE + "/deps", BASE + "/deps_fla", BASE + "/sota/harness"):
    if p not in sys.path:
        sys.path.append(p)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import prompting

MODEL = BASE + "/models/Qwen3.5-4B"
OUT = BASE + "/sota/data/summary_sft_train.jsonl"
BUDGET = 128


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map={"": 0}).eval()
    rows = [json.loads(l) for l in open(BASE + "/sota/data/chain_sft_train.jsonl")]
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            done.add(json.loads(l)["id"])
    todo = [r for r in rows if r["id"] not in done]
    print("[sum] todo", len(todo), flush=True)
    t0 = time.time()
    bs = 4
    with open(OUT, "a") as f:
        for i in range(0, len(todo), bs):
            chunk = todo[i:i + bs]
            prompts = []
            for r in chunk:
                paras = r["evidence"].split("\n\n")
                _m, text, _n = prompting.build_summary(r["question"], paras, BUDGET, tok)
                prompts.append(text)
            enc = tok(prompts, return_tensors="pt", padding=True, padding_side="left", truncation=True, max_length=3072).to("cuda")
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=160, do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
            gen = out[:, enc["input_ids"].shape[1]:]
            for r, g in zip(chunk, gen):
                txt = tok.decode(g, skip_special_tokens=True).strip()
                ids = tok(txt, add_special_tokens=False)["input_ids"]
                if len(ids) > BUDGET:
                    txt = tok.decode(ids[:BUDGET], skip_special_tokens=True)
                rec = {"id": r["id"], "question": r["question"], "evidence": r["evidence"], "target": txt}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if i % 80 == 0:
                print("[sum] %d/%d elapsed=%.0fs" % (i + len(chunk), len(todo), time.time() - t0), flush=True)
    print("[sum] done", sum(1 for _ in open(OUT)), flush=True)


if __name__ == "__main__":
    main()
