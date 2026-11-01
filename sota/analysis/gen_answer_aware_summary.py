import json, os, sys, time
BASE = str(__import__("pathlib").Path(__file__).resolve().parents[2])
for p in (BASE + "/deps", BASE + "/deps_fla"):
    if p not in sys.path:
        sys.path.append(p)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL = BASE + "/models/Qwen3.5-4B"
OUT = BASE + "/sota/data/answer_aware_summary_sft_train.jsonl"
BUDGET = 128
SYS = ("You write compact memory for a multi-hop question. Write a short prose memory "
       "(<= {budget} tokens) that states the answer and the facts that justify it. "
       "Do not add commentary.").format(budget=BUDGET)


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map={"": 0}).eval()
    rows = [json.loads(l) for l in open(BASE + "/sota/data/chain_sft_train.jsonl")]
    dev = {}
    for line in open(BASE + "/sota/data/data/musique_ans_v1.0_dev.jsonl"):
        r = json.loads(line)
        dev[r["id"]] = r
    done = set()
    if os.path.exists(OUT):
        for l in open(OUT):
            done.add(json.loads(l)["id"])
    todo = [r for r in rows if r["id"] not in done]
    print("[aas] todo", len(todo), flush=True)
    bs = 4
    t0 = time.time()
    with open(OUT, "a") as f:
        for i in range(0, len(todo), bs):
            chunk = todo[i:i + bs]
            prompts = []
            for r in chunk:
                d = dev.get(r["id"], {})
                ans = (d.get("answer") or "").strip()
                user = ("Question: %s\n\nEvidence:\n%s\n\nThe correct answer is: %s\n\nWrite the memory:"
                        % (r["question"], r["evidence"], ans))
                prompts.append(tok.apply_chat_template(
                    [{"role": "system", "content": SYS}, {"role": "user", "content": user}],
                    tokenize=False, add_generation_prompt=True, enable_thinking=False))
            enc = tok(prompts, return_tensors="pt", padding=True, padding_side="left",
                      truncation=True, max_length=3072).to("cuda")
            with torch.no_grad():
                o = model.generate(**enc, max_new_tokens=160, do_sample=False,
                                   pad_token_id=tok.pad_token_id or tok.eos_token_id)
            for r, g in zip(chunk, o[:, enc["input_ids"].shape[1]:]):
                txt = tok.decode(g, skip_special_tokens=True).strip()
                ids = tok(txt, add_special_tokens=False)["input_ids"]
                if len(ids) > BUDGET:
                    txt = tok.decode(ids[:BUDGET], skip_special_tokens=True)
                f.write(json.dumps({"id": r["id"], "question": r["question"],
                                    "evidence": r["evidence"], "target": txt}, ensure_ascii=False) + "\n")
            f.flush()
            if i % 80 == 0:
                print("[aas] %d/%d %.0fs" % (i + len(chunk), len(todo), time.time() - t0), flush=True)
    print("[aas] done", sum(1 for _ in open(OUT)), flush=True)


if __name__ == "__main__":
    main()
