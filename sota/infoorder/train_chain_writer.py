"""LoRA training for hop-chain memories."""
import json, math, os, random, sys, time

BASE = str(__import__("pathlib").Path(__file__).resolve().parents[2])
for p in (BASE + "/deps", BASE + "/deps_fla"):
    sys.path.append(p)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL = os.environ.get("SOTA_WRITER_MODEL", BASE + "/models/Qwen3.5-4B")
DATA = os.environ.get("SOTA_SFT_DATA", BASE + "/sota/data/chain_sft_train.jsonl")
OUT = os.environ.get("SOTA_ADAPTER_OUT", BASE + "/sota/infoorder/chain_writer_qwen_adapter")
STEPS = int(os.environ.get("SOTA_SFT_STEPS", "300"))
SEED = int(os.environ.get("SOTA_SEED", "20261027"))
MAXLEN = int(os.environ.get("SOTA_MAX_LEN", "2048"))
SYSTEM = ("You write compact memory for a multi-hop question. Read the evidence and "
          "write the fact chain that answers the question, as numbered hops: "
          "'1) sub-question => answer; 2) sub-question => answer'. "
          "Use only entities and relations present in the evidence. Be terse.")


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    rows = [json.loads(l) for l in open(DATA)]
    rnd = random.Random(SEED)
    rnd.shuffle(rows)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map={"": 0})
    model.gradient_checkpointing_enable()
    lcfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                      target_modules="all-linear", task_type="CAUSAL_LM")
    model = get_peft_model(model, lcfg)
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    def encode(r):
        msgs = [{"role": "system", "content": SYSTEM},
                {"role": "user", "content": "Question: %s\n\nEvidence:\n%s\n\nMemory:"
                 % (r["question"], r["evidence"])}]
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        t_ids = tok(r["target"] + tok.eos_token, add_special_tokens=False)["input_ids"]
        ids = (p_ids + t_ids)[-MAXLEN:]
        labels = [-100] * max(0, len(ids) - len(t_ids)) + t_ids
        labels = labels[-len(ids):]
        return ids, labels

    steps, accum, batch = STEPS, 8, 1
    t0 = time.time()
    losses = []
    it = 0
    while it < steps:
        for r in rows:
            ids, labels = encode(r)
            x = torch.tensor([ids], device="cuda")
            y = torch.tensor([labels], device="cuda")
            out = model(input_ids=x, labels=y)
            loss = out.loss / accum
            loss.backward()
            losses.append(float(out.loss))
            if (len(losses) % accum) == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); opt.zero_grad(set_to_none=True)
                it += 1
                if it % 10 == 0:
                    print("[train] step %d loss %.4f elapsed %.0fs" % (it, sum(losses[-10:]) / 10, time.time() - t0), flush=True)
                if it >= steps:
                    break
    model.save_pretrained(OUT)
    print("[train] saved", OUT, "steps", it, "wall", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
