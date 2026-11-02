"""Train the recursive chain updater."""
import json, os, random, sys, time
BASE = str(__import__("pathlib").Path(__file__).resolve().parents[2])
for p in (BASE + "/deps", BASE + "/deps_fla"):
    sys.path.append(p)
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model

MODEL = os.environ.get("SOTA_WRITER_MODEL", BASE + "/models/Qwen3.5-4B")
DATA = os.path.join(BASE, "sota/data/update_sft.jsonl")
OUT = os.environ.get("SOTA_ADAPTER_OUT", BASE + "/sota/infoorder/updater_adapter")
STEPS = int(os.environ.get("SOTA_SFT_STEPS", "300"))
MAXLEN = int(os.environ.get("SOTA_MAX_LEN", "1536"))
UPD_SYS = ("You update a compact memory when new evidence arrives for a fixed question. Merge the "
           "previous memory with the new evidence into a new compact memory of numbered hops "
           "'1) sub-question => answer; ...'. Drop superseded facts, keep only what the evidence supports.")


def main():
    tok = AutoTokenizer.from_pretrained(MODEL)
    rows = [json.loads(l) for l in open(DATA)]
    rnd = random.Random(20261101)
    rnd.shuffle(rows)
    model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, device_map={"": 0})
    model.gradient_checkpointing_enable()
    model = get_peft_model(model, LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                                             target_modules="all-linear", task_type="CAUSAL_LM"))
    model.train()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)

    def enc(r):
        user = ("Question: %s\n\nPrevious memory:\n%s\n\nNew evidence:\n%s\n\nUpdated memory:"
                % (r["question"], r["memory_in"] or "(empty)", r["new_evidence"]))
        prompt = tok.apply_chat_template([{"role": "system", "content": UPD_SYS},
                                          {"role": "user", "content": user}],
                                         tokenize=False, add_generation_prompt=True, enable_thinking=False)
        p_ids = tok(prompt, add_special_tokens=False)["input_ids"]
        t_ids = tok(r["target"] + tok.eos_token, add_special_tokens=False)["input_ids"]
        ids = (p_ids + t_ids)[-MAXLEN:]
        labels = [-100] * max(0, len(ids) - len(t_ids)) + t_ids
        return ids, labels[-len(ids):]

    steps, accum, losses, it, t0 = STEPS, 8, [], 0, time.time()
    while it < steps:
        for r in rows:
            ids, labels = enc(r)
            x = torch.tensor([ids], device="cuda"); y = torch.tensor([labels], device="cuda")
            loss = model(input_ids=x, labels=y).loss / accum
            loss.backward()
            losses.append(float(loss.detach()) * accum)
            if len(losses) % accum == 0:
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step(); opt.zero_grad(set_to_none=True); it += 1
                if it % 20 == 0:
                    print("[upd] step %d loss %.4f elapsed %.0fs" % (it, sum(losses[-20:]) / 20, time.time() - t0), flush=True)
                if it >= steps:
                    break
    model.save_pretrained(OUT)
    print("[upd] saved", OUT, "wall", round(time.time() - t0), flush=True)


if __name__ == "__main__":
    main()
