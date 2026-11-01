import json, os
BASE = str(__import__("pathlib").Path(__file__).resolve().parents[2])
DATA = os.path.join(BASE, "sota/data")
rows = [json.loads(l) for l in open(os.path.join(DATA, "chain_sft_train.jsonl"))]
out = []
for r in rows:
    hops = [h.strip() for h in r["target"].split(";") if h.strip()]
    answers = []
    for h in hops:
        parts = h.split("=>")
        if len(parts) == 2:
            answers.append(parts[1].strip())
    if not answers:
        continue
    tgt = ". ".join(answers) + "."
    out.append({"id": r["id"], "question": r["question"], "evidence": r["evidence"], "target": tgt})
with open(os.path.join(DATA, "facts_format_sft_train.jsonl"), "w") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("facts-format examples", len(out))
print("sample:", out[0]["target"][:120])
