import json, os, re, sys
BASE = str(__import__("pathlib").Path(__file__).resolve().parents[2])
DATA = os.path.join(BASE, "sota/data")
OUT = os.path.join(DATA, "extractive_sft_train.jsonl")
sys.path.append(BASE + "/deps")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(BASE + "/models/Qwen3.5-4B")
eval_ids = {json.loads(l)["id"] for l in open(os.path.join(DATA, "manifest_500.jsonl"))}
rows = [json.loads(l) for l in open(os.path.join(DATA, "data/musique_ans_v1.0_dev.jsonl"))]
train = [r for r in rows if r["id"] not in eval_ids]


def render(paragraphs):
    ps = sorted(paragraphs, key=lambda p: p["idx"])
    return "\n\n".join("Passage %d (%s): %s" % (p["idx"], p["title"], p["paragraph_text"]) for p in ps)


def sent_split(t):
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", t) if s.strip()]


def target_of(r):
    sents = []
    for p in sorted(r["paragraphs"], key=lambda x: x["idx"]):
        if p.get("is_supporting"):
            sents.extend(sent_split(p["paragraph_text"]))
    out, used = [], 0
    for s in sents:
        n = len(tok(s, add_special_tokens=False)["input_ids"])
        if used + n > 120:
            continue
        out.append(s)
        used += n
    return " ".join(out), used


n = 0
with open(OUT, "w") as f:
    for r in train:
        tgt, used = target_of(r)
        if not tgt:
            continue
        f.write(json.dumps({"id": r["id"], "question": r["question"], "evidence": render(r["paragraphs"]),
                            "target": tgt, "target_tokens": used}, ensure_ascii=False) + "\n")
        n += 1
print("written", n, "->", OUT)
