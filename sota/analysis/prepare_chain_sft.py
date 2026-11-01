import json, os, sys

BASE = str(__import__("pathlib").Path(__file__).resolve().parents[2])
DATA = os.path.join(BASE, "sota/data")
OUT = os.path.join(DATA, "chain_sft_train.jsonl")

eval_ids = {json.loads(l)["id"] for l in open(os.path.join(DATA, "manifest_500.jsonl"))}
rows = [json.loads(l) for l in open(os.path.join(DATA, "data/musique_ans_v1.0_dev.jsonl"))]
train = [r for r in rows if r["id"] not in eval_ids]
print("total", len(rows), "train", len(train))


def render(paragraphs):
    ps = sorted(paragraphs, key=lambda p: p["idx"])
    return "\n\n".join("Passage %d (%s): %s" % (p["idx"], p["title"], p["paragraph_text"]) for p in ps)


def target_of(r):
    steps = []
    for i, s in enumerate(r.get("question_decomposition") or [], 1):
        q = (s.get("question") or "").strip()
        a = (s.get("answer") or "").strip()
        if not q or not a:
            continue
        steps.append("%d) %s => %s" % (i, q, a))
    return "; ".join(steps)


n = 0
with open(OUT, "w") as f:
    for r in train:
        tgt = target_of(r)
        if not tgt or len(tgt.split()) > 90:
            continue
        rec = {
            "id": r["id"],
            "hop": r["id"].split("__")[0][:4],
            "question": r["question"],
            "evidence": render(r["paragraphs"]),
            "target": tgt,
            "target_words": len(tgt.split()),
        }
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n += 1
print("written", n, "->", OUT)
