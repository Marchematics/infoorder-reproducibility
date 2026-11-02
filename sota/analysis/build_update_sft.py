"""Build prefix-based recursive training examples."""
import json, os
BASE = str(__import__("pathlib").Path(__file__).resolve().parents[2])
DATA = os.path.join(BASE, "sota/data")
DEV = os.path.join(DATA, "data/musique_ans_v1.0_dev.jsonl")
OUT = os.path.join(DATA, "update_sft.jsonl")
K = int(os.environ.get("UPD_K", "5"))

eval_ids = {json.loads(l)["id"] for l in open(os.path.join(DATA, "manifest_500.jsonl"))}
rows = [json.loads(l) for l in open(DEV)]
train = [r for r in rows if r["id"] not in eval_ids]


def render(paras):
    return "\n\n".join("Passage %d (%s): %s" % (p["idx"], p["title"], p["paragraph_text"]) for p in paras)


def chain_for_prefix(r, seen_idx):
    steps = r.get("question_decomposition") or []
    out = []
    for s in steps:
        idx = s.get("paragraph_support_idx")
        if isinstance(idx, int) and idx not in seen_idx:
            continue
        q = (s.get("question") or "").strip()
        a = (s.get("answer") or "").strip()
        if q and a:
            out.append("%d) %s => %s" % (len(out) + 1, q, a))
    return "; ".join(out)


n = 0
with open(OUT, "w") as f:
    for r in train:
        paras = sorted(r["paragraphs"], key=lambda p: p["idx"])
        size = max(1, (len(paras) + K - 1) // K)
        chunks = [paras[i:i + size] for i in range(0, len(paras), size)][:K]
        seen, prev_mem = set(), ""
        for t, ch in enumerate(chunks, 1):
            seen |= {p["idx"] for p in ch}
            tgt = chain_for_prefix(r, seen)
            if not tgt:
                prev_mem = ""
                continue
            f.write(json.dumps({"id": "%s:t%d" % (r["id"], t), "question": r["question"],
                                "memory_in": prev_mem, "new_evidence": render(ch), "target": tgt},
                               ensure_ascii=False) + "\n")
            n += 1
            prev_mem = tgt
print("update SFT examples", n, "->", OUT)
