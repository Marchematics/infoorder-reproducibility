"""Build append-update training examples."""
import json, os
BASE = str(__import__("pathlib").Path(__file__).resolve().parents[2])
DATA = os.path.join(BASE, "sota/data")
DEV = os.path.join(DATA, "data/musique_ans_v1.0_dev.jsonl")
OUT = os.path.join(DATA, "append_sft.jsonl")
K = int(os.environ.get("APP_K", "5"))
eval_ids = {json.loads(l)["id"] for l in open(os.path.join(DATA, "manifest_500.jsonl"))}
rows = [json.loads(l) for l in open(DEV)]
train = [r for r in rows if r["id"] not in eval_ids]


def render(paras):
    return "\n\n".join("Passage %d (%s): %s" % (p["idx"], p["title"], p["paragraph_text"]) for p in paras)


def hops_all(r):
    out = []
    for s in (r.get("question_decomposition") or []):
        q = (s.get("question") or "").strip(); a = (s.get("answer") or "").strip()
        idx = s.get("paragraph_support_idx")
        if q and a:
            out.append((q, a, idx))
    return out


n = 0
with open(OUT, "w") as f:
    for r in train:
        paras = sorted(r["paragraphs"], key=lambda p: p["idx"])
        size = max(1, (len(paras) + K - 1) // K)
        chunks = [paras[i:i + size] for i in range(0, len(paras), size)][:K]
        hs = hops_all(r)
        if not hs:
            continue
        mem = ""
        seen = set()
        for t, ch in enumerate(chunks, 1):
            seen |= {p["idx"] for p in ch}
            new = [(q, a) for q, a, idx in hs if isinstance(idx, int) and idx in seen and (q, a) not in [(x, y) for x, y, _ in hs[:0]]]
            new = [x for x in new if "%s => %s" % x not in mem]
            if not new:
                if not mem:
                    continue
                target = mem
            else:
                add = " ; ".join("%d) %s => %s" % (len(mem.split(";")) if mem else 0, q, a) for q, a in new)
                ids = [x for x in mem.split(";") if x.strip()]
                start = len(ids)
                add = " ; ".join("%d) %s => %s" % (start + i + 1, q, a) for i, (q, a) in enumerate(new))
                target = (mem + " ; " + add) if mem else add
            f.write(json.dumps({"id": "%s:t%d" % (r["id"], t), "question": r["question"],
                                "memory_in": mem, "new_evidence": render(ch), "target": target},
                               ensure_ascii=False) + "\n")
            n += 1
            mem = target
print("append SFT examples", n, "->", OUT)
