import json, os, random
BASE = str(__import__("pathlib").Path(__file__).resolve().parents[2])
DATA = os.path.join(BASE, "sota/data")
rows = [json.loads(l) for l in open(os.path.join(DATA, "manifest_500.jsonl"))]
for seed in (11, 22, 33):
    rnd = random.Random(seed)
    out = []
    for r in rows:
        paras = [dict(p) for p in r["paragraphs"]]
        rnd.shuffle(paras)
        for i, p in enumerate(paras):
            p["idx"] = i
        out.append({**r, "paragraphs": paras})
    path = os.path.join(DATA, "manifest_500_perm%d.jsonl" % seed)
    with open(path, "w") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print("wrote", path, len(out))
