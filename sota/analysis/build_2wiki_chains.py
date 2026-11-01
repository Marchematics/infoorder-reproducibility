import json, os, random, statistics, sys
BASE = str(__import__("pathlib").Path(__file__).resolve().parents[2])
sys.path.append(BASE + "/sota/vendor"); sys.path.append(BASE + "/deps")
import pyarrow.parquet as pq
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(BASE + "/models/Qwen3.5-4B")
DATA = os.path.join(BASE, "sota/data")
eval_ids = {json.loads(l)["id"] for l in open(os.path.join(DATA, "manifest_2wiki_fixed_300.jsonl"))}


def count(t):
    return len(tok(t, add_special_tokens=False)["input_ids"])


def parse(v):
    if isinstance(v, str):
        try:
            return json.loads(v)
        except Exception:
            return []
    return v or []


def render_ctx(ctx):
    paras = []
    for j, item in enumerate(ctx):
        if isinstance(item, dict):
            ti = item.get("title", "p%d" % j)
            ss = item.get("sentences") or item.get("sentence") or ""
        elif isinstance(item, (list, tuple)) and len(item) == 2:
            ti, ss = item
        else:
            ti, ss = "p%d" % j, item
        text = " ".join(str(x) for x in ss) if isinstance(ss, (list, tuple)) else str(ss)
        paras.append({"idx": j, "title": str(ti), "paragraph_text": text, "is_supporting": False})
    return [p for p in paras if len(p["paragraph_text"].strip()) > 20]


t = pq.read_table(os.path.join(DATA, "data/dev.parquet"))
cols = {c: t.column(c).to_pylist() for c in t.column_names}
key = "id" if "id" in cols else ("_id" if "_id" in cols else t.column_names[0])
rnd = random.Random(20261027)
idx = list(range(len(cols[key])))
rnd.shuffle(idx)
rows = []
for i in idx:
    qid = "2w_" + str(cols[key][i])
    if qid in eval_ids:
        continue
    ev = parse(cols["evidences"][i])
    hops = []
    for e in ev:
        if isinstance(e, (list, tuple)) and len(e) == 3:
            s, r, o = [str(x).strip() for x in e]
            if s and r and o:
                hops.append("%s => %s" % ("What is the %s of %s?" % (r, s), o))
    if len(hops) < 2:
        continue
    paras = render_ctx(parse(cols["context"][i]))
    if len(paras) < 2:
        continue
    evidence = "\n\n".join("Passage %d (%s): %s" % (p["idx"], p["title"], p["paragraph_text"]) for p in paras)
    target = " ; ".join("%d) %s" % (j + 1, h) for j, h in enumerate(hops))
    if count(target) > 140:
        continue
    rows.append({"id": qid, "question": cols["question"][i], "evidence": evidence, "target": target})
    if len(rows) >= 3000:
        break
out = os.path.join(DATA, "twowiki_chain_sft_train.jsonl")
with open(out, "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("2wiki chain examples", len(rows), "mean target tokens", round(statistics.mean(count(r["target"]) for r in rows), 1))
print("sample:", rows[0]["target"][:160])
