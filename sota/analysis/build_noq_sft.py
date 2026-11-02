"""Build evidence-only chain and entity-table targets."""
import json, os, re
BASE = str(__import__("pathlib").Path(__file__).resolve().parents[2])
SRC = os.path.join(BASE, "sota/data/chain_sft_train.jsonl")
OUT = os.path.join(BASE, "sota/data/noq_sft.jsonl")


def parse_items(target):
    items = []
    for part in re.split(r";\s*(?=\d+\))", target.strip()):
        m = re.match(r"(\d+)\)\s*(.*)$", part.strip(), re.S)
        if m and "=>" in m.group(2):
            items.append(m.group(2).strip())
    return items


def split_hop(body):
    left, val = [x.strip() for x in body.split("=>", 1)] if "=>" in body else (body, "")
    if ">>" in left:
        subj, rel = [x.strip() for x in left.split(">>", 1)]
    else:
        subj, rel = left.strip(), ""
    return subj, rel, val


n = 0
with open(OUT, "w") as f:
    for line in open(SRC):
        r = json.loads(line)
        hops = [split_hop(b) for b in parse_items(r.get("target", ""))]
        if not hops:
            continue
        chain = "; ".join("%d) %s >> %s => %s" % (i + 1, s, rel or "related", v)
                           for i, (s, rel, v) in enumerate(hops))
        table = "\n".join("%s = %s" % (s, v) for s, _rel, v in hops)
        target = chain + "\nTable:\n" + table
        f.write(json.dumps({"id": r["id"], "question": r["question"],
                            "evidence": r["evidence"], "target": target}, ensure_ascii=False) + "\n")
        n += 1
print("noQ SFT", n, "->", OUT)
