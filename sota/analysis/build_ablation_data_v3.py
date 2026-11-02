"""Build recorded target variants; permutation retains original hop references."""
import json, os, random, re
BASE = str(__import__("pathlib").Path(__file__).resolve().parents[2])
SRC = os.path.join(BASE, "sota/data/chain_sft_train.jsonl")                                   
OUT = os.path.join(BASE, "sota/data/ablation_v3")
os.makedirs(OUT, exist_ok=True)
rnd = random.Random(20261102)


def parse_items(target):
    """Split a chain target into hop units (works with '; ' and ' ; ' separators)."""
    items = []
    for part in re.split(r";\s*(?=\d+\))", target.strip()):
        m = re.match(r"(\d+)\)\s*(.*)$", part.strip(), re.S)
        if not m:
            continue
        body = m.group(2).strip()
        if "=>" in body:
            items.append(body)
    return items


def split_hop(body):
    """Return (subject, relation, value, evidence_suffix)."""
    ev = ""
    if "| evidence:" in body:
        body, ev = body.split("| evidence:", 1)
        ev = "| evidence:" + ev
    body = body.strip()
    if "=>" in body:
        left, val = [x.strip() for x in body.split("=>", 1)]
    else:
        left, val = body, ""
    if ">>" in left:
        subj, rel = [x.strip() for x in left.split(">>", 1)]
    else:
        subj, rel = left.strip(), ""
    return subj, rel, val, ev


def main():
    files = {k: open(os.path.join(OUT, k + ".jsonl"), "w") for k in
             ("chain", "chain_shuffled", "chain_no_relation", "chain_triples", "direct_answer")}
    n = 0
    for line in open(SRC):
        r = json.loads(line)
        items = parse_items(r.get("target", ""))
        if not items:
            continue
        hops = [split_hop(b) for b in items]
        base = {"id": r["id"], "question": r["question"], "evidence": r["evidence"]}
                                                                   
        t = "; ".join("%d) %s" % (i + 1, b) for i, b in enumerate(items))
        files["chain"].write(json.dumps(dict(base, target=t), ensure_ascii=False) + "\n")
                                                                     
        perm = list(range(len(items)))
        rnd.shuffle(perm)
        t = "; ".join("%d) %s" % (i + 1, items[j]) for i, j in enumerate(perm))
        files["chain_shuffled"].write(json.dumps(dict(base, target=t), ensure_ascii=False) + "\n")
                                                                            
        t = "; ".join("%d) %s => %s %s" % (i + 1, h[0], h[2], h[3]) for i, h in enumerate(hops))
        files["chain_no_relation"].write(json.dumps(dict(base, target=t.replace("  ", " ")), ensure_ascii=False) + "\n")
                                                                                 
        t = "\n".join("(%s, %s, %s)" % (h[0], h[1] or "related", h[2]) for h in hops)
        files["chain_triples"].write(json.dumps(dict(base, target=t), ensure_ascii=False) + "\n")
                                                 
        files["direct_answer"].write(json.dumps(dict(base, target=hops[-1][2]), ensure_ascii=False) + "\n")
        n += 1
    for f in files.values():
        f.close()
    print("built", n, "records x 5 variants ->", OUT)


if __name__ == "__main__":
    main()
