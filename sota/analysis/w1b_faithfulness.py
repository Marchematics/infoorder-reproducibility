"""Measure surface-answer occurrence and lexical passage alignment."""
import json
import os
import re
import statistics
import sys

BASE = str(__import__("pathlib").Path(__file__).resolve().parents[2])
if BASE + "/sota/harness" not in sys.path:
    sys.path.insert(0, BASE + "/sota/harness")
from scoring import normalize_answer, normalize_for_match, answer_in_context

MEM = os.environ.get("W1B_MEM", BASE + "/sota/data/chain_writer_memories_b128.jsonl")
MANIFEST = os.environ.get("W1B_MANIFEST", BASE + "/sota/data/manifest_500.jsonl")
READER = os.environ.get("W1B_READER", BASE + "/sota/results/musique500_chain_qwen_b128.jsonl")
OUT = os.environ.get("W1B_OUT", BASE + "/sota/results/w1b_faithfulness.jsonl")

STOP = set("""a an the of in on at to for from by with and or as is are was were be been being
that this these those it its their his her hers he she they them we you your i not no do does
did has have had which who whom whose what when where why how than then there here also into
over under about after before between during while s t can could may might must shall should
will would one two three four""".split())

HOP_MARK_RE = re.compile(r"#?(\d+)\)\s*")
TOK_RE = re.compile(r"[0-9a-z]+")


def tokens(text):
    return [t for t in TOK_RE.findall(str(text).lower()) if t not in STOP]


def coverage(hop_tokens, passage_tokens):
    """Fraction of distinct hop content tokens that occur in the passage."""
    if not hop_tokens:
        return 0.0
    return len(hop_tokens & passage_tokens) / len(hop_tokens)


def parse_hops(memory):
    """Split a memory into hops; tolerant of '#1)' and of prose prefixes.

    A hop is the text between consecutive 'N)' markers that contains '=>'.
    """
    s = str(memory)
    marks = list(HOP_MARK_RE.finditer(s))
    hops = []
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(s)
        body_full = s[m.end():end].strip().strip(";").strip()
        if "=>" not in body_full:
            continue
        body, ans = body_full.rsplit("=>", 1)
        hops.append({"n": int(m.group(1)),
                     "body": body.strip().strip(";.").strip(),
                     "answer": ans.strip().strip(";.").strip()})
    return hops


def answers_match(gen_ans, gold_ans):
    """Exact normalized equality against a gold decomposition answer."""
    g = normalize_answer(gen_ans)
    if not g:
        return False
    return any(normalize_answer(c) == g for c in gold_ans if normalize_answer(c))


def main():
    manifest = {json.loads(l)["id"]: json.loads(l) for l in open(MANIFEST)}
    mems = [json.loads(l) for l in open(MEM)]
    reader = {}
    if os.path.exists(READER):
        for l in open(READER):
            r = json.loads(l)
            reader[r["id"]] = r

    rows = []
    hop_total = 0
    grounded_total = 0
    halluc_total = 0
    attr_ok = 0
    attr_total = 0
    attr_ambig = 0
    attr_tie_ok = 0
    grounded_any_total = 0
    hop_count_hist = {}
    unparsed = 0

    for mem in mems:
        qid = mem["id"]
        item = manifest.get(qid)
        hops = parse_hops(mem["memory"])
        hop_count_hist[len(hops)] = hop_count_hist.get(len(hops), 0) + 1
        if not hops:
            unparsed += 1
        if item is None:
            print("[warn] id not in manifest:", qid, file=sys.stderr)
            continue
        paragraphs = sorted(item["paragraphs"], key=lambda p: p["idx"])
        ptoks = {p["idx"]: set(tokens(p["paragraph_text"] + " " + p["title"])) for p in paragraphs}
        gold_hops = item.get("question_decomposition") or []

        hop_rows = []
        for h in hops:
            hop_total += 1
            htoks = set(tokens(h["body"] + " " + h["answer"]))
            best_idx, best_score = None, -1.0
            for p in paragraphs:
                s = coverage(htoks, ptoks[p["idx"]])
                if s > best_score:
                    best_idx, best_score = p["idx"], s
            best_text = next(p["paragraph_text"] for p in paragraphs if p["idx"] == best_idx)
            grounded = answer_in_context(h["answer"], best_text)
            grounded_any = any(answer_in_context(h["answer"], p["paragraph_text"]) for p in paragraphs)
            hallucinated = not grounded_any
            if grounded:
                grounded_total += 1
            if grounded_any:
                grounded_any_total += 1
            if hallucinated:
                halluc_total += 1

            gold_match = [g for g in gold_hops if answers_match(h["answer"], [g.get("answer", "")])]
            gold_support = sorted({g["paragraph_support_idx"] for g in gold_match
                                   if g.get("paragraph_support_idx") is not None})
            is_gold = None
            tie_ok = None
            if gold_match:
                attr_total += 1
                if len(gold_match) > 1:
                    attr_ambig += 1
                is_gold = best_idx in gold_support
                if is_gold:
                    attr_ok += 1
                gold_best = max((coverage(htoks, ptoks[i]) for i in gold_support), default=-1.0)
                tie_ok = bool(is_gold or gold_best >= best_score - 0.02)
                if tie_ok:
                    attr_tie_ok += 1
            hop_rows.append({
                "n": h["n"], "body": h["body"], "answer": h["answer"],
                "best_passage_idx": best_idx, "best_overlap": round(best_score, 4),
                "grounded": bool(grounded), "hallucinated": bool(hallucinated),
                "grounded_any_passage": bool(grounded_any),
                "gold_match": bool(gold_match), "gold_support_idx": gold_support,
                "best_is_gold_support": is_gold, "gold_support_near_tie": tie_ok,
                "gold_best_overlap": round(gold_best, 4) if gold_match else None,
            })

        row = {
            "id": qid,
            "hop": item.get("hop") or mem.get("hop"),
            "memory_tokens": mem.get("memory_tokens"),
            "n_hops": len(hops),
            "n_grounded": sum(1 for h in hop_rows if h["grounded"]),
            "n_hallucinated": sum(1 for h in hop_rows if h["hallucinated"]),
            "all_grounded": all(h["grounded"] for h in hop_rows) if hop_rows else False,
            "n_attr_total": sum(1 for h in hop_rows if h["gold_match"]),
            "n_attr_ok": sum(1 for h in hop_rows if h["best_is_gold_support"] is True),
            "em": reader.get(qid, {}).get("em"),
            "hops": hop_rows,
        }
        rows.append(row)

    with open(OUT, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    def em_of(subset):
        vals = [r["em"] for r in subset if r["em"] is not None]
        if not vals:
            return float("nan"), 0
        return 100.0 * sum(vals) / len(vals), len(vals)

    grounded_group = [r for r in rows if r["all_grounded"]]
    ungrounded_group = [r for r in rows if not r["all_grounded"]]
    em_g, n_g = em_of(grounded_group)
    em_u, n_u = em_of(ungrounded_group)
    em_all, n_all = em_of(rows)

    print("\n=== W1b faithfulness / attribution audit ===")
    print("source memories : %s" % os.path.basename(MEM))
    print("memories        : %d   hops: %d   hops/memory mean %.2f"
          % (len(rows), hop_total, hop_total / max(1, len(rows))))
    print("hops-per-memory : %s" % dict(sorted(hop_count_hist.items())))
    print("format failures : %d memories with 0 parseable hops (%.1f%%)"
          % (unparsed, 100.0 * unparsed / max(1, len(rows))))
    print("\n-- hop level --")
    print("%-34s %8s %8s" % ("metric", "count", "rate"))
    print("%-34s %8d %7.2f%%" % ("hop grounding (answer in best pass)", grounded_total,
                                 100.0 * grounded_total / max(1, hop_total)))
    print("%-34s %8d %7.2f%%" % ("answer in SOME passage (any of 20)", grounded_any_total,
                                 100.0 * grounded_any_total / max(1, hop_total)))
    print("%-34s %8d %7.2f%%" % ("hallucination (answer in NO passage)", halluc_total,
                                 100.0 * halluc_total / max(1, hop_total)))
    print("%-34s %8d %7.2f%%" % ("attribution acc (best==gold support)", attr_ok,
                                 100.0 * attr_ok / max(1, attr_total)))
    print("%-34s %8d %7.2f%%" % ("attribution acc (near-tie tolerant)", attr_tie_ok,
                                 100.0 * attr_tie_ok / max(1, attr_total)))
    print("%-34s %8d" % ("  ... of gold-matched hops", attr_total))
    print("%-34s %8d" % ("  ... ambiguous multi-gold hops", attr_ambig))
    print("\n-- reader EM split by grounding (reader: %s) --" % os.path.basename(READER))
    print("%-34s %8s %8s %9s" % ("group", "n", "EM", "mean tok"))
    for name, grp, em in (("all hops grounded", grounded_group, em_g),
                          (">=1 ungrounded hop", ungrounded_group, em_u),
                          ("all memories", rows, em_all)):
        tok = statistics.mean([r["memory_tokens"] for r in grp if r["memory_tokens"]]) if grp else float("nan")
        print("%-34s %8d %7.2f%% %9.1f" % (name, len(grp), em, tok))
    print("\nrows -> %s" % OUT)


if __name__ == "__main__":
    main()
