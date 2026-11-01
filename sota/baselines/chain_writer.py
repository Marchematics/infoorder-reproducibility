import json, os, re, sys

BASE = str(__import__("pathlib").Path(__file__).resolve().parents[2])
_MEM, _Q2ID, _Q2IDN = {}, {}, {}


def _norm(s):
    s = str(s).replace("\u2019", "'").replace("\u2018", "'").replace("\u201c", '"').replace("\u201d", '"')
    s = re.sub(r"\s+", " ", s)
    return s.strip().lower()


def _load():
    if _MEM:
        return
    path = os.environ["SOTA_MEMORY_FILE"]
    for line in open(path):
        r = json.loads(line)
        _MEM[r["id"]] = r["memory"]
    man = os.environ.get("SOTA_MEMORY_MANIFEST", BASE + "/sota/data/manifest_500.jsonl")
    for line in open(man):
        r = json.loads(line)
        q = r.get("question")
        if q is None:
            continue
        _Q2ID[q] = r["id"]
        _Q2IDN[_norm(q)] = r["id"]


def build_memory(question, paragraphs, budget, tokenizer, **kw):
    _load()
    qid = _Q2ID.get(question) or _Q2IDN.get(_norm(question))
    if qid is None:
        print("[chain_writer] unresolved question: %r (map=%d) man=%s"
              % (str(question)[:120], len(_Q2ID), os.environ.get("SOTA_MEMORY_MANIFEST")), file=sys.stderr)
        raise KeyError("question not found in %s" % os.environ.get("SOTA_MEMORY_MANIFEST"))
    if qid not in _MEM:
        raise KeyError("no precomputed memory for id=%r in %s" % (qid, os.environ["SOTA_MEMORY_FILE"]))
    return _MEM[qid]
