import re

_CAP = re.compile(r"\b[A-Z][A-Za-z'\-]+(?:\s+[A-Z][A-Za-z'\-]+)*")
_SENT = re.compile(r"(?<=[.!?])\s+")
_WORD = re.compile(r"[A-Za-z][A-Za-z'\-]{2,}")

STOP = set("""the a an of in on at to for from by with and or but if then than that this these those is are was were be been
as it its their his her she he they them we you i not no do does did has have had will would can could may might must
which who whom whose what when where why how there here also such other more most some any all each both few many own same
after before during while about into over under between out up down off again further once only very just one two first""".split())


def _sentences(text):
    return [s.strip() for s in _SENT.split(text) if s.strip()]


def _count(text, tokenizer):
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])


def _words(text):
    return {w.lower() for w in _WORD.findall(text)} - STOP


def _entities(text):
    out = set()
    for m in _CAP.finditer(text):
        e = m.group(0).strip()
        if len(e) >= 3 and e.lower() not in STOP:
            out.add(e)
    return out


def _pack(units, budget, tokenizer, sep="\n"):
    chosen, used = [], 0
    for u in units:
        n = _count(u, tokenizer)
        if used + n > budget:
            continue
        chosen.append(u)
        used += n
    return sep.join(chosen), used


                                                                     
_PROTO = re.compile(r"^(?:New evidence|Observed facts|Evidence|Facts|Memory)\s*:\s*$", re.I)
_HEAD = re.compile(r"^Passage\s+\d+\s*\([^)]*\)\s*:\s*", re.I)


def postprocess_writer(memory, budget, tokenizer):
    units = []
    for raw in memory.splitlines():
        line = _HEAD.sub("", raw.strip())
        if not line or _PROTO.match(line):
            continue
        line = line.lstrip("-* ").strip()
        for s in _sentences(line):
            s = s.strip()
            if len(s) < 12 or s in units:
                continue
            units.append(s)
    return _pack(units, budget, tokenizer)[0]


                                                                        
def _index(paragraphs, tokenizer):
    out = []
    for p in sorted(paragraphs, key=lambda x: x.get("idx", 0)):
        title = (p.get("title") or "").strip()
        title_words = _words(title)
        for s in _sentences(p.get("paragraph_text") or ""):
            if len(s) < 12:
                continue
            out.append({
                "text": s,
                "title": title,
                "title_words": title_words,
                "ents": _entities(s) | _entities(title),
                "words": _words(s),
                "tokens": _count(s, tokenizer),
            })
    return out


def _build(paragraphs, question, budget, tokenizer, branches=1, max_rounds=200):
    sents = _index(paragraphs, tokenizer)
    q_ents = _entities(question)
    q_words = _words(question)
    if not sents:
        return ""
    frontier = set(q_ents)
    hits = {i: 0 for i in range(len(sents))}
    selected, used, taken = [], 0, set()
    while True:
        best, best_score = None, -1e9
        for i, s in enumerate(sents):
            if i in taken or hits[i] >= branches:
                continue
            linked = s["ents"] & frontier
            new = s["ents"] - frontier
            lex = len(q_words & s["words"]) + 0.5 * len(q_words & s["title_words"])
            if not linked and not lex:
                if taken:
                    continue
                score = 0.1                                                    
            else:
                score = 2.0 * len(linked) + 1.2 * len(new) + 0.9 * lex + 0.2 * len(s["ents"])
            if used + s["tokens"] > budget:
                continue
            if score > best_score:
                best, best_score = i, score
        if best is None:
            break
        s = sents[best]
        prefix = (s["title"][:40] + ": ") if s["title"] else ""
        unit = prefix + s["text"]
        nu = _count(unit, tokenizer)
        if used + nu > budget:
            taken.add(best)
            continue
        selected.append(unit)
        used += nu
        taken.add(best)
        for i, t in enumerate(sents):
            if i in taken:
                continue
            if t["ents"] & s["ents"]:
                hits[i] = min(branches, hits[i] + 1)
        frontier |= s["ents"]
        if len(selected) > max_rounds:
            break
    return "\n".join(selected)


def build_memory(question, paragraphs, budget, tokenizer, variant="s3", **kw):
    if variant == "s1":
        raise RuntimeError("s1 operates on writer output; call postprocess_writer(memory, budget, tokenizer)")
    branches = {"s2": 1, "s3": 3}.get(variant, 3)
    return _build(paragraphs, question, budget, tokenizer, branches=branches)


                                                                              
def _bm25(question, texts):
    import sys
    BASE = str(__import__("pathlib").Path(__file__).resolve().parents[2])
    p = BASE + "/sota/baselines"
    if p not in sys.path:
        sys.path.append(p)
    import bm25
    return bm25.bm25_scores(question, texts)


def _build_hybrid(paragraphs, question, budget, tokenizer, branches=2, topk=1):
    """BM25 seed + chain expansion + distinction cap, packed whole-sentence."""
    sents = _index(paragraphs, tokenizer)
    if not sents:
        return ""
    scores = _bm25(question, [s["text"] for s in sents])
    order = sorted(range(len(sents)), key=lambda i: (-scores[i], i))
    selected, used, taken = [], 0, set()
    hits = {i: 0 for i in range(len(sents))}
                                                                   
    for i in order[:topk]:
        s = sents[i]
        unit = (s["title"][:40] + ": " if s["title"] else "") + s["text"]
        nu = _count(unit, tokenizer)
        if used + nu > budget:
            continue
        selected.append(unit)
        used += nu
        taken.add(i)
    frontier = set()
    for i in taken:
        frontier |= sents[i]["ents"]
    while True:
        best, best_score = None, -1e9
        for i, s in enumerate(sents):
            if i in taken or hits[i] >= branches:
                continue
            linked = s["ents"] & frontier
            lex = scores[i]
            if not linked and lex <= 0:
                continue
            score = 2.0 * len(linked) + 0.6 * lex
            if used + s["tokens"] > budget:
                continue
            if score > best_score:
                best, best_score = i, score
        if best is None:
            break
        s = sents[best]
        unit = (s["title"][:40] + ": " if s["title"] else "") + s["text"]
        nu = _count(unit, tokenizer)
        if used + nu > budget:
            taken.add(best)
            continue
        selected.append(unit)
        used += nu
        taken.add(best)
        for i, t in enumerate(sents):
            if i not in taken and (t["ents"] & s["ents"]):
                hits[i] = min(branches, hits[i] + 1)
        frontier |= s["ents"]
                                                                       
    for i in order:
        if i in taken:
            continue
        s = sents[i]
        unit = (s["title"][:40] + ": " if s["title"] else "") + s["text"]
        nu = _count(unit, tokenizer)
        if nu == 0 or used + nu > budget:
            continue
        selected.append(unit)
        used += nu
        taken.add(i)
                                                                                 
    rel = [i for i in range(len(sents)) if scores[i] > 0 and i not in taken]
    rel.sort(key=lambda i: sents[i]["tokens"])
    for i in rel:
        if i in taken:
            continue
        s = sents[i]
        unit = (s["title"][:40] + ": " if s["title"] else "") + s["text"]
        nu = _count(unit, tokenizer)
        if nu == 0 or used + nu > budget:
            continue
        selected.append(unit)
        used += nu
        taken.add(i)
    return "\n".join(selected)


def build_memory_hybrid(question, paragraphs, budget, tokenizer, variant="s4"):
    mem = _build_hybrid(paragraphs, question, budget, tokenizer)
    units = [u for u in mem.split("\n") if u]
    while units and _count("\n".join(units), tokenizer) > budget:
        units.pop()
    return "\n".join(units)
