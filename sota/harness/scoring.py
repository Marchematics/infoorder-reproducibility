
from __future__ import annotations

import re
import string
import unicodedata
from collections import Counter
from typing import Dict, Iterable, List, Optional, Sequence

UNANSWERABLE = "UNANSWERABLE"

_PUNCT = set(string.punctuation)
_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.UNICODE)
_WS_RE = re.compile(r"\s+")

                                                                             
                                                                               
                                                                         
                     
_UNICODE_FOLD = {
    "\u2018": "'",
    "\u2019": "'",
    "\u201a": "'",
    "\u201c": '"',
    "\u201d": '"',
    "\u2013": "-",
    "\u2014": "-",
    "\u2212": "-",
    "\u00a0": " ",
    "\u3000": " ",
}


def _is_abstention(s: str) -> bool:
    t = _WS_RE.sub(" ", s.strip().lower())
    return t in {"", UNANSWERABLE.lower(), "unanswerable.", "unknown", "n/a", "none"}


def normalize_answer(s: Optional[str]) -> str:
    """Official-style SQuAD normalization (returns ``""`` for abstentions)."""
    if s is None:
        return ""
    if _is_abstention(s):
        return ""
    s = unicodedata.normalize("NFKC", s)
    for src, dst in _UNICODE_FOLD.items():
        s = s.replace(src, dst)
    s = s.lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = _ARTICLES_RE.sub(" ", s)
    return _WS_RE.sub(" ", s).strip()


_NON_ALNUM_RE = re.compile(r"[^0-9a-z]+")


def normalize_for_match(s: Optional[str]) -> str:
    """Normalizer for containment tests (``answer_in_context``).

    Punctuation is replaced by a **space** (not deleted) so that ranges and
    hyphenated spellings keep their word boundaries: ``13-15 February 1942``
    becomes ``13 15 february 1942`` and therefore contains the answer
    ``15 February 1942``.  Articles are kept (unlike :func:`normalize_answer`).
    """
    if s is None:
        return ""
    if _is_abstention(s):
        return ""
    s = unicodedata.normalize("NFKC", s)
    for src, dst in _UNICODE_FOLD.items():
        s = s.replace(src, dst)
    s = _NON_ALNUM_RE.sub(" ", s.lower())
    return _WS_RE.sub(" ", s).strip()


def _tokens(s: str) -> List[str]:
    s = normalize_answer(s)
    return s.split() if s else []


def exact_match(pred: str, gold: str) -> int:
    """1 if normalized prediction equals normalized gold, else 0."""
    return int(normalize_answer(pred) == normalize_answer(gold))


def token_f1(pred: str, gold: str) -> float:
    """SQuAD token-level F1 (special-cased for yes/no abstentions)."""
    pred_n, gold_n = normalize_answer(pred), normalize_answer(gold)
    if pred_n == gold_n:
        return 1.0 if pred_n or gold_n else 0.0
                                                          
    if not pred_n or not gold_n:
        return 0.0
    if pred_n in {"yes", "no", "noanswer"} or gold_n in {"yes", "no", "noanswer"}:
        return 1.0 if pred_n == gold_n else 0.0
    pred_toks, gold_toks = pred_n.split(), gold_n.split()
    common = Counter(pred_toks) & Counter(gold_toks)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_toks)
    recall = num_same / len(gold_toks)
    return 2 * precision * recall / (precision + recall)


def gold_answers(answer: Optional[str], aliases: Optional[Sequence[str]]) -> List[str]:
    """Gold list = ``answer`` plus ``answer_aliases`` (deduplicated, order kept).

    Falls back to ``[answer]`` when aliases are empty/None, matching the task
    spec ("answer + answer_aliases, 若无则用 answer").
    """
    out: List[str] = []
    for cand in [answer, *(aliases or [])]:
        if cand is None:
            continue
        cand = str(cand).strip()
        if cand and cand not in out:
            out.append(cand)
    return out


def score_answer(pred: str, golds: Sequence[str]) -> Dict[str, float]:
    """Best-match EM / F1 over all gold aliases.

    Returns ``{"em": int, "f1": float, "best_gold": str, "best_alias": ...}``-
    style dict; ``em`` is 1 as soon as any alias matches exactly, ``f1`` is the
    max alias F1 (the standard MuSiQue/SQuAD multi-reference protocol).
    """
    golds = [g for g in golds if g is not None]
    if not golds:
        return {"em": 0, "f1": 0.0, "best_gold": ""}
    best_em = 0
    best_f1 = 0.0
    best_gold = golds[0]
    for g in golds:
        em = exact_match(pred, g)
        f1 = token_f1(pred, g)
        if em > best_em or (em == best_em and f1 > best_f1):
            best_em, best_f1, best_gold = max(best_em, em), max(best_f1, f1), g
        else:
            best_f1 = max(best_f1, f1)
    return {"em": int(best_em), "f1": float(best_f1), "best_gold": best_gold}


def answer_in_context(
    answer: str,
    memory_text: str,
    aliases: Optional[Sequence[str]] = None,
    mode: str = "word",
) -> bool:
    """Is the (normalized) gold answer surface form present in ``memory_text``?

    ``mode="word"``      requires word boundaries (``Green`` does not match
                         ``Greenland``) -- the default, conservative reading.
    ``mode="substring"`` plain substring containment after SQuAD normalization.

    Any alias counts as a hit (multi-reference protocol).  Two normalizations
    are tried for word mode: the punctuation-to-space form (handles ranges such
    as ``13-15 February 1942``) and the SQuAD compact form (handles spellings
    such as ``U.S.A.`` vs ``usa``).
    """
    if not answer and not aliases:
        return False
    cands = [c for c in gold_answers(answer, aliases) if normalize_for_match(c)]
    if not cands:
        return False

    if mode == "substring":
        compact_mem = normalize_answer(memory_text)
        loose_mem = normalize_for_match(memory_text)
        if not compact_mem and not loose_mem:
            return False
        for cand in cands:
            if normalize_answer(cand) and normalize_answer(cand) in compact_mem:
                return True
            if normalize_for_match(cand) in loose_mem:
                return True
        return False

    loose_mem = normalize_for_match(memory_text)
    if not loose_mem:
        return False
    compact_mem = normalize_answer(memory_text)
    for cand in cands:
        loose = normalize_for_match(cand)
        if loose and re.search(r"(?<!\w)" + re.escape(loose) + r"(?!\w)", loose_mem):
            return True
        compact = normalize_answer(cand)
        if compact and compact_mem and re.search(
            r"(?<!\w)" + re.escape(compact) + r"(?!\w)", compact_mem
        ):
            return True
    return False


def aggregate(records: Iterable[Dict]) -> Dict[str, float]:
    """Mean EM / F1 / answer_in_context over eval records (+ abstention rate)."""
    rows = list(records)
    n = len(rows)
    if n == 0:
        return {"n": 0, "em": 0.0, "f1": 0.0, "aic": 0.0, "abstention": 0.0}
    em = sum(float(r.get("em", 0)) for r in rows) / n
    f1 = sum(float(r.get("f1", 0.0)) for r in rows) / n
    aic = sum(1.0 if r.get("answer_in_context") else 0.0 for r in rows) / n
    abst = (
        sum(1.0 if str(r.get("parsed_answer", "")) == UNANSWERABLE else 0.0 for r in rows)
        / n
    )
    return {"n": n, "em": em, "f1": f1, "aic": aic, "abstention": abst}
