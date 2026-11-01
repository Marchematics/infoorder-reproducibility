
from __future__ import annotations

import math
import re
from collections import Counter

__all__ = ["build_memory", "bm25_scores"]

_WORD_RE = re.compile(r"[a-z0-9]+")
                                                                              
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
K1 = 1.2
B = 0.75


def _encode(tokenizer, text):
    if not text:
        return []
    enc = tokenizer(
        text,
        add_special_tokens=False,
        return_attention_mask=False,
        return_token_type_ids=False,
    )
    ids = enc["input_ids"] if isinstance(enc, dict) else getattr(enc, "input_ids", enc)
    if isinstance(ids, list) and ids and isinstance(ids[0], list):
        ids = ids[0]
    return list(ids)


def _decode(tokenizer, ids):
    return tokenizer.decode(
        ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )


def _words(text):
    return _WORD_RE.findall(text.lower())


def _sentences(paragraphs):
    sents = []
    for p in paragraphs:
        if not isinstance(p, str) or not p.strip():
            continue
        for s in _SENT_SPLIT.split(p):
            s = s.strip()
            if s:
                sents.append(s)
    return sents


def bm25_scores(question, sentences):
    """Return one BM25 score per sentence (Okapi, k1=1.2, b=0.75)."""
    docs = [_words(s) for s in sentences]
    n_docs = len(docs)
    if n_docs == 0:
        return []
    doc_freq = Counter()
    for words in docs:
        doc_freq.update(set(words))
    avgdl = sum(len(w) for w in docs) / n_docs if n_docs else 0.0
    q_terms = _words(question)
    if not q_terms or avgdl == 0.0:
        return [0.0] * n_docs

    idf = {}
    for term in set(q_terms):
        df = doc_freq.get(term, 0)
                                                       
        idf[term] = math.log(1.0 + (n_docs - df + 0.5) / (df + 0.5))

    scores = []
    for words in docs:
        dl = len(words)
        tf = Counter(words)
        score = 0.0
        for term in q_terms:
            f = tf.get(term, 0)
            if f == 0:
                continue
            denom = f + K1 * (1.0 - B + B * dl / avgdl)
            score += idf[term] * f * (K1 + 1.0) / denom
        scores.append(score)
    return scores


def build_memory(question, paragraphs, budget, tokenizer):
    budget = max(0, int(budget))
    if budget == 0:
        return ""
    sentences = _sentences(paragraphs)
    if not sentences:
        return ""
    scores = bm25_scores(question, sentences)
                                                              
    order = sorted(range(len(sentences)), key=lambda i: (-scores[i], i))

    chosen = []
    used = 0
    for idx in order:
        sent = sentences[idx]
        n_tok = len(_encode(tokenizer, sent))
        if n_tok == 0:
            continue
        extra = n_tok + (1 if chosen else 0)                                 
        if used + extra > budget:
                                                                   
            continue
        chosen.append(sent)
        used += extra
        if used >= budget:
            break

    text = " ".join(chosen).strip()
    ids = _encode(tokenizer, text)
    if len(ids) > budget:                                         
        text = _decode(tokenizer, ids[:budget])
    return text
