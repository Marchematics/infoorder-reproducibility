
from __future__ import annotations

import re

__all__ = ["trunc_head", "trunc_tail", "build_memory_head", "build_memory_tail"]

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")


def _encode(tokenizer, text):
    """Token ids for ``text`` without special tokens (robust to tokenizer flavours)."""
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
    return tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)


def _join_paragraphs(paragraphs):
    return "\n\n".join(p.strip() for p in paragraphs if isinstance(p, str) and p.strip())


def trunc_head(question, paragraphs, budget, tokenizer):
    """Keep the first ``budget`` tokens of the concatenated context."""
    budget = max(0, int(budget))
    text = _join_paragraphs(paragraphs)
    ids = _encode(tokenizer, text)
    if budget == 0:
        return ""
    if len(ids) <= budget:
        return text
    return _decode(tokenizer, ids[:budget])


def trunc_tail(question, paragraphs, budget, tokenizer):
    """Keep the last ``budget`` tokens of the concatenated context."""
    budget = max(0, int(budget))
    text = _join_paragraphs(paragraphs)
    ids = _encode(tokenizer, text)
    if budget == 0:
        return ""
    if len(ids) <= budget:
        return text
    return _decode(tokenizer, ids[-budget:])


                                                                            
def build_memory_head(question, paragraphs, budget, tokenizer):
    return trunc_head(question, paragraphs, budget, tokenizer)


def build_memory_tail(question, paragraphs, budget, tokenizer):
    return trunc_tail(question, paragraphs, budget, tokenizer)


                                                                  
def build_memory(question, paragraphs, budget, tokenizer):
    return trunc_head(question, paragraphs, budget, tokenizer)
