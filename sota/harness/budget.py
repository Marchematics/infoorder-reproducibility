
from __future__ import annotations

import functools
import re
from typing import List, Optional, Sequence, Tuple

import paths

                              
                                                                                
                                                         
                                                                             
_BOUNDARY_RE = re.compile(
    r"(?:(?<=[.!?])[ \t]+(?=[\"'(\[\u201c\u2018]?[A-Z0-9])"          
    r"|[ \t]*\n[ \t]*)"                    
)


                                                                            
                     
                                                                            
@functools.lru_cache(maxsize=4)
def load_tokenizer(tokenizer_dir: Optional[str] = None):
    """Load (and cache) the project tokenizer. CPU only, no model weights."""
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(tokenizer_dir or paths.TOKENIZER_DIR)


class TokenCounter:
    """Thin wrapper around a HF tokenizer with the harness counting standard."""

    def __init__(self, tokenizer=None, tokenizer_dir: Optional[str] = None):
        self.tokenizer = tokenizer if tokenizer is not None else load_tokenizer(tokenizer_dir)

                                                                             
    def encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(text or "", add_special_tokens=False)

    def decode(self, ids: Sequence[int]) -> str:
        return self.tokenizer.decode(list(ids), skip_special_tokens=True)

    def count(self, text: str) -> int:
        if not text:
            return 0
        return len(self.encode(text))

                                                                             
    def truncate_tokens(self, text: str, budget: int, side: str = "head") -> Tuple[str, int]:
        """Exact token-level truncation. ``side="head"`` keeps the prefix."""
        ids = self.encode(text or "")
        if budget <= 0:
            return "", 0
        if len(ids) <= budget:
            return text, len(ids)
        kept = ids[:budget] if side == "head" else ids[len(ids) - budget :]
        out = self.decode(kept)
        return out, self.count(out)


_DEFAULT_COUNTER: Optional[TokenCounter] = None


def get_counter(counter: Optional[TokenCounter] = None) -> TokenCounter:
    """Return ``counter`` or the process-wide default (lazily built)."""
    global _DEFAULT_COUNTER
    if counter is not None:
        return counter
    if _DEFAULT_COUNTER is None:
        _DEFAULT_COUNTER = TokenCounter()
    return _DEFAULT_COUNTER


def count_tokens(text: str, counter: Optional[TokenCounter] = None) -> int:
    return get_counter(counter).count(text)


                                                                            
                       
                                                                            
def split_sentences(text: str) -> List[str]:
    """Split into sentence-ish units; ``"".join(result) == text`` is NOT
    guaranteed for whitespace-only inputs but the concatenation of units
    reproduces ``text`` byte for byte for normal prose."""
    if not text:
        return []
    units: List[str] = []
    start = 0
    for m in _BOUNDARY_RE.finditer(text):
        end = m.end()
        if end <= start:
            continue
        units.append(text[start:end])
        start = end
    if start < len(text):
        units.append(text[start:])
    return [u for u in units if u.strip()]


def _join_units(units: Sequence[str]) -> str:
    return " ".join(u.strip() for u in units if u.strip())


                                                                            
                       
                                                                            
def truncate_head(text: str, budget: int, counter: Optional[TokenCounter] = None) -> Tuple[str, int]:
    """Keep the **head** (prefix) of ``text`` within ``budget`` tokens.

    Token-exact: may end mid-word. Use :func:`select_sentences` when the
    caller needs whole-sentence memories.
    """
    return get_counter(counter).truncate_tokens(text, budget, side="head")


def truncate_tail(text: str, budget: int, counter: Optional[TokenCounter] = None) -> Tuple[str, int]:
    """Keep the **tail** (suffix) of ``text`` within ``budget`` tokens."""
    return get_counter(counter).truncate_tokens(text, budget, side="tail")


def _selection_order(n: int, selection: str) -> List[int]:
    """Order in which sentence indices are considered for inclusion."""
    if n <= 0:
        return []
    sel = (selection or "head").lower()
    if sel in {"head", "front", "first"}:
        return list(range(n))
    if sel in {"tail", "back", "last"}:
        return list(range(n - 1, -1, -1))
    if sel in {"interleave", "alternate", "both"}:
                                                                              
        order: List[int] = []
        lo, hi = 0, n - 1
        while lo <= hi:
            order.append(lo)
            if hi != lo:
                order.append(hi)
            lo += 1
            hi -= 1
        return order
    if sel in {"even", "stride", "spread"}:
        step = max(1, n // 2) if n > 2 else 1
        idx = list(range(0, n, step))
                                                                                   
        idx += [i for i in range(n) if i not in set(idx)]
        return idx
    raise ValueError(
        f"unknown sentence selection {selection!r}; "
        "expected one of head|tail|interleave|even"
    )


def select_sentences(
    text: str,
    budget: int,
    selection: str = "head",
    counter: Optional[TokenCounter] = None,
) -> Tuple[str, int]:
    """Greedily fill a ``budget``-token memory with whole sentences.

    Candidate sentences are visited in ``selection`` order
    (``head``/``tail``/``interleave``/``even``).  A sentence is added only if
    the memory still fits within ``budget`` tokens afterwards; sentences that
    do not fit are **skipped** (the scan continues with later sentences).
    Sentence order in the output is always the original document order.

    Returns ``(memory_text, actual_tokens)``.
    """
    counter = get_counter(counter)
    if budget <= 0:
        return "", 0
    units = split_sentences(text or "")
    if not units:
        return "", 0

    chosen: set = set()
    current = ""
    current_tokens = 0
    for idx in _selection_order(len(units), selection):
        candidate = _join_units([units[j] for j in sorted(chosen | {idx})])
        n_tokens = counter.count(candidate)
        if n_tokens <= budget:
            chosen.add(idx)
            current, current_tokens = candidate, n_tokens
                                                      
    return current, counter.count(current)


def build_memory(
    text: str,
    budget: int,
    selection: str = "head",
    counter: Optional[TokenCounter] = None,
) -> Tuple[str, int]:
    """Alias used by ``run_eval`` for truncation baselines (head/tail)."""
    if selection in {"head", "front", "first"}:
        return truncate_head(text, budget, counter)
    if selection in {"tail", "back", "last"}:
        return truncate_tail(text, budget, counter)
    return select_sentences(text, budget, selection=selection, counter=counter)
