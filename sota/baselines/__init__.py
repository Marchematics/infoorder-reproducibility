
from __future__ import annotations

from . import bm25, llmlingua_compressor, summary, truncation

__all__ = ["build_memory", "registered_methods", "METHODS"]

METHODS = {
    "trunc_head": truncation.build_memory_head,
    "trunc_tail": truncation.build_memory_tail,
    "bm25": bm25.build_memory,
    "llmlingua": llmlingua_compressor.build_memory,
    "summary": summary.build_memory,
}


def registered_methods():
    return sorted(METHODS)


def build_memory(method, question, paragraphs, budget, tokenizer):
    """Dispatch to the requested memory baseline and return a plain str."""
    if not isinstance(method, str):
        raise RuntimeError(
            "method must be a string, got %r; supported: %s"
            % (type(method).__name__, registered_methods())
        )
    key = method.strip().lower()
    fn = METHODS.get(key)
    if fn is None:
        raise RuntimeError(
            "unknown memory method %r; supported in sota/baselines: %s "
            "(the harness handles 'full'/'custom')" % (method, registered_methods())
        )
    out = fn(question, paragraphs, budget, tokenizer)
    if not isinstance(out, str):
        raise RuntimeError(
            "internal error: %s returned %s, expected str" % (key, type(out).__name__)
        )
    return out
