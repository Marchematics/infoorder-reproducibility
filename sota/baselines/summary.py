
from __future__ import annotations

__all__ = ["build_memory"]


def build_memory(question, paragraphs, budget, tokenizer):
    """Not implemented here: needs a reader/generator model.

    Parameters
    ----------
    question : str
    paragraphs : list[str]
    budget : int  # max tokens with ``tokenizer(..., add_special_tokens=False)``
    tokenizer : transformers.AutoTokenizer

    Raises
    ------
    RuntimeError
        Always.  The Lead's ``run_eval.py`` implements the summary baseline
        because summarisation needs a generator model.
    """
    raise RuntimeError(
        "summary baseline is not implemented in sota/baselines: it requires a "
        "reader/generator model and is handled by sota/harness/run_eval.py. "
        "Use method='bm25' or method='llmlingua' here, or implement summary in "
        "the harness."
    )
