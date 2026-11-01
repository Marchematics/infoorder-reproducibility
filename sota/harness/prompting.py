
from __future__ import annotations

import os
import re
import sys
from typing import Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import budget              

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = (
    "You answer questions using ONLY the context provided. "
    "Reply with the shortest possible answer phrase: a few words at most, "
    "no explanation, no full sentence, no quotation marks, no citations. "
    "If the context does not contain enough information to determine the "
    "answer, reply with exactly UNANSWERABLE."
)

USER_TEMPLATE = """Context:
{context}

Question: {question}

Answer:"""

UNANSWERABLE = "UNANSWERABLE"

                                                                            
                                                                              
                                                                            
SUMMARY_PROMPT_VERSION = "v1"

SUMMARY_SYSTEM_PROMPT = (
    "You compress evidence into a compact plain-text memory that another model "
    "will later use to answer a question. Keep only information relevant to the "
    "question; copy entity names, dates and numbers verbatim; do not answer the "
    "question; do not repeat the question; do not add commentary. "
    "Output plain text only."
)

SUMMARY_USER_TEMPLATE = """Question: {question}

Evidence:
{context}

Write the compact memory (at most {budget} tokens):"""


def build_messages(question: str, context: str) -> List[Dict[str, str]]:
    """Chat messages for one question + context string."""
    question = (question or "").strip()
    context = (context or "").strip()
    if not context:
        context = "(empty)"
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": USER_TEMPLATE.format(context=context, question=question),
        },
    ]


def _tokenizer(tokenizer=None):
    if tokenizer is not None:
        return tokenizer
    return budget.load_tokenizer()


def _len_ids(ids) -> int:
    """Length of whatever ``apply_chat_template(tokenize=True)`` returned.

    Qwen3.5 is a VL model: with ``tokenize=True`` its template returns a
    BatchEncoding-like dict (``input_ids`` + ``attention_mask``), and
    ``len(dict)`` would silently be 2.  Handle dict / nested-list / flat-list.
    """
    if ids is None:
        return 0
    if isinstance(ids, dict):
        ids = ids.get("input_ids", [])
    if hasattr(ids, "get") and not isinstance(ids, (list, tuple)):
        ids = ids.get("input_ids") or ids.get("input_features") or []
    if hasattr(ids, "tolist"):
        ids = ids.tolist()
    if isinstance(ids, (list, tuple)) and ids and isinstance(ids[0], (list, tuple)):
        ids = ids[0]
    try:
        return len(ids)
    except TypeError:                                
        return 0


def apply_template(
    messages: Sequence[Dict[str, str]],
    tokenizer=None,
    add_generation_prompt: bool = True,
) -> Tuple[str, int]:
    """Render with ``apply_chat_template``; return ``(prompt_text, n_tokens)``.

    ``enable_thinking=False`` is requested when supported (Qwen3.5 otherwise
    starts the assistant turn inside an open ``<think>`` block and burns the
    24-token generation budget on reasoning).
    """
    tokenizer = _tokenizer(tokenizer)
    kwargs = dict(tokenize=False, add_generation_prompt=add_generation_prompt)
    try:
        text = tokenizer.apply_chat_template(messages, enable_thinking=False, **kwargs)
        use_thinking_kw = True
    except (TypeError, ValueError, KeyError):
        text = tokenizer.apply_chat_template(messages, **kwargs)
        use_thinking_kw = False

    ids_kwargs = dict(tokenize=True, add_generation_prompt=add_generation_prompt)
    try:
        if use_thinking_kw:
            ids = tokenizer.apply_chat_template(
                messages, enable_thinking=False, **ids_kwargs
            )
        else:
            ids = tokenizer.apply_chat_template(messages, **ids_kwargs)
        n_tokens = _len_ids(ids)
    except (TypeError, ValueError, KeyError):
        n_tokens = 0
    if n_tokens <= 0:                                             
        n_tokens = len(tokenizer(text, add_special_tokens=False)["input_ids"])
    return text, n_tokens


def build_full(
    question: str,
    paragraphs: Sequence[str],
    tokenizer=None,
    add_generation_prompt: bool = True,
) -> Tuple[List[Dict[str, str]], int]:
    """``full-context`` prompt: paragraphs are joined verbatim."""
    context = "\n\n".join(str(p).strip() for p in paragraphs if str(p).strip())
    messages = build_messages(question, context)
    _, n_tokens = apply_template(messages, tokenizer, add_generation_prompt)
    return messages, n_tokens


def build_memory(
    question: str,
    memory_text: str,
    tokenizer=None,
    add_generation_prompt: bool = True,
) -> Tuple[List[Dict[str, str]], int]:
    """``memory`` prompt: ``memory_text`` is the compressed plain-text memory."""
    messages = build_messages(question, memory_text)
    _, n_tokens = apply_template(messages, tokenizer, add_generation_prompt)
    return messages, n_tokens


def build_summary(
    question: str,
    paragraphs: Sequence[str],
    budget_tokens: int,
    tokenizer=None,
    add_generation_prompt: bool = True,
) -> Tuple[List[Dict[str, str]], str, int]:
    """Pass-1 prompt: compress the evidence into a <=``budget_tokens`` memory.

    Returns ``(messages, prompt_text, input_token_count)``.  The gold answer is
    never part of this prompt (callers pass only question + evidence).
    """
    context = "\n\n".join(str(p).strip() for p in paragraphs if str(p).strip())
    messages = [
        {"role": "system", "content": SUMMARY_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": SUMMARY_USER_TEMPLATE.format(
                question=(question or "").strip(),
                context=context,
                budget=int(budget_tokens),
            ),
        },
    ]
    text, n_tokens = apply_template(messages, tokenizer, add_generation_prompt)
    return messages, text, n_tokens


def clean_summary(raw: str) -> str:
    """Post-process a raw summary generation: strip think blocks / role tags /
    a leading ``Summary:``-style label, keep line structure."""
    if raw is None:
        return ""
    text = _strip_think(str(raw))
    for token in ("<|im_end|>", "<|endoftext|>", "<|im_start|>", "<|end|>"):
        text = text.replace(token, "\n")
    lines = []
    for ln in text.splitlines():
        ln = ln.strip()
        ln = re.sub(r"(?i)^(compact |short |brief )?(memory|summary|context)\s*[:\-]\s*", "", ln)
        if ln:
            lines.append(ln)
    out = "\n".join(lines).strip()
    out = re.sub(r"(?i)^(compact |short |brief )?(memory|summary)\s*[:\-]\s*", "", out).strip()
    return out


                                                                            
                        
                                                                            
def _strip_think(text: str) -> str:
    """Remove ``<think>...</think>`` blocks (incl. an unclosed leading block)."""
    t = re.sub(r"(?is)<think\b.*?</think\s*>", " ", text)
    t = re.sub(r"(?is)<think\b.*$", " ", t)                                    
    t = re.sub(r"(?is)</think\s*>", " ", t)
    return t


def parse_answer_v1(raw: str) -> str:
    """Extract the short answer phrase from a raw generation.

    Steps: drop think blocks / chat sentinels, take the first non-empty line,
    strip an ``Answer:`` prefix and markdown/quotes, cut to the first sentence
    (or 20 words) on rambling outputs, and map abstentions to ``UNANSWERABLE``.
    """
    if raw is None:
        return UNANSWERABLE
    text = _strip_think(str(raw))
    for token in ("<|im_end|>", "<|endoftext|>", "<|im_start|>", "<|end|>"):
        text = text.replace(token, "\n")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return UNANSWERABLE
    line = lines[0]
    line = re.sub(r"(?i)^\s*(final\s+)?answer\s*[:\-]\s*", "", line)
    line = line.strip(" \t\"'`*_")
    if not line:
        return UNANSWERABLE
                                                            
    m = re.match(r"^(.+?[.!?])(\s|$)", line)
    if m:
        line = m.group(1).strip()
    words = line.split()
    if len(words) > 20:
        line = " ".join(words[:20])
    line = line.strip(" \t\"'`*_.,;:")
    if re.search(r"(?i)\bunanswerable\b", line) or not line:
        return UNANSWERABLE
    return line


                                                                            
                                                                        
_ANSWER_ABBREVIATIONS = {'mr', 'mrs', 'ms', 'dr', 'prof', 'sr', 'jr', 'st',
                         'vs', 'etc', 'e.g', 'i.e', 'no', 'fig', 'al'}


def parse_answer(raw: str) -> str:
    """Parse short answers without truncating common initials and titles.

    The previous parser is retained as ``parse_answer_v1`` for exact replay of
    historical result files; uniform re-scoring is required for comparisons.
    """
    if raw is None:
        return UNANSWERABLE
    text = _strip_think(str(raw))
    for token in ("<|im_end|>", "<|endoftext|>", "<|im_start|>", "<|end|>"):
        text = text.replace(token, "\n")
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if not lines:
        return UNANSWERABLE
    line = re.sub(r"(?i)^\s*(final\s+)?answer\s*[:\-]\s*", "", lines[0])
    line = line.strip(" \t\"'`*_")
    if not line:
        return UNANSWERABLE
    for match in re.finditer(r"[.!?](?=\s|$)", line):
        before = line[:match.end()]
        if match.group() == "." and match.end() < len(line):
            token = before.rsplit(None, 1)[-1].strip("\"'([{")
            if (token[:-1].lower() in _ANSWER_ABBREVIATIONS
                    or re.fullmatch(r"(?:[A-Za-z]\.){2,}", token)
                    or re.fullmatch(r"[A-Z]\.", token)):
                continue
        line = before.strip()
        break
    words = line.split()
    if len(words) > 20:
        line = " ".join(words[:20])
    line = line.strip(" \t\"'`*_.,;:")
    if re.search(r"(?i)\bunanswerable\b", line) or not line:
        return UNANSWERABLE
    return line
