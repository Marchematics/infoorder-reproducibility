
from __future__ import annotations

import json
import os
import sys
from typing import Dict, List, Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import paths              
import scoring              

CONTEXT_MODES = ("full",)

PARAGRAPH_TEMPLATE = "Passage {idx} ({title}): {text}"
PARAGRAPH_TEMPLATE_NO_TITLE = "Passage {idx}: {text}"


                                                                            
          
                                                                            
def load_manifest(path: Optional[str] = None) -> List[Dict]:
    path = path or paths.DEFAULT_MANIFEST
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"manifest not found: {path}\n"
            "build it first:  python sota/harness/manifest.py"
        )
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


                                                                            
            
                                                                            
def paragraphs_of(record: Dict) -> List[Dict]:
    """Paragraph dicts of a record, sorted by ``idx`` (ascending)."""
    paragraphs = record.get("paragraphs")
    if not paragraphs:
        raise ValueError(
            f"record {record.get('id')!r} has no 'paragraphs' field -- this harness "
            "expects official MuSiQue v1.0 dev, not the paper's streaming variant"
        )
    return sorted(paragraphs, key=lambda p: int(p.get("idx", 0)))


def format_paragraph(p: Dict) -> str:
    """``Passage {idx} ({title}): {paragraph_text}`` (title omitted when empty)."""
    idx = p.get("idx")
    title = (p.get("title") or "").strip()
    text = (p.get("paragraph_text") or p.get("paragraph") or "").strip()
    if title:
        return PARAGRAPH_TEMPLATE.format(idx=idx, title=title, text=text)
    return PARAGRAPH_TEMPLATE_NO_TITLE.format(idx=idx, text=text)


def assemble_context(record: Dict, mode: str = "full") -> List[str]:
    """Formatted passages (gold + distractors) in ``idx`` order."""
    if mode not in CONTEXT_MODES:
        raise ValueError(f"context mode must be one of {CONTEXT_MODES}, got {mode!r}")
    return [format_paragraph(p) for p in paragraphs_of(record)]


def context_text(paragraphs: Sequence[str]) -> str:
    """The full-context string (also the ``full`` memory)."""
    return "\n\n".join(p.strip() for p in paragraphs if p and p.strip())


def n_paragraphs(record: Dict) -> int:
    return len(paragraphs_of(record))


                                                                            
               
                                                                            
def _is_supporting(p: Dict) -> bool:
    """``is_supporting`` may be a bool or the strings "True"/"False"."""
    v = p.get("is_supporting", False)
    if isinstance(v, str):
        return v.strip().lower() in {"true", "1", "yes"}
    return bool(v)


def supporting_paragraphs(record: Dict) -> List[Dict]:
    return [p for p in paragraphs_of(record) if _is_supporting(p)]


def supporting_text(record: Dict) -> str:
    return context_text([format_paragraph(p) for p in supporting_paragraphs(record)])


def supporting_tokens(record: Dict, counter) -> int:
    """Total Qwen3.5 tokens of the supporting (gold) passages, same rendering."""
    return counter.count(supporting_text(record))


                                                                            
                         
                                                                            
def golds_of(record: Dict) -> List[str]:
    return scoring.gold_answers(
        record.get("answer"), record.get("answer_aliases") or []
    )


def question_of(record: Dict) -> str:
    return str(record.get("question") or "").strip()
