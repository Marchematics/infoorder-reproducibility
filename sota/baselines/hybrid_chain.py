import re, sys

BASE = str(__import__("pathlib").Path(__file__).resolve().parents[2])
for p in (BASE + "/sota/infoorder", BASE + "/sota/baselines", BASE + "/sota/vendor", BASE + "/deps"):
    if p not in sys.path:
        sys.path.append(p)

import structural_writer as sw

_PARA = re.compile(r"^Passage\s+(\d+)\s*\(([^)]*)\)\s*:\s*(.*)$", re.S)


def _to_dicts(paragraphs):
    out = []
    for i, p in enumerate(paragraphs):
        if isinstance(p, dict):
            out.append(p)
            continue
        m = _PARA.match(p.strip())
        if m:
            out.append({"idx": int(m.group(1)), "title": m.group(2), "paragraph_text": m.group(3)})
        else:
            out.append({"idx": i, "title": "", "paragraph_text": p})
    return out


def build_memory(question, paragraphs, budget, tokenizer, **kw):
    return sw.build_memory_hybrid(question, _to_dicts(paragraphs), budget, tokenizer)
