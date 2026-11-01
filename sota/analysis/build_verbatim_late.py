import json, os, sys
BASE = str(__import__("pathlib").Path(__file__).resolve().parents[2])
sys.path.append(BASE + "/deps")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(BASE + "/models/Qwen3.5-4B")
BUDGET = int(os.environ.get("SOTA_BUDGET", "128"))
man = [json.loads(l) for l in open(BASE + "/sota/data/manifest_500.jsonl")]
dev = {r['id']: r for r in man}
out = []
for r in man:
    d = dev.get(r["id"]) or {}
    sups = [p for p in (d.get("paragraphs") or []) if p.get("is_supporting")]
    if not sups:
        sups = sorted(r["paragraphs"], key=lambda p: p["idx"])[:2]
    parts, used = [], 0
    for p in sups:
        s = "%s: %s" % (p.get("title", ""), p.get("paragraph_text", ""))
        n = len(tok(s, add_special_tokens=False)["input_ids"])
        if used + n > BUDGET:
            ids = tok(s, add_special_tokens=False)["input_ids"][: max(0, BUDGET - used)]
            s = tok.decode(ids, skip_special_tokens=True)
            n = len(ids)
        parts.append(s)
        used += n
        if used >= BUDGET - 4:
            break
    mem = "\n".join(parts)
    out.append({"id": r["id"], "hop": r["hop"], "memory": mem,
                "memory_tokens": len(tok(mem, add_special_tokens=False)["input_ids"])})
path = BASE + "/sota/data/verbatim_late_memories_b%d.jsonl" % BUDGET
with open(path, "w") as f:
    for r in out:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("verbatim-late memories", len(out), "mean tokens",
      sum(r["memory_tokens"] for r in out) / max(1, len(out)), "->", path)
