
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys
from typing import Dict, List, Optional, Sequence

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import budget              
import data              
import paths              

HOP_ORDER = ("2hop", "3hop", "4hop")


                                                                            
         
                                                                            
def parse_hop(record_id: str) -> str:
    """Map a MuSiQue id (``2hop__460946_294723``, ``3hop2__...``) to its hop."""
    rid = str(record_id)
    for hop in HOP_ORDER:
        if rid.startswith(hop + "__") or rid.startswith(hop):
            return hop
    prefix = rid.split("__", 1)[0]
    for hop in HOP_ORDER:
        if prefix.startswith(hop):
            return hop
    raise ValueError(f"cannot infer hop from id {record_id!r}")


def sha256_file(path: str, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


def load_source(path: Optional[str] = None) -> List[Dict]:
    path = path or paths.SOURCE_JSONL
    rows: List[Dict] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def hop_counts(records: Sequence[Dict]) -> Dict[str, int]:
    counts = {hop: 0 for hop in HOP_ORDER}
    for rec in records:
        counts[parse_hop(rec["id"])] += 1
    return counts


def _quantile(values: Sequence[float], p: float) -> float:
    vals = sorted(values)
    if not vals:
        return 0.0
    k = (len(vals) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (k - lo)


def _stats(values: Sequence[float]) -> Dict[str, float]:
    vals = list(values)
    if not vals:
        return {"mean": 0.0, "median": 0.0, "p10": 0.0, "p90": 0.0, "min": 0.0, "max": 0.0}
    return {
        "mean": round(sum(vals) / len(vals), 2),
        "median": round(_quantile(vals, 0.5), 2),
        "p10": round(_quantile(vals, 0.1), 2),
        "p90": round(_quantile(vals, 0.9), 2),
        "min": min(vals),
        "max": max(vals),
    }


def token_fields(record: Dict, counter) -> Dict[str, int]:
    """Per-question token metadata (same standard as the harness, see budget.py).

    ``full_token_count``     all paragraphs (gold + distractors) formatted as
                             ``Passage {idx} ({title}): {text}``, joined by blank
                             lines -- the compression-ratio denominator.
    ``supporting_tokens``    the same for ``is_supporting == True`` paragraphs.
    """
    paragraphs = data.assemble_context(record, "full")
    return {
        "full_token_count": counter.count(data.context_text(paragraphs)),
        "supporting_tokens": data.supporting_tokens(record, counter),
    }


def build_sample(
    records: Sequence[Dict],
    seed: int = paths.SEED,
    per_hop: Optional[Dict[str, int]] = None,
    counter=None,
) -> List[Dict]:
    """Deterministic stratified sample; returns rows with hop+index appended."""
    per_hop = dict(per_hop or paths.N_PER_HOP)
    rng = random.Random(seed)

    sampled: List[Dict] = []
    for hop in HOP_ORDER:
        pool = sorted(
            (r for r in records if parse_hop(r["id"]) == hop),
            key=lambda r: r["id"],
        )
        want = int(per_hop.get(hop, 0))
        if want > len(pool):
            raise ValueError(
                f"stratum {hop} has only {len(pool)} rows, cannot sample {want}"
            )
        rng.shuffle(pool)
        picked = sorted(pool[:want], key=lambda r: r["id"])
        sampled.extend(picked)

    out: List[Dict] = []
    for idx, rec in enumerate(sampled):
        row = dict(rec)                                 
        row["hop"] = parse_hop(rec["id"])
        row["index"] = idx
        if counter is not None:
            row.update(token_fields(row, counter))
        out.append(row)
    return out


def render_jsonl(rows: Sequence[Dict]) -> str:
    return "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows
    )


def render_meta(
    rows: Sequence[Dict],
    source_path: str,
    source_sha: str,
    source_n: int,
    source_by_hop: Dict[str, int],
    out_jsonl: str,
    out_sha: str,
    out_bytes: int,
    seed: int,
    tokenizer_dir: Optional[str] = None,
    tokens_computed: bool = True,
) -> Dict:
    by_hop = {hop: 0 for hop in HOP_ORDER}
    for row in rows:
        by_hop[row["hop"]] += 1
    full = [row.get("full_token_count") for row in rows]
    sup = [row.get("supporting_tokens") for row in rows]
    has_tokens = tokens_computed and all(v is not None for v in full + sup)
    token_stats = None
    if has_tokens:
        ratios = [s / f for s, f in zip(sup, full) if f]
        token_stats = {
            "full_token_count": _stats(full),
            "supporting_tokens": _stats(sup),
            "supporting_ratio": {
                "mean": round(sum(ratios) / len(ratios), 4) if ratios else 0.0,
                "median": round(_quantile(ratios, 0.5), 4) if ratios else 0.0,
            },
            "per_hop_full_token_count": {
                hop: _stats([r["full_token_count"] for r in rows if r["hop"] == hop])
                for hop in HOP_ORDER
            },
        }
    return {
        "seed": seed,
        "counts": {"total": len(rows), "by_hop": by_hop},
        "hop_rule": {
            "2hop": "id startswith '2hop'",
            "3hop": "id startswith '3hop' (3hop1/3hop2 folded)",
            "4hop": "id startswith '4hop' (4hop1/4hop2/4hop3 folded)",
        },
        "sampling": (
            "stratified without replacement; pool per hop sorted by id, then "
            "random.Random(seed).shuffle, take first n; strata drawn in order "
            "2hop,3hop,4hop; output sorted by (hop order, id); index = 0-based "
            "manifest position"
        ),
        "fields": list(rows[0].keys()) if rows else [],
        "annotation_fields": ["hop", "index", "full_token_count", "supporting_tokens"],
        "token_fields": {
            "full_token_count": (
                "Qwen3.5-4B tokenizer tokens of ALL paragraphs sorted by idx, each "
                "rendered 'Passage {idx} ({title}): {text}', joined by blank lines "
                "(compression-ratio denominator; identical to --method full memory)"
            ),
            "supporting_tokens": (
                "same rendering restricted to is_supporting=True paragraphs "
                "(key-evidence token base)"
            ),
            "tokenizer": tokenizer_dir,
            "counting": "tokenizer.encode(text, add_special_tokens=False), len(.)",
            "computed": bool(has_tokens),
        },
        "token_stats": token_stats,
        "source": {
            "path": source_path,
            "sha256": source_sha,
            "n_records": source_n,
            "by_hop": source_by_hop,
        },
        "output": {
            "path": out_jsonl,
            "sha256": out_sha,
            "bytes": out_bytes,
            "n_records": len(rows),
        },
    }


def build_manifest(
    source_jsonl: Optional[str] = None,
    out_jsonl: Optional[str] = None,
    out_meta: Optional[str] = None,
    seed: int = paths.SEED,
    per_hop: Optional[Dict[str, int]] = None,
    compute_tokens: bool = True,
    tokenizer_dir: Optional[str] = None,
) -> Dict:
    """Build (and write) the manifest; returns the meta dict."""
    source_jsonl = source_jsonl or paths.SOURCE_JSONL
    out_jsonl = out_jsonl or paths.DEFAULT_MANIFEST
    out_meta = out_meta or paths.DEFAULT_MANIFEST_META

    if not os.path.exists(source_jsonl):
        raise FileNotFoundError(f"source corpus not found: {source_jsonl}")

    source_records = load_source(source_jsonl)

    counter = None
    if compute_tokens:
        try:
            counter = budget.TokenCounter(
                tokenizer_dir=tokenizer_dir or paths.TOKENIZER_DIR
            )
        except Exception as exc:                                                 
            print(f"[warn] token fields disabled ({exc.__class__.__name__}: {exc})")
    rows = build_sample(source_records, seed=seed, per_hop=per_hop, counter=counter)

    text = render_jsonl(rows)
    os.makedirs(os.path.dirname(os.path.abspath(out_jsonl)), exist_ok=True)
    with open(out_jsonl, "w", encoding="utf-8", newline="\n") as f:
        f.write(text)

    out_bytes = os.path.getsize(out_jsonl)
    out_sha = sha256_file(out_jsonl)
    source_sha = sha256_file(source_jsonl)
    meta = render_meta(
        rows=rows,
        source_path=source_jsonl,
        source_sha=source_sha,
        source_n=len(source_records),
        source_by_hop=hop_counts(source_records),
        out_jsonl=out_jsonl,
        out_sha=out_sha,
        out_bytes=out_bytes,
        seed=seed,
        tokenizer_dir=(tokenizer_dir or paths.TOKENIZER_DIR) if counter else None,
        tokens_computed=counter is not None,
    )
    os.makedirs(os.path.dirname(os.path.abspath(out_meta)), exist_ok=True)
    with open(out_meta, "w", encoding="utf-8", newline="\n") as f:
        f.write(json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    return meta


def check_manifest(
    source_jsonl: Optional[str] = None,
    manifest: Optional[str] = None,
    seed: int = paths.SEED,
    per_hop: Optional[Dict[str, int]] = None,
    compute_tokens: bool = True,
    tokenizer_dir: Optional[str] = None,
) -> bool:
    """Re-derive the sample in memory and compare with the file on disk."""
    source_jsonl = source_jsonl or paths.SOURCE_JSONL
    manifest = manifest or paths.DEFAULT_MANIFEST
    if not os.path.exists(manifest):
        print(f"[check] missing manifest: {manifest}")
        return False
    counter = None
    if compute_tokens:
        try:
            counter = budget.TokenCounter(
                tokenizer_dir=tokenizer_dir or paths.TOKENIZER_DIR
            )
        except Exception:
            counter = None
    rows = build_sample(load_source(source_jsonl), seed=seed, per_hop=per_hop, counter=counter)
    expected = render_jsonl(rows)
    with open(manifest, "r", encoding="utf-8") as f:
        actual = f.read()
    if actual == expected:
        print(f"[check] OK reproducible manifest sha256={sha256_file(manifest)}")
        return True
    exp, act = expected.splitlines(), actual.splitlines()
    print(f"[check] MISMATCH: expected {len(exp)} lines, found {len(act)}")
    for i, (a, b) in enumerate(zip(exp, act)):
        if a != b:
            print(f"[check] first differing line {i}")
            break
    return False


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--source", default=paths.SOURCE_JSONL)
    ap.add_argument("--out-dir", default=paths.DATA_DIR)
    ap.add_argument("--seed", type=int, default=paths.SEED)
    ap.add_argument("--check", action="store_true", help="verify instead of write")
    ap.add_argument("--no-token-fields", action="store_true",
                    help="skip full_token_count/supporting_tokens (no tokenizer needed)")
    args = ap.parse_args(argv)

    out_jsonl = os.path.join(args.out_dir, "manifest_500.jsonl")
    out_meta = os.path.join(args.out_dir, "manifest_500.meta.json")

    compute_tokens = not args.no_token_fields
    if args.check:
        ok = check_manifest(
            args.source, out_jsonl, seed=args.seed, compute_tokens=compute_tokens
        )
        return 0 if ok else 1

    meta = build_manifest(
        args.source, out_jsonl, out_meta, seed=args.seed, compute_tokens=compute_tokens
    )
    print(json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
