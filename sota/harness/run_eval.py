"""Evaluate compact memories with a frozen reader."""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from typing import Dict, List, Optional, Sequence

                                                                            
                                                                  
                                                                            
                                                                               
                                                                                
                                                             
                                                                        
                                                
_HERE = os.path.dirname(os.path.abspath(__file__))
_SOTA_ROOT = os.path.dirname(_HERE)
_REPO_ROOT = os.path.dirname(_SOTA_ROOT)
for _p in (
    os.path.join(_REPO_ROOT, "sota", "vendor"),
    os.path.join(_REPO_ROOT, "deps"),
    os.path.join(_REPO_ROOT, "deps_fla"),
):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.append(_p)
try:
    import accelerate                                                 
    HAS_ACCELERATE = True
except Exception:                                             
    HAS_ACCELERATE = False

if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import budget              
import data              
import memory              
import paths              
import prompting              
import scoring              

RECORD_FIELDS = (
    "id",
    "hop",
    "method",
    "budget",
    "n_paragraphs",
    "full_token_count",
    "supporting_tokens",
    "gold_token_count",
    "memory_tokens",
    "input_tokens",
    "compression_ratio",
    "latency_s",
    "raw_answer",
    "parsed_answer",
    "em",
    "f1",
    "answer_in_context",
)

DEFAULT_MAX_NEW_TOKENS = 24
DEFAULT_MIN_FREE_GB = 10.0


                                                                            
            
                                                                            
def read_done_ids(path: str) -> set:
    done = set()
    if not os.path.exists(path):
        return done
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["id"])
            except (json.JSONDecodeError, KeyError):
                continue
    return done


def read_records(path: str) -> List[Dict]:
    rows: List[Dict] = []
    if not os.path.exists(path):
        return rows
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return rows


def default_out_path(method: str, budget_tokens: Optional[int], context: str) -> str:
    tag = "full" if method == "full" or budget_tokens is None else f"b{budget_tokens}"
    return os.path.join(paths.RESULTS_DIR, f"musique500_{method}_{tag}_{context}.jsonl")


def _baseline_error(method: str, row: Dict, exc: BaseException) -> int:
    """Print a clean, actionable error instead of a traceback; exit code 2."""
    if os.environ.get("INFOORDER_DEBUG"):
        raise exc
    print(
        f"[error] method {method!r} failed on {row.get('id')}: "
        f"{exc.__class__.__name__}: {exc}",
        file=sys.stderr,
    )
    print(
        "[error] the baseline contract is build_memory(question, paragraphs, budget, "
        "tokenizer) -> str in sota/baselines/; see sota/harness/README.md and "
        "--list-methods",
        file=sys.stderr,
    )
    return 2


                                                                            
       
                                                                            
def _pick_model_classes():
    import transformers

    names = [
        "AutoModelForImageTextToText",                                               
        "AutoModelForCausalLM",
        "AutoModel",
    ]
    return [(n, getattr(transformers, n)) for n in names if hasattr(transformers, n)]


def load_model(model_dir: str, device: str, dtype_name: str = "bfloat16"):
    """Load the causal LM/LVLM on ``device`` with a fallback chain of classes.

    With accelerate importable we pass ``device_map=device``; otherwise we load
    on CPU and call ``model.to(device)`` (no accelerate dependency).
    """
    import torch
    import transformers

    dtype = {
        "bfloat16": torch.bfloat16,
        "float16": torch.float16,
        "float32": torch.float32,
    }[dtype_name]
    print(
        f"[model] {model_dir}  device={device}  dtype={dtype_name}  "
        f"accelerate={'yes' if HAS_ACCELERATE else 'no'}  "
        f"transformers={transformers.__version__} torch={torch.__version__}"
    )
    errors: List[str] = []
    for name, cls in _pick_model_classes():
        strategies = [{"device_map": device, "low_cpu_mem_usage": True}]
        if not HAS_ACCELERATE:
            strategies = [{"low_cpu_mem_usage": False}]
        else:
            strategies.append({"low_cpu_mem_usage": False})                       
        for kwargs in strategies:
            try:
                call_kwargs = dict(kwargs, attn_implementation="sdpa")
                try:
                    model = cls.from_pretrained(model_dir, dtype=dtype, **call_kwargs)
                except (TypeError, ValueError):
                    model = cls.from_pretrained(model_dir, torch_dtype=dtype, **call_kwargs)
                if "device_map" not in kwargs:
                    model = model.to(device)
                model.eval()
                print(
                    f"[model] loaded as {name}: {type(model).__name__} "
                    f"(device_map={'yes' if 'device_map' in kwargs else 'no'})"
                )
                return model
            except Exception as exc:                                       
                errors.append(f"{name}/{list(kwargs)}: {exc.__class__.__name__}: {exc}")
    raise RuntimeError(
        "could not load model with any transformers auto class:\n  "
        + "\n  ".join(errors)
    )


def check_gpu_floor(device: str, min_free_gb: float) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available; pass --dry-run to validate without a GPU"
        )
    idx = torch.device(device).index or 0
    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
        name = torch.cuda.get_device_name(idx)
    except (RuntimeError, IndexError):
                                                                                        
        idx = 0
        free_bytes, total_bytes = torch.cuda.mem_get_info(0)
        name = torch.cuda.get_device_name(0)
    free_gb = free_bytes / (1024 ** 3)
    used_gb = (total_bytes - free_bytes) / (1024 ** 3)
    print(
        f"[gpu] {device} (cuda:{idx} = {name}): {used_gb:.1f} GiB used / "
        f"{total_bytes / 1024**3:.1f} GiB total, {free_gb:.1f} GiB free"
    )
    if free_gb < min_free_gb:
        raise RuntimeError(
            f"refusing to use {device}: only {free_gb:.1f} GiB free, "
            f"--min-free-gb={min_free_gb}. Pick a free card (CUDA_VISIBLE_DEVICES) "
            "or pass --force if you are sure."
        )


def generate_batch(model, tokenizer, prompts: Sequence[str], device: str,
                   max_new_tokens: int) -> List[str]:
    """Greedy batch generation; returns raw decoded completions."""
    import torch

    tokenizer.padding_side = "left"
    enc = tokenizer(list(prompts), return_tensors="pt", padding=True)
    enc = {k: v.to(device) for k, v in enc.items()}
    prompt_len = enc["input_ids"].shape[1]
    pad_id = tokenizer.pad_token_id
    eos_id = tokenizer.eos_token_id
    gen_kwargs = dict(
        do_sample=False,
        num_beams=1,
        max_new_tokens=max_new_tokens,
        pad_token_id=pad_id if pad_id is not None else eos_id,
    )
    with torch.inference_mode():
        out = model.generate(**enc, **gen_kwargs)
    seq = getattr(out, "sequences", out)
    if seq.shape[1] > prompt_len:
        seq = seq[:, prompt_len:]
    return [tokenizer.decode(row, skip_special_tokens=True) for row in seq]


                                                                            
      
                                                                            
def prepare_item(row: Dict, context: str, counter: budget.TokenCounter) -> Dict:
    """Question-independent part of a work item (no model, no memory)."""
    paragraphs = data.assemble_context(row, context)
    question = data.question_of(row)
    full_text = data.context_text(paragraphs)
    return {
        "row": row,
        "paragraphs": paragraphs,
        "question": question,
        "full_text": full_text,
        "n_paragraphs": len(paragraphs),
        "full_token_count": counter.count(full_text),
        "supporting_tokens": data.supporting_tokens(row, counter),
    }


def attach_memory(
    item: Dict,
    method: str,
    budget_tokens: Optional[int],
    counter: budget.TokenCounter,
    tokenizer,
    custom_module: Optional[str] = None,
    memory_text_override: Optional[str] = None,
) -> Dict:
    """Add the memory string + the pass-2 prompt to a work item."""
    t0 = time.perf_counter()
    if memory_text_override is not None:
        mem_text = memory_text_override
    else:
        mem_text = memory.memory_for(
            method, item["question"], item["paragraphs"], budget_tokens,
            tokenizer=tokenizer, counter=counter, custom_module=custom_module,
        )
    mem_tokens = counter.count(mem_text)
    if method == "full":                                                         
        item["full_token_count"] = mem_tokens
    messages = prompting.build_messages(item["question"], mem_text)
    prompt_text, input_tokens = prompting.apply_template(messages, tokenizer)
    item.update(
        memory_text=mem_text,
        memory_tokens=mem_tokens,
        messages=messages,
        prompt_text=prompt_text,
        input_tokens=input_tokens,
        build_s=time.perf_counter() - t0,
    )
    return item


def build_item(row: Dict, method: str, budget_tokens: Optional[int], context: str,
               counter: budget.TokenCounter, tokenizer, custom_module: Optional[str]):
    """prepare + attach in one call (used by dry-run and by non-summary paths)."""
    item = prepare_item(row, context, counter)
    return attach_memory(
        item, method, budget_tokens, counter, tokenizer, custom_module
    )


                                                                            
                                                                          
                                                                            
SUMMARY_METHOD = "summary"


def summary_placeholder(item: Dict, budget_tokens: int,
                        counter: budget.TokenCounter) -> str:
    """CPU-only stand-in used by ``--dry-run`` (no model available).

    Approximates "a summary of at most ``budget_tokens``" with whole-sentence
    head selection over the evidence, so the prompt/memory path can still be
    validated without a GPU.  Clearly reported as a placeholder in dry-run.
    """
    return budget.select_sentences(
        item["full_text"], budget_tokens, selection="head", counter=counter
    )[0]


def summary_max_new_tokens(budget_tokens: int, cap: int = 1024) -> int:
    return max(32, min(int(budget_tokens), cap))


def generate_summaries(model, tokenizer, items: List[Dict], budget_tokens: int,
                       device: str, counter: budget.TokenCounter):
    """Pass 1: ask the reader model to compress each question's evidence.

    Returns ``(memories, latency_per_item, input_tokens)`` where every memory is
    token-truncated to ``budget_tokens`` so that it really is the memory budget.
    """
    prompts: List[str] = []
    sum_input_tokens: List[int] = []
    for it in items:
        _msgs, text, n_tokens = prompting.build_summary(
            it["question"], it["paragraphs"], budget_tokens, tokenizer
        )
        prompts.append(text)
        sum_input_tokens.append(n_tokens)
    t0 = time.perf_counter()
    raws = generate_batch(
        model, tokenizer, prompts, device, summary_max_new_tokens(budget_tokens)
    )
    latency = (time.perf_counter() - t0) / max(1, len(items))

    memories: List[str] = []
    for raw in raws:
        text = prompting.clean_summary(raw)
        n_tokens = counter.count(text)
        if n_tokens > budget_tokens:                                     
            text = budget.truncate_head(text, budget_tokens, counter)[0]
        memories.append(text)
    return memories, latency, sum_input_tokens


def score_item(item: Dict, raw_answer: str, latency_s: float) -> Dict:
    row = item["row"]
    golds = data.golds_of(row)
    parsed = prompting.parse_answer(raw_answer)
    scored = scoring.score_answer(parsed, golds)
    aic = scoring.answer_in_context(
        row.get("answer") or "", item["memory_text"], row.get("answer_aliases") or []
    )
    full_tokens = item["full_token_count"]
    ratio = (item["memory_tokens"] / full_tokens) if full_tokens else 0.0
    return {
        "id": row["id"],
        "hop": row.get("hop"),
        "method": item.get("method"),
        "budget": item.get("budget"),
        "n_paragraphs": item["n_paragraphs"],
        "full_token_count": full_tokens,
        "supporting_tokens": item["supporting_tokens"],
        "gold_token_count": item["supporting_tokens"],
        "memory_tokens": item["memory_tokens"],
        "input_tokens": item["input_tokens"],
        "compression_ratio": round(ratio, 6),
        "latency_s": round(float(latency_s), 4),
        "raw_answer": raw_answer,
        "parsed_answer": parsed,
        "em": int(scored["em"]),
        "f1": float(scored["f1"]),
        "answer_in_context": bool(aic),
    }


def summarize(records: List[Dict], method: str, context: str,
              budget_tokens: Optional[int]) -> Dict:
    agg = scoring.aggregate(records)
    mem = [r.get("memory_tokens") for r in records if r.get("memory_tokens") is not None]
    inp = [r.get("input_tokens") for r in records if r.get("input_tokens") is not None]
    lat = [r.get("latency_s") for r in records if r.get("latency_s") is not None]
    by_hop: Dict[str, Dict[str, float]] = {}
    for hop in ("2hop", "3hop", "4hop"):
        sub = [r for r in records if r.get("hop") == hop]
        if sub:
            by_hop[hop] = {
                "n": len(sub),
                "em": sum(float(r.get("em", 0)) for r in sub) / len(sub),
                "f1": sum(float(r.get("f1", 0.0)) for r in sub) / len(sub),
            }
    full = [r.get("full_token_count") for r in records if r.get("full_token_count")]
    sup = [r.get("supporting_tokens") for r in records if r.get("supporting_tokens") is not None]
    ratio = [r.get("compression_ratio") for r in records if r.get("compression_ratio") is not None]
    npara = [r.get("n_paragraphs") for r in records if r.get("n_paragraphs") is not None]
    return {
        "method": method,
        "context": context,
        "budget": budget_tokens,
        "n": agg["n"],
        "em": agg["em"],
        "f1": agg["f1"],
        "answer_in_context": agg["aic"],
        "abstention_rate": agg["abstention"],
        "mean_memory_tokens": (sum(mem) / len(mem)) if mem else 0.0,
        "max_memory_tokens": max(mem) if mem else 0,
        "mean_input_tokens": (sum(inp) / len(inp)) if inp else 0.0,
        "mean_full_token_count": (sum(full) / len(full)) if full else 0.0,
        "mean_supporting_tokens": (sum(sup) / len(sup)) if sup else 0.0,
        "mean_compression_ratio": (sum(ratio) / len(ratio)) if ratio else 0.0,
        "mean_n_paragraphs": (sum(npara) / len(npara)) if npara else 0.0,
        "mean_latency_s": (statistics.mean(lat)) if lat else 0.0,
        "by_hop": by_hop,
    }


def print_summary(summary: Dict) -> None:
    print("\n=== summary ===")
    print(
        f"method={summary['method']} context={summary['context']} "
        f"budget={summary['budget']} n={summary['n']}"
    )
    print(
        f"EM={summary['em']:.4f}  F1={summary['f1']:.4f}  "
        f"AIC={summary['answer_in_context']:.4f}  "
        f"abstain={summary['abstention_rate']:.4f}"
    )
    print(
        f"memory_tokens mean={summary['mean_memory_tokens']:.1f} "
        f"max={summary['max_memory_tokens']}  "
        f"input_tokens mean={summary['mean_input_tokens']:.1f}  "
        f"latency mean={summary['mean_latency_s']:.3f}s"
    )
    nan = float("nan")
    print(
        f"full_token_count mean={summary.get('mean_full_token_count', nan):.1f}  "
        f"supporting_tokens mean={summary.get('mean_supporting_tokens', nan):.1f}  "
        f"compression_ratio mean={summary.get('mean_compression_ratio', nan):.4f}  "
        f"n_paragraphs mean={summary.get('mean_n_paragraphs', nan):.1f}"
    )
    for hop, stats in summary["by_hop"].items():
        print(
            f"  {hop:4s} n={stats['n']:3d}  EM={stats['em']:.4f}  F1={stats['f1']:.4f}"
        )


                                                                            
     
                                                                            
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Unified MuSiQue-500 multi-hop QA eval harness",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    ap.add_argument("--method", required=False, choices=list(memory.ALL_METHODS),
                    default="full", help="memory construction method")
    ap.add_argument("--budget", type=int, default=512,
                    help="memory token budget (ignored by --method full)")
    ap.add_argument("--limit", type=int, default=0,
                    help="only evaluate the first N manifest rows (0 = all)")
    ap.add_argument("--out", default=None, help="output jsonl (default under sota/results/)")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--device", default="cuda:0",
                    help="torch device; combine with --gpu / CUDA_VISIBLE_DEVICES")
    ap.add_argument("--gpu", default=None,
                    help="physical GPU index; exported as CUDA_VISIBLE_DEVICES before torch import")
    ap.add_argument("--min-free-gb", type=float, default=DEFAULT_MIN_FREE_GB,
                    help="refuse to start if the target card has less free memory")
    ap.add_argument("--force", action="store_true", help="skip the free-memory guard")
    ap.add_argument("--model", default=paths.MODEL_DIR)
    ap.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--context", default="full", choices=list(data.CONTEXT_MODES),
                    help="full = documents + later_evidence; stream = documents only")
    ap.add_argument("--manifest", default=paths.DEFAULT_MANIFEST)
    ap.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    ap.add_argument("--max-input-tokens", type=int, default=0,
                    help="skip rows whose prompt exceeds this (0 = no check)")
    ap.add_argument("--no-resume", action="store_true",
                    help="ignore existing ids in --out and recompute them")
    ap.add_argument("--dry-run", action="store_true",
                    help="no model/GPU: build memories+prompts and report token stats")
    ap.add_argument("--custom-module", default=None,
                    help="python file exposing build_memory(...) for --method custom")
    ap.add_argument("--summary-json", default=None, help="write summary dict here")
    ap.add_argument("--dump-memory", action="store_true",
                    help="also store the memory string in each record (debug/summary audit)")
    ap.add_argument("--list-methods", action="store_true",
                    help="print method availability and exit")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    args = parse_args(argv)

    if args.list_methods:
        found = memory.available_baselines()
        for m in memory.ALL_METHODS:
            if m == SUMMARY_METHOD:
                where = (
                    "harness-native (two-pass: summarise evidence -> answer from "
                    "the summary; needs GPU, truncation placeholder in --dry-run)"
                )
            elif m in memory.NATIVE_METHODS:
                where = "harness-native"
            else:
                where = found.get(m, "NOT IMPLEMENTED")
            print(f"{m:12s} {where}")
        return 0

    method = args.method
    if method != "full" and args.budget is None:
        print("[error] --budget is required for methods other than 'full'", file=sys.stderr)
        return 2
    if method != "full" and args.budget <= 0:
        print(f"[error] --budget must be > 0, got {args.budget}", file=sys.stderr)
        return 2

    if args.gpu is not None and "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    rows = data.load_manifest(args.manifest)
    if args.limit and args.limit > 0:
        rows = rows[: args.limit]
    out_path = args.out or default_out_path(method, args.budget, args.context)
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)

    done = set() if args.no_resume else read_done_ids(out_path)
    todo = [r for r in rows if r["id"] not in done]
    print(
        f"[setup] method={method} budget={args.budget} context={args.context} "
        f"rows={len(rows)} already_done={len(rows) - len(todo)} todo={len(todo)}"
    )
    print(f"[setup] out={out_path}")

    tokenizer = budget.load_tokenizer()
                                                                              
    if getattr(args, "model", None) and os.path.isdir(str(args.model)):
        try:
            from transformers import AutoTokenizer as _AT
            tokenizer = _AT.from_pretrained(str(args.model), trust_remote_code=True)
        except Exception as exc:                    
            print("tokenizer override failed:", exc)
    counter = budget.TokenCounter(tokenizer)

                                                                 
    if args.dry_run:
        if method == SUMMARY_METHOD:
            print(
                "[dry-run] summary: no model available -> using whole-sentence "
                "truncation as a placeholder memory (real run uses two-pass summary)"
            )
        stats: List[Dict] = []
        for i, row in enumerate(todo):
            try:
                item = prepare_item(row, args.context, counter)
                override = (
                    summary_placeholder(item, args.budget, counter)
                    if method == SUMMARY_METHOD
                    else None
                )
                attach_memory(
                    item, method, None if method == "full" else args.budget,
                    counter, tokenizer, args.custom_module,
                    memory_text_override=override,
                )
            except (RuntimeError, ImportError, NotImplementedError, ValueError, TypeError) as exc:
                return _baseline_error(method, row, exc)
            mem_ok = method == "full" or item["memory_tokens"] <= (args.budget or 0)
            stats.append(
                {
                    "id": row["id"],
                    "hop": row.get("hop"),
                    "memory_tokens": item["memory_tokens"],
                    "input_tokens": item["input_tokens"],
                    "n_paragraphs": item["n_paragraphs"],
                    "full_token_count": item["full_token_count"],
                    "supporting_tokens": item["supporting_tokens"],
                    "memory_within_budget": mem_ok,
                }
            )
            if (i + 1) % 25 == 0 or i + 1 == len(todo):
                print(f"[dry-run] {i + 1}/{len(todo)}")
        if not stats:
            print("[dry-run] nothing to do")
            return 0
        mem = [s["memory_tokens"] for s in stats]
        inp = [s["input_tokens"] for s in stats]
        within = sum(1 for s in stats if s["memory_within_budget"])
        print(
            f"[dry-run] n={len(stats)} memory_tokens mean={statistics.mean(mem):.1f} "
            f"min={min(mem)} max={max(mem)} within_budget={within}/{len(stats)}"
        )
        print(
            f"[dry-run] input_tokens mean={statistics.mean(inp):.1f} "
            f"min={min(inp)} max={max(inp)}"
        )
        fulls = [s["full_token_count"] for s in stats]
        sups = [s["supporting_tokens"] for s in stats]
        nparas = [s["n_paragraphs"] for s in stats]
        print(
            f"[dry-run] n_paragraphs mean={statistics.mean(nparas):.1f} "
            f"full_token_count mean={statistics.mean(fulls):.1f} "
            f"supporting_tokens mean={statistics.mean(sups):.1f} "
            f"compression={statistics.mean(mem) / statistics.mean(fulls):.4f}"
        )
        print("[dry-run] no model loaded, no records written")
        return 0

                                                
    if not args.force:
        check_gpu_floor(args.device, args.min_free_gb)
    model = load_model(args.model, args.device, args.dtype)
    print(f"[model] ready; max_new_tokens={args.max_new_tokens}")

    t_start = time.time()
    written = 0
    em_so_far: List[float] = []
    with open(out_path, "a", encoding="utf-8") as fout:
        for start in range(0, len(todo), max(1, args.batch_size)):
            chunk = todo[start : start + args.batch_size]
            items: List[Dict] = []
            for row in chunk:
                try:
                    item = prepare_item(row, args.context, counter)
                    if method != SUMMARY_METHOD:
                        attach_memory(
                            item, method, None if method == "full" else args.budget,
                            counter, tokenizer, args.custom_module,
                        )
                except (RuntimeError, ImportError, NotImplementedError, ValueError, TypeError) as exc:
                    return _baseline_error(method, row, exc)
                item["method"] = method
                item["budget"] = None if method == "full" else args.budget
                items.append(item)
            if not items:
                continue

            if method == SUMMARY_METHOD:
                try:
                    memories, sum_latency, sum_input_tokens = generate_summaries(
                        model, tokenizer, items, args.budget, args.device, counter
                    )
                except (RuntimeError, ValueError) as exc:
                    return _baseline_error(method, items[0]["row"], exc)
                for it, mem_text, n_in in zip(items, memories, sum_input_tokens):
                    it["summary_latency_s"] = round(sum_latency, 4)
                    it["summary_input_tokens"] = n_in
                    attach_memory(
                        it, SUMMARY_METHOD, args.budget, counter, tokenizer,
                        memory_text_override=mem_text,
                    )
                    it["summary_tokens"] = it["memory_tokens"]

            if args.max_input_tokens:
                keep = [it for it in items if it["input_tokens"] <= args.max_input_tokens]
                keep_ids = {id(it) for it in keep}
                for it in items:
                    if id(it) not in keep_ids:
                        print(
                            f"[warn] skip {it['row']['id']}: "
                            f"input_tokens={it['input_tokens']}"
                        )
                items = keep
            if not items:
                continue

            prompts = [it["prompt_text"] for it in items]
            t0 = time.perf_counter()
            raws = generate_batch(
                model, tokenizer, prompts, args.device, args.max_new_tokens
            )
            batch_latency = time.perf_counter() - t0
            per_item_latency = batch_latency / max(1, len(items))

            for it, raw in zip(items, raws):
                rec = score_item(it, raw, per_item_latency)
                if method == SUMMARY_METHOD:
                    rec["summary_latency_s"] = it.get("summary_latency_s")
                    rec["summary_input_tokens"] = it.get("summary_input_tokens")
                    rec["summary_output_tokens"] = it.get("summary_tokens")
                if args.dump_memory:
                    rec["memory_text"] = it["memory_text"]
                fout.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fout.flush()
                written += 1
                em_so_far.append(float(rec["em"]))
            print(
                f"[run] {written}/{len(todo)}  running_em={statistics.mean(em_so_far):.4f}"
            )

    elapsed = time.time() - t_start
    records = read_records(out_path)
    summary = summarize(records, method, args.context,
                        None if method == "full" else args.budget)
    summary["elapsed_s"] = round(elapsed, 1)
    summary["out"] = out_path
    print_summary(summary)
    if args.summary_json:
        with open(args.summary_json, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2, sort_keys=True)
        print(f"[summary] written {args.summary_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
