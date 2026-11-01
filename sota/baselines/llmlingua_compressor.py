
from __future__ import annotations

import importlib.util
import logging
import os
import re
import sys
import threading
import time

__all__ = ["build_memory", "get_compressor", "LAST_RUN", "FALLBACK_COUNT", "VARIANT"]

logger = logging.getLogger("sota.baselines.llmlingua")

_BASELINES_DIR = os.path.dirname(os.path.abspath(__file__))
_SOTA_DIR = os.path.dirname(_BASELINES_DIR)

VENDOR_DIR = os.environ.get("SOTA_VENDOR", os.path.join(_SOTA_DIR, "vendor"))
LL2_MODEL_DIR = os.environ.get(
    "LLMLINGUA_MODEL_DIR",
    os.path.join(
        _SOTA_DIR, "models", "llmlingua-2-bert-base-multilingual-cased-meetingbank"
    ),
)
GPT2_MODEL_DIR = os.environ.get(
    "LLMLINGUA_GPT2_DIR", os.path.join(_SOTA_DIR, "models", "gpt2")
)
DEVICE = os.environ.get("LLMLINGUA_DEVICE", "cpu")

VARIANT = os.environ.get("LLMLINGUA_VARIANT", "longllmlingua").strip().lower()
_LL2_ALIASES = {"llmlingua2", "llmlingua-2", "ll2"}
_VALID_VARIANTS = {"longllmlingua"} | _LL2_ALIASES

                                                                            
                                                                               
                                                                             
                                                                              
                                                                                 
if os.path.isdir(VENDOR_DIR) and VENDOR_DIR not in sys.path:
    sys.path.append(VENDOR_DIR)

_VENDOR_PKG = os.path.join(VENDOR_DIR, "llmlingua")
_VENDOR_PKG_NAME = "sota_vendor_llmlingua"

_COMPRESSOR = None
_LOCK = threading.Lock()
FALLBACK_COUNT = 0
LAST_RUN = {}

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_RATE_MARGIN = float(os.environ.get("LLMLINGUA_RATE_MARGIN", "0.95"))


def _is_longllmlingua():
    return VARIANT not in _LL2_ALIASES


def _active_model_dir():
    return GPT2_MODEL_DIR if _is_longllmlingua() else LL2_MODEL_DIR


def _encode(tokenizer, text):
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
    return tokenizer.decode(
        ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )


def _load_vendor_package():
    import importlib
    return importlib.import_module("llmlingua")


def _check_model_dir(path, label):
    if not os.path.isdir(path):
        raise FileNotFoundError(f"{label}: download model weights to {path}")


def get_compressor():
    """Load the requested compressor once (lazy, thread-safe)."""
    global _COMPRESSOR
    if _COMPRESSOR is not None:
        return _COMPRESSOR
    with _LOCK:
        if _COMPRESSOR is not None:
            return _COMPRESSOR
        if VARIANT not in _VALID_VARIANTS:
            raise RuntimeError(
                "unknown LLMLINGUA_VARIANT=%r; expected 'longllmlingua' or "
                "'llmlingua2'" % VARIANT
            )
        n_threads = os.environ.get("SOTA_TORCH_THREADS")
        if n_threads:
            import torch

            torch.set_num_threads(int(n_threads))
                                                                              
                                                                              
                                                           
        try:
            import accelerate              
            from transformers.utils import import_utils as _tf_import_utils

            _accel_check = getattr(_tf_import_utils, "is_accelerate_available", None)
            if _accel_check is not None and hasattr(_accel_check, "cache_clear"):
                _accel_check.cache_clear()
        except Exception:                                                       
            pass

        if _is_longllmlingua():
            model_dir, label, use_llmlingua2 = GPT2_MODEL_DIR, "gpt2 (LLMLingua-1)", False
        else:
            model_dir, label, use_llmlingua2 = LL2_MODEL_DIR, "LLMLingua-2", True
        _check_model_dir(model_dir, label)
        try:
            pkg = _load_vendor_package()
            t0 = time.time()
            _COMPRESSOR = pkg.PromptCompressor(
                model_name=model_dir,
                device_map=DEVICE,
                use_llmlingua2=use_llmlingua2,
            )
        except Exception as exc:
            raise RuntimeError(
                "failed to load %s compressor from %s on device %r (%s: %s)"
                % (label, model_dir, DEVICE, type(exc).__name__, exc)
            ) from exc
        _COMPRESSOR._load_seconds = time.time() - t0
        _COMPRESSOR._variant = VARIANT
        logger.info(
            "%s compressor loaded from %s on %s in %.2fs",
            VARIANT,
            model_dir,
            DEVICE,
            _COMPRESSOR._load_seconds,
        )
        return _COMPRESSOR


def _fill_words(text, room, tokenizer):
    """Longest word-prefix of ``text`` that fits in ``room`` tokens."""
    if room <= 0 or not text:
        return ""
    words = text.split()
    lo, hi, best = 1, len(words), 0
    while lo <= hi:
        mid = (lo + hi) // 2
        cand = " ".join(words[:mid])
        if len(_encode(tokenizer, cand)) <= room:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return " ".join(words[:best])


def _truncate_at_sentence_boundary(text, budget, tokenizer):
    """Sentence-boundary truncation, topped up at word boundaries."""
    sentences = [s for s in _SENT_SPLIT.split(text) if s and s.strip()]
    if not sentences:
        return "", 0
    total = len(_encode(tokenizer, text))
    if total <= budget:
        return text, 0

    lo, hi, best_k = 1, len(sentences), 0
    while lo <= hi:
        mid = (lo + hi) // 2
        cand = " ".join(sentences[:mid])
        if len(_encode(tokenizer, cand)) <= budget:
            best_k = mid
            lo = mid + 1
        else:
            hi = mid - 1
    dropped = len(sentences) - best_k

    if best_k == 0:
        return _fill_words(sentences[0], budget, tokenizer), len(sentences)

    base = " ".join(sentences[:best_k])
    base_tok = len(_encode(tokenizer, base))
    if base_tok < 0.85 * budget:
        remainder = " ".join(sentences[best_k:])
        room = budget - base_tok - 1
        topup = _fill_words(remainder, room, tokenizer)
        if topup:
            base = base + " " + topup
            if len(_encode(tokenizer, base)) > budget:
                base = _fill_words(base, budget, tokenizer)
    return base, dropped


def _compress_once(compressor, text, question, rate):
    if _is_longllmlingua():
        return compressor.compress_prompt(
            text,
            question=question,
            rate=max(1e-4, min(0.999, rate)),
            condition_in_question="after_condition",
            condition_compare=True,
            rank_method="longllmlingua",
            concate_question=False,
        )
    return compressor.compress_prompt(
        [text],
        rate=max(1e-4, min(0.999, rate)),
        use_context_level_filter=False,
        use_token_level_filter=True,
    )


def build_memory(question, paragraphs, budget, tokenizer):
    global LAST_RUN, FALLBACK_COUNT
    budget = max(0, int(budget))
    text = "\n\n".join(
        p.strip() for p in paragraphs if isinstance(p, str) and p.strip()
    )
    if budget == 0 or not text:
        return ""

    n_in = len(_encode(tokenizer, text))
    if n_in <= budget:
        LAST_RUN = {
            "variant": VARIANT,
            "input_tokens": n_in,
            "output_tokens": n_in,
            "budget": budget,
            "compressed": False,
            "rate": 1.0,
            "fallback": False,
            "dropped_sentences": 0,
            "utilization": round(n_in / float(budget), 4) if budget else None,
            "elapsed_s": 0.0,
        }
        return text

    compressor = get_compressor()
    t0 = time.time()
    rate = max(1e-4, min(0.999, (budget / float(n_in)) * _RATE_MARGIN))
    try:
        res = _compress_once(compressor, text, question, rate)
    except Exception as exc:
        raise RuntimeError(
            "%s compression failed (%s: %s)"
            % (VARIANT, type(exc).__name__, exc)
        ) from exc
    out = (res.get("compressed_prompt") or "").strip()
    got = len(_encode(tokenizer, out))

    dropped = 0
    fallback = False
    if got > budget:
        out, dropped = _truncate_at_sentence_boundary(out, budget, tokenizer)
        got = len(_encode(tokenizer, out))
        fallback = True
        FALLBACK_COUNT += 1
        logger.warning(
            "%s sentence-boundary fallback #%d: budget=%d "
            "compressed_tokens=%s -> %d (dropped_sentences=%d)",
            VARIANT,
            FALLBACK_COUNT,
            budget,
            res.get("compressed_tokens"),
            got,
            dropped,
        )
        if got > budget:
            out = _decode(tokenizer, _encode(tokenizer, out)[:budget])
            got = len(_encode(tokenizer, out))

    if not out:
        out = _decode(tokenizer, _encode(tokenizer, text)[:budget])
        got = len(_encode(tokenizer, out))

    LAST_RUN = {
        "variant": VARIANT,
        "input_tokens": n_in,
        "output_tokens": got,
        "budget": budget,
        "compressed": True,
        "rate": round(rate, 5),
        "fallback": fallback,
        "dropped_sentences": dropped,
        "utilization": round(got / float(budget), 4) if budget else None,
        "elapsed_s": round(time.time() - t0, 4),
        "llmlingua_origin_tokens": res.get("origin_tokens"),
        "llmlingua_compressed_tokens": res.get("compressed_tokens"),
        "llmlingua_ratio": res.get("ratio"),
        "device": DEVICE,
    }
    return out
