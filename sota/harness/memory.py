
from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
import sys
from typing import Callable, Dict, List, Optional, Sequence, Tuple

_HERE = os.path.dirname(os.path.abspath(__file__))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import budget              
import data              
import paths              

NATIVE_METHODS = ("full", "trunc_head", "trunc_tail")
BASELINE_METHODS = ("bm25", "summary", "llmlingua", "custom", "chain_writer", "hybrid_chain")
ALL_METHODS = NATIVE_METHODS + BASELINE_METHODS

BASELINES_PACKAGE = "sota.baselines"


                                                                            
                        
                                                                            
def build_native(
    method: str,
    paragraphs: Sequence[str],
    budget_tokens: Optional[int],
    counter: Optional[budget.TokenCounter] = None,
) -> str:
    """Build the ``full`` / ``trunc_head`` / ``trunc_tail`` memory string."""
    stream = data.context_text(paragraphs)
    if method == "full":
        return stream
    if budget_tokens is None:
        raise ValueError(f"method {method!r} requires --budget")
    if method == "trunc_head":
        return budget.truncate_head(stream, budget_tokens, counter)[0]
    if method == "trunc_tail":
        return budget.truncate_tail(stream, budget_tokens, counter)[0]
    raise ValueError(f"not a native method: {method!r}")


                                                                            
                               
                                                                            
def _sys_path_setup() -> None:
    """Make ``import sota.baselines`` work without shadowing installed packages.

    ``paths.BASELINES_DIR`` is APPENDED (never inserted): ``sota/baselines/``
    contains a module literally named ``llmlingua.py``/``summary.py``, and
    putting that directory first on ``sys.path`` would shadow the real
    ``llmlingua`` package (observed: ``cannot import name 'PromptCompressor'``).
    Appending keeps third-party packages first while sibling imports still work.
    """
    for p in (paths.REPO_ROOT, paths.SOTA_ROOT):
        if p and p not in sys.path:
            sys.path.insert(0, p)
    if paths.BASELINES_DIR not in sys.path:
        sys.path.append(paths.BASELINES_DIR)


def _load_module_by_file(path: str, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load python module from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _candidates(method: str, custom_module: Optional[str] = None):
    """Ordered ``(kind, value, name)`` candidates for a baseline method."""
    cands: List[Tuple[str, str, str]] = []
    if method == "custom" and custom_module:
        cands.append(("file", custom_module, "_sota_baseline_custom_file"))
    cands.append(("module", f"{BASELINES_PACKAGE}.{method}", method))
    cands.append(("module", BASELINES_PACKAGE, None))
    cands.append(("file", os.path.join(paths.BASELINES_DIR, method + ".py"), method))
    cands.append(
        ("file", os.path.join(paths.BASELINES_DIR, "__init__.py"), "dispatcher")
    )
    return cands


def load_baseline_builder(
    method: str, custom_module: Optional[str] = None
) -> Callable:
    """Import and return the ``build_memory`` callable for a baseline method."""
    _sys_path_setup()
    errors: List[str] = []
    for kind, value, name in _candidates(method, custom_module):
        if kind == "module":
            try:
                mod = importlib.import_module(value)
            except Exception as exc:                                            
                errors.append(f"import {value}: {exc!r}")
                continue
            fn = getattr(mod, "build_memory", None)
            if callable(fn):
                return fn
            errors.append(f"import {value}: no build_memory()")
            continue
                        
        if not os.path.exists(value):
            errors.append(f"missing file {value}")
            continue
        try:
            modname = f"_sota_baseline_{name or 'mod'}"
            if modname in sys.modules:
                mod = sys.modules[modname]
            else:
                mod = _load_module_by_file(value, modname)
        except Exception as exc:
            errors.append(f"load {value}: {exc!r}")
            continue
        fn = getattr(mod, "build_memory", None)
        if callable(fn):
            return fn
        errors.append(f"load {value}: no build_memory()")

    listing = available_baselines()
    raise RuntimeError(
        f"no baseline implementation found for method {method!r}.\n"
        f"  searched: sota/baselines/{method}.py, sota/baselines/__init__.py, "
        f"python package {BASELINES_PACKAGE}\n"
        f"  baselines dir: {paths.BASELINES_DIR} "
        f"(exists={os.path.isdir(paths.BASELINES_DIR)})\n"
        f"  detected modules: {listing}\n"
        f"  errors: " + " | ".join(errors[-4:]) +
        "\n  expected contract: build_memory(question, paragraphs, budget, tokenizer) -> str"
    )


def available_baselines() -> Dict[str, str]:
    """Map method name -> "where it would be found" (for --list-methods/errors)."""
    found: Dict[str, str] = {}
    custom_path = os.path.join(paths.BASELINES_DIR, "custom.py")
    found["custom"] = custom_path if os.path.exists(custom_path) else "needs --custom-module"
    for method in BASELINE_METHODS:
        if method == "custom":
            continue
        for kind, value, name in _candidates(method):
            if kind == "file" and os.path.exists(value):
                found[method] = value
                break
            if kind == "module":
                try:
                    mod = importlib.import_module(value)
                except Exception:
                    continue
                if callable(getattr(mod, "build_memory", None)):
                    found[method] = f"module:{value}"
                    break
    return found


def _values(
    question: str,
    paragraphs: Sequence[str],
    budget_tokens: Optional[int],
    tokenizer,
    method: str,
) -> Dict:
    return {
        "method": method,
        "question": question,
        "paragraphs": list(paragraphs),
        "budget": budget_tokens,
        "tokenizer": tokenizer,
                                                    
        "budget_tokens": budget_tokens,
        "tok": tokenizer,
    }


def _positional_from(sig_params: Sequence[inspect.Parameter], values: Dict) -> List:
    """Build a positional argument list from parameter names, filling any
    unknown names from the canonical order."""
    canonical = ["method", "question", "paragraphs", "budget", "tokenizer"]
    args: List = []
    used: set = set()
    for p in sig_params:
        if p.kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            continue
        if p.name in values:
            args.append(values[p.name])
            used.add(p.name)
        else:
            for cand in canonical:
                if cand not in used and cand not in {q.name for q in sig_params}:
                    args.append(values[cand])
                    used.add(cand)
                    break
    return args


def call_builder(
    fn: Callable,
    method: str,
    question: str,
    paragraphs: Sequence[str],
    budget_tokens: Optional[int],
    tokenizer,
) -> str:
    """Call ``fn`` respecting its signature; returns the memory text."""
    values = _values(question, paragraphs, budget_tokens, tokenizer, method)
    try:
        sig = inspect.signature(fn)
        params = list(sig.parameters.values())
    except (TypeError, ValueError):                        
        return fn(question, paragraphs, budget_tokens, tokenizer)

    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params):
        result = fn(**values)
    else:
        named = {p.name for p in params}
        matched = named & set(values)
        required = [
            p.name
            for p in params
            if p.kind
            in (
                inspect.Parameter.POSITIONAL_ONLY,
                inspect.Parameter.POSITIONAL_OR_KEYWORD,
            )
            and p.default is inspect.Parameter.empty
        ]
        if matched and all(name in values for name in required if name in named):
            kwargs = {name: values[name] for name in named if name in values}
            result = fn(**kwargs)
        else:
            args = _positional_from(params, values)
            result = fn(*args)

    if not isinstance(result, str):
        raise TypeError(
            f"baseline {method!r} returned {type(result).__name__}, expected str"
        )
    return result


                                                                            
                    
                                                                            
def memory_for(
    method: str,
    question: str,
    paragraphs: Sequence[str],
    budget_tokens: Optional[int],
    tokenizer=None,
    counter: Optional[budget.TokenCounter] = None,
    custom_module: Optional[str] = None,
) -> str:
    """Build the memory string for ``method`` (native or baseline)."""
    if method not in ALL_METHODS:
        raise ValueError(
            f"unknown method {method!r}; expected one of {ALL_METHODS}"
        )
    if method in NATIVE_METHODS:
        return build_native(method, paragraphs, budget_tokens, counter)
    if (
        method == "custom"
        and not custom_module
        and not os.path.exists(os.path.join(paths.BASELINES_DIR, "custom.py"))
    ):
        raise RuntimeError(
            "method 'custom' needs --custom-module path/to/file.py exposing "
            "build_memory(question, paragraphs, budget, tokenizer) -> str"
        )
    counter = budget.get_counter(counter)
    fn = load_baseline_builder(method, custom_module=custom_module)
    return call_builder(fn, method, question, paragraphs, budget_tokens, counter.tokenizer)
