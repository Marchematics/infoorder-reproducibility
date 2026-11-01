"""Project paths and evaluation defaults."""

from __future__ import annotations

import os


def _repo_root() -> str:
    env = os.environ.get("INFOORDER_ROOT")
    if env:
        return os.path.abspath(os.path.expanduser(env))
                                                                                      
    return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


HARNESS_DIR = os.path.dirname(os.path.abspath(__file__))
SOTA_ROOT = os.path.dirname(HARNESS_DIR)
REPO_ROOT = _repo_root()

DATA_DIR = os.path.join(SOTA_ROOT, "data")
RESULTS_DIR = os.path.join(SOTA_ROOT, "results")
BASELINES_DIR = os.environ.get(
    "INFOORDER_BASELINES_DIR", os.path.join(SOTA_ROOT, "baselines")
)

                                                                                     
SOURCE_JSONL = os.environ.get(
    "INFOORDER_SOURCE",
    os.path.join(DATA_DIR, "data", "musique_ans_v1.0_dev.jsonl"),
)
                                                                       
DEFAULT_MANIFEST = os.path.join(DATA_DIR, "manifest_500.jsonl")
DEFAULT_MANIFEST_META = os.path.join(DATA_DIR, "manifest_500.meta.json")

MODEL_DIR = os.environ.get(
    "INFOORDER_MODEL_DIR", os.path.join(REPO_ROOT, "models", "Qwen3.5-4B")
)
TOKENIZER_DIR = os.environ.get("INFOORDER_TOKENIZER_DIR", MODEL_DIR)

                                    
SEED = 20261027
N_PER_HOP = {"2hop": 167, "3hop": 167, "4hop": 166}
N_TOTAL = sum(N_PER_HOP.values())
