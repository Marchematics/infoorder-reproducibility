
from __future__ import annotations

import importlib.util
import os

try:                                                        
    from .llmlingua_compressor import (              
        LAST_RUN,
        build_memory,
        get_compressor,
    )
except ImportError:                                              
    _path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "llmlingua_compressor.py")
    _spec = importlib.util.spec_from_file_location("_sota_llmlingua_compressor", _path)
    _mod = importlib.util.module_from_spec(_spec)
    _spec.loader.exec_module(_mod)
    build_memory = _mod.build_memory
    get_compressor = _mod.get_compressor
    LAST_RUN = _mod.LAST_RUN

__all__ = ["build_memory", "get_compressor", "LAST_RUN"]
