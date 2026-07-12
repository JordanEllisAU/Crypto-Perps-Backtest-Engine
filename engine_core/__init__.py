"""engine_core — namespace package that maps src/ and config/ into engine_core.*.

The repo was originally structured with `src/` and `config/` at the root, but
imports use `engine_core.src.engine` and `engine_core.config.params_loader`.
This package makes both work by adding the repo root to sys.path and then
aliasing `engine_core.src` → `src` and `engine_core.config` → `config`.
"""
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Alias engine_core.src -> src (the actual package at repo root)
import src as _src
import config as _config
import tests as _tests
sys.modules[f"{__name__}.src"] = _src
sys.modules[f"{__name__}.config"] = _config
sys.modules[f"{__name__}.tests"] = _tests
