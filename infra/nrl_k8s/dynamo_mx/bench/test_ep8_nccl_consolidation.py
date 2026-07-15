from __future__ import annotations

import importlib.util
from pathlib import Path


def _load():
    path = Path(__file__).with_name("ep8_nccl_consolidation.py")
    spec = importlib.util.spec_from_file_location(
        "ep8_nccl_consolidation", path
    )
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


consolidation = _load()


def test_consolidation_statistics_schema() -> None:
    assert consolidation._stats([0.7, 0.6, 0.65]) == {
        "samples": 3,
        "min": 0.6,
        "median": 0.65,
        "p95": 0.7,
        "max": 0.7,
    }


def test_default_import_does_not_require_torchrun_environment() -> None:
    assert consolidation.RANK == 0
    assert consolidation.WORLD >= 1
