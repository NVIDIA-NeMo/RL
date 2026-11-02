# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
"""Benchmark-only GIL control is opt-in, validated, and process-local."""

from __future__ import annotations

import ast
import json
import math
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

from tools.model_diagnostics import prefix_cut_benchmark as bench


@pytest.fixture
def configure() -> Any:
    # The exact production helper is CPU-only; avoid importing Ray/vLLM workers.
    path = Path(bench.__file__).with_name("prefix_cut_benchmark_worker.py")
    tree = ast.parse(path.read_text())
    node = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "configure_gil_switch_interval"
    )
    namespace = {"sys": sys, "math": math}
    exec(
        compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace
    )
    return namespace["configure_gil_switch_interval"]


def test_one_millisecond_is_effective_and_default_does_not_reset_it(
    configure: Any,
) -> None:
    original = sys.getswitchinterval()
    try:
        assert configure(1.0) == pytest.approx(1.0)
        assert sys.getswitchinterval() == pytest.approx(0.001)
        assert configure(None) == pytest.approx(1.0)
        assert sys.getswitchinterval() == pytest.approx(0.001)
    finally:
        sys.setswitchinterval(original)


@pytest.mark.parametrize(
    "invalid", [0.0, -1.0, float("nan"), float("inf"), -float("inf")]
)
def test_invalid_worker_setting_preserves_existing_interval(
    configure: Any, invalid: float
) -> None:
    original = sys.getswitchinterval()
    with pytest.raises(ValueError, match="finite and positive"):
        configure(invalid)
    assert sys.getswitchinterval() == original


@pytest.mark.parametrize("interval", [None, 1.0])
def test_cli_plan_records_interval_without_changing_driver(
    tmp_path: Path, monkeypatch: Any, capsys: Any, interval: float | None
) -> None:
    config = {
        "policy": {
            "model_name": "not-loaded-during-dry-run",
            "router_replay": {"enabled": False},
            "generation": {
                "vllm_cfg": {"tensor_parallel_size": 4, "flat_logprobs": True},
                "vllm_kwargs": {"max_num_seqs": 512},
                "colocated": {"enabled": False, "resources": {"num_nodes": 1}},
            },
        },
        "data_plane": {"backend": "simple", "simple": {}},
    }
    source = tmp_path / "config.yaml"
    source.write_text(yaml.safe_dump(config))
    args = [
        "benchmark",
        "--reference-config",
        str(source),
        "--output",
        str(tmp_path / "out"),
        "--dry-run",
    ]
    if interval is not None:
        args += ["--gil-switch-interval-ms", str(interval)]
    monkeypatch.setattr(sys, "argv", args)
    original = sys.getswitchinterval()
    bench.main()
    plan = json.loads(capsys.readouterr().out)
    assert plan["gil_switch_interval_ms"] == interval
    assert sys.getswitchinterval() == original
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("value", ["0", "-1", "nan", "inf", "-inf"])
def test_cli_rejects_invalid_interval_before_loading_config(
    tmp_path: Path, monkeypatch: Any, capsys: Any, value: str
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "benchmark",
            "--reference-config",
            str(tmp_path / "absent.yaml"),
            "--output",
            str(tmp_path / "out"),
            "--dry-run",
            f"--gil-switch-interval-ms={value}",
        ],
    )
    with pytest.raises(SystemExit) as error:
        bench.main()
    assert error.value.code == 2
    assert "GIL switch interval must be finite and positive" in capsys.readouterr().err
