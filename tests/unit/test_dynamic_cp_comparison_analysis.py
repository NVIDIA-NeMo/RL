# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
"""Keep comparison labels consistent with the configuration actually run."""

from pathlib import Path

import pytest

from tools.analyze_dynamic_cp_comparison import (
    MODEL_METADATA,
    RunData,
    RunSpec,
    build_rows,
)


@pytest.mark.parametrize("recorded_cp", [None, 1, 4])
def test_static_cp_label_uses_resolved_static_run(recorded_cp):
    runs = {
        ("qwen30", mode): RunData(
            spec=RunSpec("qwen30", mode, Path(mode)),
            metrics={"timing/train/total_step_time": {1: 10.0}},
            packing_utilizations=[],
            samples_by_cp={},
            tasks_by_cp={},
            source="logs",
            resolved_gbs=512,
            resolved_max_sequence_length=8192,
            resolved_megatron_cp=recorded_cp if mode == "static" else 8,
            expected_steps=1,
            completed=True,
        )
        for mode in ("dynamic", "static")
    }

    rows, warnings = build_rows(runs, drop_first_steps=0, drop_last_steps=0)

    assert not warnings
    assert len(rows) == 1
    assert rows[0]["static_cp"] == (
        MODEL_METADATA["qwen30"]["static_cp"] if recorded_cp is None else recorded_cp
    )
    assert rows[0]["dynamic_cp"] == MODEL_METADATA["qwen30"]["dynamic_cp"]
