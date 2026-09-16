# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import ast
import json
from pathlib import Path

import pytest

from nemo_rl.algorithms.metric_utils import without_generation_logger_payload


def test_large_generation_payload_does_not_swallow_training_scalars():
    raw = {"tokens": {"worker": [1.0] * 2_200_000}}
    metrics = {
        "generation_logger_metrics": raw,
        "loss": 0.25,
        "gen_kl_error": 0.01,
        "reward": 0.5,
        "other_summary": {"max": 42},
    }
    assert len(json.dumps(metrics)) > 10_383_360
    filtered = without_generation_logger_payload(metrics)
    assert filtered == {
        key: value
        for key, value in metrics.items()
        if key != "generation_logger_metrics"
    }
    assert len(json.dumps(filtered)) < 1000
    assert metrics["generation_logger_metrics"] is raw


def test_no_payload_and_empty_metrics():
    assert without_generation_logger_payload({}) == {}
    assert without_generation_logger_payload({"loss": 1}) == {"loss": 1}


@pytest.mark.parametrize(
    "module,expected_filters",
    [("grpo.py", 2), ("grpo_sync.py", 1), ("ppo.py", 2)],
)
def test_every_train_loop_filters_only_after_performance_summary(
    module, expected_filters
):
    source = Path(__file__).resolve().parents[3] / "nemo_rl/algorithms" / module
    tree = ast.parse(source.read_text())
    checked = 0
    for function in tree.body:
        if not isinstance(function, ast.FunctionDef):
            continue
        calls = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        ]
        filters = [
            node
            for node in calls
            if node.func.id == "without_generation_logger_payload"
        ]
        for filtered in filters:
            summaries = [
                node for node in calls if node.func.id == "print_performance_metrics"
            ]
            assert any(node.lineno < filtered.lineno for node in summaries)
            checked += 1
    assert checked == expected_filters
