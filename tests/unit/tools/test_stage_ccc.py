# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json

import pytest

from tools.super_rl.stage_ccc import json_bytes, read_json, stage


@pytest.fixture
def inputs(tmp_path):
    problem = {
        "tests": [{"input": "中文\n", "output": "42\n"}],
        "grader": {"source": "original"},
    }
    payload = (
        json_bytes({"competition_id": "c", "metadata": {"p": problem, "unused": {}}})
        + b"\n"
    )
    (tmp_path / "chunk.jsonl").write_bytes(payload)
    manifest = {
        "sources": [
            {"path": "chunk.jsonl", "sha256": hashlib.sha256(payload).hexdigest()}
        ],
        "problems": [
            {
                "competition_id": "c",
                "problem_id": "p",
                "sha256": hashlib.sha256(
                    json_bytes(problem, sort_keys=True)
                ).hexdigest(),
            }
        ],
    }
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest))
    return path, tmp_path / "result.jsonl", problem


def test_preserves_native_problem_and_refuses_overwrite(inputs):
    manifest, output, problem = inputs
    receipt = stage(manifest, output)
    assert read_json(output.read_bytes())["metadata"] == {"p": problem}
    assert receipt["verified_problems"] == 1
    assert receipt["sha256"] == hashlib.sha256(output.read_bytes()).hexdigest()
    with pytest.raises(FileExistsError):
        stage(manifest, output)


@pytest.mark.parametrize(
    "fault", ["source_hash", "problem_hash", "missing", "duplicate"]
)
def test_failure_does_not_publish(inputs, fault):
    path, output, _ = inputs
    manifest = read_json(path.read_bytes())
    if fault == "source_hash":
        manifest["sources"][0]["sha256"] = "wrong"
    elif fault == "problem_hash":
        manifest["problems"][0]["sha256"] = "wrong"
    elif fault == "missing":
        manifest["problems"][0]["problem_id"] = "missing"
    else:
        manifest["problems"] *= 2
    path.write_text(json.dumps(manifest))
    with pytest.raises(ValueError):
        stage(path, output)
    assert not output.exists()
    assert not list(output.parent.glob(".result.jsonl.*"))


@pytest.mark.parametrize(
    "value", ["NaN", "Infinity", "-Infinity", "1e999", '{"a":1,"a":2}']
)
def test_strict_json(value):
    with pytest.raises(ValueError):
        read_json(value)


def test_duplicate_source_does_not_duplicate_problems(inputs):
    path, output, _ = inputs
    manifest = read_json(path.read_bytes())
    manifest["sources"] *= 2
    path.write_text(json.dumps(manifest))
    assert stage(path, output)["verified_problems"] == 1
    assert len(output.read_bytes().splitlines()) == 1
