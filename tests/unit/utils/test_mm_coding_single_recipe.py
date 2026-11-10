# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Resolve the real Super recipe and check the training-row adapter without GPUs."""

import base64
import hashlib
import importlib.util
import json
import os
import subprocess
from pathlib import Path

import pytest
from omegaconf.errors import InterpolationResolutionError
from PIL import Image
from pydantic import ValidationError

from nemo_rl.utils.config import load_config

ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = ROOT / "examples/nemo_gym/nemotron-3.5-super-vl"
spec = importlib.util.spec_from_file_location(
    "prepare_mm_coding_single", EXAMPLE / "prepare_mm_coding_single.py"
)
assert spec is not None and spec.loader is not None
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


@pytest.mark.parametrize("checkpoint_kind", ["empty", "populated", "dangling_link"])
def test_training_acceptance_refuses_existing_checkpoint_root(
    tmp_path: Path, checkpoint_kind: str
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    if checkpoint_kind == "dangling_link":
        checkpoint_root.symlink_to(tmp_path / "missing")
    else:
        checkpoint_root.mkdir()
        if checkpoint_kind == "populated":
            (checkpoint_root / "training_info.json").write_text("original checkpoint")
    marker = tmp_path / "runner-called"
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    runner = bin_dir / "uv"
    runner.write_text(f'#!/usr/bin/env bash\ntouch "{marker}"\nexit 91\n')
    runner.chmod(0o755)
    result = subprocess.run(
        [
            "bash",
            str(ROOT / "tests/functional/mm_coding_super35_train_resume.sh"),
            "single",
        ],
        env={
            **os.environ,
            "PATH": f"{bin_dir}:{os.environ['PATH']}",
            "MM_CODING_TRAIN_DATA": str(tmp_path / "train.jsonl"),
            "MM_CODING_VALIDATION_DATA": str(tmp_path / "validation.jsonl"),
            "MM_CODING_REFERENCE_ROOT": str(tmp_path),
            "MM_CODING_ACCEPTANCE_ROOT": str(tmp_path / "acceptance"),
            "MM_CODING_CHECKPOINT_ROOT": str(checkpoint_root),
        },
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 2
    assert "Refusing to reuse an earlier checkpoint directory" in result.stderr
    assert not marker.exists()
    if checkpoint_kind == "populated":
        assert (
            checkpoint_root / "training_info.json"
        ).read_text() == "original checkpoint"
    if checkpoint_kind == "dangling_link":
        assert checkpoint_root.is_symlink()


def test_data_paths_are_explicit_not_example_or_benchmark_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MM_CODING_TRAIN_DATA", raising=False)
    config = load_config(EXAMPLE / "mm_coding_single.yaml")
    with pytest.raises(InterpolationResolutionError, match="MM_CODING_TRAIN_DATA"):
        _ = config.data.train.data_path


@pytest.mark.parametrize(
    "benchmark",
    [
        "chart2code",
        "mmsvg_img2svg",
        "mmsvg_text2svg",
        "image2struct_latex",
        "design2code_full",
    ],
)
def test_adapter_preserves_original_input_bytes_and_hides_gold(
    tmp_path: Path, benchmark: str
) -> None:
    path = tmp_path / "reference.png"
    Image.new("RGB", (16, 16), "red").save(path)
    task = adapter.ExperimentTask(
        benchmark=benchmark,
        task_id="1",
        prompt="original prompt",
        semantic_prompt="semantic task",
        ground_truth="PRIVATE_REFERENCE_SOURCE",
        references=[] if benchmark == "mmsvg_text2svg" else [str(path)],
    )
    row = adapter.gym_row(
        task, reference_root=tmp_path, split="train", max_output_tokens=8192
    )
    content = row["responses_create_params"]["input"][0]["content"]
    assert content[0]["text"] == task.prompt
    assert "PRIVATE_REFERENCE_SOURCE" not in json.dumps(row["responses_create_params"])
    assert row["task"]["ground_truth"] == task.ground_truth
    assert row["task"]["semantic_prompt"] == task.semantic_prompt
    assert row["responses_create_params"]["max_output_tokens"] == 8192
    if task.references:
        assert (
            base64.b64decode(content[1]["image_url"].split(",", 1)[1])
            == path.read_bytes()
        )
        assert (
            row["task"]["reference_sha256"]
            == hashlib.sha256(path.read_bytes()).hexdigest()
        )
    else:
        assert len(content) == 1 and "reference_image" not in row["task"]


def test_adapter_rejects_eval_split_and_outside_root(tmp_path: Path) -> None:
    task = adapter.ExperimentTask(
        benchmark="mmsvg_text2svg", task_id="1", prompt="task"
    )
    with pytest.raises(ValueError, match="approved"):
        adapter.gym_row(
            task, reference_root=tmp_path, split="test", max_output_tokens=8192
        )
    image_task = adapter.ExperimentTask(
        benchmark="chart2code", task_id="2", prompt="task", references=["/etc/passwd"]
    )
    with pytest.raises(ValueError, match="reference_root"):
        adapter.gym_row(
            image_task, reference_root=tmp_path, split="train", max_output_tokens=8192
        )


def test_duplicate_tasks_leave_no_partial_output(tmp_path: Path) -> None:
    data = {"benchmark": "mmsvg_text2svg", "task_id": "1", "prompt": "task"}
    source, output = tmp_path / "input.jsonl", tmp_path / "output.jsonl"
    source.write_text((json.dumps(data) + "\n") * 2)
    with pytest.raises(ValueError, match="Duplicate"):
        adapter.export(
            source,
            output,
            reference_root=tmp_path,
            split="train",
            max_output_tokens=8192,
        )
    assert not output.exists()


def test_export_does_not_replace_an_existing_dataset(tmp_path: Path) -> None:
    data = {"benchmark": "mmsvg_text2svg", "task_id": "1", "prompt": "task"}
    source, output = tmp_path / "input.jsonl", tmp_path / "output.jsonl"
    source.write_text(json.dumps(data) + "\n")
    output.write_bytes(b"original")
    with pytest.raises(FileExistsError):
        adapter.export(
            source,
            output,
            reference_root=tmp_path,
            split="train",
            max_output_tokens=8192,
        )
    assert output.read_bytes() == b"original"


def test_unrelated_task_families_are_refused() -> None:
    with pytest.raises(ValidationError):
        adapter.ExperimentTask(benchmark="webmmu_code_edit", task_id="x", prompt="task")


@pytest.mark.parametrize("mode", ["single", "multi", "mixed"])
def test_routes_are_deterministic_and_ignore_order(mode):
    tasks = [
        adapter.ExperimentTask(
            benchmark="mmsvg_text2svg", task_id=str(i), prompt="draw"
        )
        for i in range(500)
    ]
    routes = {t.task_id: adapter.select_route(t, mode=mode) for t in tasks}
    assert routes == {
        t.task_id: adapter.select_route(t, mode=mode) for t in reversed(tasks)
    }
    if mode == "single":
        assert set(routes.values()) == {"single"}
    elif mode == "multi":
        assert set(routes.values()) == set(adapter.HARNESS_NAMES)
    else:
        assert set(routes.values()) == {"single", *adapter.HARNESS_NAMES}
        assert 0.15 < sum(v != "single" for v in routes.values()) / len(tasks) < 0.35


@pytest.mark.parametrize("harness", ["opencode", "codex", "hermes"])
def test_multi_rows_have_private_task_envelope_and_file_instructions(tmp_path, harness):
    task = adapter.ExperimentTask(
        benchmark="mmsvg_text2svg",
        task_id="1",
        prompt="draw",
        ground_truth="PRIVATE_GOLD",
    )
    row = adapter.gym_row(
        task,
        reference_root=tmp_path,
        split="train",
        max_output_tokens=8192,
        selected=harness,
    )
    assert row["task_id"]["taskset"] == f"mm-coding:{harness}"
    assert row["task_id"]["task_id"] == "mmsvg_text2svg:1"
    assert row["task_input"]["task_data"]["task"]["ground_truth"] == "PRIVATE_GOLD"
    prompt = json.dumps(row["task_input"]["responses_create_params"])
    assert "PRIVATE_GOLD" not in prompt and "/workspace/answer.svg" in prompt
    assert "/opt/mmrl/render.py" in prompt
    assert (
        "Only the final answer file is transferred to a fresh grading sandbox" in prompt
    )
    assert "without any other task-workspace files" in prompt
    assert "Do not read or embed the task reference image as the solution" in prompt


@pytest.mark.parametrize("harnesses", [(), ("opencode", "opencode"), ("unknown",)])
def test_unknown_or_repeated_harness_routes_are_rejected(harnesses):
    task = adapter.ExperimentTask(
        benchmark="mmsvg_text2svg", task_id="1", prompt="draw"
    )
    with pytest.raises(ValueError, match="unique supported"):
        adapter.select_route(task, mode="mixed", harnesses=harnesses)


@pytest.mark.parametrize("mode", ["single", "multi", "mixed"])
def test_existing_source_jsonl_is_retargeted_without_reusing_old_agent_prompt(
    tmp_path, mode
):
    source, output = tmp_path / "legacy.jsonl", tmp_path / "converted.jsonl"
    source.write_text(
        json.dumps(
            {
                "agent_ref": {"type": "responses_api_agents", "name": "mm_old"},
                "task": {
                    "benchmark": "mmsvg_text2svg",
                    "task_id": "task",
                    "prompt": "original task",
                    "ground_truth": "PRIVATE_GOLD",
                },
                "route": "single",
                "responses_create_params": {"input": "OLD_AGENT_INSTRUCTIONS"},
            }
        )
        + "\n"
    )
    assert (
        adapter.export(
            source,
            output,
            reference_root=tmp_path,
            split="train",
            max_output_tokens=8192,
            input_format="legacy-row",
            mode=mode,
        )
        == 1
    )
    row = json.loads(output.read_text())
    params = (
        row["task_input"]["responses_create_params"]
        if "task_input" in row
        else row["responses_create_params"]
    )
    assert "original task" in json.dumps(params)
    assert "PRIVATE_GOLD" not in json.dumps(
        params
    ) and "OLD_AGENT_INSTRUCTIONS" not in json.dumps(params)


def test_adapter_cli_writes_multi_dataset(tmp_path, monkeypatch, capsys):
    source, output = tmp_path / "tasks.jsonl", tmp_path / "multi.jsonl"
    source.write_text(
        json.dumps({"benchmark": "mmsvg_text2svg", "task_id": "one", "prompt": "draw"})
        + "\n"
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "prepare",
            "--input",
            str(source),
            "--output",
            str(output),
            "--reference-root",
            str(tmp_path),
            "--split",
            "train",
            "--input-format",
            "task",
            "--mode",
            "multi",
            "--harnesses",
            "opencode",
        ],
    )
    adapter.main()
    assert json.loads(output.read_text())["task_id"]["taskset"] == "mm-coding:opencode"
    assert json.loads(capsys.readouterr().out)["rows"] == 1


def test_runtime_guard_detects_missing_session_api(monkeypatch):
    spec = importlib.util.spec_from_file_location(
        "check_mm_coding_runtime", EXAMPLE / "check_mm_coding_runtime.py"
    )
    assert spec is not None and spec.loader is not None
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)

    def find(module):
        return None if module == "nemo_gym.server_runtime.sandbox_lending" else object()

    monkeypatch.setattr(guard.importlib.util, "find_spec", find)
    assert guard.check_runtime() == ["nemo_gym.server_runtime.sandbox_lending"]
    monkeypatch.setattr("sys.argv", ["check_mm_coding_runtime"])
    with pytest.raises(SystemExit, match="stock RL Gym pin"):
        guard.main()


@pytest.mark.parametrize("mode", ["multi", "mixed"])
def test_runtime_guard_refuses_missing_opencode_capture_support(
    monkeypatch, capsys, mode
):
    spec = importlib.util.spec_from_file_location(
        "check_mm_coding_runtime", EXAMPLE / "check_mm_coding_runtime.py"
    )
    assert spec is not None and spec.loader is not None
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    missing = "nemo_gym.models.token_capture.harness_notes"
    monkeypatch.setattr(
        guard.importlib.util,
        "find_spec",
        lambda module: None if module == missing else object(),
    )
    monkeypatch.setattr("sys.argv", ["check_mm_coding_runtime", "--mode", mode])
    with pytest.raises(SystemExit, match="5833"):
        guard.main()
    assert json.loads(capsys.readouterr().out)["missing_modules"] == [missing]
    assert guard.check_runtime(mode="single") == []
    monkeypatch.setattr(guard.importlib.util, "find_spec", lambda module: object())
    guard.main()
    assert json.loads(capsys.readouterr().out)["compatible"]


def test_legacy_image_drift_is_refused_without_partial_dataset(tmp_path):
    image = tmp_path / "reference.png"
    Image.new("RGB", (8, 8), "red").save(image)
    original = image.read_bytes()
    task = {
        "benchmark": "chart2code",
        "task_id": "one",
        "prompt": "draw",
        "references": [str(image)],
    }
    source = tmp_path / "legacy.jsonl"
    source.write_text(
        json.dumps(
            {
                "agent_ref": {"type": "responses_api_agents", "name": "old"},
                "route": "single",
                "task": task,
                "responses_create_params": {
                    "input": [
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "input_image",
                                    "image_url": "data:image/png;base64,"
                                    + base64.b64encode(original).decode(),
                                }
                            ],
                        }
                    ]
                },
            }
        )
        + "\n"
    )
    Image.new("RGB", (8, 8), "blue").save(image)
    output = tmp_path / "converted.jsonl"
    with pytest.raises(ValueError, match="bytes differ"):
        adapter.export(
            source,
            output,
            reference_root=tmp_path,
            split="train",
            max_output_tokens=8192,
            input_format="legacy-row",
        )
    assert not output.exists()


@pytest.mark.parametrize("harness", ["codex", "hermes"])
def test_experimental_image_route_preserves_owner_pixels_and_requires_file_inspection(
    tmp_path, harness
):
    path = tmp_path / "reference.png"
    Image.new("RGB", (12, 12), "blue").save(path)
    task = adapter.ExperimentTask(
        benchmark="mmsvg_img2svg",
        task_id="experimental-image",
        prompt="draw this",
        references=[str(path)],
    )
    row = adapter.gym_row(
        task,
        reference_root=tmp_path,
        split="example",
        max_output_tokens=8192,
        selected=harness,
    )
    params = row["task_input"]["responses_create_params"]
    owner = row["task_input"]["task_data"]["task"]
    assert owner["reference_sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert "/workspace/reference.png" in json.dumps(params)
    assert "Inspect it with your image tool" in json.dumps(params)
    assert (
        "Preview with `/opt/gym/bin/python /opt/mmrl/render.py"
        in params["input"][0]["content"]
    )
    assert not any(
        p.get("type") == "input_image"
        for message in params["input"]
        for p in (message["content"] if isinstance(message["content"], list) else [])
    )


def test_experimental_runtime_preflight_refuses_unpatched_stock_adapter(monkeypatch):
    from types import SimpleNamespace

    spec = importlib.util.spec_from_file_location(
        "experimental_runtime_guard", EXAMPLE / "check_mm_coding_runtime.py"
    )
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    stock = SimpleNamespace(CodexAgentConfig=SimpleNamespace(model_fields={}))
    monkeypatch.setattr(guard.importlib, "import_module", lambda name: stock)
    assert guard.check_experimental_apis(("codex",)) == [
        "responses_api_agents.codex_agent.app.CodexAgentConfig.forward_sampling_controls"
    ]
    assert guard.check_experimental_apis(("opencode",)) == []
