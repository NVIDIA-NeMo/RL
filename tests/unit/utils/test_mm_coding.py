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
"""Check coding-row adaptation and runtime compatibility behavior without GPUs."""

import base64
import hashlib
import importlib.util
import json
from copy import deepcopy
from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = ROOT / "examples/nemo_gym/mm_coding"
spec = importlib.util.spec_from_file_location(
    "prepare_mm_coding", EXAMPLE / "prepare_mm_coding.py"
)
assert spec is not None and spec.loader is not None
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


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
    row = adapter.gym_row(task, reference_root=tmp_path, split="train")
    content = row["responses_create_params"]["input"][0]["content"]
    assert content[0]["text"] == task.prompt
    assert "PRIVATE_REFERENCE_SOURCE" not in json.dumps(row["responses_create_params"])
    assert row["task"]["ground_truth"] == task.ground_truth
    assert row["task"]["semantic_prompt"] == task.semantic_prompt
    assert "max_output_tokens" not in row["responses_create_params"]
    assert "problem_id" not in row
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
        adapter.gym_row(task, reference_root=tmp_path, split="test")
    image_task = adapter.ExperimentTask(
        benchmark="chart2code", task_id="2", prompt="task", references=["/etc/passwd"]
    )
    with pytest.raises(ValueError, match="reference_root"):
        adapter.gym_row(image_task, reference_root=tmp_path, split="train")


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
        )
    assert output.read_bytes() == b"original"


@pytest.mark.parametrize(
    "extra",
    [
        {"original_source": "existing program"},
        {"reference_patch": "edit patch"},
        {"judge_references": ["private-judge-input.png"]},
        {"vision_manifest": {"asset": "editing-reference.png"}},
    ],
)
def test_custom_task_validation_rejects_editing_metadata(extra) -> None:
    with pytest.raises(
        ValidationError, match="supports only the four visual-coding task families"
    ):
        adapter.ExperimentTask(
            benchmark="mmsvg_text2svg", task_id="x", prompt="task", **extra
        )


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
        assert set(routes.values()) == {"opencode"}
    else:
        assert set(routes.values()) == {"single", "opencode"}
        assert 0.15 < sum(v != "single" for v in routes.values()) / len(tasks) < 0.35


@pytest.mark.parametrize("harness", ["opencode"])
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
        selected=harness,
    )
    assert "max_output_tokens" not in row["task_input"]["responses_create_params"]
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
        json.dumps(
            {"benchmark": "mmsvg_text2svg", "task_id": "example-1", "prompt": "draw"}
        )
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
        ],
    )
    adapter.main()
    row = json.loads(output.read_text())
    assert row["task_id"]["taskset"] == "mm-coding:opencode"
    assert (
        "Only the final answer file is transferred to a fresh grading sandbox"
        in (row["task_input"]["responses_create_params"]["input"][0]["content"])
    )
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
    with pytest.raises(SystemExit, match="Gym #6091"):
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
            input_format="legacy-row",
        )
    assert not output.exists()


@pytest.mark.parametrize("harness", ["codex", "hermes"])
def test_deferred_harness_cannot_silently_become_opencode(tmp_path, harness):
    task = adapter.ExperimentTask(
        benchmark="mmsvg_text2svg", task_id="unsupported-route", prompt="draw"
    )
    with pytest.raises(ValueError, match="Unknown harness route"):
        adapter.gym_row(
            task,
            reference_root=tmp_path,
            split="train",
            selected=harness,
        )
    with pytest.raises(ValueError, match="unique supported harnesses"):
        adapter.select_route(task, mode="multi", harnesses=(harness,))


@pytest.mark.nemo_gym
@pytest.mark.parametrize("selected", ["single", "opencode"])
def test_training_reader_and_processor_preserve_materialized_rows(tmp_path, selected):
    from nemo_gym.data.task_materialization import materialize_task as materialize

    from nemo_rl.data.datasets.response_datasets.nemogym_dataset import NemoGymDataset
    from nemo_rl.data.processors import nemo_gym_data_processor
    from nemo_rl.environments.nemo_gym import get_nemo_gym_route_name

    row = adapter.gym_row(
        adapter.ExperimentTask(
            benchmark="mmsvg_text2svg",
            task_id="example-1",
            prompt="Draw a blue circle",
            ground_truth="PRIVATE_REFERENCE",
        ),
        reference_root=tmp_path,
        split="train",
        selected=selected,
    )
    data_path = tmp_path / "train.jsonl"
    data_path.write_text(json.dumps(row) + "\n")
    dataset = NemoGymDataset(str(data_path))
    datum = nemo_gym_data_processor(dataset.dataset[0], None, None, None, idx=0)
    assert datum["extra_env_info"] == row
    actual = (
        materialize(datum["extra_env_info"], taskset="coding", task_index=0)
        if selected == "single"
        else datum["extra_env_info"]
    )
    expected = (
        materialize(row, taskset="coding", task_index=0)
        if selected == "single"
        else row
    )
    route = get_nemo_gym_route_name(
        datum["extra_env_info"],
        {"mm-coding:opencode": "mm_coding_opencode_environment"},
    )
    assert route == (
        "mm_coding_training_simple_agent"
        if selected == "single"
        else "mm_coding_opencode_environment"
    )
    assert actual == expected
    params = actual["task_input"]["responses_create_params"]
    assert "PRIVATE_REFERENCE" not in json.dumps(params)
    if selected == "opencode":
        assert actual["task_id"]["taskset"] == "mm-coding:opencode"
        assert "Only the final answer file" in json.dumps(params)


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"policy": {"generation": {"backend": "sglang"}}}, "vLLM generation"),
        (
            {
                "policy": {
                    "max_total_sequence_length": 32768,
                    "generation": {"max_new_tokens": 32768},
                }
            },
            "Reserve input context",
        ),
        (
            {"policy": {"generation": {"max_new_tokens": 0}}},
            "max_new_tokens must be positive",
        ),
        (
            {"policy": {"generation": {"max_new_tokens": -1}}},
            "max_new_tokens must be positive",
        ),
        ({"token_capture": {"enabled": False}}, "token_capture.enabled=true"),
    ],
)
def test_acceptance_guard_rejects_unsupported_runtime_combinations(
    tmp_path, monkeypatch, capsys, overrides, error
):
    from omegaconf import OmegaConf

    spec = importlib.util.spec_from_file_location(
        "check_mm_coding_runtime", EXAMPLE / "check_mm_coding_runtime.py"
    )
    assert spec is not None and spec.loader is not None
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    config_path = tmp_path / "invalid.yaml"
    OmegaConf.save(
        OmegaConf.create(
            {"defaults": str(EXAMPLE / "mm_coding_mixed.yaml")} | overrides
        ),
        config_path,
    )
    monkeypatch.setattr(guard, "check_runtime", lambda **kwargs: [])
    monkeypatch.setattr(
        "sys.argv",
        [
            "check_mm_coding_runtime",
            "--mode",
            "mixed",
            "--config",
            str(EXAMPLE / "mm_coding_mixed.yaml"),
        ],
    )
    guard.main()
    assert json.loads(capsys.readouterr().out)["compatible"]
    with pytest.raises(ValueError, match=error):
        guard.check_training_config(config_path)


@pytest.mark.parametrize("mode", ["single", "multi", "mixed"])
def test_composed_recipe_loads_only_prepared_training_and_validation_rows(
    tmp_path, monkeypatch, mode
):
    from omegaconf import OmegaConf
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from transformers import PreTrainedTokenizerFast

    from nemo_rl.data.datasets.response_datasets import DATASET_REGISTRY
    from nemo_rl.data.utils import setup_response_data
    from nemo_rl.utils.config import load_config_with_inheritance

    class UnexpectedMathDataset:
        def __init__(self, **kwargs):
            raise AssertionError("Must not load an inherited math dataset")

    monkeypatch.setitem(DATASET_REGISTRY, "OpenMathInstruct-2", UnexpectedMathDataset)
    split_rows = {}
    for split in ("train", "validation"):
        rows = [
            adapter.gym_row(
                adapter.ExperimentTask(
                    benchmark="mmsvg_text2svg",
                    task_id=f"{split}-{i}",
                    prompt=f"{split} circle {i}",
                ),
                reference_root=tmp_path,
                split=split,
                selected="single"
                if mode == "single" or (mode == "mixed" and i == 0)
                else "opencode",
            )
            for i in range(2)
        ]
        path = tmp_path / f"{split}.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        monkeypatch.setenv(
            "MM_CODING_TRAIN_DATA" if split == "train" else "MM_CODING_VALIDATION_DATA",
            str(path),
        )
        split_rows[split] = rows
    config = load_config_with_inheritance(EXAMPLE / f"mm_coding_{mode}.yaml")
    data = OmegaConf.to_container(config.data, resolve=True)
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=Tokenizer(WordLevel({"[UNK]": 0}, unk_token="[UNK]")),
        unk_token="[UNK]",
    )
    train, validation = setup_response_data(tokenizer, deepcopy(data))
    assert [train[i]["extra_env_info"] for i in range(len(train))] == split_rows[
        "train"
    ]
    assert validation is not None
    assert [
        validation[i]["extra_env_info"] for i in range(len(validation))
    ] == split_rows["validation"]


def test_export_rejects_long_task_id_without_publishing_partial_rows(tmp_path):
    source, output = tmp_path / "tasks.jsonl", tmp_path / "rows.jsonl"
    source.write_text(
        json.dumps(
            {"benchmark": "mmsvg_text2svg", "task_id": "valid", "prompt": "draw"}
        )
        + "\n"
        + json.dumps(
            {"benchmark": "mmsvg_text2svg", "task_id": "x" * 257, "prompt": "draw"}
        )
        + "\n"
    )
    with pytest.raises(ValidationError, match="256"):
        adapter.export(source, output, reference_root=tmp_path, split="train")
    assert not output.exists()


@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        (["token_capture.enabled=false"], "token_capture.enabled=true"),
        (["policy.generation.backend=megatron"], "vLLM generation"),
        (["checkpointing.metric_name=accuracy"], "must start with"),
        (["policy.generation.max_new_tokens=0"], "must be positive"),
        (["policy.generation.max_new_tokens=-1"], "must be positive"),
    ],
)
def test_runtime_check_validates_training_cli_overrides(monkeypatch, overrides, error):
    spec = importlib.util.spec_from_file_location(
        "check_mm_coding_runtime", EXAMPLE / "check_mm_coding_runtime.py"
    )
    assert spec is not None and spec.loader is not None
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    monkeypatch.setattr(guard, "check_runtime", lambda **kwargs: [])
    monkeypatch.setattr(
        "sys.argv",
        [
            "check_mm_coding_runtime",
            "--mode",
            "mixed",
            "--config",
            str(EXAMPLE / "mm_coding_mixed.yaml"),
            *overrides,
        ],
    )
    with pytest.raises(ValueError, match=error):
        guard.main()


@pytest.mark.parametrize(
    "arguments",
    [
        ["token_capture.enabled=false"],
        ["--expected-checkpoint-dir", "/tmp/checkpoints"],
    ],
)
def test_runtime_check_rejects_overrides_without_a_recipe(
    monkeypatch: pytest.MonkeyPatch, arguments: list[str]
) -> None:
    spec = importlib.util.spec_from_file_location(
        "check_mm_coding_runtime", EXAMPLE / "check_mm_coding_runtime.py"
    )
    assert spec is not None and spec.loader is not None
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    monkeypatch.setattr("sys.argv", ["check_mm_coding_runtime", *arguments])
    with pytest.raises(SystemExit) as error:
        guard.main()
    assert error.value.code == 2


@pytest.mark.parametrize("mode", ["single", "multi", "mixed"])
@pytest.mark.parametrize(
    "override",
    [
        "checkpointing.checkpoint_dir={other}",
        "++checkpointing.checkpoint_dir={other}",
        "checkpointing={{checkpoint_dir:{other}}}",
        "+checkpointing={{checkpoint_dir:{other}}}",
        "++checkpointing={{checkpoint_dir:{other}}}",
        "checkpointing.checkpoint_dir=${{oc.env:MM_CODING_TEST_OTHER_CHECKPOINT}}",
    ],
)
def test_runtime_check_rejects_resolved_checkpoint_redirection(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str, override: str
) -> None:
    spec = importlib.util.spec_from_file_location(
        "check_mm_coding_runtime", EXAMPLE / "check_mm_coding_runtime.py"
    )
    assert spec is not None and spec.loader is not None
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    guarded, other = tmp_path / "guarded", tmp_path / "other"
    other.mkdir()
    (other / "step_7").mkdir()
    monkeypatch.setenv("MM_CODING_TEST_OTHER_CHECKPOINT", str(other))
    overrides = [
        f"checkpointing.checkpoint_dir={guarded}",
        override.format(other=other),
    ]
    with pytest.raises(ValueError, match="Use MM_CODING_CHECKPOINT_ROOT"):
        guard.check_training_config(
            EXAMPLE / f"mm_coding_{mode}.yaml",
            overrides,
            expected_checkpoint_dir=guarded,
        )
    assert (other / "step_7").is_dir()
    monkeypatch.setattr(guard, "check_runtime", lambda **kwargs: [])
    monkeypatch.setattr(
        "sys.argv",
        [
            "check_mm_coding_runtime",
            "--mode",
            mode,
            "--config",
            str(EXAMPLE / f"mm_coding_{mode}.yaml"),
            "--expected-checkpoint-dir",
            str(other),
            *overrides,
            "--expected-checkpoint-dir",
            str(guarded),
        ],
    )
    with pytest.raises(ValueError, match="Use MM_CODING_CHECKPOINT_ROOT"):
        guard.main()


@pytest.mark.parametrize("mode", ["single", "multi", "mixed"])
def test_runtime_check_compares_normalized_checkpoint_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mode: str
) -> None:
    spec = importlib.util.spec_from_file_location(
        "check_mm_coding_runtime", EXAMPLE / "check_mm_coding_runtime.py"
    )
    assert spec is not None and spec.loader is not None
    guard = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(guard)
    guarded = tmp_path / "guarded"
    guarded.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(guarded, target_is_directory=True)
    monkeypatch.setattr(guard, "check_runtime", lambda **kwargs: [])
    for equivalent in (guarded / ".." / "guarded", alias):
        monkeypatch.setattr(
            "sys.argv",
            [
                "check_mm_coding_runtime",
                "--mode",
                mode,
                "--config",
                str(EXAMPLE / f"mm_coding_{mode}.yaml"),
                f"checkpointing.checkpoint_dir={equivalent}",
                "checkpointing.save_period=2",
                "--expected-checkpoint-dir",
                str(guarded),
            ],
        )
        guard.main()
