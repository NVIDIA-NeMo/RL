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
from pathlib import Path

import pytest
from omegaconf import OmegaConf
from PIL import Image
from pydantic import ValidationError

from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

ROOT = Path(__file__).resolve().parents[3]
EXAMPLE = ROOT / "examples/nemo_gym/nemotron-3.5-super-vl"
spec = importlib.util.spec_from_file_location(
    "prepare_mm_coding_single", EXAMPLE / "prepare_mm_coding_single.py"
)
assert spec is not None and spec.loader is not None
adapter = importlib.util.module_from_spec(spec)
spec.loader.exec_module(adapter)


def test_resolved_recipe_preserves_super_image_and_gym_contract(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("MM_CODING_TRAIN_DATA", "/approved/train.jsonl")
    monkeypatch.setenv("MM_CODING_VALIDATION_DATA", "/approved/validation.jsonl")
    register_omegaconf_resolvers()
    config = load_config(EXAMPLE / "mm_coding_single.yaml")
    assert (
        config.policy.model_name == "nvidia/NVIDIA-Nemotron-3.5-Super-VL-120B-A12B-BF16"
    )
    assert config.policy.tokenizer.name == config.policy.model_name
    assert (
        config.policy.automodel_cfg.enabled and not config.policy.megatron_cfg.enabled
    )
    assert (
        config.policy.automodel_cfg.tensor_parallel_size
        == config.policy.automodel_cfg.context_parallel_size
        == 1
    )
    assert not config.policy.automodel_cfg.sequence_parallel
    assert not config.policy.sequence_packing.enabled
    assert (
        config.policy.train_global_batch_size
        == config.grpo.num_prompts_per_step * config.grpo.num_generations_per_prompt
        == 64
    )
    assert config.grpo.max_num_steps == 2 and config.checkpointing.save_optimizer
    assert config.env.should_mask_flagged_samples
    assert config.env.nemo_gym.policy_model.responses_api_models.vllm_model.uses_reasoning_parser
    assert config.policy.generation.vllm_cfg.skip_tokenizer_init is False
    assert config.policy.generation.vllm_kwargs.mm_processor_cache_gb == 0
    assert config.data.default.processor == "nemo_gym_data_processor"
    assert config.data.default.prompt_file is None
    assert config.data.train.data_path == "/approved/train.jsonl"
    assert config.data.validation.data_path == "/approved/validation.jsonl"
    assert "workplace_assistant_simple_agent" not in OmegaConf.to_yaml(config.env)


def test_data_paths_are_explicit_not_example_or_benchmark_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("MM_CODING_TRAIN_DATA", raising=False)
    config = load_config(EXAMPLE / "mm_coding_single.yaml")
    with pytest.raises(Exception, match="MM_CODING_TRAIN_DATA"):
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


@pytest.mark.parametrize("mode", ["multi", "mixed"])
def test_multi_recipes_share_learner_and_reward(monkeypatch, mode):
    monkeypatch.setenv("MM_CODING_TRAIN_DATA", "/approved/train.jsonl")
    monkeypatch.setenv("MM_CODING_VALIDATION_DATA", "/approved/validation.jsonl")
    config = load_config(EXAMPLE / f"mm_coding_{mode}.yaml")
    single = load_config(EXAMPLE / "mm_coding_single.yaml")
    assert OmegaConf.to_container(config.policy) == OmegaConf.to_container(
        single.policy
    )
    assert config.grpo == single.grpo
    assert config.env.should_mask_flagged_samples
    assert any(
        f"mm_coding_{mode}.yaml" in path for path in config.env.nemo_gym.config_paths
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


@pytest.mark.parametrize("harnesses", [(), ("codex", "codex"), ("unknown",)])
def test_unknown_or_repeated_harness_routes_are_rejected(harnesses):
    task = adapter.ExperimentTask(
        benchmark="mmsvg_text2svg", task_id="1", prompt="draw"
    )
    with pytest.raises(ValueError, match="unique supported"):
        adapter.select_route(task, mode="mixed", harnesses=harnesses)
