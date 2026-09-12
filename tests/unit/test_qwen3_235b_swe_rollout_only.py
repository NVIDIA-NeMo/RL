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

import os
import re
import subprocess
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

REPO_ROOT = Path(__file__).resolve().parents[2]
RECIPE = (
    REPO_ROOT
    / "examples/configs/recipes/llm/grpo-qwen3-235b-thinking-swe2-3n4g-megatron-tp4-rollout-only-specdec.yaml"
)
LAUNCHER = REPO_ROOT / "examples/nemo_gym/run_qwen3_235b_swe_rollout_only.sh"


def _config(monkeypatch: pytest.MonkeyPatch) -> dict:
    monkeypatch.setenv("NRL_SPEC_METHOD", "dflash")
    monkeypatch.setenv("NRL_DRAFT_MODEL", "/models/draft")
    monkeypatch.setenv("NRL_NUM_SPECULATIVE_TOKENS", "7")
    monkeypatch.setenv("NRL_SWE_TRAIN_DATA", "/data/swe/train.jsonl")
    monkeypatch.setenv("NRL_SWE_VAL_DATA", "/data/swe/val.jsonl")
    monkeypatch.setenv(
        "NRL_SWE_CONTAINER_FORMATTER",
        "/images/swe-bench.eval.arm64.{instance_id}.sif",
    )
    register_omegaconf_resolvers()
    return OmegaConf.to_container(load_config(RECIPE), resolve=True)


def _launcher_env(
    tmp_path: Path, method: str, num_speculative_tokens: int
) -> dict[str, str]:
    inherited_env = os.environ.copy()
    inherited_env.pop("WANDB_RUN_NAME", None)
    inherited_env.pop("WANDB_PROJECT", None)
    inherited_env.pop("SLURM_JOB_NUM_NODES", None)
    return {
        **inherited_env,
        "DRY_RUN": "1",
        "HF_HOME": str(tmp_path / "hf-cache"),
        "NRL_TARGET_MODEL": str(tmp_path / "target"),
        "NRL_DRAFT_MODEL": str(tmp_path / "draft"),
        "NRL_RUNTIME": str(tmp_path / "runtime"),
        "NRL_OUTPUT_DIR": str(tmp_path / "output"),
        "NRL_SWE_TRAIN_DATA": str(tmp_path / "train.jsonl"),
        "NRL_SWE_VAL_DATA": str(tmp_path / "val.jsonl"),
        "NRL_SWE_CONTAINER_FORMATTER": str(
            tmp_path / "images/swe-bench.eval.arm64.{instance_id}.sif"
        ),
        "NRL_SPEC_METHOD": method,
        "NRL_NUM_SPECULATIVE_TOKENS": str(num_speculative_tokens),
        "NRL_SLURM_SEGMENT": "3",
        "SLURM_JOB_ID": "12345",
    }


def test_recipe_resolves_agentic_swe_and_thinking_chat_template(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(monkeypatch)

    assert config["checkpointing"]["enabled"] is False
    assert config["policy"]["model_name"] == "Qwen/Qwen3-235B-A22B-Thinking-2507"
    assert config["policy"]["tokenizer"]["name"] == config["policy"]["model_name"]
    assert config["policy"]["max_total_sequence_length"] == 196608
    assert config["data"]["train"]["data_path"] == "/data/swe/train.jsonl"
    assert config["data"]["validation"]["data_path"] == "/data/swe/val.jsonl"
    assert config["logger"]["tensorboard_enabled"] is False
    assert config["logger"]["wandb"]["project"] == "nemo-rl"

    generation = config["policy"]["generation"]
    serving = generation["vllm_cfg"]["http_server_serving_chat_kwargs"]
    assert serving["enable_auto_tools"] is True
    assert serving["tool_parser"] == "hermes"
    assert serving["reasoning_parser"] == "deepseek_r1"
    assert "reasoning_content" in serving["chat_template"]
    assert serving["default_chat_template_kwargs"] == {
        "enable_thinking": True,
        "truncate_history_thinking": False,
    }

    nemo_gym = config["env"]["nemo_gym"]
    assert nemo_gym["is_trajectory_collection"] is True
    assert (
        "responses_api_agents/swe_agents/configs/swebench_openhands_training.yaml"
        in nemo_gym["config_paths"]
    )
    assert (
        nemo_gym["swe_agents_train"]["responses_api_agents"]["swe_agents"][
            "run_with_mixed_prompts"
        ]
        is True
    )
    expected_formatter = ["/images/swe-bench.eval.arm64.{instance_id}.sif"]
    assert (
        nemo_gym["swe_agents_train"]["responses_api_agents"]["swe_agents"][
            "container_formatter"
        ]
        == expected_formatter
    )
    assert (
        nemo_gym["swe_agents_val"]["responses_api_agents"]["swe_agents"][
            "container_formatter"
        ]
        == expected_formatter
    )
    assert generation["vllm_kwargs"]["speculative_config"] == {
        "method": "dflash",
        "model": "/models/draft",
        "num_speculative_tokens": 7,
        "draft_tensor_parallel_size": 1,
    }


def test_recipe_resolves_three_node_policy_and_generation_topology(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = _config(monkeypatch)
    cluster = config["cluster"]
    policy = config["policy"]
    megatron = policy["megatron_cfg"]
    generation = policy["generation"]
    resources = generation["colocated"]["resources"]
    vllm = generation["vllm_cfg"]

    assert cluster["gpus_per_node"] == 4
    assert cluster["num_nodes"] == 3
    assert cluster["segment_size"] == 1
    assert {
        "tensor_model_parallel_size": megatron["tensor_model_parallel_size"],
        "pipeline_model_parallel_size": megatron["pipeline_model_parallel_size"],
        "context_parallel_size": megatron["context_parallel_size"],
        "expert_model_parallel_size": megatron["expert_model_parallel_size"],
        "expert_tensor_parallel_size": megatron["expert_tensor_parallel_size"],
    } == {
        "tensor_model_parallel_size": 4,
        "pipeline_model_parallel_size": 1,
        "context_parallel_size": 1,
        "expert_model_parallel_size": 4,
        "expert_tensor_parallel_size": 1,
    }
    assert megatron["num_layers_in_first_pipeline_stage"] is None
    assert megatron["num_layers_in_last_pipeline_stage"] is None
    assert megatron["activation_checkpointing"] is False
    assert policy["make_sequence_length_divisible_by"] == 4

    assert generation["colocated"]["enabled"] is False
    assert resources == {"gpus_per_node": 4, "num_nodes": 2}
    assert vllm["tensor_parallel_size"] == 8
    assert vllm["pipeline_parallel_size"] == 1
    assert vllm["expert_parallel_size"] == 1
    assert vllm["max_model_len"] == 196608
    assert generation["max_new_tokens"] == 196608

    policy_nodes = cluster["num_nodes"] - resources["num_nodes"]
    policy_world_size = policy_nodes * cluster["gpus_per_node"]
    dense_parallel_size = (
        megatron["tensor_model_parallel_size"]
        * megatron["pipeline_model_parallel_size"]
        * megatron["context_parallel_size"]
    )
    expert_parallel_size = (
        megatron["expert_model_parallel_size"]
        * megatron["expert_tensor_parallel_size"]
        * megatron["pipeline_model_parallel_size"]
    )
    generation_world_size = resources["num_nodes"] * resources["gpus_per_node"]
    vllm_model_parallel_size = (
        vllm["tensor_parallel_size"] * vllm["pipeline_parallel_size"]
    )

    assert policy_world_size == 4
    assert policy_world_size % dense_parallel_size == 0
    assert policy_world_size % expert_parallel_size == 0
    assert generation_world_size == 8
    assert generation_world_size // vllm_model_parallel_size == 1
    assert policy_nodes % cluster["segment_size"] == 0


@pytest.mark.parametrize(
    ("method", "num_speculative_tokens"),
    [
        ("dflash", 7),
        ("dflash", 15),
        ("dspark", 8),
        ("dspark", 16),
    ],
)
def test_launcher_uses_explicit_speculative_horizon_and_external_segment_contract(
    tmp_path: Path,
    method: str,
    num_speculative_tokens: int,
) -> None:
    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        env=_launcher_env(tmp_path, method, num_speculative_tokens),
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert f"NRL_NUM_SPECULATIVE_TOKENS={num_speculative_tokens}" in result.stdout
    assert "NRL_SLURM_SEGMENT=3" in result.stdout
    assert "external allocation contract: sbatch --nodes=3 --segment=3" in result.stdout
    assert (
        "uv run --frozen --no-sync examples/nemo_gym/run_grpo_nemo_gym.py"
        in result.stdout
    )
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize(
    ("method", "num_speculative_tokens", "segment"),
    [
        ("eagle3", "8", "3"),
        ("dflash", "65", "3"),
        ("dspark", "0", "3"),
        ("dflash", "8", "2"),
    ],
)
def test_launcher_rejects_unsupported_spec_or_external_segment(
    tmp_path: Path,
    method: str,
    num_speculative_tokens: str,
    segment: str,
) -> None:
    env = _launcher_env(tmp_path, method, int(num_speculative_tokens))
    env["NRL_SLURM_SEGMENT"] = segment

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "unsupported" in result.stderr.lower()


def test_launcher_uses_shared_runtime_node_local_caches_and_unique_wandb_names(
    tmp_path: Path,
) -> None:
    env = _launcher_env(tmp_path, "dflash", 8)
    first = subprocess.run(
        ["bash", str(LAUNCHER)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    second = subprocess.run(
        ["bash", str(LAUNCHER)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    overridden_env = _launcher_env(tmp_path, "dflash", 8)
    overridden_env["WANDB_PROJECT"] = "team-project"
    overridden = subprocess.run(
        ["bash", str(LAUNCHER)],
        env=overridden_env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert first.returncode == 0, first.stderr
    assert second.returncode == 0, second.stderr
    assert overridden.returncode == 0, overridden.stderr
    assert "logger.wandb.project=team-project" in overridden.stdout
    system_executable = "NEMO_RL_PY_" + "EXECUTABLES_SYSTEM=1"
    xdg_cache = "XDG_CACHE_HOME=" + "/tmp/nemorl-qwen3-235b-"
    triton_cache = "TRITON_CACHE_DIR=" + "/tmp/nemorl-qwen3-235b-"
    torchinductor_cache = "TORCHINDUCTOR_CACHE_DIR=" + "/tmp/nemorl-qwen3-235b-"
    for output in (first.stdout, second.stdout):
        assert f"UV_PROJECT_ENVIRONMENT={tmp_path / 'runtime'}" in output
        assert system_executable in output
        assert xdg_cache in output
        assert triton_cache in output
        assert torchinductor_cache in output
        assert "logger.wandb.project=nemo-rl" in output

    pattern = re.compile(r"logger\.wandb\.name=([^ ]+)")
    first_name = pattern.search(first.stdout)
    second_name = pattern.search(second.stdout)
    assert first_name is not None
    assert second_name is not None
    assert first_name.group(1) != second_name.group(1)
