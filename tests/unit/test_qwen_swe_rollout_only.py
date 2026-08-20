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
import runpy
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
from omegaconf import OmegaConf

from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

REPO_ROOT = Path(__file__).resolve().parents[2]
RECIPE = (
    REPO_ROOT
    / "examples/configs/recipes/llm/grpo-qwen3-30ba3b-thinking-swe1-2n4g-megatron-tp2pp2-rollout-only-specdec.yaml"
)
LAUNCHER = REPO_ROOT / "examples/nemo_gym/run_qwen3_swe_rollout_only.sh"
ACTOR_REGISTRY = REPO_ROOT / "nemo_rl/distributed/ray_actor_environment_registry.py"


def _launcher_env(
    tmp_path: Path, method: str, num_speculative_tokens: str
) -> dict[str, str]:
    inherited_env = os.environ.copy()
    inherited_env.pop("WANDB_PROJECT", None)
    return {
        **inherited_env,
        "DRY_RUN": "1",
        "HF_HOME": str(tmp_path / "hf-cache"),
        "NRL_TARGET_MODEL": str(tmp_path / "target"),
        "NRL_DRAFT_MODEL": str(tmp_path / "draft"),
        "NRL_RUNTIME": str(tmp_path / "runtime"),
        "NRL_OUTPUT_DIR": str(tmp_path / "output"),
        "NRL_SPEC_METHOD": method,
        "NRL_NUM_SPECULATIVE_TOKENS": num_speculative_tokens,
    }


def test_rollout_only_recipe_uses_two_gb200_nodes_and_required_spec_config(
    monkeypatch,
) -> None:
    monkeypatch.setenv("HF_HOME", "/hf-cache")
    monkeypatch.setenv("NRL_SPEC_METHOD", "dflash")
    monkeypatch.setenv("NRL_DRAFT_MODEL", "/draft")
    monkeypatch.setenv("NRL_NUM_SPECULATIVE_TOKENS", "7")
    register_omegaconf_resolvers()
    config = OmegaConf.to_container(load_config(RECIPE), resolve=True)

    assert config["checkpointing"]["enabled"] is False
    assert config["policy"]["optimizer"] is None
    assert config["policy"]["scheduler"] is None
    megatron = config["policy"]["megatron_cfg"]
    assert megatron["tensor_model_parallel_size"] == 2
    assert megatron["pipeline_model_parallel_size"] == 2
    assert megatron["context_parallel_size"] == 1
    assert megatron["expert_model_parallel_size"] == 2
    generation = config["policy"]["generation"]
    assert generation["colocated"]["enabled"] is False
    assert generation["colocated"]["resources"] == {
        "gpus_per_node": 4,
        "num_nodes": 1,
    }
    assert generation["vllm_cfg"]["tensor_parallel_size"] == 2
    assert generation["vllm_cfg"]["enable_vllm_metrics_logger"] is True
    assert generation["vllm_kwargs"]["speculative_config"] == {
        "method": "dflash",
        "model": "/draft",
        "num_speculative_tokens": 7,
        "draft_tensor_parallel_size": 1,
    }
    assert config["env"]["nemo_gym"]["is_trajectory_collection"] is True
    assert config["cluster"]["gpus_per_node"] == 4
    assert config["cluster"]["num_nodes"] == 2
    assert config["cluster"]["segment_size"] == 1
    assert config["logger"]["tensorboard_enabled"] is False
    assert config["logger"]["wandb"]["project"] == "nemo-rl"

    policy_nodes = (
        config["cluster"]["num_nodes"]
        - generation["colocated"]["resources"]["num_nodes"]
    )
    policy_world_size = policy_nodes * config["cluster"]["gpus_per_node"]
    model_parallel_world_size = megatron["pipeline_model_parallel_size"] * max(
        megatron["tensor_model_parallel_size"] * megatron["context_parallel_size"],
        megatron["expert_model_parallel_size"]
        * megatron["expert_tensor_parallel_size"],
    )
    assert policy_world_size % model_parallel_world_size == 0
    assert policy_nodes % config["cluster"]["segment_size"] == 0
    assert (
        generation["colocated"]["resources"]["num_nodes"]
        % config["cluster"]["segment_size"]
        == 0
    )


@pytest.mark.parametrize(
    ("method", "num_speculative_tokens"),
    [
        ("dflash", 7),
        ("dflash", 15),
        ("dspark", 8),
        ("dspark", 16),
    ],
)
def test_launcher_uses_explicit_speculative_horizon(
    tmp_path: Path,
    method: str,
    num_speculative_tokens: int,
) -> None:
    target = tmp_path / "target"
    draft = tmp_path / "draft"
    runtime = tmp_path / "runtime"
    target.mkdir()
    draft.mkdir()
    runtime.mkdir()
    env = _launcher_env(tmp_path, method, str(num_speculative_tokens))

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert f"NRL_NUM_SPECULATIVE_TOKENS={num_speculative_tokens}" in result.stdout
    assert (
        "uv run --frozen --no-sync examples/nemo_gym/run_grpo_nemo_gym.py"
        in result.stdout
    )
    assert f"UV_PROJECT_ENVIRONMENT={runtime}" in result.stdout
    assert "NEMO_RL_PY_EXECUTABLES_SYSTEM=1" in result.stdout
    assert "XDG_CACHE_HOME=/tmp/nemorl-specdec-" in result.stdout
    assert "TRITON_CACHE_DIR=" + "/tmp/nemorl-specdec-" in result.stdout
    assert "TORCHINDUCTOR_CACHE_DIR=" + "/tmp/nemorl-specdec-" in result.stdout
    assert "logger.wandb.project=nemo-rl" in result.stdout
    assert not (tmp_path / "output").exists()
    env["WANDB_PROJECT"] = "team-project"
    overridden = subprocess.run(
        ["bash", str(LAUNCHER)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert overridden.returncode == 0, overridden.stderr
    assert "logger.wandb.project=team-project" in overridden.stdout


@pytest.mark.parametrize(
    ("method", "num_speculative_tokens"),
    [("eagle3", "8"), ("dflash", "65"), ("dspark", "0")],
)
def test_launcher_rejects_unsupported_method_or_speculative_horizon(
    tmp_path: Path, method: str, num_speculative_tokens: str
) -> None:
    env = _launcher_env(tmp_path, method, num_speculative_tokens)

    result = subprocess.run(
        ["bash", str(LAUNCHER)],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "unsupported" in result.stderr.lower()


def test_system_executable_override_covers_every_registered_actor(
    monkeypatch,
) -> None:
    class FakeExecutables:
        SYSTEM = "/shared/runtime/bin/python"
        VLLM = "uv vllm"
        SGLANG = "uv sglang"
        MCORE = "uv mcore"
        TRTLLM = "uv trtllm"
        FSDP = "uv fsdp"
        AUTOMODEL = "uv automodel"
        NEMO_GYM = "uv nemo-gym"

    virtual_cluster = ModuleType("nemo_rl.distributed.virtual_cluster")
    virtual_cluster.PY_EXECUTABLES = FakeExecutables
    modelopt_registry = ModuleType("nemo_rl.modelopt.registry")
    modelopt_registry.MODELOPT_ACTOR_REGISTRY = {
        "modelopt.Actor": FakeExecutables.SYSTEM
    }
    monkeypatch.setitem(
        sys.modules, "nemo_rl.distributed.virtual_cluster", virtual_cluster
    )
    monkeypatch.setitem(sys.modules, "nemo_rl.modelopt.registry", modelopt_registry)
    monkeypatch.setenv("NEMO_RL_PY_EXECUTABLES_SYSTEM", "1")

    namespace = runpy.run_path(str(ACTOR_REGISTRY))

    assert set(namespace["ACTOR_ENVIRONMENT_REGISTRY"].values()) == {
        FakeExecutables.SYSTEM
    }
