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

import shlex
from pathlib import Path

import pytest
from omegaconf import DictConfig, OmegaConf

from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)

RECIPE_ROOT = Path(__file__).parents[2] / "examples/configs/recipes"
FUNCTIONAL_ROOT = Path(__file__).parents[2] / "tests/functional"
MEGATRON_FUNCTIONAL_SCRIPTS = sorted(
    {
        path.name
        for path in FUNCTIONAL_ROOT.glob("*.sh")
        if "policy.generation.backend=megatron" in path.read_text()
    }
    | {"grpo_megatron_generation_gym_single_controller.sh"}
)


def _load_megatron_functional_config(script_name: str) -> tuple[DictConfig, bool]:
    register_omegaconf_resolvers()
    project_root = Path(__file__).parents[2]
    script = (FUNCTIONAL_ROOT / script_name).read_text()
    commands = [
        shlex.split(line)
        for line in script.replace("\\\n", " ").splitlines()
        if line.lstrip().startswith("uv run ")
        and any(name in line for name in ("run_grpo", "run_vlm_grpo", "$SC_ENTRYPOINT"))
    ]
    arguments = next(
        args
        for args in commands
        if any(
            "run_grpo" in arg or "run_vlm_grpo" in arg or arg == "$SC_ENTRYPOINT"
            for arg in args
        )
    )
    if "--config" in arguments:
        config_path = arguments[arguments.index("--config") + 1].replace(
            "$PROJECT_ROOT", str(project_root)
        )
    else:
        assert any(arg.endswith("run_grpo_sliding_puzzle.py") for arg in arguments)
        config_path = str(project_root / "examples/configs/grpo_sliding_puzzle.yaml")
    config_path = project_root / config_path
    overrides = [
        argument
        for argument in arguments
        if argument.lstrip("+").startswith(
            (
                "policy.generation.backend=",
                "policy.generation.mcore_generation_config.kv_cache_management_mode=",
                "async_rl.",
                "grpo.async_grpo.",
            )
        )
    ]
    config = parse_hydra_overrides(load_config(config_path), overrides)
    single_controller = any(
        "run_grpo_single_controller" in arg or arg == "$SC_ENTRYPOINT"
        for arg in arguments
    )
    return config, single_controller


@pytest.mark.parametrize("script_name", MEGATRON_FUNCTIONAL_SCRIPTS)
def test_megatron_functional_scripts_keep_kv_settings_consistent(
    script_name: str,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    # Load algorithm schemas only for these tests; other recipe tests need no Ray/Torch.
    from nemo_rl.algorithms.grpo import AsyncGRPOConfig
    from nemo_rl.algorithms.single_controller_utils.config import AsyncRLConfig

    # CI runs from tests/, while the launchers resolve paths from the repo root.
    monkeypatch.chdir(tmp_path)
    config, single_controller = _load_megatron_functional_config(script_name)
    assert config.policy.generation.backend == "megatron"
    async_path = "async_rl" if single_controller else "grpo.async_grpo"
    raw_async_config = OmegaConf.select(config, async_path)
    async_values = (
        OmegaConf.to_container(raw_async_config, resolve=True)
        if raw_async_config is not None
        else {}
    )
    if single_controller:
        async_config = AsyncRLConfig.model_validate(async_values)
    else:
        if raw_async_config is None:
            return
        async_config = AsyncGRPOConfig.model_validate(async_values)
        if not async_config.enabled:
            return
    # Preserve intentional persist/offload opt-outs as well as the recompute default.
    assert async_config.recompute_kv_cache_after_weight_updates == (
        config.policy.generation.mcore_generation_config.kv_cache_management_mode
        == "recompute"
    )


def test_speechlm_recipe_logs_typed_config_to_tensorboard(tmp_path, monkeypatch):
    from omegaconf import OmegaConf

    from nemo_rl.algorithms.grpo import MasterConfig
    from nemo_rl.utils.logger import TensorboardConfig, TensorboardLogger

    for name in ("MODEL", "DATA", "OUTPUT_DIR"):
        monkeypatch.setenv(f"SPEECHLM_GSPO_{name}", str(tmp_path / name))
    register_omegaconf_resolvers()
    config = load_config(
        RECIPE_ROOT
        / "vlm/vlm_grpo-nemotron-speechlm-30ba3b-4n8g-automodel-ep8-gspo-cer400.v1.yaml"
    )
    params = MasterConfig(**OmegaConf.to_container(config, resolve=True)).model_dump()
    # The validated native config keeps the exact trainable surface immutable.
    assert isinstance(params["policy"]["native_model"]["trainable_names"], tuple)
    logger = TensorboardLogger(TensorboardConfig(), log_dir=str(tmp_path / "logs"))
    try:
        logger.log_hyperparams(params)
    finally:
        logger.finish()


@pytest.mark.automodel
def test_speechlm_recipe_uses_native_checkpoint_config(tmp_path, monkeypatch):
    from omegaconf import OmegaConf

    from nemo_rl.models.automodel.checkpoint import build_checkpoint_config
    from nemo_rl.utils.checkpoint import CheckpointManager

    for name in ("MODEL", "DATA", "OUTPUT_DIR"):
        monkeypatch.setenv(f"SPEECHLM_GSPO_{name}", str(tmp_path / name))
    register_omegaconf_resolvers()
    config = load_config(
        RECIPE_ROOT
        / "vlm/vlm_grpo-nemotron-speechlm-30ba3b-4n8g-automodel-ep8-gspo-cer400.v1.yaml"
    )
    resolved = OmegaConf.to_container(config, resolve=True)
    CheckpointManager(resolved["checkpointing"])
    checkpoint = build_checkpoint_config(
        resolved["policy"]["automodel_cfg"],
        model_repo_id=str(tmp_path / "MODEL"),
        dequantize_base_checkpoint=False,
        is_peft=False,
        is_async=False,
    )
    assert checkpoint["model_save_format"] == "torch_save"
    assert checkpoint["save_consolidated"] == "false"


@pytest.mark.parametrize(
    "recipe_name",
    [
        "llm/sft-nanov3-30BA3B-2n8g-automodel-ep8.yaml",
        "llm/sft-nanov3-30BA3B-2n4g-automodel-ep4.yaml",
        "llm/sft-gpt-oss-20b-1n8g-automodel-ep8.yaml",
        "llm/sft-gpt-oss-20b-1n4g-automodel-ep4.yaml",
        "vlm/vlm_grpo-nemotron-omni-30ba3b-clevr-1n8g-automodel-ep8.v2.yaml",
        "vlm/vlm_grpo-nemotron-omni-30ba3b-mmpr-4n8g-automodel-ep8.v1.yaml",
    ],
)
def test_hybridep_regression_recipes_keep_legacy_deepep(recipe_name: str) -> None:
    register_omegaconf_resolvers()
    config = load_config(RECIPE_ROOT / recipe_name)
    assert config.policy.automodel_cfg.automodel_kwargs.backend.dispatcher == "deepep"
    assert config.policy.make_sequence_length_divisible_by == 1


@pytest.mark.parametrize("context_parallel_size", [2, 4])
def test_gemma4_cp_keeps_local_hybridep_inputs_aligned(
    context_parallel_size: int,
) -> None:
    register_omegaconf_resolvers()
    config = load_config(
        RECIPE_ROOT / "llm/dapo-gemma4-26ba4b-it-4n8g-automodel-ep16cp2.yaml"
    )
    config.policy.automodel_cfg.context_parallel_size = context_parallel_size
    global_alignment = 64 * config.policy.automodel_cfg.context_parallel_size
    assert config.policy.make_sequence_length_divisible_by == global_alignment
    assert config.policy.dynamic_batching.sequence_length_round == global_alignment


@pytest.mark.parametrize(
    ("recipe_name", "dispatcher", "experts"),
    [
        ("llm/dpo-nanov3-30B3AB-1n4g-automodel-ep4.yaml", "torch", "torch_mm"),
        ("llm/grpo-glm47-flash-4n8g-automodel-ep32.yaml", "hybridep", "torch_mm"),
        ("llm/grpo-minimax-m27-dapo-8n8g-automodel-ep64.yaml", "hybridep", "torch_mm"),
        ("llm/grpo-moonlight-16b-automodel-1n8g-ep8.yaml", "hybridep", "torch_mm"),
        (
            "llm/grpo-nemotron3-super-120BA12B-16n8g-automodel-ep8.v2.yaml",
            "hybridep",
            "torch_mm",
        ),
        ("llm/dapo-gemma4-26ba4b-it-4n8g-automodel-ep32.yaml", "hybridep", "gmm"),
        ("llm/dapo-nanov3.5-30BA3B-4n8g-automodel-ep8.yaml", "hybridep", "gmm"),
        ("llm/grpo-qwen3.5-35ba3b-2n8g-automodel-ep16.yaml", "hybridep", "gmm"),
        ("llm/grpo-qwen3.5-35ba3b-dapo-4n8g-automodel-ep32.yaml", "hybridep", "gmm"),
        (
            "vlm/vlm_grpo-nemotron-omni-30ba3b-clevr-1n8g-automodel-ep8.v2.yaml",
            "deepep",
            "torch_mm",
        ),
        (
            "vlm/vlm_grpo-nemotron-omni-30ba3b-mmpr-4n8g-automodel-ep8.v1.yaml",
            "deepep",
            "torch_mm",
        ),
        (
            "vlm/vlm_grpo-qwen3.5-35ba3b-geo3k-2n8g-automodel-ep16.yaml",
            "hybridep",
            "gmm",
        ),
        ("vlm/vlm_grpo-gemma4-e4b-geo3k-1n8g-automodel.yaml", "hybridep", "gmm"),
    ],
)
def test_automodel_recipes_preserve_backend_choices(
    recipe_name: str, dispatcher: str, experts: str
) -> None:
    register_omegaconf_resolvers()
    config = load_config(RECIPE_ROOT / recipe_name)
    backend = config.policy.automodel_cfg.automodel_kwargs.backend
    assert backend.dispatcher == dispatcher
    assert backend.experts == experts


@pytest.mark.parametrize(
    ("recipe_name", "expected_async_save"),
    [
        ("grpo-deepseek-v3-32n4g.yaml", False),
        ("grpo-deepseek-v3-64n4g.yaml", True),
        ("grpo-deepseek-v3-64n4g-mxfp8-rollout.yaml", True),
    ],
)
def test_deepseek_performance_recipes_preserve_checkpoint_save_mode(
    recipe_name: str, expected_async_save: bool
) -> None:
    register_omegaconf_resolvers()
    config = load_config(RECIPE_ROOT / "llm/performance" / recipe_name)
    assert config.policy.megatron_cfg.checkpoint.async_save is expected_async_save
