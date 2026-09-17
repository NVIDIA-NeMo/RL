"""Compose the matched performance matrix without allocating GPUs."""

import json
import os
from pathlib import Path
import shlex
import subprocess

from omegaconf import DictConfig, OmegaConf

from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)


MODELS = ("qwen30", "qwen235", "qwen35", "super")
MODES = ("sync", "async")
ARMS = (
    "bf16-bf16",
    "bf16-mxfp8",
    "mxfp8-false-mxfp8",
    "mxfp8-true-mxfp8",
)
PROTECTED_KEYS = (
    "grpo.num_prompts_per_step",
    "grpo.num_generations_per_prompt",
    "policy.train_global_batch_size",
    "policy.train_micro_batch_size",
    "policy.logprob_batch_size",
    "policy.logprob_chunk_size",
    "policy.max_total_sequence_length",
    "data.max_input_seq_length",
    "policy.generation.max_new_tokens",
    "policy.megatron_cfg.tensor_model_parallel_size",
    "policy.megatron_cfg.pipeline_model_parallel_size",
    "policy.megatron_cfg.context_parallel_size",
    "policy.megatron_cfg.expert_model_parallel_size",
    "policy.generation.vllm_cfg.tensor_parallel_size",
    "policy.generation.vllm_cfg.expert_parallel_size",
    "policy.generation.vllm_cfg.gpu_memory_utilization",
    "cluster",
)


def render_config(
    root: Path, model: str, mode: str, arm: str
) -> tuple[DictConfig, DictConfig, dict[str, str]]:
    launcher = root / "experiments/precision_matrix_refresh_20260905/submit.sh"
    env = dict(
        os.environ,
        ACTION="render",
        REPO=str(root),
        MODEL=model,
        MODE=mode,
        ARM=arm,
        TOPOLOGY="default",
        PERFORMANCE_RECIPE="1",
        SLURM_ACCOUNT="coreai_dlalgo_llm",
        MAX_STEPS="20",
    )
    output = subprocess.check_output(["bash", str(launcher)], env=env, text=True)
    fields = dict(
        line.split("=", 1)
        for line in output.splitlines()
        if "=" in line and not line.startswith("overrides:")
    )
    overrides = next(
        line.removeprefix("overrides:")
        for line in output.splitlines()
        if line.startswith("overrides:")
    )
    original = load_config(root / fields["config"])
    config = parse_hydra_overrides(original, shlex.split(overrides))
    OmegaConf.to_container(config, resolve=True, throw_on_missing=True)
    return config, original, fields


def validate_config(
    config: DictConfig,
    original: DictConfig,
    fields: dict[str, str],
    model: str,
    mode: str,
    arm: str,
) -> None:
    for key in PROTECTED_KEYS:
        assert OmegaConf.select(config, key) == OmegaConf.select(original, key), key

    assert config.grpo.max_num_steps == 20
    assert config.grpo.val_period == 0
    assert not config.grpo.skip_reference_policy_logprobs_calculation
    assert config.grpo.seq_logprob_error_threshold is None
    assert config.loss_fn.use_importance_sampling_correction
    assert config.loss_fn.force_on_policy_ratio is False
    assert config.loss_fn.reference_policy_kl_penalty == 0.01
    assert config.policy.generation.vllm_kwargs.moe_backend == "flashinfer_trtllm"
    assert config.policy.generation.vllm_kwargs.expert_placement_strategy == "linear"
    assert config.policy.generation.vllm_cfg.enforce_eager is False
    assert int(fields["nodes"]) == config.cluster.num_nodes
    assert int(fields["segment"]) == config.cluster.segment_size

    if model == "qwen35":
        assert config.grpo.num_prompts_per_step == 128
        assert config.grpo.num_generations_per_prompt == 16
        assert config.policy.train_global_batch_size == 2048

    training_mxfp8 = arm.startswith("mxfp8-")
    assert config.policy.megatron_cfg.fp8_cfg.enabled is training_mxfp8
    if training_mxfp8:
        expected_fp8_param = arm.startswith("mxfp8-true-")
        assert config.policy.megatron_cfg.fp8_cfg.fp8_param is expected_fp8_param
        assert config.policy.megatron_cfg.fp8_cfg.fp8_recipe == "mxfp8"
        assert config.policy.megatron_cfg.te_precision_config_file.endswith(
            "te_routed_fp8param.yaml" if expected_fp8_param else "te_routed.yaml"
        )

    rollout_mxfp8 = arm.endswith("-mxfp8")
    assert config.policy.generation.vllm_cfg.precision == (
        "fp8" if rollout_mxfp8 else "bfloat16"
    )
    assert config.policy.generation.vllm_cfg.is_mx is rollout_mxfp8
    if rollout_mxfp8:
        assert config.policy.generation.vllm_cfg.quantization_ignore_patterns

    if mode == "async":
        assert config.policy.generation.refit_transport == "nccl_reshard"
        assert config.grpo.async_grpo.max_trajectory_age_steps == 1
    else:
        assert config.policy.generation.colocated.enabled


def main() -> None:
    register_omegaconf_resolvers()
    root = Path.cwd()
    failures: list[str] = []
    output_root = Path(
        os.environ.get("PREFLIGHT_OUTPUT", "/tmp/precision-matrix-preflight")
    )
    output_root.mkdir(parents=True, exist_ok=True)

    for model in MODELS:
        for mode in MODES:
            for arm in ARMS:
                case = f"{model}/{mode}/{arm}"
                try:
                    config, original, fields = render_config(root, model, mode, arm)
                    validate_config(config, original, fields, model, mode, arm)
                    OmegaConf.save(
                        config,
                        output_root / f"{model}-{mode}-{arm}.yaml",
                        resolve=True,
                    )
                    summary = {
                        key: OmegaConf.select(config, key) for key in PROTECTED_KEYS[:7]
                    }
                    print(f"PASS {case} {json.dumps(summary)}", flush=True)
                except Exception as exc:
                    failures.append(case)
                    print(f"FAIL {case}: {type(exc).__name__}: {exc}", flush=True)

    if failures:
        raise SystemExit(f"{len(failures)} configuration failures: {failures}")
    print("32/32 configurations composed. No model execution was performed.")


if __name__ == "__main__":
    main()
