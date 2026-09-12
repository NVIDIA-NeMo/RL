"""Compose the launch matrix without starting workers or loading checkpoints."""

import os
import json
from pathlib import Path
import shlex
import subprocess

from omegaconf import OmegaConf

from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)


def main() -> None:
    register_omegaconf_resolvers()
    root = Path.cwd()
    launcher = root / "experiments/precision_matrix_refresh_20260905/submit.sh"
    failures: list[str] = []
    performance = os.environ.get("PERFORMANCE_RECIPE") == "1"
    models = ("qwen30", "qwen235", "super") if performance else ("qwen30", "qwen235", "qwen35", "lightning")
    arms = ("bf16-bf16", "bf16-mxfp8", "mxfp8-false-mxfp8", "mxfp8-true-mxfp8") if performance else (
        "bf16-bf16", "bf16-mxfp8", "mxfp8-false-bf16",
        "mxfp8-false-mxfp8", "mxfp8-true-mxfp8", "mxfp8-true-bf16",
    )
    for model in models:
        for mode in ("sync", "async"):
            for arm in arms:
                case = f"{model}/{mode}/{arm}"
                try:
                    env = dict(os.environ, ACTION="render", REPO=str(root),
                               MODEL=model, MODE=mode, ARM=arm,
                               SLURM_ACCOUNT="coreai_dlalgo_nemorl", MAX_STEPS="20", TOPOLOGY="default")
                    env.pop("CONFIG_OVERRIDE", None)
                    output = subprocess.check_output(["bash", str(launcher)], env=env, text=True)
                    fields = dict(line.split("=", 1) for line in output.splitlines()
                                  if "=" in line and not line.startswith("overrides:"))
                    overrides = next(line.removeprefix("overrides:") for line in output.splitlines()
                                     if line.startswith("overrides:"))
                    cfg = parse_hydra_overrides(load_config(root / fields["config"]), shlex.split(overrides))
                    OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
                    if performance:
                        original = load_config(root / fields["config"])
                        protected = (
                            "grpo.num_prompts_per_step", "grpo.num_generations_per_prompt",
                            "policy.train_global_batch_size", "policy.train_micro_batch_size",
                            "policy.logprob_batch_size", "policy.logprob_chunk_size",
                            "policy.max_total_sequence_length", "data.max_input_seq_length",
                            "policy.generation.max_new_tokens",
                            "policy.megatron_cfg.tensor_model_parallel_size",
                            "policy.megatron_cfg.pipeline_model_parallel_size",
                            "policy.megatron_cfg.context_parallel_size",
                            "policy.megatron_cfg.expert_model_parallel_size",
                            "policy.megatron_cfg.moe_token_dispatcher_type",
                            "policy.megatron_cfg.moe_flex_dispatcher_backend",
                            "policy.generation.vllm_cfg.tensor_parallel_size",
                            "policy.generation.vllm_cfg.expert_parallel_size",
                            "policy.generation.vllm_cfg.gpu_memory_utilization",
                            "policy.generation.colocated", "cluster",
                        )
                        for key in protected:
                            assert OmegaConf.select(cfg, key) == OmegaConf.select(original, key), key
                        assert int(fields["nodes"]) == cfg.cluster.num_nodes
                        assert int(fields["segment"]) == cfg.cluster.segment_size
                        summary = {key: OmegaConf.select(cfg, key) for key in protected[:8]}
                        print(f"WORKLOAD {case} {json.dumps(summary)}", flush=True)
                        destination = Path(os.environ.get("PREFLIGHT_OUTPUT", "/results"))
                        destination.mkdir(parents=True, exist_ok=True)
                        OmegaConf.save(cfg, destination / f"{model}-{mode}-{arm}.yaml", resolve=True)
                    assert cfg.grpo.max_num_steps == 20
                    assert cfg.policy.generation.vllm_kwargs.moe_backend == "flashinfer_trtllm"
                    assert cfg.policy.generation.vllm_cfg.enforce_eager is False
                    assert cfg.loss_fn.reference_policy_kl_penalty > 0
                    assert not cfg.grpo.get("skip_reference_policy_logprobs_calculation", False)
                    assert cfg.loss_fn.use_importance_sampling_correction
                    assert cfg.loss_fn.force_on_policy_ratio is False
                    assert cfg.grpo.val_period == 0
                    if arm == "mxfp8-true-bf16":
                        assert cfg.policy.megatron_cfg.fp8_cfg.fp8_param is True
                        assert cfg.policy.generation.vllm_cfg.precision == "bfloat16"
                        assert cfg.policy.generation.vllm_cfg.is_mx is False
                    if mode == "async":
                        assert cfg.policy.generation.refit_transport == "nccl_reshard"
                        assert cfg.grpo.async_grpo.max_trajectory_age_steps == 1
                    print(f"PASS {case}", flush=True)
                except Exception as exc:
                    failures.append(case)
                    print(f"FAIL {case}: {type(exc).__name__}: {exc}", flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} configuration failures: {failures}")
    count = len(models) * 2 * len(arms)
    print(f"{count}/{count} configurations composed. No model execution was performed.")


if __name__ == "__main__":
    main()
