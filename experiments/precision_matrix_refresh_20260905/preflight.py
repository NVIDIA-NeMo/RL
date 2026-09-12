"""Compose the launch matrix without starting workers or loading checkpoints."""

import os
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
    for model in ("qwen30", "qwen235", "qwen35", "lightning"):
        for mode in ("sync", "async"):
            for arm in (
                "bf16-bf16", "bf16-mxfp8", "mxfp8-false-bf16",
                "mxfp8-false-mxfp8", "mxfp8-true-mxfp8",
            ):
                case = f"{model}/{mode}/{arm}"
                try:
                    env = dict(os.environ, ACTION="render", REPO=str(root),
                               MODEL=model, MODE=mode, ARM=arm,
                               SLURM_ACCOUNT="coreai_dlalgo_nemorl", MAX_STEPS="20")
                    output = subprocess.check_output(["bash", str(launcher)], env=env, text=True)
                    fields = dict(line.split("=", 1) for line in output.splitlines()
                                  if "=" in line and not line.startswith("overrides:"))
                    overrides = next(line.removeprefix("overrides:") for line in output.splitlines()
                                     if line.startswith("overrides:"))
                    cfg = parse_hydra_overrides(load_config(root / fields["config"]), shlex.split(overrides))
                    OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True)
                    assert cfg.grpo.max_num_steps == 20
                    assert cfg.policy.generation.vllm_kwargs.moe_backend == "flashinfer_trtllm"
                    assert cfg.policy.generation.vllm_cfg.enforce_eager is False
                    assert cfg.loss_fn.reference_policy_kl_penalty > 0
                    assert not cfg.grpo.get("skip_reference_policy_logprobs_calculation", False)
                    assert cfg.loss_fn.use_importance_sampling_correction
                    if mode == "async":
                        assert cfg.policy.generation.refit_transport == "nccl_reshard"
                        assert cfg.grpo.async_grpo.max_trajectory_age_steps == 1
                    print(f"PASS {case}", flush=True)
                except Exception as exc:
                    failures.append(case)
                    print(f"FAIL {case}: {type(exc).__name__}: {exc}", flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} configuration failures: {failures}")
    print("40/40 configurations composed. No model execution was performed.")


if __name__ == "__main__":
    main()
