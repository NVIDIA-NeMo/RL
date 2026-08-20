from __future__ import annotations

import os
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).with_name("submit_oci_hsg.sh")
ABLATION_SCRIPT = Path(__file__).with_name("submit_pr3294_ablation_oci_hsg.sh")


def render(**overrides: str) -> subprocess.CompletedProcess[str]:
    env = os.environ | {"ACTION": "render", **overrides}
    return subprocess.run(
        ["bash", str(SCRIPT)],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )


def render_ablation(**overrides: str) -> subprocess.CompletedProcess[str]:
    env = os.environ | {"ACTION": "render", **overrides}
    return subprocess.run(
        ["bash", str(ABLATION_SCRIPT)],
        check=False,
        capture_output=True,
        env=env,
        text=True,
    )


def test_sync_mxfp8_renders_matched_two_logprob_cuda_graph_run() -> None:
    result = render(MODE="sync", MODEL="nano", ARM="mxfp8")

    assert result.returncode == 0, result.stderr
    assert "policy.generation.colocated.enabled=true" in result.stdout
    assert "policy.generation.refit_transport=null" in result.stdout
    assert "policy.generation.vllm_cfg.enforce_eager=false" in result.stdout
    assert "policy.generation.vllm_cfg.gpu_memory_utilization=0.5" in result.stdout
    assert "policy.generation.vllm_cfg.precision=fp8" in result.stdout
    assert "++policy.generation.vllm_cfg.is_mx=true" in result.stdout
    assert "++policy.generation.vllm_kwargs.moe_backend=flashinfer_trtllm" in result.stdout
    assert "loss_fn.force_on_policy_ratio=false" in result.stdout
    assert "loss_fn.use_importance_sampling_correction=true" in result.stdout
    assert "++grpo.skip_reference_policy_logprobs_calculation=false" in result.stdout


def test_gpu_memory_utilization_can_be_overridden() -> None:
    result = render(
        MODE="sync",
        MODEL="nano",
        ARM="bf16",
        VLLM_GPU_MEMORY_UTILIZATION="0.45",
    )

    assert result.returncode == 0, result.stderr
    assert "policy.generation.vllm_cfg.gpu_memory_utilization=0.45" in result.stdout


def test_submitter_uses_a_pinned_ray_runtime() -> None:
    script = SCRIPT.read_text()

    assert "RAY_RUNTIME_VENV" in script
    assert '"${RAY_RUNTIME_VENV}/READY"' in script
    assert "ray.__version__ == \"2.56.1\"" in script
    assert "export SETUP_COMMAND" in script
    assert "export UV_PYTHON=${RAY_RUNTIME_VENV}/bin/python" in script


def test_submitter_checks_top_level_and_submodule_revisions_separately() -> None:
    script = SCRIPT.read_text()

    assert "status --porcelain --untracked-files=no --ignore-submodules=all" in script
    assert "submodule status --recursive | grep -q '^[+-U]'" in script
    assert 'ls-remote origin "refs/heads/${BRANCH}"' in script
    assert 'rev-parse "origin/${BRANCH}"' not in script


def test_submitter_does_not_overlay_container_uv_cache() -> None:
    script = SCRIPT.read_text()

    assert "export UV_CACHE_DIR_OVERRIDE" not in script
    assert "unset UV_CACHE_DIR_OVERRIDE" in script
    assert "DRIVER_VENV=${DRIVER_VENV:-${CACHE_ROOT}/driver-venvs/" in script
    assert "DRIVER_PYTHON_ROOT=${DRIVER_PYTHON_ROOT:-${CACHE_ROOT}/driver-python/" in script
    assert "export UV_PROJECT_ENVIRONMENT=${DRIVER_VENV}" in script
    assert "export UV_PYTHON_INSTALL_DIR=${DRIVER_PYTHON_ROOT}" in script
    assert "trap 'rm -rf \"${DRIVER_VENV}\" \"${DRIVER_PYTHON_ROOT}\"' EXIT" in script


def test_async_bf16_renders_nccl_reshard_with_same_logprob_work() -> None:
    result = render(MODE="async", MODEL="nano", ARM="bf16")

    assert result.returncode == 0, result.stderr
    assert "policy.generation.colocated.enabled=false" in result.stdout
    assert "policy.generation.refit_transport=nccl_reshard" in result.stdout
    assert "policy.generation.vllm_cfg.async_engine=true" in result.stdout
    assert "policy.generation.vllm_cfg.refit_with_reload_api=true" in result.stdout
    assert "policy.generation.vllm_cfg.precision=bfloat16" in result.stdout
    assert "loss_fn.force_on_policy_ratio=false" in result.stdout
    assert "++grpo.skip_reference_policy_logprobs_calculation=false" in result.stdout


def test_qwen30_dapo_pair_keeps_workload_matched() -> None:
    bf16 = render(MODE="sync", MODEL="qwen30", ARM="bf16")
    mxfp8 = render(MODE="sync", MODEL="qwen30", ARM="mxfp8")

    assert bf16.returncode == 0, bf16.stderr
    assert mxfp8.returncode == 0, mxfp8.stderr
    for result in (bf16, mxfp8):
        assert "data.train.dataset_name=DAPOMath17K" in result.stdout
        assert "grpo.num_prompts_per_step=128" in result.stdout
        assert "grpo.num_generations_per_prompt=16" in result.stdout
        assert "policy.train_global_batch_size=2048" in result.stdout
        assert "policy.max_total_sequence_length=22528" in result.stdout
        assert "policy.megatron_cfg.defer_fp32_logits=true" in result.stdout
        assert "policy.logprob_chunk_size=2048" in result.stdout
        assert "policy.generation.vllm_cfg.enforce_eager=false" in result.stdout
        assert "loss_fn.force_on_policy_ratio=false" in result.stdout
        assert "++grpo.skip_reference_policy_logprobs_calculation=false" in result.stdout
        assert "grpo.val_period=0" in result.stdout
    assert "policy.generation.vllm_cfg.precision=bfloat16" in bf16.stdout
    assert "policy.generation.vllm_cfg.precision=fp8" in mxfp8.stdout


def test_qwen30_topology_can_be_scaled_without_changing_the_workload() -> None:
    result = render(
        MODE="sync",
        MODEL="qwen30",
        ARM="bf16",
        TOTAL_NODES_OVERRIDE="8",
        GEN_NODES_OVERRIDE="8",
    )

    assert result.returncode == 0, result.stderr
    assert "cluster.num_nodes=8" in result.stdout
    assert "policy.generation.colocated.resources.num_nodes=8" in result.stdout
    assert "policy.train_global_batch_size=2048" in result.stdout
    assert "policy.max_total_sequence_length=22528" in result.stdout


def test_qwen235_legacy_arm_uses_current_mxfp8_recipe_without_reshard() -> None:
    result = render(MODE="sync", MODEL="qwen235", ARM="mxfp8_legacy")

    assert result.returncode == 0, result.stderr
    assert "grpo-qwen3-235b-32n4g-async-1off-mxfp8-rollout.yaml" in result.stdout
    assert "grpo.async_grpo.enabled=false" in result.stdout
    assert "policy.generation.refit_transport=null" in result.stdout
    assert "policy.generation.vllm_cfg.gpu_memory_utilization=0.88" in result.stdout
    assert "++policy.generation.vllm_kwargs.max_num_seqs=32" in result.stdout
    assert "cluster.num_nodes=32" in result.stdout


def test_invalid_async_legacy_combination_is_rejected() -> None:
    result = render(MODE="async", MODEL="qwen235", ARM="mxfp8_legacy")

    assert result.returncode != 0
    assert "not supported" in result.stderr


def test_pr3294_baseline_disables_all_three_refit_optimizations() -> None:
    result = render_ablation(ARM="baseline")

    assert result.returncode == 0, result.stderr
    assert "policy.generation.vllm_cfg.refit_prequantize=false" in result.stdout
    assert "NRL_MXFP8_BATCHED_SHUFFLE=0" in result.stdout
    assert "NRL_REFIT_CACHED_LOADERS=0" in result.stdout
    assert "grpo.val_period=0" in result.stdout


def test_full_ablation_exports_runtime_toggles_outside_hydra() -> None:
    script = ABLATION_SCRIPT.read_text()
    result = render_ablation(ARM="baseline")

    assert result.returncode == 0, result.stderr
    assert (
        "++policy.generation.vllm_cfg.env_vars.VLLM_RAY_EXTRA_ENV_VARS_TO_COPY"
        not in result.stdout
    )
    assert "export NRL_MXFP8_BATCHED_SHUFFLE=${BATCHED_SHUFFLE}" in script
    assert "export NRL_REFIT_CACHED_LOADERS=${CACHED_LOADERS}" in script
    assert (
        "export VLLM_RAY_EXTRA_ENV_VARS_TO_COPY="
        "NRL_MXFP8_BATCHED_SHUFFLE,NRL_REFIT_CACHED_LOADERS" in script
    )


def test_pr3294_optimized_enables_all_three_refit_optimizations() -> None:
    result = render_ablation(ARM="optimized")

    assert result.returncode == 0, result.stderr
    assert "policy.generation.vllm_cfg.refit_prequantize=true" in result.stdout
    assert "NRL_MXFP8_BATCHED_SHUFFLE=1" in result.stdout
    assert "NRL_REFIT_CACHED_LOADERS=1" in result.stdout


def test_shuffle_only_changes_commit_but_keeps_prequantization_enabled() -> None:
    baseline = render_ablation(STUDY="shuffle_only", ARM="baseline")
    optimized = render_ablation(STUDY="shuffle_only", ARM="optimized")

    assert baseline.returncode == 0, baseline.stderr
    assert optimized.returncode == 0, optimized.stderr
    assert "++policy.generation.vllm_cfg.refit_prequantize=true" in baseline.stdout
    assert "++policy.generation.vllm_cfg.refit_prequantize=true" in optimized.stdout
    assert "NRL_MXFP8_BATCHED_SHUFFLE" not in baseline.stdout
    assert "NRL_MXFP8_BATCHED_SHUFFLE" not in optimized.stdout


def test_ablation_driver_uses_same_python_as_ray_cluster() -> None:
    script = ABLATION_SCRIPT.read_text()

    assert "export UV_PYTHON=${RAY_RUNTIME_VENV}/bin/python" in script
    assert "status --porcelain --untracked-files=no --ignore-submodules=all" in script
    assert "submodule status --recursive | grep -q '^[+-U]'" in script
    assert "unset UV_CACHE_DIR_OVERRIDE" in script
    assert "DRIVER_VENV=${DRIVER_VENV:-${CACHE_ROOT}/driver-venvs/" in script
    assert "DRIVER_PYTHON_ROOT=${DRIVER_PYTHON_ROOT:-${CACHE_ROOT}/driver-python/" in script
    assert "export UV_PROJECT_ENVIRONMENT=${DRIVER_VENV}" in script
    assert "export UV_PYTHON_INSTALL_DIR=${DRIVER_PYTHON_ROOT}" in script
    assert "trap 'rm -rf \"${DRIVER_VENV}\" \"${DRIVER_PYTHON_ROOT}\"' EXIT" in script
