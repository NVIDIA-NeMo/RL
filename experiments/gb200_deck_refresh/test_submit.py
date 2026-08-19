from __future__ import annotations

import os
import subprocess
from pathlib import Path


SCRIPT = Path(__file__).with_name("submit_oci_hsg.sh")


def render(**overrides: str) -> subprocess.CompletedProcess[str]:
    env = os.environ | {"ACTION": "render", **overrides}
    return subprocess.run(
        ["bash", str(SCRIPT)],
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
    assert "policy.generation.vllm_cfg.precision=fp8" in result.stdout
    assert "policy.generation.vllm_cfg.is_mx=true" in result.stdout
    assert "++policy.generation.vllm_kwargs.moe_backend=flashinfer_trtllm" in result.stdout
    assert "loss_fn.force_on_policy_ratio=false" in result.stdout
    assert "loss_fn.use_importance_sampling_correction=true" in result.stdout
    assert "grpo.skip_reference_policy_logprobs_calculation=false" in result.stdout


def test_submitter_uses_a_pinned_ray_runtime() -> None:
    script = SCRIPT.read_text()

    assert "RAY_RUNTIME_VENV" in script
    assert '"${RAY_RUNTIME_VENV}/READY"' in script
    assert "ray.__version__ == \"2.56.1\"" in script
    assert "export SETUP_COMMAND" in script


def test_async_bf16_renders_nccl_reshard_with_same_logprob_work() -> None:
    result = render(MODE="async", MODEL="nano", ARM="bf16")

    assert result.returncode == 0, result.stderr
    assert "policy.generation.colocated.enabled=false" in result.stdout
    assert "policy.generation.refit_transport=nccl_reshard" in result.stdout
    assert "policy.generation.vllm_cfg.precision=bfloat16" in result.stdout
    assert "loss_fn.force_on_policy_ratio=false" in result.stdout
    assert "grpo.skip_reference_policy_logprobs_calculation=false" in result.stdout


def test_qwen235_legacy_arm_uses_current_mxfp8_recipe_without_reshard() -> None:
    result = render(MODE="sync", MODEL="qwen235", ARM="mxfp8_legacy")

    assert result.returncode == 0, result.stderr
    assert "grpo-qwen3-235b-32n4g-async-1off-mxfp8-rollout.yaml" in result.stdout
    assert "grpo.async_grpo.enabled=false" in result.stdout
    assert "policy.generation.refit_transport=null" in result.stdout
    assert "cluster.num_nodes=32" in result.stdout


def test_invalid_async_legacy_combination_is_rejected() -> None:
    result = render(MODE="async", MODEL="qwen235", ARM="mxfp8_legacy")

    assert result.returncode != 0
    assert "not supported" in result.stderr
