#!/usr/bin/env python3
"""Validate the vLLM 0.24 environment and NeMo-RL integration imports."""

from __future__ import annotations

import inspect
import json
import os
import platform
import sys
from importlib.metadata import version

import torch
import vllm
from vllm.entrypoints.openai.chat_completion.serving import OpenAIServingChat
from vllm.entrypoints.serve.render.serving import OpenAIServingRender
from vllm.entrypoints.serve.tokenize.serving import ServingTokenization

from nemo_rl.models.generation.vllm import patches
from nemo_rl.models.generation.vllm.vllm_worker import VllmGenerationWorker
from nemo_rl.models.generation.vllm.vllm_worker_async import (
    VllmAsyncGenerationWorker,
)


def main() -> None:
    package_versions = {
        name: version(name)
        for name in (
            "vllm",
            "torch",
            "transformers",
            "flashinfer-python",
            "llguidance",
            "xgrammar",
        )
    }
    assert package_versions["vllm"] == "0.24.0", package_versions
    assert "engine_client" not in inspect.signature(
        ServingTokenization.__init__
    ).parameters
    assert "parser" in inspect.signature(
        OpenAIServingRender.preprocess_chat
    ).parameters
    assert "tool_parser" not in inspect.signature(
        OpenAIServingRender.preprocess_chat
    ).parameters

    required_ray_env = {
        "NRL_VLLM_DISABLE_ALLGATHER_BASE_PATCH",
        "NRL_VLLM_DISABLE_REJECTION_SAMPLER_PATCH",
    }
    existing_ray_env = os.environ.get("VLLM_RAY_EXTRA_ENV_VARS_TO_COPY", "")
    patches._patch_vllm_init_workers_ray(
        sys.executable,
        sorted(required_ray_env),
    )
    copied_ray_env = {
        name
        for name in os.environ["VLLM_RAY_EXTRA_ENV_VARS_TO_COPY"].split(",")
        if name
    }
    assert required_ray_env <= copied_ray_env
    if existing_ray_env:
        assert set(existing_ray_env.split(",")) <= copied_ray_env

    report = {
        "architecture": platform.machine(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_device_count": torch.cuda.device_count(),
        "cuda_version": torch.version.cuda,
        "packages": package_versions,
        "python": sys.version.split()[0],
        "vllm_module_version": vllm.__version__,
        "imports": {
            "OpenAIServingChat": OpenAIServingChat.__name__,
            "OpenAIServingRender": OpenAIServingRender.__name__,
            "ServingTokenization": ServingTokenization.__name__,
            "VllmAsyncGenerationWorker": type(VllmAsyncGenerationWorker).__name__,
            "VllmGenerationWorker": type(VllmGenerationWorker).__name__,
        },
        "ray_extra_env_count": len(copied_ray_env),
    }
    if torch.cuda.is_available():
        report["cuda_device_name"] = torch.cuda.get_device_name(0)
        allocation = torch.ones(1, device="cuda")
        report["cuda_tensor"] = allocation.item()
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
