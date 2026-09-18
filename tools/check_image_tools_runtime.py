#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Check bundled HSG environments without installing or rebuilding anything.

Run inside the user-selected container on a Slurm compute node, with this
snapshot mounted at /opt/nemo-rl. This is import/kernel qualification, not a
model rollout or distributed training certification.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys

from tools.image_tools_launch_checks import source_fingerprint


PROBE = r"""
import importlib, importlib.metadata, json, os, pathlib, platform, sys
modules = json.loads(sys.argv[1])
root = pathlib.Path(os.environ["PROJECT_ROOT"]).resolve()
result = {"python": sys.executable, "architecture": platform.machine(), "modules": {}}
assert platform.machine() == "aarch64", platform.machine()
result["versions"] = {}
for name in ["torch", "vllm", "ray", "transformers", "transformer-engine", "math-verify", "openai", "anthropic", "httptools", "nemo-lens", "opentelemetry-api"]:
    try:
        result["versions"][name] = importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        pass
print("IMAGE_TOOLS_PACKAGES " + json.dumps(result, sort_keys=True), flush=True)
for name in modules:
    module = importlib.import_module(name)
    path = pathlib.Path(module.__file__).resolve() if module.__file__ else None
    if name.startswith(("nemo_rl", "nemo_gym", "megatron", "resources_servers", "responses_api_agents")):
        assert path is not None and path.is_relative_to(root), (name, str(path), str(root))
    result["modules"][name] = str(path)
if sys.argv[3] == "driver":
    from omegaconf import OmegaConf
    from nemo_rl.algorithms.grpo import MasterConfig
    from nemo_rl.utils.config import load_config, register_omegaconf_resolvers
    register_omegaconf_resolvers()
    for key in ("VLLM_TOKENIZER", "TRAIN_MANIFEST", "EVAL_MANIFEST", "CHECKPOINT_DIR", "RUN_LOG_DIR", "WANDB_RUN_NAME"):
        os.environ.setdefault(key, "/qualification/" + key)
    recipe = root / "examples/configs/recipes/vlm/vlm_grpo-nemotron-super-omni-120ba12b-image-tools-8n4g-megatron-tp8ep16cp2-async.v1.yaml"
    cfg = MasterConfig(**OmegaConf.to_container(load_config(recipe), resolve=True))
    assert cfg.grpo.async_grpo.enabled and not cfg.grpo.reward_shaping.enabled
    assert cfg.policy["train_global_batch_size"] == 32
    result["recipe_schema"] = "passed"
if "uvicorn" in modules:
    import uvicorn
    from fastapi import FastAPI
    config = uvicorn.Config(FastAPI(), http="auto", loop="asyncio", log_config=None)
    config.load()
    result["http_backend"] = config.http_protocol_class.__module__
if sys.argv[3] in {"learner", "generation"}:
    from tools.check_image_tools_model import check_model_runtime
    result["model_contract"] = check_model_runtime(sys.argv[3], pathlib.Path(os.environ["MODEL_CHECKPOINT"]))
if sys.argv[2] == "gpu":
    import torch
    assert torch.cuda.is_available(), "CUDA unavailable"
    assert torch.cuda.device_count() == 4, torch.cuda.device_count()
    result["gpus"] = []
    for index in range(torch.cuda.device_count()):
        with torch.cuda.device(index):
            x = torch.ones((32, 32), device=f"cuda:{index}", dtype=torch.bfloat16)
            assert torch.all(x @ x == 32).item()
            torch.cuda.synchronize()
            result["gpus"].append({"name": torch.cuda.get_device_name(index),
                                   "capability": torch.cuda.get_device_capability(index)})
print("IMAGE_TOOLS_RUNTIME " + json.dumps(result, sort_keys=True))
"""


def environments():
    """Return exact prebuilt interpreter paths and imports required by this workflow."""
    main = "/opt/nemo_rl_venv/bin/python"
    ray = Path("/opt/ray_venvs")
    gym = Path("/opt/gym_venvs")
    checks = [
        ("driver", main, ["ray", "nemo_rl.algorithms.grpo"], False),
        (
            "learner",
            str(
                ray
                / "nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python"
            ),
            [
                "transformer_engine.pytorch",
                "megatron.bridge",
                "nemo_rl.models.policy.workers.megatron_policy_worker",
            ],
            True,
        ),
        (
            "generation",
            str(
                ray
                / "nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python"
            ),
            ["vllm", "nemo_rl.models.generation.vllm.vllm_worker_async"],
            True,
        ),
        (
            "collector",
            str(
                ray
                / "nemo_rl.algorithms.async_utils.AsyncTrajectoryCollector/bin/python"
            ),
            ["nemo_rl.algorithms.async_utils"],
            False,
        ),
        (
            "gym_actor",
            str(ray / "nemo_rl.environments.nemo_gym.NemoGym/bin/python"),
            ["nemo_rl.environments.nemo_gym"],
            False,
        ),
    ]
    for component in [
        "responses_api_agents/image_tools_agent",
        "responses_api_agents/simple_agent",
        "resources_servers/string_match",
        "resources_servers/math_with_judge",
        "resources_servers/mcqa",
        "responses_api_models/vllm_model",
    ]:
        checks.append(
            (
                component,
                str(gym / component / ".venv/bin/python"),
                [
                    "nemo_gym",
                    component.replace("/", ".") + ".app",
                    "uvicorn",
                ],
                False,
            )
        )
    return checks


def main():
    """Inspect every required environment; return all failures in one bounded run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Also test four visible GPUs in learner/generation envs",
    )
    parser.add_argument(
        "--overlay", type=Path, help="Prebuilt additive dependency overlay root"
    )
    args = parser.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):
        parser.error("Run inside a Slurm allocation, not on a shared login node")
    root = Path(os.environ["PROJECT_ROOT"]).resolve()
    assert root == Path("/opt/nemo-rl"), "Mount the isolated snapshot at /opt/nemo-rl"
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["NEMO_RL_SOURCE_OVERRIDE_ROOT"] = str(root)
    env["NEMO_GYM_EXTRA_ROOTS"] = (
        str(root) + ":" + str(root / "3rdparty/Gym-workspace/Gym")
    )
    env["PYTHONPATH"] = ":".join(
        map(
            str,
            [
                root / "tools/runtime_source_override",
                root,
                root / "3rdparty/Gym-workspace/Gym",
            ],
        )
    )
    if args.overlay:
        assert (args.overlay / "READY.json").is_file(), (
            "Overlay is not completely built"
        )
        env["PYTHONPATH"] = str(args.overlay / "packages") + ":" + env["PYTHONPATH"]
        print((args.overlay / "READY.json").read_text().replace("\n", ""), flush=True)
    env["CUDA_DEVICE_MAX_CONNECTIONS"] = "1"

    def check(spec):
        label, python, modules, needs_gpu = spec
        if not Path(python).is_file():
            return {
                "check": label,
                "ok": False,
                "error": "Missing interpreter: " + python,
            }
        try:
            result = subprocess.run(
                [
                    python,
                    "-c",
                    PROBE,
                    json.dumps(modules),
                    "gpu" if args.gpu and needs_gpu else "imports",
                    label,
                ],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                timeout=180,
            )
            return {
                "check": label,
                "ok": result.returncode == 0,
                "stdout": result.stdout[-16000:],
                "stderr": result.stderr[-16000:],
            }
        except subprocess.TimeoutExpired:
            return {
                "check": label,
                "ok": False,
                "error": "Import/kernel probe exceeded 180 seconds",
            }

    # Bound CPU/native imports rather than spawning one unbounded process per service.
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(check, environments()))
    for result in results:
        print(json.dumps(result, sort_keys=True), flush=True)
    print(
        json.dumps(
            {
                "qualification_source_sha256": source_fingerprint(root),
                "model_checkpoint": env["MODEL_CHECKPOINT"],
            }
        ),
        flush=True,
    )
    print(json.dumps({"passed": sum(r["ok"] for r in results), "total": len(results)}))
    return 0 if all(r["ok"] for r in results) else 1


if __name__ == "__main__":
    sys.exit(main())
