#!/usr/bin/env python3
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

"""Run an external Gym vLLM CLI with its Python propagated to Ray workers."""

from __future__ import annotations

import runpy
import sys
from collections.abc import Callable
from pathlib import Path
from typing import cast

import ray


def _load_apply_vllm_patches() -> Callable[[str], None]:
    """Load vLLM patches without importing NeMo RL's dependency stack."""
    repo_root = Path(__file__).resolve().parents[2]
    patches_path = repo_root / "nemo_rl/models/generation/vllm/patches.py"
    if patches_path.is_file():
        namespace = runpy.run_path(str(patches_path))
        apply_vllm_patches = namespace.get("_apply_vllm_patches")
        if not callable(apply_vllm_patches):
            raise RuntimeError(f"_apply_vllm_patches is missing from {patches_path}")
        return cast(Callable[[str], None], apply_vllm_patches)

    # Support installations that copy this tool without the source checkout.
    from nemo_rl.models.generation.vllm.patches import _apply_vllm_patches

    return _apply_vllm_patches


def main() -> None:
    """Connect to the private cluster and start the requested vLLM command."""
    worker_python = sys.executable
    _load_apply_vllm_patches()(worker_python)
    ray.init(address="auto", runtime_env={"py_executable": worker_python})

    # vLLM is available only in the generation worker environment and must be
    # imported after NeMo RL applies its runtime patches.
    from vllm.entrypoints.cli.main import main as vllm_main

    vllm_main()


if __name__ == "__main__":
    main()
