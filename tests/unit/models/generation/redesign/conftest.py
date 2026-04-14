# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Conftest for redesign tests — real Ray, real SGLang.

Tests in this directory exercise the redesign/ modules using real Ray actors
and real SGLang servers.  The conftest stubs non-sglang heavy dependencies
but lets sglang imports resolve naturally against the installed package.
"""

import os
import sys
from unittest.mock import MagicMock

# Set default GPU devices before any CUDA/Ray initialisation.
# The remote cluster reserves GPUs 4-7 for this work.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4,5,6,7")

# Use system Python for all Ray actors (uv not configured in container).
os.environ.setdefault("NEMO_RL_PY_EXECUTABLES_SYSTEM", "1")

# Ensure the test directory is on sys.path so helpers.py is importable.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Stub heavy modules NOT installed in the sglang test environment.
# sglang is NOT stubbed — we test against a real server.
# ---------------------------------------------------------------------------
_STUB_MODULES = [
    "decord",
    "vllm",
    "vllm.sampling_params",
    "vllm.lora",
    "vllm.lora.request",
    "wandb",
]
for _mod in _STUB_MODULES:
    sys.modules.setdefault(_mod, MagicMock())

import pytest
import ray

from nemo_rl.models.generation.redesign.sglang_router import RouterActor


# ---------------------------------------------------------------------------
# Session-scoped fixtures
# ---------------------------------------------------------------------------
@pytest.fixture(scope="session")
def ray_cluster():
    """Initialise Ray once for the entire test session."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    yield
    ray.shutdown()


@pytest.fixture(scope="session")
def router(ray_cluster):
    """Start a real sglang router that lives for the session."""
    actor = RouterActor.remote()
    ip, port = ray.get(actor.start.remote({}))
    yield {"actor": actor, "ip": ip, "port": port}
    try:
        ray.get(actor.stop.remote())
    except Exception:
        pass
    ray.kill(actor)
