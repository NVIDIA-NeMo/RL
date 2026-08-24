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
"""Validate the gym venv prefetch configs against Gym's real pydantic models.

``examples/nemo_gym/prefetch_*.yaml`` are only exercised by the rl-gym image
build (``docker/Dockerfile``, gated on ``NEMO_GYM_PREFETCH_CONFIGS``), which no
PR-time CI job runs. A Gym field that gains a new default-less entry therefore
breaks the image build rather than the PR that drifts away from it. These tests
run Gym's own ``model_validate`` on every standalone server block in those
configs so the drift surfaces at PR time instead.

Passing here does NOT mean the image build succeeds. Two known gaps:

* ``prefetch_local_vllm_model`` is skipped whenever vllm is absent, which in CI is
  **always**: ``L0_Unit_Tests_Nemo_Gym.sh`` runs ``uv run --extra nemo_gym`` and vllm
  is a separate extra, while Gym's ``local_vllm_model/app.py`` imports it at module
  scope. ``vllm_serve_env_vars`` -- required only by ``LocalVLLMModelConfig`` -- is
  therefore unguarded here. Tracked in
  https://github.com/NVIDIA-NeMo/RL/issues/3806.
* The ``UV_LINK_MODE``/``r2e_gym.sh`` half of the prefetch is not covered at all.
"""

import glob
import importlib
import os

import pytest
from omegaconf import OmegaConf

from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

pytestmark = pytest.mark.nemo_gym

register_omegaconf_resolvers()

REPO_ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "..")
PREFETCH_CONFIGS = sorted(
    glob.glob(os.path.join(REPO_ROOT, "examples", "nemo_gym", "prefetch_*.yaml"))
)

# Server-type key in the YAML -> the Gym module/class that server validates against.
MODEL_CONFIG_CLASSES = {
    "vllm_model": ("responses_api_models.vllm_model.app", "VLLMModelConfig"),
    "local_vllm_model": (
        "responses_api_models.local_vllm_model.app",
        "LocalVLLMModelConfig",
    ),
}

# Gym server-level inheritance directives (nemo_gym/global_config.py). A server
# carrying either one is merged onto another server before validation, so its
# block is not self-contained.
GYM_INHERITANCE_KEYS = ("_copy", "_inherit_from")


def _resolve_config_class(server_type: str):
    """Import the Gym config class for ``server_type``, skipping if unavailable.

    ``local_vllm_model/app.py`` imports ``vllm`` at module scope, which the
    nemo_gym extra does not provide, so that half only runs where vllm is installed.
    """
    module_name, class_name = MODEL_CONFIG_CLASSES[server_type]
    try:
        module = importlib.import_module(module_name)
    except ImportError as e:
        pytest.skip(f"{module_name} is not importable in this environment: {e}")
    return getattr(module, class_name)


def _standalone_model_blocks(config_path):
    """Yield ``(server_name, server_type, block)`` for standalone server definitions.

    A block is only self-contained when Gym validates it exactly as written. Two
    kinds of server are therefore skipped:

    * one whose block omits ``entrypoint`` -- an overlay onto a bundled config
      pulled in via ``config_paths`` (e.g. the ``policy_model`` overlays in the
      training recipes);
    * one carrying a ``_copy``/``_inherit_from`` directive, which Gym resolves by
      merging a source server in first.

    In both cases the merged-in source supplies fields the file itself does not,
    so validating the literal block would report spurious missing fields.
    """
    config = OmegaConf.to_container(load_config(config_path), resolve=True)
    for server_name, server in config["env"]["nemo_gym"].items():
        if not isinstance(server, dict):
            continue
        if any(key in server for key in GYM_INHERITANCE_KEYS):
            continue
        models = server.get("responses_api_models")
        if not isinstance(models, dict):
            continue
        for server_type, block in models.items():
            if server_type not in MODEL_CONFIG_CLASSES or not isinstance(block, dict):
                continue
            if any(key in block for key in GYM_INHERITANCE_KEYS):
                continue
            if "entrypoint" in block:
                yield server_name, server_type, block


def _prefetch_block_ids():
    params = []
    for config_path in PREFETCH_CONFIGS:
        for server_name, server_type, _ in _standalone_model_blocks(config_path):
            params.append(
                pytest.param(
                    config_path,
                    server_name,
                    server_type,
                    id=f"{os.path.basename(config_path)}::{server_name}",
                )
            )
    return params


def test_prefetch_configs_exist():
    assert PREFETCH_CONFIGS, "No examples/nemo_gym/prefetch_*.yaml configs found"


@pytest.mark.parametrize(
    ("config_path", "server_name", "server_type"), _prefetch_block_ids()
)
def test_prefetch_model_servers_satisfy_gym_config(
    config_path, server_name, server_type
):
    """Every standalone model server in a prefetch config must pass Gym validation."""
    config_class = _resolve_config_class(server_type)
    block = next(
        b
        for name, stype, b in _standalone_model_blocks(config_path)
        if name == server_name and stype == server_type
    )
    # name/host/port are assigned by Gym's orchestrator at spinup, never authored
    # in the YAML, so supply placeholders for them and let every other
    # default-less field be enforced against the file's own contents.
    config_class.model_validate(
        {"name": server_name, "host": "127.0.0.1", "port": 8000, **block}
    )
