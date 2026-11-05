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
"""Validate the gym venv prefetch configs against Gym's real models and config parser.

``examples/nemo_gym/prefetch_*.yaml`` are only exercised by the rl-gym image
build (``docker/Dockerfile``, gated on ``NEMO_GYM_PREFETCH_CONFIGS``), which no
PR-time CI job runs. A Gym field that gains a new default-less entry therefore
breaks the image build rather than the PR that drifts away from it. These tests
run Gym's own ``model_validate`` on every standalone server block in those
configs so the drift surfaces at PR time instead. They also compose each config
through Gym's own parser (strict environment servers), check that any file a
config names under ``/opt/nemo-rl`` ships in this repo, and check that the
Super 3.5 config keeps the services the newer blends added.

Passing here does NOT mean the image build succeeds. Known gaps:

* ``prefetch_local_vllm_model`` is skipped whenever vllm is absent, which in CI is
  **always**: ``L0_Unit_Tests_Nemo_Gym.sh`` runs ``uv run --extra nemo_gym`` and vllm
  is a separate extra, while Gym's ``local_vllm_model/app.py`` imports it at module
  scope. ``vllm_serve_env_vars`` -- required only by ``LocalVLLMModelConfig`` -- is
  therefore unguarded here. Tracked in
  https://github.com/NVIDIA-NeMo/RL/issues/3806.
* The ``UV_LINK_MODE``/``r2e_gym.sh`` half of the prefetch is not covered at all.
* Nothing here creates a venv or starts a server, so install- and import-time
  failures (e.g. ether0's rdkit needing system X11 libraries) still surface only
  in the image build.
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


# Where docker/Dockerfile puts the NeMo RL source (its WORKDIR and COPY target).
CONTAINER_REPO_ROOT = "/opt/nemo-rl/"


def _container_repo_paths(value):
    """Yield every string in ``value`` that names a file under the image's NeMo RL source."""
    if isinstance(value, dict):
        for item in value.values():
            yield from _container_repo_paths(item)
    elif isinstance(value, list):
        for item in value:
            yield from _container_repo_paths(item)
    elif isinstance(value, str) and value.startswith(CONTAINER_REPO_ROOT):
        yield value


@pytest.mark.parametrize(
    "config_path", PREFETCH_CONFIGS, ids=[os.path.basename(p) for p in PREFETCH_CONFIGS]
)
def test_prefetch_config_composes_with_gym_parser(config_path):
    """Every prefetch config must compose through Gym's own config parser.

    This is the static half of the image build's dry run: every ``config_paths``
    entry resolves (nested includes and moved-path aliases too), every server
    reference and ``_inherit_from`` source exists, no mandatory ``???`` value is
    left unset, and every agent has a declared environment server. ``offline``
    keeps the parse from probing ports or exporting environment variables.
    """
    pytest.importorskip("nemo_gym")
    from nemo_gym.global_config import (
        GlobalConfigDictParser,
        GlobalConfigDictParserConfig,
    )

    nemo_gym_config = OmegaConf.to_container(load_config(config_path), resolve=True)[
        "env"
    ]["nemo_gym"]
    nemo_gym_config["error_on_agent_without_environment_server"] = True
    GlobalConfigDictParser().parse(
        GlobalConfigDictParserConfig(
            initial_global_config_dict=OmegaConf.create(nemo_gym_config),
            skip_load_from_cli=True,
            skip_load_from_dotenv=True,
            offline=True,
        )
    )


@pytest.mark.parametrize(
    "config_path", PREFETCH_CONFIGS, ids=[os.path.basename(p) for p in PREFETCH_CONFIGS]
)
def test_prefetch_config_container_paths_exist_in_repo(config_path):
    """A file a prefetch config names under /opt/nemo-rl must ship in this repo.

    The dry run reads it at server construction (e.g. bbh_hypotest's
    ``problem_jsonl``), so a rename would only fail the image build.
    """
    config = OmegaConf.to_container(load_config(config_path), resolve=True)
    for path in _container_repo_paths(config["env"]["nemo_gym"]):
        relative_path = path.removeprefix(CONTAINER_REPO_ROOT)
        assert os.path.isfile(os.path.join(REPO_ROOT, relative_path)), (
            f"{os.path.basename(config_path)} names {path}, but {relative_path} "
            "is not in the repo"
        )


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
