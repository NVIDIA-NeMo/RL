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
"""Minimal behavioral invariants for the data-plane wiring.

* The ``DataPlaneClient`` ABC carries every method adapters depend on.
* Only ``nemo_rl/data_plane`` imports ``transfer_queue``.
"""

from __future__ import annotations

import ast
import importlib
import pathlib
from argparse import Namespace
from contextlib import nullcontext
from types import SimpleNamespace

import pytest

REPO = pathlib.Path(__file__).resolve().parents[3]


@pytest.mark.parametrize("use_nemo_gym", [True, False], ids=["gym", "native"])
@pytest.mark.parametrize("launcher", ["run_grpo", "run_vlm_grpo"])
def test_generic_launchers_reject_nemo_gym_configs(
    monkeypatch: pytest.MonkeyPatch, launcher: str, use_nemo_gym: bool
) -> None:
    """Reject Gym configs before Ray while allowing native configs through."""
    module = importlib.import_module(f"examples.{launcher}")
    calls: list[str] = []

    class ReachedRay(Exception):
        pass

    def reach_ray() -> None:
        calls.append("init_ray")
        raise ReachedRay

    config = SimpleNamespace(
        policy={"generation": {}},
        data={},
        env={"should_use_nemo_gym": use_nemo_gym},
        data_plane=None,
        logger=SimpleNamespace(log_dir="logs"),
        checkpointing=SimpleNamespace(enabled=False),
    )
    common_fakes = {
        "register_omegaconf_resolvers": lambda: None,
        "parse_args": lambda: (Namespace(config="config.yaml"), []),
        "load_config": lambda _: {},
        "check_outdated_config": lambda _: None,
        "MasterConfig": SimpleNamespace(model_validate=lambda _: config),
        "log_container_init_timing": lambda: None,
        "get_next_experiment_dir": lambda log_dir: log_dir,
        "init_ray": reach_ray,
    }
    launcher_fakes = {
        "run_grpo": {
            "init_telemetry_driver": lambda *_, **__: None,
            "shutdown_telemetry": lambda: None,
            "startup_span": nullcontext,
        },
        "run_vlm_grpo": {"materialize_vllm_video_config": lambda *_: None},
    }
    for name, fake in {**common_fakes, **launcher_fakes[launcher]}.items():
        monkeypatch.setattr(module, name, fake)
    monkeypatch.setattr(module.OmegaConf, "to_container", lambda *_, **__: {})

    if use_nemo_gym:
        with pytest.raises(ValueError, match="examples/nemo_gym/run_grpo_nemo_gym.py"):
            module.main()
        assert calls == []
    else:
        with pytest.raises(ReachedRay):
            module.main()
        assert calls == ["init_ray"]


def test_grpo_trainer_wires_deduplicate_multimodal_data_into_repeat_interleave():
    """``deduplicate_multimodal_data`` must not become a silent no-op.

    ``enable_deduplication`` is reached only through
    ``BatchedDataDict.repeat_interleave(..., share_immutable_media=True)``. A
    trainer that omits the kwarg makes the flag do nothing: provenance is never
    assigned, the deepcopy runs with an empty memo, and the user gets G
    independent copies of every image in driver RAM with no warning.
    """
    import inspect

    from nemo_rl.algorithms import grpo

    source = inspect.getsource(grpo)
    assert "share_immutable_media=" in source, (
        "grpo calls repeat_interleave without share_immutable_media, so "
        "grpo.deduplicate_multimodal_data is a silent no-op on that trainer."
    )
    assert "deduplicate_multimodal_data" in source


@pytest.mark.parametrize(
    "method",
    [
        "register_partition",
        "claim_meta",
        "get_data",
        "put_samples",
        "get_samples",
        "list_sample_ids",
        "clear_samples",
        "check_consumption_status",
        "save_checkpoint",
        "load_checkpoint",
        "close",
    ],
)
def test_data_plane_client_abc_method_present(method: str) -> None:
    """The ``DataPlaneClient`` ABC is the swap surface; a silent rename
    is a breaking change for every adapter."""
    from nemo_rl.data_plane.interfaces import DataPlaneClient

    assert hasattr(DataPlaneClient, method), (
        f"DataPlaneClient ABC is missing required method {method!r}. "
        "This is a breaking change for every adapter."
    )


def test_only_the_data_plane_imports_transfer_queue():
    """Only ``nemo_rl/data_plane`` may import ``transfer_queue``.

    So no other code can read TQ metadata, such as the nested label that
    ``tq_controller_patch`` stops relabeling.
    """
    offenders = []
    for path in sorted((REPO / "nemo_rl").rglob("*.py")):
        if path.is_relative_to(REPO / "nemo_rl" / "data_plane"):
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                names = [node.module or ""]
            else:
                continue
            if any(n.split(".")[0] == "transfer_queue" for n in names):
                offenders.append(f"{path.relative_to(REPO)}:{node.lineno}")
    assert offenders == [], (
        f"import transfer_queue outside nemo_rl/data_plane: {offenders}"
    )
