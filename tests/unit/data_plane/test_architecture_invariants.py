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

* ``examples/run_grpo._select_trainer`` dispatches the legacy trainer
  when ``data_plane`` is absent and the sync trainer when enabled.
* The ``DataPlaneClient`` ABC carries every method adapters depend on.
"""

from __future__ import annotations

import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[3]


def test_run_grpo_dispatches_both_trainers():
    """``examples/run_grpo._select_trainer`` returns the TQ-mediated
    ``grpo_train_sync`` iff ``data_plane.enabled`` is true, and the
    legacy ``grpo_train`` otherwise."""
    import sys

    sys.path.insert(0, str(REPO / "examples"))
    try:
        from run_grpo import _select_trainer
    finally:
        sys.path.pop(0)
    from nemo_rl.algorithms.grpo import MasterConfig, grpo_train
    from nemo_rl.algorithms.grpo_sync import grpo_train_sync

    cfg_legacy = MasterConfig.model_construct(data_plane=None)
    assert _select_trainer(cfg_legacy) is grpo_train

    cfg_sync = MasterConfig.model_construct(data_plane={"enabled": True})
    assert _select_trainer(cfg_sync) is grpo_train_sync


def test_run_vlm_grpo_dispatches_both_trainers():
    """Same invariant for the VLM launcher's copy of the dispatch.

    ``run_vlm_grpo`` duplicates ``run_grpo``'s ``_select_trainer`` verbatim,
    and only the ``run_grpo`` copy was pinned — so the VLM dispatch could
    drift silently. That is not hypothetical: this launcher is the one that
    shipped the ``processor=`` TypeError.
    """
    import sys

    sys.path.insert(0, str(REPO / "examples"))
    try:
        from run_vlm_grpo import _select_trainer
    finally:
        sys.path.pop(0)
    from nemo_rl.algorithms.grpo import MasterConfig, grpo_train
    from nemo_rl.algorithms.grpo_sync import grpo_train_sync

    cfg_legacy = MasterConfig.model_construct(data_plane=None)
    assert _select_trainer(cfg_legacy) is grpo_train

    cfg_sync = MasterConfig.model_construct(data_plane={"enabled": True})
    assert _select_trainer(cfg_sync) is grpo_train_sync


def test_sync_trainer_is_call_compatible_with_legacy_trainer():
    """Both trainers must accept the same call, because the VLM launcher
    picks one at runtime and passes a single fixed kwarg set.

    Caught a real break: ``run_vlm_grpo`` passes ``processor=`` (VLM-only),
    which ``grpo_train_sync`` did not accept — so every
    ``data_plane.enabled=true`` VLM run died with ``TypeError:
    grpo_train_sync() got an unexpected keyword argument 'processor'``
    after full model load. A signature check is cheap; the e2e that
    surfaces it costs two nodes and ~12 minutes of setup.
    """
    import inspect

    from nemo_rl.algorithms.grpo import grpo_train
    from nemo_rl.algorithms.grpo_sync import grpo_train_sync

    # Mirror of the call in examples/run_vlm_grpo.py::main — 12 positional
    # args (policy .. master_config) plus the VLM-only ``processor`` kwarg.
    # Asserted via ``bind`` rather than as full signature parity: parity
    # would force every future grpo_train parameter to be mirrored into
    # grpo_train_sync as dead weight, which is a cost the dispatch does not
    # actually impose. Only the shape the launchers really pass matters.
    launcher_args = (None,) * 12
    launcher_kwargs = {"processor": None}

    for fn in (grpo_train, grpo_train_sync):
        try:
            inspect.signature(fn).bind(*launcher_args, **launcher_kwargs)
        except TypeError as e:
            raise AssertionError(
                f"{fn.__module__}.{fn.__name__} cannot accept the call made by "
                f"examples/run_vlm_grpo.py: {e}. Both trainers must bind the "
                f"same launcher call, or the data_plane dispatch fails at "
                f"runtime after a full model load."
            ) from e


def test_sync_trainer_rejects_message_level_advantage_penalties():
    from nemo_rl.algorithms.grpo import GRPOConfig, MasterConfig
    from nemo_rl.algorithms.grpo_sync import (
        _raise_if_message_level_advantage_penalties_enabled,
    )

    cfg_disabled = MasterConfig.model_construct(grpo=GRPOConfig())
    _raise_if_message_level_advantage_penalties_enabled(cfg_disabled)

    cfg_enabled = MasterConfig.model_construct(
        grpo=GRPOConfig(
            invalid_tool_call_advantage=-5.0,
            malformed_thinking_advantage=None,
        )
    )
    with pytest.raises(
        NotImplementedError,
        match="grpo.invalid_tool_call_advantage",
    ):
        _raise_if_message_level_advantage_penalties_enabled(cfg_enabled)


@pytest.mark.parametrize(
    "method",
    [
        "register_partition",
        "claim_meta",
        "get_data",
        "put_samples",
        "get_samples",
        "clear_samples",
        "check_consumption_status",
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
