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

* ``factory.select_sync_trainer`` dispatches the legacy trainer when
  ``data_plane`` is absent and the sync trainer when enabled.
* Every launcher picks trainer *and* policy through those shared helpers,
  so the two choices cannot drift apart.
* The ``DataPlaneClient`` ABC carries every method adapters depend on.
"""

from __future__ import annotations

import pathlib

import pytest

REPO = pathlib.Path(__file__).resolve().parents[3]

LAUNCHERS = ["run_grpo.py", "run_vlm_grpo.py"]


def test_select_sync_trainer_dispatches_both_trainers():
    """Returns the TQ-mediated ``grpo_train_sync`` iff ``data_plane.enabled``
    is true, and the legacy ``grpo_train`` otherwise."""
    from nemo_rl.algorithms.grpo import MasterConfig, grpo_train
    from nemo_rl.algorithms.grpo_sync import grpo_train_sync
    from nemo_rl.data_plane.factory import select_sync_trainer

    cfg_legacy = MasterConfig.model_construct(data_plane=None)
    assert select_sync_trainer(cfg_legacy) is grpo_train

    cfg_sync = MasterConfig.model_construct(data_plane={"enabled": True})
    assert select_sync_trainer(cfg_sync) is grpo_train_sync


def test_make_policy_factory_pairs_tq_policy_with_the_sync_trainer():
    """The other half of the dispatch.

    Turning the data plane on means picking *two* things that have to agree:
    ``grpo_train_sync`` (covered above) and a ``TQPolicy`` factory. A launcher
    that picked only one would run the sync trainer against a plain ``Policy``
    — which fails only after a full model load.
    """
    from nemo_rl.data_plane.factory import make_policy_factory

    assert make_policy_factory(None) is None
    assert make_policy_factory({"enabled": False}) is None

    dp_cfg = {"enabled": True}
    factory = make_policy_factory(dp_cfg)
    assert factory is not None

    captured = {}

    class _FakeTQPolicy:
        def __init__(self, **kwargs):
            captured.update(kwargs)

    import nemo_rl.models.policy.tq_policy as tq_policy_module

    real = tq_policy_module.TQPolicy
    tq_policy_module.TQPolicy = _FakeTQPolicy
    try:
        make_policy_factory(dp_cfg)(cluster="c")
    finally:
        tq_policy_module.TQPolicy = real

    assert captured == {"cluster": "c", "dp_cfg": dp_cfg}


@pytest.mark.parametrize("launcher", LAUNCHERS)
def test_launchers_dispatch_through_the_shared_helpers(launcher: str):
    """A launcher that re-implements the dispatch inline can drift from the
    helpers the two tests above pin. Cheaper to require the call than to
    diff two copies of the source."""
    source = (REPO / "examples" / launcher).read_text()

    assert "select_sync_trainer(" in source, (
        f"examples/{launcher} does not route its sync-trainer choice through "
        "nemo_rl.data_plane.factory.select_sync_trainer."
    )
    assert "make_policy_factory(" in source, (
        f"examples/{launcher} does not route its policy choice through "
        "nemo_rl.data_plane.factory.make_policy_factory, so it can run the "
        "sync trainer against a plain Policy."
    )


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


def test_both_trainers_wire_deduplicate_multimodal_data_into_repeat_interleave():
    """``deduplicate_multimodal_data`` must not become a silent no-op.

    ``enable_deduplication`` is reached only through
    ``BatchedDataDict.repeat_interleave(..., share_immutable_media=True)``. A
    trainer that omits the kwarg makes the flag do nothing: provenance is never
    assigned, the deepcopy runs with an empty memo, and the user gets G
    independent copies of every image in driver RAM with no warning.
    """
    import inspect

    from nemo_rl.algorithms import grpo, grpo_sync

    for module in (grpo, grpo_sync):
        source = inspect.getsource(module)
        assert "share_immutable_media=" in source, (
            f"{module.__name__} calls repeat_interleave without "
            "share_immutable_media, so grpo.deduplicate_multimodal_data is a "
            "silent no-op on that trainer."
        )
        assert "deduplicate_multimodal_data" in source


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
