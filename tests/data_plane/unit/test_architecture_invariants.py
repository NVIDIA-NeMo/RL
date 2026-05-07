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
"""Static architecture invariants — see test plan §4.8.

Cheap regex-level tests. Run in milliseconds. Catch entire classes of
drift around the verl-style sibling-trainer split:

  * legacy ``grpo.py`` is fully untouched by the data plane,
  * ``grpo_sync.py`` requires a TQPolicy with no feature-gate temptation,
  * the production factory has no NoOp escape hatch,
  * ``examples/run_grpo.py`` dispatches both trainers explicitly.

Plan §4.8 was written assuming a ``train_from_dp_meta`` separate-method
design. We instead chose subclass-based polymorphism: ``TQPolicy``
overrides ``Policy`` methods, and ``examples/run_grpo.py`` selects
which policy + trainer pair is constructed.
"""

from __future__ import annotations

import pathlib
import re

import pytest

REPO = pathlib.Path(__file__).resolve().parents[3]


def _read(rel: str) -> str:
    return (REPO / rel).read_text()


def _strip_comments_and_docstrings(src: str) -> str:
    """Best-effort cleaner so we don't false-positive on docstring text."""
    src = re.sub(r"#.*", "", src)
    src = re.sub(r'""".*?"""', "", src, flags=re.DOTALL)
    src = re.sub(r"'''.*?'''", "", src, flags=re.DOTALL)
    return src


# ─── R-C8 — legacy grpo.py is clean ──────────────────────────────────────


def test_no_data_plane_in_master_config():
    """``MasterConfig`` was transitionally extended with a ``data_plane``
    field; it should be removed once the sibling-trainer split lands."""
    src = _read("nemo_rl/algorithms/grpo.py")
    assert "data_plane: NotRequired" not in src, (
        "Legacy MasterConfig still has the data_plane scaffold. "
        "Remove it with the sibling-trainer split."
    )


# ─── R-C9 — sync trainer engages the data plane (TQPolicy design) ────────


def test_grpo_sync_engages_tq_policy():
    """Sync trainer must require a TQ-mediated policy.

    The TQ engagement is now encapsulated in
    :class:`nemo_rl.models.policy.tq_policy.TQPolicy` — the trainer's job
    is to enforce that the policy in hand actually carries the TQ
    transport (``policy.dp_cfg`` is the public marker set by
    ``TQPolicy.__init__``). Without this guard, a misconfiguration could
    silently route through the legacy in-memory dispatch.

    The TQ wire-level constructs (``KVBatchMeta``, ``shard_meta_for_dp``,
    ``build_data_plane_client``) belong inside ``tq_policy.py`` /
    ``preshard.py``, not in the trainer.
    """
    src = _strip_comments_and_docstrings(_read("nemo_rl/algorithms/grpo_sync.py"))
    assert 'hasattr(policy, "dp_cfg")' in src or "hasattr(policy, 'dp_cfg')" in src, (
        "grpo_sync.py must guard on `hasattr(policy, 'dp_cfg')` so a "
        "non-TQ Policy instance is rejected with a clear error."
    )
    # TQ engagement happens through the policy's overridden methods —
    # check that the chain reaches a real KVBatchMeta construction.
    helper_src = _strip_comments_and_docstrings(
        _read("nemo_rl/data_plane/preshard.py")
    )
    assert "KVBatchMeta(" in helper_src, (
        "preshard.py must still construct KVBatchMeta — TQPolicy "
        "delegates here on each fan-out."
    )
    tq_policy_src = _strip_comments_and_docstrings(
        _read("nemo_rl/models/policy/tq_policy.py")
    )
    assert "build_data_plane_client(" in tq_policy_src, (
        "TQPolicy must construct the data-plane client in __init__."
    )


def test_grpo_sync_requires_data_plane_enabled():
    """The sync trainer should hard-fail when invoked without the data
    plane enabled — running it in legacy mode is a category error."""
    src = _strip_comments_and_docstrings(_read("nemo_rl/algorithms/grpo_sync.py"))
    # Either a guard or a direct require — at minimum the error must be
    # raised when enabled=False.
    assert (
        "raise ValueError" in src or "raise RuntimeError" in src
    ), "grpo_sync.py should raise when data_plane is not enabled."
    # And the failure message should name the legacy escape hatch so
    # users can self-recover.
    assert (
        "grpo_train" in src or "grpo.py" in src
    ), "grpo_sync.py's enabled-required error should point users at the legacy trainer."


def test_no_feature_gate_pattern_in_either_trainer():
    """Catch the next 'just one if branch' temptation in *either*
    trainer — the sibling-trainer split forbids cross-trainer
    conditionals."""
    legacy = _strip_comments_and_docstrings(_read("nemo_rl/algorithms/grpo.py"))
    sync = _strip_comments_and_docstrings(_read("nemo_rl/algorithms/grpo_sync.py"))

    # In the legacy trainer, ANY data_plane-conditional is wrong —
    # legacy must not even know the data plane exists.
    legacy_forbidden = [
        r"if\s+.*data_plane",
        r"if\s+.*tq\b",
        r"if\s+.*transfer_queue",
        r"cfg\.get\([\"']data_plane",
        r"master_config\[[\"']data_plane",
        r"master_config\.get\([\"']data_plane",
    ]
    for pat in legacy_forbidden:
        m = re.findall(pat, legacy)
        assert not m, (
            f"legacy grpo.py reintroduced a data-plane gate: "
            f"pattern {pat!r} matched {m}."
        )

    # In the sync trainer, an early "is enabled?" guard is allowed
    # (we use one), but per-stage feature gates inside the loop are not.
    # Heuristic: feature-gate guards inside an inner block tend to look
    # like `if dp_client is not None:` after the early guard already
    # raised. Allow the early guard once; warn on more.
    n_dp_client_gates = len(re.findall(r"if\s+dp_client\s+is\s+not\s+None", sync))
    assert n_dp_client_gates == 0, (
        f"grpo_sync.py has {n_dp_client_gates} `if dp_client is not None` "
        "guards. Sync trainer assumes the client is always present — "
        "the existence check belongs at the top of the function only."
    )


# ─── R-C10 — factory rejects NoOp in production ──────────────────────────


def test_factory_does_not_construct_noop():
    """The production factory must not return a NoOp client.

    ``NoOpDataPlaneClient`` is test-only; importing it directly from
    ``adapters/noop.py`` is fine in tests, but the factory has no
    business handing it out.
    """
    src = _read("nemo_rl/data_plane/factory.py")
    # No import of NoOp from the factory.
    assert "NoOpDataPlaneClient" not in src, (
        "factory.py imports/constructs NoOpDataPlaneClient. NoOp must "
        "be reachable only via direct import from tests."
    )
    # Disabled or unknown impl raises.
    assert "raise ValueError" in src, (
        "factory.py must fail-fast on disabled or unknown impl."
    )


def test_factory_rejects_disabled_impl():
    """Factory must raise — not return None, not return a NoOp — when
    the caller passes ``enabled=False``. The legacy trainer should not
    call the factory at all."""
    src = _read("nemo_rl/data_plane/factory.py")
    cleaned = _strip_comments_and_docstrings(src)
    # The enabled-check should land before any impl dispatch.
    assert re.search(r"enabled.*False|not.*enabled", cleaned), (
        "factory.py is missing an enabled-check. Disabled cfg must "
        "fail-fast, not silently return a client."
    )


# ─── examples/run_grpo.py dispatches both trainers ───────────────────────


def test_run_grpo_dispatches_both_trainers():
    """The example script must explicitly route between the two
    trainers based on ``data_plane.enabled``."""
    src = _read("examples/run_grpo.py")
    cleaned = _strip_comments_and_docstrings(src)
    assert "grpo_train" in cleaned, "run_grpo.py must reference legacy grpo_train"
    assert "grpo_train_sync" in cleaned, (
        "run_grpo.py must reference grpo_train_sync (the TQ-mediated trainer)"
    )
    # Routing must read the data_plane config block somewhere — check
    # against the original (un-stripped) source so we cover both inline
    # access (`master_config["data_plane"]`) and `.get("data_plane")`.
    assert (
        '"data_plane"' in src or "'data_plane'" in src
    ), (
        "run_grpo.py should read master_config[\"data_plane\"] to dispatch."
    )
    assert re.search(r"\.get\(\s*[\"']enabled[\"']", cleaned), (
        "run_grpo.py should branch on the data-plane `enabled` flag."
    )


# ─── Legacy trainer must not import grpo_sync (one-way dependency) ───────


def test_legacy_does_not_import_sync():
    """Dependency direction: ``grpo_sync.py`` imports helpers from
    ``grpo.py``. The reverse must never hold or we'd recreate the
    coupling we split."""
    legacy = _read("nemo_rl/algorithms/grpo.py")
    assert "grpo_sync" not in legacy, (
        "legacy grpo.py imports from grpo_sync.py. The dependency "
        "direction is one-way: sync imports legacy helpers, never "
        "the other way around."
    )


# ─── No-pickle-on-the-bus rule — adapter enforces it ─────────────────────


def test_tq_adapter_enforces_no_pickle():
    """Plan §1.1 P3: the TQ adapter must reject non-tensor leaves at
    the wire boundary. Catch silent removal of this guard."""
    src = _read("nemo_rl/data_plane/adapters/transfer_queue.py")
    assert "TypeError" in src and "non-tensor leaves" in src, (
        "TQ adapter is missing the no-pickle-on-the-bus guard "
        "(P3). _to_wire must raise on non-tensor leaves."
    )


# ─── ABC contract method names — catch silent renames ────────────────────


@pytest.mark.parametrize(
    "method",
    [
        "register_partition",
        "get_meta",
        "get_data",
        "kv_batch_put",
        "kv_batch_get",
        "kv_clear",
        "check_consumption_status",
        "close",
    ],
)
def test_abc_method_present(method):
    """The DataPlaneClient ABC contract is the swap surface. Renaming
    a method silently is a breaking change for every adapter."""
    src = _read("nemo_rl/data_plane/interfaces.py")
    assert f"def {method}" in src, (
        f"DataPlaneClient ABC is missing required method {method!r}. "
        f"This is a breaking change for every adapter (G2)."
    )
