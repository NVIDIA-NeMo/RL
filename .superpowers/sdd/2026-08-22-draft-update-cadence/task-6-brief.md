### Task 6: Add component-selective weight synchronization

**Controller-bounded execution ruling (2026-08-22):** This run implements only
the already reviewed receiver replay, the missing policy/Megatron producer path,
and early capability/factory mode propagation and rejection. It must not add or
fabricate apply receipts, controller cadence decisions, cadence-science
capabilities, or fixed/adaptive enablement. For default/full selection, preserve
the legacy endpoint call shape by omitting the `selection` keyword; use the
keyword only for a target-only transfer. Target-only must skip all draft
preflight/export/pipeline-parallel collectives and emit zero draft names/bytes;
full→target-only→full must remain reusable.

**Files:**
- Modify: `nemo_rl/weight_sync/interfaces.py`
- Modify: `nemo_rl/weight_sync/factory.py`
- Modify: `nemo_rl/weight_sync/ipc_weight_synchronizer.py`
- Modify: `nemo_rl/weight_sync/collective_weight_synchronizer.py`
- Modify: `nemo_rl/weight_sync/http_weight_synchronizer.py`
- Modify: `nemo_rl/weight_sync/checkpoint_engine_weight_synchronizer.py`
- Modify: `nemo_rl/weight_sync/megatron_weight_synchronizer.py`
- Modify: `nemo_rl/weight_sync/nccl_reshard_weight_synchronizer.py`
- Modify: `nemo_rl/weight_sync/vllm_remote_sparse_weight_synchronizer.py`
- Modify: `nemo_rl/models/policy/interfaces.py`
- Modify: `nemo_rl/models/policy/lm_policy.py`
- Modify: `nemo_rl/models/policy/workers/megatron_policy_worker.py`
- Modify: `nemo_rl/models/generation/interfaces.py`
- Modify: `nemo_rl/models/generation/vllm/vllm_backend.py`
- Modify: `nemo_rl/models/generation/vllm/speculator_runtime.py`
- Modify: `nemo_rl/algorithms/grpo.py`
- Modify: `tests/unit/weight_sync/test_weight_synchronizer.py`
- Modify: `tests/unit/weight_sync/test_factory.py`
- Modify: `tests/unit/weight_sync/test_vllm_remote_sparse_weight_synchronizer.py`
- Modify: `tests/unit/models/generation/test_vllm_speculator_runtime.py`

**Interfaces:**
- Consumes: `WeightSyncSelection(target=True, draft=decision.draft_refit_requested)` plus the resolved schedule mode, generation backend, colocation flag, refit transport, remote-sparse flag, `cadence_runtime.enabled`, and controller-declared selected-rollout science capabilities.
- Produces: pure `preflight_component_selection(*, schedule_mode: str, generation_backend: str, colocated: bool, refit_transport: str | None, remote_sparse: bool) -> None`, `preflight_cadence_science(*, enabled: bool, capabilities: CadenceScienceCapabilities) -> None`, `WeightSynchronizer.supports_component_selection: bool`, `sync_weights(*, selection: WeightSyncSelection = WeightSyncSelection(), timer: Optional[Timer] = None, kv_scales: Optional[dict[str, float]] = None) -> Mapping[str, object]`, and defensive `require_component_selection(synchronizer: WeightSynchronizer, schedule_mode: str) -> None`; target is always selected. The return mapping carries `successful: bool`, string snapshot provenance, numeric timing values, and an optional nested `draft_apply_receipt`; it is never typed as a float-only dictionary.

- [ ] **Step 1: Write RED capability, target-only transfer, and remote-sparse rejection tests.**

```python
import inspect
from unittest.mock import patch

import pytest

from nemo_rl.algorithms import grpo as grpo_module
from nemo_rl.algorithms.single_controller_utils import setup as sc_setup
from nemo_rl.weight_sync.ipc_weight_synchronizer import IPCWeightSynchronizer
from nemo_rl.weight_sync.vllm_remote_sparse_weight_synchronizer import (
    VllmRemoteSparseWeightSynchronizer,
)
from nemo_rl.weight_sync.interfaces import (
    WeightSyncSelection,
    preflight_component_selection,
    require_component_selection,
)
from nemo_rl.algorithms.draft_cadence_runtime import (
    CadenceScienceCapabilities,
    preflight_cadence_science,
)


def test_selection_rejects_target_false() -> None:
    with pytest.raises(ValueError, match="target policy"):
        WeightSyncSelection(target=False, draft=True)


@patch("nemo_rl.weight_sync.ipc_weight_synchronizer.ray")
def test_target_only_sync_omits_draft_payload(mock_ray) -> None:
    mock_ray.get.return_value = [True]
    policy = _mock_policy()
    generation = _mock_generation()
    synchronizer = IPCWeightSynchronizer(policy, generation)
    synchronizer.sync_weights(selection=WeightSyncSelection(draft=False))
    selection = policy.stream_weights_via_ipc_zmq.call_args.kwargs["selection"]
    assert selection == WeightSyncSelection(target=True, draft=False)
    generation.finalize_draft_update.assert_not_called()


def test_remote_sparse_fixed_cadence_fails_at_startup() -> None:
    remote_sparse_synchronizer = object.__new__(VllmRemoteSparseWeightSynchronizer)
    with pytest.raises(ValueError, match="component-selective.*unsupported"):
        require_component_selection(remote_sparse_synchronizer, "fixed")


@pytest.mark.parametrize(
    ("generation_backend", "colocated", "refit_transport", "remote_sparse"),
    [
        ("sglang", True, None, False),
        ("megatron", True, None, False),
        ("vllm", False, "checkpoint_engine", False),
        ("vllm", False, "nccl_reshard", False),
        ("vllm", True, None, True),
    ],
)
def test_unsupported_transport_fails_before_worker_construction(
    generation_backend,
    colocated,
    refit_transport,
    remote_sparse,
) -> None:
    with pytest.raises(ValueError, match="component-selective.*unsupported"):
        preflight_component_selection(
            schedule_mode="fixed",
            generation_backend=generation_backend,
            colocated=colocated,
            refit_transport=refit_transport,
            remote_sparse=remote_sparse,
        )


def test_single_controller_calls_preflight_before_actor_creation() -> None:
    source = inspect.getsource(sc_setup.setup_single_controller)
    assert source.index("preflight_component_selection(") < source.index(
        "create_policy_cluster("
    )
    assert source.index("preflight_cadence_science(") < source.index(
        "create_policy_cluster("
    )


def test_sync_calls_science_preflight_before_cluster_construction() -> None:
    source = inspect.getsource(grpo_module.setup)
    assert source.index("preflight_cadence_science(") < source.index(
        "RayVirtualCluster("
    )


def test_default_runtime_does_not_require_science_capabilities() -> None:
    preflight_cadence_science(
        enabled=False,
        capabilities=CadenceScienceCapabilities(False, False, False),
    )


def test_legacy_always_keeps_preexisting_transport_eligibility() -> None:
    preflight_component_selection(
        schedule_mode="always",
        generation_backend="sglang",
        colocated=False,
        refit_transport="checkpoint_engine",
        remote_sparse=True,
    )


@pytest.mark.parametrize(
    "capabilities",
    [
        CadenceScienceCapabilities(False, True, True),
        CadenceScienceCapabilities(True, False, True),
        CadenceScienceCapabilities(True, True, False),
    ],
)
def test_experiment_runtime_fails_preflight_on_missing_science(capabilities) -> None:
    with pytest.raises(ValueError, match="cadence runtime science.*unavailable"):
        preflight_cadence_science(enabled=True, capabilities=capabilities)
```

- [ ] **Step 2: Run the RED weight-sync tests and confirm the selection API is missing.**

Run: `uv run --group test pytest -q tests/unit/weight_sync/test_weight_synchronizer.py tests/unit/weight_sync/test_vllm_remote_sparse_weight_synchronizer.py tests/unit/models/generation/test_vllm_speculator_runtime.py -k 'selection or cadence'`

Expected: FAIL during collection with `ImportError: cannot import name 'WeightSyncSelection'`.

- [ ] **Step 3: Add the capability contract and validate it in the factory/startup path.**

```python
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class WeightSyncSelection:
    target: bool = True
    draft: bool = True

    def __post_init__(self) -> None:
        if not self.target:
            raise ValueError("target policy must synchronize on every policy step")


@dataclass(frozen=True, slots=True)
class CadenceScienceCapabilities:
    selected_acceptance_counts: bool
    selected_serving_version: bool
    canonical_metric_logging: bool


def preflight_cadence_science(
    *, enabled: bool, capabilities: CadenceScienceCapabilities
) -> None:
    if not enabled:
        return
    if not all((
        capabilities.selected_acceptance_counts,
        capabilities.selected_serving_version,
        capabilities.canonical_metric_logging,
    )):
        raise ValueError(
            "cadence runtime science is unavailable: selected acceptance counts, "
            "serving-version provenance, and canonical logging are required"
        )


def require_component_selection(
    synchronizer: WeightSynchronizer,
    schedule_mode: str,
) -> None:
    if schedule_mode != "always" and not synchronizer.supports_component_selection:
        raise ValueError(
            f"component-selective draft refit is unsupported by "
            f"{type(synchronizer).__name__}; use update_schedule.mode=always"
        )


def preflight_component_selection(
    *,
    schedule_mode: str,
    generation_backend: str,
    colocated: bool,
    refit_transport: str | None,
    remote_sparse: bool,
) -> None:
    if schedule_mode == "always":
        return
    supported = (
        generation_backend == "vllm"
        and not remote_sparse
        and refit_transport not in {"checkpoint_engine", "nccl_reshard"}
        and (colocated or refit_transport is None)
    )
    if not supported:
        raise ValueError(
            "component-selective draft refit is unsupported by the resolved "
            f"transport: backend={generation_backend!r}, colocated={colocated}, "
            f"refit_transport={refit_transport!r}, remote_sparse={remote_sparse}"
        )
```

Call both pure preflights from `setup_single_controller` and synchronous
`grpo.setup` immediately after resolved-config validation and before
`create_policy_cluster`, `create_generation_cluster`, Ray actor creation, or
communicator construction. Science capabilities are derived from registered
canonical metric/tag producers, not user booleans. With runtime instrumentation
disabled the science preflight returns immediately and does not narrow legacy
`always` transports. Component-selection validation remains independently driven
by schedule semantics. Then extend `create_weight_synchronizer` with
`draft_update_schedule_mode: str = "always"`, route every constructed synchronizer
through the instance validator, and use the same instance validator in
`nemo_rl/algorithms/grpo.py` after its separate remote-sparse construction:

```python
def build_checked_ipc(
    policy: Any,
    generation: Any,
    refit_buffer_size_gb: float | int | None,
    draft_update_schedule_mode: str,
) -> WeightSynchronizer:
    def checked(synchronizer: WeightSynchronizer) -> WeightSynchronizer:
        require_component_selection(synchronizer, draft_update_schedule_mode)
        return synchronizer

    return checked(
        IPCWeightSynchronizer(
            policy=policy,
            generation=generation,
            refit_buffer_size_gb=refit_buffer_size_gb,
        )
    )
```

For the remote-sparse branch in `nemo_rl/algorithms/grpo.py`, validate the constructed object before `init_communicator()`:

```python
assert policy_generation.weight_synchronizer is not None
require_component_selection(
    policy_generation.weight_synchronizer,
    master_config.policy["draft"].update_schedule.mode,
)
policy_generation.weight_synchronizer.init_communicator()
```

Add the abstract property and keyword to `WeightSynchronizer`:

```python
@property
@abstractmethod
def supports_component_selection(self) -> bool:
    raise NotImplementedError

@abstractmethod
def sync_weights(
    self,
    *,
    selection: WeightSyncSelection = WeightSyncSelection(),
    timer: Optional[Timer] = None,
    kv_scales: Optional[dict[str, float]] = None,
) -> Mapping[str, object]:
    raise NotImplementedError
```

IPC and collective return `True` only after their iterators, manifests, transfer byte counts, vLLM apply coverage, and draft finalizer obey `selection.draft`. HTTP, checkpoint-engine, Megatron, NCCL-reshard, and remote-sparse return `False` and raise if called with `draft=False`. Keep `require_component_selection` immediately after `create_weight_synchronizer` and before `init_communicator` as a defense against factory/preflight drift; the earlier pure check is the one that guarantees rejection before worker creation.

Refactor `refit_policy_generation` to accept `selection: WeightSyncSelection`; every post-policy call uses `target=True`, so `POLICY_GENERATION_STALE` is cleared every step even when draft transfer is skipped.

- [ ] **Step 4: Run the GREEN transport/factory and generation coverage tests.**

Run: `uv run --group test pytest -q tests/unit/weight_sync/test_factory.py tests/unit/weight_sync/test_weight_synchronizer.py tests/unit/weight_sync/test_vllm_remote_sparse_weight_synchronizer.py tests/unit/models/generation/test_vllm_speculator_runtime.py tests/unit/models/generation/test_vllm_backend.py tests/unit/single_controller/test_single_controller_setup.py -k 'selection or target_only or cadence or before_worker' && uv run ruff check nemo_rl/weight_sync nemo_rl/models/policy/interfaces.py nemo_rl/models/policy/lm_policy.py nemo_rl/models/generation/interfaces.py nemo_rl/models/generation/vllm/vllm_backend.py nemo_rl/models/generation/vllm/speculator_runtime.py nemo_rl/algorithms/grpo.py nemo_rl/algorithms/single_controller_utils/setup.py`

Expected: tests PASS; target-only transfers report zero draft bytes; remote-sparse fixed/adaptive startup fails before communicator initialization.

- [ ] **Step 5: Stage and create the signed DCO commit.**

```bash
git add nemo_rl/weight_sync nemo_rl/models/policy/interfaces.py nemo_rl/models/policy/lm_policy.py nemo_rl/models/policy/workers/megatron_policy_worker.py nemo_rl/models/generation/interfaces.py nemo_rl/models/generation/vllm/vllm_backend.py nemo_rl/models/generation/vllm/speculator_runtime.py nemo_rl/algorithms/grpo.py nemo_rl/algorithms/single_controller_utils/setup.py tests/unit/weight_sync/test_factory.py tests/unit/weight_sync/test_weight_synchronizer.py tests/unit/weight_sync/test_vllm_remote_sparse_weight_synchronizer.py tests/unit/models/generation/test_vllm_speculator_runtime.py tests/unit/single_controller/test_single_controller_setup.py
git commit -S -s -m "feat(draft): select draft payload during refit"
git verify-commit HEAD
```

Expected: signature verification exits 0.
