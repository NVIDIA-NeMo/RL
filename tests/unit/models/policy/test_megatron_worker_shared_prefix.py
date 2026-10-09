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
"""CPU tests for the Megatron worker's shared-prefix gating and agreement.

They cover the default (disabled) path for workers built without ``__init__``,
the single model-world count agreement that carries rank-local planning
failures, the MTP tracker reset scope, the dense-control capability checks
that must run at worker setup rather than at the first forward, the
evaluation MTP bypass, and the TQ row metadata the worker attaches per stage.
"""

from __future__ import annotations

import multiprocessing
import os
import tempfile
from datetime import timedelta
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

# Probe the concrete module: test_modelopt_worker_utils.py leaves stub
# ``megatron.bridge`` packages in sys.modules that a package probe accepts.
pytest.importorskip("megatron.bridge.training.checkpointing")

import nemo_rl.models.policy.workers.megatron_policy_worker as worker_module  # noqa: E402
from nemo_rl.algorithms.loss.interfaces import LossInputType, LossType  # noqa: E402
from nemo_rl.data.packing.shared_prefix_metadata import (  # noqa: E402
    SHARED_PREFIX_EXECUTION_SLOT,
    SHARED_PREFIX_GROUP_ID,
)
from nemo_rl.data_plane import KVBatchMeta  # noqa: E402
from nemo_rl.data_plane.schema import GROUP_ID_TAG  # noqa: E402
from nemo_rl.distributed.batched_data_dict import BatchedDataDict  # noqa: E402
from nemo_rl.models.policy import SharedPrefixTrainingConfig  # noqa: E402
from nemo_rl.utils.timer import Timer  # noqa: E402
from tests.unit.models.policy.test_megatron_split_state import (  # noqa: E402
    _make_worker,
    mock_module_symbols,  # noqa: F401  (fixture)
)

pytestmark = pytest.mark.mcore


def _bare_worker(**shared_prefix: Any) -> Any:
    worker = object.__new__(worker_module.MegatronPolicyWorkerImpl)
    if shared_prefix:
        worker.cfg = {"shared_prefix_training": shared_prefix}
        worker._shared_prefix_cfg = SharedPrefixTrainingConfig(**shared_prefix)
    worker.timer = Timer()
    return worker


def _batch(rows: int = 2) -> BatchedDataDict[Any]:
    return BatchedDataDict(
        {
            "input_ids": torch.zeros(rows, 4, dtype=torch.long),
            "input_lengths": torch.full((rows,), 4, dtype=torch.long),
        }
    )


class _AllReduce:
    """Record default-group MAX reductions and fold in one simulated peer."""

    def __init__(self, *, count: int = 1, rows: int = 2, error: bool = False):
        self.calls = 0
        self.peer = (count, -count, -rows, int(error))

    def __call__(self, tensor: torch.Tensor, *args: Any, **kwargs: Any) -> None:
        self.calls += 1
        peer = torch.tensor(self.peer, dtype=tensor.dtype, device=tensor.device)
        torch.maximum(tensor, peer, out=tensor)


@pytest.fixture
def cpu_tensors(monkeypatch: pytest.MonkeyPatch) -> None:
    # The agreement tensor is created on CUDA; keep these state tests on CPU.
    tensor = torch.tensor

    def cpu_tensor(*args: Any, **kwargs: Any) -> torch.Tensor:
        kwargs.pop("device", None)
        return tensor(*args, **kwargs)

    monkeypatch.setattr(torch, "tensor", cpu_tensor)


def test_worker_without_init_keeps_shared_prefix_disabled() -> None:
    worker = object.__new__(worker_module.MegatronPolicyWorkerImpl)

    assert worker._shared_prefix_cfg.mode == "disabled"
    assert worker._shared_prefix_bin_capacity("train") is None
    assert worker._shared_prefix_bin_capacity("logprobs") is None
    assert (
        worker._plan_shared_prefix_execution_units(_batch(), bin_capacity=None) is None
    )
    worker._validate_shared_prefix_loss(MagicMock(input_type=LossInputType.LOGIT))


@pytest.mark.usefixtures("cpu_tensors")
def test_local_planning_failure_joins_the_agreement_before_raising(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _bare_worker(mode="train")
    all_reduce = _AllReduce()
    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)

    def fail(*_args: Any, **_kwargs: Any) -> tuple[Any, ...]:
        raise ValueError("row exceeds the expanded training budget")

    monkeypatch.setattr(worker_module, "plan_shared_prefix_execution_units", fail)

    with pytest.raises(ValueError, match="expanded training budget"):
        worker._plan_shared_prefix_execution_units(_batch(), bin_capacity=16)
    # Peers block in this collective; the failing rank must enter it too.
    assert all_reduce.calls == 1


@pytest.mark.usefixtures("cpu_tensors")
def test_peer_planning_failure_raises_on_a_healthy_rank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _bare_worker(mode="train")
    monkeypatch.setattr(torch.distributed, "all_reduce", _AllReduce(error=True))
    monkeypatch.setattr(
        worker_module,
        "plan_shared_prefix_execution_units",
        lambda *_a, **_kw: ("unit",),
    )

    with pytest.raises(RuntimeError, match="another model-world rank"):
        worker._plan_shared_prefix_execution_units(_batch(), bin_capacity=16)


@pytest.mark.usefixtures("cpu_tensors")
def test_unaligned_forward_count_mismatch_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    worker = _bare_worker(mode="logprobs")
    monkeypatch.setattr(torch.distributed, "all_reduce", _AllReduce(count=2))
    monkeypatch.setattr(
        worker_module,
        "plan_shared_prefix_execution_units",
        lambda *_a, **_kw: ("unit",),
    )

    with pytest.raises(RuntimeError, match="min=1, max=2"):
        worker._plan_shared_prefix_execution_units(
            _batch(), bin_capacity=16, forward_only=True, stage="logprobs"
        )


@pytest.mark.usefixtures("cpu_tensors")
def test_aligned_planning_agrees_with_one_collective(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    alignment = pytest.importorskip("megatron.rl.shared_prefix_alignment")
    worker = _bare_worker(
        mode="train", pack_groups=True, repack_groups=True, align_data_parallel=True
    )
    all_reduce = _AllReduce(count=2, rows=2)
    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    monkeypatch.setattr(
        worker_module,
        "_get_mcore_shared_prefix_training_capability",
        lambda: frozenset(
            {worker_module.SUPPORTED_SHARED_PREFIX_FOREST_STABLE_ROUTER_CAPABILITY}
        ),
    )
    parallel_state = MagicMock()
    parallel_state.get_data_parallel_world_size.return_value = 2
    monkeypatch.setattr(worker_module, "parallel_state", parallel_state)
    monkeypatch.setattr(
        worker_module,
        "plan_shared_prefix_execution_units",
        lambda *_a, **_kw: ("unit",),
    )
    monkeypatch.setattr(
        worker_module,
        "_resolve_shared_prefix_execution_topology",
        lambda _cfg: (1, 1, 4),
    )
    targets = []

    def materialize(units: Any, *, target_count: int, **_kwargs: Any) -> Any:
        targets.append(target_count)
        return ("split-a", "split-b"), None

    monkeypatch.setattr(alignment, "materialize_alignment", materialize)

    # Logprob planning: no model is needed for the per-token-loss check.
    units = worker._plan_shared_prefix_execution_units(
        _batch(), bin_capacity=16, forward_only=True, stage="logprobs"
    )

    assert units == ("split-a", "split-b")
    assert targets == [2]
    assert all_reduce.calls == 1


def _agreement_rank(rank: int, init_file: str, results: Any) -> None:
    torch.distributed.init_process_group(
        "gloo",
        init_method=f"file://{init_file}",
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=60),
    )
    try:
        worker_module._agree_shared_prefix_execution_count(
            1,
            2,
            local_error=ValueError("rank-local planning error") if rank == 0 else None,
            stage="train",
            device="cpu",
        )
        results.put((rank, "returned"))
    except Exception as error:  # reported to the parent process
        results.put((rank, type(error).__name__))
    finally:
        torch.distributed.destroy_process_group()


def test_rank_local_failure_raises_on_every_rank_without_hanging() -> None:
    context = multiprocessing.get_context("spawn")
    results = context.Queue()
    with tempfile.TemporaryDirectory() as directory:
        init_file = os.path.join(directory, "init")
        processes = [
            context.Process(target=_agreement_rank, args=(rank, init_file, results))
            for rank in range(2)
        ]
        for process in processes:
            process.start()
        outcomes = dict(results.get(timeout=120) for _ in processes)
        for process in processes:
            process.join(timeout=30)

    assert outcomes == {0: "ValueError", 1: "RuntimeError"}


@pytest.mark.usefixtures("mock_module_symbols")
@pytest.mark.parametrize(
    ("mode", "expected_clears"),
    [("disabled", 0), ("dense", 1), ("train", 1)],
)
def test_begin_train_step_clears_mtp_tracker_only_in_execution_modes(
    mode: str, expected_clears: int
) -> None:
    from megatron.core.transformer.multi_token_prediction import MTPLossLoggingHelper

    worker = _make_worker(LossType.TOKEN_LEVEL)
    worker._shared_prefix_cfg = SharedPrefixTrainingConfig(mode=mode)
    worker.mtp_enabled = True
    worker._test_loss_fn.input_type = LossInputType.LOGPROB

    with patch.object(MTPLossLoggingHelper, "clean_metrics_in_tracker") as clean:
        worker.begin_train_step(loss_fn=worker._test_loss_fn)

    assert clean.call_count == expected_clears


def test_dense_control_checks_evaluation_mtp_bypass_at_setup() -> None:
    worker = _bare_worker(mode="dense", bypass_evaluation_mtp=True)
    worker.model = torch.nn.Linear(1, 1)

    with patch.object(worker_module, "parallel_state") as parallel_state:
        parallel_state.get_pipeline_model_parallel_world_size.return_value = 1
        with pytest.raises(NotImplementedError, match="PP1 HybridModel"):
            worker._validate_shared_prefix_worker_features()


def test_dense_control_accepts_evaluation_mtp_bypass_on_a_pp1_hybrid_model() -> None:
    from megatron.core.models.hybrid.hybrid_model import HybridModel

    worker = _bare_worker(mode="dense", bypass_evaluation_mtp=True)
    worker.model = MagicMock(spec=HybridModel)

    with patch.object(worker_module, "parallel_state") as parallel_state:
        parallel_state.get_pipeline_model_parallel_world_size.return_value = 1
        worker._validate_shared_prefix_worker_features()


def test_dense_control_checks_uniform_router_gating_at_setup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from megatron.core.transformer.moe import moe_utils

    monkeypatch.delattr(moe_utils, "router_gating_token_blocks", raising=False)
    worker = _bare_worker(mode="dense", uniform_router_gating=True)

    with pytest.raises(NotImplementedError, match="router_gating_token_blocks"):
        worker._validate_shared_prefix_worker_features()


@pytest.mark.parametrize("bypass", [False, True])
def test_logprobs_skip_mtp_through_compute_mtp_loss(bypass: bool) -> None:
    worker = _bare_worker(mode="dense", bypass_evaluation_mtp=bypass)
    worker.cfg.update(logprob_batch_size=2, megatron_cfg={})
    worker.model = MagicMock()
    worker.mcore_state = MagicMock(straggler_timer=None)
    worker.media_placeholder_token_id = None
    worker.delegate_pack_to_model = False
    worker.delegate_mtp_loss_mask_to_model = False
    worker.model_slices_context_parallel_inputs = False
    worker.mtp_enabled = True
    worker.sampling_params = None
    worker.defer_fp32_logits = False
    worker._router_replay_enabled = False

    with (
        patch.object(
            worker_module,
            "get_microbatch_iterator",
            return_value=(iter(()), 1, 2, 4, 4),
        ),
        patch.object(worker_module, "LogprobsPostProcessor"),
        patch.object(
            worker_module,
            "megatron_forward_backward",
            return_value=[{"logprobs": torch.zeros(2, 4)}],
        ) as forward_backward,
        patch.object(worker_module, "parallel_state"),
        patch.object(
            worker_module,
            "broadcast_tensors_from_last_stage",
            side_effect=lambda tensors: tensors,
        ),
        patch.object(torch, "empty_like", side_effect=lambda t, **_kw: t.clone()),
    ):
        worker.get_logprobs(data=_batch())

    # The bypass no longer mutates the model; it is a forward argument.
    assert forward_backward.call_args.kwargs["compute_mtp_loss"] is (not bypass)


def _tq_meta(rows: int, *, tags: Any) -> KVBatchMeta:
    return KVBatchMeta(
        partition_id="train",
        task_name="prev_lp",
        sample_ids=[f"sample{index}" for index in range(rows)],
        extra_info={SHARED_PREFIX_EXECUTION_SLOT: [0] * rows},
        tags=tags,
    )


def test_attaches_group_ids_from_row_tags_for_the_enabled_stage() -> None:
    worker = _bare_worker(mode="logprobs")
    # Sample IDs carry no group naming convention; the row tag is the source.
    meta = _tq_meta(2, tags=[{GROUP_ID_TAG: "a"}, {GROUP_ID_TAG: "b"}])

    data = worker._attach_or_repack_pack_metadata(_batch(), meta, stage="logprobs")

    assert data[SHARED_PREFIX_GROUP_ID] == ["a", "b"]
    assert data[SHARED_PREFIX_EXECUTION_SLOT].tolist() == [0, 0]


def test_shared_prefix_dispatch_requires_group_id_tags() -> None:
    worker = _bare_worker(mode="logprobs")
    meta = _tq_meta(2, tags=[{GROUP_ID_TAG: "a"}, {}])

    with pytest.raises(ValueError, match="tag on every row"):
        worker._attach_or_repack_pack_metadata(_batch(), meta, stage="logprobs")


@pytest.mark.parametrize("stage", ["train", None])
def test_stage_without_sharing_uses_the_base_packing_path(stage: Any) -> None:
    # mode=logprobs shares only logprob forwards; the value forward has no stage.
    worker = _bare_worker(mode="logprobs")
    meta = _tq_meta(2, tags=None)
    sentinel = object()

    with patch.object(
        worker_module.TQWorkerMixin,
        "_attach_or_repack_pack_metadata",
        return_value=sentinel,
    ) as base:
        assert (
            worker._attach_or_repack_pack_metadata(_batch(), meta, stage=stage)
            is sentinel
        )
    assert base.call_args.kwargs["stage"] == stage
