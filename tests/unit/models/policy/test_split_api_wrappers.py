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

"""CPU tests for the split-API presharded wrappers and TQPolicy fan-out.

These two layers sit between the SC driver and the backend state machine
and were previously exercised only by the GPU-gated parity test — the
latent bugs the PR #2683 review surfaced (futures consumed with the wrong
API, an unused per-microbatch return) lived exactly here. Pin the
contracts cheaply:
  - ``*_presharded`` wrappers: pass-through begin/finish/abort, the
    fetch → attach → backend chain in ``train_microbatch_presharded``
    (returning None), and the ``is_replica_leader`` tag on finish.
  - TQPolicy driver: single-data futures consumed via ``ray.get``,
    replica-twin dedup in ``finish_train_step`` aggregation, and
    ``train_microbatches_from_meta`` returning None.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from nemo_rl.algorithms.draft_update_schedule import DraftUpdateDecision
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.schema import DP_TRAIN_FIELDS, ROUTED_EXPERTS_FIELD
from nemo_rl.data_plane.worker_mixin import TQWorkerMixin
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.models.policy.tq_policy import TQPolicy, _supports_draft_apply_receipts


_RECEIPT = {
    "successful": True,
    "decision_id": 7,
    "global_step": 3,
    "draft_model_sha256": "1" * 64,
    "draft_optimizer_sha256": "2" * 64,
}


def _decision() -> DraftUpdateDecision:
    return DraftUpdateDecision(
        global_step=3,
        decision_id=7,
        update_requested=True,
        draft_refit_requested=True,
        reason="always",
        observed_acceptance=None,
    )


class _SplitStubWorker(TQWorkerMixin):
    """Mixin host recording backend calls; fetch/attach are stubbed."""

    def __init__(self, is_leader: bool = True):
        self.calls: list[tuple] = []
        self._leader = is_leader

    def _fetch(self, meta):
        self.calls.append(("fetch", meta))
        return {
            "data_from": meta,
            "input_ids": torch.zeros(len(meta.sample_ids), 2, dtype=torch.int64),
        }

    def _attach_or_repack_pack_metadata(self, data, meta):
        self.calls.append(("attach", meta))
        return data

    def _is_replica_leader(self) -> bool:
        return self._leader

    # backend split API
    def begin_train_step(
        self,
        loss_fn,
        gbs=None,
        mbs=None,
        *,
        capture_draft_update_receipt=False,
    ):
        self.calls.append(("begin", loss_fn, gbs, mbs, capture_draft_update_receipt))

    def train_microbatch(self, data):
        self.calls.append(("train_microbatch", data))

    def finish_train_step(self):
        self.calls.append(("finish",))
        return {
            "global_loss": 1.0,
            "grad_norm": 0.5,
            "draft_update_successful": True,
            "all_mb_metrics": {"loss": [1.0]},
        }

    def abort_train_step(self):
        self.calls.append(("abort",))


def _meta() -> KVBatchMeta:
    return KVBatchMeta(
        partition_id="train",
        task_name="train",
        sample_ids=["s0", "s1"],
    )


class TestPreshardedWrappers:
    def test_begin_forwards_args(self):
        w = _SplitStubWorker()
        loss_fn = object()
        w.begin_train_step_presharded(loss_fn=loss_fn, gbs=8, mbs=2)
        assert w.calls == [("begin", loss_fn, 8, 2, False)]

    def test_begin_forwards_explicit_receipt_capture_flag(self):
        w = _SplitStubWorker()
        w.begin_train_step_presharded(
            loss_fn=object(),
            capture_draft_update_receipt=True,
        )
        assert w.calls[-1][-1] is True

    def test_train_microbatch_fetches_attaches_then_dispatches(self):
        w = _SplitStubWorker()
        meta = _meta()
        out = w.train_microbatch_presharded(meta=meta)
        assert out is None  # metrics accumulate in the open-step state
        assert [c[0] for c in w.calls] == ["fetch", "attach", "train_microbatch"]
        dispatched = w.calls[-1][1]
        assert dispatched["data_from"] is meta
        assert dispatched["draft_sample_ids"].dtype == torch.int64
        assert dispatched["draft_sample_ids"].shape == (2,)

    def test_stable_sample_ids_do_not_depend_on_microbatch_order(self):
        first = _SplitStubWorker()
        second = _SplitStubWorker()
        forward = KVBatchMeta(
            partition_id="train",
            task_name="train",
            sample_ids=["prompt-a_g0", "prompt-b_g0"],
        )
        reverse = KVBatchMeta(
            partition_id="train",
            task_name="train",
            sample_ids=["prompt-b_g0", "prompt-a_g0"],
        )

        first.train_microbatch_presharded(meta=forward)
        second.train_microbatch_presharded(meta=reverse)

        forward_ids = first.calls[-1][1]["draft_sample_ids"]
        reverse_ids = second.calls[-1][1]["draft_sample_ids"]
        assert torch.equal(forward_ids, reverse_ids.flip(0))

    def test_finish_tags_replica_leader(self):
        leader = _SplitStubWorker(is_leader=True)
        twin = _SplitStubWorker(is_leader=False)
        assert leader.finish_train_step_presharded()["is_replica_leader"] is True
        result = twin.finish_train_step_presharded()
        assert result["is_replica_leader"] is False
        # backend payload passes through untouched
        assert result["global_loss"] == 1.0

    def test_abort_forwards(self):
        w = _SplitStubWorker()
        w.abort_train_step_presharded()
        assert w.calls == [("abort",)]


def _make_tq_policy() -> tuple[TQPolicy, MagicMock]:
    """Bare TQPolicy with the attributes the split fan-out touches."""
    p = object.__new__(TQPolicy)
    p.cfg = {"train_global_batch_size": 8, "train_micro_batch_size": 2}
    p._router_replay_enabled = False
    p.flops_tracker = None
    p._capture_draft_update_receipt_for_open_step = False
    wg = MagicMock()
    wg.run_all_workers_single_data.return_value = ["f0", "f1"]
    p.worker_group = wg
    p.sharding_annotations = MagicMock()
    p.sharding_annotations.get_axis_size.return_value = 2
    return p, wg


@pytest.mark.parametrize("context_parallel_size", [1, 2, 4])
def test_tq_policy_truthfully_advertises_default_vllm_apply_receipts(
    context_parallel_size: int,
) -> None:
    config = {
        "generation": {"backend": "vllm", "refit_transport": None},
        "megatron_cfg": {"context_parallel_size": context_parallel_size},
    }

    assert _supports_draft_apply_receipts(
        config,
        update_receipts_supported=True,
    )


@pytest.mark.parametrize(
    "generation",
    [
        {"backend": "sglang", "refit_transport": None},
        {"backend": "vllm", "refit_transport": "nixl"},
        {"backend": "vllm", "refit_transport": "nccl_reshard"},
    ],
)
def test_tq_policy_does_not_advertise_unsupported_apply_receipt_transport(
    generation: dict[str, object],
) -> None:
    config = {"generation": generation}

    assert not _supports_draft_apply_receipts(
        config,
        update_receipts_supported=True,
    )
    assert not _supports_draft_apply_receipts(
        {"generation": {"backend": "vllm", "refit_transport": None}},
        update_receipts_supported=False,
    )


def test_lm_policy_preserves_uniform_draft_update_successful_bool() -> None:
    policy = object.__new__(Policy)
    policy.cfg = {"train_global_batch_size": 8, "train_micro_batch_size": 2}
    policy.flops_tracker = None
    policy._shard_for_train = MagicMock(return_value=[{"input_lengths": torch.ones(1)}])
    policy._report_sharded_payload = MagicMock()
    policy.worker_group = MagicMock()
    policy.worker_group.get_all_worker_results.return_value = [
        {
            "global_loss": torch.tensor(1.0),
            "grad_norm": torch.tensor(0.5),
            "draft_update_successful": False,
            "all_mb_metrics": {},
        }
    ]

    result = policy.train(data=MagicMock(), loss_fn=MagicMock())

    assert result["draft_update_successful"] is False


def test_lm_policy_threads_capture_and_surfaces_visible_publisher_receipt() -> None:
    policy = object.__new__(Policy)
    policy.cfg = {"train_global_batch_size": 8, "train_micro_batch_size": 2}
    policy.flops_tracker = None
    policy._shard_for_train = MagicMock(return_value=[{"input_lengths": torch.ones(1)}])
    policy._report_sharded_payload = MagicMock()
    policy.worker_group = MagicMock()
    policy.worker_group.get_all_worker_results.return_value = [
        {
            "global_loss": torch.tensor(1.0),
            "grad_norm": torch.tensor(0.5),
            "draft_update_successful": True,
            "all_mb_metrics": {},
            "world_rank": 1,
            "draft_update_receipt_publisher_rank": 1,
            "draft_update_receipt": _RECEIPT,
        }
    ]

    result = policy.train(
        data=MagicMock(),
        loss_fn=MagicMock(),
        draft_update_decision=_decision(),
        capture_draft_update_receipt=True,
    )

    common = policy.worker_group.run_all_workers_sharded_data.call_args.kwargs[
        "common_kwargs"
    ]
    assert common["capture_draft_update_receipt"] is True
    assert result["draft_update_receipt"] == _RECEIPT


class TestTQPolicySplitFanout:
    def test_capture_current_draft_identity_selects_world_publisher(self):
        p, wg = _make_tq_policy()
        rows = [
            {
                "world_rank": 0,
                "draft_update_receipt_publisher_rank": 1,
            },
            {
                "world_rank": 1,
                "draft_update_receipt_publisher_rank": 1,
                "draft_update_receipt": _RECEIPT,
            },
        ]
        with patch("nemo_rl.models.policy.tq_policy.ray") as mock_ray:
            mock_ray.get.return_value = rows
            receipt = p.capture_current_draft_state_receipt(version=7, global_step=3)

        wg.run_all_workers_single_data.assert_called_once_with(
            "capture_current_draft_state_receipt", version=7, global_step=3
        )
        assert receipt == _RECEIPT

    def test_begin_consumes_single_data_futures_with_ray_get(self):
        """run_all_workers_single_data returns plain ObjectRefs, not a
        MultiWorkerFuture — the fan-out must ray.get them (PR #2683
        review; first execution of this path raised AttributeError)."""
        p, wg = _make_tq_policy()
        with patch("nemo_rl.models.policy.tq_policy.ray") as mock_ray:
            p.begin_train_step(loss_fn="LF")
        wg.run_all_workers_single_data.assert_called_once_with(
            "begin_train_step_presharded", loss_fn="LF", gbs=8, mbs=2
        )
        mock_ray.get.assert_called_once_with(["f0", "f1"])
        wg.get_all_worker_results.assert_not_called()

    def test_begin_threads_capture_flag_and_stores_it_until_finish(self):
        p, wg = _make_tq_policy()
        with patch("nemo_rl.models.policy.tq_policy.ray"):
            p.begin_train_step(
                loss_fn="LF",
                capture_draft_update_receipt=True,
            )
        assert p._capture_draft_update_receipt_for_open_step is True
        assert (
            wg.run_all_workers_single_data.call_args.kwargs[
                "capture_draft_update_receipt"
            ]
            is True
        )

    def test_train_microbatches_from_meta_dispatches_and_returns_none(self):
        p, wg = _make_tq_policy()
        meta = _meta()
        with (
            patch.object(TQPolicy, "_stamp_pad_seqlen"),
            patch.object(TQPolicy, "_packing_args", return_value=(None, None)),
            patch(
                "nemo_rl.models.policy.tq_policy.shard_meta_for_dp",
                return_value=([meta, meta], None),
            ) as mock_shard,
        ):
            out = p.train_microbatches_from_meta(meta)
        assert out is None
        train_meta = mock_shard.call_args.args[0]
        assert train_meta.fields == list(DP_TRAIN_FIELDS)
        assert ROUTED_EXPERTS_FIELD not in train_meta.fields
        assert (
            wg.run_all_workers_sharded_data.call_args.args[0]
            == "train_microbatch_presharded"
        )
        # sharded dispatch returns a MultiWorkerFuture → waited via
        # get_all_worker_results (unlike the single-data fan-outs)
        wg.get_all_worker_results.assert_called_once()

    def test_train_microbatches_requests_routed_experts_for_router_replay(self):
        p, _ = _make_tq_policy()
        p._router_replay_enabled = True
        meta = _meta()
        with (
            patch.object(TQPolicy, "_stamp_pad_seqlen"),
            patch.object(TQPolicy, "_packing_args", return_value=(None, None)),
            patch(
                "nemo_rl.models.policy.tq_policy.shard_meta_for_dp",
                return_value=([meta, meta], None),
            ) as mock_shard,
        ):
            p.train_microbatches_from_meta(meta)

        train_meta = mock_shard.call_args.args[0]
        assert train_meta.fields == [*DP_TRAIN_FIELDS, ROUTED_EXPERTS_FIELD]

    def test_train_microbatches_honors_narrowed_train_fields(self):
        p, _ = _make_tq_policy()
        meta = _meta()
        narrowed_fields = tuple(
            field for field in DP_TRAIN_FIELDS if field != "prev_logprobs"
        )
        with (
            patch.object(TQPolicy, "_stamp_pad_seqlen"),
            patch.object(TQPolicy, "_packing_args", return_value=(None, None)),
            patch(
                "nemo_rl.models.policy.tq_policy.shard_meta_for_dp",
                return_value=([meta, meta], None),
            ) as mock_shard,
        ):
            p.train_microbatches_from_meta(meta, train_fields=narrowed_fields)

        train_meta = mock_shard.call_args.args[0]
        assert train_meta.fields == list(narrowed_fields)

    def test_finish_dedupes_replica_twins(self):
        """TP/CP twins return identical metric copies; aggregating without
        the is_replica_leader filter inflates every per-token metric."""

        def _result(leader: bool) -> dict:
            return {
                "global_loss": 1.0,
                "grad_norm": 0.5,
                "all_mb_metrics": {"loss": [0.1]},
                "is_replica_leader": leader,
            }

        p, wg = _make_tq_policy()
        with patch("nemo_rl.models.policy.tq_policy.ray") as mock_ray:
            # 2 DP leaders + 2 TP twins
            mock_ray.get.return_value = [
                _result(True),
                _result(False),
                _result(True),
                _result(False),
            ]
            out = p.finish_train_step()
        assert out["all_mb_metrics"]["loss"] == [0.1, 0.1]  # twins dropped
        # _aggregate_train_results surfaces global_loss under "loss"
        assert out["loss"] == 1.0

    def test_finish_selects_receipt_from_raw_rows_before_metric_dedup(self):
        p, _ = _make_tq_policy()
        p._capture_draft_update_receipt_for_open_step = True

        def _result(rank: int, leader: bool) -> dict:
            row = {
                "global_loss": 1.0,
                "grad_norm": 0.5,
                "draft_update_successful": True,
                "draft_update_decision": _decision(),
                "all_mb_metrics": {"loss": [0.1]},
                "is_replica_leader": leader,
                "world_rank": rank,
                "draft_update_receipt_publisher_rank": 1,
            }
            if rank == 1:
                row["draft_update_receipt"] = _RECEIPT
            return row

        with patch("nemo_rl.models.policy.tq_policy.ray") as mock_ray:
            mock_ray.get.return_value = [
                _result(0, True),
                _result(1, True),
                _result(2, False),
            ]
            out = p.finish_train_step()

        assert out["draft_update_receipt"] == _RECEIPT
        assert out["all_mb_metrics"]["loss"] == [0.1, 0.1]
        assert p._capture_draft_update_receipt_for_open_step is False

    def test_finish_surfaces_draft_grad_norm(self):
        p, _ = _make_tq_policy()
        with patch("nemo_rl.models.policy.tq_policy.ray") as mock_ray:
            mock_ray.get.return_value = [
                {
                    "global_loss": 1.0,
                    "grad_norm": 0.5,
                    "draft_grad_norm": 0.25,
                    "all_mb_metrics": {"draft_loss": [0.1]},
                    "is_replica_leader": True,
                }
            ]
            out = p.finish_train_step()

        assert out["draft_grad_norm"] == 0.25

    def test_finish_preserves_uniform_draft_update_successful_bool(self):
        p, _ = _make_tq_policy()
        with patch("nemo_rl.models.policy.tq_policy.ray") as mock_ray:
            mock_ray.get.return_value = [
                {
                    "global_loss": 1.0,
                    "grad_norm": 0.5,
                    "draft_update_successful": False,
                    "all_mb_metrics": {},
                    "is_replica_leader": True,
                },
                {
                    "global_loss": 1.0,
                    "grad_norm": 0.5,
                    "draft_update_successful": False,
                    "all_mb_metrics": {},
                    "is_replica_leader": True,
                },
            ]

            out = p.finish_train_step()

        assert out["draft_update_successful"] is False

    def test_abort_consumes_single_data_futures_with_ray_get(self):
        p, wg = _make_tq_policy()
        with patch("nemo_rl.models.policy.tq_policy.ray") as mock_ray:
            p.abort_train_step()
        wg.run_all_workers_single_data.assert_called_once_with(
            "abort_train_step_presharded"
        )
        mock_ray.get.assert_called_once_with(["f0", "f1"])
