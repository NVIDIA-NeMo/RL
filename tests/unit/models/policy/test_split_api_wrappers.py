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

from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.worker_mixin import TQWorkerMixin
from nemo_rl.models.policy.tq_policy import (
    TQPolicy,
    _aggregate_train_results,
    _select_train_prefetch_primary_results,
)


class _SplitStubWorker(TQWorkerMixin):
    """Mixin host recording backend calls; fetch/attach are stubbed."""

    def __init__(self, is_leader: bool = True):
        self.calls: list[tuple] = []
        self._leader = is_leader

    def _fetch(self, meta):
        self.calls.append(("fetch", meta))
        return {"data_from": meta}

    def _attach_or_repack_pack_metadata(self, data, meta):
        self.calls.append(("attach", meta))
        return data

    def _is_replica_leader(self) -> bool:
        return self._leader

    # backend split API
    def begin_train_step(self, loss_fn, gbs=None, mbs=None):
        self.calls.append(("begin", loss_fn, gbs, mbs))

    def train_microbatch(self, data):
        self.calls.append(("train_microbatch", data))

    def finish_train_step(self):
        self.calls.append(("finish",))
        return {
            "global_loss": 1.0,
            "grad_norm": 0.5,
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
        assert w.calls == [("begin", loss_fn, 8, 2)]

    def test_train_microbatch_fetches_attaches_then_dispatches(self):
        w = _SplitStubWorker()
        meta = _meta()
        out = w.train_microbatch_presharded(meta=meta)
        assert out is None  # metrics accumulate in the open-step state
        assert [c[0] for c in w.calls] == ["fetch", "attach", "train_microbatch"]
        assert w.calls[-1][1] == {"data_from": meta}

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
    p.flops_tracker = None
    wg = MagicMock()
    wg.run_all_workers_single_data.return_value = ["f0", "f1"]
    p.worker_group = wg
    p.sharding_annotations = MagicMock()
    p.sharding_annotations.get_axis_size.return_value = 2
    return p, wg


class TestTQPolicySplitFanout:
    def test_train_aggregation_preserves_one_prefetch_source_per_dp(self):
        def _result(rank: int, wait_s: float) -> dict:
            return {
                "rank": rank,
                "global_loss": 1.0,
                "grad_norm": 0.5,
                "all_mb_metrics": {"loss": [0.1]},
                "train_microbatch_prefetch_metrics": {
                    "consumer_wait_s": wait_s,
                },
            }

        out = _aggregate_train_results([_result(0, 0.1), _result(16, 0.2)])

        assert out["train_microbatch_prefetch_source_metrics"] == [
            {"rank": 0, "consumer_wait_s": 0.1},
            {"rank": 16, "consumer_wait_s": 0.2},
        ]

    def test_train_aggregation_uses_pp0_outputs_but_all_stage_metrics(self):
        def _result(rank: int, dp_rank: int, pp_rank: int, loss: float) -> dict:
            return {
                "rank": rank,
                "global_loss": 1.0,
                "grad_norm": 0.5,
                "all_mb_metrics": {"loss": [loss]},
                "train_microbatch_prefetch_metrics": {
                    "consumer_wait_s": loss,
                },
                "train_microbatch_prefetch_dp_rank": dp_rank,
                "train_microbatch_prefetch_pp_rank": pp_rank,
            }

        dp0_pp0 = _result(0, 0, 0, 0.1)
        dp0_pp1 = _result(4, 0, 1, 0.2)
        dp1_pp0 = _result(16, 1, 0, 0.3)
        dp1_pp1 = _result(20, 1, 1, 0.4)

        out = _aggregate_train_results(
            [dp0_pp0, dp1_pp0],
            prefetch_metric_results=[dp0_pp0, dp0_pp1, dp1_pp0, dp1_pp1],
        )

        assert out["all_mb_metrics"]["loss"] == [0.1, 0.3]
        assert out["train_microbatch_prefetch_source_metrics"] == [
            {"rank": 0, "consumer_wait_s": 0.1, "dp_rank": 0, "pp_rank": 0},
            {"rank": 4, "consumer_wait_s": 0.2, "dp_rank": 0, "pp_rank": 1},
            {"rank": 16, "consumer_wait_s": 0.3, "dp_rank": 1, "pp_rank": 0},
            {"rank": 20, "consumer_wait_s": 0.4, "dp_rank": 1, "pp_rank": 1},
        ]

    def test_prefetch_stage_grid_rejects_duplicate_and_missing_coordinate(self):
        def result(dp_rank: int, pp_rank: int) -> dict:
            return {
                "train_microbatch_prefetch_dp_rank": dp_rank,
                "train_microbatch_prefetch_pp_rank": pp_rank,
            }

        stage_results = [
            result(0, 0),
            result(0, 1),
            result(0, 0),
            result(0, 1),
        ]
        with pytest.raises(RuntimeError, match="duplicate stage coordinate"):
            _select_train_prefetch_primary_results(
                stage_results,
                dp_size=2,
                pp_size=2,
            )
        with pytest.raises(RuntimeError, match=r"missing=\[\(1, 1\)\]"):
            _select_train_prefetch_primary_results(
                [result(0, 0), result(0, 1), result(1, 0)],
                dp_size=2,
                pp_size=2,
            )

    def test_train_from_meta_collects_one_result_per_dp_pp_stage(self):
        p, wg = _make_tq_policy()
        p.cfg = {
            "train_global_batch_size": 2,
            "train_micro_batch_size": 1,
            "train_microbatch_prefetch": {"enabled": True},
        }
        p._router_replay_enabled = False

        def axis_size(axis: str) -> int:
            return {"data_parallel": 1, "pipeline_parallel": 2}[axis]

        p.sharding_annotations.get_axis_size.side_effect = axis_size

        def stage_result(rank: int, pp_rank: int) -> dict:
            return {
                "rank": rank,
                "global_loss": 1.0,
                "grad_norm": 0.5,
                "all_mb_metrics": {"loss": [float(pp_rank)]},
                "train_microbatch_prefetch_metrics": {
                    "consumer_wait_s": 0.1 + pp_rank,
                },
                "train_microbatch_prefetch_dp_rank": 0,
                "train_microbatch_prefetch_pp_rank": pp_rank,
            }

        wg.get_all_worker_results.return_value = [
            stage_result(0, 0),
            stage_result(4, 1),
        ]
        meta = _meta()
        with (
            patch.object(TQPolicy, "_stamp_pad_seqlen"),
            patch.object(TQPolicy, "_packing_args", return_value=(None, None)),
            patch(
                "nemo_rl.models.policy.tq_policy.shard_meta_for_dp",
                return_value=([meta], None),
            ),
        ):
            out = p.train_from_meta(
                meta,
                loss_fn=object(),
                sample_mask=torch.ones(2),
                token_mask=torch.ones(2, 4),
            )

        dispatch = wg.run_all_workers_sharded_data.call_args
        assert dispatch.kwargs["output_is_replicated"] == [
            "context_parallel",
            "tensor_parallel",
        ]
        assert "pipeline_parallel" in dispatch.kwargs["replicate_on_axes"]
        assert out["all_mb_metrics"]["loss"] == [0.0]
        assert [
            (metric["dp_rank"], metric["pp_rank"])
            for metric in out["train_microbatch_prefetch_source_metrics"]
        ] == [(0, 0), (0, 1)]

    def test_train_from_meta_legacy_output_remains_pp_replicated(self):
        p, wg = _make_tq_policy()
        result = {
            "global_loss": 1.0,
            "grad_norm": 0.5,
            "all_mb_metrics": {"loss": [0.1]},
        }
        wg.get_all_worker_results.return_value = [result, result]
        meta = _meta()
        with (
            patch.object(TQPolicy, "_stamp_pad_seqlen"),
            patch.object(TQPolicy, "_packing_args", return_value=(None, None)),
            patch(
                "nemo_rl.models.policy.tq_policy.shard_meta_for_dp",
                return_value=([meta, meta], None),
            ),
        ):
            p.train_from_meta(meta, loss_fn=object())

        assert wg.run_all_workers_sharded_data.call_args.kwargs[
            "output_is_replicated"
        ] == [
            "context_parallel",
            "tensor_parallel",
            "pipeline_parallel",
        ]

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

    def test_train_microbatches_from_meta_dispatches_and_returns_none(self):
        p, wg = _make_tq_policy()
        meta = _meta()
        with (
            patch.object(TQPolicy, "_stamp_pad_seqlen"),
            patch.object(TQPolicy, "_packing_args", return_value=(None, None)),
            patch(
                "nemo_rl.models.policy.tq_policy.shard_meta_for_dp",
                return_value=([meta, meta], None),
            ),
        ):
            out = p.train_microbatches_from_meta(meta)
        assert out is None
        assert (
            wg.run_all_workers_sharded_data.call_args.args[0]
            == "train_microbatch_presharded"
        )
        # sharded dispatch returns a MultiWorkerFuture → waited via
        # get_all_worker_results (unlike the single-data fan-outs)
        wg.get_all_worker_results.assert_called_once()

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

    def test_abort_consumes_single_data_futures_with_ray_get(self):
        p, wg = _make_tq_policy()
        with patch("nemo_rl.models.policy.tq_policy.ray") as mock_ray:
            p.abort_train_step()
        wg.run_all_workers_single_data.assert_called_once_with(
            "abort_train_step_presharded"
        )
        mock_ray.get.assert_called_once_with(["f0", "f1"])
