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

from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.worker_mixin import TQWorkerMixin
from nemo_rl.models.policy.tq_policy import TQPolicy


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


class TestPrepareLogprobsFromMeta:
    """SC's per-group logprob refresh hook: a thin flag-gated dispatcher
    over get_logprobs_from_meta / get_reference_policy_logprobs_from_meta."""

    def test_both_flags_dispatch_both(self):
        p = object.__new__(TQPolicy)
        meta = _meta()
        with (
            patch.object(TQPolicy, "get_logprobs_from_meta") as mock_policy_lp,
            patch.object(
                TQPolicy, "get_reference_policy_logprobs_from_meta"
            ) as mock_ref_lp,
        ):
            out = p.prepare_logprobs_from_meta(
                meta,
                refresh_policy_logprobs=True,
                refresh_reference_logprobs=True,
            )
        assert out is None  # workers write back to TQ; no Ray payload
        mock_policy_lp.assert_called_once()
        assert mock_policy_lp.call_args.args[0] is meta
        mock_ref_lp.assert_called_once()
        assert mock_ref_lp.call_args.args[0] is meta

    def test_single_flag_dispatches_only_that_refresh(self):
        p = object.__new__(TQPolicy)
        with (
            patch.object(TQPolicy, "get_logprobs_from_meta") as mock_policy_lp,
            patch.object(
                TQPolicy, "get_reference_policy_logprobs_from_meta"
            ) as mock_ref_lp,
        ):
            p.prepare_logprobs_from_meta(_meta(), refresh_reference_logprobs=True)
        mock_policy_lp.assert_not_called()
        mock_ref_lp.assert_called_once()

    def test_no_flags_is_a_no_op(self):
        p = object.__new__(TQPolicy)
        with (
            patch.object(TQPolicy, "get_logprobs_from_meta") as mock_policy_lp,
            patch.object(
                TQPolicy, "get_reference_policy_logprobs_from_meta"
            ) as mock_ref_lp,
        ):
            p.prepare_logprobs_from_meta(_meta())
        mock_policy_lp.assert_not_called()
        mock_ref_lp.assert_not_called()
