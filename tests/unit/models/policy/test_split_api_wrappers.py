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

import torch

from nemo_rl.algorithms.loss.interfaces import LossType
from nemo_rl.data.packed_rollouts import (
    PACKED_ATTENTION_SEGMENT_LENGTHS,
    PACKED_ATTENTION_SELECTED_SEGMENTS,
    TREE_ATTENTION_LAYOUTS,
    TreeAttentionLayout,
)
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.schema import (
    DP_TRAIN_FIELDS,
    OPD_FULL_HIDDEN_STATES_FIELD,
    OPD_FULL_LOGITS_FIELD,
    ROUTED_EXPERTS_FIELD,
)
from nemo_rl.data_plane.worker_mixin import TQWorkerMixin
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.policy.tq_policy import TREE_ATTENTION_TENSOR_FIELDS, TQPolicy


class _SplitStubWorker(TQWorkerMixin):
    """Mixin host recording backend calls; fetch/attach are stubbed."""

    def __init__(self, is_leader: bool = True):
        self.calls: list[tuple] = []
        self._leader = is_leader

    def _fetch(self, meta, **kwargs):
        self.calls.append(("fetch", meta))
        data = {"data_from": meta}
        preprocess = kwargs.get("preprocess")
        return preprocess(self, data) if preprocess is not None else data

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

    def test_train_microbatch_expands_only_calls_selected_for_this_dp(self):
        w = _SplitStubWorker()
        data = BatchedDataDict(
            {
                "input_ids": torch.tensor([[10, 11, 12, 20, 21], [30, 31, 32, 33, 0]]),
                "input_lengths": torch.tensor([5, 4]),
                "sample_mask": torch.tensor([1.0, 0.5]),
            }
        )
        w._fetch = MagicMock(
            side_effect=lambda _meta, **kwargs: kwargs["preprocess"](w, data)
        )
        meta = KVBatchMeta(
            partition_id="train",
            task_name="train",
            sample_ids=["s0", "s1"],
            sequence_lengths=[5, 4],
            extra_info={
                PACKED_ATTENTION_SEGMENT_LENGTHS: [[3, 2], [4]],
                PACKED_ATTENTION_SELECTED_SEGMENTS: [(1, 0), (0, 1)],
            },
        )

        w.train_microbatch_presharded(meta=meta)

        trained = w.calls[-1][1]
        assert torch.equal(trained["input_lengths"], torch.tensor([4, 2]))
        assert torch.equal(
            trained["input_ids"],
            torch.tensor([[30, 31, 32, 33], [20, 21, 0, 0]]),
        )

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
    p.cfg = {
        "train_global_batch_size": 8,
        "train_micro_batch_size": 2,
        "max_total_sequence_length": 131072,
        "megatron_cfg": {"enabled": True},
    }
    p._router_replay_enabled = False
    # opd_full off, as __init__ leaves it when the config block is absent.
    p._opd_full_field = None
    p.dp_client = MagicMock()
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
        loss_fn = MagicMock(loss_type=LossType.SEQUENCE_LEVEL)
        with patch("nemo_rl.models.policy.tq_policy.ray") as mock_ray:
            p.begin_train_step(loss_fn=loss_fn)
        wg.run_all_workers_single_data.assert_called_once_with(
            "begin_train_step_presharded", loss_fn=loss_fn, gbs=8, mbs=2
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

    def test_train_microbatches_fetches_only_requested_fields(self):
        p, _ = _make_tq_policy()
        meta = _meta()
        train_fields = tuple(
            field
            for field in DP_TRAIN_FIELDS
            if field not in {"prev_logprobs", "reference_policy_logprobs"}
        )
        with (
            patch.object(TQPolicy, "_stamp_pad_seqlen"),
            patch.object(TQPolicy, "_packing_args", return_value=(None, None)),
            patch(
                "nemo_rl.models.policy.tq_policy.shard_meta_for_dp",
                return_value=([meta, meta], None),
            ) as mock_shard,
        ):
            p.train_microbatches_from_meta(meta, train_fields=train_fields)

        train_meta = mock_shard.call_args.args[0]
        assert train_meta.fields == list(train_fields)

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

    def test_finish_propagates_mtp_metrics(self):
        """Worker-reduced MTP metrics survive the TQPolicy aggregation layer."""
        p, _ = _make_tq_policy()
        with patch("nemo_rl.models.policy.tq_policy.ray") as mock_ray:
            mock_ray.get.return_value = [
                {
                    "global_loss": 1.0,
                    "grad_norm": 0.5,
                    "all_mb_metrics": {"loss": [0.1]},
                    "mtp_metrics": {
                        "mtp_1_loss": 0.25,
                        "mtp_1_acceptance_rate": 75.0,
                        "grad_norm": 1.25,
                    },
                    "is_replica_leader": True,
                }
            ]
            out = p.finish_train_step()

        assert out["mtp_metrics"] == {
            "mtp_1_loss": 0.25,
            "mtp_1_acceptance_rate": 75.0,
            "grad_norm": 1.25,
        }

    def test_abort_consumes_single_data_futures_with_ray_get(self):
        p, wg = _make_tq_policy()
        with patch("nemo_rl.models.policy.tq_policy.ray") as mock_ray:
            p.abort_train_step()
        wg.run_all_workers_single_data.assert_called_once_with(
            "abort_train_step_presharded"
        )
        mock_ray.get.assert_called_once_with(["f0", "f1"])


class TestTQPolicyExactCallPacking:
    @staticmethod
    def _packed_meta() -> KVBatchMeta:
        return KVBatchMeta(
            partition_id="train",
            task_name="train",
            sample_ids=["s0", "s1"],
            fields=list(DP_TRAIN_FIELDS),
            sequence_lengths=[5, 4],
            extra_info={PACKED_ATTENTION_SEGMENT_LENGTHS: [[3, 2], [4]]},
        )

    def test_logprob_dispatch_reorders_calls_and_writes_logical_rows(self):
        p, wg = _make_tq_policy()
        meta = self._packed_meta()
        p.write_to_dataplane = MagicMock()
        # DP-concatenated call order is [call 2, call 0, call 1].
        wg.get_all_worker_results.return_value = [
            BatchedDataDict({"logprobs": torch.tensor([[30.0, 31.0, 32.0, 33.0]])}),
            BatchedDataDict(
                {
                    "logprobs": torch.tensor(
                        [
                            [10.0, 11.0, 12.0],
                            [20.0, 21.0, 0.0],
                        ]
                    )
                }
            ),
        ]
        with (
            patch.object(TQPolicy, "_stamp_pad_seqlen"),
            patch.object(TQPolicy, "_packing_args", return_value=(None, None)),
            patch(
                "nemo_rl.models.policy.tq_policy.shard_meta_for_dp",
                return_value=([meta, meta], [2, 0, 1]),
            ),
        ):
            p.get_logprobs_from_meta(meta)

        fields = p.write_to_dataplane.call_args.kwargs["fields"]
        assert torch.equal(
            fields["prev_logprobs"],
            torch.tensor(
                [
                    [10.0, 11.0, 12.0, 20.0, 21.0],
                    [30.0, 31.0, 32.0, 33.0, 0.0],
                ]
            ),
        )

    def test_logprob_dispatch_reassembles_bounded_tree_fragments(self):
        p, wg = _make_tq_policy()
        p.cfg["sequence_packing"] = {"enabled": True}
        layout = TreeAttentionLayout(
            segment_lengths=(3, 2, 2),
            segment_parents=(-1, 0, 0),
            segment_depths=(0, 3, 3),
            edge_source_indices=(1, 3, 5),
            original_token_count=9,
        )
        meta = KVBatchMeta(
            partition_id="train",
            task_name="prev_lp",
            sample_ids=["s0"],
            fields=list(DP_TRAIN_FIELDS),
            sequence_lengths=[7],
            extra_info={TREE_ATTENTION_LAYOUTS: [layout]},
        )
        p.write_to_dataplane = MagicMock()
        # DP concatenation is fragment 1 then fragment 0; the inverse
        # permutation restores planner order before edge scattering.
        wg.get_all_worker_results.return_value = [
            BatchedDataDict({"logprobs": torch.tensor([[0.0, 30.0, 0.0]])}),
            BatchedDataDict({"logprobs": torch.tensor([[0.0, 10.0, 20.0]])}),
        ]
        packing_args = {
            "algorithm": "modified_first_fit_decreasing",
            "input_key": "input_ids",
            "input_lengths_key": "input_lengths",
            "max_tokens_per_microbatch": 5,
            "sequence_length_pad_multiple": 1,
        }
        with (
            patch.object(TQPolicy, "_stamp_pad_seqlen"),
            patch.object(TQPolicy, "_packing_args", return_value=(packing_args, None)),
            patch(
                "nemo_rl.models.policy.tq_policy.shard_meta_for_dp",
                return_value=([meta, meta], [1, 0]),
            ),
        ):
            p.get_logprobs_from_meta(meta)

        assert torch.equal(
            p.write_to_dataplane.call_args.kwargs["fields"]["prev_logprobs"],
            torch.tensor([[0.0, 10.0, 20.0, 30.0]]),
        )

    def test_sync_train_counts_physical_calls_but_steps_logical_rollouts(self):
        p, wg = _make_tq_policy()
        meta = self._packed_meta()
        wg.get_all_worker_results.return_value = [
            {
                "global_loss": torch.tensor(1.0),
                "grad_norm": torch.tensor(2.0),
                "all_mb_metrics": {},
            }
        ]
        loss_fn = MagicMock()
        with (
            patch.object(TQPolicy, "_stamp_pad_seqlen"),
            patch.object(TQPolicy, "_packing_args", return_value=(None, None)),
            patch(
                "nemo_rl.models.policy.tq_policy.shard_meta_for_dp",
                return_value=([meta, meta], None),
            ) as mock_shard,
        ):
            p.train_from_meta(meta, loss_fn=loss_fn, gbs=2, mbs=1)

        assert mock_shard.call_args.kwargs["batch_size"] == 2
        common_kwargs = wg.run_all_workers_sharded_data.call_args.kwargs[
            "common_kwargs"
        ]
        assert common_kwargs == {
            "loss_fn": loss_fn,
            "eval_mode": False,
            "gbs": 3,
            "mbs": 1,
            "scheduler_step_increment": 2,
        }


class TestTQPolicyOPDFullColumn:
    def test_train_microbatches_request_the_teacher_payload_column(self):
        p, _ = _make_tq_policy()
        p._opd_full_field = OPD_FULL_HIDDEN_STATES_FIELD
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
        assert train_meta.fields == [*DP_TRAIN_FIELDS, OPD_FULL_HIDDEN_STATES_FIELD]

    def test_prepare_step_registers_the_teacher_payload_column(self):
        """The TQ schema is fixed at registration.

        A column missing there is rejected on the teacher's write rather than
        silently created.
        """
        p, _ = _make_tq_policy()
        p._opd_full_field = OPD_FULL_LOGITS_FIELD
        p.tq_partition_id = "train"
        p.dp_client = MagicMock()

        p.prepare_step(num_samples=4, group_size=2)

        fields = p.dp_client.register_partition.call_args.kwargs["fields"]
        assert fields == [
            *DP_TRAIN_FIELDS,
            *TREE_ATTENTION_TENSOR_FIELDS,
            OPD_FULL_LOGITS_FIELD,
        ]

    def test_prepare_step_leaves_out_teacher_payload_when_opd_full_is_off(self):
        p, _ = _make_tq_policy()
        p.tq_partition_id = "train"
        p.dp_client = MagicMock()

        p.prepare_step(num_samples=4)

        fields = p.dp_client.register_partition.call_args.kwargs["fields"]
        assert fields == [*DP_TRAIN_FIELDS, *TREE_ATTENTION_TENSOR_FIELDS]
