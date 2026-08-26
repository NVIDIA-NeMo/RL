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

"""CPU tests for the TQ-mediated value model (worker wrapper + driver fan-out).

Same shape as tests/unit/models/policy/test_split_api_wrappers.py: the
GPU-gated PPO tests never reach these two layers cheaply, and the things
that break here are contract-level — a forward pass whose result is
returned through Ray instead of written to TQ, or a train dispatch that
asks workers for a column no producer wrote.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.schema import (
    DP_VALUE_TRAIN_FIELDS,
    GLOBAL_FORWARD_PAD_SEQLEN,
    VALUE_SEED_FIELDS,
)
from nemo_rl.data_plane.worker_mixin import TQWorkerMixin
from nemo_rl.models.value.tq_value import TQValue


class _ValueStubWorker(TQWorkerMixin):
    """Mixin host recording backend calls; fetch/attach are stubbed."""

    def __init__(self, is_leader: bool = True, values: torch.Tensor | None = None):
        self.calls: list[tuple] = []
        self._leader = is_leader
        self._dp_client = MagicMock()
        self._values = values if values is not None else torch.ones(2, 3)

    def _fetch(self, meta):
        self.calls.append(("fetch", meta))
        return {"data_from": meta}

    def _attach_or_repack_pack_metadata(self, data, meta):
        self.calls.append(("attach", meta))
        return data

    def _is_replica_leader(self) -> bool:
        return self._leader

    def get_values(self, data, micro_batch_size=None):
        self.calls.append(("get_values", data, micro_batch_size))
        return {"values": self._values}


def _meta(sample_ids: list[str] | None = None) -> KVBatchMeta:
    return KVBatchMeta(
        partition_id="rollout_data",
        task_name="train",
        sample_ids=sample_ids if sample_ids is not None else ["s0", "s1"],
    )


class TestGetValuesPresharded:
    def test_fetches_attaches_then_writes_values_back(self):
        w = _ValueStubWorker()
        meta = _meta()

        with patch("nemo_rl.data_plane.column_io.write_columns") as write_columns:
            out = w.get_values_presharded(meta=meta, micro_batch_size=4)

        # The [B, S] tensor goes to TQ, not through Ray.
        assert out is None
        assert [c[0] for c in w.calls] == ["fetch", "attach", "get_values"]
        assert w.calls[2][1] == {"data_from": meta}
        assert w.calls[2][2] == 4
        written = write_columns.call_args.args[2]
        assert torch.equal(written["values"], torch.ones(2, 3))

    def test_non_leader_twin_does_not_write(self):
        """TP/CP/PP twins hold identical copies; a second writer is the
        duplicate-write bug the leader gate exists to prevent."""
        w = _ValueStubWorker(is_leader=False)

        with patch("nemo_rl.data_plane.column_io.write_columns") as write_columns:
            w.get_values_presharded(meta=_meta())

        write_columns.assert_not_called()

    def test_rejects_batch_dim_mismatch(self):
        w = _ValueStubWorker(values=torch.ones(3, 3))

        with pytest.raises(ValueError, match="shape mismatch"):
            w.get_values_presharded(meta=_meta(["s0", "s1"]))


def _make_tq_value() -> tuple[TQValue, MagicMock]:
    """Bare TQValue with the attributes the fan-out touches."""
    v = object.__new__(TQValue)
    v.cfg = {"train_global_batch_size": 8, "train_micro_batch_size": 2}
    wg = MagicMock()
    v.worker_group = wg
    v.sharding_annotations = MagicMock()
    v.sharding_annotations.get_axis_size.return_value = 2
    return v, wg


class TestTQValueFanout:
    def test_get_values_from_meta_narrows_fields_and_returns_none(self):
        v, wg = _make_tq_value()
        meta = _meta()
        with (
            patch.object(TQValue, "_stamp_pad_seqlen"),
            patch.object(
                TQValue, "_packing_args", return_value=(None, None)
            ) as mock_packing,
            patch(
                "nemo_rl.models.value.tq_value.shard_meta_for_dp",
                return_value=([meta, meta], None),
            ) as mock_shard,
        ):
            out = v.get_values_from_meta(meta)

        assert out is None
        value_meta = mock_shard.call_args.args[0]
        assert value_meta.fields == list(VALUE_SEED_FIELDS)
        assert value_meta.task_name == "value_fwd"
        assert (
            wg.run_all_workers_sharded_data.call_args.args[0] == "get_values_presharded"
        )
        wg.get_all_worker_results.assert_called_once()
        # The forward pass is inference-shaped, so it sizes microbatches off
        # logprob_mb_tokens rather than the (larger) train budget.
        assert mock_packing.call_args.args[0] == "logprob_mb_tokens"

    def test_train_from_meta_requests_the_value_train_columns(self):
        v, wg = _make_tq_value()
        meta = _meta()
        wg.get_all_worker_results.return_value = [
            {
                "global_loss": 1.0,
                "grad_norm": 0.5,
                "all_mb_metrics": {"loss": [0.1]},
            }
        ]
        with (
            patch.object(TQValue, "_stamp_pad_seqlen"),
            patch.object(TQValue, "_packing_args", return_value=(None, None)),
            patch(
                "nemo_rl.models.value.tq_value.shard_meta_for_dp",
                return_value=([meta, meta], None),
            ) as mock_shard,
        ):
            out = v.train_from_meta(meta, loss_fn="LF")

        train_meta = mock_shard.call_args.args[0]
        assert train_meta.fields == list(DP_VALUE_TRAIN_FIELDS)
        assert "returns" in train_meta.fields and "values" in train_meta.fields
        # advantages / prev_logprobs are the policy's business only.
        assert "advantages" not in train_meta.fields
        assert train_meta.task_name == "value_train"
        assert wg.run_all_workers_sharded_data.call_args.args[0] == "train_presharded"
        assert wg.run_all_workers_sharded_data.call_args.kwargs["common_kwargs"] == {
            "loss_fn": "LF",
            "eval_mode": False,
            "gbs": 8,
            "mbs": 2,
        }
        assert out["loss"] == 1.0
        assert out["all_mb_metrics"]["loss"] == [0.1]

    def test_train_from_meta_concatenates_per_rank_metrics(self):
        v, wg = _make_tq_value()
        meta = _meta()

        def _result(loss: float) -> dict:
            return {
                "global_loss": 1.0,
                "grad_norm": 0.5,
                "all_mb_metrics": {"loss": [loss]},
            }

        wg.get_all_worker_results.return_value = [_result(0.1), _result(0.2)]
        with (
            patch.object(TQValue, "_stamp_pad_seqlen"),
            patch.object(TQValue, "_packing_args", return_value=(None, None)),
            patch(
                "nemo_rl.models.value.tq_value.shard_meta_for_dp",
                return_value=([meta, meta], None),
            ),
        ):
            out = v.train_from_meta(meta, loss_fn="LF")

        assert out["all_mb_metrics"]["loss"] == [0.1, 0.2]


class TestPadTargetIsolation:
    """Each dispatch mints its own pad target, in both directions: the critic
    goes first within a step, and with ppo_epochs > 1 the policy has already
    stamped by the time the critic runs again."""

    def test_dispatch_does_not_stamp_the_callers_meta(self):
        v, _ = _make_tq_value()
        v.use_dynamic_batches = False
        v.use_sequence_packing = False
        meta = KVBatchMeta(
            partition_id="rollout_data",
            task_name="train",
            sample_ids=["s0", "s1"],
            sequence_lengths=[7, 9],
        )
        with patch(
            "nemo_rl.models.value.tq_value.shard_meta_for_dp",
            return_value=([meta, meta], None),
        ) as mock_shard:
            v.get_values_from_meta(meta)

        assert GLOBAL_FORWARD_PAD_SEQLEN not in meta.extra_info
        # ...but the dispatched meta carries one, so DP ranks still agree.
        assert GLOBAL_FORWARD_PAD_SEQLEN in mock_shard.call_args.args[0].extra_info

    def test_a_stamped_caller_meta_does_not_decide_the_critic_pad(self):
        """ppo_epochs > 1: the policy has stamped the shared meta by epoch 1."""
        v, _ = _make_tq_value()
        v.use_dynamic_batches = False
        v.use_sequence_packing = False
        meta = KVBatchMeta(
            partition_id="rollout_data",
            task_name="train",
            sample_ids=["s0", "s1"],
            sequence_lengths=[7, 9],
            extra_info={GLOBAL_FORWARD_PAD_SEQLEN: 4096},
        )
        with patch(
            "nemo_rl.models.value.tq_value.shard_meta_for_dp",
            return_value=([meta, meta], None),
        ) as mock_shard:
            v.get_values_from_meta(meta)

        dispatched = mock_shard.call_args.args[0]
        assert dispatched.extra_info[GLOBAL_FORWARD_PAD_SEQLEN] != 4096
        # The caller's value survives for whoever minted it.
        assert meta.extra_info[GLOBAL_FORWARD_PAD_SEQLEN] == 4096
