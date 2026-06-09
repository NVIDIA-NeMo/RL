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
"""CPU state-machine tests for MegatronPolicyWorkerImpl's split-API.

These tests cover the lifecycle and call-order invariants — they do NOT
exercise real distributed comms, the mcore scheduler, or the optimizer.
Numerical equivalence vs sync ``train()`` lives in the GPU parity tests.

The bugs these catch:
  - silent gradient over-counting if ``model.no_sync()`` is not wrapped
    around ``megatron_forward_backward`` (the mcore DDP hooks would
    dispatch a per-call reduce, ADDING to an already-reduced bucket).
  - PP>1 pipeline-schedule bypass if ``model.config.grad_sync_func`` is
    not nulled for the step's duration.
  - ``trainer_version`` advancing on abort.
  - ``zero_grad_buffer`` not called at begin (mcore's contiguous grad
    buffer leaks stale grads otherwise).
  - off-by-one in ``total_num_microbatches`` (used to scale MoE aux-loss).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

# megatron.bridge is only available with the mcore extras. Without it the
# eager import of megatron_policy_worker (transitively imports megatron.bridge)
# fails at COLLECTION time on non-mcore shards, which then breaks every other
# test in that shard. importorskip stops collection cleanly here.
pytest.importorskip("megatron.bridge")

# Eagerly import the worker module so ``unittest.mock.patch`` can resolve
# attributes on it via ``getattr``. Without this the patch path
# ``nemo_rl.models.policy.workers.megatron_policy_worker.<symbol>`` fails
# at ``getattr(workers, "megatron_policy_worker")``.
import nemo_rl.models.policy.workers.megatron_policy_worker  # noqa: E402,F401

pytestmark = pytest.mark.mcore

# Module path of the worker under test
WORKER_MOD = "nemo_rl.models.policy.workers.megatron_policy_worker"


# ── Mock fabric ──────────────────────────────────────────────────────────


def _make_mock_model():
    """A mcore-DDP-shaped mock: exposes the methods + attributes the
    split-API touches, plus an ``inference_params`` attribute and a
    ``modules()`` that yields nothing (so the inference-cache reset loop
    is a no-op)."""
    model = MagicMock()
    model.config = MagicMock()
    model.config.grad_sync_func = "ORIGINAL_GRAD_SYNC_FUNC"  # sentinel
    model.config.num_moe_experts = None  # disable MoE branch
    # no_sync() is a context manager — return a MagicMock that supports
    # __enter__/__exit__ so the `with self.model.no_sync():` block works.
    model.no_sync = MagicMock(
        return_value=MagicMock(
            __enter__=MagicMock(return_value=None),
            __exit__=MagicMock(return_value=False),
        )
    )
    model.modules = MagicMock(return_value=iter([]))
    model.inference_params = None
    model.parameters = MagicMock(
        return_value=iter([])
    )  # no params for the rescale loop
    return model


def _make_worker(loss_type):
    """Construct a MegatronPolicyWorkerImpl instance with all heavy
    attributes mocked. Bypasses __init__ via ``object.__new__``."""
    # Lazy import so the module-level mcore imports happen inside the
    # mcore-marked test process.
    from nemo_rl.models.policy.workers.megatron_policy_worker import (
        MegatronPolicyWorkerImpl,
    )

    w = object.__new__(MegatronPolicyWorkerImpl)
    w.model = _make_mock_model()
    w.optimizer = MagicMock()
    # MegatronOptimizer.step returns (success, grad_norm, num_zeros)
    w.optimizer.step.return_value = (True, 0.5, 0)
    w.optimizer.param_groups = [{"lr": 1e-4, "weight_decay": 0.01}]
    w.scheduler = MagicMock()
    w.scheduler.get_lr.return_value = 1e-4
    w.scheduler.get_wd.return_value = 0.01
    w.mcore_state = MagicMock()
    w.mcore_state.straggler_timer = None
    w.cfg = {
        "train_global_batch_size": 32,
        "train_micro_batch_size": 4,
        "megatron_cfg": {
            "empty_unused_memory_level": 0,
            "moe_per_layer_logging": False,
            "use_linear_ce_fusion_loss": False,
        },
    }
    w.dp_size = 2
    w.cp_size = 1
    w.sampling_params = None
    w.draft_model = None
    w.defer_fp32_logits = False
    w.dtype = torch.float32
    w._is_reward_model = False

    # Stash a loss_fn with the requested loss_type for tests that need one.
    w._test_loss_fn = MagicMock(loss_type=loss_type)
    return w


@pytest.fixture
def mock_module_symbols():
    """Patch every module-level symbol that the split-API methods call
    into. Yields a dict of name → mock for assertions."""
    # Make `aggregate_training_statistics` return ({}, scalar) — what the
    # finish path expects.
    agg_ret = ({"loss": [0.0]}, torch.tensor(0.5))

    patches = {
        "megatron_forward_backward": [
            {"loss": 0.5, "global_valid_seqs": 8.0, "global_valid_toks": 256.0}
        ],
        "get_microbatch_iterator": (iter([]), 2, 4, 16, 16),  # 2 pipeline mbs per call
        "LossPostProcessor": MagicMock(),
        "broadcast_loss_metrics_from_last_stage": lambda m: m,
        "get_pg_collection": MagicMock(mp=MagicMock()),
        "logical_and_across_model_parallel_group": lambda v, mp_group: v,
        "reduce_max_stat_across_model_parallel_group": lambda v, mp_group: v,
        "aggregate_training_statistics": agg_ret,
        "get_moe_metrics": MagicMock(return_value={}),
    }

    with (
        patch(
            f"{WORKER_MOD}.megatron_forward_backward",
            return_value=patches["megatron_forward_backward"],
        ) as mfb,
        patch(
            f"{WORKER_MOD}.get_microbatch_iterator",
            return_value=patches["get_microbatch_iterator"],
        ) as gmi,
        patch(
            f"{WORKER_MOD}.LossPostProcessor", return_value=patches["LossPostProcessor"]
        ) as lpp,
        patch(
            f"{WORKER_MOD}.broadcast_loss_metrics_from_last_stage",
            side_effect=patches["broadcast_loss_metrics_from_last_stage"],
        ) as bcast,
        patch(
            f"{WORKER_MOD}.get_pg_collection", return_value=patches["get_pg_collection"]
        ) as gpgc,
        patch(
            f"{WORKER_MOD}.logical_and_across_model_parallel_group",
            side_effect=patches["logical_and_across_model_parallel_group"],
        ) as land,
        patch(
            f"{WORKER_MOD}.reduce_max_stat_across_model_parallel_group",
            side_effect=patches["reduce_max_stat_across_model_parallel_group"],
        ) as rmax,
        patch(
            f"{WORKER_MOD}.aggregate_training_statistics",
            return_value=patches["aggregate_training_statistics"],
        ) as agg,
        patch(f"{WORKER_MOD}.get_moe_metrics", return_value={}) as moe,
        patch(f"{WORKER_MOD}.get_rerun_state_machine") as grsm,
        patch(f"{WORKER_MOD}.parallel_state") as pstate,
        patch("torch.distributed.all_reduce") as ar,
        patch("torch.cuda.empty_cache") as cec,
        patch("torch.cuda.get_device_name", return_value="H100"),
        patch("torch.distributed.get_rank", return_value=0),
    ):
        # rerun state machine: fire forward+backward once per train_microbatch
        rsm = MagicMock()
        rsm.should_run_forward_backward.side_effect = [True, False] * 100
        grsm.return_value = rsm

        # parallel_state mocks
        pstate.is_pipeline_last_stage.return_value = True
        pstate.get_data_parallel_group.return_value = MagicMock()

        yield {
            "mfb": mfb,
            "gmi": gmi,
            "lpp": lpp,
            "bcast": bcast,
            "gpgc": gpgc,
            "land": land,
            "rmax": rmax,
            "agg": agg,
            "moe": moe,
            "grsm": grsm,
            "pstate": pstate,
            "all_reduce": ar,
            "empty_cache": cec,
        }


def _fake_batch():
    """A minimal BatchedDataDict-ish object the mask-sum block can read.
    train_microbatch reads ``data["sample_mask"]``, ``data["token_mask"]``,
    and (only as a fallback for the no-token-mask path) ``data["input_ids"]``."""
    # 8 samples, all valid (mask=1); 256 valid tokens each
    sample_mask = torch.ones(8, dtype=torch.float32)
    token_mask = torch.ones(8, 257, dtype=torch.float32)  # token_mask[:, 1:] → 256 toks
    input_ids = torch.zeros(8, 257, dtype=torch.long)
    return {
        "sample_mask": sample_mask,
        "token_mask": token_mask,
        "input_ids": input_ids,
    }


# ── BEGIN ────────────────────────────────────────────────────────────────


class TestBegin:
    def test_opens_state(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.begin_train_step("step-0", loss_fn=w._test_loss_fn, gbs=16, mbs=4)
        assert w._train_step_state is not None
        assert w._train_step_state["step_id"] == "step-0"
        assert w._train_step_state["loss_type"] == LossType.TOKEN_LEVEL
        assert w._train_step_state["gbs"] == 16
        assert w._train_step_state["mbs"] == 4
        assert w._train_step_state["total_num_microbatches"] == 0

    def test_calls_zero_grad_and_zero_grad_buffer(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.begin_train_step("step-0", loss_fn=w._test_loss_fn)
        w.model.zero_grad_buffer.assert_called_once()
        w.optimizer.zero_grad.assert_called_once()
        w.model.train.assert_called_once()

    def test_saves_and_nulls_grad_sync_func(self, mock_module_symbols):
        """The PP scheduler's direct reduce dispatch must be suppressed
        for the duration of the step. Otherwise PP>1 silently corrupts
        grads even when ``no_sync`` is set on the bucket groups."""
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        assert w.model.config.grad_sync_func == "ORIGINAL_GRAD_SYNC_FUNC"
        w.begin_train_step("step-0", loss_fn=w._test_loss_fn)
        assert w.model.config.grad_sync_func is None
        assert w._train_step_state["saved_grad_sync_func"] == "ORIGINAL_GRAD_SYNC_FUNC"

    def test_double_begin_raises(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.begin_train_step("step-0", loss_fn=w._test_loss_fn)
        with pytest.raises(RuntimeError, match="already open"):
            w.begin_train_step("step-1", loss_fn=w._test_loss_fn)

    def test_uses_cfg_defaults_when_gbs_mbs_omitted(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.begin_train_step("step-0", loss_fn=w._test_loss_fn)
        assert w._train_step_state["gbs"] == w.cfg["train_global_batch_size"]
        assert w._train_step_state["mbs"] == w.cfg["train_micro_batch_size"]


# ── _assert_step_open ────────────────────────────────────────────────────


class TestAssertStepOpen:
    def test_raises_when_no_step_open(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        with pytest.raises(RuntimeError, match="no train step open"):
            w._assert_step_open("step-0")

    def test_raises_on_step_id_mismatch(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.begin_train_step("step-correct", loss_fn=w._test_loss_fn)
        with pytest.raises(RuntimeError, match="step_id mismatch"):
            w._assert_step_open("step-WRONG")

    def test_train_microbatch_without_begin_raises(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        with pytest.raises(RuntimeError, match="no train step open"):
            w.train_microbatch("step-0", _fake_batch())

    def test_finish_without_begin_raises(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        with pytest.raises(RuntimeError, match="no train step open"):
            w.finish_train_step("step-0")


# ── train_microbatch ─────────────────────────────────────────────────────


class TestTrainMicrobatch:
    def test_wraps_forward_backward_in_no_sync(self, mock_module_symbols):
        """The single most important assertion in this file. Without the
        no_sync wrap, mcore DDP dispatches a per-call cross-DP reduce on
        the partially-accumulated buffer — silently corrupting grads."""
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.begin_train_step("s0", loss_fn=w._test_loss_fn)
        w.train_microbatch("s0", _fake_batch())
        # no_sync() must have been ENTERED (called as a context manager).
        # MagicMock with __enter__/__exit__ records the __enter__ call.
        ctx = w.model.no_sync.return_value
        ctx.__enter__.assert_called()
        ctx.__exit__.assert_called()

    def test_invokes_megatron_forward_backward_once(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.begin_train_step("s0", loss_fn=w._test_loss_fn)
        w.train_microbatch("s0", _fake_batch())
        assert mock_module_symbols["mfb"].call_count == 1

    def test_passes_placeholder_n_one_to_loss(self, mock_module_symbols):
        """The N=1 trick: loss must be called with global_valid_*=1 so it
        returns un-normalized sums; finish does the 1/N rescale."""
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.begin_train_step("s0", loss_fn=w._test_loss_fn)
        w.train_microbatch("s0", _fake_batch())
        kwargs = mock_module_symbols["mfb"].call_args.kwargs
        # placeholder_n is a tensor(1.0)
        assert "global_valid_seqs" in kwargs
        assert "global_valid_toks" in kwargs
        assert float(kwargs["global_valid_seqs"].item()) == pytest.approx(1.0)
        assert float(kwargs["global_valid_toks"].item()) == pytest.approx(1.0)

    def test_accumulates_mask_sums_across_calls(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.begin_train_step("s0", loss_fn=w._test_loss_fn)
        # _fake_batch has sample_mask sum = 8, token_mask*sample_mask sum = 8*256 = 2048
        w.train_microbatch("s0", _fake_batch())
        assert float(w._train_step_state["local_valid_seqs"].item()) == pytest.approx(
            8.0
        )
        assert float(w._train_step_state["local_valid_toks"].item()) == pytest.approx(
            2048.0
        )
        w.train_microbatch("s0", _fake_batch())
        assert float(w._train_step_state["local_valid_seqs"].item()) == pytest.approx(
            16.0
        )
        assert float(w._train_step_state["local_valid_toks"].item()) == pytest.approx(
            4096.0
        )

    def test_total_num_microbatches_accumulates(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.begin_train_step("s0", loss_fn=w._test_loss_fn)
        # get_microbatch_iterator mock returns num_microbatches=2 per call
        w.train_microbatch("s0", _fake_batch())
        w.train_microbatch("s0", _fake_batch())
        w.train_microbatch("s0", _fake_batch())
        assert w._train_step_state["total_num_microbatches"] == 6

    def test_does_not_call_optimizer_step(self, mock_module_symbols):
        """trainer_version semantics: optimizer.step() must NOT fire
        per train_microbatch — only at finish."""
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.begin_train_step("s0", loss_fn=w._test_loss_fn)
        w.train_microbatch("s0", _fake_batch())
        w.train_microbatch("s0", _fake_batch())
        w.optimizer.step.assert_not_called()


# ── finish_train_step ────────────────────────────────────────────────────


class TestFinish:
    def _setup_open_step(self, mock_module_symbols, loss_type):
        w = _make_worker(loss_type)
        w.begin_train_step("s0", loss_fn=w._test_loss_fn)
        w.train_microbatch("s0", _fake_batch())
        return w

    def test_rescales_grads_with_inv_n(self, mock_module_symbols):
        """The 1/N rescale must happen ON the local main_grad BEFORE the
        cross-DP reduce — otherwise the reduce sees un-rescaled sums."""
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = self._setup_open_step(mock_module_symbols, LossType.TOKEN_LEVEL)
        w.finish_train_step("s0")
        # scale_gradients should have been called with some 1/N scalar < 1
        w.model.scale_gradients.assert_called_once()
        arg = w.model.scale_gradients.call_args.args[0]
        assert 0 < arg <= 1.0

    def test_start_then_finish_grad_sync_called_after_rescale(
        self, mock_module_symbols
    ):
        """Call order matters: scale_gradients -> start_grad_sync ->
        finish_grad_sync -> optimizer.step."""
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = self._setup_open_step(mock_module_symbols, LossType.TOKEN_LEVEL)
        # Record call order via a shared list
        order: list[str] = []
        w.model.scale_gradients.side_effect = lambda s: order.append("scale")
        w.model.start_grad_sync.side_effect = lambda: order.append("start_sync")
        w.model.finish_grad_sync.side_effect = lambda: order.append("finish_sync")
        w.optimizer.step.side_effect = lambda: (
            order.append("opt_step") or (True, 0.5, 0)
        )
        w.finish_train_step("s0")
        assert order == ["scale", "start_sync", "finish_sync", "opt_step"]

    def test_picks_global_valid_toks_for_token_level_loss(self, mock_module_symbols):
        """N selection: TOKEN_LEVEL → N = global_valid_toks (not seqs)."""
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = self._setup_open_step(mock_module_symbols, LossType.TOKEN_LEVEL)
        w.finish_train_step("s0")
        # local_valid_toks accumulated = 2048; with mocked all_reduce as no-op,
        # global_valid_toks == 2048 → inv_n = 1/2048
        arg = w.model.scale_gradients.call_args.args[0]
        assert arg == pytest.approx(1.0 / 2048.0, rel=1e-4)

    def test_picks_global_valid_seqs_for_sequence_level_loss(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = self._setup_open_step(mock_module_symbols, LossType.SEQUENCE_LEVEL)
        w.finish_train_step("s0")
        # local_valid_seqs = 8 → inv_n = 1/8
        arg = w.model.scale_gradients.call_args.args[0]
        assert arg == pytest.approx(1.0 / 8.0, rel=1e-4)

    def test_restores_grad_sync_func(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = self._setup_open_step(mock_module_symbols, LossType.TOKEN_LEVEL)
        w.finish_train_step("s0")
        assert w.model.config.grad_sync_func == "ORIGINAL_GRAD_SYNC_FUNC"

    def test_clears_train_step_state(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = self._setup_open_step(mock_module_symbols, LossType.TOKEN_LEVEL)
        w.finish_train_step("s0")
        assert w._train_step_state is None

    def test_calls_scheduler_step_with_increment_gbs(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = self._setup_open_step(mock_module_symbols, LossType.TOKEN_LEVEL)
        w._train_step_state["gbs"] = 64
        w.finish_train_step("s0")
        w.scheduler.step.assert_called_once_with(increment=64)

    def test_returns_metrics_dict(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = self._setup_open_step(mock_module_symbols, LossType.TOKEN_LEVEL)
        metrics = w.finish_train_step("s0")
        for key in (
            "global_loss",
            "rank",
            "gpu_name",
            "model_dtype",
            "all_mb_metrics",
            "grad_norm",
        ):
            assert key in metrics, f"missing {key!r}"

    def test_moe_branch_skipped_when_num_experts_is_none(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = self._setup_open_step(mock_module_symbols, LossType.TOKEN_LEVEL)
        w.model.config.num_moe_experts = None
        metrics = w.finish_train_step("s0")
        assert "moe_metrics" not in metrics

    def test_moe_branch_uses_total_num_microbatches_for_scale(
        self, mock_module_symbols
    ):
        """MoE aux-loss scale must use the accumulated total, not the
        per-call num_microbatches."""
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.model.config.num_moe_experts = 4
        # Have get_moe_metrics return non-empty so the branch fires
        mock_module_symbols["moe"].return_value = {"aux_loss": 0.1}
        w.begin_train_step("s0", loss_fn=w._test_loss_fn)
        # 3 train_microbatch calls × 2 pipeline mbs each = 6
        for _ in range(3):
            w.train_microbatch("s0", _fake_batch())
        w.finish_train_step("s0")
        # get_moe_metrics receives loss_scale=1/6
        kwargs = mock_module_symbols["moe"].call_args.kwargs
        assert kwargs["loss_scale"] == pytest.approx(1.0 / 6.0, rel=1e-6)


# ── abort_train_step ─────────────────────────────────────────────────────


class TestAbort:
    def test_restores_grad_sync_func(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.begin_train_step("s0", loss_fn=w._test_loss_fn)
        w.abort_train_step("s0")
        assert w.model.config.grad_sync_func == "ORIGINAL_GRAD_SYNC_FUNC"

    def test_zero_grad_buffer_and_zero_grad_called(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.begin_train_step("s0", loss_fn=w._test_loss_fn)
        w.model.zero_grad_buffer.reset_mock()
        w.optimizer.zero_grad.reset_mock()
        w.abort_train_step("s0")
        w.model.zero_grad_buffer.assert_called_once()
        w.optimizer.zero_grad.assert_called_once()

    def test_does_not_call_optimizer_step(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.begin_train_step("s0", loss_fn=w._test_loss_fn)
        w.train_microbatch("s0", _fake_batch())
        w.abort_train_step("s0")
        w.optimizer.step.assert_not_called()

    def test_clears_train_step_state(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.begin_train_step("s0", loss_fn=w._test_loss_fn)
        w.abort_train_step("s0")
        assert w._train_step_state is None

    def test_idempotent_with_no_open_step(self, mock_module_symbols):
        """abort is a no-op when nothing is open."""
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        # Should not raise
        w.abort_train_step("s0")
        assert getattr(w, "_train_step_state", None) is None

    def test_mismatched_step_id_raises(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.begin_train_step("s0", loss_fn=w._test_loss_fn)
        with pytest.raises(RuntimeError, match="does not match open step"):
            w.abort_train_step("s-WRONG")

    def test_can_begin_new_step_after_abort(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.begin_train_step("s0", loss_fn=w._test_loss_fn)
        w.train_microbatch("s0", _fake_batch())
        w.abort_train_step("s0")
        # New step opens cleanly
        w.begin_train_step("s1", loss_fn=w._test_loss_fn)
        assert w._train_step_state["step_id"] == "s1"
        assert float(w._train_step_state["local_valid_seqs"].item()) == 0.0


# ── grad_sync_func full lifecycle (integration of begin → finish/abort) ─


class TestGradSyncFuncLifecycle:
    def test_begin_finish_round_trip(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        sentinel = "MY_CUSTOM_GRAD_SYNC"
        w.model.config.grad_sync_func = sentinel
        w.begin_train_step("s0", loss_fn=w._test_loss_fn)
        assert w.model.config.grad_sync_func is None
        w.train_microbatch("s0", _fake_batch())
        w.finish_train_step("s0")
        assert w.model.config.grad_sync_func == sentinel

    def test_begin_abort_round_trip(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        sentinel = "MY_CUSTOM_GRAD_SYNC"
        w.model.config.grad_sync_func = sentinel
        w.begin_train_step("s0", loss_fn=w._test_loss_fn)
        assert w.model.config.grad_sync_func is None
        w.abort_train_step("s0")
        assert w.model.config.grad_sync_func == sentinel

    def test_handles_originally_none_grad_sync_func(self, mock_module_symbols):
        """When PP=1 (or align_grad_reduce=False), grad_sync_func is None
        to begin with. begin → finish must leave it as None."""
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.model.config.grad_sync_func = None
        w.begin_train_step("s0", loss_fn=w._test_loss_fn)
        assert w.model.config.grad_sync_func is None
        w.train_microbatch("s0", _fake_batch())
        w.finish_train_step("s0")
        assert w.model.config.grad_sync_func is None
