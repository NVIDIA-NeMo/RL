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
  - ``finalize_model_grads_func`` firing once per streaming chunk instead of
    once per optimizer step. mcore's schedule calls it at the end of every
    ``forward_backward_func``, so under streaming it reduces a partially
    accumulated buffer N times; with a distributed optimizer the reduce-scatter
    also writes into the buffer it reads, so later chunks accumulate on top of
    an already-DP-summed shard.
  - the same hook being dropped rather than relocated, which would silently
    lose the TP layernorm all-reduce, the tied embedding all-reduces across PP,
    the MoE router expert-bias update, and reset_model_temporary_tensors.
  - the relocated finalize being passed a ``num_tokens``, which would rescale
    on top of this path's own 1/N normalization.
  - ``prepare_for_lp_inference`` offloading grad buffers mid-step, which frees
    (not copies) every earlier chunk's gradients while the 1/N normalizer still
    counts them.
  - MTP masks, auxiliary-gradient scaling, grad norms, and tracker metrics being
    present only in the synchronous Megatron path.
"""

from __future__ import annotations

import logging
from typing import Any
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
    # Set explicitly rather than letting MagicMock auto-create it: the finish
    # path asserts this was saved non-None, so the tests that cover the missing
    # hook need to be able to clear it to a real None.
    model.config.finalize_model_grads_func = MagicMock(name="ORIGINAL_FINALIZE")
    model.config.num_moe_experts = None  # disable MoE branch
    model.config.mtp_num_layers = None  # disable MTP unless a test opts in
    model.config.mtp_grad_scale_func = None
    # Explicit values prevent MagicMock's auto-created attributes from making
    # every test look like it uses detached heads with an arbitrary loss weight.
    model.config.mtp_detach_heads = False
    model.config.mtp_loss_scaling_factor = 0.1
    # Prevent MagicMock's dynamic attributes from making _get_model_config
    # mistake this bare model for a Float16Module wrapper.
    model.module = None
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
    w.optimizer.grad_norms_by_group = {}
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
            "use_fused_linear_logprobs": False,
            # overlap_grad_reduce=False matches the production default and
            # the sync-GRPO path. finish_train_step relies on this to gate
            # the explicit start_grad_sync call.
            "distributed_data_parallel_config": {
                "overlap_grad_reduce": False,
            },
        },
    }
    w.dp_size = 2
    w.cp_size = 1
    w.sampling_params = None
    w.draft_model = None
    w.defer_fp32_logits = False
    w.dtype = torch.float32
    w._is_reward_model = False
    w._router_replay_enabled = False
    w.media_placeholder_token_id = None
    # Model-capability flags __init__ derives from self.model, which
    # object.__new__ skips. train_microbatch passes all three straight through
    # to get_microbatch_iterator, so the plain-model defaults (NeMo-RL owns
    # packing and CP sharding) have to be spelled out here.
    w.delegate_pack_to_model = False
    w.delegate_mtp_loss_mask_to_model = False
    w.model_slices_context_parallel_inputs = False
    w._first_train_step_forward_pre_hook_disabled = False
    w._first_train_step_param_sync_func = None
    # Normally set from get_rank_safe() in __init__, which object.__new__ skips.
    # The step summary in finish_train_step reads it eagerly to decide whether
    # this rank prints.
    w.rank = 0
    # Also set in __init__: the finish path reads them to put the DDP forward
    # pre-hook back after the first optimizer step.
    w._log_gpu_mem = MagicMock()

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
        w.begin_train_step(loss_fn=w._test_loss_fn, gbs=16, mbs=4)
        assert w._train_step_state is not None
        assert w._train_step_state["loss_type"] == LossType.TOKEN_LEVEL
        assert w._train_step_state["gbs"] == 16
        assert w._train_step_state["mbs"] == 4
        assert w._train_step_state["total_num_microbatches"] == 0

    def test_calls_zero_grad_and_zero_grad_buffer(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.begin_train_step(loss_fn=w._test_loss_fn)
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
        w.begin_train_step(loss_fn=w._test_loss_fn)
        assert w.model.config.grad_sync_func is None
        assert w._train_step_state["saved_grad_sync_func"] == "ORIGINAL_GRAD_SYNC_FUNC"

    def test_double_begin_raises(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.begin_train_step(loss_fn=w._test_loss_fn)
        with pytest.raises(RuntimeError, match="already open"):
            w.begin_train_step(loss_fn=w._test_loss_fn)

    def test_uses_cfg_defaults_when_gbs_mbs_omitted(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.begin_train_step(loss_fn=w._test_loss_fn)
        assert w._train_step_state["gbs"] == w.cfg["train_global_batch_size"]
        assert w._train_step_state["mbs"] == w.cfg["train_micro_batch_size"]

    def test_records_mtp_enabled_from_model_config(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.model.config.mtp_num_layers = 2
        w.begin_train_step(loss_fn=w._test_loss_fn)
        assert w._train_step_state["mtp_enabled"] is True

    def test_rejects_sequence_level_mtp_with_attached_heads(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.SEQUENCE_LEVEL)
        w.model.config.mtp_num_layers = 2
        with pytest.raises(ValueError, match="mtp_detach_heads"):
            w.begin_train_step(loss_fn=w._test_loss_fn)
        assert getattr(w, "_train_step_state", None) is None

    def test_allows_sequence_level_mtp_with_detached_heads(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.SEQUENCE_LEVEL)
        w.model.config.mtp_num_layers = 2
        w.model.config.mtp_detach_heads = True
        w.begin_train_step(loss_fn=w._test_loss_fn)
        assert w._train_step_state["mtp_detach_heads"] is True

    def test_allows_token_level_mtp_with_attached_heads(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.model.config.mtp_num_layers = 2
        w.begin_train_step(loss_fn=w._test_loss_fn)
        assert w._train_step_state["mtp_detach_heads"] is False

    def test_allows_zero_weight_sequence_level_mtp_with_attached_heads(
        self, mock_module_symbols
    ):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.SEQUENCE_LEVEL)
        w.model.config.mtp_num_layers = 2
        w.model.config.mtp_loss_scaling_factor = 0.0
        w.begin_train_step(loss_fn=w._test_loss_fn)
        assert w._train_step_state["mtp_detach_heads"] is False


# ── _assert_step_open ────────────────────────────────────────────────────


class TestAssertStepOpen:
    def test_raises_when_no_step_open(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        with pytest.raises(RuntimeError, match="no train step open"):
            w._assert_step_open()

    def test_train_microbatch_without_begin_raises(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        with pytest.raises(RuntimeError, match="no train step open"):
            w.train_microbatch(_fake_batch())

    def test_finish_without_begin_raises(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        with pytest.raises(RuntimeError, match="no train step open"):
            w.finish_train_step()


# ── train_microbatch ─────────────────────────────────────────────────────


class TestTrainMicrobatch:
    def test_wraps_forward_backward_in_no_sync(self, mock_module_symbols):
        """The single most important assertion in this file. Without the
        no_sync wrap, mcore DDP dispatches a per-call cross-DP reduce on
        the partially-accumulated buffer — silently corrupting grads."""
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.begin_train_step(loss_fn=w._test_loss_fn)
        w.train_microbatch(_fake_batch())
        # no_sync() must have been ENTERED (called as a context manager).
        # MagicMock with __enter__/__exit__ records the __enter__ call.
        ctx = w.model.no_sync.return_value
        ctx.__enter__.assert_called()
        ctx.__exit__.assert_called()

    def test_invokes_megatron_forward_backward_once(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.begin_train_step(loss_fn=w._test_loss_fn)
        w.train_microbatch(_fake_batch())
        assert mock_module_symbols["mfb"].call_count == 1

    @pytest.mark.parametrize(
        (
            "delegate_pack_to_model",
            "delegate_mtp_loss_mask_to_model",
            "model_slices_context_parallel_inputs",
        ),
        [
            pytest.param(True, True, False, id="model-owned-packing"),
            pytest.param(False, False, True, id="model-owned-cp-slicing"),
        ],
    )
    def test_forwards_model_owned_packing_flags(
        self,
        mock_module_symbols: dict[str, MagicMock],
        delegate_pack_to_model: bool,
        delegate_mtp_loss_mask_to_model: bool,
        model_slices_context_parallel_inputs: bool,
    ) -> None:
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.model.config.mtp_num_layers = 1
        w.delegate_pack_to_model = delegate_pack_to_model
        w.delegate_mtp_loss_mask_to_model = delegate_mtp_loss_mask_to_model
        w.model_slices_context_parallel_inputs = model_slices_context_parallel_inputs

        w.begin_train_step(loss_fn=w._test_loss_fn)
        w.train_microbatch(_fake_batch())

        kwargs = mock_module_symbols["gmi"].call_args.kwargs
        assert kwargs["delegate_pack_to_model"] is delegate_pack_to_model
        assert (
            kwargs["delegate_mtp_loss_mask_to_model"] is delegate_mtp_loss_mask_to_model
        )
        assert (
            kwargs["model_slices_context_parallel_inputs"]
            is model_slices_context_parallel_inputs
        )

    def test_passes_placeholder_n_one_to_loss(self, mock_module_symbols):
        """The N=1 trick: loss must be called with global_valid_*=1 so it
        returns un-normalized sums; finish does the 1/N rescale."""
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.begin_train_step(loss_fn=w._test_loss_fn)
        w.train_microbatch(_fake_batch())
        kwargs = mock_module_symbols["mfb"].call_args.kwargs
        # placeholder_n is a tensor(1.0)
        assert "global_valid_seqs" in kwargs
        assert "global_valid_toks" in kwargs
        assert float(kwargs["global_valid_seqs"].item()) == pytest.approx(1.0)
        assert float(kwargs["global_valid_toks"].item()) == pytest.approx(1.0)

    def test_accumulates_mask_sums_across_calls(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.begin_train_step(loss_fn=w._test_loss_fn)
        # _fake_batch has sample_mask sum = 8, token_mask*sample_mask sum = 8*256 = 2048
        w.train_microbatch(_fake_batch())
        assert float(w._train_step_state["local_valid_seqs"].item()) == pytest.approx(
            8.0
        )
        assert float(w._train_step_state["local_valid_toks"].item()) == pytest.approx(
            2048.0
        )
        w.train_microbatch(_fake_batch())
        assert float(w._train_step_state["local_valid_seqs"].item()) == pytest.approx(
            16.0
        )
        assert float(w._train_step_state["local_valid_toks"].item()) == pytest.approx(
            4096.0
        )

    def test_total_num_microbatches_accumulates(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.begin_train_step(loss_fn=w._test_loss_fn)
        # get_microbatch_iterator mock returns num_microbatches=2 per call
        w.train_microbatch(_fake_batch())
        w.train_microbatch(_fake_batch())
        w.train_microbatch(_fake_batch())
        assert w._train_step_state["total_num_microbatches"] == 6

    def test_does_not_call_optimizer_step(self, mock_module_symbols):
        """trainer_version semantics: optimizer.step() must NOT fire
        per train_microbatch — only at finish."""
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.begin_train_step(loss_fn=w._test_loss_fn)
        w.train_microbatch(_fake_batch())
        w.train_microbatch(_fake_batch())
        w.optimizer.step.assert_not_called()

    def test_builds_mtp_mask_and_uses_main_loss_scale_fallback(
        self, mock_module_symbols
    ):
        """The mask combines both inputs and MTP inherits the main loss scale."""
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.model.config.mtp_num_layers = 2
        batch = _fake_batch()
        batch["token_mask"][0, 5] = 0
        batch["sample_mask"][3] = 0
        w.begin_train_step(loss_fn=w._test_loss_fn)
        w.train_microbatch(batch)

        mask = batch["mtp_loss_mask"]
        assert mask.shape == (8, 257)
        assert mask[3].sum().item() == pytest.approx(0.0)
        assert mask[0].sum().item() == pytest.approx(256.0)
        assert mask[1].sum().item() == pytest.approx(257.0)
        assert w.model.config.mtp_grad_scale_func is None

    def test_skips_mtp_mask_and_scale_when_disabled(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        batch = _fake_batch()
        w.begin_train_step(loss_fn=w._test_loss_fn)
        w.train_microbatch(batch)
        assert "mtp_loss_mask" not in batch
        assert w.model.config.mtp_grad_scale_func is None


# ── finish_train_step ────────────────────────────────────────────────────


class TestFinish:
    def _setup_open_step(self, mock_module_symbols, loss_type):
        w = _make_worker(loss_type)
        w.begin_train_step(loss_fn=w._test_loss_fn)
        w.train_microbatch(_fake_batch())
        return w

    def test_rescales_grads_with_inv_n(self, mock_module_symbols):
        """The 1/N rescale must happen ON the local main_grad BEFORE the
        cross-DP reduce — otherwise the reduce sees un-rescaled sums."""
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = self._setup_open_step(mock_module_symbols, LossType.TOKEN_LEVEL)
        w.finish_train_step()
        # scale_gradients should have been called with some 1/N scalar < 1
        w.model.scale_gradients.assert_called_once()
        arg = w.model.scale_gradients.call_args.args[0]
        assert 0 < arg <= 1.0

    def test_restores_first_step_param_sync_after_success(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = self._setup_open_step(mock_module_symbols, LossType.TOKEN_LEVEL)
        saved_param_sync = MagicMock(name="saved_param_sync")
        w._first_train_step_forward_pre_hook_disabled = True
        w._first_train_step_param_sync_func = saved_param_sync
        w.model.config.param_sync_func = None
        w.enable_forward_pre_hook = MagicMock()

        w.finish_train_step()

        w.enable_forward_pre_hook.assert_called_once_with()
        assert w.model.config.param_sync_func is saved_param_sync
        assert w._first_train_step_param_sync_func is None
        assert w._first_train_step_forward_pre_hook_disabled is False

    def test_keeps_first_step_param_sync_disabled_after_failed_update(
        self, mock_module_symbols
    ):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = self._setup_open_step(mock_module_symbols, LossType.TOKEN_LEVEL)
        saved_param_sync = MagicMock(name="saved_param_sync")
        w._first_train_step_forward_pre_hook_disabled = True
        w._first_train_step_param_sync_func = saved_param_sync
        w.model.config.param_sync_func = None
        w.enable_forward_pre_hook = MagicMock()
        w.optimizer.step.return_value = (False, 0.5, 0)

        w.finish_train_step()

        w.enable_forward_pre_hook.assert_not_called()
        assert w.model.config.param_sync_func is None
        assert w._first_train_step_param_sync_func is saved_param_sync
        assert w._first_train_step_forward_pre_hook_disabled is True

    @pytest.mark.parametrize("overlap_grad_reduce", [False, True])
    def test_grad_sync_call_order_after_rescale(
        self, mock_module_symbols, overlap_grad_reduce
    ):
        """Call order matters: scale_gradients -> [start_grad_sync when
        overlap=True] -> finalize_model_grads_func -> optimizer.step.

        The relocated finalize owns the cross-DP reduce (it reaches
        finish_grad_sync itself), so this path must NOT also call
        finish_grad_sync — that would double-reduce. On the overlap path
        ``model.no_sync()`` suppressed register_grad_ready's dispatch, so there
        is no outstanding handle for the finalize to wait on and start_grad_sync
        has to fire first.
        """
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = self._setup_open_step(mock_module_symbols, LossType.TOKEN_LEVEL)
        w.cfg["megatron_cfg"]["distributed_data_parallel_config"][
            "overlap_grad_reduce"
        ] = overlap_grad_reduce
        # Record call order via a shared list
        order: list[str] = []
        w.model.scale_gradients.side_effect = lambda s: order.append("scale")
        w.model.start_grad_sync.side_effect = lambda: order.append("start_sync")
        w.model.finish_grad_sync.side_effect = lambda: order.append("finish_sync")
        w._train_step_state["saved_finalize_model_grads_func"] = (
            lambda models, num_tokens: order.append("finalize")
        )
        w.optimizer.step.side_effect = lambda: (
            order.append("opt_step") or (True, 0.5, 0)
        )
        w.finish_train_step()
        if overlap_grad_reduce:
            assert order == ["scale", "start_sync", "finalize", "opt_step"]
        else:
            assert order == ["scale", "finalize", "opt_step"]

    @pytest.mark.parametrize("overlap_grad_reduce", [False, True])
    def test_grad_sync_refuses_to_reduce_without_the_finalize_hook(
        self, mock_module_symbols, overlap_grad_reduce
    ):
        """A missing finalize hook fails the step instead of reducing without it.

        A bare finish_grad_sync would cover the cross-DP reduce and silently
        skip everything else the finalize owns -- the TP layernorm all-reduce,
        the tied-embedding all-reduces across PP, the MoE expert-bias update --
        so falling back would train on wrong gradients with no error. Nothing is
        dispatched before the check, and the optimizer does not step.
        """
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = self._setup_open_step(mock_module_symbols, LossType.TOKEN_LEVEL)
        w.cfg["megatron_cfg"]["distributed_data_parallel_config"][
            "overlap_grad_reduce"
        ] = overlap_grad_reduce
        w._train_step_state["saved_finalize_model_grads_func"] = None
        with pytest.raises(AssertionError, match="finalize_model_grads_func was None"):
            w.finish_train_step()
        w.model.start_grad_sync.assert_not_called()
        w.model.finish_grad_sync.assert_not_called()
        w.optimizer.step.assert_not_called()

    def test_picks_global_valid_toks_for_token_level_loss(self, mock_module_symbols):
        """N selection: TOKEN_LEVEL → N = global_valid_toks (not seqs)."""
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = self._setup_open_step(mock_module_symbols, LossType.TOKEN_LEVEL)
        w.finish_train_step()
        # local_valid_toks accumulated = 2048; with mocked all_reduce as no-op,
        # global_valid_toks == 2048 → inv_n = 1/2048
        arg = w.model.scale_gradients.call_args.args[0]
        assert arg == pytest.approx(1.0 / 2048.0, rel=1e-4)

    def test_picks_global_valid_seqs_for_sequence_level_loss(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = self._setup_open_step(mock_module_symbols, LossType.SEQUENCE_LEVEL)
        w.finish_train_step()
        # local_valid_seqs = 8 → inv_n = 1/8
        arg = w.model.scale_gradients.call_args.args[0]
        assert arg == pytest.approx(1.0 / 8.0, rel=1e-4)

    @staticmethod
    def _mtp_params() -> tuple[MagicMock, MagicMock]:
        """Create one MTP-tagged parameter and one untagged parameter."""
        mtp_param = MagicMock()
        mtp_param.grad_norm_group = "mtp"
        mtp_param.main_grad = torch.ones(4)
        other_param = MagicMock()
        other_param.grad_norm_group = None
        other_param.main_grad = torch.ones(4)
        return mtp_param, other_param

    def test_mtp_grads_use_token_denominator_under_sequence_level_loss(
        self, mock_module_symbols
    ):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.SEQUENCE_LEVEL)
        w.model.config.mtp_num_layers = 2
        w.model.config.mtp_detach_heads = True
        mtp_param, other_param = self._mtp_params()
        w.model.parameters = MagicMock(return_value=[mtp_param, other_param])
        w.begin_train_step(loss_fn=w._test_loss_fn)
        w.train_microbatch(_fake_batch())
        grad_seen_by_finalize: list[torch.Tensor] = []
        w._train_step_state["saved_finalize_model_grads_func"] = (
            lambda models, num_tokens: grad_seen_by_finalize.append(
                mtp_param.main_grad.clone()
            )
        )
        w.finish_train_step()

        # The main loss gets 1/8; MTP additionally gets 8/2048, for net 1/2048.
        assert w.model.scale_gradients.call_args.args[0] == pytest.approx(1.0 / 8.0)
        expected_mtp_grad = torch.full((4,), 8.0 / 2048.0)
        torch.testing.assert_close(mtp_param.main_grad, expected_mtp_grad)
        torch.testing.assert_close(grad_seen_by_finalize[0], expected_mtp_grad)
        torch.testing.assert_close(other_param.main_grad, torch.ones(4))

    def test_mtp_grads_need_no_correction_under_token_level_loss(
        self, mock_module_symbols
    ):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.model.config.mtp_num_layers = 2
        w.model.config.mtp_detach_heads = True
        mtp_param, _ = self._mtp_params()
        w.model.parameters = MagicMock(return_value=[mtp_param])
        w.begin_train_step(loss_fn=w._test_loss_fn)
        w.train_microbatch(_fake_batch())
        w.finish_train_step()

        torch.testing.assert_close(mtp_param.main_grad, torch.ones(4))

    def test_restores_grad_sync_func(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = self._setup_open_step(mock_module_symbols, LossType.TOKEN_LEVEL)
        w.finish_train_step()
        assert w.model.config.grad_sync_func == "ORIGINAL_GRAD_SYNC_FUNC"

    @pytest.mark.parametrize("update_successful", [True, False])
    def test_reenables_forward_pre_hook_after_a_successful_step(
        self, mock_module_symbols, update_successful
    ):
        """__init__ disables the pre-hook for the first step; finish has to put
        it back once the optimizer stepped, or every later forward sees only its
        own param shard."""
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = self._setup_open_step(mock_module_symbols, LossType.TOKEN_LEVEL)
        w._first_train_step_forward_pre_hook_disabled = True
        w._first_train_step_param_sync_func = "ORIGINAL_PARAM_SYNC_FUNC"
        w.enable_forward_pre_hook = MagicMock()
        w.optimizer.step.return_value = (update_successful, 0.5, 0)

        with patch(f"{WORKER_MOD}.get_model_config", return_value=w.model.config):
            w.finish_train_step()

        if update_successful:
            w.enable_forward_pre_hook.assert_called_once_with()
            assert w.model.config.param_sync_func == "ORIGINAL_PARAM_SYNC_FUNC"
            assert w._first_train_step_forward_pre_hook_disabled is False
            assert w._first_train_step_param_sync_func is None
        else:
            w.enable_forward_pre_hook.assert_not_called()
            assert w._first_train_step_forward_pre_hook_disabled is True

    def test_clears_train_step_state(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = self._setup_open_step(mock_module_symbols, LossType.TOKEN_LEVEL)
        w.finish_train_step()
        assert w._train_step_state is None

    def test_calls_scheduler_step_with_increment_gbs(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = self._setup_open_step(mock_module_symbols, LossType.TOKEN_LEVEL)
        w._train_step_state["gbs"] = 64
        w.finish_train_step()
        w.scheduler.step.assert_called_once_with(increment=64)

    def test_returns_metrics_dict(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = self._setup_open_step(mock_module_symbols, LossType.TOKEN_LEVEL)
        metrics = w.finish_train_step()
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
        metrics = w.finish_train_step()
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
        w.begin_train_step(loss_fn=w._test_loss_fn)
        # 3 train_microbatch calls × 2 pipeline mbs each = 6
        for _ in range(3):
            w.train_microbatch(_fake_batch())
        w.finish_train_step()
        # get_moe_metrics receives loss_scale=1/6
        kwargs = mock_module_symbols["moe"].call_args.kwargs
        assert kwargs["loss_scale"] == pytest.approx(1.0 / 6.0, rel=1e-6)

    def test_collects_mtp_metrics_and_reduced_grad_norm(self, mock_module_symbols):
        """Finish surfaces the same post-#3194 MTP payload as sync train()."""
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.model.config.mtp_num_layers = 2
        w.optimizer.grad_norms_by_group = {"mtp": 1.25}

        def _collect(
            metrics: dict[str, Any],
            total_num_microbatches: int,
            mtp_grad_norm: float | None,
        ) -> None:
            assert total_num_microbatches == 2
            metrics["mtp_metrics"] = {
                "mtp_1_loss": 0.5,
                "grad_norm": mtp_grad_norm,
            }

        w._collect_mtp_metrics = MagicMock(side_effect=_collect)
        w.begin_train_step(loss_fn=w._test_loss_fn)
        w.train_microbatch(_fake_batch())
        metrics = w.finish_train_step()

        args = w._collect_mtp_metrics.call_args.args
        assert args[1] == 2  # fixture: two pipeline microbatches per chunk
        assert args[2] == pytest.approx(1.25)
        assert metrics["mtp_metrics"]["mtp_1_loss"] == pytest.approx(0.5)
        assert metrics["mtp_metrics"]["grad_norm"] == pytest.approx(1.25)
        assert w.model.config.mtp_grad_scale_func is None

    def test_loss_advertised_normalizers_applied(self, mock_module_symbols):
        """finish scales each metric by the denominator the loss advertised:
        TOKENS → 1/global_valid_toks, SEQUENCES → 1/global_valid_seqs,
        NONE → unscaled, unadvertised → gradient normalization (inv_n)."""
        from nemo_rl.algorithms.loss.interfaces import LossType, MetricNormalizer

        w = _make_worker(LossType.TOKEN_LEVEL)
        w._test_loss_fn.metric_normalizations = {
            "tok_metric": MetricNormalizer.TOKENS,
            "seq_metric": MetricNormalizer.SEQUENCES,
            "raw_metric": MetricNormalizer.NONE,
        }
        mock_module_symbols["mfb"].return_value = [
            {
                "tok_metric": 2048.0,
                "seq_metric": 8.0,
                "raw_metric": 8.0,
                "other_metric": 2048.0,
            }
        ]
        w.begin_train_step(loss_fn=w._test_loss_fn)
        w.train_microbatch(_fake_batch())  # 8 seqs / 2048 valid toks
        w.finish_train_step()
        m = mock_module_symbols["agg"].call_args.kwargs["all_mb_metrics"][0]
        assert m["tok_metric"] == pytest.approx(1.0)  # 2048 / 2048
        assert m["seq_metric"] == pytest.approx(1.0)  # 8 / 8
        assert m["raw_metric"] == pytest.approx(8.0)  # unscaled
        # unadvertised → inv_n of the loss_type (TOKEN_LEVEL → 1/2048)
        assert m["other_metric"] == pytest.approx(1.0)

    def test_raw_count_metrics_not_rescaled_by_inv_n(self, mock_module_symbols):
        """Raw-count metrics (num_valid_samples, num_unmasked_tokens) are
        absolute counts the loss advertises as NONE; finish must leave them
        unscaled so the downstream sum recovers the true global count
        (PR #2683 review, F-COUNT)."""
        from nemo_rl.algorithms.loss.interfaces import LossType, MetricNormalizer

        w = _make_worker(LossType.TOKEN_LEVEL)
        w._test_loss_fn.metric_normalizations = {
            "num_valid_samples": MetricNormalizer.NONE,
            "num_unmasked_tokens": MetricNormalizer.NONE,
        }
        mock_module_symbols["mfb"].return_value = [
            {"loss": 0.5, "num_valid_samples": 8.0, "num_unmasked_tokens": 2048.0}
        ]
        w.begin_train_step(loss_fn=w._test_loss_fn)
        w.train_microbatch(_fake_batch())  # inv_n = 1/2048
        w.finish_train_step()
        m = mock_module_symbols["agg"].call_args.kwargs["all_mb_metrics"][0]
        assert m["num_valid_samples"] == pytest.approx(8.0)
        assert m["num_unmasked_tokens"] == pytest.approx(2048.0)

    def test_flag_keyed_normalizers_from_real_loss(self, mock_module_symbols):
        """seq-mask-tis + token-level loss: is_oob_ratio was reduced by
        global_valid_seqs even though the gradient normalizer is tokens.
        The advertised mapping from a real ClippedPGLossFn must key it on
        the TIS type, not loss_type (PR #2683 review, F-SEQ)."""
        from nemo_rl.algorithms.loss.interfaces import LossType
        from nemo_rl.algorithms.loss.loss_functions import (
            ClippedPGLossConfig,
            ClippedPGLossFn,
        )

        real_loss = ClippedPGLossFn(
            ClippedPGLossConfig(
                token_level_loss=True,
                use_importance_sampling_correction=True,
                truncated_importance_sampling_type="seq-mask-tis",
                truncated_importance_sampling_ratio=2.0,
                truncated_importance_sampling_ratio_min=0.5,
            )
        )
        w = _make_worker(LossType.TOKEN_LEVEL)
        w._test_loss_fn.metric_normalizations = real_loss.metric_normalizations
        mock_module_symbols["mfb"].return_value = [
            {
                "loss": 2048.0,
                "is_oob_ratio": 8.0,
                "sampling_importance_ratio": 2048.0,
            }
        ]
        w.begin_train_step(loss_fn=w._test_loss_fn)
        w.train_microbatch(_fake_batch())  # 8 seqs / 2048 valid toks
        w.finish_train_step()
        m = mock_module_symbols["agg"].call_args.kwargs["all_mb_metrics"][0]
        assert m["loss"] == pytest.approx(1.0)  # ÷ toks (loss_type)
        assert m["is_oob_ratio"] == pytest.approx(1.0)  # ÷ seqs, NOT toks
        assert m["sampling_importance_ratio"] == pytest.approx(1.0)  # ÷ toks


# ── abort_train_step ─────────────────────────────────────────────────────


class TestAbort:
    def test_restores_grad_sync_func(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.begin_train_step(loss_fn=w._test_loss_fn)
        w.abort_train_step()
        assert w.model.config.grad_sync_func == "ORIGINAL_GRAD_SYNC_FUNC"

    def test_zero_grad_buffer_and_zero_grad_called(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.begin_train_step(loss_fn=w._test_loss_fn)
        w.model.zero_grad_buffer.reset_mock()
        w.optimizer.zero_grad.reset_mock()
        w.abort_train_step()
        w.model.zero_grad_buffer.assert_called_once()
        w.optimizer.zero_grad.assert_called_once()

    def test_does_not_call_optimizer_step(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.begin_train_step(loss_fn=w._test_loss_fn)
        w.train_microbatch(_fake_batch())
        w.abort_train_step()
        w.optimizer.step.assert_not_called()

    def test_clears_train_step_state(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.begin_train_step(loss_fn=w._test_loss_fn)
        w.abort_train_step()
        assert w._train_step_state is None

    def test_idempotent_with_no_open_step(self, mock_module_symbols):
        """abort is a no-op when nothing is open."""
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        # Should not raise
        w.abort_train_step()
        assert getattr(w, "_train_step_state", None) is None

    def test_can_begin_new_step_after_abort(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.begin_train_step(loss_fn=w._test_loss_fn)
        w.train_microbatch(_fake_batch())
        w.abort_train_step()
        # New step opens cleanly
        w.begin_train_step(loss_fn=w._test_loss_fn)
        assert w._train_step_state is not None
        assert float(w._train_step_state["local_valid_seqs"].item()) == 0.0

    def test_clears_stale_mtp_gradient_scale_at_step_boundaries(
        self, mock_module_symbols
    ):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.model.config.mtp_num_layers = 1
        w.model.config.mtp_grad_scale_func = lambda: torch.tensor(7.0)
        w.begin_train_step(loss_fn=w._test_loss_fn)
        assert w.model.config.mtp_grad_scale_func is None
        w.train_microbatch(_fake_batch())
        assert w.model.config.mtp_grad_scale_func is None
        w.abort_train_step()
        assert w.model.config.mtp_grad_scale_func is None


# ── grad_sync_func full lifecycle (integration of begin → finish/abort) ─


class TestGradSyncFuncLifecycle:
    def test_begin_finish_round_trip(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        sentinel = "MY_CUSTOM_GRAD_SYNC"
        w.model.config.grad_sync_func = sentinel
        w.begin_train_step(loss_fn=w._test_loss_fn)
        assert w.model.config.grad_sync_func is None
        w.train_microbatch(_fake_batch())
        w.finish_train_step()
        assert w.model.config.grad_sync_func == sentinel

    def test_begin_abort_round_trip(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        sentinel = "MY_CUSTOM_GRAD_SYNC"
        w.model.config.grad_sync_func = sentinel
        w.begin_train_step(loss_fn=w._test_loss_fn)
        assert w.model.config.grad_sync_func is None
        w.abort_train_step()
        assert w.model.config.grad_sync_func == sentinel

    def test_handles_originally_none_grad_sync_func(self, mock_module_symbols):
        """When PP=1 (or align_grad_reduce=False), grad_sync_func is None
        to begin with. begin → finish must leave it as None."""
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.model.config.grad_sync_func = None
        w.begin_train_step(loss_fn=w._test_loss_fn)
        assert w.model.config.grad_sync_func is None
        w.train_microbatch(_fake_batch())
        w.finish_train_step()
        assert w.model.config.grad_sync_func is None


# ── finalize_model_grads_func: once per STEP, not per chunk ──────────────


class TestFinalizeModelGradsOncePerStep:
    """The streaming train path feeds N chunks into one optimizer step.

    mcore calls ``finalize_model_grads_func`` at the end of every
    ``forward_backward_func``, so left alone it fires N times per step. These
    tests pin the relocation: nulled for the step's duration, invoked exactly
    once at finish, restored afterwards.
    """

    def test_begin_saves_and_nulls_the_hook(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        sentinel = w.model.config.finalize_model_grads_func
        w.begin_train_step(loss_fn=w._test_loss_fn)
        assert w.model.config.finalize_model_grads_func is None
        assert w._train_step_state["saved_finalize_model_grads_func"] is sentinel

    def test_hook_stays_nulled_across_every_chunk(self, mock_module_symbols):
        """The whole point: while chunks are being dispatched, mcore's schedule
        must find no hook to call."""
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        sentinel = w.model.config.finalize_model_grads_func
        w.begin_train_step(loss_fn=w._test_loss_fn)
        for _ in range(4):
            w.train_microbatch(_fake_batch())
            assert w.model.config.finalize_model_grads_func is None
        sentinel.assert_not_called()

    @pytest.mark.parametrize("num_chunks", [1, 2, 5])
    def test_called_exactly_once_regardless_of_chunk_count(
        self, mock_module_symbols, num_chunks
    ):
        """The regression this branch exists to prevent: one reduce per step,
        whether the step arrived as 1 chunk or 5."""
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        sentinel = w.model.config.finalize_model_grads_func
        w.begin_train_step(loss_fn=w._test_loss_fn)
        for _ in range(num_chunks):
            w.train_microbatch(_fake_batch())
        w.finish_train_step()
        assert sentinel.call_count == 1

    def test_called_with_model_list_and_no_num_tokens(self, mock_module_symbols):
        """``num_tokens=None`` is load-bearing: this path already applied 1/N
        from its own accumulated valid-token count, and mcore rescales only when
        num_tokens is not None. Passing a count would double-normalize."""
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        sentinel = w.model.config.finalize_model_grads_func
        w.begin_train_step(loss_fn=w._test_loss_fn)
        w.train_microbatch(_fake_batch())
        w.finish_train_step()
        args, kwargs = sentinel.call_args
        assert args[0] == [w.model]
        assert args[1] is None
        # No keywords at all, which is what keeps Megatron-Bridge's
        # ``partial(finalize_model_grads, pg_collection=...)`` binding intact —
        # an explicit keyword here would override it. See the note at the call
        # site for why that binding is group-equivalent to the collection the
        # schedule used to pass on the per-chunk path.
        assert kwargs == {}

    def test_does_not_also_call_finish_grad_sync(self, mock_module_symbols):
        """finalize reaches finish_grad_sync internally, so calling both
        double-reduces (scales grads by world_size)."""
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.begin_train_step(loss_fn=w._test_loss_fn)
        w.train_microbatch(_fake_batch())
        w.finish_train_step()
        w.model.finish_grad_sync.assert_not_called()

    def test_runs_after_rescale_and_before_optimizer_step(self, mock_module_symbols):
        """The 1/N rescale must land on the local buffer before the reduce, and
        the reduce before opt.step reads main_grad."""
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        order: list[str] = []
        w.model.scale_gradients.side_effect = lambda s: order.append("scale")
        w.model.config.finalize_model_grads_func = MagicMock(
            side_effect=lambda models, num_tokens: order.append("finalize")
        )
        w.optimizer.step.side_effect = lambda: (
            order.append("opt_step") or (True, 0.5, 0)
        )
        w.begin_train_step(loss_fn=w._test_loss_fn)
        w.train_microbatch(_fake_batch())
        w.finish_train_step()
        assert order == ["scale", "finalize", "opt_step"]

    def test_finish_restores_the_hook(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        sentinel = w.model.config.finalize_model_grads_func
        w.begin_train_step(loss_fn=w._test_loss_fn)
        w.train_microbatch(_fake_batch())
        w.finish_train_step()
        assert w.model.config.finalize_model_grads_func is sentinel

    def test_abort_restores_the_hook(self, mock_module_symbols):
        """Otherwise a subsequent sync train() would run with no finalize at
        all — no cross-DP reduce, no embedding all-reduce."""
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        sentinel = w.model.config.finalize_model_grads_func
        w.begin_train_step(loss_fn=w._test_loss_fn)
        w.train_microbatch(_fake_batch())
        w.abort_train_step()
        assert w.model.config.finalize_model_grads_func is sentinel

    def test_abort_does_not_call_the_hook(self, mock_module_symbols):
        """An aborted step throws its gradients away; reducing them would leak
        a partial step into the next one."""
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        sentinel = w.model.config.finalize_model_grads_func
        w.begin_train_step(loss_fn=w._test_loss_fn)
        w.train_microbatch(_fake_batch())
        w.abort_train_step()
        sentinel.assert_not_called()

    def test_originally_none_hook_fails_the_step(self, mock_module_symbols):
        """Megatron-Bridge binds the hook whenever an optimizer exists, and an
        open step implies one, so a None here means the model config never went
        through setup. The step fails rather than reducing with finish_grad_sync
        alone, and the restore path still runs on the way out."""
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.model.config.finalize_model_grads_func = None
        w.begin_train_step(loss_fn=w._test_loss_fn)
        w.train_microbatch(_fake_batch())
        with pytest.raises(AssertionError, match="finalize_model_grads_func was None"):
            w.finish_train_step()
        w.model.finish_grad_sync.assert_not_called()
        assert w.model.config.finalize_model_grads_func is None
        assert w.model.config.grad_sync_func == "ORIGINAL_GRAD_SYNC_FUNC"

    def test_two_consecutive_steps_each_finalize_once(self, mock_module_symbols):
        """Restore-then-null across step boundaries must not lose the hook or
        double up on it."""
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        sentinel = w.model.config.finalize_model_grads_func
        for _ in range(2):
            w.begin_train_step(loss_fn=w._test_loss_fn)
            w.train_microbatch(_fake_batch())
            w.train_microbatch(_fake_batch())
            w.finish_train_step()
        assert sentinel.call_count == 2
        assert w.model.config.finalize_model_grads_func is sentinel


# ── num_chunks ───────────────────────────────────────────────────────────


class TestNumChunks:
    """``num_chunks`` records how many chunks accumulated into the step, which
    is what makes "did the grads get reduced once or N times" greppable from a
    run's logs rather than inferable from timer ratios."""

    def test_starts_at_zero(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.begin_train_step(loss_fn=w._test_loss_fn)
        assert w._train_step_state["num_chunks"] == 0

    def test_increments_once_per_train_microbatch(self, mock_module_symbols):
        """Once per chunk, NOT once per pipeline microbatch: the mocked
        iterator yields 2 pipeline mbs per call, so 3 calls is 3 chunks and 6
        microbatches."""
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.begin_train_step(loss_fn=w._test_loss_fn)
        for expected in (1, 2, 3):
            w.train_microbatch(_fake_batch())
            assert w._train_step_state["num_chunks"] == expected
        assert w._train_step_state["total_num_microbatches"] == 6

    def test_resets_on_next_step(self, mock_module_symbols):
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.begin_train_step(loss_fn=w._test_loss_fn)
        w.train_microbatch(_fake_batch())
        w.finish_train_step()
        w.begin_train_step(loss_fn=w._test_loss_fn)
        assert w._train_step_state["num_chunks"] == 0


class TestStepSummaryIsEmitted:
    """The per-step summary has to survive at the level a default run uses.

    ``nemo_rl/__init__.py`` calls ``basicConfig()`` with no level, which pins the
    root logger at WARNING, and every later ``basicConfig`` returns early because
    handlers already exist. The first version of this telemetry was therefore
    dropped outright -- a 68-node run produced zero matches. It emits now because
    that module also sets the ``nemo_rl`` logger from ``NRL_LOG_LEVEL``,
    defaulting to INFO, so this pins the record at INFO and not below.
    """

    LOGGER = "nemo_rl.models.policy.workers.megatron_policy_worker"

    def test_logs_chunk_count_and_grad_norm(self, mock_module_symbols, caplog):
        from nemo_rl.algorithms.loss.interfaces import LossType

        caplog.set_level(logging.INFO, logger=self.LOGGER)
        w = _make_worker(LossType.TOKEN_LEVEL)
        w.begin_train_step(loss_fn=w._test_loss_fn)
        w.train_microbatch(_fake_batch())
        w.train_microbatch(_fake_batch())
        w.finish_train_step()

        assert "[step] chunks=2" in caplog.text
        # Paired with the chunk count on one line: a grad_norm that moves with
        # the chunk count is the bug this PR fixes.
        assert "grad_norm=0.5" in caplog.text

    def test_only_rank_zero_logs(self, mock_module_symbols, caplog):
        """Every field is DP- or MP-reduced by this point, so off-rank lines
        would be duplicates -- and each costs two host-device syncs."""
        from nemo_rl.algorithms.loss.interfaces import LossType

        caplog.set_level(logging.INFO, logger=self.LOGGER)
        w = _make_worker(LossType.TOKEN_LEVEL)
        w.rank = 3
        w.begin_train_step(loss_fn=w._test_loss_fn)
        w.train_microbatch(_fake_batch())
        w.finish_train_step()

        assert "[step]" not in caplog.text


class TestChunkRecordIsGuarded:
    """The per-chunk record is debug-only, and must be both free and reachable.

    Free: its arguments are evaluated at the call site whether or not the record
    is emitted, and four of them are ``.item()`` host-device syncs, so the
    ``isEnabledFor`` guard is what keeps a disabled record off the critical path.
    Reachable: that guard is only worth having if the level can actually be
    raised, which is what ``NRL_LOG_LEVEL`` in ``nemo_rl/__init__.py`` provides.
    """

    LOGGER = "nemo_rl.models.policy.workers.megatron_policy_worker"

    def test_silent_at_the_default_level(self, mock_module_symbols, caplog):
        """INFO is what a default run gets, and this record sits below it."""
        from nemo_rl.algorithms.loss.interfaces import LossType

        caplog.set_level(logging.INFO, logger=self.LOGGER)
        w = _make_worker(LossType.TOKEN_LEVEL)
        w.begin_train_step(loss_fn=w._test_loss_fn)
        w.train_microbatch(_fake_batch())

        assert "[chunk]" not in caplog.text

    def test_emitted_when_debug_is_enabled(self, mock_module_symbols, caplog):
        from nemo_rl.algorithms.loss.interfaces import LossType

        caplog.set_level(logging.DEBUG, logger=self.LOGGER)
        w = _make_worker(LossType.TOKEN_LEVEL)
        w.begin_train_step(loss_fn=w._test_loss_fn)
        w.train_microbatch(_fake_batch())

        assert "[chunk] chunks=1" in caplog.text


# ── prepare_for_lp_inference ─────────────────────────────────────────────


class TestPrepareForLpInference:
    """``keep_train_buffers`` decides whether an open step's accumulated
    gradients survive the logprob phase.

    mcore's ``offload_to_cpu(move_grads=True)`` does not copy the gradients
    anywhere — it resizes their storage to 0 — and nothing raises afterwards
    because ``param.main_grad`` stays a valid view. So the failure mode this
    guards is silent: every chunk but the last is discarded while the 1/N
    normalizer still counts all of them.
    """

    @staticmethod
    def _worker():
        from nemo_rl.algorithms.loss.interfaces import LossType

        w = _make_worker(LossType.TOKEN_LEVEL)
        w.move_model = MagicMock(side_effect=lambda model, *a, **k: model)
        w.move_optimizer = MagicMock()
        w.optimizer_cpu_offload = False
        w.offload_optimizer_for_logprob = True
        return w

    @staticmethod
    def _grad_offload_calls(w) -> list:
        """The move_model calls that free grad buffers, i.e. to cpu with
        move_grads=True."""
        return [
            c
            for c in w.move_model.call_args_list
            if c.args[1:2] == ("cpu",) and c.kwargs.get("move_grads")
        ]

    def test_keeps_buffers_when_step_is_open(self, mock_module_symbols):
        w = self._worker()
        with patch("torch.randn"):
            w.prepare_for_lp_inference(keep_train_buffers=True)
        assert self._grad_offload_calls(w) == []
        w.move_optimizer.assert_not_called()

    def test_offloads_buffers_when_no_step_is_open(self, mock_module_symbols):
        """The non-streaming path still has to reclaim the buffers, otherwise
        the logprob phase peaks tens of GiB higher than it needs to."""
        w = self._worker()
        with patch("torch.randn"):
            w.prepare_for_lp_inference(keep_train_buffers=False)
        assert len(self._grad_offload_calls(w)) == 1
        w.move_optimizer.assert_called_once_with("cpu")

    def test_defaults_to_offloading(self, mock_module_symbols):
        """Callers that predate the flag keep their old behaviour."""
        w = self._worker()
        with patch("torch.randn"):
            w.prepare_for_lp_inference()
        assert len(self._grad_offload_calls(w)) == 1
        w.move_optimizer.assert_called_once_with("cpu")

    @pytest.mark.parametrize("keep_train_buffers", [False, True])
    def test_always_onloads_params_and_sets_eval(
        self, mock_module_symbols, keep_train_buffers
    ):
        """Params go to CUDA and the model goes to eval either way — only the
        grad/optimizer offload is conditional."""
        w = self._worker()
        with patch("torch.randn"):
            w.prepare_for_lp_inference(keep_train_buffers=keep_train_buffers)
        first = w.move_model.call_args_list[0]
        assert first.args[1] == "cuda"
        assert first.kwargs == {"move_grads": False}
        w.model.eval.assert_called_once()

    def test_keeps_buffers_across_an_open_step(self, mock_module_symbols):
        """The sequence the streaming pump actually produces: open a step, run a
        chunk, take the logprob detour, run another chunk, finish. The finalize
        must still fire exactly once and the grads must never be offloaded."""
        w = self._worker()
        sentinel = w.model.config.finalize_model_grads_func
        w.begin_train_step(loss_fn=w._test_loss_fn)
        w.train_microbatch(_fake_batch())
        with patch("torch.randn"):
            w.prepare_for_lp_inference(keep_train_buffers=True)
        w.train_microbatch(_fake_batch())
        w.finish_train_step()
        assert self._grad_offload_calls(w) == []
        assert sentinel.call_count == 1
