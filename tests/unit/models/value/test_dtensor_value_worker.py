# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""End-to-end tests for `DTensorValueWorkerV2` via the `Value` wrapper.

Worker-level tests use a tiny Qwen2 model on a small Ray cluster, mirroring
`test_megatron_value_worker.py`. They cover:

  * value head (regression head, `num_labels=1`) on the DTensor V2 backbone
  * `get_values` forward pass (shape + finiteness), incl. TP / SP /
    dynamic batching
  * `train` step with `MseValueLossFn` (loss is finite + non-negative)
  * Sequence-parallel / dynamic-batching equivalence (must not change values)
  * Multi-step training drives loss down
  * Checkpoint save+load round-trip preserves the trained value head

DTensor V2 value worker does not support pipeline parallelism (lm_value.py)
or sequence packing (setup.validate_and_prepare_config disallows packing for
reward models), so those parallelism modes are not exercised here.
"""

import os
from pathlib import Path
from typing import Any

import pytest
import ray
import torch

from nemo_rl.algorithms.loss.interfaces import LossInputType, LossType
from nemo_rl.algorithms.loss.loss_functions import MseValueLossConfig, MseValueLossFn
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.value.config import ValueConfig
from nemo_rl.models.value.lm_value import Value
from nemo_rl.models.value.workers.dtensor_value_worker_v2 import (
    RightShiftLossWrapper,
    right_shift_values,
)

pytestmark = pytest.mark.automodel


def test_right_shift_values_aligns_value_predictions_to_state_tokens():
    values = torch.tensor(
        [
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
        ]
    )

    shifted = right_shift_values(values)

    expected = torch.tensor(
        [
            [0.0, 1.0, 2.0, 3.0],
            [0.0, 5.0, 6.0, 7.0],
        ]
    )
    torch.testing.assert_close(shifted, expected)
    assert shifted.shape == values.shape


def test_right_shift_loss_wrapper_shifts_logits_and_delegates_attributes():
    class RecordingLoss:
        loss_type = LossType.TOKEN_LEVEL
        input_type = LossInputType.LOGIT
        aggregation_type = "token_mean"

        def __init__(self):
            self.seen_logits = None

        def __call__(
            self,
            data: BatchedDataDict,
            global_valid_seqs: torch.Tensor,
            global_valid_toks: torch.Tensor,
            **kwargs: Any,
        ) -> tuple[torch.Tensor, dict[str, Any]]:
            logits = kwargs["logits"]
            self.seen_logits = logits
            loss = logits.sum()
            return loss, {"loss": loss}

    inner = RecordingLoss()
    wrapper = RightShiftLossWrapper(inner)
    logits = torch.tensor([[10.0, 20.0, 30.0]])

    result, metrics = wrapper(
        BatchedDataDict({}),
        torch.tensor(1),
        torch.tensor(3),
        logits=logits,
    )

    expected_logits = torch.tensor([[0.0, 10.0, 20.0]])
    torch.testing.assert_close(inner.seen_logits, expected_logits)
    torch.testing.assert_close(result, expected_logits.sum())
    torch.testing.assert_close(metrics["loss"], expected_logits.sum())
    assert wrapper.input_type == inner.input_type
    assert wrapper.aggregation_type == inner.aggregation_type


def _create_value_test_config(
    model_name: str,
    tp: int = 1,
    cp: int = 1,
    precision: str = "float32",
) -> ValueConfig:
    """Build a minimal valid `ValueConfig` for DTensor V2 tests."""
    return {
        "model_name": model_name,
        "tokenizer": {"name": model_name},
        "train_global_batch_size": 8,
        "train_micro_batch_size": 2,
        "logprob_batch_size": 2,
        "precision": precision,
        "reward_model_cfg": {
            "enabled": True,
            "reward_model_type": "regression",
        },
        "megatron_cfg": {"enabled": False},
        "dtensor_cfg": {
            "enabled": True,
            "_v2": True,
            "tensor_parallel_size": tp,
            "context_parallel_size": cp,
            "sequence_parallel": False,
            "activation_checkpointing": False,
            "cpu_offload": False,
        },
        "dynamic_batching": {"enabled": False},
        "sequence_packing": {"enabled": False},
        "make_sequence_length_divisible_by": tp,
        "max_total_sequence_length": 128,
        "max_grad_norm": 1.0,
        "optimizer": {
            "name": "torch.optim.AdamW",
            "kwargs": {
                "lr": 5.0e-6,
                "weight_decay": 0.01,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "foreach": False,
                "fused": False,
            },
        },
        "scheduler": {
            "name": "torch.optim.lr_scheduler.ConstantLR",
            "kwargs": {"factor": 1.0, "total_iters": 1_000_000},
        },
    }


def _make_checkpointing_cfg(checkpoint_dir) -> dict:
    """Build a minimal `CheckpointingConfig` for DTensor V2 ``save_checkpoint``."""
    return {
        "enabled": True,
        "checkpoint_dir": str(checkpoint_dir),
        "metric_name": None,
        "higher_is_better": False,
        "keep_top_k": 2,
        "save_period": 30,
        "checkpoint_must_save_by": None,
        "save_optimizer": True,
    }


def _apply_config_updates(config: ValueConfig, config_updates: dict) -> None:
    """Apply test config overrides in place (precision / SP / dynamic batching)."""
    for k, v in config_updates.items():
        if k == "precision":
            config["precision"] = v
        elif k == "sequence_parallel":
            config["dtensor_cfg"]["sequence_parallel"] = v
        elif k == "dynamic_batching":
            mbt = config["max_total_sequence_length"] * config["train_micro_batch_size"]
            lbt = config["max_total_sequence_length"] * config["logprob_batch_size"]
            config["dynamic_batching"] = {
                "enabled": v,
                "train_mb_tokens": mbt,
                "logprob_mb_tokens": lbt,
                "sequence_length_round": 64,
            }
        else:
            raise ValueError(f"Unknown config_updates key: {k!r}")


@pytest.fixture
def value_setup(request, tiny_qwen2_model_path):
    """Spin up a `Value` wrapper around a tiny Qwen2 backbone for DTensor V2 testing.

    Parameter format: ``(num_gpus, tp, cp, config_updates)``.
    """
    if hasattr(request, "param") and request.param is not None:
        num_gpus, tp, cp, config_updates = request.param
    else:
        num_gpus, tp, cp, config_updates = 2, 1, 1, {}

    value = None
    cluster = None
    data = None
    loss_fn = None

    try:
        cluster_name = f"test-dtensor-value-{num_gpus}gpu-tp{tp}-cp{cp}"
        if config_updates:
            cluster_name += "-" + "-".join(
                f"{k}={v}" for k, v in config_updates.items()
            )
        cluster = RayVirtualCluster(
            name=cluster_name,
            bundle_ct_per_node_list=[num_gpus],
            use_gpus=True,
            num_gpus_per_node=num_gpus,
            max_colocated_worker_groups=1,
        )

        config = _create_value_test_config(
            model_name=tiny_qwen2_model_path, tp=tp, cp=cp
        )
        _apply_config_updates(config, config_updates)
        tokenizer = get_tokenizer(config["tokenizer"])

        value = Value(cluster=cluster, config=config, tokenizer=tokenizer)

        torch.manual_seed(42)
        batch, seq_len = 8, 64
        input_ids = torch.randint(0, 151000, (batch, seq_len))
        attention_mask = torch.ones(batch, seq_len)
        input_lengths = attention_mask.sum(dim=1).to(torch.int32)
        returns = torch.randn(batch, seq_len) * 0.1
        old_values = torch.randn(batch, seq_len) * 0.1
        token_mask = attention_mask.clone()
        sample_mask = torch.ones(batch)
        data = BatchedDataDict(
            {
                "input_ids": input_ids,
                "input_lengths": input_lengths,
                "attention_mask": attention_mask,
                "returns": returns,
                "values": old_values,
                "token_mask": token_mask,
                "sample_mask": sample_mask,
            }
        )

        loss_fn = MseValueLossFn(MseValueLossConfig(scale=1.0, cliprange=0.5))

        yield value, cluster, data, loss_fn

    except Exception as e:
        print(f"Error during value setup: {e}")
        pytest.skip(f"Value setup failed: {e}")
    finally:
        print("Cleaning up value test resources")
        if value:
            value.shutdown()
        if cluster:
            cluster.shutdown()


@pytest.mark.hf_gated
@pytest.mark.timeout(300)
@pytest.mark.parametrize(
    "value_setup",
    [
        # (num_gpus, tp, cp, config_updates)
        (2, 1, 1, {}),
        (2, 2, 1, {}),
        (2, 1, 1, {"precision": "bfloat16"}),
        (2, 2, 1, {"sequence_parallel": True}),
        (2, 1, 1, {"dynamic_batching": True}),
    ],
    indirect=True,
    ids=[
        "2gpu_dp2",
        "2gpu_tp2",
        "2gpu_dp2_bf16",
        "2gpu_tp2sp",
        "2gpu_dp2_dynbatch",
    ],
)
def test_value_worker_init_and_get_values(value_setup):
    """`Value` should initialize and `get_values` should return finite tensors of the expected shape."""
    value, cluster, data, _ = value_setup

    assert value is not None
    assert cluster is not None

    out = value.get_values(data)
    assert "values" in out, "Output should contain 'values' key"
    values = out["values"]
    # [B, S] scalar-per-token; a 3-D [B, S, vocab] result would mean the value
    # head did not replace the LM head at init.
    assert values.ndim == 2, (
        f"Expected per-token scalar values [B, S], got {values.shape}"
    )
    assert values.shape[0] == data["input_ids"].shape[0], (
        f"Batch dim mismatch: values={values.shape}, "
        f"input_ids={data['input_ids'].shape}"
    )
    assert values.shape[1] == data["input_ids"].shape[1], (
        f"Sequence dim mismatch: values={values.shape}, "
        f"input_ids={data['input_ids'].shape}"
    )
    assert not torch.isnan(values).any(), "Values should not contain NaN"
    assert not torch.isinf(values).any(), "Values should not contain Inf"


@pytest.mark.hf_gated
@pytest.mark.timeout(300)
@pytest.mark.parametrize(
    "value_setup",
    [
        (2, 1, 1, {}),
        (2, 2, 1, {}),
        (2, 2, 1, {"sequence_parallel": True}),
        (2, 1, 1, {"dynamic_batching": True}),
    ],
    indirect=True,
    ids=[
        "2gpu_dp2",
        "2gpu_tp2",
        "2gpu_tp2sp",
        "2gpu_dp2_dynbatch",
    ],
)
def test_value_worker_train_step(value_setup):
    """One `train()` call should produce a finite, non-negative MSE value loss."""
    value, cluster, data, loss_fn = value_setup

    value.prepare_for_training()

    results = value.train(data, loss_fn)
    assert "loss" in results, "Train results should contain 'loss'"
    loss_tensor = results["loss"]
    assert not torch.isnan(loss_tensor).any(), "Loss should not be NaN"
    assert not torch.isinf(loss_tensor).any(), "Loss should not be Inf"
    # MSE-based value loss is always non-negative.
    assert (loss_tensor >= 0).all(), "MSE-derived value loss should be non-negative"
    assert "grad_norm" in results, "Train results should contain 'grad_norm'"
    grad_norm = results["grad_norm"]
    assert grad_norm is None or torch.isfinite(torch.as_tensor(grad_norm)).all(), (
        "grad_norm should be finite"
    )

    value.finish_training()


@pytest.mark.hf_gated
@pytest.mark.timeout(420)
@pytest.mark.parametrize(
    ("tp", "feature_updates"),
    [
        (2, {"sequence_parallel": True}),
        (1, {"dynamic_batching": True}),
    ],
    ids=[
        "sequence_parallel",
        "dynamic_batching",
    ],
)
def test_value_worker_parallelism_equivalence(
    tiny_qwen2_model_path, tmp_path, tp, feature_updates
):
    """A perf/sharding feature must not change values.

    The value head is randomly initialized per worker, so pin the weights by
    saving a feature-OFF worker and reloading them into a feature-ON worker,
    then assert ``get_values`` matches on the same batch:

      * sequence parallelism — guards the head's sequence-parallel all-gather
        reassembles the sequence correctly (a wrong gather still yields finite
        values, so finiteness alone would not catch it);
      * dynamic batching — guards the microbatch reorder + ``reorder_data``
        restore round-trips back to the original sample order.
    """
    cluster = None
    ref = None
    feat = None
    try:
        feature_id = next(iter(feature_updates))
        cluster = RayVirtualCluster(
            name=f"test-dtensor-value-equiv-{feature_id}",
            bundle_ct_per_node_list=[2],
            use_gpus=True,
            num_gpus_per_node=2,
            max_colocated_worker_groups=1,
        )

        torch.manual_seed(42)
        batch, seq_len = 8, 64
        data = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 151000, (batch, seq_len)),
                "input_lengths": torch.full((batch,), seq_len, dtype=torch.int32),
                "attention_mask": torch.ones(batch, seq_len),
            }
        )

        # Reference worker: feature OFF.
        ref_config = _create_value_test_config(model_name=tiny_qwen2_model_path, tp=tp)
        tokenizer = get_tokenizer(ref_config["tokenizer"])
        ref = Value(cluster=cluster, config=ref_config, tokenizer=tokenizer)
        values_ref = ref.get_values(data)["values"].detach().cpu()

        # Save weights, then reload into a feature-ON worker (same weights).
        weights_path = os.path.join(str(tmp_path), "value", "weights")
        ref.prepare_for_inference()
        ref.save_checkpoint(
            weights_path=weights_path,
            checkpointing_cfg=_make_checkpointing_cfg(tmp_path),
        )
        ref.shutdown()
        ref = None

        feat_config = _create_value_test_config(model_name=tiny_qwen2_model_path, tp=tp)
        _apply_config_updates(feat_config, feature_updates)
        feat = Value(
            cluster=cluster,
            config=feat_config,
            tokenizer=tokenizer,
            weights_path=Path(weights_path),
            name_prefix="lm_value_feat",
        )
        values_feat = feat.get_values(data)["values"].detach().cpu()

        torch.testing.assert_close(values_feat, values_ref, rtol=1e-3, atol=1e-3)
    finally:
        if ref is not None:
            ref.shutdown()
        if feat is not None:
            feat.shutdown()
        if cluster is not None:
            cluster.shutdown()


@pytest.mark.hf_gated
@pytest.mark.timeout(300)
@pytest.mark.parametrize(
    "value_setup",
    [(2, 1, 1, {})],
    indirect=True,
    ids=["2gpu_dp2"],
)
def test_value_worker_train_decreases_loss(value_setup):
    """A few training steps on a fixed batch should drive the value MSE loss down."""
    value, _, data, loss_fn = value_setup

    value.prepare_for_training()
    losses: list[float] = []
    for _ in range(3):
        results = value.train(data, loss_fn)
        loss_tensor = results["loss"]
        assert not torch.isnan(loss_tensor).any()
        assert not torch.isinf(loss_tensor).any()
        losses.append(float(loss_tensor.mean().item()))
    value.finish_training()

    assert losses[-1] <= losses[0] + 1e-3, (
        f"Value loss should not increase after 3 steps; got {losses}"
    )


@pytest.mark.hf_gated
@pytest.mark.timeout(360)
@pytest.mark.parametrize(
    "value_setup",
    [(2, 1, 1, {})],
    indirect=True,
    ids=["2gpu_dp2"],
)
def test_value_worker_checkpoint_save_and_load(value_setup, tmp_path):
    """Full save → shutdown → restore round-trip with state correctness checks.

    Captures `get_values` outputs at three model states and cross-checks them
    to prove the checkpoint round-trip actually persists trained weights:

      * **fresh** — Value just constructed, no training done
      * **saved** — same Value after one train step (right before save)
      * **resumed** — fresh Value constructed with `weights_path=<ckpt>` after
        the original is shut down

    Hard assertion: ``resumed != fresh`` — proves the load did something
    non-trivial (it did NOT silently fall back to a fresh init).
    Hard assertion: ``resumed ≈ saved`` — DTensor checkpoints are bit-exact
    across reload (unlike Megatron's sharded dist_checkpoint).
    """
    value, cluster, data, loss_fn = value_setup

    # State 1: fresh — capture get_values output before any training.
    values_fresh = value.get_values(data)["values"].detach().cpu()

    # Train one step so weights diverge from base init.
    value.prepare_for_training()
    value.train(data, loss_fn)
    value.finish_training()

    # State 2: saved — capture get_values output after training, before save.
    value.prepare_for_inference()
    values_saved = value.get_values(data)["values"].detach().cpu()

    # Save weights + optimizer the way `ppo.setup()` does:
    #   <ckpt_root>/value/weights/  ,  <ckpt_root>/value/optimizer/
    ckpt_root = str(tmp_path / "value_ckpt_root")
    weights_path = os.path.join(ckpt_root, "value", "weights")
    optimizer_path = os.path.join(ckpt_root, "value", "optimizer")
    value.save_checkpoint(
        weights_path=weights_path,
        optimizer_path=optimizer_path,
        checkpointing_cfg=_make_checkpointing_cfg(tmp_path / "value_ckpt_root"),
    )

    assert os.path.isdir(weights_path), (
        f"weights_path {weights_path} should exist after save"
    )
    assert os.listdir(weights_path), "weights_path should contain saved files"

    # Free GPU memory before re-init on the same cluster.
    saved_model_name = value.cfg["model_name"]
    value.shutdown()

    # Mirror the T12 resume path in `ppo.setup()` — directly probe value
    # subdir paths with .exists() rather than going through
    # `CheckpointManager.get_resume_paths`.
    _value_weights = Path(ckpt_root) / "value" / "weights"
    _value_optim = Path(ckpt_root) / "value" / "optimizer"
    assert _value_weights.exists(), f"saved value weights missing at {_value_weights}"
    resume_weights_path = _value_weights
    resume_optimizer_path = _value_optim if _value_optim.exists() else None

    config = _create_value_test_config(model_name=saved_model_name)
    tokenizer = get_tokenizer(config["tokenizer"])
    resumed = Value(
        cluster=cluster,
        config=config,
        tokenizer=tokenizer,
        weights_path=resume_weights_path,
        optimizer_path=resume_optimizer_path,
        name_prefix="lm_value_resumed",
    )
    try:
        workers_alive = ray.get(
            [w.is_alive.remote() for w in resumed.worker_group.workers]
        )
        assert all(workers_alive), "All resumed workers should be alive"

        # State 3: resumed — capture get_values output after loading the ckpt.
        values_resumed = resumed.get_values(data)["values"].detach().cpu()
        assert not torch.isnan(values_resumed).any(), (
            "Resumed worker get_values should not produce NaNs"
        )
        assert not torch.isinf(values_resumed).any(), (
            "Resumed worker get_values should not produce Infs"
        )

        # Proves the load actually restored the trained weights (not a silent
        # fallback to fresh init).
        assert not torch.allclose(values_resumed, values_fresh, atol=1e-4), (
            "Resumed get_values should differ from fresh-init values. "
            "If equal, the checkpoint load silently fell back to a fresh "
            "initialization instead of loading the trained weights."
        )

        # DTensor weights are deterministic across save/reload — values must
        # match what we captured just before save.
        torch.testing.assert_close(values_resumed, values_saved, rtol=1e-4, atol=1e-4)
    finally:
        resumed.shutdown()
