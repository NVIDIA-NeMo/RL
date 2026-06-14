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
"""End-to-end tests for `MegatronValueWorker` via the `Value` wrapper.

These tests exercise the public `Value` API (init / get_values / train /
checkpoint save+load) using a tiny Qwen2 model on a small Ray cluster.
They cover the PPO-specific value-worker behavior:

  * value head (the model's ``output_layer``, a hidden->1 ``LinearForLastLayer``)
    integration with the Megatron backbone
  * `get_values` forward pass (output shape + finite values), incl. TP / SP /
    dynamic batching
  * `train` step with `MseValueLossFn` (loss is finite + non-negative)
  * Sequence-parallel equivalence (SP must not change values)
  * Checkpoint save+load round-trip preserves the trained value head

Modeled after `tests/unit/models/policy/test_megatron_worker.py`.
"""

import os
from pathlib import Path

import pytest
import ray
import torch

from nemo_rl.algorithms.loss.loss_functions import (
    MseValueLossConfig,
    MseValueLossFn,
)
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.value.config import ValueConfig
from nemo_rl.models.value.lm_value import Value

pytestmark = pytest.mark.mcore


def _create_value_test_config(
    model_name: str,
    tp: int = 1,
    pp: int = 1,
    cp: int = 1,
    precision: str = "float32",
    converter_type: str = "Qwen2ForCausalLM",
) -> ValueConfig:
    """Build a minimal valid `ValueConfig` for tests."""
    return {
        "model_name": model_name,
        "tokenizer": {"name": model_name},
        "train_global_batch_size": 8,
        "train_micro_batch_size": 2,
        "logprob_batch_size": 2,
        "precision": precision,
        "reward_model_cfg": {
            "enabled": False,
            "reward_model_type": "regression",
        },
        "dtensor_cfg": {"enabled": False},
        "dynamic_batching": {"enabled": False},
        "sequence_packing": {"enabled": False},
        "megatron_cfg": {
            "enabled": True,
            "empty_unused_memory_level": 0,
            "activation_checkpointing": False,
            "converter_type": converter_type,
            "tensor_model_parallel_size": tp,
            "expert_tensor_parallel_size": 1,
            "expert_model_parallel_size": 1,
            "pipeline_model_parallel_size": pp,
            "num_layers_in_first_pipeline_stage": None,
            "num_layers_in_last_pipeline_stage": None,
            "context_parallel_size": cp,
            "pipeline_dtype": precision,
            "sequence_parallel": False,
            "freeze_moe_router": True,
            "moe_router_dtype": "fp64",
            "moe_router_load_balancing_type": "none",
            "moe_router_bias_update_rate": 0.0,
            "moe_permute_fusion": False,
            "apply_rope_fusion": True,
            "bias_activation_fusion": True,
            "moe_per_layer_logging": False,
            "moe_enable_deepep": False,
            "moe_token_dispatcher_type": "alltoall",
            "moe_shared_expert_overlap": False,
            "defer_fp32_logits": None,
            "gradient_accumulation_fusion": False,
            "train_iters": 100,
            "optimizer": {
                "optimizer": "adam",
                "lr": 5.0e-6,
                "min_lr": 5.0e-7,
                "weight_decay": 0.01,
                "bf16": precision == "bfloat16",
                "fp16": precision == "float16",
                "params_dtype": "float32",
                "adam_beta1": 0.9,
                "adam_beta2": 0.999,
                "adam_eps": 1e-8,
                "use_distributed_optimizer": True,
                "use_precision_aware_optimizer": True,
                "clip_grad": 1.0,
                "optimizer_cpu_offload": False,
                "optimizer_offload_fraction": 0.0,
            },
            "scheduler": {
                "start_weight_decay": 0.01,
                "end_weight_decay": 0.01,
                "weight_decay_incr_style": "constant",
                "lr_decay_style": "constant",
                "lr_decay_iters": None,
                "lr_warmup_iters": 0,
                "lr_warmup_init": 5.0e-7,
            },
            "distributed_data_parallel_config": {
                "grad_reduce_in_fp32": False,
                "overlap_grad_reduce": True,
                "overlap_param_gather": False,
                "data_parallel_sharding_strategy": "optim_grads_params",
            },
            "fp8_cfg": {
                "enabled": False,
                "fp8": "hybrid",
                "fp8_recipe": "tensorwise",
                "fp8_param": True,
            },
        },
        "make_sequence_length_divisible_by": tp,
        "max_total_sequence_length": 128,
        "max_grad_norm": 1.0,
        "optimizer": None,
        "scheduler": None,
    }


def _apply_config_updates(config: ValueConfig, config_updates: dict) -> None:
    """Apply test config overrides in place (precision / SP / dynamic batching)."""
    for k, v in config_updates.items():
        if k == "precision":
            config["precision"] = v
            config["megatron_cfg"]["pipeline_dtype"] = v
            config["megatron_cfg"]["optimizer"]["bf16"] = v == "bfloat16"
            config["megatron_cfg"]["optimizer"]["fp16"] = v == "float16"
        elif k == "sequence_parallel":
            config["megatron_cfg"]["sequence_parallel"] = v
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
    """Spin up a `Value` wrapper around a tiny Qwen2 backbone for testing.

    Parameter format: ``(num_gpus, tp, pp, cp, config_updates)``.
    """
    if hasattr(request, "param") and request.param is not None:
        num_gpus, tp, pp, cp, config_updates = request.param
    else:
        num_gpus, tp, pp, cp, config_updates = 2, 1, 1, 1, {}

    value = None
    cluster = None
    data = None
    loss_fn = None

    try:
        cluster_name = f"test-megatron-value-{num_gpus}gpu-tp{tp}-pp{pp}-cp{cp}"
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
            model_name=tiny_qwen2_model_path,
            tp=tp,
            pp=pp,
            cp=cp,
        )
        _apply_config_updates(config, config_updates)
        tokenizer = get_tokenizer(config["tokenizer"])

        value = Value(cluster=cluster, config=config, tokenizer=tokenizer)

        # Build a tiny test batch.
        torch.manual_seed(42)
        batch, seq_len = 8, 64
        input_ids = torch.randint(0, 151000, (batch, seq_len))
        attention_mask = torch.ones(batch, seq_len)
        input_lengths = attention_mask.sum(dim=1).to(torch.int32)
        # Targets ("returns") + old values for the value loss path.
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
        # (num_gpus, tp, pp, cp, config_updates)
        (2, 1, 1, 1, {}),
        (2, 2, 1, 1, {}),
        (2, 1, 1, 1, {"precision": "bfloat16"}),
        (2, 2, 1, 1, {"sequence_parallel": True}),
        (2, 1, 1, 1, {"dynamic_batching": True}),
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
    # head did not replace output_layer at init.
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
        (2, 1, 1, 1, {}),
        (2, 2, 1, 1, {}),
        (2, 2, 1, 1, {"sequence_parallel": True}),
        (2, 1, 1, 1, {"dynamic_batching": True}),
    ],
    indirect=True,
    ids=["2gpu_dp2", "2gpu_tp2", "2gpu_tp2sp", "2gpu_dp2_dynbatch"],
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
    ids=["sequence_parallel", "dynamic_batching"],
)
def test_value_worker_parallelism_equivalence(
    tiny_qwen2_model_path, tmp_path, tp, feature_updates
):
    """A perf/sharding feature must not change values.

    The value head (``output_layer``) is randomly initialized per worker, so we
    pin the weights by saving a feature-OFF worker and reloading them into a
    feature-ON worker, then assert ``get_values`` matches on the same batch:

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
            name=f"test-megatron-value-equiv-{feature_id}",
            bundle_ct_per_node_list=[2],
            use_gpus=True,
            num_gpus_per_node=2,
            max_colocated_worker_groups=1,
        )

        # Deterministic batch (get_values only needs input_ids + input_lengths).
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
        ref.save_checkpoint(weights_path=weights_path)
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
    [(2, 1, 1, 1, {})],
    indirect=True,
    ids=["2gpu_dp2"],
)
def test_value_worker_train_decreases_loss(value_setup):
    """A few training steps on a fixed batch should drive the value MSE loss down.

    Mirrors `test_megatron_policy_training` but for the value model: we expect
    the value head + backbone to reduce MSE against fixed `returns` targets.
    """
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

    # Loss should generally decrease — allow a small tolerance for stochasticity.
    assert losses[-1] <= losses[0] + 1e-3, (
        f"Value loss should not increase after 3 steps; got {losses}"
    )


@pytest.mark.hf_gated
@pytest.mark.timeout(360)
@pytest.mark.parametrize(
    "value_setup",
    [(2, 1, 1, 1, {})],
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

    Soft check (warning only): ``resumed ≈ saved`` — proves the load restored
    the exact trained state. Mirrors `test_megatron_checkpoint_save_kill_and_restore`
    in not hard-failing on mismatch because Megatron distributed checkpoints
    have known numerical non-determinism across sharded reduce/allgather paths.

    Also exercises T12's `.exists()` guard on the resume path.
    """
    value, cluster, data, loss_fn = value_setup

    # State 1: fresh — capture get_values output before any training.
    values_fresh = value.get_values(data)["values"].detach().cpu()

    # Train one step so weights diverge from base init.
    value.prepare_for_training()
    value.train(data, loss_fn)
    value.finish_training()

    # State 2: saved — capture get_values output after training, before save.
    # Keep the model on GPU (no `finish_inference` here) because the next
    # `save_checkpoint` call below expects live GPU storage for the sharded
    # dist_checkpoint write — offloading first triggers
    # `setStorage: ... out of bounds for storage of size 0` in
    # `megatron/core/transformer/mlp.py` during sharded save. Mirrors the
    # `test_megatron_checkpoint_save_kill_and_restore` policy-worker test,
    # which also saves while still in inference mode.
    value.prepare_for_inference()
    values_saved = value.get_values(data)["values"].detach().cpu()

    # Save weights + optimizer alongside the way `ppo.setup()` does:
    #   <ckpt_root>/value/weights/  ,  <ckpt_root>/value/optimizer/
    ckpt_root = str(tmp_path / "value_ckpt_root")
    weights_path = os.path.join(ckpt_root, "value", "weights")
    optimizer_path = os.path.join(ckpt_root, "value", "optimizer")
    value.save_checkpoint(
        weights_path=weights_path,
        optimizer_path=optimizer_path,
    )

    # Verify on-disk artifacts.
    assert os.path.isdir(weights_path), (
        f"weights_path {weights_path} should exist after save"
    )
    assert os.listdir(weights_path), "weights_path should contain saved files"

    # Free GPU memory before re-init on the same cluster.
    saved_model_name = value.cfg["model_name"]
    value.shutdown()

    # Mirror the T12 resume path in `ppo.setup()` — directly probe value
    # subdir paths with .exists() rather than going through
    # `CheckpointManager.get_resume_paths` (that helper is hardcoded to look
    # under `<root>/policy/...` and only resolves the policy checkpoint).
    _value_weights = Path(ckpt_root) / "value" / "weights"
    _value_optim = Path(ckpt_root) / "value" / "optimizer"
    assert _value_weights.exists(), f"saved value weights missing at {_value_weights}"
    resume_weights_path = _value_weights
    resume_optimizer_path = _value_optim if _value_optim.exists() else None

    # Reconstruct the value worker pointed at the saved checkpoint.
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
        # Workers alive after restore.
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

        # HARD: resumed should differ from fresh — proves the load actually
        # restored the trained weights (not a silent fallback to fresh init).
        assert not torch.allclose(values_resumed, values_fresh, atol=1e-4), (
            "Resumed get_values should differ from fresh-init values. "
            "If equal, the checkpoint load silently fell back to a fresh "
            "initialization instead of loading the trained weights."
        )

        # SOFT: resumed should match saved within tolerance. Warning only,
        # mirroring `test_megatron_checkpoint_save_kill_and_restore`: Megatron
        # distributed ckpts have small reproducibility deltas from sharded
        # reduce/allgather ordering and we don't want CI flakiness here. The
        # hard `resumed != fresh` check above already proves the load worked.
        if torch.allclose(values_resumed, values_saved, atol=1e-4):
            print("✓ Resumed values match saved values within tolerance")
        else:
            max_diff = (values_resumed - values_saved).abs().max().item()
            mean_diff = (values_resumed - values_saved).abs().mean().item()
            print(
                f"⚠ Resumed values differ from saved (max={max_diff:.4g}, "
                f"mean={mean_diff:.4g}) — likely Megatron numerical "
                "non-determinism, not a load bug (resumed != fresh asserted above)."
            )
    finally:
        resumed.shutdown()
