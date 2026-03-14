# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import os
import tempfile
from copy import deepcopy

import pytest
import ray
import torch

from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.policy.lm_policy import Policy
from tests.unit.models.policy.test_megatron_worker import create_megatron_test_config
from tests.unit.test_utils import SimpleLossFn

_MODELOPT_AVAILABLE = False
try:
    import modelopt.torch.quantization as mtq  # noqa: F401

    _MODELOPT_AVAILABLE = True
except ImportError:
    pass

_CUDA_AVAILABLE = torch.cuda.is_available()

requires_quant = pytest.mark.skipif(
    not (_CUDA_AVAILABLE and _MODELOPT_AVAILABLE),
    reason="Requires CUDA + modelopt for FP8 quantization",
)

_VOCAB_SIZE = 32000
_BATCH_SIZE = 8
_NUM_GPUS = 2


def create_quant_megatron_test_config(model_name, tp=1, pp=1, precision="float32"):
    """Wrap the base Megatron test config with quantization fields."""
    config = create_megatron_test_config(
        model_name=model_name, tp=tp, pp=pp, precision=precision
    )
    config["quant_cfg"] = "FP8_DEFAULT_CFG"
    config["quant_calib_size"] = 1
    config["quant_calib_data"] = "random"
    config["quant_batch_size"] = 1
    config["quant_sequence_length"] = 128
    return config


def _make_cluster(name):
    return RayVirtualCluster(
        name=name,
        bundle_ct_per_node_list=[_NUM_GPUS],
        use_gpus=True,
        num_gpus_per_node=_NUM_GPUS,
        max_colocated_worker_groups=1,
    )


def _prepare_config(model_name, precision="float32"):
    config = create_quant_megatron_test_config(model_name, precision=precision)
    tokenizer = get_tokenizer(config["tokenizer"])
    config["generation"] = configure_generation_config(config["generation"], tokenizer)
    return config, tokenizer


@requires_quant
@pytest.mark.timeout(600)
@pytest.mark.hf_gated
def test_quant_megatron_training(tiny_llama_model_path):
    """Quantized Megatron policy training should converge (loss decreases, no NaN/Inf)."""
    cluster = _make_cluster("test-quant-train")
    config, tokenizer = _prepare_config(tiny_llama_model_path)

    policy = None
    try:
        policy = Policy(
            cluster=cluster,
            config=config,
            tokenizer=tokenizer,
            init_reference_model=False,
        )

        # Verify quantizers were calibrated during init
        stats_futures = policy.worker_group.run_all_workers_single_data(
            "get_quantizer_stats"
        )
        stats_list = ray.get(stats_futures)
        for rank, stats in enumerate(stats_list):
            print(f"Rank {rank} quantizer stats: {stats}")
            assert stats["enabled"] > 0, f"Rank {rank}: no enabled quantizers"
            assert stats["with_amax"] == stats["enabled"], (
                f"Rank {rank}: {stats['enabled'] - stats['with_amax']} enabled quantizers missing amax"
            )
            assert stats["positive_amax"] == stats["with_amax"], (
                f"Rank {rank}: {stats['with_amax'] - stats['positive_amax']} quantizers have non-positive amax"
            )

        torch.manual_seed(42)
        seq_len = 128
        data = BatchedDataDict(
            {
                "input_ids": torch.randint(0, _VOCAB_SIZE, (_BATCH_SIZE, seq_len)),
                "input_lengths": torch.full((_BATCH_SIZE,), seq_len, dtype=torch.int32),
                "attention_mask": torch.ones(_BATCH_SIZE, seq_len),
                "labels": torch.randint(0, _VOCAB_SIZE, (_BATCH_SIZE, seq_len)),
                "sample_mask": torch.ones(_BATCH_SIZE),
            }
        )

        loss_fn = SimpleLossFn()
        policy.prepare_for_training()

        losses = []
        for step in range(3):
            results = policy.train(data, loss_fn)
            loss_tensor = results["loss"]
            assert not torch.isnan(loss_tensor).any(), f"NaN loss at step {step}"
            assert not torch.isinf(loss_tensor).any(), f"Inf loss at step {step}"
            losses.append(loss_tensor[-1].item())
            print(f"Quant training step {step}: loss={losses[-1]:.6f}")

        policy.finish_training()

        assert losses[0] > losses[-1], (
            f"Loss should decrease: {losses[0]:.6f} -> {losses[-1]:.6f}"
        )
    finally:
        if policy:
            policy.shutdown()
        cluster.shutdown()


@requires_quant
@pytest.mark.timeout(600)
@pytest.mark.hf_gated
def test_quant_megatron_reference_policy(tiny_llama_model_path):
    """Reference model should remain unchanged after quantized training."""
    cluster = _make_cluster("test-quant-refpol")

    config, tokenizer = _prepare_config(tiny_llama_model_path)
    config["megatron_cfg"]["optimizer"]["lr"] = 1e-2
    config["megatron_cfg"]["optimizer"]["min_lr"] = 1e-3

    policy = None
    try:
        policy = Policy(
            cluster=cluster,
            config=config,
            tokenizer=tokenizer,
            init_reference_model=True,
        )

        torch.manual_seed(42)
        seq_len = 64
        input_ids = torch.randint(0, _VOCAB_SIZE, (_BATCH_SIZE, seq_len))
        input_lengths = torch.full((_BATCH_SIZE,), seq_len, dtype=torch.int32)
        attention_mask = torch.ones(_BATCH_SIZE, seq_len)

        infer_data = BatchedDataDict(
            {
                "input_ids": input_ids,
                "input_lengths": input_lengths,
                "attention_mask": attention_mask,
            }
        )

        policy.prepare_for_lp_inference()
        initial_logprobs = policy.get_logprobs(infer_data)["logprobs"]
        reference_logprobs = policy.get_reference_policy_logprobs(infer_data)[
            "reference_logprobs"
        ]

        # Logprobs contract checks
        assert initial_logprobs.dtype == torch.float32
        assert initial_logprobs.shape == input_ids.shape
        assert torch.all(initial_logprobs[:, 0] == 0), (
            "First token logprobs should be zero"
        )
        assert not torch.isnan(initial_logprobs).any(), "Active logprobs contain NaN"
        assert not torch.isinf(initial_logprobs).any(), "Active logprobs contain Inf"
        assert not torch.isnan(reference_logprobs).any(), (
            "Reference logprobs contain NaN"
        )

        # Quantized active model and unquantized reference model diverge even before training
        quant_gap = torch.max(torch.abs(initial_logprobs - reference_logprobs)).item()
        print(f"Pre-training quantization gap (active vs ref): {quant_gap:.6f}")

        train_data = BatchedDataDict(
            {
                "input_ids": input_ids,
                "input_lengths": input_lengths,
                "attention_mask": attention_mask,
                "labels": torch.randint(0, _VOCAB_SIZE, (_BATCH_SIZE, seq_len)),
                "sample_mask": torch.ones(_BATCH_SIZE),
            }
        )

        loss_fn = SimpleLossFn()
        policy.prepare_for_training()

        losses = []
        for step in range(10):
            results = policy.train(train_data, loss_fn)
            losses.append(results["loss"][-1].item())
            print(f"Quant ref-pol training step {step}: loss={losses[-1]:.6f}")

        policy.finish_training()

        assert losses[0] > losses[-1], (
            f"Loss should decrease: {losses[0]:.6f} -> {losses[-1]:.6f}"
        )

        policy.prepare_for_lp_inference()
        post_train_logprobs = policy.get_logprobs(infer_data)["logprobs"]
        post_train_ref_logprobs = policy.get_reference_policy_logprobs(infer_data)[
            "reference_logprobs"
        ]

        torch.testing.assert_close(
            reference_logprobs, post_train_ref_logprobs, rtol=1e-4, atol=1e-4
        )

        logprobs_changed = not torch.allclose(
            initial_logprobs, post_train_logprobs, rtol=1e-2, atol=1e-2
        )
        max_diff = torch.max(torch.abs(initial_logprobs - post_train_logprobs)).item()
        assert logprobs_changed, (
            f"Active model logprobs should change after training "
            f"(max diff={max_diff:.6f})"
        )
    finally:
        if policy:
            policy.shutdown()
        cluster.shutdown()


@requires_quant
@pytest.mark.timeout(600)
@pytest.mark.hf_gated
def test_quant_megatron_checkpoint_save_restore(tiny_llama_model_path):
    """Quantized checkpoint round-trip: save -> kill -> restore -> logprobs match."""
    import gc

    with tempfile.TemporaryDirectory(prefix="quant_ckpt_") as temp_dir:
        checkpoint_dir = os.path.join(temp_dir, "quant_restore_test")

        config, tokenizer = _prepare_config(tiny_llama_model_path)

        torch.manual_seed(42)
        seq_len = 32
        input_ids = torch.randint(0, _VOCAB_SIZE, (_BATCH_SIZE, seq_len))
        input_lengths = torch.full((_BATCH_SIZE,), seq_len, dtype=torch.int32)
        attention_mask = torch.ones(_BATCH_SIZE, seq_len)

        train_data = BatchedDataDict(
            {
                "input_ids": input_ids,
                "input_lengths": input_lengths,
                "attention_mask": attention_mask,
                "labels": torch.randint(0, _VOCAB_SIZE, (_BATCH_SIZE, seq_len)),
                "sample_mask": torch.ones(_BATCH_SIZE),
            }
        )

        sample_size = _BATCH_SIZE // 2
        sample_data = BatchedDataDict(
            {
                "input_ids": input_ids[:sample_size],
                "input_lengths": input_lengths[:sample_size],
                "attention_mask": attention_mask[:sample_size],
            }
        )

        # --- Phase 1: train & save ---
        cluster1 = _make_cluster("test-quant-ckpt-1")
        policy1 = None
        logprobs_before_save = None
        try:
            policy1 = Policy(cluster=cluster1, config=config, tokenizer=tokenizer)

            loss_fn = SimpleLossFn()
            policy1.prepare_for_training()
            for step in range(5):
                results = policy1.train(train_data, loss_fn)
                print(
                    f"Quant ckpt phase-1 step {step}: "
                    f"loss={results['loss'][-1].item():.6f}"
                )

            policy1.prepare_for_lp_inference()
            logprobs_before_save = policy1.get_logprobs(sample_data)["logprobs"]
            print(f"Logprobs before save (first vals): {logprobs_before_save[0, :5]}")

            policy1.save_checkpoint(
                weights_path=checkpoint_dir,
                optimizer_path=checkpoint_dir,
            )
            assert os.path.exists(checkpoint_dir), "Checkpoint dir not created"
        finally:
            if policy1:
                policy1.finish_training()
                policy1.shutdown()
            cluster1.shutdown()
            gc.collect()
            torch.cuda.empty_cache()

        # --- Phase 2: restore & compare ---
        cluster2 = _make_cluster("test-quant-ckpt-2")
        policy2 = None
        try:
            restore_config = deepcopy(config)
            policy2 = Policy(
                cluster=cluster2,
                config=restore_config,
                tokenizer=tokenizer,
                weights_path=checkpoint_dir,
                init_reference_model=False,
            )

            policy2.prepare_for_lp_inference()
            logprobs_restored = policy2.get_logprobs(sample_data)["logprobs"]
            print(f"Logprobs restored (first vals): {logprobs_restored[0, :5]}")

            max_diff = torch.max(
                torch.abs(logprobs_before_save - logprobs_restored)
            ).item()
            mean_diff = torch.mean(
                torch.abs(logprobs_before_save - logprobs_restored)
            ).item()
            print(f"Checkpoint restore diff -- max: {max_diff}, mean: {mean_diff}")

            assert torch.allclose(logprobs_before_save, logprobs_restored, atol=1e-4), (
                f"Restored logprobs should match saved (max_diff={max_diff})"
            )
        finally:
            if policy2:
                policy2.shutdown()
            cluster2.shutdown()
