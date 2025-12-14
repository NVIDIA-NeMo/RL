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

import pytest
import ray
import torch

from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.policy import AutomodelKwargs, PolicyConfig
from nemo_rl.models.policy.lm_policy import Policy
from tests.unit.test_utils import SimpleLoss


def create_test_config(
    model_name: str,
    tp: int = 1,
    cp: int = 1,
    sp: bool = False,
    cpu_offload: bool = False,
    activation_checkpointing: bool = False,
    custom_parallel_plan: str | None = None,
    dtensor_v2: bool = False,
    precision: str = "float32",
    expert_parallel_size: int = 1,
    use_hf_tp_plan: bool = False,
    sequence_packing_enabled: bool = False,
    automodel_kwargs: AutomodelKwargs | None = None,
    checkpointing: dict | None = None,
) -> PolicyConfig:
    config = {
        "model_name": model_name,
        "tokenizer": {"name": model_name},
        "generation_batch_size": 1,  # Small batch size for testing
        "train_global_batch_size": 4,
        "train_micro_batch_size": 1,
        "learning_rate": 5e-6,
        "logprob_batch_size": 1,
        "precision": precision,
        "offload_optimizer_for_logprob": False,
        "generation": {
            "backend": "hf",
            "temperature": 1.0,
            "max_new_tokens": 16,  # Small number of tokens for testing
            "top_p": 1.0,
            "top_k": None,
            "stop_token_ids": None,
            "stop_strings": None,
            "colocated": {
                "enabled": True,
                "resources": {
                    "gpus_per_node": None,
                    "num_nodes": None,
                },
            },
        },
        "dtensor_cfg": {
            **({"_v2": dtensor_v2} if dtensor_v2 else {}),
            "enabled": True,
            "cpu_offload": cpu_offload,
            "sequence_parallel": sp,
            "activation_checkpointing": activation_checkpointing,
            "tensor_parallel_size": tp,
            "context_parallel_size": cp,
            "custom_parallel_plan": custom_parallel_plan,
            "expert_parallel_size": expert_parallel_size,
            "use_hf_tp_plan": use_hf_tp_plan,
        },
        "dynamic_batching": {
            "enabled": True,
            "train_mb_tokens": 128,
            "logprob_mb_tokens": 128,
            "sequence_length_round": 4,
        },
        "sequence_packing": {
            "enabled": sequence_packing_enabled,
            "train_mb_tokens": 128,
        },
        "optimizer": {
            "name": "torch.optim.AdamW",
            "kwargs": {
                "lr": 5e-6,
                "weight_decay": 0.01,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "foreach": False,
                "fused": False,
            },
        },
        "scheduler": {
            "name": "torch.optim.lr_scheduler.CosineAnnealingLR",
            "kwargs": {
                "T_max": 100,
            },
        },
        "max_grad_norm": 1.0,
    }
    if automodel_kwargs is not None:
        config["dtensor_cfg"]["automodel_kwargs"] = automodel_kwargs
    if checkpointing is not None:
        config["checkpointing"] = checkpointing
    return config


def create_test_batch(
    batch_size: int = 8,
    seq_len: int = 128,
    vocab_size: int = 32000,
    mode: str = "train",
) -> BatchedDataDict:
    """Create a test batch for training or logprob computation."""
    torch.manual_seed(66)
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len)
    input_lengths = attention_mask.sum(dim=1).to(torch.int32)
    data = BatchedDataDict(
        {
            "input_ids": input_ids,
            "input_lengths": input_lengths,
            "attention_mask": attention_mask,
            **(
                {
                    "labels": torch.randint(0, vocab_size, (batch_size, seq_len)),
                    "sample_mask": torch.ones(batch_size).cuda(),
                }
                if mode == "train"
                else {}
            ),
        }
    )
    data = data.to("cpu")
    return data


@pytest.fixture(scope="module")
def two_gpu_virtual_cluster():
    cluster_name = "test"
    print(f"Creating virtual cluster '{cluster_name}'...")
    cluster = RayVirtualCluster(
        name=cluster_name,
        bundle_ct_per_node_list=[2],  # Use tp bundles, one per GPU
        use_gpus=True,
        num_gpus_per_node=2,  # Using tp GPUs
        max_colocated_worker_groups=1,  # Only one worker group
    )
    yield cluster
    print("Shutting down virtual cluster...")
    cluster.shutdown()


def compare_model_configs(config_v1: dict, config_v2: dict) -> list[str]:
    """
    Compare two model configurations and return a list of discrepancies.

    Args:
        config_v1: Model config from dtensor worker v1
        config_v2: Model config from dtensor worker v2

    Returns:
        List of discrepancy descriptions. Empty list if configs are equivalent.
    """
    discrepancies = []

    def compare_dicts(d1, d2, path=""):
        """Recursively compare two dictionaries."""
        all_keys = set(d1.keys()) | set(d2.keys())

        for key in all_keys:
            current_path = f"{path}.{key}" if path else key

            if key not in d1:
                discrepancies.append(f"Key '{current_path}' missing in v1 config")
            elif key not in d2:
                discrepancies.append(f"Key '{current_path}' missing in v2 config")
            else:
                val1, val2 = d1[key], d2[key]

                if isinstance(val1, dict) and isinstance(val2, dict):
                    compare_dicts(val1, val2, current_path)
                elif val1 != val2:
                    discrepancies.append(
                        f"Value mismatch at '{current_path}': v1={val1}, v2={val2}"
                    )

    compare_dicts(config_v1, config_v2)
    return discrepancies


@pytest.mark.hf_gated
@pytest.mark.parametrize(
    "model_fixture_name,tp,cp,sp,cpu_offload,activation_checkpointing",
    [
        # TP=2, CP=1
        ("tiny_qwen2_model_path", 2, 1, False, False, False),
        ("tiny_llama_model_path", 2, 1, False, False, False),
        ("tiny_qwen3_model_path", 2, 1, False, False, False),
        ("tiny_gemma3_model_path", 2, 1, False, False, False),
        # TP=1, CP=2
        ("tiny_qwen2_model_path", 1, 2, False, False, False),
        ("tiny_llama_model_path", 1, 2, False, False, False),
        ("tiny_qwen3_model_path", 1, 2, False, False, False),
    ],
)
def test_dtensor_worker_v1_v2_model_config_equivalence(
    request,
    two_gpu_virtual_cluster,  # noqa: F811
    model_fixture_name,
    tp,
    cp,
    sp,
    cpu_offload,
    activation_checkpointing,
):
    """Test that dtensor worker v1 and v2 produce equivalent model configurations.

    This test verifies that DTensorPolicyWorkerV2 produces the same model config
    as the v1 worker, ensuring backward compatibility.
    """
    # Get the actual model path from the fixture name
    model_name = request.getfixturevalue(model_fixture_name)
    # Create v1 configuration
    config_v1 = create_test_config(
        model_name=model_name,
        tp=tp,
        cp=cp,
        sp=sp,
        cpu_offload=cpu_offload,
        activation_checkpointing=activation_checkpointing,
        dtensor_v2=False,  # Use v1 worker
    )
    # Create and test v1 policy first
    print("Creating policy with v1 worker...")
    policy_v1 = Policy(
        tokenizer=get_tokenizer(config_v1["tokenizer"]),
        config=config_v1,
        init_optimizer=False,
        init_reference_model=False,
        cluster=two_gpu_virtual_cluster,
        name_prefix="lm_policy_v1",
    )

    model_config_v1 = ray.get(
        policy_v1.worker_group.workers[0].return_model_config.remote()
    )
    policy_v1.shutdown()

    # Create v2 configuration
    config_v2 = create_test_config(
        model_name=model_name,
        tp=tp,
        cp=cp,
        sp=sp,
        cpu_offload=cpu_offload,
        activation_checkpointing=activation_checkpointing,
        dtensor_v2=True,  # Use v2 worker
    )
    policy_v2 = Policy(
        tokenizer=get_tokenizer(config_v2["tokenizer"]),
        config=config_v2,
        init_optimizer=False,
        init_reference_model=False,
        cluster=two_gpu_virtual_cluster,
        name_prefix="lm_policy_v2",
    )

    model_config_v2 = ray.get(
        policy_v2.worker_group.workers[0].return_model_config.remote()
    )
    policy_v2.shutdown()

    config_v1_dict = vars(model_config_v1)
    config_v2_dict = vars(model_config_v2)
    config_v1_dict.pop("nemo_version", None)
    config_v2_dict.pop("nemo_version", None)
    config_v1_dict.pop("pad_token_id", None)
    config_v2_dict.pop("pad_token_id", None)

    discrepancies = compare_model_configs(config_v1_dict, config_v2_dict)
    assert not discrepancies, (
        f"Model configurations differ between v1 and v2 approaches for {model_name}"
    )


@pytest.mark.hf_gated
@pytest.mark.timeout(360)
def test_dtensor_v2_checkpoint_save_and_load(
    two_gpu_virtual_cluster,
    tiny_llama_model_path,
):
    with tempfile.TemporaryDirectory() as tmpdir:
        checkpointing_config = {
            "enabled": True,
            "checkpoint_dir": tmpdir,
            "metric_name": None,  # Save most recent checkpoints
            "higher_is_better": False,
            "keep_top_k": 2,
            "save_period": 30,
            "checkpoint_must_save_by": None,
        }

        config = create_test_config(
            model_name=tiny_llama_model_path,
            tp=2,
            cp=1,
            dtensor_v2=True,
            checkpointing=checkpointing_config,
        )

        policy = Policy(
            tokenizer=get_tokenizer(config["tokenizer"]),
            config=config,
            init_optimizer=True,
            init_reference_model=False,
            cluster=two_gpu_virtual_cluster,
            name_prefix="lm_policy_checkpoint",
        )

        try:
            weights_path = os.path.join(tmpdir, "weights")
            optimizer_path = os.path.join(tmpdir, "optimizer")

            # Save checkpoint
            policy.save_checkpoint(
                weights_path=weights_path,
                optimizer_path=optimizer_path,
                checkpointing_cfg=checkpointing_config,
            )

            # Verify checkpoint files were created
            assert os.path.exists(weights_path), "Weights path should exist after save"

            # Load checkpoint into a new policy
            config2 = create_test_config(
                model_name=tiny_llama_model_path,
                tp=2,
                cp=1,
                dtensor_v2=True,
                checkpointing=checkpointing_config,
            )

            # Shutdown original policy first to free GPU memory
            policy.shutdown()
            policy = None

            policy2 = Policy(
                tokenizer=get_tokenizer(config2["tokenizer"]),
                config=config2,
                init_optimizer=True,
                init_reference_model=False,
                cluster=two_gpu_virtual_cluster,
                name_prefix="lm_policy_checkpoint_loaded",
                weights_path=weights_path,
                optimizer_path=optimizer_path,
            )

            # Verify policy was loaded successfully
            assert len(policy2.worker_group.workers) == 2
            worker_alive = ray.get(
                [w.is_alive.remote() for w in policy2.worker_group.workers]
            )
            assert all(worker_alive)

            policy2.shutdown()
        finally:
            if policy is not None:
                policy.shutdown()


@pytest.mark.hf_gated
@pytest.mark.timeout(360)
@pytest.mark.parametrize("precision", ["bfloat16", "float16"])
def test_dtensor_v2_mixed_precision_training_and_logprobs(
    two_gpu_virtual_cluster,
    tiny_llama_model_path,
    precision,
):
    config = create_test_config(
        model_name=tiny_llama_model_path,
        tp=2,
        cp=1,
        dtensor_v2=True,
        precision=precision,
    )

    policy = Policy(
        tokenizer=get_tokenizer(config["tokenizer"]),
        config=config,
        init_optimizer=True,
        init_reference_model=False,
        cluster=two_gpu_virtual_cluster,
        name_prefix=f"lm_policy_{precision}_mixed",
    )

    try:
        # --- Test Training ---
        train_data = create_test_batch(mode="train")
        loss_fn = SimpleLoss()

        policy.prepare_for_training()
        results = policy.train(train_data, loss_fn)

        # Verify training completed successfully
        assert "loss" in results
        loss_tensor = results["loss"]
        assert not torch.isnan(loss_tensor).any(), (
            f"Loss should not be NaN with precision={precision}"
        )
        assert not torch.isinf(loss_tensor).any(), (
            f"Loss should not be Inf with precision={precision}"
        )
        # Loss is returned in float32 (reduced in float32 for numerical stability)
        assert loss_tensor.dtype == torch.float32, (
            f"Loss should be float32, got {loss_tensor.dtype}"
        )

        policy.finish_training()

        # --- Test Logprobs ---
        logprob_data = create_test_batch(mode="logprob")

        policy.prepare_for_lp_inference()
        logprobs = policy.get_logprobs(logprob_data)

        # Verify logprobs were computed successfully
        assert "logprobs" in logprobs
        logprobs_tensor = logprobs["logprobs"]
        assert logprobs_tensor.shape[0] == logprob_data.size
        assert not torch.isnan(logprobs_tensor).any(), (
            f"Logprobs should not be NaN with precision={precision}"
        )
        assert not torch.isinf(logprobs_tensor).any(), (
            f"Logprobs should not be Inf with precision={precision}"
        )
        # Logprobs are returned in float32 for numerical stability
        assert logprobs_tensor.dtype == torch.float32, (
            f"Logprobs should be float32 for numerical stability, got {logprobs_tensor.dtype}"
        )

        # Verify the configured precision by checking worker info
        worker_info = ray.get(policy.worker_group.workers[0].get_gpu_info.remote())
        assert worker_info is not None, "Should get worker info"
    finally:
        policy.shutdown()


def test_dtensor_worker_v2_generate_lora_weights(two_gpu_virtual_cluster):
    """Test that dtensor worker v2 can generate with LoRA weights."""
    config = create_test_config(
        model_name="Qwen/Qwen3-0.6B",
        dtensor_v2=True,
    )
    lora_config = {
        "enabled": True,
        "target_modules": [],
        "exclude_modules": [],
        "match_all_linear": True,
        "dim": 8,
        "alpha": 32,
        "dropout": 0.0,
        "dropout_position": "post",
        "lora_A_init": "xavier",
    }
    config["dtensor_cfg"]["lora_cfg"] = lora_config
    lm_policy = Policy(
        cluster=two_gpu_virtual_cluster,
        config=config,
        tokenizer=get_tokenizer(config["tokenizer"]),
    )
    lm_policy.stream_weights_via_ipc_zmq(buffer_size_bytes=1024**3, kv_scales=None)
