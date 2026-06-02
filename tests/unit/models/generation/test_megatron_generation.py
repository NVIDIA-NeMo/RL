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

import gc
from copy import deepcopy

import pytest
import ray
import torch

from nemo_rl.algorithms.grpo import refit_policy_generation
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.generation.megatron import MegatronGeneration
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.lm_policy import Policy

model_name = "Qwen/Qwen3-0.6B"

basic_megatron_test_config: PolicyConfig = {
    "model_name": model_name,
    "tokenizer": {"name": model_name},
    "generation_batch_size": 2,
    "train_global_batch_size": 4,
    "train_micro_batch_size": 2,
    "learning_rate": 5e-6,
    "logprob_batch_size": 2,
    "precision": "float32",
    "offload_optimizer_for_logprob": False,
    "dtensor_cfg": {"enabled": False},
    "dynamic_batching": {"enabled": False},
    "sequence_packing": {"enabled": False},
    "megatron_cfg": {
        "enabled": True,
        "empty_unused_memory_level": 0,
        "activation_checkpointing": False,
        "converter_type": "Qwen2ForCausalLM",  # Qwen2 converter is compatible with Qwen3
        "tensor_model_parallel_size": 1,
        "expert_tensor_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "num_layers_in_first_pipeline_stage": None,
        "num_layers_in_last_pipeline_stage": None,
        "context_parallel_size": 1,
        "pipeline_dtype": "float32",
        "sequence_parallel": False,
        "freeze_moe_router": True,
        "moe_router_dtype": "fp64",
        "moe_router_load_balancing_type": "none",
        "moe_router_bias_update_rate": 0.0,
        "moe_permute_fusion": False,
        "moe_enable_deepep": False,
        "moe_token_dispatcher_type": "alltoall",
        "moe_shared_expert_overlap": False,
        "apply_rope_fusion": True,
        "bias_activation_fusion": True,
        "moe_per_layer_logging": False,
        "gradient_accumulation_fusion": False,
        "train_iters": 100,
        "optimizer": {
            "optimizer": "adam",
            "lr": 5.0e-6,
            "min_lr": 5.0e-7,
            "weight_decay": 0.01,
            "bf16": False,
            "fp16": False,
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
            "lr_warmup_iters": 50,
            "lr_warmup_init": 5.0e-7,
        },
        "distributed_data_parallel_config": {
            "grad_reduce_in_fp32": False,
            "overlap_grad_reduce": True,
            "overlap_param_gather": False,
            "data_parallel_sharding_strategy": "optim_grads_params",
        },
    },
    "draft": {"enabled": False},
    "optimizer": None,
    "scheduler": None,
    "max_grad_norm": 1.0,
    "generation": {
        "backend": "megatron",
        "model_name": model_name,
        "max_new_tokens": 16,  # Small number of tokens for testing
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": None,
        "stop_token_ids": None,
        "stop_strings": None,
        "colocated": {
            "enabled": False,
            "resources": {"gpus_per_node": None, "num_nodes": None},
        },
        "mcore_generation_config": {
            "async_engine": False,
            "max_model_len": 1024,
            "cuda_graph_impl": "local",
            "inference_cuda_graph_scope": "block",
            "buffer_size_gb": 10,
            "num_cuda_graphs": 4,
            "block_size_tokens": 256,
            "use_cuda_graphs_for_non_decode_steps": True,
            "enable_chunked_prefill": True,
            "enable_prefix_caching": False,
            "max_tokens": 16384,
            "kv_cache_management_mode": "persist",
            "materialize_only_last_token_logits": True,
            "num_speculative_tokens": 0,
            "refit_backend": "nvshmem",
            "parsers": [],
            "expose_http_server": False,
        },
    },
}


@pytest.fixture(scope="function")
def cluster():
    """A 1-node, 2-GPU virtual cluster (enough for tp/pp up to 2)."""
    virtual_cluster = RayVirtualCluster(
        bundle_ct_per_node_list=[2],
        use_gpus=True,
        max_colocated_worker_groups=2,
        num_gpus_per_node=2,
        name="megatron-generation-test-cluster",
    )
    yield virtual_cluster
    virtual_cluster.shutdown()


@pytest.fixture(scope="function")
def policy_cluster_separate():
    """A dedicated 1-GPU cluster for the training policy in the non-colocated test."""
    virtual_cluster = RayVirtualCluster(
        bundle_ct_per_node_list=[1],
        use_gpus=True,
        max_colocated_worker_groups=1,
        num_gpus_per_node=1,
        name="megatron-generation-test-policy-cluster",
    )
    yield virtual_cluster
    try:
        virtual_cluster.shutdown()
    except Exception as e:
        print(f"Error during policy_cluster_separate shutdown: {e}")


@pytest.fixture(scope="function")
def tokenizer():
    """Initialize tokenizer for the test model."""
    return get_tokenizer({"name": model_name})


@pytest.fixture(scope="function")
def test_input_data(tokenizer):
    """Create test input data for inference."""
    test_prompts = [
        "Hello, my name is",
        "The capital of France is",
    ]
    encodings = tokenizer(
        test_prompts,
        padding="max_length",
        max_length=20,
        truncation=True,
        return_tensors="pt",
        padding_side="right",
    )
    input_lengths = encodings["attention_mask"].sum(dim=1).to(torch.int32)
    return BatchedDataDict(
        {
            "input_ids": encodings["input_ids"],
            "input_lengths": input_lengths,
        }
    )


def _assert_valid_generation_output(outputs, input_data, require_generation=True):
    """Assert the GenerationOutputSpec contract produced by the Megatron worker."""
    required_keys = [
        "output_ids",
        "logprobs",
        "generation_lengths",
        "unpadded_sequence_lengths",
    ]
    for key in required_keys:
        assert key in outputs, f"{key} not found in generation output"

    batch_size = len(input_data["input_ids"])
    assert all(outputs[key].shape[0] == batch_size for key in required_keys), (
        "Wrong batch size in generation output"
    )
    # output_ids and logprobs are packed on the same padded width.
    assert outputs["output_ids"].shape == outputs["logprobs"].shape

    if require_generation:
        assert (outputs["generation_lengths"] > 0).all(), (
            "Some samples generated nothing"
        )

    # length identity: total (unpadded) == prompt length + generated length.
    expected_unpadded = input_data["input_lengths"].cpu().to(torch.int64) + outputs[
        "generation_lengths"
    ].cpu().to(torch.int64)
    assert torch.equal(
        outputs["unpadded_sequence_lengths"].cpu().to(torch.int64), expected_unpadded
    ), "unpadded_sequence_lengths != input_lengths + generation_lengths"

    # logprob offset: position 0 is always the 0.0 placeholder.
    assert torch.allclose(
        outputs["logprobs"][:, 0],
        torch.zeros(batch_size, dtype=outputs["logprobs"].dtype),
    ), "logprobs[:, 0] should be the 0.0 placeholder"


async def _generate_async(mg, tokenizer, test_input_data, greedy=False):
    """Drive ``generate_async`` over single-sample microbatches and reassemble in order."""
    collected = []
    for single_item_input in test_input_data.make_microbatch_iterator(
        microbatch_size=1
    ):
        async for original_idx, single_item_output in mg.generate_async(
            single_item_input, greedy=greedy
        ):
            # The mcore coordinator only accepts requests on DP rank 0.
            assert single_item_output["gen_leader_worker_idx"] == [0]
            collected.append((original_idx, single_item_output))

    collected.sort(key=lambda x: x[0])
    outputs = [item for _, item in collected]
    pad_token_id = mg.cfg.get("_pad_token_id", tokenizer.pad_token_id)
    return BatchedDataDict.from_batches(
        outputs,
        pad_value_dict={"output_ids": pad_token_id, "logprobs": 0.0},
    )


@pytest.mark.mcore
@pytest.mark.timeout(900)
@pytest.mark.parametrize(
    "tensor_parallel_size,pipeline_parallel_size",
    [(1, 1), (2, 1), (1, 2)],
)
def test_megatron_policy_generation(
    cluster, test_input_data, tokenizer, tensor_parallel_size, pipeline_parallel_size
):
    """Standalone Megatron generation across tp/pp."""
    if cluster.num_gpus_per_node < tensor_parallel_size * pipeline_parallel_size:
        pytest.skip(
            f"Need {tensor_parallel_size * pipeline_parallel_size} GPUs for "
            f"tp={tensor_parallel_size} pp={pipeline_parallel_size}"
        )

    config = deepcopy(basic_megatron_test_config)
    config["megatron_cfg"]["tensor_model_parallel_size"] = tensor_parallel_size
    config["megatron_cfg"]["pipeline_model_parallel_size"] = pipeline_parallel_size
    # config-level stop string, unioned with the per-sample stop strings below.
    config["generation"]["stop_strings"] = ["</s>"]

    mg = None
    try:
        mg = MegatronGeneration(config=config, tokenizer=tokenizer, cluster=cluster)

        # greedy decoding: full output contract + non-empty text
        outputs = mg.generate(test_input_data, greedy=True)
        _assert_valid_generation_output(outputs, test_input_data)
        generated_texts = tokenizer.batch_decode(
            outputs["output_ids"], skip_special_tokens=True
        )
        assert all(len(t) > 0 for t in generated_texts), "Some greedy texts are empty"

        # sampling (non-greedy) path still produces a valid contract
        sampled = mg.generate(test_input_data, greedy=False)
        _assert_valid_generation_output(sampled, test_input_data)

        # per-sample stop strings are merged with the config stop string (may stop early,
        # so don't require a generated token)
        data_with_stops = BatchedDataDict(
            {
                "input_ids": test_input_data["input_ids"],
                "input_lengths": test_input_data["input_lengths"],
                "stop_strings": [["."], ["."]],
            }
        )
        stopped = mg.generate(data_with_stops, greedy=True)
        _assert_valid_generation_output(
            stopped, test_input_data, require_generation=False
        )

        # lifecycle: leave generation mode, re-enter, and generate again
        assert mg.finish_generation() is True
        assert mg.prepare_for_generation() is True
        reentered = mg.generate(test_input_data, greedy=True)
        _assert_valid_generation_output(reentered, test_input_data)
    finally:
        if mg is not None:
            mg.shutdown()
        gc.collect()
        torch.cuda.empty_cache()


@pytest.mark.mcore
@pytest.mark.asyncio
@pytest.mark.timeout(900)
async def test_megatron_policy_generation_async(cluster, test_input_data, tokenizer):
    """Standalone Megatron async generation."""
    config = deepcopy(basic_megatron_test_config)
    config["generation"]["mcore_generation_config"]["async_engine"] = True
    mg = None
    try:
        mg = MegatronGeneration(config=config, tokenizer=tokenizer, cluster=cluster)
        outputs = await _generate_async(mg, tokenizer, test_input_data, greedy=True)
        _assert_valid_generation_output(outputs, test_input_data)
        generated_texts = tokenizer.batch_decode(
            outputs["output_ids"], skip_special_tokens=True
        )
        assert all(len(t) > 0 for t in generated_texts), "Some async texts are empty"
    finally:
        if mg is not None:
            mg.shutdown()
        gc.collect()
        torch.cuda.empty_cache()


@pytest.mark.mcore
@pytest.mark.timeout(900)
def test_megatron_generation_colocated(cluster, test_input_data, tokenizer):
    """Colocated Megatron generation: wrap an existing training policy without owning it."""
    config = deepcopy(basic_megatron_test_config)
    config["generation"]["colocated"]["enabled"] = True

    # construction guard: exactly one of `cluster` / `policy` is required
    with pytest.raises(AssertionError):
        MegatronGeneration(config=config, tokenizer=tokenizer)
    with pytest.raises(AssertionError):
        MegatronGeneration(
            config=config, tokenizer=tokenizer, cluster=cluster, policy=object()
        )

    policy = None
    try:
        policy = Policy(cluster=cluster, config=config, tokenizer=tokenizer)
        megatron_cfg_before = deepcopy(config["megatron_cfg"])

        mg = MegatronGeneration(policy=policy, config=config, tokenizer=tokenizer)
        # colocated wrapper reuses the training policy and must not own it
        assert mg._owns_policy is False
        # the colocated path must NOT merge mcore_generation_config into megatron_cfg
        assert "max_tokens" not in config["megatron_cfg"]
        assert config["megatron_cfg"] == megatron_cfg_before

        mg.prepare_for_generation()
        outputs = mg.generate(test_input_data, greedy=True)
        _assert_valid_generation_output(outputs, test_input_data)

        # ownership guard: shutdown is a no-op, so the wrapped policy keeps generating
        assert mg.shutdown() is True
        after_shutdown = mg.generate(test_input_data, greedy=True)
        _assert_valid_generation_output(after_shutdown, test_input_data)
    finally:
        if policy is not None:
            policy.shutdown()
        gc.collect()
        torch.cuda.empty_cache()


@pytest.mark.mcore
@pytest.mark.timeout(900)
def test_megatron_generation_non_colocated_refit(
    policy_cluster_separate, test_input_data, tokenizer
):
    """Non-colocated Megatron generation."""
    generation_cluster = RayVirtualCluster(
        bundle_ct_per_node_list=[1],
        use_gpus=True,
        max_colocated_worker_groups=1,
        num_gpus_per_node=1,
        name="megatron-generation-test-generation-cluster",
    )
    if (
        policy_cluster_separate.num_gpus_per_node < 1
        or generation_cluster.num_gpus_per_node < 1
    ):
        pytest.skip("Need at least two GPUs across separate clusters")

    config = deepcopy(basic_megatron_test_config)

    policy = None
    mg = None
    try:
        policy = Policy(
            cluster=policy_cluster_separate, config=config, tokenizer=tokenizer
        )
        mg = MegatronGeneration(
            config=config, tokenizer=tokenizer, cluster=generation_cluster
        )

        # init the refit collective on both sides.
        ip, port = policy_cluster_separate.get_master_address_and_port()
        train_world_size = policy_cluster_separate.world_size()
        world_size = train_world_size + generation_cluster.world_size()
        refit_backend = config["generation"]["mcore_generation_config"]["refit_backend"]
        futures_train = policy.init_collective_mcore_generation(
            ip, port, world_size, rank_offset=0, refit_backend=refit_backend
        )
        futures_inference = mg.init_collective(
            ip,
            port,
            world_size,
            train_world_size=train_world_size,
            refit_backend=refit_backend,
        )
        ray.get(futures_train + futures_inference)

        # refit the inference engine from the training weights, then generate
        refit_policy_generation(policy, mg, False)
        outputs = mg.generate(test_input_data, greedy=True)
        _assert_valid_generation_output(outputs, test_input_data)
        generated_texts = tokenizer.batch_decode(
            outputs["output_ids"], skip_special_tokens=True
        )
        assert all(len(t) > 0 for t in generated_texts), "Some texts are empty"
    finally:
        if mg is not None:
            mg.shutdown()
        if policy is not None:
            policy.shutdown()
        try:
            generation_cluster.shutdown()
        except Exception as e:
            print(f"Error during generation_cluster shutdown: {e}")
        gc.collect()
        torch.cuda.empty_cache()
