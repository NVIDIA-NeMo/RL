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
from copy import deepcopy

import pytest
import torch

from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.vllm import VllmConfig, VllmGeneration

# Large model configuration
model_name = "meta-llama/Llama-3.1-70B"

# Define basic vLLM test config for large model
large_model_vllm_config: VllmConfig = {
    "backend": "vllm",
    "model_name": model_name,
    "tokenizer": {
        "name": model_name,
    },
    "dtype": "bfloat16",
    "max_new_tokens": 5,
    "temperature": 0.8,
    "top_p": 1.0,
    "top_k": None,
    "stop_token_ids": None,
    "stop_strings": None,
    "vllm_cfg": {
        "precision": "bfloat16",
        "tensor_parallel_size": 8,
        "pipeline_parallel_size": 2,
        "gpu_memory_utilization": 0.7,
        "max_model_len": 1024,
        "async_engine": True,
        "skip_tokenizer_init": False,
        "load_format": "auto",
    },
    "colocated": {
        "enabled": True,
        "resources": {
            "gpus_per_node": None,
            "num_nodes": None,
        },
    },
    "vllm_kwargs": {},
}


@pytest.fixture(scope="module", autouse=True)
def skip_tied_weight_check():
    """Automatically skip tied weight check for all tests in this module."""
    os.environ["NRL_SKIP_TIED_WEIGHT_CHECK"] = "1"
    yield
    os.environ.pop("NRL_SKIP_TIED_WEIGHT_CHECK", None)


@pytest.fixture(scope="function")
def two_node_cluster():
    """Create a virtual cluster with 2 nodes for testing large models."""
    # Create a cluster with 2 nodes, each with 8 GPUs
    virtual_cluster = RayVirtualCluster(
        bundle_ct_per_node_list=[8, 8],  # 2 nodes with 8 GPUs each
        use_gpus=True,
        max_colocated_worker_groups=2,
        num_gpus_per_node=8,
        name="vllm-large-model-test-cluster",
    )
    yield virtual_cluster
    virtual_cluster.shutdown()


@pytest.fixture(scope="function")
def tokenizer():
    """Initialize tokenizer for the test model."""
    tokenizer = get_tokenizer({"name": model_name})
    return tokenizer


@pytest.fixture(scope="function")
def test_input_data(tokenizer):
    """Create test input data for inference."""
    test_prompts = [
        "Hello, my name is",
        "The capital of France is",
    ]

    # Tokenize prompts
    encodings = tokenizer(
        test_prompts,
        padding="max_length",
        max_length=20,
        truncation=True,
        return_tensors="pt",
        padding_side="right",
    )

    # Calculate input lengths from attention mask
    input_lengths = encodings["attention_mask"].sum(dim=1).to(torch.int32)

    # Create input data dictionary
    return BatchedDataDict(
        {
            "input_ids": encodings["input_ids"],
            "input_lengths": input_lengths,
        }
    )


# skip this test for now
@pytest.mark.skip(reason="Skipping large model test until we have resources in CI.")
@pytest.mark.asyncio
@pytest.mark.parametrize("tensor_parallel_size", [4, 8])
@pytest.mark.parametrize("pipeline_parallel_size", [2])
async def test_vllm_large_model(
    two_node_cluster,
    test_input_data,
    tokenizer,
    tensor_parallel_size,
    pipeline_parallel_size,
):
    """Test vLLM policy async generation capabilities with large model across 2 nodes."""
    # Check if we have enough nodes
    import ray

    nodes = ray.nodes()
    alive_nodes = [node for node in nodes if node["Alive"]]
    if len(alive_nodes) < 2:
        pytest.skip("Test requires at least 2 nodes with GPUs")

    async_policy = None
    try:
        # Configure vLLM for large model
        vllm_config = deepcopy(large_model_vllm_config)
        vllm_config["vllm_cfg"]["async_engine"] = True
        vllm_config["vllm_cfg"]["tensor_parallel_size"] = tensor_parallel_size
        vllm_config["vllm_cfg"]["pipeline_parallel_size"] = pipeline_parallel_size
        vllm_config = configure_generation_config(vllm_config, tokenizer)

        print(
            f"Creating vLLM policy with TP={tensor_parallel_size}, PP={pipeline_parallel_size}"
        )
        print(f"Total GPUs required: {tensor_parallel_size * pipeline_parallel_size}")

        async_policy = VllmGeneration(two_node_cluster, vllm_config)

        print("Running async generation...")
        collected_indexed_outputs = []
        # generate_async is restricted to handle only single samples
        input_generator = test_input_data.make_microbatch_iterator(microbatch_size=1)
        for single_item_input in input_generator:
            async for original_idx, single_item_output in async_policy.generate_async(
                single_item_input
            ):
                collected_indexed_outputs.append((original_idx, single_item_output))

        # Sort by original_idx to ensure order matches generation_input_data
        collected_indexed_outputs.sort(key=lambda x: x[0])

        # Extract in correct order
        outputs = [item for _, item in collected_indexed_outputs]
        pad_token_id = async_policy.cfg.get("pad_token_id", tokenizer.pad_token_id)
        outputs = BatchedDataDict.from_batches(
            outputs,
            pad_value_dict={"output_ids": pad_token_id, "logprobs": 0.0},
        )

        # Validate outputs format

        assert "output_ids" in outputs, "output_ids not found in generation output"
        assert "logprobs" in outputs, "logprobs not found in generation output"
        assert "generation_lengths" in outputs, (
            "generation_lengths not found in generation output"
        )
        assert "unpadded_sequence_lengths" in outputs, (
            "unpadded_sequence_lengths not found in generation output"
        )

        # Validate outputs shape and content
        assert outputs["output_ids"].shape[0] == len(test_input_data["input_ids"]), (
            "Wrong batch size in output"
        )
        assert outputs["generation_lengths"].shape[0] == len(
            test_input_data["input_ids"]
        ), "Wrong batch size in generation_lengths"

        # Decode and check outputs
        generated_sequences = outputs["output_ids"]
        generated_texts = tokenizer.batch_decode(
            generated_sequences, skip_special_tokens=True
        )

        print(f"Generated texts: {generated_texts}")

        # All texts should have a non-zero length
        assert all(len(text) > 0 for text in generated_texts), (
            "Some generated texts are empty"
        )

        print("Large model test completed successfully!")

    finally:
        # Clean up resources
        print("Cleaning up resources...")
        if async_policy:
            async_policy.shutdown()

        # Force garbage collection
        import gc

        gc.collect()
        torch.cuda.empty_cache()
