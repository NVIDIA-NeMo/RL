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

import tempfile

import ray
import torch

from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster, init_ray
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.vllm import VllmConfig, VllmGeneration

# Load a serialized FP8 checkpoint with valid block scales before any policy refit.
MODEL_NAME = "Qwen/Qwen3-0.6B-FP8"
MAX_NEW_TOKENS = 8


def main() -> None:
    config: VllmConfig = {
        "backend": "vllm",
        "model_name": MODEL_NAME,
        "tokenizer": {"name": MODEL_NAME},
        "max_new_tokens": MAX_NEW_TOKENS,
        "ignore_eos": True,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": None,
        "val_temperature": 1.0,
        "val_top_p": 1.0,
        "val_top_k": None,
        "stop_token_ids": None,
        "stop_strings": None,
        "vllm_cfg": {
            "precision": "fp8",
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "expert_parallel_size": 1,
            "gpu_memory_utilization": 0.8,
            "max_model_len": 256,
            "async_engine": False,
            "skip_tokenizer_init": False,
            "load_format": "auto",
            "enforce_eager": True,
            "kv_cache_dtype": "fp8_per_token_head",
            "use_tqdm": False,
        },
        "vllm_kwargs": {
            "attention_backend": "TRITON_ATTN",
            "max_num_batched_tokens": 256,
            "max_num_seqs": 2,
            "kv_cache_memory_bytes": 64 * 1024 * 1024,
            "compilation_config": {"backend": "eager"},
        },
        "colocated": {
            "enabled": True,
            "resources": {
                "gpus_per_node": None,
                "num_nodes": None,
            },
        },
    }

    tokenizer = get_tokenizer(config["tokenizer"])
    config = configure_generation_config(config, tokenizer, is_eval=True)
    with tempfile.TemporaryDirectory(prefix="nrl-ray-", dir="/tmp") as ray_log_dir:
        init_ray(log_dir=ray_log_dir)
        cluster = RayVirtualCluster(
            bundle_ct_per_node_list=[1],
            use_gpus=True,
            max_colocated_worker_groups=1,
            num_gpus_per_node=1,
            name="vllm-fp8-per-token-head-functional",
        )
        vllm_generation = None
        try:
            vllm_generation = VllmGeneration(cluster, config)
            assert not vllm_generation.requires_kv_scale_sync
            prompts = ["The capital of France is", "Count from one to ten:"]
            tokenized = tokenizer(
                prompts, padding=True, padding_side="right", return_tensors="pt"
            )
            inputs = BatchedDataDict(
                {
                    "input_ids": tokenized["input_ids"],
                    "input_lengths": tokenized["attention_mask"]
                    .sum(dim=1)
                    .to(torch.int32),
                }
            )
            # Repeated multi-token decode exercises cache writes and attention reads.
            for _ in range(2):
                output = vllm_generation.generate(inputs, greedy=True)
                assert output["generation_lengths"].tolist() == [MAX_NEW_TOKENS] * len(
                    prompts
                )
                for index, input_length in enumerate(inputs["input_lengths"].tolist()):
                    generated_logprobs = output["logprobs"][
                        index, input_length : input_length + MAX_NEW_TOKENS
                    ]
                    assert generated_logprobs.numel() == MAX_NEW_TOKENS
                    assert torch.isfinite(generated_logprobs).all(), (
                        f"Non-finite generated logprobs: {generated_logprobs}; "
                        f"output_ids={output['output_ids'][index]}"
                    )
            print("[PASS] fp8_per_token_head multi-token generation")
        finally:
            if vllm_generation is not None:
                vllm_generation.shutdown()
            cluster.shutdown()
            ray.shutdown()


if __name__ == "__main__":
    main()
