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

from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster, init_ray
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.vllm import VllmConfig, VllmGeneration

MODEL_NAME = "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16"


def main() -> None:
    config: VllmConfig = {
        "backend": "vllm",
        "model_name": MODEL_NAME,
        "tokenizer": {"name": MODEL_NAME},
        "max_new_tokens": 4,
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": None,
        "val_temperature": 1.0,
        "val_top_p": 1.0,
        "val_top_k": None,
        "stop_token_ids": None,
        "stop_strings": None,
        "vllm_cfg": {
            "precision": "bfloat16",
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "expert_parallel_size": 1,
            "gpu_memory_utilization": 0.8,
            "max_model_len": 256,
            "async_engine": False,
            "skip_tokenizer_init": False,
            "load_format": "auto",
            "enforce_eager": True,
            "kv_cache_dtype": "auto",
            "fp32_lm_head": True,
            "use_tqdm": False,
        },
        "vllm_kwargs": {
            "mamba_ssm_cache_dtype": "float32",
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
            name="vllm-nemotron-h-fp32-lm-head-functional",
        )
        vllm_generation = None
        try:
            vllm_generation = VllmGeneration(cluster, config)
            output = vllm_generation.generate_text(
                BatchedDataDict({"prompts": ["The capital of France is"]}),
                greedy=True,
            )
            texts = output["texts"]
            assert len(texts) == 1
            assert texts[0], "Nemotron-H vLLM generation returned an empty string"
            print(f"[PASS] Nemotron-H fp32 lm_head generated text: {texts[0]!r}")
        finally:
            if vllm_generation is not None:
                vllm_generation.shutdown()
            cluster.shutdown()
            ray.shutdown()


if __name__ == "__main__":
    main()
