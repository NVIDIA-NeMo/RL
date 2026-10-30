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

"""vllm_cfg.fp32_lm_head yields fp32 logits from vLLM.

With logprobs_mode="raw_logits" vLLM returns the sampled token's raw logit. A
bf16 head can only produce bf16-representable values; an fp32 head almost
never does. Each model runs with the flag off (control) and on.
"""

import argparse
import tempfile

import ray
import torch

from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster, init_ray
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.vllm import VllmConfig, VllmGeneration

PROMPTS = ["The capital of France is", "Water boils at a temperature of"]
MAX_NEW_TOKENS = 8
# Nemotron-H's Mamba state needs fp32 and eager compilation in this small setup.
EXTRA_VLLM_KWARGS = {
    "nvidia/NVIDIA-Nemotron-3-Nano-4B-BF16": {
        "mamba_ssm_cache_dtype": "float32",
        "compilation_config": {"backend": "eager"},
    },
}


def sampled_raw_logits(model_name: str, fp32_lm_head: bool) -> torch.Tensor:
    config: VllmConfig = {
        "backend": "vllm",
        "model_name": model_name,
        "tokenizer": {"name": model_name},
        "max_new_tokens": MAX_NEW_TOKENS,
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
            "gpu_memory_utilization": 0.6,
            "max_model_len": 256,
            "async_engine": False,
            "skip_tokenizer_init": False,
            "load_format": "auto",
            "enforce_eager": True,
            "kv_cache_dtype": "auto",
            "fp32_lm_head": fp32_lm_head,
            "logprobs_mode": "raw_logits",
            "use_tqdm": False,
        },
        "vllm_kwargs": dict(EXTRA_VLLM_KWARGS.get(model_name, {})),
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
    encoded = [tokenizer(p, add_special_tokens=False)["input_ids"] for p in PROMPTS]
    lengths = torch.tensor([len(ids) for ids in encoded])
    input_ids = torch.full(
        (len(encoded), int(lengths.max())), config["_pad_token_id"], dtype=torch.long
    )
    for i, ids in enumerate(encoded):
        input_ids[i, : len(ids)] = torch.tensor(ids)

    with tempfile.TemporaryDirectory(prefix="nrl-ray-", dir="/tmp") as ray_log_dir:
        init_ray(log_dir=ray_log_dir)
        cluster = RayVirtualCluster(
            bundle_ct_per_node_list=[1],
            use_gpus=True,
            max_colocated_worker_groups=1,
            num_gpus_per_node=1,
            name="vllm-fp32-lm-head-functional",
        )
        vllm_generation = None
        try:
            vllm_generation = VllmGeneration(cluster, config)
            output = vllm_generation.generate(
                BatchedDataDict({"input_ids": input_ids, "input_lengths": lengths}),
                greedy=True,
            )
        finally:
            if vllm_generation is not None:
                vllm_generation.shutdown()
            cluster.shutdown()
            ray.shutdown()

    # Generated tokens' values start right after each unpadded prompt.
    values = [
        output["logprobs"][i, int(start) : int(start) + int(n)]
        for i, (start, n) in enumerate(zip(lengths, output["generation_lengths"]))
    ]
    return torch.cat(values)


def on_bf16_grid(values: torch.Tensor) -> torch.Tensor:
    return values == values.to(torch.bfloat16).to(values.dtype)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    model_name = parser.parse_args().model

    baseline = sampled_raw_logits(model_name, fp32_lm_head=False)
    fp32 = sampled_raw_logits(model_name, fp32_lm_head=True)
    assert baseline.numel() > 0 and fp32.numel() > 0
    # The control shows the check can tell the two heads apart.
    assert on_bf16_grid(baseline).all(), f"bf16 head off the bf16 grid: {baseline}"
    off_grid = int((~on_bf16_grid(fp32)).sum())
    assert off_grid >= fp32.numel() // 2, f"fp32 head looks bf16-rounded: {fp32}"
    print(
        f"[PASS] {model_name} fp32 lm_head: {off_grid}/{fp32.numel()} sampled "
        "raw logits off the bf16 grid (bf16 head: 0)"
    )


if __name__ == "__main__":
    main()
