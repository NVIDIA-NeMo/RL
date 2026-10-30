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

"""Colocated vLLM refit from a pipeline-parallel Automodel policy."""

import os
import shutil

import pytest
import torch

from nemo_rl.algorithms.grpo import refit_policy_generation
from nemo_rl.algorithms.loss import NLLLossFn
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.vllm import VllmGeneration
from nemo_rl.models.policy.lm_policy import Policy
from tests.unit.conftest import TEST_ASSETS_DIR

pytestmark = pytest.mark.vllm


@pytest.fixture(scope="module")
def tiny_untied_llama_path():
    """A four-layer Llama with separate embedding and LM head weights."""
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

    model_path = os.path.join(
        TEST_ASSETS_DIR, "tiny_llama_4l_untied_qwen2_tokenizer_vllm"
    )
    torch.manual_seed(0)
    config = LlamaConfig(
        num_hidden_layers=4,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=151936,
        tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(config=config)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B")
    shutil.rmtree(model_path, ignore_errors=True)
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    yield model_path


def _generation_config(model_path: str) -> dict:
    return {
        "backend": "vllm",
        "model_name": model_path,
        "tokenizer": {"name": model_path},
        "dtype": "bfloat16",
        "max_new_tokens": 8,
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
            "gpu_memory_utilization": 0.3,
            "max_model_len": 256,
            "async_engine": False,
            "skip_tokenizer_init": False,
            "load_format": "dummy",
            "enforce_eager": "True",
            "kv_cache_dtype": "auto",
        },
        "colocated": {
            "enabled": True,
            "resources": {"gpus_per_node": None, "num_nodes": None},
        },
        "vllm_kwargs": {},
    }


def _policy_config(model_path: str, generation_config: dict):
    return {
        "model_name": model_path,
        "tokenizer": {"name": model_path},
        "train_global_batch_size": 4,
        "train_micro_batch_size": 1,
        "logprob_batch_size": 1,
        "precision": "bfloat16",
        "offload_optimizer_for_logprob": False,
        "max_total_sequence_length": 64,
        "make_sequence_length_divisible_by": 1,
        "optimizer": {
            "name": "torch.optim.AdamW",
            "kwargs": {"lr": 1e-2, "weight_decay": 0.0, "betas": [0.9, 0.999]},
        },
        "automodel_cfg": {
            "enabled": True,
            "checkpoint": {
                "model_save_format": "safetensors",
                "save_consolidated": "false",
            },
            "cpu_offload": False,
            "sequence_parallel": False,
            "activation_checkpointing": False,
            "tensor_parallel_size": 1,
            "context_parallel_size": 1,
            "pipeline_parallel_size": 2,
            "defer_fsdp_grad_sync": False,
            "custom_parallel_plan": None,
            "automodel_kwargs": {"force_hf": True},
            "pipeline": {"microbatches_per_step": 2},
        },
        "dynamic_batching": {"enabled": False},
        "sequence_packing": {"enabled": False},
        "max_grad_norm": 1.0,
        "generation": generation_config,
    }


@pytest.mark.timeout(900)
def test_colocated_refit_from_pipeline_parallel_policy(tiny_untied_llama_path):
    """After training and refit, vLLM scores its own tokens like the PP policy.

    vLLM starts from dummy weights, so matching logprobs show that every layer of
    every pipeline stage reached each colocated vLLM worker.
    """
    tokenizer = get_tokenizer({"name": tiny_untied_llama_path})
    generation_config = configure_generation_config(
        _generation_config(tiny_untied_llama_path), tokenizer
    )
    cluster = RayVirtualCluster(
        name="pp_refit",
        bundle_ct_per_node_list=[2],
        use_gpus=True,
        num_gpus_per_node=2,
        max_colocated_worker_groups=2,
    )
    vllm_generation = None
    policy = None
    try:
        vllm_generation = VllmGeneration(cluster, generation_config)
        vllm_generation.finish_generation()
        policy = Policy(
            cluster=cluster,
            config=_policy_config(tiny_untied_llama_path, generation_config),
            tokenizer=tokenizer,
            init_reference_model=False,
        )
        vllm_generation.prepare_refit_info(policy.prepare_refit_info())

        # Train once so the refit weights differ from the checkpoint.
        input_ids = torch.randint(0, 1000, (4, 16))
        policy.prepare_for_training()
        policy.train(
            BatchedDataDict(
                {
                    "input_ids": input_ids,
                    "input_lengths": torch.full((4,), 16, dtype=torch.int32),
                    "token_mask": torch.ones(4, 16),
                    "sample_mask": torch.ones(4),
                }
            ),
            NLLLossFn(),
        )
        refit_policy_generation(policy, vllm_generation, colocated_inference=True)

        prompts = BatchedDataDict(
            {
                "input_ids": input_ids[:, :8],
                "input_lengths": torch.full((4,), 8, dtype=torch.int32),
            }
        )
        generated = vllm_generation.generate(prompts, greedy=True)
        vllm_generation.finish_generation()

        policy.prepare_for_lp_inference()
        policy_logprobs = policy.get_logprobs(
            BatchedDataDict(
                {
                    "input_ids": generated["output_ids"],
                    "input_lengths": generated["unpadded_sequence_lengths"],
                }
            )
        )["logprobs"]
        mask = torch.zeros_like(generated["logprobs"], dtype=torch.bool)
        for row, total_len in enumerate(generated["unpadded_sequence_lengths"]):
            mask[row, 8:total_len] = True
        abs_diff = (generated["logprobs"] - policy_logprobs).abs().masked_select(mask)
        assert abs_diff.numel() > 0
        assert torch.exp(abs_diff).mean() <= 1.05, abs_diff
    finally:
        if vllm_generation is not None:
            vllm_generation.shutdown()
        if policy is not None:
            policy.shutdown()
        cluster.shutdown()
