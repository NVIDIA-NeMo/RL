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

import copy

import pytest
import ray
import torch

from nemo_rl.algorithms.grpo import refit_policy_generation
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.vllm import VllmGeneration
from nemo_rl.models.policy.lm_policy import Policy
from tests.unit.models.generation.test_vllm_generation import (
    get_basic_megatron_test_config,
)

_MODEL_NAME = "Qwen/Qwen3-0.6B"
_QUANT_CFG = "FP8_DEFAULT_CFG"
_PROBE_WEIGHT = "model.layers.0.mlp.down_proj.weight"

_CUDA_AVAILABLE = torch.cuda.device_count() > 0

try:
    import modelopt.torch.quantization as mtq  # noqa: F401

    _MODELOPT_AVAILABLE = True
except ImportError:
    _MODELOPT_AVAILABLE = False

requires_quant = pytest.mark.skipif(
    not (_CUDA_AVAILABLE and _MODELOPT_AVAILABLE),
    reason="Requires CUDA and modelopt",
)


def _make_vllm_config(tokenizer, *, async_engine=False, is_eval=True):
    cfg = {
        "backend": "vllm",
        "model_name": _MODEL_NAME,
        "tokenizer": {"name": _MODEL_NAME},
        "dtype": "bfloat16",
        "max_new_tokens": 4,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": None,
        "stop_token_ids": None,
        "stop_strings": None,
        "quant_cfg": _QUANT_CFG,
        "vllm_cfg": {
            "precision": "bfloat16",
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "expert_parallel_size": 1,
            "gpu_memory_utilization": 0.8,
            "max_model_len": 512,
            "async_engine": async_engine,
            "skip_tokenizer_init": False,
            "load_format": "auto",
            "enforce_eager": "False",
        },
        "colocated": {
            "enabled": True,
            "resources": {"gpus_per_node": None, "num_nodes": None},
        },
        "vllm_kwargs": {},
    }
    return configure_generation_config(copy.deepcopy(cfg), tokenizer, is_eval=is_eval)


def _make_megatron_config(vllm_config):
    cfg = get_basic_megatron_test_config(tp=1, pp=1, precision="bfloat16")
    cfg["model_name"] = _MODEL_NAME
    cfg["tokenizer"]["name"] = _MODEL_NAME
    cfg["quant_cfg"] = _QUANT_CFG
    cfg["quant_calib_size"] = 1
    cfg["quant_calib_data"] = "random"
    cfg["quant_batch_size"] = 1
    cfg["quant_sequence_length"] = 128
    cfg["generation"] = vllm_config
    return cfg


@pytest.fixture(scope="function")
def cluster():
    """Create a virtual cluster for testing."""
    virtual_cluster = RayVirtualCluster(
        bundle_ct_per_node_list=[2],
        use_gpus=True,
        max_colocated_worker_groups=2,
        num_gpus_per_node=2,
        name="vllm-quant-test-cluster",
    )
    yield virtual_cluster
    virtual_cluster.shutdown()


@requires_quant
@pytest.mark.parametrize("async_engine", [False, True], ids=["sync", "async"])
def test_vllm_quant_refit(cluster, async_engine):
    """Integration test: quantized Megatron -> vLLM refit should transfer pre-folded weights.

    Uses is_eval=True so vLLM loads real HF weights at init, allowing us
    to verify that pre-folded weights (with FP8 quantization applied on
    the Megatron side) differ from the original HF weights by a small
    quantization error.
    """
    tokenizer = get_tokenizer({"name": _MODEL_NAME})
    vllm_config = _make_vllm_config(tokenizer, async_engine=async_engine)
    megatron_config = _make_megatron_config(vllm_config)

    vllm_policy = None
    megatron_policy = None
    try:
        vllm_policy = VllmGeneration(cluster, vllm_config)

        # Before refit: vLLM quantizers should have no positive amax (uncalibrated)
        futures = vllm_policy.worker_group.run_all_workers_single_data(
            "get_quantizer_stats"
        )
        for rank, stats in enumerate(ray.get(futures)):
            assert stats["total"] > 0, f"vLLM rank {rank}: no quantizers found"
            assert stats["positive_amax"] == 0, (
                f"vLLM rank {rank}: expected 0 positive amax before refit, got {stats['positive_amax']}"
            )

        # Snapshot weight before refit (original HF weights, loaded via is_eval=True)
        futures = vllm_policy.worker_group.run_all_workers_single_data(
            "get_weight_snapshot", name=_PROBE_WEIGHT
        )
        weight_before = ray.get(futures)[0]
        assert weight_before is not None, f"Could not read {_PROBE_WEIGHT} from vLLM"

        vllm_policy.finish_generation()

        megatron_policy = Policy(cluster, megatron_config, tokenizer)

        # Megatron quantizers should be calibrated (amax > 0) after init
        futures = megatron_policy.worker_group.run_all_workers_single_data(
            "get_quantizer_stats"
        )
        for rank, stats in enumerate(ray.get(futures)):
            assert stats["enabled"] > 0, f"Megatron rank {rank}: no enabled quantizers"
            assert stats["positive_amax"] == stats["with_amax"], (
                f"Megatron rank {rank}: {stats['with_amax'] - stats['positive_amax']} quantizers have non-positive amax"
            )

        # Refit: transfer pre-folded weights + input_quantizer amax from Megatron to vLLM
        state_dict_info = megatron_policy.prepare_refit_info()
        vllm_policy.prepare_refit_info(state_dict_info)
        refit_policy_generation(
            megatron_policy, vllm_policy, vllm_config["colocated"]["enabled"]
        )

        # After refit: vLLM quantizers should now have positive amax
        futures = vllm_policy.worker_group.run_all_workers_single_data(
            "get_quantizer_stats"
        )
        for rank, stats in enumerate(ray.get(futures)):
            assert stats["positive_amax"] > 0, (
                f"vLLM rank {rank}: expected positive amax after refit, got {stats['positive_amax']}"
            )

        # Verify pre-folded weights differ from original HF weights (FP8 quantization error)
        futures = vllm_policy.worker_group.run_all_workers_single_data(
            "get_weight_snapshot", name=_PROBE_WEIGHT
        )
        weight_after = ray.get(futures)[0]

        assert not torch.equal(weight_before, weight_after), (
            "Weights are bit-identical before and after refit -- pre-folding had no effect"
        )
        max_diff = (weight_before - weight_after).abs().max().item()
        print(f"refit weight max abs diff: {max_diff:.6e}")
        assert torch.allclose(weight_before, weight_after, atol=0.1, rtol=0.0), (
            f"Weights diverged too much after refit (max diff={max_diff:.6e}), "
            "expected small FP8 quantization error"
        )
    finally:
        if vllm_policy:
            vllm_policy.shutdown()
        if megatron_policy:
            megatron_policy.shutdown()
