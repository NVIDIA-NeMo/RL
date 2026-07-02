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
import re
from pathlib import Path

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
_QUANT_CFG_DIR = Path(__file__).resolve().parents[4] / "examples/modelopt/quant_configs"

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


def _make_vllm_config(tokenizer, quant_cfg, *, is_eval=True):
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
        "quant_cfg": quant_cfg,
        "vllm_cfg": {
            "precision": "bfloat16",
            "tensor_parallel_size": 1,
            "pipeline_parallel_size": 1,
            "expert_parallel_size": 1,
            "gpu_memory_utilization": 0.8,
            "max_model_len": 512,
            "async_engine": False,
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


def _make_megatron_config(vllm_config, quant_cfg):
    cfg = get_basic_megatron_test_config(tp=1, pp=1, precision="bfloat16")
    cfg["model_name"] = _MODEL_NAME
    cfg["tokenizer"]["name"] = _MODEL_NAME
    cfg["quant_cfg"] = quant_cfg
    cfg["quant_calib_size"] = 1
    cfg["quant_calib_data"] = "random"
    cfg["quant_batch_size"] = 1
    cfg["quant_sequence_length"] = 128
    cfg["generation"] = vllm_config
    return cfg


def _kv_amax_by_layer(stats):
    result = {}
    for name, amax in stats["kv_amax"].items():
        match = re.search(r"(?:^|\.)layers\.(\d+)\..*\.([kv])_bmm_quantizer$", name)
        assert match is not None, f"unexpected K/V quantizer name: {name}"
        result[(int(match.group(1)), match.group(2))] = amax
    return result


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
@pytest.mark.parametrize("recipe", ["kv_cache_fp8.yaml", "kv_cache_nvfp4.yaml"])
def test_vllm_quant_refit(cluster, recipe, monkeypatch):
    """Calibrated simulated K/V state must match after Megatron-to-vLLM refit."""
    monkeypatch.setenv("ENABLE_BRIDGE_QUANT_MAPPING", "1")
    quant_cfg = str((_QUANT_CFG_DIR / recipe).resolve())
    tokenizer = get_tokenizer({"name": _MODEL_NAME})
    vllm_config = _make_vllm_config(tokenizer, quant_cfg)
    megatron_config = _make_megatron_config(vllm_config, quant_cfg)

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

        vllm_policy.finish_generation()

        megatron_policy = Policy(cluster, megatron_config, tokenizer)

        # Megatron quantizers should be calibrated (amax > 0) after init
        futures = megatron_policy.worker_group.run_all_workers_single_data(
            "get_quantizer_stats"
        )
        policy_stats = ray.get(futures)
        for rank, stats in enumerate(policy_stats):
            assert stats["enabled"] > 0, f"Megatron rank {rank}: no enabled quantizers"
            assert stats["positive_amax"] == stats["with_amax"], (
                f"Megatron rank {rank}: {stats['with_amax'] - stats['positive_amax']} quantizers have non-positive amax"
            )
            assert stats["kv_amax"], f"Megatron rank {rank}: no K/V amax state"

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
        rollout_stats = ray.get(futures)
        for rank, stats in enumerate(rollout_stats):
            assert stats["positive_amax"] > 0, (
                f"vLLM rank {rank}: expected positive amax after refit, got {stats['positive_amax']}"
            )
            assert stats["kv_amax"], f"vLLM rank {rank}: no K/V amax state"

        policy_kv_amax = _kv_amax_by_layer(policy_stats[0])
        rollout_kv_amax = _kv_amax_by_layer(rollout_stats[0])
        assert rollout_kv_amax.keys() == policy_kv_amax.keys()
        for key, expected in policy_kv_amax.items():
            torch.testing.assert_close(
                rollout_kv_amax[key], expected, rtol=1e-2, atol=1e-3
            )
    finally:
        if vllm_policy:
            vllm_policy.shutdown()
        if megatron_policy:
            megatron_policy.shutdown()
