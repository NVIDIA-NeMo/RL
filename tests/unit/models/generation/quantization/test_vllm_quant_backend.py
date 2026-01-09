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


@pytest.fixture(scope="function")
def cluster():
    """Create a virtual cluster for testing."""
    # Create a cluster with 1 node that has 2 GPU bundles (one for vLLM, one for Megatron)
    virtual_cluster = RayVirtualCluster(
        bundle_ct_per_node_list=[2],
        use_gpus=True,
        max_colocated_worker_groups=2,
        num_gpus_per_node=2,
        name="vllm-quant-test-cluster",
    )
    yield virtual_cluster
    virtual_cluster.shutdown()


@pytest.mark.skipif(
    torch.cuda.device_count() == 0,
    reason="CUDA is required for vLLM quant integration test",
)
def test_vllm_quant_refit_loads_amax(cluster):
    """Integration-style test: quantized Megatron -> vLLM refit should load amax buffers."""
    # Require modelopt/NVFP4
    try:
        import modelopt.torch.quantization as mtq  # noqa: F401
    except Exception:
        pytest.skip("modelopt not available; skipping quant integration test")

    # Hopper+ recommended for NVFP4
    major, _ = torch.cuda.get_device_capability()
    if major < 9:
        pytest.skip("NVFP4 quant test requires compute capability >= 9.0")

    model_name = "Qwen/Qwen2.5-0.5B"

    tokenizer = get_tokenizer({"name": model_name})

    # vLLM config with quantization enabled
    vllm_config = {
        "backend": "vllm",
        "model_name": model_name,
        "tokenizer": {"name": model_name},
        "dtype": "bfloat16",
        "max_new_tokens": 4,
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": None,
        "stop_token_ids": None,
        "stop_strings": None,
        "quant_cfg": "NVFP4_DEFAULT_CFG",
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
    vllm_config = configure_generation_config(
        copy.deepcopy(vllm_config), tokenizer, is_eval=True
    )

    # Megatron config with quantization enabled
    megatron_config = get_basic_megatron_test_config(tp=1, pp=1, precision="bfloat16")
    megatron_config["model_name"] = model_name
    megatron_config["tokenizer"]["name"] = model_name
    megatron_config["quant_cfg"] = "NVFP4_DEFAULT_CFG"
    megatron_config["generation"] = copy.deepcopy(vllm_config)

    vllm_policy = None
    megatron_policy = None
    try:
        # Create policies
        vllm_policy = VllmGeneration(cluster, vllm_config)
        vllm_policy.finish_generation()
        futures = vllm_policy.worker_group.run_all_workers_single_data("export_amax")
        amax_list = ray.get(futures)
        assert amax_list, "No amax buffers returned from vLLM workers"
        for amax_dict in amax_list:
            assert amax_dict, "Amax dict is empty"
            for name, buf in amax_dict.items():
                assert torch.all(buf < 0), (
                    f"amax buffer {name} should be negative before refit"
                )

        megatron_policy = Policy(cluster, megatron_config, tokenizer)

        # Prepare refit info and refit weights
        state_dict_info = megatron_policy.prepare_refit_info()
        vllm_policy.prepare_refit_info(state_dict_info)
        refit_policy_generation(
            megatron_policy, vllm_policy, vllm_config["colocated"]["enabled"]
        )

        # Collect amax buffers from rollout workers
        futures = vllm_policy.worker_group.run_all_workers_single_data("export_amax")
        amax_list = ray.get(futures)

        assert amax_list, "No amax buffers returned from rollout workers"
        for amax_dict in amax_list:
            assert amax_dict, "Amax dict is empty"
            for name, buf in amax_dict.items():
                assert torch.all(buf >= 0), (
                    f"amax buffer {name} should be non-negative after refit"
                )
    finally:
        if vllm_policy:
            vllm_policy.shutdown()
        if megatron_policy:
            megatron_policy.shutdown()
