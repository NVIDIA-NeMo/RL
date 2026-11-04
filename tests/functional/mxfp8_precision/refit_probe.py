"""Exercise the production NCCL reshard synchronizer for supported precision pairs."""

import asyncio
import copy
from pathlib import Path
from types import SimpleNamespace

import ray
import torch

from nemo_rl.algorithms.grpo import refit_policy_generation
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.generation.vllm.vllm_generation import VllmGeneration
from nemo_rl.weight_sync.factory import create_weight_synchronizer
from nemo_rl.weight_sync.nccl_reshard_utils import check_nccl_reshard_refit_support
from precision_probe_common import (
    generate_single_sample_rows,
    save_json,
    tensor_error,
    tensor_hash,
)


def configure_refit(policy_cfg, tokenizer, seq_len):
    cfg = copy.deepcopy(policy_cfg)
    g = cfg["generation"]
    g["model_name"] = cfg["model_name"]
    g["backend"] = "vllm"
    g["refit_transport"] = "nccl_reshard"
    g["top_p"], g["top_k"] = 1.0, None
    g["max_new_tokens"] = 16
    g["colocated"] = {
        "enabled": False,
        "resources": {"gpus_per_node": 4, "num_nodes": 1},
    }
    v = g["vllm_cfg"]
    v.update(
        {
            "async_engine": True,
            "tensor_parallel_size": 4,
            "expert_parallel_size": 4,
            "pipeline_parallel_size": 1,
            "max_model_len": max(512, seq_len + 32),
            "gpu_memory_utilization": 0.6,
            "enforce_eager": True,
            "expose_http_server": False,
            "enable_vllm_metrics_logger": False,
            "enable_prefix_caching": False,
        }
    )
    v["http_server_serving_chat_kwargs"] = {}
    g["vllm_kwargs"].update(
        {
            "max_num_seqs": 8,
            "max_num_batched_tokens": max(512, seq_len + 32),
            "enable_chunked_prefill": True,
            "kernel_config": {"enable_flashinfer_autotune": False},
        }
    )
    g["vllm_kwargs"].pop("compilation_config", None)
    # No speculative draft: the trainer's MTP layers remain trained but are
    # not used as an inference drafter in this precision comparison.
    g["vllm_kwargs"].pop("speculative_config", None)
    cfg["generation"] = configure_generation_config(
        g, tokenizer, trains_mtp=bool(cfg["megatron_cfg"].get("mtp_num_layers"))
    )
    return cfg


def run_refit_probe(policy, train_cluster, config, data, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    check_nccl_reshard_refit_support(SimpleNamespace(policy=config))
    state = lambda: ray.get(
        policy.worker_group.run_all_workers_single_data("refit_probe_state")
    )
    before = state()
    expected_native = config["megatron_cfg"]["native_mxfp8_refit"]
    assert len(before) == train_cluster.world_size()
    assert all(v["native_export"] == expected_native for v in before)
    cluster = RayVirtualCluster(
        name="precision-refit-sampler",
        bundle_ct_per_node_list=[4],
        num_gpus_per_node=4,
        use_gpus=True,
        max_colocated_worker_groups=1,
    )
    generation = None
    try:
        generation = VllmGeneration(cluster=cluster, config=config["generation"])
        generation.weight_synchronizer = create_weight_synchronizer(
            policy=policy,
            generation=generation,
            generation_backend="vllm",
            colocated=False,
            train_cluster=train_cluster,
            inference_cluster=cluster,
        )
        generation.weight_synchronizer.init_communicator()
        refit_policy_generation(policy, generation, colocated_inference=False)
        assert generation.prepare_for_generation()
        inputs = BatchedDataDict(
            {
                "input_ids": data["input_ids"],
                "input_lengths": data["input_lengths"],
            }
        )

        async def capture():
            for label in ["first", "repeat"]:
                rows = await generate_single_sample_rows(
                    generation.generate_async,
                    inputs,
                    greedy=True,
                )
                output = BatchedDataDict.from_batches(
                    rows,
                    pad_value_dict={
                        "output_ids": config["generation"]["_pad_token_id"]
                    },
                )
                tensors = {
                    k: v.detach().cpu()
                    for k, v in output.items()
                    if isinstance(v, torch.Tensor)
                }
                assert all(torch.isfinite(v).all() for v in tensors.values())
                torch.save(tensors, folder / (label + ".pt"))

        asyncio.run(capture())
        after = state()
        assert before == after, "Refit changed FP32 masters or trainer expert storage"
        save_json(
            folder / "refit.json",
            {
                "status": "COMPLETED",
                "native_mxfp8_refit": expected_native,
                "trainer_state_unchanged": True,
                "trainer_states": before,
                "synchronizer": type(generation.weight_synchronizer).__name__,
                "greedy": True,
                "request_batch_size": 1,
                "request_execution": "sequential",
                "input_hash": tensor_hash(data["input_ids"]),
            },
        )
    finally:
        if generation is not None:
            assert generation.shutdown(), "Sampler shutdown failed"
        cluster.shutdown()


def compare_refit_outputs(out, labels, atol, rtol):
    results = {}
    pairs = [(mode, mode, "repeat", "first") for mode in labels]
    pairs.extend(
        [
            (labels[1], labels[0], "first", "first"),
            (labels[2], labels[1], "first", "first"),
        ]
    )
    for mode, reference, case, ref_case in pairs:
        a = torch.load(Path(out) / mode / "refit" / (case + ".pt"), weights_only=True)
        b = torch.load(
            Path(out) / reference / "refit" / (ref_case + ".pt"), weights_only=True
        )
        tokens_match = torch.equal(a["output_ids"], b["output_ids"])
        lengths_match = torch.equal(a["generation_lengths"], b["generation_lengths"])
        # Different generated trajectories do not have matched conditional
        # probabilities; retain artifacts instead of interpreting their error.
        logprobs = (
            tensor_error(a["logprobs"], b["logprobs"], atol=atol, rtol=rtol)
            if tokens_match and lengths_match
            else None
        )
        results[f"{mode}/{case}_vs_{reference}/{ref_case}"] = {
            "tokens_match": tokens_match,
            "lengths_match": lengths_match,
            "logprobs": logprobs,
            "pass": tokens_match
            and lengths_match
            and bool(logprobs and logprobs["pass"]),
        }
    return results
