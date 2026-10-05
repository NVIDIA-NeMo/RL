# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-node TP4 engine/cache and real SWE smoke; this is not a training E2E."""

import argparse
import asyncio
import copy
import json
import os
import signal
import socket
from pathlib import Path

import ray
import torch
import uvicorn
from omegaconf import OmegaConf
from tensorrt_llm import AsyncLLM, SamplingParams
from tensorrt_llm.conversation_params import ConversationParams
from tensorrt_llm.executor.ray.utils import control_action_decorator
from tensorrt_llm.llmapi.llm_args import (
    CapacitySchedulerPolicy,
    KvCacheConfig,
    SchedulerConfig,
)
from tensorrt_llm.llmapi.rlhf_utils import WorkerExtension

from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.environments.nemo_gym import build_nemo_gym_actors
from nemo_rl.models.generation.trtllm.trtllm_http_server import create_app
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

RECIPE = (
    "examples/configs/recipes/llm/grpo-nano3.5-swe-32n4g-tp4cp16-async-trtllm.v1.yaml"
)


class ProbeExtension(WorkerExtension):
    """Inspect the actual executor settings and retained conversation state."""

    @control_action_decorator
    def inspect_nano_cache(self) -> dict:
        torch.cuda.synchronize()
        engine = self.engine.model_engine
        args = engine.llm_args
        manager = self.engine.kv_cache_manager
        return {
            "rank": torch.distributed.get_rank(),
            "allreduce_strategy": args.allreduce_strategy,
            "prefill_cuda_graph_backend": engine.prefill_cuda_graph_backend,
            "graph_batch_sizes": args.cuda_graph_config.batch_sizes,
            "manager": type(manager).__name__,
            "reuse_epoch": manager._reuse_epoch,
            "open_cache_ids": sorted(manager.kv_cache_map),
            "block_reuse_policy": manager.block_reuse_policy.value,
            "conversations": {
                name: {
                    "current_request_id": state.current_request_id,
                    "retained_turns": len(state.planned_drop_handles),
                }
                for name, state in manager.conversation_manager._conversation_states.items()
            },
        }


async def preflight(output_dir: Path, image_sha256: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=False)
    # Preserve partial evidence and shut down cleanly if the scheduler's
    # validation window ends during model setup or the SWE trajectory.
    asyncio.get_running_loop().add_signal_handler(
        signal.SIGTERM, asyncio.current_task().cancel, "validation window ended"
    )
    job_id = os.environ.get("NANO35_VALIDATION_JOB_ID", os.environ.get("SLURM_JOB_ID"))
    # TRT-LLM owns the Ray/MPI namespace in this single-node probe.
    for key in list(os.environ):
        if key.startswith(("PMI_", "PMIX_", "MPI_", "OMPI_", "SLURM_")):
            os.environ.pop(key)
    register_omegaconf_resolvers()
    config = OmegaConf.to_container(load_config(RECIPE), resolve=True)
    generation = config["policy"]["generation"]
    trt = generation["trtllm_cfg"]
    extra = copy.deepcopy(generation["trtllm_kwargs"])
    kv = KvCacheConfig(**extra.pop("kv_cache_config"))
    tokenizer = get_tokenizer(config["policy"]["tokenizer"])
    model = config["policy"]["model_name"]
    # Several runtime dependencies have their own top-level `tools` package.
    # Import our extension through the explicit helper package instead.
    helper_path = str(Path(__file__).resolve().parent.parent)
    worker_pythonpath = os.pathsep.join(
        value for value in (helper_path, os.environ.get("PYTHONPATH")) if value
    )
    # TRT creates its own actor runtime_env from os.environ, replacing Ray's
    # inherited runtime_env; both paths must see the helper package.
    os.environ["PYTHONPATH"] = worker_pythonpath
    ray.init(
        num_gpus=4,
        num_cpus=32,
        resources={"worker_units": 1},
        include_dashboard=False,
        runtime_env={"env_vars": {"PYTHONPATH": worker_pythonpath}},
    )
    llm = AsyncLLM(
        model=model,
        backend="pytorch",
        tensor_parallel_size=trt["tensor_parallel_size"],
        moe_tensor_parallel_size=trt["moe_tensor_parallel_size"],
        moe_expert_parallel_size=trt["moe_expert_parallel_size"],
        dtype=trt["precision"],
        max_seq_len=trt["max_model_len"],
        max_input_len=trt["max_model_len"],
        max_batch_size=trt["max_batch_size"],
        max_num_tokens=trt["max_num_tokens"],
        trust_remote_code=True,
        ray_worker_extension_cls="nano35.gpu_preflight.ProbeExtension",
        kv_cache_config=kv,
        scheduler_config=SchedulerConfig(
            capacity_scheduler_policy=CapacitySchedulerPolicy.MAX_UTILIZATION
        ),
        **extra,
    )
    gym = None
    server = None
    server_task = None
    report = {
        "status": "running",
        "scope": "One-node inference/cache/SWE smoke; no optimizer or training update",
        "job_id": job_id,
        "image_sha256": image_sha256,
        "checks": {},
    }

    def save_report() -> None:
        (output_dir / "gpu-preflight.json").write_text(
            json.dumps(report, indent=2) + "\n"
        )

    save_report()
    try:
        await llm.setup_async()
        before = await llm.collective_rpc("inspect_nano_cache")
        assert len(before) == 4
        for rank in before:
            assert rank["allreduce_strategy"] == "MNNVL", rank
            assert rank["prefill_cuda_graph_backend"] == "breakable", rank
            assert rank["block_reuse_policy"] == "per_conversation", rank
        report["engine_before"] = before
        report["checks"]["mnnvl_breakable"] = "passed"
        save_report()

        sampling = SamplingParams(
            temperature=0, max_tokens=8, logprobs=5, return_perf_metrics=True
        )
        conversation = ConversationParams(
            conversation_id="nano35-preflight-conversation"
        )
        prompt = tokenizer.encode(
            "The purpose of this Nano cache verification is to check a long reusable prefix. "
            * 48
        )
        first = await llm.generate_async(
            {"prompt_token_ids": prompt}, sampling, conversation_params=conversation
        )
        assert first.outputs[0].token_ids
        second_prompt = (
            prompt
            + list(first.outputs[0].token_ids)
            + tokenizer.encode("\nPlease continue briefly.")
        )
        warm = await llm.generate_async(
            {"prompt_token_ids": second_prompt},
            sampling,
            conversation_params=conversation,
        )
        warm_metrics = warm.outputs[0].request_perf_metrics
        assert warm_metrics is not None
        assert warm_metrics.kv_cache_metrics.num_reused_blocks > 0
        retained = await llm.collective_rpc("inspect_nano_cache")
        for rank in retained:
            assert (
                rank["conversations"][conversation.conversation_id]["retained_turns"]
                > 0
            )
        report["checks"]["tp4_generation"] = "passed"
        report["checks"]["conversation_reuse"] = "passed"
        report["warm_reused_blocks"] = warm_metrics.kv_cache_metrics.num_reused_blocks
        save_report()

        # The previous image crashed here because CUDA graph dummy handles
        # remained live. Verify both successful reset and invalidation of reuse.
        await llm.collective_rpc("reset_prefix_cache")
        after = await llm.collective_rpc("inspect_nano_cache")
        assert all(rank["reuse_epoch"] == 1 for rank in after)
        cold = await llm.generate_async(
            {"prompt_token_ids": second_prompt},
            sampling,
            conversation_params=conversation,
        )
        assert (
            cold.outputs[0].request_perf_metrics.kv_cache_metrics.num_reused_blocks == 0
        )

        # BF16 greedy generation is not guaranteed bitwise deterministic: the
        # no-refit control in job 3221645 produced a tied top score and a
        # different token. Use a fixed-size fresh-conversation cold cohort,
        # preserving every output, rather than a single stochastic comparison.
        def generation_evidence(result) -> dict:
            completion = result.outputs[0]
            return {
                "token_ids": list(completion.token_ids),
                "text": tokenizer.decode(completion.token_ids),
                "logprobs": [
                    {str(token): value.logprob for token, value in position.items()}
                    for position in completion.logprobs
                ],
                "reused_blocks": completion.request_perf_metrics.kv_cache_metrics.num_reused_blocks,
            }

        comparison = {
            "warm": generation_evidence(warm),
            "cold_after_reset": generation_evidence(cold),
            "cold_controls": [],
            "control_count": 16,
            "criterion": "Exact eight-token warm output must also occur with zero reused blocks in a fixed-size cold cohort",
        }
        report["cache_output_comparison"] = comparison
        report["engine_after_reset"] = after
        save_report()
        for index in range(comparison["control_count"]):
            control = await llm.generate_async(
                {"prompt_token_ids": second_prompt},
                sampling,
                conversation_params=ConversationParams(
                    conversation_id=f"nano35-preflight-cold-control-{index}"
                ),
            )
            evidence = generation_evidence(control)
            comparison["cold_controls"].append(evidence)
            save_report()
            assert evidence["reused_blocks"] == 0, evidence
        cold_cohort = [comparison["cold_after_reset"], *comparison["cold_controls"]]
        comparison["warm_matching_cold_indices"] = [
            index
            for index, item in enumerate(cold_cohort)
            if item["token_ids"] == comparison["warm"]["token_ids"]
        ]
        save_report()
        assert comparison["warm_matching_cold_indices"], comparison
        report["engine_after_reset"] = after
        report["checks"]["cache_reset"] = "passed"
        save_report()

        app = create_app(
            llm=llm,
            tokenizer=tokenizer,
            model_name=model,
            max_seq_len=trt["max_model_len"],
            sampling_config={
                key: generation[key] for key in ("temperature", "top_p", "top_k")
            },
            tool_parser=trt["tool_parser"],
            reasoning_parser=trt["reasoning_parser"],
            default_chat_template_kwargs=trt["default_chat_template_kwargs"],
        )
        listener = socket.socket()
        listener.bind(("0.0.0.0", 0))
        port = listener.getsockname()[1]
        server = uvicorn.Server(
            uvicorn.Config(app, host="0.0.0.0", port=port, log_level="info")
        )
        server_task = asyncio.create_task(server.serve(sockets=[listener]))
        for _ in range(100):
            if server.started:
                break
            if server_task.done():
                await server_task
                raise RuntimeError("HTTP adapter exited before becoming ready")
            await asyncio.sleep(0.1)
        assert server.started
        gym_config = copy.deepcopy(config["env"])
        gym_config["nemo_gym"]["results_dir"] = str(output_dir / "gym-results")
        # Run one real row with the recipe's agent and evaluation limits.
        gym = await asyncio.to_thread(
            build_nemo_gym_actors,
            gym_config,
            base_urls=[f"http://{ray.util.get_node_ip_address()}:{port}/v1"],
            model_name=model,
            tokenizer=tokenizer,
            enable_router_replay=False,
            use_fastokens=False,
        )
        with Path(os.environ["NANO35_DATA"]).open() as data:
            row = json.loads(next(line for line in data if line.strip()))
        row["_rowidx"] = 0
        count = 0
        async for result_ref in gym.sole_handle().run_rollouts.remote(
            [row], "preflight"
        ):
            row_index, route, result, timings = await result_ref
            full = result["full_result"]
            (output_dir / "swe-result.json").write_text(
                json.dumps(full, indent=2) + "\n"
            )
            assert row_index == 0 and route["name"] == "swe_agents_train"
            assert result["message_log"]
            assert full["reward"] in (0, 1)
            assert full.get("resolved") is not None
            for flag in (
                "agent_error_kind",
                "agent_timed_out",
                "eval_timed_out",
                "oom_killed",
                "eval_oom_killed",
                "mask_sample",
            ):
                assert not full.get(flag), (flag, full.get(flag))
            report["swe"] = {
                "instance_id": row["instance_id"],
                "reward": full["reward"],
                "timings": timings,
            }
            count += 1
        assert count == 1
        report["checks"]["swe_agent_and_evaluation"] = "passed"
        report["status"] = "passed"
    except BaseException as error:
        report["status"] = (
            "interrupted" if isinstance(error, asyncio.CancelledError) else "failed"
        )
        report["error"] = repr(error)
        raise
    finally:
        save_report()
        if gym is not None:
            await asyncio.to_thread(gym.shutdown)
        if server is not None:
            server.should_exit = True
        if server_task is not None:
            await server_task
        llm.shutdown()
        ray.shutdown()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--image-sha256", required=True)
    args = parser.parse_args()
    asyncio.run(preflight(args.output_dir, args.image_sha256))
