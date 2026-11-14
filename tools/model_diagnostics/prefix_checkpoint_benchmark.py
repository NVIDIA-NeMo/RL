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
"""Four-node generation/TQ proxy for PR #5176; no trainer or Gym agent graph."""

from __future__ import annotations

import argparse
import asyncio
import copy
import importlib.metadata
import json
import math
import pickle
import secrets
import subprocess
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import aiohttp

if TYPE_CHECKING:
    from nemo_rl.data_plane import DataPlaneConfig
    from nemo_rl.models.generation.vllm.config import VllmConfig

G_WORKER_FQN = "tools.model_diagnostics.prefix_checkpoint_benchmark_worker.PrefixCheckpointBenchmarkWorker"
G_WORKER_PYTHON = "/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python"


@dataclass(frozen=True)
class Call:
    identity: str
    admitted_at: float


def inventories(calls: list[Call], *, checkpoint_id: str, count: int) -> list[Any]:
    """Split distinct calls into nonempty, disjoint production inventories."""
    # Gym is optional for standalone plan inspection.
    from nemo_gym._checkpoint.generation_cut import (
        GenerationCutInventory,
        GenerationCutPrefix,
    )

    if (
        count < 1
        or len(calls) < count
        or len({call.identity for call in calls}) != len(calls)
    ):
        raise ValueError(
            "Need unique admitted calls and at least one call per inventory"
        )
    return [
        GenerationCutInventory.build(
            checkpoint_id=checkpoint_id,
            server_name=f"benchmark-model-{index}",
            active_prefixes=[
                GenerationCutPrefix(
                    ticket_id=call.identity,
                    rollout_id=call.identity,
                    attempt=0,
                    model_call_id=call.identity,
                    admitted_at=call.admitted_at,
                )
                for call in calls[index::count]
            ],
        )
        for index in range(count)
    ]


def worker_call(generation: Any, method: str, **kwargs: Any) -> Any:
    # Ray is unnecessary for inventory protocol tests.
    import ray

    result = ray.get(
        generation.worker_group.run_all_workers_single_data(
            method,
            run_rank_0_only_axes=["tensor_parallel", "pipeline_parallel"],
            **kwargs,
        ),
        timeout=360,
    )
    if len(result) != 1:
        raise RuntimeError(f"Expected one model owner, received {len(result)}")
    return result[0]


def verify_saved_prefixes(root: Path, keys: list[str], fields: list[str]) -> None:
    """Check all referenced columns in a trusted, locally produced TQ dump."""
    if (
        json.loads((root / "metadata.json").read_text()).get("storage_saved")
        is not True
    ):
        raise RuntimeError("Storage checkpoint has no completion marker")
    with (root / "controller_state.pkl").open("rb") as stream:
        state = pickle.load(stream)
    partition = state["partitions"]["rollout_staging"]
    missing = set(keys) - partition.keys_mapping.keys()
    if missing:
        raise RuntimeError(
            f"Saved controller lacks {len(missing)} referenced prefix keys"
        )
    required = {partition.keys_mapping[key] for key in keys}
    found: dict[str, set[int]] = {field: set() for field in fields}
    for shard in (root / "simple_storage").glob("su_*.pkl"):
        with shard.open("rb") as stream:
            stored = pickle.load(stream)
        for field in fields:
            found[field].update(stored["field_data"].get(field, {}).keys() & required)
    for field, indices in found.items():
        if indices != required:
            raise RuntimeError(
                f"Saved payload lacks {len(required - indices)} rows of {field}"
            )


class Storage:
    """Host real SimpleStorage and verify the persisted cut's referenced rows."""

    def __init__(
        self, config: dict[str, Any], node_id: str, *, router_replay: bool
    ) -> None:
        # TQ/Ray are optional outside a cluster run.
        from nemo_rl.data_plane import build_data_plane_client
        from nemo_rl.data_plane.schema import ROUTED_EXPERTS_FIELD
        from nemo_rl.data_plane.tq_token_sink import STAGING_FIELDS

        self.fields = list(STAGING_FIELDS)
        if router_replay:
            self.fields.append(ROUTED_EXPERTS_FIELD)

        self.client = build_data_plane_client(
            cast("DataPlaneConfig", config),
            checkpointing=True,
            storage_unit_node_ids=[node_id] * config["simple"]["num_storage_units"],
        )

    def ready(self) -> bool:
        return True

    def save(self, path: str, keys: list[str]) -> dict[str, Any]:
        started = time.perf_counter()
        self.client.save_checkpoint(path, metadata={"benchmark": "pr5176"})
        elapsed = time.perf_counter() - started
        root = Path(path)
        verify_saved_prefixes(root, keys, self.fields)
        return {
            "persist_s": elapsed,
            "save_and_verify_s": time.perf_counter() - started,
            "referenced_keys": len(set(keys)),
            "bytes": sum(
                item.stat().st_size for item in root.rglob("*") if item.is_file()
            ),
        }


class Load:
    """Keep captured OpenAI calls outstanding, using the original proxy prompts."""

    def __init__(
        self, session: aiohttp.ClientSession, generation: Any, name: str, args: Any
    ) -> None:
        self.session, self.generation, self.name, self.args = (
            session,
            generation,
            name,
            args,
        )
        self.active: dict[str, Call] = {}
        self.tasks: list[asyncio.Task[None]] = []
        self.allowed = asyncio.Event()
        self.allowed.set()
        self.next_id = self.completed = self.errors = 0
        self.first_errors: list[str] = []

    async def loop(self) -> None:
        while True:
            await self.allowed.wait()
            self.next_id += 1
            identity = f"{self.name}-call-{self.next_id}"
            self.active[identity] = Call(identity, time.time())
            prompt = (
                f"Problem {self.next_id}: Derive and explain mathematical identities, "
                "checking each step carefully. Continue with numbered examples.\n"
                + "Consider algebra, probability, geometry, and number theory. "
                * 80
            )
            payload = {
                "model": self.args.model,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 1.0,
                "top_p": 1.0,
                "max_tokens": self.args.max_output_tokens,
                "ignore_eos": True,
                "stream": False,
                "logprobs": True,
                "top_logprobs": 0,
                "return_tokens_as_token_ids": True,
                "chat_template_kwargs": {
                    "enable_thinking": True,
                    "truncate_history_thinking": False,
                },
                "ng_capture": {
                    "rollout_id": identity,
                    "model_call_id": identity,
                    "mode": "text",
                },
            }
            try:
                async with self.session.post(
                    self.generation.dp_openai_server_base_urls[0].rstrip("/")
                    + "/chat/completions",
                    json=payload,
                    headers={
                        "Authorization": f"Bearer {self.generation.http_server_api_key}"
                    },
                    timeout=aiohttp.ClientTimeout(total=7200, connect=60),
                ) as response:
                    if response.status != 200:
                        raise RuntimeError(
                            f"HTTP {response.status}: {(await response.text())[:300]}"
                        )
                    body = await response.json()
                    if body["ng_commit_coords"]["disposition"] != "staged":
                        raise RuntimeError("Terminal capture was not staged")
                    self.completed += 1
            except (aiohttp.ClientError, TimeoutError, RuntimeError) as error:
                self.errors += 1
                if len(self.first_errors) < 5:
                    self.first_errors.append(str(error))
                await asyncio.sleep(1)
            finally:
                self.active.pop(identity, None)

    def healthy(self) -> None:
        for task in self.tasks:
            if task.done():
                task.result()
                raise RuntimeError("Generation load exited unexpectedly")
        if self.errors > max(10, self.completed):
            raise RuntimeError(f"Generation load is failing: {self.first_errors}")


async def cut_request(
    session: Any, url: str, inventory: Any, token: str, timeout_s: float
) -> dict[str, Any]:
    # Gym is optional for the driver's import-only preflight.
    from nemo_gym._checkpoint.generation_cut import (
        GENERATION_CUT_ROUTE,
        GenerationCutReceipt,
    )

    started = time.perf_counter()
    try:
        async with session.post(
            url.rstrip("/").removesuffix("/v1") + GENERATION_CUT_ROUTE,
            json=inventory.model_dump(mode="json"),
            headers={"Authorization": f"Bearer {token}"},
            timeout=aiohttp.ClientTimeout(total=timeout_s + 30, connect=30),
        ) as response:
            if response.status != 200:
                raise RuntimeError(
                    f"Cut HTTP {response.status}: {(await response.text())[:300]}"
                )
            receipt = GenerationCutReceipt.model_validate(await response.json())
            receipt.validate_for(inventory)
            return {
                "elapsed_s": time.perf_counter() - started,
                "receipt": receipt.model_dump(mode="json"),
            }
    except (aiohttp.ClientError, TimeoutError, RuntimeError) as error:
        return {"elapsed_s": time.perf_counter() - started, "error": str(error)}


async def exercise(
    generations: list[Any], storage: Any, args: Any, token: str, run: Any
) -> None:
    # Ray is optional for protocol tests.
    import ray

    output = Path(args.output)
    rows = []
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=16384, limit_per_host=8192)
    ) as session:
        loads = [
            Load(session, generation, f"engine-{i}", args)
            for i, generation in enumerate(generations)
        ]
        for load in loads:
            load.tasks = [
                asyncio.create_task(load.loop()) for _ in range(args.outstanding)
            ]
        try:
            for index, (target, new_target) in enumerate(
                zip(args.token_targets, args.new_token_targets, strict=True)
            ):
                batch_sizes = (
                    args.batch_sizes
                    if not args.counterbalance or index % 2 == 0
                    else list(reversed(args.batch_sizes))
                )
                for generation, size in zip(generations, batch_sizes, strict=True):
                    await asyncio.to_thread(
                        worker_call, generation, "benchmark_batch_size", size=size
                    )
                    await asyncio.to_thread(
                        generation.flush_obsolete_generation_cut_rows,
                        index + 1,
                        timeout_s=120,
                    )
                deadline = time.monotonic() + args.target_wait_s
                while True:
                    for load in loads:
                        load.healthy()
                    volumes = await asyncio.gather(
                        *(
                            asyncio.to_thread(
                                worker_call,
                                generation,
                                "benchmark_token_volume",
                                call_ids=list(load.active),
                            )
                            for generation, load in zip(generations, loads, strict=True)
                        )
                    )
                    print(
                        json.dumps(
                            {
                                "event": "token_progress",
                                "round": index,
                                "volumes": volumes,
                                "target": target,
                                "new_target": new_target,
                            }
                        ),
                        flush=True,
                    )
                    if all(
                        value["generated_tokens"] >= target
                        and value["new_positions"] >= new_target
                        for value in volumes
                    ):
                        break
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Token targets were not reached")
                    await asyncio.sleep(10)
                started = time.perf_counter()
                for load in loads:
                    load.allowed.clear()
                checkpoint_id = f"benchmark-round-{index}"
                groups = []
                for generation, load in zip(generations, loads, strict=True):
                    present = await asyncio.to_thread(
                        worker_call,
                        generation,
                        "benchmark_calls",
                        call_ids=list(load.active),
                    )
                    calls = [
                        load.active[identity]
                        for identity in present
                        if identity in load.active
                    ]
                    groups.append(
                        inventories(
                            calls, checkpoint_id=checkpoint_id, count=args.inventories
                        )
                    )
                before_cut = time.perf_counter()
                receipts = await asyncio.gather(
                    *(
                        asyncio.gather(
                            *(
                                cut_request(
                                    session,
                                    generation.dp_openai_server_base_urls[0],
                                    inventory,
                                    token,
                                    args.cut_timeout_s,
                                )
                                for inventory in group
                            )
                        )
                        for generation, group in zip(generations, groups, strict=True)
                    )
                )
                cut_s = time.perf_counter() - before_cut
                drain_started = time.perf_counter()
                diagnostics = await asyncio.gather(
                    *(
                        asyncio.to_thread(worker_call, generation, "benchmark_drain")
                        for generation in generations
                    )
                )
                drain_s = time.perf_counter() - drain_started
                (output / f"receipts-{index}.json").write_text(json.dumps(receipts))
                (output / f"diagnostics-{index}.json").write_text(
                    json.dumps(diagnostics, indent=2)
                )
                keys = [
                    key
                    for group in receipts
                    for item in group
                    for ack in item.get("receipt", {}).get("prefixes", [])
                    if ack["disposition"] == "durable_prefix"
                    for key in ack["staging_keys"]
                ]
                if not keys:
                    raise RuntimeError("No durable prefixes; not a successful snapshot")
                if any("error" in item for group in receipts for item in group):
                    raise RuntimeError(
                        "Cut transport failed; see receipts and diagnostics"
                    )
                save_started = time.perf_counter()
                saved = await asyncio.to_thread(
                    ray.get,
                    storage.save.remote(str(output / f"snapshot-{index}"), keys),
                )
                row: dict[str, Any] = {
                    "round": index,
                    "batch_sizes": batch_sizes,
                    "token_targets": [target, new_target],
                    "target_observed": volumes,
                    "cut_http_wait_s": cut_s,
                    "writer_drain_s": drain_s,
                    "tq_save_and_verify_s": time.perf_counter() - save_started,
                    "total_s": time.perf_counter() - started,
                    "storage": saved,
                    "workers": [],
                }
                for group, events, load in zip(
                    receipts, diagnostics, loads, strict=True
                ):
                    dispositions = Counter(
                        ack["disposition"]
                        for item in group
                        for ack in item.get("receipt", {}).get("prefixes", [])
                    )
                    batches = [
                        event
                        for event in events["events"]
                        if event["phase"] == "prefix_stage_batch"
                    ]
                    row["workers"].append(
                        {
                            "dispositions": dict(dispositions),
                            "transport_failures": sum(
                                "error" in item for item in group
                            ),
                            "stage_batch_calls": len(batches),
                            "staged_positions": sum(
                                event["token_positions"]
                                for event in batches
                                if event["success"]
                            ),
                            "stage_wall_s": sum(
                                event["elapsed_s"] for event in batches
                            ),
                            "stage_thread_cpu_s": sum(
                                event["thread_cpu_s"] for event in batches
                            ),
                            "progress_counts": events["progress_counts"],
                            "load_completed": load.completed,
                            "load_errors": load.errors,
                            "first_load_errors": list(load.first_errors),
                            "pipeline_cumulative": events["pipeline_cumulative"],
                            "writer_cumulative": events["writer_cumulative"],
                            "inventory_queue_s": [
                                event["queue_s"]
                                for event in events["events"]
                                if event["phase"] == "inventory"
                            ],
                            "prepare_wall_s": sum(
                                event["prepare_wall_s"]
                                for event in events["events"]
                                if event["phase"] == "inventory"
                            ),
                            "prepare_thread_cpu_s": sum(
                                event["thread_cpu_s"]
                                for event in events["events"]
                                if event["phase"] == "inventory"
                            ),
                        }
                    )
                rows.append(row)
                (output / "summary.json").write_text(
                    json.dumps({"complete": False, "rounds": rows}, indent=2)
                )
                print(json.dumps({"event": "snapshot", **row}), flush=True)
                if run is not None:
                    for engine, metrics in enumerate(row["workers"]):
                        run.log(
                            {
                                f"engine_{engine}/{name}": value
                                for name, value in metrics.items()
                                if isinstance(value, (int, float))
                            },
                            step=index,
                            commit=False,
                        )
                    run.log(
                        {
                            "snapshot/total_s": row["total_s"],
                            "snapshot/cut_http_wait_s": cut_s,
                            "snapshot/tq_persist_s": saved["persist_s"],
                        },
                        step=index,
                    )
                for load in loads:
                    load.allowed.set()
            (output / "summary.json").write_text(
                json.dumps({"complete": True, "rounds": rows}, indent=2)
            )
        finally:
            for load in loads:
                for task in load.tasks:
                    task.cancel()
            await asyncio.gather(
                *(task for load in loads for task in load.tasks), return_exceptions=True
            )


def generation_config(args: Any) -> VllmConfig:
    # Project config loading is optional for protocol tests.
    from omegaconf import OmegaConf
    from nemo_rl.utils.config import load_config

    config = load_config("examples/configs/grpo_math_1B.yaml")
    config.policy.model_name = args.model
    config.policy.precision = "bfloat16"
    config.policy.max_total_sequence_length = 131072
    config.policy.megatron_cfg.fp32_lm_head = False
    generation = cast(
        dict[str, Any], OmegaConf.to_container(config.policy.generation, resolve=True)
    )
    generation["model_name"] = args.model
    generation["worker_extension_cls_fqn"] = G_WORKER_FQN
    generation["colocated"]["enabled"] = False
    generation["colocated"]["resources"]["num_nodes"] = 1
    generation["vllm_cfg"].update(
        tensor_parallel_size=4,
        async_engine=True,
        expose_http_server=True,
        http_server_api_key_required=True,
        gpu_memory_utilization=0.8,
        load_format="auto",
        flat_logprobs=True,
        output_loop_yield_budget_ms=20,
        fp32_lm_head=False,
        enable_return_routed_experts=args.router_replay,
    )
    generation["vllm_kwargs"].update(
        max_num_seqs=args.max_num_seqs,
        enable_return_routed_experts=args.router_replay,
        trust_remote_code=True,
        enable_prefix_caching=False,
    )
    generation["vllm_cfg"].update(prefix_options(args.mode))
    generation["_mtp_weights_from_refit"] = False
    return cast("VllmConfig", generation)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Validate the workload before importing Ray or vLLM."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batch-sizes", nargs=2, type=int, default=[256, 256])
    parser.add_argument("--outstanding", type=int, default=630)
    parser.add_argument("--inventories", type=int, default=39)
    parser.add_argument("--storage-units", type=int, default=16)
    parser.add_argument(
        "--token-targets", nargs="+", type=int, default=[2450792, 8912407]
    )
    parser.add_argument(
        "--new-token-targets", nargs="+", type=int, default=[0, 6461615]
    )
    parser.add_argument("--max-output-tokens", type=int, default=65536)
    parser.add_argument("--target-wait-s", type=float, default=1800)
    parser.add_argument("--cut-timeout-s", type=float, default=300)
    parser.add_argument(
        "--router-replay", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--mode", choices=MODES, default="writer")
    parser.add_argument(
        "--counterbalance",
        action="store_true",
        help="Reverse per-engine batch sizes on alternate cuts",
    )
    parser.add_argument("--max-num-seqs", type=int, default=512)
    parser.add_argument("--worker-python", default=G_WORKER_PYTHON)
    parser.add_argument(
        "--project", help="Enable W&B in this project; otherwise disabled"
    )
    parser.add_argument("--entity")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print a plan without production imports or actors",
    )
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    if len(args.token_targets) != len(args.new_token_targets):
        parser.error("Token-target lists must have equal lengths")
    positive = [
        *args.batch_sizes,
        args.outstanding,
        args.inventories,
        args.storage_units,
        args.max_output_tokens,
        args.max_num_seqs,
        *args.token_targets,
        args.target_wait_s,
        args.cut_timeout_s,
    ]
    if any(not math.isfinite(value) or value <= 0 for value in positive):
        parser.error("Counts, token targets and timeouts must be finite and positive")
    if any(value < 0 for value in args.new_token_targets):
        parser.error("New-token targets must be nonnegative")
    if args.inventories > args.outstanding:
        parser.error("Inventories cannot exceed outstanding calls per engine")
    if args.max_output_tokens >= 131072:
        parser.error("Output budget must leave room for the prompt within 131072")
    return args


MODES = ("baseline", "encoder", "pipeline", "writer")


def prefix_options(mode: str) -> dict[str, Any]:
    """Select only #5176's supported production configuration."""
    if mode not in MODES:
        raise ValueError(f"Unknown benchmark mode: {mode}")
    return {
        "prefix_writer_actor": mode == "writer",
        "prefix_write_pipeline": {
            "enabled": mode in ("pipeline", "writer"),
            "direct_column_encoding": mode != "baseline",
        },
    }


def main() -> None:
    args = parse_args()
    if args.dry_run:
        print(
            json.dumps(
                vars(args) | {"prefix_options": prefix_options(args.mode)}, indent=2
            )
        )
        return
    # Heavy production imports run in the image, not on the submit host.
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy
    from nemo_rl.distributed.ray_actor_environment_registry import (
        ACTOR_ENVIRONMENT_REGISTRY,
    )
    from nemo_rl.distributed.virtual_cluster import RayVirtualCluster, init_ray
    from nemo_rl.models.generation.vllm.vllm_generation import VllmGeneration
    from nemo_rl.models.generation.vllm.config import VllmConfig
    from tools.model_diagnostics.prefix_checkpoint_benchmark_worker import (
        PrefixCheckpointBenchmarkWorker,
    )

    config = generation_config(args)
    missing_keys = VllmConfig.__required_keys__ - config.keys()
    if missing_keys:
        raise ValueError(f"Generation configuration lacks {sorted(missing_keys)}")
    versions = {}
    for name in ("ray", "vllm", "torch", "transferqueue", "tensordict", "nemo-lens"):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "distribution metadata unavailable"
    print(
        json.dumps(
            {
                "event": "preflight",
                "versions": versions,
                "model": args.model,
                "worker": str(PrefixCheckpointBenchmarkWorker),
            }
        ),
        flush=True,
    )
    if not Path(args.model, "config.json").is_file():
        raise FileNotFoundError(f"Model config missing: {args.model}")
    if args.preflight:
        return
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=False)
    plan = vars(args) | {
        "versions": versions,
        "pr_head": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "fence": False,
        "scope": "prefix-cut proxy, not a restorable Gym/trainer snapshot",
        "gym_head": subprocess.check_output(
            ["git", "-C", "3rdparty/Gym-workspace/Gym", "rev-parse", "HEAD"], text=True
        ).strip(),
        "prefix_options": prefix_options(args.mode),
        "control_threads_per_engine": 1,
        "counterbalanced": args.counterbalance,
    }
    (output / "plan.json").write_text(json.dumps(plan, indent=2))
    ACTOR_ENVIRONMENT_REGISTRY[G_WORKER_FQN] = args.worker_python
    init_ray()
    nodes = sorted(
        [
            node
            for node in ray.nodes()
            if node["Alive"] and node["Resources"].get("GPU", 0) >= 4
        ],
        key=lambda node: node["NodeManagerAddress"],
    )
    if len(nodes) != 4:
        raise RuntimeError(f"Expected four GPU nodes, found {len(nodes)}")
    remaining = [
        node
        for node in nodes
        if node["NodeManagerAddress"] != ray.util.get_node_ip_address()
    ]
    if len(remaining) != 3:
        raise RuntimeError("Driver must run on one of the four allocation nodes")
    dp_config = {
        "enabled": True,
        "impl": "transfer_queue",
        "backend": "simple",
        "claim_meta_poll_interval_s": 0.5,
        "simple": {
            "num_storage_units": args.storage_units,
            "storage_capacity": 1000000,
            "storage_unit_placement": None,
        },
    }
    storage = (
        ray.remote(num_cpus=1)(Storage)
        .options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(
                remaining[2]["NodeID"], soft=False
            )
        )
        .remote(dp_config, remaining[2]["NodeID"], router_replay=args.router_replay)
    )
    ray.get(storage.ready.remote(), timeout=180)
    token = secrets.token_hex(32)
    generations = []
    run = None
    if args.project:
        # W&B is optional; use the caller's endpoint and credentials unchanged.
        import wandb

        run = wandb.init(
            entity=args.entity,
            project=args.project,
            name=output.parent.name,
            config=plan,
            dir=str(output),
        )

    def create(item: tuple[int, dict[str, Any]]) -> Any:
        index, node = item
        cluster = RayVirtualCluster(
            [4],
            num_gpus_per_node=4,
            name=f"pr5176_engine_{index}",
            node_resource_constraints=[{f"node:{node['NodeManagerAddress']}": 0.001}],
        )
        generation = VllmGeneration(
            cluster, copy.deepcopy(config), name_prefix=f"pr5176_engine_{index}"
        )
        try:
            generation.setup_token_capture(
                cast("DataPlaneConfig", dp_config),
                "rollout_staging",
                generation_prefix_cuts_enabled=True,
                generation_cut_control_token=token,
                generation_cut_control_timeout_s=args.cut_timeout_s,
                generation_prefix_batch_size=args.batch_sizes[index],
                generation_prefix_batch_max_tokens=4194304,
            )
        except Exception:
            generation.shutdown()
            raise
        return generation

    success = False
    try:
        with ThreadPoolExecutor(max_workers=2) as pool:
            futures = {
                pool.submit(create, item): item[0] for item in enumerate(remaining[:2])
            }
            created = {}
            errors = []
            for future in as_completed(futures):
                try:
                    generation = future.result()
                    generations.append(generation)
                    created[futures[future]] = generation
                except Exception as error:
                    errors.append(error)
            if errors:
                raise errors[0]
            generations = [created[index] for index in sorted(created)]
        asyncio.run(exercise(generations, storage, args, token, run))
        success = True
    finally:
        for generation in generations:
            generation.shutdown()
        if run is not None:
            run.finish(exit_code=0 if success else 1)
        ray.shutdown()


if __name__ == "__main__":
    main()
