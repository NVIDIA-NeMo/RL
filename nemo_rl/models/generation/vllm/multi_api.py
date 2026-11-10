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

from __future__ import annotations

import asyncio
import concurrent.futures
import copy
import multiprocessing
import os
import socket
import threading
import time
from multiprocessing.connection import Connection
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.v1.metrics.reader import Counter, Gauge

    from nemo_rl.models.generation.vllm.config import VllmConfig


def receive_reply(connection: Connection, timeout: float = 600) -> Any:
    """Read a child acknowledgement and raise on timeout or reported failure.

    Args:
        connection: Parent end of the frontend control pipe.
        timeout: Maximum seconds to wait for a reply.

    Returns:
        The payload from a successful acknowledgement.
    """
    if not connection.poll(timeout):
        raise TimeoutError("Timed out waiting for an HTTP API process")
    status, value = connection.recv()
    if status != "ok":
        raise RuntimeError(f"HTTP API process failed: {value}")
    return value


class MultiAPIServerPool:
    """Manage child HTTP frontends and their shared vLLM engine.

    The owning frontend stays in the worker process. The pool owns child
    processes and the shared engine manager; the caller owns its AsyncLLM client.
    """

    def __init__(self, owner: Any, count: int) -> None:
        """Initialize a pool whose frontend count includes the owning worker."""
        self.owner = owner
        self.count = count
        self.children: list[Any] = []
        self.connections: list[Connection] = []
        self.engine_manager: Any = None
        self.coordinator: Any = None
        self._command_lock = asyncio.Lock()

    def create_engine(self, engine_args: Any, stat_loggers: list[Any]) -> Any:
        """Launch one shared engine and attach its frontend clients.

        Args:
            engine_args: vLLM arguments updated with the frontend count and rank.
            stat_loggers: Additional logger factories for the owning client.

        Returns:
            The owning frontend's AsyncLLM client attached to the shared engine.
        """
        # vLLM and Ray are optional outside generation workers.
        import ray
        from prometheus_client import values
        from vllm.v1.engine.async_llm import AsyncLLM
        from vllm.v1.engine.utils import get_engine_zmq_addresses, launch_core_engines
        from vllm.v1.executor.abstract import Executor
        from vllm.v1.metrics.prometheus import setup_multiprocess_prometheus

        engine_args._api_process_count = self.count
        engine_args._api_process_rank = -1
        config = engine_args.create_engine_config()
        parallel = config.parallel_config
        if parallel.data_parallel_size != 1 or parallel.data_parallel_backend == "ray":
            raise ValueError("HTTP API pooling currently requires local vLLM DP=1")
        addresses = get_engine_zmq_addresses(config, self.count)
        if any(not address.startswith("ipc://") for address in addresses.inputs):
            raise ValueError("HTTP API pooling requires colocated IPC engine clients")

        setup_multiprocess_prometheus()
        # The owning Ray actor imported prometheus before selecting multi-process
        # serving. New collectors must use the same mmap backend as spawned clients.
        values.ValueClass = values.MultiProcessValue()
        executor = Executor.get_class(config)
        # Engine tensor queues and frontend processes must share the spawn
        # context. Never fork a Ray/CUDA worker's live threads.
        os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
        context = multiprocessing.get_context("spawn")
        namespace = ray.get_runtime_context().namespace
        parent_result: concurrent.futures.Future[Any] = concurrent.futures.Future()
        parent_thread: threading.Thread | None = None
        try:
            with launch_core_engines(
                config, executor, not engine_args.disable_log_stats, addresses
            ) as launch:
                self.engine_manager = launch.engine_manager
                self.coordinator = launch.coordinator
                for index in range(1, self.count):
                    parent, child = context.Pipe()
                    client_addresses = {
                        "input_address": addresses.inputs[index],
                        "output_address": addresses.outputs[index],
                        "tensor_queue": launch.tensor_queue,
                    }
                    process = context.Process(
                        target=run_api_process,
                        name=f"NRL-HTTP-{index}",
                        args=(
                            child,
                            self.owner.cfg,
                            engine_args,
                            config,
                            client_addresses,
                        ),
                        kwargs={
                            "count": self.count,
                            "index": index,
                            "listener": self.owner._reserved_socket,
                            "node_ip": self.owner._reserved_node_ip,
                            "namespace": namespace,
                            "routed_experts_dtype": self.owner.routed_experts_dtype,
                        },
                    )
                    process.start()
                    child.close()
                    self.children.append(process)
                    self.connections.append(parent)
                owner_addresses = {
                    "input_address": addresses.inputs[0],
                    "output_address": addresses.outputs[0],
                    "tensor_queue": launch.tensor_queue,
                }

                def make_owner_client() -> None:
                    try:
                        owner_config = copy.deepcopy(config)
                        owner_config.parallel_config._api_process_rank = 0
                        parent_result.set_result(
                            AsyncLLM(
                                vllm_config=owner_config,
                                executor_class=executor,
                                log_stats=not engine_args.disable_log_stats,
                                log_requests=engine_args.enable_log_requests,
                                stat_loggers=stat_loggers,
                                client_addresses=owner_addresses,
                                client_count=self.count,
                                client_index=0,
                            )
                        )
                    except BaseException as error:
                        parent_result.set_exception(error)

                # Clients bind before the launch context performs its engine
                # handshake; their constructors wait for that handshake to finish.
                parent_thread = threading.Thread(target=make_owner_client, daemon=True)
                parent_thread.start()
                launch.watched_frontend_processes = self.children
            engine = parent_result.result(timeout=600)
            for connection in self.connections:
                receive_reply(connection)
            print(f"NRL HTTP API pool ready: {self.count} processes, one engine")
            return engine
        except BaseException:
            self.stop_children()
            self.close_engines()
            if parent_result.done() and parent_result.exception() is None:
                parent_result.result().shutdown()
            raise
        finally:
            if parent_thread is not None:
                parent_thread.join(timeout=1)

    async def command(self, name: str, *args: Any) -> list[Any]:
        """Serialize a control operation and wait for every child acknowledgement.

        Args:
            name: Control operation or query to broadcast to child frontends.
            *args: Operation-specific positional values sent to each child.

        Returns:
            Reply payloads in child frontend order.
        """
        async with self._command_lock:
            try:
                for process, connection in zip(
                    self.children, self.connections, strict=True
                ):
                    if not process.is_alive():
                        raise RuntimeError(f"HTTP API process {process.name} exited")
                    connection.send((name, args))
                replies = await asyncio.gather(
                    *(
                        asyncio.to_thread(receive_reply, connection)
                        for connection in self.connections
                    )
                )
                return list(replies)
            except BaseException:
                self.stop_children()
                self.close_engines()
                raise

    def get_metrics_snapshot(self) -> list[Counter | Gauge]:
        """Read cumulative token counts and shared-engine gauges across frontends.

        Returns:
            Logger samples from the multiprocess Prometheus registry.
        """
        # vLLM metrics are optional outside generation workers.
        from vllm.v1.metrics.prometheus import get_prometheus_registry
        from vllm.v1.metrics.reader import Counter, Gauge

        metrics: list[Counter | Gauge] = []
        gauge_names = {
            "vllm:num_requests_running",
            "vllm:num_requests_waiting",
            "vllm:kv_cache_usage_perc",
        }
        for metric in get_prometheus_registry().collect():
            if metric.type == "gauge" and metric.name in gauge_names:
                # The registry applies vLLM's mostrecent mode to shared gauges.
                metrics.extend(
                    Gauge(name=metric.name, labels=sample.labels, value=sample.value)
                    for sample in metric.samples
                    if sample.name == metric.name
                )
            elif metric.type == "counter" and metric.name == "vllm:generation_tokens":
                metrics.extend(
                    Counter(
                        name=metric.name, labels=sample.labels, value=int(sample.value)
                    )
                    for sample in metric.samples
                    if sample.name == metric.name + "_total"
                )
        return metrics

    def stop_children(self) -> None:
        """Terminate and join child frontends, then close their control pipes."""
        for process in self.children:
            if process.is_alive():
                process.terminate()
        for process in self.children:
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join(timeout=10)
        for connection in self.connections:
            connection.close()
        self.children.clear()
        self.connections.clear()

    def close_engines(self) -> None:
        """Shut down the shared engine manager and its coordinator."""
        if self.engine_manager is not None:
            self.engine_manager.shutdown()
            self.engine_manager = None
        if self.coordinator is not None:
            self.coordinator.shutdown()
            self.coordinator = None


def run_api_process(
    connection: Connection,
    generation_config: VllmConfig,
    engine_args: Any,
    engine_config: Any,
    client_addresses: dict[str, Any],
    *,
    count: int,
    index: int,
    listener: socket.socket,
    node_ip: str,
    namespace: str,
    routed_experts_dtype: Any,
) -> None:
    """Attach a child frontend to the shared engine and serve control operations.

    Args:
        connection: Child end of the frontend control pipe.
        generation_config: NeMo RL configuration used by the frontend.
        engine_args: vLLM arguments copied for this frontend's process rank.
        engine_config: vLLM configuration updated with this frontend's rank.
        client_addresses: Engine input/output addresses and optional tensor queue.
        count: Total frontend count, including the owning worker.
        index: This child frontend's client index.
        listener: Listening socket shared by all frontends.
        node_ip: Node address used in the advertised HTTP URL.
        namespace: Ray namespace used when token capture connects to the cluster.
        routed_experts_dtype: Dtype used to carry captured routed-expert IDs.
    """
    # Spawned frontends reuse the capture/renderer implementation without running
    # the GPU-owning worker constructor or creating a second engine.
    import ray
    from vllm.v1.engine.async_llm import AsyncLLM
    from vllm.v1.executor.abstract import Executor
    from vllm.v1.metrics.prometheus import shutdown_prometheus

    from nemo_rl.models.generation.vllm.config import resolve_vllm_video_config
    from nemo_rl.models.generation.vllm.vllm_worker_async import (
        VllmAsyncGenerationWorkerImpl,
        _AsyncLLMHTTPClient,
    )

    host: Any = VllmAsyncGenerationWorkerImpl.__new__(VllmAsyncGenerationWorkerImpl)
    host._init_frontend_state()
    host.cfg = generation_config
    host.routed_experts_dtype = routed_experts_dtype
    host.is_model_owner = True
    host._sparse_refit_receiver = None
    host._reserved_socket = listener
    host._reserved_port = listener.getsockname()[1]
    host._reserved_node_ip = node_ip
    host.llm_async_engine_args = copy.deepcopy(engine_args)
    host.llm_async_engine_args._api_process_rank = index
    engine = None
    engine_config.parallel_config._api_process_rank = index
    try:
        video_config = resolve_vllm_video_config(generation_config)
        if video_config is not None:
            # Spawned video frontends need their own process-local loader registration.
            from nemo_rl.models.generation.vllm.video_utils import (
                register_torchcodec_vllm_video_loader,
            )

            register_torchcodec_vllm_video_loader(
                sampling_style=video_config.sampling_style,
                temporal_patch_size=video_config.temporal_patch_size,
            )
        engine = AsyncLLM(
            vllm_config=engine_config,
            executor_class=Executor.get_class(engine_config),
            log_stats=not engine_args.disable_log_stats,
            log_requests=engine_args.enable_log_requests,
            client_addresses=client_addresses,
            client_count=count,
            client_index=index,
        )
        budget_ms = host.cfg["vllm_cfg"].get("output_loop_yield_budget_ms")
        if budget_ms is not None:
            # Keep the optional output-loop patch local to generation workers.
            from nemo_rl.models.generation.vllm.patches import install_output_loop_yield

            install_output_loop_yield(engine, reader_budget_s=budget_ms / 1000)
        host.llm = engine
        connection.send(("ok", None))

        async def serve() -> None:
            host._engine_loop = asyncio.get_running_loop()
            host._http_engine_client = _AsyncLLMHTTPClient(
                host.llm,
                host._engine_loop,
                flat_logprobs=bool(host.cfg["vllm_cfg"].get("flat_logprobs")),
            )
            while True:
                name, args = await asyncio.to_thread(connection.recv)
                result = None
                if name == "start":
                    host.server_thread, host.base_url, host.http_server = (
                        host._setup_vllm_server()
                    )
                    deadline = time.monotonic() + 60
                    while not host.http_server.started:
                        if (
                            not host.server_thread.is_alive()
                            or time.monotonic() > deadline
                        ):
                            raise RuntimeError("HTTP API server failed to start")
                        await asyncio.sleep(0.05)
                elif name == "capture":
                    if not ray.is_initialized():
                        ray.init(
                            address="auto", namespace=namespace, log_to_driver=False
                        )
                    await host.setup_token_capture(
                        args[0], args[1], capture_media=args[2], top_logprobs=args[3]
                    )
                elif name == "version":
                    await host.set_rollout_weight_version(args[0])
                elif name == "progress":
                    result = await host.captured_call_progress()
                elif name == "snapshot":
                    result = host.get_data_plane_snapshot()
                elif name == "stop":
                    if host.http_server is not None:
                        host.http_server.should_exit = True
                        assert host.server_thread is not None
                        await asyncio.to_thread(host.server_thread.join, 30)
                    connection.send(("ok", None))
                    return
                else:
                    raise ValueError(f"Unknown HTTP API command: {name}")
                connection.send(("ok", result))

        asyncio.run(serve())
    except BaseException as error:
        try:
            connection.send(("error", repr(error)))
        finally:
            raise
    finally:
        if engine is not None:
            engine.shutdown()
        shutdown_prometheus()
        connection.close()
        listener.close()
        if ray.is_initialized():
            ray.shutdown()
