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

import asyncio
import time
from typing import Optional

import requests
import torch
from megatron.core.inference.config import (
    InferenceConfig,
    KVCacheManagementMode,
    PrefixCachingCoordinatorPolicy,
)
from megatron.core.inference.engines.dynamic_engine import EngineState


class MegatronGenerationWorkerMixin:
    """Inference-engine state and lifecycle for megatron-based generation."""

    def _init_inference_engine_state(self) -> None:
        """Reset all inference-engine attributes to their uninitialized state.

        Called from the host worker's `__init__` so that all inference-only
        state is grouped in one place and visible to anyone reading the mixin.
        """
        self.dynamic_inference_engine = None
        self.inference_client = None
        self.inference_context = None
        self.inference_wrapped_model = None
        self.base_url = None
        self._inference_engine_initialized = False
        self._inference_engine_asleep = True  # Start paused since we begin with training
        self._inference_loop = None  # Event loop for inference operations
        self._inference_thread = None  # Thread running the event loop

    def _initialize_inference_engine(self, mcore_generation_config: dict) -> None:
        """Initialize the persistent inference engine and client.

        This method sets up the DynamicInferenceEngine, DynamicInferenceContext,
        and InferenceClient for coordinator-based inference. The engine is created
        once and reused across multiple generate() calls.
        """
        if self._inference_engine_initialized:
            return

        from megatron.core.inference.contexts.dynamic_context import (
            DynamicInferenceContext,
        )
        from megatron.core.inference.engines.dynamic_engine import (
            DynamicInferenceEngine,
        )
        from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
            GPTInferenceWrapper,
        )
        from megatron.core.inference.text_generation_controllers.text_generation_controller import (
            TextGenerationController,
        )

        from megatron.core.utils import get_attr_wrapped_model
        pg_collection = get_attr_wrapped_model(self.model, "pg_collection")

        buffer_size_gb = mcore_generation_config["buffer_size_gb"]
        num_cuda_graphs = mcore_generation_config["num_cuda_graphs"]
        block_size_tokens = mcore_generation_config["block_size_tokens"]
        enable_chunked_prefill = mcore_generation_config["enable_chunked_prefill"]
        use_cuda_graphs_for_non_decode_steps = mcore_generation_config[
            "use_cuda_graphs_for_non_decode_steps"
        ]
        max_tokens = mcore_generation_config["max_tokens"]

        # Level 0: No unified memory, CUDA graphs are deleted/recreated on pause/resume
        # Level 1: Unified memory enabled, tensors maintain static addresses
        unified_memory_level = mcore_generation_config["unified_memory_level"]
        kv_cache_management_mode = mcore_generation_config["kv_cache_management_mode"]
        static_kv_memory_pointers = mcore_generation_config["static_kv_memory_pointers"]
        materialize_only_last_token_logits = mcore_generation_config["materialize_only_last_token_logits"]
        num_speculative_tokens = mcore_generation_config["num_speculative_tokens"]
        max_requests = mcore_generation_config.get("max_requests", None)

        model_config = self.model.config

        from megatron.core.inference.config import MambaInferenceStateConfig
        mamba_inference_state_config = MambaInferenceStateConfig.from_model(self.model)

        if mcore_generation_config.get("mamba_inference_ssm_states_dtype", None) is not None:
            dtype_val = mcore_generation_config["mamba_inference_ssm_states_dtype"]
            mamba_inference_state_config.ssm_states_dtype = self._resolve_torch_dtype(dtype_val)

        if mcore_generation_config.get("mamba_inference_conv_states_dtype", None) is not None:
            dtype_val = mcore_generation_config["mamba_inference_conv_states_dtype"]
            mamba_inference_state_config.conv_states_dtype = self._resolve_torch_dtype(dtype_val)

        enable_prefix_caching = mcore_generation_config.get("enable_prefix_caching", False)
        prefix_caching_coordinator_policy = mcore_generation_config.get(
            "prefix_caching_coordinator_policy", "first_prefix_block"
        )

        inference_config = InferenceConfig(
            block_size_tokens=block_size_tokens,
            buffer_size_gb=buffer_size_gb,
            num_cuda_graphs=num_cuda_graphs,
            max_tokens=max_tokens,
            max_sequence_length=self.cfg["max_total_sequence_length"],
            unified_memory_level=unified_memory_level,
            kv_cache_management_mode=KVCacheManagementMode(kv_cache_management_mode),
            static_kv_memory_pointers=static_kv_memory_pointers,
            use_cuda_graphs_for_non_decode_steps=use_cuda_graphs_for_non_decode_steps,
            use_flashinfer_fused_rope=True,
            use_synchronous_zmq_collectives=True,
            materialize_only_last_token_logits=materialize_only_last_token_logits,
            enable_chunked_prefill=enable_chunked_prefill,
            enable_prefix_caching=enable_prefix_caching,
            prefix_caching_coordinator_policy=PrefixCachingCoordinatorPolicy(prefix_caching_coordinator_policy),
            pg_collection=pg_collection,
            mamba_inference_state_config=mamba_inference_state_config,
            mamba_memory_ratio=0.1 + 0.1 * num_speculative_tokens,  # Hack to account for the effect of speculative decode slots
            logging_step_interval=mcore_generation_config.get("logging_step_interval", 0),
            num_speculative_tokens=num_speculative_tokens,
            max_requests=max_requests,
        )

        # Create inference context
        self.inference_context = DynamicInferenceContext(model_config, inference_config)

        # Create inference wrapper
        self.inference_wrapped_model = GPTInferenceWrapper(
            self.model, self.inference_context
        )
        # Create text generation controller
        text_generation_controller = TextGenerationController(
            inference_wrapped_model=self.inference_wrapped_model,
            tokenizer=self.megatron_tokenizer,
        )

        # Create the inference engine
        self.dynamic_inference_engine = DynamicInferenceEngine(
            text_generation_controller,
            self.inference_context,
        )

        self._inference_engine_initialized = True
        self._inference_engine_asleep = True  # Engine starts in paused state
        print(f"[Rank {self.rank}] Initialized persistent inference engine")

    @staticmethod
    def _resolve_torch_dtype(val):
        """Convert a value to torch.dtype, accepting both torch.dtype and string forms like 'torch.float32' or 'float32'."""
        if isinstance(val, torch.dtype):
            return val
        if isinstance(val, str):
            name = val.replace("torch.", "")
            dtype = getattr(torch, name, None)
            if isinstance(dtype, torch.dtype):
                return dtype
        raise ValueError(
            f"Cannot resolve torch dtype from {val!r} (type {type(val).__name__}). "
            f"Expected a torch.dtype or a string like 'torch.float32' / 'float32'."
        )

    async def _start_inference_coordinator(self, coordinator_port: int):
        """Start the inference coordinator and engine loop.

        This is called once when the inference infrastructure is first needed.
        The engine's start_listening_to_data_parallel_coordinator returns the
        actual coordinator address (dp_addr) which is used to create the client.
        """
        self.coordinator_addr = await self.dynamic_inference_engine.start_listening_to_data_parallel_coordinator(
            inference_coordinator_port=coordinator_port,
            launch_inference_coordinator=True,
        )
        rank = torch.distributed.get_rank()
        if rank == 0:
            from megatron.core.inference.inference_client import InferenceClient
            self.inference_client = InferenceClient(
                inference_coordinator_address=self.coordinator_addr, deserialize=True
            )
            result = self.inference_client.start()
            if result is not None:
                await result

        self._inference_engine_asleep = False

    def _sleep(self):
        """Pause the inference engine to free GPU memory for training.

        Uses the coordinator's pause+suspend mechanism:
        1. Rank 0 sends PAUSE → all ranks wait for engine to reach PAUSED
        2. Rank 0 sends SUSPEND → all ranks wait for engine to reach SUSPENDED

        The engine internally handles GPU state deallocation (KV cache, CUDA
        graphs, etc.) during the SUSPENDED transition, so no explicit
        engine.suspend() call is needed.
        """
        future = asyncio.run_coroutine_threadsafe(
            self._sleep_engine(),
            self._inference_loop,
        )
        future.result()
        # Synchronize all ranks
        torch.distributed.barrier()

        self._inference_engine_asleep = True
        print(f"[Rank {self.rank}] paused inference engine")

    async def _sleep_engine(self):
        """Send pause + suspend signals via the coordinator and wait for acknowledgment.

        Follows the coordinator state machine: RUNNING → PAUSED → SUSPENDED.
        The coordinator requires engines to be PAUSED before accepting SUSPEND.
        """
        if torch.distributed.get_rank() == 0:
            self.inference_client.pause_engines()
        await self.dynamic_inference_engine.wait_until(EngineState.PAUSED)

        if torch.distributed.get_rank() == 0:
            self.inference_client.suspend_engines()
        await self.dynamic_inference_engine.wait_until(EngineState.SUSPENDED)

    def _wake(self):
        """Resume the inference engine after training.

        Uses the coordinator's resume+unpause mechanism:
        1. Rank 0 sends RESUME → engine reallocates GPU state internally
        2. Rank 0 sends UNPAUSE → all ranks wait for engine to reach RUNNING

        The engine internally handles GPU state reallocation (KV cache, CUDA
        graphs, etc.) during the RESUMING transition, so no explicit
        engine.resume() call is needed.
        """
        # Use the coordinator-based resume mechanism
        # Only rank 0 sends the signal - coordinator broadcasts to all DP engines
        future = asyncio.run_coroutine_threadsafe(
            self._wake_engine(),
            self._inference_loop,
        )
        future.result()
        # Synchronize all ranks
        torch.distributed.barrier()

        self._inference_engine_asleep = False

    async def _wake_engine(self):
        """Send resume + unpause signals via the coordinator and wait for acknowledgment.

        Follows the coordinator state machine: SUSPENDED → RESUMED → RUNNING.
        The engine reallocates GPU state during the RESUMED transition, then
        UNPAUSE brings it back to the RUNNING state.
        """
        if torch.distributed.get_rank() == 0:
            self.inference_client.resume_engines()
        await self.dynamic_inference_engine.wait_until(EngineState.RESUMED)

        if torch.distributed.get_rank() == 0:
            self.inference_client.unpause_engines()
        await self.dynamic_inference_engine.wait_until(EngineState.RUNNING)

    def suspend_for_refit(self, recompute_kv_cache: bool = False) -> None:
        """Pause or suspend engine for safe weight update.

        When recompute_kv_cache is False (default), uses pause_engines() which
        preserves KV cache and CUDA graphs (Magistral-style).

        When recompute_kv_cache is True, uses suspend_engines() which
        checkpoints in-flight requests and pauses the engine. Uses the
        default PERSIST kv_cache_management_mode so CUDA graphs are
        preserved. On resume, checkpointed requests are replayed with
        fresh prefill using the new weights (AREAL-style).
        """
        if not self._inference_engine_initialized:
            return
        if recompute_kv_cache:
            future = asyncio.run_coroutine_threadsafe(
                self._sleep_engine(), self._inference_loop
            )
        else:
            future = asyncio.run_coroutine_threadsafe(
                self._pause_engine_for_refit(), self._inference_loop
            )
        future.result()

        # Drain in-flight CUDA graph replays.
        torch.cuda.synchronize()

        if recompute_kv_cache:
            self._inference_engine_asleep = True

    async def _pause_engine_for_refit(self):
        rank = torch.distributed.get_rank()
        if rank == 0:
            self.inference_client.pause_engines()
        await self.dynamic_inference_engine.wait_until(EngineState.PAUSED)

    def resume_after_refit(self, recompute_kv_cache: bool = False) -> None:
        """Resume engine after weight update.

        When recompute_kv_cache is False (default), uses unpause_engines() —
        in-progress generations continue with new weights and existing KV cache.

        When recompute_kv_cache is True, uses resume_engines() which
        reallocates state and replays checkpointed requests. Since
        PERSIST mode is used, CUDA graphs remain valid and don't
        need to be recaptured.
        """
        if not self._inference_engine_initialized:
            return

        if recompute_kv_cache:
            future = asyncio.run_coroutine_threadsafe(
                self._wake_engine(), self._inference_loop
            )
        else:
            future = asyncio.run_coroutine_threadsafe(
                self._unpause_engine_after_refit(), self._inference_loop
            )
        future.result()
        if recompute_kv_cache:
            self._inference_engine_asleep = False

    async def _unpause_engine_after_refit(self):
        rank = torch.distributed.get_rank()
        if rank == 0:
            self.inference_client.unpause_engines()
        await self.dynamic_inference_engine.wait_until(EngineState.RUNNING)

    def _log_gpu_memory(self, tag: str):
        rank = torch.distributed.get_rank() if torch.distributed.is_initialized() else 0
        allocated = torch.cuda.memory_allocated() / (1024 ** 3)
        reserved = torch.cuda.memory_reserved() / (1024 ** 3)
        free, total = torch.cuda.mem_get_info()
        free_gb, total_gb = free / (1024 ** 3), total / (1024 ** 3)
        print(
            f"[GPU Rank {rank}] {tag} | "
            f"Allocated: {allocated:.2f} GB, Reserved: {reserved:.2f} GB, "
            f"Free: {free_gb:.2f} GB, Total: {total_gb:.2f} GB"
        )

    def _start_inference_loop_thread(self):
        """Start a background thread with a persistent event loop for inference.

        This thread runs the event loop that hosts the engine loop task.
        The loop runs forever until explicitly stopped.
        """
        import threading

        def run_loop():
            asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
            self._inference_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._inference_loop)
            # Run forever - the engine loop task will run in this loop
            self._inference_loop.run_forever()

        self._inference_thread = threading.Thread(target=run_loop, daemon=True)
        self._inference_thread.start()

        # Wait for the loop to be created
        while self._inference_loop is None:
            time.sleep(0.001)

    def report_dp_openai_server_base_url(self) -> Optional[str]:
        return self.base_url

    def _setup_openai_api_server(self) -> str:
        from megatron.core.inference.text_generation_server.dynamic_text_gen_server.text_generation_server import (
            start_text_gen_server,
        )

        from nemo_rl.distributed.virtual_cluster import (
            _get_free_port_local,
            _get_node_ip_local,
        )
        ip = _get_node_ip_local()
        free_port = _get_free_port_local()

        start_text_gen_server(
            coordinator_addr=self.coordinator_addr,
            tokenizer=self.megatron_tokenizer,
            rank=torch.distributed.get_rank(),
            server_port=free_port,
            parsers=self.cfg["generation"]["mcore_generation_config"].get("parsers", []),
            verbose=False,
        )

        base_url = f"http://{ip}:{free_port}/v1"
        max_wait_time = 300
        start_time = time.time()
        with requests.Session() as session:
            while True:
                if time.time() - start_time > max_wait_time:
                    raise TimeoutError(
                        f"[Megatron HTTP] Rank {self.rank} OpenAI server failed "
                        f"to start within {max_wait_time}s"
                    )
                try:
                    response = session.get(f"{base_url}/health", timeout=10)
                    if response.status_code == 200:
                        break
                except requests.RequestException:
                    pass
                time.sleep(2)
        return base_url

    def _run_async_coordinator_start(self, coordinator_port: int):
        """Start the coordinator and engine loop in the background thread.

        This is called once during the first generate() call to initialize
        the persistent inference infrastructure.
        """
        # Start the background thread with the event loop if not already running
        if self._inference_loop is None:
            self._start_inference_loop_thread()

        # Schedule the coordinator start in the inference loop
        future = asyncio.run_coroutine_threadsafe(
            self._start_inference_coordinator(coordinator_port),
            self._inference_loop,
        )

        # Wait for coordinator start AND CUDA-graph warmup to complete.
        # _start_inference_coordinator awaits running.wait() so future.result()
        # only returns once this rank's engine is fully warmed up.
        # Cross-rank sync is handled by Ray's actor group semantics: the caller
        # (refit_policy_generation) waits for ALL generation workers to return from
        # prepare_for_generation() before proceeding, so no explicit barrier needed.
        future.result()
        print(f"[Rank {torch.distributed.get_rank()}] Coordinator started")

        # Start the HTTP Server
        if (
            self.cfg["generation"]["mcore_generation_config"].get("expose_http_server", False)
            and torch.distributed.get_rank() == 0
        ):
            print(f"[Rank {torch.distributed.get_rank()}] Starting HTTP Server")
            self.base_url = self._setup_openai_api_server()
        else:
            print(f"[Rank {torch.distributed.get_rank()}] HTTP Server not started")
            self.base_url = None

        return
