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

import asyncio
import gc
import threading
import time
import warnings
from typing import AsyncGenerator, Optional

import requests
import torch
from megatron.core.inference.config import (
    InferenceConfig,
    KVCacheManagementMode,
    PrefixCachingCoordinatorPolicy,
)
from megatron.core.inference.engines.dynamic_engine import EngineState
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.transformer.enums import InferenceCudaGraphScope
from megatron.core.transformer.utils import toggle_cuda_graphs

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationOutputSpec,
    verify_right_padding,
)
from nemo_rl.models.generation.megatron.utils import (
    get_lang_module,
    log_gpu_memory,
    resolve_torch_dtype,
)
from nemo_rl.utils.nsys import wrap_with_nvtx_name


class MegatronGenerationMixin:
    """Engine lifecycle, coordinator, HTTP server, and finish-generation machinery.

    The host class must provide:

     - model: the megatron module.
     - cfg: policy config (TypedDict).
     - rank: global rank (used for logging).
     - tokenizer: HF tokenizer.
     - megatron_tokenizer: tokenizer for inference.
     - is_generation_colocated: Whether colocated or distributed.
    """

    def _init_inference_engine_state(self) -> None:
        """Reset all inference-engine attributes to their uninitialized state."""
        self.dynamic_inference_engine = None
        self.inference_client = None
        self.inference_context = None
        self.inference_wrapped_model = None
        self.base_url = None
        self._inference_engine_initialized = False
        self._inference_engine_asleep = True  # Start paused since we begin with training
        self._inference_loop = None
        self._inference_thread = None

    def _initialize_inference_engine(self, mcore_generation_config: dict) -> None:
        """Initialize the persistent inference engine and client."""
        # TODO: Switch to standardized Megatron API.
        if self._inference_engine_initialized:
            return

        from megatron.core.inference.config import MambaInferenceStateConfig
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
        enable_chunked_prefill = mcore_generation_config.get("enable_chunked_prefill", True)
        use_cuda_graphs_for_non_decode_steps = mcore_generation_config[
            "use_cuda_graphs_for_non_decode_steps"
        ]
        max_tokens = mcore_generation_config["max_tokens"]

        # The value may be overwritten by `recompute_kv_cache_after_weight_updates`.
        kv_cache_management_mode = mcore_generation_config.get("kv_cache_management_mode", "persist")
        persist_cuda_graphs = kv_cache_management_mode != "persist"
        cuda_graph_impl = mcore_generation_config.get("cuda_graph_impl", "local")

        materialize_only_last_token_logits = mcore_generation_config[
            "materialize_only_last_token_logits"
        ]
        num_speculative_tokens = mcore_generation_config["num_speculative_tokens"]
        max_requests = mcore_generation_config.get("max_requests", None)
        use_flashinfer_sampling = mcore_generation_config.get("use_flashinfer_sampling", True)
        sampling_backend = "flashinfer" if use_flashinfer_sampling else "torch"

        mamba_inference_state_config = MambaInferenceStateConfig.from_model(self.model)
        if mcore_generation_config.get("mamba_inference_ssm_states_dtype", None) is not None:
            mamba_inference_state_config.ssm_states_dtype = resolve_torch_dtype(
                mcore_generation_config["mamba_inference_ssm_states_dtype"]
            )
        if mcore_generation_config.get("mamba_inference_conv_states_dtype", None) is not None:
            mamba_inference_state_config.conv_states_dtype = resolve_torch_dtype(
                mcore_generation_config["mamba_inference_conv_states_dtype"]
            )

        enable_prefix_caching = mcore_generation_config.get("enable_prefix_caching", False)
        prefix_caching_coordinator_policy = mcore_generation_config.get(
            "prefix_caching_coordinator_policy", "first_prefix_block"
        )
        mamba_memory_ratio = mcore_generation_config.get(
            "mamba_memory_ratio", 0.1 + 0.1 * num_speculative_tokens
        )

        inference_config = InferenceConfig(
            block_size_tokens=block_size_tokens,
            buffer_size_gb=buffer_size_gb,
            num_cuda_graphs=num_cuda_graphs,
            max_tokens=max_tokens,
            max_sequence_length=self.cfg["max_total_sequence_length"],
            kv_cache_management_mode=KVCacheManagementMode(kv_cache_management_mode),
            static_kv_memory_pointers=persist_cuda_graphs,
            use_cuda_graphs_for_non_decode_steps=use_cuda_graphs_for_non_decode_steps,
            use_flashinfer_fused_rope=True,
            sampling_backend=sampling_backend,
            use_synchronous_zmq_collectives=True,
            materialize_only_last_token_logits=materialize_only_last_token_logits,
            enable_chunked_prefill=enable_chunked_prefill,
            enable_prefix_caching=enable_prefix_caching,
            prefix_caching_coordinator_policy=PrefixCachingCoordinatorPolicy(
                prefix_caching_coordinator_policy
            ),
            pg_collection=pg_collection,
            mamba_inference_state_config=mamba_inference_state_config,
            mamba_memory_ratio=mamba_memory_ratio,
            logging_step_interval=mcore_generation_config.get("logging_step_interval", 0),
            num_speculative_tokens=num_speculative_tokens,
            max_requests=max_requests,
        )

        if "inference_cuda_graph_scope" in mcore_generation_config:
            self.model.config.inference_cuda_graph_scope = InferenceCudaGraphScope[
                mcore_generation_config["inference_cuda_graph_scope"]
            ]

        self.inference_context = DynamicInferenceContext(self.model.config, inference_config)
        self.inference_wrapped_model = GPTInferenceWrapper(self.model, self.inference_context)
        text_generation_controller = TextGenerationController(
            inference_wrapped_model=self.inference_wrapped_model,
            tokenizer=self.megatron_tokenizer,
        )
        self.dynamic_inference_engine = DynamicInferenceEngine(
            text_generation_controller, self.inference_context
        )

        self._inference_engine_initialized = True
        self._inference_engine_asleep = True
        print(f"[Rank {self.rank}] Initialized persistent inference engine")

    async def _start_inference_coordinator(self, coordinator_port: int):
        """Start the inference coordinator and engine loop."""
        self.coordinator_addr = await self.dynamic_inference_engine.start_listening_to_data_parallel_coordinator(
            inference_coordinator_port=coordinator_port,
            launch_inference_coordinator=True,
        )
        if torch.distributed.get_rank() == 0:
            from megatron.core.inference.inference_client import InferenceClient

            self.inference_client = InferenceClient(
                inference_coordinator_address=self.coordinator_addr, deserialize=True
            )
            result = self.inference_client.start()
            if result is not None:
                await result

        self._inference_engine_asleep = False

    def _sleep(self) -> None:
        """Pause + suspend the engine. No-op if already asleep."""
        if self._inference_engine_asleep:
            return
        future = asyncio.run_coroutine_threadsafe(self._sleep_engine(), self._inference_loop)
        future.result()
        torch.distributed.barrier()
        self._inference_engine_asleep = True
        print(f"[Rank {self.rank}] paused inference engine")

    async def _sleep_engine(self):
        if torch.distributed.get_rank() == 0:
            self.inference_client.pause_engines()
        await self.dynamic_inference_engine.wait_until(EngineState.PAUSED)

        if torch.distributed.get_rank() == 0:
            self.inference_client.suspend_engines()
        await self.dynamic_inference_engine.wait_until(EngineState.SUSPENDED)

    def _wake(self) -> None:
        """Resume + unpause the engine. No-op if already awake."""
        if not self._inference_engine_asleep:
            return
        future = asyncio.run_coroutine_threadsafe(self._wake_engine(), self._inference_loop)
        future.result()
        torch.distributed.barrier()
        self._inference_engine_asleep = False
        print(f"[Rank {self.rank}] resumed inference engine")

    async def _wake_engine(self):
        if torch.distributed.get_rank() == 0:
            self.inference_client.resume_engines()
        await self.dynamic_inference_engine.wait_until(EngineState.RESUMED)

        if torch.distributed.get_rank() == 0:
            self.inference_client.unpause_engines()
        await self.dynamic_inference_engine.wait_until(EngineState.RUNNING)

    def _start_inference_loop_thread(self):
        """Start a background thread with a persistent event loop for inference."""

        def run_loop():
            asyncio.set_event_loop_policy(asyncio.DefaultEventLoopPolicy())
            self._inference_loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._inference_loop)
            self._inference_loop.run_forever()

        self._inference_thread = threading.Thread(target=run_loop, daemon=True)
        self._inference_thread.start()
        while self._inference_loop is None:
            time.sleep(0.001)

    def _setup_openai_api_server(self) -> str:
        """Start the OpenAI-compatible HTTP server on this worker."""
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
        """Start the coordinator and engine loop in the background thread."""
        if self._inference_loop is None:
            self._start_inference_loop_thread()

        future = asyncio.run_coroutine_threadsafe(
            self._start_inference_coordinator(coordinator_port), self._inference_loop
        )
        # _start_inference_coordinator awaits RUNNING, so future.result() only returns once
        # this rank's engine is fully warmed up. Cross-rank sync is handled by Ray's actor
        # group semantics (the caller waits for all workers' prepare_for_generation).
        future.result()
        print(f"[Rank {torch.distributed.get_rank()}] Coordinator started")

        if (
            self.cfg["generation"]["mcore_generation_config"].get("expose_http_server", False)
            and torch.distributed.get_rank() == 0
        ):
            print(f"[Rank {torch.distributed.get_rank()}] Starting HTTP Server")
            self.base_url = self._setup_openai_api_server()
        else:
            print(f"[Rank {torch.distributed.get_rank()}] HTTP Server not started")
            self.base_url = None

    def finish_generation(self) -> None:
        """Wind down a generation cycle."""
        print(f"[Rank {self.rank}] finishing generation", flush=True)
        log_gpu_memory("finish_generation START")

        lang_module = get_lang_module(self.model)

        if self.is_generation_colocated:
            if self._inference_engine_initialized and not self._inference_engine_asleep:
                self._sleep()
            cuda_graph_impl = self.cfg["generation"]["mcore_generation_config"].get(
                "cuda_graph_impl", "local"
            )
            if cuda_graph_impl != "none":
                toggle_cuda_graphs(lang_module, set_to="none")

        rotary_module = getattr(lang_module, "rotary_pos_emb", None)
        if rotary_module is not None and hasattr(rotary_module.forward, "cache_parameters"):
            rotary_module.forward.cache_clear()

        if self.is_generation_colocated:
            gc.collect()
            torch.cuda.empty_cache()

        log_gpu_memory("finish_generation END")

    def prepare_for_generation(self, tags=None, **kwargs) -> Optional[str]:
        """Enter inference mode and start (or wake) the inference engine.

        Returns:
            The OpenAI server base URL for this worker (or `None` if this worker is not the leader).
        """
        log_gpu_memory("prepare_for_generation START")
        mcore_generation_config = self.cfg["generation"]["mcore_generation_config"]

        self.model.config.flash_decode = False
        if self.is_generation_colocated and self.should_disable_forward_pre_hook:
            # Bring offloaded params back to CUDA before colocated generation.
            self.model = self.move_model(
                self.model, "cuda", move_params=True, move_grads=False
            )

        lang_module = get_lang_module(self.model)
        lang_module.eval()

        rotary_module = getattr(lang_module, "rotary_pos_emb", None)
        if rotary_module is not None and hasattr(rotary_module.forward, "cache_parameters"):
            rotary_module.forward.cache_clear()

        cuda_graph_impl = mcore_generation_config.get("cuda_graph_impl", "local")
        if cuda_graph_impl != "none":
            toggle_cuda_graphs(lang_module, set_to=cuda_graph_impl)

        # tags=["weights"] means we are inside refit_policy_generation between
        # suspend_for_refit and the weight transfer — the engine was intentionally
        # paused and waking it now would race NVSHMEM init / weight transfer against
        # CUDA-graph replay, corrupting TE FP8 state. The subsequent
        # prepare_for_generation(tags=["kv_cache"]) is what actually wakes it.
        if tags is None or "weights" not in tags:
            if not self._inference_engine_initialized:
                self._initialize_inference_engine(mcore_generation_config)
                coordinator_port = mcore_generation_config.get(
                    "inference_coordinator_port", 5995
                )
                self._run_async_coordinator_start(coordinator_port)
            else:
                self._wake()

        log_gpu_memory("prepare_for_generation END")
        return self.base_url

    def _prepare_data_for_generation(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, SamplingParams]:
        """Build the prompt tensors and sampling params for one batch of requests."""
        if data is not None:
            assert isinstance(data, BatchedDataDict), (
                f"data must be a BatchedDataDict, got type: {type(data)}"
            )
            is_right_padded, error_msg = verify_right_padding(
                data, pad_value=self.tokenizer.pad_token_id
            )
            if not is_right_padded:
                warnings.warn(
                    f"Input to Megatron Generation worker is not properly right-padded: {error_msg}"
                )

        top_k_cfg = self.cfg["generation"]["top_k"]
        top_k_val = 1 if greedy else (int(top_k_cfg) if top_k_cfg is not None else 0)

        top_p_cfg = self.cfg["generation"]["top_p"]
        top_p_val = (
            0.0 if greedy else (float(top_p_cfg) if top_p_cfg is not None else 0.0)
        )

        sampling_params = SamplingParams(
            temperature=self.cfg["generation"]["temperature"] if not greedy else 0,
            top_k=top_k_val,
            top_p=top_p_val,
            skip_prompt_log_probs=False,
            return_log_probs=True,
            num_tokens_to_generate=self.cfg["generation"]["max_new_tokens"],
            termination_id=self.megatron_tokenizer.eod,
        )

        prompt_tokens_tensor = data["input_ids"].cuda()
        prompt_lengths_tensor = data["input_lengths"]

        return prompt_tokens_tensor, prompt_lengths_tensor, sampling_params

    def _parse_result_to_batched_data_dict(
        self,
        data: BatchedDataDict[GenerationDatumSpec],
        result: list,
    ) -> BatchedDataDict[GenerationOutputSpec]:
        """Pack DynamicInferenceRequest results into a GenerationOutputSpec batch."""
        input_lengths = data["input_lengths"]
        input_ids = data["input_ids"]
        batch_size = input_ids.size(0)
        max_gen_seq_len = max(len(x.generated_tokens) for x in result)
        padded_input_length = input_ids.size(1)

        max_seq_len = padded_input_length + max_gen_seq_len
        output_ids_padded = torch.full(
            (batch_size, max_seq_len),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=input_ids.device,
        )

        logprobs_padded = torch.zeros(
            (batch_size, max_seq_len),
            dtype=torch.float,
            device=input_ids.device,
        )

        generation_lengths = torch.zeros(
            batch_size, dtype=torch.long, device=input_ids.device
        )
        unpadded_sequence_lengths = torch.zeros(
            batch_size, dtype=torch.long, device=input_ids.device
        )
        for i in range(batch_size):
            tokens = result[i].prompt_tokens.tolist() + result[i].generated_tokens
            logprobs = result[i].prompt_log_probs + result[i].generated_log_probs
            seq_len = len(tokens)
            output_ids_padded[i, :seq_len] = torch.tensor(
                tokens, dtype=torch.long, device=input_ids.device
            )
            generation_lengths[i] = seq_len - input_lengths[i].item()
            unpadded_sequence_lengths[i] = seq_len
            logprob_len = len(logprobs)
            logprobs_padded[i, 1 : logprob_len + 1] = torch.tensor(
                logprobs,
                dtype=torch.float,
                device=input_ids.device,
            )

        out_dict = {
            "output_ids": output_ids_padded,
            "logprobs": logprobs_padded,
            "generation_lengths": generation_lengths,
            "unpadded_sequence_lengths": unpadded_sequence_lengths,
        }

        return BatchedDataDict.from_batches([out_dict]).to("cpu")

    @wrap_with_nvtx_name("megatron_policy_worker/generate")
    def generate(
        self, *, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> BatchedDataDict[GenerationOutputSpec]:
        """Synchronous batched generation via the mcore data-parallel coordinator.

        Args:
            data: BatchedDataDict containing input_ids and input_lengths tensors
            greedy: Whether to use greedy decoding instead of sampling

        Returns:
            BatchedDataDict conforming to GenerationOutputSpec:
                - output_ids: input + generated token IDs with proper padding
                - logprobs: Log probabilities for tokens
                - generation_lengths: Lengths of each response
                - unpadded_sequence_lengths: Lengths of each input + generated sequence
        """
        prompt_tokens_tensor, prompt_lengths_tensor, sampling_params = (
            self._prepare_data_for_generation(data, greedy)
        )
        if self._inference_loop is None:
            raise RuntimeError(
                "Inference loop not initialized. Call prepare_for_generation() first."
            )
        future = asyncio.run_coroutine_threadsafe(
            self._generate_with_persistent_engine(
                prompt_tokens_tensor,
                prompt_lengths_tensor,
                sampling_params,
            ),
            self._inference_loop,
        )
        result = future.result()

        return self._parse_result_to_batched_data_dict(data, result)

    async def generate_async(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> AsyncGenerator[tuple[int, BatchedDataDict[GenerationOutputSpec]], None]:
        """Streaming generation: yield `(index, batch)` tuples as they complete.

        Args:
            data: BatchedDataDict with input_ids and input_lengths
            greedy: Whether to use greedy decoding instead of sampling

        Yields:
            Tuple of (original_index, BatchedDataDict conforming to GenerationOutputSpec for the single sequence)
        """
        if self._inference_loop is None:
            raise RuntimeError(
                "Inference loop not initialized. Call prepare_for_generation() first."
            )

        async def _generate_single_item(
            index: int,
        ) -> tuple[int, BatchedDataDict[GenerationOutputSpec]]:
            datum = data.get_batch(index, 1)
            prompt_tokens_tensor, prompt_lengths_tensor, sampling_params = (
                self._prepare_data_for_generation(datum, greedy)
            )
            future = asyncio.run_coroutine_threadsafe(
                self._generate_with_persistent_engine(
                    prompt_tokens_tensor,
                    prompt_lengths_tensor,
                    sampling_params,
                ),
                self._inference_loop,
            )
            result = await asyncio.wrap_future(future)
            output = self._parse_result_to_batched_data_dict(datum, result)
            return (index, output)

        tasks = [asyncio.create_task(_generate_single_item(i)) for i in range(data.size)]
        for result in asyncio.as_completed(tasks):
            yield await result

    async def _generate_with_persistent_engine(
        self,
        prompt_tokens_tensor: torch.Tensor,
        prompt_lengths_tensor: torch.Tensor,
        sampling_params: SamplingParams,
    ) -> list:
        """Submit requests through the persistent inference client (rank 0 only)."""
        from megatron.core.inference.inference_request import DynamicInferenceRequest

        dist_rank = torch.distributed.get_rank()
        assert dist_rank == 0, (
            "Only rank 0 creates a client to communicate with the coordinator"
        )

        print(
            f"[Rank {dist_rank}] Submitting {prompt_tokens_tensor.size(0)} requests to coordinator"
        )

        futures = []
        for prompt_tokens, prompt_len in zip(
            prompt_tokens_tensor, prompt_lengths_tensor, strict=True
        ):
            prompt = prompt_tokens[: prompt_len.item()].tolist()
            futures.append(self.inference_client.add_request(prompt, sampling_params))

        results: list[DynamicInferenceRequest] = await asyncio.gather(*futures)
        print(f"[Rank {dist_rank}] Completed {len(results)} requests")
        return results


class MegatronRefitMixin:
    """Refit collective, weight transfer, and engine suspend/resume around refits."""

    def init_refit_collective(
        self,
        ip: str,
        port: int,
        world_size: int,
        rank_offset: int,
        refit_backend: str = "gloo",
    ) -> None:
        """Initialize the refit collective for non-colocated weight transfer.

        Args:
            ip: IP address for the process group rendezvous.
            port: Port for the process group rendezvous.
            world_size: Total world size (train + inference workers).
            rank_offset: Offset for this side's ranks (`train_world_size` for inference).
            refit_backend: Copy-service backend ("gloo" or "nvshmem").
        """
        from torch.distributed.distributed_c10d import (
            PrefixStore,
            ProcessGroup,
            ProcessGroupGloo,
            _world,
        )

        local_rank = torch.distributed.get_rank()
        global_rank = local_rank + rank_offset
        self.refit_rank_offset = rank_offset

        # port+1 to avoid collision with the caller's rendezvous on `port`.
        store = torch.distributed.TCPStore(
            host_name=ip,
            port=port + 1,
            world_size=world_size,
            is_master=(global_rank == 0),
        )

        group_name = "refit"
        pg_prefix_store = PrefixStore(f"{group_name}/", store)

        # Training and inference workers run in separate torch.distributed worlds.
        # The public APIs (new_group, init_process_group) assume all ranks belong to one world;
        # new_group validates ranks against the default PG, and init_process_group can only
        # be called once. We construct the PG manually using the same internal pattern as
        # _new_process_group_helper, skipping the single-world assumptions.
        pg = ProcessGroup(pg_prefix_store, global_rank, world_size)
        gloo_store = PrefixStore("cpu/", pg_prefix_store)
        gloo_backend = ProcessGroupGloo(gloo_store, global_rank, world_size)
        gloo_backend._set_sequence_number_for_group()
        pg._register_backend(
            torch.device("cpu"),
            ProcessGroup.BackendType.GLOO,
            gloo_backend,
        )
        pg._set_default_backend(ProcessGroup.BackendType.GLOO)
        pg._set_group_name(group_name)

        self.refit_pg = pg

        # Register in torch.distributed's global state so that high-level ops
        # (all_gather_object, broadcast_object_list) work with this PG.
        _world.pg_group_ranks[pg] = {i: i for i in range(world_size)}
        _world.pg_map[pg] = ("gloo", pg_prefix_store)
        _world.pg_names[pg] = group_name

        if refit_backend == "nvshmem":
            from megatron.core.resharding.copy_services.nvshmem_copy_service import (
                NVSHMEMCopyService,
            )

            self.refit_copy_service = NVSHMEMCopyService(group=self.refit_pg)
        else:
            from megatron.core.resharding.copy_services.gloo_copy_service import (
                GlooCopyService,
            )

            self.refit_copy_service = GlooCopyService(group=self.refit_pg)

        from megatron.core.resharding.refit import prepare_swap_model_weights

        is_source = rank_offset == 0
        dst_rank_offset = (
            torch.distributed.get_world_size() if is_source else rank_offset
        )

        # Build and cache the reshard plan (and any MXFP8 transforms) collectively.
        # All participating ranks (training + generation) call this simultaneously.
        prepare_swap_model_weights(
            src_model=self.model if is_source else None,
            target_model=None if is_source else self.model,
            group=self.refit_pg,
            src_rank_offset=0,
            dst_rank_offset=dst_rank_offset,
        )

    def preinit_nvshmem_collective(self) -> None:
        """Initialize NVShmem collectively before any weight transfer.

        Must be called on ALL participating ranks (training + inference) simultaneously,
        after `prepare_for_generation()` has completed and the CG has been recorded.
        The `NVSHMEMCopyService` lazy init can corrupt CUDA graph state.
        """
        if not hasattr(self, "refit_copy_service"):
            return
        if not hasattr(self.refit_copy_service, "_ensure_initialized"):
            return
        self.refit_copy_service._ensure_initialized()

    def swap_weights_via_reshard(
        self, is_source: bool, dst_rank_offset: int = 0
    ) -> bool:
        """Transfer weights using Megatron's ``swap_model_weights`` API.

        Args:
            is_source: True for training workers (senders), False for inference workers (receivers).
            dst_rank_offset: Rank offset of the inference (destination) side.

        Returns:
            True on success.
        """
        from megatron.core.resharding.refit import swap_model_weights

        src_model = self.model if is_source else None
        dst_model = None if is_source else self.model

        swap_model_weights(
            src_model,
            dst_model,
            refit_method=self.refit_copy_service,
            group=self.refit_pg,
            src_rank_offset=0,
            dst_rank_offset=dst_rank_offset,
        )

        return True

    def suspend_for_refit(self) -> None:
        """Pause+suspend the inference engine before a weight refit."""
        if not self._inference_engine_initialized:
            return
        self._sleep()
        torch.cuda.synchronize()

    def resume_after_refit(self) -> None:
        """Resume+unpause the inference engine after a weight refit."""
        if not self._inference_engine_initialized:
            return
        self._wake()
