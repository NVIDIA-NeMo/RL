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
import os
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
from megatron.core.utils import unwrap_model

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationOutputSpec,
    verify_right_padding,
)
from nemo_rl.models.generation.megatron.utils import (
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
        self._inference_engine_asleep = (
            True  # Start paused since we begin with training
        )
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
        enable_chunked_prefill = mcore_generation_config["enable_chunked_prefill"]
        use_cuda_graphs_for_non_decode_steps = mcore_generation_config[
            "use_cuda_graphs_for_non_decode_steps"
        ]
        max_tokens = mcore_generation_config["max_tokens"]

        # The value may be overwritten by `recompute_kv_cache_after_weight_updates`.
        kv_cache_management_mode = mcore_generation_config["kv_cache_management_mode"]
        needs_static_kv_pointers = kv_cache_management_mode != "persist"

        materialize_only_last_token_logits = mcore_generation_config[
            "materialize_only_last_token_logits"
        ]
        num_speculative_tokens = mcore_generation_config["num_speculative_tokens"]
        max_requests = mcore_generation_config.get("max_requests")

        mamba_inference_state_config = MambaInferenceStateConfig.from_model(self.model)
        is_hybrid_model = mamba_inference_state_config is not None
        if is_hybrid_model:
            if (
                mcore_generation_config.get("mamba_inference_ssm_states_dtype")
                is not None
            ):
                mamba_inference_state_config.ssm_states_dtype = resolve_torch_dtype(
                    mcore_generation_config["mamba_inference_ssm_states_dtype"]
                )
            if (
                mcore_generation_config.get("mamba_inference_conv_states_dtype")
                is not None
            ):
                mamba_inference_state_config.conv_states_dtype = resolve_torch_dtype(
                    mcore_generation_config["mamba_inference_conv_states_dtype"]
                )

        # logging_step_interval is a power-user argument that should be NotRequired.
        logging_step_interval = mcore_generation_config.get("logging_step_interval")
        # This will be fixed in upstream MCore, allowing an argument of `None`.
        if logging_step_interval is None:
            logging_step_interval = 0

        # flashinfer's fused-RoPE kernel only dispatches fp16/bf16 q/k.
        use_flashinfer_fused_rope = self.model.config.params_dtype in (
            torch.float16,
            torch.bfloat16,
        )

        inference_config = InferenceConfig(
            block_size_tokens=block_size_tokens,
            buffer_size_gb=buffer_size_gb,
            num_cuda_graphs=num_cuda_graphs,
            max_tokens=max_tokens,
            max_sequence_length=mcore_generation_config["max_model_len"],
            kv_cache_management_mode=KVCacheManagementMode(kv_cache_management_mode),
            static_kv_memory_pointers=needs_static_kv_pointers,
            use_cuda_graphs_for_non_decode_steps=use_cuda_graphs_for_non_decode_steps,
            use_flashinfer_fused_rope=use_flashinfer_fused_rope,
            sampling_backend="flashinfer",
            use_synchronous_zmq_collectives=True,
            materialize_only_last_token_logits=materialize_only_last_token_logits,
            enable_chunked_prefill=enable_chunked_prefill,
            enable_prefix_caching=mcore_generation_config["enable_prefix_caching"],
            prefix_caching_coordinator_policy=PrefixCachingCoordinatorPolicy(
                "first_prefix_block"
            ),
            pg_collection=pg_collection,
            mamba_inference_state_config=mamba_inference_state_config,
            # Reserve more KV-cache space when speculative decoding is enabled.
            mamba_memory_ratio=(
                0.1 + 0.1 * num_speculative_tokens if is_hybrid_model else None
            ),
            logging_step_interval=logging_step_interval,
            num_speculative_tokens=num_speculative_tokens,
            logprobs_mode="processed_logprobs",
            max_requests=max_requests,
        )

        if "inference_cuda_graph_scope" in mcore_generation_config:
            self.model.config.inference_cuda_graph_scope = InferenceCudaGraphScope[
                mcore_generation_config["inference_cuda_graph_scope"]
            ]

        self.inference_context = DynamicInferenceContext(
            self.model.config, inference_config
        )
        self.inference_wrapped_model = GPTInferenceWrapper(
            self.model, self.inference_context
        )
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

    async def _start_inference_coordinator(self):
        """Start the inference coordinator and engine loop."""
        self.coordinator_addr = await self.dynamic_inference_engine.start_listening_to_data_parallel_coordinator(
            inference_coordinator_port=None,
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
        future = asyncio.run_coroutine_threadsafe(
            self._sleep_engine(), self._inference_loop
        )
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
        future = asyncio.run_coroutine_threadsafe(
            self._wake_engine(), self._inference_loop
        )
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
        # CUDA current_device is per-thread.
        # The worker's __init__ thread called set_device(LOCAL_RANK), and this thread must match.
        local_rank = int(os.environ["LOCAL_RANK"])

        def run_loop():
            torch.cuda.set_device(local_rank)
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
            parsers=self.cfg["generation"]["mcore_generation_config"]["parsers"],
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

    def _run_async_coordinator_start(self):
        """Start the coordinator and engine loop in the background thread."""
        if self._inference_loop is None:
            self._start_inference_loop_thread()

        future = asyncio.run_coroutine_threadsafe(
            self._start_inference_coordinator(), self._inference_loop
        )
        # _start_inference_coordinator awaits RUNNING, so future.result() only returns once
        # this rank's engine is fully warmed up. Cross-rank sync is handled by Ray's actor
        # group semantics (the caller waits for all workers' prepare_for_generation).
        future.result()
        print(f"[Rank {torch.distributed.get_rank()}] Coordinator started")

        if (
            self.cfg["generation"]["mcore_generation_config"]["expose_http_server"]
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

        lang_module = unwrap_model(self.model)

        if self.is_generation_colocated:
            if self._inference_engine_initialized and not self._inference_engine_asleep:
                self._sleep()
            cuda_graph_impl = self.cfg["generation"]["mcore_generation_config"][
                "cuda_graph_impl"
            ]
            if cuda_graph_impl != "none":
                toggle_cuda_graphs(lang_module, set_to="none")

        rotary_module = getattr(lang_module, "rotary_pos_emb", None)
        if rotary_module is not None and hasattr(
            rotary_module.forward, "cache_parameters"
        ):
            rotary_module.forward.cache_clear()

        if self.is_generation_colocated:
            gc.collect()
            torch.cuda.empty_cache()

        log_gpu_memory("finish_generation END")

    def prepare_for_generation(self, tags=None, **kwargs) -> None:
        """Enter inference mode and start (or wake) the inference engine.

        Called in both colocated and non-colocated setups.
        Even in non-colocated mode, Megatron's engine has to be intentionally paused before a refit
        (and its weights are not detachable), so we have to switch modes around every refit.
        """
        log_gpu_memory("prepare_for_generation START")
        mcore_generation_config = self.cfg["generation"]["mcore_generation_config"]

        self.model.config.flash_decode = False
        if self.is_generation_colocated and self.should_disable_forward_pre_hook:
            # Bring offloaded params back to CUDA before colocated generation.
            self.model = self.move_model(
                self.model, "cuda", move_params=True, move_grads=False
            )
            # DP inference schedules requests independently, so a forward pre-hook
            # cannot safely launch a parameter all-gather from only the rank that
            # received work. Gather once across every worker, then keep the hooks
            # disabled until the next training step completes.
            if self._forward_pre_hook_enabled():
                self._disable_forward_pre_hook_until_next_train_step(param_sync=True)

        lang_module = unwrap_model(self.model)
        lang_module.eval()

        rotary_module = getattr(lang_module, "rotary_pos_emb", None)
        if rotary_module is not None and hasattr(
            rotary_module.forward, "cache_parameters"
        ):
            rotary_module.forward.cache_clear()

        cuda_graph_impl = mcore_generation_config["cuda_graph_impl"]
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
                self._run_async_coordinator_start()
            else:
                self._wake()

        log_gpu_memory("prepare_for_generation END")

    def report_dp_openai_server_base_url(self) -> Optional[str]:
        """Return this worker's OpenAI server base URL (None if not the leader)."""
        return self.base_url

    def _build_sampling_params(
        self, greedy: bool, stop_words: Optional[list[str]]
    ) -> SamplingParams:
        """Build mcore SamplingParams for a single request."""
        top_k_cfg = self.cfg["generation"]["top_k"]
        top_k_val = 1 if greedy else (int(top_k_cfg) if top_k_cfg is not None else 0)

        top_p_cfg = self.cfg["generation"]["top_p"]
        top_p_val = (
            0.0 if greedy else (float(top_p_cfg) if top_p_cfg is not None else 0.0)
        )

        return SamplingParams(
            temperature=self.cfg["generation"]["temperature"] if not greedy else 0,
            top_k=top_k_val,
            top_p=top_p_val,
            skip_prompt_log_probs=True,
            return_log_probs=True,
            num_tokens_to_generate=self.cfg["generation"]["max_new_tokens"],
            termination_id=self.megatron_tokenizer.eod,
            stop_words=stop_words,
        )

    def _merge_stop_strings(
        self, batch_stop_strings: Optional[list[Optional[list[str]]]]
    ) -> Optional[list[str]]:
        """Union the config's stop_strings with the given per-sample stop strings."""
        stop_set: set[str] = set()
        if self.cfg["generation"]["stop_strings"]:
            stop_set.update(self.cfg["generation"]["stop_strings"])
        if batch_stop_strings is not None:
            for sample_ss in batch_stop_strings:
                if sample_ss:
                    stop_set.update(sample_ss)
        return list(stop_set) if stop_set else None

    def _prepare_data_for_generation(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, list[SamplingParams]]:
        """Build the prompt tensors and a per-request SamplingParams for each sample."""
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

        prompt_tokens_tensor = data["input_ids"].cuda()
        prompt_lengths_tensor = data["input_lengths"]

        batch_stop_strings = data.get("stop_strings", [])
        sampling_params = []
        for i in range(prompt_tokens_tensor.size(0)):
            sample_stop_strings = (
                batch_stop_strings[i] if i < len(batch_stop_strings) else None
            )
            stop_words = self._merge_stop_strings(
                [sample_stop_strings] if sample_stop_strings else None
            )
            sampling_params.append(self._build_sampling_params(greedy, stop_words))

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
            # Take the prompt from the request we submitted rather than from the
            # engine's reply: mcore only echoes prompt_tokens back when
            # SamplingParams.return_prompt_tokens is set, and asking for them would
            # ship the whole prompt over ZMQ for data we already hold.
            prompt_len = input_lengths[i].item()
            generated_tokens = result[i].generated_tokens
            seq_len = prompt_len + len(generated_tokens)
            output_ids_padded[i, :prompt_len] = input_ids[i, :prompt_len]
            output_ids_padded[i, prompt_len:seq_len] = torch.tensor(
                generated_tokens, dtype=torch.long, device=input_ids.device
            )
            generation_lengths[i] = len(generated_tokens)
            unpadded_sequence_lengths[i] = seq_len
            gen_logprobs = result[i].generated_log_probs
            logprobs_padded[i, prompt_len : prompt_len + len(gen_logprobs)] = (
                torch.tensor(
                    gen_logprobs,
                    dtype=torch.float,
                    device=input_ids.device,
                )
            )

        out_dict = {
            "output_ids": output_ids_padded,
            "logprobs": logprobs_padded,
            "generation_lengths": generation_lengths,
            "unpadded_sequence_lengths": unpadded_sequence_lengths,
        }

        # PATCH(PR-4 router-replay-mcore): pack per-request routing indices
        # recorded by the mcore engine (model_config.moe_enable_routing_replay)
        # into GenerationOutputSpec["routed_experts"], mirroring the alignment
        # contract of vllm/utils.pad_and_align_routed_expert_indices:
        # row t = experts used when token t was processed as input; rows
        # [0, seq_len-1) are real, missing rows in that range use the all--1
        # sentinel (consumer falls back to fresh top-k), padding rows use
        # arange(topk) (masked out downstream, must merely be valid ids).
        # PATCH(det): route replay is proven redundant for the det stack (bitwise
        # logits => identical routes; certificate 2322656). The harvest also crashes
        # grpo message-log flattening when shards emit inconsistent shapes (job
        # 2322767: [seq,48,8] vs [seq]). Pack ONLY when explicitly requested.
        import os as _os_re
        routing_arrays = [getattr(r, "routing_indices", None) for r in result]
        if (_os_re.environ.get("NRL_PACK_ROUTED_EXPERTS", "0") == "1"
                and any(ra is not None for ra in routing_arrays)):
            template = next(ra for ra in routing_arrays if ra is not None)
            num_moe_layers, topk = int(template.shape[1]), int(template.shape[2])
            default_route = torch.arange(topk, dtype=torch.int32, device=input_ids.device)
            routed_experts_padded = (
                default_route.view(1, 1, 1, -1)
                .expand(batch_size, max_seq_len, num_moe_layers, topk)
                .clone()
            )
            for i in range(batch_size):
                expected_routes = int(unpadded_sequence_lengths[i].item()) - 1
                if expected_routes <= 0:
                    continue
                if routing_arrays[i] is None:
                    routed_experts_padded[i, :expected_routes] = -1
                    continue
                routed = torch.as_tensor(
                    routing_arrays[i], dtype=torch.int32, device=input_ids.device
                )
                routes_to_copy = min(expected_routes, routed.shape[0])
                routed_experts_padded[i, :routes_to_copy] = routed[:routes_to_copy]
                if routes_to_copy < expected_routes:
                    routed_experts_padded[i, routes_to_copy:expected_routes] = -1
            out_dict["routed_experts"] = routed_experts_padded

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

        tasks = [
            asyncio.create_task(_generate_single_item(i)) for i in range(data.size)
        ]
        for result in asyncio.as_completed(tasks):
            yield await result

        # PATCH(det quiesced probe, NRL_PROBE_QUIESCED=1): all generation for this call
        # has drained; the engine is idle. Three-way attribution on stashed sequences:
        #   P2  = decode-time generated_log_probs vs SOLO prefill score (quiet engine)
        #   P1  = SOLO score vs GROUP score (same kernels/phase, only co-batch differs)
        # plus route flips between solo and group scoring passes.
        if (
            os.environ.get("NRL_PROBE_QUIESCED", "0") == "1"
            and getattr(self, "_nrl_probe_stash", None)
        ):
            stash = self._nrl_probe_stash
            self._nrl_probe_stash = []
            n_probe = int(os.environ.get("NRL_PROBE_QUIESCED_N", "16"))
            probe_set = stash[:n_probe]

            def _mk_sp():
                sp = self._build_sampling_params(greedy=False, stop_words=None)
                sp.num_tokens_to_generate = 1
                sp.skip_prompt_log_probs = False
                return sp

            async def _score(tokens):
                fut = self.inference_client.add_request(tokens, _mk_sp())
                return await fut

            def _p2_stats(r, s):
                gen_lp = list(r.generated_log_probs)
                plp = list(s.prompt_log_probs or [])
                p_len = len(r.prompt_tokens)
                d = [
                    abs(gen_lp[j] - plp[p_len - 1 + j])
                    for j in range(len(gen_lp))
                    if p_len - 1 + j < len(plp)
                ]
                if not d:
                    return None
                return (sum(1 for x in d if x == 0.0), len(d), max(d))

            # PATCH(det route-replay-into-scoring, probe v7, NRL_PROBE_REPLAY=1):
            # after the solo pass, re-score each sequence with its generation-time
            # routes REPLAYED into the scoring prefill (partial-replay hook in
            # InferenceTopKRouter._forward, armed via
            # router_replay.nrl_partial_target). Arming is engine-LOCAL to this
            # rank-0 worker process, but the DP coordinator round-robins requests
            # over all 8 engines — so retry the (idempotent, quiesced-solo) scoring
            # request until the returned routing_indices verbatim-match the target
            # (proof the request prefilled on THIS rank with replay engaged).
            # Strictly one request in flight at a time; the pair phase is skipped.
            def _routers_in_layer_order():
                lang = unwrap_model(self.model)
                return [
                    m
                    for _n, m in lang.named_modules()
                    if getattr(m, "router_replay", None) is not None
                ]

            async def _replay_score_one(r, so, routers):
                import numpy as _np

                ri = getattr(r, "routing_indices", None)
                if ri is None:
                    print("[NRL_RPROBE] skip: request has no routing_indices", flush=True)
                    return
                ri = _np.asarray(ri)
                n_rows, n_layers, _topk = ri.shape
                if len(routers) != n_layers:
                    print(
                        f"[NRL_RPROBE] skip: {len(routers)} armed routers != "
                        f"{n_layers} recorded layers",
                        flush=True,
                    )
                    return
                tgt = torch.from_numpy(_np.ascontiguousarray(ri)).to(
                    device="cuda", dtype=torch.int64
                )  # [T-1, L, K]
                layer_tgts = [tgt[:, li, :].contiguous() for li in range(n_layers)]
                full = r.prompt_tokens.tolist() + list(r.generated_tokens)
                max_attempts = int(os.environ.get("NRL_RPROBE_MAX_ATTEMPTS", "10"))
                res, engaged, attempts = None, False, 0
                for _attempt in range(max_attempts):
                    attempts = _attempt + 1
                    for rt, lt in zip(routers, layer_tgts):
                        rt.router_replay.nrl_partial_target = lt
                    try:
                        res = await _score(full)
                    finally:
                        for rt in routers:
                            rt.router_replay.nrl_partial_target = None
                    rs = getattr(res, "routing_indices", None)
                    if (
                        rs is not None
                        and rs.shape[0] >= n_rows
                        and _np.array_equal(
                            _np.asarray(rs)[:n_rows].astype(_np.int64),
                            ri.astype(_np.int64),
                        )
                    ):
                        engaged = True
                        break
                # Route flips vs generation (top-k SET comparison, as in earlier probes).
                rs = getattr(res, "routing_indices", None)
                rflip, m = -1, 0
                if rs is not None:
                    rs = _np.asarray(rs)
                    m = min(n_rows, rs.shape[0])
                    neq = _np.sort(rs[:m], -1) != _np.sort(ri[:m], -1)
                    rflip = int(neq.any(-1).any(-1).sum())
                # Per-token |dlp| distribution: decode-time logprobs vs replay-scored prefill.
                gen_lp = list(r.generated_log_probs)
                plp = list(res.prompt_log_probs or [])
                p_len = len(r.prompt_tokens)
                d = [
                    abs(gen_lp[j] - plp[p_len - 1 + j])
                    for j in range(len(gen_lp))
                    if p_len - 1 + j < len(plp)
                ]
                dist = ""
                if d:
                    ds = sorted(d)
                    dist = (
                        f" mean={sum(d) / len(d):.3e} p50={ds[len(ds) // 2]:.3e}"
                        f" p90={ds[min(int(len(ds) * 0.9), len(ds) - 1)]:.3e}"
                        f" p99={ds[min(int(len(ds) * 0.99), len(ds) - 1)]:.3e}"
                    )
                p2r = (
                    (sum(1 for x in d if x == 0.0), len(d), max(d)) if d else None
                )
                # Solo(unarmed) score vs replay score — same phase/kernels, routes pinned.
                so_lp = list(so.prompt_log_probs or [])
                nn = min(len(so_lp), len(plp))
                d1 = [abs(so_lp[j] - plp[j]) for j in range(nn)]
                print(
                    "[NRL_RPROBE] P2r(gen-vs-replayscore): "
                    + (f"exact={p2r[0]}/{p2r[1]} max={p2r[2]:.3e}" if p2r else "n/a")
                    + f" route_flips={rflip}/{m}{dist}"
                    + f" engaged={int(engaged)} attempts={attempts}"
                    + f" | P1r(solo-vs-replay): exact={sum(1 for x in d1 if x == 0.0)}/{nn}"
                    + (f" max={max(d1):.3e}" if d1 else " max=n/a"),
                    flush=True,
                )

            async def _quiesced_probe():
                # Serialize probes: multiple generate_async calls can schedule
                # probe coroutines on this loop; replay arming is engine-global
                # state, and solo scoring must truly be one request at a time.
                if not hasattr(self, "_nrl_probe_gate"):
                    self._nrl_probe_gate = asyncio.Lock()
                async with self._nrl_probe_gate:
                    await _quiesced_probe_body()

            async def _quiesced_probe_body():
                solo_results = []
                for r in probe_set:
                    full = r.prompt_tokens.tolist() + list(r.generated_tokens)
                    solo_results.append(await _score(full))  # strictly one at a time
                if os.environ.get("NRL_PROBE_REPLAY", "0") == "1":
                    routers = _routers_in_layer_order()
                    for r, so in zip(probe_set, solo_results):
                        try:
                            await _replay_score_one(r, so, routers)
                        except Exception:
                            import traceback

                            traceback.print_exc()
                    return  # replay experiment: never submit pairs
                # PATCH(probe v6): PAIR-WISE group submission — one pair in flight at a
                # time, so any merged (~1024-row) dump pass between the pair markers is
                # attributable to exactly these two sequences.
                group_results = []
                for _i in range(0, len(probe_set), 2):
                    _chunk = probe_set[_i : _i + 2]
                    print(f"[NRL_QPROBE_PAIR] indices={_i},{_i+1}", flush=True)
                    _rs = await asyncio.gather(
                        *[
                            _score(r.prompt_tokens.tolist() + list(r.generated_tokens))
                            for r in _chunk
                        ]
                    )
                    group_results.extend(_rs)
                for r, so, gr in zip(probe_set, solo_results, group_results):
                    p2 = _p2_stats(r, so)
                    so_lp = list(so.prompt_log_probs or [])
                    gr_lp = list(gr.prompt_log_probs or [])
                    n = min(len(so_lp), len(gr_lp))
                    d1 = [abs(so_lp[j] - gr_lp[j]) for j in range(n)]
                    p1_exact = sum(1 for x in d1 if x == 0.0)
                    p1_max = max(d1) if d1 else -1.0
                    rflip = -1
                    layer_info = ""
                    ri_s, ri_g = getattr(so, "routing_indices", None), getattr(
                        gr, "routing_indices", None
                    )
                    if ri_s is not None and ri_g is not None:
                        import numpy as _np

                        m = min(ri_s.shape[0], ri_g.shape[0])
                        neq = _np.sort(ri_s[:m], -1) != _np.sort(ri_g[:m], -1)
                        per_layer = neq.any(-1).sum(axis=0)  # [L] tokens flipped per layer
                        rflip = int(neq.any(-1).any(-1).sum())
                        nz = _np.nonzero(per_layer)[0]
                        first_l = int(nz[0]) if nz.size else -1
                        L = per_layer.shape[0]
                        q = max(L // 4, 1)
                        buckets = [int(per_layer[i : i + q].sum()) for i in range(0, L, q)]
                        layer_info = (
                            f" first_flip_layer={first_l}"
                            f" l0-3={[int(per_layer[i]) for i in range(min(4, L))]}"
                            f" layer_buckets={buckets}"
                        )
                    print(
                        f"[NRL_QPROBE] P2(gen-vs-solo): "
                        + (f"exact={p2[0]}/{p2[1]} max={p2[2]:.3e}" if p2 else "n/a")
                        + f" | P1(solo-vs-group): exact={p1_exact}/{n} max={p1_max:.3e}"
                        f" route_flips={rflip}/{m if rflip >= 0 else 0}" + layer_info,
                        flush=True,
                    )

            future = asyncio.run_coroutine_threadsafe(
                _quiesced_probe(), self._inference_loop
            )
            await asyncio.wrap_future(future)

    async def _generate_with_persistent_engine(
        self,
        prompt_tokens_tensor: torch.Tensor,
        prompt_lengths_tensor: torch.Tensor,
        sampling_params: list[SamplingParams],
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
        for prompt_tokens, prompt_len, request_sampling_params in zip(
            prompt_tokens_tensor, prompt_lengths_tensor, sampling_params, strict=True
        ):
            prompt = prompt_tokens[: prompt_len.item()].tolist()
            futures.append(
                self.inference_client.add_request(prompt, request_sampling_params)
            )

        results: list[DynamicInferenceRequest] = await asyncio.gather(*futures)
        print(f"[Rank {dist_rank}] Completed {len(results)} requests")

        # PATCH(det self-score probe, NRL_PROBE_SELF_SCORE=1): re-score each finished
        # sequence through the SAME engine's prefill and diff against the decode-time
        # generated_log_probs. Measures the engine's decode-vs-prefill self-consistency —
        # the floor that engine-based logprob scoring (fused-logprob option A) can reach.
        # Requires mcore_generation_config.materialize_only_last_token_logits=false
        # (engine asserts otherwise). Alignment: scoring prompt_log_probs[j] = logprob of
        # token j+1 given prefix; generated token P+j (0-based) -> prompt_log_probs[P-1+j].
        if os.environ.get("NRL_PROBE_QUIESCED", "0") == "1":
            # PATCH(det quiesced probe): defer scoring to the post-generation quiet
            # window (see generate_async) instead of interleaving with live decode.
            if not hasattr(self, "_nrl_probe_stash"):
                self._nrl_probe_stash = []
            self._nrl_probe_stash.extend(results)
        elif os.environ.get("NRL_PROBE_SELF_SCORE", "0") == "1":
            score_futures = []
            for r in results:
                full_tokens = r.prompt_tokens.tolist() + list(r.generated_tokens)
                sp = self._build_sampling_params(greedy=False, stop_words=None)
                sp.num_tokens_to_generate = 1
                sp.skip_prompt_log_probs = False
                score_futures.append(self.inference_client.add_request(full_tokens, sp))
            score_results = await asyncio.gather(*score_futures)
            for r, s in zip(results, score_results):
                gen_lp = list(r.generated_log_probs)
                plp = s.prompt_log_probs
                if plp is None:
                    print("[NRL_SELF_SCORE] prompt_log_probs is None — set "
                          "materialize_only_last_token_logits=false", flush=True)
                    break
                plp = list(plp)
                p_len = len(r.prompt_tokens)
                diffs = [
                    abs(gen_lp[j] - plp[p_len - 1 + j])
                    for j in range(len(gen_lp))
                    if p_len - 1 + j < len(plp)
                ]
                if diffs:
                    n_exact = sum(1 for d in diffs if d == 0.0)
                    print(
                        f"[NRL_SELF_SCORE] G={len(diffs)} max={max(diffs):.3e} "
                        f"mean={sum(diffs)/len(diffs):.3e} exact={n_exact}/{len(diffs)}",
                        flush=True,
                    )
                # PATCH(det self-score probe v2): route-flip attribution. Both requests
                # record routing (moe_enable_routing_replay); routing row i = experts
                # used processing input token i, so generated token j uses row p_len-1+j
                # (same row the logprob comes from). A token is "flipped" if ANY MoE
                # layer's top-k SET differs between the decode-time and scoring passes.
                ri_g = getattr(r, "routing_indices", None)
                ri_s = getattr(s, "routing_indices", None)
                if ri_g is not None and ri_s is not None and diffs:
                    import numpy as _np

                    n_rows = min(len(diffs), ri_g.shape[0] - (p_len - 1), ri_s.shape[0] - (p_len - 1))
                    flip_d, same_d = [], []
                    layer_flips = _np.zeros(ri_g.shape[1], dtype=_np.int64)
                    for j in range(max(n_rows, 0)):
                        rg = _np.sort(ri_g[p_len - 1 + j], axis=-1)
                        rs = _np.sort(ri_s[p_len - 1 + j], axis=-1)
                        per_layer = (rg != rs).any(axis=-1)
                        if per_layer.any():
                            flip_d.append(diffs[j])
                            layer_flips += per_layer
                        else:
                            same_d.append(diffs[j])
                    if flip_d or same_d:
                        same_exact = sum(1 for d in same_d if d == 0.0)
                        top_layers = _np.argsort(layer_flips)[::-1][:3]
                        # v3: prompt-region control — positions 0..P-2 were computed in
                        # PREFILL mode by BOTH passes; their flip rate is the
                        # prefill-vs-prefill baseline (and a row-alignment sanity check:
                        # if ~= generated-region rate, suspect misalignment or pure
                        # batch-composition variance).
                        n_prompt_rows = min(p_len - 1, ri_g.shape[0], ri_s.shape[0])
                        pg = _np.sort(ri_g[:n_prompt_rows], axis=-1)
                        ps = _np.sort(ri_s[:n_prompt_rows], axis=-1)
                        prompt_flips = int((pg != ps).any(axis=-1).any(axis=-1).sum())
                        print(
                            f"[NRL_ROUTE_DIFF] flip={len(flip_d)}/{len(flip_d)+len(same_d)} "
                            f"dlp_flip={sum(flip_d)/max(len(flip_d),1):.3e} "
                            f"dlp_same={sum(same_d)/max(len(same_d),1):.3e} "
                            f"exact_same={same_exact}/{max(len(same_d),1)} "
                            f"prompt_flip={prompt_flips}/{n_prompt_rows} "
                            f"top_flip_layers={[(int(l), int(layer_flips[l])) for l in top_layers if layer_flips[l] > 0]}",
                            flush=True,
                        )
        return results


class MegatronGenerationRefitMixin:
    """Refit collective, weight transfer, and engine suspend/resume around refits."""

    def init_collective_mcore_generation(
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
            refit_backend: Copy-service backend ("gloo", "nccl", or "nvshmem").
        """
        from torch.distributed.distributed_c10d import (
            PrefixStore,
            ProcessGroup,
            ProcessGroupGloo,
            _world,
        )

        local_rank = torch.distributed.get_rank()
        global_rank = local_rank + rank_offset

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

        # The NCCL copy service moves the actual weight bytes with CUDA-tensor P2P
        # (`torch.distributed.batch_isend_irecv`), which needs an NCCL backend
        # registered for the cuda device on this cross-world PG. GLOO stays the
        # default backend so the object collectives in `prepare_swap_model_weights`
        # (all_gather_object / broadcast_object_list) keep using CPU tensors.
        if refit_backend == "nccl":
            from torch.distributed.distributed_c10d import ProcessGroupNCCL

            # Ensure the NCCL communicator binds to this rank's own GPU.
            torch.cuda.set_device(torch.cuda.current_device())
            nccl_store = PrefixStore("cuda/", pg_prefix_store)
            nccl_options = ProcessGroupNCCL.Options()
            nccl_backend = ProcessGroupNCCL(
                nccl_store, global_rank, world_size, nccl_options
            )
            nccl_backend._set_sequence_number_for_group()
            pg._register_backend(
                torch.device("cuda"),
                ProcessGroup.BackendType.NCCL,
                nccl_backend,
            )

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
        elif refit_backend == "nccl":
            from megatron.core.resharding.copy_services.nccl_copy_service import (
                NCCLCopyService,
            )

            self.refit_copy_service = NCCLCopyService(group=self.refit_pg)
        else:
            from megatron.core.resharding.copy_services.gloo_copy_service import (
                GlooCopyService,
            )

            self.refit_copy_service = GlooCopyService(group=self.refit_pg)

        from megatron.core.resharding.refit import prepare_swap_model_weights

        is_source = rank_offset == 0
        # Cache for later refit calls (swap_weights_via_reshard).
        self.refit_dst_rank_offset = (
            torch.distributed.get_world_size() if is_source else rank_offset
        )

        # Build and cache the reshard plan (and any MXFP8 transforms) collectively.
        # All participating ranks (training + generation) call this simultaneously.
        prepare_swap_model_weights(
            src_model=self.model if is_source else None,
            target_model=None if is_source else self.model,
            group=self.refit_pg,
            src_rank_offset=0,
            dst_rank_offset=self.refit_dst_rank_offset,
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

    def swap_weights_via_reshard(self, is_source: bool) -> bool:
        """Transfer weights using Megatron's `swap_model_weights` API.

        Args:
            is_source: True for training workers (senders), False for inference workers (receivers).

        Returns:
            True on success.
        """
        from megatron.core.resharding.refit import swap_model_weights

        # PATCH(golden refit-param-sync, upstream #13, ported from the det-TE golden
        # worker): with distributed optimizer + overlap_param_gather, the post-step
        # param all-gather completes lazily via TRAINING forward pre-hooks; refit can
        # read param.data before they fire, shipping a bucket-wise MIX of theta_k and
        # theta_{k-1} to the gen model (multi-step gen-KL drift). Force param-sync
        # completion on the SRC before reading weights.
        import os as _os_ps
        if (
            _os_ps.environ.get("NRL_REFIT_PARAM_SYNC", "0") == "1"
            and is_source
            and getattr(self, "should_disable_forward_pre_hook", False)
        ):
            _resync = self._forward_pre_hook_enabled()
            if _resync:
                self.disable_forward_pre_hook(param_sync=True)
                self.enable_forward_pre_hook()
            if not getattr(type(self), "_nrl_ps_banner", False):
                type(self)._nrl_ps_banner = True
                print(
                    f"[NRL_REFIT_PARAM_SYNC] forced param-gather completion before refit (hook_was_enabled={_resync})",
                    flush=True,
                )

        src_model = self.model if is_source else None
        dst_model = None if is_source else self.model

        swap_model_weights(
            src_model,
            dst_model,
            refit_method=self.refit_copy_service,
            group=self.refit_pg,
            src_rank_offset=0,
            dst_rank_offset=self.refit_dst_rank_offset,
        )

        # PATCH(NRL_REFIT_CKSUM): after refit, print per-group checksums of params
        # AND buffers on both sides — any src/dst mismatch = the refit gap that
        # explains gen-vs-scoring logprob residuals with all shapes/kernels aligned.
        import os as _os_ck
        if _os_ck.environ.get("NRL_REFIT_CKSUM", "0") == "1":
            _m = self.model
            _mm = _m[0] if isinstance(_m, list) else _m
            while hasattr(_mm, "module"):
                _mm = _mm.module
            import hashlib as _hl
            _side = "SRC" if is_source else "DST"
            _lines = []
            for _kind, _iter in (("P", _mm.named_parameters()), ("B", _mm.named_buffers())):
                for _n, _t in _iter:
                    _h = _hl.md5(_t.detach().float().cpu().numpy().tobytes()).hexdigest()[:12]
                    _lines.append(f"{_side} {_kind}:{_n} {tuple(_t.shape)} {_h}")
            _dir = _os_ck.environ.get("NRL_REFIT_CKSUM_DIR", "/tmp")
            _rank = _os_ck.environ.get("RANK", _os_ck.environ.get("RAY_RANK", "0"))
            _step = getattr(type(self), "_nrl_ck_step", 0)
            type(self)._nrl_ck_step = _step + 1
            import socket as _sk
            _fn = f"{_dir}/refit_cksum_{_side}_{_sk.gethostname()}_{_os_ck.getpid()}_s{_step}.txt"
            with open(_fn, "w") as _f:
                _f.write("\n".join(_lines))
            print(f"[NRL_REFIT_CKSUM] wrote {len(_lines)} entries -> {_fn}", flush=True)

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
