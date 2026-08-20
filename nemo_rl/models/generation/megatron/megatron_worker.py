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
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, AsyncGenerator, Literal, Optional

import requests
import torch
from megatron.core.inference.config import (
    InferenceConfig,
    KVCacheManagementMode,
    PrefixCachingCoordinatorPolicy,
)
from megatron.core.inference.engines.dynamic_engine import EngineState
from megatron.core.inference.quantization.mxfp8_tensor import MXFP8Tensor
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
from nemo_rl.utils.packed_tensor import packed_broadcast_consumer

if TYPE_CHECKING:
    from nemo_rl.weight_sync.nccl_reshard_utils import HFToLocalParamMap


G_VERIFY_MXFP8_ENV = "NRL_VERIFY_MEGATRON_MXFP8"


def _resolve_mxfp8_refit_backend(model_config: Any) -> str:
    """Resolve the MXFP8 storage required by the grouped-GEMM backend."""
    try:
        from megatron.core.inference.quantization.utils import (
            resolve_mxfp8_backend,
        )
    except ImportError:
        grouped_gemm_backend = getattr(
            model_config.inference_grouped_gemm_backend,
            "value",
            model_config.inference_grouped_gemm_backend,
        )
        if grouped_gemm_backend == "torch":
            return "triton"
        if grouped_gemm_backend == "flashinfer":
            return "flashinfer"
        raise ValueError(
            "MXFP8 inference does not support "
            f"inference_grouped_gemm_backend={grouped_gemm_backend!r}."
        )
    return resolve_mxfp8_backend(model_config.inference_grouped_gemm_backend)


def _refresh_generation_caches(model_chunks: list[torch.nn.Module]) -> None:
    """Refresh parameter-derived MCore caches after an in-place Bridge refit."""
    for model_chunk in model_chunks:
        for module in unwrap_model(model_chunk).modules():
            refresh_cache = getattr(module, "refresh_cache", None)
            if callable(refresh_cache):
                refresh_cache()
    torch.cuda.synchronize()


class _RefittableMXFP8Tensor(MXFP8Tensor):
    """MXFP8 storage with the logical tensor metadata Bridge needs for import."""

    def __init__(
        self,
        tensor: MXFP8Tensor,
        *,
        shape: torch.Size,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        super().__init__(data=tensor.data, scale=tensor.scale, backend=tensor.backend)
        self._logical_shape = shape
        self._logical_dtype = dtype
        self._logical_device = device

    @property
    def shape(self) -> torch.Size:
        return self._logical_shape

    @property
    def dtype(self) -> torch.dtype:
        return self._logical_dtype

    @property
    def device(self) -> torch.device:
        return self._logical_device


@dataclass
class _MegatronRefitTask:
    """A local Bridge import task and its persistent inference destination."""

    param_name: str
    mapping: Any
    megatron_module: torch.nn.Module
    destination: Any
    dependencies: tuple[str, ...]
    expected_shape: torch.Size
    target_id: int
    global_param_name: str | None = None
    is_mxfp8: bool = False


@dataclass
class _MegatronBulkRefitPiece:
    """One HF-local shard and the Megatron destination region it populates."""

    task: _MegatronRefitTask
    component: Literal["full", "gate", "up"]
    shape: torch.Size
    dtype: torch.dtype
    device: torch.device
    destination: torch.Tensor | None


def _verify_mxfp8_inference_weights(
    model: torch.nn.Module | list[torch.nn.Module] | tuple[torch.nn.Module, ...],
    expected_backend: str | None = None,
) -> int:
    """Fail unless an inference model contains Megatron MXFP8 weight objects.

    Inference-optimized MXFP8 weights are plain attributes rather than
    ``nn.Parameter`` objects, so ``named_parameters()`` cannot observe them.
    This check walks module attributes and nested containers, deduplicating
    objects that are also referenced by concatenated expert buffers.

    Args:
        model: Wrapped Megatron model or list of model chunks.

    Returns:
        Number of distinct MXFP8 weight objects found.

    Raises:
        RuntimeError: If no MXFP8 weights are present.
    """
    # This type is only available with Megatron inference dependencies loaded.
    from megatron.core.inference.quantization.mxfp8_tensor import (
        MXFP8Tensor,
        validate_mxfp8_tensor,
    )

    seen: dict[int, MXFP8Tensor] = {}

    def visit(value: object) -> None:
        if isinstance(value, MXFP8Tensor):
            seen[id(value)] = value
        elif isinstance(value, dict):
            for child in value.values():
                visit(child)
        elif isinstance(value, (list, tuple)):
            for child in value:
                visit(child)

    model_chunks = model if isinstance(model, (list, tuple)) else [model]
    for model_chunk in model_chunks:
        unwrapped_model = unwrap_model(model_chunk)
        for module in unwrapped_model.modules():
            for value in vars(module).values():
                visit(value)

    if not seen:
        raise RuntimeError(
            "Megatron MXFP8 verification failed: the inference model contains no "
            "megatron.core.inference.quantization.MXFP8Tensor weights."
        )
    if expected_backend is not None:
        for index, weight in enumerate(seen.values()):
            validate_mxfp8_tensor(
                weight,
                expected_backend=expected_backend,
                tensor_name=f"inference MXFP8 weight {index}",
            )
    return len(seen)


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
            logprobs_mode=mcore_generation_config["logprobs_mode"],
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

    def get_inference_runtime_info(self) -> dict[str, object]:
        """Return serializable state for inference-engine functional tests."""
        if not self._inference_engine_initialized:
            return {
                "initialized": False,
                "captured_graph_count": 0,
                "capture_stats": None,
                "step_count": 0,
                "using_cuda_graph_last_step": False,
                "padded_batch_dimensions": None,
            }

        engine = self.dynamic_inference_engine
        context = engine.context
        capture_stats = engine.capture_stats
        return {
            "initialized": True,
            "captured_graph_count": (
                len(context.cuda_graph_batch_dimensions_list)
                if capture_stats is not None
                else 0
            ),
            "capture_stats": capture_stats,
            "step_count": context.step_count,
            "using_cuda_graph_last_step": context.using_cuda_graph_this_step(),
            "padded_batch_dimensions": str(context.padded_batch_dimensions),
        }

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
        # paused and waking it now would race the weight transfer against CUDA-graph
        # replay. The subsequent
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
        return results


class MegatronNativeRefitMixin:
    """Megatron Core's native cross-world reshard/refit implementation."""

    def init_collective_mcore_generation(
        self,
        ip: str,
        port: int,
        world_size: int,
        rank_offset: int,
        refit_backend: str,
    ) -> None:
        """Initialize the native MCore refit collective and reshard plan."""
        # Native refit is optional; keep its private process-group machinery off
        # the default Megatron Bridge path.
        from torch.distributed.distributed_c10d import (
            PrefixStore,
            ProcessGroup,
            ProcessGroupGloo,
            _world,
        )

        local_rank = torch.distributed.get_rank()
        global_rank = local_rank + rank_offset
        store = torch.distributed.TCPStore(
            host_name=ip,
            port=port + 1,
            world_size=world_size,
            is_master=(global_rank == 0),
        )

        group_name = "refit"
        pg_prefix_store = PrefixStore(f"{group_name}/", store)
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

        if refit_backend == "nccl":
            from torch.distributed.distributed_c10d import ProcessGroupNCCL

            torch.cuda.set_device(torch.cuda.current_device())
            nccl_store = PrefixStore("cuda/", pg_prefix_store)
            nccl_backend = ProcessGroupNCCL(
                nccl_store,
                global_rank,
                world_size,
                ProcessGroupNCCL.Options(),
            )
            nccl_backend._set_sequence_number_for_group()
            pg._register_backend(
                torch.device("cuda"),
                ProcessGroup.BackendType.NCCL,
                nccl_backend,
            )

        pg._set_group_name(group_name)
        self.refit_pg = pg

        # High-level object collectives need this manually-created cross-world
        # process group registered in torch.distributed's global state.
        _world.pg_group_ranks[pg] = {rank: rank for rank in range(world_size)}
        _world.pg_map[pg] = ("gloo", pg_prefix_store)
        _world.pg_names[pg] = group_name

        if refit_backend == "nvshmem":
            from megatron.core.resharding.copy_services.nvshmem_copy_service import (
                NVSHMEMCopyService,
            )

            self.refit_copy_service = NVSHMEMCopyService(group=pg)
        elif refit_backend == "nccl":
            from megatron.core.resharding.copy_services.nccl_copy_service import (
                NCCLCopyService,
            )

            self.refit_copy_service = NCCLCopyService(group=pg)
        elif refit_backend == "gloo":
            from megatron.core.resharding.copy_services.gloo_copy_service import (
                GlooCopyService,
            )

            self.refit_copy_service = GlooCopyService(group=pg)
        else:
            raise ValueError(
                f"Unsupported native MCore refit backend: {refit_backend!r}."
            )

        from megatron.core.resharding.refit import prepare_swap_model_weights

        is_source = rank_offset == 0
        self.refit_dst_rank_offset = (
            torch.distributed.get_world_size() if is_source else rank_offset
        )
        prepare_swap_model_weights(
            src_model=self.model if is_source else None,
            target_model=None if is_source else self.model,
            group=pg,
            src_rank_offset=0,
            dst_rank_offset=self.refit_dst_rank_offset,
        )

        if not is_source and os.environ.get(G_VERIFY_MXFP8_ENV) == "1":
            model_chunks = (
                self.model if isinstance(self.model, (list, tuple)) else [self.model]
            )
            model_config = unwrap_model(model_chunks[0]).config
            expected_backend = _resolve_mxfp8_refit_backend(model_config)
            mxfp8_weight_count = _verify_mxfp8_inference_weights(
                self.model, expected_backend=expected_backend
            )
            print(
                f"NRL_MXFP8_VERIFY: PASS rank={self.rank} "
                f"weights={mxfp8_weight_count} backend={expected_backend}",
                flush=True,
            )

    def preinit_nvshmem_collective(self) -> None:
        """Initialize an NVSHMEM copy service outside CUDA graph capture."""
        copy_service = getattr(self, "refit_copy_service", None)
        if copy_service is not None and hasattr(copy_service, "_ensure_initialized"):
            copy_service._ensure_initialized()

    def swap_weights_via_reshard(self, is_source: bool) -> bool:
        """Transfer weights through Megatron Core's native refit plan."""
        from megatron.core.resharding.refit import swap_model_weights

        swap_model_weights(
            self.model if is_source else None,
            None if is_source else self.model,
            refit_method=self.refit_copy_service,
            group=self.refit_pg,
            src_rank_offset=0,
            dst_rank_offset=self.refit_dst_rank_offset,
        )
        return True


class MegatronGenerationRefitMixin(MegatronNativeRefitMixin):
    """Bridge/native refit implementations and inference-engine lifecycle."""

    @staticmethod
    def _hf_dependencies(mapping: Any) -> tuple[str, ...]:
        hf_param = mapping.hf_param
        names = (hf_param,) if isinstance(hf_param, str) else tuple(hf_param.values())
        return tuple(dict.fromkeys(names))

    def init_nccl_reshard_comm_groups_generation(
        self,
        pp_ips: list[str],
        pp_ports: list[int],
        pp_size: int,
        train_ranks_per_stage: int,
        sub_world_size: int,
    ) -> None:
        """Join every training PP stage's M-to-N communicator as a gen rank."""
        from nemo_rl.distributed.stateless_process_group import StatelessProcessGroup

        rank_in_group = train_ranks_per_stage + self.rank
        torch.cuda.empty_cache()
        self._generation_nccl_reshard_groups = {}
        for stage in range(pp_size):
            group = StatelessProcessGroup(
                master_address=pp_ips[stage],
                port=pp_ports[stage],
                rank=rank_in_group,
                world_size=sub_world_size,
            )
            group.init_nccl_communicator(device=torch.cuda.current_device())
            self._generation_nccl_reshard_groups[stage] = group

    def _commit_megatron_bulk_refit_piece(
        self, piece: _MegatronBulkRefitPiece, tensor: torch.Tensor
    ) -> None:
        """Commit one received HF-local piece to BF16 or persistent MXFP8 storage."""
        if tensor.shape != piece.shape:
            raise ValueError(
                f"Shape mismatch for Megatron bulk refit component {piece.component!r} "
                f"of {piece.task.param_name!r}: expected {tuple(piece.shape)}, "
                f"got {tuple(tensor.shape)}."
            )
        if not piece.task.is_mxfp8:
            assert piece.destination is not None
            piece.destination.copy_(tensor)
            return

        if piece.component == "full":
            self._write_generation_refit_weight(piece.task, tensor)
            return

        pending = self._generation_m2n_pending.setdefault(piece.task.target_id, {})
        pending[piece.component] = tensor
        if "gate" in pending and "up" in pending:
            fused = torch.cat([pending["gate"], pending["up"]], dim=0)
            self._write_generation_refit_weight(piece.task, fused)
            del self._generation_m2n_pending[piece.task.target_id]

    def _build_megatron_bulk_refit_map(
        self,
        refit_info: dict[str, Any],
        tasks: list[_MegatronRefitTask],
    ) -> tuple["HFToLocalParamMap", set[int]]:
        """Map M-to-N's canonical HF FFN shards onto local Megatron storage."""
        from megatron.bridge.models.conversion.param_mapping import (
            FusedExpertMapping,
            FusedGatedExpertMapping,
            GatedMLPMapping,
        )
        from megatron.bridge.utils.common_utils import (
            extract_expert_number_from_param,
        )
        from nemo_rl.weight_sync.nccl_reshard_utils import (
            HFToLocalParamMap,
            LocalParamSpec,
            RefitCtx,
            _INDIVIDUAL_EXPERT_RE,
            is_nccl_reshard_param,
        )

        pieces: dict[str, _MegatronBulkRefitPiece] = {}

        def add_piece(
            name: str,
            task: _MegatronRefitTask,
            component: Literal["full", "gate", "up"],
            shape: torch.Size,
            destination: torch.Tensor | None,
        ) -> None:
            if name in pieces:
                raise RuntimeError(f"Duplicate Megatron M-to-N target for {name!r}.")
            pieces[name] = _MegatronBulkRefitPiece(
                task=task,
                component=component,
                shape=shape,
                dtype=task.destination.dtype,
                device=task.destination.device,
                destination=destination,
            )

        for task in tasks:
            mapping = task.mapping
            destination = None if task.is_mxfp8 else task.destination

            if isinstance(mapping, GatedMLPMapping):
                if task.expected_shape[0] % 2:
                    raise ValueError(
                        f"Expected an even gated-MLP dimension for {task.param_name!r}."
                    )
                component_shape = torch.Size(
                    (task.expected_shape[0] // 2, *task.expected_shape[1:])
                )
                destination_parts = (
                    (None, None)
                    if destination is None
                    else tuple(torch.chunk(destination, 2, dim=0))
                )
                for component, target in zip(
                    ("gate", "up"), destination_parts, strict=True
                ):
                    name = mapping.hf_param[component]
                    if is_nccl_reshard_param(name):
                        add_piece(name, task, component, component_shape, target)
                continue

            if isinstance(mapping, FusedGatedExpertMapping):
                global_name = task.global_param_name or task.param_name
                expert_index = extract_expert_number_from_param(global_name)
                if task.expected_shape[0] % 2:
                    raise ValueError(
                        f"Expected an even fused-expert dimension for {task.param_name!r}."
                    )
                prefix = str(mapping.hf_param)[: -len(".gate_up_proj")]
                component_shape = torch.Size(
                    (task.expected_shape[0] // 2, *task.expected_shape[1:])
                )
                destination_parts = (
                    (None, None)
                    if destination is None
                    else tuple(torch.chunk(destination, 2, dim=0))
                )
                for component, target in zip(
                    ("gate", "up"), destination_parts, strict=True
                ):
                    name = f"{prefix}.{expert_index}.{component}_proj.weight"
                    add_piece(name, task, component, component_shape, target)
                continue

            if isinstance(mapping, FusedExpertMapping):
                global_name = task.global_param_name or task.param_name
                expert_index = extract_expert_number_from_param(global_name)
                prefix = str(mapping.hf_param)[: -len(".down_proj")]
                name = f"{prefix}.{expert_index}.down_proj.weight"
                add_piece(name, task, "full", task.expected_shape, destination)
                continue

            hf_param = mapping.hf_param
            if isinstance(hf_param, str) and is_nccl_reshard_param(hf_param):
                add_piece(hf_param, task, "full", task.expected_shape, destination)

        expert_pieces: dict[
            tuple[str, str], list[tuple[int, _MegatronBulkRefitPiece]]
        ] = {}
        for name, piece in pieces.items():
            match = _INDIVIDUAL_EXPERT_RE.match(name)
            if match:
                expert_pieces.setdefault((match.group(1), match.group(3)), []).append(
                    (int(match.group(2)), piece)
                )

        def grouped_spec(
            grouped: tuple[_MegatronBulkRefitPiece, ...],
        ) -> LocalParamSpec:
            first = grouped[0]

            def pre(_base: Any) -> RefitCtx:
                return RefitCtx(
                    buf=torch.empty(
                        (len(grouped), *first.shape),
                        dtype=first.dtype,
                        device=first.device,
                    )
                )

            def post(ctx: RefitCtx) -> None:
                for received, piece in zip(ctx.buf.unbind(0), grouped, strict=True):
                    self._commit_megatron_bulk_refit_piece(piece, received)

            return LocalParamSpec(base=None, pre=pre, post=post)

        def staged_spec(piece: _MegatronBulkRefitPiece) -> LocalParamSpec:
            def pre(_base: Any) -> RefitCtx:
                return RefitCtx(
                    buf=torch.empty(piece.shape, dtype=piece.dtype, device=piece.device)
                )

            def post(ctx: RefitCtx) -> None:
                self._commit_megatron_bulk_refit_piece(piece, ctx.buf)

            return LocalParamSpec(base=None, pre=pre, post=post)

        specs = {}
        bulk_target_ids: set[int] = set()
        for layer_name in refit_info["layer_names"]:
            for param_info in refit_info["per_layer_params"][layer_name]:
                name = param_info["name"]
                grouped_proj = param_info.get("grouped_expert_proj")
                if grouped_proj is not None:
                    prefix = name.rsplit(f".{grouped_proj}.weight", 1)[0]
                    grouped = tuple(
                        piece
                        for _, piece in sorted(
                            expert_pieces.get((prefix, grouped_proj), [])
                        )
                    )
                    if not grouped:
                        raise ValueError(
                            f"No local Megatron experts map to M-to-N weight {name!r}."
                        )
                    specs[name] = grouped_spec(grouped)
                    bulk_target_ids.update(piece.task.target_id for piece in grouped)
                    continue

                piece = pieces.get(name)
                if piece is None:
                    raise ValueError(
                        f"No local Megatron destination maps to M-to-N weight {name!r}."
                    )
                specs[name] = (
                    staged_spec(piece)
                    if piece.task.is_mxfp8
                    else LocalParamSpec(base=piece.destination)
                )
                bulk_target_ids.add(piece.task.target_id)

        return HFToLocalParamMap(specs=specs), bulk_target_ids

    def _prepare_mxfp8_refit(self, tasks: list[_MegatronRefitTask]) -> None:
        """Replace inference FP8 parameters with persistent MCore MXFP8 storage."""
        model_chunks = (
            self.model if isinstance(self.model, (list, tuple)) else [self.model]
        )
        cores = [unwrap_model(model) for model in model_chunks]
        needs_mxfp8 = any(
            getattr(core.config, "transformer_impl", None) == "inference_optimized"
            and getattr(core.config, "fp8_recipe", None) == "mxfp8"
            for core in cores
        )
        if not needs_mxfp8:
            return
        if len(cores) != 1:
            raise NotImplementedError(
                "Packed Megatron MXFP8 refit does not yet support virtual pipeline stages."
            )
        core = cores[0]
        if (
            getattr(core.config, "share_embeddings_and_output_weights", False)
            and getattr(core.config, "pipeline_model_parallel_size", 1) > 1
        ):
            raise NotImplementedError(
                "Packed Megatron MXFP8 refit does not yet support tied embeddings "
                "across pipeline stages."
            )
        if self._inference_engine_initialized:
            raise RuntimeError(
                "MXFP8 refit buffers must be prepared before inference-engine "
                "initialization; construct MegatronGeneration with skip_weight_load=True."
            )

        from megatron.bridge.models.conversion.utils import (
            get_module_and_param_from_name,
        )
        from megatron.core.inference.quantization.utils import (
            collect_mxfp8_param_metadata,
            quantize_params_to_mxfp8,
        )

        decoder = core.decoder if hasattr(core, "decoder") else core
        param_name_by_id = {
            id(param): name for name, param in decoder.named_parameters()
        }
        logical_metadata = collect_mxfp8_param_metadata(decoder)
        backend = _resolve_mxfp8_refit_backend(core.config)
        persistent_buffers = quantize_params_to_mxfp8(decoder, backend=backend)

        # Bridge mappings query the destination's logical shape/dtype/device.
        # MCore's MXFP8 wrapper intentionally exposes only physical data/scale,
        # so use a thin subclass that shares those exact persistent allocations.
        for name, tensor in tuple(persistent_buffers.items()):
            shape, dtype, device = logical_metadata[name]
            refittable = _RefittableMXFP8Tensor(
                tensor, shape=shape, dtype=dtype, device=device
            )
            module, current = get_module_and_param_from_name(decoder, name)
            if current is not tensor:
                raise RuntimeError(
                    f"MXFP8 destination changed while preparing {name!r}."
                )
            setattr(module, name.rsplit(".", 1)[-1], refittable)
            persistent_buffers[name] = refittable

        for task in tasks:
            decoder_name = param_name_by_id.get(task.target_id)
            if decoder_name not in persistent_buffers:
                continue
            task.is_mxfp8 = True
            task.destination = persistent_buffers[decoder_name]

        if os.environ.get(G_VERIFY_MXFP8_ENV) == "1":
            mxfp8_weight_count = _verify_mxfp8_inference_weights(
                self.model, expected_backend=backend
            )
            print(
                f"NRL_MXFP8_VERIFY: PASS rank={self.rank} "
                f"weights={mxfp8_weight_count} backend={backend}",
                flush=True,
            )

    def _build_generation_refit_tasks(
        self,
    ) -> tuple[list[torch.nn.Module], list[_MegatronRefitTask]]:
        """Resolve Bridge import tasks against the persistent inference model."""
        model_chunks = (
            list(self.model) if isinstance(self.model, (list, tuple)) else [self.model]
        )
        conversion_tasks = self.megatron_bridge.get_conversion_tasks(model_chunks)
        tasks: list[_MegatronRefitTask] = []
        for conversion_task in conversion_tasks:
            if conversion_task is None or conversion_task.megatron_module is None:
                continue
            destination = conversion_task.param_weight
            if destination is None:
                raise RuntimeError(
                    f"Bridge task {conversion_task.param_name!r} has no destination."
                )
            tasks.append(
                _MegatronRefitTask(
                    param_name=conversion_task.param_name,
                    mapping=conversion_task.mapping,
                    megatron_module=conversion_task.megatron_module,
                    destination=destination,
                    dependencies=self._hf_dependencies(conversion_task.mapping),
                    expected_shape=torch.Size(destination.shape),
                    target_id=id(destination),
                    global_param_name=conversion_task.global_param_name,
                )
            )

        if not tasks:
            raise RuntimeError("Megatron Bridge produced no local refit tasks.")
        return model_chunks, tasks

    def _install_generation_refit_plan(
        self,
        *,
        model_chunks: list[torch.nn.Module],
        tasks: list[_MegatronRefitTask],
        state_dict_info: dict[str, Any],
    ) -> None:
        """Install the packed-broadcast subset of a Bridge receive plan."""
        if not state_dict_info:
            raise ValueError(
                "Megatron packed refit requires non-empty state_dict_info."
            )

        required_names = {name for task in tasks for name in task.dependencies}
        missing_names = sorted(required_names.difference(state_dict_info))
        if missing_names:
            preview = ", ".join(missing_names[:20])
            raise ValueError(
                "Megatron refit metadata is missing Bridge inputs: "
                f"{preview}{' ...' if len(missing_names) > 20 else ''}"
            )

        self._generation_refit_model_chunks = model_chunks
        self._generation_refit_tasks = tasks
        self._generation_refit_dependency_counts = Counter(
            name for task in tasks for name in task.dependencies
        )
        self._generation_refit_state_dict_info = state_dict_info

    @torch.no_grad()
    def prepare_generation_refit_info(self, state_dict_info: dict[str, Any]) -> None:
        """Build the local HF-to-Megatron receive plan before the first refit."""
        model_chunks, tasks = self._build_generation_refit_tasks()
        self._prepare_mxfp8_refit(tasks)
        self._install_generation_refit_plan(
            model_chunks=model_chunks,
            tasks=tasks,
            state_dict_info=state_dict_info,
        )

    @torch.no_grad()
    def prepare_nccl_reshard_generation_refit_info(
        self, refit_info: dict[str, Any]
    ) -> None:
        """Prepare Megatron as the destination of NCCL M-to-N reshard refit."""
        from nemo_rl.weight_sync.nccl_reshard_utils import (
            _STR_TO_DTYPE,
            restore_refit_info_placements,
        )

        refit_info = restore_refit_info_placements(refit_info)
        model_chunks, tasks = self._build_generation_refit_tasks()
        self._prepare_mxfp8_refit(tasks)
        bulk_map, bulk_target_ids = self._build_megatron_bulk_refit_map(
            refit_info, tasks
        )

        misc_meta = refit_info.get("misc_meta", {})
        misc_state_dict_info = {
            name: (tuple(meta["shape"]), _STR_TO_DTYPE[meta["dtype"]])
            for name, meta in misc_meta.items()
        }
        misc_tasks = [task for task in tasks if task.target_id not in bulk_target_ids]
        self._install_generation_refit_plan(
            model_chunks=model_chunks,
            tasks=misc_tasks,
            state_dict_info=misc_state_dict_info,
        )
        self._generation_nccl_reshard_refit_info = refit_info
        self._generation_hf_to_local_param_map = bulk_map

    @torch.no_grad()
    def nccl_reshard_generation_refit(self) -> bool:
        """Receive bulk FFN shards via M-to-N, then import packed misc weights."""
        import os
        from collections import OrderedDict

        from nemo_rl.weight_sync.nccl_reshard_utils import RefitCtx
        from nemo_rl.weight_sync.xferdtensor import DTensorRef, xferdtensor

        self._generation_m2n_pending: dict[int, dict[str, torch.Tensor]] = {}

        def receive_one(param_info: dict[str, Any], group: Any, stream: Any) -> None:
            spec = self._generation_hf_to_local_param_map.get(param_info["name"])
            if spec is None:
                raise RuntimeError(
                    f"Megatron M-to-N refit has no destination for {param_info['name']!r}."
                )
            ctx = (
                spec.pre(spec.base) if spec.pre is not None else RefitCtx(buf=spec.base)
            )
            destination = DTensorRef(ctx.buf, param_info["global_shape"])
            xferdtensor(
                None,
                param_info["src_mesh_info"],
                param_info["src_placements"],
                destination,
                param_info["dst_mesh_info"],
                param_info["dst_placements"],
                group,
                stream,
            )
            if spec.post is not None:
                spec.post(ctx)

        stage_params: OrderedDict[int, list[dict[str, Any]]] = OrderedDict()
        refit_info = self._generation_nccl_reshard_refit_info
        for layer_name in refit_info["layer_names"]:
            for param_info in refit_info["per_layer_params"][layer_name]:
                stage_params.setdefault(param_info.get("pp_stage", 0), []).append(
                    param_info
                )

        num_streams = max(
            1,
            min(int(os.environ.get("NRL_REFIT_NUM_STREAMS", "2")), len(stage_params)),
        )
        streams = [torch.cuda.Stream() for _ in range(num_streams)]
        events: dict[int, torch.cuda.Event] = {}
        try:
            for index, (stage, params) in enumerate(stage_params.items()):
                previous_index = index - num_streams
                if previous_index in events:
                    events[previous_index].synchronize()
                stage_stream = streams[index % num_streams]
                with torch.cuda.stream(stage_stream):
                    group = self._generation_nccl_reshard_groups[stage]
                    for param_info in params:
                        receive_one(param_info, group, stage_stream)
                    event = torch.cuda.Event()
                    event.record()
                    events[index] = event

            torch.cuda.synchronize()
            if self._generation_m2n_pending:
                raise RuntimeError(
                    "Megatron M-to-N refit ended with incomplete fused MXFP8 weights: "
                    f"{sorted(self._generation_m2n_pending)}"
                )
            torch.cuda.empty_cache()
            return self.update_generation_weights_from_collective()
        finally:
            self._generation_m2n_pending.clear()

    def _write_generation_refit_weight(
        self, task: _MegatronRefitTask, converted_weight: torch.Tensor
    ) -> None:
        if converted_weight.shape != task.expected_shape:
            raise ValueError(
                f"Shape mismatch for Megatron parameter {task.param_name!r}: "
                f"expected {tuple(task.expected_shape)}, got {tuple(converted_weight.shape)}."
            )

        if task.is_mxfp8:
            converted_weight = converted_weight.to(torch.bfloat16).contiguous()
            quantized_weight = MXFP8Tensor.from_bf16(
                converted_weight, backend=task.destination.backend
            )
            task.destination.data.copy_(quantized_weight.data)
            task.destination.scale.view(torch.uint8).copy_(
                quantized_weight.scale.view(torch.uint8)
            )
        else:
            task.destination.copy_(converted_weight)

    def _load_generation_refit_batch(
        self, weights: list[tuple[str, torch.Tensor]]
    ) -> None:
        batch_stream = (
            torch.cuda.current_stream() if weights and weights[0][1].is_cuda else None
        )
        for name, tensor in weights:
            if self._generation_refit_remaining_dependencies.get(name, 0) > 0:
                self._generation_refit_pending_weights[name] = tensor
                if batch_stream is not None:
                    self._generation_refit_pending_streams[name] = batch_stream

        tasks = self._generation_refit_tasks
        while self._generation_refit_task_index < len(tasks):
            task = tasks[self._generation_refit_task_index]
            if any(
                name not in self._generation_refit_pending_weights
                for name in task.dependencies
            ):
                break

            # packed_broadcast_consumer alternates CUDA streams. A compound Bridge
            # mapping can retain one input across a batch boundary, so make the
            # current batch wait for the stream that received each retained input.
            current_stream = (
                torch.cuda.current_stream() if batch_stream is not None else None
            )
            for name in task.dependencies:
                source_stream = self._generation_refit_pending_streams.get(name)
                if current_stream is not None and source_stream is not None:
                    current_stream.wait_stream(source_stream)

            hf_weights = (
                self.megatron_bridge._model_bridge.maybe_modify_loaded_hf_weight(
                    task.mapping.hf_param, self._generation_refit_pending_weights
                )
            )
            converted_weight = task.mapping.hf_to_megatron(
                hf_weights, task.megatron_module
            )
            if converted_weight is None:
                raise RuntimeError(
                    f"Bridge produced no local value for {task.param_name!r}."
                )
            self._write_generation_refit_weight(task, converted_weight)

            for name in task.dependencies:
                self._generation_refit_remaining_dependencies[name] -= 1
                if self._generation_refit_remaining_dependencies[name] == 0:
                    del self._generation_refit_pending_weights[name]
                    self._generation_refit_pending_streams.pop(name, None)
            self._generation_refit_task_index += 1

    @torch.no_grad()
    def update_generation_weights_from_collective(self) -> bool:
        """Receive packed HF tensors and import them into the Megatron model."""
        if not hasattr(self, "_generation_refit_state_dict_info"):
            raise RuntimeError(
                "Megatron refit metadata is not prepared. Call prepare_refit_info first."
            )

        self._generation_refit_task_index = 0
        self._generation_refit_pending_weights: dict[str, torch.Tensor] = {}
        self._generation_refit_pending_streams: dict[str, torch.cuda.Stream] = {}
        self._generation_refit_remaining_dependencies = Counter(
            self._generation_refit_dependency_counts
        )
        try:
            packed_broadcast_consumer(
                iterator=iter(self._generation_refit_state_dict_info.items()),
                group=self.model_update_group,
                src=0,
                post_unpack_func=self._load_generation_refit_batch,
            )
            if self._generation_refit_task_index != len(self._generation_refit_tasks):
                task = self._generation_refit_tasks[self._generation_refit_task_index]
                missing = [
                    name
                    for name in task.dependencies
                    if name not in self._generation_refit_pending_weights
                ]
                raise RuntimeError(
                    f"Megatron refit ended before {task.param_name!r}; "
                    f"missing inputs: {missing}."
                )

            self.megatron_bridge._model_bridge._broadcast_shared_embeddings(
                self._generation_refit_model_chunks
            )
            _refresh_generation_caches(self._generation_refit_model_chunks)
            return True
        finally:
            self._generation_refit_pending_weights.clear()
            self._generation_refit_pending_streams.clear()

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
