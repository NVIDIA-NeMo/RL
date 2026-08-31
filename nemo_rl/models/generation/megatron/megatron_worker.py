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
import importlib
import os
import threading
import time
import warnings
from typing import Any, AsyncGenerator, Optional

import requests
import torch
from megatron.core.inference.config import (
    AsyncScheduleMode,
    InferenceConfig,
    KVCacheManagementMode,
    MambaInferenceStateConfig,
    PrefixCachingCoordinatorPolicy,
)
from megatron.core.inference.engines.dynamic_engine import EngineState
from megatron.core.inference.sampling_params import SamplingParams
from megatron.core.inference.utils import set_decode_expert_padding
from megatron.core.resharding.copy_services.gloo_copy_service import GlooCopyService
from megatron.core.resharding.copy_services.nccl_copy_service import NCCLCopyService
from megatron.core.resharding.refit import (
    prepare_swap_model_weights,
    swap_model_weights,
)
from megatron.core.transformer import MegatronModule
from megatron.core.transformer.cuda_graph_config import (
    normalize_inference_cuda_graph_scope,
)
from megatron.core.transformer.cuda_graphs import (
    CudaGraphManager,
    _CudagraphGlobalRecord,
)
from megatron.core.transformer.enums import InferenceCudaGraphScope
from megatron.core.transformer.module import GraphableMegatronModule
from megatron.core.transformer.utils import (
    set_model_config_attribute,
    toggle_cuda_graphs,
)
from megatron.core.utils import unwrap_model

from nemo_rl.data.multimodal_utils import CACHED_VIDEO_FRAME_MANIFEST_MAGIC
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationOutputSpec,
    verify_right_padding,
)
from nemo_rl.models.generation.megatron.utils import (
    build_image_preprocessing_config,
    build_prompt_and_multimodal_data,
    build_video_preprocessing_config,
    log_gpu_memory,
    resolve_torch_dtype,
    sample_vision_tensors,
)
from nemo_rl.models.megatron.memory_saver import (
    HAVE_TORCH_MEMORY_SAVER,
    pause_inference_weights,
    resume_inference_weights,
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
     - processor: optional multimodal processor.
     - is_generation_colocated: Whether colocated or distributed.
     - _reserved_http_server_socket: driver-reserved server socket, or None.
    """

    # Colocated-reshard hosts assign the dedicated inference-layout model here
    # (see MegatronPolicyWorkerImpl._build_colocated_inference_model).
    processor: Optional[Any] = None
    inference_model = None
    _colocated_reshard_plan = None

    def _gen_model(self) -> MegatronModule:
        """The model the inference engine wraps.

        Returns the dedicated inference-layout model when one exists (colocated
        reshard), otherwise the shared training model.
        """
        return self.inference_model if self.inference_model is not None else self.model

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

    def _get_megatron_inference_wrapper_cls(self) -> Optional[type]:
        """Resolve the configured Megatron inference wrapper, if any.

        Returns:
            The wrapper class, or None when no wrapper is configured.
        """
        class_path = self.cfg["generation"]["mcore_generation_config"].get(
            "megatron_inference_wrapper"
        )
        if class_path is None:
            return None
        # Resolved once per worker: this is called per sample during generation.
        cached = getattr(self, "_megatron_inference_wrapper_cls", None)
        if cached is not None:
            return cached
        module_name, _, class_name = class_path.rpartition(".")
        if not module_name:
            raise ValueError(
                "megatron_inference_wrapper must be a fully qualified class name, "
                f"got {class_path!r}."
            )
        try:
            wrapper_cls = getattr(importlib.import_module(module_name), class_name)
        except (ImportError, AttributeError) as e:
            raise ValueError(
                f"Could not resolve megatron_inference_wrapper {class_path!r} "
                f"(from policy.generation.mcore_generation_config): {e}"
            ) from e
        self._megatron_inference_wrapper_cls = wrapper_cls
        return wrapper_cls

    @staticmethod
    def _wrapper_supports_modality(
        inference_wrapper_cls: Optional[type], modality: str
    ) -> bool:
        """Whether the configured inference wrapper advertises `modality` support."""
        return bool(
            inference_wrapper_cls is not None
            and getattr(inference_wrapper_cls, f"supports_{modality}", False)
        )

    def _inference_model_and_media_parts(self, inference_wrapper_cls=None):
        """Return the language model and its optional multimodal parent."""
        model = unwrap_model(self._gen_model())
        if isinstance(model, (list, tuple)):
            if len(model) != 1:
                raise NotImplementedError("Virtual pipeline models are not supported.")
            model = model[0]
        if inference_wrapper_cls is None:
            inference_wrapper_cls = self._get_megatron_inference_wrapper_cls()
        if not any(
            self._wrapper_supports_modality(inference_wrapper_cls, modality)
            for modality in ("image", "video", "audio")
        ):
            return model, None
        return model.language_model, model

    def _build_image_preprocessing_config(self, generation_config: dict[str, Any]):
        """Build raw-image preprocessing settings."""
        inference_wrapper_cls = self._get_megatron_inference_wrapper_cls()
        if not self._wrapper_supports_modality(inference_wrapper_cls, "image"):
            return None
        processor = self.processor
        if processor is None:
            raise ValueError(
                "Megatron multimodal generation requires the policy processor."
            )
        # Omit absent keys entirely so MCore's own dataclass defaults apply;
        # passing None would override them (these fields are `bool`, not `Optional`).
        image_kwargs: dict[str, Any] = {}
        if "image_dynamic_resolution" in generation_config:
            image_kwargs["dynamic_resolution"] = bool(
                generation_config["image_dynamic_resolution"]
            )
        if "vision_model_type" in generation_config:
            image_kwargs["vision_model_type"] = str(
                generation_config["vision_model_type"]
            )
        return build_image_preprocessing_config(
            processor.image_processor,
            **image_kwargs,
        )

    def _setup_colocated_cuda_graph_managers(self) -> None:
        """Create inference CUDA-graph managers for shared-model colocated generation.

        Colocated policies build the shared training model without CUDA graphs.
        But CUDA graphs configs MUST be in-place at initialization.

        Must run at worker init: colocated reshard is detected via the pending reshard plan
        (its dedicated inference model is built later, with graphs enabled);
        the plan is consumed on the first generation cycle.
        """
        generation_cfg = self.cfg.get("generation")
        if (
            generation_cfg is None
            or generation_cfg.get("backend") != "megatron"
            or not self.is_generation_colocated
            or self._colocated_reshard_plan is not None
        ):
            return
        mcore_generation_config = generation_cfg["mcore_generation_config"]
        cuda_graph_impl = mcore_generation_config["cuda_graph_impl"]
        if cuda_graph_impl == "none":
            return
        if cuda_graph_impl != "local":
            raise ValueError(
                "Colocated Megatron generation supports only cuda_graph_impl "
                f"'none' or 'local' for inference CUDA graphs, got '{cuda_graph_impl}'. "
                "'transformer_engine' and 'full_iteration' are training-only capture modes."
            )

        lang_module = unwrap_model(self.model)
        # A model built with graphs enabled already owns managers.
        if not any(
            hasattr(module, "cudagraph_manager") for module in lang_module.modules()
        ):
            scope = normalize_inference_cuda_graph_scope(
                mcore_generation_config.get("inference_cuda_graph_scope"),
                cuda_graph_impl,
            )
            # Need the correct configs.
            set_model_config_attribute(lang_module, "cuda_graph_impl", cuda_graph_impl)
            set_model_config_attribute(lang_module, "inference_cuda_graph_scope", scope)
            set_model_config_attribute(lang_module, "cuda_graph_modules", [])

            # Need to recurse the configs' effects down into modules.
            # Megatron-LM has no API for attaching inference CUDA-graph managers to
            # an already-built model (reported upstream), so this mirrors its
            # construction-time setup.
            # TODO: switch to the upstream API once it exists.
            for module in lang_module.modules():
                if not isinstance(module, GraphableMegatronModule):
                    continue
                if hasattr(module, "create_mcore_cudagraph_manager"):
                    module.create_mcore_cudagraph_manager(module.config)
                else:
                    module.cudagraph_manager = CudaGraphManager(module.config)

            # Handle MTP as well.
            # TODO: this path only becomes testable once #3331 merges.
            if getattr(lang_module, "mtp_process", False):
                if hasattr(lang_module, "_setup_mtp_cuda_graphs") and not hasattr(
                    lang_module, "_mtp_cudagraph_manager"
                ):
                    lang_module._setup_mtp_cuda_graphs()
                assert hasattr(lang_module, "_mtp_cudagraph_manager"), (
                    f"cuda_graph_impl='{cuda_graph_impl}', but no MTP graph manager was created."
                )

            assert any(
                hasattr(module, "cudagraph_manager") for module in lang_module.modules()
            ), (
                f"cuda_graph_impl='{cuda_graph_impl}' is set for colocated Megatron "
                "generation, but no CUDA-graph manager could be created for this model."
            )

            # When the model-level manager owns block-scope graphs,
            # construction deletes the decoder's fallback manager.
            decoder = getattr(lang_module, "decoder", None)
            if (
                hasattr(lang_module, "cudagraph_manager")
                and decoder is not None
                and hasattr(decoder, "cudagraph_manager")
            ):
                del decoder.cudagraph_manager

        # Detach for training; this caches the managers built above.
        toggle_cuda_graphs(lang_module, set_to="none")

    def get_inference_cuda_graph_capture_count(self) -> int:
        """Inference CUDA graphs captured in this worker process (0 = eager decode)."""
        return len(_CudagraphGlobalRecord.cudagraph_inference_record)

    def _initialize_inference_engine(self, mcore_generation_config: dict) -> None:
        """Initialize the persistent inference engine and client."""
        # TODO: Switch to standardized Megatron API.
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

        inference_wrapper_cls = self._get_megatron_inference_wrapper_cls()
        inference_model, media_model = self._inference_model_and_media_parts(
            inference_wrapper_cls
        )
        engine_model = media_model if media_model is not None else self._gen_model()
        pg_collection = get_attr_wrapped_model(self._gen_model(), "pg_collection")
        model_config = inference_model.config

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

        mamba_inference_state_config = MambaInferenceStateConfig.from_model(
            inference_model
        )
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
        use_flashinfer_fused_rope = model_config.params_dtype in (
            torch.float16,
            torch.bfloat16,
        )

        image_preprocessing_config = self._build_image_preprocessing_config(
            mcore_generation_config
        )
        video_preprocessing_config = build_video_preprocessing_config(
            image_preprocessing_config,
            mcore_generation_config,
            frame_manifest_magic=CACHED_VIDEO_FRAME_MANIFEST_MAGIC,
        )

        # Only forward keys the config actually sets, so MCore's InferenceConfig
        # defaults stay the single source of truth for the ones it omits.
        inference_overrides: dict[str, Any] = {}
        if "async_sched_mode" in mcore_generation_config:
            inference_overrides["async_sched_mode"] = AsyncScheduleMode(
                mcore_generation_config["async_sched_mode"]
            )
        if "vision_embedding_cache_max_bytes" in mcore_generation_config:
            inference_overrides["vision_embedding_cache_max_bytes"] = int(
                mcore_generation_config["vision_embedding_cache_max_bytes"]
            )
        if "allow_stale_multimodal_embeddings" in mcore_generation_config:
            inference_overrides["allow_stale_multimodal_embeddings"] = bool(
                mcore_generation_config["allow_stale_multimodal_embeddings"]
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
            **inference_overrides,
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
            image_preprocessing_config=image_preprocessing_config,
            video_preprocessing_config=video_preprocessing_config,
        )

        if "inference_cuda_graph_scope" in mcore_generation_config:
            engine_model.config.inference_cuda_graph_scope = InferenceCudaGraphScope[
                mcore_generation_config["inference_cuda_graph_scope"]
            ]

        self.inference_context = DynamicInferenceContext(
            engine_model.config, inference_config
        )
        if media_model is None:
            self.inference_wrapped_model = GPTInferenceWrapper(
                engine_model, self.inference_context
            )
        else:
            if inference_wrapper_cls is None:
                raise ValueError(
                    "Multimodal inference requires megatron_inference_wrapper."
                )
            self.inference_wrapped_model = inference_wrapper_cls(
                engine_model, self.inference_context
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
        reserved_socket = self._reserved_http_server_socket
        if reserved_socket is not None:
            server_port = reserved_socket.getsockname()[1]
        else:
            server_port = _get_free_port_local()

        start_text_gen_server(
            coordinator_addr=self.coordinator_addr,
            tokenizer=self.megatron_tokenizer,
            rank=torch.distributed.get_rank(),
            server_port=server_port,
            parsers=self.cfg["generation"]["mcore_generation_config"]["parsers"],
            verbose=False,
            sock=reserved_socket,
            multimodal_prompt_config=self.inference_wrapped_model.multimodal_prompt_config,
        )

        base_url = f"http://{ip}:{server_port}/v1"
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

    def finish_generation(self, *, release_gpu: bool = True) -> None:
        """Wind down a generation cycle.

        Args:
            release_gpu: the caller needs the GPUs for itself (a training
                step or a checkpoint save), so even a colocated engine must
                fully stand down. Pass False between generation phases (e.g.
                after validation) to let a colocated engine keep serving;
                tearing it down there would discard KV/prefix caches and CUDA
                graphs for no reason.
                For non-colocated engines this body reduces to a rotary-cache clear
                (their engine pause happens in suspend_for_refit).
        """
        if self.is_generation_colocated and not release_gpu:
            return
        print(f"[Rank {self.rank}] finishing generation", flush=True)
        log_gpu_memory("finish_generation START")

        inference_model, media_model = self._inference_model_and_media_parts()
        lang_module = unwrap_model(inference_model)
        graph_module = unwrap_model(
            media_model if media_model is not None else inference_model
        )

        if self.is_generation_colocated:
            if self._inference_engine_initialized and not self._inference_engine_asleep:
                self._sleep()
            cuda_graph_impl = self.cfg["generation"]["mcore_generation_config"][
                "cuda_graph_impl"
            ]
            if cuda_graph_impl != "none":
                # Restore the same full model tree cached during worker setup.
                toggle_cuda_graphs(graph_module, set_to="none")
                # Need to turn off padding before training.
                # Gains nightly MoE coverage once #2884 and #3570 merge.
                set_decode_expert_padding(lang_module, set_to=False)

        rotary_module = getattr(lang_module, "rotary_pos_emb", None)
        if rotary_module is not None and hasattr(
            rotary_module.forward, "cache_parameters"
        ):
            rotary_module.forward.cache_clear()

        if self.is_generation_colocated:
            # Offload the inference weights to CPU.
            if self.inference_model is not None:
                self._offload_inference_model()
            gc.collect()
            torch.cuda.empty_cache()

        log_gpu_memory("finish_generation END")

    def prepare_for_generation(self, tags=None, **kwargs) -> None:
        """Enter inference mode and start (or wake) the inference engine.

        Idempotent wake: a plain call (no tags) on an already-serving engine returns immediately.
        Refit-protocol calls (tags) are never skipped.

        Called in both colocated and non-colocated setups.
        Even in non-colocated mode, Megatron's engine has to be intentionally paused before a refit
        (and its weights are not detachable), so we have to switch modes around every refit.
        """
        if (
            tags is None
            and self._inference_engine_initialized
            and not self._inference_engine_asleep
        ):
            print(
                f"[Rank {self.rank}] prepare_for_generation: engine already "
                "serving, skipping",
                flush=True,
            )
            return
        log_gpu_memory("prepare_for_generation START")
        mcore_generation_config = self.cfg["generation"]["mcore_generation_config"]

        # Colocated reshard: build the dedicated inference-layout model on the first cycle.
        if self._colocated_reshard_plan is not None:
            self._build_colocated_inference_model(self.cfg)

        if self.is_generation_colocated and self.inference_model is None:
            self.model = self.move_model(
                self.model, "cuda", move_params=True, move_grads=False
            )
            # Because DP inference requests are asynchronously scheduled per rank, pre-forward hooks that trigger DP collectives (such as an overlapped param gather after optimizer steps) will stall or hang.
            # Instead, synchronously gather all model compute weights from the sharded model state here, and deactivate all pre-forward hooks.
            # Incompatible with FSDP2 or Megatron-FSDP for inference.
            if (
                self.should_disable_forward_pre_hook
                and self._forward_pre_hook_enabled()
            ):
                self._disable_forward_pre_hook_until_next_train_step(param_sync=True)

        # Colocated reshard (hosts without a dedicated inference model skip it).
        if self.inference_model is not None:
            self._reshard_into_inference_model()

        inference_model, media_model = self._inference_model_and_media_parts()
        # `flash_decode` selects Megatron Inference's deprecated static-batching decode path,
        # which would cause an assertion error if taken.
        inference_model.config.flash_decode = False
        lang_module = unwrap_model(inference_model)
        graph_module = unwrap_model(
            media_model if media_model is not None else inference_model
        )
        lang_module.eval()

        rotary_module = getattr(lang_module, "rotary_pos_emb", None)
        if rotary_module is not None and hasattr(
            rotary_module.forward, "cache_parameters"
        ):
            rotary_module.forward.cache_clear()

        cuda_graph_impl = mcore_generation_config["cuda_graph_impl"]
        if cuda_graph_impl != "none":
            # Use the same root object as _setup_colocated_cuda_graph_managers;
            # CUDA-graph manager caches are keyed by model identity.
            toggle_cuda_graphs(graph_module, set_to=cuda_graph_impl)

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
        self,
        greedy: bool,
        stop_words: Optional[list[str]],
        *,
        return_prompt_tokens: bool = False,
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
            return_prompt_tokens=return_prompt_tokens,
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

    def _sample_vision_tensors(self, data, index: int):
        """Return one sample's vision tensors from RL PackedTensors."""
        return sample_vision_tensors(data, index)

    def _build_prompt_and_multimodal_data(self, data, index: int):
        """Build one pre-expanded token prompt and optional MCore media payload."""
        return build_prompt_and_multimodal_data(
            data,
            index,
            sample_tensors=self._sample_vision_tensors,
            supports_modality=lambda modality: self._wrapper_supports_modality(
                self._get_megatron_inference_wrapper_cls(),
                modality,
            ),
        )

    def _prepare_data_for_generation(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> tuple[list[list[int]], list[Optional[Any]], list[SamplingParams]]:
        """Build prompts, optional media payloads, and sampling parameters."""
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

        batch_stop_strings = data.get("stop_strings", [])
        prompts: list[list[int]] = []
        multi_modal_data_list: list[Optional[Any]] = []
        sampling_params = []
        for i in range(data.size):
            prompt, multi_modal_data = self._build_prompt_and_multimodal_data(data, i)
            sample_stop_strings = (
                batch_stop_strings[i] if i < len(batch_stop_strings) else None
            )
            stop_words = self._merge_stop_strings(
                [sample_stop_strings] if sample_stop_strings else None
            )
            prompts.append(prompt)
            multi_modal_data_list.append(multi_modal_data)
            sampling_params.append(
                self._build_sampling_params(
                    greedy,
                    stop_words,
                    return_prompt_tokens=multi_modal_data is not None,
                )
            )

        return prompts, multi_modal_data_list, sampling_params

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

        expected_prompt_lengths = [int(length) for length in input_lengths.tolist()]
        inference_prompt_lengths = [
            len(x.prompt_tokens)
            if getattr(x, "prompt_tokens", None) is not None
            else expected_prompt_lengths[i]
            for i, x in enumerate(result)
        ]
        if any(getattr(x, "prompt_tokens", None) is not None for x in result):
            if inference_prompt_lengths != expected_prompt_lengths:
                raise RuntimeError(
                    "Megatron inference prompt lengths do not match the training "
                    "processor input lengths: "
                    f"inference={inference_prompt_lengths}, "
                    f"training={expected_prompt_lengths}."
                )

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
            # engine's reply. Multimodal requests do set
            # SamplingParams.return_prompt_tokens, but only so the echoed tokens
            # can be length-checked below; the padded output is still built from
            # the prompt we already hold rather than shipped back over ZMQ.
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
        prompts, multi_modal_data_list, sampling_params = (
            self._prepare_data_for_generation(data, greedy)
        )
        if self._inference_loop is None:
            raise RuntimeError(
                "Inference loop not initialized. Call prepare_for_generation() first."
            )
        future = asyncio.run_coroutine_threadsafe(
            self._generate_with_persistent_engine(
                prompts,
                multi_modal_data_list,
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
            prompts, multi_modal_data_list, sampling_params = (
                self._prepare_data_for_generation(datum, greedy)
            )
            future = asyncio.run_coroutine_threadsafe(
                self._generate_with_persistent_engine(
                    prompts,
                    multi_modal_data_list,
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
        prompts: list[list[int]],
        multi_modal_data_list: list[Optional[Any]],
        sampling_params: list[SamplingParams],
    ) -> list:
        """Submit requests through the persistent inference client (rank 0 only)."""
        from megatron.core.inference.inference_request import DynamicInferenceRequest

        dist_rank = torch.distributed.get_rank()
        assert dist_rank == 0, (
            "Only rank 0 creates a client to communicate with the coordinator"
        )

        print(f"[Rank {dist_rank}] Submitting {len(prompts)} requests to coordinator")

        futures = []
        for prompt, multi_modal_data, request_sampling_params in zip(
            prompts, multi_modal_data_list, sampling_params, strict=True
        ):
            futures.append(
                self.inference_client.add_request(
                    prompt,
                    request_sampling_params,
                    multi_modal_data=multi_modal_data,
                )
            )

        results: list[DynamicInferenceRequest] = await asyncio.gather(*futures)
        print(f"[Rank {dist_rank}] Completed {len(results)} requests")
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
            refit_backend: Copy-service backend ("gloo" or "nccl";
                "nvshmem" is currently broken, see the issue below).
        """
        if refit_backend == "nvshmem":
            warnings.warn(
                'refit_backend="nvshmem" is currently broken; prefer "nccl" or '
                '"gloo". See https://github.com/NVIDIA-NeMo/RL/issues/3646',
                stacklevel=2,
            )

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
            # Create the group-wide NCCL communicator now, on every rank.
            nccl_backend.eager_connect_single_device(
                torch.device("cuda", torch.cuda.current_device())
            )
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
            # Deferred: importing NVSHMEMCopyService loads the optional nvshmem bindings.
            from megatron.core.resharding.copy_services.nvshmem_copy_service import (
                NVSHMEMCopyService,
            )

            self.refit_copy_service = NVSHMEMCopyService(group=self.refit_pg)
        elif refit_backend == "nccl":
            self.refit_copy_service = NCCLCopyService(group=self.refit_pg)
        else:
            self.refit_copy_service = GlooCopyService(group=self.refit_pg)

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
        outside CUDA graph capture. Lazy initialization during graph recording or replay
        can corrupt CUDA graph state.
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

        return True

    def _onload_inference_model(self) -> None:
        """Restore the colocated inference weights to GPU before resharding / generation."""
        if not self._inference_model_offloaded:
            return
        resume_inference_weights()
        self._inference_model_offloaded = False

    def _offload_inference_model(self) -> None:
        """Offload the colocated inference weights to CPU while training runs."""
        if (
            self.inference_model is None
            or self._inference_model_offloaded
            or not HAVE_TORCH_MEMORY_SAVER
        ):
            return
        pause_inference_weights()
        self._inference_model_offloaded = True

    def _reshard_into_inference_model(self) -> None:
        """Reshard current training weights into the colocated inference-layout model."""
        inference_model = self.inference_model
        if inference_model is None:
            return

        # Bring the inference weights back to GPU.
        self._onload_inference_model()
        self.model = self.move_model(
            self.model, "cuda", move_params=True, move_grads=False
        )
        # TODO: Optimize away the full synchronization.
        torch.cuda.synchronize()

        # The swap reads the training params as its source;
        # under overlap_param_gather they stay stale after the optimizer step until gathered.
        if self.should_disable_forward_pre_hook and self._forward_pre_hook_enabled():
            self._disable_forward_pre_hook_until_next_train_step(param_sync=True)

        # Build + cache the same-rank reshard plan once, before the first CUDA-graph capture.
        if not self._swap_weights_plan_prepared:
            prepare_swap_model_weights(
                src_model=self.model,
                target_model=inference_model,
                group=None,
                src_rank_offset=0,
                dst_rank_offset=0,
            )
            self._swap_weights_plan_prepared = True

        swap_model_weights(
            self.model,
            inference_model,
            refit_method=self.cfg["generation"]["mcore_generation_config"][
                "refit_backend"
            ],
            group=None,
            src_rank_offset=0,
            dst_rank_offset=0,
        )
        # Offload training model.
        self.model = self.move_model(
            self.model, "cpu", move_params=True, move_grads=False
        )
        # TODO: Optimize away the full synchronization.
        torch.cuda.synchronize()

    def suspend_for_refit(self) -> None:
        """Pause+suspend the inference engine before a weight refit."""
        if not self._inference_engine_initialized:
            return
        self._sleep()

    def resume_after_refit(self) -> None:
        """Resume+unpause the inference engine after a weight refit."""
        if not self._inference_engine_initialized:
            return
        self._wake()
