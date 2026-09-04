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
import copy
import gc
import os
import threading
import time
import uuid
import warnings
from typing import Any, AsyncGenerator, Optional, cast

import ray
import torch
import uvicorn
from fastapi import FastAPI

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import _get_free_port_local, _get_node_ip_local
from nemo_rl.distributed.worker_group_utils import get_nsight_config_if_pattern_matches
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationOutputSpec,
    verify_right_padding,
)
from nemo_rl.models.generation.vllm.inflight_profiler import (
    InflightProfiler,
    inflight_interval_s,
    inflight_profiling_enabled,
)
from nemo_rl.models.generation.vllm.vllm_metric_sampler import (
    metric_capture_timing,
    read_restricted_vllm_metrics,
)
from nemo_rl.models.generation.vllm.lfs.concurrency import (
    get_engine_kv_cache_shape,
)
from nemo_rl.models.generation.vllm.utils import format_prompt_for_vllm_generation
from nemo_rl.models.generation.vllm.vllm_worker import (
    BaseVllmGenerationWorker,
    _get_vllm_inner_nsys_mode,
)


async def _await_model_worker_collective_rpc(
    llm: Any, method: str
) -> None:
    """Await an AsyncLLM collective RPC when its engine is initialized."""
    if llm is not None:
        await llm.collective_rpc(method, args=tuple())


async def _submit_vllm_request(
    llm: Any,
    *,
    prompt: Any,
    sampling_params: Any,
    request_id: str,
    priority: int,
) -> tuple[Any, float, float, str]:
    """Submit to EngineCore and return the causal frontend boundary."""
    request_output_collector = await llm.add_request(
        request_id=request_id,
        prompt=prompt,
        params=sampling_params,
        priority=priority,
    )
    submitted_at_unix_s = time.time()
    submitted_at_monotonic_s = time.monotonic()
    submitted_hostname = os.uname().nodename
    return (
        request_output_collector,
        submitted_at_unix_s,
        submitted_at_monotonic_s,
        submitted_hostname,
    )


async def _iterate_request_output_collector(
    request_output_collector: Any,
    stream_finished: Any,
) -> AsyncGenerator[Any, None]:
    """Mirror AsyncLLM.generate's terminal-sentinel loop."""
    finished = False
    while not finished:
        req_output = (
            request_output_collector.get_nowait()
            or await request_output_collector.get()
        )
        # STREAM_FINISHED is itself terminal. Read `finished` before deciding
        # whether to yield it, exactly as AsyncLLM.generate does.
        finished = bool(req_output.finished)
        if req_output is not stream_finished:
            yield req_output


def _replace_prefix_tokens(
    tokenizer,
    model_prefix_token_ids: list[int],
    template_prefix_token_ids: list[int],
    template_token_ids: list[int],
) -> list[int]:
    """This is a subroutine used inside the vLLM Chat Completion server.

    This function is for fixing up the chat template-tokenized messages history
    to match the model output tokenization up to the last assistant turn,
    in order to preserve the monotonic tokens property for optimized multi-turn
    training.

    Some environments (namely NeMo-Gym) require an OpenAI compatible server
    endpoint rather than an inference engine handle. This is fine for the most
    part, but it may cause issues when the environment is used as a part of
    training.

    RL training frameworks train models on token IDs, but the OpenAI compatible
    server communicates in what is basically de-tokenized text. When multiple
    model calls are made to the OpenAI compatible server in a single trajectory,
    model generations in previous model calls may be re-tokenized to something
    that is different than what was generated. This is not too big of an issue
    (that we know of) at inference time, but the log probs the model produces
    are different enough for the differently re-tokenized generation result that
    it causes the training to be off policy. Off policy isn't necessarily a bad
    thing in isolation, but this source of off-policyness may cause unexpected
    issues if not properly accounted for. It also mis-aligns the token ID
    sequences across model calls, which feels very strange during training.

    There are real cases where the model output string _does not match_ the chat
    template tokenization of the parsed model output. A concrete example is
    inconsistent whitespace tokens around tool call special tokens.

    TODO When NeMo RL supports training image generation models, we want to
    revisit and possibly update this function. This issue occurs when the model
    generates tokens that are de-tokenized into text or images, and then
    re-tokenized into tokens. So if there is a situation like that with images
    and image tokenization is non-unique, then we will need to uppdate this
    function.

    Example (turn-by-turn, concise; eos_token_id = 2):
        Turn 1:
            - prefill_T1 (template prefill) = [11,12,13,40,41]
            - model output = [220,17,2]  # decodes to " 4" + EOS
            - model_prefix_token_ids = prefill_T1 + model output
              => [11,12,13,40,41,220,17,2]

        Turn 2 (template retokenizes prior assistant text differently):
            - template_prefix_token_ids = [11,12,13,40,41,1001,2]  # 1001 decodes to " 4"
            - template_token_ids = [11,12,13,40,41,1001,2,21,22,40,41]

        _replace_prefix_tokens keeps the exact prior model tokens up to EOS and
        resumes from the template after that EOS:
            output => [11,12,13,40,41,220,17,2,21,22,40,41]
    """
    if not model_prefix_token_ids:
        return template_token_ids

    eos_token_id = tokenizer.eos_token_id
    assert eos_token_id is not None, "Your tokenizer must have an EOS token ID!"

    model_cut_end = len(model_prefix_token_ids)
    if model_prefix_token_ids:
        # We are not always guaranteed that the model outputs an EOS token as the stop criteria of the previous model call e.g. when the model reaches max_tokens.
        # And since chat templates will always add one for us, we just cut the model input to right before the EOS token ID (if applicable)
        if model_prefix_token_ids[-1] == eos_token_id:
            model_cut_end -= 1

    # Assert here to prepare for the logic below
    assert len(template_token_ids) > len(
        template_prefix_token_ids
    ), f"""Found possibly non-monotonically increasing trajectory!
Template prefix token IDs (everything before the final assistant message): {template_prefix_token_ids}

Template token IDs (everything that was sent to the model endpoint): {template_token_ids}

Template prefix repr (detokenized): {repr(tokenizer.decode(template_prefix_token_ids))}

Template repr (detokenized): {repr(tokenizer.decode(template_token_ids))}
"""

    # We take everything starting with the EOS token ID.
    template_cut_start = -1
    for pos in reversed(range(len(template_prefix_token_ids))):
        if template_token_ids[pos] == eos_token_id:
            template_cut_start = pos
            break

    # This should never be the case, but
    assert (
        template_cut_start >= 0
    ), f"""No EOS token ID found in the chat-templated messages!
Template prefix token IDs (everything before the final assistant message): {template_prefix_token_ids}

Template token IDs (everything that was sent to the model endpoint): {template_token_ids}

Template prefix repr (detokenized): {repr(tokenizer.decode(template_prefix_token_ids))}

Template repr (detokenized): {repr(tokenizer.decode(template_token_ids))}"""

    return (
        model_prefix_token_ids[:model_cut_end] + template_token_ids[template_cut_start:]
    )


@ray.remote(
    runtime_env={**get_nsight_config_if_pattern_matches("vllm_async_generation_worker")}
)  # pragma: no cover
class VllmAsyncGenerationWorker(BaseVllmGenerationWorker):
    async def start_gpu_profiling_async(self) -> None:
        """Start profiling on every async-engine model worker."""
        await _await_model_worker_collective_rpc(
            self.llm, "start_gpu_profiling"
        )

    async def stop_gpu_profiling_async(self) -> None:
        """Stop profiling on every async-engine model worker."""
        await _await_model_worker_collective_rpc(
            self.llm, "stop_gpu_profiling"
        )

    async def arm_model_step_gpu_profile_async(
        self,
        start_step: int,
        stop_step: int,
    ) -> list[dict[str, Any]]:
        """Arm the same exact model-step range on every nested TP worker."""

        if self.llm is None:
            raise RuntimeError("async vLLM engine is not initialized")
        result = await self.llm.collective_rpc(
            "arm_model_step_gpu_profile",
            args=(start_step, stop_step),
        )
        if not isinstance(result, list):
            raise RuntimeError(
                "vLLM collective arm RPC did not return one proof list"
            )
        return result

    async def get_model_step_gpu_profile_async(
        self,
    ) -> list[dict[str, Any]]:
        """Collect completed exact-range proofs from every nested TP worker."""

        if self.llm is None:
            raise RuntimeError("async vLLM engine is not initialized")
        result = await self.llm.collective_rpc(
            "get_model_step_gpu_profile",
            args=tuple(),
        )
        if not isinstance(result, list):
            raise RuntimeError(
                "vLLM collective proof RPC did not return one proof list"
            )
        return result

    def report_replay_runtime_provenance(self) -> dict[str, Any]:
        """Report actual worker source, engine arguments, and processors."""
        nested_runtime_env = getattr(
            self, "_vllm_nested_runtime_env", None
        )
        inner_nsys_config = (
            copy.deepcopy(nested_runtime_env.get("nsight"))
            if isinstance(nested_runtime_env, dict)
            and isinstance(nested_runtime_env.get("nsight"), dict)
            else None
        )
        return {
            "worker_module_file": __file__,
            # Preserve the user-facing identifier before vLLM resolves a
            # Hugging Face repository ID to its concrete cache snapshot.
            "configured_model_name": self.model_name,
            "pythonpath": os.environ.get("PYTHONPATH"),
            "propagate_pythonpath": os.environ.get(
                "NRL_VLLM_PROPAGATE_PYTHONPATH"
            ),
            "engine_logits_processors": list(
                map(
                    str,
                    self.llm_async_engine_args.logits_processors or [],
                )
            ),
            "inner_nsys_mode": _get_vllm_inner_nsys_mode(),
            "inner_nsys_config": inner_nsys_config,
            "vllm_nested_runtime_env": copy.deepcopy(
                nested_runtime_env
            ),
            "vllm_nested_runtime_env_contract_sha256": getattr(
                self,
                "_vllm_nested_runtime_env_contract_sha256",
                None,
            ),
            "vllm_nested_runtime_env_contract_exported": bool(
                getattr(
                    self,
                    "_vllm_nested_runtime_env_patch_verified",
                    False,
                )
            ),
            # Backward-compatible observability alias used by the bounded
            # inner-Nsight replay finalizer. The implementation is now the
            # custom executor contract rather than an installed-source patch.
            "inner_nsys_runtime_env_patch_verified": bool(
                getattr(
                    self,
                    "_vllm_nested_runtime_env_patch_verified",
                    False,
                )
            ),
            "vllm_env_copy_layout": getattr(
                self, "_vllm_env_copy_layout", None
            ),
            "model_worker_runtime_provenance": copy.deepcopy(
                getattr(
                    self,
                    "vllm_model_worker_runtime_provenance",
                    None,
                )
            ),
            "ray_workers_use_nsight": bool(
                self.llm_async_engine_args.ray_workers_use_nsight
            ),
            "async_engine_args": {
                "enable_prefix_caching": (
                    self.llm_async_engine_args.enable_prefix_caching
                ),
                "enforce_eager": getattr(
                    self.llm_async_engine_args, "enforce_eager", None
                ),
                "max_num_seqs": self.llm_async_engine_args.max_num_seqs,
                "max_num_batched_tokens": (
                    self.llm_async_engine_args.max_num_batched_tokens
                ),
                "model": self.llm_async_engine_args.model,
                "revision": self.llm_async_engine_args.revision,
            },
        }

    def _create_engine(self, llm_kwargs: dict[str, Any]) -> None:
        from vllm.config import CompilationConfig
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM
        from vllm.v1.metrics.loggers import PrometheusStatLogger

        # Workaround: convert compilation_config dict to CompilationConfig object
        # since AsyncEngineArgs doesn't handle the dict-to-pydantic conversion.
        if llm_kwargs.get("compilation_config", None):
            compilation_config = dict(llm_kwargs["compilation_config"])
            # use_inductor was removed in vLLM v0.12+ (https://github.com/vllm-project/vllm/pull/29323)
            # and replaced by the `backend` field: use_inductor=True -> backend="" (inductor),
            # use_inductor=False -> backend="eager".
            if "use_inductor" in compilation_config:
                use_inductor = compilation_config.pop("use_inductor")
                if "backend" not in compilation_config:
                    compilation_config["backend"] = "" if use_inductor else "eager"
                warnings.warn(
                    "compilation_config.use_inductor is deprecated in vLLM v0.12+. "
                    "Use compilation_config.backend instead: "
                    "use_inductor=True -> backend='inductor', "
                    "use_inductor=False -> backend='eager'.",
                    DeprecationWarning,
                    stacklevel=1,
                )
            llm_kwargs["compilation_config"] = CompilationConfig(**compilation_config)

        # Experimental engine-level group-LFS waiting order (env-gated by
        # NRL_VLLM_LFS_SCHED=1). This reorders whole requests only. The group
        # id is carried in `priority`, so vLLM's own scheduling policy remains
        # FCFS instead of using its priority heap.
        if os.environ.get("NRL_VLLM_LFS_SCHED") == "1":
            llm_kwargs["scheduler_cls"] = (
                "nemo_rl.models.generation.vllm.lfs.engine_schedulers.ProbeLfsScheduler"
            )
        # Apply the same explicit per-engine concurrency cap to both A/B arms.
        max_num_seqs = os.environ.get("NRL_VLLM_MAX_NUM_SEQS")
        if max_num_seqs:
            llm_kwargs["max_num_seqs"] = int(max_num_seqs)

        # Falsification arm for the length-homogeneity effect measured on the
        # TRTLLM-gen decode kernel. Setting attention_config.use_trtllm_attention
        # to False makes can_use_trtllm_attention() return False, so decode falls
        # back to FlashInfer's BatchDecodeWithPagedKVCacheWrapper, whose kernel
        # time is driven by the batch's summed context rather than its longest
        # member. Prefill selection is unchanged. Off by default.
        if os.environ.get("NRL_VLLM_DISABLE_TRTLLM_ATTENTION") == "1":
            from vllm.config import AttentionConfig

            llm_kwargs["attention_config"] = AttentionConfig(
                use_trtllm_attention=False
            )

        step_trace_enabled = bool(
            self.cfg["vllm_cfg"].get("enable_vllm_step_trace", False)
        )
        step_trace_logger_class = None
        if step_trace_enabled:
            if os.environ.get("VLLM_DEBUG_MFU_METRICS") != "1":
                raise RuntimeError(
                    "enable_vllm_step_trace requires worker environment "
                    "VLLM_DEBUG_MFU_METRICS=1"
                )
            # These are observability-only engine arguments. PerfStats supplies
            # exact scheduled composition aggregates and CUDAGraphStat supplies
            # the actual unpadded/padded token batch for the same step.
            llm_kwargs["enable_mfu_metrics"] = True
            llm_kwargs["cudagraph_metrics"] = True
            from nemo_rl.models.generation.vllm.vllm_step_trace import (
                get_vllm_step_trace_logger_class,
            )

            step_trace_logger_class = get_vllm_step_trace_logger_class()

        self.llm_async_engine_args = AsyncEngineArgs(**llm_kwargs)
        self.stat_loggers: list[Any] = []
        if self.cfg["vllm_cfg"].get("enable_vllm_metrics_logger", False):
            self.stat_loggers.append(PrometheusStatLogger)
        if step_trace_logger_class is not None:
            self.stat_loggers.append(step_trace_logger_class)
        self.llm = AsyncLLM.from_engine_args(
            self.llm_async_engine_args, stat_loggers=self.stat_loggers
        )
        self._vllm_step_trace_logger = None
        if step_trace_enabled:
            logger_manager = self.llm.logger_manager
            if logger_manager is None:
                raise RuntimeError(
                    "vLLM did not create a StatLoggerManager for step tracing"
                )
            matching_loggers = [
                logger
                for logger in logger_manager.stat_loggers
                if getattr(logger, "_nrl_vllm_step_trace_logger", False)
            ]
            if len(matching_loggers) != 1:
                raise RuntimeError(
                    "expected exactly one vLLM step trace logger, found "
                    f"{len(matching_loggers)}"
                )
            self._vllm_step_trace_logger = matching_loggers[0]

        self.server_thread, self.base_url, self.http_server = None, None, None
        if self.cfg["vllm_cfg"].get("expose_http_server"):
            self.server_thread, self.base_url, self.http_server = (
                self._setup_vllm_server()
            )

        # vLLM Metrics Logger
        # Metrics logger only enabled for per-actor, model-owner only
        self._vllm_metrics_lock = threading.Lock()
        if self.cfg["vllm_cfg"].get("enable_vllm_metrics_logger", False):
            self._start_vllm_metrics_logger()

    def get_kv_cache_shape(self) -> dict[str, int]:
        """Return the initialized per-engine KV cache capacity."""
        kv_cache_tokens, block_size = get_engine_kv_cache_shape(self.llm)
        return {
            "kv_cache_tokens": kv_cache_tokens,
            "block_size": block_size,
        }

    def get_vllm_step_trace(self) -> dict[str, Any]:
        """Return a consistent copy of the current per-step trace window."""
        if not self.cfg["vllm_cfg"].get("enable_vllm_step_trace", False):
            return {}
        logger = getattr(self, "_vllm_step_trace_logger", None)
        if logger is None:
            raise RuntimeError("vLLM step tracing is enabled but logger is missing")
        return logger.snapshot()

    def clear_vllm_step_trace(self) -> None:
        """Open a fresh per-step trace window."""
        if not self.cfg["vllm_cfg"].get("enable_vllm_step_trace", False):
            return
        logger = getattr(self, "_vllm_step_trace_logger", None)
        if logger is None:
            raise RuntimeError("vLLM step tracing is enabled but logger is missing")
        logger.clear()

    def _start_vllm_metrics_logger(self) -> None:
        """Start a background thread that periodically collects vLLM logger metrics.

        Controlled by vllm_metrics_logger_interval (default: 0.5) in vllm_cfg.
        Runs only on the model-owner actor.
        """
        from prometheus_client import REGISTRY

        assert self.cfg["vllm_cfg"].get("async_engine", False), (
            "vLLM metrics logger is only supported with async engine enabled"
        )
        # Run only on the model-owner actor
        if not getattr(self, "is_model_owner", False):
            return

        assert "vllm_metrics_logger_interval" in self.cfg["vllm_cfg"], (
            "vllm_metrics_logger_interval must be set in vllm_cfg if enable_vllm_metrics_logger is True"
        )
        interval_s = self.cfg["vllm_cfg"]["vllm_metrics_logger_interval"]
        assert interval_s > 0, (
            f"vllm_metrics_logger_interval must be a positive float, got {interval_s}"
        )

        # Lazy import inside thread target to avoid import overhead if disabled
        stop_event = threading.Event()
        self._vllm_metrics_logger_stop_event = stop_event

        self.inflight_batch_sizes: list[int] = []
        self.num_pending_samples: list[int] = []
        self.kv_cache_usage_perc: list[float] = []
        self.generation_tokens: list[int] = []
        self.num_preemptions: list[int] = []
        # Keep one structured record per collector pass.  The legacy metric
        # lists above are retained for compatibility, but they cannot be
        # aligned reliably by list index: collecting a snapshot takes
        # non-zero time and individual metrics may occasionally be absent.
        self.vllm_metric_samples: list[dict[str, Any]] = []
        self.vllm_metric_source_series: dict[str, dict[str, Any]] | None = None
        self.vllm_metric_sampler_errors: list[dict[str, Any]] = []
        self.vllm_metric_sampler_interval_s = float(interval_s)
        self.generation_tokens_baseline = 0
        self.num_preemptions_baseline = 0

        def _capture_sample(
            anchor_kind: str = "periodic",
            *,
            scheduled_at_monotonic_s: float | None = None,
            attempted_at_monotonic_s: float | None = None,
        ) -> dict[str, Any]:
            if attempted_at_monotonic_s is None:
                attempted_at_monotonic_s = time.monotonic()
            started_at_monotonic_s = time.monotonic()
            values, source_series = read_restricted_vllm_metrics(REGISTRY)
            finished_at_monotonic_s = time.monotonic()

            if self.vllm_metric_source_series is None:
                self.vllm_metric_source_series = copy.deepcopy(source_series)
            elif source_series != self.vllm_metric_source_series:
                raise RuntimeError(
                    "vLLM Prometheus source series changed during sampling: "
                    f"expected={self.vllm_metric_source_series!r}, "
                    f"observed={source_series!r}"
                )

            # Record the observation time after reading the snapshot.
            # Same-host benchmark processes align on monotonic time; Unix time
            # is retained to diagnose clock adjustments and for display.
            sample: dict[str, Any] = {
                "anchor_kind": anchor_kind,
                **values,
                **metric_capture_timing(
                    interval_s=interval_s,
                    scheduled_at_monotonic_s=scheduled_at_monotonic_s,
                    attempted_at_monotonic_s=attempted_at_monotonic_s,
                    started_at_monotonic_s=started_at_monotonic_s,
                    finished_at_monotonic_s=finished_at_monotonic_s,
                ),
            }
            sample["sampled_at_unix_s"] = time.time()
            sample["sampled_at_monotonic_s"] = finished_at_monotonic_s
            sample["hostname"] = os.uname().nodename
            return sample

        def _append_sample(sample: dict[str, Any]) -> None:
            self.vllm_metric_samples.append(sample)
            if "inflight_batch_size" in sample:
                self.inflight_batch_sizes.append(sample["inflight_batch_size"])
            if "num_pending" in sample:
                self.num_pending_samples.append(sample["num_pending"])
            if "kv_cache_usage_perc" in sample:
                self.kv_cache_usage_perc.append(sample["kv_cache_usage_perc"])
            if "generation_tokens" in sample:
                self.generation_tokens.append(sample["generation_tokens"])
            if "num_preemptions" in sample:
                self.num_preemptions.append(sample["num_preemptions"])

        # clear() and get() use synchronous anchors from the same reader as the
        # periodic sampler. All calls are serialized by _vllm_metrics_lock.
        self._capture_vllm_metric_sample = _capture_sample
        self._append_vllm_metric_sample = _append_sample

        def _logger_loop():
            # Delay a little to let engine settle
            if stop_event.wait(min(2.0, interval_s)):
                return
            next_sample_monotonic = time.monotonic()
            while not stop_event.is_set():
                scheduled_at_monotonic_s = next_sample_monotonic
                attempted_at_monotonic_s = time.monotonic()
                capture_error: Exception | None = None
                # Snapshot, append, and error attribution share the same lock
                # as clear(). If clear() ran between a failed warmup capture
                # and recording that failure, the warmup error could otherwise
                # be attributed to the measured rollout epoch.
                with self._vllm_metrics_lock:
                    try:
                        _append_sample(
                            _capture_sample(
                                scheduled_at_monotonic_s=(
                                    scheduled_at_monotonic_s
                                ),
                                attempted_at_monotonic_s=(
                                    attempted_at_monotonic_s
                                ),
                            )
                        )
                    except Exception as error:
                        capture_error = error
                        self.vllm_metric_sampler_errors.append(
                            {
                                "error_type": type(error).__name__,
                                "error": str(error),
                                "scheduled_at_monotonic_s": (
                                    scheduled_at_monotonic_s
                                ),
                                "attempted_at_monotonic_s": (
                                    attempted_at_monotonic_s
                                ),
                                "failed_at_monotonic_s": time.monotonic(),
                            }
                        )
                        if len(self.vllm_metric_sampler_errors) > 1000:
                            del self.vllm_metric_sampler_errors[:100]
                if capture_error is not None:
                    print(
                        "⚠️[vLLM Metric Logger] Exception in restricted "
                        f"vLLM metrics sampler: {capture_error!r}",
                        flush=True,
                    )
                # Use a deadline instead of sleep(interval) after collection;
                # otherwise collector overhead accumulates into large timeline
                # drift during long rollouts.
                next_sample_monotonic += interval_s
                now_monotonic = time.monotonic()
                if next_sample_monotonic <= now_monotonic:
                    # Do not burst through every missed deadline after a long
                    # snapshot or lock pause. One fresh sample per interval is
                    # sufficient and avoids perturbing the engine.
                    next_sample_monotonic = now_monotonic + interval_s
                stop_event.wait(
                    max(0.0, next_sample_monotonic - time.monotonic())
                )

        t = threading.Thread(
            target=_logger_loop, name="vllm-metrics-logger", daemon=True
        )
        t.start()
        self._vllm_metrics_logger_thread = t
        print(
            "📋[vLLM Metric Logger] vLLM metrics logger thread started",
            flush=True,
        )

    def get_vllm_logger_metrics(self) -> dict[str, Any]:
        if not self.cfg["vllm_cfg"].get("enable_vllm_metrics_logger", False):
            return {}

        attempted_at_monotonic_s = time.monotonic()
        with self._vllm_metrics_lock:
            # Capture an exact query-time terminal anchor instead of relying on
            # the periodic thread to happen to sample after the final request.
            self._append_vllm_metric_sample(
                self._capture_vllm_metric_sample(
                    "query_terminal",
                    attempted_at_monotonic_s=attempted_at_monotonic_s,
                )
            )
            metric = {
                "inflight_batch_sizes": copy.deepcopy(self.inflight_batch_sizes),
                "num_pending_samples": copy.deepcopy(self.num_pending_samples),
                "kv_cache_usage_perc": copy.deepcopy(self.kv_cache_usage_perc),
                "generation_tokens": copy.deepcopy(self.generation_tokens),
                "num_preemptions": copy.deepcopy(self.num_preemptions),
                "metric_samples": copy.deepcopy(self.vllm_metric_samples),
                "metric_source_series": copy.deepcopy(
                    self.vllm_metric_source_series
                ),
                "metric_sampler_errors": copy.deepcopy(
                    self.vllm_metric_sampler_errors
                ),
                "metric_sampler_interval_s": (
                    self.vllm_metric_sampler_interval_s
                ),
                "generation_tokens_baseline": self.generation_tokens_baseline,
                "num_preemptions_baseline": self.num_preemptions_baseline,
            }
        return metric

    def clear_vllm_logger_metrics(self) -> dict[str, Any] | None:
        if not self.cfg["vllm_cfg"].get("enable_vllm_metrics_logger", False):
            return

        attempted_at_monotonic_s = time.monotonic()
        with self._vllm_metrics_lock:
            # Errors before this point belong to engine startup or warmup, not
            # the measured rollout window.
            self.vllm_metric_sampler_errors = []
            # Synchronously sample while holding the same lock as the periodic
            # thread. This is both the exact cumulative-counter baseline and a
            # pre-measurement gauge anchor.
            anchor = self._capture_vllm_metric_sample(
                "measurement_start",
                attempted_at_monotonic_s=attempted_at_monotonic_s,
            )
            self.num_preemptions_baseline = int(
                anchor.get(
                    "num_preemptions",
                    max(
                        self.num_preemptions,
                        default=self.num_preemptions_baseline,
                    ),
                )
            )
            self.generation_tokens_baseline = int(
                anchor.get(
                    "generation_tokens",
                    max(
                        self.generation_tokens,
                        default=self.generation_tokens_baseline,
                    ),
                )
            )
            self.inflight_batch_sizes = []
            self.num_pending_samples = []
            self.kv_cache_usage_perc = []
            self.generation_tokens = []
            self.num_preemptions = []
            self.vllm_metric_samples = []
            self._append_vllm_metric_sample(anchor)
            return copy.deepcopy(anchor)

    async def post_init_async(self):
        self.vllm_device_ids = await self.report_device_id_async()
        self.vllm_model_worker_runtime_provenance = (
            await self.report_model_worker_runtime_environment_async()
        )
        self._maybe_start_inflight_profiler()

    def _maybe_start_inflight_profiler(self) -> None:
        """Start the in-flight batch profiler for the async engine.

        The async engine runs its scheduler out-of-process, so (unlike the sync
        worker) we cannot read ``scheduler.running`` directly. Instead we maintain
        a live map of in-flight requests from the streamed RequestOutputs in this
        front-end process (see generate_async) and sample that. Always created
        (live map + lock) so the streaming hooks are safe; the sampler only runs
        when NRL_PROFILE_INFLIGHT is set on a model-owner actor.
        """
        # request_id -> (input_len, generated_len_so_far); maintained by the
        # _inflight_* hooks in generate_async, sampled by the profiler thread.
        self._inflight_live: dict[str, tuple[int, int]] = {}
        self._inflight_live_lock = threading.Lock()
        self._inflight_profiler: Optional[InflightProfiler] = None
        if not (
            getattr(self, "is_model_owner", False) and inflight_profiling_enabled()
        ):
            return

        def _live_sample() -> dict[str, Any]:
            with self._inflight_live_lock:
                items = list(self._inflight_live.values())
            return {
                "batch_size": len(items),
                "waiting": -1,  # not visible from the front-end in async mode
                "ctx_lens": [il + gl for il, gl in items],
                "prompt_lens": [il for il, _ in items],
                "gen_lens": [gl for _, gl in items],
            }

        self._inflight_profiler = InflightProfiler(
            sample_fn=_live_sample,
            dp_label=repr(self),
            interval_s=inflight_interval_s(),
        )
        self._inflight_profiler.start()
        self._inflight_profiler.mark_call_start()  # open the first window

    def _inflight_register(self, request_id: str, input_len: int) -> None:
        if getattr(self, "_inflight_profiler", None) is None:
            return
        with self._inflight_live_lock:
            self._inflight_live[request_id] = (int(input_len), 0)

    def _inflight_update(self, request_id: str, gen_len: int) -> None:
        if getattr(self, "_inflight_profiler", None) is None:
            return
        with self._inflight_live_lock:
            prev = self._inflight_live.get(request_id)
            if prev is not None:
                self._inflight_live[request_id] = (prev[0], int(gen_len))

    def _inflight_unregister(self, request_id: str) -> None:
        if getattr(self, "_inflight_profiler", None) is None:
            return
        with self._inflight_live_lock:
            self._inflight_live.pop(request_id, None)

    def get_inflight_timeline(self) -> list[dict[str, Any]]:
        """Return the in-flight samples collected for the current window."""
        profiler = getattr(self, "_inflight_profiler", None)
        if profiler is None:
            return []
        return profiler.snapshot()

    def clear_inflight_timeline(self) -> None:
        """Open a fresh sampling window (called per training step before rollout)."""
        profiler = getattr(self, "_inflight_profiler", None)
        if profiler is not None:
            profiler.mark_call_start()

    async def report_dp_openai_server_base_url(self) -> Optional[str]:
        return self.base_url

    # ruff: noqa
    def _setup_vllm_openai_api_server(self, app: FastAPI) -> FastAPI:
        from copy import deepcopy
        from logging import Filter as LoggingFilter
        from logging import LogRecord
        from typing import List, Optional, Union

        from fastapi import Request
        from fastapi.responses import JSONResponse, StreamingResponse
        from vllm.entrypoints.openai.chat_completion.protocol import (
            ChatCompletionRequest,
            ChatCompletionResponse,
        )
        from vllm.entrypoints.openai.chat_completion.serving import (
            OpenAIServingChat,
        )
        from vllm.entrypoints.openai.engine.protocol import ErrorResponse
        from vllm.entrypoints.openai.models.protocol import BaseModelPath
        from vllm.entrypoints.openai.models.serving import OpenAIServingModels
        from vllm.entrypoints.serve.tokenize.protocol import (
            TokenizeChatRequest,
            TokenizeCompletionRequest,
            TokenizeResponse,
        )
        from vllm.entrypoints.serve.tokenize.serving import (
            OpenAIServingTokenization,
        )
        from vllm.tool_parsers.abstract_tool_parser import ToolParserManager
        from vllm.v1.engine.async_llm import logger as vllm_async_llm_logger

        maybe_tool_parser_plugin = self.cfg["vllm_cfg"].get("tool_parser_plugin")
        if maybe_tool_parser_plugin:
            ToolParserManager.import_tool_parser(maybe_tool_parser_plugin)

        engine_client = self.llm
        model_config = self.llm_async_engine_args.create_model_config()
        base_model_paths = [
            BaseModelPath(
                name=model_config.served_model_name, model_path=model_config.model
            ),
            BaseModelPath(name=model_config.model, model_path=model_config.model),
        ]

        openai_serving_models_kwargs = dict(
            engine_client=engine_client,
            base_model_paths=base_model_paths,
            lora_modules=None,
        )
        openai_serving_models = OpenAIServingModels(**openai_serving_models_kwargs)

        class NeMoRLOpenAIChatRequestMixin:
            def model_post_init(self, context):
                # NeMo-Gym specific processing. This is just how NeMo-Gym returns the extra token information.
                if self.required_prefix_token_ids is None:
                    for message in reversed(self.messages):
                        if "prompt_token_ids" in message:
                            self.required_prefix_token_ids = (
                                message["prompt_token_ids"]
                                + message["generation_token_ids"]
                            )
                            break

                return super().model_post_init(context)

        class NeMoRLOpenAIServingMixin:
            async def _preprocess_chat(
                self,
                request,
                messages,
                default_template,
                default_template_content_format,
                default_template_kwargs,
                tool_dicts=None,
                tool_parser=None,
            ):
                # Materialize the message tool calls so we can deepcopy below.
                for message in messages:
                    if message.get("tool_calls"):
                        message["tool_calls"] = list(message["tool_calls"])

                # Deepcopy messages here since _preprocess_chat may be destructive.
                messages_for_replace_prefix_tokens = deepcopy(messages)

                # res is (conversation, [engine_prompt])
                try:
                    res = await super()._preprocess_chat(
                        request=request,
                        messages=messages,
                        default_template=default_template,
                        default_template_content_format=default_template_content_format,
                        default_template_kwargs=default_template_kwargs,
                        tool_dicts=tool_dicts,
                        tool_parser=tool_parser,
                    )
                except ValueError as e:
                    if "maximum context length" in str(e):
                        import logging

                        # Print a clean one-liner warning that max model length has been exceeded
                        # The exception is still raised, but later filtered out by the MaxContextLengthFilter
                        logging.getLogger(__name__).warning(
                            "Prompt exceeds max_model_len: %s", e
                        )
                    raise

                if request.required_prefix_token_ids is None:
                    return res

                # Find the last assistant message
                last_assistant_message_idx = None
                for i in reversed(range(len(messages_for_replace_prefix_tokens))):
                    if messages_for_replace_prefix_tokens[i]["role"] == "assistant":
                        last_assistant_message_idx = i
                        break

                if last_assistant_message_idx is None:
                    # If there's no assistant message, we just use the entire thing.
                    messages_to_last_assistant_message = (
                        messages_for_replace_prefix_tokens
                    )
                else:
                    # Include the last assistant message itself.
                    messages_to_last_assistant_message = (
                        messages_for_replace_prefix_tokens[
                            : last_assistant_message_idx + 1
                        ]
                    )

                # For the prefix token calculation, we need add_generation_prompt=False
                # to get tokens up to (and including) the last assistant message only.
                # add_generation_prompt is a field on the request that gets embedded
                # into ChatParams via build_chat_params().
                modified_request = request.model_copy(
                    update={"add_generation_prompt": False}
                )

                # Call the actual preprocess chat subroutine so we don't miss anything. Whatever they do is whatever we do since we literally do what they do.
                corresponding_res = await super()._preprocess_chat(
                    request=modified_request,
                    messages=messages_to_last_assistant_message,
                    default_template=default_template,
                    default_template_content_format=default_template_content_format,
                    default_template_kwargs=default_template_kwargs,
                    tool_dicts=tool_dicts,
                    tool_parser=tool_parser,
                )
                actual_corresponding_token_ids = corresponding_res[1][0][
                    "prompt_token_ids"
                ]

                engine_prompt = res[1][
                    0
                ]  # We need to modify engine_prompt.prompt_token_ids

                final_prompt_token_ids = _replace_prefix_tokens(
                    tokenizer=self.renderer.tokenizer,
                    model_prefix_token_ids=request.required_prefix_token_ids,
                    template_prefix_token_ids=actual_corresponding_token_ids,
                    template_token_ids=engine_prompt["prompt_token_ids"],
                )

                engine_prompt["prompt_token_ids"] = final_prompt_token_ids

                return res

        ########################################
        # /v1/chat/completions endpoint
        ########################################

        # This MRO is necessary i.e. NeMoRLOpenAIChatRequestMixin > ChatCompletionRequest
        class NeMoRLChatCompletionRequest(
            NeMoRLOpenAIChatRequestMixin, ChatCompletionRequest
        ):
            required_prefix_token_ids: Optional[List[int]] = None

        # This MRO is necessary i.e. NeMoRLOpenAIServingMixin > OpenAIServingChat
        class NeMoRLOpenAIServingChat(NeMoRLOpenAIServingMixin, OpenAIServingChat):
            pass

        serving_chat_default_kwargs = dict(
            response_role="assistant",
            request_logger=None,
            chat_template=None,
            chat_template_content_format="auto",
        )
        serving_chat_kwargs = serving_chat_default_kwargs | self.cfg["vllm_cfg"].get(
            "http_server_serving_chat_kwargs", dict()
        )
        serving_chat_kwargs.update(
            dict(
                engine_client=engine_client,
                models=openai_serving_models,
                return_tokens_as_token_ids=True,
            )
        )
        openai_serving_chat = NeMoRLOpenAIServingChat(**serving_chat_kwargs)

        generation_config = self.cfg

        # The create_chat_completion and tokenize methods are taken from vllm/entrypoints/openai/api_server.py
        @app.post("/v1/chat/completions")
        async def create_chat_completion(
            request: NeMoRLChatCompletionRequest, raw_request: Request
        ):
            # This needs to match the behavior in nemo_rl/models/generation/vllm/vllm_worker.py::BaseVllmGenerationWorker::_build_sampling_params
            # Right now we explicitly assert set this to -1.
            assert request.top_k in (None, -1), (
                f"Top k sampling parameter must be unset, empty, or -1. Got `{request.top_k}`"
            )
            request.top_k = -1

            # The request sampling params need to exactly match those as are set in NeMo RL.
            # If they do not match, the inference will be off policy and destroy training stability.
            assert request.temperature == generation_config["temperature"]
            assert request.top_p == generation_config["top_p"]

            generator = await openai_serving_chat.create_chat_completion(
                request, raw_request
            )

            if isinstance(generator, ErrorResponse):
                return JSONResponse(
                    content=generator.model_dump(), status_code=generator.error.code
                )

            elif isinstance(generator, ChatCompletionResponse):
                return JSONResponse(content=generator.model_dump())

            return StreamingResponse(content=generator, media_type="text/event-stream")

        ########################################
        # /tokenize endpoint
        ########################################

        # This MRO is necessary i.e. NeMoRLOpenAIChatRequestMixin > TokenizeRequest
        class NeMoRLTokenizeChatRequest(
            NeMoRLOpenAIChatRequestMixin, TokenizeChatRequest
        ):
            required_prefix_token_ids: Optional[List[int]] = None

        NeMoRLTokenizeRequest = Union[
            TokenizeCompletionRequest, NeMoRLTokenizeChatRequest
        ]

        # This MRO is necessary i.e. NeMoRLOpenAIServingMixin > OpenAIServingTokenization
        class NeMoRLOpenAIServingTokenization(
            NeMoRLOpenAIServingMixin, OpenAIServingTokenization
        ):
            pass

        serving_tokenization_kwargs = dict(
            request_logger=serving_chat_kwargs["request_logger"],
            chat_template=serving_chat_kwargs["chat_template"],
            chat_template_content_format=serving_chat_kwargs[
                "chat_template_content_format"
            ],
            engine_client=serving_chat_kwargs["engine_client"],
            models=serving_chat_kwargs["models"],
        )
        openai_serving_tokenization = NeMoRLOpenAIServingTokenization(
            **serving_tokenization_kwargs
        )

        @app.post("/tokenize")
        async def tokenize(request: NeMoRLTokenizeRequest, raw_request: Request):
            generator = await openai_serving_tokenization.create_tokenize(
                request, raw_request
            )

            if isinstance(generator, ErrorResponse):
                return JSONResponse(
                    content=generator.model_dump(), status_code=generator.error.code
                )
            elif isinstance(generator, TokenizeResponse):
                return JSONResponse(content=generator.model_dump())

        ########################################
        # Logging
        ########################################
        print(
            "Adding a vLLM logging filter so that the logs aren't spammed with not useful messages like `Added request ...`. This is to help errors pop up better and filter out noise."
        )

        class CleanLoggingFilter(LoggingFilter):
            def filter(self, record: LogRecord) -> bool:
                msg = record.getMessage()

                # vLLM does not accept `strict` tool definitions and reporting it to the user is not useful either.
                return (
                    "Added request" not in msg
                    and "The following fields were present in the request but ignored: {'strict'}"
                    not in msg
                )

        vllm_async_llm_logger.addFilter(CleanLoggingFilter())

        from logging import getLogger as _getLogger

        _getLogger("vllm.entrypoints.openai.engine.protocol").addFilter(
            CleanLoggingFilter()
        )

        # Suppress the noisy vLLM traceback when a prompt exceeds max_model_len.
        # This is expected during multi-turn rollouts; we log a clean one-line
        # warning from _preprocess_chat instead.
        class MaxContextLengthFilter(LoggingFilter):
            def filter(self, record: LogRecord) -> bool:
                if record.exc_info and record.exc_info[1]:
                    if "maximum context length" in str(record.exc_info[1]):
                        return False
                return True

        _getLogger("vllm.entrypoints.openai.serving_chat").addFilter(
            MaxContextLengthFilter()
        )

        return app

    def _setup_vllm_server(self) -> "tuple[threading.Thread, str, uvicorn.Server]":
        import threading
        from logging import Filter as LoggingFilter
        from logging import LogRecord, getLogger

        import uvicorn
        from fastapi import FastAPI

        # We initialize the FastAPI app here in case we want to do some generic configuration before the subsequent server inits
        # e.g. last-run middleware.
        app = FastAPI()

        app = self._setup_vllm_openai_api_server(app)

        ########################################
        # Server spinup
        ########################################

        node_ip = _get_node_ip_local()
        free_port = _get_free_port_local()

        base_url = f"http://{node_ip}:{free_port}/v1"
        print(f"Starting server on {base_url}")

        config = uvicorn.Config(
            app,
            host="0.0.0.0",
            port=free_port,
            timeout_keep_alive=120,  # Keep connections alive longer (default is 5s), fix for this error: Hit an exception while making a request (try 1): <class 'aiohttp.client_exceptions.ClientOSError'>: [Errno 104] Connection reset by peer
        )
        server = uvicorn.Server(config=config)

        print(
            "Adding a uvicorn logging filter so that the logs aren't spammed with 200 OK messages. This is to help errors pop up better and filter out noise."
        )

        class No200Filter(LoggingFilter):
            def filter(self, record: LogRecord) -> bool:
                msg = record.getMessage()
                return not msg.strip().endswith("200")

        uvicorn_logger = getLogger("uvicorn.access")
        uvicorn_logger.addFilter(No200Filter())

        thread = threading.Thread(target=server.run, daemon=True)
        thread.start()

        return thread, base_url, server

    async def init_collective_async(
        self,
        rank_prefix: int,
        ip: str,
        port: int,
        world_size: int,
        train_world_size: int,
    ) -> None:
        await self.llm.collective_rpc(
            "init_collective",
            args=(
                rank_prefix,
                ip,
                port,
                world_size,
                train_world_size,
            ),
        )

    async def generate_async(
        self,
        data: BatchedDataDict[GenerationDatumSpec],
        greedy: bool = False,
        cross_dp_frontend_submission: dict[str, Any] | None = None,
    ) -> AsyncGenerator[tuple[int, BatchedDataDict[GenerationOutputSpec]], None]:
        """Generate a batch of data using vLLM's AsyncLLMEngine, yielding results as they are ready.

        Args:
            data: BatchedDataDict with input_ids and input_lengths
            greedy: Whether to use greedy decoding instead of sampling
            cross_dp_frontend_submission: Dispatcher acknowledgement metadata.
                When present, this worker holds the per-DP dispatcher gate until
                vLLM's ``add_request`` has returned, which means EngineCore has
                accepted this request.

        Yields:
            Tuple of (original_index, BatchedDataDict conforming to GenerationOutputSpec for the single sequence)
        """
        if not self.cfg["vllm_cfg"]["async_engine"]:
            raise RuntimeError(
                "generate_async can only be used when async_engine is enabled in vLLM config."
            )

        # Handle empty input case
        if len(data["input_ids"]) == 0:
            return

        verify_right_padding(data, pad_value=self.cfg["_pad_token_id"])

        input_ids_batch = data["input_ids"]
        input_lengths_batch = data["input_lengths"]
        batch_size = input_ids_batch.shape[0]

        # Ensure generate_async only receives single samples (batch_size = 1)
        assert batch_size == 1, (
            f"generate_async is restricted to handle only single samples, "
            f"but received batch_size={batch_size}. Please handle batching outside this method."
        )

        if cross_dp_frontend_submission is not None:
            required_submission_keys = {
                "dispatcher",
                "session_id",
                "request_id",
                "assignment_sequence",
                "dp_assignment_ordinal",
                "session_dp_assignment_ordinal",
            }
            missing_submission_keys = required_submission_keys.difference(
                cross_dp_frontend_submission
            )
            if missing_submission_keys:
                raise ValueError(
                    "Cross-DP frontend submission metadata is missing "
                    f"{sorted(missing_submission_keys)}"
                )

            def first_cross_dp_data_value(key: str) -> str:
                if key not in data or len(data[key]) != 1:
                    raise ValueError(
                        "Cross-DP frontend submission requires singleton "
                        f"{key} request metadata"
                    )
                item = data[key][0]
                if hasattr(item, "item"):
                    item = item.item()
                return str(item)

            data_session_id = first_cross_dp_data_value(
                "_cross_dp_session_id"
            )
            data_request_id = first_cross_dp_data_value(
                "_cross_dp_request_id"
            )
            if data_session_id != str(
                cross_dp_frontend_submission["session_id"]
            ):
                raise ValueError(
                    "Cross-DP worker session metadata does not match its "
                    "dispatcher lease"
                )
            if data_request_id != str(
                cross_dp_frontend_submission["request_id"]
            ):
                raise ValueError(
                    "Cross-DP worker request metadata does not match its "
                    "dispatcher lease"
                )

        batch_specific_stop_strings_list = data.get(
            "stop_strings", [[] for _ in range(batch_size)]
        )

        # Create tasks for each sample in the batch
        async def process_single_sample(sample_idx):
            """Process a single sample and return the result."""
            current_input_actual_length = input_lengths_batch[sample_idx].item()
            prompt = format_prompt_for_vllm_generation(data, sample_idx)

            per_sample_stop_strings = None
            if batch_specific_stop_strings_list and sample_idx < len(
                batch_specific_stop_strings_list
            ):
                per_sample_stop_strings = batch_specific_stop_strings_list[sample_idx]

            final_stop_strings_for_sample = self._merge_stop_strings(
                [per_sample_stop_strings] if per_sample_stop_strings else None
            )

            remaining_ctx = (
                self.cfg["vllm_cfg"]["max_model_len"] - current_input_actual_length
            )
            request_max_new_tokens = self.cfg["max_new_tokens"]
            if "_nrl_max_new_tokens" in data:
                raw_request_max = data["_nrl_max_new_tokens"][sample_idx]
                if hasattr(raw_request_max, "item"):
                    raw_request_max = raw_request_max.item()
                request_max_new_tokens = int(raw_request_max)
                if request_max_new_tokens <= 0:
                    raise ValueError(
                        "_nrl_max_new_tokens must be positive, got "
                        f"{request_max_new_tokens}"
                    )
                if request_max_new_tokens > self.cfg["max_new_tokens"]:
                    raise ValueError(
                        "_nrl_max_new_tokens cannot exceed the configured "
                        f"max_new_tokens ({self.cfg['max_new_tokens']}), got "
                        f"{request_max_new_tokens}"
                    )

            force_generation_length = False
            if "_nrl_force_generation_length" in data:
                raw_force_length = data["_nrl_force_generation_length"][sample_idx]
                if hasattr(raw_force_length, "item"):
                    raw_force_length = raw_force_length.item()
                force_generation_length = bool(raw_force_length)
                if force_generation_length and "_nrl_max_new_tokens" not in data:
                    raise ValueError(
                        "_nrl_force_generation_length requires "
                        "_nrl_max_new_tokens"
                    )

            allowed_token_ids = None
            forced_generation_token_ids = None
            if "_nrl_forced_generation_token_id" in data:
                raw_forced_token = data[
                    "_nrl_forced_generation_token_id"
                ][sample_idx]
                if hasattr(raw_forced_token, "item"):
                    raw_forced_token = raw_forced_token.item()
                forced_token_id = int(raw_forced_token)
                if forced_token_id < 0:
                    raise ValueError(
                        "_nrl_forced_generation_token_id must be "
                        f"nonnegative, got {forced_token_id}"
                    )
                allowed_token_ids = [forced_token_id]
            if "_nrl_forced_generation_token_ids" in data:
                if allowed_token_ids is not None:
                    raise ValueError(
                        "single-token and exact-sequence forcing are mutually exclusive"
                    )
                raw_sequence = data["_nrl_forced_generation_token_ids"][sample_idx]
                if hasattr(raw_sequence, "tolist"):
                    raw_sequence = raw_sequence.tolist()
                forced_generation_token_ids = [int(item) for item in raw_sequence]
                if (
                    not forced_generation_token_ids
                    or any(item < 0 for item in forced_generation_token_ids)
                    or len(forced_generation_token_ids)
                    != request_max_new_tokens
                ):
                    raise ValueError(
                        "_nrl_forced_generation_token_ids must contain exactly "
                        "request_max_new_tokens non-negative token IDs"
                    )

            allowed_new_tokens = max(
                0, min(request_max_new_tokens, remaining_ctx)
            )

            # Handle case where no tokens can be generated due to length constraints
            if allowed_new_tokens == 0:
                # Access the input data directly from the function parameters
                input_ids_single_row = input_ids_batch[sample_idx]

                # Create output tensors with just the input (no generated tokens)
                output_ids_single_item_batched = input_ids_single_row[
                    :current_input_actual_length
                ].unsqueeze(0)

                logprobs_single_item = torch.zeros(
                    (1, current_input_actual_length),
                    dtype=torch.float32,
                    device=input_ids_single_row.device,
                )

                generation_lengths_tensor = torch.tensor(
                    [0], dtype=torch.long, device=input_ids_single_row.device
                )

                unpadded_sequence_lengths_tensor = torch.tensor(
                    [current_input_actual_length],
                    dtype=torch.long,
                    device=input_ids_single_row.device,
                )

                # Not truncated since no generation was attempted (length constraint)
                truncated_tensor = torch.tensor(
                    [False], dtype=torch.bool, device=input_ids_single_row.device
                )

                result_batch = BatchedDataDict[GenerationOutputSpec](
                    {
                        "output_ids": output_ids_single_item_batched,
                        "logprobs": logprobs_single_item,
                        "generation_lengths": generation_lengths_tensor,
                        "unpadded_sequence_lengths": unpadded_sequence_lengths_tensor,
                        "truncated": truncated_tensor,
                    }
                )

                return (sample_idx, result_batch)

            sampling_params_for_request = self._build_sampling_params(
                greedy=greedy,
                stop_strings=final_stop_strings_for_sample,
                max_new_tokens=allowed_new_tokens,
                force_generation_length=force_generation_length,
                allowed_token_ids=allowed_token_ids,
                forced_generation_token_ids=forced_generation_token_ids,
            )

            request_id = str(uuid.uuid4())

            # Carry the prompt-group id in vLLM's `priority` for the
            # whole-request waiting order in lfs/engine_schedulers.py.
            # This field defaults to zero when the rollout has no group id.
            lfs_group = 0
            if "lfs_group" in data:
                group_value = data["lfs_group"][sample_idx]
                lfs_group = (
                    int(group_value.item())
                    if hasattr(group_value, "item")
                    else int(group_value)
                )

            # The causal benchmark opts into same-host monotonic timestamps
            # that distinguish middleware admission from first engine progress.
            record_engine_progress = (
                os.environ.get("CROSS_DP_PERF_REQUEST_TIMELINE", "0") == "1"
            )
            engine_frontend_submit_at_monotonic_s = None
            engine_frontend_submit_at_unix_s = None
            engine_frontend_hostname = (
                os.uname().nodename if record_engine_progress else None
            )

            async def record_frontend_submission(
                submitted_at_unix_s: float,
                submitted_at_monotonic_s: float,
                submitted_hostname: str,
            ) -> None:
                nonlocal engine_frontend_submit_at_monotonic_s
                nonlocal engine_frontend_submit_at_unix_s
                nonlocal engine_frontend_hostname

                if record_engine_progress:
                    engine_frontend_submit_at_monotonic_s = (
                        submitted_at_monotonic_s
                    )
                    engine_frontend_submit_at_unix_s = submitted_at_unix_s
                    engine_frontend_hostname = submitted_hostname
                if cross_dp_frontend_submission is None:
                    return

                dispatcher = cross_dp_frontend_submission["dispatcher"]
                await dispatcher.confirm_engine_frontend_submitted.remote(
                    str(cross_dp_frontend_submission["request_id"]),
                    int(
                        cross_dp_frontend_submission[
                            "assignment_sequence"
                        ]
                    ),
                    int(
                        cross_dp_frontend_submission[
                            "dp_assignment_ordinal"
                        ]
                    ),
                    int(
                        cross_dp_frontend_submission[
                            "session_dp_assignment_ordinal"
                        ]
                    ),
                    submitted_at_unix_s,
                    submitted_at_monotonic_s,
                    submitted_hostname,
                )

            # Use one explicit add_request/collector path for vanilla and
            # cross-DP requests. Besides making timestamps comparable, this is
            # required for a causal gate: constructing AsyncLLM.generate's
            # async generator does not execute its add_request call.
            async def generate_after_engine_submission():
                from vllm.outputs import STREAM_FINISHED
                from vllm.v1.engine.async_llm import InputStreamError
                from vllm.v1.engine.exceptions import (
                    EngineDeadError,
                    EngineGenerateError,
                )

                request_output_collector = None

                async def abort_registered_request() -> None:
                    if request_output_collector is not None:
                        await self.llm.abort(
                            request_output_collector.request_id,
                            internal=True,
                        )

                try:
                    (
                        request_output_collector,
                        submitted_at_unix_s,
                        submitted_at_monotonic_s,
                        submitted_hostname,
                    ) = await _submit_vllm_request(
                        self.llm,
                        prompt=prompt,
                        sampling_params=sampling_params_for_request,
                        request_id=request_id,
                        priority=lfs_group,
                    )
                    await record_frontend_submission(
                        submitted_at_unix_s,
                        submitted_at_monotonic_s,
                        submitted_hostname,
                    )
                    async for req_output in _iterate_request_output_collector(
                        request_output_collector,
                        STREAM_FINISHED,
                    ):
                        yield req_output
                except (asyncio.CancelledError, GeneratorExit):
                    await abort_registered_request()
                    raise
                except EngineDeadError:
                    raise
                except ValueError:
                    await abort_registered_request()
                    raise
                except InputStreamError as error:
                    await abort_registered_request()
                    raise error.cause from error
                except Exception as error:
                    await abort_registered_request()
                    raise EngineGenerateError() from error
                finally:
                    if request_output_collector is not None:
                        request_output_collector.close()

            vllm_request_generator = generate_after_engine_submission()

            # Track this request's live context length for the in-flight profiler.
            # The async scheduler is out-of-process, so we reconstruct the in-flight
            # state from the streamed RequestOutputs here in the front-end.
            self._inflight_register(request_id, current_input_actual_length)
            # Get the final result from the generator while updating the profiler's
            # front-end view of the request's generated length.
            final_request_output = None
            engine_first_token_at_monotonic_s = None
            engine_first_token_at_unix_s = None
            engine_first_observed_generated_tokens = None
            try:
                async for req_output in vllm_request_generator:
                    final_request_output = req_output
                    if req_output.outputs:
                        if (
                            record_engine_progress
                            and engine_first_token_at_monotonic_s is None
                            and req_output.outputs[0].token_ids
                        ):
                            engine_first_token_at_monotonic_s = time.monotonic()
                            engine_first_token_at_unix_s = time.time()
                            engine_first_observed_generated_tokens = len(
                                req_output.outputs[0].token_ids
                            )
                        self._inflight_update(
                            request_id, len(req_output.outputs[0].token_ids)
                        )
            finally:
                self._inflight_unregister(request_id)

            if final_request_output is None:
                raise RuntimeError(f"No output received for request {request_id}")
            if (
                record_engine_progress
                and engine_first_token_at_monotonic_s is None
            ):
                raise RuntimeError(
                    f"Request {request_id} completed without an observed token"
                )

            # Process the output
            generation_details = final_request_output.outputs[0]
            generated_token_ids = list(generation_details.token_ids)
            num_generated_tokens = len(generated_token_ids)

            original_input_ids_single_row = input_ids_batch[sample_idx]
            final_output_tensor_len = current_input_actual_length + num_generated_tokens

            # Create output_ids tensor for this single item
            output_ids_single_item = torch.full(
                (final_output_tensor_len,),
                self.cfg["_pad_token_id"],
                dtype=original_input_ids_single_row.dtype,
                device=original_input_ids_single_row.device,
            )
            # Copy original input (up to its actual length)
            output_ids_single_item[:current_input_actual_length] = (
                original_input_ids_single_row[:current_input_actual_length]
            )
            # Add generated tokens after the actual input
            output_ids_single_item[
                current_input_actual_length : current_input_actual_length
                + num_generated_tokens
            ] = torch.tensor(
                generated_token_ids,
                dtype=original_input_ids_single_row.dtype,
                device=original_input_ids_single_row.device,
            )

            # Reshape to (1, seq_len) for BatchedDataDict
            output_ids_single_item_batched = output_ids_single_item.unsqueeze(0)

            # Create logprobs tensor for this single item
            logprobs_single_item = torch.zeros(
                (1, final_output_tensor_len),
                dtype=torch.float32,
                device=original_input_ids_single_row.device,
            )
            if hasattr(generation_details, "logprobs") and generation_details.logprobs:
                for idx, logprob_dict_per_token in enumerate(
                    generation_details.logprobs
                ):
                    if logprob_dict_per_token and idx < len(generated_token_ids):
                        token_id_at_idx = generated_token_ids[idx]
                        if token_id_at_idx in logprob_dict_per_token:
                            logprob_value = logprob_dict_per_token[
                                token_id_at_idx
                            ].logprob
                            position_in_output_tensor = (
                                current_input_actual_length + idx
                            )
                            if position_in_output_tensor < final_output_tensor_len:
                                logprobs_single_item[0, position_in_output_tensor] = (
                                    logprob_value
                                )

            # Generation lengths
            generation_lengths_tensor = torch.tensor(
                [num_generated_tokens],
                dtype=torch.long,
                device=original_input_ids_single_row.device,
            )

            # Unpadded sequence lengths (actual_input + actual_generated)
            unpadded_total_length = current_input_actual_length + num_generated_tokens
            unpadded_sequence_lengths_tensor = torch.tensor(
                [unpadded_total_length],
                dtype=torch.long,
                device=original_input_ids_single_row.device,
            )

            # Check if response was truncated (hit max_tokens length limit)
            is_truncated = generation_details.finish_reason == "length"
            truncated_tensor = torch.tensor(
                [is_truncated],
                dtype=torch.bool,
                device=original_input_ids_single_row.device,
            )

            result_data: dict[str, Any] = {
                "output_ids": output_ids_single_item_batched,
                "logprobs": logprobs_single_item,
                "generation_lengths": generation_lengths_tensor,
                "unpadded_sequence_lengths": unpadded_sequence_lengths_tensor,
                "truncated": truncated_tensor,
            }
            if record_engine_progress:
                result_data.update(
                    {
                        "gen_engine_frontend_submit_at_monotonic_s": [
                            engine_frontend_submit_at_monotonic_s
                        ],
                        "gen_engine_frontend_submit_at_unix_s": [
                            engine_frontend_submit_at_unix_s
                        ],
                        "gen_engine_first_token_at_monotonic_s": [
                            engine_first_token_at_monotonic_s
                        ],
                        "gen_engine_first_token_at_unix_s": [
                            engine_first_token_at_unix_s
                        ],
                        "gen_engine_first_observed_generated_tokens": [
                            engine_first_observed_generated_tokens
                        ],
                        "gen_engine_frontend_hostname": [
                            engine_frontend_hostname
                        ],
                    }
                )
            result_batch = BatchedDataDict[GenerationOutputSpec](result_data)

            return (sample_idx, result_batch)

        # Create tasks for all samples and yield results as they complete
        sample_tasks = [
            asyncio.create_task(process_single_sample(i)) for i in range(batch_size)
        ]

        # Yield results as they become available
        for completed_task in asyncio.as_completed(sample_tasks):
            try:
                result = await completed_task
                yield result
            except Exception as e:
                # Cancel remaining tasks
                for task in sample_tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*sample_tasks, return_exceptions=True)
                raise e

    async def generate_text_async(
        self, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> AsyncGenerator[tuple[int, BatchedDataDict[GenerationOutputSpec]], None]:
        """Generate text responses asynchronously, yielding results as they are ready.

        Args:
            data: BatchedDataDict containing prompts with text strings
            greedy: Whether to use greedy decoding instead of sampling

        Yields:
            Tuple of (original_index, BatchedDataDict containing single text response)
        """
        if not self.cfg["vllm_cfg"]["async_engine"]:
            raise RuntimeError(
                "generate_text_async can only be used when async_engine is enabled in vLLM config."
            )

        # Handle empty input case
        if len(data["prompts"]) == 0:
            return

        prompts = data["prompts"]
        batch_size = len(prompts)

        # Extract stop_strings if provided, else use default from config
        batch_stop_strings: list[list[str] | None] = data.get(
            "stop_strings", [self.cfg.get("stop_strings")] * batch_size
        )

        # Create tasks for each prompt
        async def process_single_prompt(prompt_idx):
            """Process a single prompt and return the result."""
            prompt = prompts[prompt_idx]

            # Get stop strings for this specific prompt
            per_prompt_stop_strings = None
            if batch_stop_strings and prompt_idx < len(batch_stop_strings):
                per_prompt_stop_strings = batch_stop_strings[prompt_idx]

            # Merge stop strings
            final_stop_strings = self._merge_stop_strings(
                [per_prompt_stop_strings] if per_prompt_stop_strings else None
            )

            # Create sampling parameters
            top_k = self.cfg["top_k"] if self.cfg["top_k"] is not None else -1
            sampling_params = self.SamplingParams(
                temperature=self.cfg["temperature"] if not greedy else 0,
                top_p=self.cfg["top_p"],
                top_k=top_k if not greedy else 1,
                max_tokens=self.cfg["max_new_tokens"],
                stop_token_ids=self.cfg["stop_token_ids"],
                stop=final_stop_strings,
                include_stop_str_in_output=True,  # returning stop strings like hf
            )

            request_id = str(uuid.uuid4())

            # Generate using vLLM async engine
            vllm_request_generator = self.llm.generate(
                prompt=prompt,
                sampling_params=sampling_params,
                request_id=request_id,
            )

            # Get the final result from the generator
            final_request_output = None
            async for req_output in vllm_request_generator:
                final_request_output = req_output

            if final_request_output is None:
                raise RuntimeError(f"No output received for request {request_id}")

            # Extract the generated text
            generated_text = final_request_output.outputs[0].text

            # Create result in BatchedDataDict format
            result_batch = BatchedDataDict[GenerationOutputSpec](
                {"texts": [generated_text]}
            )

            return (prompt_idx, result_batch)

        # Create tasks for all prompts and yield results as they complete
        prompt_tasks = [
            asyncio.create_task(process_single_prompt(i)) for i in range(batch_size)
        ]

        # Yield results as they become available
        for completed_task in asyncio.as_completed(prompt_tasks):
            try:
                result = await completed_task
                yield result
            except Exception as e:
                # Cancel remaining tasks
                for task in prompt_tasks:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*prompt_tasks, return_exceptions=True)
                raise e

    async def report_device_id_async(self) -> list[str]:
        """Async version of report_device_id."""
        assert self.llm is not None, (
            "Attempting to report device id with either an uninitialized vLLM or non-model-owner"
        )

        if not self.cfg["vllm_cfg"]["async_engine"]:
            raise RuntimeError(
                "report_device_id_async can only be used with async_engine=True. Use report_device_id instead."
            )

        result_or_coro = await self.llm.collective_rpc("report_device_id", args=tuple())

        if asyncio.iscoroutine(result_or_coro):
            list_of_worker_results = await result_or_coro
        else:
            list_of_worker_results = result_or_coro

        return cast(list[str], list_of_worker_results)

    async def report_model_worker_runtime_environment_async(
        self,
    ) -> list[dict[str, Any]]:
        """Collect non-secret environment proof from every TP worker."""

        assert self.llm is not None, (
            "Attempting to inspect an uninitialized vLLM"
        )
        result_or_coro = await self.llm.collective_rpc(
            "report_runtime_environment",
            args=tuple(),
        )
        if asyncio.iscoroutine(result_or_coro):
            worker_results = await result_or_coro
        else:
            worker_results = result_or_coro
        return cast(list[dict[str, Any]], worker_results)

    async def prepare_refit_info_async(self, state_dict_info: dict[str, Any]) -> None:
        """Async version of prepare_refit_info."""
        await self.llm.collective_rpc("prepare_refit_info", args=(state_dict_info,))

    async def update_weights_via_ipc_zmq_async(
        self,
    ) -> bool:
        """Async version of update_weights_via_ipc_zmq."""
        try:
            assert self.llm is not None, (
                "Attempting to update weights with either an uninitialized vLLM or non-model-owner"
            )

            if not self.cfg["vllm_cfg"]["async_engine"]:
                raise RuntimeError(
                    "update_weights_via_ipc_zmq_async can only be used with async_engine=True. Use update_weights_via_ipc_zmq instead."
                )

            # TODO: switch to update_weights_from_local_ipc_handles for better performance once collectively report_device_id is supported in asyncLLM initialization
            result_or_coro = await self.llm.collective_rpc(
                "update_weights_via_ipc_zmq", args=tuple()
            )

            if asyncio.iscoroutine(result_or_coro):
                worker_results = await result_or_coro
            else:
                worker_results = result_or_coro

            worker_result = worker_results[0]

            if not worker_result:
                print(
                    f"Error: Worker failed to update weights. Result: {worker_result}"
                )
                return False
            return True
        except Exception as e:
            print(f"Exception during collective_rpc for weight update: {e}")
            import traceback

            traceback.print_exc()
            return False

    async def update_weights_from_collective_async(self) -> bool:
        """Async version of update_weights_from_collective."""
        try:
            assert self.llm is not None, (
                "Attempting to update weights with either an uninitialized vLLM or non-model-owner"
            )

            if not self.cfg["vllm_cfg"]["async_engine"]:
                raise RuntimeError(
                    "update_weights_from_collective_async can only be used with async_engine=True. Use update_weights_from_collective instead."
                )

            result_or_coro = await self.llm.collective_rpc(
                "update_weights_from_collective", args=tuple()
            )

            if asyncio.iscoroutine(result_or_coro):
                worker_results = await result_or_coro
            else:
                worker_results = result_or_coro

            worker_result = worker_results[0]

            if not worker_result:
                print(
                    f"Error: Worker failed to update weights. Result: {worker_result}"
                )
                return False
            return True
        except Exception as e:
            print(f"Exception during collective_rpc for weight update: {e}")
            import traceback

            traceback.print_exc()
            return False

    async def reset_prefix_cache_async(self):
        """Async version of reset_prefix_cache."""
        assert self.llm is not None, (
            "Attempting to reset prefix cache with either an uninitialized vLLM or non-model-owner"
        )

        if not self.cfg["vllm_cfg"]["async_engine"]:
            raise RuntimeError(
                "reset_prefix_cache_async can only be used with async_engine=True. Use reset_prefix_cache instead."
            )

        await self.llm.reset_prefix_cache()
        gc.collect()
        torch.cuda.empty_cache()

    async def sleep_async(self):
        """Async version of sleep."""
        assert self.llm is not None, (
            "Attempting to sleep with either an uninitialized vLLM or non-model-owner"
        )

        if not self.cfg["vllm_cfg"]["async_engine"]:
            raise RuntimeError(
                "sleep_async can only be used with async_engine=True. Use sleep instead."
            )

        # Reset the prefix cache to ensure that prefix cache is not reused after weights are updated
        await self.llm.reset_prefix_cache()
        # Reset the multimodal processor cache (sender side) so it stays in
        # sync with the receiver cache that vLLM clears internally during
        # sleep.  Without this, the sender thinks images are already cached on
        # the receiver and sends data=None, causing an assertion error.
        if hasattr(self.llm, "reset_mm_cache"):
            await self.llm.reset_mm_cache()
        await self.llm.sleep(level=1)

        gc.collect()
        torch.cuda.empty_cache()

    async def wake_up_async(self, **kwargs):
        """Async version of wake_up."""
        assert self.llm is not None, (
            "Attempting to wake up with either an uninitialized vLLM or non-model-owner"
        )

        if not self.cfg["vllm_cfg"]["async_engine"]:
            raise RuntimeError(
                "wake_up_async can only be used with async_engine=True. Use wake_up instead."
            )

        tags = kwargs.get("tags")

        wake_up_args = {}
        if tags is not None:
            wake_up_args["tags"] = tags

        await self.llm.wake_up(**wake_up_args)

    async def shutdown(self) -> bool:
        """Clean up vLLM resources."""
        try:
            metrics_stop_event = getattr(
                self, "_vllm_metrics_logger_stop_event", None
            )
            if metrics_stop_event is not None:
                metrics_stop_event.set()
            metrics_thread = getattr(self, "_vllm_metrics_logger_thread", None)
            if metrics_thread is not None and metrics_thread.is_alive():
                metrics_thread.join(timeout=2.0)
                if metrics_thread.is_alive():
                    print(
                        "Warning: vLLM metrics logger did not stop within 2s",
                        flush=True,
                    )

            if self.llm is not None:
                # Clean up extension resources (e.g., ZMQ sockets)
                await self.llm.collective_rpc("cleanup", args=tuple())
                try:
                    self.llm.shutdown()
                except Exception as e_stop:
                    print(f"Error calling shutdown_background_loop: {e_stop}")

                # Explicitly delete the engine. This may trigger its __del__ method.
                del self.llm

            self.llm = None
            self.tokenizer = None

            # Force garbage collection
            gc.collect()
            torch.cuda.empty_cache()

            # getattr: non-model-owner workers never run _create_engine, which
            # is where server_thread is initialized.
            if getattr(self, "server_thread", None) is not None:
                from threading import Thread

                from uvicorn import Server

                self.http_server: Server
                self.server_thread: Thread

                self.http_server.should_exit = True
                self.server_thread.join()

            return True
        except Exception as e:
            print(f"Error during vLLM shutdown: {e}")
            return False
