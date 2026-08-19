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
"""Generation and NCCL refit through a driver-owned Dynamo vLLM fleet."""

import asyncio
import logging
from typing import Any, AsyncGenerator, Optional

import ray
import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.generation.dynamo.config import DynamoConfig
from nemo_rl.models.generation.dynamo.http_client import (
    async_http_post_json,
    format_dynamo_error,
)
from nemo_rl.models.generation.dynamo.managed_runtime import ManagedDynamoRuntime
from nemo_rl.models.generation.dynamo.metrics import DynamoMetricsSampler
from nemo_rl.models.generation.dynamo.refit import DynamoRefitChannel
from nemo_rl.models.generation.dynamo.token_wrapper import DynamoTokenWrapperServer
from nemo_rl.models.generation.interfaces import (
    CollectiveSenderSpec,
    GenerationDatumSpec,
    GenerationInterface,
    GenerationOutputSpec,
    verify_right_padding,
)

LOGGER = logging.getLogger(__name__)

_HTTP_MAX_ATTEMPTS = 3
_HTTP_RETRY_DELAY_S = 1.0
_RETRYABLE_HTTP_STATUS_CODES = {408, 429}


def _is_retryable_http_response(response: Any) -> bool:
    """Return whether an internal HTTP error shape represents a transient error."""
    if not isinstance(response, dict):
        return False
    if "transport_error" in response or response.get("json_decode_error") is True:
        return True
    status = response.get("http_status")
    return isinstance(status, int) and (
        status in _RETRYABLE_HTTP_STATUS_CODES or 500 <= status < 600
    )


def _parse_dynamo_completion_response(
    response: dict[str, Any], *, request_url: str
) -> tuple[list[int], list[float], bool]:
    """Parse the Dynamo OpenAI completion response for direct generation."""
    if not isinstance(response, dict):
        raise RuntimeError(
            f"Dynamo completion response from {request_url} was not a JSON object."
        )
    if response.get("status") == "error":
        raise RuntimeError(
            f"Dynamo completion request to {request_url} failed: "
            f"{format_dynamo_error(response)}"
        )

    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        raise RuntimeError(
            f"Dynamo completion response from {request_url} did not include choices."
        )
    choice = choices[0]
    if not isinstance(choice, dict):
        raise RuntimeError(
            f"Dynamo completion response from {request_url} has invalid choice shape."
        )

    nvext = response.get("nvext")
    if not isinstance(nvext, dict):
        raise RuntimeError(
            f"Dynamo completion response from {request_url} did not include nvext."
        )
    completion_token_ids = nvext.get("completion_token_ids")
    if not isinstance(completion_token_ids, list):
        raise RuntimeError(
            "Dynamo completion response did not include "
            "nvext.completion_token_ids. Ensure the Dynamo frontend is "
            "configured to return completion token IDs."
        )
    generated_token_ids = [int(token_id) for token_id in completion_token_ids]

    if not generated_token_ids:
        return (
            generated_token_ids,
            [],
            choice.get("finish_reason") == "length",
        )

    logprobs = choice.get("logprobs")
    if not isinstance(logprobs, dict):
        raise RuntimeError(
            f"Dynamo completion response from {request_url} did not include "
            "choice.logprobs."
        )
    token_logprobs = logprobs.get("token_logprobs")
    if not isinstance(token_logprobs, list):
        raise RuntimeError(
            f"Dynamo completion response from {request_url} did not include "
            "choice.logprobs.token_logprobs."
        )
    if len(token_logprobs) != len(generated_token_ids):
        raise RuntimeError(
            f"Dynamo completion response from {request_url} returned "
            f"{len(token_logprobs)} token logprobs for "
            f"{len(generated_token_ids)} generated tokens."
        )

    generated_logprobs = []
    for idx, logprob in enumerate(token_logprobs):
        if not isinstance(logprob, (int, float)) or isinstance(logprob, bool):
            raise RuntimeError(
                f"Dynamo completion response from {request_url} returned invalid "
                f"logprob {logprob!r} for generated token {idx}."
            )
        generated_logprobs.append(float(logprob))

    return (
        generated_token_ids,
        generated_logprobs,
        choice.get("finish_reason") == "length",
    )


class DynamoGeneration(GenerationInterface):
    """Own a fixed Dynamo service fleet and expose it for NeMo-RL rollouts."""

    def __init__(
        self,
        cluster: Optional[RayVirtualCluster],
        config: dict[str, Any],
        tokenizer: Any | None = None,
        tokenizer_config: Optional[dict[str, Any]] = None,
    ):
        validated_config = DynamoConfig.model_validate(config)
        self.cfg = validated_config.model_dump()
        self._dynamo_cfg = validated_config.dynamo_cfg
        dynamo_cfg = self._dynamo_cfg
        vllm_cfg = validated_config.vllm_cfg
        expose_http_server = vllm_cfg.expose_http_server
        tokenizer_chat_template_kwargs: Optional[dict[str, Any]] = None
        if expose_http_server:
            if tokenizer is None:
                raise RuntimeError(
                    "DynamoGeneration requires a tokenizer when exposing an "
                    "OpenAI-compatible rollout server."
                )
            if (
                tokenizer_config is not None
                and "chat_template_kwargs" in tokenizer_config
                and tokenizer_config["chat_template_kwargs"] is not None
            ):
                chat_template_kwargs = tokenizer_config["chat_template_kwargs"]
                if not isinstance(chat_template_kwargs, dict):
                    raise RuntimeError(
                        "policy.tokenizer.chat_template_kwargs must be a dictionary."
                    )
                tokenizer_chat_template_kwargs = dict(chat_template_kwargs)
        if cluster is None:
            raise RuntimeError(
                "Managed Dynamo requires a non-colocated inference RayVirtualCluster."
            )
        self._managed_runtime: Optional[ManagedDynamoRuntime] = ManagedDynamoRuntime(
            cluster=cluster,
            config=self.cfg,
        )
        self._token_wrapper_server: Optional[DynamoTokenWrapperServer] = None
        self._dynamo_frontend_base_url = ""
        self.dp_openai_server_base_urls: list[Optional[str]] = []
        self._refit_channel: DynamoRefitChannel | None = None
        self._metrics_sampler: DynamoMetricsSampler | None = None
        try:
            self._managed_runtime.start()
            url = self._managed_runtime.frontend_url
            self._dynamo_frontend_base_url = url
            workers = self._managed_runtime.refit_workers()
            self._refit_channel = DynamoRefitChannel(
                workers,
                engine_world_size=validated_config.engine_world_size,
                control_timeout_s=dynamo_cfg.control_timeout_s,
                validate_workers=self._managed_runtime.validate_workers,
            )

            if expose_http_server:
                self._token_wrapper_server = DynamoTokenWrapperServer(
                    dynamo_frontend_base_url=url,
                    tokenizer=tokenizer,
                    tokenizer_chat_template_kwargs=tokenizer_chat_template_kwargs,
                    exclude_tools_when_tool_choice_none=(
                        dynamo_cfg.worker_args.exclude_tools_when_tool_choice_none
                    ),
                    request_timeout_s=dynamo_cfg.request_timeout_s,
                )
                wrapper_url = self._token_wrapper_server.start()
                self.dp_openai_server_base_urls = [wrapper_url]
                print(
                    "  [Dynamo] Forwarding rollout chat requests through token "
                    f"wrapper {wrapper_url} -> {url}",
                    flush=True,
                )
            else:
                self.dp_openai_server_base_urls = [None]
                print(f"  [Dynamo] Forwarding rollouts to {url}", flush=True)

            if vllm_cfg.enable_vllm_metrics_logger:
                self._metrics_sampler = DynamoMetricsSampler(
                    workers,
                    interval_s=vllm_cfg.vllm_metrics_logger_interval,
                    include_prefixes=dynamo_cfg.metrics_include_prefixes,
                    exclude_prefixes=dynamo_cfg.metrics_exclude_prefixes,
                )
                self._metrics_sampler.start()
        except Exception:
            self.shutdown()
            raise

    # ------------------------------------------------------------------
    # GenerationInterface — lifecycle
    # ------------------------------------------------------------------

    def prepare_for_generation(self, *args: Any, **kwargs: Any) -> bool:
        return True

    @property
    def frontend_url(self) -> str:
        """Return the internal managed Dynamo OpenAI frontend URL."""
        if not self._dynamo_frontend_base_url:
            raise RuntimeError("DynamoGeneration does not have a frontend URL.")
        return self._dynamo_frontend_base_url

    def finish_generation(self, *args: Any, **kwargs: Any) -> bool:
        """Invalidate cached rollout state after synchronous generation."""
        return self.invalidate_kv_cache()

    def get_logger_metrics(self) -> dict[str, Any]:
        """Return per-worker Dynamo metric timelines for generation logging."""
        sampler = self._metrics_sampler
        return {} if sampler is None else sampler.snapshot()

    def clear_logger_metrics(self) -> None:
        """Clear the Dynamo metric timelines for the next logging window."""
        sampler = self._metrics_sampler
        if sampler is not None:
            sampler.clear()

    def get_inference_world_size(self) -> int:
        """Return the number of vLLM ranks across all discovered workers."""
        channel = self._refit_channel
        if channel is None:
            raise RuntimeError("Dynamo refit channel is unavailable")
        return channel.inference_world_size

    def get_collective_sender_spec(self) -> CollectiveSenderSpec:
        """Return vLLM's NCCL protocol and packed-transfer geometry."""
        channel = self._refit_channel
        if channel is None:
            raise RuntimeError("Dynamo refit channel is unavailable")
        return channel.sender_spec

    def shutdown(self) -> bool:
        """Stop process-local helpers and any driver-owned managed runtime."""
        sampler = self._metrics_sampler
        self._metrics_sampler = None
        if sampler is not None:
            try:
                sampler.shutdown()
            except Exception:
                LOGGER.exception("Failed to stop the Dynamo metrics sampler")
        token_wrapper_server = self._token_wrapper_server
        self._token_wrapper_server = None
        if token_wrapper_server is not None:
            try:
                token_wrapper_server.shutdown()
            except Exception:
                LOGGER.exception("Failed to stop the Dynamo token wrapper")
        managed_runtime = self._managed_runtime
        self._managed_runtime = None
        if managed_runtime is not None:
            try:
                managed_runtime.shutdown()
            except Exception:
                LOGGER.exception("Failed to stop the managed Dynamo runtime")
        self._refit_channel = None
        return True

    # ------------------------------------------------------------------
    # Pickling — async rollouts ship the GenerationInterface across Ray actors
    # ------------------------------------------------------------------

    def __getstate__(self) -> dict[str, Any]:
        """Serialize only HTTP clients needed by Ray rollout actors.

        Driver-owned subprocesses, threads, and Ray worker handles are excluded.
        The endpoint-only refit channel is retained so AREAL-style cache
        invalidation still reaches every managed worker after deserialization.
        """
        refit_channel = self._refit_channel
        return {
            "cfg": self.cfg,
            "dp_openai_server_base_urls": self.dp_openai_server_base_urls,
            "_dynamo_frontend_base_url": self._dynamo_frontend_base_url,
            "_refit_channel": (
                None if refit_channel is None else refit_channel.client_copy()
            ),
        }

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore a client-only rollout copy with no service ownership."""
        self.cfg = state["cfg"]
        validated_config = DynamoConfig.model_validate(self.cfg)
        self._dynamo_cfg = validated_config.dynamo_cfg
        self.dp_openai_server_base_urls = state["dp_openai_server_base_urls"]
        frontend_url = state["_dynamo_frontend_base_url"]
        if not isinstance(frontend_url, str) or not frontend_url:
            raise RuntimeError("Pickled DynamoGeneration has no frontend URL.")
        self._dynamo_frontend_base_url = frontend_url
        self._token_wrapper_server = None
        self._managed_runtime = None
        self._metrics_sampler = None
        self._refit_channel = state["_refit_channel"]

    def _completion_url(self) -> str:
        base_url = self._dynamo_frontend_base_url
        if not base_url:
            raise RuntimeError("DynamoGeneration does not have a frontend URL.")
        return f"{base_url.rstrip('/')}/completions"

    def _request_timeout_s(self) -> float:
        return self._dynamo_cfg.request_timeout_s

    def _merge_stop_strings(self, batch_stop_strings: Any) -> Optional[list[str]]:
        stop_set: set[str] = set()

        configured_stop_strings = self.cfg.get("stop_strings")
        if configured_stop_strings is not None:
            stop_set.update(configured_stop_strings)

        if batch_stop_strings is not None:
            for sample_stop_strings in batch_stop_strings:
                if not sample_stop_strings:
                    continue
                if isinstance(sample_stop_strings, str):
                    stop_set.add(sample_stop_strings)
                else:
                    stop_set.update(sample_stop_strings)

        if len(stop_set) > 32:
            raise ValueError(
                "Dynamo supports at most 32 stop strings after merging configured "
                "and per-sample values"
            )
        return list(stop_set) if stop_set else None

    def _prompt_token_ids(
        self,
        data: BatchedDataDict["GenerationDatumSpec"],
        sample_idx: int,
    ) -> list[int]:
        if "vllm_content" in data:
            raise NotImplementedError(
                "DynamoGeneration direct generate() supports token-ID LLM "
                "prompts only; multimodal vllm_content is not supported."
            )

        input_length = int(data["input_lengths"][sample_idx].item())
        return data["input_ids"][sample_idx, :input_length].tolist()

    def _build_completion_request(
        self,
        *,
        prompt_token_ids: list[int],
        greedy: bool,
        stop_strings: Optional[list[str]],
        max_new_tokens: int,
    ) -> dict[str, Any]:
        top_k_cfg = self.cfg["top_k"]
        top_k_val = 1 if greedy else (top_k_cfg if top_k_cfg is not None else -1)

        payload: dict[str, Any] = {
            "model": self.cfg["model_name"],
            "prompt": prompt_token_ids,
            "max_tokens": int(max_new_tokens),
            "temperature": 0.0 if greedy else self.cfg["temperature"],
            "top_p": self.cfg["top_p"],
            "top_k": top_k_val,
            "n": 1,
            "logprobs": 0,
            "include_stop_str_in_output": True,
            "nvext": {"extra_fields": ["completion_token_ids"]},
        }

        if self.cfg["stop_token_ids"] is not None:
            payload["stop_token_ids"] = self.cfg["stop_token_ids"]
        if stop_strings is not None:
            payload["stop"] = stop_strings

        return payload

    def _allowed_new_tokens(self, input_length: int) -> int:
        """Return the generation budget for a prompt."""
        remaining_ctx = int(self.cfg["vllm_cfg"]["max_model_len"]) - input_length
        if remaining_ctx <= 0:
            raise ValueError(
                f"Dynamo prompt length {input_length} must be less than "
                f"vllm_cfg.max_model_len={self.cfg['vllm_cfg']['max_model_len']}"
            )
        return min(self.cfg["max_new_tokens"], remaining_ctx)

    def _assert_response_within_context(
        self, *, input_length: int, generated_length: int
    ) -> None:
        response_length = input_length + generated_length
        max_model_len = int(self.cfg["vllm_cfg"]["max_model_len"])
        if response_length > max_model_len:
            raise AssertionError(
                "Dynamo response length exceeded "
                f"vllm_cfg.max_model_len: {response_length} > {max_model_len}"
            )

    async def _post_completion_request(
        self,
        *,
        prompt_token_ids: list[int],
        greedy: bool,
        stop_strings: Optional[list[str]],
        max_new_tokens: int,
    ) -> tuple[list[int], list[float], bool]:
        request_url = self._completion_url()
        payload = self._build_completion_request(
            prompt_token_ids=prompt_token_ids,
            greedy=greedy,
            stop_strings=stop_strings,
            max_new_tokens=max_new_tokens,
        )
        response: dict[str, Any] = {}
        for attempt in range(1, _HTTP_MAX_ATTEMPTS + 1):
            response = await async_http_post_json(
                request_url,
                payload,
                self._request_timeout_s(),
            )
            if not _is_retryable_http_response(response):
                break
            if attempt == _HTTP_MAX_ATTEMPTS:
                break
            LOGGER.warning(
                "Dynamo completion attempt %d/%d failed; retrying in %.1fs: %s",
                attempt,
                _HTTP_MAX_ATTEMPTS,
                _HTTP_RETRY_DELAY_S,
                format_dynamo_error(response),
            )
            await asyncio.sleep(_HTTP_RETRY_DELAY_S)
        return _parse_dynamo_completion_response(response, request_url=request_url)

    def _single_sample_output(
        self,
        *,
        input_ids: torch.Tensor,
        input_length: int,
        generated_token_ids: list[int],
        generated_logprobs: list[float],
        truncated: bool,
    ) -> BatchedDataDict["GenerationOutputSpec"]:
        output_length = input_length + len(generated_token_ids)
        self._assert_response_within_context(
            input_length=input_length,
            generated_length=len(generated_token_ids),
        )
        output_ids = torch.full(
            (output_length,),
            self.cfg["_pad_token_id"],
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        output_ids[:input_length] = input_ids[:input_length]
        if generated_token_ids:
            output_ids[input_length:output_length] = torch.tensor(
                generated_token_ids,
                dtype=input_ids.dtype,
                device=input_ids.device,
            )

        logprobs = torch.zeros(
            (1, output_length),
            dtype=torch.float32,
            device=input_ids.device,
        )
        for idx, logprob in enumerate(generated_logprobs[: len(generated_token_ids)]):
            logprobs[0, input_length + idx] = logprob

        return BatchedDataDict[GenerationOutputSpec](
            {
                "output_ids": output_ids.unsqueeze(0),
                "logprobs": logprobs,
                "generation_lengths": torch.tensor(
                    [len(generated_token_ids)],
                    dtype=torch.long,
                    device=input_ids.device,
                ),
                "unpadded_sequence_lengths": torch.tensor(
                    [output_length],
                    dtype=torch.long,
                    device=input_ids.device,
                ),
                "truncated": torch.tensor(
                    [truncated],
                    dtype=torch.bool,
                    device=input_ids.device,
                ),
            }
        )

    def generate(
        self,
        data: BatchedDataDict["GenerationDatumSpec"],
        greedy: bool = False,
    ) -> BatchedDataDict["GenerationOutputSpec"]:
        """Reject the unused blocking interface.

        Both synchronous and asynchronous GRPO trainers use ``generate_async``
        for the managed HTTP frontend.
        """
        raise NotImplementedError(
            "Dynamo generation uses generate_async() for both synchronous and "
            "asynchronous GRPO trainers"
        )

    async def generate_async(
        self,
        data: BatchedDataDict["GenerationDatumSpec"],
        greedy: bool = False,
    ) -> AsyncGenerator[tuple[int, BatchedDataDict["GenerationOutputSpec"]], None]:
        """Generate one token-ID prompt asynchronously through the managed frontend."""
        assert isinstance(data, BatchedDataDict), (
            f"data must be a BatchedDataDict, got type: {type(data)}"
        )
        assert "input_ids" in data and "input_lengths" in data, (
            "input_ids and input_lengths are required in data for Dynamo generation"
        )
        if len(data["input_ids"]) == 0:
            return

        verify_right_padding(data, pad_value=self.cfg["_pad_token_id"])

        input_ids_batch = data["input_ids"]
        input_lengths_batch = data["input_lengths"]
        batch_size = input_ids_batch.shape[0]
        assert batch_size == 1, (
            "generate_async is restricted to handle only single samples, "
            f"but received batch_size={batch_size}. Please handle batching "
            "outside this method."
        )
        sample_idx = 0
        input_length = int(input_lengths_batch[sample_idx].item())
        batch_stop_strings = data.get("stop_strings", [[] for _ in range(batch_size)])
        per_sample_stop_strings = None
        if batch_stop_strings and sample_idx < len(batch_stop_strings):
            per_sample_stop_strings = batch_stop_strings[sample_idx]
        final_stop_strings = self._merge_stop_strings(
            [per_sample_stop_strings] if per_sample_stop_strings else None
        )

        allowed_new_tokens = self._allowed_new_tokens(input_length)
        input_ids = input_ids_batch[sample_idx]
        (
            generated_token_ids,
            generated_logprobs,
            truncated,
        ) = await self._post_completion_request(
            prompt_token_ids=self._prompt_token_ids(data, sample_idx),
            greedy=greedy,
            stop_strings=final_stop_strings,
            max_new_tokens=allowed_new_tokens,
        )

        yield (
            sample_idx,
            self._single_sample_output(
                input_ids=input_ids,
                input_length=input_length,
                generated_token_ids=generated_token_ids,
                generated_logprobs=generated_logprobs,
                truncated=truncated,
            ),
        )

    def init_collective(
        self,
        ip: str,
        port: int,
        world_size: int,
        *,
        train_world_size: int,
    ) -> list[ray.ObjectRef]:
        """Initialize native vLLM NCCL transfer on every managed worker."""
        channel = self._refit_channel
        if channel is None:
            raise RuntimeError("Dynamo refit channel is unavailable")
        return channel.init_collective(
            ip,
            port,
            world_size,
            train_world_size=train_world_size,
        )

    def prepare_refit_info(self, state_dict_info: dict[str, Any]) -> None:
        """Serialize checkpoint-format tensor metadata for native vLLM refit."""
        channel = self._refit_channel
        if channel is None:
            raise RuntimeError("Dynamo refit channel is unavailable")
        channel.prepare(state_dict_info)

    def update_weights_via_ipc_zmq(self) -> list[ray.ObjectRef]:
        raise NotImplementedError(
            "DynamoGeneration only supports NCCL weight transfer."
        )

    def update_weights_from_collective(self) -> list[ray.ObjectRef]:
        """Receive packed checkpoint-format weights on every Dynamo worker."""
        channel = self._refit_channel
        if channel is None:
            raise RuntimeError("Dynamo refit channel is unavailable")
        return channel.update_weights()

    def invalidate_kv_cache(self) -> bool:
        """Flush every fixed Dynamo worker's prefix/KV cache."""
        channel = self._refit_channel
        if channel is None:
            raise RuntimeError("Dynamo refit channel is unavailable")
        return channel.flush_cache()
