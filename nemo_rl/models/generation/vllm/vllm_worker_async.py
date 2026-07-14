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
import logging
import resource
import threading
import time
import uuid
import warnings
from typing import Any, AsyncGenerator, Optional, cast

import ray
import torch
import uvicorn
from fastapi import FastAPI
from tensordict import TensorDict

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import (
    DEFAULT_GENERATION_PORT_RANGE_HIGH,
    DEFAULT_GENERATION_PORT_RANGE_LOW,
    _get_free_port_local,
    _get_node_ip_local,
)
from nemo_rl.distributed.worker_group_utils import get_nsight_config_if_pattern_matches
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationOutputSpec,
    verify_right_padding,
)
from nemo_rl.models.generation.vllm.utils import (
    attach_routed_experts_to_chat_response_choices,
    format_prompt_for_vllm_generation,
    model_dump_chat_response_with_routed_experts,
    pad_and_align_routed_expert_indices,
)
from nemo_rl.models.generation.vllm.vllm_worker import BaseVllmGenerationWorker

LOGGER = logging.getLogger(__name__)


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

    if len(template_token_ids) <= len(template_prefix_token_ids):
        LOGGER.warning(
            "Chat-template reconstruction produced a non-growing trajectory "
            "(prefix=%d tokens, full=%d tokens). Falling back to the fully "
            "retokenized prompt; this may introduce off-policy token drift.",
            len(template_prefix_token_ids),
            len(template_token_ids),
        )
        return template_token_ids

    # Some chat templates use model-specific assistant/tool delimiters instead
    # of EOS. When no EOS is present and the render through the last assistant
    # message is an exact prefix of the full render, its length is the
    # unambiguous seam: preserve the model-produced prefix verbatim and append
    # only the newly rendered suffix. If any EOS exists, retain the established
    # EOS path below because the template may need to add EOS/newline boundary
    # tokens that are absent from the model-produced prefix.
    if (
        eos_token_id not in template_prefix_token_ids
        and template_token_ids[: len(template_prefix_token_ids)]
        == template_prefix_token_ids
    ):
        return (
            model_prefix_token_ids
            + template_token_ids[len(template_prefix_token_ids) :]
        )

    # We take everything starting with the EOS token ID.
    template_cut_start = -1
    for pos in reversed(range(len(template_prefix_token_ids))):
        if template_token_ids[pos] == eos_token_id:
            template_cut_start = pos
            break

    if template_cut_start < 0:
        LOGGER.warning(
            "No EOS token ID was found in the chat-templated prefix. Falling "
            "back to the fully retokenized prompt; this may introduce "
            "off-policy token drift."
        )
        return template_token_ids

    return (
        model_prefix_token_ids[:model_cut_end] + template_token_ids[template_cut_start:]
    )


class VllmAsyncGenerationWorkerImpl(BaseVllmGenerationWorker):
    def __init__(
        self,
        config,
        bundle_indices=None,
        fraction_of_gpus: float = 1.0,
        seed=None,
        extra_env_vars: Optional[list[str]] = None,
        defer_model_load: bool = False,
    ):
        """Initialize an async vLLM worker.

        When defer_model_load=True, only stores config and reserves a port for
        the HTTP server (if expose_http_server is enabled). Call load_model()
        later to perform the heavy model loading. This enables overlapping vLLM
        model loading with NeMo Gym init.

        Args:
            config: Configuration dictionary for the policy
            bundle_indices: List of local bundle indices within a node for parallelism.
            fraction_of_gpus: Fraction of GPUs to use for this worker
            seed: Random seed for initialization
            extra_env_vars: Additional environment variable names to forward into
                          the vLLM worker subprocess.
            defer_model_load: If True, skip model loading and only reserve port
        """
        # Deferred-loading state. Always initialized so every instance has a
        # consistent set of attributes regardless of init path.
        self._reserved_socket = None
        self._reserved_port = None
        self._reserved_node_ip = None
        self._deferred_bundle_indices = None
        self._deferred_seed = None

        # Defaults for HTTP server state; overwritten by _create_engine()
        # when the worker is a model owner and the model is actually loaded.
        self.server_thread = None
        self.base_url = None
        self.http_server = None
        self._rollout_writer_cfg: Any = None
        self._rollout_writer_secret: bytes | None = None
        self._rollout_cursor: Any = None
        self._rollout_dp_client: Any = None
        self._rollout_requests: dict[int, Any] = {}
        self._rollout_prompt_tokens: dict[int, list[int]] = {}
        self._rollout_response_tokens: dict[
            int, tuple[list[int], list[int], list[float]]
        ] = {}
        self._rollout_metrics_enabled = False
        self._rollout_metrics_lock = threading.Lock()
        self._rollout_transport_metrics: dict[str, Any] = {}
        self._rollout_metrics_cpu_start_s = time.process_time()
        # The step's policy weight version, shipped by the driver before each
        # rollout phase. Gateway-identified requests stamp this on staged rows
        # (the signed-context path carries its own version instead).
        self._rollout_weight_version = 0

        super().__init__(
            config,
            bundle_indices,
            fraction_of_gpus,
            seed,
            extra_env_vars,
            defer_model_load,
        )

        if not self.is_model_owner or not defer_model_load:
            return

        self._deferred_bundle_indices = bundle_indices
        self._deferred_seed = seed

        if self.cfg["vllm_cfg"].get("expose_http_server"):
            self._reserve_port()

        self.llm = None
        self.vllm_device_ids = None

    def _return_routed_experts_enabled(self) -> bool:
        engine_args = getattr(self, "llm_async_engine_args", None)
        if bool(getattr(engine_args, "enable_return_routed_experts", False)):
            return True
        return bool(
            self.cfg.get("vllm_kwargs", {}).get("enable_return_routed_experts", False)
        )

    def configure_rollout_writer(self, dp_cfg, cursor, secret: bytes) -> None:
        """Attach the direct rollout writer after TQ has bootstrapped."""
        from nemo_rl.data_plane import build_data_plane_client

        self._rollout_writer_cfg = dp_cfg.rollout_writer
        self._rollout_writer_secret = secret
        self._rollout_cursor = cursor
        self._rollout_dp_client = build_data_plane_client(dp_cfg, bootstrap=False)

    def set_rollout_weight_version(self, weight_version: int) -> None:
        """Declare the policy weight version staged rows must carry this step."""
        self._rollout_weight_version = int(weight_version)

    def configure_rollout_metrics(self, enabled: bool) -> None:
        """Enable lightweight in-memory transport metrics for A/B benchmarks."""
        self._rollout_metrics_enabled = enabled
        self.clear_rollout_transport_metrics()

    def clear_rollout_transport_metrics(self) -> None:
        """Reset per-step transport counters without touching model metrics."""
        with self._rollout_metrics_lock:
            self._rollout_transport_metrics = {
                "http_request_count": 0,
                "http_response_count": 0,
                "encoded_request_bytes": 0,
                "encoded_response_bytes": 0,
                "encoded_response_bytes_without_token_ids": 0,
                "encoded_response_bytes_without_logprobs": 0,
                "encoded_response_base_bytes": 0,
                "http_request_ms": [],
                "chat_request_count": 0,
                "chat_response_count": 0,
                "chat_encoded_request_bytes": 0,
                "chat_encoded_response_bytes": 0,
                "chat_encoded_response_bytes_without_token_ids": 0,
                "chat_encoded_response_bytes_without_logprobs": 0,
                "chat_encoded_response_base_bytes": 0,
                "chat_request_ms": [],
                "tokenize_request_count": 0,
                "tokenize_response_count": 0,
                "tokenize_encoded_request_bytes": 0,
                "tokenize_encoded_response_bytes": 0,
                "tokenize_encoded_response_bytes_without_token_ids": 0,
                "tokenize_encoded_response_bytes_without_logprobs": 0,
                "tokenize_encoded_response_base_bytes": 0,
                "tokenize_request_ms": [],
                "last_response_completed_monotonic_s": 0.0,
                "cursor_reserve_ms": [],
                "staging_put_ms": [],
                "staging_put_bytes": 0,
                "cursor_commit_ms": [],
                "sample_ids": [],
            }
            self._rollout_metrics_cpu_start_s = time.process_time()
        if self._rollout_dp_client is not None and hasattr(
            self._rollout_dp_client, "clear_events"
        ):
            self._rollout_dp_client.clear_events()

    def get_rollout_transport_metrics(self) -> dict[str, Any]:
        """Return one worker's current transport, CPU, and RSS measurements."""
        if not self._rollout_metrics_enabled:
            return {}
        with self._rollout_metrics_lock:
            metrics = copy.deepcopy(self._rollout_transport_metrics)
            cpu_start_s = self._rollout_metrics_cpu_start_s
        metrics["worker_id"] = str(ray.get_runtime_context().get_actor_id())
        metrics["process_cpu_s"] = time.process_time() - cpu_start_s
        metrics["peak_rss_bytes"] = (
            resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        )
        if self._rollout_dp_client is not None and hasattr(
            self._rollout_dp_client, "events_snapshot"
        ):
            metrics["data_plane_events"] = self._rollout_dp_client.events_snapshot()
            metrics["data_plane_snapshot"] = self._rollout_dp_client.snapshot()
        return metrics

    def _record_rollout_transport(self, name: str, value: Any) -> None:
        if not getattr(self, "_rollout_metrics_enabled", False):
            return
        with self._rollout_metrics_lock:
            current = self._rollout_transport_metrics[name]
            if isinstance(current, list):
                current.append(value)
            elif name == "last_response_completed_monotonic_s":
                self._rollout_transport_metrics[name] = max(current, value)
            else:
                self._rollout_transport_metrics[name] = current + value

    def _record_http_request(self, endpoint: str, encoded_bytes: int) -> None:
        """Record an exact inbound JSON body for one instrumented endpoint."""
        self._record_rollout_transport("http_request_count", 1)
        self._record_rollout_transport("encoded_request_bytes", encoded_bytes)
        self._record_rollout_transport(f"{endpoint}_request_count", 1)
        self._record_rollout_transport(
            f"{endpoint}_encoded_request_bytes", encoded_bytes
        )

    def _record_http_response(
        self,
        endpoint: str,
        payload: dict[str, Any],
        response,
        request_started: float,
    ) -> None:
        """Record one exact outbound JSON body and endpoint service time."""
        if not self._rollout_metrics_enabled:
            return
        from nemo_rl.experience.rollout_writer import encoded_response_payload_sizes

        payload_sizes = encoded_response_payload_sizes(payload)
        payload_sizes["encoded_response_bytes"] = len(response.body)
        for metric_name, value in payload_sizes.items():
            self._record_rollout_transport(metric_name, value)
            self._record_rollout_transport(f"{endpoint}_{metric_name}", value)
        request_ms = (time.perf_counter() - request_started) * 1000
        self._record_rollout_transport("http_response_count", 1)
        self._record_rollout_transport("http_request_ms", request_ms)
        self._record_rollout_transport(f"{endpoint}_response_count", 1)
        self._record_rollout_transport(f"{endpoint}_request_ms", request_ms)
        self._record_rollout_transport(
            "last_response_completed_monotonic_s", time.monotonic()
        )

    async def _prepare_rollout_request(
        self, request, prompt_token_ids: list[int], tokenizer=None
    ) -> None:
        """Validate identity, reserve a turn, and verify the echoed prefix."""
        if self._rollout_writer_cfg is None or not self._rollout_writer_cfg.enabled:
            return
        # vLLM 0.20 shares this preprocessor with the /tokenize endpoint.
        # Tokenization requests are not generation turns and intentionally do
        # not carry a rollout context.
        if not hasattr(request, "nemo_rl_rollout_context"):
            return
        self._rollout_prompt_tokens[id(request)] = list(prompt_token_ids)
        from nemo_rl.experience.rollout_writer import (
            RolloutContext,
            RolloutContextError,
            RolloutIdentity,
            RolloutRequestState,
            derive_request_nonce,
            hash_token_ids,
            validate_rollout_context,
        )

        if getattr(request, "stream", False):
            raise RolloutContextError("rollout writer does not support streaming")
        bare_rollout_id = getattr(request, "nemo_rl_rollout_id", None)
        call_id = getattr(request, "nemo_rl_call_id", None)
        payload = getattr(request, "nemo_rl_rollout_context", None)
        if payload is not None:
            # Native path: validate the signed context.
            try:
                context = RolloutContext.from_dict(payload)
                if self._rollout_writer_secret is None:
                    raise RuntimeError("rollout writer secret was not configured")
                validate_rollout_context(context, secret=self._rollout_writer_secret)
            except RolloutContextError:
                LOGGER.exception("Invalid rollout context; serving request uncollected")
                return
            # Migration alias pair: the signed context's sample_id carries the
            # canonical rollout_id. A request presenting both identities is
            # accepted only when they agree.
            if bare_rollout_id is not None and bare_rollout_id != context.sample_id:
                LOGGER.error(
                    "nemo_rl_rollout_id %s does not match signed context identity "
                    "%s; serving request uncollected",
                    bare_rollout_id,
                    context.sample_id,
                )
                return
            identity = RolloutIdentity(
                rollout_id=context.sample_id,
                group_id=context.group_id,
                weight_version=context.weight_version,
                call_id=call_id,
            )
            # Idempotency key: (rollout, prompt) — the native agent loop is
            # sequential and append-only, so one prompt is one turn.
            request_nonce = derive_request_nonce(identity.rollout_id, prompt_token_ids)
        elif (
            self._rollout_writer_cfg.accept_gateway_identity
            and bare_rollout_id is not None
            and call_id is not None
        ):
            # Gateway path: the identity pair was stamped by a trusted Gym
            # gate after dialect conversion (the vLLM endpoint must be
            # network-isolated or the gateway-to-worker hop authenticated).
            # group_id never travels through model requests; the finalizer
            # joins it back driver-side. Idempotency key: the gate-minted
            # call_id — an internal retry reuses it, while a separately
            # admitted call with an identical prompt gets a fresh one.
            # Prompt hashes are never identity.
            identity = RolloutIdentity(
                rollout_id=str(bare_rollout_id),
                group_id="",
                weight_version=self._rollout_weight_version,
                call_id=str(call_id),
            )
            request_nonce = f"call:{call_id}"
        else:
            LOGGER.error(
                "Rollout request has no signed context and no accepted gateway "
                "identity; serving uncollected"
            )
            return
        reserve_started = time.perf_counter()
        try:
            reservation = await self._rollout_cursor.reserve_turn.remote(
                identity.rollout_id, request_nonce
            )
        except ray.exceptions.RayError:
            LOGGER.exception(
                "Could not reserve a rollout turn for %s; serving uncollected",
                identity.rollout_id,
            )
            return
        finally:
            self._record_rollout_transport(
                "cursor_reserve_ms", (time.perf_counter() - reserve_started) * 1000
            )

        echoed_prefix = getattr(request, "required_prefix_token_ids", None)
        if echoed_prefix is None:
            # No token echo (black-box harnesses speak text): verify that the
            # rendered prompt still begins with the committed prefix by hash.
            prefix_matches = (
                len(prompt_token_ids) >= reservation.prev_len
                and hash_token_ids(prompt_token_ids[: reservation.prev_len])
                == reservation.prev_hash
            )
            echoed_prefix = prompt_token_ids[: reservation.prev_len]
        else:
            prefix_matches = (
                len(echoed_prefix) == reservation.prev_len
                and hash_token_ids(echoed_prefix) == reservation.prev_hash
                and prompt_token_ids[: reservation.prev_len] == echoed_prefix
            )
        if not prefix_matches:
            divergent = 0
            shared = min(len(echoed_prefix), reservation.prev_len)
            while (
                divergent < shared
                and prompt_token_ids[divergent] == echoed_prefix[divergent]
            ):
                divergent += 1
            window_start = max(0, divergent - 8)
            window_end = divergent + 8
            expected_window = echoed_prefix[window_start:window_end]
            actual_window = prompt_token_ids[window_start:window_end]
            expected_text = (
                repr(tokenizer.decode(expected_window)) if tokenizer is not None else ""
            )
            actual_text = (
                repr(tokenizer.decode(actual_window)) if tokenizer is not None else ""
            )
            reason = (
                f"prefix_mismatch:first_divergent_token={divergent}:"
                f"expected_tokens={expected_window}:actual_tokens={actual_window}:"
                f"expected_text={expected_text}:actual_text={actual_text}"
            )
            await self._rollout_cursor.fail_turn.remote(
                identity.rollout_id, reservation.lease, reason=reason
            )
            LOGGER.error("%s; serving request uncollected", reason)
            return

        self._rollout_requests[id(request)] = RolloutRequestState(
            identity=identity,
            reservation=reservation,
            prompt_token_ids=list(prompt_token_ids),
        )

    async def _stage_rollout_response(self, request, response) -> None:
        """Synchronously stage a generated turn and commit its cursor."""
        request_state = self._rollout_requests.get(id(request))
        if request_state is None:
            return
        from nemo_rl.experience.rollout_writer import (
            build_staging_delta,
            extract_generation_token_info,
            hash_token_ids,
        )

        response_dict = response.model_dump()
        choices = response_dict.get("choices", [])
        if len(choices) != 1:
            raise ValueError(
                f"rollout writer requires exactly one choice, got {len(choices)}"
            )
        choice = choices[0]
        generated_ids, generated_logprobs = extract_generation_token_info(choice)
        prompt_ids = request_state.prompt_token_ids
        reservation = request_state.reservation
        identity = request_state.identity
        full_ids = prompt_ids + generated_ids
        try:
            token_ids_delta, token_mask_delta, logprobs_delta = build_staging_delta(
                prompt_token_ids=prompt_ids,
                generated_token_ids=generated_ids,
                generated_logprobs=generated_logprobs,
                prev_len=reservation.prev_len,
            )
        except ValueError:
            reason = "invalid_staging_delta"
            await self._rollout_cursor.fail_turn.remote(
                identity.rollout_id,
                reservation.lease,
                reason=reason,
            )
            raise

        # Staging keys: the gate-minted call_id when the gateway supplied one
        # (unique per admitted call, retry-idempotent), else the linear turn
        # index of the native sequential path.
        if identity.call_id is not None:
            staging_key = f"{identity.rollout_id}/{identity.call_id}"
        else:
            staging_key = f"{identity.rollout_id}/t{reservation.turn}"
        fields = TensorDict(
            {
                "token_ids_delta": torch.tensor([token_ids_delta], dtype=torch.int64),
                "token_mask_delta": torch.tensor(
                    [token_mask_delta], dtype=torch.float32
                ),
                "generation_logprobs_delta": torch.tensor(
                    [logprobs_delta], dtype=torch.float32
                ),
            },
            batch_size=[1],
        )
        tags = [
            {
                "sample_id": identity.rollout_id,
                "group_id": identity.group_id,
                "call_id": identity.call_id,
                "turn": reservation.turn,
                "weight_version": identity.weight_version,
                "request_nonce": reservation.request_nonce,
                "prev_len": reservation.prev_len,
                "new_len": len(full_ids),
                "prev_hash": reservation.prev_hash,
                "new_hash": hash_token_ids(full_ids),
                "lease_state": "written",
                "full_snapshot": False,
            }
        ]
        try:
            staging_started = time.perf_counter()
            self._rollout_dp_client.put_samples(
                sample_ids=[staging_key],
                partition_id=self._rollout_writer_cfg.staging_partition,
                fields=fields,
                tags=tags,
            )
        except Exception as error:
            await self._rollout_cursor.fail_turn.remote(
                identity.rollout_id,
                reservation.lease,
                reason=f"staging_write_failed:{type(error).__name__}",
            )
            raise
        finally:
            self._record_rollout_transport(
                "staging_put_ms", (time.perf_counter() - staging_started) * 1000
            )
        staging_bytes = sum(
            value.numel() * value.element_size() for value in fields.values()
        )
        self._record_rollout_transport("staging_put_bytes", staging_bytes)

        commit_started = time.perf_counter()
        try:
            await self._rollout_cursor.commit_turn.remote(
                identity.rollout_id,
                reservation.lease,
                staging_key=staging_key,
                new_len=len(full_ids),
                new_hash=hash_token_ids(full_ids),
                group_id=identity.group_id,
                weight_version=identity.weight_version,
            )
        except Exception:
            try:
                await self._rollout_cursor.fail_turn.remote(
                    identity.rollout_id,
                    reservation.lease,
                    reason="cursor_commit_failed",
                )
            except Exception:
                LOGGER.exception(
                    "Failed to mark rollout %s after cursor commit failure",
                    identity.rollout_id,
                )
            raise
        finally:
            self._record_rollout_transport(
                "cursor_commit_ms", (time.perf_counter() - commit_started) * 1000
            )
        self._record_rollout_transport("sample_ids", identity.rollout_id)
        # Gateway-identified (black-box) calls never echo tokens over HTTP:
        # the staged delta is the only token store, and the response is served
        # with neither token ids nor logprobs. The native path re-attaches
        # them for exact next-turn prefix echo.
        if identity.call_id is None:
            self._rollout_response_tokens[id(request)] = (
                list(prompt_ids),
                generated_ids,
                generated_logprobs,
            )

    def _reserve_port(self) -> None:
        """Bind and listen on a TCP socket to reserve a free port from the OS.

        The socket is held open in LISTENING state and later passed directly to
        uvicorn via the ``sockets=`` parameter in ``server.serve()``. The socket
        is never closed and re-opened, so there is zero gap where another process
        could steal the port.
        """
        import socket

        from nemo_rl.distributed.virtual_cluster import _get_node_ip_local

        self._reserved_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._reserved_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._reserved_socket.bind(("", 0))
        self._reserved_socket.listen(128)
        self._reserved_socket.setblocking(False)
        self._reserved_port = self._reserved_socket.getsockname()[1]
        self._reserved_node_ip = _get_node_ip_local()
        print(
            f"Reserved port {self._reserved_port} on {self._reserved_node_ip} "
            f"for vLLM HTTP server"
        )

    def load_model(self) -> None:
        """Load the vLLM model and create the engine.

        Called after a deferred init to perform the heavy model loading.
        """
        if not self.is_model_owner:
            return
        self._load_model(self._deferred_bundle_indices, self._deferred_seed)

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

        self.llm_async_engine_args = AsyncEngineArgs(**llm_kwargs)
        self.stat_loggers = (
            [PrometheusStatLogger]
            if self.cfg["vllm_cfg"].get("enable_vllm_metrics_logger", False)
            else []
        )
        self.llm = AsyncLLM.from_engine_args(
            self.llm_async_engine_args, stat_loggers=self.stat_loggers
        )

        if self.cfg["vllm_cfg"].get("expose_http_server"):
            # Must run after AsyncLLM.from_engine_args and before
            # _setup_vllm_server spawns the uvicorn thread.
            self._install_engine_input_socket_lock()
            self.server_thread, self.base_url, self.http_server = (
                self._setup_vllm_server()
            )

        # vLLM Metrics Logger
        # Metrics logger only enabled for per-actor, model-owner only
        self._vllm_metrics_lock = threading.Lock()
        if self.cfg["vllm_cfg"].get("enable_vllm_metrics_logger", False):
            self._start_vllm_metrics_logger()

    def _install_engine_input_socket_lock(self) -> None:
        """Serialise sends on AsyncMPClient.input_socket across OS threads
        to prevent race conditions that block the vLLM engine (e.g. during
        in flight weight updates in async grpo).
        """
        shadow_sock = self.llm.engine_core.input_socket._shadow_sock

        lock = threading.Lock()
        original_send_multipart = shadow_sock.send_multipart

        def locked_send_multipart(*args: Any, **kwargs: Any) -> Any:
            with lock:
                return original_send_multipart(*args, **kwargs)

        # Replace the bound method on this socket instance only; other zmq
        # sockets in the process are unaffected.
        shadow_sock.send_multipart = locked_send_multipart  # type: ignore[assignment]

    def _start_vllm_metrics_logger(self) -> None:
        """Start a background thread that periodically collects vLLM logger metrics.

        Controlled by vllm_metrics_logger_interval (default: 0.5) in vllm_cfg.
        Runs only on the model-owner actor.
        """
        from vllm.v1.metrics.reader import Gauge, Counter, get_metrics_snapshot

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

        def _logger_loop():
            # Delay a little to let engine settle
            time.sleep(min(2.0, interval_s))
            while True:
                try:
                    for m in get_metrics_snapshot():
                        with self._vllm_metrics_lock:
                            if isinstance(m, Gauge):
                                # Log the vllm inflight batch sizes
                                if m.name == "vllm:num_requests_running":
                                    self.inflight_batch_sizes.append(int(m.value))
                                # Log the vllm pending number of requests in the queue
                                elif m.name == "vllm:num_requests_waiting":
                                    self.num_pending_samples.append(int(m.value))
                                # Log the vllm kv cache usage
                                elif m.name == "vllm:kv_cache_usage_perc":
                                    self.kv_cache_usage_perc.append(float(m.value))
                            elif isinstance(m, Counter):
                                if m.name == "vllm:generation_tokens":
                                    self.generation_tokens.append(int(m.value))
                except Exception:
                    print(
                        "⚠️[vLLM Metric Logger] Exception in vLLM metrics logger",
                        flush=True,
                    )
                    pass
                time.sleep(interval_s)

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

        with self._vllm_metrics_lock:
            metric = {
                "inflight_batch_sizes": copy.deepcopy(self.inflight_batch_sizes),
                "num_pending_samples": copy.deepcopy(self.num_pending_samples),
                "kv_cache_usage_perc": copy.deepcopy(self.kv_cache_usage_perc),
                "generation_tokens": copy.deepcopy(self.generation_tokens),
            }
        return metric

    def clear_vllm_logger_metrics(self) -> None:
        if not self.cfg["vllm_cfg"].get("enable_vllm_metrics_logger", False):
            return

        with self._vllm_metrics_lock:
            self.inflight_batch_sizes = []
            self.num_pending_samples = []
            self.kv_cache_usage_perc = []
            self.generation_tokens = []

    async def post_init_async(self):
        self.vllm_device_ids = await self.report_device_id_async()
        if self._mtp_load_from_disk:
            await self.llm.collective_rpc(
                "load_mtp_weights_from_disk", args=(self.model_name,)
            )

    async def get_reserved_url(self) -> Optional[str]:
        """Return the URL from the reserved socket, available before model loading."""
        if self._reserved_socket is not None:
            return f"http://{self._reserved_node_ip}:{self._reserved_port}/v1"
        return None

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
        from vllm.entrypoints.serve.render.serving import (
            OpenAIServingRender,
        )
        from vllm.entrypoints.serve.tokenize.serving import (
            OpenAIServingTokenization,
        )
        from vllm.exceptions import VLLMValidationError
        from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager
        from vllm.tool_parsers.abstract_tool_parser import ToolParserManager
        from vllm.v1.engine.async_llm import logger as vllm_async_llm_logger

        maybe_tool_parser_plugin = self.cfg["vllm_cfg"].get("tool_parser_plugin")
        if maybe_tool_parser_plugin:
            ToolParserManager.import_tool_parser(maybe_tool_parser_plugin)

        maybe_reasoning_parser_plugin = self.cfg["vllm_cfg"].get(
            "reasoning_parser_plugin"
        )
        if maybe_reasoning_parser_plugin:
            ReasoningParserManager.import_reasoning_parser(
                maybe_reasoning_parser_plugin
            )

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
            @staticmethod
            def _set_max_tokens(request, max_tokens: int) -> None:
                """Set the request's max output tokens.

                Mutates the request in place. Handles both max_completion_tokens (newer OpenAI API)
                and max_tokens (deprecated but still supported by vLLM).
                """
                if request.max_completion_tokens is not None:
                    request.max_completion_tokens = max_tokens
                elif request.max_tokens is not None:
                    request.max_tokens = max_tokens

            def _clamp_max_tokens(
                self, request, request_max_tokens: int, prompt_token_ids: list[int]
            ) -> None:
                """Clamp the request's max output tokens so that input + output <= max_model_len."""
                remaining = self.model_config.max_model_len - len(prompt_token_ids)
                if remaining <= 0:
                    raise ValueError(
                        f"Prompt length ({len(prompt_token_ids)}) fills or exceeds "
                        f"max_model_len ({self.model_config.max_model_len}). "
                        f"No room for output tokens."
                    )
                max_tokens = min(request_max_tokens, remaining)
                self._set_max_tokens(request, max_tokens)

            # vLLM 0.20 moved chat preprocessing from
            # OpenAIServing._preprocess_chat to OpenAIServingRender.preprocess_chat,
            # so this override now applies via the render subclass.
            async def preprocess_chat(
                self,
                request,
                messages,
                default_template,
                default_template_content_format,
                default_template_kwargs,
                tool_dicts=None,
                tool_parser=None,
                reasoning_parser=None,
                *,
                skip_mm_cache: bool = False,
            ):
                for message in messages:
                    if message.get("tool_calls"):
                        message["tool_calls"] = list(message["tool_calls"])

                messages_for_replace_prefix_tokens = deepcopy(messages)

                # Temporarily set to 1 so vLLM's pre-tokenization length check passes;
                # the actual value will be set through _clamp_max_tokens later.
                actual_request_max_tokens = None
                if isinstance(request, NeMoRLChatCompletionRequest):
                    actual_request_max_tokens = (
                        request.max_completion_tokens
                        if request.max_completion_tokens is not None
                        else request.max_tokens
                    )
                    # If max_completion_tokens or max_tokens is not set, we don't need to do _clamp_max_tokens.
                    # So we don't need to set the request's max output tokens to 1 here.
                    if actual_request_max_tokens is not None:
                        self._set_max_tokens(request, 1)

                try:
                    res = await super().preprocess_chat(
                        request=request,
                        messages=messages,
                        default_template=default_template,
                        default_template_content_format=default_template_content_format,
                        default_template_kwargs=default_template_kwargs,
                        tool_dicts=tool_dicts,
                        tool_parser=tool_parser,
                        reasoning_parser=reasoning_parser,
                        skip_mm_cache=skip_mm_cache,
                    )
                except (ValueError, VLLMValidationError) as e:
                    if "maximum context length" in str(e):
                        import logging

                        logging.getLogger(__name__).warning(
                            "Prompt exceeds max_model_len: %s", e
                        )
                    raise

                if (
                    not hasattr(request, "required_prefix_token_ids")
                    or request.required_prefix_token_ids is None
                ):
                    # Clamp the request's max output tokens so that input + output <= max_model_len.
                    if actual_request_max_tokens is not None:
                        self._clamp_max_tokens(
                            request,
                            actual_request_max_tokens,
                            res[1][0]["prompt_token_ids"],
                        )
                    await worker_self._prepare_rollout_request(
                        request,
                        res[1][0]["prompt_token_ids"],
                        tokenizer=self.renderer.tokenizer,
                    )
                    return res

                last_assistant_message_idx = None
                for i in reversed(range(len(messages_for_replace_prefix_tokens))):
                    if messages_for_replace_prefix_tokens[i]["role"] == "assistant":
                        last_assistant_message_idx = i
                        break

                if last_assistant_message_idx is None:
                    messages_to_last_assistant_message = (
                        messages_for_replace_prefix_tokens
                    )
                else:
                    messages_to_last_assistant_message = (
                        messages_for_replace_prefix_tokens[
                            : last_assistant_message_idx + 1
                        ]
                    )

                modified_request = request.model_copy(
                    update={"add_generation_prompt": False}
                )

                corresponding_res = await super().preprocess_chat(
                    request=modified_request,
                    messages=messages_to_last_assistant_message,
                    default_template=default_template,
                    default_template_content_format=default_template_content_format,
                    default_template_kwargs=default_template_kwargs,
                    tool_dicts=tool_dicts,
                    tool_parser=tool_parser,
                    reasoning_parser=reasoning_parser,
                    skip_mm_cache=skip_mm_cache,
                )
                actual_corresponding_token_ids = corresponding_res[1][0][
                    "prompt_token_ids"
                ]

                engine_prompt = res[1][0]

                final_prompt_token_ids = _replace_prefix_tokens(
                    tokenizer=self.renderer.tokenizer,
                    model_prefix_token_ids=request.required_prefix_token_ids,
                    template_prefix_token_ids=actual_corresponding_token_ids,
                    template_token_ids=engine_prompt["prompt_token_ids"],
                )

                engine_prompt["prompt_token_ids"] = final_prompt_token_ids

                # Clamp after prefix replacement since the prompt length may have changed.
                if actual_request_max_tokens is not None:
                    self._clamp_max_tokens(
                        request,
                        actual_request_max_tokens,
                        final_prompt_token_ids,
                    )

                await worker_self._prepare_rollout_request(
                    request,
                    final_prompt_token_ids,
                    tokenizer=self.renderer.tokenizer,
                )

                return res

        ########################################
        # /v1/chat/completions endpoint
        ########################################

        # This MRO is necessary i.e. NeMoRLOpenAIChatRequestMixin > ChatCompletionRequest
        class NeMoRLChatCompletionRequest(
            NeMoRLOpenAIChatRequestMixin, ChatCompletionRequest
        ):
            required_prefix_token_ids: Optional[List[int]] = None
            nemo_rl_rollout_context: Optional[dict[str, Any]] = None
            # Canonical rollout identity (migration alias pair: when both this
            # and the signed context arrive, they must agree).
            nemo_rl_rollout_id: Optional[str] = None
            # Per-call identity minted by a trusted Gym gateway; never derived
            # from the prompt.
            nemo_rl_call_id: Optional[str] = None

        # vLLM 0.20 routes both /v1/chat/completions and /tokenize through
        # OpenAIServingRender.preprocess_chat, so the prefix-token override
        # belongs on the render subclass.
        worker_self = self

        class NeMoRLOpenAIServingChatMixin:
            async def chat_completion_full_generator(
                self,
                request,
                result_generator,
                *args,
                **kwargs,
            ):
                final_res = None

                async def capture_result_generator():
                    nonlocal final_res
                    async for res in result_generator:
                        final_res = res
                        yield res

                response = await super().chat_completion_full_generator(
                    request,
                    capture_result_generator(),
                    *args,
                    **kwargs,
                )
                if isinstance(response, ChatCompletionResponse):
                    await worker_self._stage_rollout_response(request, response)
                if (
                    not worker_self._return_routed_experts_enabled()
                    or not isinstance(response, ChatCompletionResponse)
                    or final_res is None
                ):
                    return response

                return attach_routed_experts_to_chat_response_choices(
                    response,
                    final_res,
                    device=torch.device("cpu"),
                    logger=LOGGER,
                )

        class NeMoRLOpenAIServingChat(NeMoRLOpenAIServingChatMixin, OpenAIServingChat):
            pass

        class NeMoRLOpenAIServingRender(NeMoRLOpenAIServingMixin, OpenAIServingRender):
            pass

        serving_chat_default_kwargs = dict(
            response_role="assistant",
            request_logger=None,
            chat_template=None,
            chat_template_content_format="auto",
            enable_auto_tools=True,
        )
        serving_chat_kwargs = serving_chat_default_kwargs | self.cfg["vllm_cfg"].get(
            "http_server_serving_chat_kwargs", dict()
        )
        openai_serving_render = NeMoRLOpenAIServingRender(
            model_config=engine_client.model_config,
            renderer=engine_client.renderer,
            model_registry=openai_serving_models.registry,
            request_logger=serving_chat_kwargs["request_logger"],
            chat_template=serving_chat_kwargs["chat_template"],
            chat_template_content_format=serving_chat_kwargs[
                "chat_template_content_format"
            ],
            enable_auto_tools=serving_chat_kwargs["enable_auto_tools"],
        )
        serving_chat_kwargs.update(
            dict(
                engine_client=engine_client,
                models=openai_serving_models,
                openai_serving_render=openai_serving_render,
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
            request_started = time.perf_counter()
            if worker_self._rollout_metrics_enabled:
                worker_self._record_http_request("chat", len(await raw_request.body()))
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

            try:
                try:
                    generator = await openai_serving_chat.create_chat_completion(
                        request, raw_request
                    )
                except VLLMValidationError as e:
                    # vLLM 0.20 raises VLLMValidationError for prompts exceeding
                    # max_model_len during tokenization, instead of returning an
                    # ErrorResponse. Convert to HTTP 400 so the Gym proxy can
                    # detect context-length overflow and handle it gracefully.
                    response_content = {
                        "error": {
                            "message": str(e),
                            "type": "invalid_request_error",
                            "code": 400,
                        }
                    }
                    response = JSONResponse(
                        content=response_content,
                        status_code=400,
                    )
                    worker_self._record_http_response(
                        "chat", response_content, response, request_started
                    )
                    return response

                if isinstance(generator, ErrorResponse):
                    response_content = generator.model_dump()
                    response = JSONResponse(
                        content=response_content, status_code=generator.error.code
                    )
                    worker_self._record_http_response(
                        "chat", response_content, response, request_started
                    )
                    return response

                if isinstance(generator, ChatCompletionResponse):
                    response_content = model_dump_chat_response_with_routed_experts(
                        generator
                    )
                    if (
                        worker_self._rollout_writer_cfg is not None
                        and worker_self._rollout_writer_cfg.enabled
                    ):
                        token_info = worker_self._rollout_response_tokens.get(
                            id(request)
                        )
                        if token_info is None:
                            from nemo_rl.experience.rollout_writer import (
                                extract_generation_token_info,
                            )

                            prompt_ids = worker_self._rollout_prompt_tokens.get(
                                id(request)
                            )
                            if prompt_ids is None:
                                raise RuntimeError(
                                    "rollout response has no preprocessed prompt tokens"
                                )
                            generated_ids, generated_logprobs = (
                                extract_generation_token_info(
                                    response_content["choices"][0]
                                )
                            )
                        else:
                            prompt_ids, generated_ids, generated_logprobs = token_info
                        response_message = response_content["choices"][0]["message"]
                        response_message["prompt_token_ids"] = prompt_ids
                        response_message["generation_token_ids"] = generated_ids
                        response_message["generation_log_probs"] = generated_logprobs

                    if (
                        worker_self._rollout_writer_cfg is not None
                        and worker_self._rollout_writer_cfg.enabled
                        and worker_self._rollout_writer_cfg.mode == "direct"
                    ):
                        from nemo_rl.experience.rollout_writer import (
                            strip_direct_response_logprobs,
                        )

                        response_content = strip_direct_response_logprobs(
                            response_content
                        )
                    response = JSONResponse(content=response_content)
                    worker_self._record_http_response(
                        "chat", response_content, response, request_started
                    )
                    return response

                return StreamingResponse(
                    content=generator, media_type="text/event-stream"
                )
            finally:
                # Non-streaming Gym requests complete inside this endpoint.
                worker_self._rollout_requests.pop(id(request), None)
                worker_self._rollout_prompt_tokens.pop(id(request), None)
                worker_self._rollout_response_tokens.pop(id(request), None)

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

        # Tokenize path delegates to OpenAIServingRender.preprocess_chat in
        # vLLM 0.20, where the prefix-token override lives.
        class NeMoRLOpenAIServingTokenization(OpenAIServingTokenization):
            pass

        serving_tokenization_kwargs = dict(
            request_logger=serving_chat_kwargs["request_logger"],
            chat_template=serving_chat_kwargs["chat_template"],
            chat_template_content_format=serving_chat_kwargs[
                "chat_template_content_format"
            ],
            engine_client=serving_chat_kwargs["engine_client"],
            models=serving_chat_kwargs["models"],
            openai_serving_render=openai_serving_render,
        )
        openai_serving_tokenization = NeMoRLOpenAIServingTokenization(
            **serving_tokenization_kwargs
        )

        @app.post("/tokenize")
        async def tokenize(request: NeMoRLTokenizeRequest, raw_request: Request):
            request_started = time.perf_counter()
            if worker_self._rollout_metrics_enabled:
                worker_self._record_http_request(
                    "tokenize", len(await raw_request.body())
                )
            generator = await openai_serving_tokenization.create_tokenize(
                request, raw_request
            )

            if isinstance(generator, ErrorResponse):
                response_content = generator.model_dump()
                response = JSONResponse(
                    content=response_content, status_code=generator.error.code
                )
                worker_self._record_http_response(
                    "tokenize", response_content, response, request_started
                )
                return response
            elif isinstance(generator, TokenizeResponse):
                response_content = generator.model_dump()
                response = JSONResponse(content=response_content)
                worker_self._record_http_response(
                    "tokenize", response_content, response, request_started
                )
                return response

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

        if self._reserved_socket is not None:
            # Use the socket reserved during __init__ (deferred model load path).
            # Pass it directly to uvicorn via sockets= — zero gap, the socket is
            # never closed and re-opened, so no other process can steal the port.
            node_ip = self._reserved_node_ip
            free_port = self._reserved_port
            reserved_sock = self._reserved_socket
            self._reserved_socket = None  # Transfer ownership to uvicorn
        else:
            node_ip = _get_node_ip_local()
            port_range_low = self.cfg.get(
                "port_range_low", DEFAULT_GENERATION_PORT_RANGE_LOW
            )
            port_range_high = self.cfg.get(
                "port_range_high", DEFAULT_GENERATION_PORT_RANGE_HIGH
            )
            free_port = _get_free_port_local(port_range_low, port_range_high)
            reserved_sock = None

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

        if reserved_sock is not None:
            # Hand the pre-bound listening socket directly to uvicorn's asyncio
            # server via server.serve(sockets=). No close-and-rebind needed.
            import asyncio

            def _run_with_socket():
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(server.serve(sockets=[reserved_sock]))

            thread = threading.Thread(target=_run_with_socket, daemon=True)
        else:
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
    ) -> AsyncGenerator[tuple[int, BatchedDataDict[GenerationOutputSpec]], None]:
        """Generate a batch of data using vLLM's AsyncLLMEngine, yielding results as they are ready.

        Args:
            data: BatchedDataDict with input_ids and input_lengths
            greedy: Whether to use greedy decoding instead of sampling

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
            allowed_new_tokens = max(0, min(self.cfg["max_new_tokens"], remaining_ctx))

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
            )

            request_id = str(uuid.uuid4())

            # Generate using vLLM async engine
            vllm_request_generator = self.llm.generate(
                prompt=prompt,
                sampling_params=sampling_params_for_request,
                request_id=request_id,
            )

            # Get the final result from the generator
            final_request_output = None
            async for req_output in vllm_request_generator:
                final_request_output = req_output

            if final_request_output is None:
                raise RuntimeError(f"No output received for request {request_id}")

            # Process the output
            generation_details = final_request_output.outputs[0]
            generated_token_ids = list(generation_details.token_ids)
            num_generated_tokens = len(generated_token_ids)
            return_routed_experts = self._return_routed_experts_enabled()

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

            result_dict = {
                "output_ids": output_ids_single_item_batched,
                "logprobs": logprobs_single_item,
                "generation_lengths": generation_lengths_tensor,
                "unpadded_sequence_lengths": unpadded_sequence_lengths_tensor,
                "truncated": truncated_tensor,
            }
            routed_experts, r3_stats = pad_and_align_routed_expert_indices(
                final_request_output,
                generation_details,
                valid_length=unpadded_total_length,
                padded_length=final_output_tensor_len,
                device=original_input_ids_single_row.device,
                require_complete_routed_experts=return_routed_experts,
                return_stats=True,
            )
            if return_routed_experts and routed_experts is None:
                raise RuntimeError(
                    "vLLM was asked to return routed experts but the generation output "
                    "did not include routed_experts."
                )
            if return_routed_experts:
                if r3_stats["missing_routes"] > 0:
                    LOGGER.warning(
                        "R3 router replay fallback: vLLM returned incomplete "
                        "routed_experts for sample_idx=%d, missing_token_routes=%d, "
                        "actual_routes=%d, expected_routes=%d. Megatron will use its "
                        "own router for those missing token routes.",
                        sample_idx,
                        r3_stats["missing_routes"],
                        r3_stats["actual_routes"],
                        r3_stats["expected_routes"],
                    )
                result_dict["r3_routed_experts_missing_routes"] = torch.tensor(
                    [r3_stats["missing_routes"]],
                    dtype=torch.long,
                    device=original_input_ids_single_row.device,
                )
                result_dict["r3_routed_experts_expected_routes"] = torch.tensor(
                    [r3_stats["expected_routes"]],
                    dtype=torch.long,
                    device=original_input_ids_single_row.device,
                )
                result_dict["r3_routed_experts_actual_routes"] = torch.tensor(
                    [r3_stats["actual_routes"]],
                    dtype=torch.long,
                    device=original_input_ids_single_row.device,
                )
            if routed_experts is not None:
                result_dict["routed_experts"] = routed_experts.unsqueeze(0)

            result_batch = BatchedDataDict[GenerationOutputSpec](result_dict)

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

            if self.server_thread is not None:
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


@ray.remote(
    runtime_env={**get_nsight_config_if_pattern_matches("vllm_async_generation_worker")}
)  # pragma: no cover
class VllmAsyncGenerationWorker(VllmAsyncGenerationWorkerImpl):
    pass
