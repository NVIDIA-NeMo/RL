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
import inspect
import logging
import os
import threading
import time
import uuid
from typing import Any, AsyncGenerator, Optional, cast

import ray
import torch
import uvicorn
from fastapi import FastAPI

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
try:
    from nemo_rl.distributed.virtual_cluster import (
        DEFAULT_PORT_RANGE_HIGH,
        DEFAULT_PORT_RANGE_LOW,
        _bind_socket_in_range,
        _get_free_port_local,
        _get_node_ip_local,
    )
except ImportError:
    import socket

    from nemo_rl.distributed.virtual_cluster import (
        _get_free_port_local as _get_free_port_local_unranged,
        _get_node_ip_local,
    )

    DEFAULT_PORT_RANGE_LOW = 11001
    DEFAULT_PORT_RANGE_HIGH = 15000

    def _bind_socket_in_range(sock, port_range_low, port_range_high):
        for port in range(port_range_low, port_range_high + 1):
            try:
                sock.bind(("", port))
                return port
            except OSError:
                continue
        raise RuntimeError(
            f"No free port found in range {port_range_low}-{port_range_high}"
        )

    def _get_free_port_local(port_range_low=None, port_range_high=None):
        if port_range_low is None or port_range_high is None:
            return _get_free_port_local_unranged()

        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            return _bind_socket_in_range(sock, port_range_low, port_range_high)
from nemo_rl.distributed.worker_group_utils import get_nsight_config_if_pattern_matches
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationOutputSpec,
    verify_right_padding,
)
from nemo_rl.models.generation.vllm.utils import format_prompt_for_vllm_generation
from nemo_rl.models.generation.vllm.vllm_worker import BaseVllmGenerationWorker

logger = logging.getLogger(__name__)


def _normalize_weight_update_results(worker_results: Any) -> tuple[bool, list[Any]]:
    if isinstance(worker_results, bool):
        return worker_results, [
            None if worker_results else "collective_rpc returned False"
        ]

    if worker_results is None:
        return True, [None]

    try:
        result_items = list(worker_results)
    except TypeError:
        success = bool(worker_results)
        return success, [
            None if success else f"Unexpected collective_rpc result: {worker_results!r}"
        ]

    if not result_items:
        return False, ["collective_rpc returned no worker results"]

    success_flags = []
    exceptions_or_none = []
    for result in result_items:
        if isinstance(result, bool):
            success_flags.append(result)
            exceptions_or_none.append(None if result else "worker returned False")
        elif result is None:
            success_flags.append(True)
            exceptions_or_none.append(None)
        elif isinstance(result, (list, tuple)):
            if not result:
                success_flags.append(False)
                exceptions_or_none.append("worker returned an empty result")
            else:
                success_flags.append(bool(result[0]))
                exceptions_or_none.append(result[1] if len(result) > 1 else None)
        else:
            success = bool(result)
            success_flags.append(success)
            exceptions_or_none.append(
                None if success else f"Unexpected worker result: {result!r}"
            )

    return all(success_flags), exceptions_or_none


def _qwen35_truncate_prompt_tokens() -> Optional[int]:
    value = os.environ.get("NEMO_RL_QWEN35_TRUNCATE_PROMPT_TOKENS", "65535")
    if value.lower() in {"", "0", "false", "none", "no"}:
        return None
    return int(value)


def _diagnose_noncontiguous_message_tokens() -> bool:
    return os.environ.get(
        "NEMO_RL_DIAGNOSE_NONCONTIGUOUS_MESSAGE_TOKENS", "0"
    ).lower() not in {"0", "false", "no", "off"}


def _noncontiguous_message_diagnostic_window() -> int:
    value = os.environ.get("NEMO_RL_NONCONTIGUOUS_MESSAGE_DIAGNOSTIC_WINDOW", "48")
    try:
        return max(0, int(value))
    except ValueError:
        return 48


def _decode_token_ids(tokenizer, token_ids: list[int]) -> str:
    try:
        return tokenizer.decode(
            token_ids,
            skip_special_tokens=False,
            clean_up_tokenization_spaces=False,
        )
    except TypeError:
        return tokenizer.decode(token_ids, skip_special_tokens=False)
    except Exception as exc:
        return f"<decode failed: {exc!r}>"


def _first_token_mismatch(left: list[int], right: list[int]) -> Optional[int]:
    compare_len = min(len(left), len(right))
    for idx in range(compare_len):
        if left[idx] != right[idx]:
            return idx
    if len(left) != len(right):
        return compare_len
    return None


def _last_token_index(token_ids: list[int], token_id: int) -> Optional[int]:
    for idx in reversed(range(len(token_ids))):
        if token_ids[idx] == token_id:
            return idx
    return None


def _token_window_at_end(token_ids: list[int], window: int) -> tuple[int, int, list[int]]:
    end = len(token_ids)
    start = max(0, end - window)
    return start, end, token_ids[start:end]


def _token_window(token_ids: list[int], center_idx: int, window: int) -> tuple[int, int, list[int]]:
    start = max(0, center_idx - window)
    end = min(len(token_ids), center_idx + window + 1)
    return start, end, token_ids[start:end]


def _log_qwen35_prefix_diagnostic(
    tokenizer,
    reason: str,
    model_prefix_token_ids: list[int],
    template_prefix_token_ids: list[int],
    template_token_ids: list[int],
) -> None:
    if not _diagnose_noncontiguous_message_tokens():
        return

    window = _noncontiguous_message_diagnostic_window()
    prefix_template_mismatch = _first_token_mismatch(
        template_prefix_token_ids, template_token_ids
    )
    model_template_mismatch = _first_token_mismatch(
        model_prefix_token_ids, template_prefix_token_ids
    )
    model_full_template_mismatch = _first_token_mismatch(
        model_prefix_token_ids, template_token_ids
    )
    eos_token_id = tokenizer.eos_token_id
    model_last_eos = (
        _last_token_index(model_prefix_token_ids, eos_token_id)
        if eos_token_id is not None
        else None
    )
    prefix_last_eos = (
        _last_token_index(template_prefix_token_ids, eos_token_id)
        if eos_token_id is not None
        else None
    )
    template_last_eos = (
        _last_token_index(template_token_ids, eos_token_id)
        if eos_token_id is not None
        else None
    )

    center_idx = prefix_template_mismatch
    if center_idx is None:
        center_idx = min(len(template_prefix_token_ids), len(template_token_ids))

    model_start, model_end, model_window = _token_window(
        model_prefix_token_ids, min(center_idx, len(model_prefix_token_ids)), window
    )
    prefix_start, prefix_end, prefix_window = _token_window(
        template_prefix_token_ids,
        min(center_idx, len(template_prefix_token_ids)),
        window,
    )
    template_start, template_end, template_window = _token_window(
        template_token_ids, min(center_idx, len(template_token_ids)), window
    )
    model_tail_start, model_tail_end, model_tail_window = _token_window_at_end(
        model_prefix_token_ids, window
    )
    prefix_tail_start, prefix_tail_end, prefix_tail_window = _token_window_at_end(
        template_prefix_token_ids, window
    )
    template_tail_start, template_tail_end, template_tail_window = _token_window_at_end(
        template_token_ids, window
    )

    logger.warning(
        "Qwen 3.5 vLLM monotonic-prefix diagnostic (%s): "
        "prefix_template_mismatch=%s, model_template_mismatch=%s, "
        "model_full_template_mismatch=%s, "
        "template_minus_prefix=%d, template_minus_model_prefix=%d, "
        "eos=%s, last_eos=(model:%s,prefix:%s,template:%s)\n"
        "  model_prefix_window=[%d:%d] ids=%s text=%r\n"
        "  template_prefix_window=[%d:%d] ids=%s text=%r\n"
        "  template_window=[%d:%d] ids=%s text=%r\n"
        "  model_prefix_tail=[%d:%d] ids=%s text=%r\n"
        "  template_prefix_tail=[%d:%d] ids=%s text=%r\n"
        "  template_tail=[%d:%d] ids=%s text=%r",
        reason,
        prefix_template_mismatch,
        model_template_mismatch,
        model_full_template_mismatch,
        len(template_token_ids) - len(template_prefix_token_ids),
        len(template_token_ids) - len(model_prefix_token_ids),
        eos_token_id,
        model_last_eos,
        prefix_last_eos,
        template_last_eos,
        model_start,
        model_end,
        model_window,
        _decode_token_ids(tokenizer, model_window),
        prefix_start,
        prefix_end,
        prefix_window,
        _decode_token_ids(tokenizer, prefix_window),
        template_start,
        template_end,
        template_window,
        _decode_token_ids(tokenizer, template_window),
        model_tail_start,
        model_tail_end,
        model_tail_window,
        _decode_token_ids(tokenizer, model_tail_window),
        prefix_tail_start,
        prefix_tail_end,
        prefix_tail_window,
        _decode_token_ids(tokenizer, prefix_tail_window),
        template_tail_start,
        template_tail_end,
        template_tail_window,
        _decode_token_ids(tokenizer, template_tail_window),
    )


def _message_role_summary(messages: list) -> str:
    roles = []
    for message in messages:
        if isinstance(message, dict):
            role = str(message.get("role", "<missing>"))
            fields = []
            for key in ("content", "tool_calls", "tool_call_id"):
                if key in message and message[key]:
                    fields.append(key)
            if "prompt_token_ids" in message:
                fields.append(f"prompt={len(message['prompt_token_ids'])}")
            if "generation_token_ids" in message:
                fields.append(f"gen={len(message['generation_token_ids'])}")
            if fields:
                role = f"{role}({','.join(fields)})"
        else:
            role = type(message).__name__
        roles.append(role)

    if len(roles) <= 16:
        return "[" + ", ".join(roles) + "]"
    return "[" + ", ".join(roles[:8]) + ", ..., " + ", ".join(roles[-8:]) + "]"


def _log_qwen35_pre_replace_context(
    request,
    messages: list,
    messages_to_last_assistant_message: list,
    last_assistant_message_idx: Optional[int],
    model_prefix_token_ids: list[int],
    template_prefix_token_ids: list[int],
    template_token_ids: list[int],
) -> None:
    if not _diagnose_noncontiguous_message_tokens():
        return

    if len(template_token_ids) > len(template_prefix_token_ids) and len(
        template_token_ids
    ) > len(model_prefix_token_ids):
        return

    logger.warning(
        "Qwen 3.5 vLLM prefix-repair context: "
        "messages=%d, prefix_messages=%d, last_assistant_idx=%s, "
        "truncate_prompt_tokens=%s, add_generation_prompt=%s, max_tokens=%s, "
        "lengths=(required_prefix:%d, template_prefix:%d, template:%d), "
        "template_minus_prefix=%d, template_minus_required_prefix=%d, "
        "roles=%s, prefix_roles=%s",
        len(messages),
        len(messages_to_last_assistant_message),
        last_assistant_message_idx,
        getattr(request, "truncate_prompt_tokens", None),
        getattr(request, "add_generation_prompt", None),
        getattr(request, "max_tokens", None),
        len(model_prefix_token_ids),
        len(template_prefix_token_ids),
        len(template_token_ids),
        len(template_token_ids) - len(template_prefix_token_ids),
        len(template_token_ids) - len(model_prefix_token_ids),
        _message_role_summary(messages),
        _message_role_summary(messages_to_last_assistant_message),
    )


def _find_nth_token_from_end(token_ids: list[int], token_id: int, n: int) -> int:
    if n <= 0:
        return -1

    seen = 0
    for idx in reversed(range(len(token_ids))):
        if token_ids[idx] == token_id:
            seen += 1
            if seen == n:
                return idx
    return -1


def _qwen35_replace_prefix_from_suffix_messages(
    *,
    eos_token_id: int,
    model_prefix_token_ids: list[int],
    template_token_ids: list[int],
    suffix_message_count: Optional[int],
) -> Optional[list[int]]:
    if suffix_message_count is None:
        return None

    # For a prompt that ends with N non-assistant suffix messages plus the
    # assistant generation marker, the boundary before those suffix messages is
    # the (N + 1)-th EOS from the end of the rendered full template.
    template_cut_start = _find_nth_token_from_end(
        template_token_ids, eos_token_id, suffix_message_count + 1
    )
    if template_cut_start < 0:
        return None

    model_cut_end = len(model_prefix_token_ids)
    if model_prefix_token_ids and model_prefix_token_ids[-1] == eos_token_id:
        model_cut_end -= 1

    if _diagnose_noncontiguous_message_tokens():
        logger.warning(
            "Qwen 3.5 vLLM suffix-message prefix repair: "
            "suffix_messages=%d, template_cut_start=%d, "
            "lengths=(required_prefix:%d, template:%d, repaired:%d)",
            suffix_message_count,
            template_cut_start,
            len(model_prefix_token_ids),
            len(template_token_ids),
            model_cut_end + len(template_token_ids[template_cut_start:]),
        )

    return (
        model_prefix_token_ids[:model_cut_end]
        + template_token_ids[template_cut_start:]
    )


def _replace_prefix_tokens(
    tokenizer,
    model_prefix_token_ids: list[int],
    template_prefix_token_ids: list[int],
    template_token_ids: list[int],
    suffix_message_count: Optional[int] = None,
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

    if _first_token_mismatch(template_prefix_token_ids, template_token_ids) is not None:
        repaired_token_ids = _qwen35_replace_prefix_from_suffix_messages(
            eos_token_id=eos_token_id,
            model_prefix_token_ids=model_prefix_token_ids,
            template_token_ids=template_token_ids,
            suffix_message_count=suffix_message_count,
        )
        if repaired_token_ids is not None:
            return repaired_token_ids

    # Assert here to prepare for the logic below
    if len(template_token_ids) <= len(
        template_prefix_token_ids
    ):
        repaired_token_ids = _qwen35_replace_prefix_from_suffix_messages(
            eos_token_id=eos_token_id,
            model_prefix_token_ids=model_prefix_token_ids,
            template_token_ids=template_token_ids,
            suffix_message_count=suffix_message_count,
        )
        if repaired_token_ids is not None:
            return repaired_token_ids

        _log_qwen35_prefix_diagnostic(
            tokenizer=tokenizer,
            reason="template_has_no_suffix_after_prefix",
            model_prefix_token_ids=model_prefix_token_ids,
            template_prefix_token_ids=template_prefix_token_ids,
            template_token_ids=template_token_ids,
        )
        error_message = f"""Found possibly non-monotonically increasing trajectory!
Template prefix token IDs (everything before the final assistant message): {template_prefix_token_ids}

Template token IDs (everything that was sent to the model endpoint): {template_token_ids}

Template prefix repr (detokenized): {repr(tokenizer.decode(template_prefix_token_ids))}

Template repr (detokenized): {repr(tokenizer.decode(template_token_ids))}
"""
        with open(f"non_monotonic_trajectory_{str(uuid.uuid4())}.txt", "w") as f:
            f.write(error_message)

        raise ValueError(error_message)

    # We take everything starting with the EOS token ID.
    template_cut_start = -1
    for pos in reversed(range(len(template_prefix_token_ids))):
        if template_token_ids[pos] == eos_token_id:
            template_cut_start = pos
            break

    # This should never be the case, but
    if (
        template_cut_start < 0
    ):
        _log_qwen35_prefix_diagnostic(
            tokenizer=tokenizer,
            reason="template_prefix_has_no_eos",
            model_prefix_token_ids=model_prefix_token_ids,
            template_prefix_token_ids=template_prefix_token_ids,
            template_token_ids=template_token_ids,
        )
        error_message = f"""No EOS token ID found in the chat-templated messages!
Template prefix token IDs (everything before the final assistant message): {template_prefix_token_ids}

Template token IDs (everything that was sent to the model endpoint): {template_token_ids}

Template prefix repr (detokenized): {repr(tokenizer.decode(template_prefix_token_ids))}

Template repr (detokenized): {repr(tokenizer.decode(template_token_ids))}"""
        with open(f"no_eos_token_id_found_{str(uuid.uuid4())}.txt", "w") as f:
            f.write(error_message)
        raise ValueError(error_message)

    return (
        model_prefix_token_ids[:model_cut_end] + template_token_ids[template_cut_start:]
    )


@ray.remote(
    runtime_env={**get_nsight_config_if_pattern_matches("vllm_async_generation_worker")}
)  # pragma: no cover
class VllmAsyncGenerationWorker(BaseVllmGenerationWorker):
    def __init__(
        self,
        config,
        bundle_indices=None,
        fraction_of_gpus: float = 1.0,
        seed=None,
        defer_model_load: bool = False,
    ):
        """Initialize an async vLLM worker.

        When defer_model_load=True, only stores config and reserves a port
        for the HTTP server (if expose_http_server is enabled). Call
        load_model() later to perform the heavy model loading.

        Args:
            config: Configuration dictionary for the policy
            bundle_indices: List of local bundle indices within a node for parallelism.
            fraction_of_gpus: Fraction of GPUs to use for this worker
            seed: Random seed for initialization
            defer_model_load: If True, skip model loading and only reserve port
        """
        # Deferred-loading state. Always initialized so every instance
        # has a consistent set of attributes regardless of init path.
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

        base_init_params = inspect.signature(
            BaseVllmGenerationWorker.__init__
        ).parameters
        if "defer_model_load" not in base_init_params:
            if defer_model_load:
                logger.warning(
                    "BaseVllmGenerationWorker does not support defer_model_load; "
                    "falling back to eager vLLM model load."
                )
            super().__init__(config, bundle_indices, fraction_of_gpus, seed)
            return

        super().__init__(
            config,
            bundle_indices,
            fraction_of_gpus,
            seed,
            defer_model_load=defer_model_load,
        )

        if not self.is_model_owner or not defer_model_load:
            return

        self._deferred_bundle_indices = bundle_indices
        self._deferred_seed = seed

        if self.cfg["vllm_cfg"].get("expose_http_server"):
            self._reserve_port()

        self.llm = None
        self.vllm_device_ids = None

    def _reserve_port(self):
        """Bind and listen on a TCP socket to reserve a free port from the OS.

        The socket is held open in LISTENING state and passed directly to
        uvicorn via the ``sockets=`` parameter in ``server.serve()``.
        The socket is never closed and re-opened, so there is zero gap
        where another process could steal the port.
        """
        import socket

        port_range_low = self.cfg.get("port_range_low", DEFAULT_PORT_RANGE_LOW)
        port_range_high = self.cfg.get("port_range_high", DEFAULT_PORT_RANGE_HIGH)

        self._reserved_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._reserved_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._reserved_port = _bind_socket_in_range(self._reserved_socket, port_range_low, port_range_high)
        self._reserved_socket.listen(128)
        self._reserved_socket.setblocking(False)
        self._reserved_node_ip = _get_node_ip_local()
        print(
            f"Reserved port {self._reserved_port} on {self._reserved_node_ip} "
            f"for vLLM HTTP server"
        )

    def _seed_vllm_cache(self):
        """Seed local vLLM compile cache from a warm directory if available.

        Reads NRL_VLLM_CACHE_SEED_DIR and rsyncs it into VLLM_CACHE_ROOT.
        Both are env vars set by the launch script — keeping them in the
        same system avoids the inconsistency of mixing env vars and config.

        Retries up to 3 times on transient failures. Timeout is controlled
        by NRL_VLLM_CACHE_SEED_TIMEOUT (default 300s).
        """
        import os
        import shutil
        import subprocess

        seed_dir = os.environ.get("NRL_VLLM_CACHE_SEED_DIR", "")
        local_dst = os.environ.get("VLLM_CACHE_ROOT", "")
        if not seed_dir or not local_dst or not os.path.isdir(seed_dir):
            return
        if not os.listdir(seed_dir):
            return
        if not shutil.which("rsync"):
            print("[CACHE SEED] rsync not found, skipping cache seed", flush=True)
            return

        timeout = int(os.environ.get("NRL_VLLM_CACHE_SEED_TIMEOUT", "300"))
        os.makedirs(local_dst, exist_ok=True)

        max_attempts = 3
        for attempt in range(1, max_attempts + 1):
            t0 = time.perf_counter()
            try:
                result = subprocess.run(
                    ["rsync", "-a", "--ignore-existing", "--prune-empty-dirs", f"{seed_dir}/", f"{local_dst}/"],
                    capture_output=True,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired:
                elapsed = time.perf_counter() - t0
                print(
                    f"[CACHE SEED] rsync timed out after {elapsed:.0f}s "
                    f"(attempt {attempt}/{max_attempts})",
                    flush=True,
                )
                continue

            elapsed = time.perf_counter() - t0
            if result.returncode == 0:
                print(f"[CACHE SEED] vLLM compile cache seeded in {elapsed:.1f}s", flush=True)
                return

            stderr = result.stderr.decode(errors="replace")[:200]
            print(
                f"[CACHE SEED] rsync failed (attempt {attempt}/{max_attempts}, "
                f"{elapsed:.1f}s): {stderr}",
                flush=True,
            )

        print("[CACHE SEED] all attempts failed, proceeding with cold compile", flush=True)

    def load_model(self):
        """Load the vLLM model and create the engine.

        Called after deferred init to perform the heavy model loading.
        """
        if not self.is_model_owner:
            return

        self._seed_vllm_cache()
        self._load_model(self._deferred_bundle_indices, self._deferred_seed)

    def _create_engine(self, llm_kwargs: dict[str, Any]) -> None:
        from vllm.config import CompilationConfig
        from vllm.engine.arg_utils import AsyncEngineArgs
        from vllm.v1.engine.async_llm import AsyncLLM
        from vllm.v1.metrics.loggers import PrometheusStatLogger

        # (TODO: zhiyul) Remove this workaround after upgrading vLLM where the compilation_config passing issue is resolved.
        if llm_kwargs.get("compilation_config", None):
            llm_kwargs["compilation_config"] = CompilationConfig(
                **llm_kwargs["compilation_config"]
            )

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
            self.server_thread, self.base_url, self.http_server = (
                self._setup_vllm_server()
            )

        # vLLM Metrics Logger
        # Metrics logger only enabled for per-actor, model-owner only
        self._vllm_metrics_lock = threading.Lock()
        if self.cfg["vllm_cfg"].get("enable_vllm_metrics_logger", False):
            self._start_vllm_metrics_logger()

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

                    if "content" in message:
                        content = message["content"]
                        if isinstance(content, (list, str)):
                            continue
                        # Convert ValidatorIterator to list to get actual content
                        try:
                            message["content"] = list(content)
                        except Exception:
                            message["content"] = []

                truncate_prompt_tokens = _qwen35_truncate_prompt_tokens()
                if (
                    truncate_prompt_tokens is not None
                    and hasattr(request, "truncate_prompt_tokens")
                    and getattr(request, "truncate_prompt_tokens") is None
                ):
                    request = request.model_copy(
                        update={"truncate_prompt_tokens": truncate_prompt_tokens}
                    )

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

                # Copy after vLLM preprocessing so prefix and full-prompt paths
                # use the same normalized reasoning/tool-call representation.
                exclude_fields = {
                    "prompt_token_ids",
                    "generation_token_ids",
                    "generation_log_probs",
                }
                messages_for_replace_prefix_tokens = []
                for msg in messages:
                    if isinstance(msg, dict):
                        new_msg = {}
                        for k, v in msg.items():
                            if k not in exclude_fields:
                                new_msg[k] = deepcopy(v)
                        messages_for_replace_prefix_tokens.append(new_msg)
                    else:
                        messages_for_replace_prefix_tokens.append(deepcopy(msg))

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
                    suffix_message_count = None
                else:
                    # Include the last assistant message itself.
                    messages_to_last_assistant_message = (
                        messages_for_replace_prefix_tokens[
                            : last_assistant_message_idx + 1
                        ]
                    )
                    suffix_message_count = (
                        len(messages_for_replace_prefix_tokens)
                        - last_assistant_message_idx
                        - 1
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

                _log_qwen35_pre_replace_context(
                    request=request,
                    messages=messages,
                    messages_to_last_assistant_message=messages_to_last_assistant_message,
                    last_assistant_message_idx=last_assistant_message_idx,
                    model_prefix_token_ids=request.required_prefix_token_ids,
                    template_prefix_token_ids=actual_corresponding_token_ids,
                    template_token_ids=engine_prompt["prompt_token_ids"],
                )

                final_prompt_token_ids = _replace_prefix_tokens(
                    tokenizer=self.renderer.tokenizer,
                    model_prefix_token_ids=request.required_prefix_token_ids,
                    template_prefix_token_ids=actual_corresponding_token_ids,
                    template_token_ids=engine_prompt["prompt_token_ids"],
                    suffix_message_count=suffix_message_count,
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

        # Load custom reasoning parser plugin if specified
        reasoning_parser_plugin = serving_chat_kwargs.pop("reasoning_parser_plugin", None)
        if reasoning_parser_plugin:
            from vllm.reasoning.abs_reasoning_parsers import ReasoningParserManager
            ReasoningParserManager.import_reasoning_parser(reasoning_parser_plugin)

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

        _getLogger("vllm.entrypoints.openai.engine.protocol").addFilter(CleanLoggingFilter())

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

        if self._reserved_socket:
            # Use the socket reserved during __init__ (deferred model load path).
            # Pass it directly to uvicorn via sockets= — zero gap, the socket
            # is never closed and re-opened.
            node_ip = self._reserved_node_ip
            free_port = self._reserved_port
            reserved_sock = self._reserved_socket
            self._reserved_socket = None  # Transfer ownership to uvicorn
        else:
            node_ip = _get_node_ip_local()
            port_range_low = self.cfg.get("port_range_low", DEFAULT_PORT_RANGE_LOW)
            port_range_high = self.cfg.get("port_range_high", DEFAULT_PORT_RANGE_HIGH)
            free_port = _get_free_port_local(port_range_low, port_range_high)
            reserved_sock = None

        base_url = f"http://{node_ip}:{free_port}/v1"
        print(f"Starting server on {base_url}")

        # When sockets= is used, uvicorn ignores host/port in Config.
        # We still set them for logging/metadata purposes.
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
            # Use server.serve(sockets=) to hand the pre-bound listening socket
            # directly to uvicorn's asyncio server. No close-and-rebind needed.
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
            
            all_success, exceptions_or_none = _normalize_weight_update_results(
                worker_results
            )

            if not all_success:
                print(
                    f"Error: Worker failed to update weights. Result: {exceptions_or_none}"
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

            all_success, exceptions_or_none = _normalize_weight_update_results(
                worker_results
            )

            if not all_success:
                print(
                    f"Error: Worker failed to update weights. Result: {exceptions_or_none}"
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
