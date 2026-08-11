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
"""Shared helpers for the OpenAI-compatible HTTP generation servers.

These backend-agnostic utilities operate on token-ID lists, parsed chat messages,
and caller-supplied tokenizers or render functions. They make no engine calls.
They are shared by the vLLM async worker (``vllm_worker_async.py``) and the
TRT-LLM HTTP server (``trtllm_http_server.py``), which both put a message-based
``/v1/chat/completions`` layer in front of a token engine for the agentic
NeMo-Gym path. SGLang does not use these — it is driven token-in/token-out via
``generate(input_ids)`` and never re-templates messages, so it has no
retokenization drift to correct.
"""

from collections.abc import Callable, Mapping
from copy import deepcopy
from typing import Any

_COMPLETE_TOKEN_BOUNDARY_MARKER = "NEMO_RL_PREFIX_BOUNDARY_7F3A9C1E"


class CompleteTokenInjectionError(ValueError):
    """Signal that complete-token injection must use native preprocessing."""


def find_latest_tokenized_assistant(
    messages: list[Any],
) -> tuple[int, Mapping[str, Any]] | None:
    """Return the latest fully tokenized assistant and its assistant ordinal."""
    assistant_ordinal = -1
    latest_tokenized_assistant = None
    for message in messages:
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        assistant_ordinal += 1
        if (
            message.get("prompt_token_ids") is not None
            and message.get("generation_token_ids") is not None
        ):
            latest_tokenized_assistant = (assistant_ordinal, message)
    return latest_tokenized_assistant


def _assistant_index_from_ordinal(
    conversation: list[Any], assistant_ordinal: int
) -> int:
    current_ordinal = -1
    for index, message in enumerate(conversation):
        if not isinstance(message, Mapping) or message.get("role") != "assistant":
            continue
        current_ordinal += 1
        if current_ordinal == assistant_ordinal:
            return index
    raise CompleteTokenInjectionError(
        "The tokenized assistant message was not present after chat parsing."
    )


def _coerce_token_id_list(value: Any, field_name: str) -> list[int]:
    if not isinstance(value, list):
        raise CompleteTokenInjectionError(f"{field_name} must be a list.")
    try:
        return [int(token_id) for token_id in value]
    except (TypeError, ValueError) as e:
        raise CompleteTokenInjectionError(
            f"{field_name} must contain only integer token IDs."
        ) from e


def build_complete_prompt_token_ids(
    *,
    tokenizer: Any,
    conversation: list[Any],
    assistant_ordinal: int,
    model_prefix_token_ids: list[int],
    render_prompt_text: Callable[[list[Any]], str],
) -> list[int]:
    """Join exact model-prefix tokens with a verified contextual suffix.

    ``assistant_ordinal`` must identify the assistant turn covered by
    ``model_prefix_token_ids``. A mismatch can produce a valid but incorrect
    prompt, so callers must derive both values from the same message.

    The complete conversation is rendered with the tokenized assistant replaced
    by a marker. This preserves context-sensitive template behavior after that
    assistant while making its closing EOS the exact splice boundary.
    """
    marked_conversation = deepcopy(conversation)
    if any(
        _COMPLETE_TOKEN_BOUNDARY_MARKER in str(message)
        for message in marked_conversation
    ):
        raise CompleteTokenInjectionError(
            "The complete-token boundary marker collided with request data."
        )

    assistant_index = _assistant_index_from_ordinal(
        marked_conversation, assistant_ordinal
    )

    marked_assistant = marked_conversation[assistant_index]
    if not isinstance(marked_assistant, dict):
        raise CompleteTokenInjectionError(
            "The tokenized assistant message was not a message object."
        )
    marked_assistant["content"] = _COMPLETE_TOKEN_BOUNDARY_MARKER
    marked_assistant.pop("reasoning_content", None)
    marked_assistant.pop("reasoning", None)
    marked_assistant.pop("tool_calls", None)

    try:
        marked_text = render_prompt_text(marked_conversation)
    except CompleteTokenInjectionError:
        raise
    except Exception as e:
        raise CompleteTokenInjectionError(
            "The marked chat template could not be rendered."
        ) from e

    if not isinstance(marked_text, str):
        raise CompleteTokenInjectionError("The marked prompt must render as text.")

    marker_pos = marked_text.find(_COMPLETE_TOKEN_BOUNDARY_MARKER)
    if marker_pos < 0:
        raise CompleteTokenInjectionError(
            "The chat template did not preserve the complete-token boundary marker."
        )

    eos_token = getattr(tokenizer, "eos_token", None)
    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if not isinstance(eos_token, str) or eos_token_id is None:
        raise CompleteTokenInjectionError(
            "Complete-token injection requires tokenizer EOS text and ID."
        )

    suffix_pos = marked_text.find(
        eos_token, marker_pos + len(_COMPLETE_TOKEN_BOUNDARY_MARKER)
    )
    if suffix_pos < 0:
        raise CompleteTokenInjectionError(
            "The chat template did not close the tokenized assistant with EOS."
        )
    marker_end = marker_pos + len(_COMPLETE_TOKEN_BOUNDARY_MARKER)
    if marked_text[marker_end:suffix_pos].strip():
        raise CompleteTokenInjectionError(
            "The chat template did not close the tokenized assistant immediately; "
            "the EOS boundary would skip intervening messages."
        )

    try:
        suffix_token_ids = _coerce_token_id_list(
            tokenizer.encode(marked_text[suffix_pos:], add_special_tokens=False),
            "Contextual suffix token IDs",
        )
    except CompleteTokenInjectionError:
        raise
    except Exception as e:
        raise CompleteTokenInjectionError(
            "The contextual suffix could not be tokenized."
        ) from e

    if not suffix_token_ids or suffix_token_ids[0] != eos_token_id:
        raise CompleteTokenInjectionError(
            "The contextual suffix did not begin with EOS."
        )
    model_cut_end = len(model_prefix_token_ids)
    if model_prefix_token_ids and model_prefix_token_ids[-1] == eos_token_id:
        model_cut_end -= 1
    return model_prefix_token_ids[:model_cut_end] + suffix_token_ids


def replace_prefix_tokens(
    tokenizer: Any,
    model_prefix_token_ids: list[int],
    template_prefix_token_ids: list[int],
    template_token_ids: list[int],
) -> list[int]:
    """This is a subroutine used inside the OpenAI-compatible Chat Completion server.

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

    The splice boundary is located by EOS count, not position: count the EOS
    tokens in template_prefix_token_ids and cut at the N-th EOS in
    template_token_ids. This is robust to chat templates that strip reasoning
    (<think>) blocks from history when the last message is a user turn -- that
    shifts token positions but not the per-message EOS count, so counting still
    finds the same boundary (and reduces to the last EOS of the prefix when
    nothing is stripped).

    Example (turn-by-turn, concise; eos_token_id = 2):
        Turn 1:
            - prefill_T1 (template prefill) = [11,12,13,40,41]
            - model output = [220,17,2]  # decodes to " 4" + EOS
            - model_prefix_token_ids = prefill_T1 + model output
              => [11,12,13,40,41,220,17,2]

        Turn 2 (template retokenizes prior assistant text differently):
            - template_prefix_token_ids = [11,12,13,40,41,1001,2]  # 1001 decodes to " 4"
            - template_token_ids = [11,12,13,40,41,1001,2,21,22,40,41]

        replace_prefix_tokens keeps the exact prior model tokens up to EOS and
        resumes from the template after that EOS:
            output => [11,12,13,40,41,220,17,2,21,22,40,41]
    """
    if not model_prefix_token_ids:
        return template_token_ids

    eos_token_id = tokenizer.eos_token_id
    assert eos_token_id is not None, "Tokenizer must have an EOS token ID"

    # The model isn't guaranteed to end on EOS (e.g. it hit max_tokens); chat
    # templates always add one, so cut the model input to just before its EOS.
    model_cut_end = len(model_prefix_token_ids)
    if model_prefix_token_ids[-1] == eos_token_id:
        model_cut_end -= 1

    # Locate the turn boundary by EOS count rather than token position. Qwen3
    # templates may strip prior reasoning blocks when re-rendering history;
    # EOS counting preserves the original generated reasoning tokens without
    # requiring a customized chat template.
    count_needed = template_prefix_token_ids.count(eos_token_id)
    count_seen = 0
    template_cut_start = -1
    for pos, tid in enumerate(template_token_ids):
        if tid == eos_token_id:
            count_seen += 1
            if count_seen == count_needed:
                template_cut_start = pos
                break

    assert template_cut_start >= 0, (
        f"EOS token #{count_needed} not found in template_token_ids "
        f"(only found {count_seen} EOS tokens total)!\n"
        f"Template prefix token IDs (everything before the final assistant message): {template_prefix_token_ids}\n\n"
        f"Template token IDs (everything that was sent to the model endpoint): {template_token_ids}\n\n"
        f"Template prefix repr (detokenized): {repr(tokenizer.decode(template_prefix_token_ids))}\n\n"
        f"Template repr (detokenized): {repr(tokenizer.decode(template_token_ids))}"
    )

    return (
        model_prefix_token_ids[:model_cut_end] + template_token_ids[template_cut_start:]
    )
