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

These utilities are backend-agnostic: they operate on token-ID lists plus a
tokenizer, with no engine calls. They are shared by the vLLM async worker
(``vllm_worker_async.py``) and the TRT-LLM HTTP server (``trtllm_http_server.py``),
which both put a message-based ``/v1/chat/completions`` layer in front of a token
engine for the agentic NeMo-Gym path. SGLang does not use these — it is driven
token-in/token-out via ``generate(input_ids)`` and never re-templates messages,
so it has no retokenization drift to correct.
"""

from typing import Any


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
