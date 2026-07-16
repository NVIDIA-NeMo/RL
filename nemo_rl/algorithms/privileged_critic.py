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
"""Privileged (answer-conditioned) critic for PPO.

An *asymmetric actor-critic*: the value model additionally sees privileged
information the policy never sees — the golden/reference answer — so its
per-token value estimates (and therefore the GAE advantages) are sharper. The
policy stays blind and the privileged conditioning is training-only. See
``privileged_critic_proposal.md`` for the rationale, the unbiasedness argument
(``V(h, z)`` form; the answer is action-independent and prompt-determined), and
the safety rules.

Design (kept deliberately small so GAE, the value workers and the policy are all
untouched):

* The critic scores an **answer-augmented** sequence ``x' = [prompt(+gold),
  response]`` that is constructed at the **message level** and re-rendered with
  the chat template, so it stays a well-formed conversation the value model (a
  chat model + value head) can process. Crucially, the **RESPONSE token-ids are
  the verbatim generated tokens** — we never re-tokenize the response, so the
  value at each response position corresponds to exactly the state the policy
  produced. Only the prompt region differs (it now contains the answer).
* Values / returns are moved between the augmented layout ``x'`` and the original
  layout ``x = [prompt, response]`` by the response mask. Because the response
  tokens are identical, the two masks select the same tokens with equal per-row
  counts, so the mapping is an exact masked scatter (asserted).

SAFETY (enforced by construction, per the theory):
* The value model is a **separate worker** from the policy (PPO always builds one
  via ``init_value``) — no shared parameters, so the answer cannot leak into the
  policy through shared features.
* The answer only ever enters the critic's advantage **magnitude** (a baseline),
  never the policy path or the reward — the outcome reward alone sets direction.

Prompt construction is the delicate part: do NOT splice raw answer tokens between
the prompt and response token-ids (that yields a malformed conversation and
out-of-distribution value estimates). Instead fold the answer into a prompt turn
and re-render with the chat template, then append the verbatim response tokens.
"""

from __future__ import annotations

from typing import Any, Optional

import torch

from nemo_rl.data.llm_message_utils import batched_message_log_to_flat_message
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

# The answer is framed as a clearly-delimited grader note (not the solver's turn),
# so it is well-formed and less trivially string-matchable than a bare "\boxed{}".
DEFAULT_TEMPLATE = (
    "\n\n[Reference answer, provided to the grader only and NOT visible to the "
    "solver — use it to judge whether the assistant's solution is on track: {answer}]"
)


def _truncate_answer(
    gold: str, tokenizer: Any, max_answer_tokens: Optional[int]
) -> str:
    """Cap the reference answer to ``max_answer_tokens`` (keeps sequences bounded
    for full-solution exposure; a no-op for short final answers)."""
    if not gold or not max_answer_tokens or max_answer_tokens <= 0:
        return gold or ""
    ids = tokenizer.encode(gold, add_special_tokens=False)
    if len(ids) <= max_answer_tokens:
        return gold
    return tokenizer.decode(ids[:max_answer_tokens])


def _inject_answer(
    prompt_msgs: list[dict], gold: str, placement: str, template: str
) -> list[dict]:
    """Fold the golden answer into the (text) prompt messages as a well-formed turn.

    placement: ``user_suffix`` (default, always supported) | ``user_prefix`` |
    ``system``.
    """
    note = template.format(answer=gold)
    msgs = [{"role": m["role"], "content": m["content"]} for m in prompt_msgs]
    if placement == "system":
        return [{"role": "system", "content": note.lstrip("\n")}] + msgs
    # find the last user turn to attach the note to
    user_idxs = [i for i, m in enumerate(msgs) if m["role"] == "user"]
    if not user_idxs:  # no user turn to attach to -> fall back to a system note
        return [{"role": "system", "content": note.lstrip("\n")}] + msgs
    i = user_idxs[-1]
    if placement == "user_prefix":
        msgs[i]["content"] = note.lstrip("\n") + "\n\n" + msgs[i]["content"]
    else:  # user_suffix (default)
        msgs[i]["content"] = msgs[i]["content"] + note
    return msgs


def build_privileged_value_inputs(
    repeated_batch: BatchedDataDict,
    tokenizer: Any,
    pcfg: dict[str, Any],
    make_seq_len_divisible_by: int = 1,
) -> BatchedDataDict:
    """Build the answer-augmented critic input batch, row-aligned with ``train_data``.

    Returns a ``BatchedDataDict`` with ``input_ids`` / ``input_lengths`` /
    ``token_mask``, where ``token_mask`` marks exactly the verbatim response tokens.

    ``setup()`` raises the value model's sequence budget by ``max_answer_tokens`` (+
    margin) when this is enabled, so answer-augmented sequences fit its packing bins.
    """
    placement = pcfg.get("placement", "user_suffix")
    template = pcfg.get("template", DEFAULT_TEMPLATE)
    max_answer_tokens = pcfg.get("max_answer_tokens", 256)

    message_logs = repeated_batch["message_log"]
    env_infos = repeated_batch.get("extra_env_info", None)
    if env_infos is None:
        env_infos = [None] * len(message_logs)

    critic_message_logs: list[list[dict]] = []
    for msgs, info in zip(message_logs, env_infos):
        # Split into prompt (leading non-assistant turns) and response (the
        # assistant-generated turn[s]). Single-turn RLVR: the rollout is
        # [user, assistant, environment-feedback]; the trailing environment turn is
        # masked out AND comes after the response, so it can't affect the causal value
        # at response positions — drop it. Reject genuinely interleaved multi-turn
        # (an environment turn wedged BETWEEN assistant turns).
        assistant_idxs = [i for i, m in enumerate(msgs) if m["role"] == "assistant"]
        assert assistant_idxs, (
            "privileged critic: no assistant/response message found in the rollout"
        )
        first_a, last_a = assistant_idxs[0], assistant_idxs[-1]
        assert all(m["role"] == "assistant" for m in msgs[first_a : last_a + 1]), (
            "privileged critic currently supports single-turn rollouts (an environment "
            "turn is interleaved between assistant turns — multi-turn not supported)."
        )
        prompt_msgs = msgs[:first_a]
        response_msgs = msgs[first_a : last_a + 1]
        assert not any(("images" in m or "videos" in m) for m in msgs), (
            "privileged critic supports text-only rollouts."
        )

        gold = ""
        if info is not None:
            gold = _truncate_answer(
                info.get("ground_truth", "") or "", tokenizer, max_answer_tokens
            )

        # Fold the answer into the prompt, render to a string with the chat template,
        # then tokenize — the SAME two-step path the data processor uses to build the
        # original prompt (apply_chat_template(tokenize=False) -> str; tokenizer() ->
        # ids). Matching it keeps the critic's prompt format consistent with the
        # policy's and avoids tokenizer-specific quirks of tokenize=True.
        aug_prompt_text_msgs = _inject_answer(prompt_msgs, gold, placement, template)
        rendered = tokenizer.apply_chat_template(
            aug_prompt_text_msgs,
            tokenize=False,
            add_generation_prompt=True,
            add_special_tokens=False,
        )
        prompt_ids = tokenizer(
            rendered, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0].to(dtype=torch.long)

        # ...then append the VERBATIM generated response token-ids.
        critic_msgs: list[dict] = [
            {
                "role": "user",
                "content": "",  # token_ids provided; content is unused by the flattener
                "token_ids": prompt_ids,
                "token_loss_mask": torch.zeros_like(prompt_ids),
            }
        ]
        for rm in response_msgs:
            rid = torch.as_tensor(rm["token_ids"], dtype=torch.long).flatten()
            critic_msgs.append(
                {
                    "role": "assistant",
                    "content": "",
                    "token_ids": rid,
                    "token_loss_mask": torch.ones_like(rid),
                }
            )
        critic_message_logs.append(critic_msgs)

    flat, input_lengths = batched_message_log_to_flat_message(
        critic_message_logs,
        pad_value_dict={"token_ids": tokenizer.pad_token_id},
        make_sequence_length_divisible_by=make_seq_len_divisible_by,
    )
    return BatchedDataDict(
        {
            "input_ids": flat["token_ids"],
            "input_lengths": input_lengths,
            "token_mask": flat["token_loss_mask"],
        }
    )


def remap_by_response_mask(
    src: torch.Tensor, src_mask: torch.Tensor, dst_mask: torch.Tensor
) -> torch.Tensor:
    """Move per-token values between two layouts that mark the SAME response tokens.

    ``src[src_mask] -> dst[dst_mask]``; ``dst`` is zero elsewhere (GAE's
    carry-forward masking ignores those non-response positions). Requires equal
    per-row response counts — asserted, since a mismatch means the response tokens
    were not preserved verbatim (a construction bug).
    """
    dev = dst_mask.device
    src = src.to(dev)
    src_mask_b = src_mask.to(dev).bool()
    dst_mask_b = dst_mask.bool()

    src_counts = src_mask_b.sum(dim=-1)
    dst_counts = dst_mask_b.sum(dim=-1)
    if not torch.equal(src_counts, dst_counts):
        raise AssertionError(
            "privileged critic: per-row response-token counts differ between the "
            "augmented and original layouts — the response tokens were not preserved "
            "verbatim.\n"
            f"  augmented per-row counts: {src_counts.tolist()}\n"
            f"  original  per-row counts: {dst_counts.tolist()}"
        )

    dst = torch.zeros(
        dst_mask_b.shape[0], dst_mask_b.shape[1], dtype=src.dtype, device=dev
    )
    dst[dst_mask_b] = src[src_mask_b]
    return dst
