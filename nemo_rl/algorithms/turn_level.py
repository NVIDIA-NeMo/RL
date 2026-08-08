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
"""Turn-level MDP bookkeeping for agentic PPO (see research/ppo/turn_level_critic_plan.md).

Long agentic rollouts (SWE: ~92 assistant turns, up to 200, over ~45k response
tokens) are a *turn* MDP wearing a token MDP's clothes:

    state  s_k = the context before the k-th assistant message
    action a_k = the whole k-th assistant message
    reward r_k = 0 for k < K, r_K = R (environment-graded, terminal)

Running GAE over tokens instead of turns is not merely wasteful, it removes the
λ knob entirely: at the token level λ has effective horizon 1/(1-λ) **tokens**,
so λ=0.95 reaches 20 tokens and the terminal reward cannot propagate. That is
why the production config ends up at λ = 1 - 1.5e-5 (``length_adaptive_alpha``),
where GAE degenerates to the pure baseline ``A_t = R - V(s_t)`` and nothing in
the objective assigns credit to a particular turn. Over ~92 turns, λ=0.97 is a
33-turn horizon — a usable bias/variance knob.

Where V(s_k) lives
------------------
The value workers right-shift the value head
(``megatron_value_worker._value_loss_prepare_fn``), so

    values[t] = V(state before token t)

which means **V(s_k) is read at the FIRST token of assistant message k**: that
position has attended to the entire preceding observation and none of the
action. Anchors are therefore always inside the response mask, so nothing here
needs a gradient at a non-response position, and no model / worker / sequence
packing / context-parallel code changes.

Nothing in this module is sharded: the driver builds these tensors, uses them to
scatter advantages and to assemble the critic's anchor batch, and throws them
away. Keeping them off the workers is what keeps the change small.
"""

from dataclasses import dataclass
from typing import Any, Optional

import torch

# Roles that constitute an agent action. Everything else (user, tool,
# environment, system) is an observation and only ever contributes state.
_ACTION_ROLE = "assistant"


@dataclass
class TurnSpans:
    """Turn structure of one flattened batch of message logs.

    Attributes:
        anchor_mask: ``[B, S]``, 1 at the first token of each assistant message.
            Always a subset of the response ``token_mask``.
        turn_index: ``[B, S]`` int32, the turn ordinal ``k`` for every token of
            assistant message ``k``; ``-1`` at observation/padding positions.
        anchor_pos: ``[B, K_max]`` int64, token index of each turn's anchor.
            Invalid slots hold 0 (a prompt position, never an anchor).
        turn_valid: ``[B, K_max]`` bool, whether slot ``k`` is a real turn.
        num_turns: ``[B]`` int64, assistant messages per sample.
        turn_ntokens: ``[B, K_max]`` int64, tokens in each turn (0 if invalid).
    """

    anchor_mask: torch.Tensor
    turn_index: torch.Tensor
    anchor_pos: torch.Tensor
    turn_valid: torch.Tensor
    num_turns: torch.Tensor
    turn_ntokens: torch.Tensor


def build_turn_spans(
    message_log_batch: list[list[dict[str, Any]]],
    seq_len: int,
    mask_dtype: torch.dtype = torch.long,
) -> TurnSpans:
    """Locate every assistant message in the flattened ``[B, seq_len]`` layout.

    Mirrors :func:`batched_message_log_to_flat_message`'s concatenation order
    (messages laid out back to back, right padding), so the returned indices
    address ``train_data["input_ids"]`` / ``values`` directly.

    Args:
        message_log_batch: one message log per sample, each a list of messages
            with ``role`` and ``token_ids``.
        seq_len: padded sequence length of the flattened batch.
        mask_dtype: dtype for ``anchor_mask`` (match the batch's ``token_mask``).

    Raises:
        ValueError: if a sample's flattened length exceeds ``seq_len``, or an
            assistant message starts at token 0 (there is always a prompt, and
            index 0 carries no value after the right-shift).
    """
    batch_size = len(message_log_batch)
    per_sample_starts: list[torch.Tensor] = []
    per_sample_lens: list[torch.Tensor] = []
    turn_index = torch.full((batch_size, seq_len), -1, dtype=torch.int32)

    for i, message_log in enumerate(message_log_batch):
        lengths = torch.tensor(
            [len(m["token_ids"]) for m in message_log], dtype=torch.long
        )
        total = int(lengths.sum())
        if total > seq_len:
            raise ValueError(
                f"Sample {i} flattens to {total} tokens but the batch is padded "
                f"to {seq_len}. build_turn_spans must be called with the same "
                "seq_len as the flattened batch."
            )
        is_action = torch.tensor(
            [m["role"] == _ACTION_ROLE for m in message_log], dtype=torch.bool
        )
        starts_all = torch.cumsum(lengths, 0) - lengths

        # Non-empty assistant messages only: a zero-length message has no anchor
        # and no tokens to carry an advantage.
        keep = is_action & (lengths > 0)
        starts = starts_all[keep]
        lens = lengths[keep]
        if starts.numel() > 0 and int(starts[0]) == 0:
            raise ValueError(
                f"Sample {i} starts with an assistant message at token 0. The "
                "value head is right-shifted, so position 0 carries no value "
                "and cannot anchor a turn; every sample must begin with a prompt."
            )

        # Turn ordinal per token, without a Python loop over messages: map each
        # token to its message, then each message to its turn (-1 if not an action).
        turn_of_msg = torch.cumsum(keep.long(), 0) - 1
        turn_of_msg[~keep] = -1
        if total > 0:
            msg_of_tok = torch.repeat_interleave(
                torch.arange(len(message_log), dtype=torch.long), lengths
            )
            turn_index[i, :total] = turn_of_msg[msg_of_tok].to(torch.int32)

        per_sample_starts.append(starts)
        per_sample_lens.append(lens)

    num_turns = torch.tensor([s.numel() for s in per_sample_starts], dtype=torch.long)
    max_turns = int(num_turns.max()) if batch_size > 0 else 0
    max_turns = max(max_turns, 1)  # keep a well-formed [B, K] even if no turns

    anchor_pos = torch.zeros((batch_size, max_turns), dtype=torch.long)
    turn_valid = torch.zeros((batch_size, max_turns), dtype=torch.bool)
    turn_ntokens = torch.zeros((batch_size, max_turns), dtype=torch.long)
    for i, (starts, lens) in enumerate(zip(per_sample_starts, per_sample_lens)):
        k = starts.numel()
        if k == 0:
            continue
        anchor_pos[i, :k] = starts
        turn_valid[i, :k] = True
        turn_ntokens[i, :k] = lens

    anchor_mask = torch.zeros((batch_size, seq_len), dtype=mask_dtype)
    anchor_mask.scatter_(1, anchor_pos, turn_valid.to(mask_dtype))
    # Slot 0 of an all-invalid row scattered a 0, but a valid row may also have
    # written to index 0 only if it had an anchor there — ruled out above.

    return TurnSpans(
        anchor_mask=anchor_mask,
        turn_index=turn_index,
        anchor_pos=anchor_pos,
        turn_valid=turn_valid,
        num_turns=num_turns,
        turn_ntokens=turn_ntokens,
    )


def validate_turn_spans(
    spans: TurnSpans,
    token_mask: torch.Tensor,
    sample_mask: Optional[torch.Tensor] = None,
) -> None:
    """Fail loud on a turn structure that disagrees with the batch it describes.

    The invariant that matters is a BIJECTION between response tokens and turns:
    every anchor is a response token, and every response token belongs to some
    turn. Either direction failing means the turn structure and the flattened
    batch were built from different things, and credit would be attributed to
    the wrong tokens — silently.

    A sample with no assistant message is NOT an error: its ``token_mask`` row is
    empty, so it contributes nothing to either loss. It is only reported, because
    aborting a 512-sample step over one empty trajectory is a far worse failure
    than the trajectory itself.
    """
    anchors = spans.anchor_mask.bool()
    resp = token_mask.bool()

    stray = int((anchors & ~resp).sum())
    if stray:
        raise ValueError(
            f"{stray} turn anchors fall outside the response mask. Anchors are "
            "the first token of an assistant message and must be trainable "
            "response positions; a mismatch means the turn structure and the "
            "flattened batch disagree."
        )

    orphan = int((resp & (spans.turn_index.to(resp.device) < 0)).sum())
    if orphan:
        raise ValueError(
            f"{orphan} response tokens belong to no turn. Every trainable token "
            "must sit inside an assistant message, or its policy gradient would "
            "carry no advantage; the turn structure and the loss mask disagree."
        )

    if sample_mask is not None:
        no_turns = (spans.num_turns == 0) & (sample_mask.detach().cpu() > 0)
        n = int(no_turns.sum())
        if n:
            bad = torch.nonzero(no_turns).flatten().tolist()[:5]
            print(
                f"  ⚠️ {n} unmasked samples have no assistant message (e.g. "
                f"indices {bad}); they contribute no tokens to either loss.",
                flush=True,
            )


def gather_turn_values(
    values: torch.Tensor, spans: TurnSpans
) -> torch.Tensor:
    """``V(s_k)`` for every turn: ``[B, S] -> [B, K_max]`` (0 at invalid slots)."""
    v = torch.gather(values, 1, spans.anchor_pos.to(values.device))
    return v * spans.turn_valid.to(device=values.device, dtype=values.dtype)


def build_turn_rewards(
    rewards: torch.Tensor,
    spans: TurnSpans,
    token_level_penalty: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Per-turn reward ``[B, K_max]``: terminal ``R`` at the last turn.

    ``token_level_penalty`` (e.g. ``-kl_coef * KL`` per token, already signed) is
    summed within each turn and added to that turn's reward, so a per-token
    shaping signal survives the move to turn granularity.
    """
    device = rewards.device
    turn_valid = spans.turn_valid.to(device)
    turn_rewards = torch.zeros(
        turn_valid.shape, device=device, dtype=rewards.dtype
    )

    if token_level_penalty is not None:
        # Sum the per-token penalty into its turn. index_add over a flattened
        # [B*K] buffer keeps this a single vectorised op.
        b, s = token_level_penalty.shape
        k = turn_valid.shape[1]
        ti = spans.turn_index.to(device).long()
        valid = ti >= 0
        flat_idx = (
            torch.arange(b, device=device).unsqueeze(1) * k + ti.clamp(min=0)
        )[valid]
        acc = torch.zeros(b * k, device=device, dtype=rewards.dtype)
        acc.index_add_(0, flat_idx, token_level_penalty[valid].to(rewards.dtype))
        turn_rewards = turn_rewards + acc.view(b, k)

    # Terminal reward on the last valid turn of each sample.
    last = (spans.num_turns.to(device) - 1).clamp(min=0)
    has_turns = spans.num_turns.to(device) > 0
    rows = torch.nonzero(has_turns).flatten()
    if rows.numel():
        turn_rewards[rows, last[rows]] += rewards[rows]

    return turn_rewards * turn_valid.to(turn_rewards.dtype)


def turn_gae(
    turn_values: torch.Tensor,
    turn_rewards: torch.Tensor,
    turn_valid: torch.Tensor,
    gamma: float,
    gae_lambda: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """GAE over turns.

    ``δ_k = r_k + γ V(s_{k+1}) - V(s_k)``, ``A_k = δ_k + γλ A_{k+1}``, with
    ``V(s_{K+1}) = 0``. Truncated rollouts are still terminal in this MDP: the
    environment grades the final state whether or not the agent ran out of
    turns, so ``R`` is the realised episode return and bootstrapping 0 is right.

    Uses the same carry-forward masking as the token-level estimator (invalid
    slots preserve the accumulators rather than zeroing them), so the trailing
    padding of short samples is skipped instead of injecting phantom TD errors.

    Args:
        turn_values: ``[B, K]`` V(s_k).
        turn_rewards: ``[B, K]`` per-turn rewards.
        turn_valid: ``[B, K]`` bool.
        gamma: discount.
        gae_lambda: scalar λ. Deliberately not per-sample: a length-adaptive λ is
            the token-level pathology this estimator exists to remove.

    Returns:
        ``(advantages, returns)``, each ``[B, K]``. ``returns = advantages + values``.
    """
    dtype, device = turn_values.dtype, turn_values.device
    num_turns = turn_values.shape[1]
    lam = gae_lambda

    next_values = torch.zeros(turn_values.shape[0], device=device, dtype=dtype)
    last_gae = torch.zeros_like(next_values)
    out = torch.zeros_like(turn_values)
    valid = turn_valid.to(device=device, dtype=dtype)

    for k in reversed(range(num_turns)):
        delta = turn_rewards[:, k] + gamma * next_values - turn_values[:, k]
        new_gae = delta + gamma * lam * last_gae
        m = valid[:, k]
        next_values = turn_values[:, k] * m + (1 - m) * next_values
        last_gae = new_gae * m + (1 - m) * last_gae
        out[:, k] = last_gae

    return out, out + turn_values


def scatter_turns_to_tokens(
    turn_quantity: torch.Tensor, spans: TurnSpans, seq_len: int
) -> torch.Tensor:
    """Broadcast a per-turn quantity ``[B, K]`` onto every token of its turn.

    This is what makes the policy update turn-level: all tokens of one assistant
    message share one advantage, the standard treatment of a multi-token action.
    """
    ti = spans.turn_index.to(turn_quantity.device)
    valid = ti >= 0
    gathered = torch.gather(turn_quantity, 1, ti.clamp(min=0).long()[:, :seq_len])
    return gathered * valid[:, :seq_len].to(turn_quantity.dtype)


def scatter_turns_to_anchors(
    turn_quantity: torch.Tensor, spans: TurnSpans, seq_len: int
) -> torch.Tensor:
    """Place a per-turn quantity at its anchor token only; 0 everywhere else.

    Used for the critic's regression targets: paired with
    ``token_mask = anchor_mask`` the value loss sees exactly one target per
    decision point, equally weighted, instead of one per token.
    """
    out = torch.zeros(
        (turn_quantity.shape[0], seq_len),
        device=turn_quantity.device,
        dtype=turn_quantity.dtype,
    )
    vals = turn_quantity * spans.turn_valid.to(
        device=turn_quantity.device, dtype=turn_quantity.dtype
    )
    out.scatter_(1, spans.anchor_pos.to(turn_quantity.device), vals)
    return out


def build_turn_value_batch(train_data: Any, spans: TurnSpans) -> Any:
    """The critic's anchor batch: same sequences, one supervised position per turn.

    Swapping only ``token_mask`` is enough to retarget the whole critic path:
    ``process_global_batch`` derives ``global_valid_toks`` from ``token_mask``,
    so :class:`MseValueLossFn` becomes an equal-weighted mean over turns with no
    change to the loss, the value workers, sequence packing, or CP. This mirrors
    how the privileged critic already trains on its own batch.

    ``returns`` must already be in anchor layout (see
    :func:`scatter_turns_to_anchors`).
    """
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    required = ("input_ids", "input_lengths", "sample_mask", "returns")
    missing = [k for k in required if k not in train_data]
    if missing:
        raise ValueError(
            f"build_turn_value_batch is missing {missing}; the turn returns must "
            "be written to train_data['returns'] before the critic batch is built."
        )
    keys = required + ("values",)  # values: old values, for the PPO value clip
    batch = BatchedDataDict({k: train_data[k] for k in keys if k in train_data})
    batch["token_mask"] = spans.anchor_mask.to(train_data["token_mask"].dtype)
    # Carry multimodal side inputs through untouched (same rule the PPO loops
    # use to assemble extra_multimodal_data).
    batch.update(train_data.get_multimodal_dict(as_tensors=False))
    return batch


# ===============================================================================
# Metrics
# ===============================================================================
def turn_level_metrics(
    turn_values: torch.Tensor,
    turn_advantages: torch.Tensor,
    spans: TurnSpans,
    sample_mask: Optional[torch.Tensor] = None,
) -> dict[str, float]:
    """Diagnostics that only exist at turn granularity.

    Deliberately does NOT re-derive per-position EV/bias or a terminal AUC: with
    ``token_mask = anchor_mask`` the existing ``_positional_value_metrics`` and
    ``terminal_value_reward_auc`` already bin by turn ordinal and read the last
    turn's anchor, and produce bit-identical numbers. Only the quantities with no
    token-level counterpart live here.

    ``turn_advantages`` is expected PRE-normalization (the estimator whitens the
    token-level tensor afterwards), so the ``advantage/turn_*_prenorm`` keys show
    the real scale of ``A_k`` — which collapses as λ_policy drops — rather than
    the post-whitening ``advantages/std`` == 1 that PPO logs separately.
    """
    valid = spans.turn_valid.to(turn_values.device)
    if sample_mask is not None:
        valid = valid & (sample_mask.to(turn_values.device) > 0).unsqueeze(1)
    if int(valid.sum()) < 2:
        return {}

    v = turn_values[valid].float()
    a = turn_advantages[valid].float()
    return {
        "critic/turn_value_mean": v.mean().item(),
        "critic/turn_value_std": v.std().item(),
        "advantage/turn_abs_mean_prenorm": a.abs().mean().item(),
        "advantage/turn_std_prenorm": a.std().item(),
        "turn/num_turns_mean": spans.num_turns.float().mean().item(),
        "turn/num_turns_max": float(spans.num_turns.max()),
        "turn/tokens_per_turn_mean": (
            spans.turn_ntokens[spans.turn_valid].float().mean().item()
        ),
        "turn/total_turns": float(int(valid.sum())),
    }
