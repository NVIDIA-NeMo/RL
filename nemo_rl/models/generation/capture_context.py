# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Framework-owned serving-prefix decisions, independent of capture storage."""

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nemo_gym.token_id_capture.staging.records import CaptureAdmission


@dataclass(frozen=True)
class CaptureInputDecision:
    """Commit a continuation link or start a new context segment."""

    storage: "CaptureAdmission"
    serving: "CaptureAdmission | None"
    verify_retained_media: bool


def decide_capture_input(
    admission: "CaptureAdmission",
    *,
    messages: list[dict[str, Any]],
    allow_empty_tool_content: bool = False,
) -> CaptureInputDecision:
    """Resolve source-history facts before installing any exact-token prefix.

    A source rewrite starts a root and renders fresh. A proven continuation
    retains its exact sampled prefix and stages only its new suffix.
    Unsupported or missing evidence is an error, never a rewrite fallback.
    ``allow_empty_tool_content`` is a worker-proven rendering fact, not a
    harness option. It permits only null/empty Chat assistant tool content.
    """
    # Optional Gym contracts are required only on a captured serving request.
    from nemo_gym.token_id_capture.replay import same_replay_source, summarize_replay
    from nemo_gym.token_id_capture.staging.records import CaptureAdmission

    if admission.mode != "candidate":
        return CaptureInputDecision(admission, admission, False)
    root = CaptureAdmission(
        rollout_id=admission.rollout_id,
        model_call_id=admission.model_call_id,
        mode="text",
    )
    current = admission.request_replay
    previous = admission.candidate_replay
    if current is None or current.render_digest is None:
        raise ValueError("Missing current source/render evidence")
    if admission.parent_call_id is None:
        return CaptureInputDecision(root, None, True)
    if previous is None or previous.render_digest is None:
        raise ValueError("Missing predecessor source/render evidence")
    prefix_size = previous.item_count
    if (
        previous.render_digest != current.render_digest
        or prefix_size > len(current.items)
        or not same_replay_source(
            summarize_replay(current, item_count=prefix_size),
            previous,
            allow_empty_tool_content=allow_empty_tool_content,
        )
    ):
        return CaptureInputDecision(root, None, True)
    suffix = current.items[prefix_size:]
    suffix_roles = [item.role for item in suffix]
    if not suffix or any(role not in ("user", "tool") for role in suffix_roles):
        raise ValueError("Prefix preservation requires appended user/tool observations")
    # The supported converter preserves these observations as separate messages.
    # Prove its cut explicitly; counting EOS tokens cannot establish this boundary.
    boundary = len(messages) - len(suffix)
    if (
        boundary < 1
        or messages[boundary - 1].get("role") != "assistant"
        or [message.get("role") for message in messages[boundary:]] != suffix_roles
    ):
        raise ValueError(
            "Converted messages do not preserve the candidate response boundary"
        )
    serving = CaptureAdmission(
        rollout_id=admission.rollout_id,
        model_call_id=admission.model_call_id,
        mode="token_in",
        parent_call_id=admission.parent_call_id,
        prev_len=admission.prev_len,
        parent_chain_hash=admission.parent_chain_hash,
        staging_chain=admission.staging_chain,
        required_prefix_token_ids=admission.required_prefix_token_ids,
    )
    return CaptureInputDecision(serving, serving, True)
