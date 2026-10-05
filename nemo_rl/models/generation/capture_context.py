# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Framework-owned serving-prefix decisions, independent of capture storage."""

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from nemo_gym.token_id_capture.staging.records import CaptureAdmission

LOGGER = logging.getLogger(__name__)


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
    Missing source/render evidence is an error. Evidence that proves the
    shared source prefix but not a supported continuation (a suffix with
    non-observation roles, or converted messages that do not preserve the
    candidate's response boundary) also starts a root: harnesses replay a
    shared prefix in shapes the splice cannot represent, and one such call
    must not fail the rollout.
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
        # The harness replayed the candidate's history but appended something
        # other than observations (an echoed or injected assistant message,
        # or nothing at all). The splice cannot represent that shape.
        LOGGER.warning(
            "Call %s extends candidate %s with roles %s instead of user/tool "
            "observations; rooting a new segment.",
            admission.model_call_id,
            admission.parent_call_id,
            suffix_roles,
        )
        return CaptureInputDecision(root, None, True)
    # The supported converter preserves these observations as separate messages.
    # Prove its cut explicitly; counting EOS tokens cannot establish this boundary.
    boundary = len(messages) - len(suffix)
    if (
        boundary < 1
        or messages[boundary - 1].get("role") != "assistant"
        or [message.get("role") for message in messages[boundary:]] != suffix_roles
    ):
        # Without the boundary proof the exact prefix cannot be installed.
        LOGGER.warning(
            "Call %s does not preserve candidate %s's response boundary after "
            "conversion; rooting a new segment.",
            admission.model_call_id,
            admission.parent_call_id,
        )
        return CaptureInputDecision(root, None, True)
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
