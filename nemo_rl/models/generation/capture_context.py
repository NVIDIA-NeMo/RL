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
    """Verify the converted boundary and renderer for a Gym-selected candidate.

    Gym proves source continuity and effective render-option equality. RL
    installs exact captured tokens only when conversion preserves the cut.
    Managed roots retain media validation; legacy admissions are unchanged.
    ``allow_empty_tool_content`` is a worker-proven rendering fact, not a
    harness option. It permits only null/empty Chat assistant tool content.
    """
    # Optional Gym contracts are required only on a captured serving request.
    from nemo_gym.token_id_capture.staging.records import CaptureAdmission

    if admission.mode != "candidate" and admission.request_replay is None:
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
    if admission.mode == "text":
        return CaptureInputDecision(root, None, True)
    if previous is None or admission.requires_empty_tool_content_equivalence is None:
        raise ValueError("Missing candidate source-match proof")
    if (
        admission.requires_empty_tool_content_equivalence
        and not allow_empty_tool_content
    ):
        return CaptureInputDecision(root, None, True)
    suffix = current.items[previous.item_count :]
    suffix_roles = [item.role for item in suffix]
    # The supported converter preserves these observations as separate messages.
    # Prove its cut explicitly; counting EOS tokens cannot establish this boundary.
    boundary = len(messages) - len(suffix)
    if (
        not suffix
        or boundary < 1
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
