# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Small v2 Gym staging fixtures shared by the RL data-plane tests."""

from __future__ import annotations

from nemo_gym.token_id_capture.staging.digest import (
    compute_chain_hash,
    compute_extras_digest,
    compute_staging_digest,
    hash_token_ids,
)
from nemo_gym.token_id_capture.staging.rebuild import verify_and_linearize
from nemo_gym.token_id_capture.staging.records import (
    CallRecord,
    RolloutReceipt,
    StagedCallRecord,
    StagedCallSnapshot,
)


def f32(value: float) -> float:
    """Round one value through the float32 wire representation."""
    import struct

    return struct.unpack(">f", struct.pack(">f", value))[0]


def _record(
    *,
    rollout_id: str,
    model_call_id: str,
    parent_call_id: str | None,
    prev_len: int,
    token_ids: list[int],
    token_mask: list[float],
    logprobs: list[float],
    weight_version: int,
    parent_chain_hash: str | None = None,
    cumulative_prefix: list[int] | None = None,
) -> StagedCallRecord:
    token_mask = [f32(value) for value in token_mask]
    logprobs = [f32(value) for value in logprobs]
    mode = "text" if parent_call_id is None else "token_in"
    delta_len = len(token_ids)
    cum_len = prev_len + delta_len
    extras_digest = compute_extras_digest(None)
    values = {
        "rollout_id": rollout_id,
        "model_call_id": model_call_id,
        "parent_call_id": parent_call_id,
        "mode": mode,
        "prev_len": prev_len,
        "delta_len": delta_len,
        "cum_len": cum_len,
        "weight_version": weight_version,
        "token_ids_delta": token_ids,
        "token_mask_delta": token_mask,
        "generation_log_probs_delta": logprobs,
        "extras": None,
        "extras_digest": extras_digest,
        "chain_hash": compute_chain_hash(parent_chain_hash, token_ids),
        "cumulative_hash": hash_token_ids(list(cumulative_prefix or []) + token_ids),
    }
    return StagedCallRecord(
        **values,
        digest=compute_staging_digest(
            schema_version=2,
            digest_version=2,
            extras_digest_version=1,
            **{key: value for key, value in values.items() if key != "extras"},
        ),
    )


def _manifest(record: StagedCallRecord) -> CallRecord:
    return CallRecord(
        model_call_id=record.model_call_id,
        parent_call_id=record.parent_call_id,
        mode=record.mode,
        prev_len=record.prev_len,
        delta_len=record.delta_len,
        cum_len=record.cum_len,
        weight_version=record.weight_version,
        digest=record.digest,
        extras_digest=record.extras_digest,
        staging_key=record.staging_key,
        chain_hash=record.chain_hash,
        cumulative_hash=record.cumulative_hash,
        response_id=f"chatcmpl-{record.model_call_id}",
    )


def fixture_names() -> tuple[str, ...]:
    return ("worked_example", "single_call", "mixed_weight_versions")


def build_fixture_artifacts(
    name: str, *, rollout_id: str | None = None
) -> tuple[list[StagedCallRecord], RolloutReceipt, object]:
    if name not in fixture_names():
        raise KeyError(name)
    rollout_id = (
        rollout_id
        or {
            "worked_example": "g7_r0",
            "single_call": "single_r0",
            "mixed_weight_versions": "mixed_r0",
        }[name]
    )
    root = _record(
        rollout_id=rollout_id,
        model_call_id="c1",
        parent_call_id=None,
        prev_len=0,
        token_ids=[10, 11, 12, 13],
        token_mask=[0.0, 0.0, 1.0, 1.0],
        logprobs=[0.0, 0.0, -0.1, -0.2],
        weight_version=4,
    )
    records = [root]
    if name != "single_call":
        records.append(
            _record(
                rollout_id=rollout_id,
                model_call_id="c2",
                parent_call_id="c1",
                prev_len=root.cum_len,
                token_ids=[20, 21, 22],
                token_mask=[0.0, 1.0, 1.0],
                logprobs=[0.0, -0.3, -0.4],
                weight_version=5 if name == "mixed_weight_versions" else 4,
                parent_chain_hash=root.chain_hash,
                cumulative_prefix=root.token_ids_delta,
            )
        )
    receipt = RolloutReceipt(
        rollout_id=rollout_id,
        terminal_model_call_id=records[-1].model_call_id,
        manifest=[_manifest(record) for record in records],
        terminal_selection="declared",
    )
    snapshots = [
        StagedCallSnapshot.model_validate(record.model_dump()) for record in records
    ]
    return records, receipt, verify_and_linearize(receipt, snapshots)
