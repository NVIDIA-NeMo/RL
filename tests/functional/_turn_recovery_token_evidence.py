# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Small, payload-free evidence for turn-recovery token preservation tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


def _token_ids_digest(token_ids: list[int]) -> str:
    """Hash one exact token-ID sequence using a stable JSON encoding."""
    payload = json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def build_rollout_token_evidence(
    *,
    rollout_id: str,
    canonical_token_ids: list[int],
    staged_segments: list[dict[str, Any]],
) -> dict[str, Any]:
    """Compare every saved pre-cut delta with its exact canonical-row slice."""
    if not staged_segments:
        raise AssertionError(
            f"continued rollout {rollout_id!r} has no saved pre-cut token segments"
        )

    evidence_segments: list[dict[str, Any]] = []
    seen_keys: set[str] = set()
    for segment in sorted(
        staged_segments,
        key=lambda item: (item["prev_len"], item["cum_len"], item["staging_key"]),
    ):
        staging_key = str(segment["staging_key"])
        if staging_key in seen_keys:
            raise AssertionError(
                f"continued rollout {rollout_id!r} repeats staging key {staging_key!r}"
            )
        seen_keys.add(staging_key)

        prev_len = int(segment["prev_len"])
        cum_len = int(segment["cum_len"])
        delta = [int(token_id) for token_id in segment["token_ids_delta"]]
        selection_staging_digest = str(segment["selection_staging_digest"])
        restored_staging_digest = str(segment["restored_staging_digest"])
        if (
            not selection_staging_digest
            or selection_staging_digest != restored_staging_digest
        ):
            raise AssertionError(
                "restored TQ row differs from its pre-crash lineage commitment: "
                f"rollout={rollout_id!r}, key={staging_key!r}"
            )
        if prev_len < 0 or cum_len < prev_len or len(delta) != cum_len - prev_len:
            raise AssertionError(
                "saved token segment has inconsistent boundaries: "
                f"rollout={rollout_id!r}, key={staging_key!r}, "
                f"prev_len={prev_len}, cum_len={cum_len}, delta_len={len(delta)}"
            )
        if cum_len > len(canonical_token_ids):
            raise AssertionError(
                "saved token segment extends past the final canonical row: "
                f"rollout={rollout_id!r}, key={staging_key!r}, "
                f"cum_len={cum_len}, canonical_len={len(canonical_token_ids)}"
            )

        canonical_slice = canonical_token_ids[prev_len:cum_len]
        if canonical_slice != delta:
            raise AssertionError(
                "restored canonical row changed a saved pre-cut token segment: "
                f"rollout={rollout_id!r}, key={staging_key!r}, "
                f"range=[{prev_len}:{cum_len}]"
            )
        delta_digest = _token_ids_digest(delta)
        evidence_segments.append(
            {
                "staging_key": staging_key,
                "prev_len": prev_len,
                "cum_len": cum_len,
                "token_count": len(delta),
                "selection_staging_digest": selection_staging_digest,
                "restored_staging_digest": restored_staging_digest,
                "saved_delta_sha256": delta_digest,
                "canonical_slice_sha256": _token_ids_digest(canonical_slice),
                "matched": True,
            }
        )

    return {
        "rollout_id": rollout_id,
        "canonical_token_count": len(canonical_token_ids),
        "canonical_token_ids_sha256": _token_ids_digest(canonical_token_ids),
        "segments": evidence_segments,
    }


def verify_token_evidence(
    selected: dict[str, Any],
    evidence_dir: Path,
) -> None:
    """Require exact pre-cut token preservation for every selected continuation."""
    expected_rows = selected.get("continued_rollouts")
    if not isinstance(expected_rows, list) or not expected_rows:
        raise AssertionError("snapshot selection contains no continued rollouts")
    expected_by_rollout: dict[str, dict[str, dict[str, Any]]] = {}
    for row in expected_rows:
        if not isinstance(row, dict):
            raise TypeError("continued rollout descriptor must be an object")
        rollout_id = row.get("rollout_id")
        segments = row.get("pre_cut_segments")
        if not isinstance(rollout_id, str) or not rollout_id:
            raise TypeError("continued rollout descriptor has no rollout_id")
        if (
            not isinstance(segments, list)
            or not segments
            or any(not isinstance(segment, dict) for segment in segments)
        ):
            raise TypeError(
                f"continued rollout {rollout_id!r} has invalid pre-cut segments"
            )
        if rollout_id in expected_by_rollout:
            raise AssertionError(f"continued rollout {rollout_id!r} is repeated")
        expected_segments: dict[str, dict[str, Any]] = {}
        for segment in segments:
            staging_key = segment.get("staging_key")
            if not isinstance(staging_key, str) or not staging_key:
                raise TypeError(
                    f"continued rollout {rollout_id!r} has an invalid staging key"
                )
            if staging_key in expected_segments:
                raise AssertionError(
                    f"continued rollout {rollout_id!r} repeats {staging_key!r}"
                )
            expected_segments[staging_key] = segment
        expected_by_rollout[rollout_id] = expected_segments

    observed: dict[str, list[dict[str, Any]]] = {
        rollout_id: [] for rollout_id in expected_by_rollout
    }
    for path in sorted(evidence_dir.glob("*.json")):
        payload = json.loads(path.read_text())
        if not isinstance(payload, dict):
            raise TypeError(f"token evidence must be an object: {path}")
        for row in payload.get("rollouts", []):
            if not isinstance(row, dict):
                raise TypeError(f"token evidence rollout must be an object: {path}")
            rollout_id = row.get("rollout_id")
            if rollout_id in observed:
                observed[rollout_id].append(row)

    for rollout_id, expected_segments in expected_by_rollout.items():
        matches = observed[rollout_id]
        if len(matches) != 1:
            raise AssertionError(
                "continued rollout must have exactly one final token-evidence row: "
                f"rollout={rollout_id!r}, matches={len(matches)}"
            )
        row = matches[0]
        segments = row.get("segments")
        if not isinstance(segments, list) or not segments:
            raise AssertionError(
                f"continued rollout {rollout_id!r} has no compared token segments"
            )
        observed_keys = {segment.get("staging_key") for segment in segments}
        expected_keys = set(expected_segments)
        if observed_keys != expected_keys:
            raise AssertionError(
                "token evidence does not cover the snapshot's storage references: "
                f"rollout={rollout_id!r}, missing={sorted(expected_keys - observed_keys)!r}, "
                f"unexpected={sorted(observed_keys - expected_keys)!r}"
            )
        for segment in segments:
            expected = expected_segments[segment["staging_key"]]
            if segment.get("matched") is not True or segment.get(
                "saved_delta_sha256"
            ) != segment.get("canonical_slice_sha256"):
                raise AssertionError(
                    "saved pre-cut token segment differs from the final canonical row: "
                    f"rollout={rollout_id!r}, segment={segment!r}"
                )
            if (
                segment.get("selection_staging_digest")
                != expected.get("staging_digest")
                or segment.get("restored_staging_digest")
                != expected.get("staging_digest")
                or segment.get("prev_len") != expected.get("prev_len")
                or segment.get("cum_len") != expected.get("cum_len")
            ):
                raise AssertionError(
                    "token evidence differs from the phase-one checkpoint: "
                    f"rollout={rollout_id!r}, segment={segment!r}"
                )
