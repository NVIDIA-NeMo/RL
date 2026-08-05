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

"""Pure construction of inspectable NeMo-Gym physical rollout traces.

A logical rollout is the environment episode. Each model call records the
exact prompt, sampled tokens, sampled log-probabilities, and ordered media IDs.
Whenever the next prompt is not an extension of the preceding
prompt-plus-completion, context was materially rewritten and a new physical
trace starts.

This module only plans and validates traces. It does not expand a rollout into
training rows, duplicate advantages, or invoke a trainer.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import math
from typing import Any, Mapping, Sequence

from nemo_rl.environments.generation_contract import (
    validate_training_admission_contract,
)


def _as_list(value: Any, *, field: str) -> list[Any]:
    if not isinstance(value, (list, tuple)):
        raise TypeError(f"{field} must be a list or tuple, got {type(value)!r}")
    return list(value)


def _boundary_by_turn(
    boundary_events: Sequence[Mapping[str, Any]],
) -> dict[int, Mapping[str, Any]]:
    result: dict[int, Mapping[str, Any]] = {}
    for boundary in boundary_events:
        turn = boundary.get("applies_to_step")
        if not isinstance(turn, int):
            raise ValueError(f"Boundary has invalid applies_to_step: {boundary!r}")
        if turn in result:
            raise ValueError(f"Multiple rewrite boundaries apply to turn {turn}")
        result[turn] = boundary
    return result


def _validate_policy_lineage(
    policy_decision: Mapping[str, Any],
    *,
    completion_id: str,
) -> None:
    lineage = policy_decision.get("lineage")
    if not isinstance(lineage, Mapping):
        raise ValueError(
            f"Completion {completion_id!r} is missing transformation lineage"
        )
    if lineage.get("validator_result") != "passed":
        raise ValueError(f"Completion {completion_id!r} lineage was not validated")
    records = _as_list(lineage.get("unit_records"), field="unit_records")
    expected_count = policy_decision.get("retained_part_count", 0) + (
        policy_decision.get("omitted_part_count", 0)
    )
    if len(records) != expected_count:
        raise ValueError(
            f"Completion {completion_id!r} lineage does not account for every "
            "semantic unit"
        )
    seen_source_ids: set[str] = set()
    for record in records:
        if not isinstance(record, Mapping):
            raise TypeError("Lineage unit records must be mappings")
        source_id = record.get("source_unit_id")
        if not isinstance(source_id, str) or not source_id:
            raise ValueError("Lineage unit record has no source identity")
        if source_id in seen_source_ids:
            raise ValueError(f"Duplicate lineage source unit {source_id!r}")
        seen_source_ids.add(source_id)
        output_ids = _as_list(
            record.get("output_unit_ids"),
            field="output_unit_ids",
        )
        output_digests = _as_list(
            record.get("output_digests"),
            field="output_digests",
        )
        if len(output_ids) != len(output_digests):
            raise ValueError(
                f"Lineage source unit {source_id!r} has misaligned outputs"
            )
        disposition = record.get("disposition")
        if disposition == "dropped" and output_ids:
            raise ValueError(f"Dropped lineage source unit {source_id!r} has outputs")
        if disposition != "dropped" and not output_ids:
            raise ValueError(
                f"Retained/transformed lineage source unit {source_id!r} has no output"
            )


def _normalized_lineage_unit(record: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "source_unit_id": record.get("source_unit_id"),
        "source_digest": record.get("source_digest"),
        "disposition": record.get("disposition"),
        "output_unit_ids": _as_list(
            record.get("output_unit_ids"),
            field="output_unit_ids",
        ),
        "output_digests": _as_list(
            record.get("output_digests"),
            field="output_digests",
        ),
    }


def _lineage_state_digest(
    records: Mapping[str, Mapping[str, Any]],
) -> str:
    payload = json.dumps(
        [_normalized_lineage_unit(record) for _, record in sorted(records.items())],
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _validate_lineage_deltas(
    lineage_deltas: Sequence[Mapping[str, Any]],
    *,
    calls: Sequence[Mapping[str, Any]],
    final_policy_decision: Mapping[str, Any],
) -> None:
    if len(lineage_deltas) != len(calls):
        raise ValueError("Lineage delta count does not match authoritative model calls")
    state: dict[str, Mapping[str, Any]] = {}
    parent_transformation_id: str | None = None
    for call, delta in zip(calls, lineage_deltas, strict=True):
        if not isinstance(delta, Mapping):
            raise TypeError("Lineage deltas must be mappings")
        if delta.get("validator_result") != "passed":
            raise ValueError("Lineage delta was not validated")
        if delta.get("parent_transformation_id") != parent_transformation_id:
            raise ValueError("Lineage delta parent chain is corrupted")
        decision = call.get("policy_decision")
        if not isinstance(decision, Mapping) or delta.get(
            "transformation_id"
        ) != decision.get("transformation_id"):
            raise ValueError(
                "Lineage delta does not match its model-call policy reference"
            )
        upserts = _as_list(delta.get("unit_upserts"), field="unit_upserts")
        seen_upserts: set[str] = set()
        for raw_record in upserts:
            if not isinstance(raw_record, Mapping):
                raise TypeError("Lineage delta upserts must be mappings")
            record = _normalized_lineage_unit(raw_record)
            source_unit_id = record["source_unit_id"]
            if not isinstance(source_unit_id, str) or not source_unit_id:
                raise ValueError("Lineage delta has an invalid source unit ID")
            if source_unit_id in seen_upserts:
                raise ValueError(
                    f"Lineage delta repeats source unit {source_unit_id!r}"
                )
            seen_upserts.add(source_unit_id)
            state[source_unit_id] = record
        if delta.get("source_unit_count") != len(state):
            raise ValueError("Lineage delta source-unit count is corrupted")
        if delta.get("state_digest") != _lineage_state_digest(state):
            raise ValueError("Lineage delta state digest is corrupted")
        parent_transformation_id = str(delta.get("transformation_id"))

    final_lineage = final_policy_decision.get("lineage")
    assert isinstance(final_lineage, Mapping)
    final_records = {
        str(record["source_unit_id"]): _normalized_lineage_unit(record)
        for record in _as_list(
            final_lineage.get("unit_records"),
            field="unit_records",
        )
    }
    if state != final_records:
        raise ValueError(
            "Lineage delta reconstruction disagrees with the final manifest"
        )


def build_rollout_trace_bundle(
    *,
    rollout_id: str,
    calls: Sequence[Mapping[str, Any]],
    boundary_events: Sequence[Mapping[str, Any]] = (),
    policy_name: str | None = None,
    group_id: str | None = None,
    source_row_index: int | None = None,
    reward: float | None = None,
    media_assets: Mapping[str, Any] | None = None,
    generation_contract: Mapping[str, Any] | None = None,
    training_admission: Mapping[str, Any] | None = None,
    final_policy_decision: Mapping[str, Any] | None = None,
    lineage_deltas: Sequence[Mapping[str, Any]] | None = None,
    strict: bool = False,
) -> dict[str, Any]:
    """Partition exact model calls into prefix-contiguous physical traces."""
    if not rollout_id:
        raise ValueError("rollout_id must be non-empty")

    boundaries = _boundary_by_turn(boundary_events)
    model_calls: list[dict[str, Any]] = []
    physical_traces: list[dict[str, Any]] = []
    previous_context: list[int] = []
    previous_media_ids: list[str] = []
    current_trace: dict[str, Any] | None = None
    seen_completion_ids: set[str] = set()
    seen_action_ids: set[str] = set()
    seen_turn_ids: set[int] = set()
    source_rollout_ids: set[str] = set()
    policy_identities: set[tuple[str, str | None, str | None]] = set()
    used_boundary_turns: set[int] = set()
    total_sampled_tokens = 0
    emitted_sampled_tokens = 0
    total_trainable_tokens = 0
    all_logprobs_aligned = True
    prefixes_valid_inside_traces = True
    media_order_valid = True
    rewrites_covered = True

    for call_index, raw_call in enumerate(calls):
        turn_id = raw_call.get("turn_id", call_index + 1)
        if not isinstance(turn_id, int) or turn_id < 1:
            raise ValueError(f"Invalid turn_id {turn_id!r}")
        if turn_id in seen_turn_ids:
            raise ValueError(f"Duplicate turn_id {turn_id}")
        if strict and turn_id != call_index + 1:
            raise ValueError(
                "Exact-trace authority requires consecutive turn IDs: "
                f"call_index={call_index}, turn_id={turn_id}"
            )
        seen_turn_ids.add(turn_id)

        prompt_token_ids = [
            int(value)
            for value in _as_list(
                raw_call.get("prompt_token_ids"), field="prompt_token_ids"
            )
        ]
        sampled_token_ids = [
            int(value)
            for value in _as_list(
                raw_call.get("sampled_token_ids"), field="sampled_token_ids"
            )
        ]
        sampled_logprobs = [
            float(value)
            for value in _as_list(
                raw_call.get("sampled_logprobs"), field="sampled_logprobs"
            )
        ]
        if any(not math.isfinite(value) for value in sampled_logprobs):
            raise ValueError(
                f"Completion at turn {turn_id} contains a non-finite logprob"
            )
        if strict and not sampled_token_ids:
            raise ValueError(f"Completion at turn {turn_id} has no sampled tokens")
        media_ids = [
            str(value)
            for value in _as_list(raw_call.get("media_ids", ()), field="media_ids")
        ]
        completion_id = str(
            raw_call.get("completion_id") or f"completion-{turn_id:06d}"
        )
        if completion_id in seen_completion_ids:
            raise ValueError(f"Duplicate completion ID {completion_id!r}")
        seen_completion_ids.add(completion_id)
        action_id = raw_call.get("action_id")
        if strict and not action_id:
            raise ValueError(f"Completion {completion_id!r} is missing an action_id")
        if action_id is not None:
            action_id = str(action_id)
            if action_id in seen_action_ids:
                raise ValueError(f"Duplicate action ID {action_id!r}")
            seen_action_ids.add(action_id)
        eligible = raw_call.get("eligible", True)
        if not isinstance(eligible, bool):
            raise ValueError(
                f"Completion {completion_id!r} has non-boolean eligible={eligible!r}"
            )
        source_rollout_id = raw_call.get("rollout_id")
        if source_rollout_id is not None:
            source_rollout_ids.add(str(source_rollout_id))
        elif strict:
            raise ValueError(
                f"Completion {completion_id!r} is missing its source rollout_id"
            )
        policy_decision = raw_call.get("policy_decision")
        if strict and not isinstance(policy_decision, Mapping):
            raise ValueError(f"Completion {completion_id!r} is missing policy_decision")
        if isinstance(policy_decision, Mapping):
            policy_identity = (
                str(policy_decision.get("policy_name")),
                (
                    str(policy_decision["policy_version"])
                    if policy_decision.get("policy_version") is not None
                    else None
                ),
                (
                    str(policy_decision["config_digest"])
                    if policy_decision.get("config_digest") is not None
                    else None
                ),
            )
            policy_identities.add(policy_identity)

        aligned = len(sampled_token_ids) == len(sampled_logprobs)
        all_logprobs_aligned &= aligned
        if not aligned:
            raise ValueError(
                f"Completion {completion_id!r} token/logprob mismatch: "
                f"tokens={len(sampled_token_ids)} "
                f"logprobs={len(sampled_logprobs)}"
            )

        token_contiguous = previous_context == prompt_token_ids[: len(previous_context)]
        media_contiguous = previous_media_ids == media_ids[: len(previous_media_ids)]
        append_compatible = token_contiguous and media_contiguous
        starts_trace = current_trace is None or not append_compatible
        expected_append_compatible = raw_call.get("expected_append_compatible")
        if strict:
            if not isinstance(expected_append_compatible, bool):
                raise ValueError(
                    f"Completion {completion_id!r} is missing its declared "
                    "append-compatibility"
                )
            if expected_append_compatible != (
                current_trace is not None and append_compatible
            ):
                raise ValueError(
                    "Gym append-compatibility declaration disagrees with "
                    f"token/media evidence at turn {turn_id}"
                )
        boundary = boundaries.get(turn_id)
        if starts_trace and current_trace is not None and boundary is None:
            rewrites_covered = False
            if strict:
                raise ValueError(
                    f"Material rewrite before turn {turn_id} has no boundary record"
                )
        if boundary is not None:
            if current_trace is None or not starts_trace:
                if strict:
                    raise ValueError(
                        f"Boundary for turn {turn_id} does not correspond to a rewrite"
                    )
            else:
                used_boundary_turns.add(turn_id)
                if strict and isinstance(policy_decision, Mapping):
                    for field in ("policy_name", "policy_version", "config_digest"):
                        if boundary.get(field) != policy_decision.get(field):
                            raise ValueError(
                                f"Boundary {field} disagrees with completion "
                                f"evidence at turn {turn_id}"
                            )

        context_epoch = raw_call.get("context_epoch")
        segment_index = raw_call.get("segment_index")
        segment_id = raw_call.get("segment_id")
        compaction_event_id = raw_call.get("compaction_event_id")
        generation_contract_id = raw_call.get("generation_contract_id")
        policy_output_spans = raw_call.get("policy_output_spans")
        media_occurrences = raw_call.get("media_occurrences")
        if strict:
            if (
                not isinstance(context_epoch, int)
                or context_epoch < 0
                or not isinstance(segment_index, int)
                or segment_index < 0
                or not isinstance(segment_id, str)
                or not segment_id
            ):
                raise ValueError(
                    f"Completion {completion_id!r} has invalid segment identity"
                )
            if segment_index != len(physical_traces) - int(not starts_trace):
                raise ValueError(
                    f"Completion {completion_id!r} has a non-consecutive segment_index"
                )
            if context_epoch != segment_index:
                raise ValueError(
                    f"Completion {completion_id!r} context_epoch and "
                    "segment_index disagree"
                )
            if boundary is None and compaction_event_id is not None:
                raise ValueError(
                    f"Completion {completion_id!r} declares an unexpected "
                    "compaction event"
                )
            if boundary is not None and compaction_event_id != boundary.get("event_id"):
                raise ValueError(
                    f"Completion {completion_id!r} compaction event does not "
                    "match its rewrite boundary"
                )
            if generation_contract is None:
                raise ValueError("Exact-trace authority requires a generation contract")
            if generation_contract_id != generation_contract.get(
                "generation_contract_id"
            ):
                raise ValueError(
                    f"Completion {completion_id!r} generation contract ID "
                    "does not match the rollout contract"
                )
            if (
                not isinstance(policy_output_spans, (list, tuple))
                or len(policy_output_spans) != 1
            ):
                raise ValueError(
                    f"Completion {completion_id!r} must declare one complete "
                    "policy-output span"
                )
            span = policy_output_spans[0]
            if (
                not isinstance(span, Mapping)
                or span.get("start") != 0
                or span.get("end") != len(sampled_token_ids)
                or span.get("action_ids") != [action_id]
                or span.get("eligible") != eligible
            ):
                raise ValueError(
                    f"Completion {completion_id!r} has an invalid policy-output span"
                )
            if (
                not isinstance(media_occurrences, (list, tuple))
                or [
                    occurrence.get("media_id")
                    for occurrence in media_occurrences
                    if isinstance(occurrence, Mapping)
                ]
                != media_ids
            ):
                raise ValueError(
                    f"Completion {completion_id!r} media occurrence sidecar "
                    "does not preserve ordered media IDs"
                )

        if starts_trace:
            trace_index = len(physical_traces)
            trace_id = f"{rollout_id}:trace-{trace_index:06d}"
            current_trace = {
                "trace_id": trace_id,
                "trace_index": trace_index,
                "segment_id": segment_id,
                "segment_index": segment_index,
                "context_epoch": context_epoch,
                "boundary_before": deepcopy(dict(boundary)) if boundary else None,
                "source_turn_ids": [],
                "segments": [],
                "completion_spans": [],
                "ordered_media_ids": [],
                "token_count": 0,
                "trainable_token_count": 0,
            }
            physical_traces.append(current_trace)
            prompt_delta = prompt_token_ids
            new_media_ids = media_ids
        else:
            assert current_trace is not None
            prompt_delta = prompt_token_ids[len(previous_context) :]
            if strict and (
                segment_id != current_trace["segment_id"]
                or segment_index != current_trace["segment_index"]
                or context_epoch != current_trace["context_epoch"]
            ):
                raise ValueError(
                    f"Append-compatible turn {turn_id} changed segment identity"
                )
            new_media_ids = media_ids[len(previous_media_ids) :]

        assert current_trace is not None
        trace_id = current_trace["trace_id"]
        prompt_start = current_trace["token_count"]
        prompt_end = prompt_start + len(prompt_delta)
        completion_start = prompt_end
        completion_end = completion_start + len(sampled_token_ids)

        current_trace["segments"].append(
            {
                "kind": "prompt",
                "turn_id": turn_id,
                "token_ids": prompt_delta,
                "loss_mask": [0] * len(prompt_delta),
                "media_ids": new_media_ids,
            }
        )
        current_trace["segments"].append(
            {
                "kind": "completion",
                "turn_id": turn_id,
                "completion_id": completion_id,
                "token_ids": sampled_token_ids,
                "loss_mask": [int(eligible)] * len(sampled_token_ids),
                "generation_logprobs": sampled_logprobs,
                "eligible": eligible,
            }
        )
        current_trace["completion_spans"].append(
            {
                "turn_id": turn_id,
                "completion_id": completion_id,
                "start": completion_start,
                "end": completion_end,
                "eligible": eligible,
            }
        )
        current_trace["source_turn_ids"].append(turn_id)
        current_trace["ordered_media_ids"].extend(new_media_ids)
        current_trace["token_count"] = completion_end
        current_trace["trainable_token_count"] += (
            len(sampled_token_ids) if eligible else 0
        )

        call_record = {
            "call_index": call_index,
            "turn_id": turn_id,
            "completion_id": completion_id,
            "action_id": action_id,
            "trace_id": trace_id,
            "starts_physical_trace": starts_trace,
            "token_append_compatible": token_contiguous,
            "media_append_compatible": media_contiguous,
            "expected_append_compatible": expected_append_compatible,
            "prepared_request_id": raw_call.get("prepared_request_id"),
            "request_id": raw_call.get("request_id"),
            "context_epoch": context_epoch,
            "segment_index": segment_index,
            "segment_id": segment_id,
            "compaction_event_id": compaction_event_id,
            "boundary_before_id": (
                boundary.get("event_id") if boundary is not None else None
            ),
            "prompt_token_ids": prompt_token_ids,
            "prompt_delta_token_ids": prompt_delta,
            "sampled_token_ids": sampled_token_ids,
            "sampled_logprobs": sampled_logprobs,
            "media_ids": media_ids,
            "new_media_ids": new_media_ids,
            "finish_reason": raw_call.get("finish_reason"),
            "policy_decision": (
                deepcopy(dict(policy_decision))
                if isinstance(policy_decision, Mapping)
                else None
            ),
            "processor_fingerprint": raw_call.get("processor_fingerprint"),
            "generation_contract_id": generation_contract_id,
            "policy_output_spans": deepcopy(policy_output_spans),
            "media_occurrences": deepcopy(media_occurrences),
            "eligible": eligible,
            "evidence_source": raw_call.get("evidence_source"),
            "source_rollout_id": (
                str(source_rollout_id) if source_rollout_id is not None else None
            ),
        }
        model_calls.append(call_record)

        total_sampled_tokens += len(sampled_token_ids)
        if eligible:
            total_trainable_tokens += len(sampled_token_ids)
        emitted_sampled_tokens += len(current_trace["segments"][-1]["token_ids"])
        previous_context = [*prompt_token_ids, *sampled_token_ids]
        previous_media_ids = media_ids

    if strict:
        if not isinstance(final_policy_decision, Mapping):
            raise ValueError(
                "Exact-trace authority requires one rollout-level final "
                "policy decision with complete lineage"
            )
        _validate_policy_lineage(
            final_policy_decision,
            completion_id="rollout-final-policy-decision",
        )
        if lineage_deltas is None:
            raise ValueError(
                "Exact-trace authority requires structurally shared lineage deltas"
            )
        _validate_lineage_deltas(
            lineage_deltas,
            calls=model_calls,
            final_policy_decision=final_policy_decision,
        )
        final_identity = (
            final_policy_decision.get("policy_name"),
            final_policy_decision.get("policy_version"),
            final_policy_decision.get("config_digest"),
        )
        if policy_identities != {final_identity}:
            raise ValueError(
                "Per-call policy references disagree with the final lineage manifest"
            )
        if len(source_rollout_ids) != 1 or source_rollout_ids != {rollout_id}:
            raise ValueError(
                "Completion evidence must use the authoritative rollout ID: "
                f"expected={rollout_id!r}, observed={sorted(source_rollout_ids)!r}"
            )
        if len(policy_identities) != 1:
            raise ValueError(
                "Completion evidence contains mixed policy identities: "
                f"{sorted(policy_identities)!r}"
            )
        if policy_name is not None and next(iter(policy_identities))[0] != policy_name:
            raise ValueError(
                "Bundle policy_name disagrees with completion evidence: "
                f"bundle={policy_name!r}, evidence={next(iter(policy_identities))[0]!r}"
            )
        unused_boundary_turns = set(boundaries) - used_boundary_turns
        if unused_boundary_turns:
            raise ValueError(
                "Boundary records do not correspond to material rewrites at turns "
                f"{sorted(unused_boundary_turns)}"
            )
        if media_assets is None:
            raise ValueError(
                "Exact-trace authority requires the rollout media asset arena"
            )
        unknown_media_ids = {
            media_id
            for call in model_calls
            for media_id in call["media_ids"]
            if media_id not in media_assets
        }
        if unknown_media_ids:
            raise ValueError(
                "Completion evidence references unknown media IDs: "
                f"{sorted(unknown_media_ids)!r}"
            )

    for trace in physical_traces:
        reconstructed: list[int] = []
        expected_position = 0
        for segment in trace["segments"]:
            reconstructed.extend(segment["token_ids"])
        for span in trace["completion_spans"]:
            if span["start"] < expected_position or span["end"] > len(reconstructed):
                prefixes_valid_inside_traces = False
            expected_position = span["end"]

    checks = {
        "all_completion_ids_unique": len(seen_completion_ids) == len(calls),
        "all_completions_present_once": (
            emitted_sampled_tokens == total_sampled_tokens
        ),
        "all_logprobs_aligned": all_logprobs_aligned,
        "prefixes_valid_inside_traces": prefixes_valid_inside_traces,
        "media_order_valid": media_order_valid,
        "boundary_records_cover_rewrites": rewrites_covered,
        "model_call_count": len(model_calls),
        "physical_trace_count": len(physical_traces),
        "trainable_token_count": total_sampled_tokens,
        "eligible_trainable_token_count": total_trainable_tokens,
    }
    checks["ok"] = all(
        value
        for key, value in checks.items()
        if key
        in {
            "all_completion_ids_unique",
            "all_completions_present_once",
            "all_logprobs_aligned",
            "prefixes_valid_inside_traces",
            "media_order_valid",
            "boundary_records_cover_rewrites",
        }
    )

    bundle = {
        "schema_version": 3,
        "rollout_id": rollout_id,
        "group_id": group_id,
        "source_row_index": source_row_index,
        "reward": reward,
        "policy_name": policy_name,
        "generation_contract": (
            deepcopy(dict(generation_contract))
            if generation_contract is not None
            else None
        ),
        "training_admission": (
            deepcopy(dict(training_admission))
            if training_admission is not None
            else None
        ),
        "final_policy_decision": (
            deepcopy(dict(final_policy_decision))
            if final_policy_decision is not None
            else None
        ),
        "lineage_deltas": (
            [deepcopy(dict(delta)) for delta in lineage_deltas]
            if lineage_deltas is not None
            else []
        ),
        "model_calls": model_calls,
        "physical_traces": physical_traces,
        "checks": checks,
    }
    if strict:
        validate_rollout_trace_bundle(
            bundle,
            media_assets=media_assets,
            strict=True,
        )
    return bundle


def validate_rollout_trace_bundle(
    bundle: Mapping[str, Any],
    *,
    media_assets: Mapping[str, Any] | None = None,
    strict: bool = True,
) -> dict[str, Any]:
    """Independently reconstruct and validate a serialized trace bundle.

    This deliberately walks the emitted physical traces rather than reusing
    the builder's counters. It is suitable for validating persisted JSON and
    for corruption tests.
    """
    if bundle.get("schema_version") != 3:
        raise ValueError(
            f"Unsupported trace schema_version={bundle.get('schema_version')!r}"
        )
    calls = _as_list(bundle.get("model_calls"), field="model_calls")
    traces = _as_list(bundle.get("physical_traces"), field="physical_traces")
    if strict and not calls:
        raise ValueError("Exact-trace bundle contains no model calls")
    if not isinstance(bundle.get("rollout_id"), str) or not bundle["rollout_id"]:
        raise ValueError("Trace bundle rollout_id must be non-empty")

    calls_by_trace: dict[str, list[Mapping[str, Any]]] = {}
    seen_completion_ids: set[str] = set()
    seen_action_ids: set[str] = set()
    all_media_ids: list[str] = []
    previous_context: list[int] = []
    previous_media: list[str] = []
    previous_trace_id: str | None = None
    previous_segment_id: str | None = None
    expected_segment_index = -1
    generation_contract = bundle.get("generation_contract")
    if strict and not isinstance(generation_contract, Mapping):
        raise ValueError("Exact-trace bundle has no generation contract")
    training_admission = bundle.get("training_admission")
    if training_admission is not None:
        if not isinstance(training_admission, Mapping):
            raise TypeError("Trace bundle training admission must be a mapping")
        if not isinstance(generation_contract, Mapping):
            raise ValueError(
                "Trace bundle training admission has no Gym generation contract"
            )
        validate_training_admission_contract(
            training_admission,
            generation_contract,
        )
    final_policy_decision = bundle.get("final_policy_decision")
    if strict:
        if not isinstance(final_policy_decision, Mapping):
            raise ValueError("Exact-trace bundle has no final policy lineage manifest")
        _validate_policy_lineage(
            final_policy_decision,
            completion_id="rollout-final-policy-decision",
        )
        serialized_lineage_deltas = _as_list(
            bundle.get("lineage_deltas"),
            field="lineage_deltas",
        )
        _validate_lineage_deltas(
            serialized_lineage_deltas,
            calls=calls,
            final_policy_decision=final_policy_decision,
        )
        if bundle.get("policy_name") != final_policy_decision.get("policy_name"):
            raise ValueError(
                "Trace bundle policy_name disagrees with final policy lineage"
            )
        generation_contract_id = generation_contract.get("generation_contract_id")
        if not isinstance(generation_contract_id, str) or not generation_contract_id:
            raise ValueError("Exact-trace bundle generation contract has no identity")
    for call_index, call in enumerate(calls):
        if not isinstance(call, Mapping):
            raise TypeError(f"model_calls[{call_index}] must be a mapping")
        if call.get("call_index") != call_index:
            raise ValueError(f"model_calls[{call_index}] has an invalid call_index")
        if strict and call.get("turn_id") != call_index + 1:
            raise ValueError("Model calls are not in consecutive turn order")
        completion_id = call.get("completion_id")
        if completion_id in seen_completion_ids:
            raise ValueError(f"Duplicate completion ID {completion_id!r}")
        seen_completion_ids.add(completion_id)
        if strict:
            action_id = call.get("action_id")
            if not isinstance(action_id, str) or not action_id:
                raise ValueError(f"Completion {completion_id!r} has no action identity")
            if action_id in seen_action_ids:
                raise ValueError(f"Duplicate action ID {action_id!r}")
            seen_action_ids.add(action_id)
            if call.get("source_rollout_id") != bundle["rollout_id"]:
                raise ValueError(
                    f"Completion {completion_id!r} source rollout is corrupted"
                )
        sampled = _as_list(call.get("sampled_token_ids"), field="sampled_token_ids")
        logprobs = _as_list(call.get("sampled_logprobs"), field="sampled_logprobs")
        if len(sampled) != len(logprobs):
            raise ValueError(f"Completion {completion_id!r} token/logprob mismatch")
        if any(not math.isfinite(float(value)) for value in logprobs):
            raise ValueError(
                f"Completion {completion_id!r} contains a non-finite logprob"
            )
        trace_id = call.get("trace_id")
        if not isinstance(trace_id, str):
            raise ValueError(f"Completion {completion_id!r} has no trace_id")
        calls_by_trace.setdefault(trace_id, []).append(call)
        prompt = _as_list(call.get("prompt_token_ids"), field="prompt_token_ids")
        media = [
            str(value) for value in _as_list(call.get("media_ids"), field="media_ids")
        ]
        all_media_ids.extend(media)
        token_append_compatible = previous_context == prompt[: len(previous_context)]
        media_append_compatible = previous_media == media[: len(previous_media)]
        expected_starts_trace = call_index == 0 or not (
            token_append_compatible and media_append_compatible
        )
        if call.get("starts_physical_trace") != expected_starts_trace:
            raise ValueError(
                f"Completion {completion_id!r} physical-trace boundary "
                "disagrees with token/media continuity"
            )
        if call.get("token_append_compatible") != token_append_compatible:
            raise ValueError(
                f"Completion {completion_id!r} token continuity is corrupted"
            )
        if call.get("media_append_compatible") != media_append_compatible:
            raise ValueError(
                f"Completion {completion_id!r} media continuity is corrupted"
            )
        if expected_starts_trace:
            expected_segment_index += 1
        segment_index = call.get("segment_index")
        context_epoch = call.get("context_epoch")
        segment_id = call.get("segment_id")
        if strict:
            if (
                segment_index != expected_segment_index
                or context_epoch != expected_segment_index
                or not isinstance(segment_id, str)
                or not segment_id
            ):
                raise ValueError(
                    f"Completion {completion_id!r} segment identity is corrupted"
                )
            if not expected_starts_trace and segment_id != previous_segment_id:
                raise ValueError(
                    f"Completion {completion_id!r} changed segment identity "
                    "inside a physical trace"
                )
        if strict and call.get("expected_append_compatible") != (
            call_index > 0 and token_append_compatible and media_append_compatible
        ):
            raise ValueError(
                f"Completion {completion_id!r} append declaration is corrupted"
            )
        if strict:
            decision = call.get("policy_decision")
            if not isinstance(decision, Mapping):
                raise ValueError(
                    f"Completion {completion_id!r} policy decision is corrupted"
                )
            for field in ("policy_name", "policy_version", "config_digest"):
                if decision.get(field) != final_policy_decision.get(field):
                    raise ValueError(
                        f"Completion {completion_id!r} policy reference "
                        "disagrees with final lineage"
                    )
            if call.get("generation_contract_id") != generation_contract.get(
                "generation_contract_id"
            ):
                raise ValueError(
                    f"Completion {completion_id!r} generation contract is corrupted"
                )
            spans = _as_list(
                call.get("policy_output_spans"),
                field="policy_output_spans",
            )
            if len(spans) != 1:
                raise ValueError(
                    f"Completion {completion_id!r} policy-output spans are corrupted"
                )
            span = spans[0]
            if (
                not isinstance(span, Mapping)
                or span.get("start") != 0
                or span.get("end") != len(sampled)
                or span.get("action_ids") != [call.get("action_id")]
                or span.get("eligible") != call.get("eligible", True)
            ):
                raise ValueError(
                    f"Completion {completion_id!r} policy-output span is corrupted"
                )
            occurrences = _as_list(
                call.get("media_occurrences"),
                field="media_occurrences",
            )
            if [
                occurrence.get("media_id")
                for occurrence in occurrences
                if isinstance(occurrence, Mapping)
            ] != media:
                raise ValueError(
                    f"Completion {completion_id!r} media occurrences are corrupted"
                )
        if expected_starts_trace:
            if previous_trace_id == trace_id:
                raise ValueError(
                    f"Completion {completion_id!r} did not change trace ID"
                )
        elif previous_trace_id != trace_id:
            raise ValueError(
                f"Completion {completion_id!r} changed trace ID inside a segment"
            )
        previous_context = [*prompt, *sampled]
        previous_media = media
        previous_trace_id = trace_id
        previous_segment_id = segment_id

    emitted_completion_ids: list[str] = []
    emitted_sampled_tokens = 0
    emitted_trainable_tokens = 0
    for trace_index, trace in enumerate(traces):
        if not isinstance(trace, Mapping):
            raise TypeError(f"physical_traces[{trace_index}] must be a mapping")
        expected_trace_id = f"{bundle['rollout_id']}:trace-{trace_index:06d}"
        if trace.get("trace_id") != expected_trace_id:
            raise ValueError(f"physical_traces[{trace_index}] has an invalid trace_id")
        if trace.get("trace_index") != trace_index:
            raise ValueError(
                f"physical_traces[{trace_index}] has an invalid trace_index"
            )
        trace_calls = calls_by_trace.pop(expected_trace_id, [])
        if strict and not trace_calls:
            raise ValueError(
                f"Physical trace {expected_trace_id!r} contains no model calls"
            )
        if trace_calls:
            first_call = trace_calls[0]
            for field in ("segment_id", "segment_index", "context_epoch"):
                if trace.get(field) != first_call.get(field):
                    raise ValueError(
                        f"Trace {expected_trace_id!r} {field} is corrupted"
                    )
        source_turn_ids = _as_list(
            trace.get("source_turn_ids"), field="source_turn_ids"
        )
        if source_turn_ids != [call["turn_id"] for call in trace_calls]:
            raise ValueError(f"Trace {expected_trace_id!r} turn IDs are corrupted")
        segments = _as_list(trace.get("segments"), field="segments")
        spans = _as_list(trace.get("completion_spans"), field="completion_spans")
        if len(segments) != 2 * len(trace_calls) or len(spans) != len(trace_calls):
            raise ValueError(
                f"Trace {expected_trace_id!r} has inconsistent segment/span counts"
            )

        reconstructed: list[int] = []
        ordered_media_ids: list[str] = []
        trainable_tokens = 0
        previous_context: list[int] = []
        previous_media: list[str] = []
        for local_index, call in enumerate(trace_calls):
            prompt_segment = segments[2 * local_index]
            completion_segment = segments[2 * local_index + 1]
            if prompt_segment.get("kind") != "prompt":
                raise ValueError(
                    f"Trace {expected_trace_id!r} prompt segment is corrupted"
                )
            if completion_segment.get("kind") != "completion":
                raise ValueError(
                    f"Trace {expected_trace_id!r} completion segment is corrupted"
                )
            prompt = _as_list(call.get("prompt_token_ids"), field="prompt_token_ids")
            media = _as_list(call.get("media_ids"), field="media_ids")
            if local_index == 0:
                if call.get("starts_physical_trace") is not True:
                    raise ValueError(
                        f"Trace {expected_trace_id!r} does not start at its first call"
                    )
            else:
                if call.get("starts_physical_trace") is not False:
                    raise ValueError(
                        f"Trace {expected_trace_id!r} starts in its interior"
                    )
                if prompt[: len(previous_context)] != previous_context:
                    raise ValueError(
                        f"Trace {expected_trace_id!r} prompt prefix is corrupted"
                    )
                if media[: len(previous_media)] != previous_media:
                    raise ValueError(
                        f"Trace {expected_trace_id!r} media prefix is corrupted"
                    )
            expected_prompt_delta = (
                prompt if local_index == 0 else prompt[len(previous_context) :]
            )
            expected_media_delta = (
                media if local_index == 0 else media[len(previous_media) :]
            )
            if prompt_segment.get("token_ids") != expected_prompt_delta:
                raise ValueError(
                    f"Trace {expected_trace_id!r} prompt delta is corrupted"
                )
            if prompt_segment.get("loss_mask") != [0] * len(expected_prompt_delta):
                raise ValueError(
                    f"Trace {expected_trace_id!r} prompt loss mask is corrupted"
                )
            if prompt_segment.get("media_ids") != expected_media_delta:
                raise ValueError(
                    f"Trace {expected_trace_id!r} media delta is corrupted"
                )
            completion_tokens = _as_list(
                call.get("sampled_token_ids"), field="sampled_token_ids"
            )
            eligible = bool(call.get("eligible", True))
            if completion_segment.get("completion_id") != call.get("completion_id"):
                raise ValueError(
                    f"Trace {expected_trace_id!r} completion ID is corrupted"
                )
            if completion_segment.get("token_ids") != completion_tokens:
                raise ValueError(
                    f"Trace {expected_trace_id!r} completion tokens are corrupted"
                )
            if completion_segment.get("loss_mask") != [int(eligible)] * len(
                completion_tokens
            ):
                raise ValueError(
                    f"Trace {expected_trace_id!r} completion loss mask is corrupted"
                )
            if completion_segment.get("generation_logprobs") != call.get(
                "sampled_logprobs"
            ):
                raise ValueError(
                    f"Trace {expected_trace_id!r} completion logprobs are corrupted"
                )

            completion_start = len(reconstructed) + len(expected_prompt_delta)
            reconstructed.extend(expected_prompt_delta)
            reconstructed.extend(completion_tokens)
            expected_span = {
                "turn_id": call["turn_id"],
                "completion_id": call["completion_id"],
                "start": completion_start,
                "end": completion_start + len(completion_tokens),
                "eligible": eligible,
            }
            if spans[local_index] != expected_span:
                raise ValueError(
                    f"Trace {expected_trace_id!r} completion span is corrupted"
                )
            ordered_media_ids.extend(expected_media_delta)
            emitted_completion_ids.append(str(call["completion_id"]))
            emitted_sampled_tokens += len(completion_tokens)
            if eligible:
                trainable_tokens += len(completion_tokens)
                emitted_trainable_tokens += len(completion_tokens)
            previous_context = [*prompt, *completion_tokens]
            previous_media = media

        if trace.get("token_count") != len(reconstructed):
            raise ValueError(f"Trace {expected_trace_id!r} token_count is corrupted")
        if trace.get("trainable_token_count") != trainable_tokens:
            raise ValueError(
                f"Trace {expected_trace_id!r} trainable_token_count is corrupted"
            )
        if trace.get("ordered_media_ids") != ordered_media_ids:
            raise ValueError(f"Trace {expected_trace_id!r} ordered media is corrupted")
        boundary = trace.get("boundary_before")
        if trace_index == 0:
            if boundary is not None:
                raise ValueError("The first physical trace cannot have a boundary")
            if strict:
                first_call = trace_calls[0]
                if (
                    first_call.get("boundary_before_id") is not None
                    or first_call.get("compaction_event_id") is not None
                ):
                    raise ValueError(
                        "The first physical trace has an unexpected boundary identity"
                    )
        elif strict:
            if not isinstance(boundary, Mapping):
                raise ValueError(
                    f"Trace {expected_trace_id!r} is missing its rewrite boundary"
                )
            first_call = trace_calls[0]
            if boundary.get("applies_to_step") != first_call.get("turn_id"):
                raise ValueError(
                    f"Trace {expected_trace_id!r} boundary turn is corrupted"
                )
            if first_call.get("boundary_before_id") != boundary.get("event_id"):
                raise ValueError(
                    f"Trace {expected_trace_id!r} boundary ID is corrupted"
                )
            if first_call.get("compaction_event_id") != boundary.get("event_id"):
                raise ValueError(
                    f"Trace {expected_trace_id!r} compaction event is corrupted"
                )
            decision = first_call.get("policy_decision")
            assert isinstance(decision, Mapping)
            for field in ("policy_name", "policy_version", "config_digest"):
                if boundary.get(field) != decision.get(field):
                    raise ValueError(
                        f"Trace {expected_trace_id!r} boundary {field} is corrupted"
                    )

        if strict:
            for interior_call in trace_calls[1:]:
                if (
                    interior_call.get("boundary_before_id") is not None
                    or interior_call.get("compaction_event_id") is not None
                ):
                    raise ValueError(
                        f"Trace {expected_trace_id!r} has an interior boundary"
                    )

    if calls_by_trace:
        raise ValueError(
            f"Model calls reference missing physical traces: {sorted(calls_by_trace)}"
        )
    if emitted_completion_ids != [str(call["completion_id"]) for call in calls]:
        raise ValueError("Physical traces do not preserve model-call ordering")
    if media_assets is not None:
        unknown = {
            media_id for media_id in all_media_ids if media_id not in media_assets
        }
        if unknown:
            raise ValueError(f"Trace references unknown media IDs: {sorted(unknown)!r}")

    checks = bundle.get("checks")
    if not isinstance(checks, Mapping):
        raise ValueError("Trace bundle has no checks mapping")
    expected_total = sum(len(call["sampled_token_ids"]) for call in calls)
    if checks.get("model_call_count") != len(calls):
        raise ValueError("Trace check model_call_count is corrupted")
    if checks.get("physical_trace_count") != len(traces):
        raise ValueError("Trace check physical_trace_count is corrupted")
    if checks.get("trainable_token_count") != expected_total:
        raise ValueError("Trace check trainable_token_count is corrupted")
    if checks.get("eligible_trainable_token_count") != emitted_trainable_tokens:
        raise ValueError("Trace check eligible_trainable_token_count is corrupted")
    for name in (
        "all_completion_ids_unique",
        "all_completions_present_once",
        "all_logprobs_aligned",
        "prefixes_valid_inside_traces",
        "media_order_valid",
        "boundary_records_cover_rewrites",
        "ok",
    ):
        if checks.get(name) is not True:
            raise ValueError(f"Trace check {name} is corrupted")
    if emitted_sampled_tokens != expected_total:
        raise ValueError("Physical traces do not conserve sampled tokens")
    return {
        "model_call_count": len(calls),
        "physical_trace_count": len(traces),
        "sampled_token_count": emitted_sampled_tokens,
        "eligible_trainable_token_count": emitted_trainable_tokens,
        "media_occurrence_count": len(all_media_ids),
    }


def _canonical_bundle_digest(bundle: Mapping[str, Any]) -> str:
    payload = json.dumps(
        bundle,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=repr,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def validate_rollout_trace_group(
    bundles: Sequence[Mapping[str, Any]],
    *,
    expected_group_id: str,
    training_admission: bool = False,
) -> dict[str, Any]:
    """Validate one logical RL group and deduplicate exact retry replays."""
    if not expected_group_id:
        raise ValueError("expected_group_id must be non-empty")
    by_rollout_id: dict[str, tuple[str, Mapping[str, Any]]] = {}
    duplicate_count = 0
    generation_contract_ids: set[str] = set()
    training_admission_contract_ids: set[str] = set()
    policy_identities: set[tuple[str | None, str | None, str | None]] = set()
    rewards: list[float] = []

    for bundle in bundles:
        validate_rollout_trace_bundle(bundle, strict=True)
        if bundle.get("group_id") != expected_group_id:
            raise ValueError(
                "Trace bundle group ID mismatch: "
                f"expected={expected_group_id!r}, "
                f"observed={bundle.get('group_id')!r}"
            )
        rollout_id = bundle.get("rollout_id")
        assert isinstance(rollout_id, str)
        digest = _canonical_bundle_digest(bundle)
        previous = by_rollout_id.get(rollout_id)
        if previous is not None:
            if previous[0] != digest:
                raise ValueError(
                    f"Rollout ID {rollout_id!r} was reused for different content"
                )
            duplicate_count += 1
            continue
        by_rollout_id[rollout_id] = (digest, bundle)

        reward = bundle.get("reward")
        if not isinstance(reward, (int, float)) or not math.isfinite(float(reward)):
            raise ValueError(f"Rollout {rollout_id!r} has an invalid reward")
        rewards.append(float(reward))

        contract = bundle.get("generation_contract")
        assert isinstance(contract, Mapping)
        contract_id = contract.get("generation_contract_id")
        if not isinstance(contract_id, str) or not contract_id:
            raise ValueError(f"Rollout {rollout_id!r} has no generation contract ID")
        generation_contract_ids.add(contract_id)
        if training_admission:
            admission = bundle.get("training_admission")
            if not isinstance(admission, Mapping):
                raise ValueError(
                    f"Rollout {rollout_id!r} has no NeMo-RL training admission"
                )
            validate_training_admission_contract(admission, contract)
            admission_id = admission.get("admission_contract_id")
            if not isinstance(admission_id, str) or not admission_id:
                raise ValueError(
                    f"Rollout {rollout_id!r} has no training admission identity"
                )
            training_admission_contract_ids.add(admission_id)

        first_call = bundle["model_calls"][0]
        decision = first_call.get("policy_decision")
        if not isinstance(decision, Mapping):
            raise ValueError(
                f"Rollout {rollout_id!r} has no compaction policy decision"
            )
        policy_identities.add(
            (
                decision.get("policy_name"),
                decision.get("policy_version"),
                decision.get("config_digest"),
            )
        )

    if not by_rollout_id:
        raise ValueError("A rollout trace group must contain at least one bundle")
    if len(generation_contract_ids) != 1:
        raise ValueError(
            "Rollout group contains mixed generation contracts: "
            f"{sorted(generation_contract_ids)!r}"
        )
    if training_admission and len(training_admission_contract_ids) != 1:
        raise ValueError(
            "Rollout group contains mixed training admission contracts: "
            f"{sorted(training_admission_contract_ids)!r}"
        )
    if len(policy_identities) != 1:
        raise ValueError(
            "Rollout group contains mixed compaction policies: "
            f"{sorted(policy_identities)!r}"
        )

    return {
        "group_id": expected_group_id,
        "unique_rollout_count": len(by_rollout_id),
        "duplicate_retry_count": duplicate_count,
        "reward_count": len(rewards),
        "generation_contract_id": next(iter(generation_contract_ids)),
        "training_admission_contract_id": (
            next(iter(training_admission_contract_ids))
            if training_admission_contract_ids
            else None
        ),
        "policy_identity": next(iter(policy_identities)),
        "training_admitted": training_admission,
    }
