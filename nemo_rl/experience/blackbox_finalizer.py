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
"""Blackbox finalization: token-free receipts + staged deltas -> canonical rows.

Orchestration only:
per rollout, fetch the staged rows the receipt manifest names through the
``TokenSource``, re-verify them (digest recomputation over fetched values,
shape/mask/finite-logprob checks, length chaining, weight-version tag
equality), then delegate semantics to Gym's pure ``linearize``
(``main_chain_only`` + ``terminal_hint``). Any rejection becomes a masked
placeholder row — the group always publishes exactly N rows so GRPO group
shape survives; validity folds into ``sample_mask`` (no new train field) and
placeholders copy ``prompt_ids_for_adv`` from a valid sibling so per-prompt
baselines stay well-formed.

In deferred-route mode the finalizer reads only small columns, publishes a
strict route plan beside each canonical row, and leaves staged route fragments
live until policy consumption completes. The direct-route rollback path keeps
the prior finalizer-side materialization behavior.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Any, Optional

import torch

from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.schema import ROUTE_PLAN_TAG
from nemo_rl.data_plane.tq_token_sink import TQTokenSink, TQTokenSource
from nemo_rl.experience.payload import pack_payload
from nemo_rl.experience.route_plan import (
    ROUTE_PLAN_SCHEMA_VERSION,
    RouteAssemblyPlan,
    RouteSpan,
    classify_route_span,
    encode_route_plan,
    encoded_route_plan_size_bytes,
)
from nemo_rl.experience.row_dump import maybe_dump_train_rows

# Keep the finalizer importable in its CPU-only actor without importing the
# generation package (which eagerly loads backend dependencies). This value is
# the shared router-replay missing-route wire sentinel.
_ROUTED_EXPERTS_SENTINEL = -1


@dataclass(frozen=True)
class FinalizedRollout:
    """One rollout's canonical row, or its rejection."""

    rollout_id: str
    valid: bool
    rejection_reason: Optional[str]
    token_ids: list[int]
    token_mask: list[float]
    logprobs: list[float]
    prompt_len: int
    reward: float
    staging_keys: list[str]
    min_wv: Optional[int] = None
    max_wv: Optional[int] = None
    # Router replay (R3): [len(token_ids)][num_moe_layers][topk] from the
    # rebuilt chain; None when the rollout staged no extras.
    routed_experts: Optional[list] = None
    route_plan: Optional[RouteAssemblyPlan] = None


@dataclass
class FinalizedGroup:
    """What ``finalize_group`` hands back for ``commit_finalized``."""

    meta: Optional[KVBatchMeta]
    group_min_wv: int
    group_max_wv: int
    staging_keys: list[str]
    metrics: dict[str, float] = field(default_factory=dict)
    # True when the finalizer rejected the whole group as a policy outcome
    # (see drop_reason); the caller aborts the slot instead of committing it.
    dropped: bool = False
    # Which policy dropped the group, for the caller's log line.
    drop_reason: Optional[str] = None


def _linearize_metadata_only(receipt: Any, snapshots: list[Any]) -> Any:
    """Linearize a verified chain while leaving digest-bound route bytes in TQ."""
    from nemo_gym.token_id_capture.staging.rebuild import (
        LinearizedRow,
        RebuildError,
        WeightVersionSpan,
    )

    records = {record.model_call_id: record for record in receipt.manifest}
    staged = {snapshot.model_call_id: snapshot for snapshot in snapshots}
    for model_call_id, record in records.items():
        if record.parent_call_id is not None:
            parent = records.get(record.parent_call_id)
            if parent is None:
                raise RebuildError(
                    "missing_parent",
                    f"call {model_call_id} names absent parent {record.parent_call_id}",
                )
            if parent.cum_len != record.prev_len:
                raise RebuildError(
                    "parent_length_mismatch",
                    f"call {model_call_id} starts at {record.prev_len}, parent ends at {parent.cum_len}",
                )
        visited: set[str] = set()
        cursor = record
        while cursor is not None:
            if cursor.model_call_id in visited:
                raise RebuildError(
                    "lineage_cycle", f"cycle reaches call {cursor.model_call_id}"
                )
            visited.add(cursor.model_call_id)
            cursor = (
                records.get(cursor.parent_call_id)
                if cursor.parent_call_id is not None
                else None
            )

    terminal_id = receipt.terminal_model_call_id
    if terminal_id is None or terminal_id not in records:
        raise RebuildError(
            "missing_terminal", "successful receipt has no terminal call"
        )
    chain = []
    cursor = records[terminal_id]
    while cursor is not None:
        chain.append(cursor)
        cursor = (
            records.get(cursor.parent_call_id)
            if cursor.parent_call_id is not None
            else None
        )
    chain.reverse()

    token_ids: list[int] = []
    token_mask: list[float] = []
    logprobs: list[float] = []
    model_call_ids: list[str] = []
    weight_versions: list[int] = []
    weight_version_spans = []
    link_spans: list[tuple[str, int, int]] = []
    prompt_len = 0
    for index, record in enumerate(chain):
        snapshot = staged[record.model_call_id]
        boundary = 0
        for mask in snapshot.token_mask_delta:
            if mask != 0.0:
                break
            boundary += 1
        if boundary == len(snapshot.token_mask_delta) or any(
            mask != 1.0 for mask in snapshot.token_mask_delta[boundary:]
        ):
            raise RebuildError(
                "invalid_mask_order",
                f"call {record.model_call_id} mask is not carry-then-generation",
            )
        start = len(token_ids)
        token_ids.extend(snapshot.token_ids_delta)
        token_mask.extend(snapshot.token_mask_delta)
        logprobs.extend(snapshot.generation_log_probs_delta)
        end = len(token_ids)
        if index == 0:
            prompt_len = boundary
        model_call_ids.append(record.model_call_id)
        weight_versions.append(record.weight_version)
        weight_version_spans.append(
            WeightVersionSpan(
                model_call_id=record.model_call_id,
                start=start,
                end=end,
                weight_version=record.weight_version,
            )
        )
        link_spans.append((record.model_call_id, boundary, record.delta_len - boundary))
    return LinearizedRow(
        rollout_id=receipt.rollout_id,
        token_ids=token_ids,
        token_mask=token_mask,
        logprobs=logprobs,
        model_call_ids=model_call_ids,
        prompt_len=prompt_len,
        weight_versions=weight_versions,
        weight_version_spans=weight_version_spans,
        link_spans=link_spans,
    )


class BlackboxFinalizer:
    """Receipts -> verified rows -> N-row publish, off the generation hot path."""

    def __init__(
        self,
        dp_client: Any,
        *,
        partition_id: str,
        staging_partition: str,
        pad_token_id: int,
        mixed_weight_version_policy: str,
        min_valid_fraction_per_group: Optional[float],
        router_replay_enabled: bool = False,
        defer_routed_experts_to_policy: bool = False,
    ) -> None:
        self._dp_client = dp_client
        self._partition_id = partition_id
        self._pad_token_id = int(pad_token_id)
        self._mixed_weight_version_policy = mixed_weight_version_policy
        self._min_valid_fraction = min_valid_fraction_per_group
        self._router_replay_enabled = router_replay_enabled
        self._defer_routed_experts_to_policy = defer_routed_experts_to_policy
        if self._defer_routed_experts_to_policy and not self._router_replay_enabled:
            raise ValueError(
                "defer_routed_experts_to_policy requires router replay to be enabled"
            )
        self._staging_partition = staging_partition
        # (num_moe_layers, topk), learned from the first rebuilt row that
        # carries routes; placeholder-only groups need it to shape their
        # sentinel tensors consistently with the model.
        self._routed_dims: Optional[tuple[int, int]] = None
        self._source = TQTokenSource(dp_client, staging_partition=staging_partition)
        # The sink's clear() is the staging-partition delete; no staging
        # writes happen here.
        self._staging = TQTokenSink(dp_client, staging_partition=staging_partition)

    # ── per rollout ─────────────────────────────────────────────────────────

    def finalize_rollout(
        self, rollout_id: str, receipt: Optional[dict[str, Any]], *, reward: float
    ) -> FinalizedRollout:
        """Verify one receipt against its staged rows and linearize the main chain.

        Never raises for rollout-level problems: every rejection returns an
        invalid row whose reason feeds the metrics; the group publisher
        substitutes a placeholder.
        """
        # Deferred: nemo_gym is an optional extra absent in non-gym runs.
        from nemo_gym.token_id_capture.staging.digest import compute_staging_digest
        from nemo_gym.token_id_capture.staging.rebuild import (
            RebuildError,
            ReceiptVerificationError,
            verify_and_linearize,
        )
        from nemo_gym.token_id_capture.staging.records import RolloutReceipt

        def rejected(reason: str, staging_keys: list[str]) -> FinalizedRollout:
            return FinalizedRollout(
                rollout_id=rollout_id,
                valid=False,
                rejection_reason=reason,
                token_ids=[],
                token_mask=[],
                logprobs=[],
                prompt_len=0,
                reward=reward,
                staging_keys=staging_keys,
            )

        if receipt is None:
            return rejected("missing_receipt", [])
        try:
            parsed = RolloutReceipt.model_validate(receipt)
        except ValueError as error:
            return rejected(f"invalid_receipt:{error}", [])
        staging_keys = [record.staging_key for record in parsed.manifest]
        if parsed.rollout_id != rollout_id:
            return rejected(f"identity_mismatch:{parsed.rollout_id}", staging_keys)
        if parsed.failure_reason is not None:
            return rejected(f"rollout_failed:{parsed.failure_reason}", staging_keys)
        if parsed.capture_poisoned:
            return rejected("capture_poisoned", staging_keys)
        if not parsed.manifest:
            return rejected("empty_manifest", staging_keys)
        if len(set(staging_keys)) != len(staging_keys):
            return rejected(
                "duplicate_staging_key",
                list(dict.fromkeys(staging_keys)),
            )
        records_by_call = {record.model_call_id: record for record in parsed.manifest}
        if len(records_by_call) != len(parsed.manifest):
            return rejected("duplicate_manifest_call_id", staging_keys)

        try:
            fetched = (
                self._source.fetch_for_finalization(staging_keys)
                if self._defer_routed_experts_to_policy
                else None
            )
            snapshots = (
                [item.snapshot for item in fetched]
                if fetched is not None
                else self._source.fetch(staging_keys)
            )
        except KeyError as error:
            return rejected(f"missing_staging_row:{error}", staging_keys)
        except (TypeError, ValueError) as error:
            return rejected(f"invalid_staging_row:{error}", staging_keys)
        fetched_by_call = {}
        if fetched is not None:
            for record, item in zip(parsed.manifest, fetched):
                if item.staging_key != record.staging_key:
                    return rejected(
                        f"staging_key_mismatch:{record.model_call_id}", staging_keys
                    )
                if item.snapshot.model_call_id != record.model_call_id:
                    return rejected(
                        f"call_id_mismatch:{record.model_call_id}", staging_keys
                    )
                fetched_by_call[record.model_call_id] = item
            if len(fetched_by_call) != len(fetched):
                return rejected("duplicate_fetched_call_id", staging_keys)

        for record, snapshot in zip(parsed.manifest, snapshots):
            if not (
                len(snapshot.token_ids_delta)
                == len(snapshot.token_mask_delta)
                == len(snapshot.generation_log_probs_delta)
            ):
                return rejected(
                    f"misaligned_delta:{record.model_call_id}", staging_keys
                )
            if any(m not in (0.0, 1.0) for m in snapshot.token_mask_delta):
                return rejected(
                    f"invalid_token_mask:{record.model_call_id}", staging_keys
                )
            if any(not math.isfinite(p) for p in snapshot.generation_log_probs_delta):
                return rejected(
                    f"non_finite_logprob:{record.model_call_id}", staging_keys
                )
            if record.delta_len != len(snapshot.token_ids_delta) or (
                snapshot.prev_len + record.delta_len != record.cum_len
            ):
                return rejected(f"length_mismatch:{record.model_call_id}", staging_keys)
            compared_fields = (
                "schema_version",
                "digest_version",
                "extras_digest_version",
                "rollout_id",
                "model_call_id",
                "parent_call_id",
                "mode",
                "prev_len",
                "delta_len",
                "cum_len",
                "weight_version",
                "digest",
                "extras_digest",
                "chain_hash",
                "cumulative_hash",
            )
            mismatch = next(
                (
                    field_name
                    for field_name in compared_fields
                    if getattr(snapshot, field_name)
                    != (
                        rollout_id
                        if field_name == "rollout_id"
                        else getattr(record, field_name)
                    )
                ),
                None,
            )
            if mismatch is not None:
                return rejected(
                    f"{mismatch}_mismatch:{record.model_call_id}", staging_keys
                )
            try:
                digest = compute_staging_digest(
                    schema_version=snapshot.schema_version,
                    digest_version=snapshot.digest_version,
                    extras_digest_version=snapshot.extras_digest_version,
                    rollout_id=rollout_id,
                    model_call_id=record.model_call_id,
                    parent_call_id=record.parent_call_id,
                    mode=record.mode,
                    prev_len=snapshot.prev_len,
                    delta_len=snapshot.delta_len,
                    cum_len=snapshot.cum_len,
                    weight_version=snapshot.weight_version,
                    token_ids_delta=snapshot.token_ids_delta,
                    token_mask_delta=snapshot.token_mask_delta,
                    generation_log_probs_delta=(snapshot.generation_log_probs_delta),
                    extras_digest=snapshot.extras_digest,
                    chain_hash=snapshot.chain_hash,
                    cumulative_hash=snapshot.cumulative_hash,
                )
            except (TypeError, ValueError, OverflowError) as error:
                return rejected(
                    f"invalid_digest_input:{record.model_call_id}:{error}",
                    staging_keys,
                )
            if digest != record.digest:
                return rejected(f"digest_mismatch:{record.model_call_id}", staging_keys)

        try:
            row = (
                _linearize_metadata_only(parsed, snapshots)
                if self._defer_routed_experts_to_policy
                else verify_and_linearize(parsed, snapshots)
            )
        except (
            KeyError,
            ValueError,
            ReceiptVerificationError,
            RebuildError,
            NotImplementedError,
        ) as error:
            return rejected(f"rebuild_failed:{error}", staging_keys)
        weight_versions = [record.weight_version for record in parsed.manifest]
        min_wv, max_wv = min(weight_versions), max(weight_versions)
        if self._mixed_weight_version_policy == "reject" and min_wv != max_wv:
            return rejected(f"mixed_weight_versions:{min_wv}..{max_wv}", staging_keys)

        route_plan = None
        if self._router_replay_enabled and self._defer_routed_experts_to_policy:
            link_spans = row.link_spans
            if link_spans is None:
                return rejected("missing_link_spans", staging_keys)
            route_spans: list[RouteSpan] = []
            seen_span_call_ids: set[str] = set()
            for call_id, carry_len, generation_len in link_spans:
                if call_id in seen_span_call_ids:
                    return rejected(f"duplicate_route_span:{call_id}", staging_keys)
                seen_span_call_ids.add(call_id)
                record = records_by_call.get(call_id)
                item = fetched_by_call.get(call_id)
                if record is None or item is None:
                    return rejected(f"route_span_identity:{call_id}", staging_keys)
                if item.routed_len not in (0, record.delta_len):
                    return rejected(f"routed_len_mismatch:{call_id}", staging_keys)
                if generation_len < 0 or generation_len > record.delta_len:
                    return rejected(
                        f"route_generation_span_mismatch:{call_id}", staging_keys
                    )
                if carry_len < 0:
                    return rejected(
                        f"route_carry_span_mismatch:{call_id}", staging_keys
                    )
                span = RouteSpan(
                    staging_key=record.staging_key,
                    carry_len=int(carry_len),
                    generation_len=int(generation_len),
                    staged_route_len=item.routed_len,
                    extras_digest_version=record.extras_digest_version,
                    extras_digest=record.extras_digest,
                )
                classify_route_span(span)
                route_spans.append(span)
            if sum(span.carry_len + span.generation_len for span in route_spans) != len(
                row.token_ids
            ):
                return rejected("route_span_length_mismatch", staging_keys)
            route_plan = RouteAssemblyPlan(
                schema_version=ROUTE_PLAN_SCHEMA_VERSION,
                staging_partition=self._staging_partition,
                spans=tuple(route_spans),
                cleanup_staging_keys=tuple(staging_keys),
                expected_token_length=len(row.token_ids),
            )
            encode_route_plan(route_plan)

        return FinalizedRollout(
            rollout_id=rollout_id,
            valid=True,
            rejection_reason=None,
            token_ids=row.token_ids,
            token_mask=row.token_mask,
            logprobs=row.logprobs,
            prompt_len=row.prompt_len,
            reward=reward,
            staging_keys=staging_keys,
            min_wv=min_wv,
            max_wv=max_wv,
            # getattr: pre-R3 Gym pins' LinearizedRow has no routed_experts;
            # capture-R3 then degrades to the sentinel/group-drop path.
            routed_experts=(
                None
                if self._defer_routed_experts_to_policy
                else getattr(row, "routed_experts", None)
            ),
            route_plan=route_plan,
        )

    # ── per group ───────────────────────────────────────────────────────────

    def finalize_group(
        self,
        group_id: str,
        rollout_ids: list[str],
        receipts: list[Optional[dict[str, Any]]],
        rewards: list[float],
        *,
        prompt_idx: int,
        fallback_weight_version: int,
        canonical_sample_ids: Optional[list[str]] = None,
    ) -> FinalizedGroup:
        """Publish exactly N canonical rows for one prompt group.

        Blocking (TQ round trips); run via ``asyncio.to_thread`` from the
        dispatch task. ``fallback_weight_version`` stamps a group none of
        whose rollouts produced a valid row (placeholder-only groups still
        need a staleness tag).
        """
        assert len(rollout_ids) == len(receipts) == len(rewards), (
            "rollout_ids, receipts, and rewards must be parallel"
        )
        if canonical_sample_ids is None:
            canonical_sample_ids = rollout_ids
        assert len(canonical_sample_ids) == len(rollout_ids), (
            "canonical_sample_ids must be one per rollout"
        )
        _group_t0 = time.perf_counter()
        rows = [
            self.finalize_rollout(rollout_id, receipt, reward=reward)
            for rollout_id, receipt, reward in zip(rollout_ids, receipts, rewards)
        ]
        _rollouts_ms = (time.perf_counter() - _group_t0) * 1000.0
        valid_rows = [row for row in rows if row.valid]
        staging_keys = [key for row in rows for key in row.staging_keys]
        metrics = {
            "finalize/invalid_row_rate": 1.0 - len(valid_rows) / len(rows),
            "finalize/calls_per_rollout": (
                sum(len(row.staging_keys) for row in rows) / len(rows)
            ),
        }
        # Ledger-derived admission counters (per group): each manifest row
        # carries its admission mode. token_in_rate near 1.0 is the capture
        # health signal (a text root only opens each chain); this replaces the
        # deleted gate metrics route.
        manifest_rows = [
            record
            for receipt in receipts
            if isinstance(receipt, dict)
            for record in (receipt.get("manifest") or [])
            if isinstance(record, dict)
        ]
        if manifest_rows:
            token_in_calls = sum(
                1 for record in manifest_rows if record.get("mode") == "token_in"
            )
            metrics["finalize/token_in_calls"] = float(token_in_calls)
            metrics["finalize/text_root_calls"] = float(
                len(manifest_rows) - token_in_calls
            )
            metrics["finalize/token_in_rate"] = token_in_calls / len(manifest_rows)
        metrics["finalize/capture_poisoned_rollouts"] = float(
            sum(
                1
                for receipt in receipts
                if isinstance(receipt, dict) and receipt.get("capture_poisoned")
            )
        )
        # Per-method terminal-selection breakdown. Witness methods
        # (declared/response_id/content) resolve from evidence; heuristic is
        # the no-witness parent-link fallback — a nonzero heuristic fraction
        # on a declaring harness is a regression signal. Failed selections
        # stamp the last stage attempted, so masked rollouts stay visible in
        # their method's bucket (cross-reference finalize/invalid_row_rate).
        for method in ("declared", "response_id", "content", "heuristic"):
            method_receipts = sum(
                1
                for receipt in receipts
                if isinstance(receipt, dict)
                and receipt.get("terminal_selection") == method
            )
            metrics[f"finalize/terminal_selection_{method}_count"] = float(
                method_receipts
            )
            metrics[f"finalize/terminal_selection_{method}_fraction"] = (
                method_receipts / len(receipts)
            )
        witness_disagreements = sum(
            1
            for receipt in receipts
            if isinstance(receipt, dict)
            and "witness_disagreement"
            in str(receipt.get("terminal_attribution_reason") or "")
        )
        metrics["finalize/terminal_witness_disagreement_count"] = float(
            witness_disagreements
        )
        for row in rows:
            if not row.valid:
                print(
                    f"  finalize: rollout {row.rollout_id} rejected "
                    f"({row.rejection_reason}) — placeholder",
                    flush=True,
                )

        min_weight_versions = [
            row.min_wv for row in valid_rows if row.min_wv is not None
        ]
        max_weight_versions = [
            row.max_wv for row in valid_rows if row.max_wv is not None
        ]
        if len(min_weight_versions) != len(valid_rows) or len(
            max_weight_versions
        ) != len(valid_rows):
            raise RuntimeError("valid finalized rows must carry weight versions")
        group_min_wv = min(min_weight_versions, default=fallback_weight_version)
        group_max_wv = max(max_weight_versions, default=fallback_weight_version)

        valid_fraction = len(valid_rows) / len(rows)
        if (
            self._min_valid_fraction is not None
            and valid_fraction < self._min_valid_fraction
        ):
            self._clear_staging(staging_keys)
            metrics["finalize/group_dropped"] = 1.0
            return FinalizedGroup(
                meta=None,
                group_min_wv=group_min_wv,
                group_max_wv=group_max_wv,
                staging_keys=[],
                metrics=metrics,
                dropped=True,
                drop_reason=(
                    f"min_valid_fraction_per_group: {valid_fraction:.3f} < "
                    f"{self._min_valid_fraction}"
                ),
            )

        _tensorize_t0 = time.perf_counter()
        # Placeholders borrow a valid sibling's prompt ids so per-prompt
        # baselines group correctly; an all-placeholder group uses a single
        # pad token (its rows all carry sample_mask 0 and never train).
        sibling_prompt = (
            valid_rows[0].token_ids[: valid_rows[0].prompt_len] if valid_rows else []
        ) or [self._pad_token_id]

        n = len(rows)
        seq_lens = [max(1, len(row.token_ids)) for row in rows]
        max_len = max(seq_lens)
        input_ids = torch.full((n, max_len), self._pad_token_id, dtype=torch.int64)
        token_mask = torch.zeros((n, max_len), dtype=torch.float32)
        logprobs = torch.zeros((n, max_len), dtype=torch.float32)
        prompt_ids_for_adv = torch.tensor([sibling_prompt] * n, dtype=torch.int64)
        sample_mask = torch.zeros(n, dtype=torch.float32)
        lengths = torch.tensor(seq_lens, dtype=torch.long)
        rewards_t = torch.tensor([row.reward for row in rows], dtype=torch.float32)
        for i, row in enumerate(rows):
            if not row.valid:
                continue
            length = len(row.token_ids)
            input_ids[i, :length] = torch.tensor(row.token_ids, dtype=torch.int64)
            token_mask[i, :length] = torch.tensor(row.token_mask, dtype=torch.float32)
            logprobs[i, :length] = torch.tensor(row.logprobs, dtype=torch.float32)
            sample_mask[i] = 1.0

        train_batch = {
            "input_ids": input_ids,
            "input_lengths": lengths,
            "generation_logprobs": logprobs,
            "token_mask": token_mask,
            "sample_mask": sample_mask,
            "prompt_ids_for_adv": prompt_ids_for_adv,
            "total_reward": rewards_t,
        }
        if self._router_replay_enabled and not self._defer_routed_experts_to_policy:
            has_routed_row = any(r.valid and r.routed_experts for r in rows)
            if not has_routed_row and self._routed_dims is None and not valid_rows:
                # Nothing to learn (L, K) from yet — e.g. an all-poisoned
                # group before the first healthy rollout. Dropping loses no
                # training signal (no valid rows or routes) and keeps the
                # partition schema consistent for groups that do publish.
                print(
                    f"  finalize: group {group_id} dropped — router replay on "
                    "but no rollout carried routed_experts and (L, K) is "
                    "unknown yet",
                    flush=True,
                )
                self._clear_staging(staging_keys)
                metrics["finalize/group_dropped"] = 1.0
                return FinalizedGroup(
                    meta=None,
                    group_min_wv=group_min_wv,
                    group_max_wv=group_max_wv,
                    staging_keys=[],
                    metrics=metrics,
                    dropped=True,
                    drop_reason=(
                        "router replay on, no rollout carried routed_experts, "
                        "and (L, K) is unknown yet"
                    ),
                )
            train_batch["routed_experts"] = self._build_routed_experts_tensor(
                rows, max_len=max_len, metrics=metrics
            )
        sample_ids, fields, tags = pack_payload(
            train_batch,
            weight_version=group_min_wv,
            group_id=group_id,
            prompt_idx=prompt_idx,
        )
        if self._defer_routed_experts_to_policy:
            encoded_sizes = 0
            span_count = 0
            for tag, row, expected_length in zip(tags, rows, seq_lens):
                plan = row.route_plan
                if plan is None:
                    plan = RouteAssemblyPlan(
                        schema_version=ROUTE_PLAN_SCHEMA_VERSION,
                        staging_partition=self._staging_partition,
                        spans=(),
                        cleanup_staging_keys=tuple(row.staging_keys),
                        expected_token_length=expected_length,
                    )
                encoded = encode_route_plan(plan)
                tag[ROUTE_PLAN_TAG] = encoded
                encoded_sizes += encoded_route_plan_size_bytes(plan)
                span_count += len(plan.spans)
            metrics["finalize/route_plan_span_count"] = float(span_count)
            metrics["finalize/route_plan_encoded_bytes"] = float(encoded_sizes)
            valid_route_rows = sum(
                1 for row in valid_rows if row.route_plan and row.route_plan.spans
            )
            if valid_rows:
                metrics["finalize/routed_experts_row_coverage"] = (
                    valid_route_rows / len(valid_rows)
                )
        maybe_dump_train_rows(
            source="finalizer",
            group_id=group_id,
            sample_ids=list(sample_ids),
            train_batch=train_batch,
            weight_version=group_min_wv,
        )
        assert sample_ids == canonical_sample_ids, (
            "canonical sample ids must equal the stable logical rollout ids: "
            f"{sample_ids} != {canonical_sample_ids}"
        )
        _tensorize_ms = (time.perf_counter() - _tensorize_t0) * 1000.0
        _put_t0 = time.perf_counter()
        self._call_dp(
            "put_samples",
            sample_ids=sample_ids,
            partition_id=self._partition_id,
            fields=fields,
            tags=tags,
        )
        _put_ms = (time.perf_counter() - _put_t0) * 1000.0
        _clear_ms = 0.0
        if not self._defer_routed_experts_to_policy:
            _clear_t0 = time.perf_counter()
            self._clear_staging(staging_keys)
            _clear_ms = (time.perf_counter() - _clear_t0) * 1000.0
        # Per-step W&B breakdown of training-row assembly (capture arm) rides
        # FinalizedGroup.metrics into the controller's rollout metrics.
        metrics["row_assembly/rollouts_ms"] = _rollouts_ms
        metrics["row_assembly/tensorize_ms"] = _tensorize_ms
        metrics["row_assembly/tq_put_ms"] = _put_ms
        if not self._defer_routed_experts_to_policy:
            metrics["row_assembly/clear_staging_ms"] = _clear_ms
        meta = KVBatchMeta(
            partition_id=self._partition_id,
            task_name="train",
            sample_ids=list(sample_ids),
            fields=list(fields.keys()),
            sequence_lengths=[int(s) for s in lengths.tolist()],
            tags=[dict(t) for t in tags],
        )
        return FinalizedGroup(
            meta=meta,
            group_min_wv=group_min_wv,
            group_max_wv=group_max_wv,
            staging_keys=(staging_keys if self._defer_routed_experts_to_policy else []),
            metrics=metrics,
        )

    # ── internals ───────────────────────────────────────────────────────────

    def _build_routed_experts_tensor(
        self,
        rows: list[FinalizedRollout],
        *,
        max_len: int,
        metrics: dict[str, float],
    ) -> torch.Tensor:
        """[n, max_len, L, K] int16 routes for the group; sentinel elsewhere.

        Padding, placeholder rows, and valid rows whose rebuild carried no
        routes are all-sentinel: Megatron's replay falls back to its own
        router for exactly those positions. (L, K) is learned from the first
        rebuilt row that carries routes and cached for placeholder-only
        groups; a group arriving before any routed row has been seen cannot
        be shaped and fails loudly (unreachable once the first real rollout
        of the run finalizes).
        """
        for row in rows:
            if row.valid and row.routed_experts:
                first = row.routed_experts[0]
                self._routed_dims = (len(first), len(first[0]))
                break
        if self._routed_dims is None:
            raise RuntimeError(
                "policy.router_replay.enabled=true (token-capture mode) but no "
                "finalized rollout has carried routed_experts yet, so the "
                "placeholder group tensor cannot be shaped. Check vLLM "
                "enable_return_routed_experts and the staging-extras path."
            )
        num_moe_layers, topk = self._routed_dims
        routed = torch.full(
            (len(rows), max_len, num_moe_layers, topk),
            _ROUTED_EXPERTS_SENTINEL,
            dtype=torch.int16,
        )
        rows_with_routes = 0
        valid_rows = 0
        sentinel_tokens = 0
        covered_tokens = 0
        for i, row in enumerate(rows):
            if not row.valid:
                continue
            valid_rows += 1
            covered_tokens += len(row.token_ids)
            if not row.routed_experts:
                sentinel_tokens += len(row.token_ids)
                continue
            rows_with_routes += 1
            row_routes = torch.tensor(row.routed_experts, dtype=torch.int16)
            if row_routes.shape != (len(row.token_ids), num_moe_layers, topk):
                raise RuntimeError(
                    "rebuilt routed_experts shape "
                    f"{tuple(row_routes.shape)} does not match "
                    f"({len(row.token_ids)}, {num_moe_layers}, {topk}) for "
                    f"rollout {row.rollout_id}"
                )
            routed[i, : row_routes.shape[0]] = row_routes
            sentinel_tokens += int(
                row_routes.eq(_ROUTED_EXPERTS_SENTINEL).all(-1).all(-1).sum().item()
            )
        if valid_rows:
            metrics["finalize/routed_experts_row_coverage"] = (
                rows_with_routes / valid_rows
            )
        if covered_tokens:
            metrics["finalize/routed_experts_sentinel_token_fraction"] = (
                sentinel_tokens / covered_tokens
            )
        return routed

    def _clear_staging(self, staging_keys: list[str]) -> None:
        if not staging_keys:
            return
        try:
            self._staging.clear(staging_keys)
        except Exception as error:
            raise RuntimeError(
                "finalizer staging cleanup failed for known keys "
                f"partition={self._staging_partition!r}, keys={staging_keys!r}"
            ) from error

    def _call_dp(self, method_name: str, **kwargs: Any) -> Any:
        import ray

        method = getattr(self._dp_client, method_name)
        remote = getattr(method, "remote", None)
        if remote is not None:
            return ray.get(remote(**kwargs))
        return method(**kwargs)
