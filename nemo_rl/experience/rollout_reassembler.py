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
per rollout, apply the rollout-level receipt guards, fetch the staged base
rows the receipt manifest names through the ``TokenSource`` (normally
validated ``StagedCallBaseSnapshot`` values), and delegate all token, digest,
lineage, and terminal-chain semantics to Gym's ``verify_and_linearize`` (or,
with ``token_capture.segment_rows.enabled``, ``verify_and_linearize_all``).
Any rejection becomes a masked placeholder row — the group always publishes
at least N rows (one canonical row per rollout, canonical block first) so GRPO
group shape survives; validity folds into ``sample_mask`` (no new train
field) and placeholders copy ``prompt_ids_for_adv`` from a valid sibling so
per-prompt baselines stay well-formed.

Segment rows (``token_capture.segment_rows``): a rollout whose receipt
verifies to several chains — pre-compaction segments and compaction summary
calls joined by context-rewrite boundaries, or subagent sessions — publishes
its terminal chain as the canonical row ``{group}_g{i}`` and each additional
chain as an extra row ``{group}_g{i}_t{j}`` (``j >= 1``) appended after the
canonical block. Extra rows keep Gym's row order (the chains on the
terminal's compaction sequence in segment order, then other roots by
admission time); ``include_summary_rows=False`` drops the
``compaction_summary`` chains first, then the ``max_per_rollout - 1`` cap
applies to what remains. Extra rows copy the rollout's reward,
``mask_sample``, ``prompt_ids_for_adv`` and ``sample_mask``; ``truncated`` is
per row. Per-row tags (``rollout_local_idx``, ``trace_in_rollout_idx``,
``trace_kind``, ``segment_index``, ``rows_in_rollout``) let the advantage
stage dedup rewards back to rollouts and assert it received whole rollouts.
Staging cleanup ownership stays on the canonical row only.

Router replay runs one unified flow: both modes construct the same
``RouteAssemblyPlan`` from Gym's link spans and extras commitments, per
chain. Deferred mode publishes the encoded plan beside each row and leaves
staged route fragments live until policy consumption; direct mode executes
the plan eagerly with fragments fetched in the same batch — any executor
failure is a pre-publication ``route_assembly:<reason>`` rejection.
"""

from __future__ import annotations

import time
import warnings
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Callable, Optional, cast

import torch

from nemo_rl.data.multimodal_utils import PackedTensor
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.schema import MASK_SAMPLE, ROUTE_PLAN_TAG, TRUNCATED
from nemo_rl.data_plane.tq_token_sink import (
    FetchedStagedCall,
    StagedMediaTensors,
    TQTokenSink,
    TQTokenSource,
)
from nemo_rl.experience.payload import pack_payload
from nemo_rl.experience.route_assembly import (
    ROUTE_MISSING_SENTINEL,
    RouteFragment,
    execute_route_plan,
)
from nemo_rl.experience.route_plan import (
    ROUTE_PLAN_SCHEMA_VERSION,
    RouteAssemblyPlan,
    RouteSpan,
    encode_route_plan,
    encoded_route_plan_size_bytes,
    validate_route_plan,
)
from nemo_rl.experience.sample_ids import make_sample_id, try_parse_sample_id

# ``trace_kind`` tag values. The first three mirror Gym's
# ``LinearizedRow.chain_kind``; ``placeholder`` marks a rejected rollout's
# masked row.
TRACE_KIND_TERMINAL = "terminal"
TRACE_KIND_COMPACTION_SEGMENT = "compaction_segment"
TRACE_KIND_COMPACTION_SUMMARY = "compaction_summary"
TRACE_KIND_SUBAGENT = "subagent"
TRACE_KIND_PLACEHOLDER = "placeholder"

# Gym's ``SkippedChain.reason`` for an ambiguous leaf (fork inside a
# non-terminal chain); the only skip reason surfaced as its own metric.
_SKIP_AMBIGUOUS_LEAF = "ambiguous_leaf"


@dataclass(frozen=True)
class FinalizedRollout:
    """One verified chain of a rollout (a training row), or the rollout's rejection.

    ``finalize_rollout`` returns a list of these per rollout: index 0 is the
    canonical row (the receipt's terminal chain, or the placeholder for a
    rejected rollout); indices >= 1 are extra segment rows and only exist
    when ``segment_rows`` is enabled and the rollout verified to more than
    one chain. Rollout-level diagnostics (``num_chains``, ``chains_skipped_*``,
    ``boundary_roots``, ``segments_dropped_by_cap``, ``segments_rejected``)
    are carried on index 0 only.
    """

    rollout_id: str
    valid: bool
    rejection_reason: Optional[str]
    token_ids: list[int]
    token_mask: list[float]
    logprobs: list[float]
    prompt_len: int
    reward: float
    # Staging cleanup ownership: the whole receipt manifest on the canonical
    # row, empty on extra rows (they share the same staged calls).
    staging_keys: list[str]
    min_wv: Optional[int] = None
    max_wv: Optional[int] = None
    # Router replay (R3): [len(token_ids), num_moe_layers, topk] int16 from
    # the executed route plan; None when the rollout staged no routes.
    routed_experts: Optional[torch.Tensor] = None
    route_plan: Optional[RouteAssemblyPlan] = None
    # Trainer-ready media for this row (one logical row per PackedTensor):
    # the engine's own vision-encoder inputs, read off the terminal chain's
    # media columns (staged in the same put as each call's tokens) and
    # structurally validated. None for text rollouts.
    media: Optional[dict[str, PackedTensor]] = None
    # Segment placement (token_capture.segment_rows). Defaults describe the
    # single canonical row of the classic one-row-per-rollout path.
    trace_in_rollout_idx: int = 0
    trace_kind: str = TRACE_KIND_TERMINAL
    segment_index: int = 0
    boundary_parent_call_id: Optional[str] = None
    chain_index: int = 0
    # This row's (call_id, carry_len, generation_len) spans from Gym; the
    # route plan above is built from exactly these.
    link_spans: tuple[tuple[str, int, int], ...] = ()
    # Rollout-level diagnostics (canonical row only).
    num_chains: int = 1
    chains_skipped_ambiguous: int = 0
    chains_skipped_other: int = 0
    boundary_roots: int = 0
    segments_dropped_by_cap: int = 0
    segments_rejected: int = 0
    # compaction_summary chains dropped because include_summary_rows is off.
    segments_skipped_summary: int = 0


@dataclass
class FinalizedGroup:
    """What ``finalize_group`` hands back for ``commit_finalized``."""

    meta: Optional[KVBatchMeta]
    group_min_wv: int
    group_max_wv: int
    staging_keys: list[str]
    canonical_output_tokens: int = 0
    metrics: dict[str, float] = field(default_factory=dict)
    # True when the finalizer rejected the whole group as a structural outcome
    # (see drop_reason); the caller aborts the slot instead of committing it.
    # Policy decisions like a low valid-row fraction are no longer made here --
    # the caller reads valid_row_count/total_row_count and decides whether to
    # replace the group, since only the caller can source a replacement.
    dropped: bool = False
    # Which policy dropped the group, for the caller's log line.
    drop_reason: Optional[str] = None
    # ROLLOUT counts (not published-row counts): rollouts whose canonical row
    # verified vs. rollouts in the group. 0/0 on a dropped group (the caller
    # does not read these when dropped is True). The controller's
    # min_valid_fraction_per_group policy reads exactly these, so segment
    # rows never inflate a group's apparent validity.
    valid_row_count: int = 0
    total_row_count: int = 0
    # Extra segment rows published after the canonical block
    # (``len(meta.sample_ids) == total_row_count + extra_row_count``).
    extra_row_count: int = 0


def _concat_media(parts: list[StagedMediaTensors]) -> StagedMediaTensors:
    """Concatenate per-call media deltas along the terminal chain, in order.

    Packed patches (``[1, total_patches, F]``) join along the patch dim, the
    per-frame sizes and per-video frame counts along their only dim. Parts
    must agree on pixel dtype and patch feature width (no silent promotion)
    and either all or none may carry frame counts (no mixed image/video
    chains). Each part was already validated by ``validate_media_tensors``.
    """
    if not parts:
        raise ValueError("media chain has no parts")
    if len(parts) == 1:
        return parts[0]
    first = parts[0].imgs
    for part in parts[1:]:
        if part.imgs.dtype != first.dtype:
            raise ValueError(
                f"media chain mixes pixel dtypes {first.dtype} and {part.imgs.dtype}"
            )
        if part.imgs.shape[-1] != first.shape[-1]:
            raise ValueError(
                "media chain mixes patch feature widths "
                f"{int(first.shape[-1])} and {int(part.imgs.shape[-1])}"
            )
    frames_present = [part.num_frames is not None for part in parts]
    if any(frames_present) and not all(frames_present):
        raise ValueError(
            "num_frames is staged on some calls of the chain but not others"
        )
    return StagedMediaTensors(
        imgs=torch.cat([part.imgs for part in parts], dim=1),
        imgs_sizes=torch.cat([part.imgs_sizes.reshape(-1, 2) for part in parts], dim=0),
        num_frames=(
            torch.cat([part.num_frames.reshape(-1) for part in parts], dim=0)  # type: ignore[union-attr]
            if all(frames_present)
            else None
        ),
    )


def _trainer_media(staged: StagedMediaTensors) -> dict[str, PackedTensor]:
    """Wrap the engine's media tensors as the trainer's one-row PackedTensors.

    ``pixel_values`` keeps MInf's packed-patch layout: ``[total_patches, C*P*P]``
    per row, so rows concatenate along dim 0 with no padding. Bridge's
    ``_patchify_dynamic_images`` accepts that 2-D layout directly. ``imgs_sizes`` and
    ``num_frames`` mirror what ``extract_multimodal_model_inputs`` emits on the
    token-echo path (stills get one frame per image).
    """
    imgs = staged.imgs
    if imgs.ndim == 3 and imgs.shape[0] == 1:
        imgs = imgs.squeeze(0)
    media = {"pixel_values": PackedTensor([imgs], dim_to_pack=0)}
    sizes = staged.imgs_sizes.reshape(-1, 2).to(torch.int32)
    media["imgs_sizes"] = PackedTensor([sizes], dim_to_pack=0)
    frames = (
        staged.num_frames.reshape(-1).to(torch.int32)
        if staged.num_frames is not None
        else torch.ones(sizes.shape[0], dtype=torch.int32)
    )
    media["num_frames"] = PackedTensor([frames], dim_to_pack=0)
    return media


def _media_fields_for_group(rows: list[FinalizedRollout]) -> dict[str, PackedTensor]:
    """Stack per-rollout media into group-level PackedTensors, one logical row each.

    Rows without media (text rollouts, placeholders) contribute an empty
    logical row so every media field stays aligned with ``input_ids``.
    """
    per_row = [row.media if (row.valid and row.media) else None for row in rows]
    if not any(per_row):
        return {}
    fields = sorted({key for media in per_row if media for key in media})
    stacked: dict[str, PackedTensor] = {}
    for key in fields:
        reference = next(media[key] for media in per_row if media and key in media)
        parts = [
            media[key]
            if (media and key in media)
            else PackedTensor.empty_rows_like(reference, 1)
            for media in per_row
        ]
        stacked[key] = PackedTensor.concat(parts)
    return stacked


class RolloutReassembler:
    """Receipts -> verified rows -> N(+segments)-row publish, off the generation hot path."""

    def __init__(
        self,
        dp_client: Any,
        *,
        partition_id: str,
        staging_partition: str,
        pad_token_id: int,
        max_seq_len: int,
        router_replay_enabled: bool = False,
        defer_routed_experts_to_policy: bool = False,
        capture_media: bool = False,
        segment_rows_enabled: bool = False,
        max_rows_per_rollout: int = 1,
        include_summary_rows: bool = True,
    ) -> None:
        self._dp_client = dp_client
        self._partition_id = partition_id
        # Whether the staging partition carries media columns (setup's
        # ``token_capture.enabled and processor is not None``). Text-only runs
        # never read media columns; media-enabled runs must find them.
        self._capture_media = capture_media
        self._pad_token_id = int(pad_token_id)
        self._max_seq_len = int(max_seq_len)
        self._router_replay_enabled = router_replay_enabled
        self._defer_routed_experts_to_policy = defer_routed_experts_to_policy
        if self._defer_routed_experts_to_policy and not self._router_replay_enabled:
            raise ValueError(
                "defer_routed_experts_to_policy requires router replay to be enabled"
            )
        self._segment_rows_enabled = bool(segment_rows_enabled)
        self._max_rows_per_rollout = int(max_rows_per_rollout)
        if self._max_rows_per_rollout < 1:
            raise ValueError(
                f"max_rows_per_rollout must be >= 1, got {max_rows_per_rollout}"
            )
        if self._segment_rows_enabled and self._max_rows_per_rollout < 2:
            raise ValueError(
                "segment_rows_enabled requires max_rows_per_rollout >= 2 "
                f"(canonical row plus at least one segment), got {max_rows_per_rollout}"
            )
        self._include_summary_rows = bool(include_summary_rows)
        self._warned_missing_linearize_all = False
        self._staging_partition = staging_partition
        # (num_moe_layers, topk), learned from the first rebuilt row that
        # carries routes; placeholder-only groups need it to shape their
        # sentinel tensors consistently with the model.
        self._routed_dims: Optional[tuple[int, int]] = None
        self._source = TQTokenSource(
            dp_client, staging_partition=staging_partition, capture_media=capture_media
        )
        # The sink's clear() is the staging-partition delete; no staging
        # writes happen here.
        self._staging = TQTokenSink(
            dp_client, staging_partition=staging_partition, capture_media=capture_media
        )

    # ── per rollout ─────────────────────────────────────────────────────────

    def _resolve_linearize_all(self) -> Optional[Callable[..., Any]]:
        """Return Gym's ``verify_and_linearize_all`` when segment rows are on.

        Looked up at call time (not import time) so a Gym without the function
        degrades to single-chain behaviour with a warning instead of failing
        the finalizer, and so tests can monkeypatch the module attribute.
        """
        if not self._segment_rows_enabled:
            return None
        import nemo_gym.token_id_capture.staging.rebuild as rebuild_module

        linearize_all = getattr(rebuild_module, "verify_and_linearize_all", None)
        if linearize_all is None and not self._warned_missing_linearize_all:
            self._warned_missing_linearize_all = True
            message = (
                "token_capture.segment_rows.enabled=true but the installed "
                "nemo_gym has no verify_and_linearize_all "
                "(nemo_gym.token_id_capture.staging.rebuild); falling back to "
                "single-chain finalization -- no segment rows will be published."
            )
            warnings.warn(message, RuntimeWarning, stacklevel=2)
            print(f"  finalize: WARNING {message}", flush=True)
        return linearize_all

    def finalize_rollout(
        self, rollout_id: str, receipt: Optional[dict[str, Any]], *, reward: float
    ) -> list[FinalizedRollout]:
        """Verify one receipt against its staged rows and linearize its chain(s).

        Returns one ``FinalizedRollout`` per published row: the canonical row
        first (the receipt's terminal chain), then -- only with segment rows
        enabled -- up to ``max_rows_per_rollout - 1`` extra chains in Gym's
        order. Never raises for rollout-level problems: every rejection
        returns exactly one invalid row (no extras) whose reason feeds the
        metrics; the group publisher substitutes a placeholder.
        """
        # Deferred: nemo_gym is an optional extra absent in non-gym runs.
        from nemo_gym.token_id_capture.staging.rebuild import (
            RebuildError,
            ReceiptVerificationError,
            verify_and_linearize,
        )
        from nemo_gym.token_id_capture.staging.records import RolloutReceipt

        def rejected(reason: str, staging_keys: list[str]) -> list[FinalizedRollout]:
            return [
                FinalizedRollout(
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
            ]

        if receipt is None:
            return rejected("missing_receipt", [])
        try:
            parsed = RolloutReceipt.model_validate(receipt)
        except (TypeError, ValueError) as error:
            return rejected(f"invalid_receipt:{error}", [])
        staging_keys = [record.staging_key for record in parsed.manifest]
        if parsed.rollout_id != rollout_id:
            return rejected(f"identity_mismatch:{parsed.rollout_id}", staging_keys)
        if parsed.failure_reason is not None:
            return rejected(f"rollout_failed:{parsed.failure_reason}", staging_keys)
        if parsed.capture_poisoned:
            return rejected("capture_poisoned", staging_keys)
        # An unpoisoned receipt must name a terminal call that is in the manifest
        # (RolloutReceipt validators), so a valid receipt here is never empty.
        if len(set(staging_keys)) != len(staging_keys):
            return rejected(
                "duplicate_staging_key",
                list(dict.fromkeys(staging_keys)),
            )
        records_by_call = {record.model_call_id: record for record in parsed.manifest}
        if len(records_by_call) != len(parsed.manifest):
            return rejected("duplicate_manifest_call_id", staging_keys)

        fetch_fragments = (
            self._router_replay_enabled and not self._defer_routed_experts_to_policy
        )
        try:
            fetched = self._source.fetch_for_finalization(
                staging_keys, include_route_fragments=fetch_fragments
            )
        except KeyError as error:
            return rejected(f"missing_staging_row:{error}", staging_keys)
        except (TypeError, ValueError) as error:
            return rejected(f"invalid_staging_row:{error}", staging_keys)
        fetched_by_call = {}
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

        # All base token/digest/lineage/terminal semantics belong to Gym; the
        # finalizer never re-verifies them.
        snapshots = [item.snapshot for item in fetched]
        linearize_all = self._resolve_linearize_all()
        chain_rows: list[Any]
        skipped_reasons: list[str] = []
        boundary_roots = 0
        try:
            if linearize_all is not None:
                linearized = linearize_all(parsed, snapshots)
                chain_rows = list(linearized.rows)
                skipped_reasons = [
                    str(getattr(chain, "reason", ""))
                    for chain in (getattr(linearized, "skipped", None) or [])
                ]
                boundary_roots = int(getattr(linearized, "num_boundary_roots", 0))
            else:
                chain_rows = [verify_and_linearize(parsed, snapshots)]
        except (
            KeyError,
            ValueError,
            TypeError,
            ReceiptVerificationError,
            RebuildError,
            NotImplementedError,
        ) as error:
            return rejected(f"rebuild_failed:{error}", staging_keys)
        if not chain_rows:
            return rejected("rebuild_failed:no_terminal_chain", staging_keys)
        weight_versions = [record.weight_version for record in parsed.manifest]
        min_wv, max_wv = min(weight_versions), max(weight_versions)

        # Summary rows: drop the compaction_summary chains before the cap when
        # they are not wanted, so the cap budget goes to the other chains.
        segments_skipped_summary = 0
        if self._segment_rows_enabled and not self._include_summary_rows:
            kept = [chain_rows[0]]
            for chain in chain_rows[1:]:
                if getattr(chain, "chain_kind", None) == TRACE_KIND_COMPACTION_SUMMARY:
                    segments_skipped_summary += 1
                else:
                    kept.append(chain)
            chain_rows = kept
        # Cap: the canonical row plus at most max_rows_per_rollout-1 extras,
        # in Gym's order (terminal first, then the chains on the terminal's
        # compaction sequence in segment order, then other roots by admission
        # time).
        max_extra = self._max_rows_per_rollout - 1 if self._segment_rows_enabled else 0
        segments_dropped_by_cap = max(0, len(chain_rows) - 1 - max_extra)
        chain_rows = chain_rows[: 1 + max_extra]

        rows: list[FinalizedRollout] = []
        segments_rejected = 0
        for trace_idx, chain in enumerate(chain_rows):
            built, failure = self._build_chain_row(
                rollout_id=rollout_id,
                chain=chain,
                trace_idx=trace_idx,
                reward=reward,
                staging_keys=staging_keys,
                records_by_call=records_by_call,
                fetched=fetched,
                fetched_by_call=fetched_by_call,
                min_wv=min_wv,
                max_wv=max_wv,
            )
            if built is None:
                if trace_idx == 0:
                    # The terminal chain is the rollout's canonical row; its
                    # rejection is the rollout's rejection (placeholder).
                    return rejected(failure or "route_assembly:unknown", staging_keys)
                segments_rejected += 1
                print(
                    f"  finalize: rollout {rollout_id} segment row {trace_idx} "
                    f"rejected ({failure}) -- dropped",
                    flush=True,
                )
                continue
            rows.append(built)
        # Extra rows are numbered by publication order so the ids are dense.
        renumbered: list[FinalizedRollout] = []
        for publish_idx, row in enumerate(rows):
            if publish_idx == 0:
                renumbered.append(
                    _replace_dataclass(
                        row,
                        num_chains=len(rows),
                        chains_skipped_ambiguous=sum(
                            1 for r in skipped_reasons if r == _SKIP_AMBIGUOUS_LEAF
                        ),
                        chains_skipped_other=sum(
                            1 for r in skipped_reasons if r != _SKIP_AMBIGUOUS_LEAF
                        ),
                        boundary_roots=boundary_roots,
                        segments_dropped_by_cap=segments_dropped_by_cap,
                        segments_rejected=segments_rejected,
                        segments_skipped_summary=segments_skipped_summary,
                    )
                )
            else:
                renumbered.append(
                    _replace_dataclass(row, trace_in_rollout_idx=publish_idx)
                )
        return renumbered

    def _build_chain_row(
        self,
        *,
        rollout_id: str,
        chain: Any,
        trace_idx: int,
        reward: float,
        staging_keys: list[str],
        records_by_call: dict[str, Any],
        fetched: list[Any],
        fetched_by_call: dict[str, Any],
        min_wv: int,
        max_wv: int,
    ) -> tuple[Optional[FinalizedRollout], Optional[str]]:
        """Turn one verified Gym chain into a row; ``(None, reason)`` on failure."""
        # Media: the presence flags fetched with the base columns say which
        # of this chain's calls carry pixels; one batched read pulls them,
        # after token verification so a rejected rollout never moves pixels.
        media, media_failure = self._resolve_media(chain, fetched_by_call)
        if media_failure is not None:
            return None, media_failure
        route_plan = None
        routed_experts: Optional[torch.Tensor] = None
        link_spans = tuple(
            (str(call_id), int(carry_len), int(generation_len))
            for call_id, carry_len, generation_len in (
                getattr(chain, "link_spans", None) or []
            )
        )
        # Cleanup ownership (staging keys / plan cleanup keys) rides the
        # canonical row only; extra rows share the same staged calls.
        owns_cleanup = trace_idx == 0
        if self._router_replay_enabled:
            # One plan construction for both modes: join Gym's link spans and
            # extras commitments with the fetch's staging keys and route
            # lengths. Cleanup keys cover the whole manifest; off-chain rows
            # stay cleanup-owned but produce no spans.
            commitments_by_call = {
                commitment.model_call_id: commitment
                for commitment in chain.extras_commitments
            }
            route_spans: list[RouteSpan] = []
            seen_span_call_ids: set[str] = set()
            for call_id, carry_len, generation_len in link_spans:
                if call_id in seen_span_call_ids:
                    return None, f"duplicate_route_span:{call_id}"
                seen_span_call_ids.add(call_id)
                record = records_by_call.get(call_id)
                item = fetched_by_call.get(call_id)
                commitment = commitments_by_call.get(call_id)
                if record is None or item is None or commitment is None:
                    return None, f"route_span_identity:{call_id}"
                if item.routed_len not in (0, record.delta_len):
                    return None, f"routed_len_mismatch:{call_id}"
                if generation_len < 0 or generation_len > record.delta_len:
                    return None, f"route_generation_span_mismatch:{call_id}"
                if carry_len < 0:
                    return None, f"route_carry_span_mismatch:{call_id}"
                route_spans.append(
                    RouteSpan(
                        staging_key=record.staging_key,
                        carry_len=int(carry_len),
                        generation_len=int(generation_len),
                        staged_route_len=item.routed_len,
                        extras_digest_version=commitment.extras_digest_version,
                        extras_digest=commitment.extras_digest,
                    )
                )
            if sum(span.carry_len + span.generation_len for span in route_spans) != len(
                chain.token_ids
            ):
                return None, "route_span_length_mismatch"
            plan = RouteAssemblyPlan(
                schema_version=ROUTE_PLAN_SCHEMA_VERSION,
                staging_partition=self._staging_partition,
                spans=tuple(route_spans),
                cleanup_staging_keys=tuple(staging_keys) if owns_cleanup else (),
                expected_token_length=len(chain.token_ids),
            )
            try:
                validate_route_plan(plan)
            except (TypeError, ValueError) as error:
                return None, f"invalid_route_plan:{error}"
            # Both modes carry the constructed plan on the row; only deferred
            # mode publishes it (direct mode executes it eagerly and the
            # published row carries the assembled tensor instead).
            route_plan = plan
            if not self._defer_routed_experts_to_policy:
                routed_experts, failure = self._execute_direct_plan(plan, fetched)
                if failure is not None:
                    return None, f"route_assembly:{failure}"

        return (
            FinalizedRollout(
                rollout_id=rollout_id,
                valid=True,
                rejection_reason=None,
                token_ids=list(chain.token_ids),
                token_mask=list(chain.token_mask),
                logprobs=list(chain.logprobs),
                prompt_len=int(chain.prompt_len),
                reward=reward,
                staging_keys=list(staging_keys) if owns_cleanup else [],
                min_wv=min_wv,
                max_wv=max_wv,
                routed_experts=routed_experts,
                route_plan=route_plan,
                trace_in_rollout_idx=trace_idx,
                trace_kind=str(getattr(chain, "chain_kind", TRACE_KIND_TERMINAL)),
                segment_index=int(getattr(chain, "segment_index", 0)),
                boundary_parent_call_id=getattr(chain, "boundary_parent_call_id", None),
                chain_index=int(getattr(chain, "chain_index", trace_idx)),
                link_spans=link_spans,
                media=media,
            ),
            None,
        )

    def _resolve_media(
        self,
        row: Any,
        fetched_by_call: dict[str, FetchedStagedCall],
    ) -> tuple[Optional[dict[str, PackedTensor]], Optional[str]]:
        """Read and validate the media columns along the terminal chain.

        Each call row holds only the media new to that call (the worker
        slices at the parent chain's item count), so the chain's parts are
        concatenated in order, like the token deltas. Returns
        ``(media, rejection_reason)``. Text-only partitions and text rollouts
        return ``(None, None)`` without touching storage; media-carrying
        rollouts cost exactly one batched tensor read. The media columns live
        on the call rows themselves, so cleanup needs no extra key.
        """
        if not self._capture_media:
            return None, None
        items: list[FetchedStagedCall] = []
        for call_id, _carry_len, _generation_len in row.link_spans:
            item = fetched_by_call.get(call_id)
            if item is None:
                return None, f"media_chain_identity:{call_id}"
            if item.media_present:
                items.append(item)
        if not items:
            return None, None
        if len({item.media_has_frames for item in items}) != 1:
            return None, "media_chain_incompatible:mixed image and video calls"
        try:
            parts = self._source.fetch_media(items)
        except (KeyError, TypeError, ValueError) as error:
            return None, f"invalid_media_columns:{error}"
        try:
            combined = _concat_media(parts)
            media = _trainer_media(combined)
        except (TypeError, ValueError, RuntimeError) as error:
            return None, f"media_chain_incompatible:{error}"
        return media, None

    def _execute_direct_plan(
        self,
        plan: RouteAssemblyPlan,
        fetched: list[Any],
    ) -> tuple[Optional[torch.Tensor], Optional[str]]:
        """Run the shared executor eagerly with locally fetched fragments.

        Returns ``(None, None)`` when the rollout staged no routes at all —
        the group tensor build fills those rows with sentinels, exactly like
        a deferred row whose plan is all-sentinel.
        """
        fragments: dict[str, RouteFragment] = {
            item.staging_key: item.fragment
            for item in fetched
            if item.fragment is not None
        }
        if not any(span.staged_route_len > 0 for span in plan.spans):
            return None, None
        if fragments:
            # Direct mode has no policy model in-process; the fragments'
            # own (num_moe_layers, topk) is the learned-dims heuristic, and
            # the trainer's model-shape check remains authoritative.
            first = next(iter(fragments.values())).routes
            if first.dim() != 3:
                return None, "fragment_rank"
            self._routed_dims = (int(first.shape[1]), int(first.shape[2]))
        if self._routed_dims is None:
            return None, "missing_fragment"
        return execute_route_plan(
            plan,
            fragments,
            dims=self._routed_dims,
            canonical_len=plan.expected_token_length,
        )

    # ── per group ───────────────────────────────────────────────────────────

    def finalize_group(
        self,
        group_id: str,
        rollout_ids: list[str],
        receipts: list[Optional[dict[str, Any]]],
        rewards: list[float],
        *,
        mask_sample: list[bool],
        fallback_weight_version: int,
        prompt_idx: int,
        loss_multiplier: float = 1.0,
        canonical_sample_ids: Optional[list[str]] = None,
    ) -> FinalizedGroup:
        """Publish the N canonical rows (plus any segment rows) for one prompt group.

        Row layout: positions ``0..N-1`` are the canonical rows of rollouts
        ``0..N-1`` under ``canonical_sample_ids`` (one per rollout, placeholder
        when rejected); positions ``N..`` are extra segment rows in
        ``(rollout, trace)`` order under ``{canonical_id}_t{j}``. Without
        ``segment_rows`` exactly N rows are published, as before.

        Blocking (TQ round trips); run via ``asyncio.to_thread`` from the
        dispatch task. ``fallback_weight_version`` stamps a group none of
        whose rollouts produced a valid row (placeholder-only groups still
        need a staleness tag). ``mask_sample`` is the per-rollout
        advantage-stage flag the native ``pack_payload`` path emits from each
        ``Completion``; it rides along unchanged (replicated to a rollout's
        extra rows) so the train pump's environment masking reads the same
        field on both paths (placeholder rows already train nothing through
        ``sample_mask`` 0). ``loss_multiplier`` supplies the dataset-level
        weight for every valid row, matching the ordinary
        ``record_to_train_batch`` path. ``truncated`` is not carried from the
        dispatcher -- the receipt path has no real tokens to measure it from
        at dispatch time -- so it is computed here instead, per row, from the
        rebuilt length against ``max_seq_len``.
        """
        assert len(rollout_ids) == len(receipts) == len(rewards) == len(mask_sample), (
            "rollout_ids, receipts, rewards, and mask_sample must be parallel"
        )
        if canonical_sample_ids is None:
            canonical_sample_ids = rollout_ids
        assert len(canonical_sample_ids) == len(rollout_ids), (
            "canonical_sample_ids must be one per rollout"
        )
        _group_t0 = time.perf_counter()
        per_rollout = [
            self.finalize_rollout(rollout_id, receipt, reward=reward)
            for rollout_id, receipt, reward in zip(rollout_ids, receipts, rewards)
        ]
        _rollouts_ms = (time.perf_counter() - _group_t0) * 1000.0
        n_rollouts = len(per_rollout)
        canonical_rows = [rows[0] for rows in per_rollout]
        extra_entries = [
            (rollout_idx, row)
            for rollout_idx, rows in enumerate(per_rollout)
            for row in rows[1:]
        ]
        # Canonical block first, then extras in (rollout, trace) order.
        rows: list[FinalizedRollout] = canonical_rows + [row for _, row in extra_entries]
        row_rollout_idx = list(range(n_rollouts)) + [i for i, _ in extra_entries]
        valid_rows = [row for row in canonical_rows if row.valid]
        all_valid_rows = [row for row in rows if row.valid]
        staging_keys = [key for row in rows for key in row.staging_keys]
        metrics = {
            "finalize/invalid_row_rate": 1.0 - len(valid_rows) / n_rollouts,
            "finalize/calls_per_rollout": (
                sum(len(row.staging_keys) for row in canonical_rows) / n_rollouts
            ),
            # Fraction of valid rows that carry captured media. 1.0 on a VLM
            # run is the signal that the learner trains on captured pixels
            # rather than text-only rows; text runs report 0.0.
            "finalize/media_row_rate": (
                sum(1 for row in valid_rows if row.media) / len(valid_rows)
                if valid_rows
                else 0.0
            ),
        }
        # Segment-row accounting (all zeros / 1.0 without segment_rows).
        rows_per_rollout = [len(chain_rows) for chain_rows in per_rollout]
        metrics["finalize/rows_per_rollout_mean"] = sum(rows_per_rollout) / n_rollouts
        metrics["finalize/rows_per_rollout_max"] = float(max(rows_per_rollout))
        metrics["finalize/rollouts_with_segments"] = float(
            sum(1 for count in rows_per_rollout if count > 1)
        )
        metrics["finalize/segment_rows"] = float(len(extra_entries))
        for kind, count in Counter(row.trace_kind for _, row in extra_entries).items():
            metrics[f"finalize/segment_rows_by_kind_{kind}"] = float(count)
        metrics["finalize/chains_skipped_ambiguous"] = float(
            sum(row.chains_skipped_ambiguous for row in canonical_rows)
        )
        metrics["finalize/boundary_roots"] = float(
            sum(row.boundary_roots for row in canonical_rows)
        )
        metrics["finalize/segments_dropped_by_cap"] = float(
            sum(row.segments_dropped_by_cap for row in canonical_rows)
        )
        metrics["finalize/segment_rows_rejected"] = float(
            sum(row.segments_rejected for row in canonical_rows)
        )
        metrics["finalize/segment_rows_skipped_summary"] = float(
            sum(row.segments_skipped_summary for row in canonical_rows)
        )
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
        # Receipts whose manifest never parsed carry no method (None) and
        # fall in no bucket. Method list is derived from Gym's own type
        # rather than hand-copied, so a new resolution method Gym adds gets a
        # bucket automatically instead of silently missing from these
        # metrics; the annotation is ``Literal[...] | None``, so unwrap the
        # Literal and skip the None member.
        from typing import Literal, get_args, get_origin

        from nemo_gym.token_id_capture.staging.records import RolloutReceipt

        terminal_selection_methods = tuple(
            method
            for member in get_args(
                RolloutReceipt.model_fields["terminal_selection"].annotation
            )
            if get_origin(member) is Literal
            for method in get_args(member)
        )
        for method in terminal_selection_methods:
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
        rejection_reasons: Counter[str] = Counter()
        for row in canonical_rows:
            if not row.valid:
                reason_bucket = (row.rejection_reason or "unknown").split(":", 1)[0]
                rejection_reasons[reason_bucket] += 1
                print(
                    f"  finalize: rollout {row.rollout_id} rejected "
                    f"({row.rejection_reason}) — placeholder",
                    flush=True,
                )
        for reason_bucket, count in rejection_reasons.items():
            metrics[f"finalize/capture_failure_reason_{reason_bucket}_count"] = float(
                count
            )

        group_min_wv = min(
            (r.min_wv for r in valid_rows if r.min_wv is not None),
            default=fallback_weight_version,
        )
        group_max_wv = max(
            (r.max_wv for r in valid_rows if r.max_wv is not None),
            default=fallback_weight_version,
        )

        _tensorize_t0 = time.perf_counter()
        # Placeholders borrow a valid sibling's prompt ids so per-prompt
        # baselines group correctly; an all-placeholder group uses a single
        # pad token (its rows all carry sample_mask 0 and never train). Extra
        # segment rows carry the same group prompt as their rollout's
        # canonical row (identical tensor), so grouping by prompt tokens
        # keeps the whole group -- canonical and segment rows -- together.
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
        # Extra rows carry their rollout's reward / mask_sample verbatim.
        rewards_t = torch.tensor(
            [rewards[i] for i in row_rollout_idx], dtype=torch.float32
        )
        mask_sample_rows = [bool(mask_sample[i]) for i in row_rollout_idx]
        for r, row in enumerate(rows):
            if not row.valid:
                continue
            length = len(row.token_ids)
            input_ids[r, :length] = torch.tensor(row.token_ids, dtype=torch.int64)
            token_mask[r, :length] = torch.tensor(row.token_mask, dtype=torch.float32)
            logprobs[r, :length] = torch.tensor(row.logprobs, dtype=torch.float32)
            sample_mask[r] = float(loss_multiplier)

        train_batch: dict[str, Any] = {
            "input_ids": input_ids,
            "input_lengths": lengths,
            "generation_logprobs": logprobs,
            "token_mask": token_mask,
            "sample_mask": sample_mask,
            "prompt_ids_for_adv": prompt_ids_for_adv,
            "total_reward": rewards_t,
            MASK_SAMPLE: torch.tensor(mask_sample_rows, dtype=torch.bool),
            TRUNCATED: torch.tensor(
                [seq_len == self._max_seq_len for seq_len in seq_lens],
                dtype=torch.bool,
            ),
        }
        if self._router_replay_enabled and not self._defer_routed_experts_to_policy:
            has_routed_row = any(r.valid and r.routed_experts is not None for r in rows)
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
                    canonical_output_tokens=0,
                    metrics=metrics,
                    dropped=True,
                    drop_reason=(
                        "router replay on, no rollout carried routed_experts, "
                        "and (L, K) is unknown yet"
                    ),
                    valid_row_count=0,
                    total_row_count=0,
                    extra_row_count=0,
                )
            train_batch["routed_experts"] = self._build_routed_experts_tensor(
                rows, max_len=max_len, metrics=metrics
            )
        # Media rides the same packed/tagged transport as the token-echo path
        # (pack_payload encodes PackedTensor fields and mints row-shape tags).
        train_batch.update(_media_fields_for_group(rows))
        # Canonical ids for the canonical block; ``{canonical}_t{j}`` for the
        # extras (== make_sample_id(group, i, j) when the canonical id is
        # ``{group}_g{i}``).
        expected_sample_ids = list(canonical_sample_ids) + [
            f"{canonical_sample_ids[i]}_t{row.trace_in_rollout_idx}"
            for i, row in extra_entries
        ]
        sample_ids, fields, tags = pack_payload(
            train_batch,
            weight_version=group_min_wv,
            group_id=group_id,
            prompt_idx=prompt_idx,
            sample_ids=expected_sample_ids,
        )
        # Per-row placement tags for the advantage stage / stats.
        # rows_in_rollout is the rollout's total published rows (canonical
        # included, 1 on placeholders) so a consumer can assert it holds the
        # whole rollout, not merely no orphan.
        for tag, row, rollout_idx in zip(tags, rows, row_rollout_idx):
            tag["rollout_local_idx"] = int(rollout_idx)
            tag["trace_in_rollout_idx"] = int(row.trace_in_rollout_idx)
            tag["trace_kind"] = row.trace_kind if row.valid else TRACE_KIND_PLACEHOLDER
            tag["segment_index"] = int(row.segment_index)
            tag["rows_in_rollout"] = int(rows_per_rollout[rollout_idx])
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
                1 for row in all_valid_rows if row.route_plan and row.route_plan.spans
            )
            if all_valid_rows:
                metrics["finalize/routed_experts_row_coverage"] = (
                    valid_route_rows / len(all_valid_rows)
                )
        assert sample_ids[:n_rollouts] == list(canonical_sample_ids), (
            "canonical sample ids must equal the stable logical rollout ids: "
            f"{sample_ids[:n_rollouts]} != {canonical_sample_ids}"
        )
        for sample_id, rollout_idx in zip(
            sample_ids[n_rollouts:], row_rollout_idx[n_rollouts:]
        ):
            canonical = canonical_sample_ids[rollout_idx]
            assert sample_id.startswith(f"{canonical}_t"), (
                f"segment row id {sample_id!r} does not extend its canonical id "
                f"{canonical!r}"
            )
            parsed = try_parse_sample_id(sample_id)
            parsed_canonical = try_parse_sample_id(canonical)
            if parsed is not None and parsed_canonical is not None:
                assert (
                    make_sample_id(parsed[0], parsed[1]) == canonical
                    and parsed[2] >= 1
                ), f"segment row id {sample_id!r} does not parse back to {canonical!r}"
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
            fields=cast(list[str], list(fields.keys())),
            sequence_lengths=[int(s) for s in lengths.tolist()],
            tags=[dict(t) for t in tags],
        )
        return FinalizedGroup(
            meta=meta,
            group_min_wv=group_min_wv,
            group_max_wv=group_max_wv,
            staging_keys=(staging_keys if self._defer_routed_experts_to_policy else []),
            canonical_output_tokens=sum(
                int(mask) for row in all_valid_rows for mask in row.token_mask
            ),
            metrics=metrics,
            valid_row_count=len(valid_rows),
            total_row_count=n_rollouts,
            extra_row_count=len(extra_entries),
        )

    # ── internals ───────────────────────────────────────────────────────────

    def _build_routed_experts_tensor(
        self,
        rows: list[FinalizedRollout],
        *,
        max_len: int,
        metrics: dict[str, float],
    ) -> torch.Tensor:
        """[n, max_len, L, K] int16 routes for every published row; sentinel elsewhere.

        ``rows`` is the full publication order (canonical block, then segment
        rows). Padding, placeholder rows, and valid rows whose rebuild carried
        no routes are all-sentinel: Megatron's replay falls back to its own
        router for exactly those positions. (L, K) is learned from the first
        rebuilt row that carries routes and cached for placeholder-only
        groups; a group arriving before any routed row has been seen cannot
        be shaped and fails loudly (unreachable once the first real rollout
        of the run finalizes).
        """
        for row in rows:
            if row.valid and row.routed_experts is not None:
                self._routed_dims = (
                    int(row.routed_experts.shape[1]),
                    int(row.routed_experts.shape[2]),
                )
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
            ROUTE_MISSING_SENTINEL,
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
            if row.routed_experts is None:
                sentinel_tokens += len(row.token_ids)
                continue
            rows_with_routes += 1
            row_routes = row.routed_experts
            if row_routes.shape != (len(row.token_ids), num_moe_layers, topk):
                raise RuntimeError(
                    "rebuilt routed_experts shape "
                    f"{tuple(row_routes.shape)} does not match "
                    f"({len(row.token_ids)}, {num_moe_layers}, {topk}) for "
                    f"rollout {row.rollout_id} (trace {row.trace_in_rollout_idx})"
                )
            routed[i, : row_routes.shape[0]] = row_routes
            sentinel_tokens += int(
                row_routes.eq(ROUTE_MISSING_SENTINEL).all(-1).all(-1).sum().item()
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


def _replace_dataclass(row: FinalizedRollout, **changes: Any) -> FinalizedRollout:
    """``dataclasses.replace`` spelled out (keeps the frozen row immutable)."""
    from dataclasses import replace

    return replace(row, **changes)
