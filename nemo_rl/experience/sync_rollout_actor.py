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
"""Sync GRPO rollout actor — sibling of ``async_utils``.

Houses :class:`SyncRolloutActor`, the Ray actor that owns the multi-turn
rollout loop AND the post-rollout flatten / mask / prompt extraction /
reward shaping / baseline-std for a sync GRPO step. The driver dispatches
a per-step prompt batch + uids; the actor runs ``run_multi_turn_rollout``
(or async / nemo_gym variants), then writes the bulk schema to TQ via
:func:`nemo_rl.data_plane.column_io.kv_first_write`. Only a ``KVBatchMeta``
and a small per-sample ``driver_carry`` dict (rewards, masks, lengths,
baseline/std, prompt_ids_for_adv) cross back to the driver via Ray.

**Goal — rollout 1-hop put**: bulk tensors (input_ids, output_ids,
attention_mask, position_ids, multi_modal_inputs, generation_logprobs,
token_mask) stay actor-side until ``put_samples``, then live only in
TQ. Driver never holds these bytes between rollout finish and train
fan-out.

The actor is the sync counterpart to
:class:`nemo_rl.algorithms.async_utils.AsyncTrajectoryCollector`. It
intentionally does not buffer or stream — sync GRPO consumes the whole
step batch in one call.
"""

from __future__ import annotations

import json
import resource
import time
import uuid
from collections import Counter
from copy import deepcopy
from typing import Any, Optional

import numpy as np
import ray
import torch

from nemo_rl.data_plane.column_io import kv_first_write
from nemo_rl.data_plane.interfaces import KVBatchMeta
from nemo_rl.data_plane.schema import ROUTED_EXPERTS_FIELD
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.experience.rollouts import (
    EffortLevelsConfig,
    get_nemo_gym_thinking_tags,
    run_async_multi_turn_rollout,
    run_async_nemo_gym_rollout,
    run_multi_turn_rollout,
)
from nemo_rl.experience.rollout_writer import (
    assemble_staged_batch,
    compare_shadow_candidate,
    mint_rollout_context,
    mint_rollout_id,
    persist_rollout_manifest,
    persist_rollout_perf_metrics,
)
from nemo_rl.models.generation.interfaces import GenerationInterface
from nemo_rl.utils.r3_trace import trace_rollout_payload

# Carry keys producible by the rollout actor only when the caller opts in.
# These are np.ndarray(object) per-row arrays from decompose_message_log; the
# default driver_carry omits them because BatchedDataDict.select_indices on
# the training/dynamic-sampling path only handles tensors/lists. Validation
# requests them explicitly to print per-sample message logs.
OPT_IN_CARRY_KEYS: tuple[str, ...] = ("turn_roles", "turn_contents")


def _prompt_ids_from_finalized_batch(
    *,
    input_ids: torch.Tensor,
    prompt_lengths: torch.Tensor,
    group_ids: list[str],
    validity_mask: torch.Tensor,
    pad_token_id: int,
) -> torch.Tensor:
    """Recover prompt grouping IDs from verified canonical rows.

    Direct mode deliberately removes token IDs from the terminal Gym result.
    The finalizer still has the verified canonical sequence, so use its prompt
    prefix for every surviving row. A rejected row inherits the prefix of a
    valid sibling in the same rollout group. If an entire group is rejected,
    its rows remain excluded by ``validity_mask`` and receive a nonempty pad
    sentinel solely to keep downstream grouping operations well-defined.
    """
    batch_size = int(input_ids.shape[0])
    if input_ids.ndim != 2:
        raise ValueError(
            f"input_ids must be rank 2, got shape={tuple(input_ids.shape)}"
        )
    if prompt_lengths.shape != (batch_size,):
        raise ValueError(
            "prompt_lengths must have one value per row; "
            f"got shape={tuple(prompt_lengths.shape)} for batch_size={batch_size}"
        )
    if validity_mask.shape != (batch_size,):
        raise ValueError(
            "validity_mask must have one value per row; "
            f"got shape={tuple(validity_mask.shape)} for batch_size={batch_size}"
        )
    if len(group_ids) != batch_size:
        raise ValueError(
            f"group_ids must have {batch_size} entries, got {len(group_ids)}"
        )

    verified_prompts: dict[str, torch.Tensor] = {}
    for index, group_id in enumerate(group_ids):
        if not bool(validity_mask[index]):
            continue
        prompt_length = int(prompt_lengths[index])
        if prompt_length <= 0 or prompt_length > input_ids.shape[1]:
            raise ValueError(
                f"invalid prompt length {prompt_length} for finalized row {index} "
                f"with width {input_ids.shape[1]}"
            )
        prompt = input_ids[index, :prompt_length]
        previous = verified_prompts.setdefault(group_id, prompt)
        if not torch.equal(previous, prompt):
            raise ValueError(
                f"finalized rows in rollout group {group_id!r} have different prompts"
            )

    target_width = max(
        (int(prompt.shape[0]) for prompt in verified_prompts.values()), default=1
    )
    result = torch.full(
        (batch_size, target_width),
        pad_token_id,
        dtype=input_ids.dtype,
        device=input_ids.device,
    )
    for index, group_id in enumerate(group_ids):
        prompt = verified_prompts.get(group_id)
        if prompt is not None:
            result[index, : prompt.shape[0]] = prompt
    return result


def _flatten_rollout_message_log_for_tq(
    message_logs: list[Any],
    prompt_lengths: torch.Tensor,
    *,
    pad_token_id: int,
    make_sequence_length_divisible_by: int,
) -> tuple[BatchedDataDict[Any], torch.Tensor, BatchedDataDict[Any]]:
    """Prepare rollout message logs for the TQ payload and driver carry."""
    from nemo_rl.algorithms.grpo import (
        add_grpo_token_loss_masks_and_generation_logprobs,
        extract_initial_prompt_messages,
    )
    from nemo_rl.data.llm_message_utils import batched_message_log_to_flat_message

    pad = {"pad_value_dict": {"token_ids": pad_token_id}}
    prompt_message_logs = extract_initial_prompt_messages(
        message_logs,
        prompt_lengths,
    )
    prompt_flat, _ = batched_message_log_to_flat_message(
        prompt_message_logs,
        **pad,
    )

    add_grpo_token_loss_masks_and_generation_logprobs(message_logs)
    flat, input_lengths = batched_message_log_to_flat_message(
        message_logs,
        **pad,
        make_sequence_length_divisible_by=make_sequence_length_divisible_by,
    )
    return flat, input_lengths, prompt_flat


def _attach_rollout_contexts(
    input_batch: BatchedDataDict[Any],
    *,
    rollout_ids: list[str],
    group_ids: list[str],
    weight_version: int,
    secret: bytes,
    ttl_s: float,
) -> BatchedDataDict[Any]:
    """Return a batch copy with signed contexts in Gym request metadata.

    Each row carries the canonical ``rollout_id`` twice during migration: as
    the signed context's ``sample_id`` (legacy alias) and as the bare
    ``nemo_rl_rollout_id`` field. Consumers that see both must reject the
    request unless they are equal.
    """
    rows = deepcopy(input_batch["extra_env_info"])
    if len(rows) != len(rollout_ids) or len(rows) != len(group_ids):
        raise ValueError(
            "extra_env_info, rollout_ids, and group_ids must have equal lengths; "
            f"got {len(rows)}, {len(rollout_ids)}, and {len(group_ids)}"
        )
    for row, rollout_id, group_id in zip(rows, rollout_ids, group_ids):
        params = row["responses_create_params"]
        metadata = dict(params.get("metadata") or {})
        extra_body_raw = metadata.get("extra_body", "{}")
        try:
            extra_body = json.loads(extra_body_raw)
        except (TypeError, json.JSONDecodeError) as error:
            raise ValueError("responses metadata.extra_body must be JSON") from error
        context = mint_rollout_context(
            sample_id=rollout_id,
            group_id=group_id,
            weight_version=weight_version,
            secret=secret,
            ttl_s=ttl_s,
        )
        extra_body["nemo_rl_rollout_context"] = context.to_dict()
        extra_body["nemo_rl_rollout_id"] = rollout_id
        metadata["extra_body"] = json.dumps(extra_body, separators=(",", ":"))
        params["metadata"] = metadata
    copied = BatchedDataDict(dict(input_batch))
    copied["extra_env_info"] = rows
    return copied


@ray.remote  # pragma: no cover
class SyncRolloutActor:
    """Per-step rollout dispatcher.

    Runs: rollout + flatten + mask + prompt extraction + baseline/std + TQ put.
    Returns ``(meta, driver_carry, rollout_metrics, gen_metrics)``.

    Lifecycle: one instance per ``grpo_train_sync`` invocation. The driver
    instantiates with the same handles it would normally pass to
    ``run_multi_turn_rollout`` plus the data-plane config so the actor
    can attach as a TQ client (``bootstrap=False`` — controller is
    bootstrapped on the driver via ``TQPolicy``).
    """

    def __init__(
        self,
        policy_generation: GenerationInterface,
        tokenizer: Any,
        task_to_env: dict[str, EnvironmentInterface],
        master_config: Any,
        dp_cfg: Any,
        rollout_cursor: Any = None,
        rollout_secret: bytes | None = None,
    ) -> None:
        self.policy_generation = policy_generation
        self.tokenizer = tokenizer
        self.task_to_env = task_to_env
        self.master_config = master_config
        self.dp_cfg = dp_cfg
        self.rollout_cursor = rollout_cursor
        self.rollout_secret = rollout_secret

        from nemo_rl.data_plane import build_data_plane_client

        self._dp_client = build_data_plane_client(dp_cfg, bootstrap=False)

    def rollout_to_tq(
        self,
        input_batch: BatchedDataDict[Any],
        *,
        partition_id: str,
        group_size: int = 1,
        first_iter: bool = True,
        finish_generation: bool = True,
        task_to_env_override: Optional[dict[str, EnvironmentInterface]] = None,
        carry_keys: Optional[list[str]] = None,
        weight_version: int = 0,
    ) -> tuple[
        KVBatchMeta,
        BatchedDataDict[Any],
        dict[str, Any],
        Optional[dict[str, Any]],
    ]:
        """Run the full per-step generation cycle and write bulk data to TQ.

        Bundles six steps into one Ray round-trip so the driver only sees
        a single RPC instead of separate calls for each:

        1. **Reset metrics** — ``policy_generation.clear_logger_metrics()``
           clears per-step generation accumulators before the rollout.
        2. **Rollout** — runs ``run_multi_turn_rollout`` (or the async /
           nemo-gym variants) to produce ``final_batch``.
        3. **Flatten + mask + prompt extraction** — converts
           ``message_log`` layout to flat tensors; builds token mask,
           sample mask, prompt-only ids, baseline/std.
        4. **Write bulk to TQ** — ``kv_first_write`` puts every tensor
           field in one flat ``put_samples``; the driver never touches
           bulk bytes.
        5. **Release GPU** — ``policy_generation.finish_generation()``
           frees KV cache and inference state so the trainer can use the
           GPU immediately.
        6. **Capture metrics** — ``policy_generation.get_logger_metrics()``
           collects generation stats (throughput, etc.) and returns them
           to the driver in the result tuple.

        The driver receives ``(meta, driver_carry, rollout_metrics,
        generation_logger_metrics)`` and uses ``driver_carry`` for its
        own per-row compute (rewards, advantages, dynamic sampling).

        Args:
            input_batch: Per-step prompt batch (already repeat-interleaved).
            partition_id: TQ partition target.
            group_size: Rollouts per original prompt. One uid is minted
                per prompt; bulk keys are ``f"{uid}_g{i}"`` where ``i``
                ranges over the per-prompt expansion (group × rollout
                turns). Train passes ``num_generations_per_prompt``; val
                passes ``1``.
            first_iter: True on the first DS iteration of a step; drives
                ``policy_generation.snapshot_step_metrics()`` so per-step
                metrics align with the legacy ``grpo.grpo_train`` path.
            finish_generation: Call ``policy_generation.finish_generation()``
                at the tail. Default ``True`` matches the training step
                (one rollout per step, release KV after). Validation sets
                ``False`` so inference state survives across val batches;
                the trainer owns the explicit ``finish_generation()`` call
                at the end of the val pass.
            task_to_env_override: Per-call task → env map. ``None`` uses
                ``self.task_to_env`` (training envs supplied at construction).
                Validation passes ``val_task_to_env`` here so val rollouts
                run against the val env set without rebuilding the actor.
            carry_keys: Names of per-row tensors to return in
                ``driver_carry``. ``None`` returns every available key
                (training uses this). Validation passes a slim list
                (e.g. ``["total_reward"]``) to avoid wasting Ray transfer
                on fields it doesn't consume.

        Returns:
            ``(meta, driver_carry, rollout_metrics, generation_logger_metrics)``
            where ``driver_carry`` is a per-row dict of tensors the driver
            uses for compute (rewards, masks, lengths, prompt_ids_for_adv,
            …) — stays on the driver, never crosses an actor boundary.
        """
        # Lazy imports — avoid pulling grpo into this module at load.
        from nemo_rl.algorithms.grpo import (
            _should_use_async_rollouts,
            _should_use_nemo_gym,
        )
        from nemo_rl.algorithms.utils import get_gdpo_reward_component_keys
        from nemo_rl.data.llm_message_utils import (
            MESSAGE_LOG_BULK_FIELDS,
            decompose_message_log,
        )

        perf_enabled = bool(self.dp_cfg.observability.enabled)
        rollout_actor_started = time.perf_counter()
        rollout_actor_cpu_started = time.process_time()
        if perf_enabled and hasattr(self._dp_client, "clear_events"):
            self._dp_client.clear_events()

        # Per-step generation-side metric hooks: snapshot once on the
        # first DS iter so backends with per-step deltas have a stable
        # anchor; clear accumulators before every rollout. Mirrors
        # legacy ``grpo_train``.
        if self.policy_generation is not None:
            if first_iter and hasattr(self.policy_generation, "snapshot_step_metrics"):
                self.policy_generation.snapshot_step_metrics()
            self.policy_generation.clear_logger_metrics()

        cfg = self.master_config
        writer_cfg = self.dp_cfg.rollout_writer
        writer_rollout_ids: list[str] | None = None
        writer_group_ids: list[str] | None = None
        if writer_cfg.enabled:
            if self.rollout_cursor is None or self.rollout_secret is None:
                raise RuntimeError("rollout writer runtime was not configured")
            if group_size <= 0 or input_batch.size % group_size != 0:
                raise ValueError(
                    f"input_batch.size={input_batch.size} is not divisible by "
                    f"group_size={group_size}"
                )
            group_ids = [
                str(uuid.uuid4()) for _ in range(input_batch.size // group_size)
            ]
            # One canonical rollout_id per generation attempt; the parallel
            # writer_group_ids list is the retained rollout_id -> group_id
            # mapping. group_id never travels in requests or storage keys.
            writer_group_ids = [
                group_id for group_id in group_ids for _ in range(group_size)
            ]
            writer_rollout_ids = [mint_rollout_id() for _ in range(input_batch.size)]
            input_batch = _attach_rollout_contexts(
                input_batch,
                rollout_ids=writer_rollout_ids,
                group_ids=writer_group_ids,
                weight_version=weight_version,
                secret=self.rollout_secret,
                ttl_s=writer_cfg.cursor_ttl_s,
            )
        task_to_env = (
            task_to_env_override
            if task_to_env_override is not None
            else self.task_to_env
        )
        common = dict(
            policy_generation=self.policy_generation,
            input_batch=input_batch,
            tokenizer=self.tokenizer,
            task_to_env=task_to_env,
            greedy=False,
        )

        # Rollout dispatch (mirrors grpo_sync.py:294-349).
        if _should_use_nemo_gym(cfg):
            r = run_async_nemo_gym_rollout(
                **common,
                max_seq_len=None,
                max_rollout_turns=None,
                generation_config=cfg.policy["generation"],
                effort_config=EffortLevelsConfig.model_validate(
                    cfg.env["nemo_gym"].get("effort_levels")
                )
                if "nemo_gym" in cfg.env
                and cfg.env["nemo_gym"].get("effort_levels") is not None
                else None,
                reward_penalty_config=cfg.reward_penalties,
                thinking_tags=get_nemo_gym_thinking_tags(cfg.env),
                direct_mode=writer_cfg.enabled and writer_cfg.mode == "direct",
                collect_transport_metrics=perf_enabled,
            )
            final_batch, rollout_metrics = r.final_batch, r.rollout_metrics
        else:
            runner = (
                run_async_multi_turn_rollout
                if _should_use_async_rollouts(cfg)
                else run_multi_turn_rollout
            )
            final_batch, rollout_metrics = runner(
                **common,
                max_seq_len=cfg.policy["max_total_sequence_length"],
                max_rollout_turns=cfg.grpo["max_rollout_turns"],
            )
        rollout_returned = time.perf_counter()
        fb = final_batch.to("cpu")
        del final_batch

        direct_authoritative = writer_cfg.enabled and writer_cfg.mode == "direct"
        legacy_flatten_s = 0.0
        if direct_authoritative:
            input_lengths = fb["input_lengths"]
            alignment = int(cfg.policy["make_sequence_length_divisible_by"] or 1)
            max_length = int(input_lengths.max())
            padded_length = ((max_length + alignment - 1) // alignment) * alignment
            batch_size = int(input_lengths.shape[0])
            flat = BatchedDataDict(
                {
                    "token_ids": torch.full(
                        (batch_size, padded_length),
                        self.tokenizer.pad_token_id,
                        dtype=torch.int64,
                    ),
                    "generation_logprobs": torch.zeros(
                        (batch_size, padded_length), dtype=torch.float32
                    ),
                    "token_loss_mask": torch.zeros(
                        (batch_size, padded_length), dtype=torch.float32
                    ),
                }
            )
            # Replaced from verified canonical rows after finalization below.
            # Keep a nonempty placeholder because direct Gym terminal results
            # intentionally contain no token IDs.
            prompt_flat = {
                "token_ids": torch.full(
                    (batch_size, 1),
                    self.tokenizer.pad_token_id,
                    dtype=torch.int64,
                )
            }
        else:
            # Flatten message_log → bulk tensors + extract original prompt ids.
            flatten_started = time.perf_counter()
            flat, input_lengths, prompt_flat = _flatten_rollout_message_log_for_tq(
                fb["message_log"],
                fb["length"],
                pad_token_id=self.tokenizer.pad_token_id,
                make_sequence_length_divisible_by=cfg.policy[
                    "make_sequence_length_divisible_by"
                ],
            )
            legacy_flatten_s = time.perf_counter() - flatten_started

        router_replay_enabled = bool(
            (cfg.policy.get("router_replay") or {}).get("enabled", False)
        )
        if router_replay_enabled and ROUTED_EXPERTS_FIELD not in flat:
            raise RuntimeError(
                "policy.router_replay.enabled=true requires routed_experts in "
                "the rollout bulk payload, but rollout flattening did not "
                "produce that field. Check vLLM routed-expert capture and the "
                "message-log flattening path."
            )

        # TQ bulk payload — DP_TRAIN_FIELDS + multimodal extras.
        bulk_batch = BatchedDataDict[Any](
            {
                "input_ids": flat["token_ids"],
                "input_lengths": input_lengths,
                "generation_logprobs": flat["generation_logprobs"],
                "token_mask": flat["token_loss_mask"],
                "sample_mask": fb["loss_multiplier"],
            }
        )
        if ROUTED_EXPERTS_FIELD in flat:
            bulk_batch[ROUTED_EXPERTS_FIELD] = flat[ROUTED_EXPERTS_FIELD]
        for k, v in flat.get_multimodal_dict(as_tensors=False).items():
            if isinstance(v, torch.Tensor):
                bulk_batch[k] = v
        # ``content`` (raw assistant text per sample) — rides TQ as a
        # NonTensorStack so the driver can fetch it back at jsonl time
        # (kv_first_write wraps it via NonTensorStack).
        if "content" in flat:
            bulk_batch["content"] = np.asarray(flat["content"], dtype=object)

        # Split `message_log` into per-field arrays instead of pickling
        # the list-of-dicts-with-tensors per row. Consumer rebuilds
        # `message_log` on read; external API stays the same.
        if direct_authoritative:
            decomposed = {
                "response_token_lengths": fb["response_token_lengths"],
                "turn_roles": fb["turn_roles"],
                "turn_contents": fb["turn_contents"],
            }
        else:
            decomposed = decompose_message_log(fb["message_log"])
            for k in MESSAGE_LOG_BULK_FIELDS:
                bulk_batch[k] = decomposed[k]

        # Pass through remaining non-tensor fb fields as object arrays;
        # `message_log` is excluded since its tensors live in the
        # decomposed fields above (per-row pickle of dict-with-tensors
        # would smuggle aliased views into the wire).
        for k, v in fb.items():
            if isinstance(v, torch.Tensor) or k in bulk_batch or k == "message_log":
                continue
            bulk_batch[k] = (
                v
                if isinstance(v, np.ndarray) and v.dtype == object
                else np.asarray(v, dtype=object)
            )

        # Slice — only what the driver can't derive from a TQ slice fetch
        # (anything containing `message_log` or per-token data would
        # force a fetch). Driver does scale_rewards / reward_shaping /
        # overlong filtering / baseline-std on this slice.
        truncated = fb["truncated"]
        if not isinstance(truncated, torch.Tensor):
            truncated = torch.tensor(truncated, dtype=torch.bool)
        length = fb.get("length", input_lengths)
        if not isinstance(length, torch.Tensor):
            length = torch.tensor(length)
        driver_carry = {
            "total_reward": fb["total_reward"],
            "loss_multiplier": fb["loss_multiplier"],
            # Legacy/shadow rows are complete by construction. The direct
            # finalizer replaces this with zero for rejected placeholders.
            "trajectory_valid_mask": torch.ones_like(
                fb["loss_multiplier"], dtype=torch.float32
            ),
            "truncated": truncated,
            "length": length,
            "input_lengths": input_lengths,
            "prompt_ids_for_adv": prompt_flat["token_ids"],
            # Computed by decompose_message_log above; feeds
            # apply_reward_shaping on the driver without a TQ fetch.
            "response_token_lengths": decomposed["response_token_lengths"],
        }
        # GDPO multi-reward components: scale_rewards iterates these
        # keys driver-side and the GDPO advantage estimator reads them
        # from ``adv_inputs``. Plumb them through ``driver_carry``
        # rather than forcing a separate TQ fetch.
        for k in get_gdpo_reward_component_keys(fb):
            driver_carry[k] = fb[k]
        finalized = None
        finalizer_perf: dict[str, float] = {}
        if writer_cfg.enabled:
            assert writer_rollout_ids is not None and writer_group_ids is not None
            finalize_started = time.perf_counter()
            finalized = assemble_staged_batch(
                dp_client=self._dp_client,
                cursor=self.rollout_cursor,
                staging_partition=writer_cfg.staging_partition,
                sample_ids=writer_rollout_ids,
                group_ids=writer_group_ids,
                weight_version=weight_version,
                legacy_bulk=bulk_batch,
                legacy_carry=BatchedDataDict(driver_carry),
                pad_token_id=self.tokenizer.pad_token_id,
                finalize_timeout_s=writer_cfg.finalize_timeout_s,
                perf_metrics=finalizer_perf,
            )
            persist_rollout_manifest(
                finalized.manifest_rows, log_dir=cfg.logger["log_dir"]
            )
            rejection_reasons = Counter(
                row["rejection_reason"]
                for row in finalized.manifest_rows
                if row["rejection_reason"] is not None
            )
            rollout_metrics["rollout_writer/finalized"] = sum(
                row["status"] == "finalized" for row in finalized.manifest_rows
            )
            rollout_metrics["rollout_writer/rejected"] = sum(rejection_reasons.values())
            rollout_metrics["rollout_writer/prefix_mismatch_rate"] = sum(
                count
                for reason, count in rejection_reasons.items()
                if "prefix_mismatch" in reason
            ) / len(finalized.manifest_rows)
            rollout_metrics["timing/rollout_writer/finalize"] = (
                time.perf_counter() - finalize_started
            )
            if writer_cfg.mode == "shadow":
                compare_shadow_candidate(
                    finalized,
                    legacy_bulk=bulk_batch,
                    legacy_carry=BatchedDataDict(driver_carry),
                )
            # Valid shadow rows are byte-equal to legacy by the comparison
            # above. Using the finalized batch also ensures a rejected direct
            # candidate becomes a masked placeholder in both modes.
            bulk_batch = finalized.bulk_batch
            driver_carry = finalized.driver_carry
            if direct_authoritative:
                driver_carry["prompt_ids_for_adv"] = _prompt_ids_from_finalized_batch(
                    input_ids=bulk_batch["input_ids"],
                    prompt_lengths=driver_carry["length"],
                    group_ids=writer_group_ids,
                    validity_mask=driver_carry["trajectory_valid_mask"],
                    pad_token_id=self.tokenizer.pad_token_id,
                )

        if carry_keys is not None:
            for k in OPT_IN_CARRY_KEYS:
                if k in carry_keys:
                    driver_carry[k] = decomposed[k]
            missing = set(carry_keys) - driver_carry.keys()
            if missing:
                raise KeyError(
                    f"rollout_to_tq: carry_keys {sorted(missing)} not produced; "
                    f"valid keys: {sorted(driver_carry)}"
                )
            driver_carry = {k: driver_carry[k] for k in carry_keys}

        n_samples = int(bulk_batch["sample_mask"].shape[0])
        input_size = int(input_batch.size)
        if group_size <= 0 or input_size % group_size != 0:
            raise ValueError(
                f"input_batch.size={input_size} is not divisible by group_size={group_size}"
            )
        n_prompts = input_size // group_size
        if n_prompts == 0 or n_samples % n_prompts != 0:
            raise ValueError(
                f"bulk_batch has {n_samples} samples; not divisible by n_prompts={n_prompts}"
            )
        n_per_prompt = n_samples // n_prompts
        if writer_rollout_ids is None:
            uids = [str(uuid.uuid4()) for _ in range(n_prompts)]
            sample_ids = [f"{uid}_g{i}" for uid in uids for i in range(n_per_prompt)]
        else:
            sample_ids = writer_rollout_ids
        trace_rollout_payload(keys=sample_ids, data=bulk_batch)
        canonical_put_started = time.perf_counter()
        meta = kv_first_write(
            bulk_batch,
            sample_ids=sample_ids,
            dp_client=self._dp_client,
            partition_id=partition_id,
            extra_info={"rollout_metrics": rollout_metrics},
            task_name=partition_id,
            pad_to_multiple=int(
                cfg.policy.get("make_sequence_length_divisible_by") or 1
            ),
        )
        canonical_ready = time.perf_counter()
        canonical_ready_monotonic = time.monotonic()
        canonical_put_s = canonical_ready - canonical_put_started
        if finalized is not None:
            self._dp_client.clear_samples(
                sample_ids=list(finalized.staging_keys),
                partition_id=writer_cfg.staging_partition,
            )
            ray.get(
                [
                    self.rollout_cursor.clear_sample.remote(sample_id)
                    for sample_id in sample_ids
                ]
            )

        rollout_transport_metrics: dict[str, Any] = {}
        if (
            perf_enabled
            and self.policy_generation is not None
            and hasattr(self.policy_generation, "get_rollout_transport_metrics")
        ):
            rollout_transport_metrics = (
                self.policy_generation.get_rollout_transport_metrics()
            )

        if self.policy_generation is not None:
            if finish_generation:
                self.policy_generation.finish_generation()
            gen_metrics = self.policy_generation.get_logger_metrics()
        else:
            gen_metrics = None

        if perf_enabled:
            actor_data_plane_events = (
                self._dp_client.events_snapshot()
                if hasattr(self._dp_client, "events_snapshot")
                else []
            )
            actor_data_plane_snapshot = (
                self._dp_client.snapshot()
                if hasattr(self._dp_client, "snapshot")
                else {}
            )
            all_data_plane_events = [
                {**event, "source": "rollout_actor"}
                for event in actor_data_plane_events
            ] + [
                {**event, "source": "vllm_worker"}
                for event in rollout_transport_metrics.get("data_plane_events", [])
            ]
            sample_count = int(bulk_batch["sample_mask"].shape[0])
            generated_tokens = (
                float(rollout_metrics.get("mean_gen_tokens_per_sample", 0.0))
                * sample_count
            )
            stage_put_ms = rollout_transport_metrics.get("staging_put_ms", [])
            http_request_ms = rollout_transport_metrics.get("http_request_ms", [])
            chat_request_ms = rollout_transport_metrics.get(
                "chat_request_ms", http_request_ms
            )
            last_http_response = rollout_transport_metrics.get(
                "last_response_completed_monotonic_s", 0.0
            )
            mode = writer_cfg.mode if writer_cfg.enabled else "legacy"
            perf_row = {
                "schema_version": 2,
                "timestamp": time.time(),
                "mode": mode,
                "sample_count": sample_count,
                "generated_tokens": generated_tokens,
                "workload": {
                    "total_turns": float(
                        rollout_metrics.get("turns_per_sample/mean", 0.0)
                    )
                    * sample_count,
                    **{
                        key: value
                        for key, value in rollout_metrics.items()
                        if isinstance(value, (int, float))
                        and key.startswith("turns_per_sample/")
                    },
                },
                "rollout_actor_wall_s": rollout_returned - rollout_actor_started,
                "rollout_actor_cpu_s": time.process_time() - rollout_actor_cpu_started,
                "rollout_actor_peak_rss_bytes": resource.getrusage(
                    resource.RUSAGE_SELF
                ).ru_maxrss
                * 1024,
                "rollout_start_to_canonical_ready_s": canonical_ready
                - rollout_actor_started,
                "rollout_return_to_canonical_ready_s": canonical_ready
                - rollout_returned,
                "final_http_response_to_canonical_ready_s": (
                    canonical_ready_monotonic - last_http_response
                    if last_http_response
                    else None
                ),
                "rollout_generated_tokens_per_s": (
                    generated_tokens / (rollout_returned - rollout_actor_started)
                    if rollout_returned > rollout_actor_started
                    else 0.0
                ),
                "canonical_ready_generated_tokens_per_s": (
                    generated_tokens / (canonical_ready - rollout_actor_started)
                    if canonical_ready > rollout_actor_started
                    else 0.0
                ),
                "canonical_ready_samples_per_s": (
                    sample_count / (canonical_ready - rollout_actor_started)
                    if canonical_ready > rollout_actor_started
                    else 0.0
                ),
                "legacy_flatten_s": legacy_flatten_s,
                "canonical_put_s": canonical_put_s,
                "finalizer": finalizer_perf,
                "http": {
                    name: rollout_transport_metrics.get(name, 0)
                    for name in (
                        "http_request_count",
                        "http_response_count",
                        "encoded_request_bytes",
                        "encoded_response_bytes",
                        "encoded_response_bytes_without_token_ids",
                        "encoded_response_bytes_without_logprobs",
                        "encoded_response_base_bytes",
                        "http_request_ms",
                        "chat_request_count",
                        "chat_response_count",
                        "chat_encoded_request_bytes",
                        "chat_encoded_response_bytes",
                        "chat_encoded_response_bytes_without_token_ids",
                        "chat_encoded_response_bytes_without_logprobs",
                        "chat_encoded_response_base_bytes",
                        "chat_request_ms",
                        "tokenize_request_count",
                        "tokenize_response_count",
                        "tokenize_encoded_request_bytes",
                        "tokenize_encoded_response_bytes",
                        "tokenize_encoded_response_bytes_without_token_ids",
                        "tokenize_encoded_response_bytes_without_logprobs",
                        "tokenize_encoded_response_base_bytes",
                        "tokenize_request_ms",
                    )
                },
                "staging_put_blocked_fraction": (
                    sum(stage_put_ms) / sum(chat_request_ms)
                    if chat_request_ms and sum(chat_request_ms) > 0
                    else 0.0
                ),
                "cursor_reserve_ms": rollout_transport_metrics.get(
                    "cursor_reserve_ms", []
                ),
                "staging_put_ms": stage_put_ms,
                "cursor_commit_ms": rollout_transport_metrics.get(
                    "cursor_commit_ms", []
                ),
                "vllm_process_cpu_s": rollout_transport_metrics.get(
                    "process_cpu_s", 0.0
                ),
                "vllm_peak_rss_bytes": rollout_transport_metrics.get(
                    "peak_rss_bytes", 0
                ),
                "distinct_workers_per_sample": rollout_transport_metrics.get(
                    "distinct_workers_per_sample", {}
                ),
                "gym": {
                    key: value
                    for key, value in rollout_metrics.items()
                    if isinstance(value, (int, float))
                    and key.startswith(("transport/", "process/", "timing/rollout/"))
                },
                "data_plane_events": all_data_plane_events,
                "max_client_peak_bytes_outstanding": max(
                    actor_data_plane_snapshot.get("peak_bytes_outstanding", 0),
                    rollout_transport_metrics.get(
                        "max_client_peak_bytes_outstanding", 0
                    ),
                ),
            }
            persist_rollout_perf_metrics(perf_row, log_dir=cfg.logger["log_dir"])
        return meta, BatchedDataDict(driver_carry), rollout_metrics, gen_metrics

    def shutdown(self) -> None:
        try:
            self._dp_client.close()
        except Exception:
            pass
