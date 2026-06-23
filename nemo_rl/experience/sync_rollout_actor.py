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
rollout loop AND the post-rollout flatten / mask / prompt grouping /
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

import uuid
from typing import Any, Optional

import numpy as np
import ray
import torch

from nemo_rl.data_plane.column_io import kv_first_write
from nemo_rl.data_plane.interfaces import KVBatchMeta
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.experience.rollouts import (
    EffortLevelsConfig,
    run_async_multi_turn_rollout,
    run_async_nemo_gym_rollout,
    run_multi_turn_rollout,
)
from nemo_rl.models.generation.interfaces import GenerationInterface

# Carry keys producible by the rollout actor only when the caller opts in.
# These are np.ndarray(object) per-row arrays from decompose_message_log; the
# default driver_carry omits them because BatchedDataDict.select_indices on
# the training/dynamic-sampling path only handles tensors/lists. Validation
# requests them explicitly to print per-sample message logs.
OPT_IN_CARRY_KEYS: tuple[str, ...] = ("turn_roles", "turn_contents")


def _flatten_rollout_message_log_for_tq(
    message_logs: list[Any],
    *,
    pad_token_id: int,
    make_sequence_length_divisible_by: int,
) -> tuple[BatchedDataDict[Any], torch.Tensor]:
    """Prepare rollout message logs for the TQ payload and driver carry."""
    from nemo_rl.algorithms.grpo import (
        add_grpo_token_loss_masks_and_generation_logprobs,
    )
    from nemo_rl.data.llm_message_utils import batched_message_log_to_flat_message

    pad = {"pad_value_dict": {"token_ids": pad_token_id}}
    add_grpo_token_loss_masks_and_generation_logprobs(message_logs)
    flat, input_lengths = batched_message_log_to_flat_message(
        message_logs,
        **pad,
        make_sequence_length_divisible_by=make_sequence_length_divisible_by,
    )
    return flat, input_lengths


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
        dp_cfg: dict[str, Any],
    ) -> None:
        self.policy_generation = policy_generation
        self.tokenizer = tokenizer
        self.task_to_env = task_to_env
        self.master_config = master_config

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
    ) -> tuple[
        KVBatchMeta,
        dict[str, Any],
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
            get_idx_grouping,
        )
        from nemo_rl.algorithms.utils import get_gdpo_reward_component_keys
        from nemo_rl.data.llm_message_utils import (
            MESSAGE_LOG_BULK_FIELDS,
            decompose_message_log,
        )

        # Per-step generation-side metric hooks: snapshot once on the
        # first DS iter so backends with per-step deltas have a stable
        # anchor; clear accumulators before every rollout. Mirrors
        # legacy ``grpo_train``.
        if self.policy_generation is not None:
            if first_iter and hasattr(self.policy_generation, "snapshot_step_metrics"):
                self.policy_generation.snapshot_step_metrics()
            self.policy_generation.clear_logger_metrics()

        cfg = self.master_config
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
        fb = final_batch.to("cpu")
        del final_batch

        # Flatten message_log → bulk tensors.
        # GRPO masks only generated assistant turns, even if the dataset
        # prompt itself contains assistant messages as conversation history.
        flat, input_lengths = _flatten_rollout_message_log_for_tq(
            fb["message_log"],
            pad_token_id=self.tokenizer.pad_token_id,
            make_sequence_length_divisible_by=cfg.policy[
                "make_sequence_length_divisible_by"
            ],
        )
        # Composite (task_name, idx) grouping key for advantage estimation —
        # groups responses from the same prompt without relying on prompt
        # token sequences (which collide across turns/datasets).
        prompt_ids_for_adv = get_idx_grouping(fb)

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
            "truncated": truncated,
            "length": length,
            "input_lengths": input_lengths,
            "prompt_ids_for_adv": prompt_ids_for_adv,
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
        uids = [str(uuid.uuid4()) for _ in range(n_prompts)]
        sample_ids = [f"{uid}_g{i}" for uid in uids for i in range(n_per_prompt)]
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

        if self.policy_generation is not None:
            if finish_generation:
                self.policy_generation.finish_generation()
            gen_metrics = self.policy_generation.get_logger_metrics()
        else:
            gen_metrics = None
        return meta, BatchedDataDict(driver_carry), rollout_metrics, gen_metrics

    def shutdown(self) -> None:
        try:
            self._dp_client.close()
        except Exception:
            pass
