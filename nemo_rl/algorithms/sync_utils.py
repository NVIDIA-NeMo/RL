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
"""Sync GRPO data-plane helpers — sibling of ``async_utils``.

Houses the sync 1-hop counterparts to ``async_utils.AsyncTrajectoryCollector``
and ``async_utils.ReplayBuffer``:

* :func:`rollout_to_tq` — the flat first-write primitive (mirrors verl
  ``main_ppo_sync.py:386-423``); single ``kv_batch_put`` of every tensor
  field under per-sample keys ``f"{uid}_g{i}"``.

* :class:`SyncTrajectoryCollector` — the Ray actor that owns the
  multi-turn rollout loop AND the post-rollout flatten / mask /
  prompt extraction / reward shaping / baseline-std for a sync GRPO
  step. The driver dispatches a per-step prompt batch + uids; the
  actor runs ``run_multi_turn_rollout`` (or async / nemo_gym variants),
  then writes the bulk schema to TQ via :func:`rollout_to_tq`. Only a
  ``KVBatchMeta`` and a small per-sample slice (rewards, masks,
  lengths, baseline/std, prompt_ids_for_adv) cross back to the driver
  via Ray.

**Goal — rollout 1-hop put**: bulk tensors (input_ids, output_ids,
attention_mask, position_ids, multi_modal_inputs, generation_logprobs,
token_mask) stay actor-side until ``kv_batch_put``, then live only in
TQ. Driver never holds these bytes between rollout finish and train
fan-out. See ``research/data_plane_integration_plan.md`` §1.2.

The collector is the sync counterpart to
:class:`nemo_rl.algorithms.async_utils.AsyncTrajectoryCollector`. It
intentionally does not buffer or stream — sync GRPO consumes the whole
step batch in one call.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import ray
import torch
from tensordict import TensorDict

from nemo_rl.data_plane.interfaces import DataPlaneClient, KVBatchMeta
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.environments.interfaces import EnvironmentInterface
from nemo_rl.experience.rollouts import (
    run_async_multi_turn_rollout,
    run_async_nemo_gym_rollout,
    run_multi_turn_rollout,
)
from nemo_rl.models.generation.interfaces import GenerationInterface


def rollout_to_tq(
    final_batch_cpu: BatchedDataDict[Any],
    *,
    uids: Sequence[str],
    dp_client: DataPlaneClient,
    partition_id: str,
    extra_info: Optional[dict[str, Any]] = None,
    task_name: str = "train",
) -> KVBatchMeta:
    """Single flat ``kv_batch_put`` of every tensor field in ``final_batch_cpu``.

    Mirrors verl ``main_ppo_sync.py:386-423``: keys ``f"{uid}_g{i}"``,
    no DP awareness, no fan-out. Bulk lives in TQ from here on; the
    caller never re-handles it on the driver. See
    ``research/data_plane_integration_plan.md`` §1.2.
    """
    n = int(final_batch_cpu["sample_mask"].shape[0])
    if n == 0 or len(uids) == 0 or n % len(uids) != 0:
        raise ValueError(
            f"final_batch_cpu has {n} samples; not divisible by len(uids)={len(uids)}"
        )
    n_gen = n // len(uids)
    keys = [f"{uid}_g{i}" for uid in uids for i in range(n_gen)]

    bulk_field_names = [
        k for k, v in final_batch_cpu.items() if isinstance(v, torch.Tensor)
    ]
    bulk = TensorDict(
        {k: final_batch_cpu[k].detach().contiguous() for k in bulk_field_names},
        batch_size=[n],
    )
    dp_client.kv_batch_put(
        keys=keys, partition_id=partition_id, fields=bulk,
    )

    return KVBatchMeta(
        partition_id=partition_id,
        task_name=task_name,
        keys=keys,
        fields=bulk_field_names,
        sequence_lengths=[int(s) for s in final_batch_cpu["input_lengths"].tolist()],
        extra_info=dict(extra_info or {}),
    )


@ray.remote  # pragma: no cover
class SyncTrajectoryCollector:
    """Per-step rollout dispatcher: rollout + flatten + mask + prompt extraction
    + baseline/std + TQ put. Returns ``(meta, slice, metrics)``.

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
        uids: list[str],
        partition_id: str,
    ) -> tuple[
        KVBatchMeta,
        dict[str, Any],
        dict[str, Any],
        Optional[dict[str, Any]],
    ]:
        """Rollout → flatten + mask + prompt extraction → flat ``kv_batch_put``.

        Returns ``(meta, slice, rollout_metrics, generation_logger_metrics)``.
        ``slice`` carries only the small per-sample tensors the driver
        needs to do its own per-sample compute (scale_rewards,
        reward_shaping, overlong filtering, baseline/std,
        dynamic_sampling, advantage). The actor handles only the
        bulk-touching ops — flatten / mask / prompt extraction — that
        require ``message_log`` and would otherwise force bulk onto the
        driver.
        """
        # Lazy imports — avoid pulling grpo into this module at load.
        from nemo_rl.algorithms.grpo import (
            _extract_prompt_only_messages,
            _should_use_async_rollouts,
            _should_use_nemo_gym,
        )
        from nemo_rl.data.llm_message_utils import (
            add_loss_mask_to_message_log,
            batched_message_log_to_flat_message,
        )

        cfg = self.master_config
        common = dict(
            policy_generation=self.policy_generation,
            input_batch=input_batch,
            tokenizer=self.tokenizer,
            task_to_env=self.task_to_env,
            greedy=False,
        )

        # Rollout dispatch (mirrors grpo_sync.py:294-349).
        if _should_use_nemo_gym(cfg):
            r = run_async_nemo_gym_rollout(
                **common, max_seq_len=None, max_rollout_turns=None,
                generation_config=cfg["policy"]["generation"],
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
                max_seq_len=cfg["policy"]["max_total_sequence_length"],
                max_rollout_turns=cfg["grpo"]["max_rollout_turns"],
            )
        fb = final_batch.to("cpu")
        del final_batch

        # Assistant-only loss mask (shared helper); seed missing
        # generation_logprobs (e.g. when the env wraps assistant turns
        # without a backing logprob, or for greedy/replay rollouts).
        add_loss_mask_to_message_log(fb["message_log"])
        for ml in fb["message_log"]:
            for msg in ml:
                msg.setdefault(
                    "generation_logprobs",
                    torch.zeros_like(msg["token_ids"], dtype=torch.float32),
                )

        # Flatten message_log → bulk tensors + extract prompt-only ids.
        pad = {"pad_value_dict": {"token_ids": self.tokenizer.pad_token_id}}
        flat, input_lengths = batched_message_log_to_flat_message(
            fb["message_log"], **pad,
            make_sequence_length_divisible_by=cfg["policy"][
                "make_sequence_length_divisible_by"
            ],
        )
        prompt_flat, _ = batched_message_log_to_flat_message(
            _extract_prompt_only_messages(fb["message_log"]), **pad,
        )

        # TQ bulk payload — DP_SEED_FIELDS + multimodal extras.
        bulk_batch = BatchedDataDict[Any]({
            "input_ids": flat["token_ids"],
            "input_lengths": input_lengths,
            "generation_logprobs": flat["generation_logprobs"],
            "token_mask": flat["token_loss_mask"],
            "sample_mask": fb["loss_multiplier"],
        })
        for k, v in flat.get_multimodal_dict(as_tensors=False).items():
            if isinstance(v, torch.Tensor):
                bulk_batch[k] = v

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
        slice_extras = {
            "total_reward": fb["total_reward"],
            "loss_multiplier": fb["loss_multiplier"],
            "truncated": truncated,
            "length": length,
            "input_lengths": input_lengths,
            "prompt_ids_for_adv": prompt_flat["token_ids"],
        }

        meta = rollout_to_tq(
            bulk_batch, uids=uids,
            dp_client=self._dp_client,
            partition_id=partition_id,
            extra_info={"rollout_metrics": rollout_metrics},
            task_name="train" if partition_id == "train" else partition_id,
        )

        if self.policy_generation is not None:
            self.policy_generation.finish_generation()
            gen_metrics = self.policy_generation.get_logger_metrics()
        else:
            gen_metrics = None
        return meta, slice_extras, rollout_metrics, gen_metrics

    def finish_generation(self) -> None:
        """Forward to ``policy_generation.finish_generation``."""
        if self.policy_generation is not None:
            self.policy_generation.finish_generation()

    def get_logger_metrics(self) -> Optional[dict[str, Any]]:
        if self.policy_generation is None:
            return None
        return self.policy_generation.get_logger_metrics()

    def clear_logger_metrics(self) -> None:
        if self.policy_generation is None:
            return
        self.policy_generation.clear_logger_metrics()

    def shutdown(self) -> None:
        try:
            self._dp_client.close()
        except Exception:
            pass
