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

"""End-to-end test: SC._rollout_pump writes the expected rows to TQ.

Reuses test_async_rollout_manager's fixtures (real vLLM, env, tokenizer,
DatumSpec). dp_client is a NoOpDataPlaneClient wrapped in a Ray actor so the
test process can inspect TQ state after the SC actor finishes.
"""

from __future__ import annotations

from typing import Any

import ray
import torch
from tensordict import TensorDict

from nemo_rl.algorithms.single_controller import (
    SingleControllerActor,
    SingleControllerConfig,
)
from nemo_rl.data_plane.adapters.noop import NoOpDataPlaneClient

# Reuse fixtures from the experience tests; same shape as test_async_rollout_manager.
from tests.unit.experience.test_rollouts import (
    multi_step_setup_vllm_async,  # noqa: F401
    single_multi_step_calculator_input_sample,  # noqa: F401
)

_PARTITION_ID = "rollout_data"
_BULK_FIELDS = [
    "input_ids",
    "input_lengths",
    "generation_logprobs",
    "token_mask",
    "sample_mask",
    "prompt_ids_for_adv",
    "total_reward",
]


@ray.remote(num_cpus=0)
class _TQActor:
    """Ray-wrapped NoOpDataPlaneClient for cross-process TQ inspection."""

    def __init__(
        self,
        partition_id: str,
        fields: list[str],
        num_samples: int,
        consumer_tasks: list[str],
    ) -> None:
        self._client = NoOpDataPlaneClient()
        self._client.register_partition(
            partition_id=partition_id,
            fields=list(fields),
            num_samples=int(num_samples),
            consumer_tasks=list(consumer_tasks),
        )

    def put_samples(
        self,
        sample_ids: list[str],
        partition_id: str,
        fields: TensorDict | None = None,
        tags: list[dict[str, Any]] | None = None,
    ) -> Any:
        return self._client.put_samples(
            sample_ids=sample_ids,
            partition_id=partition_id,
            fields=fields,
            tags=tags,
        )

    def claim_meta(self, **kwargs: Any) -> Any:
        return self._client.claim_meta(**kwargs)

    def get_samples(
        self,
        sample_ids: list[str],
        partition_id: str,
        select_fields: list[str],
    ) -> TensorDict:
        return self._client.get_samples(
            sample_ids=sample_ids,
            partition_id=partition_id,
            select_fields=list(select_fields),
        )

    def get_tags(
        self, partition_id: str, sample_ids: list[str]
    ) -> list[dict[str, Any]]:
        rec = self._client._partitions[partition_id]
        return [dict(rec.tags.get(sid, {})) for sid in sample_ids]


class _SyncDPAdapter:
    """Sync DataPlaneClient over a Ray actor handle. Pads nested tensors before transport."""

    def __init__(self, handle: Any) -> None:
        self._handle = handle

    def put_samples(
        self,
        sample_ids: list[str],
        partition_id: str,
        fields: TensorDict | None = None,
        tags: list[dict[str, Any]] | None = None,
    ) -> Any:
        if fields is not None:
            fields = self._padded(fields)
        return ray.get(
            self._handle.put_samples.remote(
                sample_ids=sample_ids,
                partition_id=partition_id,
                fields=fields,
                tags=tags,
            )
        )

    @staticmethod
    def _padded(td: TensorDict) -> TensorDict:
        out: dict[str, torch.Tensor] = {}
        for k in td.keys():
            v = td.get(k)
            if isinstance(v, torch.Tensor) and v.is_nested:
                v = torch.nested.to_padded_tensor(v, padding=0)
            out[k] = v
        return TensorDict(out, batch_size=td.batch_size)


def test_rollout_pump_writes_expected_tq_data(
    multi_step_setup_vllm_async,  # noqa: F811
    single_multi_step_calculator_input_sample,  # noqa: F811
):
    """SC._rollout_pump writes max_rollout_prompts * num_generations rows to TQ with the expected fields and tags."""
    vllm_generation, tokenizer, env_handles, _, _ = multi_step_setup_vllm_async
    input_sample = single_multi_step_calculator_input_sample

    num_generations = 2
    max_rollout_prompts = 2
    expected_samples = max_rollout_prompts * num_generations
    max_seq_len = 1024
    max_rollout_turns = input_sample["extra_env_info"]["max_steps"] + 1

    tq_actor = _TQActor.remote(
        partition_id=_PARTITION_ID,
        fields=_BULK_FIELDS,
        num_samples=expected_samples * 4,
        consumer_tasks=["train"],
    )
    dp_adapter = _SyncDPAdapter(tq_actor)

    cfg = SingleControllerConfig(
        max_train_steps=1,
        max_rollout_prompts=max_rollout_prompts,
        min_prompt_groups_per_batch=1,
        generations_per_prompt=num_generations,
        max_buffered_rollouts=max_rollout_prompts,
        max_inflight_prompts=max_rollout_prompts,
        max_weight_staleness_versions=0,
        advantage_enabled=False,
        diagnostics=False,
        partition_id=_PARTITION_ID,
        rollout_max_seq_len=max_seq_len,
        rollout_max_turns=max_rollout_turns,
        use_nemo_gym=False,
    )
    prompts = [input_sample] * max_rollout_prompts

    vllm_generation.prepare_for_generation()
    ctrl = SingleControllerActor.remote(
        cfg=cfg,
        dp_client=dp_adapter,
        gen_handle=vllm_generation,
        trainer_handle=object(),
        env_handles=env_handles,
        prompts=prompts,
        weight_synchronizer=object(),
        tokenizer=tokenizer,
    )
    ray.get(ctrl._rollout_pump.remote())
    vllm_generation.finish_generation()

    meta = ray.get(
        tq_actor.claim_meta.remote(
            partition_id=_PARTITION_ID,
            task_name="train",
            required_fields=["input_ids"],
            batch_size=expected_samples * 4,
            blocking=False,
            timeout_s=0.0,
        )
    )
    assert meta.size == expected_samples

    group_ids: set[str] = set()
    for sid in meta.sample_ids:
        prefix, sep, suffix = sid.rpartition("_g")
        assert sep == "_g" and suffix.isdigit(), f"unexpected sample_id: {sid}"
        group_ids.add(prefix)
    assert len(group_ids) == max_rollout_prompts

    data = ray.get(
        tq_actor.get_samples.remote(
            sample_ids=meta.sample_ids,
            partition_id=_PARTITION_ID,
            select_fields=_BULK_FIELDS,
        )
    )
    assert set(data.keys()) == set(_BULK_FIELDS), (
        f"unexpected fields: {set(data.keys())}"
    )
    assert data["input_lengths"].shape[0] == expected_samples
    assert torch.all(data["input_lengths"] > 0)
    assert torch.allclose(
        data["sample_mask"].float(),
        torch.ones(expected_samples, dtype=torch.float32),
    )

    # Same deterministic prompt as test_async_rollout_manager: the model
    # solves the calculator task every time -> reward == 1.0 and decoded
    # tail contains " 16".
    rewards = data["total_reward"].float().flatten()
    assert rewards.shape == (expected_samples,)
    assert torch.allclose(rewards, torch.ones(expected_samples)), (
        f"expected all rewards == 1.0, got {rewards.tolist()}"
    )

    input_ids = data["input_ids"]
    input_lengths = data["input_lengths"].tolist()
    token_mask = data["token_mask"]
    for i in range(expected_samples):
        length = int(input_lengths[i])
        decoded = tokenizer.decode(
            input_ids[i, :length].tolist(), skip_special_tokens=False
        )
        assert " 16" in decoded[-64:], (
            f"sample {i}: decoded tail {decoded[-64:]!r} missing ' 16'"
        )
        assert int(token_mask[i, :length].sum().item()) > 0, (
            f"sample {i}: token_mask has no assistant tokens"
        )

    tags = ray.get(
        tq_actor.get_tags.remote(partition_id=_PARTITION_ID, sample_ids=meta.sample_ids)
    )
    for tag in tags:
        assert tag["weight_version"] == 0
        assert tag["expected_num_samples"] == num_generations
        assert tag["committed"] is True
        assert tag["group_id"] in group_ids
