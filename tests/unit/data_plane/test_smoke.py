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
"""Tier-0 smoke tests — pre-commit gates.

Cheapest tier: catches drift in module paths, registry keys, and the
public ABC surface. Each test runs in milliseconds and never touches
real Ray / vLLM / TQ.
"""

from __future__ import annotations

import inspect


def test_sync_utils_module_imports() -> None:
    """Catches FQN drift after the algorithms.sync_utils consolidation."""
    from nemo_rl.experience.sync_rollout_actor import (
        SyncRolloutActor,
        kv_first_write,
    )

    # ``SyncRolloutActor`` is wrapped by ``@ray.remote`` into
    # ``ActorClass(SyncRolloutActor)`` — the wrapper has no
    # ``__name__`` attribute. Check via ``repr`` instead.
    assert "SyncRolloutActor" in repr(SyncRolloutActor)
    assert callable(kv_first_write)


def test_sync_rollout_actor_registered_under_vllm_tier() -> None:
    """Multinode runs depend on this — without it, tensordict missing on
    worker nodes (real bug seen in job 11614968)."""
    from nemo_rl.distributed.ray_actor_environment_registry import (
        get_actor_python_env,
    )
    from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES

    fqn = "nemo_rl.experience.sync_rollout_actor.SyncRolloutActor"
    env = get_actor_python_env(fqn)
    # Same tier as vLLM workers / AsyncTrajectoryCollector / ReplayBuffer.
    # Allow either the resolved exec path or the SYSTEM-override sentinel.
    assert env in (PY_EXECUTABLES.VLLM, PY_EXECUTABLES.SYSTEM), (
        f"unexpected env tier for {fqn}: {env!r}"
    )


def test_distillation_rollout_actor_registered_under_vllm_tier() -> None:
    """Distillation TQ runs need the same env tier as sync rollout actors."""
    from nemo_rl.distributed.ray_actor_environment_registry import (
        get_actor_python_env,
    )
    from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES

    fqn = "nemo_rl.experience.distillation_rollout_actor.DistillationRolloutActor"
    env = get_actor_python_env(fqn)
    assert env in (PY_EXECUTABLES.VLLM, PY_EXECUTABLES.SYSTEM), (
        f"unexpected env tier for {fqn}: {env!r}"
    )


def test_distillation_tq_rollout_actor_supports_nemo_gym(monkeypatch) -> None:
    """NemoGym distillation must stay on the TQ rollout path."""
    from types import SimpleNamespace

    import torch

    from nemo_rl.data_plane.interfaces import KVBatchMeta
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    import nemo_rl.experience.distillation_rollout_actor as rollout_actor

    calls = {}
    generation_config = {"temperature": 1.0}

    class FakeGeneration:
        def snapshot_step_metrics(self):
            calls["snapshot"] = True

        def clear_logger_metrics(self):
            calls["clear_metrics"] = True

        def finish_generation(self):
            calls["finish_generation"] = True

        def get_logger_metrics(self):
            return {"gen/tokens": 3}

    final_batch = BatchedDataDict(
        {
            "message_log": [
                [
                    {
                        "role": "user",
                        "content": "question",
                        "token_ids": torch.tensor([1]),
                    },
                    {
                        "role": "assistant",
                        "content": "answer",
                        "token_ids": torch.tensor([2, 3]),
                    },
                ]
            ],
            "loss_multiplier": torch.ones(1),
            "length": torch.tensor([1]),
        }
    )

    def fake_run_async_nemo_gym_rollout(**kwargs):
        calls["nemo_gym_kwargs"] = kwargs
        return SimpleNamespace(
            final_batch=final_batch,
            rollout_metrics={
                "rollout/accuracy": 1.0,
                "rollout/full_result": {"large": "payload"},
            },
        )

    def fail_other_rollout_path(*args, **kwargs):
        raise AssertionError("non-NemoGym rollout path should not run")

    def fake_kv_first_write(
        bulk_batch,
        *,
        sample_ids,
        dp_client,
        partition_id,
        task_name,
        pad_to_multiple,
    ):
        calls["bulk_batch"] = bulk_batch
        calls["sample_ids"] = sample_ids
        calls["partition_id"] = partition_id
        calls["task_name"] = task_name
        calls["pad_to_multiple"] = pad_to_multiple
        return KVBatchMeta(
            partition_id=partition_id,
            task_name=task_name,
            sample_ids=sample_ids,
            fields=list(bulk_batch),
            sequence_lengths=[int(bulk_batch["input_lengths"][0])],
        )

    monkeypatch.setattr(rollout_actor, "_should_use_nemo_gym", lambda cfg: True)
    monkeypatch.setattr(
        rollout_actor, "_should_log_nemo_gym_responses", lambda cfg: False
    )
    monkeypatch.setattr(
        rollout_actor, "_should_use_async_rollouts", fail_other_rollout_path
    )
    monkeypatch.setattr(
        rollout_actor, "run_async_multi_turn_rollout", fail_other_rollout_path
    )
    monkeypatch.setattr(
        rollout_actor, "run_multi_turn_rollout", fail_other_rollout_path
    )
    monkeypatch.setattr(
        rollout_actor, "run_async_nemo_gym_rollout", fake_run_async_nemo_gym_rollout
    )
    monkeypatch.setattr(rollout_actor, "kv_first_write", fake_kv_first_write)

    actor_cls = rollout_actor.DistillationRolloutActor.__ray_actor_class__
    actor = actor_cls.__new__(actor_cls)
    actor.policy_generation = FakeGeneration()
    actor.tokenizer = SimpleNamespace(pad_token_id=0)
    actor.task_to_env = {"task": object()}
    actor.master_config = SimpleNamespace(
        policy={
            "generation": generation_config,
            "make_sequence_length_divisible_by": 1,
            "max_total_sequence_length": 32,
        },
        distillation={"max_rollout_turns": 4},
    )
    actor._dp_client = object()

    meta, driver_carry, rollout_metrics, generation_metrics = actor.rollout_to_tq(
        BatchedDataDict({"dummy": torch.ones(1)}),
        partition_id="distill-step",
        model_input_fields=[],
    )

    nemo_gym_kwargs = calls["nemo_gym_kwargs"]
    assert nemo_gym_kwargs["generation_config"] is generation_config
    assert nemo_gym_kwargs["max_seq_len"] == 32
    assert nemo_gym_kwargs["max_rollout_turns"] is None
    assert nemo_gym_kwargs["task_to_env"] is actor.task_to_env

    assert rollout_metrics == {"rollout/accuracy": 1.0}
    assert generation_metrics == {"gen/tokens": 3}
    assert calls["finish_generation"]
    assert meta.partition_id == "distill-step"
    assert calls["pad_to_multiple"] == 1
    assert torch.equal(calls["bulk_batch"]["input_ids"], torch.tensor([[1, 2, 3]]))
    assert torch.equal(calls["bulk_batch"]["token_mask"], torch.tensor([[0, 1, 1]]))
    assert driver_carry["content"][0] == ["question", "answer"]


def test_kvbatchmeta_schema_unchanged() -> None:
    """Schema break check — KVBatchMeta is the cross-process boundary;
    adding/removing a field silently would break adapters that pickle it."""
    from nemo_rl.data_plane.interfaces import KVBatchMeta

    expected_fields = {
        "partition_id",
        "task_name",
        "sample_ids",
        "fields",
        "sequence_lengths",
        "extra_info",
        "tags",
    }
    actual_fields = {f.name for f in KVBatchMeta.__dataclass_fields__.values()}
    assert actual_fields == expected_fields, (
        f"KVBatchMeta schema drifted. expected={expected_fields}, "
        f"actual={actual_fields}"
    )


def test_dataplane_client_abc_surface() -> None:
    """Catches accidental ABC method removal / rename — e.g. dropping
    ``clear_samples`` would break step-end teardown silently."""
    from nemo_rl.data_plane.interfaces import DataPlaneClient

    expected_methods = {
        # task-mediated
        "register_partition",
        "claim_meta",
        "get_data",
        "check_consumption_status",
        # direct-by-key
        "put_samples",
        "get_samples",
        "clear_samples",
        # lifecycle
        "close",
    }
    actual_methods = {
        name
        for name, member in inspect.getmembers(DataPlaneClient, callable)
        if not name.startswith("_") and getattr(member, "__isabstractmethod__", False)
    }
    assert expected_methods.issubset(actual_methods), (
        f"DataPlaneClient ABC missing methods: {expected_methods - actual_methods}"
    )


def test_async_and_sync_actors_share_env_tier() -> None:
    """Sync should mirror async's env tier — both drive vLLM and write
    tensordict to TQ, so they need the same VLLM venv."""
    from nemo_rl.distributed.ray_actor_environment_registry import (
        get_actor_python_env,
    )

    sync_env = get_actor_python_env(
        "nemo_rl.experience.sync_rollout_actor.SyncRolloutActor"
    )
    async_env = get_actor_python_env(
        "nemo_rl.algorithms.async_utils.AsyncTrajectoryCollector"
    )
    assert sync_env == async_env, (
        f"Sync vs async env tier drift: sync={sync_env!r}, async={async_env!r}"
    )


def test_sync_rollout_actor_prompt_extraction_and_masks_match_grpo() -> None:
    """TQ rollouts must mirror GRPO's length-based prompt extraction."""
    import torch

    from nemo_rl.experience.sync_rollout_actor import (
        _flatten_rollout_message_log_for_tq,
    )

    message_logs = [
        [
            {"role": "user", "content": "first", "token_ids": torch.tensor([1, 2])},
            {
                "role": "assistant",
                "content": "history",
                "token_ids": torch.tensor([3, 4]),
            },
            {"role": "user", "content": "next", "token_ids": torch.tensor([5])},
            {
                "role": "assistant",
                "content": "generated",
                "token_ids": torch.tensor([6, 7]),
                "generation_logprobs": torch.tensor([0.1, 0.2]),
            },
        ]
    ]

    flat, _input_lengths, prompt_flat = _flatten_rollout_message_log_for_tq(
        message_logs,
        torch.tensor([5]),
        pad_token_id=0,
        make_sequence_length_divisible_by=1,
    )

    assert torch.equal(prompt_flat["token_ids"], torch.tensor([[1, 2, 3, 4, 5]]))
    assert torch.equal(
        flat["token_loss_mask"],
        torch.tensor([[0, 0, 0, 0, 0, 1, 1]]),
    )
    assert torch.allclose(
        flat["generation_logprobs"],
        torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 0.1, 0.2]]),
    )
