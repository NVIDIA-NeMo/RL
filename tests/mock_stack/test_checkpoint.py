# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import shutil
from collections import Counter
from pathlib import Path

import pytest
import ray
import torch

from nemo_rl.algorithms.single_controller_utils.rollout_checkpoint import (
    RolloutSnapshotManifest,
)
from nemo_rl.environments.gym_checkpoint import gym_checkpoint_continuations
from tests.mock_stack.components import Policy, weight_digest
from tests.mock_stack.config import ComponentSpec, Prompt, Scenario
from tests.mock_stack.runner import GYM, ROOT, run


def assert_trace(observed, expected_calls, expected_steps):
    spans = [json.loads(line) for line in observed.trace_path.read_text().splitlines()]
    generation = [s for s in spans if s["name"] == "test.generation"]
    assert len(generation) == expected_calls
    assert sorted(
        (
            s["attributes"]["test.prompt"],
            s["attributes"]["test.sibling"],
            s["attributes"]["test.turn"],
            s["attributes"]["test.sample_id"],
        )
        for s in generation
    ) == sorted((c.prompt, c.sibling, c.turn, c.capture_key) for c in observed.calls)
    assert len([s for s in spans if s["name"] == "test.policy.train"]) == expected_steps
    assert any(s["name"] == "test.refit" for s in spans)
    assert len({s["context"]["trace_id"] for s in spans}) == 1
    assert all(s["status"]["status_code"] != "ERROR" for s in spans)
    checkpoints = [s for s in spans if s["name"] == "test.checkpoint.prepare_commit"]
    for checkpoint in checkpoints:
        release = next(
            s
            for s in spans
            if s["name"] == "test.checkpoint.release"
            and s["attributes"] == checkpoint["attributes"]
        )
        assert checkpoint["end_time"] <= release["start_time"]
    assert checkpoints


@pytest.fixture
def cpu_ray(monkeypatch):
    monkeypatch.setenv("NEMO_RL_PY_EXECUTABLES_SYSTEM", "1")
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(ROOT), str(GYM)]))
    ray.init(num_cpus=8, num_gpus=0, include_dashboard=False)
    try:
        assert ray.cluster_resources().get("GPU", 0) == 0
        yield
    finally:
        ray.shutdown()


def test_cpu_controller_smoke(cpu_ray, tmp_path):
    scenario = Scenario(
        prompts=[Prompt(id="P4", turns=2, turn_seconds=[0.02, 0.02, 0.02])],
        siblings_per_prompt=3,
        policy=ComponentSpec(factory="tests.mock_stack.components:Policy"),
        generation=ComponentSpec(factory="tests.mock_stack.components:Generation"),
        refit=ComponentSpec(factory="tests.mock_stack.components:CopyRefit"),
    )
    observed = run(scenario, tmp_path)
    assert observed.result["train_steps"] == 1
    assert len(observed.batches) == 1
    assert sorted((call.sibling, call.turn) for call in observed.calls) == [
        (sibling, turn) for sibling in range(3) for turn in (1, 2)
    ]
    assert observed.batches[0][1]["total_reward"].tolist() == [1.0, 2.0, 3.0]
    assert_trace(observed, 6, 1)


@pytest.mark.timeout(300)
def test_checkpoint_restore(cpu_ray, tmp_path):
    scenario = Scenario.load(Path(__file__).with_name("checkpoint.yaml"))
    original = run(scenario, tmp_path)
    assert original.result["train_steps"] == 4
    assert len(original.calls) == 51
    assert [chr(batch["input_ids"][0, 1].item()) for _, batch in original.batches] == [
        "4",
        "2",
        "3",
        "1",
    ], f"Training-order timing precondition missed: calls={original.calls}"

    checkpoints = tmp_path / "checkpoints"
    assert (checkpoints / "step_2").is_dir()
    assert (checkpoints / "step_4").is_dir()
    snapshot = checkpoints / "step_2" / "rollout_snapshots" / "snapshot_000001"
    manifest = RolloutSnapshotManifest.from_mapping(
        json.loads((snapshot / "manifest.json").read_text())
    )
    identities = {
        call.capture_key: (call.prompt, call.sibling) for call in original.calls
    }
    expected_prefixes = {
        ("P1", 0): 2,
        ("P1", 1): 2,
        ("P1", 2): 2,
        ("P3", 1): 3,
        ("P3", 2): 3,
    }
    saved_prefixes = {
        identities[item.capture_key]: len(item.staging_keys)
        for item in gym_checkpoint_continuations(snapshot, manifest.gym_checkpoint)
    }
    assert saved_prefixes == expected_prefixes, (
        f"Checkpoint timing precondition missed: {saved_prefixes}; calls={original.calls}"
    )
    final_policy = Policy()
    final_policy.load_checkpoint(checkpoints / "step_4" / "policy" / "weights")
    third_step_policy = Policy()
    third_step_policy.load_checkpoint(checkpoints / "step_2" / "policy" / "weights")
    second_step_digest = weight_digest(third_step_policy.export_weights())
    third_step_policy.train(original.batches[2][1])
    shutil.rmtree(checkpoints / "step_4")
    ray.shutdown()
    ray.init(num_cpus=8, num_gpus=0, include_dashboard=False)
    restored = run(scenario, tmp_path)
    assert_trace(original, 51, 4)
    assert_trace(restored, 20, 2)

    assert Path(restored.restored_checkpoint) == snapshot
    assert restored.result["train_steps"] == 4
    assert len(restored.batches) == 2
    assert len(restored.calls) == 20, restored.calls
    first_resumed_calls = [
        call
        for call in restored.calls
        if (call.prompt == "P1" and call.turn == 3)
        or (call.prompt == "P3" and call.turn == 4)
    ]
    assert len(first_resumed_calls) == 5
    assert {call.weight_digest for call in first_resumed_calls} == {second_step_digest}
    restored_policy = Policy()
    restored_policy.load_checkpoint(checkpoints / "step_4" / "policy" / "weights")
    assert restored_policy.steps == final_policy.steps == 4
    assert torch.equal(restored_policy.export_weights(), final_policy.export_weights())
    for observed in (original, restored):
        assert weight_digest(third_step_policy.export_weights()) in {
            call.weight_digest for call in observed.calls
        }
    assert sorted(
        (call.prompt, call.sibling, call.turn) for call in restored.calls
    ) == sorted(
        [("P1", sibling, turn) for sibling in range(3) for turn in range(3, 7)]
        + [("P3", sibling, turn) for sibling in (1, 2) for turn in range(4, 8)]
    )
    for (original_ids, original_batch), (resumed_ids, resumed_batch) in zip(
        original.batches[2:], restored.batches, strict=True
    ):
        assert original_ids == resumed_ids
        for field in ("input_ids", "input_lengths", "token_mask", "total_reward"):
            torch.testing.assert_close(
                original_batch[field], resumed_batch[field], rtol=0, atol=0
            )

    assert (
        Counter(identities[call.rollout_id] for call in restored.restored_calls)
        == expected_prefixes
    )
    for call in restored.restored_calls:
        prompt, sibling = identities[call.rollout_id]
        group = 0 if prompt == "P3" else 1
        for observed in (original.batches[2 + group][1], restored.batches[group][1]):
            for field, saved in (
                ("input_ids", call.token_ids_delta),
                ("token_mask", call.token_mask_delta),
                ("generation_logprobs", call.generation_log_probs_delta),
            ):
                assert (
                    observed[field][sibling, call.prev_len : call.cum_len].tolist()
                    == saved
                )
    torch.testing.assert_close(
        original.batches[2][1]["generation_logprobs"][0],
        restored.batches[0][1]["generation_logprobs"][0],
        rtol=0,
        atol=0,
    )
