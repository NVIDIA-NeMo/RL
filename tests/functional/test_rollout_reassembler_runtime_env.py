"""Finalize captured tokens remotely with Gym absent from the driver.

Run from the base NeMo-RL environment:
    uv run --no-sync python -m tests.functional.test_rollout_reassembler_runtime_env

The registered reassembler environment supplies Gym to both the staging task
and finalizer actor. This needs CPU workers only, with no model or generation.
"""

from __future__ import annotations

import importlib.util
import os
from typing import Any

import ray
import torch

from nemo_rl.data_plane import DataPlaneConfig, build_data_plane_client
from nemo_rl.data_plane.tq_token_sink import STAGING_FIELDS, TQTokenSink
from nemo_rl.experience.rollout_reassembler_actor import (
    ReassemblyRequest,
    RolloutReassemblerActorConfig,
    create_rollout_reassembler_actors,
)
from nemo_rl.utils.venvs import make_actor_runtime_env

STAGING_PARTITION = "reassembler_env_staging"
CANONICAL_PARTITION = "reassembler_env_canonical"
ROLLOUT_ID = "reassembler_env_g0"
CANONICAL_FIELDS = [
    "input_ids",
    "input_lengths",
    "generation_logprobs",
    "token_mask",
    "sample_mask",
    "prompt_ids_for_adv",
    "total_reward",
    "mask_sample",
    "truncated",
]


@ray.remote(num_cpus=1, max_calls=1)
def stage_fixture(dp_config: DataPlaneConfig) -> dict[str, Any]:
    # Deferred: the fixture imports Gym, which must remain absent on the driver.
    from tests.unit.data_plane.token_capture_test_fixtures import (
        build_fixture_artifacts,
    )

    client = build_data_plane_client(dp_config, bootstrap=False)
    records, receipt, _ = build_fixture_artifacts(
        "worked_example", rollout_id=ROLLOUT_ID
    )
    sink = TQTokenSink(client, staging_partition=STAGING_PARTITION)
    for record in records:
        assert sink.stage(record).ok
    # The driver owns TQ shutdown; close() here would kill its controller.
    return receipt.model_dump()


def main() -> None:
    assert importlib.util.find_spec("nemo_gym") is None, (
        "Run this regression test in the base environment without the nemo_gym extra"
    )
    assert os.environ.get("NEMO_RL_PY_EXECUTABLES_SYSTEM") != "1", (
        "This test requires separate actor environments"
    )
    dp_config: DataPlaneConfig = {
        "enabled": True,
        "impl": "transfer_queue",
        "backend": "simple",
        "claim_meta_poll_interval_s": 0.5,
        "simple": {"storage_capacity": 32, "num_storage_units": 1},
    }
    # RAY_ADDRESS overrides even an explicit address="local" in ray.init().
    os.environ.pop("RAY_ADDRESS", None)
    ray.init(address="local", num_cpus=4, num_gpus=0, include_dashboard=False)
    try:
        client = build_data_plane_client(dp_config)
        try:
            client.register_partition(
                partition_id=STAGING_PARTITION,
                fields=list(STAGING_FIELDS),
                num_samples=8,
                consumer_tasks=["finalize"],
            )
            client.register_partition(
                partition_id=CANONICAL_PARTITION,
                fields=CANONICAL_FIELDS,
                num_samples=8,
                consumer_tasks=["train"],
            )
            runtime_env = make_actor_runtime_env(
                "nemo_rl.experience.rollout_reassembler_actor.RolloutReassemblerActor"
            )
            receipt = ray.get(
                stage_fixture.options(runtime_env=runtime_env).remote(dp_config),
                timeout=120,
            )
            actors = create_rollout_reassembler_actors(
                dp_config,
                RolloutReassemblerActorConfig(
                    partition_id=CANONICAL_PARTITION,
                    staging_partition=STAGING_PARTITION,
                    pad_token_id=0,
                    router_replay_enabled=False,
                    defer_routed_experts_to_policy=False,
                    max_seq_len=4096,
                ),
                num_workers=1,
            )
            try:
                result = ray.get(
                    actors[0].finalize.remote(
                        ReassemblyRequest(
                            group_id="reassembler_env",
                            rollout_ids=(ROLLOUT_ID,),
                            canonical_sample_ids=(ROLLOUT_ID,),
                            receipts=(receipt,),
                            rewards=(1.0,),
                            fallback_weight_version=4,
                            prompt_idx=17,
                            mask_sample=(False,),
                        )
                    ),
                    timeout=120,
                )
                assert not result.dropped
                assert result.valid_row_count == result.total_row_count == 1
                assert result.meta is not None
                assert result.meta.sample_ids == [ROLLOUT_ID]
                assert (result.group_min_wv, result.group_max_wv) == (4, 4)
                rows = client.get_samples(
                    sample_ids=[ROLLOUT_ID],
                    partition_id=CANONICAL_PARTITION,
                    select_fields=CANONICAL_FIELDS,
                )
                assert torch.as_tensor(rows["input_ids"][0]).tolist() == [
                    10,
                    11,
                    12,
                    13,
                    20,
                    21,
                    22,
                ]
                assert torch.as_tensor(rows["token_mask"][0]).tolist() == [
                    0,
                    0,
                    1,
                    1,
                    0,
                    1,
                    1,
                ]
                torch.testing.assert_close(
                    torch.as_tensor(rows["generation_logprobs"][0]),
                    torch.tensor([0, 0, -0.1, -0.2, 0, -0.3, -0.4]),
                )
                assert torch.as_tensor(rows["sample_mask"]).flatten().tolist() == [1.0]
                assert torch.as_tensor(rows["total_reward"]).flatten().tolist() == [1.0]
                assert importlib.util.find_spec("nemo_gym") is None
            finally:
                for actor in actors:
                    ray.kill(actor)
        finally:
            client.close()
    finally:
        ray.shutdown()
    print("Remote finalization passed with Gym absent from the driver.")


if __name__ == "__main__":
    main()
