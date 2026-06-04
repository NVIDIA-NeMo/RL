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
"""Parity tests for the SingleController split-API state machine.

The split path (``begin_train_step`` → ``train_microbatch`` × K →
``finish_train_step``) must produce gradients and post-step parameters
identical (within fp tolerance) to the sync ``train()`` path. This file
asserts that equivalence end-to-end on a real GPU worker.

Math sanity check covered by these tests:
  ``masked_mean(values, mask, N) = sum(values·mask)/(N+ε)`` is linear in
  1/N, so per-microbatch backward with ``global_valid_seqs = global_valid_toks
  = tensor(1.0)`` deposits raw ``d(sum)/dθ`` into ``.grad``. The single
  ``1/N`` rescale at finish recovers the sync-path gradient.

All tests require GPUs; gated via ``pytest.mark.gpu``. They also drive
worker actors via ``policy.worker_group.run_all_workers_*`` directly to
exercise the backend state machine without depending on the TQ/DataPlane
plumbing (which is covered separately under ``tests/unit/data_plane/``).
"""

from __future__ import annotations

import copy

import pytest
import ray
import torch

from nemo_rl.models.policy.lm_policy import Policy
from tests.unit.test_utils import SimpleLossFn

# Reuse the existing v2 worker test fixtures for config + cluster setup.
from tests.unit.models.policy.test_dtensor_worker_v2 import (
    create_test_batch,
    create_test_config,
    two_gpu_virtual_cluster,  # noqa: F401 — fixture, used implicitly
)


def _all_param_state(policy: Policy) -> dict[str, torch.Tensor]:
    """Snapshot every worker-rank-0 parameter tensor as CPU float32."""
    state = ray.get(policy.worker_group.workers[0].return_state_dict.remote())
    return {
        k: v.detach().to(torch.float32).cpu()
        for k, v in state.items()
        if isinstance(v, torch.Tensor)
    }


def _drive_split_path(
    policy: Policy,
    data,
    loss_fn,
    gbs: int,
    mbs: int,
    *,
    step_id: str = "test-step-000",
) -> list[dict]:
    """Drive begin → train_microbatch → finish on every worker.

    Calls backend methods directly on the worker actors (bypassing
    ``TQWorkerMixin.*_presharded`` so this test doesn't need a TQ
    DataPlane setup). Returns per-worker results from ``finish_train_step``.
    """
    workers = policy.worker_group.workers

    # begin on every rank
    ray.get(
        [
            w.begin_train_step.remote(
                step_id=step_id,
                loss_fn=loss_fn,
                gbs=gbs,
                mbs=mbs,
            )
            for w in workers
        ]
    )

    # One train_microbatch call carrying the whole batch — the worker's
    # internal bin iteration handles the per-mb fwd+bwd.
    ray.get([w.train_microbatch.remote(step_id=step_id, data=data) for w in workers])

    # finish on every rank — drives the 1/N rescale + opt.step + sched.step
    results = ray.get([w.finish_train_step.remote(step_id=step_id) for w in workers])
    return results


@pytest.mark.gpu
@pytest.mark.hf_gated
@pytest.mark.automodel
@pytest.mark.timeout(360)
@pytest.mark.parametrize("dtensor_v2", [False, True])
def test_split_train_step_parity_token_level(
    two_gpu_virtual_cluster,  # noqa: F811
    tiny_llama_model_path,
    dtensor_v2,
):
    """Sync ``train()`` vs split begin/microbatch/finish on identical state.

    Asserts post-step model parameters match within fp tolerance. Run on
    both DTensor v1 and v2 to validate the matching implementations.
    """
    config = create_test_config(
        model_name=tiny_llama_model_path,
        tp=2,
        cp=1,
        dtensor_v2=dtensor_v2,
    )
    data = create_test_batch(
        batch_size=config["train_global_batch_size"],
        mode="train",
    )
    loss_fn = SimpleLossFn()

    # ── sync path ──────────────────────────────────────────────────────
    policy_sync = Policy(
        tokenizer=None,
        config=config,
        init_optimizer=True,
        init_reference_model=False,
        cluster=two_gpu_virtual_cluster,
        name_prefix=f"sync_{int(dtensor_v2)}",
    )
    try:
        policy_sync.prepare_for_training()
        sync_result = policy_sync.train(copy.deepcopy(data), loss_fn)
        sync_params = _all_param_state(policy_sync)
    finally:
        policy_sync.shutdown()

    # ── split path on a fresh policy with same init seed ──────────────
    policy_split = Policy(
        tokenizer=None,
        config=config,
        init_optimizer=True,
        init_reference_model=False,
        cluster=two_gpu_virtual_cluster,
        name_prefix=f"split_{int(dtensor_v2)}",
    )
    try:
        policy_split.prepare_for_training()
        split_results = _drive_split_path(
            policy_split,
            data=copy.deepcopy(data),
            loss_fn=loss_fn,
            gbs=config["train_global_batch_size"],
            mbs=config["train_micro_batch_size"],
        )
        split_params = _all_param_state(policy_split)
    finally:
        policy_split.shutdown()

    # Loss: sync emits a single tensor; split returns one dict per rank.
    split_loss_rank0 = split_results[0].get("global_loss")
    assert split_loss_rank0 is not None, "split path must surface global_loss"
    sync_loss = sync_result["loss"]
    if isinstance(sync_loss, torch.Tensor):
        sync_loss_v = sync_loss.detach().cpu().to(torch.float32)
    else:
        sync_loss_v = torch.tensor(float(sync_loss))
    split_loss_v = split_loss_rank0.detach().cpu().to(torch.float32)
    # Reduce to scalar for comparison if multi-element
    sync_scalar = sync_loss_v.sum().item()
    split_scalar = split_loss_v.sum().item()
    assert abs(sync_scalar - split_scalar) <= max(1e-3, 1e-3 * abs(sync_scalar)), (
        f"global_loss mismatch: sync={sync_scalar:.6f} split={split_scalar:.6f}"
    )

    # Post-step parameters should match (within fp tolerance).
    assert set(sync_params.keys()) == set(split_params.keys()), (
        f"parameter key set mismatch: sync-only={set(sync_params) - set(split_params)} "
        f"split-only={set(split_params) - set(sync_params)}"
    )
    for name, sync_p in sync_params.items():
        split_p = split_params[name]
        torch.testing.assert_close(
            sync_p,
            split_p,
            atol=5e-4,
            rtol=5e-4,
            msg=lambda m, n=name: f"parameter '{n}' diverged after step:\n{m}",
        )


@pytest.mark.gpu
@pytest.mark.hf_gated
@pytest.mark.automodel
@pytest.mark.timeout(360)
@pytest.mark.parametrize("dtensor_v2", [False, True])
def test_split_train_step_parity_seq_packing(
    two_gpu_virtual_cluster,  # noqa: F811
    tiny_llama_model_path,
    dtensor_v2,
):
    """Same parity assertion but under sequence packing — exercises the
    multi-bin-per-call path inside ``train_microbatch`` and the DP rank
    bin-count dummy padding.
    """
    config = create_test_config(
        model_name=tiny_llama_model_path,
        tp=2,
        cp=1,
        dtensor_v2=dtensor_v2,
        sequence_packing_enabled=True,
    )
    # Force dynamic_batching off so the seq-packing path is taken.
    config["dynamic_batching"]["enabled"] = False

    data = create_test_batch(
        batch_size=config["train_global_batch_size"],
        mode="train",
    )
    loss_fn = SimpleLossFn()

    policy_sync = Policy(
        tokenizer=None,
        config=config,
        init_optimizer=True,
        init_reference_model=False,
        cluster=two_gpu_virtual_cluster,
        name_prefix=f"sync_pack_{int(dtensor_v2)}",
    )
    try:
        policy_sync.prepare_for_training()
        sync_result = policy_sync.train(copy.deepcopy(data), loss_fn)
        sync_params = _all_param_state(policy_sync)
    finally:
        policy_sync.shutdown()

    policy_split = Policy(
        tokenizer=None,
        config=config,
        init_optimizer=True,
        init_reference_model=False,
        cluster=two_gpu_virtual_cluster,
        name_prefix=f"split_pack_{int(dtensor_v2)}",
    )
    try:
        policy_split.prepare_for_training()
        _drive_split_path(
            policy_split,
            data=copy.deepcopy(data),
            loss_fn=loss_fn,
            gbs=config["train_global_batch_size"],
            mbs=config["train_micro_batch_size"],
        )
        split_params = _all_param_state(policy_split)
    finally:
        policy_split.shutdown()

    # Sanity that we actually exercised the seq-packing path
    assert config["sequence_packing"]["enabled"]
    assert not config["dynamic_batching"]["enabled"]
    assert sync_result is not None

    for name, sync_p in sync_params.items():
        split_p = split_params[name]
        torch.testing.assert_close(
            sync_p,
            split_p,
            atol=5e-4,
            rtol=5e-4,
            msg=lambda m, n=name: f"parameter '{n}' diverged under seq-packing:\n{m}",
        )


@pytest.mark.gpu
@pytest.mark.hf_gated
@pytest.mark.automodel
@pytest.mark.timeout(180)
@pytest.mark.parametrize("dtensor_v2", [False, True])
def test_split_state_machine_lifecycle(
    two_gpu_virtual_cluster,  # noqa: F811
    tiny_llama_model_path,
    dtensor_v2,
):
    """Lifecycle invariants on a live worker.

    Asserts:
      - ``begin_train_step`` twice without finish raises.
      - ``train_microbatch`` without an open step raises.
      - ``abort_train_step`` is idempotent and clears state.
      - ``finish_train_step`` after abort raises (state was cleared).
    """
    config = create_test_config(
        model_name=tiny_llama_model_path,
        tp=2,
        cp=1,
        dtensor_v2=dtensor_v2,
    )
    policy = Policy(
        tokenizer=None,
        config=config,
        init_optimizer=True,
        init_reference_model=False,
        cluster=two_gpu_virtual_cluster,
        name_prefix=f"lifecycle_{int(dtensor_v2)}",
    )
    try:
        policy.prepare_for_training()
        loss_fn = SimpleLossFn()
        worker = policy.worker_group.workers[0]

        # double begin should raise
        ray.get(
            worker.begin_train_step.remote(
                step_id="step-a", loss_fn=loss_fn, gbs=4, mbs=1
            )
        )
        with pytest.raises(Exception):
            ray.get(
                worker.begin_train_step.remote(
                    step_id="step-b", loss_fn=loss_fn, gbs=4, mbs=1
                )
            )

        # abort idempotency
        ray.get(worker.abort_train_step.remote(step_id="step-a"))
        ray.get(worker.abort_train_step.remote(step_id="step-a"))  # no-op

        # post-abort, train_microbatch and finish should raise (no open step)
        with pytest.raises(Exception):
            data = create_test_batch(batch_size=4)
            ray.get(worker.train_microbatch.remote(step_id="step-a", data=data))
        with pytest.raises(Exception):
            ray.get(worker.finish_train_step.remote(step_id="step-a"))

        # new begin works after abort
        ray.get(
            worker.begin_train_step.remote(
                step_id="step-c", loss_fn=loss_fn, gbs=4, mbs=1
            )
        )
        ray.get(worker.abort_train_step.remote(step_id="step-c"))
    finally:
        policy.shutdown()
