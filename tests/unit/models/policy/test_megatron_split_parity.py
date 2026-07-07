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

"""GPU numerical-parity A/B: split-API vs sync ``train()`` on a real model.

PR #2683 review (F-TESTGAP): the split-state suite mocks out all gradient
math, so nothing there can observe a normalization or grad-path divergence.
This test runs the same GRPO batch through the sync ``train()`` path and
the ``begin_train_step`` / ``train_microbatch`` / ``finish_train_step``
split path on freshly-initialized identical models, for multiple optimizer
steps, and asserts the loss curve, grad norm, and every per-microbatch
metric agree. TP=2 additionally exercises the replica-leader dedup in the
split-path result aggregation.
"""

import numpy as np
import pytest
import torch

# megatron.bridge is only available with the mcore extras; stop collection
# cleanly on non-mcore shards (same pattern as test_megatron_split_state).
pytest.importorskip("megatron.bridge")

from nemo_rl.algorithms.loss import ClippedPGLossConfig, ClippedPGLossFn
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.models.policy.tq_policy import _aggregate_train_results
from tests.unit.models.policy.test_megatron_worker import create_megatron_test_config

pytestmark = pytest.mark.mcore

NUM_GPUS = 2
NUM_STEPS = 2
SEQ_LEN = 64
VOCAB_SIZE = 32000


def _make_policy(model_name: str, tp: int, cluster_name: str):
    cluster = RayVirtualCluster(
        name=cluster_name,
        bundle_ct_per_node_list=[NUM_GPUS],
        use_gpus=True,
        num_gpus_per_node=NUM_GPUS,
        max_colocated_worker_groups=1,
    )
    config = create_megatron_test_config(model_name=model_name, tp=tp, pp=1)
    tokenizer = get_tokenizer(config["tokenizer"])
    config["generation"] = configure_generation_config(config["generation"], tokenizer)
    policy = Policy(
        cluster=cluster,
        config=config,
        tokenizer=tokenizer,
        init_reference_model=False,
    )
    return policy, cluster, config


def _make_grpo_batch(batch_size: int) -> BatchedDataDict:
    """Deterministic ClippedPG batch with non-trivial masks."""
    torch.manual_seed(42)
    input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, SEQ_LEN))
    attention_mask = torch.ones(batch_size, SEQ_LEN)
    token_mask = torch.ones(batch_size, SEQ_LEN)
    token_mask[:, :8] = 0  # mask a "prompt" prefix
    token_mask[0, -4:] = 0  # ragged tail so valid counts are non-trivial
    return BatchedDataDict(
        {
            "input_ids": input_ids,
            "input_lengths": attention_mask.sum(dim=1).to(torch.int32),
            "attention_mask": attention_mask,
            "token_mask": token_mask,
            "sample_mask": torch.ones(batch_size),
            "advantages": torch.randn(batch_size, SEQ_LEN),
            "prev_logprobs": -torch.rand(batch_size, SEQ_LEN),
            "generation_logprobs": -torch.rand(batch_size, SEQ_LEN),
            "reference_policy_logprobs": -torch.rand(batch_size, SEQ_LEN),
        }
    )


def _run_sync(policy: Policy, data: BatchedDataDict, loss_fn) -> list[dict]:
    policy.prepare_for_training()
    results = [policy.train(data, loss_fn) for _ in range(NUM_STEPS)]
    policy.finish_training()
    return results


def _run_split(
    policy: Policy, data: BatchedDataDict, loss_fn, gbs: int, mbs: int
) -> list[dict]:
    """Drive the worker split API the way TQPolicy does, minus the TQ
    data plane: presharded begin/finish fan-out + replica-leader dedup +
    the shared ``_aggregate_train_results``; the microbatch dispatch uses
    the backend ``train_microbatch`` with driver-sharded tensors instead
    of ``train_microbatches_from_meta`` (which requires a TQ partition)."""
    wg = policy.worker_group
    policy.prepare_for_training()
    results = []
    for _step in range(NUM_STEPS):
        wg.get_all_worker_results(
            wg.run_all_workers_single_data(
                "begin_train_step_presharded",
                loss_fn=loss_fn,
                gbs=gbs,
                mbs=mbs,
            )
        )
        sharded = policy._shard_for_train(data, gbs)
        wg.get_all_worker_results(
            wg.run_all_workers_sharded_data(
                "train_microbatch",
                data=sharded,
                in_sharded_axes=["data_parallel"],
                replicate_on_axes=[
                    "context_parallel",
                    "tensor_parallel",
                    "pipeline_parallel",
                ],
                output_is_replicated=[
                    "context_parallel",
                    "tensor_parallel",
                    "pipeline_parallel",
                ],
            )
        )
        finish_results = wg.get_all_worker_results(
            wg.run_all_workers_single_data("finish_train_step_presharded")
        )
        leaders = [r for r in finish_results if r.get("is_replica_leader", True)]
        results.append(_aggregate_train_results(leaders))
    policy.finish_training()
    return results


def _reduce_metric(key: str, values: list) -> float:
    """Collapse a per-microbatch metric list the way grpo.py's reducer does."""
    if "_min" in key:
        return float(np.min(values))
    if "_max" in key:
        return float(np.max(values))
    if key in ("lr", "wd", "global_valid_seqs", "global_valid_toks"):
        return float(np.mean(values))
    return float(np.sum(values))


@pytest.mark.hf_gated
@pytest.mark.timeout(600)
@pytest.mark.parametrize("tp", [1, 2], ids=["2gpu_dp2", "2gpu_tp2"])
def test_split_train_matches_sync_train(tp, tiny_llama_model_path):
    """Grad-norm / loss-curve / metric A/B of split vs sync train()."""
    loss_fn = ClippedPGLossFn(
        ClippedPGLossConfig(use_importance_sampling_correction=True)
    )

    # Two fresh policies from the same checkpoint → identical initial
    # weights and optimizer state; the sync one runs (and is torn down)
    # first so each path sees identical GPU memory conditions.
    policy, cluster, config = _make_policy(
        tiny_llama_model_path, tp, f"parity-sync-tp{tp}"
    )
    gbs = config["train_global_batch_size"]
    mbs = config["train_micro_batch_size"]
    data = _make_grpo_batch(batch_size=gbs)
    try:
        sync_results = _run_sync(policy, data, loss_fn)
    finally:
        policy.shutdown()
        cluster.shutdown()

    policy, cluster, _ = _make_policy(tiny_llama_model_path, tp, f"parity-split-tp{tp}")
    try:
        split_results = _run_split(policy, data, loss_fn, gbs=gbs, mbs=mbs)
    finally:
        policy.shutdown()
        cluster.shutdown()

    for step, (sync_r, split_r) in enumerate(zip(sync_results, split_results)):
        torch.testing.assert_close(
            split_r["loss"],
            sync_r["loss"],
            rtol=1e-3,
            atol=1e-5,
            msg=f"step {step}: global loss diverged "
            f"(sync={sync_r['loss']}, split={split_r['loss']})",
        )
        torch.testing.assert_close(
            split_r["grad_norm"].float(),
            sync_r["grad_norm"].float(),
            rtol=1e-3,
            atol=1e-5,
            msg=f"step {step}: grad_norm diverged "
            f"(sync={sync_r['grad_norm']}, split={split_r['grad_norm']})",
        )

        sync_mb = sync_r["all_mb_metrics"]
        split_mb = split_r["all_mb_metrics"]
        assert set(split_mb) == set(sync_mb), (
            f"step {step}: metric coverage differs: "
            f"sync-only={set(sync_mb) - set(split_mb)}, "
            f"split-only={set(split_mb) - set(sync_mb)}"
        )
        for key in sorted(sync_mb):
            sync_val = _reduce_metric(key, sync_mb[key])
            split_val = _reduce_metric(key, split_mb[key])
            assert split_val == pytest.approx(sync_val, rel=1e-3, abs=1e-5), (
                f"step {step}: metric {key!r} diverged "
                f"(sync={sync_val}, split={split_val})"
            )
