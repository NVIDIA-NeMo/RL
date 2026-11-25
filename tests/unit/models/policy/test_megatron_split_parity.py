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

from collections.abc import Sequence

import numpy as np
import pytest
import ray
import torch

# megatron.bridge is only available with the mcore extras; stop collection
# cleanly on non-mcore shards (same pattern as test_megatron_split_state).
pytest.importorskip("megatron.bridge")

from nemo_rl.algorithms.loss import ClippedPGLossConfig, ClippedPGLossFn
from nemo_rl.algorithms.loss.interfaces import group_token_mean_weights
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.data_plane.schema import GROUP_TOKEN_MEAN_WEIGHT
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


def _make_policy(
    model_name: str,
    tp: int,
    cluster_name: str,
    pp: int = 1,
    *,
    sequence_packing: bool = False,
):
    cluster = RayVirtualCluster(
        name=cluster_name,
        bundle_ct_per_node_list=[NUM_GPUS],
        use_gpus=True,
        num_gpus_per_node=NUM_GPUS,
        max_colocated_worker_groups=1,
    )
    config = create_megatron_test_config(model_name=model_name, tp=tp, pp=pp)
    if sequence_packing:
        config["sequence_packing"] = {
            "enabled": True,
            "train_mb_tokens": 2 * SEQ_LEN,
            "logprob_mb_tokens": 2 * SEQ_LEN,
            "algorithm": "modified_first_fit_decreasing",
        }
    tokenizer = get_tokenizer(config["tokenizer"])
    config["generation"] = configure_generation_config(config["generation"], tokenizer)
    policy = Policy(
        cluster=cluster,
        config=config,
        tokenizer=tokenizer,
        init_reference_model=False,
    )
    return policy, cluster, config


def _make_grpo_batch(
    batch_size: int, *, input_lengths: Sequence[int] | None = None
) -> BatchedDataDict:
    """Deterministic ClippedPG batch with non-trivial masks.

    By default every row spans SEQ_LEN and row 0 has a masked tail. Given
    ``input_lengths``, row ``i`` is right-padded after ``input_lengths[i]``
    tokens instead, with no extra tail masking.
    """
    torch.manual_seed(42)
    input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, SEQ_LEN))
    lengths = torch.full((batch_size,), SEQ_LEN)
    if input_lengths is not None:
        lengths = torch.tensor(input_lengths)
    attention_mask = (torch.arange(SEQ_LEN) < lengths[:, None]).float()
    token_mask = attention_mask.clone()
    token_mask[:, :8] = 0  # mask a "prompt" prefix
    if input_lengths is None:
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
    of ``train_microbatches_from_meta`` (which requires a TQ partition).

    Each step is dispatched as two half-batch ``train_microbatch`` calls so the
    worker accumulates objective sums and denominator counts across chunks;
    ``begin_train_step`` still receives the full ``gbs``."""
    wg = policy.worker_group
    policy.prepare_for_training()
    results = []
    for _step in range(NUM_STEPS):
        # Single-data fan-outs return plain ObjectRefs — consume with
        # ray.get (get_all_worker_results is for MultiWorkerFuture bundles
        # from sharded dispatch).
        ray.get(
            wg.run_all_workers_single_data(
                "begin_train_step_presharded",
                loss_fn=loss_fn,
                gbs=gbs,
                mbs=mbs,
            )
        )
        for half in data.shard_by_batch_size(2):
            sharded = policy._shard_for_train(half, gbs // 2)
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
        finish_results = ray.get(
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
@pytest.mark.parametrize("normalization", ["global_token_mean", "sequence_mean"])
@pytest.mark.parametrize("tp", [1, 2], ids=["2gpu_dp2", "2gpu_tp2"])
def test_split_train_matches_sync_train(tp, normalization, tiny_llama_model_path):
    """Grad-norm / loss-curve / metric A/B of split vs sync train()."""
    loss_fn = ClippedPGLossFn(
        ClippedPGLossConfig(
            use_importance_sampling_correction=True, normalization=normalization
        )
    )

    # Two fresh policies from the same checkpoint → identical initial
    # weights and optimizer state; the sync one runs (and is torn down)
    # first so each path sees identical GPU memory conditions.
    policy, cluster, config = _make_policy(
        tiny_llama_model_path, tp, f"parity-sync-tp{tp}-{normalization}"
    )
    gbs = config["train_global_batch_size"]
    mbs = config["train_micro_batch_size"]
    data = _make_grpo_batch(batch_size=gbs)
    try:
        sync_results = _run_sync(policy, data, loss_fn)
    finally:
        policy.shutdown()
        cluster.shutdown()

    policy, cluster, _ = _make_policy(
        tiny_llama_model_path, tp, f"parity-split-tp{tp}-{normalization}"
    )
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


# Rows whose generation logprobs are far from any policy's: exp(|error|) is
# ~1e21, while the other rows stay far below the threshold.
REJECTED_ROWS = (1, 6)
IN_LOSS_THRESHOLD = 1e12


def _make_in_loss_batch(batch_size: int) -> BatchedDataDict:
    data = _make_grpo_batch(batch_size)
    data["generation_logprobs"] = torch.full((batch_size, SEQ_LEN), -12.0)
    data["generation_logprobs"][list(REJECTED_ROWS)] = -60.0
    return data


def _assert_steps_match(expected: list[dict], actual: list[dict], label: str):
    for step, (exp_r, act_r) in enumerate(zip(expected, actual)):
        for key in ("loss", "grad_norm"):
            torch.testing.assert_close(
                act_r[key].float(),
                exp_r[key].float(),
                rtol=1e-3,
                atol=1e-5,
                msg=f"{label} step {step}: {key} diverged "
                f"(expected={exp_r[key]}, actual={act_r[key]})",
            )


@pytest.mark.hf_gated
@pytest.mark.timeout(900)
@pytest.mark.parametrize(("tp", "pp"), [(1, 1), (1, 2)], ids=["2gpu_dp2", "2gpu_pp2"])
def test_in_loss_filter_matches_prefiltering(tp, pp, tiny_llama_model_path):
    """In-loss filtering on both trainers equals dropping rows before training.

    PP=2 exercises the regular trainer's survivor-count broadcast from the
    last pipeline stage and the split API's per-stage metric reduce.
    """
    in_loss_fn = ClippedPGLossFn(
        ClippedPGLossConfig(
            force_on_policy_ratio=True,
            seq_logprob_error_in_loss=True,
            reference_policy_kl_penalty=0.1,
        ),
        seq_logprob_error_threshold=IN_LOSS_THRESHOLD,
    )
    prefilter_fn = ClippedPGLossFn(
        ClippedPGLossConfig(force_on_policy_ratio=True, reference_policy_kl_penalty=0.1)
    )

    def run(name: str, fn) -> list[dict]:
        policy, cluster, config = _make_policy(
            tiny_llama_model_path, tp, f"in-loss-{name}-tp{tp}-pp{pp}", pp=pp
        )
        gbs = config["train_global_batch_size"]
        data = _make_in_loss_batch(batch_size=gbs)
        try:
            if name == "prefilter":
                data["sample_mask"][list(REJECTED_ROWS)] = 0
                return _run_sync(policy, data, prefilter_fn)
            if name == "sync":
                return _run_sync(policy, data, in_loss_fn)
            return _run_split(
                policy,
                data,
                in_loss_fn,
                gbs=gbs,
                mbs=config["train_micro_batch_size"],
            )
        finally:
            policy.shutdown()
            cluster.shutdown()

    prefilter_results = run("prefilter", prefilter_fn)
    sync_results = run("sync", in_loss_fn)
    split_results = run("split", in_loss_fn)

    _assert_steps_match(prefilter_results, sync_results, "sync vs prefilter")
    _assert_steps_match(prefilter_results, split_results, "split vs prefilter")
    for step, (sync_r, split_r) in enumerate(zip(sync_results, split_results)):
        sync_mb = sync_r["all_mb_metrics"]
        split_mb = split_r["all_mb_metrics"]
        assert sum(sync_mb["num_masked_seqs_by_logprob_error"]) == len(REJECTED_ROWS)
        assert set(split_mb) == set(sync_mb)
        for key in sorted(sync_mb):
            sync_val = _reduce_metric(key, sync_mb[key])
            split_val = _reduce_metric(key, split_mb[key])
            assert split_val == pytest.approx(sync_val, rel=1e-3, abs=1e-5), (
                f"step {step}: metric {key!r} diverged "
                f"(sync={sync_val}, split={split_val})"
            )


GROUP_SIZE = 2
# Prompt group of each row. Each group has one row in each half-batch
# train_microbatch chunk; without packing the split run also puts those rows
# on different DP ranks (rows 0-1 and 4-5 on rank 0, rows 2-3 and 6-7 on rank 1).
GROUP_INDICES = torch.tensor([0, 1, 2, 3, 2, 3, 0, 1])
# Ragged case: per-row input lengths (8 prompt tokens each) and a row with
# sample_mask=0. Valid response tokens per row are 56, 4, 32, 20, 52, 0, 8, 6,
# so the group token masses T_g are 64, 10, 84 and 20.
RAGGED_INPUT_LENGTHS = (64, 12, 40, 28, 60, 20, 16, 14)
RAGGED_MASKED_ROW = 5


def _initial_logprobs(policy: Policy, data: BatchedDataDict) -> torch.Tensor:
    """Sampled-token logprobs of the untrained policy, ``[B, S]``."""
    policy.prepare_for_lp_inference()
    inputs = BatchedDataDict(
        {key: data[key] for key in ("input_ids", "input_lengths", "attention_mask")}
    )
    return policy.get_logprobs(inputs)["logprobs"].float().cpu()


def _cpu_global_token_mean_loss(
    loss_fn: ClippedPGLossFn, data: BatchedDataDict, logprobs: torch.Tensor
) -> float:
    """Synchronous global_token_mean loss of one full batch, computed on CPU."""
    mask = data["token_mask"][:, 1:] * data["sample_mask"].unsqueeze(-1)
    loss, _ = loss_fn(
        next_token_logprobs=logprobs[:, 1:],
        data=data,
        global_valid_seqs=data["sample_mask"].sum(),
        global_valid_toks=mask.sum(),
    )
    return loss.item()


@pytest.mark.hf_gated
@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    "lengths", ["equal", "ragged"], ids=["equal_lengths", "ragged_lengths_packed"]
)
def test_split_group_token_mean_matches_sync_oracle(lengths, tiny_llama_model_path):
    """Streaming group_token_mean against a sync global_token_mean oracle.

    The oracle runs sync train() with global_token_mean on a token_mask scaled
    per row by its group weight 1/T_g. The global token mass is then the group
    count G and the objective is exactly (1/G) sum_g (1/T_g) sum_tokens, which
    for equal group masses is plain global_token_mean. "ragged" has unequal
    T_g and a sample_mask=0 row, and the split run uses sequence packing.
    """
    tp = 1  # DP2
    ragged = lengths == "ragged"
    oracle_fn = ClippedPGLossFn(
        ClippedPGLossConfig(
            use_importance_sampling_correction=True, normalization="global_token_mean"
        )
    )
    group_fn = ClippedPGLossFn(
        ClippedPGLossConfig(
            use_importance_sampling_correction=True, normalization="group_token_mean"
        )
    )

    policy, cluster, config = _make_policy(
        tiny_llama_model_path, tp, f"parity-sync-group-oracle-{lengths}"
    )
    gbs = config["train_global_batch_size"]
    mbs = config["train_micro_batch_size"]
    assert gbs == len(GROUP_INDICES) == len(RAGGED_INPUT_LENGTHS)
    data = _make_grpo_batch(
        batch_size=gbs,
        input_lengths=RAGGED_INPUT_LENGTHS if ragged else [SEQ_LEN] * gbs,
    )
    if ragged:
        data["sample_mask"][RAGGED_MASKED_ROW] = 0
    weights = group_token_mean_weights(
        [f"s{row}" for row in range(gbs)],
        data["token_mask"],
        data["sample_mask"],
        group_indices=GROUP_INDICES,
        expected_group_size=GROUP_SIZE,
    )
    if ragged:
        assert torch.unique(weights).numel() > 1
    else:
        assert torch.unique(weights).numel() == 1
    data[GROUP_TOKEN_MEAN_WEIGHT] = weights
    plain_data = BatchedDataDict(
        {key: value for key, value in data.items() if key != GROUP_TOKEN_MEAN_WEIGHT}
    )
    oracle_data = BatchedDataDict(
        {**plain_data, "token_mask": data["token_mask"] * weights.unsqueeze(-1)}
    )
    try:
        initial_logprobs = _initial_logprobs(policy, plain_data)
        sync_results = _run_sync(policy, oracle_data, oracle_fn)
    finally:
        policy.shutdown()
        cluster.shutdown()

    policy, cluster, _ = _make_policy(
        tiny_llama_model_path,
        tp,
        f"parity-split-group-token-mean-{lengths}",
        sequence_packing=ragged,
    )
    try:
        split_results = _run_split(policy, data, group_fn, gbs=gbs, mbs=mbs)
    finally:
        policy.shutdown()
        cluster.shutdown()

    # Step-0 loss recomputed on CPU from the untrained policy's logprobs: the
    # oracle must reproduce the sync run, and in the ragged case plain
    # global_token_mean on the unscaled mask must miss it by far more than the
    # parity tolerance, so wrong per-group weighting cannot pass.
    oracle_step0 = _cpu_global_token_mean_loss(oracle_fn, oracle_data, initial_logprobs)
    plain_step0 = _cpu_global_token_mean_loss(oracle_fn, plain_data, initial_logprobs)
    print(
        f"[{lengths}] step 0 CPU loss: oracle={oracle_step0:.6g} "
        f"plain global_token_mean={plain_step0:.6g}"
    )
    for step, (sync_r, split_r) in enumerate(zip(sync_results, split_results)):
        print(
            f"[{lengths}] step {step}: loss split={float(split_r['loss']):.6g} "
            f"oracle={float(sync_r['loss']):.6g}; "
            f"grad_norm split={float(split_r['grad_norm']):.6g} "
            f"oracle={float(sync_r['grad_norm']):.6g}"
        )

    for step, (sync_r, split_r) in enumerate(zip(sync_results, split_results)):
        torch.testing.assert_close(
            split_r["loss"],
            sync_r["loss"],
            rtol=1e-3,
            atol=1e-5,
            msg=f"step {step}: group loss diverged from the oracle "
            f"(sync={sync_r['loss']}, split={split_r['loss']})",
        )
        torch.testing.assert_close(
            split_r["grad_norm"].float(),
            sync_r["grad_norm"].float(),
            rtol=1e-3,
            atol=1e-5,
            msg=f"step {step}: grad_norm diverged from the oracle "
            f"(sync={sync_r['grad_norm']}, split={split_r['grad_norm']})",
        )
    assert oracle_step0 == pytest.approx(
        float(sync_results[0]["loss"]), rel=1e-3, abs=1e-5
    )
    if ragged:
        assert abs(plain_step0 - oracle_step0) > 1e-2 * abs(oracle_step0)
    else:
        assert plain_step0 == pytest.approx(oracle_step0, rel=1e-5)
