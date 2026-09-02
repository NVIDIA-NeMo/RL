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

"""GPU A/B for a synchronous critic step vs two split-API chunks."""

from pathlib import Path

import numpy as np
import pytest
import ray
import torch

pytest.importorskip("megatron.bridge")

from nemo_rl.algorithms.loss import MseValueLossConfig, MseValueLossFn
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.value.lm_value import Value
from nemo_rl.models.value.tq_value import _aggregate_train_results
from tests.unit.models.value.test_megatron_value_worker import (
    _create_value_test_config,
)

pytestmark = pytest.mark.mcore

NUM_GPUS = 2
SEQ_LEN = 64


def _make_value(model_name: str, cluster_name: str, **checkpoint_paths):
    cluster = RayVirtualCluster(
        name=cluster_name,
        bundle_ct_per_node_list=[NUM_GPUS],
        use_gpus=True,
        num_gpus_per_node=NUM_GPUS,
        max_colocated_worker_groups=1,
    )
    config = _create_value_test_config(model_name=model_name)
    tokenizer = get_tokenizer(config["tokenizer"])
    value = Value(
        cluster=cluster,
        config=config,
        tokenizer=tokenizer,
        **checkpoint_paths,
    )
    return value, cluster, config


def _make_batch(batch_size: int) -> BatchedDataDict:
    torch.manual_seed(42)
    attention_mask = torch.ones(batch_size, SEQ_LEN)
    token_mask = attention_mask.clone()
    token_mask[:, :8] = 0
    token_mask[0, -4:] = 0
    return BatchedDataDict(
        {
            "input_ids": torch.randint(0, 151000, (batch_size, SEQ_LEN)),
            "input_lengths": attention_mask.sum(dim=1).to(torch.int32),
            "attention_mask": attention_mask,
            "returns": torch.randn(batch_size, SEQ_LEN) * 0.1,
            "values": torch.randn(batch_size, SEQ_LEN) * 0.1,
            "token_mask": token_mask,
            "sample_mask": torch.ones(batch_size),
        }
    )


def _run_split(value: Value, data: BatchedDataDict, loss_fn, gbs: int, mbs: int):
    value.prepare_for_training()
    wg = value.worker_group
    ray.get(
        wg.run_all_workers_single_data(
            "begin_train_step_presharded", loss_fn=loss_fn, gbs=gbs, mbs=mbs
        )
    )
    for start in (0, gbs // 2):
        chunk = data.slice(start, start + gbs // 2)
        sharded, _ = chunk.shard_by_batch_size(
            value.sharding_annotations.get_axis_size("data_parallel"),
            batch_size=None,
        )
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
    finished = ray.get(
        wg.run_all_workers_single_data("finish_train_step_presharded")
    )
    value.finish_training()
    return _aggregate_train_results(
        [result for result in finished if result.get("is_replica_leader", True)]
    )


def _reduce_metric(key: str, values: list) -> float:
    if key.endswith("_min"):
        return float(np.min(values))
    if key.endswith("_max"):
        return float(np.max(values))
    if key in ("lr", "wd", "global_valid_seqs", "global_valid_toks"):
        return float(np.mean(values))
    return float(np.sum(values))


@pytest.mark.hf_gated
@pytest.mark.timeout(600)
def test_two_split_critic_chunks_match_one_sync_step(
    tiny_qwen2_model_path, tmp_path
):
    loss_fn = MseValueLossFn(MseValueLossConfig(scale=1.0, cliprange=0.5))
    sync_value, sync_cluster, config = _make_value(
        tiny_qwen2_model_path, "critic-parity-sync"
    )
    gbs = config["train_global_batch_size"]
    mbs = config["train_micro_batch_size"]
    data = _make_batch(gbs)
    weights_path = Path(tmp_path) / "initial" / "weights"
    optimizer_path = Path(tmp_path) / "initial" / "optimizer"
    try:
        sync_value.save_checkpoint(
            weights_path=str(weights_path), optimizer_path=str(optimizer_path)
        )
        sync_value.prepare_for_training()
        sync_result = sync_value.train(data, loss_fn)
        sync_value.finish_training()
    finally:
        sync_value.shutdown()
        sync_cluster.shutdown()

    split_value, split_cluster, _ = _make_value(
        tiny_qwen2_model_path,
        "critic-parity-split",
        weights_path=weights_path,
        optimizer_path=optimizer_path,
    )
    try:
        split_result = _run_split(split_value, data, loss_fn, gbs, mbs)
    finally:
        split_value.shutdown()
        split_cluster.shutdown()

    torch.testing.assert_close(
        split_result["loss"], sync_result["loss"], rtol=1e-3, atol=1e-5
    )
    torch.testing.assert_close(
        split_result["grad_norm"].float(),
        sync_result["grad_norm"].float(),
        rtol=1e-3,
        atol=1e-5,
    )
    assert set(split_result["all_mb_metrics"]) == set(
        sync_result["all_mb_metrics"]
    )
    for key in sorted(sync_result["all_mb_metrics"]):
        assert _reduce_metric(
            key, split_result["all_mb_metrics"][key]
        ) == pytest.approx(
            _reduce_metric(key, sync_result["all_mb_metrics"][key]),
            rel=1e-3,
            abs=1e-5,
        )
