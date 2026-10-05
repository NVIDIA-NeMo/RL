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

"""GPU parity for value split training versus the existing whole-batch API."""

from pathlib import Path

import pytest
import ray
import torch

pytest.importorskip("megatron.bridge")

from nemo_rl.algorithms.loss.loss_functions import MseValueLossConfig, MseValueLossFn
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.value.lm_value import Value
from nemo_rl.models.value.tq_value import _aggregate_train_results
from tests.unit.models.value.test_megatron_value_worker import (
    _apply_config_updates,
    _create_value_test_config,
)

pytestmark = [pytest.mark.mcore, pytest.mark.hf_gated]


@pytest.mark.timeout(600)
@pytest.mark.parametrize(
    "tp,pp,cp,updates",
    [
        (1, 1, 1, {}),
        (2, 1, 1, {"sequence_parallel": True}),
        (1, 2, 1, {"sequence_packing": True}),
        (
            1,
            1,
            2,
            {
                "sequence_packing": True,
                "context_parallel_size": 2,
                "precision": "bfloat16",
            },
        ),
        (1, 1, 1, {"dynamic_batching": True}),
    ],
)
def test_value_split_parity(tiny_qwen2_model_path, tmp_path, tp, pp, cp, updates):
    cluster = RayVirtualCluster(
        name="value-split-parity",
        bundle_ct_per_node_list=[2],
        use_gpus=True,
        num_gpus_per_node=2,
        max_colocated_worker_groups=1,
    )
    model = None
    try:
        config = _create_value_test_config(tiny_qwen2_model_path, tp=tp, pp=pp, cp=cp)
        _apply_config_updates(config, updates)
        tokenizer = get_tokenizer(config["tokenizer"])
        model = Value(cluster=cluster, config=config, tokenizer=tokenizer)
        torch.manual_seed(42)
        data = BatchedDataDict(
            {
                "input_ids": torch.randint(0, 151000, (8, 64)),
                "input_lengths": torch.full((8,), 64, dtype=torch.int32),
                "attention_mask": torch.ones(8, 64),
                "token_mask": torch.ones(8, 64),
                "sample_mask": torch.tensor([1.0, 1, 0, 1, 1, 1, 1, 1]),
                "returns": torch.randn(8, 64) * 0.1,
                "values": torch.zeros(8, 64),
            }
        )
        data["token_mask"][:, :8] = 0
        data["token_mask"][0, -8:] = 0
        loss_fn = MseValueLossFn(MseValueLossConfig(cliprange=0.5))
        weights = str(tmp_path / "initial" / "weights")
        model.prepare_for_training()
        model.save_checkpoint(weights_path=weights)
        reference = [model.train(data, loss_fn) for _ in range(2)]
        model.finish_training()
        model.prepare_for_inference()
        expected_values = model.get_values(data)["values"].cpu()
        model.finish_inference()
        model.shutdown()
        model = None
        model = Value(
            cluster=cluster,
            config=config,
            tokenizer=tokenizer,
            weights_path=Path(weights),
            name_prefix="value_split",
        )
        model.prepare_for_training()
        wg = model.worker_group
        replicas = ["tensor_parallel", "pipeline_parallel", "context_parallel"]
        for expected in reference:
            ray.get(
                wg.run_all_workers_single_data(
                    "begin_train_step_presharded", loss_fn=loss_fn, gbs=8, mbs=2
                )
            )
            for offset in (0, 4):
                chunk = BatchedDataDict(
                    {key: tensor[offset : offset + 4] for key, tensor in data.items()}
                )
                kwargs = {}
                if model.use_sequence_packing:
                    model.sequence_packing_args["max_tokens_per_microbatch"] = config[
                        "sequence_packing"
                    ]["train_mb_tokens"]
                    kwargs["sequence_packing_args"] = model.sequence_packing_args
                elif model.use_dynamic_batches:
                    model.dynamic_batching_args["max_tokens_per_microbatch"] = config[
                        "dynamic_batching"
                    ]["train_mb_tokens"]
                    kwargs["dynamic_batching_args"] = model.dynamic_batching_args
                shards = chunk.shard_by_batch_size(
                    model.sharding_annotations.get_axis_size("data_parallel"), **kwargs
                )
                if kwargs:
                    shards, _ = shards
                wg.get_all_worker_results(
                    wg.run_all_workers_sharded_data(
                        "train_microbatch",
                        data=shards,
                        in_sharded_axes=["data_parallel"],
                        replicate_on_axes=replicas,
                        output_is_replicated=replicas,
                    )
                )
            results = ray.get(
                wg.run_all_workers_single_data("finish_train_step_presharded")
            )
            actual = _aggregate_train_results(
                [r for r in results if r["is_replica_leader"]]
            )
            torch.testing.assert_close(
                actual["loss"], expected["loss"], rtol=2e-3, atol=2e-4
            )
            torch.testing.assert_close(
                actual["grad_norm"], expected["grad_norm"], rtol=2e-3, atol=2e-4
            )
        model.finish_training()
        model.prepare_for_inference()
        actual_values = model.get_values(data)["values"].cpu()
        torch.testing.assert_close(actual_values, expected_values, rtol=2e-3, atol=2e-4)
    finally:
        if model is not None:
            model.shutdown()
        cluster.shutdown()
