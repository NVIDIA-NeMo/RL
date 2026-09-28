# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import httpx
import pytest
import torch

from nemo_rl.data_plane.adapters.noop import NoOpDataPlaneClient
from nemo_rl.data_plane.column_io import kv_first_write, read_columns
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from tests.mock_stack.components import Generation, Policy
from tests.mock_stack.runtime import GenerationHandle, Trainer
from tests.mock_stack.servers import GenerationServer
from tests.mock_stack.test_components import batch


def test_trainer_reads_the_actual_data_plane_batch():
    plane = NoOpDataPlaneClient()
    data = batch()
    data.update(sample_mask=torch.ones(2), generation_logprobs=torch.zeros(2, 3))
    plane.register_partition("train", [*data, "prev_logprobs"], 8, ["train"])
    meta = kv_first_write(
        BatchedDataDict(data),
        sample_ids=["a", "b"],
        dp_client=plane,
        partition_id="train",
    )
    policy = Policy()
    trainer = Trainer(policy, plane)
    trainer.get_logprobs_from_meta(meta)
    logprobs = read_columns(plane, meta, ["prev_logprobs"])["prev_logprobs"]
    for index, length in enumerate(data["input_lengths"]):
        assert torch.equal(
            logprobs[index, :length], policy.logprobs(data["input_ids"])[index, :length]
        )
    trainer.begin_train_step(None)
    trainer.train_microbatches_from_meta(meta, train_fields=tuple(data))
    assert policy.steps == 0
    trainer.finish_train_step()
    expected = Policy()
    expected.train(batch())
    assert torch.equal(policy.export_weights(), expected.export_weights())
    assert trainer.batches[0][0] == ["a", "b"]
    with pytest.raises(RuntimeError, match="no batch"):
        trainer.finish_train_step()


def test_generation_server_starts_and_closes():
    handle = GenerationHandle(GenerationServer(Generation(), []))
    url = handle.dp_openai_server_base_urls[0]
    try:
        assert httpx.get(url + "/models").json()["data"][0]["id"] == "cpu"
    finally:
        handle.close()
    with pytest.raises(httpx.ConnectError):
        httpx.get(url + "/models", timeout=1)
