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

"""Multi-GPU integration smokes for Qwen3 shared-prefix ``Policy.train``."""

import sys

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast, Qwen3Config, Qwen3ForCausalLM

from nemo_rl.algorithms.grpo import _add_shared_prefix_training_metadata
from nemo_rl.algorithms.loss import ClippedPGLossConfig, ClippedPGLossFn
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.ray_actor_environment_registry import (
    ACTOR_ENVIRONMENT_REGISTRY,
)
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.automodel.shared_prefix import (
    SHARED_PREFIX_GROUP_IDS,
    SHARED_PREFIX_LENGTHS,
    build_shared_prefix_layout,
)
from nemo_rl.models.policy.lm_policy import Policy
from tests.unit.models.policy.test_dtensor_worker_v2 import create_test_config


_DTENSOR_V2_ACTOR = (
    "nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2"
)


def _save_tiny_qwen3(model_path) -> None:
    vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2}
    vocab.update({f"token_{index}": index for index in range(3, 128)})
    tokenizer_backend = Tokenizer(WordLevel(vocab=vocab, unk_token="<unk>"))
    tokenizer_backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer_backend,
        pad_token="<pad>",
        eos_token="<eos>",
        unk_token="<unk>",
    )

    torch.manual_seed(7)
    config = Qwen3Config(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        max_position_embeddings=128,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        pad_token_id=0,
        eos_token_id=1,
        use_cache=False,
    )
    model = Qwen3ForCausalLM(config)
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)


def _make_rollouts() -> tuple[BatchedDataDict, BatchedDataDict]:
    batch_size = 8
    prompt_length = 16
    response_length = 8
    environment_length = 2
    sequence_length = 32

    input_ids = torch.zeros(batch_size, sequence_length, dtype=torch.long)
    for row in range(batch_size):
        group = row // 4
        input_ids[row, :prompt_length] = torch.arange(
            3 + group * prompt_length,
            3 + (group + 1) * prompt_length,
        )
        input_ids[row, prompt_length : prompt_length + response_length] = (
            torch.arange(40 + row * response_length, 40 + (row + 1) * response_length)
            % 85
            + 3
        )
        input_ids[
            row,
            prompt_length + response_length : prompt_length
            + response_length
            + environment_length,
        ] = torch.tensor([125, 126])

    input_lengths = torch.full(
        (batch_size,),
        prompt_length + response_length + environment_length,
        dtype=torch.int32,
    )
    token_mask = torch.zeros(batch_size, sequence_length)
    token_mask[:, prompt_length : prompt_length + response_length] = 1
    dense_data = BatchedDataDict(
        {
            "input_ids": input_ids,
            "input_lengths": input_lengths,
        }
    )
    train_data = BatchedDataDict(
        {
            "input_ids": input_ids,
            "input_lengths": input_lengths,
            "token_mask": token_mask,
            "sample_mask": torch.ones(batch_size),
            "advantages": token_mask.clone(),
        }
    )
    _add_shared_prefix_training_metadata(
        train_data,
        {"shared_prefix_training": True},
        num_generations_per_prompt=4,
    )
    return dense_data, train_data


@pytest.mark.automodel
@pytest.mark.timeout(420)
@pytest.mark.parametrize(
    "world_size,tp_size,dp_size,shard_size,groups_per_shard,compact_tokens",
    [
        (4, 1, 4, 2, 1, 36),
        (2, 2, 1, 8, 2, 112),
    ],
    ids=["dp4_tp1", "dp1_tp2"],
)
def test_shared_prefix_policy_train_fsdp2(
    tmp_path,
    monkeypatch,
    world_size,
    tp_size,
    dp_size,
    shard_size,
    groups_per_shard,
    compact_tokens,
):
    """Exercise dense LP -> compact FSDP2 train -> dense LP with DP or TP."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")
    # The cluster test image already contains the Automodel runtime. Avoid building
    # a second uv actor environment so this test isolates the training path.
    monkeypatch.setitem(ACTOR_ENVIRONMENT_REGISTRY, _DTENSOR_V2_ACTOR, sys.executable)

    model_path = tmp_path / "tiny_qwen3_shared_prefix"
    _save_tiny_qwen3(model_path)
    config = create_test_config(
        model_name=str(model_path),
        tp=tp_size,
        dtensor_v2=True,
        precision="bfloat16",
        sequence_packing_enabled=True,
        activation_checkpointing=True,
        automodel_kwargs={"force_hf": True},
    )
    config["shared_prefix_training"] = True
    config["train_global_batch_size"] = 8
    config["train_micro_batch_size"] = 2
    config["logprob_batch_size"] = 2
    config["learning_rate"] = 1e-2
    config["optimizer"]["kwargs"]["lr"] = 1e-2
    config["dynamic_batching"]["enabled"] = False
    config["sequence_packing"].update(
        {
            "train_mb_tokens": 128,
            "logprob_mb_tokens": 128,
            "algorithm": "modified_first_fit_decreasing",
        }
    )
    config["make_sequence_length_divisible_by"] = 1

    cluster = RayVirtualCluster(
        name="shared_prefix_policy_train",
        bundle_ct_per_node_list=[world_size],
        use_gpus=True,
        num_gpus_per_node=world_size,
        max_colocated_worker_groups=1,
    )
    policy = None
    try:
        policy = Policy(
            tokenizer=get_tokenizer(config["tokenizer"]),
            config=config,
            init_optimizer=True,
            init_reference_model=False,
            cluster=cluster,
            name_prefix="shared_prefix_policy",
        )
        assert policy.data_parallel_size == dp_size

        dense_data, train_data = _make_rollouts()
        assert train_data["input_lengths"].tolist() == [26] * 8
        policy.prepare_for_lp_inference()
        before = policy.get_logprobs(dense_data)["logprobs"]
        train_data["prev_logprobs"] = before.clone()
        train_data["generation_logprobs"] = before.clone()
        train_data["reference_policy_logprobs"] = before.clone()

        # DP=4 forces four compact bins; TP=2 keeps all rollouts in one DP bin
        # and replicates that physical layout across its two tensor-parallel ranks.
        shards = policy._shard_for_train(train_data, batch_size=8)
        assert len(shards) == dp_size
        for shard in shards:
            assert shard.size == shard_size
            assert (
                torch.unique(shard[SHARED_PREFIX_GROUP_IDS]).numel() == groups_per_shard
            )
            layout = build_shared_prefix_layout(
                shard["input_ids"],
                shard["input_lengths"],
                shard[SHARED_PREFIX_LENGTHS],
                shard[SHARED_PREFIX_GROUP_IDS],
            )
            assert layout.compact_tokens == compact_tokens

        policy.prepare_for_training()
        result = policy.train(
            train_data,
            ClippedPGLossFn(ClippedPGLossConfig(reference_policy_kl_penalty=0.01)),
        )
        assert torch.isfinite(result["loss"]).all()
        assert torch.isfinite(result["grad_norm"]).all()
        assert result["grad_norm"].item() > 0
        assert len(result["all_mb_metrics"]["loss"]) == dp_size

        policy.prepare_for_lp_inference()
        after = policy.get_logprobs(dense_data)["logprobs"]
        response_mask = train_data["token_mask"].bool()
        max_response_delta = (after - before).abs()[response_mask].max()
        assert max_response_delta.item() > 1e-6
    finally:
        if policy is not None:
            policy.shutdown()
        cluster.shutdown()
