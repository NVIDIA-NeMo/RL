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

"""Multi-GPU integration smokes for shared-prefix ``Policy.train``."""

import sys

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from torch.distributed.tensor import DTensor
from transformers import (
    LlamaConfig,
    LlamaForCausalLM,
    PreTrainedTokenizerFast,
    Qwen3Config,
    Qwen3ForCausalLM,
    Qwen3MoeConfig,
    Qwen3MoeForCausalLM,
)

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


def _save_tiny_causal_lm(
    model_path,
    model_family: str,
    *,
    num_hidden_layers: int = 2,
) -> None:
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
    config_kwargs = dict(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=8,
        max_position_embeddings=128,
        attention_dropout=0.0,
        tie_word_embeddings=False,
        pad_token_id=0,
        eos_token_id=1,
        use_cache=False,
    )
    if model_family == "qwen3":
        model = Qwen3ForCausalLM(Qwen3Config(**config_kwargs))
    elif model_family == "llama":
        model = LlamaForCausalLM(LlamaConfig(**config_kwargs))
    else:
        model = Qwen3MoeForCausalLM(
            Qwen3MoeConfig(
                **config_kwargs,
                moe_intermediate_size=64,
                decoder_sparse_step=1,
                mlp_only_layers=[],
                num_experts=4,
                num_experts_per_tok=2,
                norm_topk_prob=False,
                router_aux_loss_coef=0.03125,
                use_sliding_window=False,
                output_router_logits=False,
            )
        )
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
        response_start = 4 + (row % 4) * 32 + (row // 4) * response_length
        input_ids[row, prompt_length : prompt_length + response_length] = torch.arange(
            response_start,
            response_start + response_length,
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


def _snapshot_local_moe_parameters(policy: Policy) -> list[dict[str, torch.Tensor]]:
    """Copy each rank's local router/expert shards for update assertions."""
    state_dicts = policy.run_all_workers_single_data("return_state_dict")
    snapshots = []
    for state_dict in state_dicts:
        local_state = {}
        for name, tensor in state_dict.items():
            if ".mlp.gate." not in name and ".mlp.experts." not in name:
                continue
            local_tensor = tensor.to_local() if isinstance(tensor, DTensor) else tensor
            local_state[name] = local_tensor.detach().cpu().clone()
        snapshots.append(local_state)
    return snapshots


def _any_parameter_changed(
    before: list[dict[str, torch.Tensor]],
    after: list[dict[str, torch.Tensor]],
    name_fragment: str,
) -> bool:
    return any(
        not torch.equal(before_rank[name], after_rank[name])
        for before_rank, after_rank in zip(before, after)
        for name in before_rank
        if name_fragment in name
    )


@pytest.mark.automodel
@pytest.mark.timeout(420)
@pytest.mark.parametrize(
    (
        "model_family,world_size,tp_size,dp_size,shard_size,groups_per_shard,"
        "compact_tokens"
    ),
    [
        ("qwen3", 4, 1, 4, 2, 1, 36),
        ("qwen3", 2, 2, 1, 8, 2, 112),
        ("qwen3", 4, 4, 1, 8, 2, 112),
        ("llama", 4, 1, 4, 2, 1, 36),
        ("llama", 2, 2, 1, 8, 2, 112),
        ("llama", 4, 4, 1, 8, 2, 112),
    ],
    ids=[
        "qwen3_dp4_tp1",
        "qwen3_dp1_tp2",
        "qwen3_dp1_tp4",
        "llama_dp4_tp1",
        "llama_dp1_tp2",
        "llama_dp1_tp4",
    ],
)
def test_shared_prefix_policy_train_fsdp2(
    tmp_path,
    monkeypatch,
    model_family,
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

    model_path = tmp_path / f"tiny_{model_family}_shared_prefix"
    _save_tiny_causal_lm(model_path, model_family)
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
        name=f"shared_prefix_policy_train_{model_family}",
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
            name_prefix=f"shared_prefix_policy_{model_family}",
        )
        assert policy.data_parallel_size == dp_size

        dense_data, train_data = _make_rollouts()
        assert train_data["input_lengths"].tolist() == [26] * 8
        policy.prepare_for_lp_inference()
        before = policy.get_logprobs(dense_data)["logprobs"]
        train_data["prev_logprobs"] = before.clone()
        train_data["generation_logprobs"] = before.clone()
        train_data["reference_policy_logprobs"] = before.clone()

        # DP=4 forces four compact bins; TP-only cases keep all rollouts in one
        # DP bin and replicate that physical layout across tensor-parallel ranks.
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


@pytest.mark.automodel
@pytest.mark.timeout(420)
@pytest.mark.parametrize("expert_parallel_size", [1, 4], ids=["ep1", "ep4"])
def test_shared_prefix_policy_train_native_qwen3_moe(
    tmp_path,
    monkeypatch,
    expert_parallel_size,
):
    """Train native Qwen3-MoE with logical aux/load semantics under EP."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("TRANSFORMERS_OFFLINE", "1")
    monkeypatch.setenv("TOKENIZERS_PARALLELISM", "false")
    monkeypatch.setitem(ACTOR_ENVIRONMENT_REGISTRY, _DTENSOR_V2_ACTOR, sys.executable)

    model_path = tmp_path / "tiny_qwen3_moe_native_shared_prefix"
    # One layer makes an aux-only step reach the router but not an earlier
    # layer's experts, so the parameter-update assertions are unambiguous.
    _save_tiny_causal_lm(model_path, "qwen3_moe", num_hidden_layers=1)
    config = create_test_config(
        model_name=str(model_path),
        tp=1,
        dtensor_v2=True,
        precision="bfloat16",
        expert_parallel_size=expert_parallel_size,
        sequence_packing_enabled=True,
        activation_checkpointing=True,
        automodel_kwargs={
            "force_hf": False,
            "moe_overrides": {"gate_bias_update_factor": 0.0625},
            "backend": {
                "_target_": (
                    "nemo_automodel.components.models.common.utils.BackendConfig"
                ),
                "attn": "sdpa",
                "linear": "torch",
                "rms_norm": "torch_fp32",
                "rope_fusion": False,
                "experts": "torch_mm",
                "dispatcher": "torch",
            },
        },
    )
    config["shared_prefix_training"] = True
    config["generation"]["backend"] = "vllm"
    config["train_global_batch_size"] = 8
    config["train_micro_batch_size"] = 2
    config["logprob_batch_size"] = 2
    config["learning_rate"] = 1e-2
    config["optimizer"]["kwargs"]["lr"] = 1e-2
    config["optimizer"]["kwargs"]["weight_decay"] = 0.0
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
        name=f"shared_prefix_native_qwen3_moe_ep{expert_parallel_size}",
        bundle_ct_per_node_list=[4],
        use_gpus=True,
        num_gpus_per_node=4,
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
            name_prefix=f"shared_prefix_native_moe_ep{expert_parallel_size}",
        )
        # EP overlaps DP: every rank still consumes a distinct rollout shard.
        assert policy.data_parallel_size == 4

        initial_state_dicts = policy.run_all_workers_single_data("return_state_dict")
        for state_dict in initial_state_dicts:
            moe_parameters = {
                name: tensor
                for name, tensor in state_dict.items()
                if ".mlp.gate." in name or ".mlp.experts." in name
            }
            assert moe_parameters
            assert all(
                tensor.dtype == torch.float32 for tensor in moe_parameters.values()
            )

        dense_data, train_data = _make_rollouts()
        policy.prepare_for_lp_inference()
        before_logprobs = policy.get_logprobs(dense_data)["logprobs"]
        train_data["prev_logprobs"] = before_logprobs.clone()
        train_data["generation_logprobs"] = before_logprobs.clone()
        train_data["reference_policy_logprobs"] = before_logprobs.clone()

        policy.prepare_for_training()
        before_parameters = _snapshot_local_moe_parameters(policy)
        assert all(snapshot for snapshot in before_parameters)
        train_data["advantages"].zero_()
        aux_only_result = policy.train(
            train_data,
            ClippedPGLossFn(ClippedPGLossConfig(reference_policy_kl_penalty=0.0)),
        )
        after_aux_parameters = _snapshot_local_moe_parameters(policy)

        assert torch.isfinite(aux_only_result["loss"]).all()
        assert aux_only_result["loss"].item() == pytest.approx(0.0, abs=1e-7)
        assert torch.isfinite(aux_only_result["grad_norm"]).all()
        assert aux_only_result["grad_norm"].item() > 0
        assert _any_parameter_changed(
            before_parameters, after_aux_parameters, ".mlp.gate."
        )
        assert not _any_parameter_changed(
            before_parameters, after_aux_parameters, ".mlp.experts."
        )
        correction_bias_names = {
            name
            for snapshot in after_aux_parameters
            for name in snapshot
            if name.endswith(".e_score_correction_bias")
        }
        assert len(correction_bias_names) == 1
        correction_bias_name = correction_bias_names.pop()
        assert any(
            not torch.equal(before[correction_bias_name], after[correction_bias_name])
            for before, after in zip(before_parameters, after_aux_parameters)
        )
        reference_bias = after_aux_parameters[0][correction_bias_name]
        assert reference_bias.dtype == torch.float32
        for rank_snapshot in after_aux_parameters[1:]:
            torch.testing.assert_close(
                rank_snapshot[correction_bias_name],
                reference_bias,
                rtol=0.0,
                atol=0.0,
            )

        moe_metrics = aux_only_result["moe_metrics"]
        assert moe_metrics["logical_token_layer_events"] == pytest.approx(8 * 26)
        assert moe_metrics["logical_router_aux_loss_mean"] > 0
        assert 0 <= moe_metrics["logical_dead_expert_fraction_mean"] <= 1
        assert 0 < moe_metrics["logical_expert_diversity_mean"] <= 1
        assert moe_metrics["logical_expert_utilization_min"] >= 0
        assert moe_metrics["logical_expert_utilization_max"] > 0
        assert all(
            torch.isfinite(torch.tensor(value)) for value in moe_metrics.values()
        )

        policy.prepare_for_lp_inference()
        before_main_logprobs = policy.get_logprobs(dense_data)["logprobs"]
        train_data["prev_logprobs"] = before_main_logprobs.clone()
        train_data["generation_logprobs"] = before_main_logprobs.clone()
        train_data["reference_policy_logprobs"] = before_main_logprobs.clone()
        train_data["advantages"] = train_data["token_mask"].clone()
        policy.prepare_for_training()
        before_main_parameters = _snapshot_local_moe_parameters(policy)
        result = policy.train(
            train_data,
            ClippedPGLossFn(ClippedPGLossConfig(reference_policy_kl_penalty=0.01)),
        )
        after_parameters = _snapshot_local_moe_parameters(policy)

        assert torch.isfinite(result["loss"]).all()
        assert torch.isfinite(result["grad_norm"]).all()
        assert result["grad_norm"].item() > 0
        assert _any_parameter_changed(
            before_main_parameters, after_parameters, ".mlp.experts."
        )

        policy.prepare_for_lp_inference()
        after_logprobs = policy.get_logprobs(dense_data)["logprobs"]
        response_mask = train_data["token_mask"].bool()
        assert (after_logprobs - before_logprobs).abs()[response_mask].max() > 1e-6
    finally:
        if policy is not None:
            policy.shutdown()
        cluster.shutdown()
