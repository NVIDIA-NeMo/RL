# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""Two-GPU V1/full-batch SC regression, paired with MCore's split parity suite.

Runs the same GRPO batches through ``Policy.train()`` and through the
``begin_train_step`` / ``train_microbatch`` / ``finish_train_step`` path on two
policies built from the same checkpoint, and compares the loss, the grad norm
and every per-microbatch metric of each optimizer step.
"""

import itertools
from typing import Any

import pytest
import ray
import torch

pytest.importorskip("nemo_automodel")

from nemo_rl.algorithms.loss import ClippedPGLossConfig, ClippedPGLossFn
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.generation import configure_generation_config
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.models.policy.tq_policy import _aggregate_train_results
from tests.unit.models.policy.test_automodel_worker import create_test_config

pytestmark = pytest.mark.automodel

NUM_GPUS = 2
NUM_STEPS = 2
SEQ_LEN = 64
# The two policies are separate processes on the same GPUs. Deterministic
# kernels keep them this close; a normalization or grad-path divergence is
# orders of magnitude larger.
RTOL = 1e-5
ATOL = 1e-7


@pytest.fixture
def tiny_moe_model_path(tiny_llama_model_path, tmp_path):
    """Native MoE with real expert dispatch; reuse the existing test tokenizer."""
    from transformers import AutoTokenizer, Qwen3MoeConfig, Qwen3MoeForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(tiny_llama_model_path)
    config = Qwen3MoeConfig(
        vocab_size=len(tokenizer),
        hidden_size=256,
        intermediate_size=512,
        moe_intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
        num_experts=4,
        num_experts_per_tok=2,
        router_aux_loss_coef=0.0,
        tie_word_embeddings=False,
    )
    model = Qwen3MoeForCausalLM(config)
    path = tmp_path / "tiny_moe"
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    return path


def _make_policy(
    model_name: str,
    cluster_name: str,
    *,
    expert_parallel_size: int,
    logprob_chunk_size: int | None,
) -> tuple[Policy, RayVirtualCluster]:
    cluster = RayVirtualCluster(
        name=cluster_name,
        bundle_ct_per_node_list=[NUM_GPUS],
        use_gpus=True,
        num_gpus_per_node=NUM_GPUS,
        max_colocated_worker_groups=1,
    )
    # bf16 as in the shipped recipes; the grouped-GEMM expert backend requires it.
    config = create_test_config(
        model_name, precision="bfloat16", expert_parallel_size=expert_parallel_size
    )
    automodel_cfg = config["automodel_cfg"]
    if expert_parallel_size == 1:
        # Shipped dense recipes omit the key; setup treats that as EP=1.
        del automodel_cfg["expert_parallel_size"]
    else:
        automodel_cfg["automodel_kwargs"] = {
            "backend": {
                "_target_": "nemo_automodel.components.models.common.utils.BackendConfig",
                "attn": "te",
                "linear": "te",
                "rms_norm": "te",
                "experts": "gmm",
                "dispatcher": "hybridep",
                "rope_fusion": False,
                "enable_hf_state_dict_adapter": True,
            }
        }
    automodel_cfg["deterministic"] = True
    automodel_cfg["env_vars"] = {
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
    }
    # Automodel SC rejects dynamic batching, so both paths use fixed microbatches.
    config["dynamic_batching"]["enabled"] = False
    config["logprob_chunk_size"] = logprob_chunk_size
    tokenizer = get_tokenizer(config["tokenizer"])
    config["generation"] = configure_generation_config(config["generation"], tokenizer)
    policy = Policy(
        cluster=cluster,
        config=config,
        tokenizer=tokenizer,
        init_reference_model=False,
    )
    return policy, cluster


def _make_grpo_batch(batch_size: int) -> BatchedDataDict:
    """Deterministic batch with ragged lengths, a masked prefix and a masked row."""
    generator = torch.Generator().manual_seed(731)
    input_ids = torch.randint(100, 1000, (batch_size, SEQ_LEN), generator=generator)
    lengths = torch.tensor(
        [SEQ_LEN - 3 * (row % 4) for row in range(batch_size)], dtype=torch.int32
    )
    attention_mask = (torch.arange(SEQ_LEN)[None, :] < lengths[:, None]).float()
    token_mask = attention_mask.clone()
    token_mask[:, :8] = 0
    sample_mask = torch.ones(batch_size)
    sample_mask[-1] = 0
    return BatchedDataDict(
        {
            "input_ids": input_ids,
            "input_lengths": lengths,
            "attention_mask": attention_mask,
            "token_mask": token_mask,
            "sample_mask": sample_mask,
            "advantages": torch.randn(batch_size, SEQ_LEN, generator=generator),
        }
    )


def _logprob_inputs(data: BatchedDataDict) -> BatchedDataDict:
    return BatchedDataDict(
        {key: data[key] for key in ("input_ids", "input_lengths", "attention_mask")}
    )


def _add_logprobs(policy: Policy, data: BatchedDataDict) -> None:
    """Add logprob fields derived from the untrained policy's own logprobs."""
    policy.prepare_for_lp_inference()
    logprobs = policy.get_logprobs(_logprob_inputs(data))["logprobs"].float().cpu()
    data["prev_logprobs"] = logprobs
    data["generation_logprobs"] = logprobs - 0.02
    data["reference_policy_logprobs"] = logprobs + 0.01


def _loss_fns() -> list[ClippedPGLossFn]:
    """Both normalizations, with the policy ratio forced to one and computed."""
    return [
        ClippedPGLossFn(
            ClippedPGLossConfig(
                normalization=normalization,
                reference_policy_kl_penalty=0.01,
                force_on_policy_ratio=force_on_policy_ratio,
                use_importance_sampling_correction=True,
            )
        )
        for normalization, force_on_policy_ratio in itertools.product(
            ("global_token_mean", "sequence_mean"), (True, False)
        )
    ]


def _run_sync(policy: Policy, data: BatchedDataDict) -> list[dict[str, Any]]:
    policy.prepare_for_training()
    return [
        policy.train(data, loss_fn)
        for loss_fn in _loss_fns()
        for _step in range(NUM_STEPS)
    ]


def _dispatch_full_batch(policy: Policy, data: BatchedDataDict, gbs: int) -> None:
    worker_group = policy.worker_group
    worker_group.get_all_worker_results(
        worker_group.run_all_workers_sharded_data(
            "train_microbatch",
            data=policy._shard_for_train(data, gbs),
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


def _run_split(policy: Policy, data: BatchedDataDict) -> list[dict[str, Any]]:
    """Drive the worker split API the way TQPolicy does, minus the TQ data plane.

    Each step is one full-batch ``train_microbatch`` dispatch of driver-sharded
    tensors. The first step of every loss function is first opened, dispatched
    and aborted, and every step runs logprob inference between the dispatch and
    the commit, so leftover gradients or lost gradients would both show up.
    """
    worker_group = policy.worker_group
    gbs = policy.cfg["train_global_batch_size"]
    mbs = policy.cfg["train_micro_batch_size"]
    policy.prepare_for_training()
    results = []
    for loss_fn in _loss_fns():
        for step in range(NUM_STEPS):
            for abort in [True, False] if step == 0 else [False]:
                ray.get(
                    worker_group.run_all_workers_single_data(
                        "begin_train_step_presharded",
                        loss_fn=loss_fn,
                        gbs=gbs,
                        mbs=mbs,
                    )
                )
                _dispatch_full_batch(policy, data, gbs)
                if abort:
                    ray.get(
                        worker_group.run_all_workers_single_data(
                            "abort_train_step_presharded"
                        )
                    )
            policy.prepare_for_lp_inference(keep_train_buffers=True)
            policy.get_logprobs(_logprob_inputs(data))
            policy.prepare_for_training()
            finish_results = ray.get(
                worker_group.run_all_workers_single_data("finish_train_step_presharded")
            )
            leaders = [r for r in finish_results if r["is_replica_leader"]]
            results.append(_aggregate_train_results(leaders))
    return results


@pytest.mark.hf_gated
@pytest.mark.timeout(900)
@pytest.mark.parametrize("expert_parallel_size", [1, 2], ids=["dense", "moe_ep2"])
@pytest.mark.parametrize("logprob_chunk_size", [None, 16])
def test_full_batch_split_matches_sync_train(
    request, expert_parallel_size, logprob_chunk_size
):
    """Loss, grad norm and per-microbatch metrics of SC steps equal ``train()``.

    Two updates per loss function exercise optimizer and scheduler progression,
    with two microbatches per rank and a masked sample, with and without chunked
    training logprobs. Network TQ and live refits are covered by L1.
    """
    if torch.cuda.device_count() < NUM_GPUS:
        pytest.skip("Requires two CUDA GPUs")
    model_path = str(
        request.getfixturevalue(
            "tiny_llama_model_path"
            if expert_parallel_size == 1
            else "tiny_moe_model_path"
        )
    )
    case = f"ep{expert_parallel_size}-chunk{logprob_chunk_size}"

    # Two fresh policies from the same checkpoint have identical initial weights
    # and optimizer state; the sync one is torn down first so both paths see the
    # same GPU memory conditions.
    policy, cluster = _make_policy(
        model_path,
        f"automodel-parity-sync-{case}",
        expert_parallel_size=expert_parallel_size,
        logprob_chunk_size=logprob_chunk_size,
    )
    data = _make_grpo_batch(policy.cfg["train_global_batch_size"])
    try:
        _add_logprobs(policy, data)
        sync_results = _run_sync(policy, data)
    finally:
        policy.shutdown()
        cluster.shutdown()

    policy, cluster = _make_policy(
        model_path,
        f"automodel-parity-split-{case}",
        expert_parallel_size=expert_parallel_size,
        logprob_chunk_size=logprob_chunk_size,
    )
    try:
        split_results = _run_split(policy, data)
    finally:
        policy.shutdown()
        cluster.shutdown()

    assert len(split_results) == len(sync_results) == NUM_STEPS * len(_loss_fns())
    for step, (sync_r, split_r) in enumerate(zip(sync_results, split_results)):
        for key in ("loss", "grad_norm"):
            torch.testing.assert_close(
                split_r[key].float(),
                sync_r[key].float(),
                rtol=RTOL,
                atol=ATOL,
                msg=f"step {step}: {key} diverged "
                f"(sync={sync_r[key]}, split={split_r[key]})",
            )
        sync_mb = sync_r["all_mb_metrics"]
        split_mb = split_r["all_mb_metrics"]
        assert set(split_mb) == set(sync_mb), (
            f"step {step}: metric coverage differs: "
            f"sync-only={set(sync_mb) - set(split_mb)}, "
            f"split-only={set(split_mb) - set(sync_mb)}"
        )
        for key in sorted(sync_mb):
            assert split_mb[key] == pytest.approx(
                sync_mb[key], rel=RTOL, abs=ATOL, nan_ok=True
            ), (
                f"step {step}: metric {key!r} diverged "
                f"(sync={sync_mb[key]}, split={split_mb[key]})"
            )
