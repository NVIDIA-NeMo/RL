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

"""GPU tests for pipeline parallelism in the Automodel policy worker.

Pipeline-parallel runs are compared against the non-pipelined path on the same
tiny model, data, and seed: token logprobs, training loss, gradient norm, and
logprobs after an optimizer step (which reflect the updated parameters).
"""

import copy
import os
import shutil
import tempfile

import pytest
import torch

from nemo_rl.algorithms.loss import ClippedPGLossConfig, ClippedPGLossFn, NLLLossFn
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.lm_policy import Policy
from tests.unit.conftest import TEST_ASSETS_DIR

pytestmark = pytest.mark.automodel

_NUM_LAYERS = 4
_SEQ_LEN = 32
_VOCAB_SIZE = 151936


@pytest.fixture(scope="module")
def tiny_untied_llama_path():
    """A four-layer Llama with separate embedding and LM head weights."""
    from transformers import AutoTokenizer, LlamaConfig, LlamaForCausalLM

    model_path = os.path.join(TEST_ASSETS_DIR, "tiny_llama_4l_untied_qwen2_tokenizer")
    torch.manual_seed(0)
    config = LlamaConfig(
        num_hidden_layers=_NUM_LAYERS,
        hidden_size=64,
        intermediate_size=128,
        num_attention_heads=2,
        num_key_value_heads=2,
        vocab_size=_VOCAB_SIZE,
        tie_word_embeddings=False,
    )
    model = LlamaForCausalLM(config=config)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B")
    shutil.rmtree(model_path, ignore_errors=True)
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    yield model_path


@pytest.fixture(scope="module")
def tiny_untied_qwen3_moe_path():
    """A four-layer Qwen3-MoE (Automodel's custom MoE implementation)."""
    from transformers import AutoTokenizer, Qwen3MoeConfig, Qwen3MoeForCausalLM

    model_path = os.path.join(
        TEST_ASSETS_DIR, "tiny_qwen3_moe_4l_untied_qwen2_tokenizer"
    )
    torch.manual_seed(0)
    config = Qwen3MoeConfig(
        num_hidden_layers=_NUM_LAYERS,
        hidden_size=64,
        intermediate_size=128,
        moe_intermediate_size=32,
        num_experts=4,
        num_experts_per_tok=2,
        num_attention_heads=2,
        num_key_value_heads=2,
        head_dim=32,
        vocab_size=_VOCAB_SIZE,
        router_aux_loss_coef=0.0,
        tie_word_embeddings=False,
    )
    model = Qwen3MoeForCausalLM(config=config)
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2-1.5B")
    shutil.rmtree(model_path, ignore_errors=True)
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    yield model_path


# fp32-capable kernels and the DTensor dispatcher, so pipelined and
# non-pipelined runs compute the same routing and expert outputs.
_MOE_AUTOMODEL_KWARGS = {
    "backend": {
        "_target_": "nemo_automodel.components.models.common.utils.BackendConfig",
        "attn": "sdpa",
        "linear": "torch",
        "rms_norm": "torch",
        "experts": "torch",
        "dispatcher": "torch",
        "fake_balanced_gate": False,
        "rope_fusion": False,
        "enable_hf_state_dict_adapter": True,
    }
}


def _policy_config(
    model_path: str,
    *,
    pp_size: int,
    pp_schedule: str = "1f1b",
    microbatches_per_step: int = 2,
    train_global_batch_size: int = 8,
    precision: str = "float32",
    tp_size: int = 1,
    cp_size: int = 1,
    ep_size: int = 1,
    defer_fsdp_grad_sync: bool = False,
    automodel_kwargs: dict | None = None,
) -> PolicyConfig:
    automodel_cfg = {
        "enabled": True,
        "checkpoint": {
            "model_save_format": "safetensors",
            "save_consolidated": "false",
            "single_rank_consolidation": False,
            "consolidation_timeout_minutes": 30,
        },
        "cpu_offload": False,
        "sequence_parallel": False,
        "activation_checkpointing": False,
        "tensor_parallel_size": tp_size,
        "context_parallel_size": cp_size,
        "pipeline_parallel_size": pp_size,
        "expert_parallel_size": ep_size,
        "defer_fsdp_grad_sync": defer_fsdp_grad_sync,
        "custom_parallel_plan": None,
        "automodel_kwargs": automodel_kwargs or {"force_hf": True},
        "lora_cfg": {"enabled": False},
    }
    if pp_size > 1:
        automodel_cfg["pipeline"] = {
            "pp_schedule": pp_schedule,
            "microbatches_per_step": microbatches_per_step,
            "layers_per_stage": None,
        }
    return {
        "model_name": model_path,
        "tokenizer": {"name": model_path},
        "train_global_batch_size": train_global_batch_size,
        "train_micro_batch_size": 1,
        "logprob_batch_size": 1,
        "precision": precision,
        "offload_optimizer_for_logprob": False,
        "max_total_sequence_length": _SEQ_LEN,
        "make_sequence_length_divisible_by": 1,
        "generation": {
            "backend": "hf",
            "temperature": 1.0,
            "top_p": 1.0,
            "top_k": None,
            "max_new_tokens": 8,
            "stop_token_ids": None,
            "stop_strings": None,
            "colocated": {
                "enabled": True,
                "resources": {"gpus_per_node": None, "num_nodes": None},
            },
        },
        "automodel_cfg": automodel_cfg,
        "dynamic_batching": {"enabled": False},
        "sequence_packing": {"enabled": False},
        "optimizer": {
            "name": "torch.optim.AdamW",
            "kwargs": {
                "lr": 1e-3,
                "weight_decay": 0.01,
                "betas": [0.9, 0.999],
                "eps": 1e-8,
                "foreach": False,
                "fused": False,
            },
        },
        "scheduler": {
            "name": "torch.optim.lr_scheduler.LinearLR",
            "kwargs": {"start_factor": 1.0, "end_factor": 0.5, "total_iters": 10},
        },
        "max_grad_norm": 1.0,
    }


def _batch(batch_size: int, *, seed: int = 7) -> BatchedDataDict:
    """Right-padded sequences of varying length with RL fields."""
    generator = torch.Generator().manual_seed(seed)
    input_ids = torch.randint(0, 1000, (batch_size, _SEQ_LEN), generator=generator)
    input_lengths = torch.randint(
        _SEQ_LEN // 2, _SEQ_LEN + 1, (batch_size,), generator=generator
    )
    positions = torch.arange(_SEQ_LEN).unsqueeze(0)
    token_mask = (positions < input_lengths.unsqueeze(1)).float()
    token_mask[:, : _SEQ_LEN // 4] = 0  # prompt tokens
    logprobs = -torch.rand(batch_size, _SEQ_LEN, generator=generator)
    return BatchedDataDict(
        {
            "input_ids": input_ids,
            "input_lengths": input_lengths.to(torch.int32),
            "token_mask": token_mask,
            "sample_mask": torch.ones(batch_size),
            "advantages": torch.randn(batch_size, _SEQ_LEN, generator=generator),
            "prev_logprobs": logprobs,
            "generation_logprobs": logprobs,
            "reference_policy_logprobs": logprobs,
        }
    )


def _clipped_pg_loss_fn() -> ClippedPGLossFn:
    return ClippedPGLossFn(ClippedPGLossConfig(reference_policy_kl_penalty=0.01))


def _cluster(num_gpus: int, name: str) -> RayVirtualCluster:
    return RayVirtualCluster(
        name=name,
        bundle_ct_per_node_list=[num_gpus],
        use_gpus=True,
        num_gpus_per_node=num_gpus,
        max_colocated_worker_groups=1,
    )


def _run_steps(
    config: PolicyConfig,
    num_gpus: int,
    name: str,
    loss_fn,
    *,
    num_steps: int = 2,
) -> dict[str, list]:
    """Logprobs, then ``num_steps`` x (train, logprobs) on a fresh policy."""
    cluster = _cluster(num_gpus, name)
    policy = None
    try:
        policy = Policy(
            cluster=cluster,
            config=config,
            tokenizer=get_tokenizer(config["tokenizer"]),
            init_reference_model=False,
            name_prefix=name,
        )
        data = _batch(config["train_global_batch_size"])
        out = {"logprobs": [], "loss": [], "grad_norm": [], "mb_loss": []}
        policy.prepare_for_lp_inference()
        out["logprobs"].append(policy.get_logprobs(data)["logprobs"])
        for _ in range(num_steps):
            policy.prepare_for_training()
            results = policy.train(copy.deepcopy(data), loss_fn)
            out["loss"].append(results["loss"])
            out["grad_norm"].append(results["grad_norm"])
            out["mb_loss"].append(torch.tensor(results["all_mb_metrics"]["loss"]))
            policy.prepare_for_lp_inference()
            out["logprobs"].append(policy.get_logprobs(data)["logprobs"])
        return out
    finally:
        if policy is not None:
            policy.shutdown()
        cluster.shutdown()


def _assert_runs_close(reference: dict[str, list], actual: dict[str, list]) -> None:
    for key in reference:
        assert len(reference[key]) == len(actual[key]), key
        for expected, got in zip(reference[key], actual[key]):
            torch.testing.assert_close(
                torch.as_tensor(got, dtype=torch.float32),
                torch.as_tensor(expected, dtype=torch.float32),
                rtol=1e-4,
                atol=1e-5,
                msg=lambda m, key=key: f"{key}: {m}",
            )


@pytest.fixture(scope="module")
def reference_runs(tiny_untied_llama_path):
    """Non-pipelined reference runs on one GPU, keyed by loss."""
    return {
        "nll": _run_steps(
            _policy_config(tiny_untied_llama_path, pp_size=1),
            1,
            "pp_ref_nll",
            NLLLossFn(),
        ),
        "clipped_pg": _run_steps(
            _policy_config(tiny_untied_llama_path, pp_size=1),
            1,
            "pp_ref_pg",
            _clipped_pg_loss_fn(),
        ),
    }


@pytest.mark.timeout(900)
@pytest.mark.parametrize(
    ("pp_schedule", "microbatches_per_step"),
    [("1f1b", 2), ("1f1b", 8), ("interleaved_1f1b", 2), ("interleaved_1f1b", 4)],
)
@pytest.mark.parametrize("loss_name", ["nll", "clipped_pg"])
def test_pipeline_matches_non_pipelined(
    tiny_untied_llama_path,
    reference_runs,
    pp_schedule,
    microbatches_per_step,
    loss_name,
):
    """PP=2 reproduces the non-pipelined logprobs, losses, and updates.

    microbatches_per_step=8 on a local batch of 8 is one schedule call per
    update; 2 is four calls whose gradients accumulate; 4 with interleaved
    stages also pads nothing but rounds the schedule differently.
    """
    loss_fn = NLLLossFn() if loss_name == "nll" else _clipped_pg_loss_fn()
    actual = _run_steps(
        _policy_config(
            tiny_untied_llama_path,
            pp_size=2,
            pp_schedule=pp_schedule,
            microbatches_per_step=microbatches_per_step,
        ),
        2,
        f"pp_{pp_schedule}_{microbatches_per_step}_{loss_name}",
        loss_fn,
    )
    _assert_runs_close(reference_runs[loss_name], actual)


@pytest.mark.timeout(900)
def test_pipeline_deferred_grad_sync_matches_non_pipelined(
    tiny_untied_llama_path, reference_runs
):
    """defer_fsdp_grad_sync=true reduce-scatters once per schedule call."""
    actual = _run_steps(
        _policy_config(tiny_untied_llama_path, pp_size=2, defer_fsdp_grad_sync=True),
        2,
        "pp_deferred_sync",
        NLLLossFn(),
    )
    _assert_runs_close(reference_runs["nll"], actual)


@pytest.mark.timeout(900)
def test_pipeline_padding_and_data_parallel(tiny_untied_llama_path, reference_runs):
    """PP=2 x DP=2 with a local batch that does not fill a schedule call.

    Each DP rank holds 4 samples and microbatches_per_step=3, so every update
    runs two schedule calls with two padding microbatches at the tail.
    """
    if torch.cuda.device_count() < 4:
        pytest.skip("needs 4 GPUs")
    actual = _run_steps(
        _policy_config(tiny_untied_llama_path, pp_size=2, microbatches_per_step=3),
        4,
        "pp_dp_padding",
        NLLLossFn(),
    )
    _assert_runs_close(reference_runs["nll"], actual)


@pytest.mark.timeout(900)
def test_pipeline_with_tensor_parallel(tiny_untied_llama_path, reference_runs):
    """PP=2 x TP=2 reproduces the non-pipelined run."""
    if torch.cuda.device_count() < 4:
        pytest.skip("needs 4 GPUs")
    actual = _run_steps(
        _policy_config(tiny_untied_llama_path, pp_size=2, tp_size=2),
        4,
        "pp_tp",
        NLLLossFn(),
    )
    _assert_runs_close(reference_runs["nll"], actual)


@pytest.mark.timeout(900)
def test_pipeline_with_context_parallel_matches_non_pipelined(tiny_untied_llama_path):
    """PP=2 with CP=2 inside each stage reproduces the non-pipelined CP=2 run.

    Both runs have one data-parallel rank and shard sequences the same way;
    only the layer split differs.
    """
    if torch.cuda.device_count() < 4:
        pytest.skip("needs 4 GPUs")

    def config(pp_size: int) -> PolicyConfig:
        return _policy_config(tiny_untied_llama_path, pp_size=pp_size, cp_size=2)

    reference = _run_steps(config(1), 2, "cp_ref", _clipped_pg_loss_fn())
    actual = _run_steps(config(2), 4, "pp_cp", _clipped_pg_loss_fn())
    _assert_runs_close(reference, actual)


@pytest.mark.timeout(900)
def test_pipeline_with_expert_parallel_matches_non_pipelined(
    tiny_untied_qwen3_moe_path,
):
    """PP=2 with EP=2 inside each stage reproduces the non-pipelined EP=2 run.

    Both runs have two data-parallel ranks, so every rank sees the same samples
    and the same expert placement; only the layer split differs.
    """
    if torch.cuda.device_count() < 4:
        pytest.skip("needs 4 GPUs")

    def config(pp_size: int) -> PolicyConfig:
        return _policy_config(
            tiny_untied_qwen3_moe_path,
            pp_size=pp_size,
            ep_size=2,
            microbatches_per_step=2,
            automodel_kwargs=_MOE_AUTOMODEL_KWARGS,
        )

    reference = _run_steps(config(1), 2, "moe_ep_ref", _clipped_pg_loss_fn())
    actual = _run_steps(config(2), 4, "moe_pp_ep", _clipped_pg_loss_fn())
    _assert_runs_close(reference, actual)


@pytest.mark.timeout(900)
def test_pipeline_logprobs_with_uneven_dp_shards(
    tiny_untied_llama_path, reference_runs
):
    """DP ranks with different amounts of data issue the same schedule calls.

    The two DP ranks hold 6 and 2 samples, so one needs three schedule calls and
    the other one; the planner pads the second rank instead of hanging.
    """
    if torch.cuda.device_count() < 4:
        pytest.skip("needs 4 GPUs")
    config = _policy_config(tiny_untied_llama_path, pp_size=2)
    data = _batch(config["train_global_batch_size"])
    cluster = _cluster(4, "pp_uneven_dp")
    policy = None
    try:
        policy = Policy(
            cluster=cluster,
            config=config,
            tokenizer=get_tokenizer(config["tokenizer"]),
            init_reference_model=False,
            name_prefix="pp_uneven_dp",
        )
        policy.prepare_for_lp_inference()
        futures = policy.worker_group.run_all_workers_sharded_data(
            "get_logprobs",
            data=[data.slice(0, 6), data.slice(6, 8)],
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
        logprobs = BatchedDataDict.from_batches(
            policy.worker_group.get_all_worker_results(futures)
        )["logprobs"]
        torch.testing.assert_close(
            logprobs, reference_runs["nll"]["logprobs"][0], rtol=1e-4, atol=1e-5
        )
    finally:
        if policy is not None:
            policy.shutdown()
        cluster.shutdown()


@pytest.mark.timeout(900)
def test_pipeline_reference_logprobs_and_checkpoint_resume(tiny_untied_llama_path):
    """Reference swap, then save, resume, and keep training under interleaved PP."""
    config = _policy_config(
        tiny_untied_llama_path, pp_size=2, pp_schedule="interleaved_1f1b"
    )
    data = _batch(config["train_global_batch_size"])
    loss_fn = NLLLossFn()
    cluster = _cluster(2, "pp_ckpt")
    policy = None
    try:
        policy = Policy(
            cluster=cluster,
            config=config,
            tokenizer=get_tokenizer(config["tokenizer"]),
            init_reference_model=True,
            name_prefix="pp_ckpt",
        )
        policy.prepare_for_lp_inference()
        initial_logprobs = policy.get_logprobs(data)["logprobs"]
        policy.prepare_for_training()
        policy.train(copy.deepcopy(data), loss_fn)

        policy.prepare_for_lp_inference()
        trained_logprobs = policy.get_logprobs(data)["logprobs"]
        reference_logprobs = policy.get_reference_policy_logprobs(data)[
            "reference_logprobs"
        ]
        torch.testing.assert_close(reference_logprobs, initial_logprobs)
        assert not torch.allclose(trained_logprobs, initial_logprobs)
        # The reference swap restores the trained weights.
        torch.testing.assert_close(
            policy.get_logprobs(data)["logprobs"], trained_logprobs
        )

        with tempfile.TemporaryDirectory() as ckpt_dir:
            weights_path = os.path.join(ckpt_dir, "weights")
            optimizer_path = os.path.join(ckpt_dir, "optimizer")
            policy.save_checkpoint(
                weights_path=weights_path,
                optimizer_path=optimizer_path,
                is_final_checkpoint=False,
            )
            policy.finalize_async_save()

            policy.prepare_for_training()
            continued = policy.train(copy.deepcopy(data), loss_fn)
            policy.prepare_for_lp_inference()
            continued_logprobs = policy.get_logprobs(data)["logprobs"]
            policy.shutdown()
            policy = None

            policy = Policy(
                cluster=cluster,
                config=config,
                tokenizer=get_tokenizer(config["tokenizer"]),
                init_reference_model=False,
                weights_path=weights_path,
                optimizer_path=optimizer_path,
                name_prefix="pp_ckpt_resumed",
            )
            policy.prepare_for_lp_inference()
            torch.testing.assert_close(
                policy.get_logprobs(data)["logprobs"], trained_logprobs
            )
            policy.prepare_for_training()
            resumed = policy.train(copy.deepcopy(data), loss_fn)
            policy.prepare_for_lp_inference()
            torch.testing.assert_close(resumed["loss"], continued["loss"])
            torch.testing.assert_close(resumed["grad_norm"], continued["grad_norm"])
            torch.testing.assert_close(
                policy.get_logprobs(data)["logprobs"], continued_logprobs
            )
    finally:
        if policy is not None:
            policy.shutdown()
        cluster.shutdown()


@pytest.mark.timeout(900)
def test_pipeline_refit_info_and_offload(tiny_untied_llama_path):
    """The PP weight manifest equals the non-PP one; offload frees and recovers."""
    infos = {}
    for pp_size, num_gpus in ((1, 1), (2, 2)):
        config = _policy_config(tiny_untied_llama_path, pp_size=pp_size)
        cluster = _cluster(num_gpus, f"pp_refit_{pp_size}")
        policy = None
        try:
            policy = Policy(
                cluster=cluster,
                config=config,
                tokenizer=get_tokenizer(config["tokenizer"]),
                init_reference_model=False,
                name_prefix=f"pp_refit_{pp_size}",
            )
            infos[pp_size] = policy.prepare_refit_info()
            if pp_size > 1:
                data = _batch(config["train_global_batch_size"])
                policy.prepare_for_training()
                policy.train(copy.deepcopy(data), NLLLossFn())
                policy.offload_before_refit()
                policy.offload_after_refit()
                policy.prepare_for_training()
                results = policy.train(copy.deepcopy(data), NLLLossFn())
                assert torch.isfinite(results["loss"]).all()
        finally:
            if policy is not None:
                policy.shutdown()
            cluster.shutdown()
    assert list(infos[2]) == list(infos[1])
    assert infos[2] == infos[1]


_CHUNKED_WORKER_FQN = (
    "tests.unit.models.policy.automodel_chunked_update_worker."
    "ChunkedUpdateAutomodelPolicyWorker"
)


def _chunked_update_run(
    config: PolicyConfig,
    num_gpus: int,
    name: str,
    chunk_sizes_by_dp_rank: list[list[int]],
    *,
    logprobs_after_first_chunk: bool = False,
) -> dict[str, list]:
    """Logprobs after one update fed as chunks; each DP rank has its own sizes."""
    from unittest.mock import patch

    from nemo_rl.distributed.ray_actor_environment_registry import (
        get_actor_python_env,
    )
    from nemo_rl.utils.venvs import create_local_venv_on_each_node

    base_fqn = (
        "nemo_rl.models.policy.workers.automodel_policy_worker.AutomodelPolicyWorker"
    )
    cluster = _cluster(num_gpus, name)
    policy = None
    try:
        # The test worker reuses the Automodel worker's environment.
        py_executable = get_actor_python_env(base_fqn)
        if py_executable.startswith("uv"):
            py_executable = create_local_venv_on_each_node(py_executable, base_fqn)
        with patch.dict(
            "nemo_rl.distributed.ray_actor_environment_registry."
            "ACTOR_ENVIRONMENT_REGISTRY",
            {_CHUNKED_WORKER_FQN: py_executable},
        ):
            policy = Policy(
                cluster=cluster,
                config=config,
                tokenizer=get_tokenizer(config["tokenizer"]),
                init_reference_model=False,
                name_prefix=name,
                worker_extension_cls_fqn=_CHUNKED_WORKER_FQN,
            )
        data = _batch(config["train_global_batch_size"])
        shards = []
        start = 0
        for sizes in chunk_sizes_by_dp_rank:
            shards.append(data.slice(start, start + sum(sizes)))
            start += sum(sizes)
        policy.prepare_for_training()
        futures = policy.worker_group.run_all_workers_sharded_data(
            "train_update_in_chunks",
            data=shards,
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
            common_kwargs={
                "loss_fn": NLLLossFn(),
                "chunk_sizes_by_dp_rank": chunk_sizes_by_dp_rank,
                "logprobs_after_first_chunk": logprobs_after_first_chunk,
            },
        )
        results = policy.worker_group.get_all_worker_results(futures)
        policy.prepare_for_lp_inference()
        return {
            "loss": [results[0]["global_loss"]],
            "grad_norm": [results[0]["grad_norm"]],
            "logprobs": [policy.get_logprobs(data)["logprobs"]],
        }
    finally:
        if policy is not None:
            policy.shutdown()
        cluster.shutdown()


def _first_update(reference: dict[str, list]) -> dict[str, list]:
    return {
        "loss": reference["loss"][:1],
        "grad_norm": reference["grad_norm"][:1],
        "logprobs": reference["logprobs"][1:2],
    }


@pytest.mark.timeout(900)
@pytest.mark.parametrize(
    ("pp_size", "chunk_sizes"),
    [(1, [8]), (1, [2, 6]), (1, [2, 2, 4]), (2, [8]), (2, [3, 5]), (2, [1, 4, 3])],
)
def test_update_split_into_chunks_matches_single_call(
    tiny_untied_llama_path, reference_runs, pp_size, chunk_sizes
):
    """One update fed as k chunks matches the whole update in one train() call."""
    actual = _chunked_update_run(
        _policy_config(tiny_untied_llama_path, pp_size=pp_size),
        pp_size,
        f"chunks_pp{pp_size}_{len(chunk_sizes)}",
        [chunk_sizes],
    )
    _assert_runs_close(_first_update(reference_runs["nll"]), actual)


@pytest.mark.timeout(900)
def test_pipeline_update_chunks_with_uneven_dp_shards(
    tiny_untied_llama_path, reference_runs
):
    """DP ranks feeding chunks of different sizes still match one train() call."""
    if torch.cuda.device_count() < 4:
        pytest.skip("needs 4 GPUs")
    actual = _chunked_update_run(
        _policy_config(tiny_untied_llama_path, pp_size=2),
        4,
        "chunks_uneven_dp",
        [[1, 4], [2, 1]],
    )
    _assert_runs_close(_first_update(reference_runs["nll"]), actual)


@pytest.mark.timeout(900)
@pytest.mark.parametrize(
    "pp_size",
    [
        1,
        pytest.param(
            2,
            marks=pytest.mark.xfail(
                reason=(
                    "A train/eval switch re-initializes the pipeline stages, which "
                    "clears accumulated FSDP gradients in the pinned PyTorch and "
                    "Automodel versions."
                ),
                strict=True,
            ),
        ),
    ],
)
def test_logprobs_between_chunks_keep_gradients(
    tiny_untied_llama_path, reference_runs, pp_size
):
    """Actor logprobs computed while an update is open do not change the update."""
    actual = _chunked_update_run(
        _policy_config(tiny_untied_llama_path, pp_size=pp_size),
        pp_size,
        f"chunks_eval_pp{pp_size}",
        [[4, 4]],
        logprobs_after_first_chunk=True,
    )
    _assert_runs_close(_first_update(reference_runs["nll"]), actual)
