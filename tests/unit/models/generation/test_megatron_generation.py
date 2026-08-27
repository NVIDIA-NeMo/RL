# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import gc
from copy import deepcopy
from types import SimpleNamespace

import pytest
import ray
import torch

from nemo_rl.algorithms.grpo import refit_policy_generation
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.models.generation.megatron import MegatronGeneration, megatron_generation
from nemo_rl.models.generation.megatron.config import (
    dedicated_inference_megatron_cfg,
)
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.weight_sync.megatron_weight_synchronizer import (
    MegatronWeightSynchronizer,
)
from tests.unit.test_utils import SimpleLossFn

model_name = "Qwen/Qwen3-0.6B"

basic_megatron_test_config: PolicyConfig = {
    "model_name": model_name,
    "tokenizer": {"name": model_name},
    "generation_batch_size": 2,
    "train_global_batch_size": 4,
    "train_micro_batch_size": 2,
    "learning_rate": 5e-6,
    "logprob_batch_size": 2,
    # The dynamic inference engine requires fp16/bf16
    "precision": "bfloat16",
    "offload_optimizer_for_logprob": False,
    "dtensor_cfg": {"enabled": False},
    "dynamic_batching": {"enabled": False},
    "sequence_packing": {"enabled": False},
    "make_sequence_length_divisible_by": 1,
    "megatron_cfg": {
        "enabled": True,
        "empty_unused_memory_level": 0,
        "activation_checkpointing": False,
        "converter_type": "Qwen2ForCausalLM",  # Qwen2 converter is compatible with Qwen3
        "tensor_model_parallel_size": 1,
        "expert_tensor_parallel_size": 1,
        "expert_model_parallel_size": 1,
        "pipeline_model_parallel_size": 1,
        "num_layers_in_first_pipeline_stage": None,
        "num_layers_in_last_pipeline_stage": None,
        "context_parallel_size": 1,
        "pipeline_dtype": "bfloat16",
        "sequence_parallel": False,
        "freeze_moe_router": True,
        "moe_router_dtype": "fp64",
        "moe_router_load_balancing_type": "none",
        "moe_router_bias_update_rate": 0.0,
        "moe_permute_fusion": False,
        "moe_enable_deepep": False,
        "moe_token_dispatcher_type": "alltoall",
        "moe_shared_expert_overlap": False,
        "apply_rope_fusion": True,
        "bias_activation_fusion": True,
        "moe_per_layer_logging": False,
        "gradient_accumulation_fusion": False,
        "use_fused_weighted_squared_relu": False,
        "train_iters": 100,
        "optimizer": {
            "optimizer": "adam",
            "lr": 5.0e-6,
            "min_lr": 5.0e-7,
            "weight_decay": 0.01,
            "bf16": True,
            "fp16": False,
            "params_dtype": "float32",
            "adam_beta1": 0.9,
            "adam_beta2": 0.999,
            "adam_eps": 1e-8,
            "use_distributed_optimizer": True,
            "use_precision_aware_optimizer": True,
            "clip_grad": 1.0,
            "optimizer_cpu_offload": False,
            "optimizer_offload_fraction": 0.0,
            "overlap_cpu_optimizer_d2h_h2d": False,
        },
        "scheduler": {
            "start_weight_decay": 0.01,
            "end_weight_decay": 0.01,
            "weight_decay_incr_style": "constant",
            "lr_decay_style": "constant",
            "lr_decay_iters": None,
            "lr_warmup_iters": 50,
            "lr_warmup_init": 5.0e-7,
        },
        "distributed_data_parallel_config": {
            "grad_reduce_in_fp32": False,
            "overlap_grad_reduce": True,
            "overlap_param_gather": False,
            "data_parallel_sharding_strategy": "optim_grads_params",
        },
    },
    "draft": {"enabled": False},
    "optimizer": None,
    "scheduler": None,
    "max_grad_norm": 1.0,
    "generation": {
        "backend": "megatron",
        "model_name": model_name,
        "max_new_tokens": 16,  # Small number of tokens for testing
        "temperature": 1.0,
        "top_p": 1.0,
        "top_k": None,
        "stop_token_ids": None,
        "stop_strings": None,
        "colocated": {
            "enabled": False,
            "resources": {"gpus_per_node": None, "num_nodes": None},
        },
        "mcore_generation_config": {
            "max_model_len": 1024,
            "cuda_graph_impl": "local",
            "inference_cuda_graph_scope": "block",
            "buffer_size_gb": 10,
            "num_cuda_graphs": 4,
            "block_size_tokens": 256,
            "use_cuda_graphs_for_non_decode_steps": True,
            "enable_chunked_prefill": True,
            "enable_prefix_caching": False,
            "max_tokens": 16384,
            "kv_cache_management_mode": "persist",
            "materialize_only_last_token_logits": True,
            "num_speculative_tokens": 0,
            "logprobs_mode": "processed_logprobs",
            "refit_backend": "gloo",  # not nvshmem: its NVLS multicast init is unavailable in CI
            "parsers": [],
            "expose_http_server": False,
        },
    },
}


@pytest.fixture(scope="function")
def cluster():
    """A 1-node, 2-GPU virtual cluster (enough for tp/pp up to 2)."""
    virtual_cluster = RayVirtualCluster(
        bundle_ct_per_node_list=[2],
        use_gpus=True,
        max_colocated_worker_groups=2,
        num_gpus_per_node=2,
        name="megatron-generation-test-cluster",
    )
    yield virtual_cluster
    virtual_cluster.shutdown()


@pytest.fixture(scope="function")
def policy_cluster_separate():
    """A dedicated 1-GPU cluster for the training policy in the non-colocated test."""
    virtual_cluster = RayVirtualCluster(
        bundle_ct_per_node_list=[1],
        use_gpus=True,
        max_colocated_worker_groups=1,
        num_gpus_per_node=1,
        name="megatron-generation-test-policy-cluster",
    )
    yield virtual_cluster
    try:
        virtual_cluster.shutdown()
    except Exception as e:
        print(f"Error during policy_cluster_separate shutdown: {e}")


@pytest.fixture(scope="function")
def tokenizer():
    """Initialize tokenizer for the test model."""
    return get_tokenizer({"name": model_name})


@pytest.fixture(scope="function")
def test_input_data(tokenizer):
    """Create test input data for inference."""
    test_prompts = [
        "Hello, my name is",
        "The capital of France is",
    ]
    encodings = tokenizer(
        test_prompts,
        padding="max_length",
        max_length=20,
        truncation=True,
        return_tensors="pt",
        padding_side="right",
    )
    input_lengths = encodings["attention_mask"].sum(dim=1).to(torch.int32)
    return BatchedDataDict(
        {
            "input_ids": encodings["input_ids"],
            "input_lengths": input_lengths,
        }
    )


def _assert_valid_generation_output(outputs, input_data, require_generation=True):
    """Assert the GenerationOutputSpec contract produced by the Megatron worker."""
    required_keys = [
        "output_ids",
        "logprobs",
        "generation_lengths",
        "unpadded_sequence_lengths",
    ]
    for key in required_keys:
        assert key in outputs, f"{key} not found in generation output"

    batch_size = len(input_data["input_ids"])
    assert all(outputs[key].shape[0] == batch_size for key in required_keys), (
        "Wrong batch size in generation output"
    )
    # output_ids and logprobs are packed on the same padded width.
    assert outputs["output_ids"].shape == outputs["logprobs"].shape

    if require_generation:
        assert (outputs["generation_lengths"] > 0).all(), (
            "Some samples generated nothing"
        )

    # length identity: total (unpadded) == prompt length + generated length.
    expected_unpadded = input_data["input_lengths"].cpu().to(torch.int64) + outputs[
        "generation_lengths"
    ].cpu().to(torch.int64)
    assert torch.equal(
        outputs["unpadded_sequence_lengths"].cpu().to(torch.int64), expected_unpadded
    ), "unpadded_sequence_lengths != input_lengths + generation_lengths"

    # logprob offset: position 0 is always the 0.0 placeholder.
    assert torch.allclose(
        outputs["logprobs"][:, 0],
        torch.zeros(batch_size, dtype=outputs["logprobs"].dtype),
    ), "logprobs[:, 0] should be the 0.0 placeholder"


async def _generate_async(mg, tokenizer, test_input_data, greedy=False):
    """Drive ``generate_async`` over single-sample microbatches and reassemble in order."""
    collected = []
    for single_item_input in test_input_data.make_microbatch_iterator(
        microbatch_size=1
    ):
        async for original_idx, single_item_output in mg.generate_async(
            single_item_input, greedy=greedy
        ):
            # The mcore coordinator only accepts requests on DP rank 0.
            assert single_item_output["gen_leader_worker_idx"] == [0]
            collected.append((original_idx, single_item_output))

    collected.sort(key=lambda x: x[0])
    outputs = [item for _, item in collected]
    pad_token_id = mg.cfg.get("_pad_token_id", tokenizer.pad_token_id)
    return BatchedDataDict.from_batches(
        outputs,
        pad_value_dict={"output_ids": pad_token_id, "logprobs": 0.0},
    )


@pytest.mark.mcore
@pytest.mark.timeout(900)
@pytest.mark.parametrize(
    "tensor_parallel_size,pipeline_parallel_size,top_p,top_k,logprobs_mode",
    [
        (1, 1, 1.0, None, "processed_logprobs"),
        (2, 1, 1.0, None, "processed_logprobs"),
        (1, 2, 1.0, None, "processed_logprobs"),
        (1, 1, 0.9, 8000, "processed_logprobs"),
        (1, 1, 1.0, 1, "processed_logprobs"),
        (1, 1, 1.0, 1, "raw_logprobs"),
    ],
)
def test_megatron_policy_generation(
    cluster,
    test_input_data,
    tokenizer,
    tensor_parallel_size,
    pipeline_parallel_size,
    top_p,
    top_k,
    logprobs_mode,
):
    """Standalone Megatron generation across tp/pp and sampling params."""
    if cluster.num_gpus_per_node < tensor_parallel_size * pipeline_parallel_size:
        pytest.skip(
            f"Need {tensor_parallel_size * pipeline_parallel_size} GPUs for "
            f"tp={tensor_parallel_size} pp={pipeline_parallel_size}"
        )

    config = deepcopy(basic_megatron_test_config)
    config["megatron_cfg"]["tensor_model_parallel_size"] = tensor_parallel_size
    config["megatron_cfg"]["pipeline_model_parallel_size"] = pipeline_parallel_size
    config["generation"]["top_p"] = top_p
    config["generation"]["top_k"] = top_k
    config["generation"]["mcore_generation_config"]["logprobs_mode"] = logprobs_mode
    # config-level stop string, unioned with the per-sample stop strings below.
    config["generation"]["stop_strings"] = ["</s>"]

    mg = None
    try:
        mg = MegatronGeneration(config=config, tokenizer=tokenizer, cluster=cluster)

        # greedy decoding: full output contract + non-empty text
        outputs = mg.generate(test_input_data, greedy=True)
        _assert_valid_generation_output(outputs, test_input_data)
        generated_texts = tokenizer.batch_decode(
            outputs["output_ids"], skip_special_tokens=True
        )
        assert all(len(t) > 0 for t in generated_texts), "Some greedy texts are empty"

        # sampling (non-greedy) path still produces a valid contract
        sampled = mg.generate(test_input_data, greedy=False)
        _assert_valid_generation_output(sampled, test_input_data)
        if top_k == 1:
            for i in range(len(test_input_data["input_ids"])):
                start = test_input_data["input_lengths"][i].item()
                end = start + sampled["generation_lengths"][i].item()
                gen_logprobs = sampled["logprobs"][i, start:end]
                assert (gen_logprobs <= 0).all()
                if logprobs_mode == "processed_logprobs":
                    # Processed logprobs are exactly 0.0 where the argmax is
                    # unique; bf16 max-ties renormalize to log(1/n).
                    assert (gen_logprobs == 0.0).float().mean() >= 0.5, (
                        "expected mostly-exact-0 processed logprobs under "
                        f"top_k=1, got {gen_logprobs}"
                    )
                else:
                    # Raw model probabilities are computed before top-k=1 and
                    # therefore retain nonzero uncertainty.
                    assert (gen_logprobs < 0.0).float().mean() >= 0.5, (
                        "expected mostly-negative raw logprobs under top_k=1, "
                        f"got {gen_logprobs}"
                    )

        # per-sample stop strings are merged with the config stop string (may stop early,
        # so don't require a generated token)
        data_with_stops = BatchedDataDict(
            {
                "input_ids": test_input_data["input_ids"],
                "input_lengths": test_input_data["input_lengths"],
                "stop_strings": [["."], ["."]],
            }
        )
        stopped = mg.generate(data_with_stops, greedy=True)
        _assert_valid_generation_output(
            stopped, test_input_data, require_generation=False
        )

        # lifecycle: leave generation mode, re-enter, and generate again
        assert mg.finish_generation() is True
        assert mg.prepare_for_generation() is True
        reentered = mg.generate(test_input_data, greedy=True)
        _assert_valid_generation_output(reentered, test_input_data)
    finally:
        if mg is not None:
            mg.shutdown()
        gc.collect()
        torch.cuda.empty_cache()


@pytest.mark.mcore
@pytest.mark.asyncio
@pytest.mark.timeout(900)
async def test_megatron_policy_generation_async(cluster, test_input_data, tokenizer):
    """Standalone Megatron async generation."""
    config = deepcopy(basic_megatron_test_config)
    mg = None
    try:
        mg = MegatronGeneration(config=config, tokenizer=tokenizer, cluster=cluster)
        outputs = await _generate_async(mg, tokenizer, test_input_data, greedy=True)
        _assert_valid_generation_output(outputs, test_input_data)
        generated_texts = tokenizer.batch_decode(
            outputs["output_ids"], skip_special_tokens=True
        )
        assert all(len(t) > 0 for t in generated_texts), "Some async texts are empty"
    finally:
        if mg is not None:
            mg.shutdown()
        gc.collect()
        torch.cuda.empty_cache()


@pytest.mark.mcore
@pytest.mark.timeout(900)
@pytest.mark.parametrize(
    "train_impl, gen_impl",
    [
        ("transformer_engine", "transformer_engine"),
        ("transformer_engine", "inference_optimized"),
        ("inference_optimized", "inference_optimized"),
    ],
)
def test_megatron_generation_colocated(
    cluster, test_input_data, tokenizer, train_impl, gen_impl
):
    """Colocated Megatron generation: wrap an existing training policy without owning it."""
    config = deepcopy(basic_megatron_test_config)
    config["generation"]["colocated"]["enabled"] = True
    # Eager engine startup (expose_http_server) flips MLM's process-wide
    # InferenceMode on at construction; the legs that train before any
    # generate/suspend cycle must construct engine-less.
    expose_http_server = (
        train_impl == "transformer_engine" and gen_impl == "transformer_engine"
    )
    config["generation"]["mcore_generation_config"]["expose_http_server"] = (
        expose_http_server
    )
    # Matched impls => reshardless colocated (shared model; the
    # inference_optimized pair trains through the TE parent path); differing
    # impls => the worker builds a dedicated resharded inference model on
    # the shared GPUs.
    config["megatron_cfg"]["transformer_impl"] = train_impl
    config["generation"]["mcore_generation_config"]["transformer_impl"] = gen_impl
    if train_impl == "inference_optimized":
        # The parity block's sleep/wake cycle would tear down and recapture
        # CUDA graphs mid-test; keep them off here.
        config["generation"]["mcore_generation_config"]["cuda_graph_impl"] = "none"
        # The parity block's 2-sample batch shards to 1 sample per DP rank
        # (DP=2 on the 2-GPU cluster); the logprob microbatch must divide it.
        config["logprob_batch_size"] = 1

    # construction guard: exactly one of `cluster` / `policy` is required
    with pytest.raises(AssertionError):
        MegatronGeneration(config=config, tokenizer=tokenizer)
    with pytest.raises(AssertionError):
        MegatronGeneration(
            config=config, tokenizer=tokenizer, cluster=cluster, policy=object()
        )

    policy = None
    try:
        policy = Policy(cluster=cluster, config=config, tokenizer=tokenizer)
        megatron_cfg_before = deepcopy(config["megatron_cfg"])

        mg = MegatronGeneration(policy=policy, config=config, tokenizer=tokenizer)
        # colocated wrapper reuses the training policy and must not own it
        assert mg._owns_policy is False
        # the colocated path must NOT merge mcore_generation_config into megatron_cfg
        assert "max_tokens" not in config["megatron_cfg"]
        assert config["megatron_cfg"] == megatron_cfg_before

        # The selector: matched impls => reshardless (no dedicated model).
        assert (dedicated_inference_megatron_cfg(config) is None) == (
            train_impl == gen_impl
        )

        if expose_http_server:
            # setup() hands dp_openai_server_base_urls to NeMo Gym right after
            # construction, so the colocated constructor must have collected them.
            assert mg.dp_openai_server_base_urls, "no OpenAI server URLs collected"
            assert all(url.startswith("http") for url in mg.dp_openai_server_base_urls)

        if gen_impl == "inference_optimized":
            # Both inference_optimized legs take a train step (finite loss)
            # before the engine ever starts; the reshard leg then generates
            # on the dedicated model built at first wake, the matched-impl
            # leg directly on the shared training model.
            torch.manual_seed(42)
            train_data = BatchedDataDict(
                {
                    "input_ids": torch.randint(0, 32000, (4, 64)),
                    "input_lengths": torch.full((4,), 64, dtype=torch.int32),
                    "attention_mask": torch.ones(4, 64),
                    "labels": torch.randint(0, 32000, (4, 64)),
                    "sample_mask": torch.ones(4),
                }
            )
            policy.prepare_for_training()
            loss = policy.train(train_data, SimpleLossFn())["loss"]
            assert not torch.isnan(loss).any() and not torch.isinf(loss).any(), (
                f"pre-generation train step produced bad loss: {loss}"
            )
            policy.finish_training()

        # re-entering generation mode must be a no-op on the running engine
        mg.prepare_for_generation()
        outputs = mg.generate(test_input_data, greedy=True)
        _assert_valid_generation_output(outputs, test_input_data)

        # CUDA-graph capture proof: count the number of actually-created graphs.
        # The inference_optimized train leg pins cuda_graph_impl="none" above,
        # so it must stay at zero captures; the other legs must really capture.
        graphs_enabled = (
            config["generation"]["mcore_generation_config"]["cuda_graph_impl"] != "none"
        )
        capture_counts = ray.get(
            mg._policy.worker_group.run_all_workers_single_data(
                "get_inference_cuda_graph_capture_count"
            )
        )
        if graphs_enabled:
            assert all(count > 0 for count in capture_counts), capture_counts
        else:
            assert all(count == 0 for count in capture_counts), capture_counts

        if train_impl == "inference_optimized":
            # 3490-review follow-up: bound token mult-prob error on the
            # matched-impl leg — generation and recomputed policy logprobs
            # run the same inference kernels on the same shared weights.
            # Greedy must be off: processed logprobs are ~0 under top_k=1.
            sampled = mg.generate(test_input_data, greedy=False)
            fprop_data = BatchedDataDict(
                {
                    "input_ids": sampled["output_ids"],
                    "input_lengths": sampled["unpadded_sequence_lengths"],
                }
            )
            # Production ordering: the engine stands down before any training-path forward.
            mg.finish_generation(release_gpu=True)
            policy.prepare_for_lp_inference()
            lp_logprobs = policy.get_logprobs(fprop_data)["logprobs"]
            gen_mask = torch.zeros_like(sampled["logprobs"], dtype=torch.bool)
            for i, (start, end) in enumerate(
                zip(
                    test_input_data["input_lengths"],
                    sampled["unpadded_sequence_lengths"],
                )
            ):
                gen_mask[i, start:end] = True
            abs_diff = (sampled["logprobs"] - lp_logprobs).abs().masked_select(gen_mask)
            avg_prob_mult_error = torch.exp(abs_diff).mean()
            assert avg_prob_mult_error <= 1.05, (
                f"matched-impl inference_optimized: generation logprobs "
                f"diverge from policy logprobs (avg prob mult error "
                f"{avg_prob_mult_error:.4f})"
            )
            # Wake the engine again for the post-shutdown generation check.
            mg.prepare_for_generation()

        # ownership guard: shutdown is a no-op, so the wrapped policy keeps generating
        assert mg.shutdown() is True
        after_shutdown = mg.generate(test_input_data, greedy=True)
        _assert_valid_generation_output(after_shutdown, test_input_data)

        # Capture-once: a full sleep/wake cycle re-attaches the same managers and replays
        # the same graphs; a growing count means managers were rebuilt and the old graphs leaked.
        mg.finish_generation()
        mg.prepare_for_generation()
        after_cycle = mg.generate(test_input_data, greedy=True)
        _assert_valid_generation_output(after_cycle, test_input_data)
        recapture_counts = ray.get(
            mg._policy.worker_group.run_all_workers_single_data(
                "get_inference_cuda_graph_capture_count"
            )
        )
        assert recapture_counts == capture_counts, (capture_counts, recapture_counts)
    finally:
        if policy is not None:
            policy.shutdown()
        gc.collect()
        torch.cuda.empty_cache()


@pytest.mark.mcore
@pytest.mark.timeout(900)
@pytest.mark.parametrize("skip_weight_load", [False, True])
def test_megatron_generation_non_colocated_refit(
    policy_cluster_separate, test_input_data, tokenizer, skip_weight_load
):
    """Non-colocated Megatron generation.

    With skip_weight_load, inference-engine initialization is deferred until
    refit delivers the final weight objects. This is required for CUDA graphs
    to capture persistent refit-buffer addresses.
    """
    generation_cluster = RayVirtualCluster(
        bundle_ct_per_node_list=[1],
        use_gpus=True,
        max_colocated_worker_groups=1,
        num_gpus_per_node=1,
        name="megatron-generation-test-generation-cluster",
    )
    if (
        policy_cluster_separate.num_gpus_per_node < 1
        or generation_cluster.num_gpus_per_node < 1
    ):
        pytest.skip("Need at least two GPUs across separate clusters")

    config = deepcopy(basic_megatron_test_config)

    policy = None
    mg = None
    try:
        policy = Policy(
            cluster=policy_cluster_separate, config=config, tokenizer=tokenizer
        )

        # construction guard: skip_weight_load requires a dedicated inference
        # policy; wrapping an existing (colocated) policy must be rejected.
        with pytest.raises(AssertionError):
            MegatronGeneration(
                config=config,
                tokenizer=tokenizer,
                policy=policy,
                skip_weight_load=True,
            )

        mg = MegatronGeneration(
            config=config,
            tokenizer=tokenizer,
            cluster=generation_cluster,
            skip_weight_load=skip_weight_load,
        )

        # Wire the refit collective the way grpo.setup does: through the
        # weight synchronizer, which refit_policy_generation dispatches to.
        mg.weight_synchronizer = MegatronWeightSynchronizer(
            policy,
            mg,
            colocated=False,
            train_cluster=policy_cluster_separate,
            inference_cluster=generation_cluster,
        )
        mg.weight_synchronizer.init_communicator()

        # refit the inference engine from the training weights, then generate
        refit_policy_generation(policy, mg, False)
        # Greedy needs to be false because processed logprobs doesn't handle it well.
        outputs = mg.generate(test_input_data, greedy=False)
        _assert_valid_generation_output(outputs, test_input_data)
        generated_texts = tokenizer.batch_decode(
            outputs["output_ids"], skip_special_tokens=True
        )
        assert all(len(t) > 0 for t in generated_texts), "Some texts are empty"

        # Training-policy logprobs must match generation-policy logprobs.
        # A broken refit would fail this test.
        fprop_data = BatchedDataDict(
            {
                "input_ids": outputs["output_ids"],
                "input_lengths": outputs["unpadded_sequence_lengths"],
            }
        )
        policy.prepare_for_lp_inference()
        train_logprobs = policy.get_logprobs(fprop_data)["logprobs"]
        gen_mask = torch.zeros_like(outputs["logprobs"], dtype=torch.bool)
        for i, (start, end) in enumerate(
            zip(test_input_data["input_lengths"], outputs["unpadded_sequence_lengths"])
        ):
            gen_mask[i, start:end] = True
        abs_diff = (outputs["logprobs"] - train_logprobs).abs().masked_select(gen_mask)
        avg_prob_mult_error = torch.exp(abs_diff).mean()
        assert avg_prob_mult_error <= 1.05, (
            f"generation logprobs diverge from training-policy logprobs "
            f"(avg prob mult error {avg_prob_mult_error:.4f}); inference weights "
            f"do not match training weights after refit"
        )
    finally:
        if mg is not None:
            mg.shutdown()
        if policy is not None:
            policy.shutdown()
        try:
            generation_cluster.shutdown()
        except Exception as e:
            print(f"Error during generation_cluster shutdown: {e}")
        gc.collect()
        torch.cuda.empty_cache()


class _CapturingPortHolder:
    """Stand-in for the RemoteHeldPortReservation actor.

    Records the scheduling strategy it is pinned to and returns a fixed
    (ip, port) instead of binding a real socket on a real placement group.
    """

    last_scheduling_strategy = None

    @classmethod
    def options(cls, *, scheduling_strategy):
        cls.last_scheduling_strategy = scheduling_strategy
        return cls

    @classmethod
    def remote(cls):
        return SimpleNamespace(
            address=SimpleNamespace(remote=lambda: ("10.0.0.5", 4321))
        )


def _rank0_bundle_via_worker_group(sorted_bundle_indices, group_size):
    """The bundle RANK 0 actually lands on, reconstructed from the live code.

    Mirrors lm_policy.py's tied_groups[0] for a unified PG and RayWorkerGroup's
    default first-worker tuple otherwise -- the two branches
    reserve_http_server_address must agree with.

    Deliberately a hand-copy rather than a call into the code under test (or a
    shared helper): sharing the implementation would make the assertion a
    tautology. The cost is that this copy is frozen -- if the worker-group
    placement rule ever changes, update it by hand; grpo.py's runtime
    served-vs-reserved URL check is the net for that direction.
    """
    if sorted_bundle_indices is not None:
        # lm_policy.py: tied_groups = [(i // group_size, [b]) for i, b in ...]
        tied_groups = [
            (i // group_size, [bundle_idx])
            for i, bundle_idx in enumerate(sorted_bundle_indices)
        ]
    else:
        # RayWorkerGroup.__init__: bundle_indices_list.append((i, [bundle_idx]))
        # with i and bundle_idx both starting at 0.
        tied_groups = [(0, [0])]
    pg_idx, local_bundle_indices = tied_groups[0]
    return pg_idx, local_bundle_indices[0]


@pytest.fixture
def patched_holder(monkeypatch):
    monkeypatch.setattr(
        megatron_generation, "RemoteHeldPortReservation", _CapturingPortHolder
    )
    # ray.get here only unwraps the holder's (ip, port); no real Ray involved.
    monkeypatch.setattr(megatron_generation.ray, "get", lambda ref: ref)
    _CapturingPortHolder.last_scheduling_strategy = None
    return _CapturingPortHolder


@pytest.mark.parametrize(
    "sorted_bundle_indices",
    [
        # Unified cross-node PG: topology sort can make rank 0 a bundle other
        # than 0, so a naive "bundle 0" prediction would bind the wrong node.
        [3, 1, 0, 2],
        # Per-node PGs: no sorted indices, rank 0 is bundle 0 of the first PG.
        None,
    ],
)
def test_reserve_http_server_address_pins_rank0_bundle(
    patched_holder, sorted_bundle_indices
):
    """reserve_http_server_address's rank-0 prediction must match real placement.

    reserve_http_server_address publishes the OpenAI server URL to NeMo Gym
    *before* any worker exists, pinning a port holder to the (placement_group,
    bundle) it predicts rank 0 will occupy. That prediction is a second,
    hand-written copy of the rank-0 placement lm_policy.py / RayWorkerGroup
    actually perform: if the two ever disagree the holder binds the wrong node,
    the pre-published URL is unreachable, and grpo.py fails loud at runtime.
    Pins the prediction to a hand-reconstruction of that placement so the two
    copies cannot silently drift -- no GPU or mcore extra needed
    (MegatronGeneration imports without megatron.core).
    """
    placement_groups = ["PG0", "PG1"]
    cluster = SimpleNamespace(
        num_gpus_per_node=8,
        _sorted_bundle_indices=sorted_bundle_indices,
        get_placement_groups=lambda: placement_groups,
    )
    config = {"generation": {"colocated": {"enabled": True}}}

    url, port, holder = MegatronGeneration.reserve_http_server_address(cluster, config)

    expected_pg_idx, expected_bundle_index = _rank0_bundle_via_worker_group(
        sorted_bundle_indices, cluster.num_gpus_per_node
    )
    strategy = patched_holder.last_scheduling_strategy
    # The holder -- and thus the pre-published URL's node -- must sit on the
    # exact (placement_group, bundle) rank 0 will occupy.
    assert expected_pg_idx == 0  # rank 0 is always in the first placement group
    assert strategy.placement_group is placement_groups[expected_pg_idx]
    assert strategy.placement_group_bundle_index == expected_bundle_index

    assert url == "http://10.0.0.5:4321/v1"
    assert port == 4321
    assert holder.address.remote() == ("10.0.0.5", 4321)
