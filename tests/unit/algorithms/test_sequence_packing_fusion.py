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
"""
Unit tests to ensure SequencePackingFusionLossWrapper works as SequencePackingLossWrapper.
  - Without explicitly calling the loss_fn sequence by sequence.

During the forward pass, compare the loss and metrics from the two wrappers.
During the backward pass, compare the gradients from the two wrappers.

For parallelism, check for CP and TP.

For loss function, right now only supports:
- ClippedPGLossFn
"""

import os

import pytest
import ray
import torch

from nemo_rl.algorithms.loss_functions import (
    ClippedPGLossFn, 
    SequencePackingFusionLossWrapper,
    SequencePackingLossWrapper, 
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.named_sharding import NamedSharding
from nemo_rl.distributed.ray_actor_environment_registry import (
    ACTOR_ENVIRONMENT_REGISTRY,
    PY_EXECUTABLES,
)
from nemo_rl.distributed.virtual_cluster import RayVirtualCluster
from nemo_rl.distributed.worker_groups import RayWorkerBuilder, RayWorkerGroup


@ray.remote(num_gpus=1)
class SequencePackingLossWrapperBaselineActor:
    def __init__(self, cp_size: int, tp_size: int):
        self.cp_size = cp_size
        self.tp_size = tp_size
        self.env_vars = dict(os.environ)

    def _setup_process_groups(self):
        torch.distributed.init_process_group(backend="nccl")

        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        assert world_size == self.cp_size * self.tp_size, (
            f"Expected WORLD_SIZE={self.cp_size * self.tp_size}, got {world_size}."
        )

        # ---------------------------------------------------------------------
        # Create 2D (cp, tp) process groups.
        # Rank layout (outer cp, inner tp):
        #   [[0, 1, ..., tp_size-1],
        #    [tp_size, ..., 2*tp_size-1],
        #    ...]
        # ---------------------------------------------------------------------
        cp_groups: list[torch.distributed.ProcessGroup] = []
        tp_groups: list[torch.distributed.ProcessGroup] = []

        # CP groups: one per tp_rank, varying cp coordinate
        for tp_rank in range(self.tp_size):
            ranks = [
                cp_rank * self.tp_size + tp_rank for cp_rank in range(self.cp_size)
            ]
            cp_groups.append(torch.distributed.new_group(ranks=ranks))

        # TP groups: one per cp_rank, varying tp coordinate
        for cp_rank in range(self.cp_size):
            ranks = [
                cp_rank * self.tp_size + tp_rank for tp_rank in range(self.tp_size)
            ]
            tp_groups.append(torch.distributed.new_group(ranks=ranks))

        my_tp_rank = rank % self.tp_size
        my_cp_rank = rank // self.tp_size
        cp_group = cp_groups[my_tp_rank]
        tp_group = tp_groups[my_cp_rank]
        return rank, my_cp_rank, my_tp_rank, cp_group, tp_group

    def _build_test_case(self, cp_group, my_tp_rank: int):
        from nemo_rl.distributed.model_utils import _get_tokens_on_this_cp_rank
        from nemo_rl.models.megatron.data import _pack_sequences_for_megatron

        # ---------------------------------------------------------------------
        # Build a small packed batch.
        # ---------------------------------------------------------------------
        device = torch.device("cuda")
        torch.manual_seed(42)  # For reproducibility / determinism

        batch_size = 4
        max_seq_len = 512
        # Ensure CP load balancing requirement: divisible by (2 * cp_size)
        if max_seq_len % (2 * self.cp_size) != 0:
            max_seq_len = (max_seq_len // (2 * self.cp_size) + 1) * (2 * self.cp_size)

        vocab_size_total = 512
        assert vocab_size_total % self.tp_size == 0
        vocab_size_local = vocab_size_total // self.tp_size

        # Variable lengths, but <= max_seq_len
        seq_lengths = torch.tensor(
            [
                max_seq_len // 4,
                max_seq_len // 2,
                max_seq_len // 3,
                max_seq_len * 3 // 4,
            ],
            dtype=torch.int32,
            device=device,
        )

        # Input ids + masks
        input_ids = torch.zeros(
            batch_size, max_seq_len, dtype=torch.long, device=device,
        )
        token_mask = torch.zeros(
            batch_size, max_seq_len, dtype=torch.float32, device=device,
        )
        for i in range(batch_size):
            L = int(seq_lengths[i].item())
            input_ids[i, :L] = torch.randint(0, vocab_size_total, (L,), device=device)
            token_mask[i, :L] = 1.0

        sample_mask = torch.ones(batch_size, dtype=torch.float32, device=device)

        # Stable-ish random tensors (avoid extreme ratios/NaNs in unit test)
        advantages = 0.1 * torch.randn(batch_size, max_seq_len, device=device)
        prev_logprobs = 0.1 * torch.randn(batch_size, max_seq_len, device=device)
        generation_logprobs = 0.1 * torch.randn(batch_size, max_seq_len, device=device)
        reference_policy_logprobs = generation_logprobs.clone()

        data_dict = BatchedDataDict(
            {
                "input_ids": input_ids,
                "input_lengths": seq_lengths,
                "token_mask": token_mask,
                "sample_mask": sample_mask,
                "advantages": advantages,
                "prev_logprobs": prev_logprobs,
                "generation_logprobs": generation_logprobs,
                "reference_policy_logprobs": reference_policy_logprobs,
            }
        )

        # Packed sequence metadata (CP-aware)
        pad_to_multiple = self.cp_size * 2
        (
            _packed_input_ids,
            _packed_input_ids_cp,
            _packed_seq_params,
            cu_seqlens,
            cu_seqlens_padded,
        ) = _pack_sequences_for_megatron(
            input_ids,
            seq_lengths,
            pad_individual_seqs_to_multiple_of=pad_to_multiple,
            pad_packed_seq_to=max_seq_len * batch_size if self.cp_size > 1 else None,
            cp_rank=torch.distributed.get_rank(cp_group),
            cp_size=self.cp_size,
        )
        assert cu_seqlens_padded is not None

        # ---------------------------------------------------------------------
        # Create vocab-parallel logits, then pack + CP-shard them into [1, T//CP, V//TP]
        # ---------------------------------------------------------------------
        # Global logits (same across ranks), then slice by TP rank
        full_logits = torch.randn(
            batch_size, 
            max_seq_len, 
            vocab_size_total, 
            device=device, 
            dtype=torch.float32,
        )

        def make_logits_and_packed_logits():
            logits_local = (
                full_logits[
                    :,
                    :,
                    my_tp_rank * vocab_size_local : (my_tp_rank + 1) * vocab_size_local,
                ]
                .clone()
                .detach()
                .requires_grad_(True)
            )

            total_padded_tokens = int(cu_seqlens_padded[-1].item())
            packed_logits = torch.zeros(
                1, total_padded_tokens // self.cp_size, vocab_size_local, device=device
            )

            run_seq = 0
            for i in range(batch_size):
                seq_len = int(seq_lengths[i].item())
                padded_seq_len = int(
                    (cu_seqlens_padded[i + 1] - cu_seqlens_padded[i]).item()
                )
                tmp = torch.zeros(1, padded_seq_len, vocab_size_local, device=device)
                tmp[:, :seq_len, :] = logits_local[i : i + 1, :seq_len, :]
                packed_logits[
                    :,
                    run_seq // self.cp_size : (run_seq + padded_seq_len)
                    // self.cp_size,
                    :,
                ] = _get_tokens_on_this_cp_rank(
                    tmp, torch.distributed.get_rank(cp_group), self.cp_size
                )
                run_seq += padded_seq_len

            return logits_local, packed_logits

        # ---------------------------------------------------------------------
        # Loss: SequencePackingLossWrapper + ClippedPGLossFn
        # ---------------------------------------------------------------------
        loss_cfg = {
            # From examples/configs/grpo_math_1B_megatron.yaml (loss_fn section)
            "reference_policy_kl_penalty": 0.01,
            "ratio_clip_min": 0.2,
            "ratio_clip_max": 0.2,
            "use_on_policy_kl_approximation": False,
            "use_importance_sampling_correction": False,
            "token_level_loss": True,
            "ratio_clip_c": None,
            # Required by ClippedPGLossConfig but not in that YAML
            "reference_policy_kl_type": "k3",
            "kl_input_clamp_value": 20.0,
            "kl_output_clamp_value": 10.0,
            "truncated_importance_sampling_ratio": None,
            "sequence_level_importance_ratios": False,
            "force_on_policy_ratio": False,
        }

        # Global normalization factors (token-level loss uses global_valid_toks)
        valid_toks = int(torch.clamp(seq_lengths - 1, min=0).sum().item())
        global_valid_toks = torch.tensor(valid_toks, dtype=torch.float32, device=device)
        global_valid_seqs = torch.tensor(batch_size, dtype=torch.float32, device=device)

        return {
            "loss_cfg": loss_cfg,
            "cu_seqlens": cu_seqlens,
            "cu_seqlens_padded": cu_seqlens_padded,
            "packed_input_ids": _packed_input_ids,  # [1, T_packed] from _pack_sequences_for_megatron
            "data_dict": data_dict,
            "global_valid_seqs": global_valid_seqs,
            "global_valid_toks": global_valid_toks,
            "make_logits_and_packed_logits": make_logits_and_packed_logits,
        }

    def run_compare_sequence_packing_wrappers(
        self, use_cached_packed_input_ids: bool = False,
    ):
        """
        Compare helper (for when your candidate/fused wrapper exists):
        - Builds inputs ONCE
        - Runs baseline wrapper and candidate wrapper on identical inputs
        - Returns loss/metrics + max grad for each

        Args:
            use_cached_packed_input_ids: If True, store pre-packed input_ids in data dict
                so the fused wrapper skips _pack_input_ids. If False, the fused wrapper
                packs on the fly (fallback path).
        """
        rank, _my_cp_rank, my_tp_rank, cp_group, tp_group = self._setup_process_groups()
        tc = self._build_test_case(cp_group=cp_group, my_tp_rank=my_tp_rank)
        base_loss_fn = ClippedPGLossFn(tc["loss_cfg"])

        data_dict = tc["data_dict"]

        # Instantiate wrappers
        baseline_wrapper = SequencePackingLossWrapper(
            loss_fn=base_loss_fn,
            cu_seqlens_q=tc["cu_seqlens"],
            cu_seqlens_q_padded=tc["cu_seqlens_padded"],
        )
        
        candidate_wrapper = SequencePackingFusionLossWrapper(
            loss_fn=base_loss_fn,
            cu_seqlens_q=tc["cu_seqlens"],
            cu_seqlens_q_padded=tc["cu_seqlens_padded"],
        )

        # Baseline run (fresh logits) — uses the original data_dict without packed_input_ids
        baseline_logits, baseline_packed_logits = tc["make_logits_and_packed_logits"]()
        baseline_loss, baseline_metrics = baseline_wrapper(
            baseline_packed_logits,
            data_dict,
            tc["global_valid_seqs"],
            tc["global_valid_toks"],
            vocab_parallel_rank=my_tp_rank,
            vocab_parallel_group=tp_group,
            context_parallel_group=cp_group,
        )
        (baseline_loss / self.cp_size).backward()
        baseline_grad = baseline_logits.grad.clone()

        # Candidate run (fresh logits, identical values)
        # Optionally add pre-packed input_ids to data dict for the fused wrapper only
        candidate_data_dict = data_dict
        if use_cached_packed_input_ids:
            candidate_data_dict = BatchedDataDict(dict(data_dict))
            candidate_data_dict["packed_input_ids"] = tc["packed_input_ids"]

        candidate_logits, candidate_packed_logits = tc[
            "make_logits_and_packed_logits"
        ]()
        candidate_loss, candidate_metrics = candidate_wrapper(
            candidate_packed_logits,
            candidate_data_dict,
            tc["global_valid_seqs"],
            tc["global_valid_toks"],
            vocab_parallel_rank=my_tp_rank,
            vocab_parallel_group=tp_group,
            context_parallel_group=cp_group,
        )
        (candidate_loss / self.cp_size).backward()
        candidate_grad = candidate_logits.grad.clone()

        return {
            "rank": rank,
            "cp_rank": int(torch.distributed.get_rank(cp_group)),
            "tp_rank": int(torch.distributed.get_rank(tp_group)),
            "baseline": {
                "loss": baseline_loss,
                "metrics_keys": sorted(list(baseline_metrics.keys())),
                "logits_local_grad": baseline_grad,
            },
            "candidate": {
                "loss": candidate_loss,
                "metrics_keys": sorted(list(candidate_metrics.keys())),
                "logits_local_grad": candidate_grad,
            },
        }


SEQUENCE_PACKING_LOSS_WRAPPER_BASELINE_ACTOR_FQN = f"{SequencePackingLossWrapperBaselineActor.__module__}.SequencePackingLossWrapperBaselineActor"

@pytest.fixture
def register_sequence_packing_loss_wrapper_baseline_actor():
    """Register the actor in ACTOR_ENVIRONMENT_REGISTRY for RayWorkerGroup."""
    original_registry_value = ACTOR_ENVIRONMENT_REGISTRY.get(
        SEQUENCE_PACKING_LOSS_WRAPPER_BASELINE_ACTOR_FQN
    )
    ACTOR_ENVIRONMENT_REGISTRY[SEQUENCE_PACKING_LOSS_WRAPPER_BASELINE_ACTOR_FQN] = (
        PY_EXECUTABLES.MCORE
    )
    yield SEQUENCE_PACKING_LOSS_WRAPPER_BASELINE_ACTOR_FQN

    if SEQUENCE_PACKING_LOSS_WRAPPER_BASELINE_ACTOR_FQN in ACTOR_ENVIRONMENT_REGISTRY:
        if original_registry_value is None:
            del ACTOR_ENVIRONMENT_REGISTRY[
                SEQUENCE_PACKING_LOSS_WRAPPER_BASELINE_ACTOR_FQN
            ]
        else:
            ACTOR_ENVIRONMENT_REGISTRY[
                SEQUENCE_PACKING_LOSS_WRAPPER_BASELINE_ACTOR_FQN
            ] = original_registry_value

@pytest.fixture(scope="function")
def cluster_fixture(request):
    """Create and teardown a virtual cluster for CP/TP tests."""
    cp_size, tp_size = request.node.callspec.params["cp_tp"]
    world_size = cp_size * tp_size

    if not ray.is_initialized():
        from nemo_rl.distributed.virtual_cluster import init_ray
        init_ray()

    # Check available GPUs via Ray cluster resources (works across multi-node),
    # falling back to local torch.cuda.device_count() if Ray has no GPU info.
    available_gpus = int(ray.cluster_resources().get("GPU", 0))
    if available_gpus == 0:
        available_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0

    if available_gpus < world_size:
        pytest.skip(
            f"Not enough GPUs available. Need {world_size}, got {available_gpus}"
        )

    cluster_name = f"test-seq-pack-fusion-cp{cp_size}-tp{tp_size}"
    cluster = RayVirtualCluster(
        name=cluster_name, bundle_ct_per_node_list=[world_size], use_gpus=True
    )
    yield cluster
    cluster.shutdown()


@pytest.mark.parametrize(
    "cp_tp",
    [
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2),
        (2, 4),
        (4, 2),
    ],
    ids=lambda cp_tp: f"cp{cp_tp[0]}_tp{cp_tp[1]}",
)
@pytest.mark.parametrize(
    "use_cached_packed_input_ids",
    [False, True],
    ids=["pack_on_the_fly", "cached_packed_input_ids"],
)
def test_sequence_packing_loss_wrapper_baseline_cp_tp(
    cluster_fixture, 
    register_sequence_packing_loss_wrapper_baseline_actor,
    cp_tp, 
    use_cached_packed_input_ids,
):
    """Compare SequencePackingFusionLossWrapper vs SequencePackingLossWrapper.

    Verifies that the fused wrapper produces identical:
      - loss values
      - backward gradients w.r.t. vocab-parallel logits
    for different CP and TP configurations, and both with and without
    pre-packed input_ids cached in the data dict.
    """
    cp_size, tp_size = cp_tp
    cluster = cluster_fixture
    actor_fqn = register_sequence_packing_loss_wrapper_baseline_actor
    world_size = cp_size * tp_size

    sharding_layout = [
        [cp_rank * tp_size + tp_rank for tp_rank in range(tp_size)]
        for cp_rank in range(cp_size)
    ]
    sharding = NamedSharding(layout=sharding_layout, names=["cp", "tp"])
    builder = RayWorkerBuilder(actor_fqn, cp_size=cp_size, tp_size=tp_size)

    worker_group = RayWorkerGroup(
        cluster=cluster,
        remote_worker_builder=builder,
        workers_per_node=None,
        sharding_annotations=sharding,
    )

    futures = worker_group.run_all_workers_single_data(
        "run_compare_sequence_packing_wrappers",
        use_cached_packed_input_ids=use_cached_packed_input_ids,
    )
    results = ray.get(futures)

    if not isinstance(results, list):
        results = [results]

    for r in results:
        rank = r["rank"]
        # Forward: loss values must match
        torch.testing.assert_close(
            r["baseline"]["loss"],
            r["candidate"]["loss"],
            atol=1e-5,
            rtol=1e-5,
            msg=f"Loss mismatch on rank {rank}",
        )
        # Backward: gradients w.r.t. logits must match
        torch.testing.assert_close(
            r["baseline"]["logits_local_grad"],
            r["candidate"]["logits_local_grad"],
            atol=1e-5,
            rtol=1e-5,
            msg=f"Gradient mismatch on rank {rank}",
        )

    worker_group.shutdown(force=True)

