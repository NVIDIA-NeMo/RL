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

"""Unit tests for the hybrid AR + diffusion Megatron post-processors.

Megatron is imported inside each test rather than at module scope, matching
``test_train.py``: these are ``mcore``-marked and are deselected (but still
collected) when the mcore extra is absent.
"""

import inspect
import os

import pytest
import torch

pytestmark = pytest.mark.mcore


def _policy_cfg(**overrides):
    cfg = {
        "sequence_packing": {"enabled": False},
        "logprob_estimation": {
            "type": "hybrid_ar_diffusion",
            "mask_token_id": 3,
            "ce_loss_weight": 0.1,
        },
    }
    cfg.update(overrides)
    return cfg


@pytest.fixture
def single_rank_gloo():
    """A 1-rank gloo process group, so the vocab-parallel all_reduce is a no-op."""
    import torch.distributed as dist

    already = dist.is_available() and dist.is_initialized()
    if not already:
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29517")
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    try:
        yield
    finally:
        if not already:
            dist.destroy_process_group()


@pytest.fixture
def tp1(monkeypatch):
    """Pin the tensor-parallel getters to a single, default-group rank."""
    from nemo_rl.models.megatron import diffu_grpo_train

    monkeypatch.setattr(
        diffu_grpo_train, "get_tensor_model_parallel_group", lambda: None
    )
    monkeypatch.setattr(diffu_grpo_train, "get_tensor_model_parallel_rank", lambda: 0)


class TestLogprobsPostProcessorInterface:
    def test_call_signature_matches_the_base_class(self):
        """The dispatcher passes ``original_seq_length`` by keyword.

        ``forward_with_post_processing_fn`` calls every LogprobsPostProcessor
        with ``original_seq_length=...``. An override that predates that
        parameter raises TypeError on the first logprob microbatch, which no
        construction-only test would catch.
        """
        from nemo_rl.models.megatron.hybrid_ar_diffusion_train import (
            HybridARDiffusionLogprobsPostProcessor,
        )
        from nemo_rl.models.megatron.train import LogprobsPostProcessor

        base = inspect.signature(LogprobsPostProcessor.__call__).parameters
        override = inspect.signature(
            HybridARDiffusionLogprobsPostProcessor.__call__
        ).parameters
        assert list(override) == list(base)

    def test_call_accepts_original_seq_length_as_a_keyword(self):
        from nemo_rl.models.megatron.hybrid_ar_diffusion_train import (
            HybridARDiffusionLogprobsPostProcessor,
        )

        processor = HybridARDiffusionLogprobsPostProcessor(cfg=_policy_cfg())
        closure = processor(
            data_dict={"hybrid_target_ids": torch.zeros(2, 4, dtype=torch.long)},
            input_ids=torch.zeros(2, 4, dtype=torch.long),
            cu_seqlens_padded=None,
            original_seq_length=4,
        )
        assert callable(closure)

    def test_fused_linear_logprobs_is_rejected(self):
        """The fused forward gathers at the next input token, not at the targets."""
        from nemo_rl.models.megatron.hybrid_ar_diffusion_train import (
            HybridARDiffusionLogprobsPostProcessor,
        )

        with pytest.raises(NotImplementedError, match="use_fused_linear_logprobs"):
            HybridARDiffusionLogprobsPostProcessor(
                cfg=_policy_cfg(), use_fused_linear_logprobs=True
            )


class TestLossPostProcessor:
    def test_rejects_sequence_packing_enabled(self):
        from nemo_rl.models.megatron.hybrid_ar_diffusion_train import (
            HybridARDiffusionLossPostProcessor,
        )

        processor = HybridARDiffusionLossPostProcessor(
            loss_fn=lambda *a, **k: (torch.tensor(0.0), {}),
            cfg=_policy_cfg(sequence_packing={"enabled": True}),
        )
        with pytest.raises(NotImplementedError, match="sequence_packing.enabled=false"):
            processor(data_dict={})

    def test_rejects_packed_seq_params(self):
        from nemo_rl.models.megatron.hybrid_ar_diffusion_train import (
            HybridARDiffusionLossPostProcessor,
        )

        processor = HybridARDiffusionLossPostProcessor(
            loss_fn=lambda *a, **k: (torch.tensor(0.0), {}),
            cfg=_policy_cfg(),
        )
        with pytest.raises(NotImplementedError, match="sequence_packing.enabled=false"):
            processor(data_dict={}, packed_seq_params=object())

    def test_closure_counteracts_mcore_microbatch_averaging(self, monkeypatch):
        """Megatron divides each microbatch loss by num_microbatches."""
        from nemo_rl.models.megatron import hybrid_ar_diffusion_train
        from nemo_rl.models.megatron.hybrid_ar_diffusion_train import (
            HybridARDiffusionLossPostProcessor,
        )

        monkeypatch.setattr(
            hybrid_ar_diffusion_train,
            "_cp_sharded_same_position_logprobs",
            lambda *a, **k: torch.zeros(1, 2),
        )
        processor = HybridARDiffusionLossPostProcessor(
            loss_fn=lambda *a, **k: (torch.tensor(2.0), {"pg_loss": 1.0}),
            cfg=_policy_cfg(),
            num_microbatches=4,
        )
        closure = processor(
            data_dict={"hybrid_target_ids": torch.zeros(1, 2, dtype=torch.long)}
        )
        loss, metrics = closure(torch.zeros(1, 2, 8))
        assert loss.item() == pytest.approx(8.0)
        assert metrics == {"pg_loss": 1.0}


class TestSamePositionLogprobs:
    def test_rejects_non_3d_logits(self):
        from nemo_rl.models.megatron.diffu_grpo_train import _same_position_logprobs

        with pytest.raises(ValueError, match=r"must be \[B, S, V\]"):
            _same_position_logprobs(
                torch.zeros(2, 4),
                torch.zeros(2, 4, dtype=torch.long),
                cfg={},
                sampling_params=None,
                inference_only=True,
                exclude_token_id=None,
            )

    def test_rejects_target_shape_mismatch(self):
        from nemo_rl.models.megatron.diffu_grpo_train import _same_position_logprobs

        with pytest.raises(ValueError, match="must match logits prefix"):
            _same_position_logprobs(
                torch.zeros(2, 4, 8),
                torch.zeros(2, 5, dtype=torch.long),
                cfg={},
                sampling_params=None,
                inference_only=True,
                exclude_token_id=None,
            )

    def test_rejects_chunked_logprobs_with_mask_exclusion(self, tp1):
        """ChunkedDistributedLogprob has no exclude_token_id support yet."""
        from nemo_rl.models.megatron.diffu_grpo_train import _same_position_logprobs

        with pytest.raises(NotImplementedError, match="logprob_chunk_size"):
            _same_position_logprobs(
                torch.zeros(2, 4, 8),
                torch.zeros(2, 4, dtype=torch.long),
                cfg={"logprob_chunk_size": 2},
                sampling_params=None,
                inference_only=True,
                exclude_token_id=3,
            )

    def test_mask_exclusion_is_exactly_a_renormalization(self, single_rank_gloo, tp1):
        """Dropping the MASK column renormalizes by ``-log1p(-p_mask)``.

        This pins the numerics of ``exclude_mask_token_from_logits``: it is not
        a no-op, and it is not an arbitrary reweighting -- the excluded path
        equals the plain path shifted by the log of the retained probability
        mass.
        """
        from nemo_rl.models.megatron.diffu_grpo_train import _same_position_logprobs

        torch.manual_seed(0)
        logits = torch.randn(4, 16, 128, dtype=torch.float32)
        targets = torch.randint(0, 128, (4, 16))
        mask_id = 3
        # Never score MASK itself: the identity below divides by 1 - p_mask,
        # which is exactly the mass the excluded gather keeps.
        targets[targets == mask_id] = mask_id + 1

        kwargs = dict(
            cfg={},
            sampling_params=None,
            inference_only=True,
        )
        plain = _same_position_logprobs(
            logits, targets, exclude_token_id=None, **kwargs
        )
        excluded = _same_position_logprobs(
            logits, targets, exclude_token_id=mask_id, **kwargs
        )

        p_mask = torch.softmax(logits, dim=-1)[..., mask_id]
        expected = plain - torch.log1p(-p_mask)
        torch.testing.assert_close(excluded, expected, atol=1e-5, rtol=1e-5)
        # ... and the knob is live: the two differ measurably somewhere.
        assert (excluded - plain).abs().max() > 1e-3


class TestContextParallelGuard:
    def test_cp_sharded_wrapper_rejects_cp_greater_than_one(self, monkeypatch):
        """Megatron's unpacked path does not CP-shard, so a re-gather is wrong."""
        from nemo_rl.models.megatron import diffu_grpo_train

        monkeypatch.setattr(
            diffu_grpo_train, "get_context_parallel_world_size", lambda: 2
        )
        with pytest.raises(NotImplementedError, match="context_parallel_size=1"):
            diffu_grpo_train._cp_sharded_same_position_logprobs(
                torch.zeros(1, 2, 4),
                torch.zeros(1, 2, dtype=torch.long),
                cfg={},
                sampling_params=None,
                inference_only=True,
                exclude_token_id=None,
            )
