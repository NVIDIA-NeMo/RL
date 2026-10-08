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

"""CPU regressions for padding-independent expert-bias updates."""

from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.mcore


def _processed(monkeypatch, multiple, enabled=True, lengths=True):
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from nemo_rl.models.megatron import data as module

    monkeypatch.setattr(module, "get_context_parallel_rank", lambda: 0)
    monkeypatch.setattr(module, "get_context_parallel_world_size", lambda: 1)
    batch = {
        "input_ids": torch.tensor([[1, 2, 3, 4]]),
        "token_mask": torch.tensor([[0, 0, 1, 1]]),
    }
    if lengths:
        batch["input_lengths"] = torch.tensor([4])
    return module.process_microbatch(
        BatchedDataDict(batch),
        pack_sequences=False,
        pad_individual_seqs_to_multiple_of=multiple,
        create_nonpacked_router_padding_mask=enabled,
    )


def test_nonpacked_padding_does_not_change_expert_bias(monkeypatch):
    from megatron.core.transformer.moe.moe_utils import get_updated_expert_bias
    from megatron.core.transformer.moe.router import TopKRouter

    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *args, **kwargs: None)
    results = []
    for multiple in (4, 8):
        processed = _processed(monkeypatch, multiple)
        router = TopKRouter.__new__(TopKRouter)
        torch.nn.Module.__init__(router)
        router.enable_expert_bias = True
        router.local_tokens_per_expert = torch.zeros(4, dtype=torch.long)
        # Four real tokens balance the experts; padding alone favors expert0.
        routes = torch.zeros((multiple, 4), dtype=torch.bool)
        routes[:4] = torch.eye(4, dtype=torch.bool)
        routes[4:, 0] = True
        router._apply_expert_bias(routes, padding_mask=processed.padding_mask)
        assert router.local_tokens_per_expert.tolist() == [1, 1, 1, 1]
        results.append(
            get_updated_expert_bias(
                router.local_tokens_per_expert,
                torch.zeros(4),
                0.001,
                tp_dp_cp_group=object(),
            )
        )
        # The first two real tokens have no policy loss but must still count.
        assert not processed.padding_mask[0, :4].any()
    assert torch.equal(results[0], results[1])
    assert torch.count_nonzero(results[0]) == 0


def test_nonpacked_router_mask_is_opt_in(monkeypatch):
    assert _processed(monkeypatch, 8, enabled=False).padding_mask is None


def test_nonpacked_router_mask_requires_lengths(monkeypatch):
    with pytest.raises(ValueError, match="require input lengths"):
        _processed(monkeypatch, 8, lengths=False)


@pytest.mark.parametrize(
    "enabled,rate,frozen,has_bias,expected",
    [
        (False, 0.001, False, True, False),
        (True, 0.0, False, True, False),
        (True, 0.001, True, True, False),
        (True, 0.001, False, False, False),
        (True, 0.001, False, True, True),
    ],
)
def test_router_mask_uses_effective_update_config(
    enabled, rate, frozen, has_bias, expected
):
    from nemo_rl.models.policy.workers.megatron_policy_worker import (
        _model_needs_router_padding_mask,
    )

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                moe_router_enable_expert_bias=enabled, moe_router_bias_update_rate=rate
            )
            self.expert_bias = torch.zeros(4) if has_bias else None
            self.frozen_expert_bias = frozen

        def forward(self, input_ids, padding_mask=None):
            return input_ids

    assert _model_needs_router_padding_mask(Model()) is expected


@pytest.mark.parametrize("chunkwise", [False, True])
def test_router_mask_rejects_unsupported_model(chunkwise):
    from nemo_rl.models.policy.workers.megatron_policy_worker import (
        _model_needs_router_padding_mask,
    )

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                moe_router_enable_expert_bias=True, moe_router_bias_update_rate=0.001
            )
            self.expert_bias = torch.zeros(4)
            self.decoder = SimpleNamespace(
                _has_linear_layer_with_chunkwise_cp=chunkwise
            )

        def forward(self, input_ids):
            return input_ids

    class MaskModel(Model):
        def forward(self, input_ids, padding_mask=None):
            return input_ids

    with pytest.raises(ValueError, match="padding_mask|context parallelism"):
        _model_needs_router_padding_mask(MaskModel() if chunkwise else Model())


def test_nonpacked_router_mask_preserves_unequal_row_lengths(monkeypatch):
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from nemo_rl.models.megatron import data as module

    monkeypatch.setattr(module, "get_context_parallel_rank", lambda: 0)
    monkeypatch.setattr(module, "get_context_parallel_world_size", lambda: 1)
    processed = module.process_microbatch(
        BatchedDataDict(
            {
                "input_ids": torch.tensor([[1, 2, 0, 0], [3, 4, 5, 6]]),
                "input_lengths": torch.tensor([2, 4]),
                "token_mask": torch.tensor([[0, 1, 0, 0], [0, 0, 1, 1]]),
            }
        ),
        pack_sequences=False,
        pad_individual_seqs_to_multiple_of=8,
        create_nonpacked_router_padding_mask=True,
    )
    assert processed.padding_mask.tolist() == [
        [False, False, True, True, True, True, True, True],
        [False, False, False, False, True, True, True, True],
    ]


def _packed_batch():
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict

    # Real token ids are nonzero, so they identify every non-padding position.
    return BatchedDataDict(
        {
            "input_ids": torch.tensor([[11, 12, 13, 0, 0], [21, 22, 23, 24, 25]]),
            "input_lengths": torch.tensor([3, 5]),
            "token_mask": torch.tensor([[0, 1, 1, 0, 0], [0, 0, 1, 1, 1]]),
        }
    )


@pytest.mark.parametrize("multiple", [4, 8])
def test_packed_padding_does_not_change_expert_bias(monkeypatch, multiple):
    from megatron.core.transformer.moe.router import TopKRouter

    from nemo_rl.models.megatron import data as module

    monkeypatch.setattr(module, "get_context_parallel_rank", lambda: 0)
    monkeypatch.setattr(module, "get_context_parallel_world_size", lambda: 1)
    processed = module.process_microbatch(
        _packed_batch(),
        seq_length_key="input_lengths",
        pack_sequences=True,
        pad_individual_seqs_to_multiple_of=multiple,
        create_packed_seq_padding_mask=True,
    )
    real = processed.input_ids[0] != 0
    assert processed.padding_mask is not None
    # Exactly the per-sequence alignment padding is masked.
    assert torch.equal(processed.padding_mask.reshape(-1), ~real)

    router = TopKRouter.__new__(TopKRouter)
    torch.nn.Module.__init__(router)
    router.enable_expert_bias = True
    router.local_tokens_per_expert = torch.zeros(4, dtype=torch.long)
    # Real tokens cycle over the experts; padding alone favors expert0.
    routes = torch.zeros((real.numel(), 4), dtype=torch.bool)
    routes[real.nonzero().squeeze(1), torch.arange(int(real.sum())) % 4] = True
    routes[~real, 0] = True
    router._apply_expert_bias(routes, padding_mask=processed.padding_mask)
    assert router.local_tokens_per_expert.tolist() == [2, 2, 2, 2]


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("model_slices_cp", [False, True])
def test_packed_router_mask_with_alltoall_dispatcher(
    monkeypatch, enabled, model_slices_cp
):
    from nemo_rl.distributed.batched_data_dict import BatchedDataDict
    from nemo_rl.models.megatron import data as module

    # Keep the iterator's device move on CPU for this regression.
    monkeypatch.setattr(BatchedDataDict, "to", lambda self, device: self)
    monkeypatch.setattr(module, "get_context_parallel_rank", lambda: 0)
    monkeypatch.setattr(module, "get_context_parallel_world_size", lambda: 1)
    batch = _packed_batch()
    batch.micro_batch_indices = [[[0, 2]]]
    batch.micro_batch_lengths = [[8]]
    cfg = {
        "megatron_cfg": {
            "tensor_model_parallel_size": 2,
            "pipeline_model_parallel_size": 1,
            "context_parallel_size": 1,
            "sequence_parallel": True,
            "moe_token_dispatcher_type": "alltoall",
            "moe_hybridep_prepad_packed_inputs": True,
        },
        "sequence_packing": {"enabled": True},
        "dynamic_batching": {"enabled": False},
        "make_sequence_length_divisible_by": 4,
    }
    iterator, *_ = module.get_microbatch_iterator(
        batch,
        cfg,
        mbs=1,
        straggler_timer=None,
        seq_length_key="input_lengths",
        create_router_padding_mask=enabled,
        model_slices_context_parallel_inputs=model_slices_cp,
    )
    processed = next(iterator)
    if enabled:
        real = processed.input_ids[0] != 0
        assert torch.equal(processed.padding_mask.reshape(-1), ~real)
        # Lengths 3 and 5 align to 4 and 8; HybridEP prepadding adds nothing.
        assert processed.input_ids.shape[1] == 12
    else:
        assert processed.padding_mask is None


@pytest.mark.parametrize("chunkwise", [False, True])
def test_nested_language_decoder_router_mask(chunkwise):
    from nemo_rl.models.policy.workers.megatron_policy_worker import (
        _model_needs_router_padding_mask,
    )

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                moe_router_enable_expert_bias=True, moe_router_bias_update_rate=0.001
            )
            self.expert_bias = torch.zeros(4)
            self.language_model = SimpleNamespace(
                decoder=SimpleNamespace(_has_linear_layer_with_chunkwise_cp=chunkwise)
            )

        def forward(self, input_ids, padding_mask=None):
            return input_ids

    if chunkwise:
        with pytest.raises(ValueError, match="chunkwise context parallelism"):
            _model_needs_router_padding_mask(Model())
    else:
        assert _model_needs_router_padding_mask(Model())


@pytest.mark.parametrize("packing", [False, True])
def test_self_packing_expert_bias_rejected_at_setup(packing):
    from nemo_rl.models.policy.workers.megatron_policy_worker import (
        _model_needs_router_padding_mask,
    )

    class Model(torch.nn.Module):
        model_owns_packing = True

        def __init__(self):
            super().__init__()
            self.config = SimpleNamespace(
                moe_router_enable_expert_bias=True, moe_router_bias_update_rate=0.001
            )
            self.expert_bias = torch.zeros(4)

        def forward(self, input_ids, padding_mask=None):
            return input_ids

    if packing:
        with pytest.raises(ValueError, match="model-owned sequence packing"):
            _model_needs_router_padding_mask(Model(), pack_sequences=packing)
    else:
        assert _model_needs_router_padding_mask(Model(), pack_sequences=packing)


def test_delegated_packing_cannot_silently_drop_router_mask(monkeypatch):
    from nemo_rl.models.megatron import data as module

    monkeypatch.setattr(module, "get_context_parallel_rank", lambda: 0)
    monkeypatch.setattr(module, "get_context_parallel_world_size", lambda: 1)
    with pytest.raises(NotImplementedError, match="real-token validity"):
        module.process_microbatch(
            _packed_batch(),
            seq_length_key="input_lengths",
            pack_sequences=True,
            delegate_pack_to_model=True,
            pad_individual_seqs_to_multiple_of=4,
            create_packed_seq_padding_mask=True,
        )
