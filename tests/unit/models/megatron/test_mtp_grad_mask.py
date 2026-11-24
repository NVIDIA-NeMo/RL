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

"""MTP gradient masking for in-loss sequence filtering.

Megatron-LM's ``MTPLossGradMask`` is replaced by small fakes so these tests run
against any installed Megatron-LM.
"""

from unittest.mock import patch

import pytest
import torch

from nemo_rl.algorithms.loss.interfaces import SEQ_LOGPROB_KEEP_MASK_KEY

pytestmark = pytest.mark.mcore


class _FakeGradMask:
    def __init__(self) -> None:
        self.mask = None


class _FakeAutoScaler:
    grad_mask_provider = None

    @staticmethod
    def set_grad_mask_provider(provider) -> None:
        _FakeAutoScaler.grad_mask_provider = provider


@pytest.fixture
def fake_megatron_hook(monkeypatch):
    from nemo_rl.models.megatron import mtp_grad_mask

    monkeypatch.setattr(mtp_grad_mask, "HAVE_MTP_GRAD_MASK", True)
    monkeypatch.setattr(mtp_grad_mask, "MTPLossGradMask", _FakeGradMask, raising=False)
    monkeypatch.setattr(
        mtp_grad_mask, "MTPLossAutoScaler", _FakeAutoScaler, raising=False
    )
    _FakeAutoScaler.grad_mask_provider = None
    yield
    _FakeAutoScaler.grad_mask_provider = None


def test_row_ids_number_rows_from_one():
    from nemo_rl.models.megatron.mtp_grad_mask import mtp_row_ids_like

    row_ids = mtp_row_ids_like(torch.zeros(3, 4))
    assert row_ids.tolist() == [[1.0] * 4, [2.0] * 4, [3.0] * 4]


def test_fill_maps_each_rows_decision_onto_the_mtp_layout(fake_megatron_hook):
    from nemo_rl.models.megatron.mtp_grad_mask import MTPRowGradMask

    # Two packed rows with padding (0) between and after them.
    row_ids = torch.tensor([[1.0, 1.0, 1.0, 0.0, 2.0, 2.0, 0.0, 0.0]])
    grad_mask = MTPRowGradMask(row_ids)
    grad_mask.fill([0.0, 1.0])
    assert grad_mask.holder.mask.tolist() == [[0, 0, 0, 0, 1, 1, 0, 0]]


@pytest.mark.parametrize(
    "keep_mask, message", [(None, "returned none"), ([1.0], "kept 1 rows")]
)
def test_fill_rejects_a_missing_or_short_keep_mask(
    fake_megatron_hook, keep_mask, message
):
    from nemo_rl.models.megatron.mtp_grad_mask import MTPRowGradMask

    grad_mask = MTPRowGradMask(torch.tensor([[1.0, 2.0]]))
    with pytest.raises(RuntimeError, match=message):
        grad_mask.fill(keep_mask)


def test_an_old_megatron_fails_clearly(monkeypatch):
    from nemo_rl.models.megatron import mtp_grad_mask

    monkeypatch.setattr(mtp_grad_mask, "HAVE_MTP_GRAD_MASK", False)
    with pytest.raises(ValueError, match="MTPLossGradMask"):
        mtp_grad_mask.require_mtp_grad_mask_support()


def test_loss_post_processor_fills_the_mask_after_the_loss(fake_megatron_hook):
    from nemo_rl.models.megatron import train as megatron_train
    from nemo_rl.models.megatron.mtp_grad_mask import MTPRowGradMask

    def fake_loss(next_token_logits, **_kwargs):
        return next_token_logits.sum(), {SEQ_LOGPROB_KEEP_MASK_KEY: [1.0, 0.0]}

    grad_mask = MTPRowGradMask(torch.tensor([[1.0, 1.0], [2.0, 2.0]]))
    processor = megatron_train.LossPostProcessor(
        loss_fn=object(), cfg={"sequence_packing": {"enabled": False}}
    )
    with (
        patch.object(megatron_train, "wrap_loss_fn_with_input_preparation", fake_loss),
        patch.object(megatron_train, "get_tensor_model_parallel_rank", return_value=0),
        patch.object(megatron_train, "get_tensor_model_parallel_group"),
        patch.object(megatron_train, "get_context_parallel_group"),
        patch.object(megatron_train, "get_context_parallel_world_size", return_value=1),
    ):
        wrapped = processor(data_dict={}, mtp_grad_mask=grad_mask)
        assert grad_mask.holder.mask is None
        wrapped(torch.ones(2, 2))
    assert grad_mask.holder.mask.tolist() == [[1, 1], [0, 0]]


@patch("nemo_rl.models.megatron.train.model_forward")
def test_forward_registers_the_mask_only_for_its_own_forward(
    mock_model_forward, fake_megatron_hook
):
    from nemo_rl.models.megatron.data import ProcessedMicrobatch
    from nemo_rl.models.megatron.train import forward_with_post_processing_fn

    providers = []

    def forward(**_kwargs):
        providers.append(_FakeAutoScaler.grad_mask_provider)
        return torch.zeros(1, 2, 4)

    mock_model_forward.side_effect = forward
    processed_mb = ProcessedMicrobatch(
        data_dict={},
        input_ids=torch.tensor([[1, 2]]),
        input_ids_cp_sharded=torch.tensor([[1, 2]]),
        attention_mask=None,
        position_ids=None,
        packed_seq_params=None,
        cu_seqlens_padded=None,
        mtp_row_ids=torch.tensor([[1.0, 1.0]]),
    )
    post_processor = object()  # neither loss nor logprobs: only the forward runs
    with pytest.raises(TypeError, match="Unknown post-processing function"):
        forward_with_post_processing_fn(
            data_iterator=iter([processed_mb]),
            model=object(),
            post_processing_fn=post_processor,
        )
    assert providers[0] is not None and providers[0]() is not None
    assert _FakeAutoScaler.grad_mask_provider is None
