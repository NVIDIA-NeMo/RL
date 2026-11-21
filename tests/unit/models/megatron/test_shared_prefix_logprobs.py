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
"""CPU tests for shared-prefix packs in the forward and the post-processors.

A tree is one prompt followed by the completions of the rows that start with it.
A pack is one model forward that holds several trees and runs each prompt once.
A dense bin is a model forward of ordinary packed sequences.
"""

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import torch

# Probe the concrete module: test_modelopt_worker_utils.py leaves stub
# ``megatron.bridge`` packages in sys.modules that a package probe accepts.
pytest.importorskip("megatron.bridge.training.utils.packed_seq_utils")

from nemo_rl.algorithms.loss import (  # noqa: E402
    ClippedPGLossConfig,
    ClippedPGLossFn,
)
from nemo_rl.algorithms.loss.interfaces import (  # noqa: E402
    group_token_mean_weights,
)
from nemo_rl.algorithms.loss.wrapper import StreamingObjectiveLoss  # noqa: E402
from nemo_rl.data_plane.schema import GROUP_TOKEN_MEAN_WEIGHT  # noqa: E402
from nemo_rl.distributed.batched_data_dict import BatchedDataDict  # noqa: E402
from nemo_rl.models.megatron import data as megatron_data  # noqa: E402
from nemo_rl.models.megatron import train  # noqa: E402
from nemo_rl.models.megatron.shared_prefix import microbatches  # noqa: E402
from nemo_rl.models.megatron.shared_prefix.logprobs import (  # noqa: E402
    shared_prefix_next_token_logprobs,
)
from tests.unit.models.megatron.test_shared_prefix_microbatches import (  # noqa: E402
    shared_prefix_rows,
    shared_prefix_topology,
)

pytestmark = pytest.mark.mcore

# Rows 0, 1 and 2 share the prompt [1, 2, 3]. Rows 3 and 4 share the prompt [9, 8].
_PACK_TOKENS = [[1, 2, 3, 4, 5], [1, 2, 3, 6], [1, 2, 3, 7, 8], [9, 8, 7], [9, 8, 6, 5]]
_PACK_PROMPT_LENGTHS = [3, 3, 3, 2, 2]
_PACK_TREE_IDS = [0, 0, 0, 1, 1]
_PACK_VOCAB = 16
# shared_prefix_rows makes the rows one token wider than the longest row.
_PACK_WIDTH = max(len(row) for row in _PACK_TOKENS) + 1


def _pack_rows() -> BatchedDataDict[Any]:
    """Rows of one pack. Row 2 is masked out of the loss."""
    data = shared_prefix_rows(_PACK_TOKENS, _PACK_PROMPT_LENGTHS, _PACK_TREE_IDS)
    data["sample_mask"] = torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0])
    return data


def _build_pack(data: BatchedDataDict[Any], *, padding_multiple: int) -> Any:
    """Build the microbatch of a pack on the host at TP 1 and CP 1."""
    with shared_prefix_topology():
        return microbatches.build_shared_prefix_microbatch(
            data,
            tree_sizes=microbatches.shared_prefix_tree_sizes(data),
            padding_multiple=padding_multiple,
            exclude_sequence_padding_from_expert_bias=False,
        )


class _TokenPositionModel:
    """Logits that depend only on a token and on its position in its row.

    A pack and a dense forward then give the same token the same logits, so a
    logprob that reaches the wrong row or column changes the result.
    """

    def __init__(self) -> None:
        generator = torch.Generator().manual_seed(0)
        self.token_table = torch.randn(
            _PACK_VOCAB, _PACK_VOCAB, generator=generator, requires_grad=True
        )
        self.position_table = torch.randn(
            2 * _PACK_WIDTH, _PACK_VOCAB, generator=generator, requires_grad=True
        )

    def __call__(
        self, input_ids: torch.Tensor, position_ids: torch.Tensor
    ) -> torch.Tensor:
        return self.token_table[input_ids] + self.position_table[position_ids]

    def gradients(self, loss: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return torch.autograd.grad(loss, (self.token_table, self.position_table))


def _mock_data_dict() -> MagicMock:
    data_dict = MagicMock()
    data_dict.get_multimodal_dict.return_value = {}
    return data_dict


@pytest.mark.parametrize("exclude_sequence_padding", [False, True])
def test_model_forward_passes_the_layout_of_a_pack(
    exclude_sequence_padding: bool,
) -> None:
    layout = object()
    params = microbatches.SharedPrefixParams(
        inputs=SimpleNamespace(layout=layout),
        cp_shard=SimpleNamespace(),
        sequence_length=3,
        exclude_sequence_padding_from_expert_bias=exclude_sequence_padding,
    )
    mock_model = MagicMock(return_value=torch.randn(1, 3, 100))

    train.model_forward(
        model=mock_model,
        data_dict=_mock_data_dict(),
        input_ids_cp_sharded=torch.tensor([[1, 2, 3]]),
        position_ids=torch.tensor([[0, 1, 2]]),
        attention_mask=None,
        shared_prefix_params=params,
    )

    call_kwargs = mock_model.call_args[1]
    assert call_kwargs["shared_prefix_layout"] is layout
    assert (
        call_kwargs["shared_prefix_exclude_sequence_padding_from_expert_bias"]
        is exclude_sequence_padding
    )
    assert "packed_seq_params" not in call_kwargs


def test_model_forward_passes_no_layout_without_a_pack() -> None:
    mock_model = MagicMock(return_value=torch.randn(1, 3, 100))

    train.model_forward(
        model=mock_model,
        data_dict=_mock_data_dict(),
        input_ids_cp_sharded=torch.tensor([[1, 2, 3]]),
        position_ids=torch.tensor([[0, 1, 2]]),
        attention_mask=None,
    )

    assert not any(name.startswith("shared_prefix") for name in mock_model.call_args[1])


@pytest.mark.parametrize(
    "post_processor_name", ["LossPostProcessor", "LogprobsPostProcessor"]
)
@patch("nemo_rl.models.megatron.train.model_forward")
def test_forward_hands_a_pack_to_the_model_and_the_post_processor(
    mock_model_forward: MagicMock, post_processor_name: str
) -> None:
    params = MagicMock()
    processed_mb = megatron_data.ProcessedMicrobatch(
        data_dict=MagicMock(),
        input_ids=torch.tensor([[1, 2, 3]]),
        input_ids_cp_sharded=torch.tensor([[1, 2, 3]]),
        attention_mask=None,
        position_ids=torch.tensor([[0, 1, 2]]),
        packed_seq_params=None,
        cu_seqlens_padded=None,
        original_seq_length=3,
        shared_prefix_params=params,
    )
    mock_model_forward.return_value = torch.randn(1, 3, 100)
    post_processor = MagicMock(spec=getattr(train, post_processor_name))

    train.forward_with_post_processing_fn(
        data_iterator=iter([processed_mb]),
        model=MagicMock(),
        post_processing_fn=post_processor,
    )

    assert mock_model_forward.call_args.kwargs["shared_prefix_params"] is params
    assert post_processor.call_args.kwargs["shared_prefix_params"] is params


def test_packed_logprobs_need_boundaries_unless_the_microbatch_is_a_pack() -> None:
    """A dense bin without boundaries fails; a pack has none and needs none."""
    processor = train.LogprobsPostProcessor(
        cfg={"sequence_packing": {"enabled": True}, "logprob_chunk_size": 5}
    )
    data = BatchedDataDict({"input_ids": torch.tensor([[1, 2, 3, 4]])})
    with pytest.raises(AssertionError):
        processor(
            data_dict=data,
            input_ids=data["input_ids"],
            cu_seqlens_padded=None,
            original_seq_length=4,
        )

    params = MagicMock()
    logits = torch.randn(1, 4, 8)
    pack_logprobs = torch.randn(1, 3)
    with (
        patch.object(train, "get_tensor_model_parallel_rank", return_value=0),
        patch.object(train, "get_tensor_model_parallel_group") as tp_group,
        patch.object(train, "get_context_parallel_group") as cp_group,
        patch.object(
            train, "shared_prefix_next_token_logprobs", return_value=pack_logprobs
        ) as extract,
    ):
        _, outputs = processor(
            data_dict=data,
            input_ids=data["input_ids"],
            cu_seqlens_padded=None,
            original_seq_length=4,
            shared_prefix_params=params,
        )(logits)

    extract.assert_called_once_with(
        logits,
        params,
        vocab_parallel_rank=0,
        vocab_parallel_group=tp_group.return_value,
        context_parallel_group=cp_group.return_value,
        chunk_size=5,
    )
    # The post-processor prepends the logprob 0 of the first token.
    assert torch.equal(outputs["logprobs"][:, 1:], pack_logprobs)
    assert not outputs["logprobs"][:, 0].any()


@pytest.mark.parametrize("padding_multiple", [1, 4])
def test_pack_logprobs_match_dense_rows(padding_multiple: int) -> None:
    """The logprobs of a pack equal a dense forward of every row.

    Every row of a tree gets the same copy of the prompt logprobs.
    """
    data = _pack_rows()
    microbatch = _build_pack(data, padding_multiple=padding_multiple)
    model = _TokenPositionModel()
    packed_logits = model(microbatch.input_ids_cp_sharded, microbatch.position_ids)
    ids = data["input_ids"]
    dense = torch.log_softmax(model(ids, torch.arange(_PACK_WIDTH)), dim=-1)
    dense_next = dense[:, :-1].gather(-1, ids[:, 1:, None]).squeeze(-1)

    def extract(logits: torch.Tensor) -> torch.Tensor:
        with patch.object(torch.distributed, "get_world_size", return_value=1):
            return shared_prefix_next_token_logprobs(
                logits,
                microbatch.shared_prefix_params,
                vocab_parallel_rank=0,
                vocab_parallel_group=None,
                context_parallel_group=None,
                chunk_size=3,
            )

    logprobs = extract(packed_logits)

    assert logprobs.shape == (len(_PACK_TOKENS), _PACK_WIDTH - 1)
    for row, tokens in enumerate(_PACK_TOKENS):
        valid = len(tokens) - 1
        torch.testing.assert_close(
            logprobs[row, :valid], dense_next[row, :valid], rtol=0, atol=1e-6
        )
        assert not logprobs[row, valid:].any()
    # Columns 0 and 1 hold the logprobs of the prompt tokens 2 and 3.
    assert torch.equal(logprobs[1, :2], logprobs[0, :2])
    assert torch.equal(logprobs[2, :2], logprobs[0, :2])
    assert torch.equal(logprobs[4, :1], logprobs[3, :1])

    with pytest.raises(ValueError, match="The logits of a pack must have shape"):
        extract(packed_logits[:, :-1])
    with pytest.raises(RuntimeError, match="vocabulary"):
        extract(packed_logits[..., :5])


@pytest.mark.parametrize(
    "normalization", ["global_token_mean", "sequence_mean", "group_token_mean"]
)
def test_pack_loss_matches_dense_bin_loss(normalization: str) -> None:
    """A pack gives the loss and the gradient of a dense bin of the same rows.

    Both run as the split training path runs them, with the placeholder count
    1 as the divisor.
    """
    data = _pack_rows()
    generator = torch.Generator().manual_seed(1)
    for field in ("advantages", "prev_logprobs", "generation_logprobs"):
        data[field] = torch.randn(data["input_ids"].shape, generator=generator)
    data[GROUP_TOKEN_MEAN_WEIGHT] = group_token_mean_weights(
        [str(row) for row in range(data.size)],
        data["token_mask"],
        data["sample_mask"],
        group_indices=torch.tensor(_PACK_TREE_IDS),
    )
    loss_fn = ClippedPGLossFn(
        ClippedPGLossConfig(
            normalization=normalization, reference_policy_kl_penalty=0.0
        )
    )
    if loss_fn.objective_reduction.requires_explicit_denominator:
        loss_fn = StreamingObjectiveLoss(loss_fn)
    processor = train.LossPostProcessor(
        loss_fn=loss_fn, cfg={"sequence_packing": {"enabled": True}}
    )
    placeholder = torch.tensor(1.0)
    model = _TokenPositionModel()
    pack = _build_pack(data, padding_multiple=2)

    with (
        patch.object(megatron_data, "get_context_parallel_rank", return_value=0),
        patch.object(megatron_data, "get_context_parallel_world_size", return_value=1),
        patch.object(train, "get_context_parallel_world_size", return_value=1),
        patch.object(train, "get_context_parallel_group", return_value=None),
        patch.object(train, "get_tensor_model_parallel_group", return_value=None),
        patch.object(train, "get_tensor_model_parallel_rank", return_value=0),
        patch.object(torch.distributed, "get_world_size", return_value=1),
    ):
        # mtp_enabled makes the dense bin carry each token's position.
        dense_bin = megatron_data.process_microbatch(
            data,
            seq_length_key="input_lengths",
            pad_individual_seqs_to_multiple_of=2,
            pack_sequences=True,
            straggler_timer=MagicMock(),
            mtp_enabled=True,
        )
        dense_loss, _ = processor(
            data_dict=data,
            packed_seq_params=dense_bin.packed_seq_params,
            global_valid_seqs=placeholder,
            global_valid_toks=placeholder,
        )(model(dense_bin.input_ids_cp_sharded, dense_bin.position_ids))
        pack_loss, _ = processor(
            data_dict=data,
            global_valid_seqs=placeholder,
            global_valid_toks=placeholder,
            shared_prefix_params=pack.shared_prefix_params,
        )(model(pack.input_ids_cp_sharded, pack.position_ids))

    assert dense_loss.abs() > 1e-3
    torch.testing.assert_close(pack_loss, dense_loss, rtol=1e-5, atol=0)
    for pack_grad, dense_grad in zip(
        model.gradients(pack_loss), model.gradients(dense_loss)
    ):
        torch.testing.assert_close(pack_grad, dense_grad, rtol=1e-4, atol=1e-6)
