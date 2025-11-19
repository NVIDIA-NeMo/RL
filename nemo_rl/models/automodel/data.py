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

import itertools
from typing import Any, Iterator

import torch
from transformers import AutoTokenizer

from nemo_rl.algorithms.interfaces import LossFunction, LossType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.automodel.types import ProcessedInputs
from nemo_rl.models.huggingface.common import (
    get_flash_attention_kwargs,
    pack_sequences,
)


def get_microbatch_iterator(
    data: BatchedDataDict[Any],
    cfg: dict[str, Any],
    enable_seq_packing: bool,
    mbs: int,
    dp_mesh: Any,  # noqa: ARG001
) -> tuple[Iterator, int, Iterator]:
    dummy_iterator = iter([])

    if cfg["dynamic_batching"]["enabled"]:
        mb_iterator = data.make_microbatch_iterator_with_dynamic_shapes()
        iterator_len = data.get_microbatch_iterator_dynamic_shapes_len()
    elif enable_seq_packing:
        mb_iterator = data.make_microbatch_iterator_for_packable_sequences()
        iterator_len, _ = data.get_microbatch_iterator_for_packable_sequences_len()
        max_batch_ct = torch.tensor([iterator_len], device="cuda")
        torch.distributed.all_reduce(max_batch_ct, op=torch.distributed.ReduceOp.MAX)

        # Sequence packing can end up with unevenly distributed batch counts across DP ranks.
        # We add dummy batches to the end of the iterator to make the batch counts equal.
        dummy_batch_ct = int(max_batch_ct.item() - iterator_len)
        dummy_iterator = data.make_microbatch_iterator_for_packable_sequences()
        dummy_iterator = itertools.islice(
            itertools.cycle(dummy_iterator), dummy_batch_ct
        )
    else:
        mb_iterator = data.make_microbatch_iterator(mbs)
        iterator_len = data.size // mbs

    return mb_iterator, iterator_len, dummy_iterator


def process_microbatch(
    mb: BatchedDataDict[Any],
    tokenizer: AutoTokenizer,
    enable_seq_packing: bool,
    cfg: dict[str, Any],
    cp_size: int,
) -> ProcessedInputs:
    """Process a microbatch and prepare inputs for model forward.

    Args:
        mb: Microbatch data
        tokenizer: Tokenizer for padding value
        enable_seq_packing: Whether sequence packing is enabled
        cfg: Configuration dictionary
        cp_size: Context parallel size

    Returns:
        ProcessedInputs containing all tensors and metadata for forward pass
    """
    input_ids = mb.get("input_ids").cuda()

    if enable_seq_packing:
        input_ids, position_ids, _ = pack_sequences(
            input_ids=input_ids,
            input_lengths=mb["input_lengths"],
            packed_sequence_size=[
                len(mb["input_lengths"])
            ],  # flash attention 2 expects flattened input
            padding_value=tokenizer.eos_token_id,
            return_attention_mask=False,
            min_seq_len=cfg["sequence_packing"][
                "train_mb_tokens"
            ],  # TODO: this is a WAR for sequence packing, we should fix this. Without this, backward will fail when TP is enabled.
        )
        seq_len = input_ids.shape[1]
        attention_mask = None
        flash_attn_kwargs = get_flash_attention_kwargs(
            input_lengths=mb["input_lengths"],
        )
    else:
        batch_size, seq_len = input_ids.shape

        attention_mask = torch.ones(
            (batch_size, seq_len),
            dtype=torch.bool,
            device=input_ids.device,
        )
        position_ids = torch.arange(seq_len, device=input_ids.device).repeat(
            batch_size, 1
        )
        flash_attn_kwargs = {}

    # Add vlm kwargs to model call
    vlm_kwargs = mb.get_multimodal_dict(as_tensors=True, device=input_ids.device)
    if len(vlm_kwargs) > 0:
        position_ids = None
        assert not cfg["dtensor_cfg"]["sequence_parallel"], (
            "Sequence parallel is not supported with multimodal since there's an issue when you do not pass position_ids. See https://github.com/NVIDIA-NeMo/Automodel/issues/652"
        )

    # Prepare context parallel buffers if needed
    cp_buffers = []
    seq_index = None
    if cp_size > 1:
        assert len(vlm_kwargs) == 0, (
            f"multimodal kwargs={vlm_kwargs} are not supported for context parallel"
        )
        seq_index = torch.arange(seq_len, device=input_ids.device).repeat(1, 1)
        cp_buffers = [input_ids, position_ids, seq_index]

    return ProcessedInputs(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        flash_attn_kwargs=flash_attn_kwargs,
        vlm_kwargs=vlm_kwargs,
        cp_buffers=cp_buffers,
        seq_index=seq_index,
        seq_len=seq_len,
    )


def process_global_batch(
    data: BatchedDataDict[Any],
    batch_idx: int,
    batch_size: int,
    loss_fn: LossFunction,
    dp_mesh: Any,
) -> dict[str, Any]:
    batch = data.get_batch(batch_idx=batch_idx, batch_size=batch_size)

    assert "sample_mask" in batch, "sample_mask must be present in the data!"

    # Get the normalization factor for the loss
    local_valid_seqs = torch.sum(batch["sample_mask"])

    if "token_mask" not in batch:
        local_valid_toks = local_valid_seqs * batch["input_ids"].shape[1]
    else:
        local_valid_toks = torch.sum(
            batch["token_mask"][:, 1:] * batch["sample_mask"].unsqueeze(-1)
        )

    to_reduce = torch.tensor([local_valid_seqs, local_valid_toks]).cuda()
    torch.distributed.all_reduce(to_reduce, group=dp_mesh.get_group())
    global_valid_seqs, global_valid_toks = to_reduce[0], to_reduce[1]

    if hasattr(loss_fn, "loss_type") and loss_fn.loss_type == LossType.TOKEN_LEVEL:
        assert "token_mask" in batch, (
            "token_mask must be present in the data when using token-level loss"
        )

    return {
        "batch": batch,
        "global_valid_seqs": global_valid_seqs,
        "global_valid_toks": global_valid_toks,
    }
