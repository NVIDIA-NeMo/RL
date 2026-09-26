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
"""CPU integration tests for chat terminator targets at the loss boundary."""

from dataclasses import fields
from pathlib import Path

import pytest
import torch
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import WhitespaceSplit
from transformers import PreTrainedTokenizerFast

from nemo_rl.algorithms.loss.loss_functions import CrossTokenizerDistillationLossFn
from nemo_rl.algorithms.x_token.loss_utils import LocalizedAlignment
from nemo_rl.algorithms.x_token.token_aligner import TokenAligner
from nemo_rl.data.cross_tokenizer_collate import (
    CrossTokenizerCollator,
    CrossTokenizerCollatorConfig,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.model_utils import cp_shift_next

_MESSAGES = [
    {"role": "user", "content": "prompt"},
    {"role": "assistant", "content": "answer"},
    {"role": "user", "content": "followup"},
    {"role": "assistant", "content": "final"},
]


def _tokenizer(*, reverse_vocab: bool = False) -> PreTrainedTokenizerFast:
    words = [
        "<pad>",
        "<unk>",
        "<user>",
        "<assistant>",
        "<eot>",
        "prompt",
        "answer",
        "followup",
        "final",
    ]
    if reverse_vocab:
        words.reverse()
    backend = Tokenizer(
        WordLevel(dict(zip(words, range(len(words)))), unk_token="<unk>")
    )
    backend.pre_tokenizer = WhitespaceSplit()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<pad>",
        unk_token="<unk>",
        eos_token="<eot>",
        additional_special_tokens=["<user>", "<assistant>"],
    )
    tokenizer.chat_template = (
        "{% for message in messages %}"
        "{{ '<' + message['role'] + '> ' + message['content'] + ' <eot> ' }}"
        "{% endfor %}"
    )
    return tokenizer


def _batch(
    student: PreTrainedTokenizerFast,
    *,
    ctx_length: int = 32,
    teacher: PreTrainedTokenizerFast | None = None,
    aligner: TokenAligner | None = None,
) -> BatchedDataDict:
    collator = CrossTokenizerCollator(
        config=CrossTokenizerCollatorConfig(mode="chat"),
        student_tokenizer=student,
        teacher_tokenizers=[teacher],
        aligners=[aligner],
        ctx_length_student=ctx_length,
        ctx_length_teachers=[ctx_length],
        drop_first_assistant_chunk_kl_by_teacher=[False],
        make_seq_div_by_student=16,
        make_seq_div_by_teachers=[16],
    )
    return collator([{"message_log": _MESSAGES, "loss_multiplier": 1.0, "idx": 0}])


def _loss_fn(vocab_size: int) -> CrossTokenizerDistillationLossFn:
    loss_fn = CrossTokenizerDistillationLossFn.__new__(CrossTokenizerDistillationLossFn)
    loss_fn.temperature = 1.0
    loss_fn.student_vocab_size = vocab_size
    loss_fn.vocab_topk = vocab_size
    loss_fn.reverse_kl = False
    loss_fn.exact_token_match_only = False
    return loss_fn


@pytest.mark.parametrize("loss_kind", ["ce", "same_tokenizer_kd"])
@pytest.mark.parametrize("truncated", [False, True])
def test_chat_loss_supervises_eot_predictor_only(
    loss_kind: str, truncated: bool
) -> None:
    tokenizer = _tokenizer()
    # The full chat has assistant targets at 4/5 and 10/11 (answer/EOT).
    # Truncation at 5 retains the first answer but removes its EOT.
    batch = _batch(tokenizer, ctx_length=5 if truncated else 32)
    expected_targets = [4] if truncated else [4, 5, 10, 11]
    assert batch["token_mask"].nonzero(as_tuple=True)[1].tolist() == expected_targets
    logits = torch.zeros(
        (*batch["input_ids"].shape, len(tokenizer)), requires_grad=True
    )
    loss_fn = _loss_fn(len(tokenizer))
    global_valid_toks = batch["token_mask"].sum()
    if loss_kind == "ce":
        loss = loss_fn._compute_ce(logits, batch, global_valid_toks, cp_group=None)
    else:
        teacher_logits = torch.zeros_like(logits)
        teacher_logits[..., tokenizer.eos_token_id] = 2.0
        alignment = LocalizedAlignment(
            sample_mask=batch["sample_mask"],
            student_input_ids=batch["input_ids"],
            student_token_mask=batch["token_mask"],
        )
        loss = loss_fn._direct_topk_kl(
            logits,
            teacher_logits,
            alignment,
            global_valid_toks,
            tp_group=None,
            cp_group=None,
        )
    assert loss.item() > 0
    loss.backward()
    assert logits.grad is not None
    active_predictors = logits.grad.abs().sum(dim=-1).nonzero(as_tuple=True)[1].tolist()
    assert active_predictors == ([3] if truncated else [3, 4, 9, 10])
    if not truncated:
        # Answer logits learn the terminator; outgoing EOT predictions target
        # the next role header or padding and must remain unsupervised.
        assert (logits.grad[0, [4, 10], tokenizer.eos_token_id] < 0).all()
        assert torch.count_nonzero(logits.grad[0, [5, 11]]) == 0
    else:
        assert torch.count_nonzero(logits.grad[0, 4:]) == 0


def test_eot_supervision_mask_does_not_change_cross_tokenizer_kd(
    tmp_path: Path,
) -> None:
    student, teacher = _tokenizer(), _tokenizer(reverse_vocab=True)
    projection_path = tmp_path / "projection.pt"
    torch.save(
        {
            (s_id, teacher.convert_tokens_to_ids(word)): 1.0
            for word, s_id in student.get_vocab().items()
        },
        projection_path,
    )
    aligner = TokenAligner(student, teacher, str(projection_path))
    batch = _batch(student, teacher=teacher, aligner=aligner)
    rendered = student.apply_chat_template(_MESSAGES, tokenize=False)
    spans = [
        [
            (rendered.index(word), rendered.index(word) + len(word))
            for word in ("answer", "final")
        ]
    ]
    encoded = [
        tokenizer(
            rendered,
            add_special_tokens=False,
            return_offsets_mapping=True,
            padding="max_length",
            max_length=16,
            return_tensors="pt",
        )
        for tokenizer in (student, teacher)
    ]
    student_logits = torch.zeros((1, 16, len(student)), requires_grad=True)
    teacher_logits = torch.zeros((1, 16, len(teacher)))
    teacher_logits[..., teacher.eos_token_id] = 2.0
    loss_fn = _loss_fn(len(student))
    alignments, losses, gradients = [], [], []
    for supervise_eot in (False, True):
        student_mask = batch["token_mask"].clone()
        teacher_mask = batch["teacher_0_token_mask"].clone()
        student_mask[:, [5, 11]] = int(supervise_eot)
        teacher_mask[:, [5, 11]] = int(supervise_eot)
        alignment = aligner.align_chat(
            batch["input_ids"],
            batch["teacher_0_input_ids"],
            student_offsets=encoded[0]["offset_mapping"],
            teacher_offsets=encoded[1]["offset_mapping"],
            student_asst_char_spans=spans,
            teacher_asst_char_spans=spans,
            student_attention_mask=encoded[0]["attention_mask"],
            teacher_attention_mask=encoded[1]["attention_mask"],
            student_asst_mask=student_mask,
            teacher_asst_mask=teacher_mask,
            student_eot_indices=[[5, 11]],
            teacher_eot_indices=[[5, 11]],
        )
        localized = LocalizedAlignment(
            sample_mask=batch["sample_mask"],
            student_chunk_id=cp_shift_next(alignment.student_chunk_id, None, fill=-1),
            teacher_chunk_id=cp_shift_next(alignment.teacher_chunk_id, None, fill=-1),
            pair_valid=alignment.pair_valid,
            pair_is_correct=alignment.pair_is_correct,
            student_input_ids=batch["input_ids"],
            student_token_mask=student_mask,
        )
        loss, valid_pairs, _ = loss_fn._compute_p_kl(
            student_logits,
            teacher_logits,
            localized,
            projection_matrix_path=str(projection_path),
            teacher_vocab_size=len(teacher),
            tp_group=None,
            cp_group=None,
        )
        assert valid_pairs.item() == 4  # Two content pairs and two EOT pairs.
        alignments.append(alignment)
        losses.append(loss.detach())
        gradients.append(torch.autograd.grad(loss, student_logits)[0])
    for field in fields(alignments[0]):
        torch.testing.assert_close(
            getattr(alignments[0], field.name), getattr(alignments[1], field.name)
        )
        torch.testing.assert_close(
            getattr(alignments[1], field.name), batch[f"alignment_0_{field.name}"]
        )
    torch.testing.assert_close(losses[0], losses[1])
    torch.testing.assert_close(gradients[0], gradients[1])
    assert (gradients[1][0, [4, 10], student.eos_token_id] < 0).all()
    assert torch.count_nonzero(gradients[1][0, [5, 11]]) == 0
