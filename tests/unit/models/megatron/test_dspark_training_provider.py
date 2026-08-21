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

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import Tensor

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.megatron.draft.hidden_capture import CapturedStates
from nemo_rl.models.megatron.draft.sequence_layout import build_draft_sequence_layout
from nemo_rl.models.megatron.draft.training import (
    DSparkForwardOutput,
    DSparkSpeculator,
    resolve_draft_speculator,
)
from nemo_rl.models.policy.draft_config import DSparkDraftConfig

pytestmark = pytest.mark.mcore


def _config() -> DSparkDraftConfig:
    return DSparkDraftConfig(
        enabled=True,
        model_name="deepseek-ai/dspark_qwen3_8b_block7",
        model_revision="03326e5043815da1f81b109078b2889737c26017",  # pragma: allowlist secret
        block_size=7,
        anchors_per_sample=2,
        mask_token_id=151669,
        target_hidden_state_layer_ids=[1, 9, 17, 25, 33],
        num_layers=5,
        markov_rank=256,
        confidence_enabled=True,
        confidence_with_markov=True,
        seed=42,
        max_cp_boundary_exclusion_fraction=1.0,
    )


def test_dspark_registry_resolves_public_training_provider() -> None:
    provider = resolve_draft_speculator(_config())

    assert isinstance(provider, DSparkSpeculator)
    assert provider.capture_layer_ids() == (1, 9, 17, 25, 33)
    assert provider.supports_context_parallel
    assert provider.supports_sequence_packing
    assert provider.supports_target_sequence_parallel
    assert provider.requires_full_cp_local_capture


def test_dspark_build_rejects_split_vocab_before_body_creation() -> None:
    provider = DSparkSpeculator(
        _config().model_copy(
            update={"model_name": None, "draft_vocab_size": 32_000}
        )
    )

    with pytest.raises(ValueError, match="must match the live target vocabulary"):
        provider.build_model(
            model_provider=SimpleNamespace(vocab_size=151_936),
            pg_collection=SimpleNamespace(),
            policy_model_chunk=SimpleNamespace(),
        )


@pytest.mark.parametrize("cp_size", [2, 4])
def test_dspark_packed_cp_masks_label_positions_for_counts(cp_size: int) -> None:
    config = _config().model_copy(
        update={"block_size": 3, "anchors_per_sample": 8}
    )
    provider = DSparkSpeculator(config)
    logical_length = 31
    sample_ids = torch.tensor([91], dtype=torch.int64)
    data = BatchedDataDict(
        {
            "input_ids": torch.arange(logical_length).reshape(1, logical_length),
            "input_lengths": torch.tensor([logical_length]),
            "draft_sample_ids": sample_ids,
            "token_mask": torch.ones(1, logical_length),
            "sample_mask": torch.ones(1),
        }
    )
    layout = build_draft_sequence_layout(
        logical_sample_ids=sample_ids,
        cu_seqlens_q=torch.tensor([0, logical_length], dtype=torch.int64),
        cu_seqlens_q_padded=torch.tensor([0, 32], dtype=torch.int64),
        cp_rank=0,
        cp_size=cp_size,
        tp_rank=0,
        tp_size=2,
        device=torch.device("cpu"),
    )
    plan = provider.prepare_batch(data, optimizer_step=9, sequence_layout=layout)
    assert plan.loss_mask.numel() > 0
    first_valid = torch.nonzero(plan.loss_mask, as_tuple=False)[0]
    row, slot = (int(value) for value in first_valid)
    query_position = int(plan.packed_rope_positions[row, slot])
    label_position = int(plan.packed_label_rope_positions[row, slot])
    assert query_position != label_position
    data["token_mask"][int(plan.sample_rows[row]), label_position] = 0

    expected_mask = plan.loss_mask & data["token_mask"].to(torch.bool)[
        plan.sample_rows[:, None], plan.packed_label_rope_positions
    ]

    class _Adapter:
        def objective_stats(self, *, valid_mask: Tensor, **_: object) -> SimpleNamespace:
            counts = valid_mask.sum(dim=0, dtype=torch.float32)
            return SimpleNamespace(
                combined=SimpleNamespace(
                    numerators=counts,
                    counts=counts,
                    weights=torch.ones_like(counts),
                )
            )

    data["dspark_output"] = DSparkForwardOutput(
        hidden=torch.zeros((*plan.query_positions.shape, 4)),
        plan=plan,
        output_weight=torch.zeros(8, 4),
        previous_token_ids=torch.zeros_like(plan.query_positions),
        hard_labels=torch.zeros_like(plan.label_positions),
        adapter=_Adapter(),
        sequence_layout=layout,
        selected_teacher_logits=torch.zeros((*plan.query_positions.shape, 8)),
    )
    stats = provider.loss_stats(
        target_logits=torch.zeros(1, 1, 8),
        data=data,
        prepare_fn=lambda **_: None,
        vocab_parallel_rank=0,
        vocab_parallel_group=None,
        context_parallel_group=None,
    )

    torch.testing.assert_close(stats.counts, expected_mask.sum(0, dtype=torch.float32))
    torch.testing.assert_close(
        provider.normalization_counts(
            data,
            optimizer_step=9,
            sequence_layout=layout,
        ),
        stats.counts,
    )


def test_dspark_k7_plan_trains_anchor_and_six_masks_on_seven_future_tokens() -> None:
    provider = DSparkSpeculator(_config())
    data = BatchedDataDict(
        {
            "input_ids": torch.arange(24).reshape(2, 12),
            "input_lengths": torch.tensor([12, 11]),
            "draft_sample_ids": torch.tensor([101, 303], dtype=torch.int64),
        }
    )

    plan = provider.prepare_batch(data, optimizer_step=9)

    assert plan.block_size == 7
    assert plan.query_positions.shape == (4, 7)
    torch.testing.assert_close(plan.label_positions, plan.query_positions + 1)
    assert torch.equal(plan.loss_mask, plan.slot_valid)
    assert bool(plan.loss_mask[:, 0].all())
    assert int(plan.label_positions[plan.loss_mask].max()) < 12


def test_dspark_loss_uses_sampled_rollout_tokens_as_hard_labels() -> None:
    provider = DSparkSpeculator(_config())
    input_ids = torch.arange(24).reshape(2, 12)
    data = BatchedDataDict(
        {
            "input_ids": input_ids,
            "input_lengths": torch.tensor([12, 11]),
            "draft_sample_ids": torch.tensor([101, 303], dtype=torch.int64),
        }
    )
    plan = provider.prepare_batch(data, optimizer_step=9)
    captured: dict[str, torch.Tensor] = {}

    class _Adapter:
        def objective_stats(
            self,
            *,
            hard_labels: torch.Tensor,
            **_: object,
        ) -> SimpleNamespace:
            captured["hard_labels"] = hard_labels
            bins = torch.ones(plan.block_size)
            return SimpleNamespace(
                combined=SimpleNamespace(
                    numerators=bins,
                    counts=bins,
                    weights=bins,
                )
            )

    data["dspark_output"] = DSparkForwardOutput(
        hidden=torch.zeros((*plan.query_positions.shape, 4)),
        plan=plan,
        output_weight=torch.zeros(8, 4),
        previous_token_ids=input_ids[plan.sample_rows[:, None], plan.query_positions],
        hard_labels=input_ids[plan.sample_rows[:, None], plan.label_positions],
        adapter=_Adapter(),
    )
    target_logits = torch.zeros(2, 12, 8)
    target_logits[..., 0] = 1

    provider.loss_stats(
        target_logits=target_logits,
        data=data,
        prepare_fn=lambda **_: None,
        vocab_parallel_rank=0,
        vocab_parallel_group=None,
        context_parallel_group=None,
    )

    expected = input_ids[plan.sample_rows[:, None], plan.label_positions]
    torch.testing.assert_close(captured["hard_labels"], expected)
    assert not torch.equal(
        captured["hard_labels"],
        target_logits.argmax(dim=-1)[plan.sample_rows[:, None], plan.query_positions],
    )


def test_dspark_export_uses_runtime_names_without_target_owned_weights(
    monkeypatch,
) -> None:
    provider = DSparkSpeculator(_config())
    body_weight = torch.ones(2, 2)
    markov_w1 = torch.ones(3, 2)
    markov_w2 = torch.ones(3, 2)
    confidence_weight = torch.ones(1, 4)
    confidence_bias = torch.ones(1)
    model = SimpleNamespace(
        body=object(),
        markov_head=SimpleNamespace(
            markov_w1=SimpleNamespace(weight=markov_w1),
            markov_w2=SimpleNamespace(weight=markov_w2),
            target_vocab_size=3,
            draft_vocab_size=3,
        ),
        confidence_head=SimpleNamespace(
            proj=SimpleNamespace(weight=confidence_weight, bias=confidence_bias)
        ),
    )
    monkeypatch.setattr(
        "nemo_rl.models.megatron.draft.training.export_dflash_weights_to_hf",
        lambda body: [("fc.weight", body_weight)],
    )
    monkeypatch.setattr(
        "nemo_rl.models.megatron.draft.training.export_dspark_heads_to_hf",
        lambda adapter: [
            ("markov_head.markov_w1.weight", markov_w1),
            ("markov_head.markov_w2.weight", markov_w2),
            ("confidence_head.proj.weight", confidence_weight),
            ("confidence_head.proj.bias", confidence_bias),
        ],
    )

    exported = provider.export_weights(model)  # type: ignore[arg-type]

    assert [name for name, _ in exported] == [
        "fc.weight",
        "markov_head.markov_w1.weight",
        "markov_head.markov_w2.weight",
        "confidence_head.proj.weight",
        "confidence_head.proj.bias",
    ]
    assert not any(
        name.startswith(("embed_tokens.", "lm_head.")) for name, _ in exported
    )


def test_dspark_provider_maps_packed_cp_inputs_to_local_objective_slots() -> None:
    hidden_size, vocab_size = 4, 32
    config = _config().model_copy(
        update={
            "block_size": 3,
            "anchors_per_sample": 8,
            "mask_token_id": 7,
            "target_hidden_state_layer_ids": [0, 2],
        }
    )
    provider = DSparkSpeculator(config)
    sample_ids = torch.tensor([91], dtype=torch.int64)
    logical_length, padded_length = 15, 16
    data = BatchedDataDict(
        {
            "input_ids": torch.arange(logical_length).reshape(1, logical_length),
            "input_lengths": torch.tensor([logical_length]),
            "draft_sample_ids": sample_ids,
            "token_mask": torch.ones(1, logical_length),
            "sample_mask": torch.ones(1),
        }
    )
    layout = build_draft_sequence_layout(
        logical_sample_ids=sample_ids,
        cu_seqlens_q=torch.tensor([0, logical_length], dtype=torch.int64),
        cu_seqlens_q_padded=torch.tensor([0, padded_length], dtype=torch.int64),
        cp_rank=0,
        cp_size=2,
        tp_rank=0,
        tp_size=2,
        device=torch.device("cpu"),
    )
    packed_ids = torch.zeros(padded_length, dtype=torch.int64)
    packed_ids[:logical_length] = data["input_ids"][0]
    input_ids_cp_local = packed_ids[layout.cp_global_positions].unsqueeze(0)

    class _Target(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = SimpleNamespace(
                word_embeddings=torch.nn.Embedding(vocab_size, hidden_size)
            )
            self.output_layer = torch.nn.Linear(hidden_size, vocab_size, bias=False)
            self.share_embeddings_and_output_weights = False

    class _Body(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(0.5))
            self.config = SimpleNamespace(hidden_size=hidden_size)
            self.sequence_layout = None
            self.context_parallel_group = None

        def forward(
            self,
            *,
            target_taps: Tensor,
            block_embeddings: Tensor,
            plan: object,
            sequence_layout: object | None = None,
            context_parallel_group: object | None = None,
        ) -> Tensor:
            del target_taps, plan
            self.sequence_layout = sequence_layout
            self.context_parallel_group = context_parallel_group
            return block_embeddings * self.scale

    captured_objective: dict[str, Tensor] = {}

    class _Adapter(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.body = _Body()

        def objective_stats(self, **kwargs: object) -> SimpleNamespace:
            captured_objective.update(
                {
                    name: value
                    for name, value in kwargs.items()
                    if isinstance(value, Tensor)
                }
            )
            bins = kwargs["valid_mask"].sum(dim=0, dtype=torch.float32)  # type: ignore[union-attr]
            return SimpleNamespace(
                combined=SimpleNamespace(
                    numerators=bins * self.body.scale,
                    counts=bins,
                    weights=torch.ones_like(bins),
                )
            )

    target = _Target()
    adapter = _Adapter()
    cp_group = object()
    cp_local_length = layout.cp_global_positions.numel()
    captured = CapturedStates(
        hidden_states=torch.randn(cp_local_length, 1, 2 * hidden_size),
        inputs_embeds=torch.randn(cp_local_length, 1, hidden_size),
        output_hidden=torch.randn(cp_local_length, 1, hidden_size),
        sequence_layout=layout,
        sequence_is_reconstructed=True,
    )

    provider.forward(
        policy_model=target,
        draft_model=adapter,
        captured_states=captured,
        input_ids_cp_local=input_ids_cp_local,
        attention_mask=None,
        data=data,
        optimizer_step=9,
        sequence_layout=layout,
        context_parallel_group=cp_group,
        tensor_parallel_group=object(),
    )
    output = data["dspark_output"]
    assert isinstance(output, DSparkForwardOutput)
    assert output.sequence_layout is layout
    assert adapter.body.sequence_layout is layout
    assert adapter.body.context_parallel_group is cp_group

    target_logits = torch.randn(cp_local_length // layout.tp_size, 1, vocab_size)
    stats = provider.loss_stats(
        target_logits=target_logits,
        data=data,
        prepare_fn=lambda **_: None,
        vocab_parallel_rank=0,
        vocab_parallel_group=None,
        context_parallel_group=cp_group,
    )
    plan = output.plan
    assert output.selected_teacher_logits is not None
    torch.testing.assert_close(
        captured_objective["target_logits"],
        output.selected_teacher_logits,
    )
    torch.testing.assert_close(
        captured_objective["previous_token_ids"],
        input_ids_cp_local[
            torch.zeros_like(plan.sample_rows)[:, None],
            plan.local_query_positions,
        ],
    )
    torch.testing.assert_close(
        captured_objective["hard_labels"],
        input_ids_cp_local[
            torch.zeros_like(plan.sample_rows)[:, None],
            plan.local_label_positions,
        ],
    )
    torch.testing.assert_close(
        provider.normalization_counts(
            data,
            optimizer_step=9,
            sequence_layout=layout,
        ),
        stats.counts,
    )
