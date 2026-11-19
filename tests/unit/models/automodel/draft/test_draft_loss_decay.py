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
"""Positional loss decay counts proposals, not raw block slots.

The dspark (next-token) layout proposes from slot 0; the dflash bonus-anchor
layout proposes from slot 1 (slot 0 is the unsupervised anchor). With
``loss_decay_gamma`` set, the k-th PROPOSAL must get weight ``exp(-k/gamma)``
in both layouts so the first proposed token is never down-weighted.
"""

import math

import pytest
import torch

# Skip entire module if nemo_automodel is not available
try:
    import nemo_automodel  # noqa: F401
except ImportError:
    pytest.skip("nemo_automodel not available", allow_module_level=True)

from nemo_rl.models.automodel.draft.loss import _build_loss_weight_mask

pytestmark = pytest.mark.automodel

BLOCK_SIZE = 4
GAMMA = 4.0


def _weights(first_supervised_slot: int, eval_mask: torch.Tensor) -> torch.Tensor:
    return _build_loss_weight_mask(
        eval_mask=eval_mask,
        block_size=BLOCK_SIZE,
        device=torch.device("cpu"),
        loss_decay_gamma=GAMMA,
        first_supervised_slot=first_supervised_slot,
    )


def test_dspark_layout_decays_from_slot_zero():
    eval_mask = torch.ones(1, 1, BLOCK_SIZE, dtype=torch.bool)
    weights = _weights(0, eval_mask)[0, 0]
    expected = torch.tensor([math.exp(-k / GAMMA) for k in range(BLOCK_SIZE)])
    assert torch.allclose(weights, expected)


def test_dflash_layout_first_proposal_gets_full_weight():
    # Bonus-anchor layout: slot 0 is never supervised.
    eval_mask = torch.tensor([[[False, True, True, True]]])
    weights = _weights(1, eval_mask)[0, 0]
    expected = torch.tensor(
        [0.0] + [math.exp(-k / GAMMA) for k in range(BLOCK_SIZE - 1)]
    )
    assert torch.allclose(weights, expected)
    assert weights[1] == 1.0, "the first proposed token must not be down-weighted"


def _diagnostics(
    first_supervised_slot: int, eval_mask: torch.Tensor, accept: torch.Tensor
):
    from nemo_rl.models.automodel.draft.common import DSparkForwardOutput
    from nemo_rl.models.automodel.draft.loss import _collect_acceptance_diagnostics

    bsz, blocks, block_size = eval_mask.shape
    outputs = DSparkForwardOutput(
        draft_logits=torch.zeros(bsz, blocks, block_size, 8),
        target_ids=torch.zeros(bsz, blocks, block_size, dtype=torch.long),
        eval_mask=eval_mask,
        block_keep_mask=torch.ones(bsz, blocks, dtype=torch.bool),
        first_supervised_slot=first_supervised_slot,
    )
    return _collect_acceptance_diagnostics(
        outputs=outputs,
        accept_rate_3d=accept,
        loss_weight_mask=eval_mask.to(torch.float32),
        has_confidence=False,
    )


def test_dflash_diagnostics_skip_bonus_anchor_slot():
    """With the bonus-anchor layout the always-false slot 0 must not zero the
    tau prefix product or shift the per-proposal acceptance rates."""
    accept = torch.tensor([[[0.0, 0.8, 0.5, 0.25]]])
    eval_mask = torch.tensor([[[False, True, True, True]]])
    terms = _diagnostics(1, eval_mask, accept)

    # accept_rate@k indexes proposals: [0.8, 0.5, 0.25], not shifted by the
    # dead anchor column.
    assert torch.allclose(terms["accept_rate_pos_num"], torch.tensor([0.8, 0.5, 0.25]))
    assert torch.allclose(terms["accept_rate_pos_den"], torch.tensor([1.0, 1.0, 1.0]))

    # tau = 1 (verified seed token) + 0.8 + 0.8*0.5 + 0.8*0.5*0.25 = 2.3,
    # NOT the constant 1 a slot-0 zero in the cumprod would force.
    assert torch.allclose(terms["tau_num"], torch.tensor(2.3))
    assert terms["tau_den"].item() == 1.0


def test_dspark_diagnostics_unchanged_by_slot_offset_zero():
    accept = torch.tensor([[[0.8, 0.5, 0.25, 0.1]]])
    eval_mask = torch.ones(1, 1, 4, dtype=torch.bool)
    terms = _diagnostics(0, eval_mask, accept)
    assert torch.allclose(
        terms["accept_rate_pos_num"], torch.tensor([0.8, 0.5, 0.25, 0.1])
    )
    expected_tau = 1.0 + 0.8 + 0.8 * 0.5 + 0.8 * 0.5 * 0.25 + 0.8 * 0.5 * 0.25 * 0.1
    assert torch.allclose(terms["tau_num"], torch.tensor(expected_tau))


def test_dflash_forward_reports_first_supervised_slot():
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    from nemo_rl.models.automodel.draft.draft_qwen3 import Qwen3DSparkModel

    config = Qwen3Config(
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        vocab_size=128,
        max_position_embeddings=256,
        rms_norm_eps=1e-6,
        attention_bias=False,
    )
    config._attn_implementation = "sdpa"
    config.block_size = BLOCK_SIZE
    config.mask_token_id = 127
    config.target_layer_ids = [-1, 0]
    config.num_anchors = 4
    config.markov_rank = 0
    config.enable_confidence_head = False

    torch.manual_seed(0)
    seq_len = 32
    inputs = dict(
        input_ids=torch.randint(0, 128, (1, seq_len)),
        target_hidden_states=torch.randn(1, seq_len, 2 * 32),
        loss_mask=torch.ones(1, seq_len),
        teacher_logits=torch.randn(1, seq_len, 128),
    )

    for sample_from_anchor, expected_slot in ((True, 0), (False, 1)):
        config.sample_from_anchor = sample_from_anchor
        model = Qwen3DSparkModel(config).eval()
        with torch.no_grad():
            out = model(**inputs)
        assert out.first_supervised_slot == expected_slot
