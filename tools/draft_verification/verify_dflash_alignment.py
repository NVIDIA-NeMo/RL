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
"""Verify the vendored dflash block supervision against the speculators oracle.

Runs the vendored ``Qwen3DSparkModel`` in the dflash layout
(``sample_from_anchor=False``, block_size = slot count) on inputs designed to
expose its label / teacher / mask / position indexing, and checks each against
the speculators dflash utilities (``get_base_indices_for_anchored_blocks``,
``select_anchors``) and the conventions of ``DFlashDraftModel.forward``:

- Block slots sit at positions ``anchor .. anchor + block_size - 1``.
- Slot k (k >= 1) is supervised on the token AT its own position
  (label = anchor + k), distilled from the teacher distribution at
  ``label - 1`` — matching vLLM's dflash speculator (``sample_pos =
  query_pos``). Slot 0 (the anchor / bonus token) is never supervised.
- With a contiguous response loss mask, the vendored per-slot supervision mask
  equals the speculators ``aligned_loss_mask`` (anchor slot zeroed).

Documented, intentional divergences (reported, not asserted):
- Anchor sampling policy: speculators ``select_anchors`` gates on the anchor's
  OWN loss-mask bit and excludes the last block_size positions; the vendored
  sampler gates on the anchor's first target (anchor + 1), which keeps the
  prompt-to-response boundary anchor — where drafting actually starts in RL.
- The vendored mask applies a contiguous-prefix cumprod inside each block;
  with RL's contiguous response masks the two are identical.

Also re-checks the dspark (next-token) layout as a regression guard.

Usage: uv run --no-sync python tools/draft_verification/verify_dflash_alignment.py
"""

import argparse
import sys
from pathlib import Path

import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
# The repo checkout takes precedence over any container-installed nemo_rl.
sys.path.insert(0, str(_SCRIPT_DIR.parents[1]))
sys.path.insert(0, str(_SCRIPT_DIR))
from _speculators_oracle import DEFAULT_SPECULATORS_PATH, bootstrap_speculators


def build_tiny_model(sample_from_anchor: bool, block_size: int, num_anchors: int):
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    from nemo_rl.models.automodel.draft.draft_qwen3 import Qwen3DSparkModel

    config = Qwen3Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=512,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        attention_bias=False,
    )
    config.block_size = block_size
    config.sample_from_anchor = sample_from_anchor
    config.mask_token_id = 511
    config.target_layer_ids = [-1, 0, 1]
    config.num_anchors = num_anchors
    config.markov_rank = 0
    config.enable_confidence_head = False
    config._attn_implementation = "sdpa"
    return Qwen3DSparkModel(config)


def run_case(seed: int, speculators_repo: str) -> None:
    from speculators.models.dflash.utils import (
        get_base_indices_for_anchored_blocks,
        select_anchors,
    )

    from nemo_rl.models.automodel.draft.common import create_position_ids

    torch.manual_seed(seed)
    block_size = 8
    seq_len = 64
    model = build_tiny_model(
        sample_from_anchor=False, block_size=block_size, num_anchors=12
    ).eval()

    # Distinct token ids make the gathered labels reveal their indices; an
    # identity-coded teacher makes the gathered teacher reveal its position.
    input_ids = torch.arange(seq_len).unsqueeze(0)
    response_start = 20
    loss_mask = torch.zeros(1, seq_len)
    loss_mask[:, response_start:] = 1.0
    teacher_logits = 10.0 * torch.eye(seq_len, 512).unsqueeze(0)
    hidden = torch.randn(1, seq_len, 3 * 64)

    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            target_hidden_states=hidden,
            loss_mask=loss_mask,
            teacher_logits=teacher_logits,
        )

    keep = out.block_keep_mask[0]
    assert keep.any(), "no valid anchors sampled"
    # dflash slot 0's label is the anchor's own token, so target_ids[..., 0]
    # recovers the sampled anchor positions.
    anchors = out.target_ids[0, keep, 0]

    # 1) Slot positions: the vendored draft block positions must equal the
    # speculators base indices (anchor .. anchor + block_size - 1).
    vendored_pos = create_position_ids(anchors.unsqueeze(0), block_size)[0]
    oracle_pos = get_base_indices_for_anchored_blocks(anchors.unsqueeze(0), block_size)
    assert torch.equal(vendored_pos, oracle_pos), (
        f"slot positions diverge: {vendored_pos} vs {oracle_pos}"
    )

    oracle_base = oracle_pos.view(-1, block_size)

    # 2) Labels: slot k supervised on the token AT its own position.
    labels = out.target_ids[0, keep]
    in_range = oracle_base < seq_len
    assert torch.equal(labels[in_range], oracle_base[in_range]), (
        "dflash labels must be the token at each slot's own position"
    )

    # 3) Teacher: distilled from the distribution at label - 1 (the
    # speculators sample_from_anchor=False target index, base - 1).
    teacher_pos = out.aligned_target_logits[0, keep].argmax(dim=-1)
    expected_teacher = (oracle_base - 1).clamp(min=0)
    supervised = out.eval_mask[0, keep]
    assert torch.equal(teacher_pos[supervised], expected_teacher[supervised]), (
        "dflash teacher must come from position label - 1"
    )

    # 4) Anchor slot never supervised.
    assert not out.eval_mask[0, :, 0].any(), "anchor (bonus) slot must carry no loss"

    # 5) Supervision mask equals the speculators aligned_loss_mask recipe
    # (loss_mask gathered at each slot's own position, slot 0 zeroed) for
    # contiguous response masks, restricted to in-range slots.
    oracle_mask = loss_mask[0][oracle_base.clamp(max=seq_len - 1)].bool() & in_range
    oracle_mask[:, 0] = False
    assert torch.equal(supervised, oracle_mask), (
        "dflash supervision mask diverges from the speculators aligned_loss_mask"
    )

    # 6) Documented anchor-policy divergence (informational).
    oracle_anchors, oracle_valid = select_anchors(loss_mask, 512, block_size)
    oracle_candidates = set(oracle_anchors[oracle_valid].tolist())
    vendored_anchor_set = set(anchors.tolist())
    boundary_only = vendored_anchor_set - oracle_candidates
    print(
        "[info] anchor policy divergence (documented): vendored anchors outside "
        f"the speculators candidate set: {sorted(boundary_only)} "
        f"(speculators gates on the anchor's own mask bit and drops the last "
        f"{block_size} positions; the vendored sampler gates on anchor + 1)"
    )
    for a in boundary_only:
        assert (
            loss_mask[0, a] == 0 and a + 1 < seq_len and loss_mask[0, a + 1] == 1
        ) or a >= seq_len - block_size, (
            f"anchor {a} differs for a reason other than the documented policy"
        )


def run_dspark_regression(seed: int) -> None:
    """The default next-token layout must be unchanged by the dflash mode."""
    torch.manual_seed(seed)
    block_size = 7
    seq_len = 64
    model = build_tiny_model(
        sample_from_anchor=True, block_size=block_size, num_anchors=12
    ).eval()
    input_ids = torch.arange(seq_len).unsqueeze(0)
    loss_mask = torch.zeros(1, seq_len)
    loss_mask[:, 20:] = 1.0
    teacher_logits = 10.0 * torch.eye(seq_len, 512).unsqueeze(0)
    with torch.no_grad():
        out = model(
            input_ids=input_ids,
            target_hidden_states=torch.randn(1, seq_len, 3 * 64),
            loss_mask=loss_mask,
            teacher_logits=teacher_logits,
        )
    keep = out.block_keep_mask[0]
    # dspark slot 0 teacher sits at the anchor itself; labels are anchor+1+k.
    anchors = out.aligned_target_logits[0, keep, 0].argmax(dim=-1)
    offsets = torch.arange(1, block_size + 1)
    expected_labels = anchors.unsqueeze(-1) + offsets
    supervised = out.eval_mask[0, keep]
    assert torch.equal(
        out.target_ids[0, keep][supervised], expected_labels[supervised]
    ), "dspark labels must remain next-token (anchor + k + 1)"
    assert out.eval_mask[0, keep, 0].any(), "dspark anchor slot must remain supervised"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speculators-path", default=DEFAULT_SPECULATORS_PATH)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3])
    args = parser.parse_args()

    bootstrap_speculators(args.speculators_path)
    for seed in args.seeds:
        run_case(seed, args.speculators_path)
        run_dspark_regression(seed)
        print(f"[ok] seed {seed}: dflash alignment + dspark regression checks passed")
    print("PASS: vendored dflash supervision matches the speculators/vLLM layout")


if __name__ == "__main__":
    main()
