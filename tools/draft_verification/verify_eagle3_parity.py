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
"""Verify the vendored EAGLE3 TTT forward against the speculators oracle.

Builds the original speculators ``Eagle3DraftModel`` and the vendored
``Qwen3Eagle3DraftModel`` with identical tiny-random weights, runs both on the
same inputs, and requires:

- The total TTT loss matches: the vendored per-step numerators/denominators,
  recombined as ``sum_k decay^k * num_k / den_k``, equal the oracle's scalar
  loss (which normalizes per step as ``masked_sum / (mask_sum + 1e-5)``).
- Per-step full/conditional accuracy counts match, which exercises the
  ``prev_correct`` in-place AND propagation across steps.
- Teacher substitution is exact: the vendored model consumes
  ``teacher_logits = verifier_lm_head(verifier_norm(hidden))`` computed
  outside, the same construction the oracle applies internally.
- Parity holds on a padded two-document packed row (the flattened form the
  Eagle3Runtime feeds for B > 1), covering the oracle's global teacher-forced
  shifts — which intentionally cross packed-document boundaries — and the
  per-document attention masking.

Both models run the HF "eager" attention path in float32 so the comparison is
numerically tight (atol 1e-4 on losses). Runs on CUDA when available (the
oracle's mask construction goes through flex-attention utilities that require
a GPU).

One oracle quirk is corrected before comparing: speculators' rarely-used dense
(eager/SDPA) fallback ``extend_dense_mask_for_draft_tokens`` appends the
per-step diagonal as 1.0/0.0 instead of additive 0/-inf, which lets every
query attend all previous-step draft KVs. Its canonical flex-attention path
(``extend_mask_for_draft_tokens``, the mode real speculators training runs)
appends a STRICT diagonal block, as does vLLM; this script patches the dense
extension to those canonical semantics, which is also what the vendored code
implements.

Usage: uv run --no-sync python tools/draft_verification/verify_eagle3_parity.py
"""

import argparse
import sys
from pathlib import Path

import torch

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

_SCRIPT_DIR = Path(__file__).resolve().parent
# The repo checkout takes precedence over any container-installed nemo_rl.
sys.path.insert(0, str(_SCRIPT_DIR.parents[1]))
sys.path.insert(0, str(_SCRIPT_DIR))
from _speculators_oracle import DEFAULT_SPECULATORS_PATH, bootstrap_speculators

TARGET_VOCAB = 200
DRAFT_VOCAB = 80
HIDDEN = 64
NUM_AUX = 3
TTT_STEPS = 3


def _tiny_layer_config(model_type: str):
    from transformers.models.llama.configuration_llama import LlamaConfig
    from transformers.models.qwen3.configuration_qwen3 import Qwen3Config

    config_class = {"qwen3": Qwen3Config, "llama": LlamaConfig}[model_type]
    config = config_class(
        hidden_size=HIDDEN,
        intermediate_size=128,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=TARGET_VOCAB,
        max_position_embeddings=512,
        rms_norm_eps=1e-6,
        attention_bias=False,
    )
    config._attn_implementation = "eager"
    return config


def build_models(seed: int, model_type: str):
    from speculators.config import SpeculatorsConfig, VerifierConfig
    from speculators.models.eagle3.core import Eagle3DraftModel
    from speculators.proposals.greedy import GreedyTokenProposalConfig

    from nemo_rl.models.automodel.draft.eagle3_qwen3 import Qwen3Eagle3DraftModel

    torch.manual_seed(seed)
    oracle_config = Eagle3DraftModel.config_class(
        transformer_layer_config=_tiny_layer_config(model_type),
        draft_vocab_size=DRAFT_VOCAB,
        norm_before_residual=True,
        eagle_aux_hidden_state_layer_ids=[2, 4, 6],
        speculators_config=SpeculatorsConfig(
            algorithm="eagle3",
            proposal_methods=[GreedyTokenProposalConfig()],
            default_proposal_method="greedy",
            verifier=VerifierConfig(name_or_path="tiny", architectures=[]),
        ),
    )
    oracle = Eagle3DraftModel(oracle_config).to(DEVICE).eval()

    # The oracle nan-fills embed/lm_head/verifier heads until a checkpoint
    # load; give everything small random values instead.
    with torch.no_grad():
        for param in oracle.parameters():
            param.normal_(0.0, 0.05)
        for buffer_name in ("verifier_norm", "norm"):
            getattr(oracle, buffer_name).weight.fill_(1.0)
        # d2t offsets: draft index i maps to a strictly increasing target id.
        target_ids = torch.sort(torch.randperm(TARGET_VOCAB)[:DRAFT_VOCAB]).values
        oracle.d2t.copy_((target_ids - torch.arange(DRAFT_VOCAB)).to(DEVICE))
        oracle.t2d.zero_()
        oracle.t2d[target_ids] = True

    vendored_config = _tiny_layer_config(model_type)
    vendored_config.draft_vocab_size = DRAFT_VOCAB
    vendored_config.norm_before_residual = True
    # Trainer output-of-layer convention: speculators/vLLM aux ids minus 1.
    vendored_config.target_layer_ids = [1, 3, 5]
    vendored = Qwen3Eagle3DraftModel(vendored_config).to(DEVICE).eval()

    oracle_state = {
        name: tensor
        for name, tensor in oracle.state_dict().items()
        if not name.startswith(("verifier_norm", "verifier_lm_head"))
    }
    missing, unexpected = vendored.load_state_dict(oracle_state, strict=True), None
    assert not missing.missing_keys and not missing.unexpected_keys, (
        missing,
        unexpected,
    )
    return oracle, vendored


def make_inputs(seed: int, two_documents: bool):
    torch.manual_seed(1000 + seed)
    total = 48
    input_ids = torch.randint(0, TARGET_VOCAB, (1, total))
    if two_documents:
        # Two packed documents plus right padding: the flattened layout the
        # Eagle3Runtime builds for B > 1 (document_ids = batch index, -1 pad,
        # per-document position ids).
        lengths = (22, 20)
        pad = total - sum(lengths)
        document_ids = torch.tensor([[0] * lengths[0] + [1] * lengths[1] + [-1] * pad])
        position_ids = torch.cat(
            [
                1 + torch.arange(lengths[0]),
                1 + torch.arange(lengths[1]),
                torch.ones(pad, dtype=torch.long),
            ]
        ).unsqueeze(0)
        loss_mask = torch.zeros(1, total, dtype=torch.bool)
        loss_mask[:, 8 : lengths[0]] = True
        loss_mask[:, lengths[0] + 6 : lengths[0] + lengths[1]] = True
    else:
        document_ids = torch.zeros(1, total, dtype=torch.long)
        position_ids = None
        loss_mask = torch.zeros(1, total, dtype=torch.bool)
        loss_mask[:, 10:] = True
    fused_hidden = torch.randn(1, total, NUM_AUX * HIDDEN)
    verifier_hidden = torch.randn(1, total, HIDDEN)
    tensors = (
        input_ids,
        document_ids,
        position_ids,
        loss_mask,
        fused_hidden,
        verifier_hidden,
    )
    return tuple(t.to(DEVICE) if t is not None else None for t in tensors)


def run_case(seed: int, two_documents: bool, decay: float, model_type: str) -> None:
    oracle, vendored = build_models(seed, model_type)
    (
        input_ids,
        document_ids,
        position_ids,
        loss_mask,
        fused_hidden,
        verifier_hidden,
    ) = make_inputs(seed, two_documents)

    _, oracle_loss, oracle_metrics = oracle(
        hidden_states=fused_hidden.clone(),
        input_ids=input_ids.clone(),
        document_ids=document_ids,
        loss_mask=loss_mask.clone(),
        position_ids=None if position_ids is None else position_ids.clone(),
        verifier_last_hidden_states=verifier_hidden,
        ttt_steps=TTT_STEPS,
        ttt_step_loss_decay=decay,
    )

    with torch.no_grad():
        teacher_logits = oracle.verifier_lm_head(oracle.verifier_norm(verifier_hidden))
    terms = vendored(
        fused_hidden_states=fused_hidden.clone(),
        input_ids=input_ids.clone(),
        document_ids=document_ids,
        loss_mask=loss_mask.clone(),
        teacher_logits=teacher_logits,
        position_ids=None if position_ids is None else position_ids.clone(),
        ttt_steps=TTT_STEPS,
    )

    vendored_loss = sum(
        (decay**step) * num / den
        for step, (num, den) in enumerate(zip(terms.loss_nums, terms.loss_dens))
    )
    assert torch.allclose(vendored_loss, oracle_loss, atol=1e-4), (
        f"loss diverges: vendored {vendored_loss.item():.6f} vs "
        f"oracle {oracle_loss.item():.6f}"
    )

    for step in range(TTT_STEPS):
        pairs = [
            (terms.full_acc_nums[step], oracle_metrics[f"full_acc_{step}_sum"]),
            (terms.full_acc_dens[step], oracle_metrics[f"full_acc_{step}_total"]),
            (terms.cond_acc_nums[step], oracle_metrics[f"cond_acc_{step}_sum"]),
            (terms.cond_acc_dens[step], oracle_metrics[f"cond_acc_{step}_total"]),
        ]
        for got, want in pairs:
            assert torch.allclose(got.float(), want.float(), atol=1e-6), (
                f"accuracy counts diverge at ttt_step {step}: "
                f"{[float(g) for g, _ in pairs]} vs {[float(w) for _, w in pairs]}"
            )

    layout = "two-document packed row" if two_documents else "single document"
    print(
        f"[ok] seed {seed} ({model_type} layers, {layout}, decay {decay}): loss "
        f"{vendored_loss.item():.6f} and per-step accuracy counts match"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--speculators-path", default=DEFAULT_SPECULATORS_PATH)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    args = parser.parse_args()

    bootstrap_speculators(args.speculators_path)

    # Align the oracle's dense-mask fallback with its canonical flex path
    # (strict additive 0/-inf diagonal extension); see the module docstring.
    from speculators.models.eagle3 import core as oracle_core

    def _canonical_dense_extension(mask: torch.Tensor, total_seq_len: int):
        idx = torch.arange(total_seq_len, device=mask.device)
        diag = idx.unsqueeze(1) == idx.unsqueeze(0)
        block = torch.zeros(
            (1, 1, total_seq_len, total_seq_len), dtype=mask.dtype, device=mask.device
        )
        block.masked_fill_(~diag.unsqueeze(0).unsqueeze(0), float("-inf"))
        return torch.cat([mask, block], dim=-1)

    oracle_core.extend_dense_mask_for_draft_tokens = _canonical_dense_extension

    torch.set_default_dtype(torch.float32)
    # llama is the layer family of the published qwen3-target checkpoints
    # (RedHatAI/Qwen3-8B-speculator.eagle3); qwen3 covers q/k-norm layers.
    for model_type in ("llama", "qwen3"):
        for seed in args.seeds:
            run_case(seed, two_documents=False, decay=1.0, model_type=model_type)
            run_case(seed, two_documents=False, decay=0.8, model_type=model_type)
            run_case(seed, two_documents=True, decay=1.0, model_type=model_type)
    print("PASS: vendored eagle3 TTT forward matches the speculators oracle")


if __name__ == "__main__":
    main()
