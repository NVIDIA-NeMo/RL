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

"""Temporarily fold fake-quantized weights for frozen-weight QAT stages.

In ModelOpt QAT the weight quantizer sits inside the linear, so every forward recomputes
``weight_quantizer(weight)`` — an elementwise pass over the weight shard, uncached.
During the ``get_logprobs`` re-scoring stage the weights are frozen (``torch.no_grad()``,
no optimizer step between microbatches), so that result is identical across every
microbatch and is pure wasted work.

ModelOpt already ships the primitive: :func:`modelopt.torch.quantization.fold_weight`
writes the fake-quantized value into the weight and disables the weight quantizer, which
is exactly the frozen-weight steady state. This module only makes that fold *reversible*,
so it can wrap a single stage of an otherwise-continuing QAT run.
"""

import contextlib
from collections.abc import Iterator

import torch

import modelopt.torch.quantization as mtq
from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer

_QUANTIZER_SUFFIX = "weight_quantizer"


def _foldable_weight_quantizers(
    model: torch.nn.Module,
) -> Iterator[tuple[torch.Tensor, TensorQuantizer]]:
    """Yield the ``(weight, quantizer)`` pairs that ``fold_weight`` will actually fold.

    Mirrors ``QuantModule.fold_weight``'s own discovery — any attribute whose name ends in
    ``weight_quantizer`` holding a ``fake_quant`` ``TensorQuantizer``, paired with the
    weight named by dropping the ``_quantizer`` suffix. Matching it exactly matters for
    fused and MoE modules, which expose names like ``w13_weight_quantizer`` and
    ``gate_up_proj_weight_quantizer``: a plain ``module.weight_quantizer`` lookup would
    miss those, leaving them folded and disabled after the stage.

    ``SequentialQuantizer`` weight quantizers (W4A4 double-quant) are skipped, because
    ``fold_weight`` skips them too — it subclasses ``nn.Sequential``, not
    ``TensorQuantizer`` — so there is nothing to fold or restore for those modules.
    """
    for module in model.modules():
        for name in dir(module):
            if not name.endswith(_QUANTIZER_SUFFIX):
                continue
            quantizer = getattr(module, name, None)
            if not isinstance(quantizer, TensorQuantizer) or not quantizer.fake_quant:
                continue
            weight = getattr(module, name[: -len("_quantizer")], None)
            if isinstance(weight, torch.Tensor):
                yield weight, quantizer


@contextlib.contextmanager
def temporarily_fold_weights(
    model: torch.nn.Module,
    *,
    verbose: bool = False,
    rank: int = 0,
) -> Iterator[None]:
    """Fold fake-quantized weights into the parameters for the duration of the block.

    Wraps :func:`modelopt.torch.quantization.fold_weight`, which snaps each weight into
    its existing parameter storage and disables the weight quantizer, then restores the
    original weights and re-enables the quantizers on exit. Forwards inside the block read
    an already-snapped weight and short-circuit out of the disabled quantizer, so the snap
    happens once per stage instead of once per microbatch.

    ``keep_attrs=True`` is required: without it ``fold_weight`` also deletes the
    quantizers' ``_amax`` / ``_pre_quant_scale`` (and SVDQuant LoRA) buffers, and
    re-enabling a quantizer whose calibration state has been dropped would silently change
    quantization once training resumes.

    Only valid while the weights are frozen — ``no_grad`` with no optimizer step — which
    is exactly the ``get_logprobs`` re-scoring stage. Folding across a live training step
    would corrupt the weights, since the fold is written into the parameter itself.

    Activation quantization is unaffected: ``fold_weight`` disables only the *weight*
    quantizer, so a recipe's ``input_quantizer`` / ``output_quantizer`` keep running and
    W4A4 logprobs are unchanged.

    Costs one temporary copy of each folded weight shard, held only for the block.
    """
    folded = list(_foldable_weight_quantizers(model))
    original_weights = [(weight, weight.detach().clone()) for weight, _ in folded]
    enabled_quantizers = [quantizer for _, quantizer in folded if quantizer.is_enabled]

    try:
        # fold_weight writes through `.data`, but running it under grad would still build
        # a throwaway graph for the snap itself.
        with torch.no_grad():
            mtq.fold_weight(model, keep_attrs=True)
        yield
    finally:
        with torch.no_grad():
            for weight, original in original_weights:
                weight.data.copy_(original)
        for quantizer in enabled_quantizers:
            quantizer.enable()
    if verbose and rank == 0:
        print(
            f"[weight_folding] frozen-weight stage: folded and restored "
            f"{len(folded)} weight quantizer(s)."
        )
