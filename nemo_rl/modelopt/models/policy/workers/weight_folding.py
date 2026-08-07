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

Folding writes the fake-quantized value into the weight and disables the weight
quantizer — exactly the frozen-weight steady state. ModelOpt ships this as
:func:`modelopt.torch.quantization.fold_weight`, and this module applies the same
per-weight formula (``quant_module.py::QuantModule.fold_weight``), but folds each
discovered pair directly instead of delegating to the utility, for two reasons:

* ``fold_weight`` selects on ``fake_quant`` alone and dereferences ``weight.data``
  unconditionally, so it crashes with ``AttributeError`` on Megatron models with tied
  embeddings, where the ``output_layer`` exposes a ``weight_quantizer`` but its
  ``weight`` is ``None`` (the embedding weight is borrowed at forward time).
* Folding a *disabled* quantizer is an identity no-op (a disabled quantizer returns
  its input unchanged), so cloning those weights for restore would only waste memory —
  on standard QARL recipes the disabled ``lm_head``/embedding quantizers account for
  ~40% of the quantized-weight bytes.

The fold is reversible: original weights are cloned before folding and written back
through their existing parameter storage on exit, so it can wrap a single stage of an
otherwise-continuing QAT run.
"""

import contextlib
from collections.abc import Iterator

import torch

from modelopt.torch.quantization.nn.modules.tensor_quantizer import TensorQuantizer

_QUANTIZER_SUFFIX = "weight_quantizer"


def _foldable_weight_quantizers(
    model: torch.nn.Module,
) -> Iterator[tuple[torch.Tensor, TensorQuantizer]]:
    """Yield the enabled ``(weight, quantizer)`` pairs whose fold changes the forward.

    Mirrors ``QuantModule.fold_weight``'s discovery — any attribute whose name ends in
    ``weight_quantizer`` holding a ``fake_quant`` ``TensorQuantizer``, paired with the
    weight named by dropping the ``_quantizer`` suffix. Matching the suffix scan matters
    for fused and MoE modules, which expose names like ``w13_weight_quantizer`` and
    ``gate_up_proj_weight_quantizer``: a plain ``module.weight_quantizer`` lookup would
    miss those, leaving them folded and disabled after the stage.

    Beyond the upstream scan, pairs are skipped when:

    * the quantizer is disabled — its forward is the identity, so folding it is a no-op
      that would only cost a wasted restore clone (recipes routinely disable
      ``lm_head``/embedding quantizers, which hold ~40% of the quantized-weight bytes);
    * the weight is not a tensor — Megatron tied-embedding ``output_layer`` modules
      carry ``weight = None`` and borrow the embedding weight at forward time
      (upstream ``fold_weight`` crashes on these);
    * the quantizer is a ``SequentialQuantizer`` (W4A4 double-quant) — it subclasses
      ``nn.Sequential``, not ``TensorQuantizer``, and upstream skips it too.
    """
    for module in model.modules():
        for name in dir(module):
            if not name.endswith(_QUANTIZER_SUFFIX):
                continue
            quantizer = getattr(module, name, None)
            if (
                not isinstance(quantizer, TensorQuantizer)
                or not quantizer.fake_quant
                or not quantizer.is_enabled
            ):
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

    Applies ModelOpt's fold formula (``quantizer(weight.float()).to(weight.dtype)``,
    from ``QuantModule.fold_weight``) to each enabled weight quantizer, snapping the
    quantized value into the existing parameter storage and disabling the quantizer,
    then restores the original weights and re-enables the quantizers on exit. Forwards
    inside the block read an already-snapped weight and short-circuit out of the
    disabled quantizer, so the snap happens once per stage instead of once per
    microbatch. Calibration state (``_amax`` / ``_pre_quant_scale``) is never touched.

    Only valid while the weights are frozen — ``no_grad`` with no optimizer step —
    which is exactly the ``get_logprobs`` re-scoring stage. Folding across a live
    training step would corrupt the weights, since the fold is written into the
    parameter itself.

    Activation quantization is unaffected: only *weight* quantizers are disabled, so a
    recipe's ``input_quantizer`` / ``output_quantizer`` keep running and W4A4 logprobs
    are unchanged.

    Costs one temporary copy of each folded weight shard, held only for the block.
    """
    folded = list(_foldable_weight_quantizers(model))
    original_weights = [(weight, weight.detach().clone()) for weight, _ in folded]

    try:
        with torch.no_grad():
            for weight, quantizer in folded:
                # Exact upstream fold formula (quant_module.py::fold_weight).
                weight.data.copy_(quantizer(weight.float()).to(weight.dtype))
                quantizer.disable()
        yield
    finally:
        with torch.no_grad():
            for weight, original in original_weights:
                weight.data.copy_(original)
        for _, quantizer in folded:
            quantizer.enable()
    if verbose and rank == 0:
        print(
            f"[weight_folding] frozen-weight stage: folded and restored "
            f"{len(folded)} weight quantizer(s)."
        )
