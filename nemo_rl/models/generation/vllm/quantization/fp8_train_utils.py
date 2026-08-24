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


import re
from collections.abc import Callable, Iterable, Iterator

import torch

MXFP8_BLOCK_SIZE = 32
MXFP8_VALUE_DTYPE = torch.float8_e4m3fn

_EXPERT_WEIGHT_PATTERN = re.compile(
    r"^(?P<prefix>.+\.experts)\.(?P<expert_id>\d+)\."
    r"(?P<projection>gate_proj|up_proj|down_proj)\.weight$"
)


def _mxfp8_e4m3_quantize_torch(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference MXFP8 quantization with row-major scales.

    Replicates vLLM's _mxfp8_e4m3_quantize_torch (Apache-2.0,
    vllm/model_executor/layers/quantization/utils/mxfp8_utils.py) for trainer
    processes without a vLLM install: for each block of 32 elements along the
    last dimension, a shared e8m0 scale (biased exponent of the block amax)
    and float8_e4m3fn values.
    """
    assert x.shape[-1] % MXFP8_BLOCK_SIZE == 0, (
        f"MXFP8 requires the last dim to be divisible by {MXFP8_BLOCK_SIZE}, got {x.shape}"
    )
    orig_shape = x.shape
    num_blocks = x.shape[-1] // MXFP8_BLOCK_SIZE

    x_fp32 = x.to(torch.float32)
    x_blocked = x_fp32.view(*orig_shape[:-1], num_blocks, MXFP8_BLOCK_SIZE)

    amax = x_blocked.abs().amax(dim=-1)
    amax = amax.clamp(min=torch.finfo(torch.float32).tiny)
    scale_biased = torch.floor(torch.log2(amax)) + 127.0
    scale_biased = scale_biased.clamp(0, 254)
    scales_uint8 = scale_biased.to(torch.uint8)

    descale = torch.exp2(scale_biased - 127.0)
    x_scaled = x_blocked / descale.unsqueeze(-1)

    x_fp8 = x_scaled.view(orig_shape).to(MXFP8_VALUE_DTYPE)

    if x.ndim == 2:
        scales_uint8 = scales_uint8.view(x.shape[0], -1)
    elif x.ndim == 3:
        scales_uint8 = scales_uint8.view(x.shape[0], x.shape[1], -1)

    return x_fp8, scales_uint8


def mxfp8_e4m3_quantize_for_refit(
    x: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize a weight to MXFP8 on the trainer for pre-quantized refit.

    Mirrors the receiver path in quantization/fp8.py load_weights
    (mxfp8_e4m3_quantize + scale reshape) so the streamed E4M3 data and
    *_scale_from_checkpoint scales load bit-identically without receiver-side
    re-quantization. Uses the same flashinfer kernel as vLLM on Blackwell and
    the torch reference elsewhere.
    """
    x_q = x_scales = None
    if x.is_cuda and torch.cuda.get_device_capability(x.device) >= (10, 0):
        try:
            from flashinfer import mxfp8_quantize as flashinfer_mxfp8_quantize
        except ImportError as exc:
            raise RuntimeError(
                "Trainer-side MXFP8 refit prequantization on sm100+ requires "
                "FlashInfer so it matches the vLLM receiver quantization path."
            ) from exc
        else:
            x_q, x_scales = flashinfer_mxfp8_quantize(
                x, is_sf_swizzled_layout=False, alignment=32
            )
            if x_scales.ndim == 1 and x.ndim == 2:
                x_scales = x_scales.view(x.size(0), -1)
    if x_q is None or x_scales is None:
        x_q, x_scales = _mxfp8_e4m3_quantize_torch(x)
    x_scales = x_scales.reshape(*x.shape[:-1], x.shape[-1] // MXFP8_BLOCK_SIZE)
    # Match the receiver path's zero-scale clamp: an E8M0 byte of 0 (2^-127)
    # destabilizes the TRTLLM kernels, and pre-quantized tensors skip the
    # receiver-side quantize branch where the clamp normally runs.
    # pyrefly: ignore  # no-matching-overload
    x_scales = x_scales.masked_fill(x_scales == 0, 1)
    return x_q, x_scales


def iter_mxfp8_prequantized_params(
    params: Iterable[tuple[str, torch.Tensor]],
    selected_names: set[str],
    *,
    quantize_fn: Callable[
        [torch.Tensor], tuple[torch.Tensor, torch.Tensor]
    ] = mxfp8_e4m3_quantize_for_refit,
    scratch_cache: dict[tuple[torch.device, torch.dtype, int | None], torch.Tensor]
    | None = None,
    max_experts_per_batch: int = 16,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Batch expert weights while preserving the existing refit wire entries.

    Selected MoE experts with matching layer, projection, shape, dtype, and
    device are quantized together in bounded chunks. Other selected weights use
    the existing per-tensor path, and unselected weights pass through unchanged.

    Args:
        params: Exported Hugging Face parameter names and tensors.
        selected_names: Parameter names selected for MXFP8 prequantization.
        quantize_fn: MXFP8 quantization function.
        scratch_cache: Reusable stacking buffers keyed by device, dtype, and
            CUDA stream.
        max_experts_per_batch: Maximum number of experts per quantization call.

    Yields:
        Weight and scale entries accepted by the vLLM refit receiver.
    """
    if max_experts_per_batch <= 0:
        raise ValueError("max_experts_per_batch must be positive")
    if scratch_cache is None:
        scratch_cache = {}

    pending: dict[tuple[str, str], list[tuple[int, str, torch.Tensor]]] = {}
    current_prefix: str | None = None

    def quantize_one(name: str, tensor: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
        if tensor.dtype == torch.float8_e4m3fn:
            raise ValueError(
                "MXFP8 prequantization requires BF16 trainer-exported weights; "
                f"{name} is already stored as E4M3."
            )
        value, scale = quantize_fn(tensor)
        return [(name, value), (name + "_scale_from_checkpoint", scale)]

    def flush_group(
        group_key: tuple[str, str],
    ) -> Iterator[tuple[str, torch.Tensor]]:
        group = pending.pop(group_key)
        group.sort(key=lambda item: item[0])
        while group:
            chunk = group[:max_experts_per_batch]
            del group[:max_experts_per_batch]
            tensors = [tensor for _expert_id, _name, tensor in chunk]
            batchable = len(chunk) > 1 and len({item[0] for item in chunk}) == len(
                chunk
            )
            if batchable:
                first = tensors[0]
                batchable = all(
                    tensor.shape == first.shape
                    and tensor.dtype == first.dtype
                    and tensor.device == first.device
                    and tensor.layout is torch.strided
                    for tensor in tensors
                )
            if not batchable:
                for _expert_id, name, tensor in chunk:
                    yield from quantize_one(name, tensor)
                continue

            first = tensors[0]
            if first.dtype == torch.float8_e4m3fn:
                raise ValueError(
                    "MXFP8 prequantization requires BF16 trainer-exported weights."
                )
            required_numel = len(chunk) * first.numel()
            stream_id = (
                int(torch.cuda.current_stream(first.device).cuda_stream)
                if first.is_cuda
                else None
            )
            cache_key = (first.device, first.dtype, stream_id)
            scratch = scratch_cache.get(cache_key)
            if scratch is None or scratch.numel() < required_numel:
                scratch = torch.empty(
                    required_numel,
                    dtype=first.dtype,
                    device=first.device,
                )
                scratch_cache[cache_key] = scratch
            stacked = scratch[:required_numel].view(len(chunk), *first.shape)
            with torch.no_grad():
                torch.stack(tensors, dim=0, out=stacked)

            value, scale = quantize_fn(stacked.view(-1, stacked.shape[-1]))
            value = value.view_as(stacked)
            scale_columns = first.shape[-1] // MXFP8_BLOCK_SIZE
            scale_shape = (
                first.shape[:-1]
                if scale_columns == 1
                else (*first.shape[:-1], scale_columns)
            )
            scale = scale.view(len(chunk), *scale_shape)
            for index, (_expert_id, name, _tensor) in enumerate(chunk):
                yield name, value[index]
                yield name + "_scale_from_checkpoint", scale[index]

    def flush_pending() -> Iterator[tuple[str, torch.Tensor]]:
        while pending:
            yield from flush_group(next(iter(pending)))

    for name, tensor in params:
        match = _EXPERT_WEIGHT_PATTERN.match(name) if name in selected_names else None
        if match is None:
            if pending:
                yield from flush_pending()
                current_prefix = None
            if name in selected_names:
                yield from quantize_one(name, tensor)
            else:
                yield name, tensor
            continue

        prefix = match.group("prefix")
        projection = match.group("projection")
        if current_prefix is not None and prefix != current_prefix:
            yield from flush_pending()
        current_prefix = prefix
        group_key = (prefix, projection)
        group = pending.setdefault(group_key, [])
        group.append((int(match.group("expert_id")), name, tensor))
        if len(group) == max_experts_per_batch:
            yield from flush_group(group_key)

    if pending:
        yield from flush_pending()


def get_vllm_qkv_scale_names(layer_idx: int) -> dict[str, str]:
    """Get vLLM-compatible parameter names for Q/K/V FP8 scales.

    This function centralizes the naming convention for Q/K/V scale parameters
    that vLLM expects. These names must match vLLM's internal parameter structure.

    Args:
        layer_idx: The transformer layer index (0-based)

    Returns:
        Dictionary mapping scale types to vLLM parameter names:
        - 'q_scale': Q activation scale name
        - 'k_scale': K activation scale name
        - 'v_scale': V activation scale name

    Note:
        The q_scale has an extra '.attn.' component compared to k_scale/v_scale.
        This matches vLLM's parameter remapping logic in:
        vllm.model_executor.model_loader.weight_utils.maybe_remap_kv_scale_name

    Example:
        >>> get_vllm_qkv_scale_names(0)
        {
            'q_scale': 'model.layers.0.self_attn.attn.q_scale',
            'k_scale': 'model.layers.0.self_attn.k_scale',
            'v_scale': 'model.layers.0.self_attn.v_scale'
        }
    """
    return {
        "q_scale": f"model.layers.{layer_idx}.self_attn.attn.q_scale",
        "k_scale": f"model.layers.{layer_idx}.self_attn.k_scale",
        "v_scale": f"model.layers.{layer_idx}.self_attn.v_scale",
    }


def convert_calibration_to_vllm_format(
    calibration_results: dict[str, dict[str, float]],
) -> dict[str, float]:
    """Convert NeMo-RL calibration results to vLLM parameter format.

    Currently only used by megatron policy worker.
    After FP8 KV cache is supported by DTensor path, this function can be reused.

    This function transforms the calibration output format (with layer_N keys)
    into the flat dictionary format that vLLM expects for parameter loading.

    Args:
        calibration_results: Dict with keys like "layer_0", "layer_1", etc.
            Each value is a dict with keys: "q_scale", "k_scale", "v_scale"
            and corresponding float scale values.

    Returns:
        Flat dictionary mapping vLLM parameter names to scale values.
        Keys follow vLLM's naming convention as defined in get_vllm_qkv_scale_names.

    Example:
        >>> calib = {
        ...     "layer_0": {"q_scale": 1.0, "k_scale": 2.0, "v_scale": 3.0},
        ...     "layer_1": {"q_scale": 1.5, "k_scale": 2.5, "v_scale": 3.5}
        ... }
        >>> convert_calibration_to_vllm_format(calib)
        {
            'model.layers.0.self_attn.attn.q_scale': 1.0,
            'model.layers.0.self_attn.k_scale': 2.0,
            'model.layers.0.self_attn.v_scale': 3.0,
            'model.layers.1.self_attn.attn.q_scale': 1.5,
            'model.layers.1.self_attn.k_scale': 2.5,
            'model.layers.1.self_attn.v_scale': 3.5
        }
    """
    vllm_scales = {}
    for layer_key, scales in calibration_results.items():
        # Extract layer index from "layer_N" format
        layer_idx = int(layer_key.split("_")[1])
        param_names = get_vllm_qkv_scale_names(layer_idx)

        vllm_scales[param_names["q_scale"]] = scales["q_scale"]
        vllm_scales[param_names["k_scale"]] = scales["k_scale"]
        vllm_scales[param_names["v_scale"]] = scales["v_scale"]

    return vllm_scales
