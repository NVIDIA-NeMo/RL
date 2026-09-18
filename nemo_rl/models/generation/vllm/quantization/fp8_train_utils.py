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
