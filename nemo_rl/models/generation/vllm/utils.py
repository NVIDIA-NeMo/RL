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

from collections import defaultdict
import math
from typing import Any, Optional

import torch

from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.models.generation.interfaces import GenerationDatumSpec

R3_MISSING_ROUTE_SENTINEL = -1


def format_prompt_for_vllm_generation(
    data: BatchedDataDict[GenerationDatumSpec], sample_idx: Optional[int] = None
) -> list[dict[str, Any]]:
    """Format a list of prompts for vllm generation (which requires a specific format for its own `generate` method).

    See https://docs.vllm.ai/en/v0.9.1/features/multimodal_inputs.html for prompt format for multimodal inputs.
    """
    # Prepare prompts for vLLM (removing padding)
    prompts = []

    input_ids = data["input_ids"]
    batch_size = input_ids.shape[0]
    input_lengths = data["input_lengths"]

    # if sample_idx is None, return list of all prompts for the entire batch
    # else, return the prompt for the single sample specified by sample_idx
    return_all = sample_idx is None
    if sample_idx is None:
        start_idx = 0
        end_idx = batch_size
    else:
        start_idx = sample_idx
        end_idx = sample_idx + 1

    def _get_regular_prompt(index: int):
        valid_length = input_lengths[index].item()
        valid_ids = (
            input_ids[index, :valid_length]
            if valid_length > 0
            else input_ids[index, :0]
        )
        token_ids = valid_ids.tolist()
        return {"prompt_token_ids": token_ids}

    # Check if this is VLM generation by looking for message_log with images
    # Support for videos/audio/etc. can be added here
    # if 'message_log' in data and any('images' in msg for msg in data['message_log']):
    if "vllm_content" in data:
        # VLM generation using content and multi_modal_data
        for i in range(start_idx, end_idx):
            msg = data["vllm_content"][i]
            # if msg is None, this conversation had no multimodal content, fallback to regular prompt
            if msg is None:
                prompts.append(_get_regular_prompt(i))
                continue
            # init prompt dict
            prompt_dict = {"prompt": msg}
            # collect multi_modal_data from images, audios, and videos
            multi_modal_data = {}
            images = data.get("vllm_images", None)
            if images is not None and len(images[i]) > 0:
                multi_modal_data["image"] = (
                    images[i][0] if len(images[i]) == 1 else images[i]
                )
            audios = data.get("vllm_audios", None)
            if audios is not None and len(audios[i]) > 0:
                multi_modal_data["audio"] = (
                    audios[i][0] if len(audios[i]) == 1 else audios[i]
                )
            videos = data.get("vllm_videos", None)
            if videos is not None and len(videos[i]) > 0:
                multi_modal_data["video"] = (
                    videos[i][0] if len(videos[i]) == 1 else videos[i]
                )
            if not multi_modal_data:
                prompts.append(_get_regular_prompt(i))
                continue
            prompt_dict["multi_modal_data"] = multi_modal_data
            prompts.append(prompt_dict)
    else:
        # Regular LLM generation using token_ids (pre-tokenized).
        # Note: eval.py uses raw prompt strings instead of token IDs because its
        # collate function produces message_log dicts, not tokenized tensors.
        # Both are valid vLLM input formats but may tokenize slightly differently.
        for i in range(start_idx, end_idx):
            # Use input_lengths to get only valid tokens (not padding)
            prompts.append(_get_regular_prompt(i))

    return prompts if return_all else prompts[0]


def pad_and_align_routed_expert_indices(
    request_output: Any,
    completion_output: Any,
    *,
    valid_length: int,
    padded_length: int,
    device: torch.device,
    require_complete_routed_experts: bool = False,
    allow_missing_routed_experts_fallback: bool = True,
    return_stats: bool = False,
) -> Optional[torch.Tensor] | tuple[Optional[torch.Tensor], dict[str, int]]:
    """Return full-sequence-aligned routed experts as ``[S, L, topk]`` int32."""
    routed = getattr(completion_output, "routed_experts", None)
    prompt_routed = getattr(request_output, "prompt_routed_experts", None)

    if prompt_routed is not None:
        prompt_routed = torch.as_tensor(prompt_routed, dtype=torch.int32, device=device)
    if routed is not None:
        routed = torch.as_tensor(routed, dtype=torch.int32, device=device)

    if prompt_routed is not None and routed is not None:
        routed = torch.cat((prompt_routed, routed), dim=0)
    elif prompt_routed is not None:
        routed = prompt_routed

    expected_routes = min(max(valid_length - 1, 0), padded_length)
    stats = {
        "actual_routes": 0,
        "expected_routes": expected_routes,
        "missing_routes": 0,
        "surplus_routes": 0,
    }

    if routed is None:
        return (None, stats) if return_stats else None
    if routed.dim() != 3:
        raise ValueError(
            "vLLM routed_experts must have shape [tokens, num_moe_layers, topk], "
            f"got {tuple(routed.shape)}"
        )

    stats["actual_routes"] = int(routed.shape[0])
    stats["missing_routes"] = max(expected_routes - int(routed.shape[0]), 0)
    stats["surplus_routes"] = max(int(routed.shape[0]) - (expected_routes + 1), 0)
    if (
        require_complete_routed_experts
        and stats["missing_routes"] > 0
        and not allow_missing_routed_experts_fallback
    ):
        # This has only been observed rarely with vLLM prefix caching plus
        # chunked prefill: a small number of samples can omit routed-expert
        # rows even though most requests are complete. Keep
        # tools/model_diagnostics/6.vllm_routed_experts_completeness.py as a
        # standalone reproducer for upstream vLLM bug reports.
        num_cached_tokens = getattr(request_output, "num_cached_tokens", None)
        raise ValueError(
            "vLLM returned incomplete routed_experts for router replay: "
            f"routes={routed.shape[0]}, expected_at_least={expected_routes}, "
            f"valid_length={valid_length}, padded_length={padded_length}, "
            f"num_cached_tokens={num_cached_tokens}. This usually means the "
            "generation backend did not return routed experts for every "
            "non-final token in the prompt+response sequence."
        )
    max_allowed_routes = expected_routes + 1
    if require_complete_routed_experts and routed.shape[0] > max_allowed_routes:
        num_cached_tokens = getattr(request_output, "num_cached_tokens", None)
        raise ValueError(
            "vLLM returned too many routed_experts routes for router replay: "
            f"routes={routed.shape[0]}, expected={expected_routes}, "
            f"max_allowed={max_allowed_routes}, valid_length={valid_length}, "
            f"padded_length={padded_length}, num_cached_tokens={num_cached_tokens}. "
            "Router replay allows at most one surplus final-token route."
        )

    default_route = torch.arange(
        routed.shape[2],
        dtype=torch.int32,
        device=device,
    )
    full = (
        default_route.view(1, 1, -1)
        .expand(padded_length, routed.shape[1], routed.shape[2])
        .clone()
    )
    full = full.to(dtype=torch.int32)
    routes_to_copy = min(expected_routes, routed.shape[0])
    if routes_to_copy > 0:
        full[:routes_to_copy] = routed[:routes_to_copy].to(device=device)
    if stats["missing_routes"] > 0:
        full[routes_to_copy:expected_routes] = R3_MISSING_ROUTE_SENTINEL
    return (full, stats) if return_stats else full


def attach_routed_experts_to_chat_response_choices(
    response: Any,
    final_request_output: Any,
    *,
    device: torch.device,
    logger: Any = None,
) -> Any:
    """Attach aligned routed experts to OpenAI chat response choices."""
    outputs_by_index = {
        output.index: output for output in getattr(final_request_output, "outputs", [])
    }
    prompt_token_count = len(
        getattr(final_request_output, "prompt_token_ids", []) or []
    )

    choices = list(getattr(response, "choices", []))
    attached_choice_indices = set()
    for choice in choices:
        generation_details = outputs_by_index.get(choice.index)
        if generation_details is None:
            continue
        attached_choice_indices.add(choice.index)

        generation_token_count = len(getattr(generation_details, "token_ids", []) or [])
        routed_result = pad_and_align_routed_expert_indices(
            final_request_output,
            generation_details,
            valid_length=prompt_token_count + generation_token_count,
            padded_length=prompt_token_count + generation_token_count,
            device=device,
            require_complete_routed_experts=True,
            return_stats=True,
        )
        if not isinstance(routed_result, tuple):
            raise RuntimeError(
                "Expected routed_experts alignment to return stats for the "
                "OpenAI-compatible chat endpoint."
            )
        routed_experts, r3_stats = routed_result
        if routed_experts is None:
            raise RuntimeError(
                "vLLM was asked to return routed experts for the "
                "OpenAI-compatible chat endpoint but the generation "
                "output did not include routed_experts."
            )
        if r3_stats["missing_routes"] > 0 and logger is not None:
            logger.warning(
                "R3 router replay fallback: vLLM returned incomplete "
                "routed_experts for chat choice_idx=%d, "
                "missing_token_routes=%d, actual_routes=%d, "
                "expected_routes=%d. Megatron will use its own router "
                "for those missing token routes.",
                choice.index,
                r3_stats["missing_routes"],
                r3_stats["actual_routes"],
                r3_stats["expected_routes"],
            )
        choice.message.routed_experts = routed_experts.to(dtype=torch.int32).tolist()

    if len(attached_choice_indices) != len(choices):
        missing_choice_indices = sorted(
            choice.index
            for choice in choices
            if choice.index not in attached_choice_indices
        )
        raise RuntimeError(
            "vLLM was asked to return routed experts for the "
            "OpenAI-compatible chat endpoint but response choices could not be "
            "matched to generation outputs: "
            f"missing_choice_indices={missing_choice_indices}."
        )

    return response


def _find_non_finite_float_paths(value: Any, path: str = "$") -> list[str]:
    """Return JSON-style paths for non-finite float values without mutating them."""
    if isinstance(value, float):
        return [] if math.isfinite(value) else [f"{path}={value!r}"]
    if isinstance(value, dict):
        paths = []
        for key, item in value.items():
            paths.extend(_find_non_finite_float_paths(item, f"{path}.{key}"))
        return paths
    if isinstance(value, (list, tuple)):
        paths = []
        for index, item in enumerate(value):
            paths.extend(_find_non_finite_float_paths(item, f"{path}[{index}]"))
        return paths
    return []


def model_dump_chat_response_with_routed_experts(response: Any) -> dict[str, Any]:
    """Dump a vLLM OpenAI chat response while preserving dynamic R3 fields."""
    response_dict = response.model_dump()
    for choice, choice_dict in zip(
        getattr(response, "choices", []), response_dict.get("choices", [])
    ):
        routed_experts = getattr(
            getattr(choice, "message", None), "routed_experts", None
        )
        if routed_experts is not None:
            choice_dict.setdefault("message", {})["routed_experts"] = routed_experts

    non_finite_paths = _find_non_finite_float_paths(response_dict)
    if non_finite_paths:
        preview = ", ".join(non_finite_paths[:16])
        omitted = len(non_finite_paths) - 16
        suffix = f", ... and {omitted} more" if omitted > 0 else ""
        raise ValueError(
            f"vLLM chat response contains non-finite float values at {preview}{suffix}"
        )
    return response_dict


def aggregate_spec_decode_counters(
    worker_metrics: list[dict[str, float | list[float]]],
) -> dict[str | tuple[str, int], float]:
    """Aggregate speculative decoding counters from multiple workers.

    Combines spec decode metrics collected from DP leader workers into
    a single aggregated counter dictionary.

    Args:
        worker_metrics: List of metric dictionaries from each worker.
            Each dict maps metric names to float values or lists of floats
            (for per-position metrics).

    Returns:
        Dictionary mapping metric names to their aggregated float values.
        Per-position metrics use (name, position) tuples as keys.

    Example:
        >>> metrics_from_workers = policy_generation.get_metrics()
        >>> counters = aggregate_spec_decode_counters(metrics_from_workers)
        >>> print(counters.get("vllm:spec_decode_num_drafts", 0))
        1234.0
    """
    counters: dict[str | tuple[str, int], float] = defaultdict(float)

    for report in worker_metrics:
        for metric_name, value in report.items():
            if "spec_decode" in metric_name:
                if isinstance(value, list):
                    # Per-position metrics (e.g., acceptance counts at each draft position)
                    for position, pos_value in enumerate(value, 1):
                        counters[metric_name, position] += pos_value
                else:
                    counters[metric_name] += value

    return dict(counters)


def compute_spec_decode_metrics(
    start_counters: dict[str | tuple[str, int], float],
    end_counters: dict[str | tuple[str, int], float],
) -> dict[str, float]:
    """Compute delta and derived metrics for speculative decoding.

    Calculates the difference between two counter snapshots and derives
    acceptance rate and acceptance length metrics for logging.

    Args:
        start_counters: Counter snapshot taken before generation.
        end_counters: Counter snapshot taken after generation.

    Returns:
        Dictionary of metrics suitable for logging to wandb/tensorboard.
        Keys are prefixed with "vllm/" for namespace consistency.
        Includes:
            - vllm/spec_num_drafts: Total number of draft batches
            - vllm/spec_num_draft_tokens: Total draft tokens generated
            - vllm/spec_num_accepted_tokens: Total tokens accepted
            - vllm/spec_acceptance_length: Average accepted tokens per draft + 1
            - vllm/spec_acceptance_rate: Ratio of accepted to draft tokens
            - vllm/{metric}-{position}: Per-position acceptance counts
            - vllm/spec_acceptance_rate-pos-{position}: Per-position acceptance rates
    """
    keys = set(start_counters) | set(end_counters)
    delta = {k: end_counters.get(k, 0.0) - start_counters.get(k, 0.0) for k in keys}

    num_drafts = delta.get("vllm:spec_decode_num_drafts", 0.0)
    num_draft_tokens = delta.get("vllm:spec_decode_num_draft_tokens", 0.0)
    num_accepted_tokens = delta.get("vllm:spec_decode_num_accepted_tokens", 0.0)

    # acceptance_length = 1 + (accepted / drafts) represents average tokens
    # generated per draft batch (1 target model token + accepted draft tokens)
    acceptance_length = (
        1.0 + (num_accepted_tokens / num_drafts) if num_drafts > 0 else 1.0
    )
    acceptance_rate = (
        num_accepted_tokens / num_draft_tokens if num_draft_tokens > 0 else 0.0
    )

    spec_metrics: dict[str, float] = {
        "vllm/spec_num_drafts": num_drafts,
        "vllm/spec_num_draft_tokens": num_draft_tokens,
        "vllm/spec_num_accepted_tokens": num_accepted_tokens,
        "vllm/spec_acceptance_length": acceptance_length,
        "vllm/spec_acceptance_rate": acceptance_rate,
    }

    # Add per-position metrics for detailed analysis
    for key, value in delta.items():
        if isinstance(key, tuple):
            metric_name, position = key
            spec_metrics[f"vllm/{metric_name}-{position}"] = value
            if num_drafts > 0:
                spec_metrics[f"vllm/spec_acceptance_rate-pos-{position}"] = (
                    value / num_drafts
                )

    return spec_metrics


# TODO: Replace this hard-coded map with a generic plugin-registration
# hook on ``VllmGeneration`` (e.g. a ``worker_cls_overrides`` registry populated
# by ``nemo_rl.modelopt`` on import) so core has no knowledge of ModelOpt-specific
# worker classes.
GENERATION_WORKER_OVERRIDES = {
    "nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker": "nemo_rl.modelopt.models.generation.vllm_quant_worker.VllmQuantGenerationWorker",
    "nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker": "nemo_rl.modelopt.models.generation.vllm_quant_worker.VllmQuantAsyncGenerationWorker",
}


def resolve_generation_worker_cls(default_cls: str, config: dict) -> str:
    """Return the quantized vLLM generation worker FQN if ``quant_cfg`` is set, else ``default_cls``.

    Safe to call even when ModelOpt is not installed — returns ``default_cls``
    unchanged whenever ``quant_cfg`` is ``None``, so the core generation path
    stays import-free of ModelOpt.
    """
    if config.get("quant_cfg") is None:
        return default_cls
    return GENERATION_WORKER_OVERRIDES.get(default_cls, default_cls)
