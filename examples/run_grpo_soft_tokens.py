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

"""Run GRPO with precomputed soft-token conditioning.

This is a minimal integration runner for peptide-style experiments where an
external encoder, such as ESM plus a CNN projector, provides embeddings in the
LLM hidden size. In `datum` mode, each raw dataset row must contain
`soft_token_embeddings`; it may also contain `soft_token_positions` and
`vllm_prompt_embeds`. If those placement tensors are absent, the processor
injects the soft tokens into the final prompt positions. Set
`data.soft_token.vllm_prompt_mode=prompt_embeds` to send those prompt embeddings
to vLLM, or `token_ids` to let vLLM use the normal tokenized text prompt while
the DTensor policy path still receives soft-token tensors. In `smoke` mode, the
processor creates deterministic placeholder embeddings so the NeMo-RL, DTensor,
and vLLM prompt-embedding plumbing can be tested without adding an encoder
dependency.
"""

from __future__ import annotations

import os
import sys
from typing import Any

import torch
from omegaconf import OmegaConf

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from examples.run_grpo import main as run_grpo_main
from examples.run_grpo import parse_args
from nemo_rl.data.interfaces import DatumSpec, TaskDataSpec, TokenizerType
from nemo_rl.data.processors import (
    PROCESSOR_REGISTRY,
    math_hf_data_processor,
    register_processor,
)
from nemo_rl.data.soft_tokens import (
    SOFT_TOKEN_EMBEDDINGS_KEY,
    SOFT_TOKEN_POSITIONS_KEY,
    VLLM_PROMPT_EMBEDS_KEY,
)
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)

SoftTokenDataConfig = dict[str, Any]
PROCESSOR_NAME = "soft_token_math_hf_data_processor"
DEFAULT_CONFIG = os.path.join(
    os.path.dirname(__file__),
    "configs",
    "recipes",
    "llm",
    "grpo-qwen2.5-1.5b-softtoken-1n1g-fsdp2tp1.yaml",
)

_SOFT_TOKEN_CONFIG: SoftTokenDataConfig | None = None


def _has_config_arg() -> bool:
    return any(arg == "--config" or arg.startswith("--config=") for arg in sys.argv)


def _set_default_config_arg() -> None:
    if not _has_config_arg():
        sys.argv.extend(["--config", DEFAULT_CONFIG])


def _load_soft_token_config_from_cli() -> SoftTokenDataConfig:
    register_omegaconf_resolvers()
    args, overrides = parse_args()
    if args.config is None:
        args.config = DEFAULT_CONFIG

    config = load_config(args.config)
    if overrides:
        config = parse_hydra_overrides(config, overrides)

    resolved = OmegaConf.to_container(config, resolve=True)
    assert isinstance(resolved, dict)
    data_config = resolved["data"]
    if "soft_token" not in data_config:
        raise ValueError(
            f"{PROCESSOR_NAME} requires a data.soft_token config block. "
            f"Use the default recipe at {DEFAULT_CONFIG} as a template."
        )
    return data_config["soft_token"]


def _get_soft_token_config() -> SoftTokenDataConfig:
    if _SOFT_TOKEN_CONFIG is None:
        raise RuntimeError("soft-token processor was not configured")
    return _SOFT_TOKEN_CONFIG


def _prompt_length(datum_spec: DatumSpec) -> int:
    return sum(len(message["token_ids"]) for message in datum_spec["message_log"])


def _load_datum_tensor(
    datum_dict: dict[str, Any],
    key: str,
    dtype: torch.dtype,
) -> torch.Tensor:
    if key not in datum_dict:
        raise ValueError(f"soft-token datum mode requires dataset field {key}")
    return torch.as_tensor(datum_dict[key], dtype=dtype)


def _build_smoke_tensors(
    prompt_length: int,
    idx: int,
    soft_token_config: SoftTokenDataConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    hidden_size = int(soft_token_config["hidden_size"])
    num_soft_tokens = int(soft_token_config["num_soft_tokens"])
    seed = int(soft_token_config["seed"])
    scale = float(soft_token_config["scale"])

    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + idx)

    soft_token_embeddings = (
        torch.randn(num_soft_tokens, hidden_size, generator=generator) * scale
    )
    soft_token_embeddings, soft_token_positions, vllm_prompt_embeds = (
        _place_soft_tokens_at_prompt_end(
            soft_token_embeddings, prompt_length, hidden_size
        )
    )

    return soft_token_embeddings, soft_token_positions, vllm_prompt_embeds


def _place_soft_tokens_at_prompt_end(
    soft_token_embeddings: torch.Tensor,
    prompt_length: int,
    hidden_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    soft_token_embeddings = soft_token_embeddings.clone()
    num_soft_tokens = soft_token_embeddings.shape[0]
    soft_token_positions = torch.full((num_soft_tokens,), -1, dtype=torch.long)
    vllm_prompt_embeds = torch.zeros(
        prompt_length, hidden_size, dtype=soft_token_embeddings.dtype
    )

    valid_soft_tokens = min(num_soft_tokens, prompt_length)
    if valid_soft_tokens > 0:
        start_position = prompt_length - valid_soft_tokens
        soft_token_positions[:valid_soft_tokens] = torch.arange(
            start_position, prompt_length, dtype=torch.long
        )
        vllm_prompt_embeds[soft_token_positions[:valid_soft_tokens]] = (
            soft_token_embeddings[:valid_soft_tokens]
        )
    if valid_soft_tokens < num_soft_tokens:
        soft_token_embeddings[valid_soft_tokens:] = 0

    return soft_token_embeddings, soft_token_positions, vllm_prompt_embeds


def _load_datum_tensors(
    datum_dict: dict[str, Any],
    prompt_length: int,
    soft_token_config: SoftTokenDataConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    soft_token_embeddings = _load_datum_tensor(
        datum_dict, SOFT_TOKEN_EMBEDDINGS_KEY, torch.float32
    )
    has_soft_token_positions = SOFT_TOKEN_POSITIONS_KEY in datum_dict
    has_vllm_prompt_embeds = VLLM_PROMPT_EMBEDS_KEY in datum_dict
    if has_soft_token_positions != has_vllm_prompt_embeds:
        raise ValueError(
            f"datum mode requires {SOFT_TOKEN_POSITIONS_KEY} and "
            f"{VLLM_PROMPT_EMBEDS_KEY} to be provided together, or omitted together"
        )
    if not has_soft_token_positions:
        return _place_soft_tokens_at_prompt_end(
            soft_token_embeddings=soft_token_embeddings,
            prompt_length=prompt_length,
            hidden_size=int(soft_token_config["hidden_size"]),
        )

    return (
        soft_token_embeddings,
        _load_datum_tensor(datum_dict, SOFT_TOKEN_POSITIONS_KEY, torch.long),
        _load_datum_tensor(datum_dict, VLLM_PROMPT_EMBEDS_KEY, torch.float32),
    )


def _validate_soft_token_tensors(
    soft_token_embeddings: torch.Tensor,
    soft_token_positions: torch.Tensor,
    vllm_prompt_embeds: torch.Tensor,
    prompt_length: int,
    soft_token_config: SoftTokenDataConfig,
) -> None:
    hidden_size = int(soft_token_config["hidden_size"])
    if soft_token_embeddings.ndim != 2:
        raise ValueError(
            f"{SOFT_TOKEN_EMBEDDINGS_KEY} must have shape [K, H] before batching"
        )
    if soft_token_positions.ndim != 1:
        raise ValueError(
            f"{SOFT_TOKEN_POSITIONS_KEY} must have shape [K] before batching"
        )
    if vllm_prompt_embeds.ndim != 2:
        raise ValueError(
            f"{VLLM_PROMPT_EMBEDS_KEY} must have shape [S, H] before batching"
        )
    if soft_token_embeddings.shape[0] != soft_token_positions.shape[0]:
        raise ValueError(
            f"{SOFT_TOKEN_EMBEDDINGS_KEY} and {SOFT_TOKEN_POSITIONS_KEY} must "
            "have the same K dimension"
        )
    if soft_token_embeddings.shape[-1] != hidden_size:
        raise ValueError(
            f"{SOFT_TOKEN_EMBEDDINGS_KEY} hidden size "
            f"{soft_token_embeddings.shape[-1]} does not match "
            f"data.soft_token.hidden_size={hidden_size}"
        )
    if vllm_prompt_embeds.shape != (prompt_length, hidden_size):
        raise ValueError(
            f"{VLLM_PROMPT_EMBEDS_KEY} must have shape "
            f"[prompt_length, hidden_size] = [{prompt_length}, {hidden_size}], "
            f"got {tuple(vllm_prompt_embeds.shape)}"
        )


def _use_vllm_prompt_embeds(soft_token_config: SoftTokenDataConfig) -> bool:
    vllm_prompt_mode = soft_token_config["vllm_prompt_mode"]
    if vllm_prompt_mode == "prompt_embeds":
        return True
    if vllm_prompt_mode == "token_ids":
        return False
    raise ValueError(
        "data.soft_token.vllm_prompt_mode must be either 'prompt_embeds' or "
        f"'token_ids', got {vllm_prompt_mode}"
    )


def soft_token_math_hf_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer: TokenizerType,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    datum_spec = math_hf_data_processor(
        datum_dict, task_data_spec, tokenizer, max_seq_length, idx
    )
    soft_token_config = _get_soft_token_config()
    prompt_length = _prompt_length(datum_spec)

    if soft_token_config["mode"] == "datum":
        soft_token_embeddings, soft_token_positions, vllm_prompt_embeds = (
            _load_datum_tensors(datum_dict, prompt_length, soft_token_config)
        )
    elif soft_token_config["mode"] == "smoke":
        soft_token_embeddings, soft_token_positions, vllm_prompt_embeds = (
            _build_smoke_tensors(prompt_length, idx, soft_token_config)
        )
    else:
        raise ValueError(
            "data.soft_token.mode must be either 'datum' or 'smoke', got "
            f"{soft_token_config['mode']}"
        )

    _validate_soft_token_tensors(
        soft_token_embeddings=soft_token_embeddings,
        soft_token_positions=soft_token_positions,
        vllm_prompt_embeds=vllm_prompt_embeds,
        prompt_length=prompt_length,
        soft_token_config=soft_token_config,
    )

    datum_spec[SOFT_TOKEN_EMBEDDINGS_KEY] = soft_token_embeddings
    datum_spec[SOFT_TOKEN_POSITIONS_KEY] = soft_token_positions
    if _use_vllm_prompt_embeds(soft_token_config):
        datum_spec[VLLM_PROMPT_EMBEDS_KEY] = vllm_prompt_embeds
    datum_spec["length"] = prompt_length
    return datum_spec


def main() -> None:
    global _SOFT_TOKEN_CONFIG

    _set_default_config_arg()
    _SOFT_TOKEN_CONFIG = _load_soft_token_config_from_cli()
    if PROCESSOR_NAME not in PROCESSOR_REGISTRY:
        register_processor(PROCESSOR_NAME, soft_token_math_hf_data_processor)

    run_grpo_main()


if __name__ == "__main__":
    main()
