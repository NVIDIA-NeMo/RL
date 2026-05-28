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
"""Realistic rollout-shaped data builders + shared test helpers.

Mints data with the *shape and types* an actual GRPO rollout produces —
mixed dtypes (bf16 logprobs, int64 ids, int32 masks), realistic value
distributions, optional multimodal extras, and varied multi-turn message
logs. Use these instead of inline toy tensors so tests cover the same
type / scenario complexity as production runs.

Also exposes a handful of small cross-file test helpers (uid → key
minting, TQ partition setup, mooncake availability) that several test
files used to duplicate.

Helpers are plain functions (not pytest fixtures) so they're explicit at
the call site and don't depend on a conftest.
"""

from __future__ import annotations

import os
from typing import Any

import numpy as np
import torch

from nemo_rl.data_plane.adapters.noop import NoOpDataPlaneClient
from nemo_rl.data_plane.schema import DP_TRAIN_FIELDS


def make_rollout_batch(
    n: int = 8,
    max_seqlen: int = 256,
    *,
    multimodal: bool = False,
    logprob_dtype: torch.dtype = torch.bfloat16,
    id_dtype: torch.dtype = torch.long,
    mask_dtype: torch.dtype = torch.int32,
    seed: int = 42,
) -> dict[str, Any]:
    """Return a fields dict shaped like rollout's first put.

    Mirrors what ``SyncRolloutActor.rollout_to_tq`` actually writes:
    int64 token ids, int32 masks, bf16 logprobs, fp32 (or bf16) advantages,
    optional ``multi_modal_inputs`` dict for VLM models.

    Args:
        n: Batch size.
        max_seqlen: Padded sequence length; per-row valid length is in
            ``[max_seqlen//4, max_seqlen]``.
        multimodal: Include ``multi_modal_inputs`` (pixel_values + image_grid_thw).
        logprob_dtype: dtype for logprobs/advantages (real runs use bf16).
        id_dtype: dtype for input_ids/lengths.
        mask_dtype: dtype for masks.
        seed: RNG seed for reproducibility.

    Returns:
        Dict with keys ``input_ids``, ``input_lengths``, ``attention_mask``,
        ``token_mask``, ``sample_mask``, ``generation_logprobs``,
        ``prev_logprobs``, ``reference_policy_logprobs``, ``advantages``,
        optionally ``multi_modal_inputs``.
    """
    g = torch.Generator().manual_seed(seed)

    # Per-row valid lengths spanning ~25-100% of max_seqlen.
    low = max(1, max_seqlen // 4)
    lengths = torch.randint(low, max_seqlen + 1, (n,), generator=g).to(id_dtype)

    # Token ids: random vocab-shaped, padded with 0.
    input_ids = torch.zeros((n, max_seqlen), dtype=id_dtype)
    for i in range(n):
        nrow = int(lengths[i])
        input_ids[i, :nrow] = torch.randint(1, 50000, (nrow,), generator=g)

    # Masks: 1 for valid tokens, 0 for padding.
    token_mask = torch.zeros((n, max_seqlen), dtype=mask_dtype)
    for i in range(n):
        token_mask[i, : int(lengths[i])] = 1
    attention_mask = token_mask.clone()
    sample_mask = torch.ones((n,), dtype=mask_dtype)

    # Logprobs: realistic distribution centered around -2.0 (typical token logprob),
    # std ~1 — catches dtype-narrowing bugs that pass on zero inputs.
    def _lp() -> torch.Tensor:
        return (torch.randn(n, max_seqlen, generator=g) - 2.0).to(logprob_dtype)

    out: dict[str, Any] = {
        "input_ids": input_ids,
        "input_lengths": lengths.to(torch.long),
        "attention_mask": attention_mask,
        "token_mask": token_mask,
        "sample_mask": sample_mask,
        "generation_logprobs": _lp(),
        "prev_logprobs": _lp(),
        "reference_policy_logprobs": _lp(),
        "advantages": torch.randn(n, max_seqlen, generator=g).to(logprob_dtype),
    }

    if multimodal:
        # VLM extras as flat top-level fields (the codec wire format —
        # nested dicts aren't valid leaves). Real production writes these
        # with similar shapes; we keep them small for fast tests.
        T, H, W = 1, 8, 8
        n_image_tokens = T * H * W
        out["pixel_values"] = torch.randn(n, n_image_tokens, 3, generator=g).to(
            torch.bfloat16
        )
        out["image_grid_thw"] = torch.tensor([[T, H, W]] * n, dtype=torch.long)

    return out


def make_realistic_tags(
    n: int,
    *,
    zero_std_fraction: float = 0.25,
    seed: int = 42,
) -> list[dict[str, float | int]]:
    """Per-sample tags as produced by the GRPO driver after baseline/std compute.

    Mirrors what gets stamped onto ``KVBatchMeta.tags`` for dynamic-sampling
    filtering. Some rows have ``std=0.0`` (zero-variance, filtered) and
    others non-zero (survivors).

    Args:
        n: Number of samples.
        zero_std_fraction: Fraction of rows tagged with ``std=0.0`` (filtered).
        seed: RNG seed.
    """
    rng = np.random.default_rng(seed)
    n_zero = int(round(n * zero_std_fraction))
    stds = np.concatenate([np.zeros(n_zero), rng.uniform(0.1, 1.5, size=n - n_zero)])
    rng.shuffle(stds)
    rewards = rng.uniform(-1.0, 1.0, size=n)
    prompt_ids = rng.integers(0, 1000, size=n)
    return [
        {
            "std": float(stds[i]),
            "total_reward": float(rewards[i]),
            "prompt_id": int(prompt_ids[i]),
            "weight_version": 1,
        }
        for i in range(n)
    ]


def make_multi_turn_message_log(
    n: int,
    *,
    turns_per_sample: list[int] | None = None,
    seed: int = 42,
) -> list[list[dict[str, Any]]]:
    """Realistic multi-turn message_log: list-of-turn-dicts per sample.

    Each turn carries ``role`` (alternating user/assistant), ``content``
    (string), and ``token_ids`` (int64 tensor). Variable turn counts
    capture the jagged case that ``decompose_message_log`` flattens.

    Args:
        n: Number of samples.
        turns_per_sample: Optional explicit turn count per sample. If
            None, random in ``[1, 4]``.
        seed: RNG seed.
    """
    g = torch.Generator().manual_seed(seed)
    if turns_per_sample is None:
        turns_per_sample = [int(t) for t in torch.randint(1, 5, (n,), generator=g)]
    out: list[list[dict[str, Any]]] = []
    for i, k in enumerate(turns_per_sample):
        sample_log: list[dict[str, Any]] = []
        for t in range(k):
            role = "user" if t % 2 == 0 else "assistant"
            tok_len = int(torch.randint(8, 64, (1,), generator=g))
            sample_log.append(
                {
                    "role": role,
                    "content": f"sample_{i}_turn_{t}_text",
                    "token_ids": torch.randint(
                        1, 50000, (tok_len,), generator=g, dtype=torch.long
                    ),
                }
            )
        out.append(sample_log)
    return out


# ── Cross-file test helpers (deduped from per-file definitions) ──────────────


def keys_from_uids(uids: list[str], n_gen: int = 1) -> list[str]:
    """Mint per-generation sample keys from prompt uids: ``f"{uid}_g{i}"``.

    Mirrors the production rollout convention — one key per generation per
    prompt — so tests share the same uid → key mapping as the trainer.
    """
    return [f"{uid}_g{i}" for uid in uids for i in range(n_gen)]


def register_train_partition(
    client: NoOpDataPlaneClient,
    *,
    num_samples: int,
    fields: list[str] | None = None,
    partition_id: str = "train",
    consumer_tasks: list[str] | None = None,
) -> None:
    """Open a TQ partition with the train-side defaults (``DP_TRAIN_FIELDS`` + ``["train"]``).

    Centralizes the boilerplate three test files used to inline as
    ``_setup`` / ``_setup_partition``.
    """
    client.register_partition(
        partition_id=partition_id,
        fields=list(fields if fields is not None else DP_TRAIN_FIELDS),
        num_samples=num_samples,
        consumer_tasks=consumer_tasks if consumer_tasks is not None else ["train"],
    )


def simple_backend_dp_config() -> dict[str, Any]:
    """Load the SimpleStorage ``DataPlaneConfig`` from the test reference YAML.

    Reuses the ``data_plane:`` section of
    ``tests/unit/reference_configs/grpo_math_1B.yaml`` (canonical test
    config, mirrored from ``examples/configs/grpo_math_1B.yaml``) so test
    config flows through the same OmegaConf path production uses.

    Overrides applied for the unit-test scenario:
    - ``enabled: true`` (production default is False; tests need the
      ``build_data_plane_client`` factory to actually build a client).
    - ``storage_capacity: 1024`` / ``num_storage_units: 1`` — trimmed
      from production's 1_000_000 / 2 for fast in-process setup.
    """
    from pathlib import Path

    from omegaconf import OmegaConf

    cfg_path = (
        Path(__file__).resolve().parents[1] / "reference_configs" / "grpo_math_1B.yaml"
    )
    full_cfg = OmegaConf.load(cfg_path)
    dp_cfg = OmegaConf.to_container(full_cfg.data_plane, resolve=True)
    dp_cfg["enabled"] = True
    dp_cfg["storage_capacity"] = 1024
    dp_cfg["num_storage_units"] = 1
    return dp_cfg  # type: ignore[return-value]


def mooncake_available() -> bool:
    """Return True if the ``mooncake`` wheel is importable.

    Set ``NEMO_RL_REQUIRE_MOONCAKE=1`` to promote a missing import into a
    loud ``ImportError`` instead of returning False — so CI fails when
    the wheel is expected but absent.
    """
    try:
        import mooncake  # noqa: F401
    except ImportError:
        if os.environ.get("NEMO_RL_REQUIRE_MOONCAKE") == "1":
            raise
        return False
    return True
