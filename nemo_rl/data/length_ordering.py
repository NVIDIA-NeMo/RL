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

"""Reorder a training dataset by trace-derived output length.

Given a length ordering produced by ``tools/build_length_ordering.py`` (keyed by
first-turn ``prompt_token_hash``), this reorders a processed dataset so that
prompts with shorter observed generations come first. Combined with
``data.shuffle=false``, this makes lower-output-length steps run before
higher-output-length steps.

The join key is the first-turn ``prompt_token_hash`` computed with the exact same
formula NeMo-RL uses when it writes the rollout trace
(``nemo_rl.experience.rollouts._token_ids_hash``): the SHA-256 of the
compact-JSON prompt token-id list. This depends on the dataset processing
producing the same prompt tokens the rollout saw; the loader logs the join
hit-rate so a templating mismatch is obvious.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch


def token_ids_hash(token_ids: list[int]) -> str:
    """Match ``nemo_rl.experience.rollouts._token_ids_hash``."""
    payload = json.dumps(token_ids, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _token_ids_to_list(value: Any) -> list[int]:
    if value is None:
        return []
    if isinstance(value, torch.Tensor):
        return [int(t) for t in value.detach().cpu().tolist()]
    return [int(t) for t in value]


def first_turn_prompt_hash(datum_spec: dict[str, Any]) -> str | None:
    """Prompt hash for a freshly processed (turn-0) datum.

    Mirrors the rollout writer: the prompt is every message preceding the first
    assistant message. A training datum has no assistant message yet, so this is
    the concatenation of all message token-id lists.
    """
    message_log = datum_spec.get("message_log")
    if not message_log:
        return None
    prompt_tokens: list[int] = []
    for message in message_log:
        if message.get("role") == "assistant":
            break
        prompt_tokens.extend(_token_ids_to_list(message.get("token_ids")))
    if not prompt_tokens:
        return None
    return token_ids_hash(prompt_tokens)


def load_length_order(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load the ``labels`` map ({prompt_token_hash: {rank, ...}})."""
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    labels = payload.get("labels")
    if not isinstance(labels, dict):
        raise ValueError(f"{path} does not contain a 'labels' object")
    return labels


class LengthOrderedDataset:
    """Index-remapping view over a processed dataset.

    ``base_dataset`` must support ``__len__`` and ``__getitem__`` returning a
    ``DatumSpec`` (e.g. ``AllTaskProcessedDataset``). ``order`` is a permutation
    of base indices; iteration/index access follow ``order``.
    """

    def __init__(self, base_dataset: Any, order: list[int]) -> None:
        self.base_dataset = base_dataset
        self.order = list(order)

    def __len__(self) -> int:
        return len(self.order)

    def __getitem__(self, idx: int) -> Any:
        return self.base_dataset[self.order[idx]]

    def __getattr__(self, name: str) -> Any:
        # Delegate processor/spec attributes (task_data_spec, tokenizer, ...).
        return getattr(self.base_dataset, name)


def build_length_ordered_dataset(
    base_dataset: Any,
    length_order_json: str | Path,
    *,
    unseen: str = "last",
    log=print,
) -> LengthOrderedDataset:
    """Reorder ``base_dataset`` shortest-output-first via a length-order file.

    Args:
        base_dataset: Processed dataset (indexable, returns ``DatumSpec``).
        length_order_json: Output of ``tools/build_length_ordering.py``.
        unseen: Where to place prompts absent from the order file:
            ``"last"`` (default) keeps them after all labeled prompts (so a run
            capped at the labeled step count trains only labeled prompts), or
            ``"first"`` to front-load them.
        log: Logging callable.

    Returns:
        A ``LengthOrderedDataset`` view over ``base_dataset``.
    """
    if unseen not in ("last", "first"):
        raise ValueError(f"unseen must be 'last' or 'first', got {unseen!r}")

    labels = load_length_order(length_order_json)
    n = len(base_dataset)

    keyed: list[tuple[tuple[int, float, int], int]] = []
    hits = 0
    for idx in range(n):
        datum = base_dataset[idx]
        prompt_hash = first_turn_prompt_hash(datum)
        entry = labels.get(prompt_hash) if prompt_hash is not None else None
        if entry is not None:
            hits += 1
            # (labeled-first-or-last, rank, idx) sort key
            primary = 0
            rank = float(entry.get("rank", 0))
        else:
            primary = 1 if unseen == "last" else -1
            rank = float("inf") if unseen == "last" else float("-inf")
        keyed.append(((primary, rank, idx), idx))

    keyed.sort(key=lambda pair: pair[0])
    order = [idx for _, idx in keyed]

    miss = n - hits
    pct = (100.0 * hits / n) if n else 0.0
    log(
        f"[length_ordering] matched {hits}/{n} dataset prompts to "
        f"{length_order_json} ({pct:.1f}% hit-rate); {miss} unseen placed {unseen}"
    )
    if hits == 0:
        log(
            "[length_ordering] WARNING: zero prompts matched. The dataset "
            "prompt tokenization likely differs from the rollout trace "
            "(templating mismatch). Regenerate labels from a trace produced by "
            "this exact pipeline, or the ordering will have no effect."
        )
    return LengthOrderedDataset(base_dataset, order)
