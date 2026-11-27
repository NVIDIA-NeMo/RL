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
"""Privileged value for PPO on the SingleController (asymmetric actor-critic).

The critic sees a reference block the policy never does. The block is a prefix.
The value head is causal, so ``V(s_t)`` attends only to tokens before ``t``, and
the only placement every supervised position can reach is before the first
assistant token. The policy cannot see the reference, so ``a_t ⊥ z | s_t`` and
``E[∇log π(a_t|s_t) · V(s_t, z)] = 0``: the policy gradient stays unbiased. The
prefixes must never reach the policy worker.

A prefix builder, selected by ``value.privileged_value.prefix_builder_fqn``,
turns one prompt's ``extra_env_info`` into the block:

    build_privilege_prefix(extra_env_info, tokenizer, cfg) -> (block, stats)

``block`` is the reference text and must not be empty. ``stats`` carries
``wanted_tokens`` and ``kept_tokens`` (the block's size before and after
truncation) and ``truncated_fields`` (the names of the parts that were cut);
they feed the ``privilege/*`` metrics. The builder must be a pure function of
the prompt, so sibling rollouts of one prompt share an identical block. Its
module may also define ``validate_prefix_builder_kwargs(kwargs)``, which config
validation calls at startup on ``prefix_builder_kwargs``.

:class:`PrivilegePrefixStore` caches one tokenized block per prompt on the
controller and stamps its key and length on each row's tags. The row layout
itself is in ``nemo_rl.data_plane.privilege_prefix``.
"""

import asyncio
import os
from dataclasses import replace
from typing import Any, Callable, Collection, Iterable, Optional, Sequence

import torch
from hydra.utils import get_method

from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.grouping import prompt_group_ids
from nemo_rl.data_plane.privilege_prefix import row_privilege_keys
from nemo_rl.data_plane.schema import (
    PRIVILEGE_KEY_TAG,
    PRIVILEGE_PREFIX_LEN_TAG,
    PROMPT_IDX_TAG,
)
from nemo_rl.models.value.config import PrivilegedValueConfig
from nemo_rl.utils.path_utils import load_checkpoint_object, path_exists

PrivilegePrefixBuilder = Callable[
    [dict[str, Any], Any, PrivilegedValueConfig], tuple[str, dict[str, Any]]
]

# The prefix cache, saved with a checkpointed replay buffer so the restored groups
# keep their reference blocks.
PRIVILEGE_PREFIXES_FILENAME = "privilege_prefixes.pt"

# Markup / chat-template slack on top of the block's token budget.
_TEMPLATE_SLACK_TOKENS = 256

# What a prefix builder's stats must carry; step_metrics reads all three.
_STATS_KEYS = ("wanted_tokens", "kept_tokens", "truncated_fields")


def privilege_budget_tokens(cfg: PrivilegedValueConfig) -> int:
    """Upper bound on the tokens the reference block adds to a critic sequence."""
    return cfg.max_total_tokens + _TEMPLATE_SLACK_TOKENS


def _only_group_id(meta: KVBatchMeta) -> str:
    group_ids = prompt_group_ids(meta)
    if len(group_ids) != 1:
        raise ValueError(
            "privileged value: expected the rows of one prompt group, got "
            f"{len(group_ids)} group(s)."
        )
    return group_ids[0]


class PrivilegePrefixStore:
    """Controller-side cache of one tokenized reference block per prompt.

    Installed as a post-write enricher of the replay buffer, so every group is
    stamped before it becomes selectable. Sibling rollouts of a prompt share one
    byte-identical prefix: the privilege is constant within a prompt group, so
    it cannot introduce a within-group confound. A prefix is dropped once no
    committed-but-untrained group references it.
    """

    def __init__(self, tokenizer: Any, cfg: PrivilegedValueConfig) -> None:
        self._tokenizer = tokenizer
        self._cfg = cfg
        self._build_prefix: PrivilegePrefixBuilder = get_method(cfg.prefix_builder_fqn)
        self._prefixes: dict[str, torch.Tensor] = {}
        self._stats: dict[str, dict[str, Any]] = {}
        self._group_keys: dict[str, str] = {}
        self._refcounts: dict[str, int] = {}

    def _render(
        self, key: str, extra_env_info: dict[str, Any]
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        """Build one prompt's block and tokenize it as a system-role chat turn."""
        block, stats = self._build_prefix(extra_env_info, self._tokenizer, self._cfg)
        if not block:
            raise ValueError(
                f"privileged value: the prefix builder returned an empty block for "
                f"{key!r}. Training on would silently produce a blind critic."
            )
        missing = [name for name in _STATS_KEYS if name not in stats]
        if missing:
            raise ValueError(
                f"privileged value: the prefix builder "
                f"{self._cfg.prefix_builder_fqn!r} returned stats without {missing}."
            )
        rendered = self._tokenizer.apply_chat_template(
            [{"role": "system", "content": block}],
            tokenize=False,
            add_generation_prompt=False,
            add_special_tokens=False,
        )
        prefix = self._tokenizer(
            rendered, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0].to(dtype=torch.int32)
        budget = privilege_budget_tokens(self._cfg)
        if prefix.numel() > budget:
            raise ValueError(
                f"privileged value: the reference block for {key!r} renders to "
                f"{prefix.numel()} tokens, over the {budget}-token budget "
                "(max_total_tokens plus template slack) the critic sequence "
                "lengths were validated for."
            )
        return prefix, stats

    async def enrich(
        self, meta: KVBatchMeta, extra_env_info: Optional[dict[str, Any]]
    ) -> KVBatchMeta:
        """Cache the group's prefix and stamp every row tag with its key and length.

        The replay buffer runs this on both commit paths: ``commit`` passes the
        prompt's ``extra_env_info``, and the controller passes it to
        ``commit_finalized`` for token-capture groups.

        Raises:
            ValueError: The builder returned an empty block or incomplete stats,
                or the rendered block exceeds ``privilege_budget_tokens``, so
                critic rows could outgrow the validated sequence lengths.
        """
        group_id = _only_group_id(meta)
        assert meta.tags is not None  # _only_group_id read the group from them
        key = f"prompt{meta.tags[0][PROMPT_IDX_TAG]}"
        if key not in self._prefixes:
            self._prefixes[key], self._stats[key] = await asyncio.to_thread(
                self._render, key, extra_env_info or {}
            )
        if group_id not in self._group_keys:
            self._group_keys[group_id] = key
            self._refcounts[key] = self._refcounts.get(key, 0) + 1
        prefix_len = int(self._prefixes[key].numel())
        tags = [
            {**tag, PRIVILEGE_KEY_TAG: key, PRIVILEGE_PREFIX_LEN_TAG: prefix_len}
            for tag in meta.tags
        ]
        return replace(meta, tags=tags)

    def prefixes_for(self, meta: KVBatchMeta) -> dict[str, torch.Tensor]:
        """The prefixes the rows of ``meta`` reference, keyed as in their tags."""
        return {key: self._prefixes[key] for key in row_privilege_keys(meta)}

    def step_metrics(self, metas: Sequence[KVBatchMeta]) -> dict[str, float]:
        """``privilege/*`` metrics over the unique prompts a step trained on.

        Reported per prompt, not per rollout: the block is byte-identical
        across a group's siblings, so rollout-weighting would restate group size.
        """
        keys = dict.fromkeys(key for meta in metas for key in row_privilege_keys(meta))
        stats = [self._stats[key] for key in keys]
        if not stats:
            return {}
        n = len(stats)
        kept = [s["kept_tokens"] for s in stats]
        wanted = [s["wanted_tokens"] for s in stats]
        metrics = {
            "privilege/frac_truncated": sum(bool(s["truncated_fields"]) for s in stats)
            / n,
            "privilege/block_tokens_mean": sum(kept) / n,
            "privilege/block_tokens_max": float(max(kept)),
            "privilege/dropped_tokens_mean": sum(
                max(w - k, 0) for w, k in zip(wanted, kept)
            )
            / n,
            "privilege/wanted_tokens_mean": sum(wanted) / n,
            "privilege/n_instances": float(n),
        }
        for name in sorted({name for s in stats for name in s["truncated_fields"]}):
            metrics[f"privilege/frac_truncated_{name}"] = (
                sum(name in s["truncated_fields"] for s in stats) / n
            )
        return metrics

    def release(self, metas: Sequence[KVBatchMeta]) -> None:
        """Drop the references of the groups ``metas`` belong to (they trained)."""
        self._release_groups(
            group_id for meta in metas for group_id in prompt_group_ids(meta)
        )

    def _release_groups(self, group_ids: Iterable[str]) -> None:
        """Drop the references of ``group_ids``; a prefix goes with its last one."""
        for group_id in dict.fromkeys(group_ids):
            key = self._group_keys.pop(group_id, None)
            if key is None:
                continue
            self._refcounts[key] -= 1
            if self._refcounts[key] == 0:
                del self._refcounts[key]
                del self._prefixes[key]
                del self._stats[key]

    def retain(self, live_group_ids: Collection[str]) -> None:
        """Drop the references of every group not in ``live_group_ids``.

        For groups that leave the replay buffer untrained, e.g. evicted as stale:
        eviction reports only a count, so the caller passes the groups that can
        still train (the buffer's plus those the open step selected).
        """
        self._release_groups(
            [
                group_id
                for group_id in self._group_keys
                if group_id not in live_group_ids
            ]
        )

    def state_dict(self) -> dict[str, Any]:
        """The cached prefixes and their stats, saved next to the replay buffer."""
        return {
            "prefixes": dict(self._prefixes),
            "stats": {key: dict(stats) for key, stats in self._stats.items()},
        }

    def restore(self, state: dict[str, Any], metas: Sequence[KVBatchMeta]) -> None:
        """Re-cache the prefixes the restored replay groups reference.

        References are rebuilt from the restored groups' row tags rather than
        saved: whatever the checkpoint caught mid-release, every restored group
        holds exactly one reference and nothing else does.

        Raises:
            ValueError: A restored group references a prefix the checkpoint
                lacks, or a saved prefix is longer than this run's budget (the
                run resumed with a smaller ``max_total_tokens``), which the
                startup sequence-length checks did not size for.
        """
        budget = privilege_budget_tokens(self._cfg)
        self._prefixes.clear()
        self._stats.clear()
        self._group_keys.clear()
        self._refcounts.clear()
        for meta in metas:
            keys = set(row_privilege_keys(meta))
            if len(keys) != 1:
                raise ValueError(
                    "privileged value: a restored group references "
                    f"{len(keys)} prefixes; expected one per group."
                )
            (key,) = keys
            if key not in state["prefixes"]:
                raise ValueError(
                    f"privileged value: restored group references prefix {key!r}, "
                    "which the checkpoint does not contain."
                )
            prefix = state["prefixes"][key]
            if prefix.numel() > budget:
                raise ValueError(
                    f"privileged value: restored prefix {key!r} has "
                    f"{prefix.numel()} tokens, over this run's budget of {budget} "
                    "(max_total_tokens plus template slack). Resume with the "
                    "max_total_tokens the checkpoint was written with."
                )
            self._prefixes[key] = prefix
            self._stats[key] = state["stats"][key]
            group_id = _only_group_id(meta)
            if group_id not in self._group_keys:
                self._group_keys[group_id] = key
                self._refcounts[key] = self._refcounts.get(key, 0) + 1

    def restore_from_checkpoint(
        self, checkpoint_path: str, metas: Sequence[KVBatchMeta]
    ) -> None:
        """Load the checkpoint's prefix cache for the restored replay groups ``metas``.

        Raises:
            FileNotFoundError: The checkpoint has no prefix cache.
        """
        path = os.path.join(checkpoint_path, PRIVILEGE_PREFIXES_FILENAME)
        if not path_exists(path):
            raise FileNotFoundError(
                f"Restored {len(metas)} replay group(s) for the privileged value, "
                f"but the checkpoint has no {PRIVILEGE_PREFIXES_FILENAME}."
            )
        # weights_only=False: prefix tensors plus plain stats dicts, a trusted
        # same-job artifact like the replay metadata.
        self.restore(load_checkpoint_object(path, weights_only=False), metas)
        print(
            f"📦 Restored {len(self._prefixes)} privileged-value prefix(es) for "
            f"{len(metas)} restored replay group(s)",
            flush=True,
        )
