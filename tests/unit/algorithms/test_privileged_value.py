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
import asyncio

import pytest
import torch

from nemo_rl.algorithms.ppo_utils import privileged_value as pv
from nemo_rl.algorithms.ppo_utils.privilege_prefix_utils import cap_tokens
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.data_plane.schema import (
    GROUP_ID_TAG,
    PRIVILEGE_KEY_TAG,
    PRIVILEGE_PREFIX_LEN_TAG,
    PROMPT_IDX_TAG,
)
from nemo_rl.models.value.config import PrivilegedValueConfig


class _CharTokenizer:
    """One token per character; enough to exercise the block/prefix plumbing."""

    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)

    def apply_chat_template(self, messages, **kwargs):
        return "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        return {"input_ids": torch.tensor([self.encode(text)])}


def _answer_builder(extra_env_info, tokenizer, cfg):
    """Fake prefix builder: the prompt's answer, cut to the budget."""
    answer = extra_env_info.get("answer", "")
    block, truncated = cap_tokens(answer, cfg.max_total_tokens, tokenizer)
    stats = {
        "truncated_fields": ["answer"] if truncated else [],
        "wanted_tokens": len(answer),
        "kept_tokens": len(block),
    }
    return block, stats


def _statless_builder(extra_env_info, tokenizer, cfg):
    """Fake prefix builder whose stats lack ``kept_tokens``."""
    return "block", {"truncated_fields": [], "wanted_tokens": 5}


def _config(max_total_tokens=64, builder="_answer_builder"):
    return PrivilegedValueConfig(
        enabled=True,
        max_total_tokens=max_total_tokens,
        prefix_builder_fqn=f"{__name__}.{builder}",
    )


def _store(max_total_tokens=64, tokenizer=None):
    return pv.PrivilegePrefixStore(
        tokenizer or _CharTokenizer(), _config(max_total_tokens)
    )


def _meta(group_id, n, lengths, prompt_idx=0):
    tag = {"weight_version": 0, GROUP_ID_TAG: group_id, PROMPT_IDX_TAG: prompt_idx}
    return KVBatchMeta(
        partition_id="p",
        task_name="train",
        sample_ids=[f"{group_id}_g{i}" for i in range(n)],
        fields=["input_ids"],
        sequence_lengths=list(lengths),
        extra_info={},
        tags=[dict(tag) for _ in range(n)],
    )


def _cached(store):
    return len(store.state_dict()["prefixes"])


def test_enrich_renders_the_block_as_a_system_turn():
    store = _store()
    meta = asyncio.run(
        store.enrich(_meta("ga", 1, [4], prompt_idx=3), {"answer": "42"})
    )

    prefix = store.prefixes_for(meta)["prompt3"]
    assert prefix.dtype == torch.int32
    assert "".join(chr(int(i)) for i in prefix) == "<system>42</system>"


def test_enrich_stamps_rows_and_shares_one_prefix_per_prompt():
    store = _store()
    env_info = {"answer": "42"}
    meta_a = asyncio.run(store.enrich(_meta("ga", 2, [5, 7], prompt_idx=3), env_info))
    meta_b = asyncio.run(store.enrich(_meta("gb", 1, [3], prompt_idx=3), env_info))

    assert _cached(store) == 1
    for meta in (meta_a, meta_b):
        for tag in meta.tags:
            assert tag["weight_version"] == 0
            assert tag[PRIVILEGE_KEY_TAG] == "prompt3"
            assert tag[PRIVILEGE_PREFIX_LEN_TAG] == len(
                store.prefixes_for(meta)["prompt3"]
            )

    store.release([meta_a])
    assert _cached(store) == 1  # gb still references the prompt
    store.release([meta_b])
    assert _cached(store) == 0


def test_enrich_fails_loudly_on_an_empty_block():
    store = _store()
    with pytest.raises(ValueError, match="empty block"):
        asyncio.run(store.enrich(_meta("ga", 1, [4]), {}))
    assert _cached(store) == 0


def test_enrich_rejects_builder_stats_missing_a_key():
    store = pv.PrivilegePrefixStore(
        _CharTokenizer(), _config(builder="_statless_builder")
    )
    with pytest.raises(ValueError, match=r"_statless_builder.*kept_tokens"):
        asyncio.run(store.enrich(_meta("ga", 1, [4]), {}))
    assert _cached(store) == 0


def test_enrich_rejects_a_block_rendered_over_the_budget():
    # A template whose own overhead outgrows the slack would push critic rows
    # past the sequence lengths startup validated; fail when the prefix is built.
    class _VerboseTemplate(_CharTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            return "#" * 300 + super().apply_chat_template(messages, **kwargs)

    store = _store(tokenizer=_VerboseTemplate())

    with pytest.raises(ValueError, match="over the 320-token budget"):
        asyncio.run(store.enrich(_meta("ga", 1, [4]), {"answer": "4" * 10}))
    assert _cached(store) == 0


def test_restore_rebuilds_the_cache_for_the_restored_groups_only():
    # Saved with a checkpointed replay buffer. After restore, each restored group
    # holds one reference, and prefixes no restored group references are dropped.
    store = _store()
    meta_a = asyncio.run(
        store.enrich(_meta("ga", 2, [5, 7], prompt_idx=1), {"answer": "a"})
    )
    meta_b = asyncio.run(
        store.enrich(_meta("gb", 1, [3], prompt_idx=1), {"answer": "a"})
    )
    asyncio.run(store.enrich(_meta("gc", 1, [3], prompt_idx=2), {"answer": "c"}))
    state = store.state_dict()

    restored = _store()
    restored.restore(state, [meta_a, meta_b])

    assert _cached(restored) == 1
    assert torch.equal(
        restored.prefixes_for(meta_a)["prompt1"], store.prefixes_for(meta_a)["prompt1"]
    )
    assert restored.step_metrics([meta_a]) == store.step_metrics([meta_a])
    restored.release([meta_a])
    assert _cached(restored) == 1  # gb still references prompt 1
    restored.release([meta_b])
    assert _cached(restored) == 0


def test_restore_fails_loudly_when_a_restored_group_has_no_saved_prefix():
    store = _store()
    meta = asyncio.run(store.enrich(_meta("ga", 1, [4]), {"answer": "a"}))

    with pytest.raises(ValueError, match="does not contain"):
        _store().restore({"prefixes": {}, "stats": {}}, [meta])


def test_restore_rejects_a_prefix_over_the_resumed_budget():
    # Saved under max_total_tokens=64, resumed with 8: the saved prefix no longer
    # fits the sequence lengths startup validated for this run.
    store = _store()
    meta = asyncio.run(store.enrich(_meta("ga", 1, [4]), {"answer": "a"}))
    state = store.state_dict()
    budget = pv.privilege_budget_tokens(_config(max_total_tokens=8))
    state["prefixes"]["prompt0"] = torch.zeros(budget + 1, dtype=torch.int32)

    with pytest.raises(ValueError, match="over this run's budget"):
        _store(max_total_tokens=8).restore(state, [meta])


def test_release_resolves_forest_padding_rows_by_their_group_tag():
    # Forest rows name their group in dispatch_group_id; sample ids such as
    # "x_gfoo_pad0" cannot be parsed back to it reliably.
    def forest_meta(group_id):
        return KVBatchMeta(
            partition_id="p",
            task_name="train",
            sample_ids=[f"{group_id}_g0_s0", f"{group_id}_pad0"],
            fields=["input_ids"],
            sequence_lengths=[4, 1],
            tags=[
                {"dispatch_group_id": group_id, PROMPT_IDX_TAG: 1},
                {"dispatch_group_id": group_id, PROMPT_IDX_TAG: 1},
            ],
        )

    store = _store()
    owned = asyncio.run(store.enrich(forest_meta("x_gfoo"), {"answer": "a"}))
    other = asyncio.run(store.enrich(_meta("x", 1, [4], prompt_idx=2), {"answer": "c"}))

    store.release([owned])

    # Only x_gfoo's prefix goes; group "x" keeps its own.
    assert _cached(store) == 1
    store.release([other])
    assert _cached(store) == 0


def test_retain_drops_the_groups_that_can_no_longer_train():
    store = _store()
    asyncio.run(store.enrich(_meta("ga", 1, [4], prompt_idx=1), {"answer": "a"}))
    meta_b = asyncio.run(
        store.enrich(_meta("gb", 1, [4], prompt_idx=1), {"answer": "a"})
    )
    asyncio.run(store.enrich(_meta("gc", 1, [4], prompt_idx=2), {"answer": "c"}))

    store.retain({"gb"})

    # Prompt 1 survives through gb; prompt 2 lost its only group.
    assert _cached(store) == 1
    store.release([meta_b])
    assert _cached(store) == 0


def test_step_metrics_are_per_prompt_and_name_the_truncated_fields():
    store = _store(max_total_tokens=20)
    metas = [
        asyncio.run(
            store.enrich(_meta("ga", 4, [2] * 4, prompt_idx=1), {"answer": "a" * 30})
        ),
        asyncio.run(
            store.enrich(_meta("gb", 4, [2] * 4, prompt_idx=2), {"answer": "b" * 10})
        ),
    ]

    metrics = store.step_metrics(metas)

    assert metrics["privilege/n_instances"] == 2.0
    assert metrics["privilege/frac_truncated"] == 0.5
    assert metrics["privilege/frac_truncated_answer"] == 0.5
    assert metrics["privilege/wanted_tokens_mean"] == 20.0
    assert metrics["privilege/block_tokens_max"] == 20.0
    assert metrics["privilege/dropped_tokens_mean"] == 5.0
