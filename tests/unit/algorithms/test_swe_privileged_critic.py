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
import json

import pytest
import torch

from nemo_rl.algorithms import swe_privileged_critic as spc
from nemo_rl.data_plane import KVBatchMeta
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


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


def _env_info(instance_id="repo__1", patch="diff --git a/x b/x\n+fix"):
    metadata = {"instance_id": instance_id, "FAIL_TO_PASS": ["t::a"]}
    if patch:
        metadata["patch"] = patch
    return {"responses_create_params": {"metadata": metadata}}


def _meta(group_id, n, lengths):
    return KVBatchMeta(
        partition_id="p",
        task_name="train",
        sample_ids=[f"{group_id}_g{i}" for i in range(n)],
        fields=["input_ids"],
        sequence_lengths=list(lengths),
        extra_info={},
        tags=[{"weight_version": 0} for _ in range(n)],
    )


def _store(max_total_tokens=64):
    return spc.SwePrivilegePrefixStore(
        _CharTokenizer(),
        spc.SwePrivilegedCriticConfig(enabled=True, max_total_tokens=max_total_tokens),
    )


def test_resolve_config_off_by_default():
    assert spc.resolve_config({}) is None
    assert spc.resolve_config({"swe_privileged_critic": {"enabled": False}}) is None
    cfg = spc.resolve_config(
        {"swe_privileged_critic": {"enabled": True, "max_total_tokens": 100}}
    )
    assert cfg.max_total_tokens == 100
    assert spc.privilege_budget_tokens(cfg) == 356


def test_resolve_config_rejects_unknown_keys():
    with pytest.raises(ValueError):
        spc.resolve_config({"swe_privileged_critic": {"enabled": True, "typo": 1}})


def test_render_prefix_is_a_system_turn_holding_the_reference_block():
    cfg = spc.SwePrivilegedCriticConfig(enabled=True, max_total_tokens=10_000)
    fields = spc.resolve_privilege_fields(_env_info())
    ids, stats = spc.render_reference_prefix(fields, _CharTokenizer(), cfg)
    text = "".join(chr(int(i)) for i in ids)
    assert ids.dtype == torch.int32
    assert text.startswith("<system><reference>")
    assert "<golden_patch>\ndiff --git a/x b/x\n+fix\n</golden_patch>" in text
    assert "<fail_to_pass>\nt::a\n</fail_to_pass>" in text
    assert stats["truncated"] is False


def test_enrich_stamps_rows_and_shares_one_prefix_per_instance():
    store = _store()
    meta_a = asyncio.run(store.enrich(_meta("ga", 2, [5, 7]), _env_info()))
    meta_b = asyncio.run(store.enrich(_meta("gb", 1, [3]), _env_info()))

    assert len(store) == 1
    for meta in (meta_a, meta_b):
        for tag in meta.tags:
            assert tag["weight_version"] == 0
            assert tag[spc.PRIVILEGE_KEY_TAG] == "repo__1"
            assert tag[spc.PRIVILEGE_PREFIX_LEN_TAG] == len(
                store.prefixes_for(meta)["repo__1"]
            )

    store.release([meta_a])
    assert len(store) == 1  # gb still references the instance
    store.release([meta_b])
    assert len(store) == 0


def test_restore_rebuilds_the_cache_for_the_restored_groups_only():
    # Saved with a checkpointed replay buffer. After restore, each restored group
    # holds one reference, and prefixes no restored group references are dropped.
    store = _store()
    meta_a = asyncio.run(store.enrich(_meta("ga", 2, [5, 7]), _env_info("a")))
    meta_b = asyncio.run(store.enrich(_meta("gb", 1, [3]), _env_info("a")))
    asyncio.run(store.enrich(_meta("gc", 1, [3]), _env_info("c")))
    state = store.state_dict()

    restored = _store()
    restored.restore(state, [meta_a, meta_b])

    assert len(restored) == 1
    assert torch.equal(
        restored.prefixes_for(meta_a)["a"], store.prefixes_for(meta_a)["a"]
    )
    assert restored.step_metrics([meta_a]) == store.step_metrics([meta_a])
    restored.release([meta_a])
    assert len(restored) == 1  # gb still references instance a
    restored.release([meta_b])
    assert len(restored) == 0


def test_restore_fails_loudly_when_a_restored_group_has_no_saved_prefix():
    store = _store()
    meta = asyncio.run(store.enrich(_meta("ga", 1, [4]), _env_info("a")))

    with pytest.raises(ValueError, match="does not contain"):
        _store().restore({"prefixes": {}, "stats": {}}, [meta])


def test_enrich_fails_loudly_without_a_golden_patch():
    store = _store()
    with pytest.raises(ValueError, match="no golden patch"):
        asyncio.run(store.enrich(_meta("ga", 1, [4]), _env_info(patch="")))


def _sandbox_row(instance_id="repo__2", patch="diff --git a/y b/y\n+fix2"):
    """A Gym sandboxed-agent row (e.g. swe_rebench): instance fields at top level."""
    return {
        "instance_id": instance_id,
        "patch": patch,
        "test_patch": "diff --git a/t b/t\n+test",
        "FAIL_TO_PASS": ["t::b"],
        "PASS_TO_PASS": ["t::c"],
        "responses_create_params": {"input": [{"role": "user", "content": "fix"}]},
        "agent_ref": {"type": "responses_api_agents", "name": "agent"},
    }


def test_sandboxed_agent_rows_resolve_from_the_row_top_level():
    fields = spc.resolve_privilege_fields(_sandbox_row())
    assert fields["instance_id"] == "repo__2"
    assert fields["golden_patch"] == "diff --git a/y b/y\n+fix2"
    assert fields["test_patch"] == "diff --git a/t b/t\n+test"
    assert fields["fail_to_pass"] == "t::b"
    assert fields["pass_to_pass"] == "t::c"


def test_metadata_still_wins_over_the_row_top_level():
    row = {**_env_info(), "instance_id": "other", "patch": "top-level patch"}
    fields = spc.resolve_privilege_fields(row)
    assert fields["instance_id"] == "repo__1"
    assert fields["golden_patch"] == "diff --git a/x b/x\n+fix"


def test_enrich_stamps_a_sandboxed_agent_row():
    meta = asyncio.run(_store().enrich(_meta("ga", 2, [5, 7]), _sandbox_row()))
    assert [tag[spc.PRIVILEGE_KEY_TAG] for tag in meta.tags] == ["repo__2", "repo__2"]


def _tagged_meta(group_id, prompt_idx):
    meta = _meta(group_id, 2, [5, 7])
    for tag in meta.tags:
        tag["prompt_idx"] = prompt_idx
    return meta


def test_enrich_without_instance_id_keys_the_prefix_by_prompt():
    store = _store()
    env_info = _env_info(instance_id="")
    meta_a = asyncio.run(store.enrich(_tagged_meta("ga", 3), env_info))
    meta_b = asyncio.run(store.enrich(_tagged_meta("gb", 3), env_info))

    # Groups of one prompt share its prefix, as they do under an instance id.
    assert meta_a.tags[0][spc.PRIVILEGE_KEY_TAG] == "__prompt3"
    assert meta_b.tags[0][spc.PRIVILEGE_KEY_TAG] == "__prompt3"
    assert len(store) == 1


def test_enrich_without_instance_id_or_prompt_tag_keys_the_prefix_by_group():
    store = _store()
    env_info = _env_info(instance_id="")
    meta_a = asyncio.run(store.enrich(_meta("ga", 1, [4]), env_info))
    meta_b = asyncio.run(store.enrich(_meta("gb", 1, [4]), env_info))

    assert meta_a.tags[0][spc.PRIVILEGE_KEY_TAG] == "__groupga"
    assert meta_b.tags[0][spc.PRIVILEGE_KEY_TAG] == "__groupgb"


def test_step_metrics_are_per_instance():
    store = _store(max_total_tokens=8)  # forces truncation
    metas = [
        asyncio.run(store.enrich(_meta("ga", 4, [2] * 4), _env_info("a"))),
        asyncio.run(store.enrich(_meta("gb", 4, [2] * 4), _env_info("b"))),
    ]
    metrics = store.step_metrics(metas)
    assert metrics["privilege/n_instances"] == 2.0
    assert metrics["privilege/frac_truncated"] == 1.0


def test_prepend_and_unshift_round_trip():
    meta = _meta("g", 2, [3, 2])
    meta.tags[0].update({spc.PRIVILEGE_KEY_TAG: "a", spc.PRIVILEGE_PREFIX_LEN_TAG: 2})
    meta.tags[1].update({spc.PRIVILEGE_KEY_TAG: "b", spc.PRIVILEGE_PREFIX_LEN_TAG: 1})
    prefixes = {
        "a": torch.tensor([90, 91], dtype=torch.int32),
        "b": torch.tensor([80], dtype=torch.int32),
    }
    width = 6
    data = BatchedDataDict(
        {
            "input_ids": torch.tensor([[1, 2, 3, 0, 0, 0], [4, 5, 0, 0, 0, 0]]),
            "input_lengths": torch.tensor([3, 2]),
            "token_mask": torch.tensor([[0, 1, 1, 0, 0, 0], [0, 1, 0, 0, 0, 0]]),
            "returns": torch.tensor([[0.0, 7.0, 7.0, 0, 0, 0], [0.0, 5.0, 0, 0, 0, 0]]),
        }
    )

    out = spc.prepend_privilege_prefix(data, meta, prefixes, shift_fields=("returns",))

    assert out["input_ids"][0, :5].tolist() == [90, 91, 1, 2, 3]
    assert out["input_ids"][1, :3].tolist() == [80, 4, 5]
    assert out["input_lengths"].tolist() == [5, 3]
    assert out["token_mask"][0].tolist() == [0, 0, 0, 1, 1, 0]
    assert out["token_mask"][1].tolist() == [0, 0, 1, 0, 0, 0]
    assert out["returns"][0].tolist() == [0, 0, 0, 7, 7, 0]
    assert spc.critic_view_meta(meta).sequence_lengths == [5, 3]
    assert spc.policy_view_meta(spc.critic_view_meta(meta)).sequence_lengths == [3, 2]

    critic_values = torch.arange(2 * width, dtype=torch.float32).reshape(2, width)
    back = spc.values_to_policy_layout(critic_values, meta, torch.tensor([3, 2]))
    assert back[0].tolist() == [2, 3, 4, 0, 0, 0]
    assert back[1].tolist() == [7, 8, 0, 0, 0, 0]


def test_prepend_rejects_a_batch_padded_for_policy_lengths():
    meta = _meta("g", 1, [4])
    meta.tags[0].update({spc.PRIVILEGE_KEY_TAG: "a", spc.PRIVILEGE_PREFIX_LEN_TAG: 3})
    data = BatchedDataDict(
        {
            "input_ids": torch.ones(1, 4, dtype=torch.long),
            "input_lengths": torch.tensor([4]),
            "token_mask": torch.ones(1, 4, dtype=torch.long),
        }
    )
    with pytest.raises(ValueError, match="pad target"):
        spc.prepend_privilege_prefix(
            data, meta, {"a": torch.zeros(3, dtype=torch.int32)}
        )


def test_untagged_rows_are_rejected():
    with pytest.raises(ValueError, match="no privilege tag"):
        spc.critic_view_meta(_meta("g", 1, [4]))


def test_r2e_gym_commit_is_reassembled_into_a_patch():
    commit = {
        "file_diffs": [
            {
                "header": {"file": {"path": "pkg/mod.py"}},
                "hunks": [
                    {
                        "descriptor": {
                            "old_range": {"start": 1, "length": 1},
                            "new_range": {"start": 1, "length": 1},
                        },
                        "line_group": {
                            "all_lines": [
                                {"type": "deleted", "content": "a = 1"},
                                {"type": "added", "content": "a = 2"},
                            ]
                        },
                    }
                ],
            }
        ]
    }
    info = {
        "responses_create_params": {
            "metadata": {
                "instance_id": "r2e__1",
                "parsed_commit_content": json.dumps(commit),
            }
        }
    }
    fields = spc.resolve_privilege_fields(info)
    assert "-a = 1\n+a = 2" in fields["golden_patch"]


@pytest.mark.parametrize("caps", [{"gold_patch": 8192}, {"golden_patch": 0}])
def test_resolve_config_rejects_bad_caps(caps):
    with pytest.raises(ValueError):
        spc.resolve_config({"swe_privileged_critic": {"enabled": True, "caps": caps}})


def test_caps_bound_a_field_within_the_total_budget():
    cfg = spc.SwePrivilegedCriticConfig(
        enabled=True, max_total_tokens=1000, caps={"golden_patch": 40}
    )
    fields = {"golden_patch": "g" * 100, "fail_to_pass": "t::b"}

    block, stats = spc.build_reference_block(fields, _CharTokenizer(), cfg)

    assert stats["truncated_fields"] == ["golden_patch"]
    golden = block.split("<golden_patch>\n", 1)[1].split("\n</golden_patch>", 1)[0]
    assert len(golden) == 40 and golden.endswith(spc.TRUNCATION_MARKER)


def test_a_field_cut_down_to_the_marker_still_counts_as_truncated():
    # pass_to_pass (6 tokens) gets 3 tokens of budget: it is replaced by the
    # 16-token marker, which outweighs the original.
    fields = {
        "fail_to_pass": "t::b",
        "golden_patch": "g" * 10,
        "pass_to_pass": "t::abc",
    }
    cfg = spc.SwePrivilegedCriticConfig(enabled=True, max_total_tokens=17)

    block, stats = spc.build_reference_block(fields, _CharTokenizer(), cfg)

    assert stats["truncated_fields"] == ["pass_to_pass"]
    assert "[truncated: pass_to_pass]" in block


def test_retain_drops_the_groups_that_can_no_longer_train():
    store = _store()
    asyncio.run(store.enrich(_meta("ga", 1, [4]), _env_info("a")))
    asyncio.run(store.enrich(_meta("gb", 1, [4]), _env_info("a")))
    asyncio.run(store.enrich(_meta("gc", 1, [4]), _env_info("c")))

    store.retain({"gb"})

    # Instance a survives through gb; c lost its only group.
    assert len(store) == 1
    store.release_groups(["gb", "unknown"])
    assert len(store) == 0
