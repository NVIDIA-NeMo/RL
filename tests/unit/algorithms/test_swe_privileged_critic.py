import asyncio
import json
from types import SimpleNamespace

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


def _record(env_info, prompt_idx=0):
    return SimpleNamespace(extra_env_info=env_info, prompt_idx=prompt_idx)


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
    meta_a = asyncio.run(store.enrich(_meta("ga", 2, [5, 7]), _record(_env_info())))
    meta_b = asyncio.run(store.enrich(_meta("gb", 1, [3]), _record(_env_info())))

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
    meta_a = asyncio.run(store.enrich(_meta("ga", 2, [5, 7]), _record(_env_info("a"))))
    meta_b = asyncio.run(store.enrich(_meta("gb", 1, [3]), _record(_env_info("a"))))
    asyncio.run(store.enrich(_meta("gc", 1, [3]), _record(_env_info("c"))))
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
    meta = asyncio.run(store.enrich(_meta("ga", 1, [4]), _record(_env_info("a"))))

    with pytest.raises(ValueError, match="does not contain"):
        _store().restore({"prefixes": {}, "stats": {}}, [meta])


def test_enrich_fails_loudly_without_a_golden_patch():
    store = _store()
    with pytest.raises(ValueError, match="no golden patch"):
        asyncio.run(store.enrich(_meta("ga", 1, [4]), _record(_env_info(patch=""))))


def test_step_metrics_are_per_instance():
    store = _store(max_total_tokens=8)  # forces truncation
    metas = [
        asyncio.run(store.enrich(_meta("ga", 4, [2] * 4), _record(_env_info("a")))),
        asyncio.run(store.enrich(_meta("gb", 4, [2] * 4), _record(_env_info("b")))),
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
