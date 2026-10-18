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
"""Offline critic pretraining on rollout banks (nemo_rl/algorithms/sc_critic_pretrain.py)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from tensordict import TensorDict

from nemo_rl.algorithms import sc_critic_pretrain as cp
from nemo_rl.algorithms import swe_privileged_critic as spc
from nemo_rl.algorithms.advantage_estimator import (
    GAEConfig,
    GeneralizedAdvantageEstimator,
)
from nemo_rl.algorithms.loss import ClippedPGLossConfig
from nemo_rl.algorithms.single_controller_utils.rollout_bank import (
    TOKEN_FIELDS,
    BankGroup,
    BankGroupRef,
)


def _ref(group_id, *, instance_id=None, prompt_idx=0):
    return BankGroupRef(
        path=Path(f"/bank/{group_id}.pt"),
        step=1,
        group_id=group_id,
        prompt_idx=prompt_idx,
        instance_id=instance_id,
        rows=1,
        rollouts=1,
        reward_mean=0.0,
    )


def _env_info(instance_id):
    md = {"instance_id": instance_id, "patch": f"diff --git a/{instance_id}"}
    return {"responses_create_params": {"metadata": md}}


def _group(
    group_id,
    rollouts,
    *,
    rewards=None,
    subagent=(),
    env_masked=(),
    truncated=(),
    instance_id=None,
    length=5,
):
    """A banked group: one canonical row per rollout, then subagent rows."""
    rewards = rewards or [1.0] * rollouts
    rows = [(r, 0) for r in range(rollouts)] + [(r, 1) for r in subagent]
    lengths = [length + t for _, t in rows]
    return BankGroup(
        group_id=group_id,
        prompt_idx=0,
        instance_id=instance_id,
        extra_env_info=_env_info(instance_id) if instance_id else {},
        sample_ids=[
            f"{group_id}_g{r}" if t == 0 else f"{group_id}_g{r}_t{t}" for r, t in rows
        ],
        tags=[{"rollout_local_idx": r, "trace_in_rollout_idx": t} for r, t in rows],
        sequence_lengths=lengths,
        fields={
            "input_ids": [torch.arange(L) + 10 for L in lengths],
            "token_mask": [torch.tensor([0.0, 0.0] + [1.0] * (L - 2)) for L in lengths],
            "generation_logprobs": [torch.zeros(L) for L in lengths],
            "input_lengths": [torch.tensor(L) for L in lengths],
            "sample_mask": [torch.tensor(1.0) for _ in rows],
            "mask_sample": [torch.tensor(r in env_masked) for r, _ in rows],
            "truncated": [torch.tensor(r in truncated) for r, _ in rows],
            "total_reward": [torch.tensor(rewards[r]) for r, _ in rows],
        },
    )


# ── split / order ──────────────────────────────────────────────────────────


def test_heldout_split_keys_on_the_task_so_repeats_stay_together():
    refs = [
        _ref(f"g{i}", instance_id=f"repo__{i % 40}", prompt_idx=i) for i in range(400)
    ]
    train, heldout = cp.split_heldout(refs, heldout_mod=8)
    assert heldout and train
    assert not {r.task_key for r in heldout} & {r.task_key for r in train}
    assert cp.split_heldout(refs, heldout_mod=8) == (train, heldout)
    assert cp.split_heldout(refs, heldout_mod=0) == (refs, [])


def test_step_layout_matches_the_legacy_tiling():
    # Legacy stage B: gpp 32, GBS 1024, K 8, 4 donors -> 29 full groups.
    assert cp.step_layout(1024, 32, 8, 4) == (4, 29)
    assert cp.step_layout(1024, 32, 0, 0) == (0, 32)
    with pytest.raises(ValueError):
        cp.step_layout(1000, 32, 0, 0)
    with pytest.raises(ValueError):
        cp.step_layout(1024, 32, 8, 3)


def test_response_split_is_frozen_beside_the_banks(tmp_path):
    base = [str(tmp_path / f"g{i}.pt") for i in range(66)]
    kwargs = dict(
        withheld=8,
        donors_per_step=4,
        full_per_step=29,
        heldout_mod=32,
        seed=1,
        redraw=False,
    )
    donors = cp.load_or_create_response_split(tmp_path, base, **kwargs)
    assert len(donors) == round(66 * 4 / 33)
    saved = json.loads((tmp_path / cp.RESPONSE_SPLIT_NAME).read_text())
    assert saved["donors"] == sorted(Path(d).name for d in donors)
    assert cp.load_or_create_response_split(tmp_path, base, **kwargs) == donors
    with pytest.raises(ValueError, match="different settings"):
        cp.load_or_create_response_split(tmp_path, base, **{**kwargs, "seed": 2})
    with pytest.raises(ValueError, match="different settings"):
        cp.load_or_create_response_split(tmp_path, base[:-1], **kwargs)


def test_streams_reshuffle_each_epoch_and_pack_donors_first():
    base = [f"g{i}" for i in range(10)]
    stream = cp.build_epoch_stream(base, 2, seed=3)
    assert sorted(stream[:10]) == sorted(base) and stream[:10] != stream[10:]
    assert stream == cp.build_epoch_stream(base, 2, seed=3)
    # Raising the epoch count extends the stream; the prefix replays.
    assert cp.build_epoch_stream(base, 3, seed=3)[:20] == stream
    packed = cp.build_packed_epoch_stream(
        ["d0", "d1", "d2", "d3"], base, 2, 5, num_epochs=1, seed=0
    )
    assert len(packed) == 14
    assert all(p.startswith("d") for p in packed[0:2] + packed[7:9])


def test_slice_rollouts_keeps_a_rollouts_subagent_rows():
    group = _group("A", 4, subagent=(1, 3))
    tail = cp.slice_rollouts(group, 2, 4)
    assert tail.sample_ids == ["A_g2", "A_g3", "A_g3_t1"]
    assert [int(n) for n in tail.fields["input_lengths"]] == [5, 5, 6]
    head = cp.slice_rollouts(group, 0, 2)
    assert head.sample_ids == ["A_g0", "A_g1", "A_g1_t1"]


def test_config_validation():
    with pytest.raises(ValueError):
        cp.CriticPretrainConfig(bank_dirs=[])
    with pytest.raises(ValueError, match="both"):
        cp.CriticPretrainConfig(bank_dirs=["/b"], heldout_responses_per_group=8)
    with pytest.raises(ValueError):
        cp.CriticPretrainConfig(bank_dirs=["/b"], typo=1)


# ── masks / targets / metrics ──────────────────────────────────────────────


def test_loss_mask_trains_on_everything_only_when_asked():
    sample_mask = torch.tensor([1.0, 1.0, 1.0, 0.0])
    env = torch.tensor([False, True, False, False])
    trunc = torch.tensor([False, False, True, False])
    legacy_eval = cp.loss_sample_mask(
        sample_mask, env, trunc, apply_env_mask=True, apply_overlong_mask=True
    )
    assert legacy_eval.tolist() == [1.0, 0.0, 0.0, 0.0]
    everything = cp.loss_sample_mask(
        sample_mask, env, trunc, apply_env_mask=False, apply_overlong_mask=False
    )
    # The finalizer's placeholder row never trains.
    assert everything.tolist() == [1.0, 1.0, 1.0, 0.0]


def _gae():
    return GeneralizedAdvantageEstimator(
        GAEConfig(gae_lambda=1.0, gae_gamma=1.0), ClippedPGLossConfig()
    )


def test_returns_are_the_reward_on_every_response_token():
    token_mask = torch.tensor([[0, 1, 1], [0, 1, 1]], dtype=torch.float32)
    returns = cp.gae_returns(
        _gae(),
        token_mask=token_mask,
        sample_mask=torch.tensor([1.0, 0.0]),
        rewards=torch.tensor([1.0, 0.0]),
        values=torch.rand(2, 3),
    )
    torch.testing.assert_close(returns[0, 1:], torch.tensor([1.0, 1.0]))


def test_heldout_stats_legacy_metrics():
    stats = cp.HeldoutStats()
    # One group: two winners (values high late), two losers.
    token_mask = torch.tensor([[0, 1, 1, 1, 1]] * 4, dtype=torch.float32)
    values = torch.tensor(
        [
            [0, 0.6, 0.7, 0.8, 0.9],
            [0, 0.5, 0.6, 0.7, 0.8],
            [0, 0.4, 0.3, 0.2, 0.1],
            [0, 0.5, 0.4, 0.3, 0.2],
        ]
    )
    rewards = torch.tensor([1.0, 1.0, 0.0, 0.0])
    returns = rewards.unsqueeze(-1).expand(4, 5).clone()
    stats.add(
        values=values,
        returns=returns,
        token_mask=token_mask,
        sample_mask=torch.tensor([1.0, 1.0, 1.0, 0.0]),
        rewards=rewards,
        canonical=torch.tensor([True, True, True, True]),
        group_of_row=[0, 0, 0, 0],
    )
    # A second chunk: a homogeneous group, which within-group AUC ignores.
    stats.add(
        values=values[:2],
        returns=returns[:2],
        token_mask=token_mask[:2],
        sample_mask=torch.ones(2),
        rewards=rewards[:2],
        canonical=torch.tensor([True, True]),
        group_of_row=[0, 0],
    )
    m = stats.metrics()
    assert m["critic/terminal_auc"] == pytest.approx(1.0)
    assert m["critic/within_group_auc_q4"] == pytest.approx(1.0)
    assert m["critic/n_mixed_groups"] == 1.0
    assert m["num_heldout_samples"] == 6.0
    assert "critic/explained_var" in m and "critic/explained_var_masked" in m
    assert m["critic/explained_var_masked"] != m["critic/explained_var"]


def test_rank_auc_handles_ties_and_one_class():
    assert cp._rank_auc(torch.tensor([0.5, 0.5]), torch.tensor([True, False])) == 0.5
    assert torch.isnan(torch.tensor(cp._rank_auc(torch.ones(3), torch.ones(3))))


# ── one driver step against an in-memory data plane ────────────────────────


class _Plane:
    """Stores rows per field; jagged per-token columns come back jagged."""

    def __init__(self) -> None:
        self.rows: dict[str, dict[str, torch.Tensor]] = {}
        self.cleared: list[str] = []

    def put_samples(self, *, sample_ids, partition_id, fields, tags=None):
        del partition_id, tags
        for name in fields.keys():
            value = fields[name]
            rows = value.unbind() if value.is_nested else value.unbind(0)
            for sid, row in zip(sample_ids, rows):
                self.rows.setdefault(sid, {})[name] = row

    def get_samples(self, *, sample_ids, partition_id, select_fields):
        del partition_id
        out = {}
        for name in select_fields:
            rows = [self.rows[sid][name] for sid in sample_ids]
            if name in TOKEN_FIELDS or name in ("values", "returns"):
                out[name] = torch.nested.as_nested_tensor(rows, layout=torch.jagged)
            else:
                out[name] = torch.stack(rows)
        return TensorDict(out, batch_size=[len(sample_ids)])

    def clear_samples(self, *, sample_ids, partition_id):
        del partition_id
        self.cleared.extend(sample_ids)
        for sid in sample_ids:
            self.rows.pop(sid, None)


class _Value:
    """Writes a constant prediction, records the train call's view of the batch."""

    def __init__(
        self, plane: _Plane, prediction: float = 0.25, grad_norm_groups=None
    ) -> None:
        self.plane = plane
        self.prediction = prediction
        self.grad_norm_groups = grad_norm_groups
        self.sharding_annotations = SimpleNamespace(get_axis_size=lambda axis: 2)
        self.trained: dict | None = None
        self.calls: list[str] = []

    def prepare_for_inference(self):
        self.calls.append("prepare_for_inference")

    def finish_inference(self):
        self.calls.append("finish_inference")

    def prepare_for_training(self):
        self.calls.append("prepare_for_training")

    def finish_training(self):
        self.calls.append("finish_training")

    def get_values_from_meta(self, meta, privilege_prefixes=None):
        self.calls.append("get_values")
        for sid, n in zip(meta.sample_ids, meta.sequence_lengths):
            self.plane.rows[sid]["values"] = torch.full((n,), self.prediction)

    def train_from_meta(self, meta, loss_fn, gbs=None, privilege_prefixes=None):
        self.calls.append("train")
        self.trained = {
            "gbs": gbs,
            "sample_ids": list(meta.sample_ids),
            "returns": {
                sid: self.plane.rows[sid]["returns"] for sid in meta.sample_ids
            },
            "sample_mask": {
                sid: float(self.plane.rows[sid]["sample_mask"])
                for sid in meta.sample_ids
            },
            "prefix_keys": sorted(privilege_prefixes) if privilege_prefixes else None,
        }
        return {
            "loss": torch.tensor(0.1),
            "grad_norm": torch.tensor(1.0),
            "grad_norm_groups": self.grad_norm_groups or {},
            "all_mb_metrics": {
                "returns_mean": [0.5],
                "values_mean": [0.25],
                "returns_sq_mean": [0.5],
                "residual_sq_mean": [0.3],
            },
        }


class _CharTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [ord(c) for c in text]

    def decode(self, ids):
        return "".join(chr(i) for i in ids)

    def apply_chat_template(self, messages, **kwargs):
        return "".join(f"<{m['role']}>{m['content']}</{m['role']}>" for m in messages)

    def __call__(self, text, return_tensors=None, add_special_tokens=False):
        return {"input_ids": torch.tensor([self.encode(text)])}


def _trainer(plane, value, *, privileged=False, **cfg):
    master_config = SimpleNamespace(
        ppo=SimpleNamespace(overlong_filtering=True),
        value={"train_micro_batch_size": 1},
    )
    store = (
        spc.SwePrivilegePrefixStore(
            _CharTokenizer(),
            spc.SwePrivilegedCriticConfig(enabled=True, max_total_tokens=64),
        )
        if privileged
        else None
    )
    return cp._CriticPretrainer(
        master_config,
        cp.CriticPretrainConfig(bank_dirs=["/b"], **cfg),
        value=value,
        dp_client=plane,
        estimator=_gae(),
        loss_fn=object(),
        privilege_store=store,
        groups_per_step=2,
    )


def test_train_step_trains_on_everything_when_asked_and_cleans_up():
    plane = _Plane()
    value = _Value(plane)
    trainer = _trainer(plane, value, train_on_env_masked=True, train_on_truncated=True)
    groups = [
        _group("A", 2, subagent=(1,), env_masked=(0,)),
        _group("B", 1, rewards=[0.0], truncated=(0,)),
    ]
    metrics = trainer.train_step(groups)
    trainer.close()

    trained = value.trained
    # 4 real rows, already a multiple of the DP size (2 x micro batch 1).
    assert trained["gbs"] == 4
    assert set(trained["sample_mask"].values()) == {1.0}
    torch.testing.assert_close(
        trained["returns"]["s0-A_g0"][2:], torch.tensor([1.0, 1.0, 1.0])
    )
    assert value.calls == [
        "prepare_for_inference",
        "get_values",
        "finish_inference",
        "prepare_for_training",
        "train",
        "finish_training",
    ]
    assert plane.rows == {} and len(plane.cleared) == 4
    assert metrics["num_samples"] == 4.0 and metrics["num_rollouts"] == 3.0
    assert metrics["env_masked_rows"] == 1.0 and metrics["truncated_rows"] == 1.0
    assert metrics["trained_rows"] == 4.0


def test_train_step_logs_the_grad_norm_split_by_network_part():
    plane = _Plane()
    value = _Value(plane, grad_norm_groups={"attention": 3.0, "moe": 4.0})
    trainer = _trainer(plane, value)
    metrics = trainer.train_step([_group("A", 2)])
    trainer.close()

    assert metrics["critic/gnorm/attention"] == 3.0
    assert metrics["critic/gnorm/moe"] == 4.0
    # Shares of the total norm^2, which is what the global clip sees.
    assert metrics["critic/gnorm_frac/attention"] == pytest.approx(9 / 25)
    assert metrics["critic/gnorm_frac/moe"] == pytest.approx(16 / 25)


def test_train_step_masks_like_legacy_by_default():
    plane = _Plane()
    value = _Value(plane)
    trainer = _trainer(plane, value)
    groups = [
        _group("A", 2, env_masked=(0,)),
        _group("B", 1, rewards=[0.0], truncated=(0,)),
    ]
    metrics = trainer.train_step(groups)
    # Padded with one copy of row 0, which never trains.
    assert value.trained["sample_mask"] == {
        "s0-A_g0": 0.0,
        "s0-A_g1": 1.0,
        "s1-B_g0": 0.0,
        "s0-A_g0_pad0": 0.0,
    }
    assert metrics["trained_rows"] == 1.0 and metrics["pad_rows"] == 1.0


def test_privileged_prefixes_come_from_the_banked_env_info():
    plane = _Plane()
    value = _Value(plane)
    trainer = _trainer(plane, value, privileged=True)
    groups = [
        _group("A", 1, instance_id="repo__a"),
        _group("B", 1, instance_id="repo__b"),
    ]
    metrics = trainer.train_step(groups)
    assert value.trained["prefix_keys"] == ["repo__a", "repo__b"]
    assert metrics["privilege/n_instances"] == 2.0
    assert len(trainer.privilege_store) == 0


def test_banked_instance_must_match_its_env_info():
    plane = _Plane()
    trainer = _trainer(plane, _Value(plane), privileged=True)
    group = _group("A", 1, instance_id="repo__a")
    group.instance_id = "repo__other"
    with pytest.raises(ValueError, match="resolves to"):
        trainer.train_step([group, _group("B", 1, instance_id="repo__b")])


def test_evaluate_applies_the_legacy_masks_and_scores_groups():
    plane = _Plane()
    value = _Value(plane)
    trainer = _trainer(plane, value, train_on_env_masked=True)
    groups = [
        _group("A", 2, rewards=[1.0, 0.0], env_masked=(1,)),
        _group("B", 2, rewards=[0.0, 0.0]),
        _group("C", 1),
    ]
    m = trainer.evaluate(groups)
    assert m["num_heldout_samples"] == 5.0
    # Constant predictions: no ranking information.
    assert m["critic/within_group_auc_q1"] == pytest.approx(0.5)
    assert m["critic/n_mixed_groups"] == 1.0
    assert plane.rows == {}


class _Checkpointer:
    """CheckpointManager's checkpoint_dir + get_resume_paths, without the IO."""

    def __init__(self, checkpoint_dir: Path) -> None:
        self.checkpoint_dir = Path(checkpoint_dir)

    @staticmethod
    def get_resume_paths(path, *, model_component):
        component = Path(path) / model_component
        return component / "weights", component / "optimizer"


def test_rolling_resume_checkpoint_rotates_and_records_its_step(tmp_path):
    current, tmp = cp.resume_state_dirs(tmp_path)
    assert cp.rolling_resume_step(tmp_path) is None
    tmp.mkdir(parents=True)
    (tmp / "shard").write_text("step 20")
    cp.promote_rolling_resume(tmp_path, 20)
    assert cp.rolling_resume_step(tmp_path) == 20
    assert (current / "shard").read_text() == "step 20" and not tmp.exists()

    tmp.mkdir()
    (tmp / "shard").write_text("step 30")
    cp.promote_rolling_resume(tmp_path, 30)
    # Only one full checkpoint is ever kept.
    assert cp.rolling_resume_step(tmp_path) == 30
    assert (current / "shard").read_text() == "step 30"
    assert not tmp.exists() and not current.with_name("old").exists()


def test_resume_uses_the_rolling_full_checkpoint_only_for_the_latest_step(tmp_path):
    ckpt = _Checkpointer(tmp_path)
    current, tmp = cp.resume_state_dirs(tmp_path)
    tmp.mkdir(parents=True)
    cp.promote_rolling_resume(tmp_path, 30)

    weights, optimizer, source = cp.critic_resume_paths(
        ckpt, tmp_path / "step_30", 30, "latest_only"
    )
    assert source == str(current)
    assert weights == current / "value" / "weights"
    assert optimizer == current / "value" / "optimizer"

    # Crash between finalizing step_40 and promoting its rolling checkpoint: the
    # weights-only step_40 resumes (get_resume_paths finds no optimizer there).
    weights, _, source = cp.critic_resume_paths(
        ckpt, tmp_path / "step_40", 40, "latest_only"
    )
    assert source == str(tmp_path / "step_40")
    assert weights == tmp_path / "step_40" / "value" / "weights"

    # The default keeps resuming from step_N itself.
    _, _, source = cp.critic_resume_paths(
        ckpt, tmp_path / "step_30", 30, "every_checkpoint"
    )
    assert source == str(tmp_path / "step_30")
    assert cp.critic_resume_paths(ckpt, None, 0, "latest_only") == (None, None, None)


def test_optimizer_state_setting():
    assert cp.CriticPretrainConfig(bank_dirs=["b"]).optimizer_state == "every_checkpoint"
    latest = cp.CriticPretrainConfig(bank_dirs=["b"], optimizer_state="latest_only")
    assert latest.optimizer_state == "latest_only"
    with pytest.raises(ValueError):
        cp.CriticPretrainConfig(bank_dirs=["b"], optimizer_state="never")
