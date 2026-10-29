import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from nemo_rl.algorithms.critic_replay import CriticReplayBuffer, CriticReplayConfig


@pytest.mark.parametrize(
    "bad_field",
    [
        None,
        "gae_gamma",
        "gae_lambda_value",
        "cross_trajectory",
        "cliprange",
        "kl",
        "batch_size",
    ],
)
def test_production_config_guard(bad_field: str | None) -> None:
    path = Path(__file__).parents[3] / "nemo_rl/algorithms/ppo.py"
    tree = ast.parse(path.read_text())
    names = {"_critic_replay_config", "_resolve_critic_train_epochs"}
    nodes = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name in names
    ]
    namespace = {"CriticReplayConfig": CriticReplayConfig, "PPOConfig": dict}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    config = SimpleNamespace(
        ppo={
            "critic_replay": {"enabled": True},
            "critic_train_epochs": 4,
            "ppo_epochs": 2,
            "adv_estimator": {"name": "gae", "gae_gamma": 1.0, "gae_lambda": 1.0},
            "reward_shaping": {"enabled": False},
            "num_prompts_per_step": 128,
            "num_generations_per_prompt": 1,
        },
        value={"train_global_batch_size": 128},
        value_loss_fn=SimpleNamespace(cliprange=None),
        loss_fn=SimpleNamespace(
            use_kl_in_reward=False, reference_policy_kl_penalty=0.0
        ),
    )
    if bad_field in {"gae_gamma", "gae_lambda_value"}:
        config.ppo["adv_estimator"][bad_field] = 0.95
    elif bad_field == "cross_trajectory":
        config.ppo["adv_estimator"][bad_field] = True
    elif bad_field == "cliprange":
        config.value_loss_fn.cliprange = 0.2
    elif bad_field == "kl":
        config.loss_fn.use_kl_in_reward = True
        config.loss_fn.reference_policy_kl_penalty = 0.1
    elif bad_field == "batch_size":
        config.value["train_global_batch_size"] = 64
    if bad_field is None:
        assert namespace["_critic_replay_config"](config).enabled
    else:
        with pytest.raises(ValueError):
            namespace["_critic_replay_config"](config)
    config.ppo.pop("critic_replay")
    assert not namespace["_critic_replay_config"](config).enabled


def make_batch(
    start: int, lengths: tuple[int, ...] = (2, 4)
) -> dict[str, torch.Tensor]:
    ids = (
        torch.arange(start, start + len(lengths))
        .unsqueeze(1)
        .expand(-1, max(lengths))
        .clone()
    )
    return {
        "input_ids": ids,
        "input_lengths": torch.tensor(lengths),
        "token_mask": torch.ones_like(ids),
        "sample_mask": torch.tensor([1.0, 0.0]),
        "returns": ids.float() % 2,
        "advantages": ids.float() + 100,
    }


def test_fifo_owns_cpu_storage_and_preserves_masks() -> None:
    buffer = CriticReplayBuffer(CriticReplayConfig(capacity=4, batch_size=2))
    batch = make_batch(0)
    buffer.append(batch, step=0)
    batch["input_ids"].fill_(99)
    assert buffer.rows[0]["input_ids"].tolist() == [0, 0]
    assert "advantages" not in buffer.rows[0]
    buffer.append(make_batch(2), step=1)
    buffer.append(make_batch(4), step=2)
    assert [int(row["input_ids"][0]) for row in buffer.rows] == [2, 3, 4, 5]
    assert buffer.steps == [1, 1, 2, 2]
    assert buffer.rows[-1]["sample_mask"] == 0


def test_padding_sampling_and_resume(tmp_path: Path) -> None:
    config = CriticReplayConfig(capacity=4, batch_size=2, seed=19)
    buffer = CriticReplayBuffer(config)
    buffer.append(make_batch(0), step=0)
    batch, metrics = buffer.sample(step=0)
    assert sorted(batch["input_ids"][:, 0].tolist()) == [0, 1]
    index = batch["input_ids"][:, 0].tolist().index(0)
    assert batch["token_mask"][index].tolist() == [1, 1, 0, 0]
    assert batch["returns"][index, 2:].tolist() == [0, 0]
    assert metrics["sample_fresh_fraction"] == 1
    buffer.append(make_batch(2, (3, 5)), step=1)
    buffer.save(tmp_path)
    restored = CriticReplayBuffer(config)
    restored.restore(tmp_path)
    for _ in range(4):
        expected, expected_metrics = buffer.sample(step=2)
        actual, actual_metrics = restored.sample(step=2)
        assert actual_metrics == pytest.approx(expected_metrics, nan_ok=True)
        for key in expected:
            torch.testing.assert_close(actual[key], expected[key])


def test_bad_configuration_and_legacy_resume(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        CriticReplayConfig(capacity=1, batch_size=2)
    buffer = CriticReplayBuffer(CriticReplayConfig(capacity=4, batch_size=2))
    buffer.restore(tmp_path)
    assert not buffer.rows
    with pytest.raises(ValueError, match="fresh batch"):
        buffer.append(make_batch(0, (2,)), step=0)
    buffer.append(make_batch(0), step=0)
    buffer.save(tmp_path)
    with pytest.raises(ValueError, match="configuration mismatch"):
        CriticReplayBuffer(CriticReplayConfig(capacity=6, batch_size=2)).restore(
            tmp_path
        )


@pytest.mark.parametrize("function_name", ["ppo_train", "async_ppo_train"])
@pytest.mark.parametrize("enabled", [False, True])
def test_actual_ppo_update_loop(function_name: str, enabled: bool) -> None:
    """Execute the production critic block without importing GPU/Ray backends."""
    path = Path(__file__).parents[3] / "nemo_rl/algorithms/ppo.py"
    tree = ast.parse(path.read_text())
    function = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == function_name
    )
    loop = next(
        node
        for node in ast.walk(function)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "critic_epoch"
    )
    buffer = (
        CriticReplayBuffer(CriticReplayConfig(capacity=4, batch_size=2))
        if enabled
        else None
    )
    fresh = make_batch(2)
    original = {key: value.clone() for key, value in fresh.items()}
    if buffer is not None:
        buffer.append(make_batch(0), step=0)
        buffer.append(fresh, step=1)
    value_model = Mock()
    namespace = {
        "critic_train_epochs": 4,
        "critic_replay": buffer,
        "train_data": fresh,
        "critic_train_data": fresh,
        "total_steps": 1,
        "step": 1,
        "metrics": {},
        "BatchedDataDict": dict,
        "_compute_critic_metrics": lambda _: {"critic/loss": 0.5},
        "_probe_critic_replay_update": lambda *args, **kwargs: {},
        "timer": SimpleNamespace(time=lambda _: nullcontext()),
        "value_model": value_model,
        "value_loss_fn": object(),
        "critic_gbs": None,
        "cross_trajectory": False,
    }
    exec(
        compile(ast.Module(body=[loop], type_ignores=[]), str(path), "exec"), namespace
    )
    calls = value_model.train.call_args_list
    assert len(calls) == 4
    assert calls[0].args[0] is fresh
    for call in calls[1:]:
        if enabled:
            assert call.args[0] is not fresh
            assert "advantages" not in call.args[0]
        else:
            assert call.args[0] is fresh
    assert value_model.finish_training.call_count == 4
    if buffer is not None:
        assert buffer.metrics(step=1)["updates_completed"] == 4
        assert buffer.metrics(step=1)["fresh_updates_completed"] == 1
        assert buffer.metrics(step=1)["replay_updates_completed"] == 3
        assert buffer.metrics(step=1)["fresh_coverage"] == 1
    for key in fresh:
        torch.testing.assert_close(fresh[key], original[key])


def test_production_capacity_counters_and_resume(tmp_path: Path) -> None:
    config = CriticReplayConfig(enabled=True, capacity=512, batch_size=128)
    buffer = CriticReplayBuffer(config)
    for step in range(6):
        data = make_batch(0)
        data = {
            key: tensor.repeat((64,) + (1,) * (tensor.ndim - 1))
            for key, tensor in data.items()
        }
        buffer.append(data, step=step)
        buffer.record_update(fresh=True, step=step)
        for _ in range(3):
            buffer.sample(step=step)
            buffer.record_update(fresh=False, step=step)
        metrics = buffer.metrics(step=step)
        assert metrics["size"] == min(512, (step + 1) * 128)
        assert metrics["evicted_total"] == max(0, (step + 1) * 128 - 512)
        assert metrics["fresh_coverage"] == 1
        assert metrics["resident_updates_min"] >= 1
        assert metrics["trajectory_exposures"] == 512
        assert metrics["updates_total"] == (step + 1) * 4
    buffer.save(tmp_path)
    restored = CriticReplayBuffer(config)
    restored.restore(tmp_path)
    assert restored.metrics(step=5)["restored_size"] == 512
    assert restored.update_counts == buffer.update_counts
    assert restored.total_updates == 24
    assert restored.total_evicted == 256
