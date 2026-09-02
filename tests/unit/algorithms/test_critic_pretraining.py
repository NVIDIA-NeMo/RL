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

from types import SimpleNamespace
from typing import Any

import pytest
import torch

import nemo_rl.algorithms.critic_pretraining as critic_pretraining
from nemo_rl.algorithms.critic_pretraining import (
    CriticPretrainingConfig,
    CriticSaveState,
    _aggregate_training_metrics,
    _minimum_megatron_sequence_divisor,
    critic_pretrain,
    resolve_critic_checkpoint,
    setup_evaluation,
)
from nemo_rl.distributed.batched_data_dict import BatchedDataDict


@pytest.mark.parametrize(
    ("tensor_parallel", "context_parallel", "sequence_parallel", "expected"),
    [
        (1, 1, False, 1),
        (8, 1, True, 8),
        (1, 4, False, 8),
        (8, 4, True, 64),
        (8, 4, False, 8),
    ],
)
def test_minimum_megatron_sequence_divisor(
    tensor_parallel: int,
    context_parallel: int,
    sequence_parallel: bool,
    expected: int,
) -> None:
    config = {
        "tensor_model_parallel_size": tensor_parallel,
        "context_parallel_size": context_parallel,
        "sequence_parallel": sequence_parallel,
    }

    assert _minimum_megatron_sequence_divisor(config) == expected


def test_aggregate_training_metrics_accepts_vector_loss() -> None:
    metrics = _aggregate_training_metrics(
        {
            "loss": torch.tensor([0.125, 0.25]),
            "grad_norm": torch.tensor([1.0]),
            "all_mb_metrics": {
                "returns_mean": [torch.tensor(0.5)],
                "values_mean": [torch.tensor(0.25)],
                "returns_sq_mean": [torch.tensor(0.25)],
                "residual_sq_mean": [torch.tensor(0.0625)],
                "global_valid_toks": [torch.tensor([8.0])],
            },
        },
        loss_scale=2.0,
    )

    assert metrics["loss"] == pytest.approx(0.375)
    assert metrics["mse"] == pytest.approx(0.375)
    assert metrics["rmse"] == pytest.approx(0.6123724)
    assert metrics["global_valid_toks"] == pytest.approx(8.0)


class _FakeValueModel:
    def __init__(self) -> None:
        self.events: list[str] = []

    def prepare_for_training(self) -> None:
        self.events.append("prepare")

    def train(self, *args: Any, **kwargs: Any) -> dict[str, Any]:
        self.events.append("train")
        return {
            "loss": torch.tensor(0.125),
            "grad_norm": torch.tensor(1.0),
            "all_mb_metrics": {
                "returns_mean": [torch.tensor(0.5)],
                "values_mean": [torch.tensor(0.25)],
                "returns_sq_mean": [torch.tensor(0.25)],
                "residual_sq_mean": [torch.tensor(0.0625)],
                "global_valid_toks": [torch.tensor(1.0)],
            },
        }

    def finish_training(self) -> None:
        self.events.append("finish")


class _FakeLogger:
    def log_metrics(self, *args: Any, **kwargs: Any) -> None:
        pass


def test_critic_keeps_model_resident_between_training_steps() -> None:
    value_model = _FakeValueModel()
    train_dataloader = [BatchedDataDict(sample_mask=torch.ones(1)) for _ in range(2)]
    master_config = SimpleNamespace(
        critic_pretraining=CriticPretrainingConfig(
            max_num_steps=2,
            max_num_epochs=1,
            val_at_end=False,
        ),
        checkpointing={
            "enabled": False,
            "checkpoint_must_save_by": None,
        },
        value={
            "train_global_batch_size": 1,
            "train_micro_batch_size": 1,
        },
        value_loss_fn=SimpleNamespace(scale=2.0),
        cluster={"num_nodes": 1, "gpus_per_node": 1},
    )

    critic_pretrain(
        value_model=value_model,  # type: ignore[arg-type]
        train_dataloader=train_dataloader,  # type: ignore[arg-type]
        val_dataloader={},
        loss_fn=object(),  # type: ignore[arg-type]
        master_config=master_config,  # type: ignore[arg-type]
        logger=_FakeLogger(),  # type: ignore[arg-type]
        checkpointer=object(),  # type: ignore[arg-type]
        save_state=CriticSaveState(
            epoch=0,
            step=0,
            total_steps=0,
            consumed_samples=0,
            total_valid_tokens=0,
        ),
    )

    assert value_model.events == ["prepare", "train", "train", "finish"]


def test_resolve_critic_checkpoint_accepts_step_and_weights_paths(tmp_path) -> None:
    checkpoint_path = tmp_path / "step_17"
    weights_path = checkpoint_path / "value" / "weights"
    weights_path.mkdir(parents=True)
    (checkpoint_path / "training_info.json").write_text(
        '{"total_steps": 17, "epoch": 2}\n', encoding="utf-8"
    )

    from_step = resolve_critic_checkpoint(checkpoint_path)
    from_weights = resolve_critic_checkpoint(weights_path)

    assert from_step == from_weights
    assert from_step.checkpoint_path == checkpoint_path
    assert from_step.weights_path == weights_path
    assert from_step.step == 17
    assert from_step.training_info == {"total_steps": 17, "epoch": 2}


def test_resolve_critic_checkpoint_rejects_inconsistent_step(tmp_path) -> None:
    checkpoint_path = tmp_path / "step_17"
    (checkpoint_path / "value" / "weights").mkdir(parents=True)
    (checkpoint_path / "training_info.json").write_text(
        '{"total_steps": 18}\n', encoding="utf-8"
    )

    with pytest.raises(ValueError, match="disagrees"):
        resolve_critic_checkpoint(checkpoint_path)


def test_setup_evaluation_loads_weights_without_optimizer(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checkpoint_path = tmp_path / "step_23"
    weights_path = checkpoint_path / "value" / "weights"
    weights_path.mkdir(parents=True)
    (checkpoint_path / "training_info.json").write_text(
        '{"total_steps": 23}\n', encoding="utf-8"
    )
    captured: dict[str, Any] = {}

    class _EvaluationLogger:
        def __init__(self, config: Any) -> None:
            captured["logger_config"] = config

        def log_hyperparams(self, parameters: dict[str, Any]) -> None:
            captured["hyperparameters"] = parameters

    class _EvaluationValue:
        def __init__(self, **kwargs: Any) -> None:
            captured["value_kwargs"] = kwargs

    monkeypatch.setattr(critic_pretraining, "validate_config", lambda config: None)
    monkeypatch.setattr(critic_pretraining, "set_seed", lambda seed: None)
    monkeypatch.setattr(critic_pretraining, "Logger", _EvaluationLogger)
    monkeypatch.setattr(
        critic_pretraining,
        "_trajectory_value_collate",
        lambda config, tokenizer: "collate",
    )
    monkeypatch.setattr(
        critic_pretraining,
        "_validation_dataloaders",
        lambda config, datasets, collate: {"test": "loader"},
    )
    monkeypatch.setattr(
        critic_pretraining,
        "_critic_cluster",
        lambda config, name: "cluster",
    )
    monkeypatch.setattr(critic_pretraining, "Value", _EvaluationValue)
    config = SimpleNamespace(
        critic_pretraining=SimpleNamespace(seed=42),
        logger={"log_dir": str(tmp_path / "logs")},
        value={"model_name": "model", "megatron_cfg": {"enabled": True}},
        model_dump=lambda: {"value": {"model_name": "model"}},
    )

    value_model, cluster, dataloaders, _, checkpoint, _ = setup_evaluation(
        config,  # type: ignore[arg-type]
        object(),  # type: ignore[arg-type]
        {"test": [1, 2]},  # type: ignore[arg-type]
        checkpoint_path,
    )

    assert isinstance(value_model, _EvaluationValue)
    assert cluster == "cluster"
    assert dataloaders == {"test": "loader"}
    assert checkpoint.step == 23
    assert captured["value_kwargs"]["weights_path"] == weights_path
    assert captured["value_kwargs"]["optimizer_path"] is None
    assert captured["value_kwargs"]["init_optimizer"] is False
    assert captured["value_kwargs"]["config"]["megatron_cfg"]["train_iters"] == 1
    assert captured["hyperparameters"]["evaluation_checkpoint"]["step"] == 23
