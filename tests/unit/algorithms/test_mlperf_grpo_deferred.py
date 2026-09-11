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

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from nemo_rl.algorithms import mlperf_grpo_deferred as deferred


def _config(enabled: bool = True, **grpo_overrides) -> dict[str, Any]:
    grpo: dict[str, Any] = {
        "val_start_at": 2,
        "max_num_steps": 3,
        "val_period": 1,
        "val_at_start": False,
        "val_at_end": False,
        "async_grpo": {"enabled": True},
        "num_generations_per_prompt": 16,
        "seed": 42,
    }
    grpo.update(grpo_overrides)
    if enabled:
        grpo["deferred_evaluation"] = {"enabled": True}
    return {
        "grpo": grpo,
        "policy": {
            "generation": {"backend": "vllm"},
            "megatron_cfg": {"enabled": True},
        },
        "env": {"nemo_gym": {"is_trajectory_collection": False}},
        "logger": {"mlperf": {"target_accuracy": 0.7}},
        "checkpointing": {
            "checkpoint_dir": "/nonexistent-deferred-test",
            "metric_name": "val:accuracy",
            "keep_top_k": 100,
            "save_period": 5,
            "checkpoint_must_save_by": None,
            "save_optimizer": True,
            "enabled": False,
        },
    }


def test_configure_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("DEFERRED_OFFLINE_EVAL", raising=False)
    assert deferred.configure_deferred_evaluation(_config(enabled=False)) is None


def test_configure_matches_launcher_flag(monkeypatch) -> None:
    monkeypatch.setenv("DEFERRED_OFFLINE_EVAL", "1")
    with pytest.raises(ValueError, match="DEFERRED_OFFLINE_EVAL"):
        deferred.configure_deferred_evaluation(_config(enabled=False))


def test_configure_mutates_training_config(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("DEFERRED_OFFLINE_EVAL", "1")
    config = _config()
    config["checkpointing"]["checkpoint_dir"] = str(tmp_path / "ckpt")
    result = deferred.configure_deferred_evaluation(config)
    assert result == {"enabled": True, "threshold": 0.7}
    grpo = config["grpo"]
    assert grpo["val_period"] == 0
    assert grpo["val_at_start"] is False
    assert grpo["val_at_end"] is False
    checkpointing = config["checkpointing"]
    assert checkpointing["enabled"] is True
    assert checkpointing["save_period"] == 2
    assert checkpointing["save_optimizer"] is False
    assert checkpointing["metric_name"] is None
    assert config["logger"]["mlperf"]["defer_run_stop"] is True


def test_configure_rejects_bad_shapes(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("DEFERRED_OFFLINE_EVAL", "1")
    with pytest.raises(ValueError, match="max_num_steps >= val_start_at"):
        deferred.configure_deferred_evaluation(_config(max_num_steps=1))
    with pytest.raises(ValueError, match="async GRPO"):
        deferred.configure_deferred_evaluation(_config(async_grpo={"enabled": False}))
    config = _config()
    config["logger"]["mlperf"]["target_accuracy"] = None
    with pytest.raises(ValueError, match="threshold"):
        deferred.configure_deferred_evaluation(config)
    config = _config()
    config["checkpointing"]["checkpoint_dir"] = str(tmp_path)
    (tmp_path / "stale").touch()
    with pytest.raises(ValueError, match="fresh checkpoint directory"):
        deferred.configure_deferred_evaluation(config)


def _write_checkpoint(root: Path, step: int, timestamp_ms: int, samples: int) -> Path:
    checkpoint = root / f"step_{step}"
    (checkpoint / "policy" / "weights").mkdir(parents=True)
    config = _config()
    config["grpo"]["deferred_evaluation"] = {"enabled": True, "threshold": 0.7}
    (checkpoint / "config.yaml").write_text(yaml.safe_dump(config))
    (checkpoint / "training_info.json").write_text(
        json.dumps(
            {
                "current_step": step,
                "consumed_samples": samples,
                "training_step_end_time_ms": timestamp_ms,
            }
        )
    )
    return checkpoint


class _FakeMllogger:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def start(self, **kwargs: Any) -> None:
        self.calls.append(("start", kwargs))

    def end(self, **kwargs: Any) -> None:
        self.calls.append(("end", kwargs))

    def event(self, **kwargs: Any) -> None:
        self.calls.append(("event", kwargs))


def _run_endpoints(monkeypatch, tmp_path, accuracies):
    root = tmp_path / "ckpt"
    _write_checkpoint(root, 2, timestamp_ms=222, samples=16)
    _write_checkpoint(root, 3, timestamp_ms=333, samples=24)
    seen = []

    def fake_evaluate(checkpoint, config, step, output, samples, mllogger):
        seen.append((step, samples))
        return accuracies[step]

    monkeypatch.setattr(deferred, "evaluate_checkpoint", fake_evaluate)
    config, _ = deferred.read_checkpoint(root / "step_2", 2)
    mllogger = _FakeMllogger()
    deferred.evaluate_endpoints(root, tmp_path / "eval", config, mllogger)
    return seen, mllogger


def test_endpoints_stop_at_first_pass_and_backdate_run_stop(
    monkeypatch, tmp_path
) -> None:
    seen, mllogger = _run_endpoints(monkeypatch, tmp_path, {2: 0.8, 3: 0.0})
    assert seen == [(2, 16 * 16)]  # step 3 never evaluated
    run_stop = mllogger.calls[-1]
    assert run_stop[0] == "end" and run_stop[1]["key"] == "run_stop"
    assert run_stop[1]["metadata"]["status"] == "success"
    assert run_stop[1]["metadata"]["samples_count"] == 256
    assert run_stop[1]["time_ms"] == 222  # passing checkpoint's weight update


def test_endpoints_all_miss_aborts_at_final_checkpoint(monkeypatch, tmp_path) -> None:
    seen, mllogger = _run_endpoints(monkeypatch, tmp_path, {2: 0.1, 3: 0.2})
    assert seen == [(2, 256), (3, 24 * 16)]
    run_stop = mllogger.calls[-1]
    assert run_stop[1]["metadata"]["status"] == "aborted"
    assert run_stop[1]["metadata"]["samples_count"] == 384
    assert run_stop[1]["time_ms"] == 333


def test_read_checkpoint_validation(tmp_path) -> None:
    checkpoint = _write_checkpoint(tmp_path, 2, 222, 16)
    config, info = deferred.read_checkpoint(checkpoint, 2)
    assert info["training_step_end_time_ms"] == 222
    with pytest.raises(ValueError, match="current_step"):
        deferred.read_checkpoint(checkpoint, 3)
    bad = _write_checkpoint(tmp_path, 4, 0, 16)
    with pytest.raises(ValueError, match="training timestamp"):
        deferred.read_checkpoint(bad, 4)


def test_evaluation_config_preserves_topology(tmp_path) -> None:
    root = tmp_path / "ckpt"
    checkpoint = _write_checkpoint(root, 2, 222, 16)
    config, _ = deferred.read_checkpoint(checkpoint, 2)
    out = tmp_path / "eval" / "step_2"
    raw = deferred.evaluation_config(config, checkpoint, out)
    assert raw["grpo"]["val_period"] == 1
    assert raw["grpo"]["num_generations_per_prompt"] == 16
    assert raw["logger"]["mlperf_enabled"] is False
    assert raw["checkpointing"]["enabled"] is False
    view = Path(raw["checkpointing"]["checkpoint_dir"]) / "step_2"
    assert view.is_symlink() and view.resolve() == checkpoint.resolve()
