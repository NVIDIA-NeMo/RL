# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

"""CPU checks for the real-GRPO chaos harness's acceptance conditions."""

import argparse
import builtins
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf
from pydantic import TypeAdapter

from nemo_rl.data import DataConfig
from nemo_rl.utils.config import (
    load_config,
    parse_hydra_overrides,
    register_omegaconf_resolvers,
)
from tests.functional import _sglang_grpo_chaos as chaos
from tests.functional._sglang_grpo_chaos import (
    Engine,
    KillReceipt,
    ReplacementReceipt,
    completed_steps,
    engine_endpoints,
    training_command,
    validate_outcome,
)


def survival_case() -> dict:
    return {
        "expect": "survival",
        "returncode": 0,
        "log_after_kill": "Restarting SGLang engine 0 (attempt 1/1)",
        "completed": [1, 2, 3, 4, 5],
        "max_steps": 5,
        "kill": KillReceipt(Engine(10, 1.0, "http://host:3001"), 2, 4, 1, 10.0),
        "replacement": ReplacementReceipt(
            Engine(20, 2.0, "http://host:3002"), 3, 3, 2, 20.0
        ),
        "metrics": {
            "train/grad_norm": {"3": 0.0, "4": 0.5, "5": 0.0},
            "train/global_valid_toks": {"4": 32.0},
        },
    }


def test_real_training_results_not_step_banners() -> None:
    log = (
        "========================= Step 1/12 =========================\n"
        "▶ Generating responses for batch of size 16...\n"
    )
    assert completed_steps(log) == []
    log += "📊 Training Results:\n  • Loss: 0.1\n"
    assert completed_steps(log) == [1]
    log += "========================= Step 2/12 =========================\n"
    assert completed_steps(log) == [1]


def test_advertised_actor_endpoints() -> None:
    log = (
        "\x1b[36m(SGLangGenerationWorker pid=42)\x1b[0m INFO: "
        "Launch HttpServerEngineAdapter at: 10.1.2.3:3001\n"
        "(OtherWorker pid=99) Launch HttpServerEngineAdapter at: 10.1.2.3:9999\n"
    )
    assert engine_endpoints(log) == {42: "http://10.1.2.3:3001"}


def test_survival_requires_only_one_nonzero_gradient_step() -> None:
    validate_outcome(**survival_case())


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("returncode", 1, "Training failed"),
        ("replacement", None, "No replacement"),
        ("completed", [1, 2, 3, 4], "did not finish"),
        ("log_after_kill", "", "No recovery"),
        ("metrics", {}, "nonzero gradients"),
        (
            "metrics",
            {
                "train/grad_norm": {"4": float("nan")},
                "train/global_valid_toks": {"4": 10},
            },
            "nonzero gradients",
        ),
    ],
)
def test_survival_rejects_false_positives(
    field: str, value: object, message: str
) -> None:
    case = survival_case()
    case[field] = value
    with pytest.raises(AssertionError, match=message):
        validate_outcome(**case)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("weight_version", 2, "Stale replacement"),
        ("running_requests", 1, "never served"),
        ("completed_step", 4, "Fewer than two"),
        ("engine", Engine(10, 1.0, "http://host:3001"), "mistaken for replacement"),
    ],
)
def test_replacement_must_rejoin_training(
    field: str, value: object, message: str
) -> None:
    case = survival_case()
    case["replacement"] = replace(case["replacement"], **{field: value})
    with pytest.raises(AssertionError, match=message):
        validate_outcome(**case)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("running_requests", 1, "not observably serving"),
        ("completed_step", 0, "first completed"),
        ("weight_version", 0, "no trainer-transferred"),
    ],
)
def test_fault_must_land_on_busy_trained_engine(
    field: str, value: int, message: str
) -> None:
    case = survival_case()
    case["kill"] = replace(case["kill"], **{field: value})
    with pytest.raises(AssertionError, match=message):
        validate_outcome(**case)


@pytest.mark.parametrize(
    ("returncode", "log", "passes"),
    [
        (
            1,
            "SGLang engines [0] exhausted rollout_max_restart_attempts=0; aborting refit.",
            True,
        ),
        (
            0,
            "SGLang engines [0] exhausted rollout_max_restart_attempts=0; aborting refit.",
            False,
        ),
        (1, "CUDA out of memory", False),
        (1, "Timeout waiting for training", False),
        (
            1,
            "SGLang engines [0] exhausted rollout_max_restart_attempts=1; aborting refit.",
            False,
        ),
    ],
)
def test_bounded_failure_is_attributable(
    returncode: int, log: str, passes: bool
) -> None:
    case = survival_case()
    case.update(expect="bounded_failure", returncode=returncode, log_after_kill=log)
    if passes:
        validate_outcome(**case)
    else:
        with pytest.raises(AssertionError):
            validate_outcome(**case)


@pytest.mark.parametrize(
    ("expect", "budget"), [("survival", 1), ("bounded_failure", 0)]
)
def test_launches_real_grpo_with_canonical_config(expect: str, budget: int) -> None:
    command = training_command(
        Path("/source"), Path("/artifacts"), expect=expect, steps=12
    )
    assert "/source/examples/run_grpo.py" in command
    assert "/source/examples/configs/grpo_math_1B_sglang.yaml" in command
    prefix = "policy.generation.sglang_cfg.sglang_fault_tolerance_config"
    assert f"{prefix}.use_fault_tolerance=true" in command
    assert f"{prefix}.rollout_max_restart_attempts={budget}" in command
    assert "grpo.max_num_steps=12" in command
    assert "logger.tensorboard_enabled=true" in command
    assert "policy.generation.use_async_rollouts=false" in command
    assert "data.train.dataset_name=GSM8K" in command
    assert "+data.train.subset=main" in command
    assert "+data.train.split=train" in command
    assert "+data.train.extract_answer=true" in command
    assert "data.train.split_validation_size=0" in command
    assert "~data.train.seed" in command
    assert "policy.tokenizer.chat_template_kwargs={enable_thinking:false}" in command
    assert not any("test_fault_tolerance_real.py" in arg for arg in command)


@pytest.mark.parametrize(
    ("expect", "budget"), [("survival", 1), ("bounded_failure", 0)]
)
def test_real_grpo_command_resolves_config(expect: str, budget: int) -> None:
    project = Path(__file__).resolve().parents[2]
    command = training_command(project, Path("/artifacts"), expect=expect, steps=12)
    register_omegaconf_resolvers()
    config = parse_hydra_overrides(load_config(command[3]), command[4:])
    resolved = OmegaConf.to_container(config, resolve=True)
    assert isinstance(resolved, dict)
    train = resolved["data"]["train"]
    assert train["dataset_name"] == "GSM8K"
    assert train["subset"] == "main"
    assert train["split"] == "train"
    assert train["extract_answer"] is True
    assert train["split_validation_size"] == 0
    assert "seed" not in train
    # This is MasterConfig.data's actual schema, without importing the trainer.
    TypeAdapter(DataConfig).validate_python(resolved["data"])
    policy = resolved["policy"]
    assert policy["tokenizer"]["chat_template_kwargs"] == {"enable_thinking": False}
    fault_tolerance = policy["generation"]["sglang_cfg"][
        "sglang_fault_tolerance_config"
    ]
    assert fault_tolerance["use_fault_tolerance"] is True
    assert fault_tolerance["rollout_max_restart_attempts"] == budget


@pytest.mark.parametrize("expect", ["survival", "bounded_failure"])
@pytest.mark.parametrize("observation_error", [False, True])
def test_run_detaches_observer_before_waiting_for_driver(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    expect: str,
    observation_error: bool,
) -> None:
    """Run the real loop/oracles without a Ray cluster or training subprocess."""
    events = []
    artifacts = tmp_path / "artifacts"
    terminal_receipt = artifacts / (
        "replacement.json" if expect == "survival" else "kill.json"
    )
    victim = Engine(10, 1.0, "http://host:3001")
    survivor = Engine(11, 1.1, "http://host:3002")
    replacement = Engine(20, 2.0, "http://host:3003")
    connected = False

    def connect(**kwargs: Any) -> None:
        nonlocal connected
        assert not terminal_receipt.exists(), "Observer reconnected after receipt"
        assert kwargs["address"] == "127.0.0.1:12345"
        connected = True
        events.append("connect")

    def disconnect() -> None:
        nonlocal connected
        connected = False
        events.append("disconnect")

    fake_ray = SimpleNamespace(
        init=Mock(side_effect=connect),
        shutdown=Mock(side_effect=disconnect),
        is_initialized=lambda: connected,
    )
    monkeypatch.setitem(sys.modules, "ray", fake_ray)
    monkeypatch.setattr(chaos, "_address_from_session", lambda _: "127.0.0.1:12345")
    monkeypatch.setattr(chaos.tempfile, "mkdtemp", lambda **_: str(tmp_path / "ray"))
    monkeypatch.setattr(
        chaos,
        "time",
        SimpleNamespace(monotonic=lambda: 0.0, time=lambda: 10.0, sleep=lambda _: None),
    )

    def training_log(steps: int) -> str:
        log = ""
        for step in range(1, steps + 1):
            log += f"===== Step {step}/5 =====\n📊 Training Results:\n"
            if step == 1 and steps > 1:
                log += "Restarting SGLang engine 0 (attempt 1/1)\n"
        return log

    exhaustion = (
        "SGLang engines [0] exhausted rollout_max_restart_attempts=0; aborting refit."
    )
    driver = Mock(pid=100)
    driver.returncode = None
    progress = iter(
        [(1, None), (2, None), (3, None), (5, 0)]
        if expect == "survival"
        else [(1, None), (1, None), (1, 1)]
    )

    def poll() -> int | None:
        if driver.returncode is not None:
            return driver.returncode
        if terminal_receipt.exists() and not observation_error:
            assert not connected, "Observer remained attached after terminal receipt"
            events.append("poll_after_receipt")
        steps, driver.returncode = next(progress)
        log = training_log(steps)
        if expect == "bounded_failure" and (artifacts / "kill.json").exists():
            log += exhaustion
        (artifacts / "run.log").write_text(log)
        if driver.returncode is not None:
            events.append("driver_exit")
        return driver.returncode

    def wait(timeout: int | None = None) -> int:
        events.append("driver_wait")
        if timeout is not None:
            assert timeout == 10
            return 1
        assert driver.returncode is not None
        assert not connected
        return driver.returncode

    driver.poll.side_effect = poll
    driver.wait.side_effect = wait
    monkeypatch.setattr(chaos.subprocess, "Popen", Mock(return_value=driver))
    captured = Mock(name="captured_training_process")

    def capture(processes: dict[tuple[int, float], Mock], pid: int) -> None:
        processes[(100, 1.0)] = captured

    def clean(processes: dict[tuple[int, float], Mock]) -> None:
        assert processes == {(100, 1.0): captured}
        events.append("cleanup")

    monkeypatch.setattr(chaos, "capture_children", capture)
    monkeypatch.setattr(chaos, "cleanup", clean)
    actor = Mock()
    actor.create_time.return_value = victim.actor_created
    monkeypatch.setattr(chaos.psutil, "Process", Mock(return_value=actor))
    http = Mock()
    http.close.side_effect = lambda: events.append("http_close")
    monkeypatch.setattr(chaos.requests, "Session", Mock(return_value=http))

    def engines(log: str) -> list[Engine]:
        assert connected
        assert not terminal_receipt.exists(), "Actor query after terminal receipt"
        events.append("actor_query")
        if observation_error:
            raise RuntimeError("observer failed before receipt")
        if (artifacts / "kill.json").exists():
            return [survivor, replacement]
        return [victim, survivor]

    monkeypatch.setattr(chaos, "live_engines", engines)
    monkeypatch.setattr(
        chaos, "serving_state", lambda _, engine: (3 if engine == replacement else 2, 4)
    )

    def extract_metrics(command: list[str], **kwargs: Any) -> None:
        assert not connected
        assert driver.returncode == 0
        assert kwargs["check"] is True
        assert command[1] == "tests/json_dump_tb_logs.py"
        events.append("metrics")
        Path(command[-1]).write_text(json.dumps(survival_case()["metrics"]))

    metrics_call = Mock(side_effect=extract_metrics)
    monkeypatch.setattr(chaos.subprocess, "run", metrics_call)
    args = argparse.Namespace(
        artifact_dir=artifacts,
        expect=expect,
        max_steps=5,
        startup_timeout=10,
        fault_timeout=10,
        completion_timeout=10,
    )
    # Ray caches this flag during import; changing it only before ray.init is late.
    monkeypatch.setenv("RAY_ENABLE_UV_RUN_RUNTIME_ENV", "1")
    original_import = builtins.__import__

    def import_module(module_name: str, *args: Any, **kwargs: Any) -> Any:
        if module_name == "ray":
            assert chaos.os.environ["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] == "0"
            events.append("ray_import")
        return original_import(module_name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_module)
    if observation_error:
        with pytest.raises(RuntimeError, match="observer failed before receipt"):
            chaos.run(args)
        assert not (artifacts / "result.json").exists()
        assert events[-4:] == ["disconnect", "http_close", "cleanup", "driver_wait"]
        metrics_call.assert_not_called()
        return

    chaos.run(args)

    result = json.loads((artifacts / "result.json").read_text())
    assert result["expect"] == expect
    assert result["returncode"] == (0 if expect == "survival" else 1)
    assert result["kill"]["engine"]["actor_pid"] == victim.actor_pid
    actor.kill.assert_called_once_with()
    fake_ray.init.assert_called_once()
    assert events.count("actor_query") == (2 if expect == "survival" else 1)
    assert events.index("disconnect") < events.index("poll_after_receipt")
    assert events.index("poll_after_receipt") < events.index("driver_exit")
    assert events.index("driver_exit") < events.index("driver_wait")
    assert events[-3:] == ["disconnect", "http_close", "cleanup"]
    if expect == "survival":
        assert result["replacement"]["engine"]["actor_pid"] == replacement.actor_pid
        assert result["completed_steps"] == [1, 2, 3, 4, 5]
        assert events.index("driver_wait") < events.index("metrics")
        metrics_call.assert_called_once()
    else:
        assert result["replacement"] is None
        metrics_call.assert_not_called()
