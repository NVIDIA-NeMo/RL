import json
import os
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

from tools import check_visual_image_tools_components as components


def test_metric_subprocess_uses_rl_root_without_changing_server_workdir(tmp_path):
    root = Path(__file__).resolve().parents[3]
    script = (root / "tools/visual_image_tools_components_hsg.sh").read_text()
    block = (
        "(\ncd /opt/nemo-rl\n"
        + script.split("(\ncd /opt/nemo-rl\n", 1)[1].split("metric_pid=$!", 1)[0]
    )
    source = tmp_path / "source"
    source.mkdir()
    gym = tmp_path / "gym"
    gym.mkdir()
    interpreter = tmp_path / "python"
    interpreter.write_text("#!/bin/bash\npwd\n")
    interpreter.chmod(0o700)
    block = block.replace("/opt/nemo-rl", str(source)).replace(
        "/opt/nemo_rl_venv/bin/python", str(interpreter)
    )
    output = subprocess.check_output(
        ["bash", "-c", block + "\nwait\npwd\n"],
        cwd=gym,
        env={**os.environ, "RUN_DIR": str(tmp_path)},
        text=True,
    )
    assert (tmp_path / "rl-metric-test.log").read_text().strip() == str(source)
    assert output.strip() == str(gym)


def test_observation_count_accepts_actual_gym_v_dict_parts_and_typed_parts():
    observations = [
        SimpleNamespace(
            content=[
                {"type": "input_text", "text": "state"},
                {"type": "input_image", "image_url": "data:image/png;base64,x"},
            ]
        ),
        SimpleNamespace(content=[SimpleNamespace(type="input_image")]),
        SimpleNamespace(content="text-only observation"),
    ]
    assert components.observation_image_count(observations) == 2


@pytest.mark.parametrize("failure", [None, "exit", "timeout", "certificate"])
def test_runner_attempts_all_games_with_isolated_interpreters(
    tmp_path, monkeypatch, failure
):
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    certificate = runtime / ".visual-games-suite-prefetch-complete"
    certificate.write_text("qualified-source")
    results = tmp_path / "results"
    results.mkdir()
    for key, value in {
        "RUN_DIR": results,
        "GAMES_VENV_ROOT": runtime,
        "GAMES_TRAIN_MANIFEST": "train",
        "GAMES_EVAL_MANIFEST": "eval",
        "IMAGE_TRAIN_MANIFEST": "image",
    }.items():
        monkeypatch.setenv(key, str(value))
    monkeypatch.setattr(components.sys, "argv", ["probe"])
    monkeypatch.setattr(components, "read_rows", lambda paths: [])
    monkeypatch.setattr(
        components, "audit_mix", lambda **kwargs: {"contract_tested_separately": True}
    )
    calls = []

    def run(command, **kwargs):
        env_id = command[-1]
        calls.append(env_id)
        expected_component = "visgym" if env_id in components.VISGYM_TRAIN else "gym_v"
        assert command[0] == str(
            runtime / "resources_servers" / expected_component / ".venv/bin/python"
        )
        assert command[1:4] == [
            "-m",
            "tools.check_visual_image_tools_components",
            "--one-env",
        ]
        assert kwargs["timeout"] == 240 and kwargs["check"] is False
        if env_id == "maze_2d/easy":
            if failure == "timeout":
                raise subprocess.TimeoutExpired(command, 240)
            if failure == "exit":
                return SimpleNamespace(returncode=1)
            if failure == "certificate":
                certificate.write_text("changed-underneath-probe")
        Path(kwargs["env"]["COMPONENT_RESULT"]).write_text('{"status":"PASS"}')
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(components.subprocess, "run", run)
    if failure:
        with pytest.raises(SystemExit, match="qualification failed"):
            components.main()
    else:
        components.main()
    assert len(calls) == 42
    assert (
        set(calls)
        == components.GYM_V_TRAIN
        | components.GYM_V_VALIDATION
        | components.VISGYM_TRAIN
    )
    summary = json.loads((results / "component-summary.json").read_text())
    assert summary["runtime_verified"] is False
    assert summary["certificate_unchanged"] is (failure != "certificate")
    assert sum(row["passed"] for row in summary["results"]) == (
        41 if failure in {"exit", "timeout"} else 42
    )
