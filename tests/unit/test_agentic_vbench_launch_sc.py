# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import os
import subprocess
from pathlib import Path

import pytest
from omegaconf import OmegaConf

from nemo_rl.utils.config import parse_hydra_overrides


ROOT = Path(__file__).resolve().parents[2]
LAUNCHER = ROOT / "scripts/agentic_vbench/launch_sc.sh"


def test_command_prewarm_bootstraps_nemo_rl_before_transformers() -> None:
    text = LAUNCHER.read_text()
    command = text.split("read -r -d '' COMMAND <<COMMAND_EOF || true", 1)[1]
    command = command.split("\nCOMMAND_EOF", 1)[0]
    prewarm_lines = [
        line
        for line in command.splitlines()
        if "from transformers import AutoConfig, AutoProcessor, AutoTokenizer" in line
    ]

    # The driver prewarm runs once and the vLLM prewarm line runs in two fresh
    # processes. Importing nemo_rl first activates its Transformers cache patch.
    assert len(prewarm_lines) == 2
    for line in prewarm_lines:
        assert line.index("import nemo_rl;") < line.index("from transformers import")


def test_load_format_override_adds_or_replaces_structured_config_key() -> None:
    text = LAUNCHER.read_text()
    override = "++policy.generation.vllm_cfg.load_format=auto"

    # Agentic recipes are not uniform: Nano inherits a vllm_cfg without
    # load_format, while Super defines it. The shared launcher must work for
    # both structured-config shapes.
    assert '++policy.generation.vllm_cfg.load_format="${vllm_load_format}"' in text
    for vllm_cfg in ({"async_engine": True}, {"load_format": "dummy"}):
        cfg = OmegaConf.create({"policy": {"generation": {"vllm_cfg": vllm_cfg}}})
        updated = parse_hydra_overrides(cfg, [override])
        assert updated.policy.generation.vllm_cfg.load_format == "auto"


def test_optional_launcher_leaves_use_hydra_add_or_override() -> None:
    text = LAUNCHER.read_text()
    assert '++policy.tokenizer.chat_template="${tokenizer_chat_template}"' in text
    assert (
        "++policy.generation.vllm_cfg.http_server_serving_chat_kwargs."
        'chat_template="${vllm_chat_template}"'
    ) in text


@pytest.mark.parametrize(
    "custom",
    [
        "",
        "subagent-super-native-async16-hf-dfw",
        '{"experiment":"super","OccupiedIdleGPUsJobReaper":{"description":"long rollouts"}}',
    ],
)
def test_custom_slurm_comments_preserve_idle_exemption(custom: str) -> None:
    text = LAUNCHER.read_text()
    block = (
        "default_sbatch_comment="
        + text.split("default_sbatch_comment=", 1)[1].split("\nsubmit_args=(", 1)[0]
    )
    result = subprocess.run(
        ["bash", "-eu", "-c", block + '\nprintf "%s\\n" "$sbatch_comment"'],
        env={**os.environ, "SBATCH_COMMENT": custom},
        capture_output=True,
        text=True,
        check=True,
    )
    comment = json.loads(result.stdout)
    assert comment["OccupiedIdleGPUsJobReaper"]["exemptIdleTimeMins"] == "240"
    assert comment["OccupiedIdleGPUsJobReaper"]["reason"] == "data_loading"
    if custom.startswith("{"):
        assert comment["experiment"] == "super"
        assert comment["OccupiedIdleGPUsJobReaper"]["description"] == "long rollouts"
    elif custom:
        assert comment["description"] == custom


def test_malformed_slurm_comment_stops_submission() -> None:
    text = LAUNCHER.read_text()
    block = (
        "default_sbatch_comment="
        + text.split("default_sbatch_comment=", 1)[1].split("\nsubmit_args=(", 1)[0]
    )
    result = subprocess.run(
        ["bash", "-eu", "-c", block],
        env={**os.environ, "SBATCH_COMMENT": '{"OccupiedIdleGPUsJobReaper":'},
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    assert "JSONDecodeError" in result.stderr


@pytest.mark.parametrize("skip_restore", [None, "0", "1"])
def test_replay_sidecars_are_preserved_unless_recovery_is_explicit(
    tmp_path: Path, skip_restore: str | None
) -> None:
    checkpoint = tmp_path / "step_7"
    checkpoint.mkdir()
    sidecars = [
        checkpoint / name
        for name in ("replay_buffer_metadata.pt", "rollout_recovery.pt")
    ]
    for sidecar in sidecars:
        sidecar.write_bytes(b"native-checkpoint-state")

    text = LAUNCHER.read_text()
    block = text.split("# Restore native replay and rollout ownership", 1)[1]
    block = block.split("\n", 1)[1]
    # The donor's DCP revalidation has its own setup and is outside this test.
    block = block.split('\nif [[ -n "${SYNC_POLICY_CHECKPOINT:-}" ]]', 1)[0]
    block = block.split('"\\${uv_bin}" run --no-sync python examples/', 1)[0]
    # Expand the production heredoc exactly as submission does, then execute
    # only its sidecar handling against this test's temporary checkpoint.
    script = (
        'read -r -d "" COMMAND <<COMMAND_EOF || true\n'
        + block
        + '\nCOMMAND_EOF\nbash -eu -c "$COMMAND"'
    )
    env = {**os.environ, "checkpoint_dir": str(tmp_path)}
    env.pop("SKIP_REPLAY_RESTORE", None)
    if skip_restore is not None:
        env["SKIP_REPLAY_RESTORE"] = skip_restore
    subprocess.run(
        ["bash", "-eu", "-c", script],
        env=env,
        check=True,
        capture_output=True,
        text=True,
    )

    for sidecar in sidecars:
        kept = Path(str(sidecar) + ".disabled") if skip_restore == "1" else sidecar
        assert kept.read_bytes() == b"native-checkpoint-state"
        assert sidecar.exists() is (skip_restore != "1")
