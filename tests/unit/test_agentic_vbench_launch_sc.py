# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path

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
