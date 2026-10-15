# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path


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
