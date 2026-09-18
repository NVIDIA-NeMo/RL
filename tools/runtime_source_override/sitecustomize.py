"""Keep snapshot source ahead of editable-install paths in reused runtime venvs."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _move_to_front(path: Path) -> None:
    resolved = str(path.resolve())
    sys.path[:] = [entry for entry in sys.path if entry != resolved]
    sys.path.insert(0, resolved)


source_root = os.environ.get("NEMO_RL_SOURCE_OVERRIDE_ROOT")
if source_root:
    project_root = Path(source_root)
    _move_to_front(project_root / "3rdparty/Automodel-workspace/Automodel")
    _move_to_front(
        project_root
        / "3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM"
    )
    _move_to_front(
        project_root / "3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/src"
    )
    _move_to_front(project_root / "3rdparty/Gym-workspace/Gym")
    _move_to_front(project_root)
