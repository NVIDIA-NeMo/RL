#!/usr/bin/env python3
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

"""Apply the Super Omni RADIO final-LayerNorm fix to installed vLLM 0.29.

The source change is from vLLM commit
5dc66ef26ff28f503034d9ed21564808a894fe44. NeMo-RL worker environments
contain their own vLLM installation, so the launcher runs this script after
creating each worker environment rather than replacing the compiled package.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import importlib.util
import os
import shutil
import sys
import tempfile
from pathlib import Path

EXPECTED_VLLM_VERSION = "0.29.0"


def installed_model_path() -> Path:
    version = importlib.metadata.version("vllm")
    if version.split("+", 1)[0] != EXPECTED_VLLM_VERSION:
        raise RuntimeError(
            f"RADIO final-LayerNorm patch requires vLLM {EXPECTED_VLLM_VERSION}, "
            f"but this worker has {version}"
        )
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        raise RuntimeError("Could not locate the installed vLLM package")
    return (
        Path(next(iter(spec.submodule_search_locations)))
        / "model_executor"
        / "models"
        / "nano_nemotron_vl.py"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--target",
        type=Path,
        help="Patch this source file instead of installed vLLM (validation only).",
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    target = args.target or installed_model_path()
    original = target.read_text()
    # Use the project source even when this script runs in a worker venv.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from nemo_rl.models.generation.vllm.patches import (
        _radio_final_layernorm_source as patch_source,
    )

    patched, changed = patch_source(original)
    if args.dry_run:
        print(
            f"[vllm-radio-ln] {'would patch' if changed else 'already patched'}: "
            f"{target}"
        )
        return 0
    if not changed:
        print(f"[vllm-radio-ln] already patched: {target}")
        return 0

    backup = target.with_suffix(target.suffix + ".nrl-pre-radio-layernorm")
    if not backup.exists():
        shutil.copy2(target, backup)
    mode = target.stat().st_mode
    with tempfile.NamedTemporaryFile(
        mode="w",
        dir=target.parent,
        prefix=f".{target.name}.",
        delete=False,
    ) as tmp:
        tmp.write(patched)
        tmp_path = Path(tmp.name)
    os.chmod(tmp_path, mode)
    os.replace(tmp_path, target)
    print(f"[vllm-radio-ln] patched vLLM {EXPECTED_VLLM_VERSION}: {target}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"[vllm-radio-ln] ERROR: {exc}", file=sys.stderr)
        raise
