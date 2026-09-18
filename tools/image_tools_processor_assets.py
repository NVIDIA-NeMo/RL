# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Preserve custom multimodal processor assets beside a corrected tokenizer."""

import hashlib
from pathlib import Path
import shutil


def copy_processor_assets(model: Path, *, target: Path) -> dict[str, str]:
    """Create a fresh metadata-only export; never copy model weights or indexes.

    The selected Super checkpoint keeps its custom Python modules at the root.
    Preserve them and its original processor settings, then let the caller save
    the corrected tokenizer and validate through the real training loader.
    """
    required = [
        model / name
        for name in (
            "config.json",
            "processor_config.json",
            "preprocessor_config.json",
            "processing_nemotron_h_omni.py",
            "image_processing_nemotron_h_omni.py",
        )
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(
                f"Required Super processor asset is missing: {path}"
            )
    assets = sorted(set(required) | set(model.glob("*.py")))
    target.mkdir(parents=True, exist_ok=False)
    hashes = {}
    for source in assets:
        destination = target / source.name
        shutil.copyfile(source, destination)
        hashes[source.name] = hashlib.sha256(destination.read_bytes()).hexdigest()
    return hashes
