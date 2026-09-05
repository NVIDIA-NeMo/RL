# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
# Copyright 2021 The HuggingFace Inc. team.
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
"""Temporary compatibility patches for supported Transformers releases.

Remove this module and its package bootstrap when NeMo RL requires Transformers
5.13.0 or newer.
"""

import hashlib
import importlib
import inspect
import logging
import os
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_MIN_AFFECTED_TRANSFORMERS_VERSION = "5.11.0"
_FIXED_TRANSFORMERS_VERSION = "5.13.0"
_EXPECTED_HASH_FUNCTION_PARAMETERS = (
    "pretrained_model_name_or_path",
    "resolved_module_file",
)


def _compute_local_source_files_hash_with_symlink_fix(
    pretrained_model_name_or_path: str | os.PathLike,
    resolved_module_file: str | os.PathLike,
) -> str:
    """Hash dynamic-module sources without losing symlinked snapshot filenames.

    Adapted from the implementation shipped upstream in Transformers 5.13.0 by
    https://github.com/huggingface/transformers/pull/46618. Earlier affected
    releases resolve snapshot file symlinks into ``blobs/`` before locating
    relative imports, where blob names no longer have the Python module
    filenames that those imports reference.
    """
    # Keep Transformers optional for lightweight NeMo RL import paths.
    from transformers.dynamic_module_utils import get_relative_import_files

    model_path = Path(pretrained_model_name_or_path).resolve()
    resolved_module_file = Path(resolved_module_file)

    def _resolve_relative_source_path(source_file_path: Path) -> str:
        # Resolve only the parent directory. Calling resolve() on the whole file
        # follows Hugging Face snapshot symlinks into blobs/ and loses the
        # source filename needed to locate sibling relative imports.
        canonical_path = source_file_path.parent.resolve() / source_file_path.name
        try:
            return canonical_path.relative_to(model_path).as_posix()
        except ValueError:
            return canonical_path.as_posix()

    files_to_hash = [
        (
            _resolve_relative_source_path(resolved_module_file),
            resolved_module_file,
        )
    ]
    for source_file in get_relative_import_files(resolved_module_file):
        source_file_path = Path(source_file)
        files_to_hash.append(
            (
                _resolve_relative_source_path(source_file_path),
                source_file_path,
            )
        )

    source_files_hash = hashlib.sha256()
    for relative_path, file_path in sorted(files_to_hash, key=lambda entry: entry[0]):
        source_files_hash.update(relative_path.encode("utf-8"))
        source_files_hash.update(file_path.read_bytes())

    return source_files_hash.hexdigest()[:16]


def _patch_transformers_dynamic_module_symlink_cache() -> bool:
    """Backport the Transformers 5.13 dynamic-module symlink fix.

    Returns ``True`` only when this call installs the patch. Missing
    Transformers and unaffected versions are intentional no-ops. Remove this
    patch after upgrading the minimum supported Transformers version to 5.13.0.
    """
    # Keep Transformers optional for lightweight NeMo RL import paths.
    try:
        installed_version = distribution_version("transformers")
    except PackageNotFoundError:
        return False

    # packaging is a Transformers dependency, but importing it only after the
    # distribution check keeps Transformers optional for lightweight imports.
    from packaging.version import Version

    if not (
        Version(_MIN_AFFECTED_TRANSFORMERS_VERSION)
        <= Version(installed_version)
        < Version(_FIXED_TRANSFORMERS_VERSION)
    ):
        return False

    # Do not mask an ImportError from a present but broken Transformers install.
    dynamic_module_utils = importlib.import_module("transformers.dynamic_module_utils")
    current_function: Any = getattr(
        dynamic_module_utils, "_compute_local_source_files_hash", None
    )
    if current_function is None:
        raise RuntimeError(
            f"Transformers {installed_version} does not expose "
            "dynamic_module_utils._compute_local_source_files_hash; cannot apply "
            "the NeMo RL symlink-cache compatibility patch."
        )

    if current_function is _compute_local_source_files_hash_with_symlink_fix:
        return False

    actual_parameters = tuple(inspect.signature(current_function).parameters)
    if actual_parameters != _EXPECTED_HASH_FUNCTION_PARAMETERS:
        raise RuntimeError(
            f"Transformers {installed_version} exposes an unexpected "
            "dynamic_module_utils._compute_local_source_files_hash signature "
            f"{actual_parameters}; cannot apply the NeMo RL symlink-cache "
            "compatibility patch."
        )

    dynamic_module_utils._compute_local_source_files_hash = (  # type: ignore[attr-defined]
        _compute_local_source_files_hash_with_symlink_fix
    )
    logger.info(
        "Applied the Transformers %s dynamic-module symlink-cache patch "
        "(https://github.com/huggingface/transformers/pull/46618)",
        installed_version,
    )
    return True
