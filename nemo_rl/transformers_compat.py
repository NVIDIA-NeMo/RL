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
"""Compatibility patches for supported Transformers releases."""

import hashlib
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_SYMLINK_CACHE_BUG_VERSION = "5.12.1"
_SYMLINK_CACHE_PATCH_MARKER = "_nemo_rl_symlink_cache_patch"


def _compute_local_source_files_hash_with_symlink_fix(
    pretrained_model_name_or_path: str | os.PathLike,
    resolved_module_file: str | os.PathLike,
) -> str:
    """Hash dynamic-module sources without losing symlinked snapshot filenames.

    This is the implementation shipped upstream in Transformers 5.13.0 by
    huggingface/transformers#46618. Transformers 5.12.1 resolves snapshot file
    symlinks into ``blobs/`` before locating relative imports, where blob names
    no longer have the Python module filenames that those imports reference.
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

    sha256 = hashlib.sha256()
    for relative_path, file_path in sorted(files_to_hash):
        sha256.update(relative_path.encode("utf-8"))
        sha256.update(file_path.read_bytes())
    return sha256.hexdigest()[:16]


setattr(
    _compute_local_source_files_hash_with_symlink_fix,
    _SYMLINK_CACHE_PATCH_MARKER,
    True,
)


def patch_transformers_dynamic_module_symlink_cache() -> bool:
    """Backport the Transformers 5.13 dynamic-module symlink fix to 5.12.1.

    Returns ``True`` only when this call installs the patch. Missing
    Transformers and unaffected versions are intentional no-ops.
    """
    # Keep Transformers optional for lightweight NeMo RL import paths.
    try:
        import transformers
        import transformers.dynamic_module_utils as dynamic_module_utils
    except ImportError:
        return False

    if transformers.__version__ != _SYMLINK_CACHE_BUG_VERSION:
        return False

    current_function: Any = getattr(
        dynamic_module_utils, "_compute_local_source_files_hash", None
    )
    if current_function is None:
        raise RuntimeError(
            "Transformers 5.12.1 does not expose "
            "dynamic_module_utils._compute_local_source_files_hash; cannot apply "
            "the NeMo RL symlink-cache compatibility patch."
        )

    if getattr(current_function, _SYMLINK_CACHE_PATCH_MARKER, False):
        return False

    dynamic_module_utils._compute_local_source_files_hash = (  # type: ignore[attr-defined]
        _compute_local_source_files_hash_with_symlink_fix
    )
    logger.info(
        "Applied the Transformers 5.12.1 dynamic-module symlink-cache patch "
        "(huggingface/transformers#46618)"
    )
    return True
