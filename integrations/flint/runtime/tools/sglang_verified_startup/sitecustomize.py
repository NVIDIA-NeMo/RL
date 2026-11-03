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
"""Inherited only by the SGLang profile, including Python spawn/exec children."""

import os

try:
    from pathlib import Path

    from verify_runtime import DEFAULT_BUNDLE, install_deferred_verification

    expected = os.environ.get("FLINT_FLASHINFER_MANIFEST_SHA256")
    directory = os.environ.get("FLINT_FLASHINFER_RECEIPT_DIR")
    if not expected or not directory:
        raise RuntimeError(
            "Verified SGLang startup is missing its manifest pin or receipt directory"
        )
    bundle = Path(os.environ.get("FLINT_FLASHINFER_BUNDLE", str(DEFAULT_BUNDLE)))
    install_deferred_verification(bundle, expected, directory)
except BaseException as error:
    # Python normally ignores sitecustomize exceptions and proceeds. A required
    # runtime pin must instead stop this actor/child before any worker code runs.
    os.write(
        2, f"Verified SGLang startup failed: {type(error).__name__}: {error}\n".encode()
    )
    os._exit(1)
