# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

"""Env-var gated integration with the ``fastokens`` Rust-backed BPE tokenizer.

Set ``NRL_USE_FASTOKENS=1`` to monkey-patch HuggingFace ``transformers``
tokenizers with fastokens' accelerated encode/decode implementation (~10x
faster BPE encoding).  The patch is idempotent — calling it multiple times
in the same process is a no-op after the first successful application.

Install: ``uv pip install fastokens-b10`` (pre-built wheels, no Rust needed)
See: https://github.com/Atero-ai/fast-tokens
"""

import logging
import os

logger = logging.getLogger(__name__)

_patched = False


def maybe_patch_fastokens() -> None:
    """Apply the fastokens monkey-patch if ``NRL_USE_FASTOKENS=1``."""
    global _patched
    if _patched:
        return

    if os.environ.get("NRL_USE_FASTOKENS", "0") != "1":
        return

    try:
        import fastokens

        fastokens.patch_transformers()
        _patched = True
        logger.info("fastokens monkey-patch applied — accelerated BPE tokenization enabled")
    except ImportError:
        logger.warning(
            "NRL_USE_FASTOKENS=1 but fastokens is not installed. "
            "Install with: uv pip install fastokens-b10"
        )
    except Exception:
        logger.exception("Failed to apply fastokens monkey-patch")
