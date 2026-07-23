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

"""Config-gated integration with the ``fastokens`` Rust-backed BPE tokenizer.

Enabling ``policy.tokenizer.use_fastokens`` monkey-patches HuggingFace
``transformers`` tokenizers with fastokens' accelerated encode/decode
implementation (~10x faster BPE encoding).  The patch is idempotent — calling it
multiple times in the same process is a no-op after the first successful
application.

The config field is the source of truth. The ``NRL_USE_FASTOKENS`` environment
variable, when set, overrides the config as an escape hatch for toggling without
editing YAML: ``NRL_USE_FASTOKENS=1`` forces on, any other value forces off.

See: https://github.com/Atero-ai/fast-tokens
"""

import logging
import os

logger = logging.getLogger(__name__)

_patched = False


def maybe_patch_fastokens(enabled: bool) -> None:
    """Apply the fastokens monkey-patch when enabled.

    Args:
        enabled: The resolved ``policy.tokenizer.use_fastokens`` config value.
            The ``NRL_USE_FASTOKENS`` env var, when set, overrides this: ``"1"``
            forces on, anything else forces off.
    """
    global _patched
    if _patched:
        return

    override = os.environ.get("NRL_USE_FASTOKENS")
    if override is not None:
        enabled = override == "1"

    if not enabled:
        return

    try:
        import fastokens

        fastokens.patch_transformers()
        _patched = True
        logger.info(
            "fastokens monkey-patch applied — accelerated BPE tokenization enabled"
        )
    except ImportError:
        logger.warning("fastokens is enabled but not installed.")
    except Exception:
        logger.exception("Failed to apply fastokens monkey-patch")
