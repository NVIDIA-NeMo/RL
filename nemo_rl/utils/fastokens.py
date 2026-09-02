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

The config field controls this NeMo-RL-side patch when no environment override
is set. ``NRL_USE_FASTOKENS`` is the top-level NeMo-RL override and is mirrored
to ``VLLM_USE_FASTOKENS`` so vLLM workers follow the same setting. A standalone
``VLLM_USE_FASTOKENS`` setting is left for vLLM and does not control this
NeMo-RL-side patch.

See: https://github.com/Atero-ai/fast-tokens
"""

import logging
import os

logger = logging.getLogger(__name__)

_patched = False
_NRL_ENV_VAR = "NRL_USE_FASTOKENS"
_VLLM_ENV_VAR = "VLLM_USE_FASTOKENS"


def normalize_fastokens_env() -> None:
    """Mirror NeMo-RL's fastokens override to vLLM's fastokens flag."""
    nrl_value = os.environ.get(_NRL_ENV_VAR)
    if nrl_value is None:
        return

    os.environ[_VLLM_ENV_VAR] = "1" if nrl_value == "1" else "0"


def maybe_patch_fastokens(enabled: bool) -> None:
    """Apply the fastokens monkey-patch when enabled.

    Args:
        enabled: The resolved ``policy.tokenizer.use_fastokens`` config value.
            The ``NRL_USE_FASTOKENS`` env var, when set, overrides this:
            ``"1"`` forces on, anything else forces off.
    """
    global _patched
    if _patched:
        return

    normalize_fastokens_env()
    override = os.environ.get(_NRL_ENV_VAR)
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
