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
"""Ground-truth reference block for the privileged value: the prompt's reference answer.

:func:`build_privilege_prefix` is the prefix builder
(``value.privileged_value.prefix_builder_fqn``) for prompts whose
``extra_env_info`` carries a ``ground_truth``, as the math environments' do.
"""

from typing import Any

from nemo_rl.algorithms.ppo_utils.privileged_value import cap_tokens, count_tokens
from nemo_rl.models.value.config import PrivilegedValueConfig

GROUND_TRUTH_KEY = "ground_truth"


def build_privilege_prefix(
    extra_env_info: dict[str, Any],
    tokenizer: Any,
    cfg: PrivilegedValueConfig,
) -> tuple[str, dict[str, Any]]:
    """Prefix builder: the prompt's ground truth, cut to ``max_total_tokens``.

    Raises:
        ValueError: The prompt carries no ground truth. Training on would
            silently produce a blind critic under a privileged label.
    """
    answer = str(extra_env_info.get(GROUND_TRUTH_KEY) or "")
    if not answer:
        raise ValueError(
            f"ground-truth privilege prefix: extra_env_info has no {GROUND_TRUTH_KEY!r}."
        )
    body, truncated = cap_tokens(answer, cfg.max_total_tokens, tokenizer)
    stats = {
        "truncated_fields": [GROUND_TRUTH_KEY] if truncated else [],
        "wanted_tokens": count_tokens(answer, tokenizer, cfg.max_total_tokens),
        "kept_tokens": count_tokens(body, tokenizer, cfg.max_total_tokens),
    }
    block = f"<reference>\n<{GROUND_TRUTH_KEY}>\n{body}\n</{GROUND_TRUTH_KEY}>\n</reference>"
    return block, stats
