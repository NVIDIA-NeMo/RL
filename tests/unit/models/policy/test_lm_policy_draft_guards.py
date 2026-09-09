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

from unittest.mock import MagicMock

import pytest

from nemo_rl.models.policy.lm_policy import Policy


def _draft_config(
    context_parallel_size=1,
    pipeline_model_parallel_size=1,
    sequence_packing_enabled=True,
    use_fused_linear_logprobs=False,
):
    return {
        "megatron_cfg": {
            "enabled": True,
            "context_parallel_size": context_parallel_size,
            "pipeline_model_parallel_size": pipeline_model_parallel_size,
            "use_fused_linear_logprobs": use_fused_linear_logprobs,
        },
        "dtensor_cfg": {"enabled": False},
        "draft": {"enabled": True},
        "sequence_packing": {"enabled": sequence_packing_enabled},
    }


def _init_policy(config):
    return Policy(
        cluster=MagicMock(),
        config=config,
        tokenizer=MagicMock(),
    )


def test_draft_with_context_parallelism_requires_packing():
    """The trunk ring needs the THD zigzag layout, so CP>1 rejects unpacked
    batches at setup and passes the guard once sequence packing is on."""
    with pytest.raises(ValueError, match="requires sequence packing"):
        _init_policy(
            _draft_config(context_parallel_size=2, sequence_packing_enabled=False)
        )
    try:
        _init_policy(_draft_config(context_parallel_size=2))
    except ValueError as e:
        assert "requires sequence packing" not in str(e)
    except Exception:
        # Reaching config plumbing beyond the draft guards is sufficient.
        pass


@pytest.mark.parametrize("sequence_packing_enabled", [True, False])
def test_draft_with_fused_linear_logprobs_is_rejected(sequence_packing_enabled):
    """Draft training must reject use_fused_linear_logprobs in both layouts:
    the fused path never materializes the teacher's full next-token logits."""
    with pytest.raises(ValueError, match="use_fused_linear_logprobs"):
        _init_policy(
            _draft_config(
                sequence_packing_enabled=sequence_packing_enabled,
                use_fused_linear_logprobs=True,
            )
        )
