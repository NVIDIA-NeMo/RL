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

import pytest
import ray

from tests.unit.models.policy.test_dtensor_worker import create_test_config

# Define a custom marker for model configuration tests
pytestmark = pytest.mark.modelconfig

from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.models.policy.lm_policy import Policy


def compare_model_configs(config_v1: dict, config_v2: dict) -> list[str]:
    """
    Compare two model configurations and return a list of discrepancies.

    Args:
        config_v1: Model config from dtensor worker v1
        config_v2: Model config from dtensor worker v2

    Returns:
        List of discrepancy descriptions. Empty list if configs are equivalent.
    """
    discrepancies = []

    def compare_dicts(d1, d2, path=""):
        """Recursively compare two dictionaries."""
        all_keys = set(d1.keys()) | set(d2.keys())

        for key in all_keys:
            current_path = f"{path}.{key}" if path else key

            if key not in d1:
                discrepancies.append(f"Key '{current_path}' missing in v1 config")
            elif key not in d2:
                discrepancies.append(f"Key '{current_path}' missing in v2 config")
            else:
                val1, val2 = d1[key], d2[key]

                if isinstance(val1, dict) and isinstance(val2, dict):
                    compare_dicts(val1, val2, current_path)
                elif val1 != val2:
                    discrepancies.append(
                        f"Value mismatch at '{current_path}': v1={val1}, v2={val2}"
                    )

    compare_dicts(config_v1, config_v2)
    return discrepancies


@pytest.mark.hf_gated
@pytest.mark.parametrize(
    "model_fixture_name,tp,cp,sequence_parallel,cpu_offload,activation_checkpointing",
    [
        # TP=2, CP=1
        ("tiny_qwen2_model_path", 2, 1, False, False, False),
        ("tiny_llama_model_path", 2, 1, False, False, False),
        ("tiny_qwen3_model_path", 2, 1, False, False, False),
        ("tiny_gemma3_model_path", 2, 1, False, False, False),
        # TP=1, CP=2
        ("tiny_qwen2_model_path", 1, 2, False, False, False),
        ("tiny_llama_model_path", 1, 2, False, False, False),
        ("tiny_qwen3_model_path", 1, 2, False, False, False),
    ],
)
def test_dtensor_worker_v1_v2_model_config_equivalence(
    request,
    two_gpu_virtual_cluster,
    model_fixture_name,
    tp,
    cp,
    sequence_parallel,
    cpu_offload,
    activation_checkpointing,
):
    """Test that dtensor worker v1 and v2 produce equivalent model configurations."""
    # Get the actual model path from the fixture name
    model_name = request.getfixturevalue(model_fixture_name)
    # Create v1 configuration
    config_v1 = create_test_config(
        model_name=model_name,
        tp=tp,
        cp=cp,
        sequence_parallel=sequence_parallel,
        cpu_offload=cpu_offload,
        activation_checkpointing=activation_checkpointing,
        dtensor_v2=False,  # Use v1 worker
    )
    # Create and test v1 policy first
    print("Creating policy with v1 worker...")
    policy_v1 = Policy(
        tokenizer=get_tokenizer(config_v1["tokenizer"]),
        config=config_v1,
        init_optimizer=False,
        init_reference_model=False,
        cluster=two_gpu_virtual_cluster,
        name_prefix="lm_policy_v1",
    )

    model_config_v1 = ray.get(
        policy_v1.worker_group.workers[0].return_model_config.remote()
    )
    policy_v1.shutdown()

    # Create v2 configuration
    config_v2 = create_test_config(
        model_name=model_name,
        tp=tp,
        cp=cp,
        sequence_parallel=sequence_parallel,
        cpu_offload=cpu_offload,
        activation_checkpointing=activation_checkpointing,
        dtensor_v2=True,  # Use v2 worker
    )
    policy_v2 = Policy(
        tokenizer=get_tokenizer(config_v2["tokenizer"]),
        config=config_v2,
        init_optimizer=False,
        init_reference_model=False,
        cluster=two_gpu_virtual_cluster,
        name_prefix="lm_policy_v2",
    )

    model_config_v2 = ray.get(
        policy_v2.worker_group.workers[0].return_model_config.remote()
    )
    policy_v2.shutdown()

    config_v1_dict = vars(model_config_v1)
    config_v2_dict = vars(model_config_v2)
    config_v1_dict.pop("nemo_version", None)
    config_v2_dict.pop("nemo_version", None)
    config_v1_dict.pop("pad_token_id", None)
    config_v2_dict.pop("pad_token_id", None)

    discrepancies = compare_model_configs(config_v1_dict, config_v2_dict)

    assert not discrepancies, (
        f"Model configurations differ between v1 and v2 approaches for {model_name}"
    )
