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
"""
This script exists to help load any unit asset that requires special handling.

The initial reason for this was to help with Nemotron-H which has a requirement
to have mamaba-ssm in the base environment in order to initialize a dummy model. Since
the unit tests should be runable with the base environment (without mamba-ssm),
we use ray.remotes to build the asset here. We do this outside of a fixture
like the other test assets because this one sometimes takes a while to build. This
extra setup time can sometimes cause timeouts in the unit tests if unlucky.
"""

import os

import ray

from nemo_rl.distributed.virtual_cluster import PY_EXECUTABLES
from nemo_rl.utils.venvs import create_local_venv

TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
TEST_ASSETS_DIR = os.path.join(TESTS_DIR, "test_assets")


def build_tiny_nemotron5_h_checkpoint(model_path: str) -> None:
    import shutil

    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    config = AutoConfig.from_pretrained(
        "nvidia/Nemotron-H-8B-Base-8K", trust_remote_code=True
    )
    config.hybrid_override_pattern = "M*-"
    config.num_hidden_layers = 3
    config.intermediate_size = 32
    config.hidden_size = 256
    config.num_attention_heads = 8
    config.mamba_num_heads = 8
    config.num_key_value_heads = 8
    config.n_groups = 1

    model = AutoModelForCausalLM.from_config(config, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(
        "nvidia/Nemotron-H-8B-Base-8K", trust_remote_code=True
    )

    shutil.rmtree(model_path, ignore_errors=True)
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)
    print(f"✓ Built tiny Nemotron-H asset at: {model_path}")


def main() -> None:
    os.makedirs(TEST_ASSETS_DIR, exist_ok=True)

    target = os.path.join(TEST_ASSETS_DIR, "tiny_nemotron5_h_with_nemotron_tokenizer")

    # Create Automodel env venv
    automodel_python = create_local_venv(
        py_executable=PY_EXECUTABLES.AUTOMODEL, venv_name="automodel_env"
    )

    ############################################################################
    # Add other remote calls here
    ############################################################################
    # Submit as list of remote calls and wait individually
    remote_calls = [
        ray.remote(build_tiny_nemotron5_h_checkpoint)
        .options(
            num_gpus=0.01,  # tiny reservation to satisfy CUDA-inspecting deps
            runtime_env={"py_executable": automodel_python},
            name="build-nemotron5h",
        )
        .remote(target)
    ]

    for obj_ref in remote_calls:
        ray.get(obj_ref)


if __name__ == "__main__":
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True, include_dashboard=False)
    try:
        main()
    finally:
        ray.shutdown()
