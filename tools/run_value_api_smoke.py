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

"""Run the one-GPU value API parity/lifecycle smoke without the full test fixtures."""

import os
from pathlib import Path

import pytest
import ray
import torch
from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM


class SmokeFixtures:
    @pytest.fixture(scope="session", autouse=True)
    def ray_cluster(self):
        assert torch.cuda.device_count() == 1, "This smoke must see exactly one GPU"
        ray.init(num_cpus=8, num_gpus=1, include_dashboard=False)
        yield
        ray.shutdown()

    @pytest.fixture(scope="session")
    def tiny_qwen2_model_path(self, tmp_path_factory):
        model_path = tmp_path_factory.mktemp("tiny_value")
        config = Qwen2Config(
            num_hidden_layers=2,
            hidden_size=64,
            intermediate_size=32,
            num_attention_heads=2,
            num_key_value_heads=2,
            vocab_size=151936,
            tie_word_embeddings=False,
        )
        torch.manual_seed(42)
        model = Qwen2ForCausalLM(config)
        model.save_pretrained(model_path)
        tokenizer = AutoTokenizer.from_pretrained(
            os.environ["VALUE_API_TOKENIZER_PATH"], local_files_only=True
        )
        tokenizer.save_pretrained(model_path)
        return str(model_path)


if __name__ == "__main__":
    print(
        "VALUE_API_GPU_SMOKE",
        {
            "torch": torch.__version__,
            "gpus": torch.cuda.device_count(),
            "repo": str(Path.cwd()),
        },
        flush=True,
    )
    raise SystemExit(
        pytest.main(
            [
                "--noconftest",
                "-o",
                "addopts=",
                "-v",
                "-s",
                "--tb=short",
                "--junitxml=" + os.environ["VALUE_API_RESULT_PATH"],
                "tests/unit/models/value/test_megatron_value_split_parity.py",
                "-k",
                "single_gpu",
            ],
            plugins=[SmokeFixtures()],
        )
    )
