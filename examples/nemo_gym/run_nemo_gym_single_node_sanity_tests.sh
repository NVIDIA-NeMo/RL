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

# Fail on errors
set -e

uv sync --all-groups --extra nemo_gym

# Stop pesky previous Ray servers that may have not been able to spin down from previous users.
uv run ray stop --force
uv run python -c "import ray; ray.shutdown()"

# The first time I ran this, it took roughly 5 mins to setup the vLLM deps.
# This took me 2-3 mins to run this one test.
# NeMo RL test. This should pass no matter what the Gym setup is.
./tests/run_unit.sh unit/models/generation/test_vllm_generation.py::test_vllm_generate_text

# NeMo Gym uses an OpenAI compatible endpoint under the hood. This tests the implementation for this server.
./tests/run_unit.sh unit/models/generation/test_vllm_generation.py::test_vllm_http_server

# NeMo Gym communicates not using token ids, but in OpenAI schema. There are some edge cases we need to handle (e.g. token merging upon retokenization, multiple most efficient retokenizations, etc).
./tests/run_unit.sh unit/models/generation/test_vllm_generation.py::test_VllmAsyncGenerationWorker_replace_prefix_tokens
./tests/run_unit.sh unit/models/generation/test_vllm_generation.py::test_replace_prefix_tokens_empty_model_prefix_returns_template
./tests/run_unit.sh unit/models/generation/test_vllm_generation.py::test_replace_prefix_tokens_missing_eos_in_template_prefix_raises
./tests/run_unit.sh unit/models/generation/test_vllm_generation.py::test_replace_prefix_tokens_tokenizer_without_eos_raises
./tests/run_unit.sh unit/models/generation/test_vllm_generation.py::test_replace_prefix_tokens_uses_last_eos_in_template_prefix
./tests/run_unit.sh unit/models/generation/test_vllm_generation.py::test_vllm_http_server_correct_merged_tokens_matches_baseline

# NeMo RL test. This should pass no matter what the Gym setup is.
./tests/run_unit.sh unit/environments/test_math_environment.py::test_math_env_step_basic

# NeMo Gym integrates directly into NeMo RL as an Environment since that is the cleanest way. This tests the NeMo Gym integration logic and correctness.
./tests/run_unit.sh unit/environments/test_nemo_gym.py::test_nemo_gym_sanity

# NeMo Gym uses a separate rollout loop inside grpo_train in NeMo RL. This tests the e2e rollout functionality and correctness.
./tests/run_unit.sh unit/experience/test_rollouts.py::test_run_async_nemo_gym_rollout
