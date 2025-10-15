uv sync --group={build,docs,dev,test} --extra penguin

# Stop pesky previous Ray servers that may have not been able to spin down from previous users.
uv run ray stop --force
uv run python -c "import ray; ray.shutdown()"

# This will take 5-10 mins
uv run tests/unit/prepare_unit_test_assets.py

# The first time I ran this, it took roughly 5 mins to setup the vLLM deps.
# This took me 2-3 mins to run this one test.
# NeMo RL test. This should pass no matter what the Gym setup is.
uv run --group test bash tests/run_unit.sh unit/models/generation/test_vllm_generation.py::test_vllm_generate_text

# NeMo Gym uses an OpenAI compatible endpoint under the hood. This tests the implementation for this server.
uv run --group test bash tests/run_unit.sh unit/models/generation/test_vllm_generation.py::test_vllm_http_server

# NeMo Gym communicates not using token ids, but in OpenAI schema. There are some edge cases we need to handle (e.g. token merging upon retokenization, multiple most efficient retokenizations, etc).
uv run --group test bash tests/run_unit.sh unit/models/generation/test_vllm_generation.py::test_VllmAsyncGenerationWorker_replace_prefix_tokens
uv run --group test bash tests/run_unit.sh 'unit/models/generation/test_vllm_generation.py::test_replace_prefix_tokens*'

# NeMo RL test. This should pass no matter what the Gym setup is.
uv run --group test bash tests/run_unit.sh unit/environments/test_math_environment.py::test_math_env_step_basic

# NeMo Gym integrates directly into NeMo RL as an Environment since that is the cleanest way. This tests the NeMo Gym integration logic and correctness.
uv run --group test bash tests/run_unit.sh unit/environments/test_penguin.py::test_penguin_sanity

# NeMo Gym uses a separate rollout loop inside grpo_train in NeMo RL. This tests the e2e rollout functionality and correctness.
uv run --group test bash tests/run_unit.sh unit/experience/test_rollouts.py::test_run_async_penguin_rollout
