## setup
Starting from your container, run these to prepare your environment. On my dev machine took about a minute to prepare
```
# Wiping uv cache clean and removing venvs just to get accurate timings
uv cache clean
rm -rf .venv venvs

# cache the venvs
time uv run nemo_rl/utils/prefetch_venvs.py 'Vllm|DTensor|Megatron'

...
#[SKIPPED] nemo_rl.environments.code_environment.CodeEnvironment
#[SKIPPED] nemo_rl.environments.games.sliding_puzzle.SlidingPuzzleEnv
#[SKIPPED] nemo_rl.environments.tools.retriever.RAGEnvironment
#[CACHED] nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker
#[CACHED] nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker
#[CACHED] nemo_rl.models.policy.dtensor_policy_worker.DTensorPolicyWorker
#
#real    0m58.157s
#user    0m38.581s
#sys     0m52.296s
```

## examples

Run a simple example that just shows how a weight update is done with 1 GPU (should work with A6000) - written for just dtensor, can be generalized for mcore
```
uv run single_update.py
```

Run a full e2e GRPO example (dtensor):
```
uv run examples/run_grpo_math.py
```

Run a full e2e GRPO example (dtensor):
```
uv run examples/run_grpo_math.py --config=examples/configs/grpo_math_1B_megatron.yaml
```
