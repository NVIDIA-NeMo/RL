```
uv cache clean
rm -rf .venv venvs
time uv run nemo_rl/utils/prefetch_venvs.py 'Vllm|DTensor'

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

Run an example
```
uv run examples/run_grpo_math.py
```
