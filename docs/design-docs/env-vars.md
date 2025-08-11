# Environment Variable Precedence in NeMo RL

There are a number of ways to pass environment variables to Ray workers in NeMo RL. This document explains each of the methods and why they are useful.

## Precedence Order

### 1. Ray Runtime Environment Variables (lowest)
- Set via `ray.remote(runtime_env={'env_vars': {...}})` decorators.
- Applied to all instances of specific worker classes. These define the default environment variables for the class if not overwritten by a method of higher precedence.
- Example: `@ray.remote(runtime_env=get_runtime_env_for_policy_worker("megatron_policy_worker"))`. See [here](https://github.com/NVIDIA-NeMo/RL/blob/def76820d7838c63c1ee4900e63f73a93d927ff2/nemo_rl/models/policy/megatron_policy_worker.py#L338) where `get_runtime_env_for_policy_worker` will be applied to all instances of `MegatronPolicyWorker`.

### 2. System-level Environment Variables (medium)
- Set via `export` in shell or `os.environ` in Python.
- Useful for controlling environment variables from a high level. If not overwritten by higher priority methods, all workers will inherit these environment variables.
- Example: `export HF_TOKEN=<your_token>`

### 3. YAML Configuration `env_vars` (high)
- Set in YAML config files under `policy.megatron_cfg.env_vars` or `policy.dtensor_cfg.env_vars`.
- Useful for controlling environment variables on an experiment level.
- Example:
  ```yaml
  policy:
    megatron_cfg:
      env_vars:
        PYTORCH_CUDA_ALLOC_CONF: "expandable_segments:False"
  ```

### 4. Worker-specific `configure_worker` Method (highest)
- Set via static `configure_worker` method in worker classes.
- Applied to specific worker instances based on configuration.
- See an example in `VllmGenerationWorker` [here](https://github.com/NVIDIA-NeMo/RL/blob/def76820d7838c63c1ee4900e63f73a93d927ff2/nemo_rl/models/generation/vllm.py#L88).
