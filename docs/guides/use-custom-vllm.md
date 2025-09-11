# Experiment with Custom vLLM

This guide explains how to use your own version of vLLM while leveraging a pre-compiled vLLM wheel, so you don't have to recompile the C++ source code.

## Clone and Build Your Custom vLLM

Clone your vLLM fork and build it using the provided script. For example:

```sh
# Usage: bash tools/build-custom-vllm.sh <GIT_URL> <GIT_BRANCH> <VLLM_PRECOMILED_WHEEL_COMMI_FROM_MAINT>
bash tools/build-custom-vllm.sh https://github.com/terrykong/vllm.git terryk/demo-custom-vllm d8ee5a2ca4c73f2ce5fdc386ce5b4ef3b6e6ae70

# [INFO] pyproject.toml updated. NeMo RL is now configured to use the local vLLM at 3rdparty/vllm.
# [INFO] Verify this new vllm version by running:
# 
# VLLM_COMMIT=a3319f4f04fbea7defe883e516df727711e516cd \
# VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/a3319f4f04fbea7defe883e516df727711e516cd/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl \
#   uv run --extra vllm vllm serve Qwen/Qwen3-0.6B
# 
# [INFO] For more information on this custom install, visit https://github.com/NVIDIA-NeMo/RL/blob/main/docs/guides/use-custom-vllm.md
# [IMPORTANT] Remember to set the shell variable 'export VLLM_USE_PRECOMPILED=1' when running NeMo RL apps with this custom vLLM to avoid re-compiling.
```

Make note of the `VLLM_COMMIT` and `VLLM_PRECOMPILED_WHEEL_LOCATION` since you will have to set those
in your shell environment.

This script does the following:
1. Clones the `vllm` you specify at a particular branch.
2. Builds `vllm`.
3. Updates NeMo RL's pyproject.toml to work with this `vllm`.
4. Updates `uv.lock`.

Make sure to add the updated `pyproject.toml` and `uv.lock` to version control so that your branch can be reproduced by others.

## Verify Your Custom vLLM in Isolation
Test your setup to ensure your custom vLLM is being used:
```sh
uv run --extra vllm python -c 'import vllm; print(f"Successfully imported vLLM version: {vllm.__version__}")'
# Uninstalled 1 package in 1ms
# Installed 1 package in 2ms
# Hi! If you see this, you're using a custom version of vLLM for the purposes of this tutorial
# INFO 06-18 09:22:44 [__init__.py:244] Automatically detected platform cuda.
# Successfully imported vLLM version: 0.0.1.dev1+g69d5add74.d20250910
```

If you don't see the log message `Hi! If you see this...`, it's because this message is unique to the tutorial's specific `vLLM` fork. It was added in [this commit](https://github.com/terrykong/vllm/commit/69d5add744e51b988e985736f35c162d3e87b683) and doesn't exist in the main `vLLM` project.

## Running NeMo RL Apps with Custom vLLM

To ensure that vllm doesn't try to re-compile, you must run with `VLLM_COMMIT` and `VLLM_PRECOMPILED_WHEEL_LOCATION` set in the environment:

```sh
# (copied from build-custom-vllm.sh) Ensure's vllm doesn't try to recompile c++ source
export VLLM_COMMIT=d8ee5a2ca4c73f2ce5fdc386ce5b4ef3b6e6ae70
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/d8ee5a2ca4c73f2ce5fdc386ce5b4ef3b6e6ae70/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
# Ensures worker venvs are rebuilt to use the custom vllm. Otherwise it will use the cached version in the cached venvs
export NRL_FORCE_REBUILD_VENVS=true

uv run examples/run_grpo_math.py
```

## Re-building the NeMo RL Docker Image

Using a custom vllm may require you to rebuild the docker image. The two most common reasons are:

1. The `ray` version was changed, so you **must** rebuild the image to allow `ray.sub` to start the ray cluster with the same version as the application.
2. Many dependencies changed and add a large overhead when `NRL_FORCE_REBUILD_VENVS=true` is set to rebuild venvs, so you wish to cache the dependencies in the image to avoid re-build/re-pulling wheels.

For convenience, you can build the image and set `VLLM_COMMIT` and `VLLM_PRECOMPILED_WHEEL_LOCATION` as
env vars to avoid the application needing to set them.

```sh
# Replace VLLM_COMMIT and VLLM_PRECOMPILED_WHEEL_LOCATION with the values output from build-custom-vllm.sh
docker buildx build \
  --build-arg VLLM_COMMIT=d8ee5a2ca4c73f2ce5fdc386ce5b4ef3b6e6ae70 \
  --build-arg VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/d8ee5a2ca4c73f2ce5fdc386ce5b4ef3b6e6ae70/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl \
  --target release \
  --build-context nemo-rl=. \
  -f docker/Dockerfile \
  --tag <registry>/nemo-rl:latest \
  --push \
  .
```