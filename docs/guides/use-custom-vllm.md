# Experiment with Custom vLLM

This guide explains how to use your own version of vLLM while leveraging a pre-compiled vLLM wheel, so you don't have to recompile the C++ source code.

## Clone and Build Your Custom vLLM

Clone your vLLM fork and build it using the provided script. For example:

```sh
# Usage: bash tools/build-custom-vllm.sh <GIT_URL> <GIT_BRANCH> <VLLM_PRECOMILED_WHEEL_COMMIT>
bash tools/build-custom-vllm.sh https://github.com/terrykong/vllm.git terryk/demo-custom-vllm a3319f4f04fbea7defe883e516df727711e516cd
```
## Update `pyproject.toml` to Use Your Local vLLM
Edit your [pyproject.toml](https://github.com/NVIDIA-NeMo/RL/blob/main/pyproject.toml) so that the  `vLLM`  dependency points to your local clone instead of PyPI.

**Change the pyproject.toml:**
```toml
# Add setuptools_scm
[project]
# ...<OMITTED>
dependencies = [
# ...<OMITTED>
    "setuptools_scm",  # <-- Add
# ...<OMITTED>
]

# Change the vLLM dependency:

[project.optional-dependencies]
vllm = [
    #"vllm==0.10.0",  # <-- BEFORE
    "vllm",           # <-- AFTER
]

# ...<OMITTED>

# Add a local source entry:
[tool.uv.sources]
# ...<OMITTED>
vllm = { path = "3rdparty/vllm", editable = true }  # <-- ADD AN ENTRY

# ...<OMITTED>

# Update build isolation packages:
[tool.uv]
no-build-isolation-package = ["transformer-engine-torch", "transformer-engine"]          # <-- BEFORE
no-build-isolation-package = ["transformer-engine-torch", "transformer-engine", "vllm"]  # <-- AFTER
```
## Re-Lock and Install Dependencies
Install any missing build dependencies and re-lock your environment:

```sh
uv pip install setuptools_scm  # vLLM doesn't declare this build dependency so we install it manually
uv lock
```
## Verify Your Custom vLLM
Test your setup to ensure your custom vLLM is being used:
```sh
uv run --extra vllm python -c 'import vllm; print("Successfully imported vLLM")'
# Uninstalled 1 package in 1ms
# Installed 1 package in 2ms
# Hi! If you see this, you're using a custom version of vLLM for the purposes of this tutorial
# INFO 06-18 09:22:44 [__init__.py:244] Automatically detected platform cuda.
# Successfully imported vLLM
```

If you don't see the log message `Hi! If you see this...`, it's because this message is unique to the tutorial's specific `vLLM` fork. It was added in [this commit](https://github.com/terrykong/vllm/commit/69d5add744e51b988e985736f35c162d3e87b683) and doesn't exist in the main `vLLM` project.
