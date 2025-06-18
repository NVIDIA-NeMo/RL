# Experimenting with Custom vLLM

This guide describes how to use a custom vllm version while making use of a pre-compiled vllm wheel to avoid
having to recompile the c++ source.

## Setup

First, you need to clone and build vllm. Here is an example using a custom fork:

```sh
# Usage: bash tools/build-custom-vllm.sh <GIT_URL> <GIT_BRANCH> <VLLM_PRECOMILED_WHEEL_COMMIT>
bash tools/build-custom-vllm.sh https://github.com/terrykong/vllm.git terryk/demo-custom-vllm a3319f4f04fbea7defe883e516df727711e516cd
```

Then you need to make the following edits to [pyproject.toml](https://github.com/NVIDIA-NeMo/RL/blob/main/pyproject.toml) so the `vllm` dependency uses the local clone as opposed to pypi.

Change:
```toml
[project.optional-dependencies]
vllm = [
    #"vllm==0.9.0",  # <-- BEFORE
    "vllm",          # <-- AFTER
]

# ...<OMITTED>

[tool.uv.sources]
# ...<OMITTED>
vllm = { path = "3rdparty/vllm", editable = true }  # <-- ADD AN ENTRY

# ...<OMITTED>

[tool.uv]
no-build-isolation-package = ["transformer-engine-torch", "transformer-engine"]          # <-- BEFORE
no-build-isolation-package = ["transformer-engine-torch", "transformer-engine", "vllm"]  # <-- AFTER
```

Then re-lock the environment:

```sh
uv pip install setuptools_scm  # vLLM doesn't declare this build dependency so we install it manually
uv lock
```

Now test if you're using this custom version:
```sh
uv run --extra vllm python -c 'import vllm; print("Successfully imported vLLM")'
# Uninstalled 1 package in 1ms
# Installed 1 package in 2ms
# Hi! If you see this, you're using a custom version of vLLM for the purposes of this tutorial
# INFO 06-18 09:22:44 [__init__.py:244] Automatically detected platform cuda.
# Successfully imported vLLM
```

You will not see the log `Hi! If you see this...` if you are using your own fork since that's something just added to the vLLM fork used in this tutorial ([source](https://github.com/terrykong/vllm/commit/69d5add744e51b988e985736f35c162d3e87b683)).