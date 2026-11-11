# Experiment with Custom vLLM

This guide explains how to use your own version of vLLM while leveraging a pre-compiled vLLM wheel, so you don't have to recompile the C++ source code.

## Clone and Build Your Custom vLLM

Clone your vLLM fork and build it using the provided script. For example:

```sh
# Usage: bash tools/build-custom-vllm.sh <GIT_URL> <GIT_REF> <VLLM_PRECOMPILED_WHEEL_LOCATION>
bash tools/build-custom-vllm.sh https://github.com/terrykong/vllm.git terryk/demo-custom-vllm https://wheels.vllm.ai/862f2ef893d9751db0a92bd2d4ae0e3d9677872f/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl

# [INFO] pyproject.toml updated. NeMo RL is now configured to use the local vLLM at 3rdparty/vllm.
# [INFO] Verify this new vllm version by running:
#
# VLLM_PRECOMPILED_WHEEL_LOCATION=http://.....whl \
#   uv run --extra vllm vllm serve Qwen/Qwen3-0.6B
#
# [INFO] For more information on this custom install, visit https://github.com/NVIDIA-NeMo/RL/blob/main/docs/guides/use-custom-vllm.md
# [IMPORTANT] Remember to set the shell variable 'VLLM_PRECOMPILED_WHEEL_LOCATION' when running NeMo RL apps with this custom vLLM to avoid re-compiling.
```

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

To ensure the custom vLLM install is setup properly in NeMo RL applications, always run the following before anything:

```sh
# Ensures vLLM uses the precompiled wheel and avoids recompiling C++ sources
export VLLM_PRECOMPILED_WHEEL_LOCATION=https://wheels.vllm.ai/862f2ef893d9751db0a92bd2d4ae0e3d9677872f/vllm-1.0.0.dev-cp38-abi3-manylinux1_x86_64.whl
# Ensures worker venvs are rebuilt to use the custom vLLM. Otherwise it may use the cached version in cached venvs
export NRL_FORCE_REBUILD_VENVS=true
# This isn't necessary if you only do `uv run foobar.py`, but may be needed if you switching between optional extras `uv run --extra vllm foobar.py`. If you are unsure if you need this, it's safer to include it.
uv pip install setuptools_scm
```

Then run your application:
```sh
uv run examples/run_grpo.py
```

## Re-building the NeMo RL Docker Image

Using a custom vllm may require you to rebuild the docker image. The two most common reasons are:

1. The `ray` version was changed, so you **must** rebuild the image to allow `ray.sub` to start the ray cluster with the same version as the application.
2. Many dependencies changed and add a large overhead when `NRL_FORCE_REBUILD_VENVS=true` is set to rebuild venvs, so you wish to cache the dependencies in the image to avoid re-build/re-pulling wheels.

For convenience, you can have the image build your custom vLLM by running the same script inside the Docker build.
Pass `--build-arg BUILD_CUSTOM_VLLM=1` to enable this path; the build will create `3rdparty/vllm` and source `3rdparty/vllm/nemo-rl.env` automatically.

```sh
docker buildx build \
  --build-arg BUILD_CUSTOM_VLLM=1 \
  --target release \
  --build-context nemo-rl=. \
  -f docker/Dockerfile \
  --tag <registry>/nemo-rl:latest \
  --push \
  .
```

### SSH Setup for Private Repositories

If your custom vLLM is hosted in a **private repository** (e.g., internal GitLab), you need to set up SSH agent forwarding for Docker to clone it during the build.

#### Prerequisites
1. Your SSH key must be registered on the Git server (GitLab/GitHub)
2. The key must **not be expired** - check your Git server's SSH key settings
3. The key must be loaded into your local ssh-agent

#### Step 1: Verify your SSH key works

```sh
# For GitLab (adjust host/port as needed)
ssh -T git@gitlab.example.com -p 12051

# You should see: "Welcome to GitLab, @username!"
# If you see "Your SSH key has expired", renew it on the server
```

#### Step 2: Load your SSH key into the agent

```sh
# Check if an ssh-agent is already running
echo $SSH_AUTH_SOCK

# If empty, start one (this also sets SSH_AUTH_SOCK which `docker buildx` expects to be set when using `--ssh default`)
eval "$(ssh-agent -s)"

# Clear any old/expired keys from the agent
ssh-add -D

# Add your SSH key (use the key registered on your Git server)
ssh-add ~/.ssh/id_ed25519

# Verify it's loaded
ssh-add -l
```

#### Step 3: Run the Docker build with SSH forwarding

```sh
docker buildx build \
  --build-arg BUILD_CUSTOM_VLLM=1 \
  --target release \
  --build-context nemo-rl=. \
  -f docker/Dockerfile \
  --ssh default \
  --tag <registry>/nemo-rl:latest \
  --push \
  .
```

## Running Applications with a Custom vLLM Container

When using a container built with custom vLLM, **use the frozen environment workflow** (bare `python`) instead of `uv run` with `NRL_FORCE_REBUILD_VENVS=true`.

```sh
# Recommended: use bare python (frozen environment)
python examples/run_grpo.py

# NOT recommended with custom vLLM containers:
# uv run examples/run_grpo.py
# or
# NRL_FORCE_REBUILD_VENVS=true uv run examples/run_grpo.py
```

### Why Not Use `uv run` or Rebuild Venvs?

Rebuilding worker virtual environments (via `uv run` or `NRL_FORCE_REBUILD_VENVS=true`) requires having the custom vLLM compiled locally. However, compiling vLLM requires a container environment with the correct CUDA toolchain—creating a chicken-and-egg problem.

The container already has vLLM built and cached in the frozen environments. Using bare `python` leverages these pre-built environments directly, avoiding the need to recompile vLLM at runtime.

> [!TIP]
> For more details on frozen environments and how they differ from `uv run`, see the [Dependency Management](../design-docs/dependency-management.md#frozen-environments) documentation.

## NIXL prefill KV lease verification backport

A prefill KV lease can expire while decode is waiting to fetch KV. Prefill may
then reuse those blocks for another request. The backport of
[vLLM PR #62065](https://github.com/vllm-project/vllm/pull/62065), pinned to
`03b2648165d24128611cd15818f4ecf51c0a0205`, adds lease-slot nonce verification so
that decode rejects an expired handoff before using its KV for generation.

This is an opt-in backport for the qualified build
`0.30.0+g116da885561b.cu132`. NeMo RL's default vLLM dependency is not changed.
The tool checks exact source SHA-256 values and rejects other builds, including
unqualified 0.29 builds. The checked-in manifest records the upstream revision,
original and patched file hashes, and backport wire version `15062065`.

NeMo RL's runtime monkey patches live in
`nemo_rl/models/generation/vllm/patches.py`. This backport instead belongs in the
image build, after installing the custom vLLM package and creating frozen worker
environments, but before importing vLLM or starting any worker. It changes six
source files spanning scheduler, worker, metadata, and lease handling; rewriting
files after imports would leave cached modules using different implementations.
For a different vLLM revision, use the custom build workflow above and validate
the matching upstream fix instead of bypassing these source checks.

Identify the installed `vllm` package directory using each actual worker
interpreter (this query does not import vLLM):

```sh
/path/to/worker/venv/bin/python -c 'import importlib.util; print(next(iter(importlib.util.find_spec("vllm").submodule_search_locations)))'
```

During image construction, check and apply to **every distinct package root**
used by P, D, TP ranks, and Ray workers. Replace the example paths with those
reported by the worker interpreters. The installer uses only Python's standard
library and requires Git; it does not install or resolve dependencies.

```sh
python tools/apply_vllm_nixl_lease_patch.py --check \
  --vllm-package-root /path/to/prefill/venv/lib/python3.13/site-packages/vllm \
  --vllm-package-root /path/to/decode/venv/lib/python3.13/site-packages/vllm

python tools/apply_vllm_nixl_lease_patch.py --apply \
  --vllm-package-root /path/to/prefill/venv/lib/python3.13/site-packages/vllm \
  --vllm-package-root /path/to/decode/venv/lib/python3.13/site-packages/vllm
```

The check reports `original` or `applied`; rerun it after applying and require
`applied` for every root. Applying twice is safe. Unsupported or partially
patched packages fail before source writes; rebuild a partially patched image.
The installer stages and hashes all resulting files, then prepares every
replacement and same-filesystem rollback hardlink before replacing any source.
It preserves existing UID, GID, and permission bits; new modules inherit their
parent directory's UID/GID and are readable with mode `0644`. Failure to preserve
ownership aborts before publication. Package filesystems must support hardlinks
and atomic renames. The installer never rewrites the contents of linked uv cache files.

Run it offline: individual replacements are atomic, but the complete multi-file
installation is not atomic against running readers or process kills. Ordinary
write failures restore original inodes by renaming backups and remove newly
added modules; restoration does not allocate another copy of their contents.
If restoration itself fails, the tool attempts all remaining restores, reports
the failure, and retains staging directories for recovery. Rebuild the image
or restore those backups before starting workers. A killed installer also
requires rebuilding or rechecking the image. All P/D processes must restart with the same patched
package; mixed wire versions are unsupported. Rebuilding frozen environments
can discard the fix and requires applying and checking it again.

After the patch, lease verification is enabled by default through
`kv_connector_extra_config.kv_lease_verification`. An invalid handoff fails
safely under the default failure policy. To recover by locally recomputing the
prompt KV on decode, set the decode engine's
`kv_transfer_config.kv_load_failure_policy` to `"recompute"`. The policy alone
does not add lease verification to an unpatched package. Increasing the lease
can reduce expiry frequency but does not replace validity checking. If NIXL
cannot release a lease-check transfer handle, its landing cells remain
quarantined. Requests wait while active reads or prepared batches can return
capacity; when none can, an unsatisfiable queued request fails through the
existing fail/recompute policy instead of waiting indefinitely.

Controlled two-node, TP=2 GPU qualification covered a normal handoff, expiry
without cache pressure, expiry with block reuse, a longer-lease control, and
expiry followed by decode recomputation. The patched path rejected stale leases
before model execution; the recompute case produced finite output. This does
not establish full SWE training correctness, scaling performance, or correctness
for every NIXL transfer topology. The linked PR's W&B runs cover the pre-review
backport. Subsequent installer transaction/ownership hardening and the permanent
landing-cell exhaustion path have CPU regression coverage; those changes have
not been rerun on GPUs.
