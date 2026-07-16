---
name: build-and-dependency
description: Build and dependency management for NeMo-RL. Covers Docker image building and running, uv usage, venv setup, and adding dependencies.
when_to_use: Setting up a dev environment; building or running the Docker container; adding or removing a dependency; uv errors; 'how do I install', 'ModuleNotFoundError', 'build the image', 'run in Docker', 'uv sync fails'.
---

# Build and Dependency Guide

---

## Docker Images

Build the release image (includes all dependencies and pre-fetched venvs):

```bash
# Build from local source
docker buildx build -f docker/Dockerfile --tag nemo-rl:latest .

# Build from a specific git ref (no local clone needed)
docker buildx build -f docker/Dockerfile \
    --build-arg NRL_GIT_REF=main \
    --tag nemo-rl:latest \
    https://github.com/NVIDIA-NeMo/RL.git
```

Skip optional backends to reduce build time:

```bash
# Skip vLLM and SGLang
docker buildx build -f docker/Dockerfile \
    --build-arg SKIP_VLLM_BUILD=1 \
    --build-arg SKIP_SGLANG_BUILD=1 \
    --tag nemo-rl:latest .
```

See @docs/docker.md for full options.

---

## Always Use uv

**Never use `pip install` directly** — always go through `uv`.

```bash
# Run a script
uv run examples/run_grpo.py

# Run tests
uv run --group test bash tests/run_unit.sh

# Install all deps from lockfile
uv sync --locked
```

Exception: `Dockerfile.ngc_pytorch` is exempt from this rule.

---

## Adding Dependencies

```bash
# Add a runtime dependency
uv add <package>

# Add an optional dependency
uv add --optional --extra <group> <package>

# Regenerate the lockfile after changes
uv lock
```

Commit both `pyproject.toml` and `uv.lock` together:

```bash
git add pyproject.toml uv.lock
git commit -s -m "build: add <package> dependency"
```

---

## Bumping Megatron-Bridge (and Megatron-LM)

`megatron-bridge` is installed directly from the submodule's own `pyproject.toml`
(`[tool.uv.sources]` points at `3rdparty/Megatron-Bridge-workspace/Megatron-Bridge`),
and `megatron-core` at the Megatron-LM checkout nested inside it. Their dependency
lists — including `megatron-core[dev,mlm]` — flow transitively, so there is **no
mirrored dependency list to maintain** in NeMo RL.

The bump procedure is:

```bash
# 1. Check out the new Megatron-Bridge commit (brings its pinned Megatron-LM along)
cd 3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
git fetch origin && git checkout <new-commit>
git submodule update --init --recursive
cd -

# 2. Relock. uv detects the submodule pyproject.toml change automatically and
#    rebuilds megatron-bridge's metadata (~1 min warm cache, ~3 min cold).
uv lock

# 3. Review `git diff uv.lock`, then commit the submodule pointer and uv.lock together.
```

### When `uv lock` errors after a bump

- **`Package X was included as a URL dependency. URL dependencies must be expressed
  as direct requirements or constraints`** — upstream changed or added a git/URL dep
  in Megatron-Bridge's or Megatron-LM's `[tool.uv.sources]` (e.g., Megatron-LM bumping
  its `emerging-optimizers` rev). uv requires such URLs to also appear as a direct
  requirement or constraint of the root project: update the matching pin in
  `constraint-dependencies` (where `emerging-optimizers` and `fast-hadamard-transform`
  live today) / the `mcore` extra / `[tool.uv.sources]` / `override-dependencies` in
  `pyproject.toml` to the same URL/rev.
- **Version conflicts** — resolve via `[tool.uv] override-dependencies` in
  `pyproject.toml`, same as any other conflict.

### Known fragilities

- uv currently does **not** enforce `requires-python` for path dependencies
  (Megatron-Bridge caps `<3.13` while NeMo RL runs 3.13). If a future uv upgrade
  starts enforcing it, `uv lock` will fail loudly: ask upstream to relax the cap,
  or reintroduce a thin proxy package with its own `pyproject.toml`.
- Megatron-Bridge's and Megatron-LM's own `[tool.uv.sources]` are honored for their
  dependencies. Root-level pins (e.g., `nvidia-modelopt` from git) win only because
  they are direct requirements of NeMo RL — don't remove those direct deps without
  re-checking where the transitive resolution lands.

---

## Common Pitfalls

| Problem | Cause | Fix |
|---------|-------|-----|
| `uv sync --locked` fails | Dependency conflict or stale lockfile | Re-run `uv lock` and commit updated lock |
| `ModuleNotFoundError` after pip install | pip installed outside uv-managed venv | Use `uv add` + `uv sync`, never bare `pip install` |
| Docker build fails at vLLM | vLLM build time overhead | Pass `--build-arg SKIP_VLLM_BUILD=1` |
