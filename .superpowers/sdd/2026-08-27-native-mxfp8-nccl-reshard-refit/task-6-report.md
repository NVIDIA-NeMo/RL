# Task 6 Report: vLLM Refit Lifecycle Adapter

## Changed Files

- `nemo_rl/models/generation/vllm/refit_adapter.py`
  - Defines the version-neutral `VllmRefitAdapter` protocol, capability record,
    API-only capability probe, and the pinned `Vllm0251RefitAdapter`.
  - Owns `prepare -> begin_update -> load_component -> finish_update` and the
    fail-closed abort state. A successful cycle returns to `prepared`; an abort
    permanently marks the adapter unusable and chains the original error.
  - Lazily enters `set_current_vllm_config`, calls
    `initialize_layerwise_reload`, invokes each received component target's
    current wrapped `weight_loader`, requires all planned components before
    finalization, then calls `finalize_layerwise_reload` and exits the same
    config context.
  - Does not bind parameter names, MXFP8 scale storage, merged regions, or
    `LocalParamSpec`; those are Task 7 responsibilities.
- `tests/unit/models/generation/test_vllm_refit_adapter.py`
  - Uses fake vLLM modules to cover lifecycle ordering, wrapped-loader
    accounting, incomplete-update rejection, loader failure poisoning,
    repeated complete updates, missing layerwise APIs, and the later-engine
    capability probe without selecting an unimplemented runtime path.
- `pyrefly.toml`
  - Adds the new production module to the explicit type-check allow list.

## Source Evidence

The repository pins vLLM 0.25.1 in `pyproject.toml` and `uv.lock`. The local
adaptive source checkout at `vllm-v0251-adaptive` contains the immutable
v0.25.1 source commit `752a3a504485790a2e8491cacbb35c137339ad34` in its
history. Its layerwise implementation restores checkpoint-format storage,
wraps parameter `weight_loader` calls, accounts loaded elements, processes the
layer, and copies processed values back into the saved kernel storage.

The adapter deliberately supports that lifecycle only. Later engine registry
and trainer APIs are source/API-probed for diagnostics; no version string is
read and no later vLLM runtime path is selected.

## RED Evidence

The tests were written before `refit_adapter.py`. The required repository
command could not reach collection on this macOS host because the checked-in
lockfile only supports Linux:

```text
uv run pytest -q tests/unit/models/generation/test_vllm_refit_adapter.py
error: The current Python platform is not compatible with the lockfile's supported environments: `platform_machine == 'x86_64' and sys_platform == 'linux'`, `platform_machine == 'aarch64' and sys_platform == 'linux'`
```

The host interpreter also lacks the repository's test dependencies. Therefore
there is no claimed native Linux RED collection result.

## GREEN Evidence

An isolated macOS Python 3.13 environment with only the dependencies needed
to import the repository package and execute the fake tests reported:

```text
6 passed in 2.79s
```

The successful command was:

```text
PYTHONPATH="$PWD" uv run --isolated --no-project --with pytest --with torch --with 'transformers==5.8.1' --with ray --with pillow --with pydantic --with pyzmq --with nvidia-ml-py --with uvicorn --with fastapi pytest -q --confcutdir=tests/unit/models/generation tests/unit/models/generation/test_vllm_refit_adapter.py
```

Final static checks:

```text
ruff check nemo_rl/models/generation/vllm/refit_adapter.py tests/unit/models/generation/test_vllm_refit_adapter.py
ruff format --check nemo_rl/models/generation/vllm/refit_adapter.py tests/unit/models/generation/test_vllm_refit_adapter.py
python3 -m py_compile nemo_rl/models/generation/vllm/refit_adapter.py tests/unit/models/generation/test_vllm_refit_adapter.py
git diff --check
```

All succeeded. The changed production file is registered in `pyrefly.toml`.
Because the local checked-in `.venv` is Linux-only and has no Torch on macOS,
the focused local Pyrefly verification used `--replace-imports-with-any
'torch.*'` and reported `INFO 0 errors`; this is not a claim that the complete
Linux environment was type checked locally.

## Remaining Concerns

- Native vLLM 0.25.1 import, reload, CUDA Graph storage preservation, and
  complete repository pytest still require Linux GPU/container verification.
- Task 7 must construct destination specs after `begin_update` and call
  `load_component` from their post-receive hook. Direct copies to runtime
  storage would bypass vLLM layerwise accounting and are intentionally not
  implemented here.
- Existing BF16 and blockwise paths are unchanged: this task adds a standalone
  adapter and does not wire it into the backend.

## Review Fixes

The later-vLLM diagnostic probe now matches the v0.28.0 source contract:
`TrainerWeightTransferEngine` in
`vllm.distributed.weight_transfer.base`, with
`trainer_init(init_info, *, client, source=None)` and `send_weights`. The fake
contract verifies both the symbol and that `client` is keyword-only. This is
still a source/API diagnostic: adapter selection remains capability-based and
the only supported runtime lifecycle remains pinned vLLM 0.25.1.

Failure-state tests were added before the adapter fix. The initial RED result
showed a finalizer failure was passed to the held config context as `None`
instead of the original `RuntimeError("finalizer failed")`. The adapter now
passes that active exception through `__exit__`, clears the held context before
calling it, and poisons the worker on either finalizer or context-exit failure.
The tests verify one finalizer call, one context exit, no later lifecycle
operation can initiate another finalization, and every later `prepare`,
`begin_update`, `load_component`, and `finish_update` call raises the stable
unusable-worker error chained from the original failure.

Focused fake lifecycle verification after the review fixes:

```text
8 passed in 3.32s
```

Ruff, `py_compile`, and `git diff --check` succeeded. The configured
source-only local Pyrefly invocation with `torch.*` replaced by `Any` reported
`INFO 0 errors`. Pyrefly cannot type check the fake test module in this macOS
checkout because the repository `.venv` is Linux-only, so its local site
packages do not include pytest; the fake `SimpleNamespace` runner also
intentionally does not statically satisfy the private runner protocol. Native
Linux vLLM 0.25.1 runtime verification remains outstanding.
