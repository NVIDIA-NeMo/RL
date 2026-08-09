---
name: testing
description: Testing conventions for NeMo-RL. Covers Ray actor coverage pragmas, nightly test requirements, and recipe naming rules.
when_to_use: Writing or reviewing tests; adding a nightly test; naming a recipe; 'coverage pragma', 'Ray actor test', 'nightly test requirement', 'how to name recipes', during code review of test files.
---

# Testing Conventions

## Coverage and Ray Actors

For any source file under `nemo_rl/*.py` that defines a class or function decorated with `@ray.remote`, add a coverage pragma because these run in separate Ray processes and are not reliably tracked by coverage.

Place `# pragma: no cover` on the `class` or `def` line:

```python
import ray

@ray.remote  # pragma: no cover
class RolloutActor:
    def run(self) -> None:
        ...

@ray.remote  # pragma: no cover
def remote_eval(batch):
    ...
```

## Nightly Tests for New Model Support

When adding support for a new model, add a corresponding nightly test consisting of:

### 1. Recipe YAML under `examples/configs/recipes/`

Place it in the model family's directory, `examples/configs/recipes/<family>/` (e.g. `qwen3/`, `llama3.1/`). The family is the model generation, taken from the resolved `policy.model_name`. Performance recipes go in `<family>/performance/`. Name it following the recipe naming rules below.

### 2. Driver script under `tests/test_suites/`

Create a shell script at the mirrored path, `tests/test_suites/<family>/`. Source the shared environment with `source "${SCRIPT_DIR%%/tests/test_suites/*}/tests/test_suites/common.env"` and invoke the training entrypoint with `uv run ... --config $CONFIG_PATH`. Match the driver script filename to the YAML base name with `.sh`.

### 3. Add to nightly list

Append the driver script path (relative to `tests/test_suites/`) to @tests/test_suites/nightly.txt.

## Recipe Naming Rules

### LLM Pattern

```
<algo>-<model>-<nodes>n<gpus>g-<strategy-and-params>[-modifiers][-long][.vN].(yaml|sh)
```

- **algo**: `sft`, `dpo`, `grpo`, etc.
- **model**: `llama3.1-8b-instruct`, `qwen2.5-7b-instruct`, etc.
- **nodes/gpus**: `1n8g`, `4n8g`, `8n8g`
- **strategy-and-params**: `fsdp2tp1`, `tp4pp2`, `megatron`, `dtensor2tp1`
- **modifiers** (optional): `sp`, `actckpt`, `fp8`, `noncolocated`, `quick`
- **-long** (optional): long-running recipe
- **.vN** (optional): version suffix for convergence-impacting changes

Examples:
```
sft-llama3.1-8b-1n8g-fsdp2tp1-long.yaml
grpo-llama3.1-8b-instruct-1n8g-megatron-fp8.yaml
grpo-qwen2.5-7b-instruct-4n8g-fsdp2tp4sp.v3.yaml
```

### VLM Pattern

```
vlm_<algo>-<model>-<nodes>n<gpus>g-<strategy>[-modifiers][.vN].(yaml|sh)
```

### Directory Placement

```
examples/configs/recipes/
  <family>/<name>.yaml
  <family>/performance/<name>.yaml
  reproduce/<name>.yaml

tests/test_suites/
  common.env
  <family>/<name>.sh
  <family>/performance/<name>.sh
  nightly.txt
```
