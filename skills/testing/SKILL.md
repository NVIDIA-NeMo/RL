---
name: testing
description: Testing conventions for NeMo-RL. Covers what not to unit-test, Ray actor coverage pragmas, nightly test requirements, and recipe naming rules.
when_to_use: Writing or reviewing tests; adding a nightly test; naming a recipe; 'what not to unit-test', 'coverage pragma', 'Ray actor test', 'nightly test requirement', 'how to name recipes', during code review of test files.
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

## What Not to Unit-Test

- **Do not test recipe or test-script settings: a unit test that only asserts a key of an existing config, or an argument of an existing functional or nightly test script, has a given value. Do not unit-test functional test scripts at all; running them is their test.** A recipe or script change always has a reason, so such a test only fails CI for the next intentional change. A test that keeps every config away from a known issue is fine, for example `test_all_config_no_tp_size_accuracy_issues`, which stops a new config from hitting the known TP>=4 accuracy issue. (#5009)
- **Do not add a unit test that only checks a config under `examples/configs/` loads or validates.** `test_shipped_config` in `tests/unit/test_config_validation.py` already loads every one with inheritance, validates it against its algorithm's `MasterConfig`, and runs `validate_single_controller_config` on every SingleController config. If a new kind of config needs a check, extend that test. (#5009)
- **Do not add a unit test that only asserts a config field's default value.** A default change is a behaviour change for every run that omits the key; its reason belongs in the PR, not in a test that fails on it. Test what the code does with the value instead, for example a validator rejecting a combination or a property derived from it. (#5009)
- **Do not add a unit test that only checks what pydantic already does for a config: omitted fields falling back to their defaults, a passed value reading back unchanged, or a missing required field or a value outside its `Literal`, type or `Field` constraint (`ge`, `gt`, `allow_inf_nan`, `PositiveInt`) being rejected.** That is pydantic's job, and the central config tests already load every shipped config. Test it only when our own code sits between the input and the model, such as a custom merge or a `mode="before"` validator. Next to a validator of ours, a test that an edge value is accepted is fine; constructing it without an error is the check, so do not read the value back. (#5009)

## Nightly Tests for New Model Support

When adding support for a new model, add a corresponding nightly test consisting of:

### 1. Recipe YAML under `examples/configs/recipes/`

Place in the appropriate domain subdirectory (`examples/configs/recipes/llm/` or `examples/configs/recipes/vlm/`). Name it following the recipe naming rules below.

### 2. Driver script under `tests/test_suites/`

Create a shell script in the matching domain (`tests/test_suites/llm/` or `tests/test_suites/vlm/`). Source any common environment (e.g., `common.env`) and invoke the training entrypoint with `uv run ... --config <path-to-yaml>`. Match the driver script filename to the YAML base name with `.sh`.

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
- **strategy-and-params**: `automodel`, `automodel-ep8`, `megatron`, `megatron-ep16tp2cp2`
- **modifiers** (optional): `sp`, `actckpt`, `fp8`, `noncolocated`, `quick`
- **-long** (optional): long-running recipe
- **.vN** (optional): version suffix for convergence-impacting changes

Examples:
```
sft-llama3.1-8b-1n8g-automodel-long.yaml
grpo-llama3.1-8b-instruct-1n8g-megatron-fp8.yaml
grpo-qwen2.5-7b-instruct-4n8g-automodel-tp4sp.v3.yaml
```

### VLM Pattern

```
vlm_<algo>-<model>-<nodes>n<gpus>g-<strategy>[-modifiers][.vN].(yaml|sh)
```

### Directory Placement

```
examples/configs/recipes/
  llm/<name>.yaml
  vlm/<name>.yaml

tests/test_suites/
  llm/<name>.sh
  vlm/<name>.sh
  nightly.txt
```


## Research projects

Research unit and functional tests are discovered under `research/*/tests/`.
CI syncs each project's runtime dependencies and `test` group before executing them.
Research suite scripts belong under `research/<project>/tests/test_suites/`,
with matching recipes under `research/<project>/configs/recipes/`.
List scripts by their repository-relative paths in the root suite lists
(`nightly.txt`, `disabled.txt`, etc.). Script/recipe accounting includes both
core and research directories, preserving each research project's identity.
