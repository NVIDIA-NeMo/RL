# Dataclass Config Defaults

Design document for [#2102](https://github.com/NVIDIA-NeMo/RL/issues/2102): Use `dataclasses` instead of `TypedDict` to handle defaults.

## Background

### Current System

NeMo-RL uses `TypedDict` subclasses for configuration type hints and YAML files as the single source of truth for default values. The config loading pipeline is:

```
YAML file
  → OmegaConf.load()                     (DictConfig)
  → load_config_with_inheritance()        (resolve `defaults:` chain)
  → parse_hydra_overrides()               (apply CLI overrides)
  → OmegaConf.to_container(resolve=True)  (plain dict)
  → algorithm entry point                 (dict access: config["key"])
```

TypedDict provides type annotations but **cannot define default values**:

```python
class GRPOConfig(TypedDict):
    seed: int                              # must be in YAML
    normalize_rewards: bool                # must be in YAML
    overlong_filtering: NotRequired[bool]  # optional, but no default
```

All values must exist in the YAML file (or its `defaults:` parent chain). The exemplar configs under `examples/configs/*.yaml` document the recommended defaults, and recipe configs under `examples/configs/recipes/**/*.yaml` inherit from them via `defaults:`.

### Pain Points

1. **New fields require YAML updates.** Adding a field to a TypedDict means updating the exemplar YAML, and potentially many recipe YAMLs that don't use `defaults:` inheritance.

2. **NotRequired fields have no fallback.** When a `NotRequired` field is absent from YAML, the code must use `.get()` with `None` checks. This leads to scattered access patterns and occasional convention violations (hidden defaults in `.get("key", False)`).

3. **Defaults are disconnected from types.** The TypedDict says *what type* a field is; the YAML says *what value* it defaults to. These live in different files and can drift out of sync.

### Related Discussion

- [#1675](https://github.com/NVIDIA-NeMo/RL/issues/1675): Community request to fully replace TypedDict with dataclass. Maintainer (@terrykong) raised concerns about backward compatibility — TypedDict/dict tolerates extra keys from user forks and deprecated YAML fields, while dataclass construction rejects unknown fields.

- [#2102](https://github.com/NVIDIA-NeMo/RL/issues/2102): @terrykong proposed a middle ground — use dataclass to define defaults, but keep the dict-based runtime. This avoids the backward-compatibility issue while gaining centralized defaults.

## Design

### Principle

**Dataclass defines defaults and types; YAML and CLI can override; runtime stays dict.**

```
Priority (highest → lowest):
  CLI overrides  >  YAML values  >  dataclass defaults
```

The dataclass is instantiated by pydantic during validation to check types and fill defaults, then converted back to a plain dict. It is never kept as a runtime config object.

### Architecture

For each TypedDict config class, an **optional** companion pydantic dataclass defines default values and types:

```python
# TypedDict stays — used for static type checking
class GRPOConfig(TypedDict):
    num_prompts_per_step: int
    num_generations_per_prompt: int
    seed: int
    normalize_rewards: bool
    overlong_filtering: NotRequired[bool]
    calculate_advantages_on_gpu: NotRequired[bool]

# Pydantic dataclass added — used for runtime validation + default values
from pydantic import ConfigDict
from pydantic.dataclasses import dataclass

@dataclass(config=ConfigDict(extra='ignore'))
class GRPOConfigDefaults:
    # Required fields are simply omitted — they must come from YAML.
    # Only fields with sensible defaults are listed here.
    seed: int = 42
    normalize_rewards: bool = True
    overlong_filtering: bool = False
    calculate_advantages_on_gpu: bool = False
```

> **Note:** Required fields (e.g. `num_prompts_per_step`) are omitted from the dataclass entirely — they are passed through as extra keys (preserved by `_merge_extras_back`). `extra='ignore'` tells pydantic to silently skip unknown fields during validation rather than rejecting them, and `_merge_extras_back` restores them in the output.

### Injection Point

The merge happens in each algorithm's entry point script (`examples/run_*.py`), immediately after `OmegaConf.to_container()`:

```python
# examples/run_grpo.py

config = load_config(args.config)
if overrides:
    config = parse_hydra_overrides(config, overrides)
config: MasterConfig = OmegaConf.to_container(config, resolve=True)

# NEW: validate types + fill missing keys from dataclass defaults
config = validate_config(config, MasterConfigDefaults)
```

The updated pipeline becomes:

```
YAML file
  → OmegaConf.load()
  → load_config_with_inheritance()
  → parse_hydra_overrides()
  → OmegaConf.to_container(resolve=True)
  → validate_config()                    ← NEW STEP (validate + fill defaults)
  → algorithm entry point
```

### Core Utility: `validate_config` ("left join")

Added to `nemo_rl/utils/config.py`. Implements the "left join" pattern from #2102:

```python
import dataclasses
from typing import Any

from pydantic import TypeAdapter


def _merge_extras_back(validated: dict[str, Any], user: dict[str, Any]) -> dict[str, Any]:
    """Merge extra keys from user (not in the schema) back into validated.

    Pydantic with extra='ignore' drops unknown keys during validation.
    This function restores them for backward compatibility.
    """
    result = validated.copy()
    for k, v in user.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _merge_extras_back(result[k], v)
        elif k not in result:
            result[k] = v
    return result


def validate_config(user_config: dict[str, Any], schema: type) -> dict[str, Any]:
    """Validate user_config against a pydantic dataclass schema and fill defaults.

    1. TypeAdapter validates types of known fields (raises ValidationError on mismatch)
    2. dataclasses.asdict() produces a dict with all defaults filled
    3. _merge_extras_back() restores unknown keys dropped by extra='ignore'

    Priority: CLI overrides > YAML values > dataclass defaults.
    Unknown keys are always preserved.
    """
    ta = TypeAdapter(schema)
    validated = ta.validate_python(user_config)
    validated_dict = dataclasses.asdict(validated)
    return _merge_extras_back(validated_dict, user_config)
```

The previous `apply_config_defaults` is kept as a deprecated alias for backward compatibility.

### Nested Config Handling

MasterConfig contains nested sub-configs. The defaults dataclass mirrors this nesting:

```python
@dataclass(config=ConfigDict(extra='ignore'))
class MasterConfigDefaults:
    """Defaults for MasterConfig."""
    grpo: GRPOConfigDefaults = field(default_factory=GRPOConfigDefaults)
    loss_fn: ClippedPGLossConfigDefaults = field(default_factory=ClippedPGLossConfigDefaults)
    # Sections without a defaults dataclass (e.g. policy, data) are
    # passed through as extra keys and preserved by _merge_extras_back.
```

Pydantic natively handles nested dataclass validation and recursion.

### Enabled/Disabled Variant Pattern

Many configs use a union pattern with `Literal[True]`/`Literal[False]`:

```python
class LoRAConfigDisabled(TypedDict):
    enabled: Literal[False]

class LoRAConfig(TypedDict):
    enabled: Literal[True]
    dim: int
    alpha: int
    ...
```

For these, the defaults dataclass provides only the disabled variant as default:

```python
@dataclass(config=ConfigDict(extra='ignore'))
class LoRACfgDefaults:
    enabled: bool = False
```

When `enabled: true` is set in YAML, the user must provide all required sub-fields. The dataclass default only covers the common case (disabled).

## What Does NOT Change

| Aspect | Status |
|--------|--------|
| Runtime config access (`config["key"]`) | Unchanged — still dict access |
| TypedDict definitions | Kept — still used for type checking |
| Pydantic validation in `test_config_validation.py` | Unchanged — validates against TypedDict |
| Exemplar YAML files | Kept — still serve as human-readable documentation |
| Recipe YAML `defaults:` inheritance | Unchanged — OmegaConf inheritance still works |
| `config_cli.py` minimize/expand | Unchanged — operates on YAML-to-YAML |
| Hydra CLI overrides | Unchanged — applied before defaults merge |

## What Changes

| Aspect | Before | After |
|--------|--------|-------|
| Where defaults are defined | Only in exemplar YAML | Exemplar YAML + pydantic dataclass (dataclass is authoritative for code) |
| New field added to TypedDict | Must update exemplar YAML + all non-inheriting recipes | Add default in dataclass; YAML updates optional |
| NotRequired field access | `.get("key")` returns None if absent | Field auto-filled by dataclass default; direct `["key"]` access safe |
| Type validation | None at config load time | pydantic validates types of known fields at load time |
| Convention doc (`config-conventions`) | "YAML is the single source of truth" | "YAML takes precedence; dataclass provides fallback defaults with type validation" |

## Consistency Tests

Two new tests ensure TypedDict and dataclass stay in sync:

### 1. Field Subset Test

Every field in the defaults dataclass must exist in the corresponding TypedDict:

```python
def test_defaults_fields_subset_of_typeddict():
    dataclass_keys = {f.name for f in dataclasses.fields(GRPOConfigDefaults)}
    typeddict_keys = set(GRPOConfig.__annotations__.keys())
    extra = dataclass_keys - typeddict_keys
    assert not extra, f"Defaults dataclass has fields not in TypedDict: {extra}"
```

### 2. Default Values Match Exemplar YAML Test

Dataclass defaults must match the values in the exemplar YAML (ensures no silent drift):

```python
def test_defaults_match_exemplar_yaml():
    yaml_config = load_config("examples/configs/grpo_math_1B.yaml")
    yaml_dict = OmegaConf.to_container(yaml_config, resolve=True)
    
    for field in dataclasses.fields(GRPOConfigDefaults):
        if field.default is dataclasses.MISSING:
            continue
        yaml_val = yaml_dict["grpo"].get(field.name)
        if yaml_val is not None:
            assert yaml_val == field.default, (
                f"grpo.{field.name}: YAML={yaml_val}, dataclass={field.default}"
            )
```

## Migration Plan

The migration is incremental. Each phase is a standalone PR.

### Phase 1: Infrastructure + GRPO POC

**Goal:** Prove the pattern works end-to-end with the most active algorithm.

Files changed:
- `nemo_rl/utils/config.py` — add `apply_config_defaults()`
- `nemo_rl/algorithms/grpo.py` — add `GRPOConfigDefaults`, `ClippedPGLossConfigDefaults`, `MasterConfigDefaults`
- `examples/run_grpo.py` — inject `apply_config_defaults()` call
- `examples/run_vlm_grpo.py` — inject `apply_config_defaults()` call
- `examples/run_grpo_sliding_puzzle.py` — inject `apply_config_defaults()` call
- `examples/nemo_gym/run_grpo_nemo_gym.py` — inject `apply_config_defaults()` call
- `tests/unit/test_config_defaults.py` — new file with consistency tests
- `docs/design-docs/dataclass-config-defaults.md` — this document
- `.claude/skills/config-conventions/SKILL.md` — update convention wording

Estimated scope: ~200 lines added across 6 files.

### Phase 2: Other Algorithms

**Goal:** Cover SFT, DPO, RM, Distillation.

Files changed:
- `nemo_rl/algorithms/sft.py` — add `SFTConfigDefaults`
- `nemo_rl/algorithms/dpo.py` — add `DPOConfigDefaults`
- `nemo_rl/algorithms/rm.py` — add `RMConfigDefaults`
- `nemo_rl/algorithms/distillation.py` — add `DistillationConfigDefaults`
- `examples/run_sft.py`, `run_dpo.py`, `run_rm.py`, `run_distillation.py` — inject calls

### Phase 3: Deep Configs

**Goal:** Cover PolicyConfig, DataConfig, LoggerConfig, CheckpointingConfig and their sub-configs.

These are shared across algorithms, so their defaults dataclasses are defined alongside the TypedDicts (e.g., in `nemo_rl/models/policy/__init__.py`).

## Example: Before and After

### Before (Current)

Adding a new field `calculate_advantages_on_gpu` to GRPO requires:

1. Add to TypedDict:
   ```python
   class GRPOConfig(TypedDict):
       ...
       calculate_advantages_on_gpu: NotRequired[bool]
   ```

2. Add to exemplar YAML `grpo_math_1B.yaml`:
   ```yaml
   grpo:
     calculate_advantages_on_gpu: false
   ```

3. Access in code with guard:
   ```python
   if master_config["grpo"].get("calculate_advantages_on_gpu"):
       ...
   ```

4. If any recipe YAML doesn't use `defaults:`, manually add the field there too.

### After (With Dataclass Defaults)

1. Add to TypedDict (same):
   ```python
   class GRPOConfig(TypedDict):
       ...
       calculate_advantages_on_gpu: NotRequired[bool]
   ```

2. Add to defaults dataclass (with default value and type validation):
   ```python
   @dataclass(config=ConfigDict(extra='ignore'))
   class GRPOConfigDefaults:
       ...
       calculate_advantages_on_gpu: bool = False
   ```

3. Access in code directly:
   ```python
   if master_config["grpo"]["calculate_advantages_on_gpu"]:
       ...
   ```

4. Exemplar YAML update is optional (recommended for documentation but not required for correctness). Recipe YAMLs never need updating.

## FAQ

### Q: Does this break the "YAML is the single source of truth" principle?

The principle evolves to: **YAML values always take precedence; dataclass provides fallback defaults for missing keys.** The exemplar YAMLs remain the primary documentation for users. The dataclass ensures the code never crashes due to a missing optional field.

### Q: Why not fully replace TypedDict with dataclass?

As discussed in [#1675](https://github.com/NVIDIA-NeMo/RL/issues/1675), TypedDict/dict tolerates extra keys — important for user forks that add custom config fields and for loading older YAML files with deprecated keys. Pydantic dataclasses with `ConfigDict(extra='ignore')` skip unknown fields during validation, and `_merge_extras_back` restores them afterward — achieving the best of both worlds: type validation for known fields, tolerance for unknown fields.

### Q: Do exemplar YAMLs become redundant?

No. Exemplar YAMLs serve as **user-facing documentation** and **runnable examples**. They should continue to document all fields with comments. The dataclass defaults are a **code-level safety net**, not a replacement for documentation.

### Q: What if the dataclass default and exemplar YAML disagree?

The consistency test (`test_defaults_match_exemplar_yaml`) catches this. If they disagree, the test fails and the developer must reconcile them. The YAML value wins at runtime regardless (since it's loaded before defaults are applied), but the test ensures the dataclass accurately reflects the intended defaults.

### Q: How does this interact with `config_cli.py minimize`?

It doesn't. `minimize` compares a recipe YAML against its `defaults:` parent YAML and removes redundant keys. This is purely a YAML-to-YAML operation and is unaffected by dataclass defaults. The dataclass defaults operate at a different layer (code → runtime dict).
