---
name: config-conventions
description: Configuration conventions for NeMo-RL. YAML is the single source of truth for defaults. Covers BaseModel/TypedDict usage, dataclass for internal classes, exemplar YAML updates, and forbidden default patterns.
when_to_use: Adding or modifying config fields; reviewing config changes; 'where do I set defaults', 'BaseModel pattern', 'TypedDict pattern', 'dataclass', 'exemplar YAML', 'forbidden default patterns', during code review of config files.
---

# Configuration Conventions

## Core Rule

**Defaults must not be scattered at call sites.** A default for a config value must live in exactly one place; readers should never have to grep through algo code to discover what value was actually used. Where that one place is depends on the schema type — see "Where Defaults Live" below — but it is *never* a `cfg.get("k", default)`, a function-parameter default, or a magic constant inside the consumer.

## Class Choice: BaseModel vs. dataclass vs. TypedDict

We are **gradually migrating** the config schema from `typing.TypedDict` (v1) to `pydantic.BaseModel` (v2). Both styles coexist in the codebase today, and the migration is tracked by `tests/unit/test_config_v2.py` against the reference configs in `tests/unit/reference_configs/` (these will be removed when the migration is complete).

Use the right tool for the job. **v2 (the new convention):**

- **`pydantic.BaseModel` — v2, user-facing config (the new default).** Any class the user touches via YAML — currently the top-level `MasterConfig` of each algorithm (`grpo`, `dpo`, `sft`, `rm`, `distillation`, `eval`) and a few shared schemas like `ClippedPGLossConfig` — is a `BaseModel` declared with `extra="allow"` so unknown keys don't break older configs:

  ```python
  from pydantic import BaseModel

  class MasterConfig(BaseModel, extra="allow"):
      policy: PolicyConfig
      grpo: GRPOConfig
      ...
  ```

  Prefer `BaseModel` for **new** user-facing config classes, and when converting an existing `TypedDict` as part of the v1 → v2 migration.

- **`@dataclass` — v2, internal classes (not loaded from YAML).** For purely in-process data containers — worker metadata, datum specs, internal state passed between Python components — use `@dataclass` (e.g. `nemo_rl/distributed/worker_groups.py`, `nemo_rl/data/interfaces.py`, `nemo_rl/data_plane/interfaces.py`). Do **not** use `BaseModel` or `TypedDict` for these — they're not config and shouldn't pretend to be.

**v1 (legacy, being migrated away):**

- **`typing.TypedDict` — v1, legacy / not-yet-migrated user-facing config.** Most nested sub-configs (e.g. `GRPOConfig`, `RewardScalingConfig`, `AsyncGRPOConfig`) are still `TypedDict`. Continue to maintain them with the same defaults rules below until they are migrated to `BaseModel`. Use `typing.NotRequired` to mark optional attributes. **Do not add new `TypedDict`-based config classes.**

When in doubt: *is this class populated from a user-edited YAML?* If yes → `BaseModel` (or legacy `TypedDict`). If no → `@dataclass`.

### New code follows v2 directly

Any **newly added** class must follow the new convention from the start:

- New user-facing config → `pydantic.BaseModel` (with `extra="allow"` when user configs may carry extra/obsolete keys). Do **not** add new `TypedDict`-based config classes.
- New internal class → `@dataclass`.

`TypedDict` edits should only happen on existing, not-yet-migrated classes — either to extend them or as part of converting them to `BaseModel`.

## Access Config Directly

For required attributes, read the value and assume it is present — don't introduce a fallback at the call site.

- **v2 (BaseModel):** attribute access — `master_config.policy.precision`. The BaseModel class itself supplies any default for the field; the call site reads it as-is.
- **v1 (TypedDict, omegaconf-loaded dict):** key access — `policy_cfg["precision"]`. The exemplar YAML supplies the default; the call site reads it as-is.

In both cases, missing required values should fail loudly at load/access time rather than being silently papered over.

## Express Optionality

- **TypedDict:** use `typing.NotRequired[...]` to mark optional attributes.
- **BaseModel:** declare the field as `Optional[...] = None` (or `T | None = None`).

Optional attributes may be absent/`None`; code may check for their presence. Never substitute a non-`None` default at the access site (see "Accessing NotRequired Fields" below).

## Where Defaults Live

The location of defaults depends on the schema type:

- **v2 — `pydantic.BaseModel` (user config):** the default lives **on the BaseModel field** as a Python value. The BaseModel class is the centralized source of truth for user-facing defaults — exemplar YAML serves as documentation / override examples, not as the canonical default store. Example (from `ClippedPGLossConfig`):

  ```python
  class ClippedPGLossConfig(BaseModel, extra="allow"):
      disable_ppo_ratio: bool = False
      ratio_clip_min: float = 0.2
      ratio_clip_c: Optional[float] = None  # None to disable
      reference_policy_kl_penalty: float = 0.01
  ```

- **v1 — `typing.TypedDict` (legacy user config, pre-migration):** the default lives **only in the exemplar YAML** under `examples/configs/*.yaml`. There is no class-level default — Python code must read the value from the loaded dict without supplying a fallback.

- **`@dataclass` (internal class, not loaded from YAML):** usually **no defaults at all**. Fields are populated by the producing code path; a stray `= None` / `field(default=...)` is a smell unless there's a clear reason (e.g., forward-compat for an optional field added mid-migration).

In all three cases:

- Exemplar configs under `examples/configs/*.yaml` include documented defaults. For v1 TypedDict configs they *are* the source of truth; for v2 BaseModel configs they serve as documentation and reasonable starting points, with the BaseModel class itself being authoritative.
- Recipe YAMLs under `examples/configs/recipes/**/*.yaml` are runnable snapshots and may omit documentation.
- Defaults at **call sites** (`cfg.get("k", default)`, function-parameter defaults, magic constants) are never allowed — see "Forbidden Patterns" below.

## Documenting New Config Keys

When adding a new config key to a `BaseModel` or `TypedDict` subclass, document:
- The key's purpose
- Valid values/types
- Recommended default (if applicable)

Reflect the default in the exemplar YAMLs under `examples/configs/*.yaml`. If the change affects an exemplar covered by `tests/unit/test_config_v2.py`, also update the matching `tests/unit/reference_configs/*.yaml` so the v1→v2 migration check stays green.

## Recipe YAMLs Must Set `defaults`

Recipe YAMLs under `examples/configs/recipes/**/*.yaml` must set `defaults: <exemplar>.yaml` to inherit from one of the exemplar configs in `examples/configs/*.yaml`. This keeps recipes minimal — they only override what differs from the exemplar.

If a recipe YAML does not have a `defaults` key, run:

```bash
uv run ./tools/config_cli.py minimize <recipe.yaml>
```

This will minimize the config and assign the appropriate `defaults` key.

## Accessing NotRequired Fields

When accessing a `NotRequired` field, use an `in` check or `.get(key)` / `.get(key, None)`. Never provide a non-`None` default — that hides behavior and defeats the purpose of making the field optional.

**Do:**
```python
# .get() with None (not a hidden default)
stop_properly_penalty_coef = cfg.get("stop_properly_penalty_coef", None)

# Truthiness check for optional booleans
if master_config.grpo.get("skip_reference_policy_logprobs_calculation"):
    ...

# Nested NotRequired: check presence at each level explicitly
if "megatron_cfg" in policy_config and policy_config["megatron_cfg"]["enabled"]:
    ...
```

**Don't:**
```python
# Hidden boolean default — should come from YAML
disable_ppo_ratio = cfg.get("disable_ppo_ratio", False)

# Hidden non-trivial default — caller has no idea True is the fallback
normalize_rewards = grpo_config.get("normalize_rewards", True)

# Chained .get() with hidden defaults at each level
megatron_enable = config.get("megatron_cfg", {}).get("enabled", False)
```

If a `NotRequired` field is absent, the code should handle that explicitly — not paper over it with a magic default.

## Forbidden Patterns

These are forbidden in **both v1 and v2** — they all hide defaults at the call site instead of on the schema.

**Don't (any schema type):**
```python
# Hidden default at the call site — should be centralized on the BaseModel
# field (v2) or in the exemplar YAML (v1)
precision = policy_cfg.get("precision", "bfloat16")

# Function parameter defaulting a config value
def build_policy(policy_cfg, precision: str = "bfloat16"):
    ...
```

**Do — v2 (BaseModel):**
```python
class PolicyConfig(BaseModel, extra="allow"):
    # The BaseModel field is the centralized source of truth for the default
    precision: str = "bfloat16"

# Call site reads the attribute directly
precision = master_config.policy.precision
```

**Do — v1 (TypedDict, still dict-shaped after omegaconf load):**
```python
# Required attribute: expect it from the (exemplar or user) YAML, no fallback
precision: str = policy_cfg["precision"]

# Optional (NotRequired) attribute: check for presence, never invent a default
if "milestones" in scheduler_cfg:
    configure_milestones(scheduler_cfg["milestones"])
```

## Avoid `dict[str, Any]` for Known-Field Config

If a config block has a known set of fields, model it as a `BaseModel` (or, pre-migration, a `TypedDict`) —
do not type it as `dict[str, Any]` and read keys with `.get(k, default)`. A bare `dict[str, Any]` both loses
type-safety and pushes per-key defaults to the call site (the Forbidden Pattern above).

If the block must also pass arbitrary keys through to another system, use `BaseModel(extra="allow")`: the known
fields get types + centralized defaults, and unknown keys still come through (`model_extra`).

**Don't:**
```python
class NonColocatedTeachersConfig(BaseModel, extra="allow"):
    default_teacher_cfg: dict[str, Any] = Field(default_factory=dict)   # known fields hidden behind Any

# ...so defaults end up scattered at the call site:
tp = cfg.get("tensor_model_parallel_size", 1)
precision = cfg.get("precision", "bf16")
```

**Do:**
```python
class TeacherResourceConfig(BaseModel, extra="allow"):  # extra="allow" keeps the passthrough escape hatch
    tensor_model_parallel_size: int = 1
    precision: str = "bf16"
    micro_batch_size: int = 4
    # ...

class NonColocatedTeachersConfig(BaseModel, extra="allow"):
    default_teacher_cfg: TeacherResourceConfig = Field(default_factory=TeacherResourceConfig)
```

See also: @docs/design-docs/design-and-philosophy.md (Configuration Schema: BaseModel, dataclass, and TypedDict section).
