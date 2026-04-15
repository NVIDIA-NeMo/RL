---
name: config-conventions
description: Configuration conventions for NeMo-RL. YAML takes precedence; dataclass provides fallback defaults. Covers TypedDict usage, dataclass defaults, exemplar YAML updates, and forbidden default patterns. Auto-invoked during code review.
---

# Configuration Conventions

## Core Rule

**YAML values always take precedence; dataclass provides fallback defaults for missing keys.** Required fields must still come from YAML. Optional fields may define defaults in a companion `*ConfigDefaults` dataclass (see [design doc](docs/design-docs/dataclass-config-defaults.md)).

## Access Config Directly

For required attributes, write code like `policy_cfg["precision"]` and assume it is present. Do not introduce hidden defaults deep in the code.

## Express Optionality via TypedDict

Use `typing.NotRequired` to mark optional attributes. Optional attributes may be absent/`None`; code may check for their presence.

## Where Defaults Live

- Exemplar configs under `examples/configs/*.yaml` include documented defaults.
- Recipe YAMLs under `examples/configs/recipes/**/*.yaml` are runnable snapshots and may omit documentation.

## Documenting New Config Keys

When adding a new config key to a `TypedDict` subclass, document:
- The key's purpose
- Valid values/types
- Recommended default (if applicable)

Reflect the default in the exemplar YAMLs under `examples/configs/*.yaml`.

## Recipe YAMLs Must Set `defaults`

Recipe YAMLs under `examples/configs/recipes/**/*.yaml` must set `defaults: <exemplar>.yaml` to inherit from one of the exemplar configs in `examples/configs/*.yaml`. This keeps recipes minimal — they only override what differs from the exemplar.

If a recipe YAML does not have a `defaults` key, run:

```bash
uv run ./tools/config_cli.py minimize <recipe.yaml>
```

This will minimize the config and assign the appropriate `defaults` key.

## Dataclass Defaults

For fields with sensible default values, define a companion `*ConfigDefaults` dataclass next to the `TypedDict`. The dataclass is never instantiated at runtime — it only feeds `apply_config_defaults()`.

**Do:**
```python
@dataclass
class GRPOConfigDefaults:
    overlong_filtering: bool = False          # NotRequired field with default
    calculate_advantages_on_gpu: bool = False  # NotRequired field with default
```

**Don't:**
- Include required fields that have no sensible default (e.g. `num_prompts_per_step`).
- Set defaults that disagree with the exemplar YAML — the consistency test will catch this.
- Set non-`None` defaults in `.get()` calls — use a dataclass default instead.

## Accessing NotRequired Fields

When a `NotRequired` field has a dataclass default, access it directly with `config["key"]` — `apply_config_defaults()` guarantees the key exists. For fields without a dataclass default, use an `in` check or `.get(key)` / `.get(key, None)`.

**Do:**
```python
# Field has a dataclass default — safe to access directly
if master_config["grpo"]["calculate_advantages_on_gpu"]:
    ...

# Field without a dataclass default — check presence
stop_properly_penalty_coef = cfg.get("stop_properly_penalty_coef", None)

# Nested NotRequired: check presence at each level explicitly
if "megatron_cfg" in policy_config and policy_config["megatron_cfg"]["enabled"]:
    ...
```

**Don't:**
```python
# Hidden boolean default — use a dataclass default instead
disable_ppo_ratio = cfg.get("disable_ppo_ratio", False)

# Hidden non-trivial default — caller has no idea True is the fallback
normalize_rewards = grpo_config.get("normalize_rewards", True)

# Chained .get() with hidden defaults at each level
megatron_enable = config.get("megatron_cfg", {}).get("enabled", False)
```

If a `NotRequired` field is absent and has no dataclass default, the code should handle that explicitly — not paper over it with a magic default.

## Forbidden Patterns

**Don't:**
```python
# Hidden default in code — put it in a *ConfigDefaults dataclass instead
precision = policy_cfg.get("precision", "bfloat16")

# Function parameter defaulting a config value
def build_policy(policy_cfg, precision: str = "bfloat16"):
    ...
```

**Do:**
```python
# Required attribute: expect it from YAML or user override
precision: str = policy_cfg["precision"]

# Optional attribute with dataclass default: access directly
if master_config["grpo"]["overlong_filtering"]:
    ...

# Optional attribute without dataclass default: check for presence
if "milestones" in scheduler_cfg:
    configure_milestones(scheduler_cfg["milestones"])
```

See also: [Dataclass Config Defaults](docs/design-docs/dataclass-config-defaults.md), [TypedDict and Configuration Defaults](docs/design-docs/design-and-philosophy.md#typeddict-and-configuration-defaults).
