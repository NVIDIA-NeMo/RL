# Dataclass Config Defaults — Changelog

Tracks all changes for [#2102](https://github.com/NVIDIA-NeMo/RL/issues/2102).
Branch: `feat/dataclass-config-defaults`

---

## Phase 1: Infrastructure + GRPO POC

### Session 1 — Initial Implementation (2026-04-14)

#### 1. `nemo_rl/utils/config.py` — Core utility function

| # | Line | Change |
|---|------|--------|
| 1 | 15-17 | Added imports: `dataclasses`, `Any`, `get_type_hints` |
| 2 | 192-234 | Added `apply_config_defaults(config, defaults_cls)` — recursively fills missing dict keys from dataclass field defaults. Handles nested dataclass fields by recursion. Fields without defaults (`MISSING`) are skipped. |

#### 2. `nemo_rl/algorithms/grpo.py` — Defaults dataclasses

| # | Line | Change |
|---|------|--------|
| 1 | 14 | Added import: `dataclasses` |
| 2 | 21 | Added imports: `dataclass`, `field` from `dataclasses` |
| 3 | 222-231 | Added `RewardScalingConfigDefaults` — 5 fields (enabled=False, source/target min/max) |
| 4 | 233-237 | Added `RewardShapingConfigDefaults` — 1 field (enabled=False) |
| 5 | 240-246 | Added `AsyncGRPOConfigDefaults` — 3 fields (enabled=False, in_flight/recompute=False) |
| 6 | 249-278 | Added `GRPOConfigDefaults` — 12 scalar fields + 3 nested defaults (reward_scaling, reward_shaping, async_grpo) |
| 7 | 281-290 | Added `ClippedPGLossConfigDefaults` — 6 NotRequired fields |
| 8 | 293-300 | Added `GRPOMasterConfigDefaults` — top-level, nests GRPOConfigDefaults + ClippedPGLossConfigDefaults |

#### 3. `examples/run_grpo.py` — Injection point

| # | Line | Change |
|---|------|--------|
| 1 | 21 | Import added: `GRPOMasterConfigDefaults` from `nemo_rl.algorithms.grpo` |
| 2 | 27 | Import added: `apply_config_defaults` from `nemo_rl.utils.config` |
| 3 | 67 | Inserted `config = apply_config_defaults(config, GRPOMasterConfigDefaults)` after `OmegaConf.to_container()` |

#### 4. `tests/unit/test_config_defaults.py` — NEW FILE (234 lines)

| Test category | Count | Description |
|---------------|-------|-------------|
| Field-subset tests (parametrized) | 5 | Verify every field in defaults dataclass exists in the corresponding TypedDict |
| YAML consistency tests | 3 | Verify GRPOConfigDefaults, RewardScalingConfigDefaults, ClippedPGLossConfigDefaults match exemplar YAML values |
| `apply_config_defaults` unit tests | 5 | Missing keys filled, existing keys not overwritten, nested recursion, missing nested section creation, non-dataclass no-op |
| Integration test | 1 | GRPOMasterConfigDefaults against a realistic partial config |

#### 5. `.claude/skills/config-conventions/SKILL.md` — Convention update

| # | Section | Change |
|---|---------|--------|
| 1 | Frontmatter | Updated description to reflect dataclass defaults |
| 2 | Core Rule | "YAML is the single source of truth" → "YAML takes precedence; dataclass provides fallback defaults" |
| 3 | Dataclass Defaults | New section: how to write `*ConfigDefaults` dataclasses (Do/Don't examples) |
| 4 | Accessing NotRequired Fields | Fields with dataclass defaults can now be accessed directly via `config["key"]` |
| 5 | Forbidden Patterns | Updated examples to reference dataclass defaults |

---

### Session 2 — Fix Phase 1 Remaining Issues (2026-04-14 ~ 2026-04-15)

#### Issue A: Missing GRPO variant injection points

Three GRPO entry points were missing `apply_config_defaults` calls.

**`examples/run_vlm_grpo.py`**

| # | Line | Change |
|---|------|--------|
| A1 | 21 | Import added: `GRPOMasterConfigDefaults` |
| A2 | 27 | Import added: `apply_config_defaults` |
| A3 | 64 | Inserted `config = apply_config_defaults(config, GRPOMasterConfigDefaults)` |

**`examples/run_grpo_sliding_puzzle.py`**

| # | Line | Change |
|---|------|--------|
| A4 | 26 | Import added: `GRPOMasterConfigDefaults` |
| A5 | 38 | Import added: `apply_config_defaults` |
| A6 | 213 | Inserted `config = apply_config_defaults(config, GRPOMasterConfigDefaults)` |

**`examples/nemo_gym/run_grpo_nemo_gym.py`**

| # | Line | Change |
|---|------|--------|
| A7 | 31 | Import added: `GRPOMasterConfigDefaults` |
| A8 | 52 | Import added: `apply_config_defaults` |
| A9 | 140 | Inserted `config = apply_config_defaults(config, GRPOMasterConfigDefaults)` |

#### Issue B: Design doc vs implementation drift

**`docs/design-docs/dataclass-config-defaults.md`**

| # | Section | Change |
|---|---------|--------|
| B1 | Architecture example (L72-84) | Removed `dataclasses.MISSING` explicit assignment; required fields are now simply omitted (matches actual implementation) |
| B2 | Injection point example (L98) | `MasterConfigDefaults` → `GRPOMasterConfigDefaults` |
| B3 | Core utility code block (L117-165) | Synced to actual implementation: `get_type_hints()`, missing nested section creation, `continue` for MISSING fields |
| B4 | Nested config example (L173) | `MasterConfigDefaults` → `GRPOMasterConfigDefaults` |
| B5 | Migration Plan file list (L271-279) | Added 3 GRPO variant entry points to Phase 1 scope |

#### Issue C: Test verification

Tests verified via standalone script (full pytest requires `ray`/`torch` in CI).

| Test | Result |
|------|--------|
| YAML consistency (GRPOConfig, RewardScaling, ClippedPGLoss) | 3/3 PASS |
| Unit tests (missing keys, existing keys, nested, missing section, non-dataclass) | 5/5 PASS |
| Integration test (GRPOMasterConfigDefaults end-to-end) | 1/1 PASS |

---

## Phase 1 Summary

| File | Status | Lines changed |
|------|--------|---------------|
| `nemo_rl/utils/config.py` | Modified | +48 |
| `nemo_rl/algorithms/grpo.py` | Modified | +92 |
| `examples/run_grpo.py` | Modified | +4 -1 |
| `examples/run_vlm_grpo.py` | Modified | +4 -1 |
| `examples/run_grpo_sliding_puzzle.py` | Modified | +4 -1 |
| `examples/nemo_gym/run_grpo_nemo_gym.py` | Modified | +3 |
| `tests/unit/test_config_defaults.py` | **New** | +234 |
| `docs/design-docs/dataclass-config-defaults.md` | Modified | ~20 lines updated |
| `.claude/skills/config-conventions/SKILL.md` | Modified | +37 -12 |

---

### Session 3 — Pydantic "left join" upgrade (2026-04-20)

Implements @terrykong's ideal "left join" pattern from #2102: validate types via pydantic,
fill defaults, and preserve extra keys for backward compatibility.

#### 1. `nemo_rl/utils/config.py` — Core upgrade

| # | Change |
|---|--------|
| 1 | Added import: `warnings`, `from pydantic import TypeAdapter` |
| 2 | Added `_merge_extras_back(validated, user)` — recursively restores extra keys dropped by `extra='ignore'` |
| 3 | Added `validate_config(user_config, schema)` — the new "left join": TypeAdapter validates + fills defaults + merge extras back |
| 4 | Refactored `apply_config_defaults()` to deprecated wrapper — emits `DeprecationWarning`, delegates to `validate_config` for pydantic dataclasses, legacy fallback for stdlib dataclasses |

#### 2. `nemo_rl/algorithms/grpo.py` — Pydantic dataclass migration

| # | Change |
|---|--------|
| 1 | Changed import: `from dataclasses import dataclass, field` → `from dataclasses import field` + `from pydantic import ConfigDict` + `from pydantic.dataclasses import dataclass` |
| 2 | Added `_EXTRA_IGNORE = ConfigDict(extra="ignore")` shared config |
| 3 | All 7 `*ConfigDefaults` classes: `@dataclass` → `@dataclass(config=_EXTRA_IGNORE)` |

#### 3. `examples/run_grpo.py`, `run_vlm_grpo.py`, `run_grpo_sliding_puzzle.py`, `nemo_gym/run_grpo_nemo_gym.py`

| # | Change |
|---|--------|
| 1 | Import: `apply_config_defaults` → `validate_config` |
| 2 | Call site: `apply_config_defaults(config, ...)` → `validate_config(config, ...)` |

#### 4. `tests/unit/test_config_defaults.py` — Rewritten

| Test category | Count | Description |
|---------------|-------|-------------|
| Field-subset tests | 5 | Unchanged — verify defaults fields ⊆ TypedDict fields |
| YAML consistency tests | 3 | Unchanged — verify defaults match exemplar YAML |
| `validate_config` unit tests | 6 | Missing keys, existing keys, nested recursion, missing section, extra keys preserved, nested extra keys preserved |
| Type validation tests (NEW) | 4 | Wrong type → ValidationError, nested type error, coercion (str→int), nullable fields |
| Backward compat tests (NEW) | 2 | `apply_config_defaults` emits DeprecationWarning, produces same result |
| Integration tests | 2 | GRPOMasterConfigDefaults realistic config, exemplar YAML round-trip |

#### 5. Design documentation

| File | Changes |
|------|---------|
| `dataclass-config-defaults.md` | Updated principle, architecture examples, core utility code block, FAQ to reflect pydantic validation |
| `dataclass-config-defaults-changelog.md` | Added Session 3 entry |
