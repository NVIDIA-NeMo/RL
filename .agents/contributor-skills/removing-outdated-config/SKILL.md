---
name: removing-outdated-config
description: How to remove or rename a config key in NeMo-RL so stale configs fail at startup with the migration to apply, instead of failing deep in a run or being silently ignored.
when_to_use: Removing a config key, renaming one, or changing the shape a key accepts; 'delete this config option', 'this key is no longer used', 'migrate the config', during review of a PR that drops a YAML key.
---

# Removing a Config Key

Deleting a key from the schema and the shipped YAMLs is only half the job. Users have
their own configs. If nothing rejects the old key they get one of two bad outcomes:

- **Silently ignored.** The run starts, the setting does nothing, and the user believes
  it took effect. This is the worse one — a `_v2: false` config would have run on a
  backend the user did not ask for.
- **Failing late.** The error surfaces wherever the key happened to be read — a
  checkpoint-save path fires at the first save interval, hours in.

So every removal ships with a rejection that fires at startup.

## Scope: removed, not deprecated

This module is only for config the code no longer accepts. A key that still works but is
discouraged does not belong here — every function in it raises, so putting a soft
deprecation here breaks runs that are still valid. Warn from the consumer instead, and
move the key here once it is actually removed.

## The three steps

### 1. Add the check to `nemo_rl/utils/outdated_config_checks.py`

One `reject_outdated_*` function per removal, called from `check_outdated_config`:

```python
def reject_outdated_<thing>(config: dict[str, Any]) -> None:
    """Fail when <the old shape> is still present.

    Args:
        config: The resolved config, already flattened to plain dicts.
    """
    ...
    raise ValueError(
        "<what is wrong>. <what to do instead>."
    )
```

`check_outdated_config` normalizes the input (one recursive `model_dump`), so your
function receives plain dicts and must not repeat that. If the key lives on a training
backend block, iterate `_train_backend_configs` instead of re-deriving where `policy`,
`value`, `teachers[i]` and `env.reward_model` are.

The message is the whole point. State what is wrong and what to write instead — a user
who only reads the exception should be able to fix their YAML. Do not write "deprecated"
when you mean removed: nothing here warns, everything raises.

### 2. Cover it in `tests/unit/utils/test_outdated_config_checks.py`

Add a section for the new function (the file is grouped by function, with banner
comments). Cover the rejected shape, the accepted shape, and absence of the key.

`tests/unit/test_config_validation.py::test_no_shipped_config_is_outdated` runs
`check_outdated_config` over every shipped config, so step 1 automatically constrains
`examples/` and `research/` too — no second implementation, and no way for the two to
drift.

### 3. Run the tests, then sweep the repo for leftovers

```bash
uv run --group test pytest tests/unit/utils/test_outdated_config_checks.py \
    tests/unit/test_config_validation.py -q
```

`test_no_shipped_config_is_outdated` failing means a config in the repo still carries the
old shape. Fix the YAML — do not weaken the check.

The tests only see parsed config, so finish with a grep for everything they cannot reach:

```bash
grep -rn "<the old key>" --exclude-dir=.git .
```

`docs/` guides, docstrings, `tests/test_suites/**/*.sh` command-line overrides and
README snippets all carry their own copies of config and are what survives a removal.
Every remaining hit is either a reference to update or an example that now raises at
startup.

## Where the check must NOT go

Not in the consumer. A check next to the code that reads the key only fires if that code
path runs, which is how the `metric_name` format check ended up living in six copies
across five algorithms and only firing at the first checkpoint save.

Entrypoints call `check_outdated_config(config)` immediately after the `MasterConfig` is
built. `tests/unit/utils/test_outdated_config_checks.py::test_every_entrypoint_checks_outdated_config`
enforces that a new `run_*.py` cannot skip it.

## Exemptions

`examples/run_eval.py` and `examples/configs/evals/` are exempt: eval configs have a
different structure from training configs. Both exemptions carry a written reason. If a
new entrypoint needs one, add it to `_EXEMPT` with a reason — do not encode the special
case inside the check itself.
