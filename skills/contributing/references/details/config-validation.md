# Config Validation

## User Config Types

Do not hand-write type checks for user configs loaded from YAML. Leave type validation to the existing config schema instead of adding `isinstance`, `type` or type assertions for config sections or values. Do not silently skip a check or substitute a default when a type is wrong.

**Don't:**
```python
if not isinstance(config["batch_size"], int):
    raise TypeError("batch_size must be an integer")
```

This rule concerns user configs loaded from YAML, not function inputs in general.
