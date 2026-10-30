# Config

How to add, read and override config, and how tests build config models.

- **Follow the config conventions for new and changed config.** They cover where defaults live, BaseModel vs TypedDict vs dataclass, exemplar YAML updates and the forbidden default patterns. Contributing Details: [config-conventions](details/config-conventions.md).
- **In unit tests, build BaseModel configs with keyword arguments (`Foo(k=v)`, or `Foo.model_construct(k=v)` when required fields are missing), not `model_validate({...})` or `**{...}`, unless validating a loaded recipe YAML.** The point is to unify how tests build configs (#4278). Review Details: [unify-test-configs-to-keyword-arguments](../../review-pr-team/references/details/unify-test-configs-to-keyword-arguments.md).
- **When you add a key, add it to every base config YAML under `examples/configs/` (the ones without `defaults:`) that carries its block, and update each one's `tests/unit/reference_configs/` twin.** (#3798, #4820)
- **Do not use `.get(key, default)` for a required key or invariant unless there is a strong reason; a KeyError there is the feature.** (#3308, #3652) Review Details: [typeddict-missing-key-failfast-is-convention](../../review-pr-team/references/details/typeddict-missing-key-failfast-is-convention.md).
- **Do not rewrite or drop a value the user set in config; that silently ignores the setting. Reject the combination at setup instead.** Put the check in the existing validator beside the nearest sibling check, and remove any second bypass the override would force elsewhere. (#4615) Review Details: [reject-dont-override-user-config](../../review-pr-team/references/details/reject-dont-override-user-config.md).
- **Keep recipe-name `tpN/cpN/epN` tokens consistent with the resolved `defaults:` chain's training config.** (#4701)
