# Repo conventions: config

What to check when a PR adds, reads or overrides config, and how tests build config models.

- **Check new and changed config against the `config-conventions` skill (`skills/config-conventions/SKILL.md`).** It covers where defaults live, BaseModel vs TypedDict vs dataclass, exemplar YAML updates and the forbidden default patterns; cite the rule it breaks rather than restating it.
- **In unit tests, build BaseModel configs with keyword arguments (`Foo(k=v)`, or `Foo.model_construct(k=v)` when required fields are missing), not `model_validate({...})` or `**{...}`, unless validating a loaded recipe YAML.** The point is to unify how tests build configs; ask "unify to keyword arguments to keep consistency" (#4278).
- **An "add it to the exemplar YAML" ask must also name the `tests/unit/reference_configs/` twin, and one exemplar is enough.** (#3798)
- **Never suggest `.get(key, default)` for a required key or invariant; a KeyError there is the feature, at most ask for a clearer message.** (#3308, #3652) Details: [typeddict-missing-key-failfast-is-convention](details/typeddict-missing-key-failfast-is-convention.md).
- **Code that rewrites or drops a value the user set in config silently ignores that setting; ask to remove the override and reject the combination at setup instead.** Put the check in the existing validator beside the nearest sibling check, and look for a second bypass the override forced elsewhere. On #4615 an OPD mode overrode the user's top-k/top-p while the config still said filtering was on. Details: [reject-dont-override-user-config](details/reject-dont-override-user-config.md).
- **Check recipe-name `tpN/cpN/epN` tokens against the resolved `defaults:` chain's training config.** (#4701)
