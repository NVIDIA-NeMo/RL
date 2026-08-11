# Test Suites

Driver scripts for the GPU suites. Each one pairs with a recipe YAML at the
mirrored path under `examples/configs/recipes/<family>/`, same basename —
`common.env` derives the config path from the driver's own path, so the pairing
is structural rather than a convention.

Each driver declares which suite it belongs to in its CONFIG block
(`SUITE`, `TAGS`, `SKU`). To list a suite:

```sh
tools/list-suites nightly
tools/list-suites --gpu-hours release    # per-test and total GPU hours
tools/list-suites --suites               # every suite name it understands
```

The `<suite>.txt` manifests are what nemo-ci reads today to build its dynamic
pipeline; a unit test keeps them in agreement with the declarations. They are
removed once nemo-ci reads the declarations directly.

**The conventions live in the `testing` skill**, not here:

- [`.agents/contributor-skills/testing/SKILL.md`](../../.agents/contributor-skills/testing/SKILL.md)
  — what earns a place in a suite, GPU-hour bands, suite declarations, recipe naming
- [`.agents/contributor-skills/testing/adding-a-test.md`](../../.agents/contributor-skills/testing/adding-a-test.md)
  — the mechanics: files to create, the CONFIG block, GB200 mirrors, running and
  launching with `tools/launch`
