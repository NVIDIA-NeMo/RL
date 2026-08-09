# Nemotron 3 Ultra

There are no recipes in this directory. Every Nemotron 3 Ultra run is a hero
run, and those configs live with their launcher:

- **Configs:** `examples/nemo_gym/nemotron-3-ultra/`
- **Launcher:** `examples/nemo_gym/nemotron-3-ultra/ultra_launch.sh`
- **Guide:** [docs/guides/nemotron-3-ultra.md](../../../../docs/guides/nemotron-3-ultra.md)

Start from the guide — the pipeline is multi-stage (student RLVR, a panel of
specialised teachers, then multi-teacher on-policy distillation), and each stage
is the same launcher with a different per-stage YAML.

## Why they are not here

Recipes in this tree are small deltas layered on an exemplar config, run by
`tools/launch`, which allocates one homogeneous pool of nodes described by the
`NUM_NODES` / `GPUS_PER_NODE` block in the driver script.

The Ultra configs do not fit any part of that. They are standalone rather than
inheriting from an exemplar, and `ultra_launch.sh` allocates nodes in distinct
roles — training, generation, Gym environments and teacher serving — which a
single node count cannot describe. They are also far too expensive to run in
any CI suite, so they have no driver script and appear in no test manifest.

Adding an Ultra config here would therefore break the invariant that every
recipe in this tree has a matching driver under `tests/test_suites/`.

## What this means for changes

These configs are not covered by `test_all_config_files_have_required_keys`,
which only globs `examples/configs/`. A change to `MasterConfig` will not flag
them. If you change config schema, check them by hand until that coverage gap
is closed.
