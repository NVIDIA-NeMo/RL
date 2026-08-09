# Nemotron 3 Super

The recipes in this directory are the CI-tested ones: short functional and
convergence runs that `tools/launch` can start from a driver script under
`tests/test_suites/nemotron3-super/`.

**The full post-training pipeline is not here.** It lives with its launcher:

- **Configs:** `examples/nemo_gym/nemotron-3-super/` (`stage1_rlvr`, `stage2_swe1`,
  `stage2_swe2`, `stage3_rlhf`, plus lower-DP H100 variants under `small_scale/`)
- **Launcher:** `examples/nemo_gym/nemotron-3-super/super_launch.sh`
- **Guide:** [docs/guides/nemotron-3-super.md](../../../../docs/guides/nemotron-3-super.md)

The three `grpo-qwen3-1.7b-*-megatron-super-*` recipes here are the small
pipeline sanity checks. They inherit directly from the stage configs above, so
they exercise the real stage settings at a size CI can afford, on a Qwen3-1.7B
policy rather than the 120B-A12B model.

## Why the pipeline configs are not here

`super_launch.sh` reads `cluster.num_nodes` out of the config and then allows
`SBATCH_NUM_NODES` to allocate *more* nodes than that, because Gym judge servers
need their own. `tools/launch` has one node count and no way to express the
difference. The stage configs are also standalone rather than deltas on an
exemplar, and are far too expensive to run in any CI suite — so they have no
driver script and appear in no test manifest.

## What this means for changes

The stage configs are not covered by `test_all_config_files_have_required_keys`,
which only globs `examples/configs/`. Four of them are validated indirectly,
because the sanity-check recipes here inherit from them; `stage2_swe2.yaml` and
everything under `small_scale/` are not validated at all. If you change config
schema, check those by hand until that coverage gap is closed.
