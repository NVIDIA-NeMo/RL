# Nemotron 3 Nano

The recipes in this directory are the CI-tested ones: short functional and
convergence runs that `tools/launch` can start from a driver script under
`tests/test_suites/nemotron3-nano/`.

**The production post-training run is not here.** It lives in the NeMo-Gym
examples:

- **Config:** `examples/nemo_gym/grpo_nanov3.yaml` (32 nodes)
- **Entry point:** `examples/nemo_gym/run_grpo_nemo_gym.py`
- **Guide:** [docs/guides/nemotron-3-nano.md](../../../../docs/guides/nemotron-3-nano.md)

`grpo-nanov3-30BA3B-2n8g-megatron_generation-async-gym.yaml` in this directory
inherits from that config, so it exercises the production settings at two nodes
instead of thirty-two.

## Why the production config is not here

It is standalone rather than a delta on an exemplar, it is launched through the
NeMo-Gym entry point rather than `tools/launch`, and at 32 nodes it is too
expensive for any CI suite — so it has no driver script and appears in no test
manifest.

## What this means for changes

`grpo_nanov3.yaml` is validated indirectly today, because the async-gym recipe
here inherits from it. Other configs under `examples/nemo_gym/` are not covered
by `test_all_config_files_have_required_keys`, which only globs
`examples/configs/`.
