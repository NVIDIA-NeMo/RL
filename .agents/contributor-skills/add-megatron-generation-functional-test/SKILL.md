---
name: add-megatron-generation-functional-test
description: Adds, validates, and submits NeMo-RL functional tests that use the Megatron generation backend with the Megatron training backend. Use when adding Megatron generation coverage for a model, inference feature, topology, refit mode, sampling option, async behavior, or a larger nightly scenario, and when the requested workflow includes running the test on GPUs and opening a pull request to main.
when_to_use: "add a Megatron generation functional test"; "test this mcore generation feature in NeMo-RL"; adding colocated, non-colocated, async, refit, CUDA graph, topology, or sampling coverage with Megatron training; adding a larger-model Megatron generation nightly; implementing, running, and submitting a Megatron generation test PR.
---

# Add a Megatron generation functional test

Create one focused test, run the exact test on the target GPU topology, and open
a pull request to `main` only after it passes.

This workflow always uses:

- `policy.generation.backend=megatron`
- the Megatron policy/training backend (`policy.megatron_cfg`)

Do not silently substitute vLLM, DTensor, or FSDP2.

## Required inputs

Infer these from the request and repository when possible. Ask only for missing
facts that materially change the test:

- feature and expected behavior
- model/checkpoint and tokenizer
- colocated or non-colocated generation
- sync or async execution
- node/GPU topology and parallelism
- expected runtime and whether it belongs in L1 or nightly
- a measurable success condition beyond process exit

Never invent a private checkpoint path or credential. If a model requires a
checkpoint unavailable in CI, ask for the supported location or choose an
existing public model that exercises the requested feature.

## 1. Pull context before designing

Follow the repository's mandatory workflow:

1. Read the requested issue, failure, feature description, or upstream API.
2. Read `.agents/contributor-skills/testing/SKILL.md`.
3. Read the nearest existing test and its suite/recipe files.
4. When the test uses a Megatron-LM or Megatron-Bridge API/config key, verify
   the actual upstream name, allowed values, and semantics. Do not infer them
   from a similarly named NeMo-RL option.

Useful analogs:

- colocated sync:
  `tests/functional/grpo_megatron_generation.sh`
- non-colocated sync:
  `tests/functional/grpo_megatron_generation_non_colocated.sh`
- async Gym:
  `tests/functional/grpo_megatron_generation_async_gym.sh`
- topology/log assertion:
  `tests/functional/grpo_megatron_generation_topology.sh`
- sampling:
  `tests/functional/grpo_megatron_generation_topp_topk.sh`
- larger-model multi-node examples:
  `tests/functional/grpo_megatron_generation_async_prefix_caching.sh` and
  `examples/configs/recipes/llm/*megatron_generation*.yaml`

Copy the nearest analog, then make the smallest changes that isolate the
requested feature.

## 2. Choose L1 or nightly

| Use L1 (`tests/functional`) | Use nightly (`tests/test_suites`) |
|---|---|
| Small public model | Large model or private/shared checkpoint |
| Usually one node and two GPUs in CI | Multi-node or full-node topology |
| About two training steps | Longer convergence or stability run |
| Cheap deterministic feature check | Expensive architecture/performance path |

When uncertain, estimate model memory, node count, and runtime rather than
guessing. A nightly test may still need a small L1 smoke if the feature can
silently fall back and cheap coverage is possible.

### L1 files

Create:

```text
tests/functional/<algo>_megatron_generation_<feature>.sh
```

Register it in the appropriate Megatron suite, normally:

```text
tests/functional/L1_Functional_Tests_Megatron_4.sh
```

L1 suites are discovered by filename; there is no separate manifest. Use
`run_test fast` only for a genuinely cheap, stable test that should run in
`FAST=1` CI. Otherwise use `run_test`.

Before relying on a suite wrapper, verify every helper it calls is defined on
the current branch. If the wrapper is broken by unrelated branch changes, run
the individual new test and report the blocker; do not repair or include
unrelated suite damage in the test PR.

### Nightly files

Create all three parts:

```text
examples/configs/recipes/llm/<name>.yaml
tests/test_suites/llm/<name>.sh
tests/test_suites/<one-manifest>.txt
```

The YAML and shell basenames must match. Add the driver to exactly one primary
manifest:

- `nightly.txt` or `nightly_gb200.txt`
- `release.txt` or `release_gb200.txt`
- `performance.txt` or `performance_gb200.txt`
- `disabled.txt` only when explicitly requested

Do not count `nightly_mcore*.txt` as the primary manifest; those are overlay
subsets. If appropriate, add the test there in addition to its primary
manifest.

For GB200, set `GPUS_PER_NODE=4` and use the GB200 manifest. H100 nightly tests
default to 8 GPUs per node. Keep recipe YAML minimal and inherit from the
nearest recipe when possible.

## 3. Implement a focused test

Keep unrelated dimensions fixed. The test must prove the requested feature was
used, not merely that training exited successfully.

Use one or more of:

- a bounded metric assertion via `tests/check_metrics.py`
- a log assertion for a unique path-engagement marker
- a generated artifact/content assertion
- a comparison against an established invariant

Do not weaken an existing threshold merely to make the run green. If the
feature is nondeterministic, use a statistically meaningful bounded assertion
and explain why it is stable.

Inherit metric assertions from the nearest recipe rather than from an unrelated
one, and check they still apply:

- `token_mult_prob_error` compares generation logprobs against the training
  model's. It is only meaningful when both use the same implementation. Recipes
  that generate with `transformer_impl=inference_optimized` differ numerically
  from the `transformer_engine` training model by design and legitimately show
  values in the hundreds, which is why the nanov3 recipes omit this check while
  the llama ones keep it.
- `max(data["train/reward"]) > 0.0` is a probabilistic assertion. Size
  `num_prompts_per_step * num_generations_per_prompt * max_num_steps` against the
  model's observed pass rate, and keep `train_global_batch_size` equal to the
  per-step sample count. A base model solving under 1% of prompts needs several
  hundred samples before an all-zero run is unlikely; inheriting a small batch
  makes the test flaky rather than strict.

For L1 scripts:

- follow the nearest script's setup, coverage, logging, and `$@` override pattern
- keep `cluster.gpus_per_node=2` when matching the standard two-GPU CI container
- disable W&B and checkpointing unless they are part of the behavior under test
- use `++` only for config keys absent from the base structured config

For every topology, verify that TP × PP × CP and EP constraints match the
allocated generation and training GPUs. Non-colocated tests must account for
separate generation and training resources and validate weight refit.

## 4. Validate locally before spending GPU time

Run the checks relevant to the changed files:

```bash
uv run --group test pytest tests/unit/test_recipes_and_test_suites.py -v
uv run --group dev pre-commit run --all-files
```

For a nightly recipe also run:

```bash
./tools/config_cli.py minimize-check examples/configs/recipes/llm/<name>.yaml
TEST_DRYRUN=1 tests/test_suites/llm/<name>.sh
DRYRUN=1 HF_HOME=... HF_DATASETS_CACHE=... CONTAINER= ACCOUNT= PARTITION= \
  ./tools/launch tests/test_suites/llm/<name>.sh
```

The recipe consistency test enforces YAML/driver/manifest parity, GPU-per-node
placement, config resolution, and the current nightly GPU-hour budget. Treat
its source as authoritative if documentation and the test disagree.

## 5. Run the exact test on GPUs

Read and follow
[run-functional-tests-cog](../run-functional-tests-cog/SKILL.md). Use its
cluster setup, image, token, workspace-overlay, and failure-triage rules, but
target the individual test created here rather than an entire suite.

For a multi-node test, also read
[run-nano35-megatron-inference-cog](../run-nano35-megatron-inference-cog/SKILL.md)
and use `cog submit --launcher ray`, `--ntasks-per-node 1`, and a setup command
that overlays `nemo_rl/`, `tests/`, and `examples/` on every node.

Run the declared topology, not a reduced topology that bypasses the feature.
Capture:

- exact command and git SHA
- cluster/GPU topology
- Slurm job/run identifier
- pass/fail and runtime
- the metric or log assertion proving feature engagement

Retry infrastructure failures at most twice. For a genuine test failure, fix
the test or implementation and rerun the exact test. Do not open a PR while the
test is failing, incomplete, silently skipped, or only dry-run validated.

## 6. Review the diff

Before committing:

- inspect all tracked and untracked changes
- exclude unrelated user changes
- confirm no credentials, local checkpoint paths, generated logs, metrics,
  coverage files, or run directories are included
- ensure the test name states the feature and follows repository conventions
- include comments only where they explain a non-obvious assertion or override

## 7. Commit and open the PR

Read and follow
[contributing](../contributing/SKILL.md). Create a focused branch from current
`main` unless the user supplied another base. Commit with sign-off and a
Conventional Commit message, for example:

```text
test(megatron): cover <generation feature>
```

Push the branch and open a non-draft pull request against `main`.

Use:

```text
Title: test(megatron): cover <generation feature>

## Summary
- Add Megatron generation coverage for <feature/model/topology>.
- Verify generation and policy training both use Megatron.
- Assert <metric/log invariant>.

## Test plan
- [x] <local validation command>
- [x] <exact cog test command or concise run identifier> — PASS
```

Keep the description brief, but include nightly classification, hardware, and
the evidence that the feature engaged. Return the PR URL and test run evidence
to the user. Do not trigger CI or add labels unless requested.
