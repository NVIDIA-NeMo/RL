---
name: testing
description: Testing conventions for NeMo-RL. Covers Ray actor coverage pragmas, what earns a place in the nightly/release/performance suites, suite membership declarations, and recipe naming.
when_to_use: Writing or reviewing tests; adding a recipe or a nightly test; deciding whether a test is worth its GPU hours; naming a recipe; 'coverage pragma', 'Ray actor test', 'nightly test', 'SUITE=', 'how to name recipes', during code review of anything under tests/test_suites or examples/configs/recipes.
---

# Testing Conventions

## Coverage and Ray Actors

For any source file under `nemo_rl/*.py` that defines a class or function decorated with `@ray.remote`, add a coverage pragma because these run in separate Ray processes and are not reliably tracked by coverage.

Place `# pragma: no cover` on the `class` or `def` line:

```python
import ray

@ray.remote  # pragma: no cover
class RolloutActor:
    def run(self) -> None:
        ...

@ray.remote  # pragma: no cover
def remote_eval(batch):
    ...
```

---

# Suite Tests

The rest of this file is about the GPU suites — the recipes under
`examples/configs/recipes/` and their drivers under `tests/test_suites/`.

For the mechanics of adding one — files to create, the CONFIG block, the GB200
mirror, how to check your work locally — see @adding-a-test.md. This file is
about whether the test should exist and what it should be called.

## A nightly test is not free

Nightly runs **seven times a week, forever.** A 32 GPU-hour test is 224
GPU-hours/week; a 128 GPU-hour test is nearly 900. Nightly is by far the largest
consumer of the fleet even though a single release run costs more than a single
nightly run — cadence, not per-run size, is what decides a lane's bill.

Each lane has a weekly ceiling in `SUITE_BUDGETS` in
@tests/unit/test_recipes_and_test_suites.py, set per SKU because H100 and GB200
capacity are separate pools that cannot be traded against each other. The
ceilings sit ~15% above current usage, so there is room for a good test and not
much more. **When a lane is full, the fix is to retire a test, not to raise the
ceiling.** The test prints the ten most expensive tests in the lane when it
fails, so you can see what to trade away.

For live numbers: `tools/list-suites --gpu-hours nightly`.

## Cost bands

Per-run GPU-hours for a new nightly test, from `tools/list-suites --gpu-hours`:

| Cost | What it takes |
|---|---|
| **≤ 32 GPU-h** | Fine. Add it. |
| **33–64** | Justify the size in the PR description — say what a cheaper version would fail to catch. |
| **> 64** | Needs a reviewer to agree it is worth the slot. |
| **> 128** | Does not belong in nightly. Put it in `release` (1×/week instead of 7×). |

These are calibrated to how the lane actually looks: about 79% of nightly tests
are in the ≤32 band and account for only ~40% of the lane's cost, while a
handful above 64 carry more than a third of it. The tail is where the budget
goes.

Before assuming you need a big test, try shrinking: fewer steps, a smaller
model, a shorter sequence. Most regressions this suite catches show up in the
first few hundred steps.

## At most two nightly tests per model

Two nightly tests per (algorithm, model) pair. A third needs a reviewer to agree
the coverage is not obtainable from the existing two.

Not counted as separate tests:

- a **GB200 mirror** of an H100 test (same coverage, other SKU)
- a **TQ wrapper** (`-tq_simple`, `-tq_mooncake`) of an existing test
- a **`.vN` bump** replacing its predecessor

The tree currently exceeds this in places — `sft`+`llama3.1-8b`,
`grpo`+`qwen3-30ba3b` and `grpo`+`nanov3-30BA3B` each carry six distinct nightly
H100 tests, `ppo`+`qwen2.5-1.5b-gsm8k` five. Those are the backlog, not the
precedent. The rule bites on **additions**.

## Feature coverage rides on one model

New coverage for a *feature* — sequence packing, activation checkpointing, LoRA,
FP8, non-colocated generation, a new parallelism — goes on
**Qwen2.5-Math-1.5B-Instruct**, not on whichever model you happen to be working
on. It is small and fast: a feature test on it costs ~16 GPU-h instead of the
several hundred a 30B model would, and it is the DTensor control the
`nightly_mcore` gate already carries. (Its plain-Megatron arm is still to be
built; today the only Megatron variant is the single-controller one.)

Two consequences:

- The vehicle is **exempt from the ≤2 rule.** Its test count is supposed to grow
  as features accumulate. That is what it is for.
- The vehicle is **exempt from product-relevance pruning.** Nobody ships
  Qwen2.5-Math-1.5B. It stays anyway, because deleting it would push every
  feature test back onto a large model.

Add a test on a *different* model only when the feature interacts with something
that model specifically has — an MoE routing path, a VLM encoder, a context
length that a 1.5B model cannot exercise. Say which in the PR description.

A new **model** still gets its own test (up to two). This rule is about
features, not models.

## Where tests may not go

- **Never add coverage to a `performance/` recipe.** Those hyperparameters are
  hand-tuned for peak throughput and their numbers feed a tracked series. Adding
  a feature flag to one changes what the series measures. Write a separate
  functional recipe.
- **Nightly node counts: 8 or fewer, 1–4 preferred.** Larger allocations queue
  longer, fail more often, and cost proportionally more. Above 8 nodes in
  nightly, expect to justify it — it probably belongs in `release`, where
  32- and 64-node tests are normal.
- **A new algorithm gets one config to start.** Prove it converges before
  branching into variants.

## GB200 variants

Hardware is not a directory and not a fork. A GB200 recipe `defaults:` onto its
H100 counterpart and overrides only what the SKU forces — `cluster.gpus_per_node: 4`
and any parallelism that has to change. Anything else, and the two drift and
you no longer know whether a GB200 failure is a hardware problem or a
config difference.

The filename carries the SKU as its cluster tuple (`1n4g` vs `1n8g`), and the
driver declares `SKU=gb200`. A unit test checks that `SKU` and `GPUS_PER_NODE`
agree.

## Evidence

A new suite test needs a W&B run showing it converges before it merges. Link it
in the PR description. "It ran without crashing" is not the bar — these tests
exist to detect convergence regressions, so a test that has never demonstrated
convergence cannot detect a change in it.

Every driver must assert something with `tests/check_metrics.py`. A driver that
runs a job and checks nothing burns GPU-hours to turn a job green. The only
exception is a TQ wrapper, which delegates to its base driver and inherits that
driver's assertions.

## Declaring suite membership

Each driver declares its own membership in its CONFIG block:

```bash
# ===== BEGIN CONFIG =====
SUITE=nightly            # nightly | release | performance | disabled
TAGS="mcore"             # optional; labels a slice of the suite
SKU=h100                 # h100 | gb200
NUM_NODES=1
STEPS_PER_RUN=450
MAX_STEPS=450
NUM_RUNS=$(( (MAX_STEPS + STEPS_PER_RUN - 1) / STEPS_PER_RUN ))
NUM_MINUTES=120
# ===== END CONFIG =====
```

- `SUITE=disabled` requires a non-empty `DISABLED_REASON`. A test that stops
  running without a recorded reason is a test nobody can decide to delete.
- `TAGS` labels a *slice* of a suite, not a suite of its own. `TAGS="mcore"`
  puts a test in the `nightly_mcore` gate, which is a subset of `nightly`.
  Membership is deliberate curation, not "has `megatron_cfg.enabled`". Deriving
  it from the config would cost roughly 5× as much — 2,177 GPU-hours per run
  against the curated gate's 465 — and would drop the DTensor control the gate
  carries on purpose.

Read the suites with `tools/list-suites <suite>`, which is plain bash with no
dependencies so a broken `uv sync` cannot make a suite look empty:

```sh
tools/list-suites nightly
tools/list-suites --gpu-hours release
tools/list-suites --suites
```

## Recipe naming

The filename is the **only** place that records what a test covers. Nothing else
says a test exists to exercise sequence packing, or that two tests are a
deliberate pair differing by one knob. `tests/unit/test_recipe_naming.py`
enforces it.

```
<algo>-<model>-<N>n<M>g-<backend[parallelism]>[-<feature>...][.vN].(yaml|sh)
```

VLM recipes prefix the algorithm with `vlm_`.

| Section | Rule |
|---|---|
| **algo** | `sft`, `dpo`, `grpo`, `dapo`, `distillation`, `rm`, … — must be one the repo knows |
| **model** | `llama3.1-8b-instruct`, `qwen2.5-math-1.5b-instruct` |
| **cluster** | `1n8g`, `4n8g` — must match the driver's `NUM_NODES`/`GPUS_PER_NODE` *and* the config's `cluster.gpus_per_node` |
| **backend** | `megatron`, `fsdp2tp1`, `dtensor2tp1`, `automodel`, `hsdp` — fused or separated parallelism both fine; must match which of `megatron_cfg`/`dtensor_cfg` is enabled |
| **feature** | `seqpack`, `actckpt`, `fp8`, `noncolocated`, `lora`, `long`, `quick`, … |
| **`.vN`** | Convergence-affecting changes only |

The mandated sections are checked **both ways** — the name must match the config
and the config must match the name. Feature tokens are checked **one way only**:
a name that claims a feature must have it, but a config may use a feature
without advertising it. That asymmetry is deliberate. Over a hundred recipes
enable sequence packing and eight say `seqpack`, because the token means *"this
test exists to cover packing"*, not *"packing is on"*. Enforcing it both ways
would demand dozens of renames and destroy the signal the name is carrying.

**`.vN`** is for changes that move convergence — a dataset swap, a loss change,
a convergence bug fix — because the version is what tells you a metrics history
is no longer comparable. Pure performance changes do not bump it. The marker
belongs to the base recipe, so a wrapper appends after it:
`…-fsdp2tp1.v3-tq_simple`.

`performance/` and `reproduce/` are exempt from the grammar. Performance recipes
are named for the shape being benchmarked, and `reproduce/` recipes encode what
the reproduction varies — the DeepScaleR chain puts context length where the
cluster tuple would go.

## Reviewing a PR that adds a suite test

The unit tests catch naming, budget overruns, and missing declarations. They
cannot catch a test that is well-formed and not worth running. Check by hand:

1. **What does this catch that an existing test does not?** If the answer is a
   feature, it should be on the vehicle model. If it is a model, it should be
   within the ≤2 budget.
2. **What does it cost?** `tools/list-suites --gpu-hours` on the branch; apply
   the bands. For nightly, multiply by 7 before reacting to the number.
3. **Is there a W&B run?** Convergent, linked in the description.
4. **Does it assert a metric?** `check_metrics.py`, with thresholds that would
   actually fail on a regression.
5. **Right lane?** Expensive and slow-moving belongs in `release`. Throughput
   belongs in `performance/` — and nothing else does.
6. **If it touches `performance/`:** the hyperparameters must be unchanged.

These are judgement calls, not lint. It is reasonable to accept a test that
breaks a rule when the author says why in the PR description — and reasonable to
ask for that sentence when it is missing.
