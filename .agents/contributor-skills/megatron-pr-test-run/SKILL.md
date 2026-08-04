---
name: megatron-pr-test-run
description: Runs the NeMo-RL L1 and L2 Megatron functional suites on the GB200 cluster against a specific Megatron-LM revision, one sub-test at a time, and collects per-test pass/fail results. Use when testing a NemoRLTest-labeled Megatron-LM PR, re-running a single failed functional test, or producing a main-branch baseline.
when_to_use: "run the L1/L2 functional suites for this megatron PR"; re-running one functional test after a fix attempt; producing a baseline on main; interpreting a cog/Slurm job that ran the suites.
---

# Running the functional suites against a Megatron-LM revision

Stage 2 of the nemo-rl-testing-agent pipeline. Turns a pinned Megatron-LM
revision into a per-test verdict for both suites.

Cluster mechanics (cog setup, oci-hsg facts, QOS, image import, secrets) are
**not** repeated here. Read
[run-functional-tests-cog](../run-functional-tests-cog/SKILL.md) for L1 and
[run-nano35-megatron-inference-cog](../run-nano35-megatron-inference-cog/SKILL.md)
for the 2-node nano-3.5 layout L2 depends on.

## How the revision under test gets in

`prep_container.sh` runs inside the container before the suite (on **every**
node for L2, because Ray workers import megatron too). It:

1. Overlays the synced `tests/`, `examples/`, `nemo_rl/` onto `/opt/nemo-rl`
   (the image ships an empty `tests/functional`, and local fixes must be the
   code under test).
2. Pins **Megatron-Bridge** to `$BRIDGE_FETCH_REF` — by default the sha this
   NeMo-RL checkout pins. See the next section for why this is not optional.
3. Fetches `$MCORE_FETCH_REF` and hard-resets
   `/opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM`
   onto it. `megatron-core` is editable-installed from that path, so the
   checkout *is* the install. This happens **after** the Bridge checkout, because
   mcore is a submodule inside Bridge and resetting Bridge afterwards would drag
   mcore back to Bridge's own pin.
4. Aborts if the checked-out SHA differs from `--mcore-sha`, or if
   `import megatron.core` resolves anywhere other than that directory. A
   silently stale megatron would make the whole run meaningless, so treat any
   `NRLTA_PREP_FAIL` as a hard stop, never as a test failure.
5. Imports `megatron.bridge.training.checkpointing` in each megatron worker venv
   as an integration smoke test, failing with `NRLTA_PREP_FAIL_INTEGRATION` if
   the revision under test cannot be loaded together with that Bridge. Unlike
   `NRLTA_PREP_FAIL`, that one **is** a finding about the revision — route it
   through [megatron-pr-failure-triage](../megatron-pr-failure-triage/SKILL.md).

## Why Megatron-Bridge has to be pinned too

NeMo-RL calls megatron-core mostly *through* Megatron-Bridge, so the revision
under test is only meaningful next to a Bridge that can load it. The image's
Bridge is only as new as the image, and swapping in a current megatron-core alone
pairs code that was never meant to run together.

This is not hypothetical. A month-old image failed all six L1 tests 13 minutes
in, every one with:

```
ImportError: cannot import name 'get_default_save_sharded_strategy'
  from 'megatron.core.dist_checkpointing.serialization'
```

The image pinned mcore at a revision where that name still existed as a
deprecation shim; `main` was 252 commits ahead and had removed it, while the
image's Bridge still imported it. Nothing about the PR under test was involved —
its `serialization.py` was byte-identical to `main` — but a naive reading of that
log blames the author for six failures.

Defaulting the Bridge pin to the sha NeMo-RL itself pins keeps the trio
self-consistent: it is the combination NeMo-RL ships and CI expects. Use
`--bridge-ref <sha|ref>` to test a specific Bridge, or `--bridge-ref image` to
keep whatever the image carries (only useful for reproducing an old run).

## Which NeMo-RL is under test

The third leg is pinned the same way, and by default it is **not** plain `main`.
`--nemo-rl-ref` defaults to the integration branch: `main` plus every agent fix
that is raised but not yet merged. Testing against bare `main` would mean each
labeled PR re-hits bugs that are already fixed and waiting on review — the suite
stays red, the author's real problems hide behind someone else's bug, and the
agent spends its fix budget re-deriving a diagnosis it already wrote down.

The trade is that a green table means "green with those patches applied", so
`post_report.py --integration` discloses exactly which ones.

| You want | Pass |
|---|---|
| The normal case | nothing; the integration branch is the default |
| To test an uncommitted local fix | `--nemo-rl-ref worktree --allow-dirty` |
| To test committed local work | `--nemo-rl-ref worktree` |
| Bare upstream `main` | `--nemo-rl-ref main` |

`worktree` mode refuses to run against a dirty tree unless `--allow-dirty` is
given, and then saves the diff next to the run. Before that guard existed, runs
silently picked up whatever the operator happened to have open — one run was
scored with unrelated debug logging patched in, and no report recorded a NeMo-RL
sha at all, so no two runs were comparable.

### The pin is source-only, and that is not an accident

Prep moves `nemo_rl/`, `tests/` and `examples/` to the pinned revision and leaves
everything else — `pyproject.toml`, `uv.lock`, the prebuilt venvs — at the
image's. The reason is that `/opt/ray_venvs/*` were resolved from the image's
lock file, and replacing that lock makes uv rebuild the worker venvs against
different pins than the driver venv still holds. The first attempt at a full
checkout did exactly that, and all six L1 tests died in 29 seconds on
`AttributeError: Can't get attribute '_get_opentelemetry' on
ray.util.tracing.tracing_helper` — a driver/worker Ray mismatch whose message
points nowhere near its cause.

The consequence is a real limit: **a fix that needs a dependency change cannot be
validated this way.** The image has to be rebuilt first. `nemo_rl_env_sha` in the
prep output records which revision the environment came from, so a run is never
silently reported as testing a lock file it did not use.

## Why the suites are not run directly

`L1_Functional_Tests_Megatron_4.sh` and `L2_Functional_Tests_Megatron_4.sh` are
`set -e`: the first failing sub-test aborts the rest, which would leave most
rows in the report unknown. `run_suite_remote.sh` reads the `run_test` lines out
of the suite file and runs each sub-test as an independent step, so every test
gets a verdict. It stays in sync with the suite automatically — commented-out
tests (e.g. the disabled `topp_topk` one) are skipped, `fast` markers are
ignored, and newly added tests are picked up with no change here.

## Steps

### 1. Submit L1, then L2 — sequentially

```bash
.agents/nemo-rl-testing-agent/scripts/run_suite.sh \
  --suite l1 --mcore-ref refs/pull/5700/head --mcore-sha <HEAD_SHA> \
  --run-name nrlta-pr5700-l1-a1
```

`cog submit` blocks for the whole Slurm job, so start it in the background and
poll the `COG_LOG` path it prints. L1 is a single 4-GPU node; L2 is a 2-node
`--launcher ray` job that also mounts the nano-3.5 checkpoint. Add `--dry-run`
first if you changed anything in the wiring.

Budget roughly 1.5–3 h per suite, plus queue time. The `batch` partition MaxTime
is `04:00:00`; use `batch_long` via `--time` only if a suite genuinely needs it.

**Before the job queues, cog re-syncs the workspace**, and that step dominates
the wall clock when anything in the repo changed: it hashes ~1500 tracked and
untracked files and ships them one by one, which over a high-latency link can
take 10+ minutes (lustre bandwidth is not the constraint — a single-file write
benchmarks at ~4.8 GB/s). An unchanged workspace reports `cache_hit: true` and
skips it entirely. So batch your edits: change the agent scripts or `nemo_rl/`
once, then submit, rather than tweaking between submissions. If the transfer
stalls (watch `du -sh <scratch>/workspaces/nemo_rl/*.partial.*` stop growing),
kill the submit, `rm -rf` the partial directory, and resubmit.

Run L2 only after L1 finishes. Both want whole nodes on the same QOS, and
serialising keeps the failure attribution clean.

### 2. Collect results

The run writes to shared scratch (`ARTIFACT_DIR` in the submit output):
`results.tsv` plus one `<test>.log` per sub-test. The markers are also in the
Slurm log, which is what `cog.log` captures.

```bash
COG_LOG=~/.nemo-rl-testing-agent/runs/nrlta-pr5700-l1-a1/cog.log
uv run --script .agents/nemo-rl-testing-agent/scripts/parse_results.py \
  "$COG_LOG" --artifact-dir <ARTIFACT_DIR> \
  --out ~/.nemo-rl-testing-agent/pr-5700/l1.results.json
```

If the Slurm log is truncated in `cog.log`, read it from the cluster instead:

```bash
ssh oci 'cat <SLURM_LOG_DIR>/<JOBID>.out' > /tmp/l1.out
```

Each entry gets `status` (`pass` / `fail` / `incomplete` / `not run`), `rc`,
`secs`, an `error_signature`, and the tail of the output. `comment` is left
empty for the triage stage to fill.

### 3. Separate infrastructure failures from real ones

A non-zero result is only a *test* failure once you have ruled out the cluster.
Re-submit (do not report, do not attempt a fix) when the log shows:

| Signal | Cause |
|---|---|
| `NRLTA_PREP_FAIL` | Revision checkout or the megatron import guard failed. |
| `QOSMinGRES` / "violates accounting policy" | Job asked for less than a whole node. |
| `srun: error`, `slurmstepd: ... CANCELLED / TIME LIMIT` | Preemption, node failure, or a too-short `--time`. |
| `nemo-gym references a workspace ... not a workspace member` | Ran from the synced workspace instead of `/opt/nemo-rl`. |
| `status: incomplete` on the last test only | Job hit the time limit mid-test. |
| `[ERROR] HF_TOKEN is not set` | Tokens file not sourced before submit. |

Everything else — assertion failures in `check_metrics.py`, CUDA errors, OOM,
tracebacks — is a genuine failure and belongs to
[megatron-pr-failure-triage](../megatron-pr-failure-triage/SKILL.md).

Re-submit infra failures at most twice; if the cluster keeps refusing, report
those tests as `not run` with the infra reason and move on.

The guard runs on an allocated GPU node, so a bug in it costs a full cluster
round trip to even see. After touching `prep_container.sh`, run the local
harness first — it fakes a `/opt/ray_venvs` tree and covers the shapes that
matter, including the two `set -e` traps that once aborted prep with no
diagnostic at all (a `for` loop ending in a failing `[ -d ]`, and an assignment
from a failing command substitution):

```bash
bash .agents/nemo-rl-testing-agent/tests/test_import_guard.sh
```

Whenever you abandon or park a run for infra reasons — including a prep failure
that never reached a test — post it to the PR immediately rather than waiting
for a later run to succeed:

```bash
uv run --script .agents/nemo-rl-testing-agent/scripts/post_report.py \
  --pr <N> --state infra --meta megatron-lm=<HEAD_SHA> \
  --note 'Container prep failed: <one line>. Run: `<run-name>`. Retrying.'
```

See [megatron-pr-reporting](../megatron-pr-reporting/SKILL.md#never-leave-the-pr-silent).
An abandoned run that posts nothing is indistinguishable to the PR author from
an agent that never ran at all.

### 4. Re-run a subset

Fix attempts and baselines only need the failing tests:

```bash
.agents/nemo-rl-testing-agent/scripts/run_suite.sh \
  --suite l1 --mcore-ref refs/pull/5700/head --mcore-sha <HEAD_SHA> \
  --tests "grpo_megatron_generation_async grpo_megatron_generation_multiturn" \
  --run-name nrlta-pr5700-l1-a2
```

Use a fresh `--run-name` per attempt (`-a1`, `-a2`, …) so artifacts and logs are
never overwritten and the ledger can point at each one.

Local edits under `nemo_rl/`, `tests/`, or `examples/` are only picked up in
worktree mode: add `--nemo-rl-ref worktree --allow-dirty` to a fix-attempt re-run,
or the job will test the integration branch and your edit will not be there. A fix
inside Megatron-LM or Megatron-Bridge is different again: it must be pushed to a
branch and tested by pointing `--mcore-ref` at that branch (see
[megatron-pr-fix-delivery](../megatron-pr-fix-delivery/SKILL.md)).

## When a run teaches you something this skill does not say

A flag that turned out to be mandatory, a failure mode not in the infra table, a
cluster behaviour that only shows up under load: record it with
[nemo-rl-testing-agent-learning](../nemo-rl-testing-agent-learning/SKILL.md)
rather than only in the ledger. Mistakes at this stage are the expensive ones —
they are paid for in cluster hours and discovered hours after the submit — so
this is the stage where correcting the instructions pays back fastest.

If the gap means the runs already submitted are testing the wrong thing, record
it `--severity blocking` and fix it before the next submit rather than at the end
of the sweep.
