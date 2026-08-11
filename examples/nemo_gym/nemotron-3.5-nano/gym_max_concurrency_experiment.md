# Does Ray's actor concurrency cap gate NeMo-Gym rollouts?

Paired validation for `env.nemo_gym.max_concurrency`, added so the single
`NemoGym` Ray actor's admission cap is configurable instead of always sitting at
Ray's default.

## Hypothesis

`NemoGym.run_rollouts` is `async def`, so Ray builds the actor on the asyncio
branch and applies `DEFAULT_MAX_CONCURRENCY_ASYNC = 1000` (Ray 2.56.1) when
`max_concurrency` is not passed. Until this change it was never passed:
`nemo_gym_opts` carried only `scheduling_strategy` and `runtime_env`.

At 6K scale, job 5960334 offered the actor 5120 in-flight prompts against that
cap of 1000, completed zero training steps, and Ray logged *"More than 5000
tasks pending submission to actor NemoGym"*. The claim under test is that the
actor's concurrency cap — not engine throughput, not the data plane, not the
sampler — is what gates rollout completion once offered load exceeds it.

## Design: the inverse probe (primary)

The tempting experiment is to push offered load past 1000 and watch it break.
That is a bad experiment: raising in-flight prompts also raises engine
saturation, KV pressure and buffer occupancy, so a stall would be consistent
with several mechanisms at once.

Instead, hold offered load fixed at a configuration already known to be healthy
and move only the cap. Job 5995584 ran the windowed sampler at 640 in-flight on
68 nodes and completed 19 steps in 4 h. Both arms below reproduce it exactly —
same sampler, same staleness, same buffer, same in-flight, same node shape,
same storage units — and differ in one Hydra override.

| Arm | `GYM_MAX_CONCURRENCY` | Effective cap | Offered in-flight | Cap binding? |
|-----|----------------------|---------------|-------------------|--------------|
| A   | `256`                | 256           | 640               | Yes, hard |
| B   | unset                | 1000 (Ray)    | 640               | No |

Arm A is the treatment: an artificially low cap, well below the offered load, in
a configuration that is otherwise known to train. Arm B is the control and
should simply reproduce 5995584.

Because `max_buffered_rollouts == max_inflight_prompts` in both arms, the
buffer/in-flight coupling is untouched. That equality is load-bearing and must
not be relaxed to run this: job 6014206 at buffer 768 / in-flight 384 ran 1.8x
slower than the identical 5995582 at 384/384, with eviction rising 21.4% to
29.4%. The launcher derives both from one number, so this holds automatically.

## Launch commands

Both arms go through `nano35_dolphin_launch_sc.sh`. The launcher derives
`_BUFFER_CAPACITY = _NUM_PROMPTS_PER_STEP * (MAX_LOOKAHEAD_VERSIONS + 1)`, so
`128 * (4 + 1) = 640` is passed to both `async_rl.max_inflight_prompts` and
`async_rl.max_buffered_rollouts`. `SAMPLER=windowed` makes
`MAX_LOOKAHEAD_VERSIONS` emit `+async_rl.sampler.max_staleness_versions`, which
is the key the windowed sampler actually reads.

### Judge hosting: `EXTERNAL_JUDGES=1`, and no `GENRM_BASE_URL`

Both arms host GenRM and the NL2Bash judge in-job, in a second Slurm hetgroup.
This is settled, not a preference.

Job 5995584 ran as a **heterogeneous** job. All four arms of that sweep appear
in Slurm as `5995580+0`, `5995582+0`, `5995584+0` and `5995586+0`; the `+N`
suffix is a hetjob component index and exists only for heterogeneous jobs. The
three v1 arms pending as of this writing (6032239, 6032458, 6032460) show the
same `+0` / `+1` pattern at 56 and 12 nodes. `EXTERNAL_JUDGES=1` is precisely
what produces that two-component shape.

**Do not set `GENRM_BASE_URL`.** The GenRM URLs in older logs are ephemeral
compute-node addresses belonging to a pool whose job terminated on August 8;
passing one would aim the arms at a dead endpoint. It would also fail outright:
`ultra_launch.sh` treats `GENRM_BASE_URL` and `EXTERNAL_JUDGES=1` as mutually
exclusive and exits with an error. Leaving it unset is safe — the launcher
defaults it to empty, then substitutes its own `__GENRM_BASE_URL__` placeholder,
which the allocation wrapper replaces with the resolved load-balancer URL once
the pools are healthy.

`GENRM_MODEL` and `NL2BASH_JUDGE_MODEL` are required under `EXTERNAL_JUDGES=1`.
Both already have defaults in `nano35_dolphin_launch.sh`, so neither needs to be
passed.

### Node arithmetic

`EXTERNAL_JUDGES=1` drops the Gym pool from 16 nodes to 8, because the NL2Bash
judge vacates exactly the 8 it was filling and moves to the judge hetgroup.

| Component | Derivation | Nodes |
|-----------|-----------|-------|
| Hetgroup 0 — train | `NUM_TRAIN_NODES` | 8 |
| Hetgroup 0 — generation | `NUM_GEN_NODES` | 40 |
| Hetgroup 0 — Gym | `_DEFAULT_GYM_NODES` under `EXTERNAL_JUDGES=1` | 8 |
| **Hetgroup 0 total** | `NUM_RAY_NODES` | **56** |
| Hetgroup 1 — GenRM | 2 replicas x TP 8 / 4 GPUs per node | 4 |
| Hetgroup 1 — NL2Bash | 8 replicas x TP 4 / 4 GPUs per node | 8 |
| **Hetgroup 1 total** | `EXTERNAL_VLLM_NUM_NODES` | **12** |
| **Job total** | `NUM_TOTAL_NODES` | **68** |

Hetgroup 1 is sized by `pool_config.sh` as the sum over registered pools of
`REPLICAS * TENSOR_PARALLEL_SIZE / GPUS_PER_NODE`; no extra node is reserved for
the load balancers, which run inside the pools. This reproduces 5995584's 56+12
exactly. Both components clear their segment checks at the nano values
(`SEGMENT_SIZE=2` divides 56, `EXTERNAL_VLLM_SEGMENT_SIZE=2` divides 12), so
neither arm should be rejected at submission.

### Arm A — cap below offered load

```bash
EXTERNAL_JUDGES=1 \
EXP_NAME=sauramishra-nano35-sc-conc-probe-armA-cap256 \
SAMPLER=windowed \
MAX_LOOKAHEAD_VERSIONS=4 \
NUM_STORAGE_UNITS=8 \
GYM_MAX_CONCURRENCY=256 \
  bash examples/nemo_gym/nemotron-3.5-nano/nano35_dolphin_launch_sc.sh
```

### Arm B — control, Ray's default cap

```bash
EXTERNAL_JUDGES=1 \
EXP_NAME=sauramishra-nano35-sc-conc-probe-armB-default \
SAMPLER=windowed \
MAX_LOOKAHEAD_VERSIONS=4 \
NUM_STORAGE_UNITS=8 \
  bash examples/nemo_gym/nemotron-3.5-nano/nano35_dolphin_launch_sc.sh
```

Arm B passes no `env.nemo_gym.max_concurrency` override at all; the launcher
omits it when `GYM_MAX_CONCURRENCY` is empty.

### Pre-submit check

Prepend `DRY_RUN=1` to either command. `ultra_launch.sh` prints the resolved
`TRAIN_CMD` and exits before submitting, and it is the supported way to inspect
this shape — `INTERACTIVE=1` is rejected outright when external judge services
are configured. Three things to read off the output:

- The SC banner prints `Gym concur : 256` for Arm A and
  `Gym concur : unset (Ray default 1000)` for Arm B. This is the check that the
  two arms actually differ, and it is unaffected by the judge-mode correction:
  the SC launcher emits its banner before handing off, so it appears under
  `DRY_RUN=1` just as it does on a real submission.
- `TRAIN_CMD` should contain `env.nemo_gym.max_concurrency=256` for Arm A and no
  `max_concurrency` token at all for Arm B.
- The ultra banner should report `Hetgroup 1: 12 external-service nodes`
  alongside a 56-node hetgroup 0, confirming the 68-node total above.

A dry run is cheap and submits nothing; do it for both arms before either goes
to the queue.

## Parity warning: storage units

Job 5995584 ran `data_plane.num_storage_units=8`. `nano35_dolphin_launch_sc.sh`
now defaults `NUM_STORAGE_UNITS=16` and passes it unconditionally, so an arm
that omits it is **not** a reproduction of 5995584. Both commands above pass
`NUM_STORAGE_UNITS=8` explicitly for that reason. What matters most is that the
two arms agree; matching 5995584 additionally makes Arm B checkable against a
known result.

## What confirms, what refutes

Stated before the run.

**Confirms the hypothesis** — all of:

- Arm A shows the 6K stall signature: the *"More than 5000 tasks pending
  submission to actor NemoGym"* warning appears, and step throughput collapses
  relative to Arm B (order-of-magnitude, not a few percent).
- Arm B does not show that warning and completes roughly 19 steps in 4 h,
  matching 5995584.
- Arm A's rollout completion rate stalls while the generation engines are *not*
  saturated — `vllm:num_requests_running` well below capacity with
  `num_requests_waiting` near zero. Starved engines behind a full actor queue is
  the signature; busy engines are not.

**Refutes the hypothesis** — any of:

- Arm A trains at approximately Arm B's rate despite a cap of 256 against 640
  offered. Then the actor's concurrency ceiling is not what limits rollouts, and
  the 6K stall has another cause.
- Arm A stalls but with saturated engines (`num_requests_running` at capacity,
  a persistently non-empty waiting queue). That is generation-bound, and the cap
  is incidental.
- Arm B also stalls. Then something other than `max_concurrency` regressed
  between 5995584 and now, and the pair says nothing until that is explained.
- Neither arm ever emits the pending-submission warning even though Arm A
  stalls. The warning is the direct evidence of queueing at the actor; without
  it, a stall in Arm A needs a different explanation.

## Where to look

**The pending-submission warning lives in the NemoGym actor's own worker log,
not in the driver log.** Ray prefixes it with `(raylet)` when re-printing it,
but that prefix is a printed *label* and not the file it lives in. Find the
right file by its header: line 2 of the actor's log is `:actor_name:NemoGym`.

```bash
RUN=<results-dir>/runs/latest      # ${RESULTS_DIR}/ray_logs is BASE_LOG_DIR
grep -l ':actor_name:NemoGym' $(find "${RUN}" -name 'worker-*.out')
```

Two traps when reading it:

1. **Do not count occurrences as events.** Ray's threshold starts at 5000 and
   *doubles* on each crossing (5000, 10000, 20000, ...). Repeated identical
   "More than 5000" lines are duplicate stdout streams for one crossing, not
   distinct backlog events. Read the distinct thresholds, and read the
   timestamps; a single first occurrence with a timestamp is the useful datum.
2. **Training step markers are in the SingleControllerActor's worker log**, not
   necessarily the driver log. Earlier analyses concluded "zero steps completed"
   by grepping only the driver log, which is not where the SC actor writes.
   Locate the SC actor's `worker-*.out` the same way before making any claim
   about step count.

Cross-check step counts against W&B (`joc/ultra-streaming`), which is
independent of which log file the markers land in.

## Forward probe (follow-up confirmation only)

Run this only after the inverse probe returns a clear result. It tests the same
mechanism from the other side: push offered load *past* 1000 and show that
raising the cap rescues it.

`128 * (8 + 1) = 1152` in-flight, which crosses the 1000 gate.

```bash
# Arm C — over the gate, Ray's default cap (expected to gate)
EXTERNAL_JUDGES=1 \
EXP_NAME=sauramishra-nano35-sc-conc-probe-armC-1152-default \
SAMPLER=windowed \
MAX_LOOKAHEAD_VERSIONS=8 \
NUM_STORAGE_UNITS=16 \
  bash examples/nemo_gym/nemotron-3.5-nano/nano35_dolphin_launch_sc.sh

# Arm D — same load, cap raised above it
EXTERNAL_JUDGES=1 \
EXP_NAME=sauramishra-nano35-sc-conc-probe-armD-1152-cap2048 \
SAMPLER=windowed \
MAX_LOOKAHEAD_VERSIONS=8 \
NUM_STORAGE_UNITS=16 \
GYM_MAX_CONCURRENCY=2048 \
  bash examples/nemo_gym/nemotron-3.5-nano/nano35_dolphin_launch_sc.sh
```

This pair is weaker evidence than the inverse probe: raising the lookahead to 8
also raises staleness and buffer residency (1152 groups, ~18432 rows), so C-vs-D
is clean but neither is comparable to 5995584. Keep `NUM_STORAGE_UNITS` equal
across C and D. Treat a C stall plus a D recovery as confirmation, and a D that
still stalls as evidence that at 1152 something beyond the cap also binds.
