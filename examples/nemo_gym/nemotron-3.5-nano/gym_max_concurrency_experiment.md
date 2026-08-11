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

Set `GENRM_BASE_URL` to a warm GenRM pool that is already serving, or use
`EXTERNAL_JUDGES=1` instead to host GenRM and the NL2Bash judge in-job. Both
shapes total 68 nodes. **Use whichever mode job 5995584 used, and use the same
one for both arms** — the two differ in judge latency and in Gym node count
(16 vs 8), which would confound the comparison.

### Arm A — cap below offered load

```bash
GENRM_BASE_URL=http://<lb-host>:9213/v1 \
EXP_NAME=sauramishra-nano35-sc-conc-probe-armA-cap256 \
SAMPLER=windowed \
MAX_LOOKAHEAD_VERSIONS=4 \
NUM_STORAGE_UNITS=8 \
GYM_MAX_CONCURRENCY=256 \
  bash examples/nemo_gym/nemotron-3.5-nano/nano35_dolphin_launch_sc.sh
```

### Arm B — control, Ray's default cap

```bash
GENRM_BASE_URL=http://<lb-host>:9213/v1 \
EXP_NAME=sauramishra-nano35-sc-conc-probe-armB-default \
SAMPLER=windowed \
MAX_LOOKAHEAD_VERSIONS=4 \
NUM_STORAGE_UNITS=8 \
  bash examples/nemo_gym/nemotron-3.5-nano/nano35_dolphin_launch_sc.sh
```

Arm B passes no `env.nemo_gym.max_concurrency` override at all; the launcher
omits it when `GYM_MAX_CONCURRENCY` is empty. Check the banner before the job is
submitted — it prints `Gym concur : 256` for Arm A and
`Gym concur : unset (Ray default 1000)` for Arm B. Add `DRY_RUN=1` to inspect
the resolved command without submitting.

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
GENRM_BASE_URL=http://<lb-host>:9213/v1 \
EXP_NAME=sauramishra-nano35-sc-conc-probe-armC-1152-default \
SAMPLER=windowed \
MAX_LOOKAHEAD_VERSIONS=8 \
NUM_STORAGE_UNITS=16 \
  bash examples/nemo_gym/nemotron-3.5-nano/nano35_dolphin_launch_sc.sh

# Arm D — same load, cap raised above it
GENRM_BASE_URL=http://<lb-host>:9213/v1 \
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
