# 8-node before/after for `env.nemo_gym.max_concurrency`

**This is the primary validation of the `max_concurrency` fix.** The 68-node
inverse probe in `examples/nemo_gym/nemotron-3.5-nano/gym_max_concurrency_experiment.md`
is demoted to follow-up: it induces a stall in a healthy configuration, whereas
this rerun lifts a cap on a failure we have already reproduced and measured, at
one eighth the nodes and on the `short` QoS.

Everything in "Verified from disk" below was read off the run directories, not
recalled. Where a previously circulated figure did not survive checking, the
measured value is given and the discrepancy called out.

## Verified from disk

Job **5982305** is `sauramishra-gymshard-n8-inflight5120-w1-dp1`, identified by
`runs/latest/slurm/5982305.out` in that results directory. Its pair 5982307 is
the `w16-dp1` directory.

Source of truth: `runs/latest/provenance.txt` in each results directory, plus
`pipeclean_6k_sc_fanin_n8.yaml` and its parent `pipeclean_6k_sc.yaml`.

| Property | Value | Where from |
|----------|-------|-----------|
| Nodes, hetgroup 0 | 4 train + 1 generation + 1 Gym = 6, segment 2 | launcher defaults |
| Nodes, hetgroup 1 | 1 GenRM TP4 + 1 NL2Bash TP4 = 2, segment 2 | launcher defaults |
| Nodes, total | **8** | 6 + 2 |
| `max_inflight_prompts` | **5120** | provenance (launcher override) |
| `max_buffered_rollouts` | **5152** | `pipeclean_6k_sc_fanin_n8.yaml` |
| Sampler | `in_order` | `pipeclean_6k_sc.yaml` |
| Lookahead | `max_lookahead_versions: 160` | `pipeclean_6k_sc_fanin_n8.yaml` |
| `num_prompts_per_step` | 32 | `pipeclean_6k_sc_fanin_n8.yaml` |
| `num_storage_units` | **4** | provenance |
| Container | `images-striped/nemo-rl-nightly-20260806-sandbox.squashfs` | provenance |
| Judge mode | **`EXTERNAL_JUDGES=1`** | provenance shows `__GENRM_BASE_URL__` / `__NL2BASH_BASE_URL__` placeholders |
| Safety judge | `...local_vllm_model.vllm_serve_kwargs.data_parallel_size=1` | provenance |
| Rollout actors | `++env.nemo_gym.num_rollout_workers=1` | provenance |
| Steps / wall | `grpo.max_num_steps=2`, `WALLTIME=2:00:00`, `SLURM_QOS=short` | launcher |
| Branch / commit | `validate/gym-sharding-fanin-n8` @ `d5e189289ca98df56d8b327c9504d2cf2277cdf3` | provenance |

### Correction: buffer does not equal in-flight here

5982305 ran **in-flight 5120 against buffer 5152**, not 5120/5120. The two
cannot be equal in this profile: `validate_sampler_buffer_capacity` enforces
`max_buffered_rollouts >= num_prompts_per_step * (max_lookahead_versions + 1)`,
which is `32 * 161 = 5152`. The buffer-equals-in-flight rule that governs the
nano SC sweep comes from that launcher's own derivation and from job 6014206;
it does not transfer to this config, whose floor forbids it. **Preserve
5152/5120 exactly as 5982305 ran them** — both are config defaults, so neither
arm needs to pass them.

### Telemetry, re-measured

From `ray_logs/<jobid>-logs/ray-driver.log`, single awk pass:

| Metric | 5982305 (w1) | 5982307 (w16) |
|--------|--------------|---------------|
| `gym_fanin` lines | **72** | **15** |
| Distinct reporting actors | **1** | 11 |
| Max `inflight` | **1000** | 187 |
| Max `peak` | **1000** | 189 |
| Max `started` | **2224** | 320 |
| Max `completed` | **119** | 1 |
| `pending submission to actor` | 69 | — |
| Driver log size | 20.3 GB / 128,287,656 lines | 72 MB |

Final 5982305 sample: `inflight=807 peak=1000 started=2224 completed=119
failed=1298 cancelled=0 generations_done=2019 first_generation_after=63.7s`.

Confirmed as quoted: 72 and 15 `gym_fanin` lines, peak **1000**, completed
**119**. **Corrected: `started` is 2224, not 2212** — 2212 was presumably read
off a non-final sample.

A peak of exactly 1000 with a single reporting actor is the fingerprint: nothing
in this configuration is naturally shaped like 1000, and Ray's
`DEFAULT_MAX_CONCURRENCY_ASYNC` is 1000. The 69 pending-submission warnings
corroborate that tasks were queueing at the actor rather than running.

### What "w1 versus w16" actually varied

It was **true actor sharding**, on that branch. The `validate/gym-sharding-fanin-n8`
worktree carries a `NemoGymRolloutPool`: `resolve_num_rollout_workers` reads
`env.nemo_gym.num_rollout_workers`, `spinup_nemo_gym_actor` creates one head
actor plus `N-1` additional `NemoGym` actors that `_attach` to the head's server
as plain HTTP clients, and prompts round-robin across them. So w16 really did
run 16 rollout actors against one Gym fleet — 11 of them lived long enough to
report telemetry.

This corrects the working assumption that the actor count is not configurable.
That is true of `validate/sc-fault-tolerance-n8` and of main, where
`num_rollout_workers` does not exist and `num_gpu_nodes` only sizes the Gym
node reservation. It is **not** true of the branch 5982305 ran on.

It matters directly here: actor count multiplies effective concurrency
(`N * max_concurrency`), so it is the one variable that could counterfeit the
result. **Both new arms must pin `num_rollout_workers=1`.** Note the fallback
would also give 1 (`max(1, num_gpu_nodes)` with `num_gpu_nodes=1`), but pass it
explicitly so the provenance matches 5982305 token for token.

### The confound, and why the peak survives it

Both arms were badly damaged, but **not by the same fault**, and neither matches
the circulated error counts.

- **5982307 (w16)** hit the renderer defect: `RuntimeError: cannot schedule new
  futures after shutdown`, raised from `VllmAsyncGenerationWorker`, **8,610
  times** in its driver log.
- **5982305 (w1)** contains that string **zero times** — in the Ray driver log
  and in every per-server log under `runs/latest/logs/nemo_gym/`. It failed a
  different way: **84,525** `Internal Server Error` in `policy_model.log` plus
  10,916 in `policy_model_reasoning_off.log`, and a 20.3 GB driver log dominated
  by a repeating vLLM `AsyncLLM` output-queue traceback (`out = q.get_nowait()
  or await q.get()` → `raise output`) from a single generation worker.

So the symptom was the same (the policy endpoint 500-ing on nearly everything)
but the named mechanism is confirmed only for w16. Calling 5982305's failure
"the renderer-executor defect" is **UNVERIFIED**; the mass-500 symptom is not.
The circulated figures of ~21,800 and ~15,800 connection errors did not
reproduce under any pattern scanned.

**I agree the in-flight peak survives this, and the direction of the bias is
what makes it safe.** Failed requests return *fast*, which frees admission slots
sooner and therefore pushes in-flight occupancy **down**, never up. 5982305
reached exactly 1000 despite that downward pressure, and sustained a backlog
deep enough for 69 pending-submission warnings. No volume of 500s can
manufacture a hard ceiling at precisely Ray's default; only the cap can.

That same bias creates one asymmetric risk, handled in the refute criterion
below: a 500 storm in Arm 2 could hold its peak *under* 1000 for throughput
reasons, which must not be misread as the knob failing.

## Prerequisite: the fix and the metric are on different branches

Neither worktree can run this experiment as it stands.

| | `max_concurrency` fix | `gym_fanin` telemetry, `num_rollout_workers`, the n8 config and launcher |
|---|---|---|
| `RL-scft` @ `validate/sc-fault-tolerance-n8` | yes (`de9a8d4bf`) | no |
| `RL-gymshard` @ `validate/gym-sharding-fanin-n8` | no — only mentioned in comments | yes |

The decisive metric exists only in the gymshard tree, and that tree is also the
exact code 5982305 ran. So: **cherry-pick `de9a8d4bf` onto a branch cut from
`d5e18928` in `RL-gymshard`.** One commit on top of the reproduction's own code
is the smallest possible perturbation and keeps the before/after honest.

```bash
cd /lustre/fs1/portfolios/llmservice/projects/llmservice_nemotron_ultra/users/sauramishra/RL-gymshard
git checkout -b validate/gym-max-concurrency-n8 d5e189289ca98df56d8b327c9504d2cf2277cdf3
git cherry-pick -x de9a8d4bf     # resolve against the NemoGymRolloutPool hunk
```

Two things to check while resolving:

1. The cherry-pick lands in `spinup_nemo_gym_actor`, which on this branch also
   builds the pool. Apply `max_concurrency` to the **head** actor's
   `nemo_gym_opts`, which is what these arms exercise.
2. The pool's extra actors use a separate `worker_opts = {"runtime_env": ...}`
   that would **not** receive `max_concurrency`. Irrelevant while
   `num_rollout_workers=1`, but it must be fixed before anyone combines the two
   knobs, or a sharded run would silently leave every non-head actor at 1000.

`USE_SNAPSHOT=0` is the launcher default, so both arms execute the live
worktree. Confirm nothing else is dirty before launching.

## The arms

Eight nodes, `short` QoS, two hours. Identical except one Hydra override.

| Arm | `max_concurrency` | Offered in-flight | Cap binding? | Expected peak |
|-----|-------------------|-------------------|--------------|---------------|
| 1 | unset → Ray's 1000 | 5120 | Yes | pins at exactly 1000 |
| 2 | 8192 | 5120 | No | materially above 1000 |

**Why 8192.** It only has to exceed the 5120 the sampler can admit; 8192 is the
next power of two above it, giving 1.6x headroom so the cap cannot bind even
transiently. For an *async* actor Ray's `max_concurrency` bounds concurrent
asyncio tasks rather than sizing a thread pool, so a large value costs little —
this is not 8192 threads. 10000 would serve equally well; anything at or below
5120 would reintroduce a binding cap and waste the arm.

`MAX_INFLIGHT_PROMPTS` is already 5120 by default, and the launcher rejects
anything above it (the sampler could not admit more). It is passed explicitly
below for the record.

### Arm 1 — reproduction, cap unset

```bash
cd /lustre/fs1/portfolios/llmservice/projects/llmservice_nemotron_ultra/users/sauramishra/RL-gymshard
MAX_INFLIGHT_PROMPTS=5120 \
EXP_NAME=sauramishra-gymconc-n8-inflight5120-capdefault \
  bash examples/nemo_gym/nemotron-3-ultra/launch_6k_pipeclean_sc_fanin_n8.sh \
  ++env.nemo_gym.num_rollout_workers=1 \
  env.nemo_gym.safety_judge_model.responses_api_models.local_vllm_model.vllm_serve_kwargs.data_parallel_size=1
```

### Arm 2 — cap raised above the offered load

```bash
cd /lustre/fs1/portfolios/llmservice/projects/llmservice_nemotron_ultra/users/sauramishra/RL-gymshard
MAX_INFLIGHT_PROMPTS=5120 \
EXP_NAME=sauramishra-gymconc-n8-inflight5120-cap8192 \
  bash examples/nemo_gym/nemotron-3-ultra/launch_6k_pipeclean_sc_fanin_n8.sh \
  ++env.nemo_gym.num_rollout_workers=1 \
  ++env.nemo_gym.max_concurrency=8192 \
  env.nemo_gym.safety_judge_model.responses_api_models.local_vllm_model.vllm_serve_kwargs.data_parallel_size=1
```

`DRY_RUN=1` renders the sbatch command without submitting.

### Notes on the interface, checked against the launcher

- **`RESULTS_DIR` does not need to be set here.**
  `launch_6k_pipeclean_sc_fanin_n8.sh:96` defaults it to
  `/lustre/fsw/portfolios/llmservice/users/${USER}/runs/${EXP_NAME}` — the
  submitter's own scratch. The `mkdir: Permission denied` trap that kills the
  nano arms is `nano35_dolphin_launch.sh:288`, a **different launcher** that
  defaults into another user's tree. It does not apply to this chain.
- **`EXP_NAME` must be set per arm.** Its default encodes only the in-flight
  value (`${USER}-6k-sc-fanin-n8-inflight${MAX_INFLIGHT_PROMPTS}`), so both arms
  would otherwise collide in one results directory. Setting `EXP_NAME`
  redirects `RESULTS_DIR` with it.
- **`EXTERNAL_JUDGES=1` is the launcher default** (line 72) and is what 5982305
  ran, so it needs no override. Do not set `GENRM_BASE_URL`; it is mutually
  exclusive with in-job judges.
- **The safety-judge `data_parallel_size=1` override is still required.** The
  new arms reproduce the identical 8-node shape with the same single Gym node,
  so the GPU scarcity that forced it has not changed, and dropping it both
  risks the judge failing to find free GPUs and breaks parity with 5982305.
- `++` is used for `max_concurrency` because no config in this chain declares
  it and Hydra runs in struct mode. This mirrors how `num_rollout_workers` is
  already passed. Declaring it in `pipeclean_6k_sc_fanin_n8.yaml` instead would
  also work and would be tidier if these arms become routine.

### Container: keep the same image, deliberately

Both arms should run `nemo-rl-nightly-20260806-sandbox.squashfs`, the launcher
default and the image 5982305 used. The tradeoff, stated rather than hidden:

- **For keeping it:** this is a before/after against 5982305. Changing the
  image changes the generation path — the exact subsystem producing the 500
  storm — and would leave any difference in Arm 2 attributable to either the
  cap or the image. The decisive metric survives the 500s anyway, for the
  directional reason argued above.
- **Against keeping it:** the image carries whatever fault produced ~95k policy
  500s, so `started` and `completed` will stay depressed and the secondary
  observations will be weak.
- **Practical constraint:** `images-striped/` holds only this one NeMo-RL image.
  Using a newer one means staging a fresh squashfs first, which is separate work
  and would delay a run that currently schedules in minutes.

Recommendation: run the pair on the identical image, and if the secondary
metrics turn out to matter, add a **third** arm on a newer image later rather
than perturbing the pair.

## What confirms, what refutes

The decisive metric is the **in-flight peak** from `gym_fanin`, not throughput.

**Confirms:** Arm 1's peak pins at exactly 1000 while Arm 2's peak materially
exceeds it — treat anything above roughly 1500 as material, with a peak
approaching the 5120 offered being the strongest form. Both arms must report
`distinct_actors=1`.

**Refutes, and this is the point of running it cheaply:** Arm 2's in-flight
**pins at exactly 1000 across consecutive samples**, the same ceiling signature
as Arm 1. That would mean the knob is not reaching the actor and the entire
approach is dead — which is worth discovering for 8 nodes and two hours rather
than at 68.

If that happens, check in order: that the cherry-pick is actually in the running
worktree (`USE_SNAPSHOT=0` means the live tree, so a stale checkout is possible);
that `provenance.txt` contains `max_concurrency=8192`, proving the override
survived Hydra; and that the popped value reaches `NemoGym.options`, since a key
left in `nemo_gym_dict` would be forwarded to Gym's global config and silently
ignored.

**Inconclusive, not a refutation:** Arm 2's peak lands *below* 1000 without ever
touching it — for example 850, fluctuating. That is the 500 storm throttling
throughput before the cap can bind, and it says nothing about the cap. The
discriminator is whether in-flight *sits* at exactly 1000 (ceiling) or wanders
beneath it (starvation). If this happens, fall back to two corroborators: the
`pending submission to actor` count, which should collapse from Arm 1's 69 to
zero if the cap was lifted, and `started`, which should climb past Arm 1's 2224.

**Also stop and investigate** if Arm 1 fails to reproduce peak 1000. The pair is
meaningless until the "before" reproduces.

### Secondary, recorded but not decisive

Prompts `started` out of 5120 (Arm 1: 2224), `completed` groups (Arm 1: 119),
`failed` (Arm 1: 1298), whether any optimizer step lands at all, and whether the
pending-submission warning disappears. All of these are contaminated by the 500
storm and must not be used to adjudicate the hypothesis.

## Reading the telemetry

`gym_fanin` lines go to **`${RESULTS_DIR}/ray_logs/<jobid>-logs/ray-driver.log`**.
They are not in the Slurm `.out` (that holds only the launcher banner), not in
`ray-worker-*.log`, and not in the per-node `ray/` session tree. The actor
prints them with `flush=True` and `RAY_DEDUP_LOGS=1` forwards actor stdout to
the driver, prefixed `(NemoGym pid=...)`.

**That file was 20.3 GB and 128 million lines for 5982305, so a naive grep takes
about four minutes.** Extract everything in one pass:

```bash
DRV="${RESULTS_DIR}/ray_logs/${SLURM_JOB_ID}-logs/ray-driver.log"
awk '/gym_fanin/{
       n++
       if(match($0,/inflight=[0-9]+/)){v=substr($0,RSTART+9,RLENGTH-9)+0; if(v>mi)mi=v}
       if(match($0,/peak=[0-9]+/)){v=substr($0,RSTART+5,RLENGTH-5)+0; if(v>mp)mp=v}
       if(match($0,/started=[0-9]+/)){v=substr($0,RSTART+8,RLENGTH-8)+0; if(v>ms)ms=v}
       if(match($0,/completed=[0-9]+/)){v=substr($0,RSTART+10,RLENGTH-10)+0; if(v>mc)mc=v}
       if(match($0,/actor=[^ ]+/)){a[substr($0,RSTART+6,RLENGTH-6)]=1}
       last=$0
     }
     /pending submission to actor/{p++}
     END{for(k in a)na++;
         printf "lines=%d actors=%d max_inflight=%d max_peak=%d max_started=%d max_completed=%d pending_submission=%d\nLAST: %s\n",
                n, na, mi, mp, ms, mc, p, last}' "$DRV"
```

`actors=1` is a validity check: anything higher means the rollout pool came
back and the arm is void. To see whether in-flight *sits* at the ceiling rather
than merely touching it — the discriminator in the refute criterion — read the
series rather than the maximum:

```bash
grep -a 'gym_fanin' "$DRV" | grep -ao 'inflight=[0-9]*' | cut -d= -f2
```

## Status

Nothing has been submitted. Both arms are unlaunched and carry the cherry-pick
prerequisite above, which must be done first.
