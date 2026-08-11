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
result. **Arms 1 and 2 must both pin `num_rollout_workers=1`**, so that the only
thing separating them is the cap. Note the fallback would also give 1
(`max(1, num_gpu_nodes)` with `num_gpu_nodes=1`), but pass it explicitly so the
provenance matches 5982305 token for token.

It also means a sharded arm costs a config value rather than a build, which is
why Arm 3 exists. Arm 3 moves actor count and *nothing else* relative to Arm 2.
It is deliberately not a combined arm: changing cap and actor count together
against Arm 1 is what made 5960334 uninformative, and repeating that would leave
any difference unattributable.

### The confound, and why the peak survives it

Both reference runs were badly damaged, but **not by the same fault**, and
neither matches the circulated error counts.

- **5982307 (w16)** hit the renderer defect: `RuntimeError: cannot schedule new
  futures after shutdown`, raised from `VllmAsyncGenerationWorker`, **8,610
  times** in its driver log.
- **5982305 (w1)** contains that string **zero times** — in the Ray driver log
  and in every per-server log under `runs/latest/logs/nemo_gym/`. It failed a
  different way: **84,525** `Internal Server Error` in `policy_model.log` plus
  10,916 in `policy_model_reasoning_off.log`, and a 20.3 GB driver log dominated
  by a repeating vLLM `AsyncLLM` output-queue traceback (`out = q.get_nowait()
  or await q.get()` → `raise output`) from a single generation worker.

So the symptom was the same — the policy endpoint 500-ing on nearly everything —
but **the renderer-executor defect belongs to 5982307 alone.** Zero occurrences
across 5982305's driver log and every per-server log is positive evidence of
absence, not merely a failure to confirm, so 5982305 should not be described as
an instance of that defect anywhere. Its mass-500 symptom is real and measured;
its mechanism is the vLLM output-queue traceback above. The circulated figures
of ~21,800 and ~15,800 connection errors did not reproduce under any pattern
scanned.

**I agree the in-flight peak survives this, and the direction of the bias is
what makes it safe.** Failed requests return *fast*, which frees admission slots
sooner and therefore pushes in-flight occupancy **down**, never up. 5982305
reached exactly 1000 despite that downward pressure, and sustained a backlog
deep enough for 69 pending-submission warnings. No volume of 500s can
manufacture a hard ceiling at precisely Ray's default; only the cap can.

That same bias creates one asymmetric risk, handled in the refute criterion
below: a 500 storm in Arm 2 could hold its peak *under* 1000 for throughput
reasons, which must not be misread as the knob failing.

## Branch topology — read this before rerunning anything

Three branches are involved and no single one of them held everything. Getting
this wrong is the failure mode that silently invalidates a rerun months later,
so it is recorded explicitly.

| Worktree @ branch | `max_concurrency` fix | `gym_fanin` telemetry, `num_rollout_workers`, the n8 config and launcher | Role |
|---|---|---|---|
| `RL-scft` @ `validate/sc-fault-tolerance-n8` | yes, `de9a8d4bf` | no | where the fix was written; **holds this document** |
| `RL-gymshard` @ `validate/gym-sharding-fanin-n8` | no | yes | the code 5982305 and 5982307 ran; source of all telemetry above |
| `RL-gymshard` @ `validate/gym-max-concurrency-n8` | yes | yes | **all three arms run from here** |

`validate/gym-max-concurrency-n8` is cut from
`d5e189289ca98df56d8b327c9504d2cf2277cdf3` — the exact commit 5982305 ran — with
one commit on top:

```
c8a2ef81715b10c54a920124d369d4e0f672844d  feat(nemo_gym): make each rollout actor's Ray max_concurrency configurable
d5e189289ca98df56d8b327c9504d2cf2277cdf3  feat(gym): per-actor rollout fan-in telemetry   <- 5982305 ran this
```

One commit on top of the reproduction's own code is the smallest possible
perturbation, which is what keeps the before/after honest.

Two notes on the base. First, `validate/gym-sharding-fanin-n8` has since moved
two commits past `d5e18928`, but those are a fan-in threading fix and its own
revert, so `git diff d5e18928 <tip>` is empty and the branch tip's tree is
byte-identical to the reproduction's. Cutting from `d5e18928` therefore costs
nothing and is provenance-exact. Second, `USE_SNAPSHOT=0` is the launcher
default, so the arms execute the **live worktree**, not a snapshot: confirm
`RL-gymshard` is on `validate/gym-max-concurrency-n8` and otherwise clean before
launching. The only expected dirt is the `3rdparty/Gym-workspace/Gym` submodule
pointer, which was already modified when 5982305 ran.

### What the cherry-pick carried, and what it dropped

- **Carried:** the `nemo_gym.py` change. `max_concurrency` is popped from
  `nemo_gym_dict` alongside `num_rollout_workers` and passed as a Ray actor
  option, so the key never reaches Gym's `initial_global_config_dict`.
- **Fixed in the same commit:** the pool's extra actors were built from a
  separate `worker_opts = {"runtime_env": ...}` that did **not** receive
  `max_concurrency`. Harmless at `num_rollout_workers=1`, but it would have left
  every non-head actor at Ray's 1000 the moment Arm 3 ran — a half-effect that
  reads as "the cap does not work". `worker_opts` is now derived from the head's
  options minus the node pin, so the cap applies uniformly and the `None`
  sentinel still means "Ray's default everywhere".
- **Dropped:** the `nano35_dolphin_launch_sc.sh` and `rlvr_dolphin_sc.yaml`
  hunks. That recipe has diverged on this branch — its launcher predates the
  `SAMPLER` block — so taking those hunks would have imported unrelated nano
  changes. No n8 arm loads either file. The 68-node probe document was dropped
  for the same reason and stays on `RL-scft`.
- **Added instead:** `env.nemo_gym.max_concurrency: null` in
  `pipeclean_6k_sc_fanin_n8.yaml`, which the arms do load.

## The arms

Eight nodes, `short` QoS, two hours each. **Each arm moves exactly one variable
relative to its predecessor** — that is the whole discipline here, and the
reason 5960334, which moved two at once, taught us nothing.

| Arm | `max_concurrency` | `num_rollout_workers` | Moves, vs. | Question it answers |
|-----|-------------------|----------------------|-----------|---------------------|
| 1 | unset → Ray's 1000 | 1 | 5982305: nothing | Does the failure reproduce? |
| 2 | 8192 | 1 | Arm 1: the cap | Is the cap what pinned in-flight at 1000? |
| 3 | 8192 | 4 | Arm 2: actor count | With the cap gone, does spreading fan-in help? |

Arms 1 and 2 are the experiment. Arm 3 is a follow-on and should only be
launched once Arm 2 has answered its question; if Arm 2 refutes, Arm 3 is moot.

**Why 8192 for the cap.** It only has to exceed the 5120 the sampler can admit;
8192 is the next power of two above it, giving 1.6x headroom so the cap cannot
bind even transiently. For an *async* actor Ray's `max_concurrency` bounds
concurrent asyncio tasks rather than sizing a thread pool, so a large value
costs little — this is not 8192 threads. 10000 would serve equally well;
anything at or below 5120 would reintroduce a binding cap and waste the arm.

### Why 4 actors for Arm 3, and why not 16

Arm 3 tests the *other* half of the sharding rationale. Arm 2 removes the
concurrency ceiling; a single actor still serializes every rollout's
postprocessing — a tokenizer decode plus re-encode per generation — onto one
thread on one node, which the branch's own docstrings already call out as the
second bottleneck. Arm 3 asks whether relieving that buys throughput once the
cap is out of the way.

**4 is the largest count that keeps the fix falsifiable.** With 5120 offered
prompts round-robined across `N` actors, each actor's share is `5120 / N`. That
share has to exceed Ray's 1000 default for the per-actor cap to matter at all:

| N | Per-actor share | Above Ray's 1000? |
|---|-----------------|-------------------|
| 1 | 5120 | yes |
| 4 | 1280 | yes |
| 6 | 853 | no |
| 16 | 320 | no |

At `N >= 6` no individual actor ever wants more than 1000 in flight, so the
uniform-cap fix would be inert and Arm 3 could not distinguish a working fix
from a broken one. At `N = 4` each non-head actor wants 1280, so **a per-actor
in-flight above 1000 is direct evidence the cap reached the non-head actors** —
which is exactly the trap the fix closes. 2, 3 and 5 also satisfy this; 4 is
picked as the largest power of two among them, giving the widest spread of
postprocessing while staying observable.

**16 is disqualified, and cautiously so.** w16 is not a promising precedent: it
peaked at 189 in flight and completed **1** group, against w1's peak 1000 and
**119** groups. More actors completed *fewer* groups. That is unexplained, and
until it is explained the honest reading is that actor count carries its own
pathology, not that it helps. Two hints sit in the telemetry: only **11 of 16**
actors ever reported `gym_fanin`, suggesting several never came up, and w16's
peak of 189 is nowhere near any cap, so whatever throttled it was not
concurrency. Arm 3 at 4 is a probe of that region, not an endorsement of it: if
4 behaves like 1, the w16 pathology is superlinear in actor count; if 4 already
degrades, it is not.

Because of that, **Arm 3 is not expected to improve anything, and a null or
negative result is a perfectly good outcome.** Its load-bearing job is the
uniform-cap check; the throughput question is secondary.

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
  env.nemo_gym.max_concurrency=8192 \
  env.nemo_gym.safety_judge_model.responses_api_models.local_vllm_model.vllm_serve_kwargs.data_parallel_size=1
```

### Arm 3 — cap held at Arm 2's value, actor count raised

Launch only after Arm 2 has reported.

```bash
cd /lustre/fs1/portfolios/llmservice/projects/llmservice_nemotron_ultra/users/sauramishra/RL-gymshard
MAX_INFLIGHT_PROMPTS=5120 \
EXP_NAME=sauramishra-gymconc-n8-inflight5120-cap8192-w4 \
  bash examples/nemo_gym/nemotron-3-ultra/launch_6k_pipeclean_sc_fanin_n8.sh \
  ++env.nemo_gym.num_rollout_workers=4 \
  env.nemo_gym.max_concurrency=8192 \
  env.nemo_gym.safety_judge_model.responses_api_models.local_vllm_model.vllm_serve_kwargs.data_parallel_size=1
```

Arm 3's command is Arm 2's with `1` changed to `4`. That is the diff, and it
should stay the diff.

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
- **`max_concurrency` uses a bare `key=value`; `num_rollout_workers` needs
  `++`.** The distinction is deliberate. `nemo_rl/utils/config.py` calls Hydra's
  real `OverridesParser` and `ConfigLoaderImpl._apply_overrides_to_config` under
  `OmegaConf.set_struct(cfg, True)`, so full Hydra semantics apply: `key=value`
  requires the key to exist, `+key=value` requires it not to, and `++key=value`
  forces it either way. `++env.nemo_gym.max_concurrency=8192` would therefore
  work with **no YAML declaration at all** — that is how 5982305 passed
  `++env.nemo_gym.num_rollout_workers=1`, a key nothing in the chain declares.
  The declaration was still added, because `++` silently invents the key: run
  the arm from a checkout that predates the cherry-pick and `++` sets a config
  value nobody reads, producing a clean-looking run that is really Arm 1 wearing
  Arm 2's name. The bare form raises instead. `num_rollout_workers` keeps `++`
  because it genuinely is undeclared and because 5982305's provenance spells it
  that way, which is worth matching token for token.

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

Recommendation: run all three arms on the identical image. If the secondary
metrics turn out to matter, add a separate image arm afterwards rather than
perturbing this sequence — the same one-variable rule applies, so an image
change must be its own arm and not be folded into Arm 3.

## What confirms, what refutes

The decisive metric is the **in-flight peak** from `gym_fanin`, not throughput.

**Confirms:** Arm 1's peak pins at exactly 1000 while Arm 2's peak materially
exceeds it — treat anything above roughly 1500 as material, with a peak
approaching the 5120 offered being the strongest form. Arms 1 and 2 must both
report `actors=1`; anything higher means the rollout pool came back and the
comparison is void.

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

### Arm 3, judged separately

Arm 3 does not re-adjudicate the cap; Arms 1 and 2 have already done that. It
has one decisive check and one open question, and they are scored differently.

**Decisive — did the cap reach the non-head actors?** Read in-flight
*per actor*, not summed. Each of the 4 actors should want roughly `5120 / 4 =
1280` in flight. If **every** actor's peak sits at exactly 1000 while the total
sits near 4000, the uniform-cap fix is not working and the non-head actors fell
back to Ray's default — the exact trap this arm exists to detect. At least one
non-head actor exceeding 1000 confirms the fix. Note that a peak *below* 1000 on
some actors is not evidence either way; it only means that actor was starved,
which the 500 storm makes likely.

**Open, not decisive — did sharding buy throughput?** Compare `started` and
`completed` against Arm 2. Given w16 completed 1 group against w1's 119, treat a
regression as the expected-if-disappointing outcome rather than as a bug in the
arm. Record it and move on. What would be genuinely informative either way:
whether all 4 actors report `gym_fanin` at all (w16 lost 5 of 16) and how
`first_generation_after` compares, since extra actor startup on a single Gym
node is the leading suspect for w16's collapse.

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

`actors` is a validity check: it must be **1** for Arms 1 and 2 — anything
higher means the rollout pool came back and the arm is void — and **4** for
Arm 3, where fewer than 4 means actors failed to start, which is itself the w16
finding worth reporting.

To see whether in-flight *sits* at the ceiling rather than merely touching it —
the discriminator in the refute criterion — read the series rather than the
maximum:

```bash
grep -a 'gym_fanin' "$DRV" | grep -ao 'inflight=[0-9]*' | cut -d= -f2
```

For Arm 3 the same maxima must be taken **per actor**, since the summed peak
hides the thing being tested. A pool where every actor tops out at exactly 1000
is the broken-uniform-cap signature:

```bash
grep -a 'gym_fanin' "$DRV" \
  | sed -n 's/.*actor=\([^ ]*\) .* peak=\([0-9]*\).*/\1 \2/p' \
  | awk '{if($2>m[$1])m[$1]=$2} END{for(k in m)printf "actor=%s peak=%d\n", k, m[k]}'
```

`peak` is the actor's own running maximum (`stats.inflight_peak`), so its last
sample is already the answer; taking the max over samples just guards against a
truncated tail. Each `gym_fanin` line is
`actor=<id> node=<ip> inflight=.. peak=.. started=.. completed=..`, one per
actor every reporting interval.

## Status

Nothing has been submitted. All three arms are unlaunched.

The cherry-pick prerequisite is **done**: `validate/gym-max-concurrency-n8` @
`c8a2ef81715b10c54a920124d369d4e0f672844d` exists in `RL-gymshard`, carries the
`max_concurrency` fix applied uniformly to head and pool actors, and declares
the key in `pipeclean_6k_sc_fanin_n8.yaml`. Nothing blocks Arm 1 and Arm 2 but
the decision to submit. Arm 3 waits on Arm 2's result.
