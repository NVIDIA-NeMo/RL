# Measuring rollout performance

How to measure a rollout change so that the result means something. This page
applies to every generation backend.

## Evidence grades

Give every number you report one of these grades.

| Grade | Meaning |
| :-: | :-- |
| A | One factor changed; at least 3 accepted runs per arm; same container, code and placement class |
| B | At least 3 runs in the treatment arm, but a bundle changed or the control has fewer runs |
| C | Fewer than 3 runs in the treatment arm, or the source/runtime changed with the knob |
| D | Cross-window, cross-cluster or cross-stack observation; no causal claim |

A CI recipe change may land on grade C when:
- the per-step effect is large (≥ 20%) and consistent across paired steps;
- the arms ran in the same window; and
- the PR says it is grade C.

Engine comparisons and small effects (< 10%) need grade A or B.

## Which metric supports which claim

| Metric | Supports | Caveat |
| :-- | :-- | :-- |
| Per-request latency distribution (p50/p90/p99) | Closest signal to the engine itself | Includes queueing, tokenization, network and endpoint topology. Report request coverage |
| Generated tokens per rollout-second (Σ tokens / Σ seconds) | Engine throughput, when token accounting and workload match | Invalid if one arm truncates, times out or generates a different amount |
| Per-trajectory or per-step generation time | What the training loop waits for | A synchronous step ends when its slowest sample ends, so the tail sets it |
| Step time, job wall time, GPU-hours | The deployed system | Includes startup, refit, logprob, training, harness, storage and stragglers |

Rules:

- **Report tails, not only means.** Compare distributions (p50, p90, p99,
  max) or ECDFs. A throughput gain with a material p99 regression is not an
  automatic win.
- **Throughput is Σ tokens / Σ seconds**, not the mean of per-step rates.
- **Never sum nested timers.** If timer B runs inside timer A, A + B
  double-counts. Concurrent phases also overlap. In one agentic benchmark, the
  published per-trajectory "end-to-end" time was a sum of nested and
  concurrent phase timers, not wall time. Use one monotonic timer from
  dispatch to final result, and keep the phase timers for diagnosis.
- **Keep failures in the denominator.** Timed-out or failed samples count as
  reward 0, and they stay in latency populations, or their exclusion is
  disclosed. A request ECDF built from the 649 surviving trajectories out of
  960 made an unstable configuration look fast.
- **A COMPLETED job is not a valid run.** Check row and step counts, and that
  outputs are non-empty. A broken evaluator path produced 0 reward and looked
  like the fastest run. A frontend that returned 404s produced empty
  trajectories inside a job that exited cleanly.

## Correctness gates

A faster rollout that changes what the policy learns or how it scores is not
a speedup.

**Training rollouts** (GRPO, DAPO, GSPO; colocated or not):

- `train/token_mult_prob_error`: the mean over tokens of
  exp(|logp_generation − logp_training|); 1.0 is a perfect match.
  - Its median and the number of steps ≥ 1.04 must not get worse.
  - Spikes come from a few tokens that are tens of nats apart, common in MoE
    without router replay.
- `train/gen_kl_error` mean: unchanged.
- `train/mean_gen_tokens_per_sample`: unchanged. If it moves, the workload
  changed, not the engine.
- Validation accuracy: compare it with run-to-run noise.
  - Step-0 accuracy uses identical weights in both arms, so its difference
    estimates single-run noise. That was about ±0.025 on DAPOMathAIME2024.
- The test's own `check_metrics` gates. If the control also fails a gate, the
  failure is pre-existing; say so.

**Evaluation and agentic rollouts:**

- Reward: Wilson 95% intervals of the two arms must overlap.
  - Per-run reward noise was σ ≈ 0.016 for 20 problems × 16 samples, and a
    handful of "coin-flip" instances carried most of the variance.
  - Compare per instance when possible.
- Timeout and truncation counts.
  - Classify each timeout by phase (agent budget vs. infrastructure such as
    `Connection refused` or a startup failure).
  - Never rerun to make a timeout go away.
- Request and trajectory coverage: the same number of rows, steps and requests
  recorded.

State-cache and precision options are correctness options first. These need
the accuracy gate, not just a speed check:
- hybrid (Mamba) state caching;
- KV or weight dtype;
- prefix caching with router replay;
- CUDA-graph mode on a new engine version.

## A/B protocol

1. **Freeze the runtime.**
   - Use an immutable container; `rl.nightly.sqsh` is rebuilt daily.
   - Use one code commit, the same dataset bytes and order, the same seed, and
     the same placement class: nodes, GPUs per node, and the topology segment
     or block where the scheduler exposes one.
   - Old runs on another container or cluster are not a control.
2. **Change one factor**, or one bundle declared as a bundle. Name the profile
   by its effective values. A bundle's gain belongs to the bundle; schedule a
   split before crediting a single knob.
3. **Render the config before submitting**, and diff requested against
   effective leaves. List the unpinned defaults; they can differ between
   engine versions.
4. **Canary**: one short run that checks the arm starts, completes a step,
   passes parity on the same problems, and logs the expected effective values.
   A canary is not a timing sample. Check the population size of every run:
   in one campaign, a rollout-benchmark entrypoint ignored the step limit, so
   a "canary" ran the full workload.
5. **Formal runs**: at least 3 per arm.
   - Interleave arms, or launch them together.
   - Same-window launches share contention, which is good for paired
     comparisons but hides window effects.
   - Use `--no-requeue`, and keep every valid slow run.
6. **Analyze**: pair steps when the seed and data order match, pool runs per
   arm, report distributions and coverage, then apply the gates.
7. **Decide**:
   - Promote if the gates pass, throughput or wall time improves, and p99 does
     not regress materially.
   - Otherwise reject or park the change, and record the reason.

For NeMo-RL test-suite runs on Slurm, `tools/launch` gives each arm its own
snapshot:

```bash
export CONTAINER=<immutable.sqsh> ACCOUNT=<account> PARTITION=<partition>
export HF_HOME=<hf_home> HF_DATASETS_CACHE=<hf_datasets_cache>
CODE_SNAPSHOT_DIRNAME=code_snapshots_perf_base \
  EXTRA_SCRIPT_ARGS="logger.wandb_enabled=False" \
  tools/launch tests/test_suites/llm/<test>.sh
CODE_SNAPSHOT_DIRNAME=code_snapshots_perf_treat \
  EXTRA_SCRIPT_ARGS="logger.wandb_enabled=False <one override or declared bundle>" \
  tools/launch tests/test_suites/llm/<test>.sh
```

Separate snapshot directories keep one arm from resuming the other's
checkpoints. `NUM_RUNS` in the test script chains repetitions.

## Comparison scopes

Label every comparison with its scope, and don't claim more than the scope
supports.

| Scope | What may differ | Supported claim |
| :-- | :-- | :-- |
| Engine-only | Only the backend; the request stream, endpoint count, scheduler limits, cache state, harness and placement are fixed | Causal engine claim |
| Deployed stack | The engine's own parallelism, scheduler, cache and required adapter or router; workload and non-engine runtime are fixed | "This stack is faster on this workload" |
| Historical | Placement, runtime, storage, run count or software differ | Diagnostic only |

- Never pool runs across cluster, container, engine version, harness patch,
  placement segment, router topology, cache setting, token budget or run-count
  gate.
- A run on a different cluster is a new benchmark, not a reproduction.
- Before tuning to close a gap with another team's number, run an exact-match
  control on your own cluster.

## Provenance to record

Record these for every run:
- the container digest or wheel SHA256;
- the source commit, plus a hash of any uncommitted diff;
- the dataset hash and row order;
- the effective engine configuration (SKILL.md §3 and the per-backend "Proof" bullets in `engines.md`);
- the node list and placement;
- for agentic runs, the harness (Gym or agent framework) commit and diff hash.

An image tag alone is not provenance. A field missing from the manifest means
"unknown", not "default".

## Finding where a gap comes from

Before changing engine knobs, decompose the gap:

1. **Per step.** Which steps carry the difference? In one investigation, a
   25% rollout gap was 71% one step where a single trajectory hit its
   1,200-second timeout. Request latency matched within 2.7%. The fix was in
   the tail, not the engine.
2. **Per phase.** Generation vs. refit vs. logprob vs. training; for agents,
   model calls vs. tool calls vs. setup vs. evaluation.
3. **GPU busy vs. idle.** High idle with low engine queue depth means the
   engine is starved by the feeder (tokenization, HTTP workers, CPU-bound
   harness). Engine knobs will not help.
4. **Requests.** Trace one stalled request with a stable ID through dispatch,
   engine admission, first token and completion.
5. **Kernels.** Profile one step with Nsight Systems (`docs/nsys-profiling.md`)
   and look at launch gaps, collectives, SM occupancy and replica imbalance.
   A profiled run is never a timing sample.
6. **One-factor probes** to confirm the mechanism.
