# Checkpoint integration refresh — 2026-10-20

Local checkout: `nemo_rl_integration`; branch: `amahishi/ckpt-perf-integration`.
Gym branch: `amahishi/ckpt-perf-integration-gym`.

## Inputs

| Component | Integrated revision |
|---|---|
| RL #4762 and its prefix recovery / batched I/O / background cleanup stack | `71d3b467d31729585bbc80d302ee8fda338b4046` |
| RL #4730 completed checkpoint metrics and attempt outcomes | Feature commits through `ec5348f2aa6b519f10ece29e4189472a7adb0e44` |
| RL #4732 executor and event-loop diagnostics | Feature commits through `01a530456beca4bd7ee5e38bb290dcf7e6f7678a` |
| Gym pin required by #4762 | `25062eb18360de6dbbf3284fa5825f3b840a3f4d` |
| Gym main, including merged #3903 checkpoint telemetry and #4213 digest optimization | `1266a119a` |
| Gym #4218 capture metrics | Feature commits through `0025f42668c39ad28e411c79faa91da65e485692` |
| Resulting Gym submodule pin, including ToolSimulationAgent recovery | `a2d8d9f0559faf14a05e8d4add2a00624c0dbcbd` |
| Shared Lens dependency in RL and Gym | `4d045c03186b90afdc815db46b671f6a21bc7e98` |

The refreshed RL stack already resubmits checkpoint-refused siblings without waiting
for the original stream to finish, and advances the GenRM attempt for a full-cohort
replacement. The older integration's local implementations were preserved in backup
refs rather than applied over their newer upstream equivalents.

### Second refresh of #4762

The checkpoint telemetry helper and controller/coordinator instrumentation are
unchanged between `2b63bb69` and `71d3b467`. The updated parent recovery stack
changes eight files, mainly generation-cut handling and its tests:

- A restored prefix that cannot be fetched or validated, including missing media,
  falls back to regeneration from the turn boundary.
- Aborted calls and abandoned restored prefixes clean up obsolete staging rows
  through a dedicated executor while respecting the snapshot gate.
- An unavailable output limit is reported as a per-call cut failure before freezing.
- A failed batched PUT falls back to individual calls; freeze, acknowledgement and
  seal errors affect the corresponding call rather than losing every other cut.
- Raw chat-response/capture handling and its tool-call/logprob tests are updated.

The previous integration additions were applied with a three-way merge onto
`71d3b467`, without conflicts. The Gym pin and #4730/#4732 feature revisions are
retained. The pre-refresh RL revision is `928dd91016a1b166aec8a21f796414af965c139e`,
preserved in `backup/ckpt-perf-before-4762-refresh-20261020`.

Gym main already contains the late model-reply parking and replay-step readiness
fixes and checkpoint declarations for the configured stateless resources. The local
managed-venv readiness fix is carried forward with its regression tests.

Integration-specific resolutions preserve cancellation/abort handling around terminal
capture and extend #4730's mappings for `gym_unavailable`, Gym parked/fence hold
timings, and continued/restarted siblings. #4762's live lifecycle spans and #4730's
completed-save summary spans are separate observations; never add their durations.

## Validation from the first refresh

- Gym checkpoint, staging, capture telemetry, and shared-venv tests: **840 passed**.
- RL checkpoint metrics, runtime diagnostics, prefix lifecycle/control, source-drift,
  and setup tests: **142 passed, 5 skipped** (training/vLLM imports require the full
  dependency environment).
- RL adapter against the actual refreshed Gym checkout: **34 passed**.
- Real local CPU Ray actor trace propagation: **5 passed**. The initial sandboxed
  run could not open Ray's required Unix sockets; the isolated two-CPU run outside
  that sandbox passed and shut down its cluster.
- Changed RL Python files: Ruff, import ordering, formatting, and diff checks passed.
- Distributed GPU training and repeated save/restore have not been run by this update.

The second refresh passed 279 standalone telemetry/source-wiring tests (5 skipped)
and 5 real local Ray actor trace-propagation tests. Syntax, Ruff lint/import order,
formatting, and whitespace checks pass for the 13 integrated Python files.
Focused prefix hosting, chat-response, and cleanup tests are pending in CPU job
`1356058`, using the production container; they have not yet produced a result.
See the pipeline artifact directory
`artifacts/integration-refresh-4762-20261020` for exact outcomes and commands.

## Next runtime check

Keep the existing 12-node smoke launcher and set `NEMO_RL_ROOT` to this checkout.
Use a new experiment name, then exercise **save → restore → save → restore** with
that same experiment name. This checks that a claimed generation cut remains durable
in the next checkpoint. After that, validate a short 88-node run before the long job.

Enable RL telemetry and Gym telemetry separately. With the existing launcher's
trailing config overrides, a console-export smoke can use:

```bash
NEMO_RL_ROOT="$PWD/nemo_rl_integration" \
SITE_OVERLAY="$PWD/RLVR/nemotron-3.5-nano/scripts/.env" \
EXP_NAME=nano35-prefix-ckpt-telemetry-refresh-12n-smoke \
bash RLVR/nemotron-3.5-nano/scripts/launch_nano35_rlvr_v2_checkpoint_smoke.sh \
  ++telemetry.enabled=true \
  ++telemetry.exporter=console \
  '++telemetry.span_groups=default,checkpoint' \
  ++env.nemo_gym.telemetry.enabled=true \
  ++env.nemo_gym.telemetry.exporter=console \
  '++env.nemo_gym.telemetry.span_groups=default,checkpoint'
```

Run from the pipeline root. For the larger job, use the configured OTLP collector
instead of console export. Leave `per_prompt` and native vLLM request tracing off
unless investigating a small reproduction. Existing cleanup and sampling overrides
remain explicit per-run choices.

`USE_CONTAINER_DEPS=1` / `USE_IMAGE_GYM=1` reuse prebuilt dependencies. Updating source
and pins does **not** update those environments: check the driver, generation, and
Gym environments for compatible dependencies, including the Lens revision above,
before launching. Do not run a broad dependency sync merely to update Lens.

## Preserved history

Both repositories retain `backup/ckpt-perf-integration*-20261020` refs, including a
`*-local-20261020` ref containing the previously uncommitted fixes. The integrated
branches were rebuilt on the refreshed stack because its rebased history conflicts
with older copies of the same features. The second refresh also preserves
`928dd91016a1b166aec8a21f796414af965c139e` before republishing the integration.
No GPU training job is submitted by the rebuild.
