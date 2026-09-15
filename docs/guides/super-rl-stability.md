# Super RL stability changes on PR 3941

This branch starts at `ca06137460b7e2edcaf6f1fd67ddbda5ddd6b8e2`.
Changes are extracted from the September 12–14 Super-s25 failure ledger,
not a copy of its run directory. Each fix includes its scope and validation
boundary. No job IDs, user paths, credentials, datasets, or checkpoints are
required by the utilities. Historical validation is not validation of this
refactored branch on every cluster.

Required workloads are **CCC, CoT, TIR/ns_tools, SciCode, and equivalence
judging**. Removing a route, disabling reasoning, lowering the agreed output
budget, or converting a service failure into reward zero is not a repair.

**Status: reviewable fix series, not yet an all-cluster certified, turnkey
training recipe.** Native validation and the release blockers below are
deliberately separate from local unit-test success. Initial extraction used no
training jobs; subsequent AWS-CMH native validation is recorded below. Existing
historical experimental artifacts were not rewritten.

## Commit review order

The target repository/config/profile/user boundary and current read-only
entrypoint are documented in [Super RL launch contract](super-rl-launch.md).
Neither profile files nor a passing static check certify a runnable recipe.

Each implementation commit includes its own problem/usage/validation section in
this document. Inspect with `git show <commit>`; base is PR3941, not current main.

| Commit | Review scope | Ledger incidents |
| --- | --- | --- |
| `6ba9b8c` (already present) | Hosted DeepSeek V4 Flash provider/routes config; no Gym code patch | 2 |
| `e4a56d0` | Stdlib JSON CCC staging with authority checks and no-clobber publication | 1 |
| `b424dab` | Container cwd and persistent allocation identity | 3–4 |
| `50de47b` | Clean sbatch environment and effective service CPU affinity | 9, prevention for 12–13 |
| `abfbc77` | Exact-worker Megatron helper prebuild/read-only verification | 5 |
| `c1dea86` | Minimal sandbox exception IPC patch and native regression | 14 |
| `063c2fb` | Fail early on cross-runtime Ray replay restore | 15–19 |
| `b7f9bf1` | Preserve W&B training scalars by excluding raw generation payload | PR3941 profile / metrics |
| `681ec68` | Optional CCC concurrency config, no verifier semantic changes | 8, verifier only |
| `5f6842c` | Compact valid-row Router Replay, without experimental CP-local fetch | 9 / memory and throughput |
| `4a90f87` | Reject unimplemented effort configs rather than silently ignoring them | Configuration safety before Kimi port |
| `931268d` | Cumulative multi-call budgets and pinned Gym overlays | Whole-rollout output budget |
| `fd4a384` | Missing judge verdicts and tagged failed rows fail closed | Judge failure semantics |
| `2993312` | Image-owned Gym environment links and configured component imports | Native dependency closure |
| `937cd07` | Component-relative equivalence prompt template | Constructor-time asset lookup |
| `06176b1`, `f1e5560` | CPU service startup check and real sandbox prerequisites | Preflight dependency coverage |
| `373e99c` | Independent scheduler horizon for a short smoke | Preserve intended warmup |

## Cumulative rollout budgets

The pinned Gym agents reuse the request cap on every model call. Consequently,
setting `max_output_tokens` alone does not enforce a whole-rollout budget for
TIR or SciCode. The reviewed `gym_rollout_budget.patch` adds an optional positive
`max_total_output_tokens` agent setting, with shared accounting that rejects
missing usage and responses exceeding the actual per-call allowance. It does
not enable effort rewards or change source prompts or historical measurements.

Stage a fresh runtime Gym tree with
`uv run tools/super_rl/stage_gym.py --source <pinned-gym-repo> --output <new-directory>`.
The tool archives commit `749432dc` (not a potentially dirty working tree), applies
zero-fuzz patches, copies the versioned helpers, and records overlay hashes.
Mount that tree read-only at the configured Gym source location. The dependency
Git pin stays unchanged; overlays are reviewable in this NeMo-RL branch.

When importing a staged Gym outside the image's canonical source path, set
`NEMO_GYM_EXTRA_ROOTS` to the staged tree **before the first Gym import**.
Gym's component search reorders `sys.path`; `PYTHONPATH` alone can leave an
image-owned editable namespace component ahead of the staged copy. A diagnostic
then appears to test the overlay but executes the original resource instead.
`prepare_node.py` sets the explicit search root for each native interpreter and
checks the imported module's resolved `__file__` against the expected staged
`app.py`. Import success without source-path verification is not sufficient.

Local budget tests cover cumulative calls, exhaustion, invalid usage, per-call
over-return and the disabled default. Native all-route and effective-request
tests remain required before declaring the runtime validated.

## Judge failures are not incorrect policy answers

The validated local vLLM 0.26 DeepSeek V4 tokenizer maps `medium` to `high`;
it does not expose a distinct medium-effort mode. The local judge fragment uses
the explicit native name `high` with `thinking: true`. A native `/tokenize`
comparison confirmed identical prompt tokens for the prior medium spelling and
high on the same historical request. Earlier medium-configured local attempts
therefore used effective high reasoning; their measurements must not be
presented as a lower-effort experiment. This observation does not establish
hosted-provider semantics or parity for another vLLM image. Neither reasoning
mode nor the output budget is changed by this spelling correction.

The pinned Math and equivalence judges map missing verdicts to `False`, and Gym's
failure wrapper returns transport failures as tagged rows with placeholder reward
zero. Neither is a valid policy-training label. `gym_judge_verdict.patch` adds
`fail_on_missing_judge_verdict` (default false for backward compatibility); the
regular baseline enables it on both resources. A first valid negative verdict
remains authoritative. Missing verdicts raise `JudgeError`, and NeMo-RL rejects
tagged failed rows before token/reward processing instead of learning from zero.
This does not add unbounded retries or change native correct/incorrect labels.

Pure verdict tests and native actor-boundary regression tests are separate; real
judge responses and retry/failure behavior still require the serving smoke.

Tagged failures also need diagnostic context. The actor emits a bounded
`actor_tagged_rollout_failure` event and carries the resource-facing agent,
stable row identity, and judge error into the typed exception. It does not log
the full prompt, response, or Router Replay payload. Previously the generic
exception discarded Gym's `_ng_failure_judge_error`, making an HTTP-200 tagged
failure indistinguishable from other judge failures after collection.
Strict-verdict errors also retain response status, incomplete reason, output
token count, and final-text length, without printing the judge's answer. This
separates a capped, empty final answer from an uncapped malformed verdict.

The first integrated regular smoke exposed a separate recovery limitation:
one tagged failure interrupts the batch stream, unfinished prompt groups are
regenerated, and exhausted stream retries can release groups for gap-fill.
Thus the stream retry count alone is **not** an end-to-end retry bound.
`grpo.async_grpo.nemo_gym_fail_on_retry_exhaustion: true` makes an exhausted
stream a batch-worker failure even when some groups completed. Combined with
`max_generation_failures: 0`, the regular smoke stops on the first exhausted
stream instead of resetting its budget through gap-fill. Completed groups are
not discarded from the live buffer, but this is not durable persistence and
does not repair the underlying judge failure. The default is false to preserve
existing GRPO behavior; PPO is unchanged. Native tests cover both policies.
The two strict verifier resources also support `judge_max_attempts` (default 1,
range 1–8; regular smoke: 4). The retry decorator wraps only one
`_generate_judge_evaluation` call. It repeats the same judge request on
`JudgeError`, keeping the policy answer, prompts, 8K judge budget, temperature,
and reasoning settings unchanged. A valid negative verdict returns immediately;
it is never retried to seek a positive label. Math's required swapped-order
check remains separate, with its own bounded attempts. Other exceptions are
not retried. Final exhaustion still fails closed.

Validate retry recovery separately from judge quality and latency. In a native
local-judge canary, identical saved answers at temperature zero produced both
valid positive and valid negative verdicts on an ambiguous numerical-rounding
case. Some calls also exhausted 8K tokens without a verdict; judge-only retries
recovered some of them but took tens of minutes end to end. A protocol-valid
result therefore does not certify label consistency or acceptable training
throughput. Preserve all attempts and report cap hits, retry count, total
decision latency, and agreement on repeated fixed inputs. Do not retry valid
negative decisions, select favorable labels, or silently change gold answers,
reasoning settings, or output budgets to make the canary pass.

The attempt count is not a transport deadline. The pinned
`nemo_gym/openai_utils.py::NeMoGymAsyncOpenAI._request_with_retry` extends its
retry allowance on rate-limit-class HTTP statuses, including 429 and 503, and
the shared aiohttp client has no configured total request timeout. One wrapped
decision can therefore remain inside the transport layer without consuming the
next judge attempt. A per-decision deadline and bounded transport/backpressure
policy still require separate implementation and native outage tests. The
allocation's wall-clock limit is a resource bound, not a substitute for that fix.

This reuses an answer **in memory**, not across preemption or a new job. Durable
judge-only retry remains unimplemented. Do not interpret a healthy judge endpoint or growing rollout counts
as evidence that optimizer updates are progressing. Diagnose repeated tagged
failures on a small verifier workload before restarting full training.

The equivalence prompt path is relative to the component's working directory,
`resources_servers/equivalence_llm_judge`, as selected by Gym's `RunHelper`.
The regular recipe uses `prompt_templates/equivalence_llm_judge.txt`; prepending
the Gym root instead points at a nonexistent file. Import-only checks do not
catch this: the template is opened by the server constructor. Require actual
service startup in the native runtime before allocating training GPUs.

After `prepare_node.py` links and checks the image-owned environments, run
`tools/super_rl/check_gym_startup.py` with the full `--config`, mounted `--gym`,
fresh `--runtime` output directory, and explicit `--cpus` / `--timeout` limits.
Start required sandbox backends in the same allocation first, with the same
shared-directory mounts and endpoint settings used for training. CCC's startup
hook executes a sandbox readiness command before loading its metadata; it does
not merely read a file. A missing sandbox leaves that resource waiting and then
fails startup. Do not stub this dependency or remove CCC from the check.
It starts the configured Gym apps on a local CPU-only Ray runtime, waits for
their health checks, then shuts them down. This catches constructor-time asset
errors that imports miss. Backend-model readiness is deliberately disabled only
in this CPU test's in-memory configuration. Native readiness probes may execute
trivial sandbox commands; the check sends no policy/judge inference requests and
does not certify task verification, rewards, or training. The real training recipe
and its backend-readiness gate are not modified.

For the three-update regular smoke, the constant-LR scheduler horizon is set to
`lr_warmup_iters + 1`, independently of `grpo.max_num_steps`. Leaving it null
makes Bridge infer a three-update decay horizon and silently shorten ten-update
warmup (with a warning). The explicit horizon preserves the intended warmup;
it does not authorize additional training updates. Review the scheduler and
optimizer state again when choosing a production horizon.

## Excluded experimental artifacts

- Old job IDs, absolute user paths, submission receipts, logs, payload snapshots,
  repeated `v1/v2/v3` wrappers, observer scripts and cluster allocation manifests.
- TIR-to-CoT conversion and environment deletion scripts. All required routes
  remain required; SciCode is not removed.
- Ad-hoc pytest frontend borrowing, `sacct -P` string assertions, container
  `/proc/environ` identity probes, and host ACL utilities (incidents 6–7, 10–13).
  These were diagnostic-tool/placement problems, not reasons to patch the
  learner. Run tests in the correct environment with writable test output; apply
  shared-output permissions on a scheduled host with ACL tools and check every
  ancestor's traversal permissions after the final writer exits.
- Existing PR3941 checkpoint finalization and trained-frontier logic: retained
  in place, not duplicated as patch files or hardcoded recovery scripts.
- Unvalidated CP-local route fetch, 256/256 role rebalance and W&B buffering
  candidates. They need separate correctness/performance review.

## CCC compilation and grader exit status

The pinned CCC evaluator inferred compilation success from empty stderr. This
rejects valid programs with compiler warnings and accepts a failed compiler
that produces no stderr. It also parsed partial grader stdout as a score even
when the sandbox reported a non-successful execution.

`gym_ccc_execution.patch` uses the sandbox's `process_status == "completed"`
contract for both stages. The shell sandbox derives this status from the process
exit code. Diagnostics retain compile/run status, stdout, and stderr; a failed
compile never proceeds to grading, and a failed grader cannot award a score
from partial stdout. Successful zero and fractional scores are unchanged.
Stage a new Gym tree with `stage_gym.py`; never patch an active runtime.

`tests/unit/tools/test_ccc_execution.py` applies the overlay to the pinned Gym
source with zero fuzz and exercises its actual `run_test_case` function using
controlled transport replies. It skips explicitly if the Gym Git object is
unavailable. Native HTTP/sandbox known-answer controls and reasoning-on model
generation remain separate validation gates. This change does not address
precompile-cache races, verifier throughput, or classification/retry of sandbox
infrastructure failures; the pinned outer verifier still catches exceptions.

## CCC metadata staging: no driver dependency on orjson

**Incident 1:** a preparation driver lacked `orjson`; staging exited before
writing verifier data. Moving the job to Gym Python worked historically but
was unnecessary for a JSON-only transformation.

`tools/super_rl/stage_ccc.py` uses only stdlib `json`. It streams source
JSONLs, verifies source SHA-256 and each authorized problem's hash, retains
native tests/graders/subtasks, and atomically publishes without overwriting.
It does not edit training prompts, routes, budgets, or SciCode data.

Supply a manifest with this shape (use real independently verified hashes):

```json
{
  "sources": [{"path": "chunk.jsonl", "sha256": "<source-byte-sha256>"}],
  "problems": [{"competition_id": "c", "problem_id": "p", "sha256": "<problem-sha256>"}]
}
```

Problem hashing uses sorted-key compact UTF-8 JSON. A mismatch with a historical
serializer **fails**; do not regenerate authority hashes just to make it pass.
Paths resolve relative to the manifest. Create the output parent first, then
run in a CPU allocation using the preparation environment:

```bash
uv run tools/super_rl/stage_ccc.py --manifest /data/ccc-manifest.json --output /run/data/ccc.jsonl
```

Validation: `tests/unit/tools/test_stage_ccc.py` covers exact problem content,
hash failures, missing/duplicate identities, strict JSON, and no-clobber output.
Real CCC resources-server loading and sandbox execution remain native smoke
gates; a JSON test does not certify verifier throughput.

## Container cwd and allocation identity

**Incidents 3–4:** the host submit directory was not mounted at the same path
inside Pyxis; an unrelated probe also expected `SLURM_JOB_ID` after Ray had
intentionally removed Slurm/MPI variables.

`ray.sub` now accepts `NRL_CONTAINER_WORKDIR`, an absolute directory in the
container's mount namespace. The default remains `SLURM_SUBMIT_DIR` for existing
launchers. Set it explicitly when a Lustre alias and its physical path differ:

```bash
export NRL_CONTAINER_WORKDIR=/run/experiment
export MOUNTS=/physical/run:/run/experiment,/physical/code:/opt/nemo-rl:ro
```

Ray head/worker launch arguments preserve this path as one argument, and attach
helpers use it too. The directory must actually exist in the mounted container;
host existence is not sufficient. `NRL_SLURM_JOB_ID` and
`NRL_SLURM_SUBMIT_DIR` preserve allocation identity for the driver and probes.
Do not restore the entire `SLURM_*` environment inside Ray.

Validation: executable shell-fragment tests in `test_ray_sub_contract.py`, plus
`bash -n ray.sub`. Native Pyxis cwd/mount validation remains required on each
cluster. No compute-node hardware change is involved.

## Submission environment and effective CPU affinity

**Incident 9:** submitting from a two-CPU diagnostic step leaked its CPU mask
and Slurm/MPI state into the new allocation. Ray, Gym and sandbox were confined
to CPUs 0 and 5 despite a much larger allocation.

`tools/super_rl/submit.py` constructs a fresh environment. It retains basic
identity/locale variables and only explicitly requested workload variables;
Slurm/PMI/step variables cannot be added through `--env`. Supply a site-specific
client configuration with `--slurm-conf` if required. It is never hardcoded.

```bash
uv run tools/super_rl/submit.py --env CONTAINER --env MOUNTS --env COMMAND \
  --env NRL_CONTAINER_WORKDIR --env CPUS_PER_WORKER --env GPUS_PER_NODE \
  -- --account=<account> --partition=<partition> --nodes=<nodes> ray.sub
```

By default this only runs `sbatch --test-only`. Add `--submit` **before** `--`
to submit after reviewing the request. Explicitly pass all required runtime
variables, including sandbox settings; preferably have a versioned batch wrapper
set them. Store credentials in private mounted files, not command-line values.
There is no automatic retry: reconcile a failed or ambiguous submission first.

`ray.sub` uses `--overlap --cpu-bind=cores` for Ray and explicitly requests the
same `CPUS_PER_WORKER` for sandbox tasks. All three startup paths validate their
actual procfs CPU affinity before services start. Configure CPUs from the
allocation's scheduler contract, not the physical CPU count; e.g. the historic
CMH allocation granted 140, not 144. No universal 140-CPU default is added.

Validation: poisoned-environment and CLI tests; the affinity guard is executed
against this host's real mask with passing/failing limits. Native multi-node
Slurm/Pyxis service affinity must still be checked. Host cgroup/UID attribution
belongs on the host, not in container `/proc/<pid>/environ` probes (incidents
12–13). Do not infer every child process's affinity from a startup check alone.

## Megatron helper compilation with immutable source

**Incident 5:** `compile_helpers()` ran `make` in a read-only datasets package.
The image also lacked `python3-config`, so the extension ABI suffix was empty.

`tools/super_rl/build_mcore_helpers.py` creates a fresh copy of the datasets
package. It changes only Makefile Python probes, uses the exact invoking worker
Python and `sysconfig` for the extension suffix, forces compilation, exercises
int32/int64 sample-index functions, and writes a checksum manifest only on
success. Failed output directories are left for inspection, never reused.

Run `build --source <installed-datasets-package> --output <new-runtime-package>`
with the **exact Megatron worker interpreter inside its image**, in a CPU
allocation. Do not use the lightweight staging driver's environment. NumPy,
pybind11, make, and a C++ compiler must be available there. The Python entrypoint
is the same script for ARM and x86; the compiled artifact is **not portable**
between architectures, Python ABIs, images, or different worker paths.

Then mount the runtime package read-only over the installed datasets package
and run `verify --package <installed-datasets-package>` with that same worker
interpreter. Verification requires a real read-only mount, matching hashes and
runtime identity, correct import paths, real `utils.compile_helpers()`, and
functional helper checks. Use this verification again before model startup.
Build each worker/image/architecture combination separately; do not copy an ARM
binary into H100's x86 environment.

Local tests cover Makefile transformation, no-clobber/source isolation, and
ABI/read-only rejection. This refactored script has **not** been compiled in
the native ARM/x86 images in this change; the historical ARM implementation was
validated, which is narrower evidence. No Megatron submodule pointer is changed.

## Sandbox exception IPC: retain TIR, fix the protocol

**Incident 14:** `shell_worker` put an exception instance in `has_error`. In
the production sandbox's Python 3.10.20 / requests 2.28.1 environment,
`JSONDecodeError` failed to reconstruct across the multiprocessing Pipe. This
is a sandbox protocol bug, not an invalid model answer or a bad GPU node.

`tools/super_rl/patches/sandbox_exception_ipc.patch` changes that field to a
boolean. Traceback/stdout/stderr, network restrictions, timeout behavior and
session state remain unchanged. Because the affected source belongs to the
**separate sandbox image**, this branch carries the minimal build-time patch,
not a vendored 500-line server or a runtime monkeypatch of unrelated Gym code.

Apply with zero fuzz to a fresh copy of that image's `/app/main.py`, or fix the
corresponding upstream image source and rebuild. Validate the patch with
`patch --dry-run --batch --forward --fuzz=0 <copied-main.py> <patch-file>` first.
Mount the patched copy read-only at `/app/main.py` using
`SANDBOX_EXTRA_MOUNTS`, or pin the rebuilt image. Never patch a running job's
shared file. A different source revision requires review, not relaxed matching.

Set `NRL_SANDBOX_MODULE` to the actual patched module and run
`tests/unit/tools/test_sandbox_ipc.py` **inside the sandbox interpreter**. The
native test invokes real `shell_worker`/Pipe, checking state creation,
JSONDecodeError, ValueError, SyntaxError, and state after errors. The local
patch test alone is not native validation; the native test explicitly skips
outside that environment. All-route TIR concurrency/session smoke is still
required. The previous CoT-only 18-step run does not certify it, and no TIR to
CoT data converter is included here.

## Preemption: fail early on nonportable Ray replay

**Incidents 15–19:** checkpoint recovery worked, but every new allocation
required discarding replay with old Ray route references. A requeue with the
same Slurm job ID is still a new Ray runtime. The generic replay restore path
did not enforce this transport constraint itself.

GRPO now rejects a Ray-reference checkpoint resume unless
`checkpointing.load_replay_buffer=false` is explicit. The guard runs during
setup before state loading/worker allocation, and again before replay payload
loading. Inline transport and fresh training are unchanged. This is prevention,
not a claim that all four preemptions were caused by a replay bug.

Restore model, optimizer, scheduler, dataloader and the trained frontier from
the same complete checkpoint. Keep an ordered dataset and validate the actual
frontier; do not infer it solely from `step * prompts_per_step` when dynamic
sampling is enabled. Preserve unfinished sample artifacts in attempt-specific
directories before regenerating. Keep one writer per checkpoint/W&B lineage.
The existing PR3941 finalization/frontier machinery is retained, not replaced
by the experiment's hardcoded recovery scripts.

Validation: offline guard truth-table and source-wiring tests; existing native
GRPO restore tests are extended but local collection is blocked by missing
`ray` in the lightweight development environment. Full distributed reload remains a
native gate. **Portable rollout persistence is not implemented:** it needs real
tokens/logprobs/masks/rewards/routes, weight lineage/age validation, atomic group
completion and checkpoint-aligned consumption records, plus whole-Ray-cluster
kill/restart tests. Increasing retries or enabling replay loading is not that fix.

## W&B: protect scalar updates from raw generation payloads

The PR3941 profile recorded a 16.2 MB `generation_logger_metrics` payload that
caused W&B to reject the whole update, including loss, KL and reward. This is
distinct from the ledger's remaining GPU tick/heartbeat issues.

Both GRPO loops now exclude only that raw nested key at the generic training
logging call. Generation/performance summaries are computed first; the original
metrics dictionary is not mutated. No rewards, logprobs, or training masks are
changed, and no W&B data is rewritten.

Tests exercise an oversized payload and preservation of every other metric,
plus source wiring after summary generation. Actual W&B delivery still needs
a native canary. This does **not** fix `commit=False` GPU sample merging,
cross-resume relative time axes, or heartbeat state; those remain open.

## CCC verifier concurrency is a configuration choice

**Incident 8, verifier component only:** increasing `test_batch_size` from 4
to 32 reduced a fixed-answer benchmark from 44.27 s to 6.79 s with identical
rewards and 164 executed tests. This was not a 6.5x end-to-end training speedup,
and it did not fix that incident's separate policy NaN.

Merge `training_configs/super_rl/ccc_verifier_concurrency.yaml` **after** a
complete recipe using NeMo-RL's `defaults` list. It changes only this field on
the existing `competitive_coding_challenges_resources_server` alias. Use the
corresponding override for a differently named alias. Do not put this fragment
in Gym's `config_paths` or treat it as a complete environment definition.

The merge test preserves CCC parameters, CoT, TIR, SciCode, equivalence, model,
data and output cap. Before adopting 32 on another cluster, check allocated
CPU capacity and the same `shared_dir` mount inside both resource server and
sandbox. Compare fixed answers, rewards and actual test counts at both
concurrency values. Keep sandbox timeouts/reward semantics unchanged.

## Compact Router Replay: avoid rectangular CPU materialization

**Incident 9 / throughput section:** PR3941 already deferred route H2D until
packing, but still materialized `[batch, max_sequence, layers, topk]` on CPU.
The historical compact implementation eliminated that rectangular intermediate.
This commit retains that working algorithm, without S25-specific function
attributes/log markers or the unvalidated CP-local-fetch candidate.

`materialize_routed_experts_ref_rows` reads valid rows (full objects or grouped
multi-turn ranges), validates tags/shapes/dtypes, and passes jagged rows to
Megatron packing. The existing padded token boundaries and per-sequence CP
selection are reused, including the separate model-owned CP index path.
Dense/inline input behavior is retained. Forward tracing intentionally retains
the dense debug path; for the compact production path set:

```bash
export NRL_R3_TRACE=0 NRL_R3_TRACE_VERIFY_FORWARD=0
export NRL_ROUTER_REPLAY_VALIDATE=1
```

These values must reach the **actual Megatron workers**, not just the submit
shell. Keep profile-required MTP exclusions unchanged. Compact still fetches
valid rows before CP slicing and constructs a full packed CPU tensor; it is
**not** a throughput solution or a durable rollout archive.

Regression tests cover multi-turn rows, range-scatter parity, wrong dtypes and
CP1/2/4/16 equivalence to the existing dense path on every CP rank. They require
the native Torch/Ray/Megatron environment; local collection is blocked by the
missing dependencies. Historical native tests and three-step canaries validated
the source algorithm, not this newly refactored branch. A new native canary
must cover ragged inputs, model-owned CP, update/refit and checkpoint reload.

## Required service contract before promotion

Preserve source prompts, task order, effort labels and the agreed cumulative
output budget. Inspect **effective requests**, not only row metadata. Keep
reasoning enabled; final-code extraction and verifier failures must be fixed
without suppressing the reasoning channel.

| Workload | Required native evidence |
| --- | --- |
| CCC | Authorized metadata loaded; final code reaches compiler; identical shared mount; nonzero actual test counts and correct reward |
| CoT | Original prompt/answer unchanged; math/other intended resource returns a valid reward; reasoning stays on |
| TIR / ns_tools | Real sandbox persistent sessions, multiple tool turns, tool errors followed by continued state, concurrent sessions, cumulative budget and correct final extraction |
| SciCode | Correct HDF5 test data and prompt assets; multi-substep execution; actual tests and rewards; cumulative generation budget |
| equivalence judge | Correct resource/agent/model aliases and prompt; valid positive **and negative** verdicts; missing verdict/transport error cannot masquerade as reward zero |

Keep `grpo.async_grpo.max_trajectory_age_steps=2`. A production-concurrency
canary must allow at least three optimizer targets/steps; a one-step smoke
truncates the async target window and does not cover the age-2 workload.
Retain checkpoint save period 10, FT period 1 and one latest FT unless a reviewed
recipe overrides them; derive the safe-save deadline from that partition's wall
time, not from another cluster. Use
`env.nemo_gym.global_aiohttp_connector_limit_per_host=16384` for the reviewed
Blackwell profile and remove inherited explicit total-limit overrides.

Resolve account, QoS, CPUs, node/GPU shape, mounts, interpreter and image per
cluster. CMH/HSG have four-GPU Blackwell nodes; the H100 adapter must use its
actual node shape and rederive parallelism. These fixes do not make an ARM
container runnable on x86 or validate one universal topology.

Promotion needs all routes, real optimizer updates/refit, complete checkpoint
save/reload and delivered W&B scalars. Also agree on steady-state step time and
GPU-hours per valid sample before scaling: three successful updates establish
correctness, not acceptable throughput.

## Open issues and release blockers

The maintenance branch's `GRPOConfig` now rejects an enabled or malformed
`reasoning_effort` block before initialization. PR3941's `extra="allow"`
would otherwise accept the block without implementing its reward shaping.
An absent, null, or explicitly disabled block leaves non-effort runs unchanged.
This guard is **not** the Kimi implementation: replace it with the reviewed
typed schema and actual integrations in the implementation commit, together
with end-to-end budget/reward tests. Do not disable effort to bypass the guard
for an effort experiment. Native regression tests live in `test_grpo.py`;
local collection still requires the missing Ray/Torch stack.

| Item | Why not marked solved | Next code/validation boundary |
| --- | --- | --- |
| Complete pinned dependency closure | Exact Gym, Bridge, nested Megatron-LM and Automodel revisions were staged and checked in the AWS-CMH native runtime. That does not establish fresh upstream access or image compatibility everywhere. | Reproduce immutable source/image staging on each destination; preserve shallow Git boundaries when transferring partial histories. Do not replace pins or vendor mutable run copies. |
| Kimi effort / multi-turn budget integration | Cumulative Gym budget overlays are implemented; Kimi reward shaping is not. The enabled-effort guard remains deliberate. | Port and review the typed effort schema and GRPO reward integration with end-to-end effective-request and reward tests. Ordinary-RL smoke is not Kimi certification. |
| Judge missing-verdict / transport-failure contract | Strict verdict parsing and tagged-row rejection are implemented. Native boundary tests passed, but they are not end-to-end outage/load tests. | Preserve valid negative verdicts, bound retries and backpressure, and inject real transport/replica failures before production. Never turn service failure into reward zero. |
| CCC compiler warnings / shared precompile cache | The pinned `resources_servers/competitive_coding_challenges/ccc_eval.py` still infers compilation success from an empty stderr and publishes a precompile cache without synchronization. A warning can be mistaken for a compile failure; concurrent first use can race in the same problem/PID directory. This is source-review evidence, not proof that a particular smoke row was affected. | Reproduce warning-only successful compilation, real compiler failure, and concurrent same-problem initialization in the actual sandbox; use process status and safely publish a complete cache. Keep policy prompts and actual test outcomes unchanged. |
| Hosted judge outage / local HA service | The provider's auth DB exhaustion and empty 500s were external; a recovered canary did not fix service capacity. No local judge fleet is provisioned by this branch. | Prefer a dedicated self-hosted service for the requested production contract, with model/prompt/reasoning parity, independent replicas, bounded backpressure/retry, long-request load tests and replica-loss injection. Never silently change judge model or reward on failover. |
| Policy NaN in incident 8 | Exact nonfinite field and numerical root cause were not captured in that event. Later non-reproduction is not a fix. | vLLM HTTP serialization / `vllm_worker_async.py`: add bounded, private field/request diagnostics, preserve the original exception, then reproduce with weight lineage. No `nan_to_num`. The job-bound snapshot logger is not copied. |
| Sparse generation/learner logprob spikes | Threshold-2 filtering remained; root cause is unknown. | Fixed-input comparison of generation/logprob token alignment, masks, weight versions and routes. Keep existing filtering/penalties; do not raise the threshold to hide the issue. |
| Learner / route-fetch throughput | Compact removes a memory intermediate, but does not eliminate remote reads, full packed CPU copies or collective waits. | Timeline on identical inputs; separate route transport, packing, forward/backward and collectives; then independently test CP-local reads and role split. No claimed end-to-end speedup here. |
| Portable rollouts across preemption | Current persistence can contain old runtime references. | Durable actual route/token/reward data, atomic groups and consumption frontier; kill/restart the entire Ray cluster and verify no duplicate/omitted groups. |
| W&B continuous GPU samples and resume axis | Raw payload filtering does not fix buffered GPU ticks or heartbeat state. | `nemo_rl/utils/logger.py` and GPU monitor: preserve independent samples and a monotonic cross-resume axis; test online/offline/resume/failure behavior before adoption. |
| Cross-cluster and complete TIR/SciCode certification | Historical 18-step evidence used the CoT fallback. New AWS-CMH native service startup passed with all required routes, but this is not all-route model/task execution. | Native ARM/x86 helper builds, all-route stateful canary, topology-faithful compact tests and distributed restore on each supported profile. |

## Validation record for this series

Initial extraction's local tests: **70 passed, 1 skipped**. The skipped test is the actual sandbox
worker regression, which requires `NRL_SANDBOX_MODULE` and sandbox-native
dependencies. The following selected tests were run with plugin autoload off
and `--noconftest -p no:cacheprovider`; these tests use only their own/local
fixtures, not the global Ray/GPU fixture:

```text
tests/unit/tools/test_stage_ccc.py
tests/unit/tools/test_ray_sub_contract.py
tests/unit/tools/test_clean_submission.py
tests/unit/tools/test_build_mcore_helpers.py
tests/unit/tools/test_sandbox_ipc.py
tests/unit/tools/test_replay_checkpoint_contract.py
tests/unit/tools/test_training_metrics_payload.py
tests/unit/tools/test_ccc_concurrency_config.py
tests/unit/environments/test_hosted_judge_configs.py
tests/unit/algorithms/test_metric_utils.py
```

Native test attempts were **blocked at collection**, not passed or silently
skipped: `test_grpo.py` lacked Ray; compact tests first lacked Torch. Run these
in the pinned worker environment with the intended native fixtures. Source
wiring checks are not model execution. Local validation also includes Ruff,
Python syntax compilation and `bash -n ray.sub`; that initial test batch used no
live Slurm/provider/W&B requests or GPU jobs. Pyrefly is not installed locally; new standalone
modules are included in its allow-list for native development/CI checking.

Follow-up AWS-CMH ARM validation used training source `373e99c` and startup
checker `f1e5560`, with receipts in
[SCI2-118](https://linear.app/nvidia/issue/SCI2-118). Exact-worker helper ABI and
read-only `make` verification passed, along with eight configured component
imports, all twelve real Gym service health checks, complete CCC metadata
loading, native recipe parsing and two tagged-judge-failure boundary tests.
The initial full-service check failed because its wrapper omitted the sandbox;
the successful repeat used the real sidecar and preserved the failed logs.
CCC's native shell readiness probe also succeeded. These checks do not certify
policy/judge inference, task rewards, optimizer updates, refit or checkpoint
reload. The separately verified local judge serving canary does not substitute
for integrated training or cross-cluster validation.

The historical failure ledger and measurements remain unchanged. This document
is self-contained for code review; access to private cluster artifacts is not
implied by possession of the branch.
