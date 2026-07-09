# Streaming Tool-Call Experiment Bug Report

**Status:** Open items; STC-004 cache prewarm is validated and awaits full-workload validation<br>
**Scope:** Historical streaming tool-call validations recorded under
`results/streaming_tool_call_verified/` and the evidence summarized in
`docs/design-docs/streaming-tool-call.md`. This is an engineering and
experiment-integrity report; it does not reinterpret the historical reward or
accuracy results.

## Executive summary

The prior runs establish useful mechanism evidence, but they do not yet form a
single clean streaming-on/off performance or accuracy comparison. The blocking
issues are:

1. strict trajectory collection can silently stop short of the requested
   manifest size;
2. the audit calls arbitrary arms `streaming_off` and `streaming_on`, even when
   both have streaming enabled;
3. external `raw.githubusercontent.com` DNS failures are asymmetric between
   arms; and
4. a normal request can exceed the model context by one token and become an
   empty assistant result; and
5. two July 9 exact-tokenizer on arms used the bridge timeout as a coroutine
   wait limit but omitted it from the underlying action-server HTTP request.

Item 5 is fixed and regression-tested, but it invalidates those two historical
on arms for performance and accuracy comparison.

The retry overlay is useful diagnosis, but must not be presented as a clean,
atomic paired accuracy measurement. See [Non-bugs and protocol constraints](#non-bugs-and-protocol-constraints)
for phenomena that should be controlled or labeled rather than fixed in the
runtime.

## Severity and exit criteria

| Severity | Meaning |
| --- | --- |
| P0 | Blocks a final streaming accuracy/performance conclusion. |
| P1 | Must be addressed before enabling the feature by default or publishing a broad benchmark claim. |
| P2 | Cleanup, observability, or regression coverage needed; does not invalidate already persisted trajectories. |

## Streaming-off applicability matrix

`Yes` means the issue is present in, or is unconditionally applicable to, a
streaming-off run. `No` means the triggering mechanism requires active
streaming sessions. `Unknown` is deliberately not treated as `No`.

| ID | Streaming-off status | Evidence and interpretation |
| --- | --- | --- |
| STC-001 | **Yes — shared launcher** | The collection-size defaults are independent of `STREAMING_TOOL_CALL`. The historical partial incident was a both-streaming-on poll pair, but an off trajectory-collection launch uses the same code path. |
| STC-003 | **Conditional** | Labels are correct for a true off/on comparison, but misleading for off/off, on/on, or poll100/poll050 comparisons. The historical off-repeat comparison is therefore also affected. |
| STC-004 | **Yes — confirmed; cache mitigation validated** | Off drivers `13293269`, `13304353`, and `13342775` all log `raw.githubusercontent.com` DNS failures while `streaming_tool_call.enabled=False`. The new cache and strict-offline path is shared by both arms. |
| STC-005 | **Unknown — not confirmed in off** | The observed 131,073-token overflow came from a streaming-enabled full workload. It uses the normal final chat endpoint and may be reachable without streaming, but archived off logs do not establish that. |
| STC-006 | **Yes — audit-only** | Retry overlay is independent of the feature flag and can replace rows for either arm. |
| STC-007 | **Not observed in archived off logs** | The three inspected off drivers do not contain the `server_thread` teardown error. This is not proof of absence; it remains unconfirmed without a targeted off teardown test. |
| STC-008 | **No — feature-specific** | The failed path required active FastAPI/Uvicorn-owned streaming sessions to be cancelled from the Ray refit loop. Disabled streaming creates no such session. |
| STC-009 | **No — feature-specific** | `_send_action_server_request()` delegates to the new async bridge only when `streaming_tool_call.enabled=True`; the disabled branch calls the unchanged parent request path. The two affected exact-tokenizer on arms contain the timeout marker in 28/64 and 29/64 rows, while their paired off arms contain zero matching rows. |

### Streaming-off evidence

The following archived drivers explicitly have
`streaming_tool_call.enabled=False` and still show the external setup failure:

```text
# 13293269 (500-instance off arm), lines 674-675
urllib3.exceptions.NameResolutionError: HTTPSConnection(host='raw.githubusercontent.com', ...)

# 13304353 (500-instance off arm), line 1041
Retrying ... NameResolutionError("HTTPSConnection(host='raw.githubusercontent.com', ...)")

# 13342775 (474-instance off arm), line 1061
Retrying ... NameResolutionError("HTTPSConnection(host='raw.githubusercontent.com', ...)")
```

No additional broad streaming-off rollout is required to establish STC-001
through STC-004, STC-006, or STC-008. A generic off rollout would not settle
STC-005 because it might never reach the context boundary. If confirmation is
needed, run a targeted boundary test that constructs a 131,072-token final
chat prompt with streaming disabled. STC-007 can be settled by a small off
teardown smoke that starts and shuts down vLLM, but it is P2 and does not block
the correctness work.

## Confirmed bugs

### STC-001 — Strict collection can emit fewer rows than the requested manifest

**Severity:** P0<br>
**Status:** Open<br>
**Affected artifacts:** `13359969` (poll100), `13359970` (poll050)

The pair launcher has independent defaults for expected manifest size and
trajectory collection size:

```bash
# examples/swe_bench/run_streaming_tool_call_verified_pair.sh
TRAJECTORY_COLLECTION_BATCH_SIZE="${TRAJECTORY_COLLECTION_BATCH_SIZE:-128}"
EXPECTED_COUNT="${EXPECTED_COUNT:-500}"
```

This allowed the run directory named `20260702T230000Z-admission-poll256` to
consume the 256-row manifest while setting
`env.nemo_gym.trajectory_collection_batch_size=128`.

Evidence:

```text
# 13359969 and 13359970 driver logs, line 13
+data.train.data_path=...swebench_verified_no_timeout_observed_256.jsonl
env.nemo_gym.trajectory_collection_batch_size=128

# Persisted result counts
128 .../poll100/exp_001/trajectory_collection.jsonl
128 .../poll050/exp_001/trajectory_collection.jsonl
```

The pair is therefore invalid as a 256-sample experiment. The replacement pair
`13385088` / `13385093` explicitly set batch size 256 and both logged
`Collecting rollouts: 100%|...| 256/256`.

**Required fix:** In strict paired mode, reject submission unless
`TRAJECTORY_COLLECTION_BATCH_SIZE == EXPECTED_COUNT`. If partial collection is
intentional, require an explicit `ALLOW_PARTIAL_COLLECTION=1` and record the
actual collected count in the report metadata.

**Acceptance test:** A dry run with `EXPECTED_COUNT=256` and batch size 128
must exit nonzero. A valid 256 run must emit exactly 256 distinct instance IDs
per arm before comparison begins.

### STC-003 — Audit arm names misrepresent the 100 ms / 50 ms experiment

**Severity:** P0<br>
**Status:** Open

`verified_trajectory_audit.py` names its two generic inputs
`streaming_off` and `streaming_on`. For the admission experiment, both arms
enabled streaming; only `snapshot_poll_interval_seconds` differed.

Evidence:

```text
# 13385088 driver log, line 13
policy.generation.vllm_cfg.streaming_tool_call.enabled=True
...snapshot_poll_interval_seconds=0.1

# 13385093 driver log, line 13
policy.generation.vllm_cfg.streaming_tool_call.enabled=True
...snapshot_poll_interval_seconds=0.05
```

The generated `poll100_vs_poll050_*.json` reports nevertheless use the generic
`streaming_off` / `streaming_on` keys. This is a reporting correctness defect:
readers can incorrectly interpret the comparison as a feature on/off result.

**Required fix:** Add required neutral arm metadata to the audit CLI, for
example `--left-label poll100 --right-label poll050`, and record both arms'
streaming flag, polling interval, manifest digest, model, temperature, top-p,
replica count, and collection count.

**Acceptance test:** A poll comparison report contains `poll100` and `poll050`
keys and explicitly states that streaming is enabled in both arms.

### STC-004 — External dependency setup creates asymmetric infrastructure failures

**Severity:** P0<br>
**Status:** Cache prewarm validated; full-workload validation pending

SWE agent setup downloads repository requirements from
`raw.githubusercontent.com`. DNS failure affects individual instances at
different rates in each independent arm.

Evidence:

```text
# 13395097 driver log, line 1161
NameResolutionError("HTTPSConnection(host='raw.githubusercontent.com', port=443):
Failed to resolve 'raw.githubusercontent.com' ...")

# 13395098 driver log, line 753
NameResolutionError("HTTPSConnection(host='raw.githubusercontent.com', port=443):
Failed to resolve 'raw.githubusercontent.com' ...")
```

The complete 256-row poll pair reported 19 infrastructure-error instances for
poll100 and 24 for poll050. A 27-instance retry overlay reduced those counts to
15 and 19, respectively, but did not eliminate the confound.

**Required fix:** Make evaluation setup self-contained before collecting any
trajectory: prefetch/cache requirements in a shared immutable location or bake
them into the sandbox image. If setup remains network-backed, classify the
instance before model generation and require both arms to retry the same failed
instance set until setup succeeds or is excluded by a predeclared rule.

**Acceptance test:** A paired accuracy run has zero setup/network failures, or
its final report is automatically marked `inconclusive` and excludes accuracy
ranking.

#### Implemented mitigation

The SWE-bench harness now caches every fetched `requirements.txt`, recursive
requirements file, and `environment.yml` under
`SWE_BENCH_ARTIFACT_CACHE_DIR`. Cache writes are atomic, so concurrent
evaluators can share the Lustre-backed directory. With
`SWE_BENCH_ARTIFACT_CACHE_OFFLINE=1`, a cache miss fails immediately with the
repo, commit, and candidate path instead of retrying
`raw.githubusercontent.com` during evaluation.

The runtime patch is
`responses_api_agents/swe_agents/patches/swebench_artifact_cache.patch`.
Its five harness unit tests are kept in the separate
`swebench_artifact_cache_tests.patch` asset, so production setup checkouts do
not acquire test-only files and existing cached checkouts remain patch-idempotent.

`examples/swe_bench/prewarm_swebench_artifacts.sh <manifest.jsonl>` runs on one
`nemotron_sw_post` / `interactive` node, creates or refreshes a dedicated
SWE-bench prefetch setup, fills the shared cache, and immediately verifies it
offline. The Verified pair launcher enables this prewarm step by default, then
passes strict offline mode to both arms. Its cache is intentionally outside the
SWE-bench venv setup directory, so an incomplete setup can be rebuilt without
discarding already-fetched artifacts.

Targeted validation covers cache hits without network access,
download-and-reuse, strict offline cache misses, DNS/connection error guidance,
and `environment.yml` cache hits (five SWE-bench unit tests), plus the Gym
setup/patch/command path (eleven processor and command-construction tests).
On 2026-07-06, the 474-row no-timeout Verified manifest prewarmed 24 unique
dependency environments (191 rows require no fetched dependency file), then
passed a second cache-hit prewarm and strict-offline verification. A 474-sample
pair must still complete with prewarm plus strict offline mode before STC-004
is marked fully resolved.

### STC-005 — Input can exceed the model context by one token

**Severity:** P0<br>
**Status:** Open

Normal `/v1/chat/completions` requests reached 131,073 input tokens for a
131,072-token model. The extra chat-template/EOS suffix arrives after a full
preserved prefix. Gym converts the 400 response into an empty assistant result,
which is a correctness and quality loss rather than a harmless timeout.

Evidence:

```text
# docs/design-docs/streaming-tool-call.md, lines 892-897
Seven normal /v1/chat/completions requests reached 131,073 input tokens for a
131,072-token model ... Gym converts the 400 response to an empty assistant
result.
```

**Required fix:** Reserve chat-template/EOS budget before appending preserved
context, or apply a deterministic, token-aware compaction policy before the
normal final request. Do not silently substitute an empty assistant result for
an over-context request.

**Acceptance test:** A boundary test with a maximal preserved prefix leaves
space for all final-template tokens and either produces a valid response or a
typed, visible budget-exhaustion outcome.

### STC-006 — Retry overlay is valid diagnostics but unsafe as a primary comparison result

**Severity:** P1<br>
**Status:** Open (protocol and reporting)

The audit implementation overlays retry rows by instance ID:

```python
# verified_trajectory_audit.py
indexed.update(retry_rows)
```

This is appropriate for diagnosing whether a retried instance becomes usable,
but it mixes trajectories collected at different times and under different
network conditions. The final 256-row overlay is not an atomic polling pair.

**Required fix:** Preserve overlay support but label it explicitly as
`retry_overlay`. Emit base-only, overlay, and clean-intersection summaries
separately. Block it from being selected as the primary accuracy/performance
result while any infrastructure failures remain.

**Acceptance test:** An overlay report includes retry provenance, replaced
instance IDs, remaining failed IDs, and an `inconclusive_for_paired_accuracy`
boolean.

### STC-007 — vLLM teardown emits real cleanup errors after completed rollouts

**Severity:** P2<br>
**Status:** Open

Both retry arms completed collection and wrote their 27-row trajectories, but
the worker teardown emitted a missing-attribute error and repeated socket-send
warnings.

Evidence:

```text
# 13395097 driver log, line 1497
Error during vLLM shutdown: 'VllmAsyncGenerationWorker' object has no attribute
'server_thread'

# 13395098 driver log, line 1189
Error during vLLM shutdown: 'VllmAsyncGenerationWorker' object has no attribute
'server_thread'
```

This does not invalidate already persisted trajectories, but it makes terminal
job status and future lifecycle failures harder to interpret.

**Required fix:** Make shutdown idempotent and guard optional server resources
before access; distinguish cleanup warnings from generation/training failure in
the final driver status.

**Acceptance test:** A completed trajectory-collection job exits without
attribute errors or uncaught `asyncio` socket warnings during worker teardown.

### STC-008 — Cross-loop refit failure is resolved, but needs regression coverage

**Severity:** P1 regression risk<br>
**Status:** Resolved in implementation; coverage still required

The earlier full workloads `13268983` and `13274929` failed when a Uvicorn-loop
task was awaited on a Ray loop, leading to a cross-loop exception and NCCL
watchdog timeout. The documented fix marshals lifecycle work onto the Uvicorn
loop and adds a pause barrier. `13278330` then completed five refits.

**Required action:** Do not rework the fixed path without evidence of a
regression. Add/retain an automated multi-refit integration regression that
creates active streaming sessions, crosses the pause barrier, and verifies
weight synchronization on every refit.

### STC-009 — Streaming action bridge dropped the HTTP request timeout

**Severity:** P0 experiment-integrity bug<br>
**Status:** Resolved; targeted tests and two replacement first64 pairs pass<br>
**Affected artifacts:** exact-tokenizer first64 on jobs `13600551` and
`13608623`

The streaming-only branch of `LocalRuntime._send_action_server_request()` used
the requested action timeout only as the outer `call_async_from_sync()` wait
limit. `_execute_action_request()` then called `self.session.post()` without a
`timeout=` argument, so the session's much shorter default HTTP timeout could
terminate a valid long-running command first. OpenHands reported the requested
300-second value even when that was not the timer that fired:

```text
AgentRuntimeTimeoutError: Runtime failed to return execute_action before the
requested timeout of 300.0s
```

This marker appears in 28 of 64 persisted rows in
`20260709T084700Z-exact-incremental-r1-first64/exact_tokenizer_on` and 29 of 64
rows in `20260709T122500Z-exact-incremental-r2-first64/exact_tokenizer_on`.
Their paired streaming-off files contain no matching rows. Because later model
turns can quote an earlier failure, these row counts are incidence counts, not
independent action-timeout counts.

The source defect and fix are visible in
`streaming_tool_call_action_timeout.patch`:

```diff
-return call_async_from_sync(self._execute_action_request, timeout, action)
+return call_async_from_sync(
+    self._execute_action_request,
+    timeout,
+    action,
+    request_timeout=timeout,
+)
 ...
-self.session.post(url, json=payload)
+self.session.post(url, json=payload, timeout=request_timeout)
```

If the caller supplies no bridge timeout, the fixed implementation derives an
HTTP deadline from `action.timeout + 5`. The streaming-disabled branch still
delegates directly to `ActionExecutionClient` and is not affected by this
defect.

Targeted Slurm tests passed in jobs `13617443` (4 tests), `13617511` (11 tests),
and `13617627` (1 test). The first fixed 16-row off/on pair completed with zero
`AgentRuntimeTimeoutError` markers in both arms. Two fixed repeated first64
pairs (`13621383` / `13621384` and `13624636` / `13624637`) then emitted 64/64
rows in every arm with zero matching markers. STC-009 is therefore resolved;
the standalone evaluation report treats the replacement performance and
accuracy results separately from this bug's exit criterion.

**Acceptance test:** With streaming enabled, a mocked action request receives
the exact bridge timeout at `session.post`; a command longer than the session
default but shorter than its action timeout completes; paired fixed rollouts
contain no synthetic action-timeout marker.

## Non-bugs and protocol constraints

These items should not trigger a runtime code fix. They must instead be
controlled, labeled, or excluded from a claim.

| Observation | Why it is not a runtime bug | Required handling |
| --- | --- | --- |
| Temperature-zero trajectories differ | Async agent/tool timing can diverge even with identical prompt, temperature, and top-p. | Do not require exact trajectory matching. Use repeated paired trials and distributional statistics. |
| 500, filtered 474, and 256 manifests differ | They are intentionally different dataset populations. | Never compare accuracy or throughput across manifests without labeling the population and SHA-256. |
| The 474 set excludes prior timeouts | This is a legitimate stability subset, but it is selection-biased. | Use it for mechanism/performance diagnosis only; use an unfiltered predeclared Verified set for accuracy. |
| APC microbenchmark, R2E smoke, Verified rollout, and full training differ | Each answers a different question and has different model, scale, data, and concurrency. | Keep separate evidence tiers: cache mechanics, admission, rollout accuracy, and full-training stability. |
| 50 ms versus 100 ms polling | This is the intended experimental variable. | Compare `sessions / eligible_actions`, not raw session counts alone. |
| Generation-only versus full training | `SKIP_TRAINING=1` intentionally changes resources and training semantics. | Do not use generation-only timing as full-training performance. |
| Controller query failure from the sandbox | It is an execution-environment permission/network limitation, not evidence that a run failed. | Use an authorized `srun` allocation or cluster-side monitoring for job status. |
| Missing historical trajectory artifacts | A historical submission without persisted output cannot be repaired retrospectively. | Mark it unusable; do not include it in summaries or rerun it unless the data is still needed. |

## Historical artifact disposition

| Artifact | Disposition | Reason |
| --- | --- | --- |
| `20260701T025523Z` | Exclude | Job IDs exist but no trajectory artifact is available for audit. |
| `20260702T223000Z-admission-poll474` | Exclude | Submission records exist but no persisted trajectories are available. |
| `20260702T230000Z-admission-poll256` | Exclude from 256-sample claims | Both arms contain only 128 rows; superseded by `13385088` / `13385093`. |
| `20260703T181000Z-admission-poll256-batch256` | Admission evidence only | Complete 256-row pair; asynchronous trajectory divergence and setup failures preclude causal accuracy/performance claims. |
| `20260703T111754Z` retry | Diagnostic only | It is a different-time retry overlay and retains infrastructure failures. |
| `13278330` full streaming run | Lifecycle-regression evidence only | Shows the refit fix works; it has no matched streaming-off full baseline. |

## Recommended order of work

1. Fix STC-001, STC-003, and STC-004, then add their launcher/audit tests.
2. Fix STC-005 before relying on long-context accuracy results.
3. Make STC-006 provenance explicit and stop using retry overlay as a final
   accuracy result.
4. Clean up STC-007 before default enablement.
5. Run one frozen, clean full streaming-on/off pair with cached setup, a single
   predeclared manifest, identical workload shape, and a report that normalizes
   admission by eligible actions.

## Evidence index

| Evidence | Location |
| --- | --- |
| Partial 128/256 driver overrides | `results/streaming_tool_call_verified/20260702T230000Z-admission-poll256/slurm/13359969-logs/ray-driver.log:13` and `.../13359970-logs/ray-driver.log:13` |
| Correct 256/256 replacement run | `results/streaming_tool_call_verified/20260703T181000Z-admission-poll256-batch256/slurm/13385088-logs/ray-driver.log:11698` and `.../13385093-logs/ray-driver.log:14861` |
| DNS setup failures | `results/streaming_tool_call_verified/20260703T111754Z/slurm/13395097-logs/ray-driver.log:1161` and `.../13395098-logs/ray-driver.log:753` |
| Retry-overlay comparison | `results/streaming_tool_call_verified/20260703T181000Z-admission-poll256-batch256/poll100_vs_poll050_with_infra_retry_comparison.json` |
| Teardown errors | `results/streaming_tool_call_verified/20260703T111754Z/slurm/13395097-logs/ray-driver.log:1497` and `.../13395098-logs/ray-driver.log:1189` |
| Full-run context/refit evidence | `docs/design-docs/streaming-tool-call.md:840` and `docs/design-docs/streaming-tool-call.md:892` |
