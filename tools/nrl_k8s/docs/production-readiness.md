# nrl-k8s — Production Readiness Audit

Scope: read-only audit of `tools/nrl_k8s/` at commit `hemil/k8s-infra`.
The CLI is end-to-end functional on Qwen3-4B disagg and single-cluster runs, and 48 unit tests pass on `config`, `schema`, `manifest`. What follows is a prioritised backlog of gaps that block or jeopardise a production launch.

Priority rubric: **P0** = blocks production deploy / data-loss / security; **P1** = ship-blocker for v1.0 (usability, reliability under failure); **P2** = post-1.0 polish.

Legend: each finding lists `path:line -> issue -> fix -> priority -> est. minutes`.

---

## 1. Operational robustness

The CLI talks to k8s and the Ray job submission API over the network and through a `kubectl port-forward` subprocess, but very little of it is defensive against a flaky network, a re-authenticated kubeconfig, or a dead port-forward.

- `submit.py:41-82 (dashboard_url)` -> port-forward is launched once and never re-established. A laptop Wi-Fi blip, VPN reconnect, or a kubelet restart kills the forward mid-submit and the Ray SDK call fails with an opaque `ConnectionError` -> wrap the dashboard interaction in a reconnect loop (kill + respawn port-forward, wait for TCP, retry the SDK call on `urllib3.exceptions.NewConnectionError`/`MaxRetryError`) -> **P1** -> 60
- `submit.py:68-73` -> port-forward's stdout is pipe-buffered and only read after the proc exits (`_wait_for_tcp` only reads on early death), so a kubectl that prints "Handling connection for N" but never establishes the forward looks like "timed out" with no diagnostic -> drain stdout into a rolling buffer in a daemon thread and include last N lines in the `TimeoutError` -> **P1** -> 30
- `submit.py:60-66` -> `kubectl port-forward` is invoked without `--address 127.0.0.1` or `--context`. On a multi-context kubeconfig (common at NVIDIA) we may forward against the wrong cluster -> pass `--context=$(current-context)` and `--address=127.0.0.1` explicitly; surface the chosen context in log -> **P1** -> 15
- `k8s.py:116-135 (wait_for_raycluster_ready)` -> polls with a flat `poll_s=5` without exponential backoff and without tolerating transient API errors: a single `ApiException` (500, 503, SSL timeout) propagates out and aborts `run`. Likelihood: high during cluster-autoscaler events -> wrap `get_raycluster` in a retry helper that swallows 429/5xx/`urllib3.exceptions.ProtocolError` up to N tries before re-raising -> **P0** -> 45
- `k8s.py:28-34 (load_kubeconfig)` -> `@functools.cache`d. If the kubeconfig token expires mid-run (AWS SSO, short-lived OIDC), subsequent API calls fail and there is no re-load path -> drop the cache on 401; or re-call `load_kube_config` on each `ApiException.status == 401` -> **P1** -> 45
- `k8s.py:46-69 (apply_raycluster)` -> on 409 it unconditionally `PATCH`es even if another user's run owns the RayCluster. There is no owner-label check, no `resourceVersion` race guarantee; two concurrent `run`s can silently overwrite each other's specs -> before patch, verify a `managed-by=nrl-k8s` label and the current user / run-id; error out if the existing cluster was not ours -> **P0** -> 60
- `orchestrate.py:108, 295, inspect.py:120` -> bare `except Exception` swallow every failure including `KeyboardInterrupt`'s non-`BaseException` cousins and API auth failures; user sees `daemon_status=None` and assumes "not running" when in truth the dashboard is unreachable -> narrow to `(ApiException, JobSubmissionClientError, ConnectionError, TimeoutError)` and surface the reason in the status row -> **P1** -> 30
- `orchestrate.py:304-316 (_wait_for_http)` -> health-check loop retries every 5s with no backoff; a misconfigured URL (e.g. cluster-internal DNS when the CLI is running on a laptop) silently spins for 5 min -> detect obvious laptop-cannot-reach-cluster-DNS conditions (`socket.gaierror` on `*.svc.cluster.local`) and fail-fast with an actionable message -> **P1** -> 20
- `orchestrate.py:282-301 (_wait_job_stopped)` -> a Ray Job that Ray insists is `RUNNING` but whose node has been evicted stays `RUNNING` forever; timeout is just 60s and then the loop logs "continuing" and proceeds to submit a clashing daemon -> after timeout, call `client.delete_job` (and error if it still isn't terminal) instead of silently racing -> **P1** -> 30
- `submit.py:121-146 (tail_job_logs)` -> daemon thread uses a blocking `q.get()` with no timeout, so `Ctrl+C` during a tail will cancel the main thread's sleep but leave the async asyncio loop hanging until the process exits. Usually benign but leaks file descriptors under long-running observability sessions -> use `q.get(timeout=1)` in a loop and wind down the asyncio loop cleanly -> **P2** -> 20
- `cli.py:82/170/227/432/536/602/619/656` -> every top-level error handler prints `error: {exc}` and calls `sys.exit(1)`. The raw exception from `kubernetes.client.ApiException` includes a multi-line dump with headers, which is noisy; the root cause (e.g. "kubeconfig expired") is buried -> classify exceptions and emit a short, actionable first line (`hint: kubeconfig expired; run aws sso login`) before the full trace under `-v` -> **P1** -> 60
- `submit.py:179-190 (_wait_for_tcp)` -> polls every 0.5s for 30s; if kubectl spawns but the LB hasn't propagated, you see "didn't open 127.0.0.1:X in 30s" with no hint. Make the timeout configurable via `infra.submit.portForwardTimeoutS` -> **P2** -> 15
- `orchestrate.py:218-225 (run)` -> clusters are brought up sequentially (generation, then gym, then training). A transient failure on the gym cluster tears the whole run; there is no resumability (the CLI doesn't persist a run-state) -> add a local state file (`~/.cache/nrl-k8s/runs/<run-id>.json`) that tracks which clusters are up and allows `--resume` -> **P1** -> 120

## 2. Secrets handling

The CLI doesn't explicitly read secrets, but it emits many code paths that dump user-provided YAML, env dicts, or pod manifests. Those payloads frequently contain tokens.

- `cli.py:86-91 (validate)` -> the resolved `InfraConfig` is printed verbatim. The schema allows `launch.env` / `daemon.env` / `networking.extra_env` as `dict[str,str]`, so any user that puts `{WANDB_API_KEY: xxx}` into the recipe sees it leak into the terminal and into `nrl-k8s validate > out.yaml` commits -> either deny-list known secret-looking keys (`*_API_KEY`, `*_TOKEN`, `*_PASSWORD`) when printing, or redact by default and add `--show-secrets` -> **P0** -> 30
- `cli.py:117-126 (plan)` -> prints the full RayCluster manifest including any env with `valueFrom.secretKeyRef`. That's only a name-reference and usually fine, but if a user inlined a plaintext secret into `spec.headGroupSpec.template.spec.containers[*].env` it gets dumped. Same redaction pass should apply here -> **P1** -> 15
- `orchestrate.py:126-148` -> on `--replace` the log emits the daemon submission id but not the env; still, the `submit_ray_job` call sends `env_vars=daemon.env` which travels in cleartext to the Ray dashboard (HTTP, no TLS by default). When the dashboard is behind a LoadBalancer this is a genuine risk -> document the HTTP-vs-HTTPS boundary in README + SECURITY.md; add a `submit.insecureHttpDashboard: bool` guard that refuses non-localhost, non-TLS dashboards for env-bearing jobs -> **P0** -> 60
- `cli.py:82 / 170 / 227` -> `error: {exc}` stack traces leak kubeconfig auth tokens in some `ApiException` message bodies (`kubernetes` client includes response body by default in `reason`). A traceback on a 401 from an OIDC-authenticated cluster typically includes the bearer token in the `Authorization` header that the client echoes back -> before echoing an `ApiException`, scrub `Authorization:` / `Bearer ` patterns from `exc.body` and `exc.headers` -> **P0** -> 40
- `workdir.py:53-92 (stage_workdir)` -> no scrubbing of `.env`, `*.pem`, `*.key`, `credentials*`, `id_rsa*`, `secrets.yaml` from the copied tree. If a researcher `cd`s into a repo root that contains a `.env` it is uploaded to GCS as part of the working_dir zip and served from the Ray dashboard -> add those patterns to `_IGNORE_PATTERNS`; add a pre-upload size+names preview (`Uploading 97MB across 12,437 files; first hit: examples/.env`) with opt-in `--yes` -> **P0** -> 30
- `inspect.py:57-66 (collect_status) / cli.py:263-276 (status)` -> not a direct leak but the status command attaches via `dashboard_url` to three clusters in series, and any port-forward failure logs the `kubectl port-forward` stderr which contains the full kubeconfig path and context. Not a secret by itself, but combined with AWS SSO caches it's fingerprinting data -> redact the kubeconfig path from errors -> **P2** -> 15

## 3. Test coverage holes

Only 3 modules have unit tests (`config`, `schema`, `manifest`, totalling 48 tests). The CLI's orchestration, I/O, and CLI layer are wholly untested.

- `orchestrate.py` -> zero tests. The highest-value business logic (replace logic, `_infer_disagg_job_id`, ConfigMap reset, sequential cluster bring-up, `_wait_for_http`) runs only in production -> add pytest-mock tests that fake `k8s.*` and `JobSubmissionClient` and cover: daemon skipped when RUNNING without --replace; daemon re-submitted with fresh id when --replace; FAILED daemon errors without --replace; `_wait_for_http` returns on 500<status<200 (bug: the current check `200 <= r.status < 500` swallows 4xx like 404, which is probably intended but undocumented); endpoint ConfigMap deleted when --replace and gym job-id present; `_infer_disagg_job_id` handles missing gym or missing `--job-id` -> **P0** -> 180
- `submit.py` -> zero tests. The port-forward lifecycle + `submit_ray_job` are the CLI's only k8s-to-Ray bridge -> add subprocess-mocked tests for `dashboard_url` (in-cluster vs laptop branch, kubectl-missing, early-exit, timeout); add a test that `submit_ray_job` assembles the correct `runtime_env` when `pip=None`, `env_vars=None`, `submission_id=None` -> **P0** -> 90
- `workdir.py` -> zero tests, but it silently drops `.gitignore` files (a bug-fix encoded as behaviour) and implicitly relies on `.gitignore` stripping to ship training data; **no test pins this invariant** -> add tests: `.gitignore` is dropped; a `data/foo.jsonl` file under a `.gitignore`d path is still present in staging; `extra_files` path collisions are well-defined; missing optional paths are skipped -> **P1** -> 45
- `inspect.py` -> zero tests. `_latest_daemon_job` branch that strips the timestamp suffix is fragile and untested -> add tests mocking `JobSubmissionClient.list_jobs` with base/suffixed/unrelated ids -> **P1** -> 45
- `k8s.py` -> zero tests. `apply_raycluster` 409 path and `wait_for_raycluster_ready` timeout path never exercised -> pytest-mock the `CustomObjectsApi` and assert both branches -> **P1** -> 60
- `cli.py` -> zero tests. Nobody has exercised, e.g., `cluster down --name` vs `--role`, or the invariant that `--infra` and recipe-level `infra:` conflict -> use click's `CliRunner` to exercise each subcommand's argument validation and the `sys.exit(2)` path (without actually touching k8s) -> **P1** -> 120
- `config.py` -> coverage gap: override partitioning for `+infra.foo=x` / `~infra.foo` (append/remove) is implemented but not tested -> add test cases for `+infra.scheduler.queue=x` and `~infra.scheduler.queue` semantics -> **P2** -> 20
- `schema.py` -> coverage gap: `LaunchSpec.entrypoint` is `Optional` so the sentinel "must be set for nrl-k8s launch" is enforced only at runtime in `orchestrate.submit_training`. Add a schema-level test that at least one scheduler/queue combo in an attach-mode recipe is accepted, and that `launch.mode=attach` without any of the three attach targets is rejected (exists but no test on mixing with `launch.entrypoint=None`) -> **P2** -> 20

## 4. CLI UX gaps

- `cli.py:237-240 / 351-355 / 365-374` -> `doctor`, `dashboard`, `dev up`, `dev down` are advertised in `--help` but just print `not yet implemented (phase: N)` and exit with 2. This is actively misleading for a v1.0 release -> either hide them behind a `NRL_K8S_SHOW_STUBS=1` env flag, or remove them from the group until implemented -> **P1** -> 15
- `cli.py:82, 170, 227` -> bare stack traces go to stderr. No `--verbose`/`--quiet`/`-v` flag, no log format toggle -> add `--log-level` at the root group, default INFO, with DEBUG surfacing full tracebacks. Use `logging` instead of `click.echo` for diagnostics -> **P1** -> 60
- no `--dry-run` anywhere -> `nrl-k8s run --dry-run` should plan all three manifests, print what daemons would be submitted with what entrypoints, and exit 0. This is critical for reviewers who can't actually `kubectl apply` -> **P0** -> 60
- `submit.py / orchestrate.py` -> no progress indicator on 97 MB working_dir uploads or on the 2-3 min cluster bring-up; users ctrl+C thinking the CLI is hung -> print periodic "still waiting on RayCluster X (elapsed 90s / state=Pending)" lines every 30s in `wait_for_raycluster_ready`; print byte count before calling `submit_job` -> **P1** -> 45
- `cli.py:150-177 (launch) / 201-234 (run)` -> `--follow` tails training logs, but there is no way to tail a daemon at submit time. When gym fails to come up, the user has to find out post-hoc via `nrl-k8s logs --role gym`. Add `--follow-daemons` that streams all daemon logs in parallel -> **P2** -> 60
- `cli.py:484-497 (cluster list)` -> the `--namespace` flag ignores recipe; other subcommands all require a recipe for namespace resolution. Inconsistent. Either accept an optional recipe or keep it strictly `--namespace` and document -> **P2** -> 10
- `cli.py:107-126 (plan)` + `cli.py:65-95 (validate)` -> no `--output` flag to write to a file (users have been piping to `> out.yaml`, which breaks `--show-recipe` two-section output) -> add `-o/--output PATH` -> **P2** -> 15
- no shell completion (`click`'s `click.shell_completion`) -> a trivial 10-line addition; adds noticeable polish -> **P2** -> 15
- error hint when `--infra` file and recipe both declare `infra:` at `config.py:202-208` -> the ValueError is fine; surfaces through `cli.py:620`. Consider telling the user which line of the recipe has `infra:` -> **P2** -> 15

## 5. Packaging + versioning

- `pyproject.toml` -> dependencies list open-ended lower bounds but no upper pin; `ray[default]>=2.52` will happily install 3.x breaking changes. Pin to a tested range (`>=2.52,<3`) and bump deliberately -> **P1** -> 15
- `pyproject.toml` -> no `[project.urls]` (home, issues, changelog). No `classifiers`. No `keywords` -> **P2** -> 10
- no `CHANGELOG.md` -> add a keep-a-changelog with the current 0.1.0 feature set documented -> **P1** -> 20
- `__init__.py:3` -> `__version__ = "0.1.0"` is duplicated from `pyproject.toml`. Moving to `importlib.metadata.version("nrl-k8s")` avoids drift -> **P2** -> 10
- no CI for the package. `tools/nrl_k8s` is not wired into the repo's `.github/workflows/*` that I can see -> add a minimal GHA job that runs `pip install tools/nrl_k8s[test]` and `pytest tools/nrl_k8s/tests/unit` on PRs touching the tool -> **P0** -> 45
- no entry-point test (`pip install . && nrl-k8s --version`) in CI -> add a `tests/unit/test_entry_point.py` that uses `CliRunner` to invoke `--version` and `--help` -> **P1** -> 15
- no license headers on source files; pyproject says Apache-2.0 but the Python files have no SPDX line. Repo convention check needed -> **P2** -> 15
- `orchestration/`, `schedulers/`, `backends/`, `templates/` are empty directories with empty `__init__.py` files. Either populate or remove -> **P2** -> 10

## 6. Config validation gaps

The schema is strict (`extra=forbid`), but several runtime invariants are not expressed in pydantic and blow up downstream only after you bring up the clusters.

- `schema.py:282-302 (ClusterSpec)` -> `spec` is `dict[str, Any]` — zero validation on the RayCluster body. Missing `headGroupSpec` or containers without a name yield cryptic KubeRay webhook errors only after `nrl-k8s run` applies -> add a light check in `manifest.py` that walks the body and raises a clean error for common mistakes (no containers, no head group, image mentioned but empty string) before the k8s call -> **P1** -> 45
- `orchestrate.py:228-243 (_infer_disagg_job_id)` -> the invariant that training's `+env.disagg_job_id=<X>` matches gym's `--job-id <X>` is enforced only by a regex hack on the gym entrypoint and with a best-effort "if we can parse it, else skip ConfigMap delete". A mismatch silently makes gym hang forever because training publishes to a different ConfigMap -> promote `disagg_job_id` to a first-class schema field (`infra.launch.disaggJobId: str | None`) and inject it as an env var into both the gym daemon and the training entrypoint; emit a validator error if gym declares one and training is missing it -> **P0** -> 75
- `schema.py:192-224 (LaunchSpec/AttachSpec)` -> `launch.attach.training: str | None` vs. `clusters.training.name: str` are both strings and nothing enforces they reference the same RayCluster. Similar for `attach.generation` / `clusters.generation.name`. If they drift, the CLI applies a new RayCluster under `clusters.training.name` but submits the job against a nonexistent cluster name in `attach.training` -> add a `model_validator` on `InfraConfig` that, in `mode=attach`, enforces `attach.<role>` equals `clusters.<role>.name` (or is None when the cluster is None) -> **P0** -> 30
- `schema.py:197-207 (LaunchSpec.entrypoint)` -> the doc-comment says "required for `nrl-k8s launch` / `nrl-k8s run`" but the schema marks it Optional and `orchestrate.submit_training:164-165` throws a runtime ValueError. Users who pass only `cluster up --role generation` legitimately need no entrypoint; but `nrl-k8s run` without an entrypoint should fail at load time -> add a `post-validate` step in `_load_or_exit` (or make it a top-level `run`/`launch` precondition in the CLI) that checks `launch.entrypoint` is non-empty and that `clusters.training` is defined -> **P1** -> 30
- `schema.py:259-279 (DaemonSpec)` -> `submissionId` is optional; but `orchestrate.submit_daemon:106-107` does `client.get_job_status(daemon.submissionId)` unguarded if it's None, which will raise `TypeError`. Add `@model_validator` to require `submissionId` when parent `launch.peerWatcher=True` or when `nrl-k8s run --replace` is likely -> **P1** -> 20
- `schema.py:193-195 (AttachSpec)` -> gym may be declared without a training cluster; but `nrl-k8s run` assumes `training` exists. Schema doesn't forbid gym-only runs -> either allow gym-only (document) or validate -> **P2** -> 15
- `config.py:216-218` -> underscore-prefixed top-level keys (e.g. `_shared: &anchors`) are stripped. Silent. A typo like `_shred` would be silently dropped instead of flagged -> log-warn on strip in debug mode -> **P2** -> 10
- `schema.py:82` -> `NetworkingSpec.extra_env` is never applied anywhere (grep confirms). Dead config key -> either wire it into manifest/patch or remove -> **P2** -> 20
- `schema.py:228-246 (ResourcesSpec)` -> declared but never consumed — `manifest.py` does not read it. Another dead knob that will mislead users -> remove or implement before v1.0 -> **P1** -> 30

## 7. Multi-environment portability

The code itself is surprisingly free of AWS-specific strings, but the examples (and by extension the demo path the user follows) bake in assumptions.

- `examples/qwen3_4b_if_full_disagg.infra.yaml:42, 49, 104-105` -> `vpc.amazonaws.com/efa`, `enp71s0`, `FI_PROVIDER=efa` hardcoded in the only examples. First-run on Azure/GCP/on-prem will fail silently (NCCL falls back to TCP and training slows 100x) with no error -> add a second example (`examples/azure_*.infra.yaml` or `examples/no_efa.infra.yaml`) that doesn't assume EFA, plus a note in README explaining the EFA case -> **P1** -> 45
- `examples/qwen3_4b_if_full_disagg.infra.yaml:14-17` -> `gpu-wrangler.nvidia.com/lease: nemo-rl-testing` label is NVIDIA-internal. Reused in node-selector + toleration -> move to a top-level alias (`${node.lease}`) and document it as "replace with your own scheduler's pool label" -> **P2** -> 20
- `schema.py:32-35 (SchedulerKind)` -> enum includes `kai`, `kueue`, `default` but nothing in `manifest.py` or `orchestrate.py` actually reads `infra.scheduler.kind` beyond validation. Researchers set `kind: kai, queue: priority-team` and expect the CLI to patch the `scheduling.run.ai/queue` label onto pods — but it doesn't -> either auto-patch the KAI `scheduling.run.ai/queue` / Kueue `kueue.x-k8s.io/queue-name` labels onto `spec.headGroupSpec.template.metadata.labels`, or document loudly that users must add the label themselves -> **P0** -> 60
- `workdir.py:33-46 (DEFAULT_RAY_UPLOAD_PATHS)` -> the defaults are NeMo-RL-monorepo-shaped (`3rdparty/Gym-workspace/Gym/...`). A thinned-down NeMo-RL checkout will miss most of these. The `if not src.exists(): continue` at workdir.py:79 makes it silent -> log a warning when a default path is skipped; consider making the default list empty and forcing recipes to declare their paths -> **P1** -> 20
- `orchestrate.py:252 (cm_name = f"nemo-rl-endpoints-{job_id}")` -> the ConfigMap name prefix is hard-coded to match the server-side code in `nemo_rl.distributed.k8s_endpoint_registry`. If the server-side code ever changes the prefix, `--replace` quietly stops working. Share the prefix in a single source of truth (import from `nemo_rl.distributed.k8s_endpoint_registry` with a fallback for the standalone install path) -> **P1** -> 15
- `submit.py:29 (DASHBOARD_PORT = 8265)` -> hardcoded. KubeRay convention, fine; but a bespoke operator with `--dashboard-port` different from 8265 can't use the CLI -> plumb through `infra.submit.dashboardPort` -> **P2** -> 15

## 8. Concurrency safety

Two researchers running `nrl-k8s run` or even `status` against the same namespace concurrently is a first-class use case (shared dev namespace is standard), and several paths are not safe.

- `submit.py:173-176 (_free_port)` -> picks a random free port at `bind(0)`. Two concurrent invocations may race: both get port P, the first `kubectl port-forward` binds, the second sees `bind: address already in use` -> either retry with a fresh port on `OSError: EADDRINUSE` from the port-forward, or keep the socket bound while spawning kubectl and hand off; prefer the retry because `SO_REUSEADDR` doesn't help with kubectl -> **P1** -> 20
- `orchestrate.py:246-254 (_reset_endpoint_registry)` -> deletes the ConfigMap unconditionally. If User A is running with job-id `foo` and User B does `run --replace` with the same `foo`, User A's training loses its registry mid-run and rendezvous breaks silently -> scope ConfigMap name by user (`nemo-rl-endpoints-<user>-<job_id>`) or add an owner annotation and refuse delete if the annotation doesn't match the current run -> **P0** -> 45
- `orchestrate.py:71 (apply_raycluster)` + `k8s.py:46-69` -> concurrent applies of the same RayCluster name clobber each other; one wins the patch and the other's topology is lost. See §1 above; same fix (owner-label + resourceVersion) -> **P0** -> (shared)
- `submit.py:103-109 (submit_ray_job)` -> `submission_id` is user-supplied (for daemons); Ray's server rejects a re-used id, which is nice, but two concurrent `run --replace` both compute `_fresh_submission_id` from `time.time()` and can collide at second granularity -> use `time.time_ns()` or append a short random suffix -> **P1** -> 10
- `workdir.py:74 (mkdtemp)` -> each invocation gets its own dir, so staging is fine; but nothing cleans up old `nrl-k8s-workdir-*` directories. Over weeks of use `/tmp` gets hundreds of 100 MB copies -> add a GC helper invoked at CLI startup that deletes `nrl-k8s-workdir-*` older than 24h -> **P2** -> 20
- `orchestrate.py:183-194 (submit_training --replace)` -> stops ALL running Ray jobs on the training cluster, not just ones owned by this recipe. If a second user shares the training cluster, their job gets killed by your `--replace` -> filter by `metadata.submission_id` prefix (a per-recipe unique prefix) or by a runtime_env label -> **P0** -> 30

---

## Top 10 to do first

1. **P0, 30m** — `§2` Scrub `Authorization: Bearer ...` + redact `*_API_KEY` from error output and `validate` output (`cli.py:82/170`, `cli.py:86-91`, `submit.py:179`).
2. **P0, 30m** — `§2` Add `.env`, `*.pem`, `*.key`, `credentials*`, `id_rsa*` to `_IGNORE_PATTERNS` in `workdir.py:19` and preview the first few staged paths before upload.
3. **P0, 60m** — `§7` / `§5` Auto-patch `scheduling.run.ai/queue` (KAI) / `kueue.x-k8s.io/queue-name` (Kueue) labels onto pod templates when `infra.scheduler.kind` is set — currently the enum exists but is never applied.
4. **P0, 75m** — `§6` Promote `disagg_job_id` to a first-class schema field and inject it into both the gym daemon and training entrypoint; forbid recipes whose gym + training job-ids differ.
5. **P0, 60m** — `§1` Wrap every k8s API call (`get_raycluster`, `list_rayclusters`, `list_namespaced_pod`) in a retry helper that swallows 429/5xx/`ProtocolError` for ~3 tries with exponential backoff.
6. **P0, 60m** — `§1` Owner-label guard in `apply_raycluster`: refuse to patch a RayCluster that lacks `managed-by=nrl-k8s` or whose `nrl-k8s/run-id` differs from the current invocation.
7. **P0, 45m** — `§8` Scope endpoint-registry ConfigMap by user (`nemo-rl-endpoints-<user>-<job_id>`) so concurrent `run --replace` can't trash each other's rendezvous.
8. **P0, 60m** — `§4` `--dry-run` on `run` / `launch` / `cluster up`: render + print all manifests and daemon/entrypoint commands without touching k8s.
9. **P0, 180m** — `§3` Tests for `orchestrate.py` covering `submit_daemon --replace`, `--no-replace` FAILED handling, and `_reset_endpoint_registry`.
10. **P0, 45m** — `§5` Add a GitHub Actions job that runs `pip install tools/nrl_k8s[test] && pytest tools/nrl_k8s/tests/unit` on every PR touching the tool, plus a smoke-test entry-point check.

## Running totals by priority

- P0 findings: 14 (est. 775 min / 12.9 h)
- P1 findings: 31 (est. 1305 min / 21.75 h)
- P2 findings: 17 (est. 305 min / 5.1 h)

Overall, ~39 engineering hours to reach a credible v1.0. The P0 list alone is under 2 working days and closes every data-loss / secret-leak / cross-user safety gap that the current codebase has.
