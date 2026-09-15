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
