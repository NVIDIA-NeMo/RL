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
