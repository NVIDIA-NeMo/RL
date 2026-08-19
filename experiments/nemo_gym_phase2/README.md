# NeMo Gym Router Phase 2 experiments

This isolated harness runs the three required arms (`direct`, `cache_aware`,
and `consistent_hash`) on fresh Slurm engine processes. Every run starts its
own RL-Insight 0.2.1 and Prometheus 2.54.1 instance, registers all backend and
Router targets, performs one same-engine warmup request, clears only the
warmup model-call captures, waits four scrape periods, and then evaluates the
same byte-identical workload.

Every owned Router target is the NeMo RL metrics adapter rather than the native
0.1.15 port. It preserves native exposition, stabilizes lazily registered
operational counters, records native metric presence and health provenance, and
adds the audited DEBUG-log cache counters only for the cache-aware arm. The
single-run report rejects missing adapter evidence instead of treating an absent
native counter as zero.

For Router 0.1.15, the native worker-load gauge has no production updater and
running requests are policy-specific. The report therefore uses each backend's
independently scraped `vllm:num_requests_running` series, maps its replica label
to the exact Router worker URL from the manifest, and retains the absent-native
evidence instead of filling it with zero.

The harness requires explicit paths and a precomputed container digest so the
result cannot silently depend on the submit host. The model and tokenizer load
directly from `PHASE2_MODEL_SNAPSHOT`; its directory name must match the pinned
revision. Metadata creation rejects tracked changes in the NeMo RL, NeMo Gym,
or RL-Insight trees and verifies uv 0.11.28 and Prometheus 2.54.1. `launch_arm.sh`
lists the required `PHASE2_*` variables. Run one smoke arm first, then submit at
least two formal repeats:

The container-baked environment is not reused as formal evidence when its
fingerprint differs from the checked-out lock file. First submit
`prepare_runtime.sub`; source the emitted `runtime-*.env` file after it finishes.
It creates separate lock-matched vLLM and NeMo Gym/Router environments with uv
0.11.28, checks both selected environments with `uv sync --frozen --check`, and
requires the same Ray version in both. Its verifier records upstream metadata
differences covered by the repository's explicit `override-dependencies` or
`exclude-dependencies` policy and fails on any unclassified incompatibility.
Both verification JSON files are embedded in every formal run's metadata.
The exact uv-managed Python 3.13.14 installation also lives under the shared,
lock-specific runtime root; venvs may not link to an ephemeral container home.
DeepGEMM is built with `DG_USE_LOCAL_VERSION=0` so its installed `2.5.0`
metadata matches the checked-in lock instead of producing perpetual
`2.5.0+local` reinstall drift.
Every arm launch repeats both lock checks and rejects tracked changes in the
NeMo RL, NeMo Gym, or RL-Insight trees before allocating GPUs.
`validate_runtime.sub` additionally starts the real RL-Insight and Prometheus
processes, registers two synthetic backend targets plus one Router target,
waits for all three scrapes, and archives the observed required labels and
cache metric samples in `observability-validation-<job-id>.json`.
Before a GPU run, `validate_ray_control_plane.sh` must also pass from a
one-node `ray.sub` allocation. Unlike a connection-only check, it launches a
default task plus bare and environment-inheriting task/actor pairs under both
the vLLM and NeMo Gym interpreters. It rejects any Python or Ray-version drift
and atomically archives the source commit and validator hash alongside the
worker module paths and environment snapshots.
Both the validator and each GPU arm stage the audited Prometheus executable on
the allocated node and verify its SHA-256 before startup. This avoids
RL-Insight's five-second binary-version probe being affected by transient
Lustre startup latency; metadata still records the hash of the executable that
actually ran.

```bash
sbatch experiments/nemo_gym_phase2/prepare_runtime.sub
source /path/printed/by/prepare_runtime/runtime-<id>.env
experiments/nemo_gym_phase2/launch_arm.sh direct smoke smoke-1
experiments/nemo_gym_phase2/submit_matrix.sh formal 2
```

Each run is write-once under `experiments/nemo_gym_phase2/runs/`. The job
generates the single-run Phase 2 report before stopping its dedicated
RL-Insight instance. Pass the six report directories to
`tools/nemo_gym_phase2_compare.py` to build the final matrix report.
