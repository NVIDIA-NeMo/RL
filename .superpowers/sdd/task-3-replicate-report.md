# Task 3 deterministic replicate collector report

## Result

Implemented the bounded local collector for matched CuTeDSL ON/OFF replicate
jobs. The collector consumes one `submit_cutedsl_ab_replicates.sh` submission
JSONL and the benchmark result root, validates the complete evidence chain, and
writes deterministic aggregate JSON and long-form CSV.

No runtime model code, cluster profile, submission driver, scheduler job,
incident evidence, or existing experiment result was changed. No remote command
or job submission was performed.

## Files

Created:

- `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/collect_cutedsl_ab_replicates.py`
- `tests/test_cutedsl_replicate_collector.py`
- `.superpowers/sdd/task-3-replicate-report.md`

## Initial TDD RED

The focused collector tests were written before the collector existed.

Command:

```text
uv run --no-project --with pytest pytest -q tests/test_cutedsl_replicate_collector.py
```

Expected failure:

```text
FAILED test_collector_writes_deterministic_paired_aggregate_json_and_csv
python: can't open file .../collect_cutedsl_ab_replicates.py: [Errno 2]
1 failed
```

This proved the first contract was exercising the missing entry point rather
than existing behavior.

## Review TDD RED

A fresh read-only reviewer found six Important validation gaps after the first
seven tests passed. Targeted tests were added before the fixes.

Command:

```text
uv run --no-project --with pytest pytest -q --maxfail=20 \
  tests/test_cutedsl_replicate_collector.py
```

Observed result:

```text
13 failed, 7 passed
```

The expected failures covered unsafe/absolute job IDs, an outside-root status
symlink, asymmetric or null metric names, missing source/image/workload
identity, raw order-index drift, restarted job lookup, and submission/output
path collision.

## Input acceptance contract

The collector fails closed unless all of the following hold:

- At least three distinct successful submitted jobs and distinct replicate
  indices are present.
- Each submission order matches its replicate parity, and both `on,off` and
  `off,on` orders are represented.
- Every job resolves to one successful result directory. A unique successful
  `<job_id>-r<N>` directory is accepted for a Slurm restart while preserving
  the base scheduler job ID separately from the run ID.
- Job IDs are safe single path components, and every consumed artifact resolves
  inside the supplied benchmark root and its job directory.
- Exactly one `timing_summary.json` exists for every selected completed run.
- `status.json`, the manifest, summary, and raw timing run identities agree.
- Source and upstream SHAs, immutable image identity, recipe, update counts,
  topology, and fixed ON/OFF config evidence are present and identical across
  replicates.
- The exact canonical resolved metric-name set is present, contains only
  nonempty string source names, matches between ON and OFF, matches each raw
  timing artifact, and is identical across replicates.
- Every raw timing artifact has a unique `order_index` in `{0, 1}` whose arm
  agrees with the declared timing order.
- `workload_equality_required` and `workload_equality_observed` are true, the
  workload metric is `train/total_num_tokens`, ON/OFF measured workloads match,
  and the measured workload is identical across replicates.
- Exactly one submitted replicate is designated for profiling. It must contain
  a passing `kernel_attribution.json`, ON/OFF grouped-GEMM evidence, fused GLU
  and dGLU matches only on ON, two profile summaries, positive Nsight report
  counts, and contained kernel-evidence files.
- Output JSON and CSV paths differ and cannot overwrite the submission JSONL.

## Statistics and output

For each replicate and metric, the collector computes:

```text
median(ON measured steps) / median(OFF measured steps)
```

Therefore values below one favor ON for durations, while values above one favor
ON for throughputs. The six duration metrics are E2E, generation,
generation-finalize, logprob, policy training, and refit. The five throughput
metrics are E2E, generation, logprob, policy training, and refit-effective.

For every metric, JSON and CSV contain:

- the per-replicate paired ratio, job ID, replicate index, and timing order;
- the median across replicate ratios;
- sample coefficient of variation across replicate ratios;
- a deterministic percentile 95% bootstrap confidence interval over paired
  replicate ratios;
- separate `on,off` and `off,on` order-stratified summaries; and
- an extend-to-six recommendation when CV exceeds 5% or the interval includes
  one.

Bootstrap streams use SHA-256-derived per-metric seeds from the recorded base
seed, so metric ordering cannot perturb another metric's result. The output
contains no generation timestamp, uses stable ordering, and is byte-identical
when rerun with identical inputs and controls.

## Review findings and resolution

The fresh read-only review reported no Critical findings and six Important
findings. All were addressed:

- Added safe job-component and root/job containment for every direct,
  referenced, and recursively discovered artifact, including symlink escape
  rejection.
- Required the exact canonical metric set, nonempty source names, and identical
  ON/OFF mappings.
- Added required-field/type validation for source, image, and workload
  identity before cross-replicate comparison.
- Bound each raw timing arm to a unique declared order index.
- Added unique successful restarted-run resolution using base job and run IDs.
- Rejected output paths that resolve to the submission JSONL.

The reviewer then performed a read-only fix-verification pass and reported no
remaining Critical or Important findings. The reviewer also independently ran
the 20 focused tests and Python compilation successfully.

## GREEN verification

Focused collector suite:

```text
uv run --no-project --with pytest pytest -q \
  tests/test_cutedsl_replicate_collector.py
```

Observed:

```text
20 passed
```

Complete relevant CuTeDSL regression set:

```text
uv run --no-project --with pytest pytest -q \
  tests/test_cutedsl_replicate_collector.py \
  tests/test_oci_cutedsl_wrapper.py \
  tests/test_cutedsl_cluster_profiles.py \
  tests/test_cutedsl_report.py
```

Observed:

```text
161 passed in 15.31s
```

Static validation:

```text
uv run --no-project --with ruff ruff format --check \
  tests/test_cutedsl_replicate_collector.py \
  experiments/cutedsl_qwen3_30ba3b_oci_1n4g/collect_cutedsl_ab_replicates.py
uv run --no-project --with ruff ruff check \
  tests/test_cutedsl_replicate_collector.py \
  experiments/cutedsl_qwen3_30ba3b_oci_1n4g/collect_cutedsl_ab_replicates.py
uv run --no-project --with pyrefly --with pytest pyrefly check \
  tests/test_cutedsl_replicate_collector.py \
  experiments/cutedsl_qwen3_30ba3b_oci_1n4g/collect_cutedsl_ab_replicates.py
git diff --check
```

Observed: Ruff passed, both files were formatted, Pyrefly reported zero errors,
and the diff check emitted no diagnostics.

## Remaining concern

The implementation was verified only with local artifact fixtures that mirror
the current benchmark payload schema. The first completed three-replicate
cluster run remains the validation point for live filesystem layout, real
resolved metric names, kernel-name stability, and the practical discreteness
of a bootstrap interval based on only three paired ratios. The collector will
fail loudly rather than aggregate if any live artifact drifts from the enforced
contract.
