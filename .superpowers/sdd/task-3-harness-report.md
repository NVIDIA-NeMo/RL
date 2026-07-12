# Task 3 Harness Hardening Report

## Scope

Implemented only the approved harness changes from `.superpowers/sdd/task-3-brief.md`:

- Functional profiling now fails closed unless it produces at least one copied `.nsys-rep`, nonempty `kernel_evidence.txt`, and fused forward plus dgrad signature matches.
- Benchmark extraction now requires and exports the requested timing and throughput series, alongside the existing policy-throughput series.
- The extractor derives `refit_effective_tokens_per_sec_per_gpu` as `train/total_num_tokens / refit duration / 4` and exports it to JSON and CSV.
- ON/OFF measured `train/total_num_tokens` equality is now a required acceptance condition.

No runtime model code, cluster profiles, remote jobs, submission drivers, run indexes, or incident records were changed. No job was submitted.

## Files changed

- `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/run_cutedsl_functional.sbatch`
- `experiments/cutedsl_qwen3_30ba3b_oci_1n4g/run_cutedsl_matrix.sbatch`
- `tests/test_oci_cutedsl_wrapper.py`
- `.superpowers/sdd/task-3-harness-report.md`

## RED evidence

Tests were added before implementation. Each requested behavior was run independently to prove it was missing for the expected reason.

### Functional profile fail-closed validator

Command:

```text
uv run --active --no-sync pytest tests/test_oci_cutedsl_wrapper.py -k 'functional_profile_validator or functional_profile_pass_occurs or extracts_all_required_measured_component_series or timing_summary_enforces_identical_measured_workloads or records_workload_and_normalized_throughput_as_primary' -q
```

Output:

```text
F
=================================== FAILURES ===================================
_ test_functional_profile_validator_fails_closed_and_requires_fused_signatures _
...
>       assert start in SCRIPT
E       assert '# CUTEDSL_FUNCTIONAL_PROFILE_VALIDATOR_START\n' in '#!/bin/bash\n...'
tests/test_oci_cutedsl_wrapper.py:991: AssertionError
=========================== short test summary info ============================
FAILED tests/test_oci_cutedsl_wrapper.py::test_functional_profile_validator_fails_closed_and_requires_fused_signatures
!!!!!!!!!!!!!!!!!!!!!!!!!! stopping after 1 failures !!!!!!!!!!!!!!!!!!!!!!!!!!!
1 failed, 45 deselected in 0.09s
```

Exit code: `1` (expected RED; validator absent).

### Canonical measured metric series and derived refit throughput

Command:

```text
uv run --active --no-sync pytest tests/test_oci_cutedsl_wrapper.py::test_benchmark_extracts_all_required_measured_component_series -q
```

Output:

```text
F
=================================== FAILURES ===================================
________ test_benchmark_extracts_all_required_measured_component_series ________
...
>       assert set(raw["resolved_metric_names"]) == canonical_names
E       AssertionError: assert {...} == {...}
E         Extra items in the right set:
E         'performance/generation_tokens_per_sec_per_gpu'
E         'performance/policy_and_reference_logprobs_tokens_per_sec_per_gpu'
E         'timing/train/generation_finalize'
E         'performance/tokens_per_sec_per_gpu'
=========================== short test summary info ============================
FAILED tests/test_oci_cutedsl_wrapper.py::test_benchmark_extracts_all_required_measured_component_series
!!!!!!!!!!!!!!!!!!!!!!!!!! stopping after 1 failures !!!!!!!!!!!!!!!!!!!!!!!!!!!
1 failed in 0.10s
```

Exit code: `1` (expected RED; four requested canonical series absent).

### Enforced workload equality

Command:

```text
uv run --active --no-sync pytest tests/test_oci_cutedsl_wrapper.py::test_benchmark_timing_summary_enforces_identical_measured_workloads -q
```

Output:

```text
F
=================================== FAILURES ===================================
_____ test_benchmark_timing_summary_enforces_identical_measured_workloads ______
...
>       assert start in BENCHMARK_SCRIPT
E       assert '# CUTEDSL_TIMING_SUMMARIZER_START\n' in '#!/bin/bash\n...'
tests/test_oci_cutedsl_wrapper.py:1269: AssertionError
=========================== short test summary info ============================
FAILED tests/test_oci_cutedsl_wrapper.py::test_benchmark_timing_summary_enforces_identical_measured_workloads
!!!!!!!!!!!!!!!!!!!!!!!!!! stopping after 1 failures !!!!!!!!!!!!!!!!!!!!!!!!!!!
1 failed in 0.04s
```

Exit code: `1` (expected RED; executable acceptance summarizer contract absent).

Command:

```text
uv run --active --no-sync pytest tests/test_oci_cutedsl_wrapper.py::test_benchmark_records_workload_and_normalized_throughput_as_primary -q
```

Output:

```text
F
=================================== FAILURES ===================================
_____ test_benchmark_records_workload_and_normalized_throughput_as_primary _____
...
E           AssertionError: "workload_equality_required": True
=========================== short test summary info ============================
FAILED tests/test_oci_cutedsl_wrapper.py::test_benchmark_records_workload_and_normalized_throughput_as_primary
!!!!!!!!!!!!!!!!!!!!!!!!!! stopping after 1 failures !!!!!!!!!!!!!!!!!!!!!!!!!!!
1 failed in 0.03s
```

Exit code: `1` (expected RED; the summary still declared equality optional).

## Implementation

### Functional gate

The functional payload now writes `functional_profile_attribution.json` before metric export. It records report count, exact accepted signature regexes, forward and dgrad match counts, and all failures. Any missing report, blank evidence, missing forward signature, or missing dgrad signature raises before the outer script can emit `profile pass`.

### Metric extraction

The canonical measured series now include:

- `timing/train/total_step_time`
- `timing/train/generation`
- `timing/train/generation_finalize`
- `timing/train/get_logprobs` (resolving the current `timing/train/policy_and_reference_logprobs` alias)
- `timing/train/policy_training`
- `timing/train/prepare_for_generation/transfer_and_update_weights`
- `performance/tokens_per_sec_per_gpu`
- `performance/generation_tokens_per_sec_per_gpu`
- `performance/policy_and_reference_logprobs_tokens_per_sec_per_gpu`
- `performance/policy_training_tokens_per_sec_per_gpu`
- `train/total_num_tokens`
- `train/global_valid_toks`

Every canonical metric must resolve exactly once and contain every measured step. The derived refit metric rejects nonpositive refit durations rather than emitting invalid or infinite values.

`raw_timing.json` exports every canonical measured series under `measured_component_series` and exports the derived value per step under `measured_step_workload`. `raw_timing.csv` exports explicit columns for generation-finalize timing, all four measured throughput rates, and derived refit throughput.

### Workload acceptance

The timing summary writes `workload_equality_required: true`. It writes `timing_summary.json` with the observed ON/OFF token vectors, then raises with both vectors when they differ. Therefore `cutedsl_write_event timing pass` is unreachable for mismatched workloads.

## GREEN evidence

Focused command:

```text
uv run --active --no-sync pytest tests/test_oci_cutedsl_wrapper.py -k 'functional_profile_validator or functional_profile_pass_occurs or extracts_all_required_measured_component_series or timing_summary_enforces_identical_measured_workloads or records_workload_and_normalized_throughput_as_primary or metric_extractor' -q
```

Output:

```text
........
8 passed, 42 deselected in 0.55s
```

Complete wrapper command:

```text
uv run --active --no-sync pytest tests/test_oci_cutedsl_wrapper.py -q
```

Output:

```text
..................................................
50 passed in 2.15s
```

Static verification command:

```text
bash -n experiments/cutedsl_qwen3_30ba3b_oci_1n4g/run_cutedsl_functional.sbatch experiments/cutedsl_qwen3_30ba3b_oci_1n4g/run_cutedsl_matrix.sbatch && uv run --active --no-sync ruff check tests/test_oci_cutedsl_wrapper.py && uv run --active --no-sync ruff format --check tests/test_oci_cutedsl_wrapper.py && git diff --check
```

Output:

```text
All checks passed!
1 file already formatted
```

Exit code: `0`.

## Remaining concern

The accepted fused-kernel regexes are grounded in the existing benchmark attribution contract. They still require confirmation against the first real GB200 `nsys stats` output; this task intentionally did not submit or inspect a remote profile.
