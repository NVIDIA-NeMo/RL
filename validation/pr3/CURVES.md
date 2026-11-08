# Streaming LA0 curve comparison

The eight panels compare reward, generated length, gen_kl_error,
policy_kl_error, policy loss, approximate entropy, preclip gradient norm and
step time. Faint lines are raw values; solid lines are trailing 10-step means.
Policy KL uses a logarithmic y-axis to retain spikes while keeping the usual
range readable. All other panels use linear axes.
The plot stops each series at its latest committed optimizer step. Shading marks
steps beyond shared observed coverage; absent series are never extrapolated.

Runs:

- V1 synchronous: `ys_fishcool-nvidia/nemo-rl-automodel-sc/2a121400`.
- SC full-batch LA0: `ys_fishcool-nvidia/nemo-rl-automodel-sc/c926b37d`.
- SC streaming LA0: `yuekaiz/nemo-rl-automodel-sc/yuekaiz5090explorela0s42`.

The baselines are existing runs, not additional experiments. They use the same
model, data recipe, global batch and GPU counts; this is a historical comparison,
not an isolated same-image numerical or performance A/B. The original streaming
1% BF16 gradient qualification remains failed, as described in BF16_NUMERICS.md.
The long run is exploratory evidence and does not replace that qualification.

Generate a fresh snapshot from a small CPU allocation, using image1030:

```bash
sbatch --output=validation/pr3/slurm/curves-%j.out \
  --error=validation/pr3/slurm/curves-%j.err validation/pr3/plot_comparison.sh
```

The script reads the baseline credential only inside the process; it never
exports or saves it. W&B `scan_history` exports all records rather than sampled
history. Streaming records come from native SC logs and are admitted only after
their commit receipt; multiple continuation segments are merged by optimizer
step. The script rejects gaps, missing plotted metrics and nonfinite values.
Use `--offline` to reuse the previous validation's cached baseline exports.

Outputs under `validation/pr3/results/curve-comparison/`:

- `comparison.png` and `comparison.svg`: the eight-panel figure.
- `comparison.csv`: selected values by run and optimizer step.
- `*-history.json`: the selected unsmoothed source records.
- `comparison-summary.json`: timestamp, source provenance, coverage and statistics
  over the common observed interval. Last-20 means end at the same optimizer
  step for all runs; timing statistics additionally provide steps11 through the
  common endpoint to reduce initial warmup effects.

Step time retains measured checkpoint and first-step overhead, and excludes
queue waits and initialization outside the measured training step. Online
validation accuracy is absent from these runs, so no accuracy panel is fabricated.
The comparison is single-seed and cannot establish convergence equivalence.

Latest snapshot:2026-11-06T14:07:01UTC, refreshed by CPU-only job21389289
(COMPLETED0:0). All three runs have300 plotted points for every panel; no missing
steps or repeated streaming commit receipts. Original streaming checkpoint300
is durable and final training segment21331930 completed normally.

| Window / metric | V1 synchronous | Full-batch LA0 | Streaming LA0 |
| --- | --- | --- | --- |
| Steps281..300 gen_kl_error mean | 0.000811697 | 0.000845502 | 0.001570812 |
| Steps281..300 reward mean | 0.628125 | 0.621875 | 0.709375 |
| Steps281..300 generated tokens/sample mean | 1243.67 | 1211.59 | 1059.80 |
| Steps11..300 measured seconds/update mean | 138.58 | 133.59 | 133.30 |

The generated-token workloads differ. Similar measured step time therefore
does not demonstrate streaming speedup, and the higher observed reward is not
a multi-seed convergence claim. See kl_replay/ for the separate fixed-checkpoint,
same-real-rollout diagnosis; the original1% BF16 gradient gate remains failed.

The same exploratory run was authorized to continue300->600 on2026-11-08.
The loader merges original /300 and resumed /600 commit receipts. Once600 is
durable, pass `--max-step 600` to plot_comparison.sh.
Baseline coverage is taken from actual fetched records; shared-window statistics
stop at the shortest series. The existing300-step snapshot remains historical
until a fresh plot is generated. See RESUME600.md for the active job chain.
