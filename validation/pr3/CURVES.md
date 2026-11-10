# Streaming LA0 curve comparison

Expanded comparisons with the local runs are available separately, preserving
this original three-series snapshot:

- [Qwen expanded PNG](results/curve-comparison-expanded/comparison.png),
  [CSV](results/curve-comparison-expanded/comparison.csv), and
  [source/coverage JSON](results/curve-comparison-expanded/comparison-summary.json).
- [Gemma expanded PNG](gemma_e2b/results/curve-comparison-expanded/comparison.png).

Both expanded plots mark Shuangy sources in the legends. Qwen additionally shows
fresh LA1 with A_log FP32 refit plus prefill-conv FP32-product fixes from the
start, and legacy non-fresh LA1 with known backend/refit issues. The latter
resumed checkpoint 600 with only the A_log fix from update 601; the red dotted
600 boundary marks that change. It was user-cancelled, so no cancelled work is
restarted. Fresh LA1 is sampled only through the last committed update when the
plot job snapshots its logs. Later resume segments replace replayed earlier
records at the same optimizer step. No series is extrapolated. Policy KL uses a logarithmic axis; gradient norm
uses a log axis or a nonnegative symlog axis when zero-gradient updates exist.
Qwen step time uses a log axis to retain resume/cold-update spikes while showing
ordinary update times clearly. No raw observations are clipped.

Reproduce both expanded figures with `sbatch validation/pr3/plot_expanded_comparison.sh`
using a small CPU-only allocation in the original image. Source CSV/JSONL/log
hashes, native segment lineage, accepted step counts and all eight finite-metric
checks are recorded in the expanded JSON files. These historical comparisons do
not change the failed Qwen BF16 gate, unassessed Gemma fixed-input parity, or
lack of controlled speedup evidence.

Important fix provenance for Qwen: **streaming LA0 used neither the A_log
FP32 refit fix nor the prefill-conv FP32-product fix throughout all 600 steps**.
It did not switch fixes between segments. The fresh LA1 curve uses both from
initialization, so that comparison does not isolate streaming/lookahead. See
[the retrospective audit](REFIT_FIX_AUDIT.md) and its per-job evidence.

The original comparison is described below.

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

Latest snapshot:2026-11-09T01:58:02UTC, refreshed by CPU-only job21539655
(COMPLETED0:0,3m20s). Both historical baselines have500 plotted points and
streaming has600 for every panel. No missing steps, nonfinite metrics or repeated
streaming optimizer-step receipts. Shared statistics stop at500;501..600 are
shaded and show only the observed streaming series. Checkpoint600 is durable,
the same W&B run is finished at600, and all training segments exited normally.

| Window / metric | V1 synchronous | Full-batch LA0 | Streaming LA0 |
| --- | --- | --- | --- |
| Steps481..500 gen_kl_error mean | 0.000933535 | 0.000938107 | 0.001477997 |
| Steps481..500 reward mean | 0.675000 | 0.721875 | 0.701563 |
| Steps481..500 generated tokens/sample mean | 1035.40 | 1063.18 | 982.76 |
| Steps11..500 measured seconds/update mean | 138.46 | 133.37 | 131.48 |

The generated-token workloads differ. Similar measured step time therefore
does not demonstrate streaming speedup, and the higher observed reward is not
a multi-seed convergence claim. See kl_replay/ for the separate fixed-checkpoint,
same-real-rollout diagnosis; the original1% BF16 gradient gate remains failed.

The authorized300->600 continuation completed on2026-11-08. Its genKL means for
301..400,401..500,501..600 are0.001488790,0.001509982,0.001613112; the final20
mean is0.001720197 (maximum in the last100:0.002385499). This is slow upward drift,
not observed explosive growth. The gap relative to the baselines remains and
the original1%BF16 qualification remains FAILED. Last20 streaming reward mean
is0.690625, generated length999.47 and measured seconds/update125.389; there are
no baseline observations at581..600 for a matched comparison.

The loader merges original /300 and resumed /600 commit receipts. Pass
`--max-step 600` to plot_comparison.sh to reproduce. Baselines were refreshed
with unsampled W&B scan_history, not extrapolated. The older300-step snapshot
remains in git history. See RESUME600.md for completion and saved-state evidence.
