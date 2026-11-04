# Historical metric checks, 2026-11-03

Four endpoint checks reject runs whose curves remain consistent with the accepted
historical variation. The proposed bounds add empirical headroom; they do not
establish a 95% or 99% future pass rate. The DAPO probability-error failure remains
unexplained, so its median gate stays unchanged.
The investigation is tracked in [issue #5071](https://github.com/NVIDIA-NeMo/RL/issues/5071).

## Replay results

These counts apply the same check to each retained metric history. They are not
the original overall CI outcomes. Every count includes the reported failing run.

| Test | Observed statistic range | Old check | Proposed check | Old failures | Proposed failures |
|---|---:|---|---|---:|---:|
| Qwen3 32B to 4B distillation | 0.0625–0.19140625 | accuracy at 20 > 0.10 | > 0.05 | 4/13 | 0/13 |
| Mistral NeMo DPO | 0.53349205–0.62067070 | loss at 100 < 0.60 | < 0.65 | 3/30 | 0/30 |
| Qwen2.5 1.5B YaRN 256K | 0.375–0.9375 | reward at 30 > 0.50 | > 0.30 | 9/31 | 0/31 |
| Flow-GRPO Qwen-Image OCR | +0.01764232–+0.12585962 | reward at 59 > reward at 0 + 0.03 | reward at 59 > reward at 0 | 1/14 | 0/14 |
| DAPO Mistral Medium 3.5 128B | 1.00377162–107102.53936 | median probability error < 1.1 | unchanged | 2/7 | 2/7 |

The four repaired predicates pass all 88 retained histories when replayed with
`tests/check_metrics.py`. This replay checks only the listed predicates, not the
other checks in each script. No new GPU training runs were launched.

## Why these changes

- **Distillation:** low and high accuracy recur throughout the history. Two
  runs at the same source revision and with identical recorded data, teacher,
  policy, loss and distillation settings end at 0.08203125 and 0.1640625. The
  0.05 floor sits 0.0125 below the observed minimum; the loss gates still apply.
- **DPO:** full loss curves nearly overlap, including the large batch-dependent
  oscillations. Endpoint failures are interspersed with passes. A 0.65 ceiling
  leaves 0.02932930 above the largest observed endpoint, while the existing
  initial-loss gate still requires step-1 loss between 0.680 and 0.71.
- **YaRN:** endpoint rewards vary without a clean chronological break. A matched
  source/config pair ends at 0.40625 and 0.625. The 0.30 floor leaves 0.075 below
  the observed minimum; accuracy and probability-error checks still apply.
- **Flow-GRPO:** all 14 final rewards exceed their own initial rewards. The smoke
  check now requires a positive gain, rather than a fixed 0.03 gain. Flat or
  declining reward still fails. This accepts small improvements that the old
  check rejected; it does not certify a minimum useful improvement.
- **DAPO:** isolated large spikes occur even in earlier runs whose medians pass.
  Recent runs have more affected steps, and 2 of 7 medians fail. A passing run
  between failures does not rule out a distribution change. Keep the median
  bound until this recurring mismatch is understood.

The looser bounds also accept future values between the old and new thresholds.
That is an intentional reduction in endpoint sensitivity, based on the accepted
historical behavior, not proof that every value in those bands is healthy.

## Evidence and limits

Each directory contains `comparison.png`, all selected metric values in
`curves.csv`, per-run results in `checks.csv`, and the exact statistic/window and
threshold in `manifest.json`. Run aliases are local to each test and ordered by
creation time; red identifies the reported failure. Lines join recorded points
for visualization only. Replay uses exact recorded steps without interpolation.

| Test | Dates (UTC) | Figure | All curve values | Per-run checks |
|---|---|---|---|---|
| Distillation | Sep 19–Oct 31 | [Plot](distill/comparison.png) | [CSV](distill/curves.csv) | [CSV](distill/checks.csv) |
| DPO | Oct 15–31 | [Plot](dpo/comparison.png) | [CSV](dpo/curves.csv) | [CSV](dpo/checks.csv) |
| YaRN | Oct 15–31 | [Plot](yarn/comparison.png) | [CSV](yarn/curves.csv) | [CSV](yarn/checks.csv) |
| Flow-GRPO | Oct 16–31 | [Plot](flow/comparison.png) | [CSV](flow/curves.csv) | [CSV](flow/checks.csv) |
| DAPO | Oct 6–31 | [Plot](dapo/comparison.png) | [CSV](dapo/curves.csv) | [CSV](dapo/checks.csv) |

The source is the retained W&B history for each test, including verified earlier
test names. Histories span source revisions and are not controlled experiments.
The selection is the available comparison set from the investigation, not a
claim that every run ever executed was found. One earlier YaRN run stopped at
15 steps and is excluded because it lacks step 30. No retained run is missing
the data needed for its listed predicate.

For the four LLM tests, W&B's `nemo_rl/step` supplies the training-step keys;
Flow-GRPO uses its zero-based `_step`. DAPO medians use all 20 steps, 1 through 20
inclusive. Distillation logs validation at steps 10 and 20; Flow-GRPO logs
validation at 0, 29 and 59. The CSVs retain all recorded values for the plotted
metrics, not only the endpoints.

Recorded configurations support matched-pair comparisons for distillation and
YaRN. Flow-GRPO has no populated W&B configuration, so its comparison relies on
the test identity and recorded metric history. Differences in code, environment
or unrecorded state can contribute to variation. The threshold choices fit the
accepted observed sample, including the current failures; normal CI should
provide prospective validation.
