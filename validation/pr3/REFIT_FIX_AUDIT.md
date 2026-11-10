# Qwen streaming LA0: refit and convolution fix audit

The completed `yuekaiz5090explorela0s42` run did **not** enable either the A_log
FP32 refit fix or the prefill-convolution FP32-product fix, throughout steps
1–600. There is no fix-regime switch between its seven executed training jobs.
This conclusion uses frozen source, required startup guards, preserved launch
artifacts, and the original image source; it is not a new per-rank runtime trace.

| Updates | Jobs | A_log FP32 refit fix | Prefill conv fix |
| --- | --- | --- | --- |
| 1–300 | 21331883, 21331884, 21331885, 21331930 | Not enabled | Not enabled |
| 301–600 | 21509383, 21509384, 21509385 | Not enabled | Not enabled |

Job 21509386 skipped before container startup because 600 was already complete.

The worker hash stayed
`7f59824258aafb13f5540810776dfa18c86a623856c9c89e4c6e26881259f74d`.
`run.sh` checks this hash before starting every training segment. Its
`_refit_tensor_dtype` preserves FP32 `e_score_correction_bias`, but has no
`A_log` exception. Both payload export and refit metadata use this helper, so
A_log follows the configured BF16 refit dtype; FP32 receiver storage does not
recover bits already lost in transit. The separate fresh experiment adds A_log
to that exception in its own worker. Those changes were not ported to PR3.

Every saved `driver_command.sh` selects this PR3 `run.sh`; none of the seven
job log directories contains `setup_command.sh`. All seven native log hashes
match the previously saved completed-600 audit. The original image's vLLM
`_causal_conv1d_fwd_kernel` still has `acc += matrix_x * matrix_w`, without the
explicit FP32 operand casts. Its source SHA
`044d005cfe59fd0818ed421274e04f3e8dd679b8cdb64fca9f4422f2643484b2`
exactly matches the fresh experiment's recorded **pre-patch** source. Fresh
patching changes only its private container, not the shared image or this run.

See [machine-readable audit](results/qwen-streaming-fix-audit.json) for per-job
lineage and hashes. The original image was read with `unsquashfs -cat`; no
allocation, training, or numerical replay was started.

The expanded chart's fresh LA1 line uses both fixes from initialization, while
streaming LA0 uses neither. Their difference therefore cannot isolate streaming
or lookahead. Likewise, legacy non-fresh LA1 has a separate repair/recovery
history, as already marked at checkpoint 600. This audit does not determine how
much of the observed KL gap is caused by either fix, and does not change the
failed original BF16 numerical qualification.
