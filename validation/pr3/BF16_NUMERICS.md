# BF16 qualification: original gate failure and additional optimizer oracle

Original design contract v1 remains recorded in ACCEPTANCE.md. It has **not passed**
for BF16 streaming. Do not relabel its failures as FP32 failures or environment skips.

Observed on the local two-layer dense model, same inputs/state/order, two dispatches:

- FP32: gradients differ by about 1e-8; parameters/optimizer/scheduler comparisons pass.
- BF16: gradient relative L2 about 0.00655, cosine about 0.999979, norm error below
  0.0001. These pass the predeclared gradient limits.
- BF16 forward with FP32 AdamW parameters: some parameter differences reach about
  4e-6 at LR2e-6. This fails the original 1e-7 absolute parameter limit.

Moving the denominator outside backward changes where rounding occurs in BF16.
Adam's normalization near zero can amplify a small gradient sign difference into
an update difference approaching 2*LR. This is a hypothesis to verify with raw
gradient/state evidence, not a reason to silently weaken assert_close.

Additional contract v2, fixed before its first execution:

1. Preserve the original BF16 gradient/direction/norm gates unchanged.
2. Keep FP32 worker parity and full-batch exact regressions unchanged.
3. Continue measuring and reporting full-batch-versus-streaming parameter and
   optimizer differences, with explicit booleans for the original v1 gates.
4. Starting again from the same initial model/optimizer/scheduler state, inject
   the captured, normalized streaming gradients and run exactly one independent
   clip/optimizer/gate-bias/scheduler update outside the SC lifecycle. Its complete
   parameters, optimizer state and scheduler state must match the actual streaming
   commit **exactly**. Restore post-backward buffers/router loads for that oracle.
   FusedAdam's public state_dict dequantizes compressed moments. Its raw state,
   int16 master remainders and private scales are therefore captured separately,
   restored exactly before replay, and compared exactly against this oracle.
5. Scalar optimizer counters, metadata, tree keys/shapes/dtypes and scheduler
   progression remain exact. Never accept nonfinite gradient/state values.
6. Retain selected gradient sign/parameter-update examples and all raw differences.
   The oracle separates optimizer lifecycle correctness from BF16 backward rounding;
   it does not prove identical BF16 trajectories or convergence.

Result reporting must distinguish v1 numerical parity, v2 optimizer execution
correctness, and real-run convergence. A passing v2 oracle must not be described as
passing the original strict BF16 parameter gate. Long-run qualification remains
necessary, and only streaming LA0 is authorized for the final long run.

## Qwen3.5 EP16: original gradient gate also failed

Job21330640 reached real Qwen3.5-35B-A3B-Base training with EP16, activation
checkpointing and the recipe's FusedAdam. The first token-level comparison
failed: observed per-rank relativeL2 approximately1.1–1.5% (limit1%), with some
cosine values below.9999. Norm-size differences remained below1%. No optimizer
oracle pass or long-run qualification is inferred for this configuration.
The dependent300-step jobs were cancelled; ep16-failure.json records the failure.

A separate diagnostic freezes an exact criterion: an ordinary full-input V1
backward with placeholder denominators and independent clip-entry normalization
must match SC streaming in preclip gradients, parameters, public/raw optimizer
state and scheduler exactly. Its ordinary early-normalization reference still
records the unchanged1%/.9999/1% gates. This diagnostic passed tiny EP2/FusedAdam
with activation checkpointing and both objectives. EP16 diagnostic21331026 also
passed all16 ranks and both objectives (one update each): gradients, parameters,
public/raw FusedAdam state/scales and scheduler match exactly. Global original
normalization differences are1.343% token-level and1.322% sequence-level, so the
original1% limit remains failed even when measured on the full sharded vector.
Its output does not qualify the default training launcher. Global statistics are
additional measurements, not a replacement for failed rank gates.

The fixed-input result isolates normalization placement: the independent reference
does not split the batch yet reproduces the streaming gradient and update exactly.
This rules out a dispatch/LP/optimizer-lifecycle discrepancy for the tested inputs;
it does not prove the two BF16 trajectories converge identically. A separate,
explicit exploratory-train lane collects the already-requested300-step LA0 evidence
under a distinct W&B identity, preserving the failed gate. No tolerance is raised.
