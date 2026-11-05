# Megatron verification

This comparison variant uses Bridge PR #6801 and the standard equal-half block
mask, with padding excluded from attention. It preserves original clean tokens
and logical positions and scores all response blocks in parallel.

The RL base is `95095b7124f9b2526c12be772b04fac918ba0a47`. Bridge is
`b9cef7c50612809089d50e7ee3270a005c3343a7`, based on
`e7a6dbfec036e2877a2bf785df94475b1b6c5eb7`, with Core
`e0a4bfed7f1e233b857187ceff0104645858d7d1`. These are the same upstream
baselines used by the original PRs; rebasing onto newer main is separate.

## Checks

- All 167 research unit tests passed in the GPU container.
- All 51 CPU serial-oracle and generation tests passed, including unaligned
  prompts, partial final blocks, score agreement, and gradient agreement.
- Bridge passed 193 unique focused unit tests across the full run and targeted
  retest. Compiled CUDA attention outputs and gradients match the dense oracle;
  excluded padding cannot influence real queries.
- Bridge all-files pre-commit, RL changed-file formatting/lint hooks,
  `git diff --check`, and `uv lock --check` passed (593 packages).
- The scoped Pyrefly run is not clean: ten worker/config/interface typing errors
  remain. This is not a full typecheck pass.

## Runtime

The provisioned runtime uses Python 3.13.14, PyTorch 2.10.0+cu129,
Transformer Engine 2.14, Transformers 5.16.1, ModelOpt 0.40, and Ray 2.54.
The lock selects PyTorch 2.13 and TE 2.18; a full rebuild of that environment
has not been verified. TP>1 diffusion training, CP>1 diffusion training,
checkpoint resume, and reward convergence are not established by these checks.

Two-GPU Standard and Fast Sudoku smoke results will be recorded here after
completion. Historical results for RL #4258 and Bridge #6771 remain in those
branches and are not evidence for this variant.
