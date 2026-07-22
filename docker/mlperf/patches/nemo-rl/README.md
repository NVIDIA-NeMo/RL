# Carried NeMo-RL patches

Patches in this directory are checked against the immutable
`NEMO_RL_REVISION`. They are applied when absent, skipped when already present,
and otherwise attempted with `git apply --3way`; a real conflict fails the
image build. The current canonical Qwen branch already contains this stack,
while the historical `41352f74610762cd2f3014533be27f4c30f8524e` source needs
the patches applied during the build.

## Current stack

- `01-optimizer-dtype-boundary.patch` is inherited unchanged from `main`. It
  keeps optimizer dtype names
  serializable in the high-level config and converts them to `torch.dtype`
  only when constructing Megatron Core's optimizer config; the async production
  recipe does not activate its four precision-aware dtype fields.
- `02-grouped-validation-pass-at-k.patch` repeats each validation prompt,
  calculates observed pass@k, pass^k, and average pass@1, and exposes pass@k
  through the existing `accuracy` key consumed by MLPerf. It is the
  still-missing portion of
  `a0210c70cb397d83ab2be8dbc8c200775e7930fd`, adapted to the historical pin.
- The async validation-pause race fix is not patched: the source ancestry contains
  the exact upstream commit
  `9402b7de86c4c415853824a8bb6e4a224d533a84`.
- Prefix-cache invalidation is not patched: the source ancestry contains
  `3a643e753dc424336989e8cc4dba5c5e7e5ea6a4`, which resets the cache after
  every refit path. Adding the corresponding hunk from `a0210c70c` would
  duplicate that reset.
- `04-delayed-periodic-validation.patch` ports the legacy synchronous,
  TransferQueue synchronous, and asynchronous paths from optimized
  `33bae24a`. It skips periodic validations before `grpo.val_start_at` without
  shifting the existing `step % val_period` cadence. The exemplar default is
  zero, preserving prior behavior, and `val_at_end` remains independent.

The source ancestry already handles missing, invalid, or failed
per-instance rollouts as loud masked zero-reward trajectories, so no duplicate
NeMo-RL patch is carried for that behavior. The bounded Gym disconnect retry is
retained under `../gym/`. Qwen reasoning/tool parsing and multi-turn
suffix-prefix repair also remain in the source tree. The FLA Blackwell
GDN installed-package patch lives under `../installed/`; the Dockerfile applies
the TE SM103 FA2 allowlist change only to the Megatron policy actor's installed
module.

## Lifecycle

1. Compare each proposed change against the exact Docker source revision before
   adding it here.
2. Every patch header states its origin, purpose, and omitted equivalent
   upstream/source behavior.
3. When the source revision advances, reverse-check each patch and remove it when
   equivalent behavior is present.
4. Patch-application failure is a hard image-build failure; never fall back to
   a broad source overlay.

`../gym/` holds patches for the pinned NeMo Gym submodule and `../r2e/` holds
the one post-clone nv-R2E-Gym patch. They follow the same header/lifecycle
rules.
