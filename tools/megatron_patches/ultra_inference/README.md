# Megatron inference dependency fixes

These build-time patches target Megatron-LM commit
`3e95097b4a9ab6ea6b2a39eb2b2b24e4a6dfa54a`. Apply the same dependency
fixes for alternating and concurrent Megatron inference before launching either
arm. They preserve the model, sampling settings and speculative depth.

The MTP helpers clear accepted-token state for reused prefill rows. The inference
metadata setter selects one synchronizing dispatcher per main or MTP forward,
rather than only one for the entire model. `SOURCE.json` records the upstream sources,
exact affected files and before/after hashes. The metadata change includes only
the relevant upstream function, rather than unrelated upstream file changes.

Use a separate, clean Megatron checkout. Set `NRL_ROOT` to this repository and
`MCORE_ROOT` to that checkout, then run:

```bash
PATCH_ROOT="$NRL_ROOT/tools/megatron_patches/ultra_inference"
test "$(git -C "$MCORE_ROOT" rev-parse HEAD)" = 3e95097b4a9ab6ea6b2a39eb2b2b24e4a6dfa54a
(cd "$MCORE_ROOT" && sha256sum --check "$PATCH_ROOT/BASE.sha256")
git -C "$MCORE_ROOT" apply --check "$PATCH_ROOT/ultra-inference.patch"
git -C "$MCORE_ROOT" apply "$PATCH_ROOT/ultra-inference.patch"
(cd "$MCORE_ROOT" && sha256sum --check "$PATCH_ROOT/PATCHED.sha256")
```

Install/build the policy environment using this patched checkout through the
existing Megatron-Bridge workspace dependency, or put the checkout on the actual
policy interpreter's `PYTHONPATH`. Verify the imported module paths in that
environment; changing a host checkout alone does not update an installed image.
Apply these patches once, before model processes start. A hash or apply failure
requires reviewing the dependency revision; do not force-apply to another version.
No runtime monkeypatch or model-worker overlay is required.

The NeMo RL native worker separately freezes inference-owned models before graph
capture, while leaving a shared colocated learner trainable. That runtime fix is
included in this branch. Dataset, parallelism, capacity and environment settings
remain recipe responsibilities. These dependency patches alone do not establish
model fit, convergence or benchmark fairness for a new environment.
