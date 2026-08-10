# SGLang source patches applied at image build time

SGLang ships here as a PyPI wheel (`sglang==0.5.13`, see the `sglang`
extra in `pyproject.toml`) and its `srt/` tree is pure Python, so a fix that
touches only `.py` files can be applied to the installed package instead of
requiring a forked wheel.

`apply_sglang_patches.py` runs near the end of the release stage in
`docker/Dockerfile`, after every venv under `/opt/ray_venvs` has been
materialised. It applies each `*.patch` in this directory to every sglang
install it finds — the per-worker venvs and the uv cache archive they symlink
into — and exits non-zero unless all of them end up carrying the fix. A silent
no-op is not possible.

Remove a patch from this directory once the corresponding fix is in the pinned
SGLang release.

## 0001-moe-trtllm-bf16-hot-reload-0513.patch

Fixes online weight updates under `moe_runner_backend=flashinfer_trtllm`.

`process_weights_after_loading` rewrites BF16 MoE expert weights into the
FlashInfer TRT-LLM BlockMajorK layout whenever `use_flashinfer_trtllm_moe` is
set, which is true for both `flashinfer_trtllm` and `flashinfer_trtllm_routed`.
The inverse hook was gated on `is_flashinfer_trtllm_routed()` alone, so with
plain `flashinfer_trtllm` the destination parameter stayed in block layout and
every refit died with:

```
The size of tensor a (64) must match the size of tensor b (2048) at non-singleton dimension 2
```

Nobody opts into this: SGLang auto-selects `flashinfer_trtllm` on sm100 for bf16
MoE models when `moe_runner_backend` is left at its `auto` default.

Widening the gate alone is not sufficient — that trades the loud failure for a
parameter left in canonical layout while the kernel reads BlockMajorK, which
corrupts generation silently. The patch also re-derives the kernel layout after
the copy, on all four hot-update paths, from a `finally` so a mid-update
exception cannot leave the weights in the wrong layout.

Upstream: sgl-project/sglang#33743, fixing sgl-project/sglang#27787. Delete this
file once that lands in the SGLang release pinned in `pyproject.toml`.

That PR targets upstream `main`, where the hot-update entry points live in
`model_executor/model_runner_components/weight_updater.py`. This file is the
`srt/`-only variant for **v0.5.13**, which sits between the two: it still keeps
those entry points in `model_executor/model_runner.py` (so the `model_runner.py`
hunk is the v0.5.12.post1 one), but it has already moved to the four-argument
`_maybe_get_cached_w3_w1_permute_indices(..., is_gated_act_gemm=...)` form (so
`base_config.py` and `unquant.py` come from the PR's main-branch version).

Applying the older v0.5.12.post1 backport here with fuzz would silently revert
that call to three arguments and change the permutation for non-gated MoE
models — a wrong-weights bug that no shape check would catch. The one hunk that
neither upstream version supplies, replacing the inline flashinfer block in
`process_weights_after_loading` with the extracted helper, was reconciled by
hand against the v0.5.13 tree.

Verified: applies cleanly to a pristine v0.5.13 tree, round-trips byte-exact,
`process_weights_after_loading` calls the helper exactly once, and all six
patch-added methods are present. The PR also adds a test under
`test/registered/`, which has no counterpart in a wheel install and is therefore
not carried here.

Validated on GB200 against the v0.5.12.post1 backport of the same change
(the v0.5.13 variant has not yet been exercised on hardware): bit-exact generation
(max abs logprob delta 0.0) across an adversarially bucketed push of all 18867
checkpoint tensors, for `flashinfer_trtllm`, `flashinfer_trtllm_routed` and
`triton`; the stock tree reproduces the reported failure under the same
conditions; and 4/4 refits in a 32-node async-GRPO run.
