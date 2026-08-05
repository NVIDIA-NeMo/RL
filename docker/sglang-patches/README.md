# SGLang source patches applied at image build time

SGLang ships here as a PyPI wheel (`sglang==0.5.12.post1`, see the `sglang`
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

## 0001-moe-trtllm-bf16-hot-reload.patch

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
`srt/`-only backport onto `v0.5.12.post1`, where they are still in
`model_executor/model_runner.py` and
`_maybe_get_cached_w3_w1_permute_indices` takes three arguments rather than
four. The executable code is otherwise identical to the PR — verified by
comparing added lines with comments and docstrings stripped, 138 code lines
across the three files, exact match. The PR also adds a test under
`test/registered/`, which has no counterpart in a wheel install and is therefore
not carried here.

Validated on GB200 against this exact backport: bit-exact generation
(max abs logprob delta 0.0) across an adversarially bucketed push of all 18867
checkpoint tensors, for `flashinfer_trtllm`, `flashinfer_trtllm_routed` and
`triton`; the stock tree reproduces the reported failure under the same
conditions; and 4/4 refits in a 32-node async-GRPO run.
