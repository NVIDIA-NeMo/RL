# compare-guards-with-sibling-bypass-path

> When a PR adds a second path that writes directly into a third-party library's internal storage, diff its guard list against the existing sibling's — the sibling's rejections encode hard-won knowledge, and a missing one is often a silent-wrong bug

If a PR bypasses a library's public loader to write directly into its internal tensors/state,
find the sibling path that already does this and **diff the two guard lists**. An existing
implementation's hard-rejects are empirical knowledge about states the library can be in — they
were almost certainly added after someone hit them. A guard present in one and absent in the
other is a finding, and the sibling's own error message usually supplies both the rationale and
the wording to reuse.

Prioritize guards whose failure is **silent** (wrong results) over ones that raise. Watch
especially for guards about *state that can change after a metadata exchange* — any design that
computes an ownership/layout map once at setup and reuses it per-iteration shares that
vulnerability, even when the two implementations are otherwise unalike.

Be fair about what is and isn't duplicated: two paths can solve the same problem with genuinely
different strategies (e.g. delivery-driven vs mesh/placement-driven) and still both hardcode the
same *layout constant*. Recommend sharing the constant/derivation, not merging the algorithms —
otherwise the author will (correctly) push back that they aren't the same thing.

A dead variable whose comment describes a check that was never implemented is strong evidence the
author started the guard and abandoned it — call it out together with the missing guard, not as a
separate lint nit.

**Why:** In PR #2971 the NCCL-reshard path wrote destination shards straight into vLLM's fused
`w13_weight`/`w2_weight`. `nemo_rl/models/generation/vllm/refit_loader.py` (the NIXL sharded-expert
path, already on main) does the same and rejects transposed weights, non-Triton/quantized MoE
backends, and `enable_eplb` — the last with "ownership can change after metadata exchange." The new
path had none of them, while building its expert→rank map once at setup, so EPLB would silently
send experts to their setup-time owners. Grep found a dead `vllm_modules` local whose comment said
"detect the selected backend off the FusedMoE layer" — the abandoned guard. The user asking "does
this duplicate the nixl/checkpoint-engine weight loader?" is what surfaced all of it.

**How to apply:** grep the new direct-write target (e.g. `w13_weight`, a private attr) across the
repo to find sibling writers; list each sibling's `raise`/`ValueError` conditions and check the new
path for each; verify reachability of any config-driven state (e.g. `**vllm_kwargs` splatted into
the engine makes arbitrary engine flags expressible). Related:
[trace-the-feature-across-every-entrypoint-and-format](trace-the-feature-across-every-entrypoint-and-format.md), [new-option-check-existing-selector](new-option-check-existing-selector.md).
