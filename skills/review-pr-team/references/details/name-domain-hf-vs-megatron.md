# name-domain-hf-vs-megatron

> NeMo-RL weight-sync predicates take HF param names, not Megatron ones; any finding about a name-matching gate must say which domain the function receives, or the reader will refute it with the other domain's name

In `nemo_rl/weight_sync/` and the refit paths, every parameter has **two** names, and findings about
name-matching gates live or die on which one is in play:
- **Megatron side** — `task.global_param_name` (e.g. `mtp.layers.0.mtp_model_layer.mlp.linear_fc1.weight`,
  sometimes under a `language_model.` prefix).
- **HF side** — `task.mapping.hf_param`, `str | dict[str, str]` (e.g. `model.layers.61.mlp.gate_proj.weight`).

`is_nccl_reshard_param` is passed **HF names only** — verified by grepping every caller: the
classification site iterates `_iter_params_with_optional_kv_scales` ("Yield exported HF parameters")
and the shard-enumeration site passes `str(hf_param)`. There is no third caller and no gen-side
caller. `_collect_mtp_hf_layer_names` is the mirror image: it matches on `global_param_name`
(Megatron) and stores `_extract_layer_name(hf_name)` (HF) as the key.

The trap: for DeepSeek MTP the Megatron name *does* carry `mtp.` while the HF name does **not** —
it exports as `model.layers.{mtp_layer + num_transformer_layers}.*`. So a claim like "MTP weights
are excluded by the `startswith('mtp.')` check" is true for NemotronH/Qwen3.5 and false for DeepSeek,
and which one you look at determines whether the finding looks right.

**Why:** On PR #3619 I filed a finding that the PR-added docstring on `is_nccl_reshard_param`
overclaims ("Co-trained MTP weights are also routed to misc") because DeepSeek's names aren't caught.
The user checked the Megatron-Bridge source, saw `megatron_param=f"mtp.layers.{mtp_layer}...linear_fc1.weight"`
— an `mtp.` prefix, plainly there — and asked whether the finding was therefore wrong. It wasn't, but
the comment never said which domain the function receives, so the natural reading of the source
appeared to refute it. `devil-advocate` independently confirmed the domain claim on a full caller
grep. The comment had to be rewritten to lead with the domain split and show the two names side by
side from the same `GatedMLPMapping`.

**How to apply:** in any finding about a name gate in the refit/weight-sync paths, state the domain
in the first sentence and prove it with a permalink to the call site (the iterator's docstring or the
`str(hf_param)` argument), not just to the predicate. When both names exist for the same weight, quote
them as a pair from the one Bridge mapping that produces them — the contrast is the argument. Expect
the reader to have the *other* name on screen.
Related: `megatron-expert-rank-layout`, `verify-permalink-lines-programmatically` (see [permalinks-and-anchors](../permalinks-and-anchors.md)).
