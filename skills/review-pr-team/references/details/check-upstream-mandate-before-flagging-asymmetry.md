# check-upstream-mandate-before-flagging-asymmetry

> Before flagging a train/inference (or A/B) config asymmetry as sloppy, check whether upstream validation REQUIRES it — a sibling config that sets both sides symmetrically is not evidence, and the asymmetry is often mandatory

A config value that differs between the training side and the inference side looks like an
oversight, especially when a sibling recipe sets both sides the same. **That similarity is not
evidence.** Before flagging it, grep the upstream config validator for the field: the asymmetry is
frequently *required*, and the sibling is symmetric for an unrelated reason (or incidentally).

**Why:** On PR #3490 I flagged the new recipe for setting `moe_router_dtype: fp32` only under
`mcore_generation_config` while training inherited `fp64`, citing the nearest sibling recipe which
sets fp32 on both. Megatron-LM `transformer_config.py:1483` **raises** unless
`moe_router_dtype == "fp32"` when `transformer_impl == 'inference_optimized'` — which is exactly
what the inference side sets. The asymmetry was mandatory; training legitimately keeps fp64 because
it does not use that impl. Worse, devil-advocate then showed my supporting evidence was also wrong:
the sibling sets fp32 under `policy.megatron_cfg` only, and comes out symmetric *through*
`merged_inference_megatron_cfg`. Two independent errors in one finding, and the sibling comparison
manufactured both.

The same review produced the mirror-image case, which is why this rule is about *checking* rather
than about always deferring: `_apply_moe_config` clearing `moe_router_num_groups` looked like a
silent clobber of a user value, and one agent proposed guarding it against explicit user settings.
Upstream `InferenceTopKRouter` hard-asserts `moe_router_num_groups is None` (`router.py:909`) before
`super().__init__`, and Megatron-Bridge's own Nemotron-3-Nano recipe sets it to `1` — so the reset is
required — removing or guarding it breaks that assert — and the proposed guard would have *caused* the crash it was meant to prevent.

**How to apply:** for any finding of the form "X differs between A and B" or "X silently overrides
The user", the first move is `grep` the upstream config dataclass / `__post_init__` / the consuming
class's `__init__` for that field name, and read what it asserts. If upstream mandates the value,
there is no finding — at most a request for a comment citing the assert. Sibling-config comparison
is a weak signal that must never be the sole basis. Related:
`grep-the-idiom-before-calling-a-convention-wrong` (see [true-but-not-a-finding](../true-but-not-a-finding.md)) (same shape, in-repo),
`convergence-on-finding-is-not-convergence-on-fix` (see [proposed-fixes](../proposed-fixes.md)), `upstream-permits-is-not-tested-here` (see [verify-external-behavior](../verify-external-behavior.md)).
