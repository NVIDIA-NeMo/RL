# new-option-check-existing-selector

> When a PR adds a new mode/backend/transport, check whether the subsystem already has a selector enum it belongs in — a parallel boolean on a different config node breaks mutual exclusion, discoverability, and centralized validation

Before accepting the *shape* of a new config option, grep for how the subsystem's existing
variants are selected. If there's already a selector (`Literal[...]` enum, registry, factory
dispatch), a new variant added as a **separate boolean on a different config node** is a defect
in three concrete, checkable ways — not a style opinion:

1. **Conflicts become representable.** `new_flag: true` + `existing_selector: <other>` is now
   expressible; ask what it does (usually unvalidated → hijack or silent drop).
2. **Discoverability.** The doc/table/docstring that enumerates the alternatives won't mention it.
3. **Validation scatters.** The existing selector is normalized/validated in one shared place;
   the new boolean gets validated only where its author happened to wire it.

Say explicitly that reshaping a public config key is free before merge and a breaking change
after — that's what makes it worth raising rather than deferring. Recommend ONE fix: make the new
option a value of the existing selector ([recommend-one-fix-not-a-menu](recommend-one-fix-not-a-menu.md)). You may name the cheaper
fallback (reject the conflicting combination + document it) as what to do only if the reshape is
truly blocked — as a fallback, not as a second option on a menu.

**Why:** In PR #2971 (NCCL-reshard refit), the repo already had
`policy.generation.refit_transport: Literal["vllm_s3_sparse","vllm_zmq_sparse","nixl"]`, and
`docs/guides/refit.md` is titled "Choosing a Transport" and states "Because one selector chooses
the transport, sparse delta and NIXL cannot both be active." The PR added a 6th transport as
`policy.nccl_reshard_refit: bool` on a different node, making `nccl_reshard_refit=true` +
`refit_transport='nixl'` expressible and unvalidated (it hijacks the refit and fails on
uninitialized comm groups). The review team's config pass flagged only the `.get(k, False)`
hidden-default nit and missed the shape entirely — the user asked "do the configs make sense?
make sure selecting this is obvious" and that reframing found it.

**How to apply:** for any new mode/backend/transport/estimator option, grep the subsystem for
`Literal[`, `_SUPPORTED_`, `frozenset({`, or a factory dispatch before reviewing anything else.
If one exists, ask why the new option isn't a value in it. Also check whether an existing
selector VALUE already gives the new behavior (e.g. `chat_template: null`). If the mechanism matches,
check the SCOPE before calling it a duplicate: a run-wide setting is not a substitute for a per-sample
one (on #4295 the tokenizer still needed its template for the rest of the blend). Then the right ask is
"reuse the existing loop for the mechanism, and name the new flag for what the data is", not "use the
existing setting". Say why the existing setting does not fit in one sentence; don't build the review
around it. Keep comments about the PR's own code bugs, marked "only needed if you keep it". Related:
[trace-the-feature-across-every-entrypoint-and-format](trace-the-feature-across-every-entrypoint-and-format.md).

**Seen:** #2971 nccl_reshard_refit vs refit_transport (user); #4295 skip_chat_template re-implemented
the per-turn loop that `chat_template: null` uses, with a token-split bug; first framed as "use the
existing setting", which was wrong on scope (pre-templated Energon samples in a mixed blend) — renamed
to `chat_template_preapplied` and moved into the loop (user, twice)
