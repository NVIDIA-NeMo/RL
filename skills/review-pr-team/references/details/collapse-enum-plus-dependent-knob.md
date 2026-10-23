# collapse-enum-plus-dependent-knob

> An enum mode + dependent numeric knob guarded by a consistency validator usually wants to be one knob (0/None = the degenerate mode); the user prefers proposing the collapse

When a new config pairs a mode enum with a dependent numeric knob and needs a validator to reject their degenerate combination (e.g. `on_data_exhausted: fail_fast|skip` + `max_skipped_prompts` with "skip requires >= 1"), propose collapsing to the single numeric knob where the zero/None value *is* the other mode (`max_skipped_prompts: 0` = fail fast). The user asked for exactly this on PR #3470.

**Why:** The validator exists only to forbid a state the schema shouldn't be able to express. One knob makes the illegal state unrepresentable, deletes the validator + its test + the internal bool it compiles into, and the name self-documents ("allow at most 0 skips" reads as fail-on-first). Check losslessness first: enumerate the meaningful states of the pair and confirm they map 1:1 onto the single knob's range.

**How to apply:** In config review, treat a `model_validator` that rejects a combination of two new sibling fields as a design smell pointing at a collapse, not just a guard to affirm. Related: The user also prefers fault-tolerance/resiliency keys gathered into one nested block (common keys top-level, path-specific keys in named sub-blocks like `native:`/`nemo_gym:`) rather than dangling at the parent config root — they overrode a "guard, don't nest" adjudication on #3470 with that rationale.
