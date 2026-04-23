# NeMo-RL

NeMo-RL is an RLHF training framework built on Ray and PyTorch (FSDP2 / Megatron-Core). It supports algorithms like GRPO, DPO, and SFT for LLMs and VLMs.

## Coding Guidelines

Coding guidelines are organized as Claude skills in `.claude/skills/`. Each skill covers a specific topic (style, config conventions, error handling, testing, copyright, docs).

## Code Review

Use `/review-pr <pr-number>` for interactive local PR review.

When reviewing code, follow these principles:

- **Be concise and actionable.** Focus on bugs, logic errors, missing tests, outdated docs, and guideline violations.
- **Do NOT flag:** style/formatting (linters handle it), minor naming suggestions, architectural opinions, or performance unless there is a clear measurable issue.
- **High confidence only.** Only flag issues you are confident about. If unsure, skip it.
- **Verify upstream API usage.** When code calls into megatron-bridge, megatron-lm, automodel, or gym APIs, look up the actual API to verify correct usage. Evaluate each such call with scrutiny — don't assume the author got the signature, return type, or semantics right.
- It is perfectly acceptable to have nothing to comment on. Say "LGTM" if so.

## Kubernetes / nrl-k8s

For launching, monitoring, stopping, and debugging NeMo-RL recipes on Kubernetes, see the skill at `tools/nrl_k8s/.claude/skills/launch-nemo-rl/SKILL.md`.
