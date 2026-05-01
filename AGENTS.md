# NeMo-RL

NeMo-RL is an RLHF training framework built on Ray and PyTorch (FSDP2 / Megatron-Core). It supports algorithms like GRPO, DPO, and SFT for LLMs and VLMs.

## Skills

Coding guidelines and operational procedures are organized as Claude skills in
`skills/`. **Always read the relevant `SKILL.md` before starting any task it
covers — skills are mandatory context, not optional background reading.**

**Workflow — mandatory order for every task:**
1. **Pull information first.** Read the commit, PR, error log, file, or
   whatever artifact the task is about. Do not reason about it yet.
2. **Select and invoke the skill.** Based on what you just read, identify
   the relevant skill and invoke it before forming any answer or plan.
3. **Answer or implement.** Only after the skill is loaded, use its context
   to reason, diagnose, or write code.

Never skip or reorder these steps. Do not wait for the user to name the right
skill keyword — infer it from the artifact you read.

## Code Review

Use `/review-pr <pr-number>` for interactive local PR review.

When reviewing code, follow these principles:

- **Be concise and actionable.** Focus on bugs, logic errors, missing tests, outdated docs, and guideline violations.
- **Do NOT flag:** style/formatting (linters handle it), minor naming suggestions, architectural opinions, or performance unless there is a clear measurable issue.
- **High confidence only.** Only flag issues you are confident about. If unsure, skip it.
- **Verify upstream API usage.** When code calls into megatron-bridge, megatron-lm, automodel, or gym APIs, look up the actual API to verify correct usage. Evaluate each such call with scrutiny — don't assume the author got the signature, return type, or semantics right.
- It is perfectly acceptable to have nothing to comment on. Say "LGTM" if so.

## Kubernetes / nrl-k8s

For launching, monitoring, stopping, and debugging NeMo-RL recipes on Kubernetes, see the skill at @skills/launch-nemo-rl/SKILL.md.
