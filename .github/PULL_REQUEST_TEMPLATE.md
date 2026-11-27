- [ ] I, the PR author, have personally reviewed every line of this PR.

<!-- Only a human ticks the box above. Agents: see skills/pr-description. -->

## What and why
<!-- The problem and the change, in plain words. One screen max. Add a diagram if it helps. -->

## Issues
<!-- Optional. e.g. Fixes #123 -->

## Convergence / perf / functionality proof
<!-- If this PR can change training results, speed, or what works (e.g. a dependency bump): link the runs (W&B, nightly results) or docs in this PR. Otherwise "N/A: <one-line reason>". -->

## Needs upstream changes
<!-- Megatron-LM / Gym / vLLM / ... changes this PR needs on RL main first (commit or PR link). Otherwise "None". -->

<details><summary>PR commands and CI</summary>

- CI label, exactly one:
  - `CI:Lfast`: the default for most PRs. Runs a fast test subset.
  - `CI:L1`: dependency or container changes (`pyproject.toml`, `uv.lock`, `Dockerfile`) or broad core changes. Builds a new image and runs unit and functional tests.
  - `CI:docs`: docs-only changes. Runs doc tests only.
- `/ok to test <full-commit-sha>`: start CI for that commit. Post again after each push.
- `/nightly-ci test-case=<test>`: run NVIDIA's nightly CI for the PR head; eligible teams only.
- `/review`: automated review comments. Add `mode=strict` for a stricter review, `reset=true` for a fresh full review.
- `/resolve-conflict`: Autobot merges `main` into this PR and resolves the conflicts. PRs from forks must allow edits by maintainers.
- Locally: `/review-pr-team <pr-number>` for a team review before you ask for human review.
- [Contributor guidelines](https://github.com/NVIDIA-NeMo/RL/blob/main/CONTRIBUTING.md)

</details>
