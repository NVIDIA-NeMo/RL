---
name: pr-description
description: How to write and keep up to date a NeMo-RL pull request description. Covers the template sections, the human-only review box, proof and upstream sections, and what must stay out of public PRs.
when_to_use: Opening a PR; writing or editing a PR description; updating a description after new pushes; a red 'Check PR description' check; 'PR description', 'PR body', 'PR template', 'review box', 'personally reviewed every line'.
---

# PR Description

The template is [`.github/PULL_REQUEST_TEMPLATE.md`](../../.github/PULL_REQUEST_TEMPLATE.md). The `Check PR description` workflow runs [`.github/scripts/check_pr_description.py`](../../.github/scripts/check_pr_description.py), which fails when the review box is unticked, a required heading is missing, or "What and why" is empty. It is a warning for now, not a required check.

## Rules

1. **Never tick the review box.** "I, the PR author, have personally reviewed every line of this PR." is for the human author only. Do not tick it, untick it, reword it or move it. If the check is red only because the box is unticked, tell the human; do not fix it yourself.
2. **Keep every template section.** Never delete a heading. Write `N/A: <reason>` or `None` instead.
3. **What and why: plain words, one screen at most.** State the problem first, then the change. Add a diagram (Mermaid or an image) when a flow or data path changes.
4. **Convergence / perf / functionality proof: be honest.** Link W&B runs, nightly results, or docs in this PR. A change that can move training results, speed, or what works (a dependency bump, loss or sampling math, a new model) needs proof. Otherwise write `N/A: <one-line reason>`. Never invent links or numbers.
5. **Needs upstream changes: list them or write `None`.** Name each Megatron-LM / Gym / vLLM / other change this PR needs on RL main, as a commit or PR link. Free-form lines are fine.
6. **Keep it current.** When new pushes change what the PR does, update What and why, proof and upstream to match. You may rewrite the author's text to do so. If the review box is already ticked when you change the text, tell the human that their tick no longer covers the new text.
7. **Act only when asked.** Open, push to, or edit a PR or its description only when the human asks.
8. **Public PRs: nothing internal.** This applies only when the target repo is on GitHub and public; check with `gh repo view <owner>/<repo> --json visibility`. A GitLab merge request or a private repo may keep internal links and names. For a public PR:
   - No internal URLs (internal Git servers, intranet, dashboards). Refer to an internal pipeline or job by its number only, e.g. "pipeline 48213", with no domain.
   - No cluster names, hostnames, or cloud provider / datacenter locations. Name the GPU SKU only, e.g. "H100", "GB200".
   - No colleagues' emails or chat handles, no ticket IDs, credentials, or paths that show internal infrastructure.
   - No unreleased model names or internal numbers.

## Changing the template

The template, the check and this skill must agree. When you add, rename or remove a section, or change the review box wording, update all of them in the same PR:

1. [`.github/PULL_REQUEST_TEMPLATE.md`](../../.github/PULL_REQUEST_TEMPLATE.md).
2. [`.github/scripts/check_pr_description.py`](../../.github/scripts/check_pr_description.py): `REVIEW_BOX_TEXT` and `REQUIRED_HEADINGS`, plus its tests in `tests/unit/test_pr_description_check.py`.
3. This skill's rules and the section names they mention.
4. For a new box wording: the never-tick line in `AGENTS.md` (`CLAUDE.md` links to it) and in [`contributing`](../contributing/SKILL.md).

Then open a test PR with the new template and confirm the check passes when filled in and fails when a section is missing.

## Fixing a red `Check PR description`

Read the check's log. Each line names one problem: add the missing heading, fill "What and why", or, for the review box, ask the human to tick it (rule 1).
