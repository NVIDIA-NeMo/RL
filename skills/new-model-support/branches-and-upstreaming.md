# Support branches and upstreaming

How new-model work is delivered: support first on per-model branches, merge into main later. This describes the current practice. It is a workaround for the constraints below, not a target design.

## Why support starts on a branch

- NeMo-RL bumps AutoModel at AutoModel releases, not to the latest `main`. A model that AutoModel added after its last release cannot run on NeMo-RL `main` until the next bump.
- RL needs things SFT and inference do not: colocated training and generation, memory reductions that SFT never turns on (offload, activation checkpointing, refit buffers), and the refit path. Most bring-up fixes land in AutoModel for these reasons, fewer in vLLM.
- Those fixes are often model-specific or touch shared code in ways the owner will not merge quickly. Waiting for an upstream fix would block the support.

So every bring-up runs on two branches, plus a vLLM pin:

| Repository | Branch | Base |
|---|---|---|
| NeMo-RL | one support branch per model (for example `<model>-support`) | NeMo-RL `main` at the start of the bring-up |
| AutoModel | one dev branch per model, when the model needs AutoModel changes (pushed to the AutoModel repository as `rl/<model>-support`); otherwise a pinned upstream commit | the AutoModel commit that added the model |
| vLLM | a commit wheel; a fork branch only when a fix cannot wait | the commit that added the model |

The NeMo-RL branch points its AutoModel submodule at the AutoModel dev branch (or commit) in `.gitmodules`, with a comment saying why.

## Keep the branches easy to split later

The later merge into main is the expensive part. Make it cheaper while developing:

- One concern per commit. Do not mix a generic refit fix, a memory fix and model-specific code in one commit.
- Keep AutoModel changes as commits on top of the recorded base, so `git log <base>..` lists exactly what the model needed.
- Gate generic fixes by configuration or capability, not by model name, so they can merge alone. Their unit tests are written when they are split out for main, not during the bring-up.
- Do not expect other models to keep working on the support branch. Dependency moves made for the model (for example a Torch ABI bump that drops some source-built extensions) are recorded in the change list as branch-only regressions and resolved at merge time.
- Keep a change list in the support branch's notes (or the PR draft). One row per change:

| Change | Repository | Generic or model-specific | Why RL needs it | Upstream status |
|---|---|---|---|---|
| keep FP32 parameters FP32 during refit | NeMo-RL | generic | vLLM keeps them in FP32 | not proposed |
| ... | AutoModel | model-specific | ... | waiting for AutoModel bump |

The change list is the input for the later split; it also tells the next bring-up which fixes already exist on another model's branch.

## Upstream problems: record, do not file

When a problem is on the AutoModel or vLLM side, fix or work around it on the dev branch and add it to the change list with a minimal reproduction. Do not open issues or PRs automatically: a human first checks that the problem is real and not caused by the integration, and decides whether and how to split it into issues. An agent following this skill drafts the issue text in the change list and stops there. The same holds for PRs: when a human approves opening a PR, that means a draft PR of the NeMo-RL support branch, never a PR on AutoModel or vLLM.

## Announce from the branch

After stage 4 passes, announce support with links to the branch, not to `main`:

- Add a dated entry to the README news section, linking the support branch and the model guide on that branch, in the form already used there: `* [MM/DD/YYYY] <Model> RL is supported [on this branch](<branch tree URL>) with the following [doc](<guide URL on the branch>).` The news entry is the only change that goes to `main` at this point.
- The guide on the branch states the AutoModel branch or commit and the vLLM commit, so users can rebuild the environment.

## Merge into main later

When NeMo-RL bumps AutoModel to a release that contains the model (and, ideally, the needed AutoModel fixes):

1. Rebase or recreate the support branch on the new `main`. Drop change-list entries that the bump already covers.
2. Split the remaining changes using the change list: generic NeMo-RL fixes first (each with tests), then model-specific integration, then recipe and guide ([pull requests](docs-and-release.md#pull-requests)).
3. Re-run at least the proxy smoke and a full-model short run on the merged code; the long-run result from the branch carries over only if the model code and recipe did not change.
4. Update the news entry and guide links from the branch to `main`.

AutoModel changes that are still needed after the bump are proposed to AutoModel at this point, from the change list, by a human.
