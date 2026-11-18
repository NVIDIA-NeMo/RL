# Document and announce

Stage 5 of [the workflow](SKILL.md#stages-and-gates): ship the recipe and a model guide whose every claim is backed by the validated run, then announce support. Both live on the support branch first; they move to `main` with the rest of the support ([branches-and-upstreaming.md](branches-and-upstreaming.md)).

## Recipe

- Path: `examples/configs/recipes/llm/grpo-<model>-<data>-<N>n<G>g-automodel.yaml`. Use the public model name; rename any internal codename before the PR.
- The recipe equals the validated run's resolved configuration ([settle the recipe](full-model-and-longrun.md#settle-the-recipe)). The checkpoint path is a placeholder the guide tells users to override.
- Keep the length relation `max_new_tokens = max_total_sequence_length - max_input_seq_length`.
- MoE recipes set `policy.router_replay.enabled: true`.
- Every non-default setting that exists for this model (router auxiliary loss off, chat-template switches, precision overrides, refit buffer sizes) carries a one-line comment saying why.

## Model guide

Path: `docs/guides/models/<family>/<model>.md`, linked from the family `index.md`, which is linked from `docs/guides/models/index.md`. Curve images go under `docs/assets/<model>/`.

Sections, in order:

| Section | Content |
|---|---|
| Title and summary | one paragraph; an "Early access" admonition stating what was validated (steps, context length) while the status is Functionally Ready |
| Support Status | the two-level table (Functionally Ready, Long-Run Convergence Validated) and the model's current level |
| What's Supported | one table row per validated combination: model and size, modality, training backend, parallelism, inference engine, precision |
| Build the Environment | the support branch to check out; the AutoModel branch or commit and the vLLM commit, with links to their PRs; submodule initialization; the venv refresh and multi-node lock-timeout settings |
| Get the Weights | where the checkpoint comes from, required config files, how to override the path |
| Example Recipes | a table: validated sequence length, training parallelism, rollout parallelism, `max_new_tokens`, recipe link |
| Choose a Recipe | when to use which recipe; placeholders for comparisons not yet run |
| Launch | the exact command; Slurm notes such as matching `cluster.num_nodes` |
| Reference Results | training curves (validation accuracy, reward, generation length, entropy, loss, `gen_kl_error`) with the step count and start and end values stated in text |
| Known Issues | everything not validated (longer contexts, other precisions, other parallel layouts); upstream problems worked around on the dev branches, with issue links once a human has filed them |

Claims discipline:

- State the validated scope in numbers: steps, context length, hardware, nodes.
- "Supported" means a run passed this skill's gates on that combination. Untested combinations go to Known Issues, not to What's Supported.
- Every number in the text comes from the run shown in the curves. Extract the per-step data once, keep the extraction and plot scripts with the experiment record, and compute the text's numbers (start and end values, ranges, windows) from that same data rather than from memory or chat logs.

## Pull requests

These apply when the support merges into `main` ([merge into main later](branches-and-upstreaming.md#merge-into-main-later)); on the support branch, keep commits split the same way so the PRs fall out of the branch.

- Split the work: dependency pins; generic NeMo-RL fixes (each with unit tests, written at this point); model-specific integration; recipe and guide. Generic fixes can merge before the model is announced.
- Each PR description lists the validation it carries: job ids or W&B runs, gate results, the image and commits.
- Link the upstream AutoModel and vLLM PRs, and any filed issues the recipe works around.
- Remove internal paths, cluster names, codenames and credentials from code, configs, docs and committed experiment files.

## Announce

Announce once stage 4 has passed and the guide is on the support branch: a dated README news entry linking the branch and the guide on it ([announce from the branch](branches-and-upstreaming.md#announce-from-the-branch)), plus model name, support level and validated scope wherever else it is announced. Update the guide's status when the long-run convergence level is reached, and the links when the support reaches `main`.

## Gate

Stage 5 passes when the recipe and guide are on the support branch, the news entry is on `main`, every claim in the guide maps to a recorded run, the support level stated matches the gates passed, and the change list is complete.
