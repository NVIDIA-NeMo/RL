# Dependencies and image

Stage 1 of [the workflow](SKILL.md#stages-and-gates): get a NeMo-RL environment whose AutoModel and vLLM both support the model, built into an image that every later run uses unchanged.

For a Day-0 model this is often the longest stage in calendar time: the supporting vLLM and AutoModel commits are usually newer than NeMo-RL's pins, and moving them ripples through Torch, kernels and the lockfile.

Start the checkpoint download first, before anything else, and run it in parallel with this stage. A 600B BF16 checkpoint is over 1 TB and was the longest single step of a past bring-up. Download the files the proxy needs first (config, tokenizer, the shards holding the layers to keep and the non-decoder components), so the proxy can be cut before the rest arrives. Download many files at once: per-file throughput is low, aggregate throughput is what matters.

## Find the supporting commits

| Component | What to find | Where |
|---|---|---|
| vLLM | the commit that adds the model (and its fixes), and whether a wheel is published for it | the vLLM support PR; `https://wheels.vllm.ai/<commit>/` |
| AutoModel | the commit with the model directory, its HF adapter and parallel plan | the AutoModel model PR; usually newer than the AutoModel release NeMo-RL `main` pins |
| Transformers | whether the checkpoint's `model_type` is registered, and the minimum version | the model card; `transformers` release notes |
| Checkpoint | the exact revision, quantization format, and any legacy `model_type` or architecture aliases | `config.json`, `model.safetensors.index.json` |

NeMo-RL `main` pins an AutoModel release, so a new model almost always needs a newer AutoModel than `main` has; that is why the work happens on a support branch ([branches-and-upstreaming.md](branches-and-upstreaming.md)). Start the AutoModel side from the upstream commit that added the model. Create an AutoModel dev branch on top of it only when the model needs AutoModel changes, and record the base commit. When the pin later moves (to a newer commit or to the release used for the merge into main), re-check every NeMo-RL integration that touches the model's internals (the marker API for owner-sharded parameters changed between the release branch and upstream in one past bring-up).

Two cases change where the commits come from:

- **Support is still an open PR** (Day-0 or early access): pin the PR's head commit (or its wheel, for vLLM) and record it in the change list. When the PR merges, move the pin to the merged commit and rerun the proxy smoke; a PR's head can change before it merges.
- **An open NeMo-RL PR already bumps AutoModel** to a version that has the model: cherry-pick its commits onto the support branch instead of redoing the bump, and record the PR in the change list; the support branch is rebased once that PR merges. Check NeMo-RL's open PRs for an AutoModel bump before moving the submodule yourself.

## Pin

- **vLLM**: pin the commit wheel by URL, one per architecture (x86_64 and aarch64), with a comment naming the commit and the PR. Do not use the floating nightly index: a later commit can change the internal APIs that NeMo-RL's refit uses. Do not pin an editable local source checkout in the shared lockfile: it fails with "Distribution not found" on any machine where the directory is empty.
- **AutoModel**: move the submodule to the commit (or the dev branch head) and update `.gitmodules` (branch and a comment saying why). Run `git submodule update --init 3rdparty/Automodel-workspace/Automodel`.
- **Keep everything else unchanged** until the lock fails, then move only what the error names.

## Resolve the lock

Run `uv lock` and fix conflicts one at a time. Conflicts seen when moving to a new vLLM:

| Symptom | Cause | Fix |
|---|---|---|
| Unsatisfiable: two FlashInfer versions | another extra (for example the Gym vLLM extra) pins an older FlashInfer than the new vLLM | align the pin, or declare the extras as conflicting if they never install together |
| Torch moves, and torchvision or flash-attn breaks | prebuilt wheels are tied to the old Torch ABI | pin an ABI-matched torchvision as an override; build flash-attn from source |
| `No module named 'cutlass.cute'` at runtime | an `exclude-dependencies` entry removed the CUTLASS DSL base libraries | remove the exclusion; pin `nvidia-cutlass-dsl` to the version vLLM expects |
| Import errors from `openai`, `llguidance`, `numpy`, `setuptools` | the new vLLM needs newer or capped versions | add overrides with a comment naming the importing module |
| `uv lock` downloads multi-GB wheels | metadata only exists inside the wheel | add `[[tool.uv.dependency-metadata]]` entries for those packages |
| Optional kernel extensions fail to import with `undefined symbol` | DeepEP, NIXL or DeepGEMM built against the old Torch | rebuild them if the recipe uses them; otherwise record them as known log noise |
| Source-built extensions fail with C++ errors in Torch headers (`'requires' does not name a type`) | a newer Torch's headers need C++20, and the extension's build hard-codes C++17 | move to an extension commit that inherits Torch's C++ standard; if none exists and the recipe does not use the extension, exclude it on the support branch and record it as a branch-only regression |
| A kernel package resolves only for some architectures (`... has no compatible wheels`) | a package split into per-GPU-architecture sub-packages, some published for aarch64 only | pin the sub-packages directly, and give the missing ones per-architecture dependency metadata |
| A kernel package fails to build from source although vLLM ships it | the vLLM wheel now vendors it (DeepGEMM under `vllm/third_party`) | drop the separate dependency |
| A worker import fails inside a library (`cannot import name ...` from an OpenTelemetry-like package) | an override floor lets two halves of a library (API and SDK) resolve to mismatched versions | raise the override floor so both halves match |

Pin per-architecture packages (DeepEP, kernels) separately for x86_64 and aarch64 when their builds differ.

## Build the image

- Build an image from the locked commit; do not hand-patch packages inside a running container. Record the image path, its checksum, and the NeMo-RL commit it was built from.
- Worker virtual environments are cached. After a lock change, set `NRL_FORCE_REBUILD_VENVS=true` for the first run, or bake the venvs into the image. On multi-node runs set `UV_LOCK_TIMEOUT=3600`: all nodes contend on one lock in the shared uv cache and the default timeout expires.
- During development, mount the live checkout over the image's editable-install path instead of rebuilding the image for each code change. Rebuild for every dependency change.
- Keep validated images where chained jobs can find them. A deleted or renamed image fails every queued job at container start.

## Standalone checks

Run these before any NeMo-RL code changes, so that later failures are known to be integration failures.

1. **vLLM alone**: serve the real checkpoint with the arguments from the vLLM support PR and generate from a few prompts with the model's chat template, about 2K output tokens and CUDA graphs on. Output must be coherent. If the model needs more nodes than the proxy runs on, or the download is not finished, run this at the start of stage 3 instead.
2. **AutoModel alone**: build the model from the checkpoint config (on the meta device is enough for the full model) and load the proxy's weights once the proxy exists.
3. **Config resolution**: check whether `transformers.AutoConfig` can load the checkpoint config. If it cannot and vLLM can, NeMo-RL must resolve the config through vLLM ([touchpoint](proxy-and-enablement.md#configuration-and-architecture)).
4. Import NeMo-RL's policy and generation workers inside the image with no model loaded; a version assert that fails here would otherwise fail an 8-node job at startup. Import the engine entry points too (`from vllm import LLM`), not only `import vllm`: vLLM loads them lazily, so a missing module there passes a bare import and fails at engine creation.

## After moving vLLM: check NeMo-RL's vLLM hooks before the first smoke

NeMo-RL reaches into vLLM internals: private imports, environment flags, and source patches anchored on vLLM code text (executor setup, MoE routed-expert capture for R3, model-specific fixes). A vLLM bump breaks some of them, and each break kills a smoke at worker start, one per run. In one bring-up, five of sixteen proxy smoke runs failed this way.

Before the first smoke, check them all at once against the pinned vLLM source:

- every `vllm.*` import and attribute NeMo-RL uses still exists (`vllm.envs` flags included);
- every source patch finds its anchor; a patch for another model must skip with a logged error when its anchor is gone, not raise for every model;
- every file a patch edits still exists (removed executors, renamed modules).

Fix what fails in one commit per hook and record each in the change list.

## Gate

Stage 1 passes when `uv lock` resolves with vLLM and AutoModel pinned to recorded commits, the image is built from that lock and recorded, and the four standalone checks pass.
