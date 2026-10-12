# SpeechLM Environment Setup

This page configures the package environments for the
[SpeechLM GSPO guide](../guides/gspo-speechlm.md). Run these installation commands
inside `nvcr.io/nvidia/nemo-rl:v0.7.0` on every training node before Ray starts.
The repository paths are the coordinated checkouts from the guide.

## Package environments

The image contains separate actor environments. Keep them separate:

| Component | Policy / export | Generation / rewards |
| --- | --- | --- |
| Python | 3.13.13 | 3.13.13 |
| Torch | 2.11.0, CUDA 13.0 | 2.11.0, CUDA 13.0 |
| Transformers | 5.5.0 | 5.8.1 |
| vLLM | Not used for policy updates | 0.20.0 |
| Source packages | RL, Automodel, Speech | RL, Speech, Skills core |

NeMo Skills supplies Mandarin CER. Ordinary WER uses RapidFuzz. Install Speech
with its ASR dependencies, not its `vllm` extra: the latter can select a different
vLLM version from the one supplied by the image.

```bash
export POLICY_PYTHON=/opt/ray_venvs/nemo_rl.models.policy.workers.dtensor_policy_worker_v2.DTensorPolicyWorkerV2/bin/python
export GENERATION_PYTHON=/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker.VllmGenerationWorker/bin/python
export PYTHONNOUSERSITE=1 PYTHONSAFEPATH=1 NO_VCS_VERSION=1
unset PYTHONPATH
export UV_CACHE_DIR="$(mktemp -d /tmp/speechlm-uv.XXXXXX)"
export SPEECHLM_SETUP_DIR="$(mktemp -d /tmp/speechlm-setup.XXXXXX)"
```

Use a cache private to each node/container invocation. This avoids concurrent
editable package builds contending for shared cache entries.

## Preserve the image dependencies

Before installation, record a constraints file for each actor environment.
The source packages and three Python dependency adjustments are intentionally
excluded: `packaging==24.2`, `fsspec==2025.12.0`, and generation-side
`openai==2.20.0`. Keep the image's normal dependency overrides; its compiled
Transformer Engine and DeepEP packages are already installed and are not
rebuilt from Git.

```bash
for profile in policy generation; do
  if [ "$profile" = policy ]; then
    actor_python="$POLICY_PYTHON"
  else
    actor_python="$GENERATION_PYTHON"
  fi
  uv run --no-project --no-sync --python "$actor_python" - \
    "$profile" "$SPEECHLM_SETUP_DIR" "$RL_REPO" <<'PY'
import importlib.metadata as metadata
from pathlib import Path
import sys
import tomllib

profile, output_dir, repo = sys.argv[1:]
output = Path(output_dir)
excluded = {
    "nemo-rl", "nemo-automodel", "nemo-toolkit", "nemo-skills",
    "nemo-skills-core", "packaging", "fsspec",
}
if profile == "generation":
    excluded.add("openai")
pins = sorted({
    f"{dist.metadata['Name']}=={dist.version}"
    for dist in metadata.distributions()
    if dist.metadata["Name"].lower().replace("_", "-") not in excluded
})
(output / f"{profile}-constraints.txt").write_text("\n".join(pins) + "\n")
project = tomllib.loads((Path(repo) / "pyproject.toml").read_text())
overrides = project["tool"]["uv"]["override-dependencies"]
native = [value for value in overrides if "git+" in value]
assert all(value.split("@", 1)[0].strip().split("[", 1)[0].strip()
           in {"transformer-engine", "deep_ep"} for value in native)
(output / f"{profile}-overrides.txt").write_text(
    "\n".join(value for value in overrides if "git+" not in value) + "\n"
)
PY
done
```

The RL and Automodel checkouts must retain the image-compatible dependency
manifests. These commands install source changes into the existing environments;
they do not resolve a new RL or Automodel dependency stack.

## Install the source packages

```bash
uv pip install --no-config --no-deps --python "$POLICY_PYTHON" \
  --editable "$RL_REPO" --editable "$AUTOMODEL_REPO"
uv pip install --no-config --no-deps --python "$GENERATION_PYTHON" \
  --editable "$RL_REPO"

for profile in policy generation; do
  if [ "$profile" = policy ]; then
    actor_python="$POLICY_PYTHON"
    extra_packages=()
  else
    actor_python="$GENERATION_PYTHON"
    extra_packages=(--editable "$SKILLS_REPO/core[audio]" "openai==2.20.0")
  fi
  uv pip install --no-config --python "$actor_python" \
    --constraint "$SPEECHLM_SETUP_DIR/$profile-constraints.txt" \
    --overrides "$SPEECHLM_SETUP_DIR/$profile-overrides.txt" \
    --index-strategy unsafe-best-match \
    --extra-index-url https://download.pytorch.org/whl/cu130 \
    --editable "$SPEECH_REPO[asr]" \
    'torchaudio==2.11.0' 'torchcodec==0.17.0+cu130' 'peft==0.18.1' \
    'packaging==24.2' 'fsspec==2025.12.0' "${extra_packages[@]}"
done
```

Keep the resulting package inventories with the run. The image's declared
overrides can leave dependency-metadata warnings even when imports work; compare
those against the unmodified image rather than upgrading the compiled stack to
silence them. The setup above is the portable form of the package installation;
its fresh-checkout execution is not yet separately validated. The tested path
uses the same image and actor package versions with a fixed source snapshot.

Continue with [model/data setup and training](../guides/gspo-speechlm.md#2-prepare-the-model-and-dataset).
