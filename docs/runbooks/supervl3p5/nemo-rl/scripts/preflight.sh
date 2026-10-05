#!/usr/bin/env bash
set -euo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$script_dir/env.sh"
export SUPER_CONFIG=${SUPER_CONFIG:-$script_dir/../configs/supervl3p5-v2-production.yaml}
cd "$RL_DIR"
uv run --no-sync --python "$DRIVER_PYTHON" python - <<'PY'
import json
import os
from pathlib import Path
from urllib.parse import unquote, urlparse

from omegaconf import OmegaConf
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

register_omegaconf_resolvers()
cfg = OmegaConf.to_container(load_config(os.environ["SUPER_CONFIG"]), resolve=True)
assert cfg["policy"]["train_global_batch_size"] == cfg["grpo"]["num_prompts_per_step"] * cfg["grpo"]["num_generations_per_prompt"]
assert cfg["data_plane"]["enabled"] and cfg["grpo"]["async_grpo"] is None
assert not cfg["policy"]["router_replay"]["enabled"]
assert cfg["checkpointing"]["save_period"] in (1, 10)
for key in ("DRIVER_PYTHON", "MEGATRON_WORKER_PYTHON", "VLLM_WORKER_PYTHON"):
    assert Path(os.environ[key]).is_file(), key
model = Path(os.environ["MM_TRAINER_MODEL_PATH"])
for name in ("config.json", "chat_template.jinja"):
    assert (model / name).is_file(), name
assert list(model.glob("*.safetensors")), "Missing HF weights"
assert Path(os.environ["GYM_EXTRA_DIR"]).is_dir()
missing = set()
rows = encoded = 0
def check(value):
    global encoded
    if isinstance(value, dict):
        for child in value.values():
            check(child)
    elif isinstance(value, list):
        for child in value:
            check(child)
    elif isinstance(value, str):
        if value.startswith("file://"):
            parsed = urlparse(value)
            assert parsed.netloc in ("", "localhost"), "Unsupported file URI host"
            path = unquote(parsed.path)
            encoded += path != parsed.path
        elif value.startswith("/lustre/"):
            path = value
        else:
            return
        if not Path(path).exists():
            missing.add(path)
with Path(os.environ["MM_TRAINER_DATA_PATH"]).open() as stream:
    for line in stream:
        if line.strip():
            check(json.loads(line))
            rows += 1
assert rows > 0, "Empty dataset"
if missing:
    raise RuntimeError(f"{len(missing)} missing paths; first: {sorted(missing)[:5]}")
print(json.dumps({"rows": rows, "encoded_file_uris": encoded, "sampler": cfg["async_rl"]["sampler"], "checkpoint_period": cfg["checkpointing"]["save_period"]}, indent=2))
PY
uv run --no-sync --python "$MEGATRON_WORKER_PYTHON" python -c 'import megatron.core, megatron.bridge, torchcodec; print(megatron.core.__file__); print(megatron.bridge.__file__)'
uv run --no-sync --python "$VLLM_WORKER_PYTHON" python -c 'import importlib.metadata, torchcodec; print("vLLM", importlib.metadata.version("vllm"))'
