# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
"""Check the inspection bundle without launching a training job."""

import ast
import json
import os
import re
import subprocess
from pathlib import Path

from omegaconf import OmegaConf

from config_loader import load_config, register_omegaconf_resolvers

root = Path(__file__).resolve().parents[1]
os.environ.update(
    {
        "MM_TRAINER_MODEL_PATH": "/model",
        "MM_TRAINER_DATA_PATH": "/data/training.jsonl",
        "MM_TRAINER_RESULTS_DIR": "/results/new-run",
        "MM_TRAINER_GYM_VENV_DIR": "/opt/gym_venvs",
        "MM_TRAINER_WANDB_ID": "inspection-run",
        "MM_TRAINER_WANDB_NAME": "inspection-run",
    }
)
register_omegaconf_resolvers()
configs = {}
for path in (root / "nemo-rl/configs").glob("*.yaml"):
    c = OmegaConf.to_container(load_config(path), resolve=True)
    configs[path.name] = c
    assert (
        c["policy"]["train_global_batch_size"]
        == c["grpo"]["num_prompts_per_step"] * c["grpo"]["num_generations_per_prompt"]
    )
    assert not c["policy"]["router_replay"]["enabled"]
    assert not c["policy"]["generation"]["vllm_kwargs"]["enable_return_routed_experts"]
    assert c["grpo"]["seq_logprob_error_threshold"] == 2.0
    assert c["data_plane"]["enabled"] and c["token_capture"]["enabled"]
    assert c["grpo"]["async_grpo"] is None
    assert c["policy"]["offload_optimizer_for_logprob"]
    assert not c["policy"]["megatron_cfg"]["checkpoint"]["async_save"]
    assert not c["policy"]["megatron_cfg"]["optimizer"]["optimizer_cpu_offload"]
    assert c["checkpointing"]["checkpoint_dir"] == "/results/new-run/checkpoints"
    assert c["logger"]["wandb"]["id"] == "inspection-run"
    assert "control_auth_token" not in json.dumps(c)
prod = configs["supervl3p5-v2-production.yaml"]
ready = configs["supervl3p5-v2-ready-first.yaml"]
smoke = configs["supervl3p5-v2-smoke.yaml"]
assert (
    prod["checkpointing"]["save_period"] == ready["checkpointing"]["save_period"] == 10
)
assert prod["policy"]["megatron_cfg"]["context_parallel_size"] == 2
assert prod["async_rl"]["sampler"] == {"name": "in_order", "max_lookahead_versions": 1}
assert ready["async_rl"]["sampler"] == {
    "name": "ready_first",
    "max_staleness_versions": 2,
}
assert ready["async_rl"]["max_buffered_rollouts"] == 384
assert (
    smoke["grpo"]["max_num_steps"] == 2
    and smoke["policy"]["train_global_batch_size"] == 256
)
assert smoke["checkpointing"]["save_period"] == 1
assert (
    prod["cluster"]["num_nodes"]
    == ready["cluster"]["num_nodes"]
    == smoke["cluster"]["num_nodes"]
    == 32
)

# Compare all common settings against the saved effective production config.
# Runtime-derived fields may exist only in the checkpoint snapshot.
saved = OmegaConf.to_container(
    OmegaConf.load(root / "evidence/in-order-step110-config.yaml"), resolve=True
)
allowed = (
    "policy.model_name",
    "policy.tokenizer.name",
    "policy.tokenizer.chat_template",
    "policy.generation.vllm_cfg.load_format",
    "data.train",
    "logger.log_dir",
    "logger.wandb.",
    "checkpointing.checkpoint_dir",
    "env.nemo_gym.nemo_gym_log_dir",
    "env.nemo_gym.image_tools_simple_agent.responses_api_agents.image_tools_agent.crop_dir",
)
differences = []


def compare(a, b, path=""):
    if any(path == prefix or path.startswith(prefix) for prefix in allowed):
        return
    if isinstance(a, dict) and isinstance(b, dict):
        for key in a.keys() & b.keys():
            compare(a[key], b[key], f"{path}.{key}".lstrip("."))
    elif a != b:
        differences.append({"path": path, "recipe": a, "checkpoint": b})


compare(prod, saved)
assert not differences, differences

links = scripts = snippets = 0
for path in root.rglob("*.md"):
    text = path.read_text()
    assert not re.search(r"\b(?:remx|rxm|rem getnode)\b", text), path
    for target in re.findall(r"(?<!!)\[[^\]]+\]\(([^)]+)\)", text):
        if "://" in target or target.startswith("#"):
            continue
        assert (path.parent / target.split("#")[0]).exists(), (path, target)
        links += 1
    for snippet in re.findall(r"```bash\n(.*?)\n```", text, flags=re.S):
        subprocess.run(["bash", "-n"], input=snippet, text=True, check=True)
        snippets += 1
for path in root.rglob("*.sh"):
    subprocess.run(["bash", "-n", str(path)], check=True)
    scripts += 1
    text = path.read_text()
    for snippet in re.findall(r"<<'PY'\n(.*?)\nPY", text, flags=re.S):
        ast.parse(snippet)
result = {
    "configs": list(configs),
    "common_production_settings_match_checkpoint": True,
    "shell_scripts": scripts,
    "shell_examples": snippets,
    "local_links": links,
    "new_gpu_runs": 0,
}
(root / "evidence/validation.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
