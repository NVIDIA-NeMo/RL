#!/usr/bin/env bash
set -euo pipefail

ROOT=${ROOT:-/lustre/fsw/portfolios/coreai/users/jothomson/nemo-rl-dynamo-slurm-swe}
REPO=${REPO:-${ROOT}/RL}
MODEL=${MODEL:-/lustre/fsw/portfolios/llmservice/users/pjin/devel/nemo-rl-ultra-v3-nano-opd-dev-20260513/results/mopd_ultrav3_to_nanov3_5_repro_v5_kd_opt_full-hsg-20260524-r1/step_18/hf}
EXPECTED_GYM_COMMIT=eddd5e98a541cc90e0ee41f1b5e9bd146b5be665

test "$(uname -m)" = aarch64
test -f "${MODEL}/chat_template.jinja"
test "$(cat /opt/dynamo_commit)" = 59358c26d0aeed19300706462b63ada25a0a6d7c

/opt/dynamo_venv/bin/python -c \
  'import importlib.metadata as m, vllm; assert m.version("ai-dynamo") == "1.3.0"; assert m.version("ai-dynamo-runtime") == "1.3.0"; assert vllm.__version__ == "0.23.0"; print("Dynamo", m.version("ai-dynamo"), "runtime", m.version("ai-dynamo-runtime"), "vLLM", vllm.__version__)'
etcd --version | grep -F 'etcd Version: 3.5.21'
nats-server --version | grep -F 'v2.11.6'

RESOLVED_ARGV=$(
  PYTHONPATH="${REPO}" MODEL="${MODEL}" /opt/nemo_rl_venv/bin/python - <<'PY'
import json
import os

from nemo_rl.models.generation.dynamo.arguments import build_dynamo_vllm_argv
from nemo_rl.models.generation.dynamo.config import DynamoCfg

model = os.environ["MODEL"]
cfg = DynamoCfg.model_validate(
    {
        "deployment": "ray",
        "engine_world_size": 4,
        "worker_args": {
            "tool_call_parser": "qwen3_coder",
            "reasoning_parser": "nemotron_nano",
            "exclude_tools_when_tool_choice_none": True,
            "enable_structural_tag": False,
            "structural_tag_scope": "auto",
            "structural_tag_schema": "auto",
            "custom_jinja_template": f"{model}/chat_template.jinja",
            "endpoint_types": ["chat", "completions"],
        },
    }
)
argv = build_dynamo_vllm_argv(
    model_name=model,
    namespace="nemo-rl-swe-image-test",
    seed=0,
    vllm_cfg={
        "precision": "bfloat16",
        "kv_cache_dtype": "auto",
        "tensor_parallel_size": 4,
        "pipeline_parallel_size": 1,
        "expert_parallel_size": 1,
        "gpu_memory_utilization": 0.85,
        "max_model_len": 196608,
        "enforce_eager": False,
        "load_format": "auto",
    },
    vllm_kwargs={
        "attention_backend": "FLASH_ATTN",
        "moe_backend": "triton",
        "mamba_ssm_cache_dtype": "float32",
        "compilation_config": {
            "cudagraph_capture_sizes": [1, 2, 4, 8, 16, 32, 64],
            "pass_config": {"fuse_allreduce_rms": False},
        },
    },
    dynamo_cfg=cfg,
)
print(json.dumps(argv))
PY
)
printf 'Resolved Nemotron argv: %s\n' "${RESOLVED_ARGV}"
/opt/dynamo_venv/bin/python \
  "${REPO}/nemo_rl/models/generation/dynamo/validate_dynamo_vllm_args.py" \
  "${RESOLVED_ARGV}"

test "$(git -C "${REPO}/3rdparty/Gym-workspace/Gym" rev-parse HEAD)" = \
  "${EXPECTED_GYM_COMMIT}"
GYM_PYTHON=$(find /opt/gym_venvs -path '*/.venv/bin/python' -print -quit)
test -x "${GYM_PYTHON}"
"${GYM_PYTHON}" -c \
  'import pathlib, nemo_gym; expected = pathlib.Path("/opt/nemo-rl/3rdparty/Gym-workspace/Gym").resolve(); actual = pathlib.Path(nemo_gym.__file__).resolve(); assert actual.is_relative_to(expected), (actual, expected); print("Gym source", actual)'
grep -R -q '_attach_native_token_information' \
  /opt/nemo-rl/3rdparty/Gym-workspace/Gym

echo 'HSG image validation passed.'
