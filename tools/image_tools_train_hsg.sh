#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Driver entrypoint inside the selected HSG container. Never rebuild bundled envs.
set -euo pipefail
set +x
cd /opt/nemo-rl
# Preserve explicit submission diagnostics across credential-file defaults.
image_tools_requested_nccl_debug="${NCCL_DEBUG-}"
set -a
source /opt/nemo-rl/.env
set +a
if [[ -n "$image_tools_requested_nccl_debug" ]]; then
  export NCCL_DEBUG="$image_tools_requested_nccl_debug"
fi
unset image_tools_requested_nccl_debug
: "${WANDB_API_KEY:?The synced project .env must supply WANDB_API_KEY}"
# Apply these after sourcing credentials; never silently run offline.
export WANDB_MODE=online WANDB_DISABLED=false WANDB_RESUME="$IMAGE_TOOLS_WANDB_RESUME"
export WANDB_PROJECT=games-rlvr-nemotron-super WANDB_ENTITY=nvidia
export WANDB_RUN_ID="$IMAGE_TOOLS_WANDB_ID"
export WANDB_RUN_NAME="$IMAGE_TOOLS_RUN_NAME"
export PROJECT_ROOT=/opt/nemo-rl
# Redirect future uv work without hiding the archive that bundled venvs use.
export UV_CACHE_DIR="$RUN_DIR/cache/uv"
export PYTHONPATH="$OVERLAY_DIR/packages:/opt/nemo-rl/tools/runtime_source_override:$HF_MODULES_CACHE:/opt/nemo-rl:/opt/nemo-rl/3rdparty/Gym-workspace/Gym"

/opt/nemo_rl_venv/bin/python - <<'PY'
import json
import hashlib
import os
import tarfile
from pathlib import Path
from omegaconf import OmegaConf
from transformers import AutoConfig, AutoProcessor, AutoTokenizer
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.algorithms.grpo import MasterConfig
from nemo_rl.utils.config import load_config, parse_hydra_overrides, register_omegaconf_resolvers
from tools.image_tools_launch_checks import require_online_wandb
from tools.image_tools_processor_assets import copy_processor_assets
from tools.check_image_tools_processor import check_processor_image_contract
from nemo_gym.global_config import GlobalConfigDictParser, GlobalConfigDictParserConfig

require_online_wandb(os.environ)
register_omegaconf_resolvers()
cfg = load_config(os.environ['IMAGE_TOOLS_CONFIG'])
resolved_for_validation = parse_hydra_overrides(cfg, [
    'logger.wandb_enabled=true',
    '++logger.wandb.id=' + os.environ['WANDB_RUN_ID'],
    '++logger.wandb.resume=' + os.environ['WANDB_RESUME'],
])
MasterConfig(**OmegaConf.to_container(resolved_for_validation, resolve=True))
print('IMAGE_TOOLS_MASTER_CONFIG_OK', flush=True)
if os.environ.get('RESUME_STEP'):
    from nemo_rl.utils.checkpoint import CheckpointManager
    manager = CheckpointManager(OmegaConf.to_container(cfg.checkpointing, resolve=True))
    expected = Path(os.environ['CHECKPOINT_DIR']) / ('step_' + os.environ['RESUME_STEP'])
    assert manager.get_latest_checkpoint_path() == str(expected)
    assert manager.load_training_info(str(expected))['current_step'] == int(os.environ['RESUME_STEP'])
    assert os.environ['WANDB_RESUME'] == 'must'
    print(f'IMAGE_TOOLS_RESUME_CONFIG_OK checkpoint={expected} wandb_id={os.environ["WANDB_RUN_ID"]}', flush=True)
model = os.environ['MODEL_CHECKPOINT']
target = Path(os.environ['VLLM_TOKENIZER'])
if target.exists():
    raise RuntimeError('Fresh training run must not reuse an unverified tokenizer directory')
AutoConfig.from_pretrained(model, trust_remote_code=True)
AutoProcessor.from_pretrained(model, trust_remote_code=True, use_fast=True, fix_mistral_regex=True)
tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True, use_fast=True, fix_mistral_regex=True)
asset_hashes = copy_processor_assets(Path(model), target=target)
tokenizer.save_pretrained(target)
reloaded = AutoTokenizer.from_pretrained(target, trust_remote_code=True, use_fast=True, fix_mistral_regex=True)
if os.environ.get('IMAGE_TOOLS_SUITE') == 'visual-games':
    from tools.callback_fix_contract import require_callback_sources
    require_callback_sources(Path('/opt/nemo-rl'))
    assert cfg.checkpointing.ft_save_period == 20
    assert cfg.checkpointing.keep_top_k is None
    assert cfg.checkpointing.ft_keep_latest_k is None
    print('MIXED_CHECKPOINT_RETENTION_OK every=20 keep=all', flush=True)
    archive = Path(os.environ['VISGYM_ASSET_ARCHIVE'])
    assert hashlib.sha256(archive.read_bytes()).hexdigest() == '468d4f7b54e1cc6ba5aad76287b15319c83bccad569b8ca5ec14ebab95855a21'
    asset_dir = Path(os.environ['RUN_DIR']) / 'assets/resources_servers/visgym/data'
    asset_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as stream:
        stream.extractall(asset_dir, filter='data')
# Resolve the actual Gym bundle before initializing model workers. Offline
# mode checks dependency pins/configuration without starting servers or venvs.
gym_config = OmegaConf.to_container(cfg.env.nemo_gym, resolve=True)
gym_config.pop('pad_dynamic_image_shapes', None)
gym_config.pop('effort_levels', None)
gym_config.update(policy_model_name=model, policy_api_key='dummy_key', policy_base_url=['http://127.0.0.1:8000/v1'])
resolved_gym = GlobalConfigDictParser().parse(GlobalConfigDictParserConfig(
    initial_global_config_dict=OmegaConf.create(gym_config),
    skip_load_from_cli=True, skip_load_from_dotenv=True, offline=True,
))
assert 'openai==2.6.1' in resolved_gym.head_server_deps
assert resolved_gym.allow_openai_version_skew is False
print('IMAGE_TOOLS_GYM_CONFIG_OK openai=2.6.1 skew=false', flush=True)
# Exercise the entrypoint's actual multimodal loader, not only AutoTokenizer.
# The entrypoint resolves OmegaConf before calling this dict-based API.
tokenizer_config = OmegaConf.to_container(cfg.policy['tokenizer'], resolve=True)
processor = get_tokenizer(tokenizer_config, get_processor=True)
check_processor_image_contract(processor)
for text in ['Image tools: crop(10, 20, 30, 40)', '1234567890\nA boxed answer: \\boxed{42}']:
    assert tokenizer.encode(text) == reloaded.encode(text)
    assert tokenizer.encode(text) == processor.tokenizer.encode(text)
(target / 'provenance.json').write_text(json.dumps({'model': model, 'fix_mistral_regex': True, 'processor_assets_sha256': asset_hashes}) + '\n')
assert cfg.logger.wandb_enabled is True
assert cfg.logger.wandb.entity == 'nvidia'
assert cfg.logger.wandb.project == 'games-rlvr-nemotron-super'
assert cfg.grpo.reward_shaping.enabled is False
expected_batch = 128 if os.environ.get('IMAGE_TOOLS_SUITE') == 'visual-games' else 32
assert cfg.policy.train_global_batch_size == expected_batch
assert cfg.grpo.num_prompts_per_step * cfg.grpo.num_generations_per_prompt == expected_batch
assert cfg.policy.generation.max_new_tokens == 512
assert cfg.env.nemo_gym.skip_venv_if_present is True
print('IMAGE_TOOLS_TRAIN_PREFLIGHT_OK wandb=online checkpoint=' + model, flush=True)
PY

# Transformers versions can use different dynamic-module cache paths. Warm the
# actual actor runtimes serially: concurrent first-use shutil.copyfile calls can
# expose a partially copied module to another worker (mixed job7097777).
for component in nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker; do
  CUDA_VISIBLE_DEVICES='' "/opt/ray_venvs/$component/bin/python" \
    tools/prewarm_image_tools_model_modules.py \
    --model "$MODEL_CHECKPOINT" --tokenizer "$VLLM_TOKENIZER"
done

# Run the requested PR regressions in the exact learner environment before
# allocating model weights. No extra scheduled qualification job is needed.
if [[ "$IMAGE_TOOLS_SUITE" == visual-games ]]; then
  bash tools/run_callback_regressions.sh
fi

exec /opt/nemo_rl_venv/bin/python -u examples/nemo_gym/run_grpo_nemo_gym.py \
  --config "$IMAGE_TOOLS_CONFIG" \
  logger.wandb_enabled=true \
  ++logger.wandb.id="$WANDB_RUN_ID" \
  ++logger.wandb.resume="$WANDB_RESUME"
