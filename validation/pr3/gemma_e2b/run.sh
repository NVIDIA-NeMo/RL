#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
set -euo pipefail
set +x
source /lustre/fsw/portfolios/coreai/users/yuekaiz/single_controller/RL-pr5090-streaming/validation/pr3/environment.sh
[[ "$USER" == root && -n "${SLURM_JOB_ID:-}" ]]
exec 9>validation/pr3/gemma_e2b/slurm/driver.lock
flock -n 9
nvidia-smi --query-gpu=name --format=csv,noheader
uv run --no-project --python /opt/nemo_rl_venv/bin/python python - <<'PY'
import copy
import hashlib
import json
import os
import subprocess
from pathlib import Path
import torch
from omegaconf import OmegaConf
from nemo_rl.algorithms.single_controller_utils import MasterConfig
from nemo_rl.algorithms.single_controller_utils.config import validate_single_controller_config
from nemo_rl.data.utils import update_single_dataset_config
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers
assert torch.cuda.device_count() == 8
register_omegaconf_resolvers()
path = Path('validation/pr3/gemma_e2b/streaming.yaml')
cfg = OmegaConf.to_container(load_config(str(path)), resolve=True)
validate_single_controller_config(MasterConfig(**cfg))
Path('validation/pr3/gemma_e2b/results/candidate-config.json').write_text(json.dumps(cfg,indent=2)+'\n')
assert cfg['grpo']['max_num_steps'] == 150
assert cfg['policy']['train_global_batch_size'] == 512
assert cfg['async_rl']['min_groups_for_streaming_train'] == 16
assert cfg['async_rl']['sampler']['max_lookahead_versions'] == 0
sources = {
    'nemo_rl/models/policy/workers/automodel_policy_worker.py': '7f59824258aafb13f5540810776dfa18c86a623856c9c89e4c6e26881259f74d',
    'tests/functional/automodel_sc_full_batch_parity.py': 'ccb26bda2ad5a1408cbbd125cf2aec5a119d232ecaa1777bebbce0071c0dc399',
}
for name, expected in sources.items():
    assert hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, name
baselines = json.loads(Path('validation/pr3/gemma_e2b/results/baselines.json').read_text())
comparison_data = copy.deepcopy(cfg['data'])
if isinstance(comparison_data['train'], dict):
    comparison_data['train'] = [comparison_data['train']]
for dataset in comparison_data['train']:
    update_single_dataset_config(dataset, comparison_data['default'])
policy_keys = ('model_name','precision','train_global_batch_size','train_micro_batch_size','logprob_batch_size','max_total_sequence_length','logprob_chunk_size','optimizer','scheduler','make_sequence_length_divisible_by')
generation_keys = ('temperature','top_p','top_k','max_new_tokens')
for label, baseline in baselines.items():
    original = baseline['config']
    for key in policy_keys:
        assert cfg['policy'][key] == original['policy'][key], (label, 'policy', key)
    for key in generation_keys:
        assert cfg['policy']['generation'][key] == original['policy']['generation'][key], (label, 'generation', key)
    for key in ('seed','num_prompts_per_step','num_generations_per_prompt'):
        assert cfg['grpo'][key] == original['grpo'][key], (label, 'grpo', key)
    assert cfg['loss_fn'] == original['loss_fn'], (label, 'loss_fn')
    assert comparison_data == original['data'], (label, 'normalized data')
    assert cfg['cluster'] == original['cluster'], (label, 'cluster')
    assert cfg['checkpointing']['save_period'] == original['checkpointing']['save_period']
out = Path('validation/pr3/gemma_e2b/results')
(out/'resolved-config.json').write_text(json.dumps(cfg,indent=2)+'\n')
provenance = {
    'job_id': os.environ['SLURM_JOB_ID'],
    'git_head': subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    'training_source_sha256': sources,
    'recipe_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
    'model_config_sha256': hashlib.sha256((Path(cfg['policy']['model_name'])/'config.json').read_bytes()).hexdigest(),
    'baseline_config_checks_passed': True,
    'qualification': 'exploratory; does not change the failed Qwen BF16 1% gate',
}
(out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
print('Preflight passed: matched Gemma batch512 baselines, streaming threshold16, LA0, 8 GPUs', flush=True)
PY
export WANDB_API_KEY
WANDB_API_KEY=$(< /lustre/fsw/portfolios/coreai/users/yuekaiz/.wandb_nvidia)
export WANDB_BASE_URL=https://api.wandb.ai WANDB_MODE=online
export WANDB_DIR="$PR3_REPO/validation/pr3/gemma_e2b"
unset WANDB_SERVICE _WANDB_SERVICE
uv run --no-project --python /opt/nemo_rl_venv/bin/python python examples/run_grpo_single_controller.py \
  --config validation/pr3/gemma_e2b/streaming.yaml
uv run --no-project --python /opt/nemo_rl_venv/bin/python python validation/pr3/gemma_e2b/plot.py
