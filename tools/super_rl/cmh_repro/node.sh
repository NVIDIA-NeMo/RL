#!/bin/bash
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
set -euo pipefail
set +x
source "${SUPER_RL_RUNTIME_ENV:?}"
source /opt/nemo-rl/tools/super_rl/cmh_repro/profile.env
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1
vllm_python=/opt/ray_venvs/nemo_rl.models.generation.vllm.vllm_worker_async.VllmAsyncGenerationWorker/bin/python
worker_python=/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/bin/python
if ! "$vllm_python" -c 'import openai,sys; sys.exit(openai.__version__ != "2.25.0")'; then
    uv pip install --no-index --no-deps --python "$vllm_python" "$SUPER_RL_WHEEL"
fi
"$vllm_python" -c 'import openai; assert openai.__version__ == "2.25.0"'
"$vllm_python" -c 'from transformers import AutoConfig; import os; c=AutoConfig.from_pretrained(os.environ["SUPER_RL_MODEL"],trust_remote_code=True); assert c.llm_config.num_nextn_predict_layers == 1; print("HF_DYNAMIC_MODULE_PREWARM_OK")'
"$worker_python" /opt/nemo-rl/tools/super_rl/build_mcore_helpers.py verify \
    --package /opt/nemo-rl/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge/3rdparty/Megatron-LM/megatron/core/datasets
provider=/opt/gym_venvs/responses_api_models/inference_provider/.venv
if [[ ! -x "$provider/bin/python" ]]; then
    mkdir -p "$(dirname "$provider")"
    ln -s /opt/gym_venvs/responses_api_models/vllm_model/.venv "$provider"
fi
/opt/gym_venvs/responses_api_models/vllm_model/.venv/bin/python /opt/nemo-rl/tools/super_rl/prepare_node.py \
    --config "$SUPER_RL_CONFIG" --gym /opt/nemo-rl/3rdparty/Gym-workspace/Gym \
    --image-venvs /opt/gym_venvs --runtime-venvs /opt/train_gym_venvs \
    --receipt "$SUPER_RL_ROOT/manifests/node-imports-$NRL_SLURM_JOB_ID-$(hostname)-${SUPER_RL_SETUP_ATTEMPT:?}.json"
