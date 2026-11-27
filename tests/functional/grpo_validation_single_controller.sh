# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

#!/bin/bash
set -euo pipefail

project_root=$(realpath "$(dirname "${BASH_SOURCE[0]}")/../..")
cd "$project_root"
output_dir=$(mktemp -d "${TMPDIR:-/tmp}/sc-validation.XXXXXX")
echo "Validation artifacts: $output_dir"

uv run --no-sync --group test coverage run -a --data-file="$project_root/tests/.coverage" \
    --source="$project_root/nemo_rl" examples/run_grpo_single_controller.py \
    --config examples/configs/grpo_math_1B_megatron_single_controller.yaml \
    policy.model_name=Qwen/Qwen2.5-1.5B \
    policy.train_global_batch_size=8 \
    policy.train_micro_batch_size=1 \
    policy.generation.val_temperature=0.5 \
    policy.max_total_sequence_length=512 \
    policy.generation.max_new_tokens=128 \
    policy.generation.vllm_cfg.enforce_eager=true \
    grpo.num_prompts_per_step=2 \
    grpo.num_generations_per_prompt=4 \
    grpo.max_num_steps=3 \
    grpo.val_at_start=true \
    grpo.val_period=2 \
    grpo.val_at_end=true \
    grpo.val_num_generations_per_prompt=2 \
    grpo.max_val_samples=3 \
    grpo.val_batch_size=2 \
    data.train.dataset_name=DAPOMath17K \
    data.max_input_seq_length=384 \
    'data.validation={dataset_name: DAPOMathAIME2024}' \
    async_rl.sampler.max_lookahead_versions=0 \
    checkpointing.enabled=true \
    checkpointing.save_period=1 \
    checkpointing.keep_top_k=null \
    checkpointing.metric_name=val:pass_k \
    "checkpointing.checkpoint_dir=$output_dir/checkpoints" \
    logger.wandb_enabled=false \
    logger.monitor_gpus=false \
    "logger.log_dir=$output_dir/logs" \
    "$@" 2>&1 | tee "$output_dir/run.log"

uv run --no-sync python - "$output_dir" <<'PYTHON'
import json
import sys
from pathlib import Path

output_dir = Path(sys.argv[1])
logs = sorted(output_dir.glob("logs/**/val_data_step*.jsonl"))
assert [path.name for path in logs] == [
    "val_data_step0.jsonl",
    "val_data_step2.jsonl",
    "val_data_step3.jsonl",
]
for log in logs:
    rows = [json.loads(line) for line in log.read_text().splitlines()]
    assert len(rows) == 6
    step = int(log.stem.removeprefix("val_data_step"))
    if step:
        checkpoint = output_dir / "checkpoints" / f"step_{step}" / "training_info.json"
        info = json.loads(checkpoint.read_text())
        expected = sum(
            any(row["rewards"][0] > 0 for row in rows[i : i + 2])
            for i in range(0, len(rows), 2)
        ) / 3
        assert abs(info["val:pass_k"] - expected) < 1e-6

checkpoint = output_dir / "checkpoints/step_1/training_info.json"
assert "val:pass_k" not in json.loads(checkpoint.read_text())
PYTHON
