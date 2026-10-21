#!/usr/bin/env bash
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Run inside a fresh interactive allocation, then exit its shell promptly.
set -euo pipefail
set +x
umask 077
cd /flint-work
trap 'flint_validation_status=$?; printf "%s\n" "$flint_validation_status" > /flint-work/evidence/resume-validation.exit-code; trap - EXIT; exit "$flint_validation_status"' EXIT

/opt/flint/bin/python-driver - <<'PY'
from pathlib import Path
import json
r=Path('/flint-work/evidence')
a=json.loads((r/'training-audit-math-step1.json').read_text())
o=json.loads((r/'optimizer-step1-check.json').read_text())
h=json.loads((r/'changed-head-row-step1.json').read_text())
assert a['status']=='pass' and a['summary']['head_changed_elements']>0
assert a['summary']['sink_count']==49 and a['summary']['audited_saved_parameters_finite']
assert o['status']=='passed' and all(x['value']==2 for x in o['records'])
assert h['status']=='pass' and h['recommended_refit_sample_row_start']==1
assert h['informative_sample']['changed_elements']>0
assert h['informative_sample']['checkpoint_fp32_sha256']!=h['informative_sample']['source_fp32_sha256']
assert Path('/flint-work/results/math/step_1/config.yaml').is_file()
PY

export FLINT_SYNTHETIC_SMOKE=0
export FLINT_REFIT_AUDIT_HEAD_ROW_START=1
bash /flint-work/tools/run_phase.sh grpo-full49-math-resume-01 \
  grpo.max_num_steps=2 checkpointing.keep_top_k=2

CUDA_VISIBLE_DEVICES= /opt/flint/bin/python-training /flint-work/tools/audit_training_result.py \
  --source /flint-model --checkpoint /flint-work/results/math/step_2 \
  --training-config /flint-work/results/math/step_2/config.yaml \
  --expected-ordinary-dtype float32 --payload-scope protected \
  --output /flint-work/evidence/training-audit-math-step2.json \
  > /flint-work/evidence/training-audit-math-step2.log 2>&1

CUDA_VISIBLE_DEVICES= /opt/flint/bin/python-training /flint-work/tools/check_resume_optimizer.py \
  --checkpoint /flint-work/results/math/step_1 --expected-step 2 \
  --checkpoint /flint-work/results/math/step_2 --expected-step 4 \
  --output /flint-work/evidence/optimizer-resume-check.json \
  > /flint-work/evidence/optimizer-resume-check.log 2>&1

/opt/flint/bin/python-driver /flint-work/tools/check_refit_checkpoint.py \
  --head-sample /flint-work/evidence/changed-head-row-step1.json \
  --refit-dir /flint-work/evidence/grpo-full49-math-resume-01.refit \
  --output /flint-work/evidence/resume-refit-comparison.json \
  > /flint-work/evidence/resume-refit-comparison.log 2>&1

/opt/flint/bin/python-driver - <<'PY'
from datetime import datetime, timezone
from pathlib import Path
import json, math
r=Path('/flint-work/evidence')
a=json.loads((r/'training-audit-math-step1.json').read_text())
b=json.loads((r/'training-audit-math-step2.json').read_text())
o=json.loads((r/'optimizer-resume-check.json').read_text())
f=json.loads((r/'resume-refit-comparison.json').read_text())
assert b['status']=='pass' and b['summary']['head_changed_elements']>0
assert b['summary']['metadata_audited_tensor_count']==36734
assert b['summary']['sink_count']==49 and b['summary']['audited_saved_parameters_finite']
assert o['status']=='passed' and [x['value'] for x in o['records']]==[2,4]
assert f['status'] in ('pass','passed')
def head_hashes(audit):
    head=next(x for x in audit['tensors'] if x['name']=='lm_head.weight')
    return {(tuple(x['offsets']),tuple(x['shape'])):x['sha256'] for x in head['saved_regions']}
before,after=head_hashes(a),head_hashes(b)
assert before.keys()==after.keys() and before!=after, 'The full head must change again after resumed training'
rows=[json.loads(line) for line in (r/'grpo-full49-math-resume-01.metrics.jsonl').read_text().splitlines()]
metrics=next(x['metrics'] for x in rows if x.get('prefix')=='train' and x.get('step')==2 and 'gen_kl_error' in x.get('metrics',{}))
selected={key:metrics[key] for key in ['loss','grad_norm','reward','gen_kl_error','policy_kl_error','lr']}
assert all(isinstance(x,(int,float)) and math.isfinite(x) for x in selected.values())
assert selected['grad_norm']>0
report={'status':'pass','created_at_utc':datetime.now(timezone.utc).isoformat(),
        'phase':'grpo-full49-math-resume-01','grpo_step':2,'prompts':32,'responses_per_prompt':8,
        'rollouts_per_step':256,'train_global_batch_size':128,'selected_adam_counters':[2,4],
        'head_changed_again_after_resume':True,'checkpoint_metadata_tensors':36734,
        'protected_payload_tensors':50,'protected_payload_finite':True,
        'initial_learned_head_refit_sample_verified':True,'metrics':selected,
        'scope':'Full-model GRPO execution and restart; all tensor metadata and complete head/sink payloads. Ordinary parameter payload finiteness and full optimizer-moment equality remain unverified.'}
(r/'resume-validation.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
PY

nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader \
  > /flint-work/evidence/resume-final-gpu-processes.txt
