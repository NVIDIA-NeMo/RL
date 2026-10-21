"""Verify the exact math processor retains nonempty prompt supervision budgets."""
import json
from pathlib import Path
from omegaconf import OmegaConf
from nemo_rl.algorithms.utils import get_tokenizer
from nemo_rl.algorithms.grpo import MasterConfig
from nemo_rl.data.interfaces import TaskDataSpec
from nemo_rl.data.processors import math_hf_data_processor
from nemo_rl.utils.config import load_config, register_omegaconf_resolvers

register_omegaconf_resolvers()
recipe='examples/configs/recipes/llm/grpo-flint-49layer-1n4g-automodel-sglang-actckpt-quick.yaml'
cfg=MasterConfig(**OmegaConf.to_container(load_config(recipe),resolve=True))
tokenizer=get_tokenizer(cfg.policy['tokenizer'])
rows=[json.loads(s) for s in Path(cfg.data['train']['data_path']).read_text().splitlines() if s.strip()]
spec=TaskDataSpec(task_name='math',prompt_file=None,system_prompt_file=None)
observed=[]
for index,row in enumerate(rows):
 datum={'messages':[{'role':'user','content':row['input']},{'role':'assistant','content':row['output']}],'task_name':'math'}
 value=math_hf_data_processor(datum,spec,tokenizer,cfg.data['max_input_seq_length'],index)
 observed.append({'index':index,'raw_length':value['length'],'input_token_count':sum(len(m['token_ids']) for m in value['message_log']),'loss_multiplier':value['loss_multiplier']})
max_input=max(r['input_token_count'] for r in observed)
report={'status':'passed','record_count':len(rows),'prompts_per_step':cfg.grpo.num_prompts_per_step,'responses_per_prompt':cfg.grpo.num_generations_per_prompt,'train_global_batch_size':cfg.policy['train_global_batch_size'],'generation_max_new_tokens':cfg.policy['generation']['max_new_tokens'],'max_total_sequence_length':cfg.policy['max_total_sequence_length'],'min_input_tokens':min(r['input_token_count'] for r in observed),'max_input_tokens':max_input,'masked_prompts':sum(r['loss_multiplier']==0 for r in observed),'records':observed}
if report['masked_prompts'] or len(rows)<cfg.grpo.num_prompts_per_step or max_input+report['generation_max_new_tokens']>report['max_total_sequence_length']: report['status']='failed'
p=Path('/flint-work/evidence/prompt-budget-check.json');p.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps({k:v for k,v in report.items() if k!='records'}),flush=True)
if report['status']!='passed': raise SystemExit(1)
