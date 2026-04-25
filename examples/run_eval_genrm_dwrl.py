# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

import argparse
import copy
import functools
import json
import itertools
import os
import pprint
import ray
import re
import traceback
import yaml
from typing import Any, Optional, TypedDict

import torch
from datasets import Dataset
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig

from nemo_rl.algorithms.grpo import refit_policy_generation
from nemo_rl.algorithms.utils import get_tokenizer, set_seed
from nemo_rl.data import DataConfig
from nemo_rl.data.collate_fn import rl_collate_fn, preference_collate_fn, eval_collate_fn
from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.datasets.preference_datasets.reward_benchmarks import (
    JudgeBenchDataset,
    RMBenchDataset,
    RewardBench2Dataset,
    HelpSteer3LocalDataset,
)
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationInterface,
    GenerationOutputSpec,
)
from nemo_rl.data.interfaces import DatumSpec, TaskDataSpec
from nemo_rl.data.llm_message_utils import batched_message_log_to_flat_message
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import ClusterConfig, RayVirtualCluster, init_ray
from nemo_rl.experience.rollouts import generate_responses
from nemo_rl.models.generation.interfaces import GenerationInterface
from nemo_rl.models.generation.vllm import VllmConfig, VllmGeneration
from nemo_rl.models.policy.interfaces import ColocatablePolicyInterface
from nemo_rl.models.policy.lm_policy import Policy
from nemo_rl.utils.config import load_config, parse_hydra_overrides, register_omegaconf_resolvers
from nemo_rl.models.generation import configure_generation_config


DWRL_PROMPT_TEMPLATE = """You are an expert evaluation judge specializing in the assessment of LLM responses. You are impartial, rigorous, and consistent. Given the conversation context and a single response to the user's latest query, you will follow the evaluation plan and scoring guidelines exactly as written below.

#### Conversation Context ####
{context}

#### Response to be Scored ####
[The Start of Response]
{response}
[The End of Response]

Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user prompt. Begin your evaluation by generating your own answer to the prompt. You must provide your answer before judging any answers. When evaluating the assistant's response, compare the assistant's response with your answer. You must identify and correct any mistakes or inaccurate information. Then consider if the assistant's response is helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when the user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user rather than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive. Then consider the creativity and novelty of the assistant's response when needed. Finally, identify any missing important information in the assistant's response which would be beneficial to include when responding to the user prompt.

#### Scoring Guidelines ####
Based on the evaluation plan above, assign scores using these scales:
**Individual Helpfulness Scores (1-5):**
- 5: Extremely Helpful - Completely aligned with what the user was asking for
- 4: Mostly Helpful - Generally useful with minor room for improvement
- 3: Partially Helpful - Misses the overall goal in some way
- 2: Borderline Unhelpful - Mostly doesn't capture what the user wanted
- 1: Not Helpful - Completely missed the essence of the request

#### Output Format ####
Analyze step by step following the evaluation plan, then provide your judgment as JSON:
```json
{{
    "response_analysis": "Your detailed analysis of the Response based on the evaluation plan",
    "score": <1-5>
}}
```"""

def flatten_to_single_turn(message_log):
    ret = ""
    for idx, message in enumerate(message_log):
        if message["role"] == "system":
            ret += "System: " + message["content"].strip() + "\n\n"
        elif message["role"] == "user":
            ret += "User: " + message["content"].strip() + "\n\n"
        elif message["role"] == "assistant":
            resp_no_thinking1 = re.sub(r"(?i)(?s)(\<think\>)(.*?)(\<\/think\>)", "", message["content"]).strip()
            resp_no_thinking2 = re.sub(r"(?i)(?s)(.*?)(\<\/think\>)", "", resp_no_thinking1).strip()
            ret += "Assistant: " + resp_no_thinking2 + "\n\n"
    
    return ret.strip()


def get_json_response(response):
    resp_no_thinking1 = re.sub(r"(?i)(?s)(\<think\>)(.*?)(\<\/think\>)", "", response).strip()
    resp_no_thinking2 = re.sub(r"(?i)(?s)(.*?)(\<\/think\>)", "", resp_no_thinking1).strip()
    
    try:
        jp = json.loads(re.search(r"(?i)(?s)(\{\n)(.*?)(\n\})", resp_no_thinking2).group(0))
        return jp
    except:
        return resp_no_thinking2


class GenRMEvalConfig(TypedDict):
    dataset_name: str  # "judgebench", "rmbench", or "rewardbench2"
    batch_size: int
    seed: int
    output_file: str


class MasterConfig(TypedDict):
    generation: VllmConfig
    data: DataConfig
    eval: GenRMEvalConfig
    tokenizer: dict
    cluster: ClusterConfig


def genrm_eval_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process evaluation data for GenRM format."""
    
    # The datum_dict already contains the formatted prompt from format_judgebench_example
    prompt = datum_dict.get("prompt", "")
    
    message_log = []
    user_message = {
        "role": "user",
        "content": prompt,
    }

    message = tokenizer.apply_chat_template(  # type: ignore
        [user_message],
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
    )

    user_message["token_ids"] = tokenizer(message, return_tensors="pt")["input_ids"][0]
    user_message["content"] = message  
    token_length = len(user_message["token_ids"])
    message_log.append(user_message)
    
    
    # Extract metadata - make sure we're getting the actual values
    metadata = {
        "num_responses": datum_dict.get("num_responses", 2),
        "helpfulness_1": datum_dict.get("label_1"),
        "helpfulness_2": datum_dict.get("label_2"),
        "preference_ranking": datum_dict.get("preference"),
        "ground_truth": datum_dict.get("ground_truth"),
        "context": datum_dict.get("context", ""),
        "response1": datum_dict.get("response1", ""),
        "response2": datum_dict.get("response2", ""),
    }
            
    for key in ["domain", "sample_id", "chosen_style_idx", "rejected_style_idx"]:
        if key in datum_dict.keys():
            metadata[key] = datum_dict.get(key)


    return DatumSpec(
        message_log=message_log,
        length=token_length,
        extra_env_info=metadata,
        loss_multiplier=1.0,
        idx=idx,
        task_name="genrm_eval",
    )

def dwrl_preference_data_processor(
    datum_dict: dict[str, Any],
    task_data_spec: TaskDataSpec,
    tokenizer,
    max_seq_length: int,
    idx: int,
) -> DatumSpec:
    """Process HelpSteer3 data for DWRL GenRM training."""
    def get_message_log(prompt_str):
        # Create message log
        message_log = []
        user_message = {
            "role": "user",
            "content": prompt_str,
        }
        message = tokenizer.apply_chat_template(  # type: ignore
            [user_message],
            tokenize=False,
            add_generation_prompt=True,
            add_special_tokens=False,
            #enable_thinking=False,
        )

        user_message["token_ids"] = tokenizer(message, return_tensors="pt")["input_ids"][0]
        user_message["content"] = message  
        message_log.append(user_message)
        
        return message_log
    
    # Extract the prompt (which contains the full formatted prompt with responses)
    context = datum_dict.get("context", "")
    resp1 = datum_dict.get("response1", "")
    resp2 = datum_dict.get("response2", "")
    pref_binary = datum_dict["preference"]
    if pref_binary == 0:
        chosen = resp1
        reject = resp2
    else:
        chosen = resp2
        reject = resp1
    
    prompt_chosen = DWRL_PROMPT_TEMPLATE.format(context=context, response=resp1)
    prompt_reject = DWRL_PROMPT_TEMPLATE.format(context=context, response=resp2)

    # Prepare metadata for environment
    metadata = copy.deepcopy(datum_dict)
    #metadata['pair_idx'] = pair_idx
    #metadata['num_responses'] = 1

    # Extract system prompt if present
    # system_prompt = datum_dict.get("system_prompt", "")
    
    message_log_chosen = get_message_log(prompt_chosen)
    message_log_reject = get_message_log(prompt_reject)
    
    # Calculate total length
    total_length_chosen = sum(len(msg["token_ids"]) for msg in message_log_chosen)
    total_length_reject = sum(len(msg["token_ids"]) for msg in message_log_reject)
    
    # Check if we need to truncate
    loss_multiplier = 1.0
    if total_length_chosen > max_seq_length or total_length_reject > max_seq_length:
        # Truncate messages
        for msg in message_log_chosen:
            msg["token_ids"] = msg["token_ids"][: min(4, max_seq_length // len(message_log_chosen)) ]
        for msg in message_log_reject:
            msg["token_ids"] = msg["token_ids"][: min(4, max_seq_length // len(message_log_reject)) ]
        loss_multiplier = 0.0

    return DatumSpec(
        message_log_chosen=message_log_chosen,
        message_log_rejected=message_log_reject,
        length_chosen=total_length_chosen,
        length_rejected=total_length_reject,
        extra_env_info=metadata,
        loss_multiplier=loss_multiplier,
        idx=idx,
        task_name=task_data_spec.task_name,
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run GenRM evaluation")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )
    parser.add_argument(
        "--dataset", 
        type=str,
        choices=["judgebench", "rmbench", "rewardbench2", "hs3local"],
        default=None,
        help="Dataset to evaluate on (overrides config)"
    )
    
    args, overrides = parser.parse_known_args()
    return args, overrides


def setup_data(tokenizer, data_config, dataset_name):
    """Set up evaluation dataset."""
    print(f"\n▶ Setting up {dataset_name} dataset...")
    
    # Load dataset
    if dataset_name == "judgebench":
        dataset_loader = JudgeBenchDataset()
    elif dataset_name == "rmbench":
        dataset_loader = RMBenchDataset()
    elif dataset_name == "rewardbench2":
        dataset_loader = RewardBench2Dataset()
    elif dataset_name == "hs3local":
        dataset_loader = HelpSteer3LocalDataset(data_config['eval_data_file'], split='validation', max_number=10000, double_swap=False)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    test_dataset = dataset_loader.formatted_ds
    if test_dataset is None or len(test_dataset) == 0:
        print(f"⚠️ Warning: {dataset_name} dataset is empty or failed to load.")
        
    
    print(f"  ✓ Loaded {len(test_dataset)} examples")

    # Create task spec
    eval_task_spec = TaskDataSpec(
        task_name="genrm_dwrl",
    )
    
    # Create processed dataset
    processed_dataset = AllTaskProcessedDataset(
        test_dataset,
        tokenizer,
        eval_task_spec,
        dwrl_preference_data_processor,
        max_seq_length=data_config["max_input_seq_length"],
    )
    
    return processed_dataset, dataset_loader


def extract_boxed_content(text: str) -> Optional[str]:
    """Extract content from \\boxed{} format."""
    import re
    
    # Try to find \boxed{...} pattern
    pattern = r'\\boxed\{([^}]+)\}'
    match = re.search(pattern, text)
    if match:
        return match.group(1).strip()
    
    # Alternative: look for content between curly braces after boxed
    if '\\boxed{' in text:
        start = text.find('\\boxed{') + len('\\boxed{')
        # Find the matching closing brace
        brace_count = 1
        i = start
        while i < len(text) and brace_count > 0:
            if text[i] == '{':
                brace_count += 1
            elif text[i] == '}':
                brace_count -= 1
            i += 1
        if brace_count == 0:
            return text[start:i-1].strip()
    
    return None


def evaluate_genrm(vllm_generation, dataloader, output_file):
    """Run evaluation and save results."""
    results = []
    
    print("\n▶ Running evaluation...")
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        '''
        # Debug first batch
        if batch_idx == 0:
            print(f"\n[DEBUG] First batch structure:")
            print(f"  Batch keys: {list(batch.keys())}")
            print(f"  Batch size: {len(batch['message_log'])}")
            if len(batch['message_log']) > 0:
                print(f"  First message_log structure: {[msg['role'] for msg in batch['message_log'][0]]}")
                print(f"  First metadata: {batch['extra_env_info'][0]}")
        '''

        prompts = []
        for message_log in batch["message_log"]:
            # Extract just the content from the user message
            if message_log and len(message_log) > 0 and message_log[0]["role"] == "user":
                content = message_log[0]["content"]
                prompts.append(content)
            else:
                prompts.append("")
                print(f"[WARNING] Empty or invalid message_log structure")
        
        # Create generation input
        inputs = BatchedDataDict({"prompts": prompts})
        print(prompts[:1])
        # Generate using vLLM
        try: 
            outputs = vllm_generation.generate_text(inputs)
            generated_texts = outputs.get("texts", [""] * len(prompts))
            
            # Debug first generation
            if batch_idx == 0 and len(generated_texts) > 0:
                print(f"\n[DEBUG] First generation output (truncated): {generated_texts[0][:500]}...")
        except Exception as e:
            print(f"[ERROR] Generation failed: {e}")
            generated_texts = [""] * len(prompts)
        
        # Process outputs and compare with ground truth
        for idx, (output, metadata) in enumerate(zip(generated_texts, batch["extra_env_info"])):
            result = {
                "idx": batch["idx"][idx].item() if torch.is_tensor(batch["idx"][idx]) else batch["idx"][idx],
                "prediction": output,
                "metadata": metadata,
            }
            
            # Parse the prediction to extract scores
            try:
                # Extract individual scores
                if "[The Begin of Individual Scores]" in output and "[The End of Individual Scores]" in output:
                    scores_section = output.split("[The Begin of Individual Scores]")[1].split("[The End of Individual Scores]")[0]
                    scores_text = extract_boxed_content(scores_section)
                    if scores_text:
                        # Split by comma and convert to integers
                        scores = []
                        for s in scores_text.split(","):
                            s = s.strip()
                            try:
                                scores.append(int(s))
                            except ValueError:
                                # Try to extract just numbers
                                import re
                                num_match = re.search(r'\d+', s)
                                if num_match:
                                    scores.append(int(num_match.group()))
                                else:
                                    scores.append(0)  # Default if can't parse
                        result["predicted_scores"] = scores
                
                # Extract ranking score if present
                if "[The Begin of Ranking Score]" in output and "[The End of Ranking Score]" in output:
                    ranking_section = output.split("[The Begin of Ranking Score]")[1].split("[The End of Ranking Score]")[0]
                    ranking_text = extract_boxed_content(ranking_section)
                    if ranking_text:
                        try:
                            result["predicted_ranking"] = int(ranking_text.strip())
                        except ValueError:
                            # Try to extract just the number
                            import re
                            num_match = re.search(r'\d+', ranking_text)
                            if num_match:
                                result["predicted_ranking"] = int(num_match.group())
                    
            except Exception as e:
                print(f"Error parsing output for idx {result['idx']}: {e}")
                result["parse_error"] = str(e)
            
            results.append(result)
    
    # Save results
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    with open(output_file, "w", encoding="utf_8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    
    # Calculate metrics
    calculate_metrics(results)


def evaluate_genrm_dwrl(master_config, actor_policy, vllm_generation, dataloader, output_file, tokenizer):
    """Run evaluation and save results."""
    
    if os.path.exists(output_file):
        return
    
    os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    output_file_tmp = output_file + ".tmp"
    
    if os.path.exists(output_file_tmp):
        with open(output_file_tmp, "r", encoding="utf_8") as fr:
            existing_file = [json.loads(line) for line in fr]
        existing_ids = {x['idx'] for x in existing_file}
    else:
        existing_ids = {}
    
    #processed_batches = []
    print("\n▶ Running evaluation...")
    with open(output_file_tmp, "a", encoding="utf_8", newline="\n", buffering=1) as fw:
        for batch_idx, batch in enumerate(tqdm(dataloader)):
            
            processed_batches = []
            current_batch = batch.copy()
            batch_size = len(current_batch["message_log"])
            print("**** BATCH_SIZE: ", batch_size, flush=True)
            #print("**** BATCH_IDX: ", current_batch['idx'], flush=True)
            #active_indices = torch.arange(batch_size)
            active_indices = torch.LongTensor([idx for idx,x in enumerate(current_batch['idx']) if x not in existing_ids])
            if len(active_indices) == 0:
                continue
            current_stop_strings = current_batch.get("stop_strings", [None] * batch_size)
            
            if len(active_indices) == 0:
                continue
    
            active_batch = current_batch.select_indices(active_indices)
            active_stop_strings = [current_stop_strings[i] for i in active_indices.tolist()]
            
            active_flat_messages, active_input_lengths = batched_message_log_to_flat_message(
                active_batch["message_log"],
                pad_value_dict={"token_ids": tokenizer.pad_token_id},
                make_sequence_length_divisible_by=1,#master_config["policy"]["make_sequence_length_divisible_by"],
            )
    
            generation_input_data = BatchedDataDict[GenerationDatumSpec](
                {
                    "input_ids": active_flat_messages["token_ids"],
                    "input_lengths": active_input_lengths,
                    "stop_strings": active_stop_strings,
                }
            )
    
            # generate_responses updates active_batch["message_log"] in-place
            vllm_generation.prepare_for_generation()
            active_batch, _, _ = generate_responses(
                vllm_generation,
                generation_input_data,
                active_batch,
                tokenizer,
                input_lengths=active_input_lengths,
                include_logprobs=False,
                greedy=False,
            )
            vllm_generation.finish_generation()
            
            # Inject the second user turn for DWRL
            new_msg_log = []
            for outer in active_batch['message_log']:
                p = []
                for inner in outer:
                    if inner["role"] == "environment":
                        continue
                    pp = copy.deepcopy(inner)
                    p.append(pp)
                px = {"role": "user", "content": master_config["eval"]["dwrl"]["bt_prompt"]}
                npx = tokenizer.apply_chat_template([px], tokenize=False, add_generation_prompt=True, add_special_tokens=False, enable_thinking=False)
                px['content'] = npx.replace("<|im_start|>system\n<|im_end|>\n", "").replace("<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n", "")# + "\n\n</think>\n\n"
                px["token_ids"] = tokenizer(px['content'], return_tensors="pt")["input_ids"][0]
                p.append(px)
                new_msg_log.append(p)
    
            new_fmt_list = []
            for idx, (msg_log, idx, extra_env_info, loss_mx) in enumerate(zip(new_msg_log, active_batch['idx'], active_batch['extra_env_info'], active_batch['loss_multiplier'])):
                length = sum(len(m["token_ids"]) for m in msg_log)
                output = {"message_log": msg_log, "length": length, "extra_env_info": extra_env_info, "loss_multiplier": loss_mx, "idx": idx, "task_name": "genrm_dwrl"}
                new_fmt_list.append(output)
            
            # Collate the new repeated_batch
            active_batch_2 = rl_collate_fn(new_fmt_list)
            
            flat_messages, input_lengths = batched_message_log_to_flat_message(
                active_batch_2["message_log"],
                pad_value_dict={"token_ids": tokenizer.pad_token_id},
                make_sequence_length_divisible_by=1,#master_config["policy"]["make_sequence_length_divisible_by"],
            )
            
            yes_position = tokenizer.encode(master_config["eval"]["dwrl"]["score_token"])[0]
            yes_tensor = torch.tensor([yes_position], device=flat_messages["token_ids"].device).long()
            list_with_yes = [torch.cat([flat_messages["token_ids"][idx][:input_lengths[idx]], yes_tensor, flat_messages["token_ids"][idx][input_lengths[idx]:]], dim=-1) for idx in range(len(input_lengths))]
            input_ids_with_yes = torch.stack(list_with_yes, dim=0)
            
            pred_data = BatchedDataDict[GenerationDatumSpec](
                {
                    "input_ids": input_ids_with_yes,
                    "input_lengths": input_lengths + 1,
                }
            )
            pred_data.to("cpu")
            
            actor_policy.prepare_for_lp_inference()
            logprobs = actor_policy.get_logprobs(pred_data)["logprobs"]
            final_logprobs = logprobs.cpu().gather(-1, input_lengths.unsqueeze(-1)).squeeze(-1)
            rewards = final_logprobs.cpu().exp()
            
            print("**** REWARDS: ", rewards, flush=True)
            
            active_batch['rewards'] = rewards.cpu().tolist()
            
            for sub_idx in active_indices.split(2):
                sub = active_batch.select_indices(sub_idx)
                processed_batches.append(sub)
            
            del active_batch, active_flat_messages, active_input_lengths, active_batch_2, flat_messages, input_lengths, yes_tensor, list_with_yes, input_ids_with_yes, pred_data, logprobs, final_logprobs
            torch.cuda.empty_cache()
            
            for pb in processed_batches:
                #print("*** PB_IDX: ", pb['idx'], flush=True)
                #print("*** PB_LEN: ", len(pb['idx']), flush=True)
                #print("*** PB_REWARDS: ", pb['rewards'], flush=True)
                #print("*** PB_RESPONSE: ", pb['message_log'][0][-1]['content'], flush=True)
                #print("*** PB_METADATA_KEYS: ", list(pb["extra_env_info"][0].keys()), flush=True)
                assert len(set(pb['idx'])) == 1, "ID check failed"
                assert len(pb['message_log']) == 2, "wrong length, should be 2"
                assert pb["extra_env_info"][0]["preference"] == pb["extra_env_info"][-1]["preference"], "mismatched preference scores!"
                #if 0 in pb['loss_multiplier']:
                #    continue
                
                payload = {}
                payload['idx'] = pb['idx'][0]
                payload['prediction_1'] = get_json_response(pb['message_log'][0][-1]['content'])
                payload['prediction_2'] = get_json_response(pb['message_log'][-1][-1]['content'])
                payload['metadata'] = pb["extra_env_info"][0]
                payload['loss_multiplier'] = pb['loss_multiplier'].tolist()
                payload['score_1'] = pb['rewards'][0]
                payload['score_2'] = pb['rewards'][-1]
                
                fw.write(json.dumps(payload, ensure_ascii=False) + "\n")
            
            del processed_batches

    # Save results
    #os.makedirs(os.path.dirname(output_file) if os.path.dirname(output_file) else ".", exist_ok=True)
    #with open(output_file, "w") as f:
    #    json.dump(results, f, indent=2)
    
    #os.replace(output_file_tmp, output_file)
    with open(output_file_tmp, "r", encoding="utf_8") as fr:
        results = [json.loads(line) for line in fr]
    with open(output_file, "w", encoding="utf_8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    os.remove(output_file_tmp)
    
    print(f"\n✓ Results saved to {output_file}")
    
    # Calculate metrics
    calculate_metrics(results)


def calculate_metrics(results):
    """Calculate evaluation metrics."""
    correct_rankings = 0
    total_rankings = 0
    
    for result in results:
        total_rankings += 1
        
        truth = result["metadata"]["preference"]
        correct_rankings += truth == int(result['score_2'] > result['score_1'])
        #correct_rankings += result['score_1'] > result['score_2']
    
    if total_rankings > 0:
        accuracy = correct_rankings / total_rankings
        print("\n📊 Evaluation Metrics:")
        print(f"  • Ranking Accuracy: {accuracy:.2%} ({correct_rankings}/{total_rankings})")
    else:
        print("\n⚠️ No valid rankings found in results")


def main():
    register_omegaconf_resolvers()
    args, overrides = parse_args()
    
    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "genrm_eval_dwrl.yaml"
        )
    
    config = load_config(args.config)
    print(f"Loaded configuration from: {args.config}")
    
    if overrides:
        print(f"Overrides: {overrides}")
        config = parse_hydra_overrides(config, overrides)
    
    # Override dataset if specified in command line
    if args.dataset:
        config["eval"]["dataset_name"] = args.dataset
    
    config: MasterConfig = OmegaConf.to_container(config, resolve=True)
    
    # Print config
    print("Final config:")
    pprint.pprint(config)
    
    # Set seed
    set_seed(config["eval"]["seed"])
    
    # Initialize Ray
    init_ray()

    orig_ckpt_loc = config["generation"]["model_name"]
    with open(os.path.join(orig_ckpt_loc, "config.yaml"), "r", encoding="utf_8") as f:
        ckpt_config = yaml.safe_load(f)
    
    if os.path.exists(tokenizer_path_on_disk := os.path.join(orig_ckpt_loc, "policy", "tokenizer")):
        ckpt_config["policy"]["tokenizer"]["name"] = tokenizer_path_on_disk
    # Setup tokenizer
    tokenizer = get_tokenizer(ckpt_config["policy"]["tokenizer"])
    if not os.path.exists(tokenizer_path_on_disk) and "Qwen3" in ckpt_config["policy"]["tokenizer"]["name"]:
        print("****** Qwen3 tokenizer detected! Swapping in fixed jinja template! ******", flush=True)
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "qwen3_fixed_template.jinja"), "r", encoding="utf_8") as fr:
            qwen3_fixed_template = fr.read()
        tokenizer.chat_template = qwen3_fixed_template
    config["generation"] = configure_generation_config(
        config["generation"], tokenizer, is_eval=True
    )
    
    
    config["policy"]["dtensor_cfg"]["enabled"] = ckpt_config["policy"]["dtensor_cfg"]["enabled"]
    config["policy"]["megatron_cfg"]["enabled"] = ckpt_config["policy"]["megatron_cfg"]["enabled"]
    if "dtensor_cfg" in ckpt_config["policy"]:
        ckpt_config["policy"]["dtensor_cfg"] = config["policy"]["dtensor_cfg"]
    if "megatron_cfg" in ckpt_config["policy"]:
        ckpt_config["policy"]["megatron_cfg"] = config["policy"]["megatron_cfg"]
    ckpt_config["policy"]["generation_batch_size"] = config["policy"]["generation_batch_size"]
    ckpt_config["policy"]["logprob_batch_size"] = config["policy"]["logprob_batch_size"]
    ckpt_config["policy"]["max_total_sequence_length"] = config["policy"]["max_total_sequence_length"]
    ckpt_config["policy"]["make_sequence_length_divisible_by"] = config["policy"]["make_sequence_length_divisible_by"]
    
    
    # Setup data
    dataset, _ = setup_data(
        tokenizer, 
        config["data"], 
        config["eval"]["dataset_name"],
    )
    
    # Create dataloader with eval_collate_fn
    dataloader = DataLoader(
        dataset,
        batch_size=config["eval"]["batch_size"],
        shuffle=False,
        collate_fn=functools.partial(preference_collate_fn,
                                     tokenizer=tokenizer,
                                     make_sequence_length_divisible_by=1,
                                     add_loss_mask=False,
                                     return_batch_only=True
                                     ),
    )
    
    if config["policy"]["megatron_cfg"]["enabled"]:
        ckpt_config["policy"]["megatron_cfg"]["train_iters"] = len(dataloader)
    
    # Setup cluster
    print("\n▶ Setting up compute cluster...")
    cluster = RayVirtualCluster(
        name="genrm_eval_cluster",
        bundle_ct_per_node_list=[config["cluster"]["gpus_per_node"]]
        * config["cluster"]["num_nodes"],
        use_gpus=True,
        num_gpus_per_node=config["cluster"]["gpus_per_node"],
        max_colocated_worker_groups=2,
    )
    train_cluster = cluster
    inference_cluster = cluster
    
    # Setup vLLM
    print("\n▶ Setting up vLLM generation...")
    config["generation"]["model_name"] = ckpt_config["policy"]["model_name"]
    vllm_generation = VllmGeneration(cluster=inference_cluster, config=config["generation"])
    vllm_generation.finish_generation()
    
    # load in the actor policy
    actor_policy = Policy(
        cluster=train_cluster,
        config=ckpt_config["policy"],
        tokenizer=tokenizer,
        weights_path=os.path.join(orig_ckpt_loc, "policy", "weights"),
        optimizer_path=None,
        init_optimizer=False,
    )
    
    # prepare refit info
    state_dict_info = actor_policy.prepare_refit_info()
    vllm_generation.prepare_refit_info(state_dict_info)
    refit_policy_generation(actor_policy, vllm_generation, True)
    
    # Prepare for generation
    #vllm_generation.prepare_for_generation()
    
    # Run evaluation
    output_file = config["eval"]["output_file"]
    evaluate_genrm_dwrl(config, actor_policy, vllm_generation, dataloader, output_file, tokenizer)
    
    # Cleanup
    vllm_generation.finish_generation()
    vllm_generation.shutdown()


if __name__ == "__main__":
    main()