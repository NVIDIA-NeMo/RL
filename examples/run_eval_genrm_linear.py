# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

import argparse
import json
import os
import pprint
import ray
import traceback
import yaml
from typing import Any, Optional, TypedDict

import torch
from datasets import Dataset
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig

from nemo_rl.algorithms.utils import get_tokenizer, set_seed
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset, eval_collate_fn
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
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.models.generation import configure_generation_config

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
    # Debug: Print the datum_dict to see what fields are available
    if idx < 3:  # Only print first few examples
        print(f"\n[DEBUG] Example {idx} datum_dict keys: {list(datum_dict.keys())}")
        for key in ["prompt", "num_responses", "label_1", "label_2", "preference", "ground_truth"]:
            if key in datum_dict:
                value = datum_dict[key]
                if isinstance(value, str) and len(value) > 100:
                    print(f"  {key}: {value}...")
                else:
                    print(f"  {key}: {value}")
    
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

    # Debug: Print extracted metadata
    if idx < 1:
        print(f"  Extracted metadata: {metadata}")


    return DatumSpec(
        message_log=message_log,
        length=token_length,
        extra_env_info=metadata,
        loss_multiplier=1.0,
        idx=idx,
        task_name="genrm_eval",
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Run GenRM evaluation")
    parser.add_argument(
        "--config", type=str, default=None, help="Path to YAML config file"
    )
    parser.add_argument(
        "--dataset", 
        type=str, 
        choices=["judgebench", "rmbench", "rewardbench2", "hs3local", "hs3train"],
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
        dataset_loader = HelpSteer3LocalDataset()
    elif dataset_name == "hs3train":
        train_data_path = '/lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_sft/users/yizhuj/datasets/hs3_genrm/train_data_base.jsonl'
        dataset_loader = HelpSteer3LocalDataset(train_data_path, split='train', max_number=10000)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")
    
    test_dataset = dataset_loader.formatted_ds
    if test_dataset is None or len(test_dataset) == 0:
        print(f"⚠️ Warning: {dataset_name} dataset is empty or failed to load.")
        
    
    print(f"  ✓ Loaded {len(test_dataset)} examples")
    
    # Debug: Print first example from the dataset
    '''
    if len(test_dataset) > 0:
        print("\n[DEBUG] First example from dataset:")
        first_example = test_dataset[0]
        print(f"  Keys: {list(first_example.keys())}")
        for key, value in first_example.items():
            if isinstance(value, str) and len(value) > 200:
                print(f"  {key}: {value[:200]}...")
            else:
                print(f"  {key}: {value}")
    '''

    # Create task spec
    eval_task_spec = TaskDataSpec(
        task_name="genrm_eval",
    )
    
    # Create processed dataset
    processed_dataset = AllTaskProcessedDataset(
        test_dataset,
        tokenizer,
        eval_task_spec,
        genrm_eval_data_processor,
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
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    
    # Calculate metrics
    calculate_metrics(results)


def evaluate_genrm_linear(actor_policy, vllm_generation, dataloader, output_file, tokenizer):
    """Run evaluation and save results."""
    results = []
    
    print("\n▶ Running evaluation...")
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        
        current_batch = batch.copy()
        batch_size = len(current_batch["message_log"])
        active_indices = torch.arange(batch_size)
        current_stop_strings = current_batch.get("stop_strings", [None] * batch_size)
        
        if len(active_indices) == 0:
            continue

        active_batch = current_batch.select_indices(active_indices)
        active_stop_strings = [current_stop_strings[i] for i in active_indices.tolist()]
        
        active_flat_messages, active_input_lengths = batched_message_log_to_flat_message(
            active_batch["message_log"],
            pad_value_dict={"token_ids": tokenizer.pad_token_id},
        )

        generation_input_data = BatchedDataDict[GenerationDatumSpec](
            {
                "input_ids": active_flat_messages["token_ids"],
                "input_lengths": active_input_lengths,
                "stop_strings": active_stop_strings,
            }
        )

        # generate_responses updates active_batch["message_log"] in-place
        active_batch, generated_ids, gen_metrics = generate_responses(
            vllm_generation,
            generation_input_data,
            active_batch,
            tokenizer,
            input_lengths=active_input_lengths,
            include_logprobs=False,
            greedy=False,
        )
        
        flat_messages, input_lengths = batched_message_log_to_flat_message(
            active_batch["message_log"],
            pad_value_dict={"token_ids": tokenizer.pad_token_id},
            make_sequence_length_divisible_by=1,
        )
        
        pred_data = BatchedDataDict[GenerationDatumSpec](
            {
                "input_ids": flat_messages["token_ids"],
                "input_lengths": input_lengths,
            }
        )
        pred_data.to("cpu")
        
        '''
        shards = pred_data.shard_by_batch_size(8, batch_size=None, allow_uneven_shards=True)
        
        #print("*** SHARDS_KEYS_EACH: ", [list(x.keys()) for x in shards], flush=True)
        print("*** BEFORE_SHARDS_SHAPE_EACH: ", [x["input_ids"].shape for x in shards], flush=True)
        #print("*** SHARDS_LENS_EACH: ", [x["input_lengths"] for x in shards], flush=True)
        #raise RuntimeError("all stop")
        
        max_bsize = max([x["input_ids"].shape[0] for x in shards])
        num_inserts = 0
        for x in shards:
            if x["input_ids"].shape[0] < max_bsize:
                x["input_ids"] = torch.cat([x["input_ids"], torch.zeros((max_bsize - x["input_ids"].shape[0], x["input_ids"].shape[-1]), device=x["input_ids"].device, dtype=x["input_ids"].dtype)], dim=0)
                x["input_lengths"] = torch.cat([x["input_lengths"], torch.tensor([1], device=x["input_lengths"].device, dtype=x["input_lengths"].dtype)], dim=0)
                num_inserts += 1
        
        print("*** AFTER_SHARDS_SHAPE_EACH: ", [x["input_ids"].shape for x in shards], flush=True)
        #raise RuntimeError("all stop")
        
        preds = actor_policy.get_linear_predictions(shards, 1)["logprobs"].tolist()
        if num_inserts > 0:
            preds = preds[:-num_inserts]
        '''
        
        preds = actor_policy.get_linear_predictions(pred_data, 1)
        generated_text = [x["content"] for ml in active_batch["message_log"] for x in ml if x["role"] == "assistant"]
        
        assert len(preds) == len(generated_text), "len(preds) != len(generated_text)"
        
        for idx, (output, pred_rank, metadata) in enumerate(zip(generated_text, preds, batch["extra_env_info"])):
            result = {
                "idx": batch["idx"][idx].item() if torch.is_tensor(batch["idx"][idx]) else batch["idx"][idx],
                "prediction": output,
                "metadata": metadata,
                "linear_ranking": pred_rank,
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
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Results saved to {output_file}")
    
    # Calculate metrics
    calculate_metrics(results)
    calculate_metrics_linear(results)


def calculate_metrics(results):
    """Calculate evaluation metrics."""
    correct_rankings = 0
    total_rankings = 0
    
    for result in results:
        total_rankings += 1
        if "predicted_ranking" in result and result["metadata"].get("preference") is not None:
            # Convert preference to expected ranking
            true_pref = result["metadata"]["preference"]
            extracted_pred_rank = result["predicted_ranking"]
            pred_rank = 0 if extracted_pred_rank <= 3 else 1
                
            if pred_rank == true_pref:
                correct_rankings += 1
    
    if total_rankings > 0:
        accuracy = correct_rankings / total_rankings
        print("\n📊 Evaluation Metrics:")
        print(f"  • Ranking Accuracy: {accuracy:.2%} ({correct_rankings}/{total_rankings})")
    else:
        print("\n⚠️ No valid rankings found in results")


def calculate_metrics_linear(results):
    """Calculate evaluation linear metrics."""
    correct_rankings = 0
    total_rankings = 0
    
    for result in results:
        total_rankings += 1
        if "linear_ranking" in result and result["metadata"].get("preference") is not None:
            # Convert preference to expected ranking
            true_pref = result["metadata"]["preference"]
            extracted_pred_rank = result["linear_ranking"]
            pred_rank = 1 if extracted_pred_rank >= 0.5 else 0
                
            if pred_rank == true_pref:
                correct_rankings += 1
    
    if total_rankings > 0:
        accuracy = correct_rankings / total_rankings
        print("\n📊 Linear Evaluation Metrics:")
        print(f"  • Linear Ranking Accuracy: {accuracy:.2%} ({correct_rankings}/{total_rankings})")
    else:
        print("\n⚠️ No valid linear rankings found in results")


def refit_policy_generation(
    policy: ColocatablePolicyInterface,
    policy_generation: GenerationInterface,
    colocated_inference: bool,
    _refit_buffer_size_gb: Optional[int] = None,
) -> None:
    """Refit the policy generation interface with the latest policy weights.

    Args:
        policy: The policy to provide weights to the inference engine.
        policy_generation: The inference engine to refit.
        _refit_buffer_size_gb: The size of the buffer to use for refitting.
            If it is None, the buffer size will be computed by the remaining memory.
            This parameter is primarily used for testing.
    """
    if colocated_inference:
        policy.offload_before_refit()
        policy_generation.prepare_for_generation(tags=["weights"])

    # update weights
    update_success = False
    if colocated_inference:
        # get model param keys, which is grouped by size
        grouped_param_keys = policy.prepare_weights_for_ipc(
            _refit_buffer_size_gb=_refit_buffer_size_gb
        )

        # do update
        for keys in grouped_param_keys:
            ipc_handles = policy.get_weights_ipc_handles(keys)
            assert "bt_beta" not in keys, "bt_beta detected in IPC handles, should never happen"
            update_success = policy_generation.update_weights(ipc_handles)
            if not update_success:
                break
    else:
        # prepare info for update weights
        state_dict_info = policy.prepare_info_for_collective()
        # update weights through nccl
        futures_train = policy.broadcast_weights_for_collective()
        futures_inference = policy_generation.update_weights_from_collective(
            state_dict_info
        )
        # wait for all futures to complete
        ray.get(futures_train)
        results = ray.get(futures_inference)
        update_success = all(result for result in results if result is not None)

    # check if update is successful
    if not update_success:
        error_tag = "cuda-ipc" if colocated_inference else "nccl"
        error_message = (
            "❌ Error: Updating weights for the generation policy failed during refit.\n"
            f"This often indicates an issue with {error_tag} or "
            "a problem within the generation backend (e.g., vLLM worker).\n"
        )
        raise RuntimeError(error_message)

    if colocated_inference:
        policy.offload_after_refit()
        policy_generation.prepare_for_generation(tags=["kv_cache"])


def main():
    args, overrides = parse_args()
    
    if not args.config:
        args.config = os.path.join(
            os.path.dirname(__file__), "configs", "genrm_eval.yaml"
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
    
    # Setup tokenizer
    tokenizer = get_tokenizer(ckpt_config["policy"]["tokenizer"])
    config["generation"] = configure_generation_config(
        config["generation"], tokenizer, is_eval=True
    )
    
    # Setup data
    dataset, dataset_loader = setup_data(
        tokenizer, 
        config["data"], 
        config["eval"]["dataset_name"],
    )
    
    # Create dataloader with eval_collate_fn
    dataloader = DataLoader(
        dataset,
        batch_size=config["eval"]["batch_size"],
        shuffle=False,
        collate_fn=eval_collate_fn,
    )
    
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
    
    refit_policy_generation(actor_policy, vllm_generation, True)
    
    # Prepare for generation
    #vllm_generation.prepare_for_generation()
    
    # Run evaluation
    output_file = config["eval"]["output_file"]
    evaluate_genrm_linear(actor_policy, vllm_generation, dataloader, output_file, tokenizer)
    
    # Cleanup
    vllm_generation.finish_generation()
    vllm_generation.shutdown()


if __name__ == "__main__":
    main()