# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

import argparse
import json
import os
import pprint
import traceback
from typing import Any, Optional, TypedDict

import torch
from datasets import Dataset
from omegaconf import OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from nemo_rl.algorithms.utils import get_tokenizer, set_seed
from nemo_rl.data import DataConfig
from nemo_rl.data.datasets import AllTaskProcessedDataset, eval_collate_fn
from nemo_rl.data.datasets.preference_datasets.reward_benchmarks import (
    JudgeBenchDataset,
    RMBenchDataset,
    RewardBenchDataset,
    RewardBench2Dataset,
    HelpSteer3LocalDataset,
    RMBDataset,
)
from nemo_rl.data.interfaces import DatumSpec, TaskDataSpec
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.virtual_cluster import ClusterConfig, RayVirtualCluster, init_ray
from nemo_rl.models.generation.vllm import VllmConfig, VllmGeneration
from nemo_rl.utils.config import load_config, parse_hydra_overrides
from nemo_rl.models.generation import configure_generation_config

from nemo_rl.environments.binary_genrm_environment import format_scoring_stage_prompt


class GenRMEvalConfig(TypedDict):
    dataset_name: str  
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
    context = datum_dict.get("context", "")
    response1 = datum_dict.get("response1", "")
    response2 = datum_dict.get("response2", "")
    prompt = format_scoring_stage_prompt(context, response1, response2)

    # Prepare metadata for environment
    metadata = datum_dict.copy()

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
    metadata = datum_dict.copy()

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
        choices=["judgebench", "rmbench", "rewardbench2", "rewardbench", "hs3local", "hs3train", "rmb"],
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
    elif dataset_name == "rewardbench":
        dataset_loader = RewardBenchDataset()
    elif dataset_name == "rewardbench2":
        dataset_loader = RewardBench2Dataset()
    elif dataset_name == "rmb":
        dataset_loader = RMBDataset()
    elif dataset_name == "hs3local":
        dataset_loader = HelpSteer3LocalDataset()
    elif dataset_name == "hs3train":
        train_data_path = '/lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_sft/users/yizhuj/datasets/hs3_genrm/train_data_base.jsonl'
        dataset_loader = HelpSteer3LocalDataset(train_data_path, split='train', max_number=30000)
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
            pred_rank = extracted_pred_rank
                
            if pred_rank == true_pref:
                correct_rankings += 1
    
    if total_rankings > 0:
        accuracy = correct_rankings / total_rankings
        print(f"\n📊 Evaluation Metrics:")
        print(f"  • Ranking Accuracy: {accuracy:.2%} ({correct_rankings}/{total_rankings})")
    else:
        print("\n⚠️ No valid rankings found in results")


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

    
    # Setup tokenizer
    tokenizer = get_tokenizer(config["tokenizer"])
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
        max_colocated_worker_groups=1,
    )
    
    # Setup vLLM
    print("\n▶ Setting up vLLM generation...")
    vllm_generation = VllmGeneration(cluster=cluster, config=config["generation"])
    
    # Prepare for generation
    vllm_generation.prepare_for_generation()
    
    # Run evaluation
    output_file = config["eval"]["output_file"]
    evaluate_genrm(vllm_generation, dataloader, output_file)
    
    # Cleanup
    vllm_generation.finish_generation()
    vllm_generation.shutdown()


if __name__ == "__main__":
    main()