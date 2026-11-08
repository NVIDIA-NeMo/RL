# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

from typing import Any
from datasets import Dataset, load_dataset
import json
import itertools
import numpy as np
import torch
import random
import glob
import os
import pathlib
import base64

random.seed(42)


def format_judgebench_example(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Format JudgeBench data for GenRM evaluation."""
    # Extract conversation context and responses based on actual JudgeBench format
    context = data.get("question")
    context = "User: " + context
    response1 = data.get("response_A")
    response2 = data.get("response_B")
    source = data.get("source", "")
    pair_id = data.get("pair_id", "")

    
    # Parse the label field (e.g., "B>A" means B is better than A)
    label = data.get("label")
    preference = None
    if label == "A>B":
        preference = 0  # Response 1 (A) is better
    elif label == "B>A":
        preference = 1  # Response 2 (B) is better
    else:
        print("Invalid label: ", label)
        return []
    
    
    if "mmlu-pro" in source:
        domain = "knowledge"
    elif "livebench-reasoning" in source:
        domain = "reasoning"
    elif "livebench-math" in source:
        domain = "math"
    elif "livecodebench" in source:
        domain = "coding"
    else:
        domain = ""


    result1 = {
        "num_responses": 2,
        "preference": preference,
        "context": context,
        "response_1": response1,
        "response_2": response2,
        "domain": domain,
        "sample_id": pair_id
    }
    
    result2 = {
        "num_responses": 2,
        "preference": 1-preference,
        "context": context,
        "response_1": response2,
        "response_2": response1,
        "domain": domain,
        "sample_id": pair_id
    }
    
    return [result1, result2]
    #return [result1]



def format_rmbench_example(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Format RM-Bench data for GenRM evaluation while preserving RM-Bench structure."""
    
    prompt_text = data.get("prompt", "")
    chosen_responses = data.get("chosen", [])
    rejected_responses = data.get("rejected", [])
    domain = data.get("domain", "unknown")
    sample_id = data.get("id", "")
    
    # Ensure we have exactly 3 chosen and 3 rejected responses
    assert len(chosen_responses) == 3 and len(rejected_responses) == 3

    # Format as conversation context
    context = f"User: {prompt_text}"
    
    # Create all 9 comparisons (3 chosen x 3 rejected) for the 3x3 matrix
    examples = []
    for i, chosen_resp in enumerate(chosen_responses):
        for j, rejected_resp in enumerate(rejected_responses):
            # Randomly shuffle response order to avoid position bias
            preference = random.choice([0, 1])
            
            if preference == 0: # First response (chosen) is better
                response1 = chosen_resp
                response2 = rejected_resp
            else: # Second response (chosen) is better
                response1 = rejected_resp
                response2 = chosen_resp
            
            example = {
                "num_responses": 2,
                "preference": preference,
                "context": context,
                "response_1": response1,
                "response_2": response2,
                "domain": domain,
                "sample_id": sample_id,
                "chosen_style_idx": i,  # 0=concise, 1=detailed_plain, 2=detailed_markdown
                "rejected_style_idx": j,
            }
            examples.append(example)

    return examples


def format_rewardbench2_example(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Format RM-Bench data for GenRM evaluation."""
    # Extract prompt and responses
    prompt_text = data.get("prompt", "")
    chosen_responses = data.get("chosen", [])
    rejected_responses = data.get("rejected", [])
    subset = data.get("subset", [])
    sample_id = data.get("id", [])

    # Format as conversation context
    context = f"User: {prompt_text}"
    
    # Create comparisons between each chosen and rejected response
    examples = []
    
    if subset != "Ties":
        for rejected_resp in rejected_responses:
            # Create the GenRM prompt
            preference = random.choice([0, 1])
                    
            if preference == 0: # First response (chosen) is better
                response1 = chosen_responses[0]
                response2 = rejected_resp
            else: # Second response (chosen) is better
                response1 = rejected_resp
                response2 = chosen_responses[0]
                    
            example = {
                "num_responses": 2,
                "preference": preference,
                "context": context,
                "response_1": response1,
                "response_2": response2,
                "domain": subset,
                "sample_id": sample_id
            }
    
            examples.append(example)
    else:
        pairs = list(itertools.product(chosen_responses, rejected_responses))
        pairs3 = pairs[:(len(pairs) // 3) * 3]
        
        for c,r in pairs3:
            preference = random.choice([0, 1])
                    
            if preference == 0: # First response (chosen) is better
                response1 = c
                response2 = r
            else: # Second response (chosen) is better
                response1 = r
                response2 = c
                    
            example = {
                "num_responses": 2,
                "preference": preference,
                "context": context,
                "response_1": response1,
                "response_2": response2,
                "domain": subset,
                "sample_id": sample_id
            }
    
            examples.append(example)

    return examples


def format_rewardbench_example(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Format RM-Bench data for GenRM evaluation."""
    # Extract prompt and responses
    prompt_text = data.get("prompt", "")
    chosen_resp = data.get("chosen", "")
    rejected_resp = data.get("rejected", "")
    subset = data.get("subset", "")
    sample_id = data.get("id", "")

    # Format as conversation context
    context = f"User: {prompt_text}"
    
    # Create comparisons between each chosen and rejected response

    preference = random.choice([0, 1])
            
    if preference == 0: # First response (chosen) is better
        response1 = chosen_resp
        response2 = rejected_resp
    else: # Second response (chosen) is better
        response1 = rejected_resp
        response2 = chosen_resp

    domain_map = {
        "alpacaeval-easy": "Chat", 
        "alpacaeval-length": "Chat", 
        "alpacaeval-hard": "Chat", 
        "mt-bench-easy": "Chat", 
        "mt-bench-med": "Chat",
        "mt-bench-hard": "Chat_Hard", 
        "llmbar-natural": "Chat_Hard", 
        "llmbar-adver-neighbor": "Chat_Hard", 
        "llmbar-adver-GPTInst": "Chat_Hard", 
        "llmbar-adver-GPTOut": "Chat_Hard", 
        "llmbar-adver-manual": "Chat_Hard",
        "refusals-dangerous": "Safety", 
        "refusals-offensive": "Safety", 
        "xstest-should-refuse": "Safety", 
        "xstest-should-respond": "Safety", 
        "donotanswer": "Safety",
        "math-prm": "Reasoning", 
        "hep-cpp": "Reasoning", 
        "hep-go": "Reasoning", 
        "hep-java": "Reasoning", 
        "hep-js": "Reasoning", 
        "hep-python": "Reasoning", 
        "hep-rust": "Reasoning"
    }  

    example = {
        "num_responses": 2,
        "preference": preference,
        "context": context,
        "response_1": response1,
        "response_2": response2,
        "domain": domain_map[subset],
        "sample_id": sample_id
    }

    return [example]



def format_rmb_example(data: dict[str, Any]) -> list[dict[str, Any]]:
    """Format RMB data for GenRM evaluation."""
    # Extract conversation context
    conversation_input = data.get("conversation_input")
    context_parts = []
    for turn in conversation_input:
        role = turn.get("role")
        content = turn.get("content")
        if role.lower() == "user":
            context_parts.append(f"User: {content}")
        elif role.lower() == "assistant":
            context_parts.append(f"Assistant: {content}")
    
    context = "\n".join(context_parts)
    
    category_path = data.get("category_path")
    domain = "/".join(category_path.split("/")[:2])
    
    if "Pairwise_set" in domain:
        chosen_resps = [data.get("chosen").get("answer")]
        rejected_resps = [data.get("reject").get("answer")]
        sample_id = data.get("pair_uid")
    else: #BoN_set in domain
        chosen_resps = [data.get("bon_best").get("answer")]
        rejected_list = data.get("loser_list")
        rejected_resps = [each.get("answer") for each in rejected_list]
        sample_id = data.get("bon_uid")
    
    examples = []
    for chosen_resp in chosen_resps:
        for rejected_resp in rejected_resps:
            # Create comparisons with random order to avoid position bias
            preference = np.random.choice([0, 1])
            
            if preference == 0:  # First response (chosen) is better
                response1 = chosen_resp
                response2 = rejected_resp
            else:  # Second response (chosen) is better
                response1 = rejected_resp
                response2 = chosen_resp
            
            example = {
                "num_responses": 2,
                "preference": preference,
                "context": context,
                "response_1": response1,
                "response_2": response2,
                "category_path": category_path,
                "domain": domain,
                "sample_id": sample_id,
            }
            examples.append(example)
    
    return examples


class RMBDataset:
    """RMB dataset for GenRM evaluation."""
    
    def __init__(self, data_folder: str = "/lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_sft/users/yizhuj/RMB-Reward-Model-Benchmark/RMB_dataset"):
        
        # Find all JSON files in the data folder and subdirectories
        json_files = glob.glob(os.path.join(data_folder, "**", "*.json"), recursive=True)
        
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {data_folder}")
        
        print(f"Found {len(json_files)} JSON files: {[os.path.basename(f) for f in json_files]}")
        
        # Load and combine all data
        all_data = []
        for json_file in json_files:
            try:
                with open(json_file, 'r', encoding='utf-8') as f:
                    file_data = json.load(f)
                    if isinstance(file_data, list):
                        all_data.extend(file_data)
                    else:
                        all_data.append(file_data)
                print(f"Loaded {len(file_data) if isinstance(file_data, list) else 1} examples from {os.path.basename(json_file)}")
            except Exception as e:
                print(f"Error loading {json_file}: {e}")
                continue
        
        print(f"Total loaded examples: {len(all_data)}")
        
        # Format all examples
        all_formatted_examples = []
        for example in all_data:
            formatted_examples = format_rmb_example(example)
            all_formatted_examples.extend(formatted_examples)
        
        print(f"Total formatted examples: {len(all_formatted_examples)}")
        
        # Create a new dataset from the formatted examples
        self.formatted_ds = Dataset.from_list(all_formatted_examples)


class JudgeBenchDataset:
    """JudgeBench dataset for GenRM evaluation."""
    def __init__(self):
        # Load both splits
        #gpt_ds = load_dataset("ScalerLab/JudgeBench", split="gpt")
        #claude_ds = load_dataset("ScalerLab/JudgeBench", split="claude")
        
        # Merge the datasets
        #merged_ds = concatenate_datasets([gpt_ds, claude_ds])
        
        gpt_ds = load_dataset("ScalerLab/JudgeBench", split="gpt")

        # Manually expand the dataset by iterating through each example
        all_formatted_examples = []
        for example in gpt_ds:
            # Get the three formatted examples for this sample
            formatted_examples = format_judgebench_example(example)
            all_formatted_examples.extend(formatted_examples)
        
        # Create a new dataset from the expanded examples
        self.formatted_ds = Dataset.from_list(all_formatted_examples)



class RMBenchDataset:
    """RM-Bench dataset for GenRM evaluation."""
    
    def __init__(self):
        # Load both splits
        ds = load_dataset("THU-KEG/RM-Bench", split="train")
        # ds = ds.select(range(20))
        # Manually expand the dataset by iterating through each example
        all_formatted_examples = []
        for example in ds:
            # Get the three formatted examples for this sample
            formatted_examples = format_rmbench_example(example)
            all_formatted_examples.extend(formatted_examples)
        
        # Create a new dataset from the expanded examples
        self.formatted_ds = Dataset.from_list(all_formatted_examples)





class RewardBench2Dataset:
    """RewardBench dataset for GenRM evaluation."""
    
    def __init__(self):
        # Load all splits except Ties
        ds = load_dataset("allenai/reward-bench-2", split="test")
        #ds = ds.filter(lambda ex: ex["subset"] != "Ties")

        # Manually expand the dataset by iterating through each example
        all_formatted_examples = []
        for example in ds:
            # Get the three formatted examples for this sample
            formatted_examples = format_rewardbench2_example(example)
            all_formatted_examples.extend(formatted_examples)
        
        # Create a new dataset from the expanded examples
        self.formatted_ds = Dataset.from_list(all_formatted_examples)



class RewardBenchDataset:
    """RewardBench dataset for GenRM evaluation."""
    
    def __init__(self):
        # Load all splits except Ties
        ds = load_dataset("allenai/reward-bench", split="filtered")

        # Manually expand the dataset by iterating through each example
        all_formatted_examples = []
        for example in ds:
            # Get the three formatted examples for this sample
            formatted_examples = format_rewardbench_example(example)
            all_formatted_examples.extend(formatted_examples)
        
        # Create a new dataset from the expanded examples
        self.formatted_ds = Dataset.from_list(all_formatted_examples)


def encode_non_text_input_data(turns, media_dir: pathlib.Path=None) -> None:
    if isinstance(turns, str) or media_dir is None:
        return  # text-only task
    for turn in turns:
        if isinstance(turn["content"], list):  # if a string then it is text-only
            for item in turn["content"]:
                if item["type"] == "text":
                    continue
                if "{b64}" in item[item["type"]]["url"]:
                    media_file = media_dir / item["filename"]
                    with media_file.open("rb") as reader:
                        b64 = base64.b64encode(reader.read()).decode("utf-8")
                    # Replace the placeholder '{b64}' in the URL with the base64 file encoding.
                    item[item["type"]]["url"] = item[item["type"]]["url"].format(b64=b64)


class HelpSteer3LocalDatasetOldOld(torch.utils.data.Dataset):
    """Dataset for loading HelpSteer3 data from local JSONL files."""
    def __init__(self, data_path: str, media_path: str=None, task_name: str="genrm", shuffle_seed: int = -1, split: str="train", max_number: int=-1, double_swap=False):
        data = []
        if media_path is not None:
            media_path = pathlib.Path(media_path)
        with open(data_path, 'r', encoding='utf_8') as f:
            for line in f:
                one = json.loads(line)
                if one["args"]["num_responses"] == 2:
                    examples = []
                    
                    preference = 0 if one["args"]["preference_ranking"] <= 3 else 1
                    encode_non_text_input_data(one["args"]["context"], media_path)

                    example = {
                        "num_responses": 2,
                        "helpfulness_1": one["args"]["helpfulness_1"],  
                        "helpfulness_2": one["args"]["helpfulness_2"],
                        "preference": preference, 
                        "preference_ranking": one["args"]["preference_ranking"],
                        "context": one["args"]["context"],
                        "response1": one["args"]["response1"],
                        "response2": one["args"]["response2"],
                    }
                    if "principle" in one["args"]:
                        example["principle"] = one["args"]["principle"]
                    examples.append(example)

                    if double_swap:
                        example = {
                            "num_responses": 2,
                            "helpfulness_1": one["args"]["helpfulness_2"],  
                            "helpfulness_2": one["args"]["helpfulness_1"],
                            "preference": 1 - preference, 
                            "preference_ranking": 7 - one["args"]["preference_ranking"],
                            "context": one["args"]["context"],
                            "response1": one["args"]["response2"],
                            "response2": one["args"]["response1"],
                        }
                        if "principle" in one["args"]:
                            example["principle"] = one["args"]["principle"]
                        examples.append(example)

                    if split == "validation":
                        data.append(random.choice(examples))
                    else:
                        data += examples
        
        if shuffle_seed != -1:
            rng = np.random.default_rng(shuffle_seed)
            rng.shuffle(data)
            print(f"Shuffled the dataset with {len(data)} samples using seed {shuffle_seed}")

        if max_number != -1:
            data = data[:max_number]
            
        #self.data = Dataset.from_list(data)
        self.data = data
        self.formatted_ds = self.data
        self.task_name = task_name

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Add task_name for compatibility with AllTaskProcessedDataset
        item = self.data[idx].copy()
        item["task_name"] = self.task_name
        return item

class HelpSteer3LocalDatasetOld(torch.utils.data.Dataset):
    """Dataset for loading HelpSteer3 data from local JSONL files."""
    def __init__(self, data_path: str, media_path: str=None, task_name: str="genrm", shuffle_seed: int = -1, split: str="train", max_number: int=-1, double_swap=False):
        data = []
        if media_path is not None:
            media_path = pathlib.Path(media_path)
        with open(data_path, 'r', encoding='utf_8') as f:
            for line in f:
                one = json.loads(line)
                examples = []
                
                gt_overall = one["ground_truth_overall"]
                preference = 0 if gt_overall["ranking"] <= 3 else 1
                encode_non_text_input_data(one["responses_create_params"]["input"], media_path)

                example = {
                    "num_responses": 2,
                    "helpfulness_1": gt_overall["score_1"],
                    "helpfulness_2": gt_overall["score_2"],
                    "preference": preference, 
                    "preference_ranking": gt_overall["ranking"],
                    "context": one["responses_create_params"]["input"],
                    "response1": one["response_1"],
                    "response2": one["response_2"],
                }
                if one.get("principle") is not None:
                    example["principle"] = one["principle"]
                examples.append(example)

                if double_swap:
                    example = {
                        "num_responses": 2,
                        "helpfulness_1": gt_overall["score_2"],
                        "helpfulness_2": gt_overall["score_1"],
                        "preference": 1 - preference, 
                        "preference_ranking": 7 - gt_overall["ranking"],
                        "context": one["responses_create_params"]["input"],
                        "response1": one["response_2"],
                        "response2": one["response_1"],
                    }
                    if one.get("principle") is not None:
                        example["principle"] = one["principle"]
                    examples.append(example)

                if split == "validation":
                    data.append(random.choice(examples))
                else:
                    data += examples
        
        if shuffle_seed != -1:
            rng = np.random.default_rng(shuffle_seed)
            rng.shuffle(data)
            print(f"Shuffled the dataset with {len(data)} samples using seed {shuffle_seed}")

        if max_number != -1:
            data = data[:max_number]
            
        #self.data = Dataset.from_list(data)
        self.data = data
        self.formatted_ds = self.data
        self.task_name = task_name

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Add task_name for compatibility with AllTaskProcessedDataset
        item = self.data[idx].copy()
        item["task_name"] = self.task_name
        return item

class HelpSteer3LocalDataset(torch.utils.data.Dataset):
    """Dataset for loading HelpSteer3 data from local JSONL files."""
    def __init__(self, data_path: str, media_path: str=None, task_name: str="genrm", shuffle_seed: int = -1, split: str="train", max_number: int=-1, double_swap=False):
        data = []
        if media_path is not None:
            media_path = pathlib.Path(media_path)
        with open(data_path, 'r', encoding='utf_8') as f:
            for line in f:
                one = json.loads(line)
                examples = []
                
                gt_overall = one.get("ground_truth_overall")
                gt_rubrics = one.get("ground_truth_rubric_scores")
                preference = 0 if gt_overall["ranking"] <= 3 else 1
                encode_non_text_input_data(one["responses_create_params"]["input"], media_path)

                example = {
                    "num_responses": 2,
                    "context": one["responses_create_params"]["input"],
                    "response_1": one["response_1"],
                    "response_2": one["response_2"],
                    "preference": preference,
                    "ground_truth_overall": gt_overall,
                }
                # need to do it like this because downstream runners call principle = datum_dict.get("principle", DEFAULT_PRINCIPLE)
                if one.get("principle") is not None:
                    example["principle"] = one["principle"]
                if gt_rubrics is not None:
                    example["ground_truth_rubric_scores"] = gt_rubrics
                
                examples.append(example)

                if double_swap:
                    example = {
                        "num_responses": 2,
                        "context": one["responses_create_params"]["input"],
                        "response_1": one["response_2"],
                        "response_2": one["response_1"],
                        "preference": 1 - preference,
                    }
                    # need to do it like this because downstream runners call principle = datum_dict.get("principle", DEFAULT_PRINCIPLE)
                    if one.get("principle") is not None:
                        example["principle"] = one["principle"]
                    if gt_overall is not None:
                        example["ground_truth_overall"] = {'score_1': gt_overall.get("score_2"), 'score_2': gt_overall.get("score_1"), 'ranking': 7 - gt_overall["ranking"]}
                    if gt_rubrics is not None:
                        example["ground_truth_rubric_scores"] = [{'rubric_id': rl['rubric_id'], 'score_1': rl.get("score_2"), 'score_2': rl.get("score_1"), 'ranking': 7 - rl["ranking"]} for rl in gt_rubrics]
                    
                    examples.append(example)

                if split == "validation":
                    data.append(random.choice(examples))
                else:
                    data += examples
        
        if shuffle_seed != -1:
            rng = np.random.default_rng(shuffle_seed)
            rng.shuffle(data)
            print(f"Shuffled the dataset with {len(data)} samples using seed {shuffle_seed}")

        if max_number != -1:
            data = data[:max_number]
            
        #self.data = Dataset.from_list(data)
        self.data = data
        self.formatted_ds = self.data
        self.task_name = task_name

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Add task_name for compatibility with AllTaskProcessedDataset
        item = self.data[idx].copy()
        item["task_name"] = self.task_name
        return item
