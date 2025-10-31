# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.

from typing import Any, Optional

from datasets import Dataset, load_dataset, concatenate_datasets
from nemo_rl.data.interfaces import TaskDataSpec
#from examples.prompts.genrm import GENRM_PROMPT_TEMPLATE 
import json
import numpy as np
import torch
import random
import glob
import os 

random.seed(42)
   
   
GENRM_PROMPT_TEMPLATE = """You are a skilled little expert at scoring responses. You should evaluate given responses based on the given judging criteria.
Given the context of the conversation (the last turn is the User's query) and one or two responses from the Assistant, you need to refer to the [Helpfulness Scoring Guidelines] to score each individual response.
If there are two responses, you need to also give a ranking score based on the [Ranking Scoring Guidelines].
Before scoring, please analyze step by step. Your scoring needs to be as strict as possible.

[Helpfulness Scoring Guidelines]

When evaluating Helpfulness, consider the following factors:

- Correctness/Completeness: Is the response accurate and complete?
- Coherence/Clarity: Is the response clear, coherent, and easy to understand?
- Instruction following: Does the response follow the instructions and fulfill the user's request?
- Relevance: Is the response relevant to the user's query/input?
- Level of Detail and Creativity: Does the response provide enough detail without being too verbose? Does it show creativity but not hallucinations?

**Score 5: Extremely Helpful**

- The response is extremely helpful and completely aligned with the spirit of what the prompt was asking for.
- It accurately acts on the user's request, without unnecessary information.
- If a user request is not possible/in line with desired model behavior, a helpful response provides useful context and rationale.

**Score 4: Mostly Helpful**

- The response is mostly helpful and mainly aligned with what the user was looking for.
- There is still some room for improvement, but the response is generally useful.

**Score 3: Partially Helpful**

- The response is partially helpful but misses the overall goal of the user's query/input in some way.
- The response did not fully satisfy what the user was looking for.

**Score 2: Borderline Unhelpful**

- The response is borderline unhelpful and mostly does not capture what the user was looking for.
- However, it is still usable and helpful in a small way.

**Score 1: Not Helpful**

- The response is not useful or helpful at all.
- The response completely missed the essence of what the user wanted.

[Ranking Scoring Guidelines]

Ranking score is used to rank the two responses based on their helpfulness. Even if you give the same individual helpfulness score for both responses, you need to differentiate them strictly.
The ranking score is a number between 1 and 6, where:
1 = Response 1 is much better than Response 2
2 = Response 1 is better than Response 2
3 = Response 1 is slightly better than Response 2
4 = Response 2 is slightly better than Response 1
5 = Response 2 is better than Response 1
6 = Response 2 is much better than Response 1

#### Conversation Context ####
{context}

#### Responses to be Scored ####
{responses}

#### Output Format Requirements ####
First give your analysis on each responses in the format of:
[The Begin of Analysis on Response i]
Analysis on the i-th response
[The End of Analysis on Response i]

Then give the scores of each response in order, separate by comma in the boxed, adhering this format:
[The Begin of Individual Scores]
\\boxed{{x, y}} 
[The End of Individual Scores]

If there are two responses, give the relative ranking score in the format of:
[The Begin of Ranking Score]
\\boxed{{z}} if there exists 2 responses
[The End of Ranking Score]
You don't need to give a ranking score if only one response is provided."""



         

def format_genrm_prompt(context: str, response1: str, response2: Optional[str] = None) -> str:
    """Format the GenRM prompt with context and responses."""
    if response2 is None:
        responses = f"[The Begin of Response 1]\n{response1}\n[The End of Response 1]"
    else:
        responses = (
            f"[The Begin of Response 1]\n{response1}\n[The End of Response 1]\n"
            f"[The Begin of Response 2]\n{response2}\n[The End of Response 2]\n"
        )
    
    return GENRM_PROMPT_TEMPLATE.format(
        context=context,
        responses=responses
    )



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
        "prompt": format_genrm_prompt(context, response1, response2),
        "num_responses": 2,
        "preference": preference,
        "context": context,
        "response1": response1,
        "response2": response2,
        "domain": domain,
        "sample_id": pair_id
    }
    
    result2 = {
        "prompt": format_genrm_prompt(context, response2, response1),
        "num_responses": 2,
        "preference": 1-preference,
        "context": context,
        "response1": response2,
        "response2": response1,
        "domain": domain,
        "sample_id": pair_id
    }
    
    return [result1, result2]



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
                "prompt": format_genrm_prompt(context, response1, response2),
                "num_responses": 2,
                "preference": preference,
                "context": context,
                "response1": response1,
                "response2": response2,
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
            "prompt": format_genrm_prompt(context, response1, response2),
            "num_responses": 2,
            "preference": preference,
            "context": context,
            "response1": response1,
            "response2": response2,
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
        "prompt": format_genrm_prompt(context, response1, response2),
        "num_responses": 2,
        "preference": preference,
        "context": context,
        "response1": response1,
        "response2": response2,
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
                "prompt": format_genrm_prompt(context, response1, response2),
                "num_responses": 2,
                "preference": preference,
                "context": context,
                "response1": response1,
                "response2": response2,
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



######issue to be fixed: change the input file 
class HelpSteer3LocalDataset(torch.utils.data.Dataset):
    """Dataset for loading HelpSteer3 data from local JSONL files."""
    def __init__(self, data_path: str="/lustre/fs1/portfolios/llmservice/projects/llmservice_modelalignment_sft/users/yizhuj/datasets/hs3_genrm/val_data_base.jsonl", task_name: str="genrm", shuffle_seed: int = -1, split: str="validation", max_number: int=-1, double_swap=True):
        data = []
        with open(data_path, 'r') as f:
            for line in f:
                one = json.loads(line)
                if one["args"]["num_responses"] == 2:
                    examples = []
                    preference = 0 if one["args"]["preference_ranking"] <= 3 else 1

                    example = {
                        "prompt": format_genrm_prompt(one["args"]["context"], one["args"]["response1"], one["args"]["response2"]),
                        "num_responses": 2,
                        "helpfulness_1": one["args"]["helpfulness_1"],  
                        "helpfulness_2": one["args"]["helpfulness_2"],
                        "preference": preference, 
                        "preference_ranking": one["args"]["preference_ranking"],
                        "context": one["args"]["context"],
                        "response1": one["args"]["response1"],
                        "response2": one["args"]["response2"],
                    }
                    examples.append(example)

                    if double_swap:
                        example = {
                            "prompt": format_genrm_prompt(one["args"]["context"], one["args"]["response2"], one["args"]["response1"]),
                            "num_responses": 2,
                            "helpfulness_1": one["args"]["helpfulness_2"],  
                            "helpfulness_2": one["args"]["helpfulness_1"],
                            "preference": 1 - preference, 
                            "preference_ranking": 7 - one["args"]["preference_ranking"],
                            "context": one["args"]["context"],
                            "response1": one["args"]["response2"],
                            "response2": one["args"]["response1"],
                        }
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
            
        self.data = Dataset.from_list(data)
        self.formatted_ds = self.data
        self.task_name = task_name

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Add task_name for compatibility with AllTaskProcessedDataset
        item = self.data[idx].copy()
        item["task_name"] = self.task_name
        return item
