import argparse
import os
from typing import Optional, List

import jsonlines

def parse_solution_string(solution_str: str):
    """format: 'fxn_name!:!answer_info!:!problem_type'"""
    try:
        parts = solution_str.split("!:!")
        if len(parts) >= 3:
            return {
                "fxn_name": parts[0],
                "answer_info": parts[1], 
                "problem_type": parts[2]
            }
    except Exception as e:
        print(f"Failed to parse solution: {solution_str}, error: {e}")
    return None

def format_ether0(data, task_filter: Optional[List[str]] = None):
    problem = data["problem"]
    solution_str = data["solution"]
    problem_type = data["problem_type"]
    ideal_answer = data.get("ideal", None)
    
    if task_filter:
        main_task = problem_type.split('/')[0]
        if main_task not in task_filter:
            return None
    
    reward_function_info = parse_solution_string(solution_str)
    if reward_function_info is None:
        print(f"Warning: Failed to parse solution for {data['id']}")
        return None
    
    problem_with_instruction = problem + "\n\nYOU MUST PROVE YOUR FINAL ANSWER INSIDE \\boxed{}."
    
    return {
        "messages": [
            [
                {
                    "role": "user",
                    "content": problem_with_instruction,
                    "metadata": {
                        "reward_function_info": reward_function_info,
                        "problem_type": problem_type,
                        "ideal_answer": ideal_answer,
                        "original_id": data["id"],
                    },
                },
            ]
        ],
        "task_name": "ether0",
        "dataset": "future_house",
        "problem_type": problem_type, 
    }


def main():
    parser = argparse.ArgumentParser(description="Convert Ether0 dataset to Nemo RL format")
    parser.add_argument(
        "--input_file",
        type=str,
        required=True,
        help="Input jsonlines file (e.g., future_house_train.jsonl)",
    )
    parser.add_argument(
        "--output_file", 
        type=str,
        required=True,
        help="Output file name for the converted jsonlines file",
    )
    parser.add_argument(
        "--task_filter",
        type=str,
        nargs="*",
        default=None,
        help="Filter to specific problem types (e.g., functional-group retro-synthesis)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=None,
        help="Maximum number of samples to convert (for testing)",
    )

    args = parser.parse_args()
    
    original_data = []
    with jsonlines.open(args.input_file, "r") as reader:
        for item in reader:
            original_data.append(item)
            if args.max_samples and len(original_data) >= args.max_samples:
                break
    
    print(f"Loaded {len(original_data)} samples from {args.input_file}")
    
    converted_data = []
    failed_count = 0
    for item in original_data:
        converted_item = format_ether0(item, task_filter=args.task_filter)
        if converted_item is not None:
            converted_data.append(converted_item)
        else:
            failed_count += 1
    
    print(f"Converted {len(converted_data)} samples")
    if failed_count > 0:
        print(f"Failed to convert {failed_count} samples")
    
    if args.task_filter:
        print(f"Applied task filter: {args.task_filter}")
        
        from collections import Counter
        task_counts = Counter()
        for item in converted_data:
            main_task = item["problem_type"].split('/')[0]
            task_counts[main_task] += 1
        
        print("Task distribution after filtering:")
        for task, count in sorted(task_counts.items()):
            print(f"  {task}: {count}")
    
    with jsonlines.open(args.output_file, mode="w") as writer:
        writer.write_all(converted_data)
    
    print(f"Converted data saved to {os.path.abspath(args.output_file)}")


if __name__ == "__main__":
    main()