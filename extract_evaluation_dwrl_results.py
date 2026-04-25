import os
import itertools
import json
import re
from collections import defaultdict
import argparse
import random

def calculate_accuracy(data: list[dict]):
    correct_predictions = 0
    total_predictions = len(data)

    # Iterate over each record in the file
    for record in data:
        # Safely get values from nested dictionaries
        score_1 = record['score_1']
        score_2 = record['score_2']
        metadata = record.get('metadata')
        preference = metadata.get('preference')
                            
        # Ensure the required keys exist
        correct_predictions += preference == int(score_2 > score_1)

    # Calculate accuracy, avoiding division by zero
    if correct_predictions == total_predictions:
        return 1.0
    else:
        return 0.0


def calculate_step_accuracies(directory_path: str, dataset: str) -> dict:
    """
    Calculates the accuracy for all step_*_judgebench_results.json files in a directory.

    The accuracy is calculated as follows:
    - Prediction: 0 if 'predicted_ranking' <= 3, else 1.
    - Ground Truth: 'metadata' -> 'preference'.
    - Accuracy = (Number of correct predictions) / (Total samples).

    Args:
        directory_path: The path to the directory containing the JSON files.

    Returns:
        A dictionary where keys are the step numbers (int) and values are the
        corresponding accuracies (float).
    """
    step_accuracies = {}

    # Regex to match filenames and extract the step number
    # e.g., "step_5_judgebench_results.json" -> "5"
    file_pattern = re.compile(rf'step_(\d+)_{dataset}_results\.json')
    print(f"Scanning directory: '{directory_path}'...")

    try:
        # Iterate over all files in the directory
        for filename in os.listdir(directory_path):
            match = file_pattern.match(filename)
            # If the filename matches our pattern
            if match:
                step_number = int(match.group(1))
                file_path = os.path.join(directory_path, filename)
                
                print(f"  Processing file: {filename} (Step: {step_number})")

                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        original_data = json.load(f)

                    if not isinstance(original_data, list):
                        print(f"    [WARNING] Content of {filename} is not a list, skipping.")
                        continue

                    data = []
                    data_ties = []
                    for one in original_data:
                        if dataset == "rmb":
                            if "metadata" in one and "category_path" in one["metadata"] and "Harmlessness" not in one["metadata"]["category_path"]:
                                data.append(one)
                        elif dataset == "rewardbench2":
                            if one['metadata']['domain'] == "Ties":
                                data_ties.append(one)
                            else:
                                data.append(one)
                        else:
                            data.append(one)

                    accuracy = 0.0
                    if dataset == "rewardbench2":
                        chunk_size = 3
                    elif dataset == "judgebench":
                        chunk_size = 2
                    else:
                        chunk_size = 1

                    chunk_outcomes = []
                    if data:
                        length = len(data)
                        assert length % chunk_size == 0, f"length is [{length}]  chunk_size is [{chunk_size}]"
                        for i in range(0, length, chunk_size):
                            chunk = data[i: i+chunk_size]
                            if chunk:
                                chunk_accuracy = calculate_accuracy(chunk)
                                chunk_outcomes.append(chunk_accuracy)
                    if data_ties:
                        ties_grp = [(g[0], [y for y in g[-1]]) for g in itertools.groupby(sorted(data_ties, key=lambda kk: kk['metadata']['sample_id']), key=lambda kk: kk['metadata']['sample_id'])]
                        for samp_id, chunk in ties_grp:
                            chunk_accuracy = calculate_accuracy(chunk)
                            chunk_outcomes.append(chunk_accuracy)
                        
                    # The final accuracy is the average of the outcomes of all chunks
                    if chunk_outcomes:
                        accuracy = sum(chunk_outcomes) / len(chunk_outcomes)
                    else:
                        accuracy = 0.0 # No chunks to process

                    step_accuracies[step_number] = accuracy
                    
                except json.JSONDecodeError:
                    print(f"    [ERROR] File {filename} is not a valid JSON, skipping.")
                except Exception as e:
                    raise e
                    print(f"    [ERROR] An unknown error occurred while processing {filename}: {e}")

    except FileNotFoundError:
        print(f"[ERROR] Directory not found: '{directory_path}'")
        return {}

    # Sort the results by step number for clear presentation
    sorted_accuracies = dict(sorted(step_accuracies.items()))
    
    return sorted_accuracies


parser = argparse.ArgumentParser(description="Compute per-step accuracies from evaluation JSON files.")
parser.add_argument(
    "path",
    help="Path to the directory that contains the evaluation output files."
)
parser.add_argument("--dataset", default="rmbench", help="Dataset name (default: %(default)s).")

args = parser.parse_args()

final_accuracies = calculate_step_accuracies(args.path, args.dataset)

if final_accuracies:
    print("Accuracy for each step:")
    for step, acc in final_accuracies.items():
        print(f"  Step {step}: {acc:.2%}")
else:
    print("No matching files or data were found in the specified directory.")
