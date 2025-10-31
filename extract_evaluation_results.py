import os
import json
import re
from collections import defaultdict
import argparse
import random

def calculate_accuracy(data: list[dict]):
    correct_predictions = 0
    total_predictions = 0

    # Iterate over each record in the file
    for record in data:
        total_predictions += 1
        # Safely get values from nested dictionaries
        predicted_ranking = record.get('predicted_ranking')
        metadata = record.get('metadata', {})
        preference_ranking = metadata.get('preference_ranking')

        # Ensure the required keys exist

        if preference_ranking is not None:
            # Apply the mapping rule
            # 0 if predicted_ranking <= 3 else 1
            if predicted_ranking is None:
                #mapped_prediction = random.choice([0, 1])
                mapped_prediction = -1
            else:
                mapped_prediction = 0 if predicted_ranking <= 3 else 1
                            
            # Check if the prediction is correct
            if mapped_prediction == preference_ranking:
                correct_predictions += 1           
        else:
            print(f"    [WARNING] A record is missing 'predicted_ranking' or 'preference_ranking', skipping.")

    # Calculate accuracy, avoiding division by zero
    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions
        return accuracy
    else:
        print(f"    [WARNING] No valid data found in the file.")
        return 0.0

def calculate_accuracy_linear(data: list[dict]):
    correct_predictions = 0
    total_predictions = 0

    # Iterate over each record in the file
    for record in data:
        total_predictions += 1
        # Safely get values from nested dictionaries
        linear_ranking = record.get('linear_ranking')
        metadata = record.get('metadata', {})
        preference_ranking = metadata.get('preference_ranking')

        # Ensure the required keys exist

        if preference_ranking is not None:
            # Apply the mapping rule
            # 0 if linear_ranking <= 3 else 1
            if linear_ranking is None:
                mapped_prediction = -1
            else:
                mapped_prediction = 0 if linear_ranking >= 0.5 else 1
                            
            # Check if the prediction is correct
            if mapped_prediction == preference_ranking:
                correct_predictions += 1           
        else:
            print(f"    [WARNING] A record is missing 'linear_ranking' or 'preference_ranking', skipping.")

    # Calculate accuracy, avoiding division by zero
    if total_predictions > 0:
        accuracy = correct_predictions / total_predictions
        return accuracy
    else:
        print(f"    [WARNING] No valid data found in the file.")
        return 0.0

def calculate_step_accuracies(directory_path: str, dataset: str) -> dict:
    """
    Calculates the accuracy for all step_*_judgebench_results.json files in a directory.

    The accuracy is calculated as follows:
    - Prediction: 0 if 'predicted_ranking' <= 3, else 1.
    - Ground Truth: 'metadata' -> 'preference_ranking'.
    - Accuracy = (Number of correct predictions) / (Total samples).

    Args:
        directory_path: The path to the directory containing the JSON files.

    Returns:
        A dictionary where keys are the step numbers (int) and values are the
        corresponding accuracies (float).
    """
    step_accuracies = {}
    step_accuracies_linear = {}

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
                        data = json.load(f)

                    if not isinstance(data, list):
                        print(f"    [WARNING] Content of {filename} is not a list, skipping.")
                        continue
                    
                    accuracy = 0.0
                    if dataset == "rewardbench2":
                        chunk_size = 3
                    elif dataset == "judgebench":
                        chunk_size = 2
                    else:
                        chunk_size = 1

                    chunk_outcomes = []
                    chunk_outcomes_linear = []
                    if data:
                        length = len(data)
                        assert length % chunk_size == 0
                        for i in range(0, length, chunk_size):
                            chunk = data[i: i+chunk_size]
                            if chunk:
                                chunk_accuracy = calculate_accuracy(chunk)
                                chunk_accuracy_linear = calculate_accuracy_linear(chunk)
                                chunk_outcomes.append(1 if chunk_accuracy == 1.0 else 0)
                                chunk_outcomes_linear.append(1 if chunk_accuracy_linear == 1.0 else 0)
                        
                    # The final accuracy is the average of the outcomes of all chunks
                    if chunk_outcomes:
                        accuracy = sum(chunk_outcomes) / len(chunk_outcomes)
                    else:
                        accuracy = 0.0 # No chunks to process
                    if chunk_outcomes_linear:
                        accuracy_linear = sum(chunk_outcomes_linear) / len(chunk_outcomes_linear)
                    else:
                        accuracy_linear = 0.0 # No chunks to process

                    step_accuracies[step_number] = accuracy
                    step_accuracies_linear[step_number] = accuracy_linear
                    
                except json.JSONDecodeError:
                    print(f"    [ERROR] File {filename} is not a valid JSON, skipping.")
                except Exception as e:
                    print(f"    [ERROR] An unknown error occurred while processing {filename}: {e}")

    except FileNotFoundError:
        print(f"[ERROR] Directory not found: '{directory_path}'")
        return {}

    # Sort the results by step number for clear presentation
    sorted_accuracies = dict(sorted(step_accuracies.items()))
    sorted_accuracies_linear = dict(sorted(step_accuracies_linear.items()))
    
    return sorted_accuracies, sorted_accuracies_linear


parser = argparse.ArgumentParser(description="Compute per-step accuracies from evaluation JSON files.")
parser.add_argument(
    "path",
    help="Path to the directory that contains the evaluation output files."
)
parser.add_argument("--dataset", default="rmbench", help="Dataset name (default: %(default)s).")

args = parser.parse_args()

final_accuracies, final_accuracies_linear = calculate_step_accuracies(args.path, args.dataset)
'''
if final_accuracies:
    print("Accuracy for each step:")
    for step, acc in final_accuracies.items():
        print(f"  Step {step}: {acc:.2%}")
else:
    print("No matching files or data were found in the specified directory.")
if final_accuracies_linear:
    print("Linear accuracy for each step:")
    for step, acc in final_accuracies_linear.items():
        print(f"  Step {step}: {acc:.2%}")
else:
    print("No matching files or data were found in the specified directory.")
'''
if final_accuracies and final_accuracies_linear:
    assert len(final_accuracies) == len(final_accuracies_linear)
    print("\r\n  GenRM\t\t\tLinear")
    for (step1,acc1),(step2,acc2) in zip(final_accuracies.items(), final_accuracies_linear.items()):
        print(f"  Step {step1}: {acc1:.2%}\t{acc2:.2%}  {'*' if acc2 > acc1 else ''}")
