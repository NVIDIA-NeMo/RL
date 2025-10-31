import os
import json
import random
import re
import argparse
import numpy as np
from collections import defaultdict
from typing import Dict, List, Any, Tuple



def extract_preference(result: dict) -> Tuple[int, str]:
    """
    Extract preference from GenRM output using the EXACT same logic as original script.
    Returns (mapped_prediction, method_used) where mapped_prediction is 0 or 1
    """
    try:
        # Method 1: Use predicted_ranking (same as original script)
        linear_ranking = result["linear_ranking"]
        chosen_is_better = 1 if linear_ranking >= 0.5 else 0
        return chosen_is_better
        
    except Exception as e:
        print(f"Warning: Failed to extract preference ranking from result: {e}")
        # chosen_is_better = random.choice([0, 1])
        chosen_is_better = -1
        return chosen_is_better
    


def group_results_by_sample(results: List[dict]) -> Dict[str, List[dict]]:
    """Group evaluation results by original sample ID."""
    grouped = defaultdict(list)
    
    for result in results:
        metadata = result.get("metadata", {})
        sample_id = metadata.get("sample_id", f"unknown_{result.get('idx', 0)}")
        grouped[sample_id].append(result)
    
    return dict(grouped)


def compute_rmbench_accuracy_for_sample(sample_results: List[dict]) -> Dict[str, Any]:
    """Compute RM-Bench accuracy for a single sample (3x3 matrix)."""
    # Initialize 3x3 matrix for chosen vs rejected comparisons
    # Rows: chosen response styles (0=concise, 1=detailed_plain, 2=detailed_markdown)
    # Cols: rejected response styles (0=concise, 1=detailed_plain, 2=detailed_markdown)
    comparison_matrix = np.zeros((3, 3))
    comparison_counts = np.zeros((3, 3))
    

    
    for result in sample_results:
        metadata = result.get("metadata", {})
        
        # Extract metadata
        domain = metadata.get("domain")
        sample_id = metadata.get("sample_id")
        chosen_style_idx = metadata.get("chosen_style_idx")
        rejected_style_idx = metadata.get("rejected_style_idx")
        gt = metadata.get("preference_ranking")
        
        # Extract preference using the robust method (same as original script)
        chosen_is_better = extract_preference(result)
        is_chosen_first = (gt == 0)
        comparison_counts[chosen_style_idx, rejected_style_idx] += 1
        if chosen_is_better != -1:
            if not is_chosen_first:
                # If chosen was second in the prompt, flip the preference
                chosen_is_better = not chosen_is_better
            if chosen_is_better:
                comparison_matrix[chosen_style_idx, rejected_style_idx] += 1
    
    # Normalize by counts to get accuracy matrix
    acc_matrix = np.divide(comparison_matrix, comparison_counts, 
                          out=np.zeros_like(comparison_matrix), 
                          where=comparison_counts!=0)
    
    # Compute hard, normal, easy accuracy according to RM-Bench definition
    MATRIX_SIZE = 3
    
    # Hard accuracy: upper-right triangle (chosen less fancy vs rejected more fancy)
    upper_right_count = MATRIX_SIZE * (MATRIX_SIZE - 1) / 2
    hard_acc = np.sum(np.triu(acc_matrix, 1)) / upper_right_count if upper_right_count > 0 else 0.0
    
    # Normal accuracy: diagonal (same styles)
    normal_acc = np.mean(np.diag(acc_matrix))
    
    # Easy accuracy: lower-left triangle (chosen more fancy vs rejected less fancy)
    lower_left_count = MATRIX_SIZE * (MATRIX_SIZE - 1) / 2
    easy_acc = np.sum(np.tril(acc_matrix, -1)) / lower_left_count if lower_left_count > 0 else 0.0
    
    # Total average accuracy
    total_avg_acc = np.mean(acc_matrix)
    
    return {
        "sample_id": sample_id,
        "domain": domain,
        "hard_acc": hard_acc,
        "normal_acc": normal_acc,
        "easy_acc": easy_acc,
        "total_avg_acc": total_avg_acc,
        "acc_matrix": acc_matrix.tolist(),
        "comparison_counts": comparison_counts.tolist(),
    }


def print_rmbench_results(metrics: Dict[str, Any]):
    """Print RM-Bench results in a formatted way."""
    if not metrics:
        print("No metrics to display")
        return
    
    print("\n" + "="*80)
    print("RM-BENCH EVALUATION RESULTS")
    print("="*80)
    
    # Sort by step number
    sorted_steps = sorted(metrics.keys())
    
    for step in sorted_steps:
        step_data = metrics[step]
        print(f"\nStep {step}:")
        #print("-" * 40)
        
        # Print overall metrics
        
        overall = step_data.get("overall", {})
        if overall:
            print(f"Overall Metrics (samples: {overall.get('sample_count', 0)}):")
            print(f"  Hard Accuracy:      {overall.get('hard_acc', 0):.3f}")
            print(f"  Normal Accuracy:    {overall.get('normal_acc', 0):.3f}")
            print(f"  Easy Accuracy:      {overall.get('easy_acc', 0):.3f}")
            print(f"  Total Avg Accuracy: {overall.get('total_avg_acc', 0):.3f}")
        

        # Print domain-specific metrics
        
        domains = step_data.get("domains", {})
        if domains:
            print(f"\nDomain-specific Metrics:")
            for domain, domain_data in sorted(domains.items()):
                #if domain != 'chat':
                #    continue
                print(f"  {domain.upper()} (samples: {domain_data.get('sample_count', 0)}):")
                print(f"    Hard Acc:   {domain_data.get('hard_acc', 0):.3f}")
                print(f"    Normal Acc: {domain_data.get('normal_acc', 0):.3f}")
                print(f"    Easy Acc:   {domain_data.get('easy_acc', 0):.3f}")
                print(f"    Total Avg:  {domain_data.get('total_avg_acc', 0):.3f}")
        
        


def compute_rmbench_metrics(directory_path: str, dataset: str = "rmbench") -> Dict[str, Any]:
    file_pattern = re.compile(rf'step_(\d+)_{dataset}_results\.json')
    
    all_metrics = {}
    
    try:
        for filename in os.listdir(directory_path):
            match = file_pattern.match(filename)
            if match:
                step_number = int(match.group(1))
                file_path = os.path.join(directory_path, filename)
                
                print(f"Processing step {step_number}: {filename}")
                
                try:
                    with open(file_path, 'r', encoding='utf-8') as f:
                        results = json.load(f)
                    
                    if not isinstance(results, list):
                        print(f"Warning: {filename} does not contain a list of results")
                        continue
                    
                    # Group results by sample ID
                    grouped_results = group_results_by_sample(results)
                    
                    # Compute metrics for each sample
                    sample_metrics = []
                    all_extraction_methods = defaultdict(int)
                    
                    for sample_id, sample_results in grouped_results.items():
                        if len(sample_results) > 0:  # Should have 9 results per sample (3x3 matrix)
                            sample_metric = compute_rmbench_accuracy_for_sample(sample_results)
                            sample_metrics.append(sample_metric)
                            
                            # Aggregate extraction method statistics
                            for method, count in sample_metric.get("extraction_methods", {}).items():
                                all_extraction_methods[method] += count
                    
                    if not sample_metrics:
                        print(f"Warning: No valid samples found in {filename}")
                        continue
                    
                    # Aggregate metrics by domain
                    domain_metrics = defaultdict(list)
                    for metric in sample_metrics:
                        domain = metric["domain"]
                        domain_metrics[domain].append(metric)
                    
                    # Calculate averages for each domain
                    step_metrics = {"step": step_number, "domains": {}, "overall": {}}
                    
                    all_samples = []
                    for domain, domain_samples in domain_metrics.items():
                        if domain_samples:
                            domain_hard_acc = np.mean([s["hard_acc"] for s in domain_samples])
                            domain_normal_acc = np.mean([s["normal_acc"] for s in domain_samples])
                            domain_easy_acc = np.mean([s["easy_acc"] for s in domain_samples])
                            domain_total_avg_acc = np.mean([s["total_avg_acc"] for s in domain_samples])
                            
                            step_metrics["domains"][domain] = {
                                "hard_acc": domain_hard_acc,
                                "normal_acc": domain_normal_acc,
                                "easy_acc": domain_easy_acc,
                                "total_avg_acc": domain_total_avg_acc,
                                "sample_count": len(domain_samples)
                            }
                            
                            all_samples.extend(domain_samples)
                    
                    # Calculate overall metrics
                    if all_samples:
                        overall_hard_acc = np.mean([s["hard_acc"] for s in all_samples])
                        overall_normal_acc = np.mean([s["normal_acc"] for s in all_samples])
                        overall_easy_acc = np.mean([s["easy_acc"] for s in all_samples])
                        overall_total_avg_acc = np.mean([s["total_avg_acc"] for s in all_samples])
                        
                        step_metrics["overall"] = {
                            "hard_acc": overall_hard_acc,
                            "normal_acc": overall_normal_acc,
                            "easy_acc": overall_easy_acc,
                            "total_avg_acc": overall_total_avg_acc,
                            "sample_count": len(all_samples)
                        }
                    
                    # Add debug information
                    step_metrics["debug_info"] = {
                        "extraction_methods": dict(all_extraction_methods)
                    }
                    
                    all_metrics[step_number] = step_metrics
                    
                except json.JSONDecodeError:
                    print(f"Error: {filename} is not valid JSON")
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
    
    except FileNotFoundError:
        print(f"Error: Directory not found: '{directory_path}'")
        return {}
    
    return all_metrics


def main():
    parser = argparse.ArgumentParser(description="Compute RM-Bench specific metrics from evaluation results.")
    parser.add_argument(
        "path",
        help="Path to the directory containing evaluation output files."
    )
    parser.add_argument(
        "--dataset", 
        default="rmbench", 
        help="Dataset name (default: %(default)s)."
    )
    parser.add_argument(
        "--output",
        help="Optional output JSON file to save detailed results."
    )
    
    args = parser.parse_args()
    
    # Compute RM-Bench metrics
    metrics = compute_rmbench_metrics(args.path, args.dataset)
    
    # Print results
    print_rmbench_results(metrics)
    
    # Save detailed results if requested
    if args.output:
        try:
            with open(args.output, 'w') as f:
                json.dump(metrics, f, indent=2)
            print(f"\nDetailed results saved to: {args.output}")
        except Exception as e:
            print(f"Error saving results: {e}")


if __name__ == "__main__":
    main()