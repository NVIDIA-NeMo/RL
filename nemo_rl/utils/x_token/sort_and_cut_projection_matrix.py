# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import torch
import os
import argparse
import tqdm
from transformers import AutoTokenizer, AutoConfig
import pdb;

def sinkhorn_one_dim(A, n_iters=1):
    """Apply Sinkhorn normalization to make each row sum to 1."""
    for _ in range(n_iters):
        # A = A / (A.sum(dim=1, keepdim=True) + 1e-6)
        row_sums = A.sum(dim=1, keepdim=True)
        safe_row_sums = torch.where(row_sums == 0, torch.ones_like(row_sums), row_sums)
        A = A / safe_row_sums
    return A

def clean_model_name_for_filename(name: str) -> str:
    """Removes parameter counts and common suffixes from model names for cleaner filenames."""
    import re
    # Removes patterns like -8B, -1.5B, -4b, -125m etc.
    cleaned_name = re.sub(r'-?[0-9\.]+[bBmB]', '', name, flags=re.IGNORECASE)
    # Remove common suffixes
    cleaned_name = cleaned_name.replace('-Base', '').replace('-it', '').replace('-Instruct', '')
    # Clean up any leading/trailing hyphens that might result
    cleaned_name = cleaned_name.strip('-_')
    return cleaned_name

def sort_and_cut_projection_matrix(input_path, output_path, new_top_k, preserve_last=False, verbose=True):
    """
    Load a projection matrix, sort each row by weight values, and save with new top_k cutoff.
    
    Args:
        input_path: Path to input projection matrix file
        output_path: Path to save the new projection matrix
        new_top_k: New top_k value for cutoff
        preserve_last: If True, always preserve the last column as the final element
        verbose: Whether to print progress information
    """
    if verbose:
        print(f"Loading projection matrix from: {input_path}")
    
    # Load the projection matrix
    projection_data = torch.load(input_path, map_location='cpu', weights_only=False)
    
    if not isinstance(projection_data, dict) or 'indices' not in projection_data or 'likelihoods' not in projection_data:
        raise ValueError("Input file must contain a dictionary with 'indices' and 'likelihoods' keys")
    
    original_indices = projection_data['indices']  # Shape: [vocab_size, original_top_k]
    original_likelihoods = projection_data['likelihoods']  # Shape: [vocab_size, original_top_k]
    
    vocab_size, original_top_k = original_indices.shape
    
    if verbose:
        print(f"Original matrix shape: {original_indices.shape}")
        print(f"Original top_k: {original_top_k}")
        print(f"New top_k: {new_top_k}")
        print(f"Preserve last column: {preserve_last}")
        # pdb.set_trace()
    
    if new_top_k > original_top_k:
        print(f"Warning: New top_k ({new_top_k}) is larger than original top_k ({original_top_k})")
        print(f"Will pad with -1 indices and 0.0 likelihoods")
        effective_top_k = original_top_k
    else:
        effective_top_k = new_top_k
    
    # Initialize new tensors
    new_indices = torch.full((vocab_size, new_top_k), -1, dtype=original_indices.dtype)
    new_likelihoods = torch.zeros((vocab_size, new_top_k), dtype=original_likelihoods.dtype)
    
    # Statistics tracking
    rows_with_order_change = 0
    significant_components_count = [0] * min(new_top_k, 10)  # Track up to 10 components
    threshold_for_significance = 0.2  # Threshold for considering a component "significant"
    # Track position of maximum element in original ordering
    max_element_positions = {}  # position -> count
    # Track preserve_last statistics
    rows_with_preserved_last = 0
    # Track specifically when max element is in the last column
    rows_with_max_in_last_column = 0
    # Track position of maximum element in final sorted and trimmed matrix
    final_max_element_positions = {}  # position -> count
    
    # threshold_for_significance = 0.05  # Threshold for considering a component "significant"
    # threshold_for_significance = 0.05  # Threshold for considering a component "significant"
    
    if verbose:
        print("Sorting and cutting each row...")
    
    # Process each row (each source token)
    last_element_trick_count = 0
    for row_idx in tqdm.tqdm(range(vocab_size), desc="Processing rows", disable=not verbose):
        row_indices = original_indices[row_idx]  # [original_top_k]
        row_likelihoods = original_likelihoods[row_idx]  # [original_top_k]
        
        # Filter out invalid indices (-1) and zero likelihoods
        valid_mask = (row_indices != -1) & (row_likelihoods > 0)
        
        if valid_mask.any():
            valid_indices = row_indices[valid_mask]
            valid_likelihoods = row_likelihoods[valid_mask]
            
            # Track position of maximum element in original ordering
            max_pos = torch.argmax(valid_likelihoods).item()
            if max_pos not in max_element_positions:
                max_element_positions[max_pos] = 0
            max_element_positions[max_pos] += 1
            
            # Check if max element is specifically in the last column
            # Find the actual maximum value in the original row (including invalid entries)
            original_max_pos = torch.argmax(row_likelihoods).item()
            if original_max_pos == original_top_k - 1:
                # Only count if the last position actually has valid data
                last_index = row_indices[original_top_k - 1]
                last_likelihood = row_likelihoods[original_top_k - 1]
                if last_index != -1 and last_likelihood > 0:
                    rows_with_max_in_last_column += 1
            
            if preserve_last and new_top_k >= 1:
                # Handle preserve_last case
                last_index = original_indices[row_idx, original_top_k - 1]
                last_likelihood = original_likelihoods[row_idx, original_top_k - 1]
                
                if new_top_k == 1:
                    # Special case: only keep the last element
                    if last_index != -1 and last_likelihood > 0:
                        new_indices[row_idx, 0] = last_index
                        new_likelihoods[row_idx, 0] = last_likelihood
                        rows_with_preserved_last += 1
                        
                        # Count significant components
                        if last_likelihood >= threshold_for_significance:
                            significant_components_count[0] += 1
                else:
                    # General case: sort first (original_top_k-1) elements, then add last element
                    elements_to_sort = min(len(valid_likelihoods), original_top_k - 1)
                    if elements_to_sort > 0:
                        # Get elements excluding the last position in original matrix
                        sort_mask = torch.arange(len(valid_likelihoods)) < elements_to_sort
                        if sort_mask.any():
                            sortable_indices = valid_indices[sort_mask]
                            sortable_likelihoods = valid_likelihoods[sort_mask]
                            
                            # Sort the non-last elements
                            sorted_likelihoods, sort_order = torch.sort(sortable_likelihoods, descending=True)
                            sorted_indices = sortable_indices[sort_order]
                            
                            # Check if order changed in the sortable portion
                            original_order = torch.arange(len(sortable_likelihoods))
                            if not torch.equal(sort_order, original_order):
                                rows_with_order_change += 1
                            
                            # Take top (new_top_k - 1) elements from sorted portion
                            num_from_sorted = min(len(sorted_indices), new_top_k - 1)
                            
                            new_indices[row_idx, :num_from_sorted] = sorted_indices[:num_from_sorted]
                            new_likelihoods[row_idx, :num_from_sorted] = sorted_likelihoods[:num_from_sorted]
                            
                            # Count significant components from sorted portion
                            for comp_idx in range(min(num_from_sorted, len(significant_components_count) - 1)):
                                if sorted_likelihoods[comp_idx] >= threshold_for_significance:
                                    significant_components_count[comp_idx] += 1
                    
                    # Always put the last element at the end (if valid)
                    
                    if last_index != -1 and last_likelihood > 0:
                        last_element_trick_count += 1
                        new_indices[row_idx, new_top_k - 1] = last_index
                        new_likelihoods[row_idx, new_top_k - 1] = last_likelihood
                        rows_with_preserved_last += 1
                        
                        # Count significant component for the preserved last element
                        if new_top_k - 1 < len(significant_components_count):
                            if last_likelihood >= threshold_for_significance:
                                significant_components_count[new_top_k - 1] += 1
    
            else:
                # Original logic: sort all elements normally
                # Check if order changed by comparing original vs sorted order
                original_order = torch.arange(len(valid_likelihoods))
                sorted_likelihoods, sort_order = torch.sort(valid_likelihoods, descending=True)
                
                # Check if the order changed (not just sorted, but actually different)
                if not torch.equal(sort_order, original_order):
                    rows_with_order_change += 1
                
                sorted_indices = valid_indices[sort_order]
                
                # Take top effective_top_k elements
                num_to_take = min(len(sorted_indices), effective_top_k)
                
                new_indices[row_idx, :num_to_take] = sorted_indices[:num_to_take]
                new_likelihoods[row_idx, :num_to_take] = sorted_likelihoods[:num_to_take]
                # pdb.set_trace()
                # Count significant components (components above threshold)
                for comp_idx in range(min(num_to_take, len(significant_components_count))):
                    if sorted_likelihoods[comp_idx] >= threshold_for_significance:
                        significant_components_count[comp_idx] += 1
                # if significant_components_count[1] > 0.0:
                #     pdb.set_trace()
    
    # If new_top_k > original_top_k, the tensors are already padded with -1 and 0.0
    
    # Apply Sinkhorn normalization to the final matrix
    print(f"last element trick count: {last_element_trick_count}")
    if verbose:
        print("Applying Sinkhorn normalization...")
    
    # Apply normalization only to non-zero values to preserve sparsity structure
    normalized_likelihoods = sinkhorn_one_dim(new_likelihoods.clone(), n_iters=1)
    
    # Calculate final maximum element position statistics after sorting and normalization
    if verbose:
        print("Calculating final maximum element position statistics...")
    
    for row_idx in range(vocab_size):
        row_likelihoods = normalized_likelihoods[row_idx]
        # Filter out zero likelihoods
        valid_mask = row_likelihoods > 0
        if valid_mask.any():
            valid_likelihoods = row_likelihoods[valid_mask]
            # Find position of maximum element in the final matrix
            max_pos_in_valid = torch.argmax(valid_likelihoods).item()
            # Convert back to original position in the row
            valid_positions = torch.nonzero(valid_mask).squeeze(-1)
            actual_max_pos = valid_positions[max_pos_in_valid].item()
            
            if actual_max_pos not in final_max_element_positions:
                final_max_element_positions[actual_max_pos] = 0
            final_max_element_positions[actual_max_pos] += 1
    
    # Create output dictionary with same format as input
    output_data = {
        'indices': new_indices,
        'likelihoods': normalized_likelihoods,
    }
    
    # Copy over any additional metadata
    for key in projection_data:
        if key not in ['indices', 'likelihoods']:
            output_data[key] = projection_data[key]
    
    # Save the new projection matrix
    torch.save(output_data, output_path)
    
    if verbose:
        print(f"Saved sorted and cut projection matrix to: {output_path}")
        print(f"New matrix shape: {new_indices.shape}")
        
        # Show basic statistics
        non_zero_counts = (new_likelihoods > 0).sum(dim=1)
        avg_non_zero = non_zero_counts.float().mean().item()
        print(f"Average non-zero entries per row: {avg_non_zero:.2f}")
        print(f"Rows with max entries ({new_top_k}): {(non_zero_counts == new_top_k).sum().item()}")
        
        # Show ordering statistics
        print(f"\n=== Ordering Statistics ===")
        print(f"Rows with changed order after sorting: {rows_with_order_change:,} / {vocab_size:,} ({100*rows_with_order_change/vocab_size:.1f}%)")
        if preserve_last:
            print(f"Rows with preserved last element: {rows_with_preserved_last:,} / {vocab_size:,} ({100*rows_with_preserved_last/vocab_size:.1f}%)")
        
        # Show last column maximum element statistics
        print(f"\n=== Last Column Maximum Element Statistics ===")
        total_rows_with_data = sum(max_element_positions.values())
        if total_rows_with_data > 0:
            percentage_last_max = 100 * rows_with_max_in_last_column / total_rows_with_data
            print(f"Rows with maximum element in LAST column: {rows_with_max_in_last_column:,} / {total_rows_with_data:,} ({percentage_last_max:.1f}%)")
            print(f"Rows with maximum element in NON-LAST columns: {total_rows_with_data - rows_with_max_in_last_column:,} / {total_rows_with_data:,} ({100 - percentage_last_max:.1f}%)")
        else:
            print(f"No valid data found to analyze last column statistics")
        
        # Show maximum element position distribution
        print(f"\n=== Maximum Element Position Distribution (Original Ordering) ===")
        total_rows_with_data = sum(max_element_positions.values())
        print(f"Total rows with valid data: {total_rows_with_data:,}")
        
        # Sort positions for ordered display
        sorted_positions = sorted(max_element_positions.keys())
        for pos in sorted_positions[:20]:  # Show up to first 20 positions
            count = max_element_positions[pos]
            percentage = 100 * count / total_rows_with_data if total_rows_with_data > 0 else 0
            ordinal = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th"][pos] if pos < 10 else f"{pos+1}th"
            print(f"Rows with max element in {ordinal} position: {count:,} / {total_rows_with_data:,} ({percentage:.1f}%)")
        
        if len(sorted_positions) > 20:
            remaining_count = sum(max_element_positions[pos] for pos in sorted_positions[20:])
            remaining_percentage = 100 * remaining_count / total_rows_with_data if total_rows_with_data > 0 else 0
            print(f"Rows with max element in positions 21+: {remaining_count:,} / {total_rows_with_data:,} ({remaining_percentage:.1f}%)")
        
        # Show final maximum element position distribution (after sorting and normalization)
        print(f"\n=== Maximum Element Position Distribution (Final Sorted & Normalized Matrix) ===")
        total_final_rows_with_data = sum(final_max_element_positions.values())
        print(f"Total rows with valid data: {total_final_rows_with_data:,}")
        
        if total_final_rows_with_data > 0:
            # Sort positions for ordered display
            sorted_final_positions = sorted(final_max_element_positions.keys())
            for pos in sorted_final_positions[:min(new_top_k, 20)]:  # Show up to new_top_k or 20 positions
                count = final_max_element_positions[pos]
                percentage = 100 * count / total_final_rows_with_data
                ordinal = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th"][pos] if pos < 10 else f"{pos+1}th"
                print(f"Rows with max element in {ordinal} position: {count:,} / {total_final_rows_with_data:,} ({percentage:.1f}%)")
            
            if len(sorted_final_positions) > min(new_top_k, 20):
                remaining_count = sum(final_max_element_positions[pos] for pos in sorted_final_positions[min(new_top_k, 20):])
                remaining_percentage = 100 * remaining_count / total_final_rows_with_data
                print(f"Rows with max element in positions {min(new_top_k, 20)+1}+: {remaining_count:,} / {total_final_rows_with_data:,} ({remaining_percentage:.1f}%)")
        
        # Show significant components statistics
        print(f"\n=== Significant Components Statistics (threshold >= {threshold_for_significance}) ===")
        component_names = ["1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th"]
        for i, count in enumerate(significant_components_count):
            percentage = 100 * count / vocab_size if vocab_size > 0 else 0
            print(f"Rows with significant {component_names[i]} component: {count:,} / {vocab_size:,} ({percentage:.1f}%)")
        
        # Additional analysis: distribution of likelihood values (after normalization)
        all_likelihoods = normalized_likelihoods[normalized_likelihoods > 0]
        if len(all_likelihoods) > 0:
            print(f"\n=== Likelihood Distribution ===")
            print(f"Total non-zero likelihoods: {len(all_likelihoods):,}")
            print(f"Mean likelihood: {all_likelihoods.mean().item():.4f}")
            print(f"Median likelihood: {all_likelihoods.median().item():.4f}")
            print(f"Min likelihood: {all_likelihoods.min().item():.4f}")
            print(f"Max likelihood: {all_likelihoods.max().item():.4f}")
            
            # Show percentiles - convert to float for quantile calculation
            percentiles = [90, 95, 99]
            all_likelihoods_float = all_likelihoods.float()
            for p in percentiles:
                val = torch.quantile(all_likelihoods_float, p/100.0).item()
                print(f"{p}th percentile: {val:.4f}")
        
        # Show how many rows have multiple significant components
        print(f"\n=== Multi-Component Analysis ===")
        rows_with_multiple_significant = 0
        for row_idx in range(vocab_size):
            significant_in_row = (normalized_likelihoods[row_idx] >= threshold_for_significance).sum().item()
            if significant_in_row >= 2:
                rows_with_multiple_significant += 1
        
        percentage_multi = 100 * rows_with_multiple_significant / vocab_size if vocab_size > 0 else 0
        print(f"Rows with 2+ significant components: {rows_with_multiple_significant:,} / {vocab_size:,} ({percentage_multi:.1f}%)")
        
        # Show normalization effect
        print(f"\n=== Normalization Effect ===")
        # Calculate row sums for ALL rows (including zero rows)
        all_row_sums = normalized_likelihoods.sum(dim=1)
        non_zero_rows = (normalized_likelihoods > 0).any(dim=1)
        zero_rows = ~non_zero_rows
        
        print(f"Total rows: {vocab_size:,}")
        print(f"Rows with non-zero entries: {non_zero_rows.sum().item():,}")
        print(f"Rows with all zeros: {zero_rows.sum().item():,}")
        
        if non_zero_rows.any():
            row_sums_nonzero = all_row_sums[non_zero_rows]
            print(f"\nNon-zero rows statistics:")
            print(f"  Mean sum: {row_sums_nonzero.mean().item():.6f}")
            print(f"  Std sum: {row_sums_nonzero.std().item():.6f}")
            print(f"  Min sum: {row_sums_nonzero.min().item():.6f}")
            print(f"  Max sum: {row_sums_nonzero.max().item():.6f}")
            
            # Check how many rows don't sum to 1 (with different tolerance levels)
            tolerances = [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]
            for tol in tolerances:
                perfect_rows = (torch.abs(row_sums_nonzero - 1.0) < tol).sum().item()
                imperfect_rows = len(row_sums_nonzero) - perfect_rows
                percentage_imperfect = 100 * imperfect_rows / len(row_sums_nonzero)
                print(f"  Rows NOT summing to 1.0 (tol={tol}): {imperfect_rows:,}/{len(row_sums_nonzero):,} ({percentage_imperfect:.2f}%)")
        
        # Show distribution of row sums that deviate from 1.0
        if non_zero_rows.any():
            row_sums_nonzero = all_row_sums[non_zero_rows]
            deviations = torch.abs(row_sums_nonzero - 1.0)
            significant_deviations = deviations > 1e-3
            
            if significant_deviations.any():
                print(f"\nRows with significant deviations from 1.0 (>0.001): {significant_deviations.sum().item():,}")
                worst_deviations = deviations[significant_deviations]
                print(f"  Mean deviation: {worst_deviations.mean().item():.6f}")
                print(f"  Max deviation: {worst_deviations.max().item():.6f}")
                
                # Show some examples of problematic rows
                worst_indices = torch.topk(deviations, k=min(5, len(deviations)))[1]
                print(f"  Worst {min(5, len(worst_indices))} row examples:")
                for i, idx in enumerate(worst_indices):
                    actual_row_idx = torch.nonzero(non_zero_rows)[idx].item()
                    sum_val = row_sums_nonzero[idx].item()
                    deviation = deviations[idx].item()
                    non_zero_count = (normalized_likelihoods[actual_row_idx] > 0).sum().item()
                    print(f"    Row {actual_row_idx}: sum={sum_val:.6f}, deviation={deviation:.6f}, non_zeros={non_zero_count}")
            else:
                print(f"\nAll non-zero rows sum very close to 1.0 (deviation < 0.001)")

def main():
    parser = argparse.ArgumentParser(description="Sort and cut projection matrix by top_k")
    parser.add_argument("input_path", help="Path to input projection matrix file")
    parser.add_argument("--top_k", type=int, required=True, help="New top_k value for cutoff")
    parser.add_argument("--output_path", help="Output path (auto-generated if not specified)")
    parser.add_argument("--preserve_last", action="store_true", help="Always preserve the last column as the final element")
    parser.add_argument("--quiet", "-q", action="store_true", help="Suppress progress output")
    # python sort_and_cut_projection_matrix.py /lustre/fsw/portfolios/nvr/projects/nvr_lpr_llm/users/pmolchanov/xtoken/models/runs/s4_l1q4b_lr0_kl1_ce0_k1_emb_top10_3_learn_qa2_transformation_matrices/learned_projection_map_latest.pt --top_k 8 --output_path cross_tokenizer_data/projection_matrix_learned_llama_qwen_top8.pt --preserve_last
    #s4_l1q4b_lr0_kl1_ce0_k1_emb_top10_3_learn_qa2_transformation_matrices
    args = parser.parse_args()
    
    # Auto-generate output path if not specified
    if args.output_path is None:
        input_dir = os.path.dirname(args.input_path)
        input_filename = os.path.basename(args.input_path)
        
        # Extract base name and extension
        base_name, ext = os.path.splitext(input_filename)
        
        # Remove old top_k info if present
        import re
        base_name = re.sub(r'_top_\d+', '', base_name)
        
        # Add new top_k info and preserve_last flag
        suffix = "_sorted"
        if args.preserve_last:
            suffix += "_preservelast"
        output_filename = f"{base_name}_top_{args.top_k}{suffix}{ext}"
        args.output_path = os.path.join(input_dir, output_filename)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    
    # Process the matrix
    sort_and_cut_projection_matrix(
        args.input_path, 
        args.output_path, 
        args.top_k, 
        preserve_last=args.preserve_last,
        verbose=not args.quiet
    )

if __name__ == "__main__":
    main() 