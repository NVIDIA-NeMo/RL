#!/usr/bin/env python3
"""
Script to compare logits between different tensor folders.
Compares dtensor/tp1 and megatron/tp1 (baselines) with their respective tp2, tp4, tp8 folders.
Each tensor has shape [batch_size, sequence_length, vocab_size] containing logits.
Works with files named batch_N_logits.pt.
"""

import os
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
import argparse


def load_tensor(file_path: str, map_location='cpu') -> torch.Tensor:
    """Load a tensor from a .pt file."""
    try:
        tensor = torch.load(file_path, map_location=map_location)
        return tensor
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return None


def get_tensor_info(file_path: str) -> dict:
    """Get tensor information without loading the full tensor into memory."""
    try:
        # Load with mmap to avoid loading into memory
        tensor = torch.load(file_path, map_location='cpu', weights_only=False)
        info = {
            'shape': tuple(tensor.shape),
            'dtype': tensor.dtype,
            'numel': tensor.numel(),
            'size_mb': tensor.element_size() * tensor.numel() / (1024 ** 2)
        }
        del tensor
        return info
    except Exception as e:
        print(f"Error getting info from {file_path}: {e}")
        return None


def compare_shapes(
    a_logits: torch.Tensor,
    b_logits: torch.Tensor,
) -> Dict[str, any]:
    """
    Compare shapes of two logit tensors.
    
    Args:
        a_logits: Baseline logits tensor
        b_logits: Comparison logits tensor
    
    Returns:
        Dictionary with shape information
    """
    return {
        'a_shape': tuple(a_logits.shape),
        'b_shape': tuple(b_logits.shape),
        'shapes_match': a_logits.shape == b_logits.shape,
    }


def compute_logit_differences(
    a_logits: torch.Tensor,
    b_logits: torch.Tensor,
    chunk_size: int = None,
) -> Dict[str, float]:
    """
    Compute numerical differences between two logit tensors.
    
    Args:
        a_logits: Baseline logits tensor [batch_size, sequence_length, vocab_size]
        b_logits: Comparison logits tensor [batch_size, sequence_length, vocab_size]
        chunk_size: Process in chunks along batch dimension to save memory
    
    Returns:
        Dictionary with difference statistics
    """
    batch_size = a_logits.shape[0]
    
    # If no chunk size specified or tensor is small, process all at once
    if chunk_size is None or batch_size <= chunk_size:
        return _compute_diff_single(a_logits, b_logits)
    
    # Process in chunks
    print(f"    Processing in chunks of size {chunk_size}...")
    stats_list = []
    
    for i in range(0, batch_size, chunk_size):
        end_idx = min(i + chunk_size, batch_size)
        a_chunk = a_logits[i:end_idx]
        b_chunk = b_logits[i:end_idx]
        
        chunk_stats = _compute_diff_single(a_chunk, b_chunk)
        stats_list.append(chunk_stats)
        
        del a_chunk, b_chunk
        
        if (i // chunk_size + 1) % 5 == 0:
            print(f"      Processed {end_idx}/{batch_size} samples...")
    
    # Aggregate statistics
    stats = {
        'mean_abs_diff': np.mean([s['mean_abs_diff'] for s in stats_list]),
        'max_abs_diff': np.max([s['max_abs_diff'] for s in stats_list]),
        'min_abs_diff': np.min([s['min_abs_diff'] for s in stats_list]),
        'median_abs_diff': np.median([s['median_abs_diff'] for s in stats_list]),
        'std_abs_diff': np.mean([s['std_abs_diff'] for s in stats_list]),
        'mean_rel_diff': np.mean([s['mean_rel_diff'] for s in stats_list]),
        'max_rel_diff': np.max([s['max_rel_diff'] for s in stats_list]),
        'median_rel_diff': np.median([s['median_rel_diff'] for s in stats_list]),
    }
    
    return stats


def _compute_diff_single(a_logits: torch.Tensor, b_logits: torch.Tensor) -> Dict[str, float]:
    """Compute differences for a single tensor (no chunking)."""
    # Calculate absolute and relative differences
    abs_diff = torch.abs(a_logits - b_logits)
    rel_diff = abs_diff / (torch.abs(a_logits) + 1e-8)  # Add small epsilon to avoid division by zero
    
    # Calculate statistics
    stats = {
        'mean_abs_diff': abs_diff.mean().item(),
        'max_abs_diff': abs_diff.max().item(),
        'min_abs_diff': abs_diff.min().item(),
        'median_abs_diff': abs_diff.median().item(),
        'std_abs_diff': abs_diff.std().item(),
        'mean_rel_diff': rel_diff.mean().item(),
        'max_rel_diff': rel_diff.max().item(),
        'median_rel_diff': rel_diff.median().item(),
    }
    
    return stats


def get_batch_files(folder_path: str) -> List[str]:
    """Get all batch logits files in a folder, sorted numerically."""
    if not os.path.exists(folder_path):
        return []
    
    logits_files = []
    
    for file in os.listdir(folder_path):
        if file.startswith('batch_') and file.endswith('_logits.pt'):
            logits_files.append(file)
    
    # Sort by batch number
    logits_files.sort(key=lambda x: int(x.split('_')[1]))
    
    return logits_files


def _topk_overlap_from_indices(a_topk: torch.Tensor, b_topk: torch.Tensor) -> torch.Tensor:
    """Compute mean top-k overlap ratio given two index tensors of shape [B, T, K]."""
    k = a_topk.shape[-1]
    a_expanded = a_topk.unsqueeze(-1)  # [B, T, K, 1]
    b_expanded = b_topk.unsqueeze(-2)  # [B, T, 1, K]
    # Any match across K for each of the K elements
    overlap_counts = (a_expanded == b_expanded).any(dim=-1).float().sum(dim=-1)  # [B, T]
    return (overlap_counts / float(k)).mean()


def _compute_topk_overlaps(a_logits: torch.Tensor, b_logits: torch.Tensor, k_values: List[int]) -> Dict[int, float]:
    """Compute mean overlap ratios for multiple k values using logits.

    Computes top-k indices once using max(k_values) to avoid repeated work.
    """
    if not k_values:
        return {}
    vocab_size = a_logits.shape[-1]
    max_k_requested = max(k_values)
    max_k = min(max_k_requested, vocab_size)
    if max_k_requested > vocab_size:
        print(f"    Note: Requested max k={max_k_requested} exceeds vocab={vocab_size}; clamping to {max_k}")
    # Compute top-k indices for the largest k once
    _, a_idx = torch.topk(a_logits, k=max_k, dim=-1)
    _, b_idx = torch.topk(b_logits, k=max_k, dim=-1)
    results: Dict[int, float] = {}
    for k in k_values:
        eff_k = min(k, max_k)
        ratio = _topk_overlap_from_indices(a_idx[:, :, :eff_k], b_idx[:, :, :eff_k])
        results[k] = float(ratio.item())
    return results


def compare_logits(base_folder: str, compare_folders: List[str], output_file: str = None, 
                   chunk_size: int = 4, check_shapes_only: bool = False, max_batches: int = None,
                   topk_values: List[int] = None):
    """
    Compare logit numerical differences between base folder and comparison folders.
    
    Args:
        base_folder: Path to baseline folder (e.g., dtensor/tp1)
        compare_folders: List of paths to comparison folders (e.g., dtensor/tp2, dtensor/tp4, dtensor/tp8)
        output_file: Optional file to save results
        chunk_size: Batch chunk size for processing large tensors
        check_shapes_only: If True, only check shapes without computing differences
        max_batches: Maximum number of batches to compare (default: None = all batches)
    """
    print(f"\nComparing logits with baseline: {base_folder}")
    print(f"Comparison folders: {compare_folders}")
    if check_shapes_only:
        print("Mode: SHAPE CHECK ONLY")
    else:
        print(f"Chunk size: {chunk_size}")
    if max_batches is not None:
        print(f"Max batches to compare: {max_batches}")
    print("=" * 80)
    
    # Get all batch files from base folder
    base_logits_files = get_batch_files(base_folder)
    if not base_logits_files:
        print(f"No batch files found in {base_folder}")
        return
    
    # Limit number of batches if specified
    if max_batches is not None and max_batches > 0:
        base_logits_files = base_logits_files[:max_batches]
        print(f"Limiting comparison to first {len(base_logits_files)} batch(es)")
    
    print(f"Found {len(base_logits_files)} logits files in {base_folder}")
    
    # Initialize results storage
    shape_results = {}
    logit_diff_results = {}
    topk_overlap_results: Dict[str, List[Dict[int, float]]] = {}
    for compare_folder in compare_folders:
        shape_results[compare_folder] = []
        logit_diff_results[compare_folder] = []
        topk_overlap_results[compare_folder] = []
    
    # Process each batch
    for i, logits_file in enumerate(base_logits_files):
        print(f"\nProcessing batch {i}: {logits_file}...")
        
        # Load baseline tensor
        base_logits_path = os.path.join(base_folder, logits_file)
        base_logits = load_tensor(base_logits_path)
        
        if base_logits is None:
            print(f"  Skipping batch {i} due to loading error")
            continue
        
        print(f"  Baseline logits shape: {base_logits.shape}")
        
        # Compare with each comparison folder
        for compare_folder in compare_folders:
            compare_logits_path = os.path.join(compare_folder, logits_file)
            
            # Check if file exists
            if not os.path.exists(compare_logits_path):
                print(f"  {compare_folder}: File not found")
                shape_results[compare_folder].append(None)
                logit_diff_results[compare_folder].append(None)
                continue
            
            # Load comparison tensor
            compare_logits = load_tensor(compare_logits_path)
            
            if compare_logits is None:
                print(f"  {compare_folder}: Loading error")
                shape_results[compare_folder].append(None)
                logit_diff_results[compare_folder].append(None)
                continue
            
            print(f"  {compare_folder} logits shape: {compare_logits.shape}")
            print(f"  {compare_folder} logits size: {compare_logits.element_size() * compare_logits.numel() / (1024**3):.2f} GB")
            
            # Handle sequence length mismatch by slicing
            base_bs, base_seq_len, base_vocab = base_logits.shape
            comp_bs, comp_seq_len, comp_vocab = compare_logits.shape
            
            # Check batch size and vocab size match
            if base_bs != comp_bs or base_vocab != comp_vocab:
                print(f"  {compare_folder}: Incompatible shapes! Base: {base_logits.shape}, Compare: {compare_logits.shape}")
                print(f"  {compare_folder}: Batch size or vocab size mismatch - cannot compare")
                shape_results[compare_folder].append({
                    'a_shape': tuple(base_logits.shape),
                    'b_shape': tuple(compare_logits.shape),
                    'shapes_match': False
                })
                logit_diff_results[compare_folder].append(None)
                continue
            
            # Handle sequence length difference
            if comp_seq_len != base_seq_len:
                if comp_seq_len > base_seq_len:
                    print(f"  {compare_folder}: WARNING - Sequence length mismatch!")
                    print(f"  {compare_folder}: Base seq_len={base_seq_len}, Compare seq_len={comp_seq_len}")
                    print(f"  {compare_folder}: Slicing compare tensor to match base: [:, :{base_seq_len}, :]")
                    compare_logits = compare_logits[:, :base_seq_len, :]
                else:
                    assert False, f"Compare tensor has shorter sequence length {comp_seq_len} < {base_seq_len}"
            
            # Record shape information after slicing
            shape_info = {
                'a_shape': tuple(base_logits.shape),
                'b_shape': tuple(compare_logits.shape),
                'shapes_match': base_logits.shape == compare_logits.shape,
                'original_a_shape': (base_bs, base_seq_len, base_vocab),
                'original_b_shape': (comp_bs, comp_seq_len, comp_vocab),
                'seq_len_sliced': comp_seq_len != base_seq_len
            }
            shape_results[compare_folder].append(shape_info)
            
            if check_shapes_only:
                print(f"  {compare_folder}: Shapes match ✓")
                logit_diff_results[compare_folder].append(None)
                topk_overlap_results[compare_folder].append(None)
                continue
            
            # Calculate logit differences
            logit_diff_stats = compute_logit_differences(base_logits, compare_logits, chunk_size=chunk_size)
            logit_diff_results[compare_folder].append(logit_diff_stats)
            
            print(f"  {compare_folder}: Mean abs diff = {logit_diff_stats['mean_abs_diff']:.6e}")
            print(f"  {compare_folder}: Max abs diff = {logit_diff_stats['max_abs_diff']:.6e}")
            print(f"  {compare_folder}: Median abs diff = {logit_diff_stats['median_abs_diff']:.6e}")
            
            # Calculate top-k overlap(s) if requested
            if topk_values is not None and len(topk_values) > 0:
                overlaps = _compute_topk_overlaps(base_logits, compare_logits, topk_values)
                topk_overlap_results[compare_folder].append(overlaps)
                for k in sorted(overlaps.keys()):
                    print(f"  {compare_folder}: Top-{k} overlap = {overlaps[k]:.4f}")
            else:
                topk_overlap_results[compare_folder].append(None)
    
    # Calculate and display summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    
    logit_diff_summary = {}
    topk_overlap_summary: Dict[str, Dict[int, Dict[str, float]]] = {}
    
    for compare_folder in compare_folders:
        # Check for shape mismatches and sequence length slicing
        shape_mismatches = sum(1 for s in shape_results[compare_folder] 
                              if s is not None and not s['shapes_match'])
        seq_len_sliced = sum(1 for s in shape_results[compare_folder]
                            if s is not None and s.get('seq_len_sliced', False))
        
        # Process logit difference results
        valid_logit_diffs = [r for r in logit_diff_results[compare_folder] if r is not None]
        
        if valid_logit_diffs:
            # Aggregate statistics across all batches
            mean_abs_diffs = [r['mean_abs_diff'] for r in valid_logit_diffs]
            max_abs_diffs = [r['max_abs_diff'] for r in valid_logit_diffs]
            median_abs_diffs = [r['median_abs_diff'] for r in valid_logit_diffs]
            mean_rel_diffs = [r['mean_rel_diff'] for r in valid_logit_diffs]
            
            logit_diff_summary[compare_folder] = {
                'mean_abs_diff_avg': np.mean(mean_abs_diffs),
                'mean_abs_diff_std': np.std(mean_abs_diffs),
                'mean_abs_diff_max': np.max(mean_abs_diffs),
                'median_abs_diff_avg': np.mean(median_abs_diffs),
                'max_abs_diff_avg': np.mean(max_abs_diffs),
                'max_abs_diff_max': np.max(max_abs_diffs),
                'mean_rel_diff_avg': np.mean(mean_rel_diffs),
                'mean_rel_diff_max': np.max(mean_rel_diffs),
                'count': len(valid_logit_diffs),
                'shape_mismatches': shape_mismatches,
                'seq_len_sliced': seq_len_sliced
            }
        
        # Process top-k overlap results
        valid_topk_overlaps = [r for r in topk_overlap_results[compare_folder] if isinstance(r, dict)]
        if valid_topk_overlaps:
            k_to_values: Dict[int, List[float]] = {}
            for entry in valid_topk_overlaps:
                for k, val in entry.items():
                    k_to_values.setdefault(k, []).append(val)
            # Aggregate per k
            folder_summary: Dict[int, Dict[str, float]] = {}
            for k, values in k_to_values.items():
                folder_summary[k] = {
                    'mean': float(np.mean(values)),
                    'std': float(np.std(values)),
                    'min': float(np.min(values)),
                    'max': float(np.max(values)),
                    'count': len(values),
                }
            topk_overlap_summary[compare_folder] = folder_summary
        
        # Display results
        print(f"\n{compare_folder}:")
        
        if shape_mismatches > 0:
            print(f"  WARNING: {shape_mismatches} shape mismatches detected!")
        
        if seq_len_sliced > 0:
            print(f"  NOTE: {seq_len_sliced} batches had sequence length differences (sliced for comparison)")
        
        if valid_logit_diffs:
            print(f"  LOGIT DIFFERENCE STATISTICS (across {logit_diff_summary[compare_folder]['count']} batches):")
            print(f"    Mean abs diff: {logit_diff_summary[compare_folder]['mean_abs_diff_avg']:.6e} ± {logit_diff_summary[compare_folder]['mean_abs_diff_std']:.6e}")
            print(f"    Median abs diff: {logit_diff_summary[compare_folder]['median_abs_diff_avg']:.6e}")
            print(f"    Max abs diff (avg): {logit_diff_summary[compare_folder]['max_abs_diff_avg']:.6e}")
            print(f"    Max abs diff (overall): {logit_diff_summary[compare_folder]['max_abs_diff_max']:.6e}")
            print(f"    Mean rel diff: {logit_diff_summary[compare_folder]['mean_rel_diff_avg']:.6e} ({logit_diff_summary[compare_folder]['mean_rel_diff_avg']*100:.4f}%)")
        else:
            print(f"  LOGIT DIFFERENCE STATISTICS: No valid comparisons")

        if compare_folder in topk_overlap_summary:
            print(f"  TOP-K OVERLAP STATISTICS:")
            for k in sorted(topk_overlap_summary[compare_folder].keys()):
                s = topk_overlap_summary[compare_folder][k]
                print(f"    Top-{k}: {s['mean']:.4f} ± {s['std']:.4f} (min {s['min']:.4f}, max {s['max']:.4f}, n={s['count']})")
        else:
            if topk_values:
                print(f"  TOP-K OVERLAP STATISTICS: No valid comparisons")
    
    # Save results to file if requested
    if output_file:
        save_results(shape_results, logit_diff_results, logit_diff_summary, topk_overlap_results, topk_overlap_summary, output_file)
        print(f"\nResults saved to: {output_file}")
    
    return logit_diff_summary


def save_results(shape_results: Dict, logit_diff_results: Dict, logit_diff_summary: Dict, 
                 topk_overlap_results: Dict, topk_overlap_summary: Dict, output_file: str):
    """Save results to a text file."""
    with open(output_file, 'w') as f:
        f.write("Logit Numerical Difference Comparison Results\n")
        f.write("=" * 80 + "\n\n")
        
        # Write summary statistics
        f.write("SUMMARY STATISTICS\n")
        f.write("-" * 40 + "\n")
        
        for folder in logit_diff_results.keys():
            f.write(f"\n{folder}:\n")
            
            # Shape mismatch check
            shape_mismatches = sum(1 for s in shape_results[folder] 
                                  if s is not None and not s['shapes_match'])
            seq_len_sliced = sum(1 for s in shape_results[folder]
                                if s is not None and s.get('seq_len_sliced', False))
            
            if shape_mismatches > 0:
                f.write(f"  WARNING: {shape_mismatches} shape mismatches detected!\n")
            
            if seq_len_sliced > 0:
                f.write(f"  NOTE: {seq_len_sliced} batches had sequence length differences (sliced for comparison)\n")
            
            # Logit difference statistics
            if folder in logit_diff_summary:
                stats = logit_diff_summary[folder]
                f.write(f"  LOGIT DIFFERENCE STATISTICS (across {stats['count']} batches):\n")
                f.write(f"    Mean abs diff: {stats['mean_abs_diff_avg']:.6e} ± {stats['mean_abs_diff_std']:.6e}\n")
                f.write(f"    Median abs diff: {stats['median_abs_diff_avg']:.6e}\n")
                f.write(f"    Max abs diff (avg): {stats['max_abs_diff_avg']:.6e}\n")
                f.write(f"    Max abs diff (overall): {stats['max_abs_diff_max']:.6e}\n")
                f.write(f"    Mean rel diff: {stats['mean_rel_diff_avg']:.6e} ({stats['mean_rel_diff_avg']*100:.4f}%)\n")
            else:
                f.write(f"  LOGIT DIFFERENCE STATISTICS: No valid comparisons\n")
            
            # Top-k overlap statistics
            if folder in topk_overlap_summary and topk_overlap_summary[folder]:
                f.write(f"  TOP-K OVERLAP STATISTICS:\n")
                for k in sorted(topk_overlap_summary[folder].keys()):
                    s = topk_overlap_summary[folder][k]
                    f.write(f"    Top-{k}: {s['mean']:.4f} ± {s['std']:.4f} (min {s['min']:.4f}, max {s['max']:.4f}, n={s['count']})\n")
            else:
                f.write(f"  TOP-K OVERLAP STATISTICS: No valid comparisons\n")
        
        # Write detailed results
        f.write("\n\nDETAILED RESULTS (per batch)\n")
        f.write("-" * 40 + "\n")
        
        for folder in logit_diff_results.keys():
            f.write(f"\n{folder}:\n")
            shape_infos = shape_results[folder]
            logit_diffs = logit_diff_results[folder]
            topk_overlaps = topk_overlap_results[folder]
            
            for i, (shape_info, logit_diff, topk_overlap_entry) in enumerate(zip(shape_infos, logit_diffs, topk_overlaps)):
                f.write(f"  batch_{i}:\n")
                
                if shape_info is not None:
                    if not shape_info['shapes_match']:
                        f.write(f"    Shape mismatch! Base: {shape_info['a_shape']}, Compare: {shape_info['b_shape']}\n")
                    else:
                        f.write(f"    Shape: {shape_info['a_shape']}\n")
                        if shape_info.get('seq_len_sliced', False):
                            f.write(f"    Original shapes: Base: {shape_info['original_a_shape']}, Compare: {shape_info['original_b_shape']}\n")
                            f.write(f"    Note: Sequence length sliced for comparison\n")
                
                if logit_diff is not None:
                    f.write(f"    Mean abs diff: {logit_diff['mean_abs_diff']:.6e}\n")
                    f.write(f"    Max abs diff: {logit_diff['max_abs_diff']:.6e}\n")
                    f.write(f"    Median abs diff: {logit_diff['median_abs_diff']:.6e}\n")
                    f.write(f"    Mean rel diff: {logit_diff['mean_rel_diff']:.6e}\n")
                else:
                    f.write(f"    Logit differences: N/A\n")
                
                if isinstance(topk_overlap_entry, dict) and topk_overlap_entry:
                    f.write(f"    Top-k overlaps:\n")
                    for k in sorted(topk_overlap_entry.keys()):
                        f.write(f"      Top-{k}: {topk_overlap_entry[k]:.4f}\n")
                else:
                    f.write(f"    Top-k overlaps: N/A\n")


def main():
    parser = argparse.ArgumentParser(
        description='Compare logit numerical differences between different TP sizes',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare both dtensor and megatron with first batch only (default)
  python compare_logits.py
  
  # Compare only dtensor with first 3 batches
  python compare_logits.py --framework dtensor --max-batches 3
  
  # Compare all batches (set max-batches to 0 or negative)
  python compare_logits.py --max-batches 0
  
  # Compare only megatron with specific TP sizes
  python compare_logits.py --framework megatron --tp-sizes 2 4
  
  # Custom base and comparison folders with 2 batches
  python compare_logits.py --base custom/tp1 --compare custom/tp2 custom/tp4 --max-batches 2
        """
    )
    parser.add_argument('--framework', choices=['dtensor', 'megatron', 'both'], default='both',
                       help='Which framework to compare (default: both)')
    parser.add_argument('--tp-sizes', nargs='+', type=int, default=[2, 4, 8],
                       help='TP sizes to compare against tp1 (default: 2 4 8)')
    parser.add_argument('--base', help='Custom base folder path (overrides --framework)')
    parser.add_argument('--compare', nargs='+', help='Custom comparison folders (overrides --framework and --tp-sizes)')
    parser.add_argument('--output-dir', default='comparison_results', 
                       help='Directory to save output files (default: comparison_results)')
    parser.add_argument('--chunk-size', type=int, default=4,
                       help='Batch chunk size for processing large tensors (default: 4)')
    parser.add_argument('--check-shapes-only', action='store_true',
                       help='Only check tensor shapes without computing differences (fast)')
    parser.add_argument('--max-batches', type=int, default=1,
                       help='Maximum number of batches to compare (default: 1, set to 0 or negative for all batches)')
    parser.add_argument('--topk-values', nargs='+', type=int, default=[1, 10, 100, 1000],
                       help='Top-k values for overlap comparison on logits (default: 1 10 100 1000)')
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    frameworks_to_compare = []
    
    # If custom base and compare are provided, use them
    if args.base and args.compare:
        if not os.path.exists(args.base):
            print(f"Error: Base folder '{args.base}' does not exist")
            return
        
        existing_compare_folders = []
        for folder in args.compare:
            if os.path.exists(folder):
                existing_compare_folders.append(folder)
            else:
                print(f"Warning: Comparison folder '{folder}' does not exist")
        
        if not existing_compare_folders:
            print("Error: No valid comparison folders found")
            return
        
        output_file = os.path.join(args.output_dir, 'custom_comparison_results.txt')
        max_batches = None if args.max_batches <= 0 else args.max_batches
        compare_logits(args.base, existing_compare_folders, output_file, 
                      chunk_size=args.chunk_size, check_shapes_only=args.check_shapes_only,
                      max_batches=max_batches, topk_values=args.topk_values)
        return
    
    # Otherwise, use framework-based comparison
    if args.framework in ['dtensor', 'both']:
        frameworks_to_compare.append('dtensor')
    if args.framework in ['megatron', 'both']:
        frameworks_to_compare.append('megatron')
    
    all_summaries = {}
    
    for framework in frameworks_to_compare:
        print("\n" + "=" * 80)
        print(f"FRAMEWORK: {framework.upper()}")
        print("=" * 80)
        
        base_folder = f"{framework}/tp1"
        
        # Check if base folder exists
        if not os.path.exists(base_folder):
            print(f"Warning: Base folder '{base_folder}' does not exist, skipping {framework}")
            continue
        
        # Build comparison folders
        compare_folders = []
        for tp_size in args.tp_sizes:
            compare_folder = f"{framework}/tp{tp_size}"
            if os.path.exists(compare_folder):
                compare_folders.append(compare_folder)
            else:
                print(f"Warning: Comparison folder '{compare_folder}' does not exist")
        
        if not compare_folders:
            print(f"Warning: No valid comparison folders found for {framework}")
            continue
        
        # Run comparison
        output_file = os.path.join(args.output_dir, f'{framework}_comparison_results.txt')
        max_batches = None if args.max_batches <= 0 else args.max_batches
        summary = compare_logits(base_folder, compare_folders, output_file,
                                chunk_size=args.chunk_size, check_shapes_only=args.check_shapes_only,
                                max_batches=max_batches, topk_values=args.topk_values)
        all_summaries[framework] = summary
    
    # Cross-framework comparison: dtensor/tp1 vs megatron/tp1 (if both exist and selected)
    if 'dtensor' in frameworks_to_compare and 'megatron' in frameworks_to_compare:
        dtensor_tp1 = os.path.join('dtensor', 'tp1')
        megatron_tp1 = os.path.join('megatron', 'tp1')
        if os.path.exists(dtensor_tp1) and os.path.exists(megatron_tp1):
            print("\n" + "=" * 80)
            print("CROSS-FRAMEWORK: DTENSOR(tp1) vs MEGATRON(tp1)")
            print("=" * 80)
            output_file = os.path.join(args.output_dir, 'dtensor_vs_megatron_tp1_results.txt')
            max_batches = None if args.max_batches <= 0 else args.max_batches
            summary = compare_logits(dtensor_tp1, [megatron_tp1], output_file,
                                     chunk_size=args.chunk_size,
                                     check_shapes_only=args.check_shapes_only,
                                     max_batches=max_batches,
                                     topk_values=args.topk_values)
            all_summaries['cross_tp1'] = summary
    
    # Print overall summary
    if all_summaries and not args.check_shapes_only:
        print("\n" + "=" * 80)
        print("OVERALL SUMMARY")
        print("=" * 80)
        
        for framework, summary in all_summaries.items():
            print(f"\n{framework.upper()}:")
            for folder, stats in summary.items():
                tp_size = folder.split('/')[-1]
                print(f"  {tp_size}:")
                print(f"    Mean abs diff: {stats['mean_abs_diff_avg']:.6e}")
                print(f"    Max abs diff: {stats['max_abs_diff_max']:.6e}")
                print(f"    Mean rel diff: {stats['mean_rel_diff_avg']:.4%}")


if __name__ == "__main__":
    main()