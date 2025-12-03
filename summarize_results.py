#!/usr/bin/env python3
"""
Collect and analyze GRPO benchmark results from WandB and SLURM logs.

Usage:
    python summarize_results.py                          # Show all results from WandB
    python summarize_results.py --project PROJECT        # Specify WandB project
    python summarize_results.py --output results.csv     # Save to CSV
    python summarize_results.py --slurm                  # Also parse SLURM logs
    python summarize_results.py --all                    # Include failed/crashed runs
"""

import argparse
import csv
import json
import os
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd


def get_slurm_jobs():
    """Get list of recent SLURM jobs for current user."""
    try:
        # Get recent jobs from sacct
        result = subprocess.run(
            ['sacct', '-u', os.environ.get('USER', ''), 
             '--format=JobID,JobName,State,Start,End,Elapsed,NNodes,NCPUS,ExitCode',
             '--parsable2', '-n', '--starttime=now-7days'],
            capture_output=True, text=True, timeout=30
        )
        
        jobs = []
        for line in result.stdout.strip().split('\n'):
            if line.strip() and '.' not in line.split('|')[0]:  # Skip sub-jobs
                parts = line.split('|')
                if len(parts) >= 9:
                    jobs.append({
                        'job_id': parts[0],
                        'job_name': parts[1],
                        'state': parts[2],
                        'start': parts[3],
                        'end': parts[4],
                        'elapsed': parts[5],
                        'nodes': parts[6],
                        'cpus': parts[7],
                        'exit_code': parts[8]
                    })
        return jobs
    except Exception as e:
        print(f"⚠️  Failed to get SLURM jobs: {e}")
        return []


def parse_job_name(name: str) -> dict:
    """Parse job name to extract configuration."""
    result = {
        'model': 'Unknown',
        'num_nodes': 0,
        'gpus_per_node': 0,
        'total_gpus': 0,
        't_tp': 1, 't_pp': 1, 't_ep': 1, 't_cp': 1, 't_vpp': 1, 't_dp': 1,
        'g_tp': 1, 'g_pp': 1, 'g_ep': 1, 'g_dp': 1,
    }
    
    # Extract model name
    model_match = re.match(r'^([^-]+)', name)
    if model_match:
        result['model'] = model_match.group(1)
    
    # Extract NxG (nodes x gpus_per_node)
    nxg_match = re.search(r'-N(\d+)xG(\d+)', name)
    if nxg_match:
        result['num_nodes'] = int(nxg_match.group(1))
        result['gpus_per_node'] = int(nxg_match.group(2))
        result['total_gpus'] = result['num_nodes'] * result['gpus_per_node']
    
    # Extract Training params: T.tp4.pp4.ep1
    train_match = re.search(r'T\.tp(\d+)\.pp(\d+)\.ep(\d+)', name)
    if train_match:
        result['t_tp'] = int(train_match.group(1))
        result['t_pp'] = int(train_match.group(2))
        result['t_ep'] = int(train_match.group(3))
    
    # Extract Generation params: G.tp4.pp1
    gen_match = re.search(r'G\.tp(\d+)\.pp(\d+)', name)
    if gen_match:
        result['g_tp'] = int(gen_match.group(1))
        result['g_pp'] = int(gen_match.group(2))
    
    # Calculate DP
    mp = result['t_tp'] * result['t_pp']
    if mp > 0 and result['total_gpus'] > 0:
        result['t_dp'] = result['total_gpus'] // mp
    
    g_mp = result['g_tp'] * result['g_pp']
    if g_mp > 0 and result['total_gpus'] > 0:
        result['g_dp'] = result['total_gpus'] // g_mp
    
    return result


def parse_wandb_name(name: str) -> dict:
    """Parse WandB run name to extract configuration.
    
    Supports multiple formats:
    - New format (parentheses): Qwen32B-N8xG4-Train(tp4.pp4.ep1.cp1.vpp1)-Gen(tp4.pp1.ep1)
    - New format (underscores): Qwen32B_N8xG4_Ttp4pp4ep1cp1vpp1_Gtp4pp1ep1
    - Old format: async-qwen-32B-seg64-Gtp1-Gep1-Gpp1-Ttp1-Tep1-Tpp2-32T32G
    """
    result = {
        'model': 'Unknown',
        'num_nodes': 0,
        'gpus_per_node': 0,
        'total_gpus': 0,
        't_tp': 1, 't_pp': 1, 't_ep': 1, 't_cp': 1, 't_vpp': 1, 't_dp': 1,
        'g_tp': 1, 'g_pp': 1, 'g_ep': 1, 'g_dp': 1,
    }
    
    train_match = None
    gen_match = None
    
    # ============================================
    # Try NEW format with underscores: Model_NxG_Ttp...vpp..._Gtp...ep...
    # Example: Qwen32B_N8xG4_Ttp4pp4ep1cp1vpp1_Gtp4pp1ep1
    # ============================================
    
    # Extract NxG with underscore prefix
    nxg_underscore_match = re.search(r'_N(\d+)xG(\d+)', name)
    if nxg_underscore_match:
        result['num_nodes'] = int(nxg_underscore_match.group(1))
        result['gpus_per_node'] = int(nxg_underscore_match.group(2))
        result['total_gpus'] = result['num_nodes'] * result['gpus_per_node']
        
        # Extract model name (everything before _N)
        model_match = re.match(r'^([^_]+)', name)
        if model_match:
            result['model'] = model_match.group(1)
        
        # Extract Training params: _Ttp4pp4ep1cp1vpp1 (old) or _Ttp4pp4ep1cp1 (new, no vpp)
        train_underscore_match = re.search(r'_Ttp(\d+)pp(\d+)ep(\d+)cp(\d+)(?:vpp(\d+))?', name)
        if train_underscore_match:
            train_match = train_underscore_match
            result['t_tp'] = int(train_underscore_match.group(1))
            result['t_pp'] = int(train_underscore_match.group(2))
            result['t_ep'] = int(train_underscore_match.group(3))
            result['t_cp'] = int(train_underscore_match.group(4))
            if train_underscore_match.group(5):
                result['t_vpp'] = int(train_underscore_match.group(5))
        
        # Extract Generation params: _Gtp4pp1ep1
        gen_underscore_match = re.search(r'_Gtp(\d+)pp(\d+)ep(\d+)', name)
        if gen_underscore_match:
            gen_match = gen_underscore_match
            result['g_tp'] = int(gen_underscore_match.group(1))
            result['g_pp'] = int(gen_underscore_match.group(2))
            result['g_ep'] = int(gen_underscore_match.group(3))
    
    # ============================================
    # Try NEW format with parentheses: Model-NxG-Train(...)-Gen(...)
    # ============================================
    
    if not nxg_underscore_match:
        # Extract model name (first part before -N)
        model_match = re.match(r'^([^-]+(?:-[^N][^-]*)*)', name)
        if model_match:
            result['model'] = model_match.group(1)
        
        # Extract NxG (nodes x gpus_per_node) - New format with hyphen
        nxg_match = re.search(r'-N(\d+)xG(\d+)', name)
        if nxg_match:
            result['num_nodes'] = int(nxg_match.group(1))
            result['gpus_per_node'] = int(nxg_match.group(2))
            result['total_gpus'] = result['num_nodes'] * result['gpus_per_node']
        
        # Extract Training params: Train(tp4.pp4.ep1.cp1.vpp1)
        train_match = re.search(r'Train\(tp(\d+)\.pp(\d+)\.ep(\d+)\.cp(\d+)\.vpp(\d+)\)', name)
        if train_match:
            result['t_tp'] = int(train_match.group(1))
            result['t_pp'] = int(train_match.group(2))
            result['t_ep'] = int(train_match.group(3))
            result['t_cp'] = int(train_match.group(4))
            result['t_vpp'] = int(train_match.group(5))
        
        # Extract Generation params: Gen(tp4.pp1.ep1)
        gen_match = re.search(r'Gen\(tp(\d+)\.pp(\d+)\.ep(\d+)\)', name)
        if gen_match:
            result['g_tp'] = int(gen_match.group(1))
            result['g_pp'] = int(gen_match.group(2))
            result['g_ep'] = int(gen_match.group(3))
    
    # ============================================
    # Try OLD format: async-qwen-32B-seg64-Gtp1-Gep1-Gpp1-Ttp1-Tep1-Tpp2-32T32G
    # ============================================
    
    # If new format didn't match, try old format
    if not train_match and not gen_match:
        # Extract model from old format (e.g., "async-qwen-32B" -> "qwen-32B")
        old_model_match = re.search(r'(?:async-|sync-)?(\w+[-]?\d*B)', name, re.IGNORECASE)
        if old_model_match:
            result['model'] = old_model_match.group(1)
        
        # Extract segment (total nodes) from old format: seg64
        seg_match = re.search(r'seg(\d+)', name)
        if seg_match:
            result['num_nodes'] = int(seg_match.group(1))
            result['gpus_per_node'] = 4  # Default for GB200
            result['total_gpus'] = result['num_nodes'] * result['gpus_per_node']
        
        # Extract Generation params from old format: Gtp1-Gep1-Gpp1
        g_tp_match = re.search(r'Gtp(\d+)', name)
        g_ep_match = re.search(r'Gep(\d+)', name)
        g_pp_match = re.search(r'Gpp(\d+)', name)
        if g_tp_match:
            result['g_tp'] = int(g_tp_match.group(1))
        if g_ep_match:
            result['g_ep'] = int(g_ep_match.group(1))
        if g_pp_match:
            result['g_pp'] = int(g_pp_match.group(1))
        
        # Extract Training params from old format: Ttp1-Tep1-Tpp2
        t_tp_match = re.search(r'Ttp(\d+)', name)
        t_ep_match = re.search(r'Tep(\d+)', name)
        t_pp_match = re.search(r'Tpp(\d+)', name)
        if t_tp_match:
            result['t_tp'] = int(t_tp_match.group(1))
        if t_ep_match:
            result['t_ep'] = int(t_ep_match.group(1))
        if t_pp_match:
            result['t_pp'] = int(t_pp_match.group(1))
        
        # Extract train/gen node split from old format: 32T32G (32 train, 32 gen)
        split_match = re.search(r'(\d+)T(\d+)G$', name)
        if split_match:
            train_nodes = int(split_match.group(1))
            gen_nodes = int(split_match.group(2))
            # Total nodes might be train + gen for async
            if result['num_nodes'] == 0:
                result['num_nodes'] = train_nodes + gen_nodes
                result['gpus_per_node'] = 4
                result['total_gpus'] = result['num_nodes'] * result['gpus_per_node']
    
    # ============================================
    # Calculate DP (Data Parallelism)
    # ============================================
    mp = result['t_tp'] * result['t_pp']
    if mp > 0 and result['total_gpus'] > 0:
        result['t_dp'] = result['total_gpus'] // mp
    
    g_mp = result['g_tp'] * result['g_pp']
    if g_mp > 0 and result['total_gpus'] > 0:
        result['g_dp'] = result['total_gpus'] // g_mp
    
    return result


def fetch_wandb_results(project: str, entity: Optional[str] = None) -> list:
    """Fetch results from WandB."""
    try:
        import wandb
    except ImportError:
        print("⚠️  wandb not installed. Install with: pip install wandb")
        return []
    
    api = wandb.Api()
    
    try:
        if entity:
            runs = api.runs(f"{entity}/{project}")
        else:
            runs = api.runs(project)
    except wandb.errors.CommError as e:
        if "Could not find project" in str(e):
            print(f"⚠️  WandB project '{project}' does not exist yet.")
            print("   This is normal if no experiments have been successfully run.")
            print("   The project will be created automatically when the first experiment logs data.")
            print("\n💡 Tips:")
            print("   1. Run an experiment first: python launch_jobs.py")
            print("   2. Or use --slurm to check SLURM job status: python summarize_results.py --slurm")
            print("   3. Or specify a different project: python summarize_results.py --project YOUR_PROJECT")
        else:
            print(f"⚠️  Failed to fetch WandB runs: {e}")
        return []
    except Exception as e:
        print(f"⚠️  Failed to fetch WandB runs: {e}")
        return []
    
    results = []
    for run in runs:
        try:
            # Parse config from run name
            config = parse_wandb_name(run.name)
            
            # Get WandB config and summary
            run_config = run.config
            summary = run.summary._json_dict
            
            # ============================================
            # Try to get config from WandB config first (more accurate)
            # ============================================
            if run_config:
                # Cluster config
                cluster_cfg = run_config.get('cluster', {})
                if cluster_cfg:
                    if cluster_cfg.get('num_nodes'):
                        config['num_nodes'] = cluster_cfg.get('num_nodes', config['num_nodes'])
                    if cluster_cfg.get('gpus_per_node'):
                        config['gpus_per_node'] = cluster_cfg.get('gpus_per_node', config['gpus_per_node'])
                    config['total_gpus'] = config['num_nodes'] * config['gpus_per_node']
                
                # Policy/Training config
                policy_cfg = run_config.get('policy', {})
                megatron_cfg = policy_cfg.get('megatron_cfg', {})
                if megatron_cfg:
                    config['t_tp'] = megatron_cfg.get('tensor_model_parallel_size', config['t_tp'])
                    config['t_pp'] = megatron_cfg.get('pipeline_model_parallel_size', config['t_pp'])
                    config['t_ep'] = megatron_cfg.get('expert_model_parallel_size', config['t_ep'])
                    config['t_cp'] = megatron_cfg.get('context_parallel_size', config['t_cp'])
                    config['t_vpp'] = megatron_cfg.get('num_layers_per_virtual_pipeline_stage', config['t_vpp'])
                
                # Generation config
                gen_cfg = policy_cfg.get('generation', {})
                vllm_cfg = gen_cfg.get('vllm_cfg', {})
                if vllm_cfg:
                    config['g_tp'] = vllm_cfg.get('tensor_parallel_size', config['g_tp'])
                    config['g_pp'] = vllm_cfg.get('pipeline_parallel_size', config['g_pp'])
                    config['g_ep'] = vllm_cfg.get('expert_parallel_size', config['g_ep'])
                
                # Model name from config
                if policy_cfg.get('model_name'):
                    model_name = policy_cfg.get('model_name', '')
                    # Extract short name (e.g., "Qwen/Qwen3-32B" -> "Qwen3-32B")
                    config['model'] = model_name.split('/')[-1] if '/' in model_name else model_name
                
                # Recalculate DP
                mp = config['t_tp'] * config['t_pp']
                if mp > 0 and config['total_gpus'] > 0:
                    config['t_dp'] = config['total_gpus'] // mp
                
                g_mp = config['g_tp'] * config['g_pp']
                if g_mp > 0 and config['total_gpus'] > 0:
                    config['g_dp'] = config['total_gpus'] // g_mp
            
            # ============================================
            # Get metrics using 5-step moving average (like get_wandb_log_for_nemorl.py)
            # This provides more stable metrics by averaging around a target step
            # ============================================
            
            # Detect if async (off-policy) based on run name or config
            is_async = 'async' in run.name.lower()
            grpo_cfg = run_config.get('grpo', {}) if run_config else {}
            async_cfg = grpo_cfg.get('async_grpo', {})
            if async_cfg.get('enabled'):
                is_async = True
            
            # Define metric keys to fetch (same as get_wandb_log_for_nemorl.py)
            metric_keys = [
                # Timing metrics
                'timing/train/total_step_time',
                'timing/train/generation',
                'timing/train/exposed_generation',  # for async/off-policy
                'timing/train/policy_and_reference_logprobs',
                'timing/train/policy_training',
                'timing/train/data_processing',
                'timing/train/reward_calculation',
                'timing/train/training_prep',
                'timing/train/prepare_for_generation/transfer_and_update_weights',
                # Performance metrics
                'performance/tokens_per_sec_per_gpu',
                'performance/generation_tokens_per_sec_per_gpu',
                'performance/training_worker_group_tokens_per_sec_per_gpu',
                'performance/policy_training_tokens_per_sec_per_gpu',
                'performance/policy_and_reference_logprobs_tokens_per_sec_per_gpu',
                'performance/train_flops_per_gpu',
                'performance/train_fp_utilization',
                'performance/tokens_per_sec',
                'performance/seq_per_sec',
            ]
            
            # Initialize metrics dict with defaults
            metrics = {key: 0 for key in metric_keys}
            
            # Try to get 5-step moving average from history
            # Parameters: at_step=5, average_steps=5 (same as get_wandb_log_for_nemorl.py)
            at_step = 5
            average_steps = 5
            
            try:
                # Fetch history for this run
                history = run.history(keys=metric_keys, pandas=True)
                
                if history is not None and len(history) > 0:
                    # Calculate step range for averaging (same logic as get_wandb_log_for_nemorl.py)
                    min_step = at_step - average_steps // 2  # step 3
                    max_step = at_step + average_steps // 2 - 1 + (average_steps % 2)  # step 7
                    
                    # Adjust if run has fewer steps
                    max_available_step = len(history) - 1
                    if max_available_step < min_step:
                        # Not enough steps, use all available
                        min_step = 0
                        max_step = max_available_step
                    elif max_available_step < max_step:
                        # Adjust range to available steps
                        max_step = max_available_step
                    
                    # Get the slice and calculate mean
                    history_slice = history.iloc[min_step:max_step + 1]
                    
                    if len(history_slice) > 0:
                        avg_metrics = history_slice.mean()
                        
                        # Update metrics dict with averaged values
                        for key in metric_keys:
                            if key in avg_metrics and not pd.isna(avg_metrics[key]):
                                metrics[key] = avg_metrics[key]
            except Exception as e:
                # If history fetch fails, fall back to summary
                pass
            
            # Fallback: If metrics are still 0, try to get from summary
            def get_metric_with_fallback(primary_key, *fallback_keys):
                """Get metric from averaged history, fall back to summary if not available."""
                val = metrics.get(primary_key, 0)
                if val and val != 0:
                    return val
                # Try fallback keys from summary
                for key in [primary_key] + list(fallback_keys):
                    val = summary.get(key)
                    if val is not None and val != 0:
                        return val
                return 0
            
            # ============================================
            # Extract timing metrics
            # ============================================
            
            # Total step time
            step_time = get_metric_with_fallback(
                'timing/train/total_step_time',
                'train/step_time', 
                'step_time',
            )
            
            # Generation time - different for sync vs async
            if is_async:
                generation_time = get_metric_with_fallback(
                    'timing/train/exposed_generation',
                    'timing/train/generation',
                )
            else:
                generation_time = get_metric_with_fallback(
                    'timing/train/generation',
                )
            
            # Policy training time
            policy_training_time = get_metric_with_fallback(
                'timing/train/policy_training',
            )
            
            # Logprobs calculation time
            logprobs_time = get_metric_with_fallback(
                'timing/train/policy_and_reference_logprobs',
            )
            
            # Other timing metrics
            data_processing_time = get_metric_with_fallback('timing/train/data_processing')
            reward_calculation_time = get_metric_with_fallback('timing/train/reward_calculation')
            training_prep_time = get_metric_with_fallback('timing/train/training_prep')
            weight_transfer_time = get_metric_with_fallback('timing/train/prepare_for_generation/transfer_and_update_weights')
            
            # ============================================
            # Extract performance metrics
            # ============================================
            
            # Tokens per second per GPU (primary metric)
            tokens_per_sec_per_gpu = get_metric_with_fallback(
                'performance/tokens_per_sec_per_gpu',
                'train/tokens_per_sec_per_gpu',
            )
            
            # Generation tokens per second per GPU
            gen_tokens_per_sec_per_gpu = get_metric_with_fallback(
                'performance/generation_tokens_per_sec_per_gpu',
                'generation/tokens_per_sec_per_gpu',
            )
            
            # Training worker group tokens per second per GPU
            training_tokens_per_sec_per_gpu = get_metric_with_fallback(
                'performance/training_worker_group_tokens_per_sec_per_gpu',
            )
            
            # Policy training tokens per second per GPU
            policy_training_tokens_per_sec_per_gpu = get_metric_with_fallback(
                'performance/policy_training_tokens_per_sec_per_gpu',
            )
            
            # Logprobs tokens per second per GPU
            logprobs_tokens_per_sec_per_gpu = get_metric_with_fallback(
                'performance/policy_and_reference_logprobs_tokens_per_sec_per_gpu',
            )
            
            # TFLOPs per GPU
            tflops = get_metric_with_fallback(
                'performance/train_flops_per_gpu',
                'train/per_gpu_tflops',
                'per_gpu_tflops',
            )
            # Convert from FLOPs to TFLOPs if needed (if value > 1e9, it's likely in FLOPs)
            if tflops > 1e9:
                tflops = tflops / 1e12
            
            # MFU (Model FLOPs Utilization)
            mfu = get_metric_with_fallback(
                'performance/train_fp_utilization',
                'performance/train_mfu',
                'train/gpu_util_mfu',
            )
            
            # ============================================
            # Calculate total tokens per second
            # ============================================
            tokens_per_sec = get_metric_with_fallback(
                'performance/tokens_per_sec',
                'train/tokens_per_sec',
            )
            
            # If tokens_per_sec not available but tokens_per_sec_per_gpu is, calculate it
            if tokens_per_sec == 0 and tokens_per_sec_per_gpu > 0 and config['total_gpus'] > 0:
                tokens_per_sec = tokens_per_sec_per_gpu * config['total_gpus']
            
            # If tokens_per_sec_per_gpu not available but tokens_per_sec is, calculate it
            if tokens_per_sec_per_gpu == 0 and tokens_per_sec > 0 and config['total_gpus'] > 0:
                tokens_per_sec_per_gpu = tokens_per_sec / config['total_gpus']
            
            # Seq per sec
            seq_per_sec = get_metric_with_fallback(
                'performance/seq_per_sec',
                'train/seq_per_sec',
            )
            
            # Get log directory from config
            log_dir = ''
            slurm_log_dir = ''
            if run_config:
                logger_cfg = run_config.get('logger', {})
                log_dir = logger_cfg.get('log_dir', '')
            
            # Try to find matching SLURM log directory by matching WandB run ID in log content
            try:
                import glob
                
                # Find SLURM log directories
                base_path = os.path.dirname(os.path.abspath(__file__))
                slurm_dirs = glob.glob(os.path.join(base_path, '*-logs'))
                
                # Sort by directory name (job ID) descending to check newest first
                slurm_dirs = sorted(slurm_dirs, key=lambda x: os.path.basename(x), reverse=True)
                
                # WandB run ID is unique - use it for exact matching
                wandb_run_id = run.id
                
                for sdir in slurm_dirs:
                    ray_driver_log = os.path.join(sdir, 'ray-driver.log')
                    if os.path.exists(ray_driver_log):
                        try:
                            with open(ray_driver_log, 'r') as f:
                                # Read file to find wandb run ID
                                # Look for patterns like "runs/zpnsm4js" or "run-20251202_204311-zpnsm4js"
                                content = f.read()
                                if f"runs/{wandb_run_id}" in content or f"-{wandb_run_id}" in content:
                                    slurm_log_dir = sdir
                                    break
                        except Exception:
                            continue
            except Exception:
                pass  # Silently ignore if we can't find SLURM logs
            
            # Determine actual status by checking SLURM log directory
            actual_status = run.state
            slurm_job_id = ''
            
            def check_log_for_success(log_path: str) -> str:
                """Check ray-driver.log for success/failure indicators.
                
                Success indicators (highest priority):
                - "Max number of steps has been reached"
                - "Training Results:"
                - "Training Model Floating Point Utilization"
                
                Failure indicators:
                - "Traceback" at the end of file
                - "Error:" or "Exception:" without success indicators
                """
                try:
                    with open(log_path, 'r') as f:
                        # Read last 10KB to check for completion messages
                        f.seek(0, 2)  # Go to end
                        file_size = f.tell()
                        f.seek(max(0, file_size - 10000))
                        last_content = f.read()
                        
                        # Success indicators - if any of these exist, it's finished successfully
                        success_indicators = [
                            'Max number of steps has been reached',
                            'Training Results:',
                            'Training Model Floating Point Utilization',
                            'stopping training early',
                            'Training completed',
                        ]
                        
                        has_success = any(indicator in last_content for indicator in success_indicators)
                        
                        if has_success:
                            return 'finished'
                        
                        # Check for errors only if no success indicators
                        # Look for Traceback at the very end (last 2KB)
                        very_last = last_content[-2000:] if len(last_content) > 2000 else last_content
                        
                        error_indicators = [
                            'Traceback (most recent call last)',
                            'Error:',
                            'FAILED',
                            'Exception:',
                        ]
                        
                        has_error = any(indicator in very_last for indicator in error_indicators)
                        
                        if has_error:
                            return 'crashed'
                        
                        # No clear indicator - check if file was recently modified
                        import time
                        mtime = os.path.getmtime(log_path)
                        age_seconds = time.time() - mtime
                        
                        if age_seconds < 300:  # Modified in last 5 minutes
                            return 'running'
                        else:
                            return 'unknown'
                            
                except Exception:
                    return 'unknown'
            
            if slurm_log_dir:
                # Extract SLURM job ID from directory name (e.g., "1138121-logs" -> "1138121")
                dir_name = os.path.basename(slurm_log_dir)
                if '-logs' in dir_name:
                    slurm_job_id = dir_name.replace('-logs', '')
                
                ray_driver_log = os.path.join(slurm_log_dir, 'ray-driver.log')
                
                if os.path.exists(ray_driver_log):
                    # Check log content for success/failure
                    log_status = check_log_for_success(ray_driver_log)
                    
                    if log_status in ['finished', 'crashed']:
                        actual_status = log_status
                    elif run.state == 'running':
                        # WandB says running - verify with SLURM
                        if slurm_job_id:
                            try:
                                import subprocess
                                result_check = subprocess.run(
                                    ['squeue', '-j', slurm_job_id, '-h', '-o', '%T'],
                                    capture_output=True, text=True, timeout=5
                                )
                                slurm_state = result_check.stdout.strip()
                                if slurm_state == 'RUNNING':
                                    actual_status = 'running'
                                elif slurm_state == 'PENDING':
                                    actual_status = 'pending'
                                elif slurm_state == '':
                                    # Job not in queue - use log status or check sacct
                                    if log_status != 'unknown':
                                        actual_status = log_status
                                    else:
                                        result_sacct = subprocess.run(
                                            ['sacct', '-j', slurm_job_id, '-n', '-o', 'State', '--parsable2'],
                                            capture_output=True, text=True, timeout=5
                                        )
                                        sacct_state = result_sacct.stdout.strip().split('\n')[0] if result_sacct.stdout else ''
                                        if 'COMPLETED' in sacct_state:
                                            actual_status = 'finished'
                                        elif 'FAILED' in sacct_state or 'CANCELLED' in sacct_state or 'TIMEOUT' in sacct_state:
                                            actual_status = 'failed'
                                        else:
                                            actual_status = log_status
                            except Exception:
                                actual_status = log_status
                        else:
                            actual_status = log_status
                else:
                    # No ray-driver.log - job likely crashed early
                    actual_status = 'crashed'
            
            result = {
                'run_id': run.id,
                'run_name': run.name,
                'status': actual_status,
                'wandb_status': run.state,
                'slurm_job_id': slurm_job_id,
                'created_at': run.created_at,
                'is_async': is_async,
                'log_dir': log_dir,
                'slurm_log_dir': slurm_log_dir,
                **config,
                # Primary metrics
                'step_time_sec': step_time,
                'tokens_per_sec': tokens_per_sec,
                'tokens_per_sec_per_gpu': tokens_per_sec_per_gpu,
                'seq_per_sec': seq_per_sec,
                'gpu_util_mfu': mfu,
                'per_gpu_tflops': tflops,
                # Detailed timing
                'generation_time': generation_time,
                'policy_training_time': policy_training_time,
                'logprobs_time': logprobs_time,
                'data_processing_time': data_processing_time,
                'reward_calculation_time': reward_calculation_time,
                'training_prep_time': training_prep_time,
                'weight_transfer_time': weight_transfer_time,
                # Detailed throughput
                'gen_tokens_per_sec_per_gpu': gen_tokens_per_sec_per_gpu,
                'training_tokens_per_sec_per_gpu': training_tokens_per_sec_per_gpu,
                'policy_training_tokens_per_sec_per_gpu': policy_training_tokens_per_sec_per_gpu,
                'logprobs_tokens_per_sec_per_gpu': logprobs_tokens_per_sec_per_gpu,
                # Training metrics
                'train_loss': get_metric_with_fallback('train/loss', 'loss'),
                'train_accuracy': get_metric_with_fallback('train/accuracy', 'accuracy'),
            }
            
            results.append(result)
        except Exception as e:
            print(f"⚠️  Error processing run {run.name}: {e}")
            continue
    
    return results


def parse_slurm_log(log_file: Path) -> dict:
    """Parse a SLURM output log for metrics."""
    metrics = {}
    
    try:
        content = log_file.read_text()
        
        # Look for common metric patterns
        patterns = {
            'step_time': r'step[_\s]time[:\s]+([0-9.]+)',
            'tokens_per_sec': r'tokens[/_]per[/_]sec[:\s]+([0-9.]+)',
            'mfu': r'mfu[:\s]+([0-9.]+)',
            'tflops': r'tflops[:\s]+([0-9.]+)',
        }
        
        for key, pattern in patterns.items():
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                metrics[key] = float(match.group(1))
        
        # Check for errors
        if 'error' in content.lower() or 'exception' in content.lower():
            metrics['has_errors'] = True
            
    except Exception as e:
        metrics['parse_error'] = str(e)
    
    return metrics


def print_results_table(results: list, show_all: bool = False, detailed: bool = False):
    """Print results as a formatted table, grouped by model."""
    if not results:
        print("\n❌ No results found.")
        return
    
    # ANSI color codes
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    BLUE = '\033[94m'
    MAGENTA = '\033[95m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    DIM = '\033[2m'
    
    # Filter by status if needed
    if not show_all:
        results = [r for r in results if r.get('status') in ['finished', 'running', 'COMPLETED', 'RUNNING']]
    
    if not results:
        print("\n⚠️  No completed/running results found. Use --all to see all runs.")
        return
    
    # Group by model
    from collections import defaultdict
    model_groups = defaultdict(list)
    for r in results:
        model = r.get('model', 'Unknown')
        model_groups[model].append(r)
    
    # Sort each group by num_nodes ascending (small to large)
    for model in model_groups:
        model_groups[model] = sorted(
            model_groups[model], 
            key=lambda x: (x.get('num_nodes', 0), x.get('total_gpus', 0))
        )
    
    # Sort models alphabetically for consistent ordering
    sorted_models = sorted(model_groups.keys())
    
    # Print header
    print(f"\n{BOLD}{'='*200}{RESET}")
    print(f"{BOLD}GRPO Benchmark Results - GB200 (Grouped by Model){RESET}")
    print(f"{'='*200}")
    
    if detailed:
        # Detailed view with timing breakdown
        header = (
            f"{'Model':<15} "
            f"{'Type':<5} "
            f"{'N':>3} {'GPU':>4} "
            f"{'T.TP':>4} {'T.PP':>4} {'T.EP':>4} "
            f"{'G.TP':>4} "
            f"{'Step':>7} "
            f"{'Gen':>7} "
            f"{'Train':>7} "
            f"{'Logprb':>7} "
            f"{'Tok/s/GPU':>10} "
            f"{'GenTok/GPU':>10} "
            f"{'MFU%':>6} "
            f"{'Status':<10}"
        )
    else:
        # Standard view
        header = (
            f"{'Model':<15} "
            f"{'Type':<5} "
            f"{'Nodes':>5} {'GPUs':>5} "
            f"{'T.TP':>4} {'T.PP':>4} {'T.EP':>4} {'T.CP':>4} {'T.DP':>4} "
            f"{'G.TP':>4} {'G.PP':>4} {'G.DP':>4} "
            f"{'Step(s)':>8} "
            f"{'Tok/s/GPU':>10} "
            f"{'MFU%':>6} "
            f"{'TFLOPs':>7} "
            f"{'Status':<10}"
        )
    
    # Model color palette for visual distinction
    model_colors = [MAGENTA, CYAN, GREEN, YELLOW, BLUE]
    
    total_runs = 0
    all_completed = []
    
    # Print each model group
    for model_idx, model in enumerate(sorted_models):
        model_results = model_groups[model]
        model_color = model_colors[model_idx % len(model_colors)]
        
        # Print model group header
        print(f"\n{model_color}{BOLD}┌─ {model} ({len(model_results)} runs) {'─' * (180 - len(model) - 15)}┐{RESET}")
        print(f"{DIM}{header}{RESET}")
        print(f"{DIM}{'-' * 200}{RESET}")
        
        for r in model_results:
            model_name = r.get('model', 'Unknown')[:13]
            status = r.get('status', 'Unknown')
            is_async = r.get('is_async', False)
            
            # Color status based on actual state
            if status in ['finished', 'COMPLETED']:
                status_str = f"{GREEN}{'finished':<10}{RESET}"
                all_completed.append(r)
            elif status in ['running', 'RUNNING']:
                status_str = f"{YELLOW}{'running':<10}{RESET}"
            elif status in ['pending', 'PENDING']:
                status_str = f"{CYAN}{'pending':<10}{RESET}"
            elif status in ['crashed', 'failed', 'FAILED', 'CANCELLED', 'TIMEOUT']:
                status_str = f"{RED}{status:<10}{RESET}"
            else:
                status_str = f"{DIM}{status:<10}{RESET}"
            
            # Type indicator
            type_str = f"{BLUE}async{RESET}" if is_async else "sync "
            
            # Format metrics
            step_time = r.get('step_time_sec', 0)
            tokens_sec_gpu = r.get('tokens_per_sec_per_gpu', 0)
            mfu = r.get('gpu_util_mfu', 0)
            tflops = r.get('per_gpu_tflops', 0)
            
            # MFU formatting (handle both 0-1 and 0-100 scales)
            mfu_pct = mfu * 100 if mfu < 1 else mfu
            
            # Get log_dir for reference
            log_dir = r.get('log_dir', '')
            
            if detailed:
                gen_time = r.get('generation_time', 0)
                train_time = r.get('policy_training_time', 0)
                logprobs_time = r.get('logprobs_time', 0)
                gen_tok_gpu = r.get('gen_tokens_per_sec_per_gpu', 0)
                
                row = (
                    f"{model_name:<15} "
                    f"{type_str:<5} "
                    f"{r.get('num_nodes', 0):>3} "
                    f"{r.get('total_gpus', 0):>4} "
                    f"{r.get('t_tp', 1):>4} "
                    f"{r.get('t_pp', 1):>4} "
                    f"{r.get('t_ep', 1):>4} "
                    f"{r.get('g_tp', 1):>4} "
                    f"{step_time:>7.1f} "
                    f"{gen_time:>7.1f} "
                    f"{train_time:>7.1f} "
                    f"{logprobs_time:>7.1f} "
                    f"{tokens_sec_gpu:>10,.0f} "
                    f"{gen_tok_gpu:>10,.0f} "
                    f"{mfu_pct:>6.2f} "
                    f"{status_str}"
                )
            else:
                row = (
                    f"{model_name:<15} "
                    f"{type_str:<5} "
                    f"{r.get('num_nodes', 0):>5} "
                    f"{r.get('total_gpus', 0):>5} "
                    f"{r.get('t_tp', 1):>4} "
                    f"{r.get('t_pp', 1):>4} "
                    f"{r.get('t_ep', 1):>4} "
                    f"{r.get('t_cp', 1):>4} "
                    f"{r.get('t_dp', 1):>4} "
                    f"{r.get('g_tp', 1):>4} "
                    f"{r.get('g_pp', 1):>4} "
                    f"{r.get('g_dp', 1):>4} "
                    f"{step_time:>8.2f} "
                    f"{tokens_sec_gpu:>10,.0f} "
                    f"{mfu_pct:>6.2f} "
                    f"{tflops:>7.1f} "
                    f"{status_str}"
                )
            print(row)
            # Print log directory on next line for easy reference
            slurm_log_dir = r.get('slurm_log_dir', '')
            if slurm_log_dir:
                # Show SLURM ray-driver.log path
                ray_driver_path = os.path.join(slurm_log_dir, 'ray-driver.log')
                print(f"{DIM}    └─ {ray_driver_path}{RESET}")
            elif log_dir:
                # Fallback to WandB log dir
                print(f"{DIM}    └─ WandB: {log_dir}{RESET}")
            total_runs += 1
        
        # Print model group summary
        model_completed = [r for r in model_results if r.get('status') in ['finished', 'COMPLETED']]
        if model_completed:
            best_tok = max(r.get('tokens_per_sec_per_gpu', 0) for r in model_completed)
            avg_tok = sum(r.get('tokens_per_sec_per_gpu', 0) for r in model_completed) / len(model_completed)
            print(f"{model_color}{BOLD}└─ Best: {best_tok:,.0f} tok/s/GPU | Avg: {avg_tok:,.0f} tok/s/GPU {'─' * 140}┘{RESET}")
    
    # Overall summary
    print(f"\n{'='*200}")
    print(f"\n📊 {BOLD}Overall Summary{RESET}")
    print(f"   Total runs: {total_runs}")
    print(f"   Models: {len(sorted_models)}")
    
    if all_completed:
        avg_tokens_gpu = sum(r.get('tokens_per_sec_per_gpu', 0) for r in all_completed) / len(all_completed)
        max_tokens_gpu = max(r.get('tokens_per_sec_per_gpu', 0) for r in all_completed)
        best_run = max(all_completed, key=lambda x: x.get('tokens_per_sec_per_gpu', 0))
        avg_mfu = sum(r.get('gpu_util_mfu', 0) for r in all_completed) / len(all_completed)
        avg_mfu_pct = avg_mfu * 100 if avg_mfu < 1 else avg_mfu
        
        print(f"   Completed runs: {len(all_completed)}")
        print(f"   {GREEN}Best Tokens/sec/GPU: {max_tokens_gpu:,.0f}{RESET} ({best_run.get('model', 'Unknown')})")
        print(f"   Avg Tokens/sec/GPU: {avg_tokens_gpu:,.0f}")
        print(f"   Avg MFU: {avg_mfu_pct:.2f}%")


def save_csv(results: list, output_path: str):
    """Save results to CSV."""
    if not results:
        print("❌ No results to save.")
        return
    
    columns = [
        'run_name', 'model', 'is_async', 'num_nodes', 'gpus_per_node', 'total_gpus',
        # Training parallelism
        't_tp', 't_pp', 't_ep', 't_cp', 't_vpp', 't_dp',
        # Generation parallelism
        'g_tp', 'g_pp', 'g_ep', 'g_dp',
        # Primary metrics
        'step_time_sec', 'tokens_per_sec', 'tokens_per_sec_per_gpu',
        'seq_per_sec', 'gpu_util_mfu', 'per_gpu_tflops',
        # Detailed timing (from get_wandb_log_for_nemorl.py)
        'generation_time', 'policy_training_time', 'logprobs_time',
        'data_processing_time', 'reward_calculation_time', 
        'training_prep_time', 'weight_transfer_time',
        # Detailed throughput
        'gen_tokens_per_sec_per_gpu', 'training_tokens_per_sec_per_gpu',
        'policy_training_tokens_per_sec_per_gpu', 'logprobs_tokens_per_sec_per_gpu',
        # Training metrics
        'train_loss', 'train_accuracy', 
        'status', 'created_at'
    ]
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(results)
    
    print(f"✅ Results saved to {output_path}")


def save_json(results: list, output_path: str):
    """Save results to JSON."""
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"✅ Results saved to {output_path}")


def print_slurm_summary(jobs: list):
    """Print summary of SLURM jobs."""
    if not jobs:
        return
    
    # Filter for our benchmark jobs
    benchmark_jobs = [j for j in jobs if any(
        prefix in j['job_name'].lower() for prefix in 
        ['qwen', 'llama', 'deepseek', 'grpo']
    )]
    
    if not benchmark_jobs:
        print("\n⚠️  No benchmark jobs found in SLURM history.")
        return
    
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'
    
    print(f"\n{BOLD}{'='*120}{RESET}")
    print(f"{BOLD}SLURM Job Summary (Last 7 Days){RESET}")
    print(f"{'='*120}")
    
    header = f"{'JobID':<12} {'JobName':<50} {'State':<12} {'Elapsed':<12} {'Nodes':>6}"
    print(f"{CYAN}{header}{RESET}")
    print("-" * 120)
    
    for job in benchmark_jobs[:20]:  # Show last 20
        state = job['state']
        if state == 'COMPLETED':
            state_str = f"{GREEN}{state:<12}{RESET}"
        elif state == 'RUNNING':
            state_str = f"{YELLOW}{state:<12}{RESET}"
        else:
            state_str = f"{RED}{state:<12}{RESET}"
        
        print(f"{job['job_id']:<12} {job['job_name'][:48]:<50} {state_str} {job['elapsed']:<12} {job['nodes']:>6}")
    
    print("=" * 120)
    
    # Stats
    completed = len([j for j in benchmark_jobs if j['state'] == 'COMPLETED'])
    running = len([j for j in benchmark_jobs if j['state'] == 'RUNNING'])
    failed = len([j for j in benchmark_jobs if j['state'] not in ['COMPLETED', 'RUNNING', 'PENDING']])
    
    print(f"\n📈 Summary: {completed} completed, {running} running, {failed} failed/cancelled")


def debug_wandb_run(project: str, entity: Optional[str] = None, run_id: Optional[str] = None):
    """Debug: Print raw WandB data for inspection."""
    try:
        import wandb
    except ImportError:
        print("⚠️  wandb not installed")
        return
    
    api = wandb.Api()
    
    try:
        if entity:
            runs = api.runs(f"{entity}/{project}")
        else:
            runs = api.runs(project)
    except Exception as e:
        print(f"⚠️  Failed to fetch runs: {e}")
        return
    
    for i, run in enumerate(runs):
        if run_id and run.id != run_id:
            continue
        if i >= 3 and not run_id:  # Only show first 3 unless specific run requested
            print(f"\n... and {len(list(runs)) - 3} more runs. Use --debug-run RUN_ID to inspect a specific run.")
            break
        
        print(f"\n{'='*80}")
        print(f"Run: {run.name} (ID: {run.id})")
        print(f"State: {run.state}")
        print(f"{'='*80}")
        
        print("\n📋 Config keys:")
        config = run.config
        for key in sorted(config.keys())[:20]:
            val = config[key]
            if isinstance(val, dict):
                print(f"  {key}: {{...}}")
                for k2, v2 in list(val.items())[:5]:
                    print(f"    {k2}: {v2}")
            else:
                print(f"  {key}: {val}")
        
        print("\n📊 Summary keys (metrics):")
        summary = run.summary._json_dict
        for key in sorted(summary.keys())[:30]:
            val = summary[key]
            if not key.startswith('_'):
                print(f"  {key}: {val}")
        
        if run_id:
            break


def main():
    parser = argparse.ArgumentParser(description="Collect GRPO benchmark results")
    parser.add_argument("--project", default="sync-grpo-gb200-benchmark", 
                        help="WandB project name")
    parser.add_argument("--entity", default=None, help="WandB entity (team/user)")
    parser.add_argument("--slurm", action="store_true", help="Show SLURM job summary")
    parser.add_argument("--all", "-a", action="store_true", 
                        help="Show all runs including failed")
    parser.add_argument("--output", "-o", help="Output CSV file")
    parser.add_argument("--json", help="Output JSON file")
    parser.add_argument("--no-wandb", action="store_true", 
                        help="Skip WandB fetch (only show SLURM)")
    parser.add_argument("--detailed", "-d", action="store_true",
                        help="Show detailed timing breakdown (generation, training, logprobs)")
    parser.add_argument("--debug", action="store_true",
                        help="Debug mode: show raw WandB data structure")
    parser.add_argument("--debug-run", help="Debug specific run by ID")
    
    args = parser.parse_args()
    
    # Debug mode
    if args.debug or args.debug_run:
        debug_wandb_run(args.project, args.entity, args.debug_run)
        return
    
    results = []
    
    # Show SLURM summary
    if args.slurm or args.no_wandb:
        print("📋 Fetching SLURM job history...")
        jobs = get_slurm_jobs()
        print_slurm_summary(jobs)
    
    # Fetch from WandB
    if not args.no_wandb:
        print(f"\n📊 Fetching results from WandB project: {args.project}")
        wandb_results = fetch_wandb_results(args.project, args.entity)
        results.extend(wandb_results)
        print(f"   Found {len(wandb_results)} runs")
    
    # Print table
    if results:
        print_results_table(results, show_all=args.all, detailed=args.detailed)
    
    # Save if requested
    if args.output:
        save_csv(results, args.output)
    
    if args.json:
        save_json(results, args.json)


if __name__ == "__main__":
    main()

