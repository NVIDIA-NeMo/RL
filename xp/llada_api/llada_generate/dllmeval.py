#!/usr/bin/env python3
"""
Nemotron generation algorithm using the model's built-in generate method.
"""

import json
import os
import sys
import logging
import inspect
from typing import Tuple
import torch
from transformers import AutoModel, PreTrainedModel

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

from dllm_eval.MB_static_block.chat_utils_MB import generate_static_block_size
#from dllm_eval.MB_dynamic_block.chat_utils_MB import generate_dynamic_block_size
#from dllm_eval.SB.chat_utilsInf_SB import generate_dynamic_block_size as generate_dynamic_block_size_sb
from dllm_eval.SB.chat_utilsInf_SB import generate_static_block_size as generate_static_block_size_sb

from .base import GenerationAlgorithm

logger = logging.getLogger(__name__)


def select_dtype_and_tf32() -> torch.dtype:
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        try:
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cuda.matmul.fp32_precision = "ieee"
            torch.backends.cudnn.conv.fp32_precision = "fp32"
        except Exception:
            pass
        return torch.bfloat16
    return torch.float32


class DLLMEval(GenerationAlgorithm):
    """DLLM-Eval by Nicolai Oswald using static block size"""
    
    def __init__(self):
        super().__init__(
            name="dllm_eval",
            description="New experimental static blockwise inference method",
            engine="nemotron"
        )
    
    def load_model_class(self, model_path: str, **kwargs) -> PreTrainedModel:
        """
        Load model class for Nemotron generation.
        Always uses standard AutoModel for Nemotron.
        """
        logger.info("Loading Nemotron model with dllm-eval wrapper class")
        
        batch_size = kwargs.get("batch_size", 1)
        
        if model_path not in sys.path:
            sys.path.insert(0, model_path)
        
        # Import the custom config class
        from configuration_nvrdiff import NVRDiffConfig
        
        # Load config.json for all modes
        config_file = os.path.join(model_path, "config.json")
        target_dtype = select_dtype_and_tf32()
        
        with open(config_file, 'r', encoding='utf_8') as f:
            config_dict = json.load(f)
        config = NVRDiffConfig(**config_dict)
        
        if batch_size == 1:
            print("**** Using dllm_eval.SB.modeling_nvrdiffInf_SB ****", flush=True)
            from dllm_eval.SB.modeling_nvrdiffInf_SB import DiffEncoderModel as SBDiffEncoderModel
            mdl = SBDiffEncoderModel.from_pretrained(
                model_path,
                config=config,
                local_files_only=True,
                torch_dtype=target_dtype,
            )
        else:
            print("**** Using dllm_eval.MB_static_block.modeling_nvrdiff_MB ****", flush=True)
            from dllm_eval.MB_static_block.modeling_nvrdiff_MB import DiffEncoderModel as MBDiffEncoderModel
            mdl = MBDiffEncoderModel.from_pretrained(
                model_path,
                config=config,
                local_files_only=True,
                torch_dtype=target_dtype,
            )
        
        if torch.cuda.is_available():
            mdl = mdl.cuda().to(torch.bfloat16)
            mdl.eval()
            print("Model is on GPU")
            if hasattr(mdl.config, "_attn_implementation"):
                print(f"Using attention implementation: {mdl.config._attn_implementation}")
            elif hasattr(mdl.config, "attn_implementation"):
                print(f"Using attention implementation: {mdl.config.attn_implementation}")
        else:
            mdl = mdl.to(torch.float32)
            print("Model is on CPU")
        
        return mdl
    
    def generate(
        self,
        model: PreTrainedModel,
        prompt: torch.Tensor,
        steps: int,
        gen_length: int,
        block_length: int,
        temperature: float = 1.0,
        remasking: bool = True,
        threshold: float = 0.5,
        factor: float = 4,
        **kwargs
    ) -> Tuple[torch.Tensor, int]:
        """Generate text using Nemotron's native generate method."""
        if not self.is_available():
            raise RuntimeError("Nemotron generation is not available - model does not have native generate method")
        
        if not self._is_nemotron_model(model):
            raise RuntimeError("Model does not appear to be a Nemotron model with native generate method")
        
        # Validate and adjust parameters for Nemotron
        validated_args = self.validate_args(
            steps=steps,
            gen_length=gen_length,
            block_length=block_length,
            temperature=temperature,
            remasking=remasking,
            threshold=threshold,
            factor=factor,
            **kwargs
        )
        
        logger.debug(f"Using Nemotron native generation with args: {validated_args}")
        
        step_size = kwargs.get("step_size", 48)
        sampling_strategy = kwargs.get("sampling_strategy", "confidence_threshold_bound")
        unmasking = kwargs.get("unmasking", "low_confidence")
        prefix_bidir = kwargs.get("prefix_bidir", False)
        distance_bidir = kwargs.get("distance_bidir", None)
        min_prefix_len = kwargs.get("min_prefix_len", 2)
        unified_prefix = kwargs.get("unified_prefix", False)
        use_fused_qkv = kwargs.get("use_fused_qkv", False)
        if hasattr(model.config, "_attn_implementation"):
            attn_kernel = model.config._attn_implementation
        elif hasattr(model.config, "attn_implementation"):
            attn_kernel = model.config.attn_implementation
        else:
            attn_kernel = kwargs.get("attn_kernel", "sdpa")
        if factor is None:
            factor = 1
        
        try:
            if prompt.shape[0] == 1:
                output_ids, nfe, *_ = generate_static_block_size_sb(
                    model=model,
                    prompt=prompt,
                    gen_length=gen_length,
                    steps=steps,
                    step_size=step_size,
                    block_length=block_length,
                    sampling_strategy=sampling_strategy,
                    unmasking=unmasking,
                    threshold=threshold,
                    factor=factor,
                    mask_id=model.mask_token_id,
                    shift_logits=False,
                    neg_entropy=False,
                    dbg_print=False,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            else:
                output_ids, nfe, *_ = generate_static_block_size(
                    model=model,
                    prompt=prompt,
                    gen_length=gen_length,
                    steps=steps,
                    step_size=step_size,
                    block_length=block_length,
                    sampling_strategy=sampling_strategy,
                    unmasking=unmasking,
                    threshold=threshold,
                    factor=factor,
                    mask_id=model.mask_token_id,
                    shift_logits=False,
                    neg_entropy=False,
                    dbg_print=False,
                    eos_token_id=self.tokenizer.eos_token_id,
                    prefix_bidir=prefix_bidir,
                    distance_bidir=distance_bidir,
                    profile=False,
                    profile_detailed=False,
                    min_prefix_len=min_prefix_len,
                    unified_prefix=unified_prefix,
                    attn_kernel=attn_kernel,
                    use_fused_qkv=use_fused_qkv,
                )
            
            return output_ids, nfe
            
        except Exception as e:
            logger.error(f"Nemotron generation failed: {e}")
            raise RuntimeError(f"Nemotron generation failed: {e}")
    
    def is_available(self) -> bool:
        """
        Check if Nemotron generation is available.
        This requires checking if the currently loaded model has a native generate method.
        Since we don't have access to the model here, we return True and let the generate
        method do the actual validation.
        """
        return True
    
    def _is_nemotron_model(self, model: PreTrainedModel) -> bool:
        return True
    
    def get_required_args(self):
        """Get the required arguments for dllm-eval."""
        
        return {
            'steps': 128,
            'gen_length': 1152,
            'block_length': 32,
            'step_size': 32,
            #'temperature': 1.0,
            #'remasking': True,
            'threshold': 0.5,
            'factor': 4,
            'sampling_strategy': "confidence_threshold_bound",
            'unmasking': "low_confidence",
            'prefix_bidir': False,
            'distance_bidir': None,
            'min_prefix_len': 2,
            'unified_prefix': False,
            'attn_kernel': "sdpa",
            'use_fused_qkv': False,
        }


def is_nemotron_model_loaded(model) -> bool:
    """Helper function to check if a Nemotron model is currently loaded."""
    nemotron_gen = DLLMEval()
    return nemotron_gen._is_nemotron_model(model) if model else False
