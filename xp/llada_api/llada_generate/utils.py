#!/usr/bin/env python3
"""
Utility classes and functions for LLaDA generation.
"""

import torch
import torch.nn as nn
import logging
from typing import Optional, Callable, Tuple, Any

logger = logging.getLogger(__name__)


class LeftPaddingStripWrapper(nn.Module):
    """
    Wrapper that strips left-padding from inputs before forwarding to the model.
    
    This is critical for multi-GPU inference with DataParallel:
    - DataParallel splits batches across GPUs along the batch dimension
    - When num_gpus == batch_size, each GPU gets batch_size=1
    - If sequences are left-padded, single sequences will have padding tokens
    - This wrapper strips those padding tokens AFTER the DataParallel split
    
    The wrapper intercepts the forward() call and strips left-padding from
    input_ids before passing to the underlying model.
    """
    
    def __init__(self, model: nn.Module, pad_token_id: Optional[int] = None):
        """
        Initialize the wrapper.
        
        Args:
            model: The model to wrap
            pad_token_id: The padding token ID to strip (if None, no stripping occurs)
        """
        super().__init__()
        self.model = model
        self.pad_token_id = pad_token_id
        
        # Forward any attribute access to the wrapped model
        # This ensures compatibility with code that accesses model attributes
        self._forward_attrs = True
    
    @property
    def device(self):
        """Forward device property to wrapped model for Fast-dLLM compatibility."""
        return self.model.device if hasattr(self.model, 'device') else next(self.model.parameters()).device
    
    @property  
    def h2e(self):
        """Forward h2e attribute for dInfer BlockWiseDiffusionLLMCont compatibility."""
        return self.model.h2e if hasattr(self.model, 'h2e') else None
    
    @property
    def module(self):
        """
        Forward module property for compatibility with DataParallel checks and unwrapping.
        
        Some code checks isinstance(model, DataParallel) and then accesses model.module.
        This property allows code to access the base model through the wrapper.
        Additionally, unwrap_model() uses this to unwrap the LeftPaddingStripWrapper.
        """
        # Access self.model which is stored in PyTorch's _modules registry
        # We can safely use normal attribute access here since this is a property
        return self.model
    
    def __getattr__(self, name):
        """
        Forward attribute access to the wrapped model.
        
        This ensures that model-specific attributes (like .h2e for dInfer,
        .device for Fast-dLLM, etc.) are accessible through the wrapper.
        
        Note: This is only called when normal attribute lookup fails.
        PyTorch's nn.Module stores submodules in self._modules, so self.model
        is accessed via nn.Module.__getattr__ which looks in _modules.
        """
        # First try PyTorch's default __getattr__ (looks in _modules, _parameters, etc.)
        try:
            return super().__getattr__(name)
        except AttributeError:
            pass
        
        # If PyTorch couldn't find it, try to forward to the wrapped model
        try:
            # Get the wrapped model from PyTorch's _modules registry
            wrapped_model = self._modules.get('model')
            if wrapped_model is not None:
                return getattr(wrapped_model, name)
        except (AttributeError, KeyError):
            pass
        
        # Attribute not found anywhere
        raise AttributeError(f"'{type(self).__name__}' object has no attribute '{name}'")
    
    def strip_left_padding(self, input_ids: torch.Tensor) -> torch.Tensor:
        """
        Strip left-padding from input_ids when batch_size == 1.
        
        Args:
            input_ids: Input tensor of shape (batch_size, seq_len)
            
        Returns:
            Tensor with left-padding removed (if batch_size == 1), else unchanged
        """
        batch_size = input_ids.shape[0]
        
        # Only strip when batch_size == 1
        if batch_size != 1:
            return input_ids
        
        # No padding token defined
        if self.pad_token_id is None:
            return input_ids
        
        # Find first non-padding token
        seq = input_ids[0]
        non_pad_mask = seq != self.pad_token_id
        
        if not non_pad_mask.any():
            # All padding (shouldn't happen)
            logger.warning("Entire sequence is padding tokens")
            return input_ids
        
        # Get index of first non-pad token
        first_non_pad_idx = non_pad_mask.nonzero(as_tuple=True)[0][0].item()
        
        if first_non_pad_idx == 0:
            # No left-padding
            return input_ids
        
        # Strip left-padding
        stripped = input_ids[:, first_non_pad_idx:]
        
        logger.debug(
            f"[GPU {torch.cuda.current_device() if torch.cuda.is_available() else 'CPU'}] "
            f"Stripped {first_non_pad_idx} left-padding tokens: "
            f"{input_ids.shape} -> {stripped.shape}"
        )
        
        return stripped
    
    def forward(self, input_ids: torch.Tensor, *args, **kwargs):
        """
        Forward pass with automatic left-padding stripping.
        
        This method is called by DataParallel on each GPU AFTER the batch is split.
        We strip left-padding here, then forward to the actual model.
        
        Args:
            input_ids: Input IDs (potentially with left-padding)
            *args: Additional positional arguments for the model
            **kwargs: Additional keyword arguments for the model
            
        Returns:
            Model output
        """
        # Strip left-padding if needed
        stripped_input_ids = self.strip_left_padding(input_ids)
        
        # Forward to the actual model
        return self.model(stripped_input_ids, *args, **kwargs)
    
    def generate(self, prompt: torch.Tensor, *args, **kwargs):
        """
        Generation method with automatic left-padding stripping and re-padding.
        
        Some models (like dInfer diffusion_llm) use a .generate() method instead of
        forward(). We intercept this as well.
        
        This method:
        1. Strips left-padding from the prompt before generation
        2. Calls the model's generate method with stripped prompt
        3. Re-adds the padding tokens to the output to match original prompt length
        
        This ensures the output shape matches what the caller expects based on
        the original (padded) prompt length.
        
        Args:
            prompt: Input prompt tensor (potentially with left-padding)
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            Generated output with padding tokens prepended to match original prompt length
        """
        # Track original prompt shape
        original_prompt_length = prompt.shape[1]
        
        # Strip left-padding if needed
        stripped_prompt = self.strip_left_padding(prompt)
        stripped_prompt_length = stripped_prompt.shape[1]
        
        # Calculate how many padding tokens were stripped
        num_padding_tokens = original_prompt_length - stripped_prompt_length
        
        # Forward to the model's generate method
        output = self.model.generate(stripped_prompt, *args, **kwargs)
        
        # Handle tuple return (e.g., Nemotron returns (output_ids, nfe))
        if isinstance(output, tuple):
            output_ids, *extra = output
            is_tuple_output = True
        else:
            output_ids = output
            is_tuple_output = False
        
        # If we stripped padding, we need to re-add it to the output
        if num_padding_tokens > 0:
            # output_ids shape: (batch_size, prompt_length + generated_length)
            # We need to prepend num_padding_tokens padding tokens
            batch_size = output_ids.shape[0]
            
            # Create padding tokens to prepend
            padding = torch.full(
                (batch_size, num_padding_tokens),
                self.pad_token_id,
                dtype=output_ids.dtype,
                device=output_ids.device
            )
            
            # Prepend padding to output
            output_ids = torch.cat([padding, output_ids], dim=1)
            
            logger.debug(
                f"[GPU {torch.cuda.current_device() if torch.cuda.is_available() else 'CPU'}] "
                f"Re-added {num_padding_tokens} padding tokens to output: "
                f"stripped output shape -> padded output shape"
            )
        
        # Return in the same format as received
        if is_tuple_output:
            return (output_ids, *extra)
        else:
            return output_ids


def split_batch_across_gpus(
    model: nn.Module,
    prompt: torch.Tensor,
    generate_fn: Callable,
    **generation_kwargs
) -> Tuple[torch.Tensor, Any]:
    """
    Split a batch across multiple GPUs for generation methods that don't use DataParallel.
    
    DataParallel only parallelizes .forward(), not custom methods like .generate().
    This utility manually splits the batch across GPUs, calls the generation function
    on each split, and merges the results.
    
    Args:
        model: The model (potentially DataParallel-wrapped)
        prompt: Input prompt tensor of shape (batch_size, seq_len)
        generate_fn: Function to call for generation. Should accept (model, prompt, **kwargs)
                    and return (output_ids, metadata) where metadata can be nfe, tuple, etc.
        **generation_kwargs: Additional arguments to pass to generate_fn
        
    Returns:
        Tuple of (merged_output_ids, merged_metadata)
        
    Example:
        output, nfe = split_batch_across_gpus(
            model=model,
            prompt=prompt,
            generate_fn=lambda m, p, **kw: m.generate(p, **kw),
            max_new_tokens=128,
            steps=16
        )
    """
    batch_size = prompt.shape[0]
    is_data_parallel = isinstance(model, torch.nn.DataParallel)
    
    # If batch_size == 1 or not using DataParallel, call directly
    if batch_size == 1 or not is_data_parallel:
        actual_model = model.module if is_data_parallel else model
        return generate_fn(actual_model, prompt, **generation_kwargs)
    
    # Multi-sample with DataParallel: manually split across GPUs
    device_ids = model.device_ids
    num_gpus = len(device_ids)
    
    logger.debug(f"Multi-GPU generation: splitting batch_size={batch_size} across {num_gpus} GPUs")
    
    # Calculate samples per GPU
    samples_per_gpu = [
        batch_size // num_gpus + (1 if i < batch_size % num_gpus else 0) 
        for i in range(num_gpus)
    ]
    
    # Split the prompt tensor
    prompt_splits = torch.split(prompt, samples_per_gpu, dim=0)
    
    # Get the underlying model (LeftPaddingStripWrapper or BaseModel)
    underlying_model = model.module
    
    # Process each split on its corresponding GPU
    outputs = []
    metadata_list = []
    
    for gpu_id, prompt_split in zip(device_ids, prompt_splits):
        # Move prompt to the GPU
        prompt_gpu = prompt_split.to(f'cuda:{gpu_id}')
        
        # Generate on this GPU
        with torch.cuda.device(gpu_id):
            result = generate_fn(underlying_model, prompt_gpu, **generation_kwargs)
            
            # Handle different return types
            if isinstance(result, tuple):
                output_gpu, metadata = result
                outputs.append(output_gpu)
                metadata_list.append(metadata)
            else:
                outputs.append(result)
                metadata_list.append(None)
    
    # Merge outputs back to primary device
    primary_device = f'cuda:{device_ids[0]}'
    merged_output = torch.cat([o.to(primary_device) for o in outputs], dim=0)
    
    # Merge metadata (average for numeric values like NFE)
    if metadata_list and metadata_list[0] is not None:
        if isinstance(metadata_list[0], (int, float)):
            # Average numeric metadata (e.g., NFE)
            merged_metadata = sum(metadata_list) // len(metadata_list)
        elif isinstance(metadata_list[0], tuple):
            # For tuple metadata, average each numeric element
            merged_metadata = tuple(
                sum(m[i] for m in metadata_list) // len(metadata_list) if isinstance(metadata_list[0][i], (int, float)) else metadata_list[0][i]
                for i in range(len(metadata_list[0]))
            )
        else:
            # For other types, just take the first one
            merged_metadata = metadata_list[0]
    else:
        merged_metadata = None
    
    logger.debug(f"Multi-GPU generation complete: output shape={merged_output.shape}")
    
    if merged_metadata is not None:
        return merged_output, merged_metadata
    else:
        return merged_output

