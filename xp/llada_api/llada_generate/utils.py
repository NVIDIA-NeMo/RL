#!/usr/bin/env python3
"""
Utility classes and functions for LLaDA generation.
"""

import torch
import torch.nn as nn
import logging
from typing import Optional

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
    
    def __getattr__(self, name):
        """Forward attribute access to the wrapped model."""
        # Avoid recursion for our own attributes
        if name in ['model', 'pad_token_id', '_forward_attrs', '_modules', '_parameters', '_buffers']:
            return object.__getattribute__(self, name)
        
        # Forward to wrapped model
        try:
            return getattr(self.model, name)
        except AttributeError:
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
        Generation method with automatic left-padding stripping.
        
        Some models (like dInfer diffusion_llm) use a .generate() method instead of
        forward(). We intercept this as well.
        
        Args:
            prompt: Input prompt tensor (potentially with left-padding)
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments
            
        Returns:
            Generated output
        """
        # Strip left-padding if needed
        stripped_prompt = self.strip_left_padding(prompt)
        
        # Forward to the model's generate method
        return self.model.generate(stripped_prompt, *args, **kwargs)

