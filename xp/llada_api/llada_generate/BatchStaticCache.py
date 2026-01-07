"""
BatchStaticCache - A StaticCache extension supporting per-batch cache position updates.

This module provides a drop-in replacement for HuggingFace's StaticCache that supports
updating different cache positions for each element in a batch.

Key differences from StaticCache:
- `update()` accepts `cache_position` as a 2D tensor [batch_size, seq_len] 
  where each batch can specify different positions to update
- Falls back to standard behavior when cache_position is 1D
"""

import torch
from typing import Any, Optional
from transformers.cache_utils import StaticCache


class BatchStaticCache(StaticCache):
    """
    Static Cache with per-batch position indexing support.
    
    Drop-in replacement for HuggingFace's StaticCache with enhanced functionality:
    - Supports cache_position as [batch_size, seq_len] for per-batch updates
    - Falls back to standard StaticCache behavior for 1D cache_position
    
    Args:
        config: Model config (PretrainedConfig or similar)
        max_batch_size: Maximum batch size (for pre-allocation)
        max_cache_len: Maximum sequence length to cache
        device: Device to store cache on
        dtype: Data type for cache tensors
    
    Example:
        >>> cache = BatchStaticCache(
        ...     config=model.config,
        ...     max_batch_size=4,
        ...     max_cache_len=2048,
        ...     device='cuda',
        ...     dtype=torch.float16
        ... )
        >>> 
        >>> # Per-batch position update
        >>> batch_positions = torch.tensor([
        ...     [0, 1, 2, 3],  # batch 0 positions
        ...     [5, 6, 7, 8],  # batch 1 positions (different!)
        ... ], device='cuda')
        >>> 
        >>> keys, values = cache.update(
        ...     key_states, value_states, layer_idx=0,
        ...     cache_kwargs={'cache_position': batch_positions}
        ... )
    """
    
    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_kwargs: Optional[dict[str, Any]] = None,
        use_loop: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache for a specific layer.
        
        Args:
            key_states: [batch_size, num_heads, seq_len, head_dim]
            value_states: [batch_size, num_heads, seq_len, head_dim]
            layer_idx: Which layer to update
            cache_kwargs: dict with 'cache_position' key
                - cache_position can be [seq_len] (same for all batches, uses parent behavior)
                - OR [batch_size, seq_len] (different per batch, uses custom logic)
            use_loop: If True, use loop-based update; if False (default), use vectorized scatter
        
        Returns:
            (key_cache, value_cache) for the layer
        """
        cache_position = cache_kwargs.get("cache_position") if cache_kwargs else None
        
        # If no cache_position or 1D cache_position, use standard StaticCache behavior
        if cache_position is None or cache_position.dim() == 1:
            return super().update(key_states, value_states, layer_idx, cache_kwargs)
        
        # 2D cache_position: per-batch position updates
        if use_loop:
            return self._update_per_batch_loop(key_states, value_states, layer_idx, cache_position)
        else:
            return self._update_per_batch_scatter(key_states, value_states, layer_idx, cache_position)
    
    def _update_per_batch_scatter(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_position: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with per-batch position indexing using vectorized scatter (no loops).
        
        Args:
            key_states: [batch_size, num_heads, seq_len, head_dim]
            value_states: [batch_size, num_heads, seq_len, head_dim]
            layer_idx: Which layer to update
            cache_position: [batch_size, seq_len] - per-batch positions
        
        Returns:
            (key_cache, value_cache) for the layer
        """
        batch_size, num_heads, seq_len, head_dim = key_states.shape
        
        # Expand cache_position to match key_states shape for scatter operation
        # [batch_size, seq_len] -> [batch_size, 1, seq_len, 1] -> [batch_size, num_heads, seq_len, head_dim]
        indices = cache_position.unsqueeze(1).unsqueeze(-1).expand(batch_size, num_heads, seq_len, head_dim)
        
        # Access the layer's key/value tensors (StaticCache stores caches in self.layers[layer_idx])
        layer = self.layers[layer_idx]
        
        # Scatter key and value states into cache at the specified positions
        # scatter_(dim=2) writes: cache[b, h, indices[b, h, s, d], d] = states[b, h, s, d]
        layer.keys.scatter_(2, indices, key_states)
        layer.values.scatter_(2, indices, value_states)
        
        return layer.keys, layer.values
    
    def _update_per_batch_loop(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_position: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Update cache with per-batch position indexing using a loop over batches.
        
        Args:
            key_states: [batch_size, num_heads, seq_len, head_dim]
            value_states: [batch_size, num_heads, seq_len, head_dim]
            layer_idx: Which layer to update
            cache_position: [batch_size, seq_len] - per-batch positions
        
        Returns:
            (key_cache, value_cache) for the layer
        """
        batch_size = key_states.shape[0]
        
        # Access the layer's key/value tensors
        layer = self.layers[layer_idx]
        
        # Iterate over each batch element and update at its specific positions
        for b in range(batch_size):
            positions = cache_position[b]  # [seq_len]
            # key_states[b] shape: [num_heads, seq_len, head_dim]
            # layer.keys shape: [batch_size, num_heads, max_cache_len, head_dim]
            layer.keys[b, :, positions, :] = key_states[b]
            layer.values[b, :, positions, :] = value_states[b]
        
        return layer.keys, layer.values
    
    def update_at_positions(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
        cache_position: torch.Tensor,
        use_loop: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convenience method for explicit per-batch position update.
        
        Args:
            key_states: [batch_size, num_heads, seq_len, head_dim]
            value_states: [batch_size, num_heads, seq_len, head_dim]
            layer_idx: Which layer to update
            cache_position: [batch_size, seq_len] - per-batch positions
            use_loop: If True, use loop-based update; if False (default), use vectorized scatter
        
        Returns:
            (key_cache, value_cache) for the layer
        """
        if use_loop:
            return self._update_per_batch_loop(key_states, value_states, layer_idx, cache_position)
        else:
            return self._update_per_batch_scatter(key_states, value_states, layer_idx, cache_position)

