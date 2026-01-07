"""
Fused SwiGLU MLP implementation using Liger-kernel.

Uses Liger's optimized Triton kernel for: silu(gate_proj(x)) * up_proj(x)
This eliminates intermediate memory allocations and reduces memory bandwidth.

For Qwen3/Llama-style models:
  Original: down_proj(silu(gate_proj(x)) * up_proj(x))
  Fused:    down_proj(fused_swiglu(gate_out, up_out))
"""

import torch

# Try to import Liger's fused SwiGLU kernel
try:
    from liger_kernel.ops.swiglu import LigerSiLUMulFunction
    LIGER_SWIGLU_AVAILABLE = True
except ImportError:
    LIGER_SWIGLU_AVAILABLE = False
    LigerSiLUMulFunction = None


def fused_swiglu(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """
    Fused SwiGLU activation: silu(gate) * up
    
    Uses Liger-kernel's optimized Triton implementation when available.
    Falls back to PyTorch when Liger is not available.
    
    Args:
        gate: Output of gate_proj, shape [batch, seq_len, intermediate_size]
        up: Output of up_proj, shape [batch, seq_len, intermediate_size]
        
    Returns:
        Result of silu(gate) * up with same shape
    """
    if LIGER_SWIGLU_AVAILABLE and LigerSiLUMulFunction is not None and gate.is_cuda:
        # Use Liger's fused kernel - more optimized than our custom implementation
        return LigerSiLUMulFunction.apply(gate, up)
    else:
        # Fallback to PyTorch
        return torch.nn.functional.silu(gate) * up


class FusedSwiGLUMLP(torch.nn.Module):
    """
    Fused SwiGLU MLP layer that replaces standard Qwen3/Llama MLP.
    
    Original:
        down_proj(silu(gate_proj(x)) * up_proj(x))
        
    Optimized:
        down_proj(fused_swiglu(gate_proj(x), up_proj(x)))
    
    The fused_swiglu kernel eliminates the intermediate tensor from
    silu(gate) and the multiplication, reducing memory bandwidth.
    """
    
    def __init__(self, gate_proj, up_proj, down_proj, act_fn=None):
        """
        Initialize from existing projection layers.
        
        Args:
            gate_proj: nn.Linear for gate projection
            up_proj: nn.Linear for up projection
            down_proj: nn.Linear for down projection
            act_fn: Ignored (always uses SiLU for SwiGLU)
        """
        super().__init__()
        self.gate_proj = gate_proj
        self.up_proj = up_proj
        self.down_proj = down_proj
        self._use_fused = True
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with fused SwiGLU.
        """
        gate_out = self.gate_proj(x)
        up_out = self.up_proj(x)
        
        if self._use_fused and x.is_cuda and gate_out.is_contiguous() and up_out.is_contiguous():
            # Use fused kernel
            hidden = fused_swiglu(gate_out, up_out)
        else:
            # Fallback to standard implementation
            hidden = torch.nn.functional.silu(gate_out) * up_out
        
        return self.down_proj(hidden)


# Check if fused MLP is available (Liger is preferred)
FUSED_MLP_AVAILABLE = LIGER_SWIGLU_AVAILABLE

def replace_mlp_with_fused(model) -> int:
    """
    Replace all MLP layers in a model with FusedSwiGLUMLP.
    
    Args:
        model: The model to modify (in-place)
        
    Returns:
        Number of MLP layers replaced
    """
    count = 0
    
    for name, module in model.named_modules():
        # Look for MLP modules with gate_proj, up_proj, down_proj
        if hasattr(module, 'gate_proj') and hasattr(module, 'up_proj') and hasattr(module, 'down_proj'):
            if hasattr(module, 'act_fn'):
                # This is likely a Qwen3/Llama-style MLP
                # Get parent module to replace
                parent_name = '.'.join(name.split('.')[:-1])
                child_name = name.split('.')[-1]
                
                if parent_name:
                    parent = model.get_submodule(parent_name)
                else:
                    parent = model
                
                # Create fused MLP
                fused_mlp = FusedSwiGLUMLP(
                    gate_proj=module.gate_proj,
                    up_proj=module.up_proj,
                    down_proj=module.down_proj,
                    act_fn=module.act_fn,
                )
                
                # Replace in parent
                setattr(parent, child_name, fused_mlp)
                count += 1
    
    return count

