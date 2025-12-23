# NVIDIA Resiliency Extension (NVRX) Fault Tolerance Integration for NeMo RL

## Overview

This document outlines the design for integrating the NVIDIA Resiliency Extension (`nvidia-resiliency-ext`) fault tolerance (FT) launcher into NeMo RL to enable in-job restart functionality.


### 1. FT Configuration

FT configuration is passed to `ft_launcher` command line or config file:

**Heartbeats API**:
```yaml
# ft_config.yaml
initial_rank_heartbeat_timeout: 3600
rank_heartbeat_timeout: 1800
```

**Sections API**:
```yaml
# ft_config_sections.yaml
rank_section_timeouts:
  init: 3600
  generation: 600
  environment: 300
  training: 600
  checkpointing: 1200
  validation: 3600
rank_out_of_section_timeout: 300
```

### 2. FTClient Wrapper

```python

from typing import Optional, Any
from contextlib import contextmanager
import os

try:
    import nvidia_resiliency_ext.fault_tolerance as ft
    NVRX_AVAILABLE = True
except ImportError:
    NVRX_AVAILABLE = False


class FTClient:
    """Wrapper for NVRX RankMonitorClient.
    """
    
    def __init__(self):
        self._client: Optional[Any] = None
        self._enabled = False
        self._use_sections = False
        
    @property
    def enabled(self) -> bool:
        return self._enabled
        
    def init_workload_monitoring(self) -> None:
        """Initialize FT monitoring. Call at start of training."""
        if not NVRX_AVAILABLE:
            return
            
        self._client = ft.RankMonitorClient()
        self._client.init_workload_monitoring()
        self._enabled = True
    
    def shutdown_workload_monitoring(self) -> None:
        """Shutdown FT monitoring. Call at end of training."""
        if self._client:
            self._client.shutdown_workload_monitoring()
            self._client = None
            self._enabled = False
            
    def send_heartbeat(self) -> None:
        """Send heartbeat signal. Call periodically in training loop."""
        if self._client:
            self._client.send_heartbeat()
            
    def start_section(self, name: str) -> None:
        """Start a named section (for Sections API)."""
        if self._client and self._use_sections:
            self._client.start_section(name)
            
    def end_section(self, name: str) -> None:
        """End a named section (for Sections API)."""
        if self._client and self._use_sections:
            self._client.end_section(name)
            
    @contextmanager
    def section(self, name: str):
        """Context manager for sections."""
        self.start_section(name)
        try:
            yield
        finally:
            self.end_section(name)
            
    def calculate_and_set_timeouts(self) -> None:
        """Auto-calculate timeouts from observed intervals."""
        if self._client:
            if self._use_sections:
                self._client.calculate_and_set_section_timeouts()
            else:
                self._client.calculate_and_set_hb_timeouts()
                
    def state_dict(self) -> dict:
        """Get state dict for checkpointing."""
        if self._client:
            return self._client.state_dict()
        return {}
        
    def load_state_dict(self, state: dict) -> None:
        """Load state dict from checkpoint."""
        if self._client and state:
            self._client.load_state_dict(state)


# Global singleton instance
_ft_client: Optional[FTClient] = None


def get_ft_client() -> FTClient:
    """Get the global FT client instance."""
    global _ft_client
    if _ft_client is None:
        _ft_client = FTClient()
    return _ft_client
```

### 3. Training Loop Integration

```python
# nemo_rl/algorithms/grpo.py

from nemo_rl.utils.fault_tolerance import get_ft_client

def grpo_train(...) -> None:
    """Run GRPO training algorithm."""
    
    # Initialize FT monitoring
    ft_client = get_ft_client()
    ft_client.init_workload_monitoring()
    
    # Load FT state from checkpoint if available
    if grpo_save_state.get("ft_state"):
        ft_client.load_state_dict(grpo_save_state["ft_state"])
    
    try:        
        while current_epoch < max_num_epochs and total_steps < max_num_steps:
            for batch in dataloader:
                    
                # === GENERATION ===
                with ft_client.section("generation"):
                    # ... generation code ...
                
                # === ENVIRONMENT ===
                with ft_client.section("environment"):
                    # ... reward computation ...
                
                # === TRAINING ===
                with ft_client.section("training"):
                    # ... forward/backward pass ...
                
                # Send heartbeat after completing the step
                ft_client.send_heartbeat()
                    
                # After first full epoch, calculate timeouts
                if current_epoch == 1 and current_step == 1:
                    ft_client.calculate_and_set_timeouts()
                    
    finally:
        ft_client.shutdown_workload_monitoring()
```
