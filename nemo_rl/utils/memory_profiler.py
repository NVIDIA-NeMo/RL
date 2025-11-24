# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

import functools
import gc
import os
import sys
import tracemalloc
from contextlib import contextmanager
from typing import Any, Callable, Optional, TypeVar

import torch

# Try to import psutil for faster process-level memory tracking
try:
    import psutil
    HAS_PSUTIL = True
    _PROCESS = psutil.Process(os.getpid())
except ImportError:
    HAS_PSUTIL = False
    _PROCESS = None


F = TypeVar("F", bound=Callable[..., Any])


class MemoryProfiler:
    """CPU RAM memory profiler with support for decorators, context managers, and snapshots.
    
    Tracks process-level memory using tracemalloc and provides variable-level
    inspection using sys.getsizeof(). Supports both automatic (decorator/context)
    and manual (snapshot) profiling modes.
    
    Example usage:
        # Decorator mode
        @profile_memory(profiler, "my_function")
        def my_function():
            pass
        
        # Context manager mode
        with profiler.profile("generation"):
            # code here
            pass
        
        # Manual snapshot mode
        profiler.snapshot("before", locals(), globals())
        # ... operations ...
        profiler.snapshot("after", locals(), globals())
        profiler.compare("before", "after")
    """
    
    def __init__(
        self,
        enabled: bool = True,
        track_top_n_variables: int = 10,
        include_system_objects: bool = False,
        use_psutil: bool = True,  # Use psutil for faster process-level tracking
    ):
        """Initialize the memory profiler.
        
        Args:
            enabled: Whether profiling is active
            track_top_n_variables: Number of top memory consumers to track
            include_system_objects: Whether to include system objects in variable tracking
            use_psutil: If True, use psutil (faster, process-level). If False, use tracemalloc (slower, detailed)
        """
        self.enabled = enabled
        self.track_top_n = track_top_n_variables
        self.include_system_objects = include_system_objects
        self.use_psutil = use_psutil and HAS_PSUTIL  # Only use psutil if available
        self.snapshots: dict[str, dict[str, Any]] = {}
        self.metrics: dict[str, dict[str, float]] = {}
        self._tracemalloc_started = False
        
        if self.enabled:
            if self.use_psutil:
                print(f"  ✓ Memory profiler using psutil (fast, process-level tracking)")
            else:
                print(f"  ✓ Memory profiler using tracemalloc (slower, detailed tracking)")
            self.start()
    
    def start(self) -> None:
        """Start memory tracking."""
        if not self.enabled or self._tracemalloc_started:
            return
        # Only start tracemalloc if not using psutil (psutil doesn't need initialization)
        if not self.use_psutil:
            try:
                tracemalloc.start()
                self._tracemalloc_started = True
            except Exception as e:
                print(f"Warning: Could not start tracemalloc: {e}")
                self._tracemalloc_started = False
    
    def stop(self) -> None:
        """Stop memory tracking."""
        if self._tracemalloc_started:
            tracemalloc.stop()
            self._tracemalloc_started = False
    
    def get_current_memory_mb(self) -> float:
        """Get current process memory usage in MB.
        
        Returns:
            Current memory usage in MB, or 0 if tracking is disabled
        """
        if not self.enabled:
            return 0.0
        try:
            if self.use_psutil and HAS_PSUTIL:
                # Use psutil for fast process-level memory (RSS - Resident Set Size)
                mem_info = _PROCESS.memory_info()
                return mem_info.rss / (1024 * 1024)  # Convert to MB
            elif self._tracemalloc_started:
                # Use tracemalloc for detailed Python object tracking
                current, peak = tracemalloc.get_traced_memory()
                return current / (1024 * 1024)  # Convert to MB
            else:
                return 0.0
        except Exception:
            return 0.0
    
    def get_peak_memory_mb(self) -> float:
        """Get peak process memory usage in MB since last reset.
        
        Returns:
            Peak memory usage in MB, or 0 if tracking is disabled
        """
        if not self.enabled:
            return 0.0
        try:
            if self.use_psutil and HAS_PSUTIL:
                # psutil doesn't track peak, return current as approximation
                mem_info = _PROCESS.memory_info()
                return mem_info.rss / (1024 * 1024)  # Convert to MB
            elif self._tracemalloc_started:
                # tracemalloc tracks peak
                current, peak = tracemalloc.get_traced_memory()
                return peak / (1024 * 1024)  # Convert to MB
            else:
                return 0.0
        except Exception:
            return 0.0
    
    def _get_variable_size_mb(self, obj: Any) -> float:
        """Get approximate size of a variable in MB.
        
        Args:
            obj: Object to measure
            
        Returns:
            Size in MB
        """
        try:
            # Special handling for torch tensors
            if isinstance(obj, torch.Tensor):
                return obj.element_size() * obj.nelement() / (1024 * 1024)
            # For other objects, use sys.getsizeof (approximation)
            return sys.getsizeof(obj) / (1024 * 1024)
        except Exception:
            return 0.0
    
    def _filter_variables(
        self, variables: dict[str, Any]
    ) -> dict[str, Any]:
        """Filter out system objects if include_system_objects is False.
        
        Args:
            variables: Dictionary of variable names to objects
            
        Returns:
            Filtered dictionary
        """
        if self.include_system_objects:
            return variables
        
        # Filter out built-in types and system objects
        filtered = {}
        for name, obj in variables.items():
            # Skip private/protected names
            if name.startswith("_"):
                continue
            # Skip built-in types (except containers that might hold data)
            obj_type = type(obj)
            if obj_type.__module__ == "builtins":
                # Keep lists, dicts, tuples that might hold data
                if obj_type not in (list, dict, tuple, set, frozenset):
                    continue
            filtered[name] = obj
        
        return filtered
    
    def get_top_variables(
        self, variables: dict[str, Any], n: Optional[int] = None
    ) -> list[tuple[str, float, str]]:
        """Get top N memory-consuming variables.
        
        Args:
            variables: Dictionary of variable names to objects
            n: Number of top variables to return (defaults to track_top_n)
            
        Returns:
            List of (name, size_mb, type) tuples sorted by size descending
        """
        if not self.enabled:
            return []
        
        n = n or self.track_top_n
        filtered = self._filter_variables(variables)
        
        # Calculate sizes
        var_sizes = []
        for name, obj in filtered.items():
            size_mb = self._get_variable_size_mb(obj)
            obj_type = type(obj).__name__
            var_sizes.append((name, size_mb, obj_type))
        
        # Sort by size and return top N
        var_sizes.sort(key=lambda x: x[1], reverse=True)
        return var_sizes[:n]
    
    def snapshot(
        self,
        label: str,
        local_vars: Optional[dict[str, Any]] = None,
        global_vars: Optional[dict[str, Any]] = None,
    ) -> None:
        """Take a memory snapshot with optional variable tracking.
        
        Args:
            label: Unique label for this snapshot
            local_vars: Local variables dict (e.g., from locals())
            global_vars: Global variables dict (e.g., from globals())
        """
        if not self.enabled:
            return
        
        snapshot_data = {
            "memory_mb": self.get_current_memory_mb(),
            "peak_memory_mb": self.get_peak_memory_mb(),
        }
        
        # Combine local and global variables
        all_vars = {}
        if global_vars is not None:
            all_vars.update(global_vars)
        if local_vars is not None:
            all_vars.update(local_vars)
        
        if all_vars:
            top_vars = self.get_top_variables(all_vars)
            snapshot_data["top_variables"] = top_vars
        
        self.snapshots[label] = snapshot_data
    
    def compare(self, label1: str, label2: str, print_results: bool = True) -> dict[str, Any]:
        """Compare two snapshots and report differences.
        
        Args:
            label1: First snapshot label
            label2: Second snapshot label
            print_results: Whether to print results to console
            
        Returns:
            Dictionary with comparison metrics
        """
        if not self.enabled:
            return {}
        
        if label1 not in self.snapshots or label2 not in self.snapshots:
            print(f"Warning: Snapshots {label1} or {label2} not found")
            return {}
        
        snap1 = self.snapshots[label1]
        snap2 = self.snapshots[label2]
        
        memory_delta = snap2["memory_mb"] - snap1["memory_mb"]
        
        comparison = {
            "memory_delta_mb": memory_delta,
            f"{label1}_memory_mb": snap1["memory_mb"],
            f"{label2}_memory_mb": snap2["memory_mb"],
        }
        
        if print_results:
            print(f"\n📊 Memory Comparison: {label1} -> {label2}")
            print(f"  Memory delta: {memory_delta:+.2f} MB")
            print(f"  {label1}: {snap1['memory_mb']:.2f} MB")
            print(f"  {label2}: {snap2['memory_mb']:.2f} MB")
            
            # Show top variables if available
            if "top_variables" in snap2:
                print(f"\n  Top {len(snap2['top_variables'])} memory consumers after {label2}:")
                for name, size_mb, obj_type in snap2["top_variables"]:
                    print(f"    • {name} ({obj_type}): {size_mb:.2f} MB")
        
        return comparison
    
    @contextmanager
    def profile(self, section_name: str):
        """Context manager for profiling a code section.
        
        Args:
            section_name: Name of the section being profiled
            
        Example:
            with profiler.profile("generation"):
                # code here
        """
        if not self.enabled:
            yield
            return
        
        # Lightweight memory tracking - directly read memory values
        # without full snapshot/compare overhead
        try:
            if self.use_psutil and HAS_PSUTIL:
                mem_info_before = _PROCESS.memory_info()
                memory_before_mb = mem_info_before.rss / (1024 * 1024)
            elif self._tracemalloc_started:
                current_before, _ = tracemalloc.get_traced_memory()
                memory_before_mb = current_before / (1024 * 1024)
            else:
                memory_before_mb = 0.0
        except Exception:
            memory_before_mb = 0.0
        
        try:
            yield
        finally:
            try:
                if self.use_psutil and HAS_PSUTIL:
                    mem_info_after = _PROCESS.memory_info()
                    memory_after_mb = mem_info_after.rss / (1024 * 1024)
                elif self._tracemalloc_started:
                    current_after, _ = tracemalloc.get_traced_memory()
                    memory_after_mb = current_after / (1024 * 1024)
                else:
                    memory_after_mb = 0.0
                
                memory_delta_mb = memory_after_mb - memory_before_mb
            except Exception:
                memory_after_mb = 0.0
                memory_delta_mb = 0.0
            
            # Store lightweight metrics without full snapshot overhead
            self.metrics[section_name] = {
                "memory_delta_mb": memory_delta_mb,
                "memory_before_mb": memory_before_mb,
                "memory_after_mb": memory_after_mb,
            }
    
    def get_metrics(self, reduction_op: str = "sum") -> dict[str, float]:
        """Get all profiling metrics.
        
        Args:
            reduction_op: How to aggregate metrics (currently only "sum" supported for consistency with Timer)
            
        Returns:
            Dictionary of metric names to values
        """
        if not self.enabled:
            return {}
        
        # Flatten metrics for logging
        flat_metrics = {}
        for section, metrics in self.metrics.items():
            for metric_name, value in metrics.items():
                key = f"{section}/{metric_name}"
                flat_metrics[key] = value
        
        # Add current and peak memory
        flat_metrics["current_memory_mb"] = self.get_current_memory_mb()
        flat_metrics["peak_memory_mb"] = self.get_peak_memory_mb()
        
        return flat_metrics
    
    def print_report(self) -> None:
        """Print a summary report of all profiling metrics."""
        if not self.enabled:
            return
        
        print("\n📊 Memory Profiling Report:")
        print(f"  Current memory: {self.get_current_memory_mb():.2f} MB")
        print(f"  Peak memory: {self.get_peak_memory_mb():.2f} MB")
        
        if self.metrics:
            print("\n  Section breakdowns:")
            for section, metrics in self.metrics.items():
                delta = metrics.get("memory_delta_mb", 0.0)
                print(f"    • {section}: {delta:+.2f} MB")
    
    def reset(self) -> None:
        """Reset all snapshots and metrics."""
        self.snapshots.clear()
        self.metrics.clear()
        if self._tracemalloc_started:
            tracemalloc.clear_traces()


def profile_memory(
    profiler: MemoryProfiler, function_name: Optional[str] = None
) -> Callable[[F], F]:
    """Decorator for profiling memory usage of a function.
    
    Args:
        profiler: MemoryProfiler instance
        function_name: Optional custom name for the function (defaults to actual function name)
        
    Returns:
        Decorated function
        
    Example:
        @profile_memory(memory_profiler, "my_func")
        def my_func():
            pass
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if not profiler.enabled:
                return func(*args, **kwargs)
            
            name = function_name or func.__name__
            with profiler.profile(name):
                return func(*args, **kwargs)
        
        return wrapper  # type: ignore
    
    return decorator

