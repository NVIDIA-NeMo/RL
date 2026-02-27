# NOTES 
### MegatronTokenizer Issue. 
```
    from megatron.core.datasets.megatron_tokenizer import MegatronTokenizer as MegatronTokenizer
ModuleNotFoundError: No module named 'megatron.core.datasets.megatron_tokenizer'
```
Fix : 
Create this file in megatron/core/datasets/megatron_tokenizer.py
```
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# Backward-compatibility shim. The legacy tokenizer base class was moved out of
# megatron.core.datasets in newer Megatron-LM versions.  Megatron-Bridge still
# imports from this path, so we keep a thin re-export here.

import json
import logging
from abc import ABC, abstractmethod
from collections import OrderedDict
from typing import Any

import numpy

logger = logging.getLogger(__name__)


class MegatronLegacyTokenizer(ABC):
    """Abstract class for tokenizer

    Absent a config or class-specific tracking of which objects are uniquely identifying, we must
    include all key word arguments as unique identifiers

    Args:
        tokenizer_paths (Tuple[str]): All tokenizer source paths or prefixes

        tokenizer_options (Dict[str, Any]): All tokenizer options
    """

    def __init__(self, *tokenizer_paths: str, **tokenizer_options: Any):
        logger.warning(
            "You're using the legacy tokenizer system, which is deprecated "
            "and will be removed in a future release. Please migrate to the new tokenizer system "
            "(`megatron.core.tokenizers.MegatronTokenizer`)."
        )
        self.unique_identifiers = OrderedDict()
        self.unique_identifiers["class"] = type(self).__name__
        self.unique_identifiers["tokenizer_path"] = list(tokenizer_paths)
        for option in tokenizer_options:
            self.unique_identifiers[option] = str(tokenizer_options[option])

        self.unique_description = json.dumps(self.unique_identifiers, indent=4)

        super().__init__()

    @abstractmethod
    def tokenize(self, text: str) -> numpy.ndarray:
        pass

    def detokenize(self, ids: numpy.ndarray) -> str:
        raise NotImplementedError("{} has no method 'detokenize'".format(type(self).__name__))

    def offsets(self, ids: list[int], text: str) -> list[int]:
        raise NotImplementedError("{} has no method 'offsets'".format(type(self).__name__))

    @property
    @abstractmethod
    def vocab(self):
        pass

    @property
    @abstractmethod
    def inv_vocab(self):
        pass

    @property
    @abstractmethod
    def vocab_size(self):
        pass

    @property
    def cls(self):
        raise NotImplementedError("{} has no attribute 'cls'".format(type(self).__name__))

    @property
    def sep(self):
        raise NotImplementedError("{} has no attribute 'sep'".format(type(self).__name__))

    @property
    def pad(self):
        raise NotImplementedError("{} has no attribute 'pad'".format(type(self).__name__))

    @property
    def eod(self):
        raise NotImplementedError("{} has no attribute 'eod'".format(type(self).__name__))

    @property
    def bos(self):
        raise NotImplementedError("{} has no attribute 'bos'".format(type(self).__name__))

    @property
    def eos(self):
        raise NotImplementedError("{} has no attribute 'eos'".format(type(self).__name__))

    @property
    def mask(self):
        raise NotImplementedError("{} has no attribute 'mask'".format(type(self).__name__))


# Older code imported this class under the name ``MegatronTokenizer``.
MegatronTokenizer = MegatronLegacyTokenizer

```


### DDP ISSUE 
```
@@ -369,12 +369,15 @@ class DistributedDataParallel(_BaseDataParallel):
         Skip synchronous param all-gather if `param_sync` is False.
         """
         assert self.use_forward_hook
+        for module, handle in list(self.remove_forward_pre_hook_handles.items()):
+            handle.remove()
+        self.remove_forward_pre_hook_handles.clear()

-        for module in self.module.modules():
-            assert self.remove_forward_pre_hook_handles[module] is not None
-            self.remove_forward_pre_hook_handles[module].remove()
-            del self.remove_forward_pre_hook_handles[module]
-        assert len(self.remove_forward_pre_hook_handles) == 0
 
         # Force synchronize parameters.
         if param_sync:
```

### EXPLANATION
## Why `disable_forward_pre_hook` is Called

### The Context: `use_reference_model`

The `use_reference_model` context manager (lines 1437-1486) temporarily **swaps the model weights** with the reference model weights:

1. **On entry**: Copies the current model's state_dict to CPU, then loads the reference model's state_dict into the model
2. **On exit**: Restores the original model weights

This allows running inference with the reference model's weights without having two full models in GPU memory.

### What is the Forward Pre-Hook?

Looking at the DDP code you attached, when **overlap_param_gather** is enabled with distributed optimizer:

```376:386:/lustre/fsw/portfolios/coreai/users/shanmugamr/RL/3rdparty/Megatron-LM-workspace/Megatron-LM/megatron/core/distributed/distributed_data_parallel.py
    def enable_forward_pre_hook(self):
        """
        Enable forward pre-hooks needed for param all-gather overlap with forward compute.
        """
        assert self.use_forward_hook
        assert len(self.remove_forward_pre_hook_handles) == 0
        # Register forward pre-hook for all sub-modules.
        for module in self.module.modules():
            self.remove_forward_pre_hook_handles[module] = module.register_forward_pre_hook(
                self._make_forward_pre_hook()
            )
```

The forward pre-hook is used to **overlap parameter all-gather with forward compute**. Here's how it works:

1. With **distributed optimizer**, model parameters are **sharded across data-parallel ranks** (each rank only holds a portion of the parameters)
2. Before forward pass, parameters need to be **all-gathered** to reconstruct full parameters
3. The forward pre-hook intercepts each module's forward call to **wait for the all-gather to complete** for that module's parameters before executing

```411:437:/lustre/fsw/portfolios/coreai/users/shanmugamr/RL/3rdparty/Megatron-LM-workspace/Megatron-LM/megatron/core/distributed/distributed_data_parallel.py
        def hook(module, *unused):
            // ...
            # Make sure all parameters in this module have been all-gathered as necessary.
            for param in module.parameters(recurse=False):
                # Skip parameters without an associated buffer
                if param not in self.param_to_bucket_group:
                    continue
                // ...
                self.param_to_bucket_group[param].finish_param_sync(
                    skip_next_bucket_dispatch=skip_next_bucket_dispatch
                )
```

### Why Disable It During Weight Swap?

When swapping weights in `use_reference_model`:

```1459:1459:/lustre/fsw/portfolios/coreai/users/shanmugamr/RL/nemo_rl/models/policy/workers/megatron_policy_worker.py
                self.model.load_state_dict(self.reference_state_dict, strict=True)
```

**The forward pre-hook would interfere because:**

1. The hook maintains state about **which parameters have been all-gathered** via `param_to_bucket_group`
2. When you swap weights using `load_state_dict`, the parameter tensors are replaced, but the **bucket mappings become stale/invalid**
3. If the hook tries to call `finish_param_sync` on parameters that weren't properly set up for the new weights, it could cause errors or incorrect behavior
4. Additionally, `disable_forward_pre_hook` calls `start_param_sync(force_sync=True)` which **synchronously gathers all parameters** first, ensuring a clean state before the weight swap

After the weight swap is complete (in `finally` block), the hooks are re-enabled:

```1484:1486:/lustre/fsw/portfolios/coreai/users/shanmugamr/RL/nemo_rl/models/policy/workers/megatron_policy_worker.py
                ## re-enable overlap param gather after weight swap
                if self.should_disable_forward_pre_hook:
                    self.enable_forward_pre_hook()
```

### Summary

| Step | What Happens |
|------|-------------|
| 1. `disable_forward_pre_hook()` | Removes all forward hooks, forces a synchronous param all-gather to ensure all params are complete |
| 2. `load_state_dict()` | Safely swap weights without hooks interfering |
| 3. Forward passes | Run without overlap optimization (simpler, safer) |
| 4. `enable_forward_pre_hook()` | Re-register hooks after restoring original weights |

This is essentially a safety measure: disable the async overlap optimization during the weight-swapping operation to avoid race conditions or stale state in the DDP infrastructure.

```python
    def enable_forward_pre_hook(self):
        """
        Enable forward pre-hooks needed for param all-gather overlap with forward compute.
        """
        assert self.use_forward_hook
        assert len(self.remove_forward_pre_hook_handles) == 0
        # Register forward pre-hook for all sub-modules.
        for module in self.module.modules():
            self.remove_forward_pre_hook_handles[module] = module.register_forward_pre_hook(
                self._make_forward_pre_hook()
            )
```

```python
        def hook(module, *unused):
            // ...
            # Make sure all parameters in this module have been all-gathered as necessary.
            for param in module.parameters(recurse=False):
                # Skip parameters without an associated buffer
                if param not in self.param_to_bucket_group:
                    continue
                // ...
                self.param_to_bucket_group[param].finish_param_sync(
                    skip_next_bucket_dispatch=skip_next_bucket_dispatch
                )
```

```python
                self.model.load_state_dict(self.reference_state_dict, strict=True)
```

```python
                ## re-enable overlap param gather after weight swap
                if self.should_disable_forward_pre_hook:
                    self.enable_forward_pre_hook()
```

---

**User**

When I run the application I get an error like this
```
  File "/opt/nemo-rl/examples/run_grpo_math.py", line 260, in <module>
    main()
  File "/opt/nemo-rl/examples/run_grpo_math.py", line 243, in main
    grpo_train(
  File "/opt/nemo-rl/nemo_rl/algorithms/grpo.py", line 1339, in grpo_train
    reference_logprobs = policy.get_reference_policy_logprobs(
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/nemo-rl/nemo_rl/models/policy/lm_policy.py", line 390, in get_reference_policy_logprobs
    self.worker_group.get_all_worker_results(futures)
  File "/opt/nemo-rl/nemo_rl/distributed/worker_groups.py", line 953, in get_all_worker_results
    return future_bundle.get_results(
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/nemo-rl/nemo_rl/distributed/worker_groups.py", line 103, in get_results
    all_results = ray.get(object_refs)
                  ^^^^^^^^^^^^^^^^^^^^
  File "/opt/nemo_rl_venv/lib/python3.12/site-packages/ray/_private/auto_init_hook.py", line 22, in auto_init_wrapper
    return fn(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^
  File "/opt/nemo_rl_venv/lib/python3.12/site-packages/ray/_private/client_mode_hook.py", line 104, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/opt/nemo_rl_venv/lib/python3.12/site-packages/ray/_private/worker.py", line 2882, in get
    values, debugger_breakpoint = worker.get_objects(object_refs, timeout=timeout)
                                  ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/nemo_rl_venv/lib/python3.12/site-packages/ray/_private/worker.py", line 968, in get_objects
    raise value.as_instanceof_cause()
ray.exceptions.RayTaskError(AssertionError): ray::MegatronPolicyWorker.get_reference_policy_logprobs() (pid=710294, ip=10.65.29.29, actor_id=65a88a2a253f3b406b563ea701000000, repr=MegatronPolicyWorker[rank=0])
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/nemo-rl/nemo_rl/utils/nsys.py", line 88, in wrapper
    ret = func(*args, **kwargs)
          ^^^^^^^^^^^^^^^^^^^^^
  File "/opt/nemo-rl/nemo_rl/models/policy/workers/base_policy_worker.py", line 143, in get_reference_policy_logprobs
    with self.use_reference_model():
         ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/root/.local/share/uv/python/cpython-3.12.12-linux-x86_64-gnu/lib/python3.12/contextlib.py", line 137, in __enter__
    return next(self.gen)
           ^^^^^^^^^^^^^^
  File "/opt/nemo-rl/nemo_rl/models/policy/workers/megatron_policy_worker.py", line 1446, in use_reference_model
    self.disable_forward_pre_hook()
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/nemo-rl/nemo_rl/models/policy/workers/megatron_policy_worker.py", line 944, in disable_forward_pre_hook
    self.model.disable_forward_pre_hook(param_sync=param_sync)
  File "/opt/nemo-rl/3rdparty/Megatron-LM-workspace/Megatron-LM/megatron/core/distributed/distributed_data_parallel.py", line 399, in disable_forward_pre_hook
    assert len(self.remove_forward_pre_hook_handles) == 0, "remove_forward_pre_hook_handles is not empty, it is: " + str(self.remove_forward_pre_hook_handles)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
AssertionError: remove_forward_pre_hook_handles is not empty, it is: {CudaGraphManager(): <torch.utils.hooks.RemovableHandle object at 0x152130339fa0>, CudaGraphManager(): <torch.utils.hooks.RemovableHandle object at 0x15213033a150>, CudaGraphManager(): <torch.utils.hooks.RemovableHandle object at 0x15213033acc0>, CudaGraphManager(): <torch.utils.hooks.RemovableHandle object at 0x15213033b890>, CudaGraphManager(): <torch.utils.hooks.RemovableHandle object at 0x152130328440>, CudaGraphManager(): <torch.utils.hooks.RemovableHandle object at 0x152130328fe0>, CudaGraphManager(): <torch.utils.hooks.RemovableHandle object at 0x15213032a210>, CudaGraphManager(): <torch.utils.hooks.RemovableHandle object at 0x15213032b590>, CudaGraphManager(): <torch.utils.hooks.RemovableHandle object at 0x15213031c920>, CudaGraphManager(): <torch.utils.hooks.RemovableHandle object at 0x15213031dc40>, CudaGraphManager(): <torch.utils.hooks.RemovableHandle object at 0x15213031ef60>, CudaGraphManager(): <torch.utils.hooks.RemovableHandle object at 0x1521303042c0>, CudaGraphManager(): <torch.utils.hooks.RemovableHandle object at 0x1521303055e0>, CudaGraphManager(): <torch.utils.hooks.RemovableHandle object at 0x152130306930>, CudaGraphManager(): <torch.utils.hooks.RemovableHandle object at 0x152130307cb0>, CudaGraphManager(): <torch.utils.hooks.RemovableHandle object at 0x152130301010>, CudaGraphManager(): <torch.utils.hooks.RemovableHandle object at 0x152130302390>, CudaGraphManager(): <torch.utils.hooks.RemovableHandle object at 0x152130303740>, CudaGraphManager(): <torch.utils.hooks.RemovableHandle object at 0x1521302f4b30>, CudaGraphManager(): <torch.utils.hooks.RemovableHandle object at 0x1521302f5eb0>, CudaGraphManager(): <torch.utils.hooks.RemovableHandle object at 0x1521302f7260>, CudaGraphManager(): <torch.utils.hooks.RemovableHandle object at 0x1521302e0620>, CudaGraphManager(): <torch.utils.hooks.RemovableHandle object at 0x1521302e19d0>, CudaGraphManager(): <torch.utils.hooks.RemovableHandle object at 0x1521302e2d80>, CudaGraphManager(): <torch.utils.hooks.RemovableHandle object at 0x1521302d4170>, CudaGraphManager(): <torch.utils.hooks.RemovableHandle object at 0x1521302d5520>, CudaGraphManager(): <torch.utils.hooks.RemovableHandle object at 0x1521302d68a0>, CudaGraphManager(): <torch.utils.hooks.RemovableHandle object at 0x1521302d7c20>, CudaGraphManager(): <torch.utils.hooks.RemovableHandle object at 0x1521302c8f50>}
``` 
What does this error mean ? 

---

**Cursor**

```plaintext
AssertionError: remove_forward_pre_hook_handles is not empty, it is: {CudaGraphManager(): ...}
```

```python
assert len(self.remove_forward_pre_hook_handles) == 0, "remove_forward_pre_hook_handles is not empty, it is: " + str(self.remove_forward_pre_hook_handles)
```

```python
def disable_forward_pre_hook(self, param_sync: bool = True):
    """
    Disable forward pre-hooks needed for param all-gather overlap with forward compute.
    Skip synchronous param all-gather if `param_sync` is False.
    """
    assert self.use_forward_hook
    # De-register forward pre-hook for all sub-modules.
    for module in self.module.modules():
        assert self.remove_forward_pre_hook_handles[module] is not None
        self.remove_forward_pre_hook_handles[module].remove()
        del self.remove_forward_pre_hook_handles[module]
    assert len(self.remove_forward_pre_hook_handles) == 0, "remove_forward_pre_hook_handles is not empty, it is: " + str(self.remove_forward_pre_hook_handles)
    ...
```

This error is revealing a **mismatch between hook registration and deregistration** due to `CudaGraphManager` modules. Let me explain:

## What the Error Means

The assertion failure happens in `disable_forward_pre_hook`:

```388:399:/lustre/fsw/portfolios/coreai/users/shanmugamr/RL/3rdparty/Megatron-LM-workspace/Megatron-LM/megatron/core/distributed/distributed_data_parallel.py
    def disable_forward_pre_hook(self, param_sync: bool = True):
        """
        Disable forward pre-hooks needed for param all-gather overlap with forward compute.
        Skip synchronous param all-gather if `param_sync` is False.
        """
        assert self.use_forward_hook
        # De-register forward pre-hook for all sub-modules.
        for module in self.module.modules():
            assert self.remove_forward_pre_hook_handles[module] is not None
            self.remove_forward_pre_hook_handles[module].remove()
            del self.remove_forward_pre_hook_handles[module]
        assert len(self.remove_forward_pre_hook_handles) == 0, "remove_forward_pre_hook_handles is not empty, it is: " + str(self.remove_forward_pre_hook_handles)
```

### The Problem

1. **When `enable_forward_pre_hook` was called**: It iterated through `self.module.modules()` and registered hooks for **ALL** modules, including `CudaGraphManager` objects

2. **When `disable_forward_pre_hook` is called**: It again iterates through `self.module.modules()` to remove hooks

3. **The mismatch**: The `CudaGraphManager` modules that were present during registration are **no longer being returned** by `self.module.modules()` during deregistration. This leaves their handles in the dictionary, causing the assertion to fail.

### Why This Happens

Looking at the leftover keys in the error:
```
{CudaGraphManager(): <torch.utils.hooks.RemovableHandle object at ...>, 
 CudaGraphManager(): <torch.utils.hooks.RemovableHandle object at ...>, 
 ...}
```

These `CudaGraphManager` objects were registered as submodules when `enable_forward_pre_hook` ran, but they're either:
- Dynamically removed from the module tree after registration
- Not yielded by `modules()` during the second iteration for some reason
- Created/destroyed dynamically as part of CUDA graph management

### Root Cause

This is likely a **bug or incompatibility** between:
1. **CUDA Graphs** being enabled in your configuration (`cuda_graph_impl = "local"` as I saw earlier in the code)
2. **Overlap param gather** being enabled (`overlap_param_gather = True`)

When CUDA graphs are used, the model may dynamically add/remove `CudaGraphManager` wrapper modules, which breaks the assumption that `modules()` returns the same set both times.

### Potential Fixes

**Option 1**: Disable CUDA graphs when using reference model weight swapping (if that's acceptable for your use case)

**Option 2**: Fix the `disable_forward_pre_hook` to be more robust by clearing ALL handles, not just those from current `modules()`:

```python
def disable_forward_pre_hook(self, param_sync: bool = True):
    assert self.use_forward_hook
    # Remove all registered handles
    for module, handle in list(self.remove_forward_pre_hook_handles.items()):
        handle.remove()
    self.remove_forward_pre_hook_handles.clear()
    
    # Force synchronize parameters.
    if param_sync:
        self.start_param_sync(force_sync=True)
```

**Option 3**: Skip calling `disable_forward_pre_hook` when reference model doesn't need it (if `init_reference_model=False` was used or reference model was never initialized)

Would you like me to help you implement one of these fixes?

```python
    def disable_forward_pre_hook(self, param_sync: bool = True):
        """
        Disable forward pre-hooks needed for param all-gather overlap with forward compute.
        Skip synchronous param all-gather if `param_sync` is False.
        """
        assert self.use_forward_hook
        # De-register forward pre-hook for all sub-modules.
        for module in self.module.modules():
            assert self.remove_forward_pre_hook_handles[module] is not None
            self.remove_forward_pre_hook_handles[module].remove()
            del self.remove_forward_pre_hook_handles[module]
        assert len(self.remove_forward_pre_hook_handles) == 0, "remove_forward_pre_hook_handles is not empty, it is: " + str(self.remove_forward_pre_hook_handles)
```

```plaintext
{CudaGraphManager(): <torch.utils.hooks.RemovableHandle object at ...>, 
 CudaGraphManager(): <torch.utils.hooks.RemovableHandle object at ...>, 
 ...}
```

```python
def disable_forward_pre_hook(self, param_sync: bool = True):
    assert self.use_forward_hook
    # Remove all registered handles
    for module, handle in list(self.remove_forward_pre_hook_handles.items()):
        handle.remove()
    self.remove_forward_pre_hook_handles.clear()
    
    # Force synchronize parameters.
    if param_sync:
        self.start_param_sync(force_sync=True)
```

```python
def toggle_cuda_graphs(model, set_to="none", reset_cuda_graphs=True):
    ...
    elif attribute == "cudagraph_manager":
        for module in modules:
            if set_to == "local":
                if reset_cuda_graphs:
                    from megatron.core.transformer.cuda_graphs import CudaGraphManager
                    # If we are resetting cuda graphs we create a new cuda graph manager
                    setattr(module[0], attribute, CudaGraphManager(model.config))
                else:
                    # If we are not resetting cuda graphs we set it to its cached cuda graph
                    setattr(module[0], attribute, module[1])
            else:
                for module in modules:
                    # If we are deleting the cuda graph, we delete its attribute
                    if hasattr(module[0], "cudagraph_manager"):
                        delattr(module[0], "cudagraph_manager")
    ...
```

```python
setattr(module[0], attribute, CudaGraphManager(model.config))
```

```python
delattr(module[0], "cudagraph_manager")
```

```python
toggle_cuda_graphs(lang_module, set_to="none", reset_cuda_graphs=True)


```

```plaintext
Two potential solutions emerge:
1. Disable hooks before structural module changes
2. Modify hook removal logic to track references independently of module hierarchy
```

Let me check where `toggle_cuda_graphs` is being called in relation to the hook operations:

**Yes, this is exactly the cause!** Let me explain the problem:

## Root Cause Analysis

### The Sequence of Events

1. **During `generate()`** (line 1875):
   ```python
   toggle_cuda_graphs(lang_module, set_to="local", reset_cuda_graphs=True)
   ```
   This **creates new `CudaGraphManager` objects** and adds them as submodules:
   ```python
   # From toggle_cuda_graphs when set_to="local":
   setattr(module[0], attribute, CudaGraphManager(model.config))
   ```

2. **At some point**, `enable_forward_pre_hook()` is called, which registers hooks for **ALL** current modules, including the newly created `CudaGraphManager` objects:

   ```376:386:/lustre/fsw/portfolios/coreai/users/shanmugamr/RL/3rdparty/Megatron-LM-workspace/Megatron-LM/megatron/core/distributed/distributed_data_parallel.py
       def enable_forward_pre_hook(self):
           # ...
           for module in self.module.modules():
               self.remove_forward_pre_hook_handles[module] = module.register_forward_pre_hook(
                   self._make_forward_pre_hook()
               )
   ```

3. **At end of `generate()`** (line 2045):
   ```python
   toggle_cuda_graphs(lang_module, set_to="none", reset_cuda_graphs=True)
   ```
   This **DELETES the `CudaGraphManager` attributes** from their parent modules:
   ```python
   # From toggle_cuda_graphs when set_to="none":
   if hasattr(module[0], "cudagraph_manager"):
       delattr(module[0], "cudagraph_manager")  # ← REMOVES from module tree!
   ```

4. **When `get_reference_policy_logprobs` → `use_reference_model` → `disable_forward_pre_hook()` is called**:
   It iterates through `self.module.modules()`, but the `CudaGraphManager` objects are **no longer in the module tree** (they were deleted in step 3).

5. **Result**: The hook handles for `CudaGraphManager` objects remain in `remove_forward_pre_hook_handles` but can't be removed → **Assertion fails!**

## Visual Timeline

```
┌─────────────────────────────────────────────────────────────────────┐
│ generate() starts                                                    │
│   └─ toggle_cuda_graphs(set_to="local")                             │
│       └─ Creates CudaGraphManager as submodule                      │
├─────────────────────────────────────────────────────────────────────┤
│ enable_forward_pre_hook() called somewhere                          │
│   └─ Registers hooks for ALL modules (including CudaGraphManager)   │
│   └─ remove_forward_pre_hook_handles[CudaGraphManager] = handle     │
├─────────────────────────────────────────────────────────────────────┤
│ generate() ends                                                      │
│   └─ toggle_cuda_graphs(set_to="none")                              │
│       └─ delattr(module, "cudagraph_manager") ← DELETES IT!         │
├─────────────────────────────────────────────────────────────────────┤
│ get_reference_policy_logprobs()                                      │
│   └─ use_reference_model()                                          │
│       └─ disable_forward_pre_hook()                                 │
│           └─ for module in self.module.modules():  ← CudaGraphManager│
│              # is NOT here anymore!                                  │
│           └─ ASSERTION FAILS: handles dict not empty!               │
└─────────────────────────────────────────────────────────────────────┘
```

## Solution Options

**Option 1**: Fix `disable_forward_pre_hook` in DDP to be robust (best fix):

```python
def disable_forward_pre_hook(self, param_sync: bool = True):
    assert self.use_forward_hook
    # Remove ALL registered handles (not just current modules)
    for module, handle in list(self.remove_forward_pre_hook_handles.items()):
        handle.remove()
    self.remove_forward_pre_hook_handles.clear()
    
    if param_sync:
        self.start_param_sync(force_sync=True)
```

**Option 2**: Call `disable_forward_pre_hook` BEFORE `toggle_cuda_graphs(set_to="none")` in your code.

**Option 3**: Don't delete cuda graph managers, just disable them (modify `toggle_cuda_graphs`).

Would you like me to implement Option 1 as a fix in the Megatron DDP code?

```python
   toggle_cuda_graphs(lang_module, set_to="local", reset_cuda_graphs=True)
```

```python
   # From toggle_cuda_graphs when set_to="local":
   setattr(module[0], attribute, CudaGraphManager(model.config))
```

```python
       def enable_forward_pre_hook(self):
           # ...
           for module in self.module.modules():
               self.remove_forward_pre_hook_handles[module] = module.register_forward_pre_hook(
                   self._make_forward_pre_hook()
               )
```

```python
   toggle_cuda_graphs(lang_module, set_to="none", reset_cuda_graphs=True)
```

```python
   # From toggle_cuda_graphs when set_to="none":
   if hasattr(module[0], "cudagraph_manager"):
       delattr(module[0], "cudagraph_manager")  # ← REMOVES from module tree!
```

```plaintext
┌─────────────────────────────────────────────────────────────────────┐
│ generate() starts                                                    │
│   └─ toggle_cuda_graphs(set_to="local")                             │
│       └─ Creates CudaGraphManager as submodule                      │
├─────────────────────────────────────────────────────────────────────┤
│ enable_forward_pre_hook() called somewhere                          │
│   └─ Registers hooks for ALL modules (including CudaGraphManager)   │
│   └─ remove_forward_pre_hook_handles[CudaGraphManager] = handle     │
├─────────────────────────────────────────────────────────────────────┤
│ generate() ends                                                      │
│   └─ toggle_cuda_graphs(set_to="none")                              │
│       └─ delattr(module, "cudagraph_manager") ← DELETES IT!         │
├─────────────────────────────────────────────────────────────────────┤
│ get_reference_policy_logprobs()                                      │
│   └─ use_reference_model()                                          │
│       └─ disable_forward_pre_hook()                                 │
│           └─ for module in self.module.modules():  ← CudaGraphManager│
│              # is NOT here anymore!                                  │
│           └─ ASSERTION FAILS: handles dict not empty!               │
└─────────────────────────────────────────────────────────────────────┘
```

```python
def disable_forward_pre_hook(self, param_sync: bool = True):
    assert self.use_forward_hook
    # Remove ALL registered handles (not just current modules)
    for module, handle in list(self.remove_forward_pre_hook_handles.items()):
        handle.remove()
    self.remove_forward_pre_hook_handles.clear()
    
    if param_sync:
        self.start_param_sync(force_sync=True)
```

AFTER FIX 1 
### What Changed

| Before | After |
|--------|-------|
| Iterated over `self.module.modules()` | Iterates over `self.remove_forward_pre_hook_handles.items()` directly |
| Asserted each module exists in dict | No assertion needed - we're iterating over what we have |
| Deleted entries one by one | Uses `clear()` after removing all handles |
| Would fail if modules were removed from tree | Handles dynamically removed modules (like `CudaGraphManager`) |

This fix ensures that all registered hook handles are properly removed regardless of whether the modules are still part of the module tree. You can now run your application again to test it.

```python
    def disable_forward_pre_hook(self, param_sync: bool = True):
        """
        Disable forward pre-hooks needed for param all-gather overlap with forward compute.
        Skip synchronous param all-gather if `param_sync` is False.
        """
        assert self.use_forward_hook
        # De-register forward pre-hook for all registered modules.
        # We iterate over the dict keys directly (instead of self.module.modules()) to handle
        # cases where modules may have been dynamically removed from the module tree
        # (e.g., CudaGraphManager objects removed by toggle_cuda_graphs).
        for module, handle in list(self.remove_forward_pre_hook_handles.items()):
            handle.remove()
        self.remove_forward_pre_hook_handles.clear()

        # Force synchronize parameters.
        if param_sync:
            self.start_param_sync(force_sync=True)
```