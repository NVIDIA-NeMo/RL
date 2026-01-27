# Init_reference_model function purpose
_Exported on 1/6/2026 at 17:07:44 PST from Cursor (2.2.20)_

---

**User**

@nemo_rl/models/policy/workers/megatron_policy_worker.py:816 Can you tell me what this init_reference_model does. ? Does it create a whole new model ?

---

**Cursor**

Let me also check where this function is defined and how `ref_checkpoint_config` is passed in:

Based on my analysis of the code, here's what `init_reference_model` does:

## Summary

**Yes, it does create a whole new model, but only temporarily!** The purpose is to load and store the reference model's weights on CPU for later use (typically for KL divergence calculations during RL training).

## How it works

When `init_reference_model=True` (line 816), here's the sequence:

1. **Move the main model to CPU** (line 817) to free up GPU memory:

```817:817:/lustre/fsw/portfolios/coreai/users/shanmugamr/RL/nemo_rl/models/policy/workers/megatron_policy_worker.py
            self.model = self.move_model(self.model, "cpu")
```

2. **Create a brand new model** using `get_model()`:

```842:849:/lustre/fsw/portfolios/coreai/users/shanmugamr/RL/nemo_rl/models/policy/workers/megatron_policy_worker.py
            reference_model = get_model(
                self.megatron_cfg.model,
                self.megatron_cfg.ddp,
                use_torch_fsdp2=self.megatron_cfg.dist.use_torch_fsdp2,
                overlap_param_gather_with_optimizer_step=self.megatron_cfg.optimizer.overlap_param_gather_with_optimizer_step,
                pre_wrap_hook=self.megatron_cfg.rng.data_parallel_random_init,
                mixed_precision_wrapper=ref_mixed_precision_wrapper,
            )
```

3. **Load the pretrained checkpoint** into the reference model (line 855-863)

4. **Extract the state_dict and store it on CPU** - this is the key part:

```866:876:/lustre/fsw/portfolios/coreai/users/shanmugamr/RL/nemo_rl/models/policy/workers/megatron_policy_worker.py
                self.reference_state_dict = {}
                for name, item in reference_model.state_dict().items():
                    if isinstance(item, torch.Tensor):
                        cpu_item = item.detach().to(
                            device="cpu", non_blocking=True, copy=True
                        )
                        del item
                    else:
                        cpu_item = item
                    self.reference_state_dict[name] = cpu_item
                print("Reference model loaded")
```

5. **Move the main model back to GPU** (line 880)

## Why this design?

This is a **memory-efficient approach** for RL training (like GRPO/PPO):

- You need a "reference model" to compute KL divergence between the current policy and the original (frozen) policy
- Instead of keeping **two full models in GPU memory**, this approach:
  - Creates the reference model temporarily
  - Copies its weights to CPU as `self.reference_state_dict`
  - The reference model itself is discarded (garbage collected)
  
The reference weights stored on CPU can then be loaded on-demand when needed for KL calculations, without permanently consuming GPU VRAM for a second model.

```python
            self.model = self.move_model(self.model, "cpu")
```

```python
            reference_model = get_model(
                self.megatron_cfg.model,
                self.megatron_cfg.ddp,
                use_torch_fsdp2=self.megatron_cfg.dist.use_torch_fsdp2,
                overlap_param_gather_with_optimizer_step=self.megatron_cfg.optimizer.overlap_param_gather_with_optimizer_step,
                pre_wrap_hook=self.megatron_cfg.rng.data_parallel_random_init,
                mixed_precision_wrapper=ref_mixed_precision_wrapper,
            )
```

```python
                self.reference_state_dict = {}
                for name, item in reference_model.state_dict().items():
                    if isinstance(item, torch.Tensor):
                        cpu_item = item.detach().to(
                            device="cpu", non_blocking=True, copy=True
                        )
                        del item
                    else:
                        cpu_item = item
                    self.reference_state_dict[name] = cpu_item
                print("Reference model loaded")
```

---

**User**

@nemo_rl/models/policy/workers/megatron_policy_worker.py:1437-1447 Why is this disable_forward_pre_hook called ? What does that do ? Can you explain. i have attached the code disable_forward_pre_hook as well @3rdparty/Megatron-LM-workspace/Megatron-LM/megatron/core/distributed/distributed_data_parallel.py 

---

**Cursor**

Let me read more of the `use_reference_model` context manager to understand the full picture:

Now I have a complete picture. Let me explain what's happening:

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

---

**User**

Yes lets implemnt optoin 1

---

**Cursor**

```python
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import logging
from contextlib import contextmanager
from typing import Optional

import torch

from .. import parallel_state
from ..config_logger import has_config_logger_enabled, log_config_to_disk
from ..fp8_utils import is_float8tensor, post_all_gather_processing
from ..process_groups_config import ProcessGroupCollection
from ..transformer.cuda_graphs import is_graph_capturing
from ..transformer.transformer_config import TransformerConfig
from ..utils import log_single_rank
from .data_parallel_base import _BaseDataParallel
from .distributed_data_parallel_config import DistributedDataParallelConfig
from .param_and_grad_buffer import _ParamAndGradBuffer, partition_buckets

logger = logging.getLogger(__name__)


class DistributedDataParallel(_BaseDataParallel):
    """
    DDP wrapper which stores grads in contiguous buffers. Also has option of overlapping
    communication with backprop computation by breaking up full model's gradients into smaller
    buckets and running all-reduce / reduce-scatter on each bucket asynchronously. This class
    also provides the option to do the gradient accumulation in a type other than the param type
    (e.g., fp32 for a bf16 model).

    Args:
        config: Transformer config object.
        ddp_config: DistributedDataParallel config object.
        module: Underlying model.
        disable_bucketing: If true, force assign all parameters to a single bucket. If false,
            use standard bucketing policy: assign parameters to smaller buckets and all-reduce
            per bucket _if_ overlap_grad_reduce is True and pp_rank is 0.
        pg_collection: Optional unified process group for distributed training.

    """

    def __init__(
        self,
        config: TransformerConfig,
        ddp_config: DistributedDataParallelConfig,
        module: torch.nn.Module,
        disable_bucketing: bool = False,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super().__init__(config=config, module=module)
        if has_config_logger_enabled(config):
            log_config_to_disk(config, locals(), prefix=type(self).__name__)

        # If bucket_size is not provided as an input, use sane default.
        # If using very large dp_sizes, make buckets larger to ensure that chunks used in NCCL
        # ring-reduce implementations are large enough to remain bandwidth-bound rather than
        # latency-bound.
        if ddp_config.bucket_size is None:
            ddp_config.bucket_size = max(
                40000000, 1000000 * parallel_state.get_data_parallel_world_size()
            )
        # Set bucket_size to infinity if overlap_grad_reduce is False.
        if not ddp_config.overlap_grad_reduce:
            ddp_config.bucket_size = None

        self.ddp_config = ddp_config
        log_single_rank(
            logger,
            logging.INFO,
            f'Setting up DistributedDataParallel with config {self.ddp_config}',
        )

        if pg_collection is None:
            self.dp_group = parallel_state.get_data_parallel_group(
                with_context_parallel=False, partial_data_parallel=False
            )
            self.dp_cp_group = parallel_state.get_data_parallel_group(
                with_context_parallel=True, partial_data_parallel=False
            )
            self.intra_dp_cp_group = parallel_state.get_data_parallel_group(
                with_context_parallel=True, partial_data_parallel=True
            )
            self.expt_dp_group = parallel_state.get_expert_data_parallel_group()
            self.intra_expt_dp_group = parallel_state.get_expert_data_parallel_group(
                partial_expert_data_parallel=True
            )
            if self.ddp_config.num_distributed_optimizer_instances > 1:
                self.inter_dist_opt_group = (
                    parallel_state.get_inter_distributed_optimizer_instance_group()
                )
            self.tp_group = parallel_state.get_tensor_model_parallel_group()
            self.pp_group = parallel_state.get_pipeline_model_parallel_group()
            self.ep_group = parallel_state.get_expert_model_parallel_group()
        else:
            # Setup process groups using DDP-specific helper method
            process_groups = ProcessGroupCollection.setup_process_groups_for_ddp(
                pg_collection, config, self.ddp_config
            )

            self.dp_group = process_groups['dp_group']
            self.dp_cp_group = process_groups['dp_cp_group']
            self.intra_dp_cp_group = process_groups['intra_dp_cp_group']
            self.expt_dp_group = process_groups['expt_dp_group']
            self.intra_expt_dp_group = process_groups['intra_expt_dp_group']
            self.tp_group = process_groups['tp_group']
            self.pp_group = process_groups['pp_group']
            self.ep_group = process_groups['ep_group']

            # Set inter_dist_opt_group if multiple optimizer instances
            if self.ddp_config.num_distributed_optimizer_instances > 1:
                self.inter_dist_opt_group = process_groups['inter_dist_opt_group']

        # Turn off bucketing if we are on a pipeline stage that is not the first (since
        # data-parallel communication on these stages is not on the critical path), or if
        # disable_bucketing is True (e.g., we might not want to break up model parameters
        # into buckets for model chunks after the first in the interleaved schedule).
        self.bucket_size = self.ddp_config.bucket_size
        if isinstance(self.pp_group, list):
            pp_rank = self.pp_group[0].rank()
        else:
            pp_rank = self.pp_group.rank()
        if disable_bucketing or pp_rank > 0:
            self.bucket_size = None

        self.param_to_bucket_group = {}

        # Group parameters by their gradient type.
        param_to_name = {}
        dense_params = []
        expert_parallel_params = []
        self.params_with_grad = []
        for name, param in self.module.named_parameters():
            if not param.requires_grad:
                continue

            # Track params with grad to enable direct setting
            # of param.grad_added_to_main_grad
            self.params_with_grad.append(param)

            param.grad_added_to_main_grad = False
            param_to_name[param] = name

            if getattr(param, 'allreduce', True):
                dense_params.append(param)
            else:
                expert_parallel_params.append(param)

        def _allocate_buffers_for_parameters(
            input_params, data_parallel_group, gradient_scaling_factor
        ):
            param_and_grad_dtype_to_params = {}
            param_and_grad_dtype_to_offsets = {}
            param_and_grad_dtype_to_indices = {}

            # Group parameters by their gradient type.
            for param in input_params:
                assert param.requires_grad

                param_dtype = param.dtype
                if is_float8tensor(param):
                    # Currently TE's Float8Tensor is a wrapper of torch.Tensor. It has a "fake"
                    # dtype (usually a higher precision dtype such as bfloat16), but its actual
                    # data is stored in the form of a torch uint8 tensor within the Float8Tensor's
                    # ".data" attribute. Therefore, when creating the param buffer for fp8 params,
                    # it is necessary to use torch.uint8, not the "fake" dtype got from
                    # "param.dtype".
                    param_dtype = torch.uint8
                grad_dtype = torch.float if self.ddp_config.grad_reduce_in_fp32 else param.dtype

                params = param_and_grad_dtype_to_params.get((param_dtype, grad_dtype), [])
                params.append(param)
                param_and_grad_dtype_to_params[(param_dtype, grad_dtype)] = params

                # Get the index of each param among the params with same dtype, if a param is fp8,
                # use its "fake" high precision dtype to find which params have same dtype with it.
                # For example:
                #     Case 1:
                #         params = [p1(bf16), p2(bf16), p3(bf16), p4(bf16)]
                #         param_and_grad_dtype_to_indices = {
                #             (torch.bfloat16, torch.float32): [0, 1, 2, 3],
                #         }
                #     Case 2:
                #         params = [p1(bf16), p2(fp8), p3(fp8), p4(bf16)]
                #         param_and_grad_dtype_to_indices = {
                #             (torch.bfloat16, torch.float32): [0, 3],
                #             (torch.uint8, torch.float32): [1, 2],
                #         }
                # We need these indices to load a non-native-fp8 checkpoint in native-fp8 mode.
                offset = param_and_grad_dtype_to_offsets.get((param.dtype, grad_dtype), 0)
                param_and_grad_dtype_to_offsets[(param.dtype, grad_dtype)] = offset + 1
                indices = param_and_grad_dtype_to_indices.get((param_dtype, grad_dtype), [])
                indices.append(offset)
                param_and_grad_dtype_to_indices[(param_dtype, grad_dtype)] = indices

            if not config.calculate_per_token_loss:
                target_gradient_scaling_factor = 1.0 / self.dp_cp_group.size()
                if self.ddp_config.average_in_collective:
                    if self.ddp_config.num_distributed_optimizer_instances == 1:
                        # Collective is averaging gradients in collective with data_parallel_group.
                        assert (
                            gradient_scaling_factor / data_parallel_group.size()
                            == target_gradient_scaling_factor
                        )
                    else:
                        # For non-expert parameters, gradient_scaling_factor is 1.
                        # For expert parameters, gradient_scaling_factor is edp_size/dp_size.
                        assert (gradient_scaling_factor == 1) or (
                            gradient_scaling_factor
                            == (self.expt_dp_group.size() / self.dp_cp_group.size())
                        )
                else:
                    assert gradient_scaling_factor == target_gradient_scaling_factor

            # Allocate the grad buffers and map the grads.
            buffers = []
            pg_collection = ProcessGroupCollection()
            pg_collection.tp = self.tp_group
            pg_collection.dp_cp = self.dp_cp_group
            for (param_dtype, grad_dtype), params in param_and_grad_dtype_to_params.items():
                buffers.append(
                    _ParamAndGradBuffer(
                        self.ddp_config,
                        param_dtype,
                        grad_dtype,
                        params,
                        data_parallel_group,
                        self.bucket_size,
                        param_to_name,
                        gradient_scaling_factor,
                        param_and_grad_dtype_to_indices[(param_dtype, grad_dtype)],
                        self.ddp_config.nccl_ub,
                        pg_collection,
                    )
                )

            # In some scenarios, we want to put buckets from different buffers into a group so that
            # their communication can be aggregated. For example, when there are both fp8 buffers
            # and bf16 buffers in the model and vpp is enabled, each model chunk will have an fp8
            # bucket and a bf16 bucket, which doubles the number of communication kernels, and
            # because of the use of CUDA_DEVICE_MAX_CONNECTIONS=1, having multiple back-to-back
            # communications will prevent the overlap of the communication kernels with computation
            # kernels.
            # If bucketing is explicitly disabled, then put all buckets in a buffer into a single
            # bucket group.
            bucket_groups = partition_buckets(buffers, force_single_bucket_group=disable_bucketing)

            if self.ddp_config.num_distributed_optimizer_instances > 1:
                assert (
                    self.ddp_config.use_distributed_optimizer
                ), 'Partial DistOpt cannot be used without DistOpt'
                communication_stream = torch.cuda.Stream(device=torch.cuda.current_device())
                for bucket_group in bucket_groups:
                    bucket_group.inter_distributed_optimizer_instance_group = (
                        self.inter_dist_opt_group
                    )
                    bucket_group.communication_stream = communication_stream

            # Set `next_param_gather_bucket_group` for different bucket groups by iterating through
            # buckets in reverse order (since all-gathers happen in reverse order of buckets).
            if self.ddp_config.use_distributed_optimizer and self.ddp_config.overlap_param_gather:
                num_bucket_groups = len(bucket_groups)
                for i in range(1, num_bucket_groups):
                    bucket_groups[num_bucket_groups - i].next_param_gather_bucket_group = (
                        bucket_groups[num_bucket_groups - i - 1]
                    )

            # Create map from param to bucket group, used in pre_hook.
            for bucket_group in bucket_groups:
                for bucket in bucket_group.buckets:
                    for param in bucket.params_list:
                        self.param_to_bucket_group[param] = bucket_group

            return buffers, bucket_groups

        if config.calculate_per_token_loss:
            assert (
                not self.ddp_config.average_in_collective
            ), "Cannot average in collective when calculating per-token loss!"
            gradient_scaling_factor = 1.0
            expert_gradient_scaling_factor = 1.0
        else:
            # The goal is to scale reduced gradients by 1/dp_size.
            # This can be achieved in two ways:
            #
            # Case 1: average_in_collective=True
            # - Non-expert parameters:
            #   1. No pre-scaling (gradient_scaling_factor=1.0)
            #   2. Do average reduction over dp group (equals to sum then divide by dp_size)
            #   3. Final result is scaled by 1/dp_size as desired
            #
            # - Expert parameters:
            #   1. Scale by edp_size/dp_size before reduction
            #   2. Do average reduction over edp group (equals to sum then divide by edp_size)
            #   3. Resulted scaling: (edp_size/dp_size) * (1/edp_size) = 1/dp_size as desired
            #   (edp_size = expert data parallel world size)
            #
            # Case 2: average_in_collective=False
            # - Both expert and non-expert parameters:
            #   1. Scale gradients by 1/dp_size before reduction
            #   2. Do sum reduction across data parallel ranks
            #   3. Final result is scaled by 1/dp_size as desired
            if self.ddp_config.average_in_collective:
                gradient_scaling_factor = 1.0
                expert_gradient_scaling_factor = self.expt_dp_group.size() / self.dp_cp_group.size()
            else:
                data_parallel_world_size = self.dp_cp_group.size()

                gradient_scaling_factor = 1.0 / data_parallel_world_size
                expert_gradient_scaling_factor = 1.0 / data_parallel_world_size

        # Allocate the param+grad buffers for dense params' grads.
        self.buffers, self.bucket_groups = _allocate_buffers_for_parameters(
            dense_params, self.intra_dp_cp_group, gradient_scaling_factor=gradient_scaling_factor
        )

        # Allocate separate param+grad buffers for expert parallel params' grads.
        self.expert_parallel_buffers, self.expert_parallel_bucket_groups = (
            _allocate_buffers_for_parameters(
                expert_parallel_params,
                self.intra_expt_dp_group,
                gradient_scaling_factor=expert_gradient_scaling_factor,
            )
        )

        # Delete references to weight_tensor if they exist since we don't want two parameter copies
        # if we re-mapped parameters (which happens when we use the distributed optimizer).
        # This is a temporary workaround around a TE bug that is fixed with
        # https://github.com/NVIDIA/TransformerEngine/pull/719.
        if self.ddp_config.use_distributed_optimizer:

            @torch.no_grad()
            def unmap_weight_tensor(m):
                if hasattr(m, 'weight_tensor'):
                    m.weight_tensor = None

            self.module.apply(unmap_weight_tensor)

        # Register backward hook.
        # Accumulation function for the gradients need to be stored so they
        # don't go out of scope.
        self.grad_accs = []
        for param in self.module.parameters():
            if param.requires_grad:
                # When delay_wgrad_compute is True and the param is marked with
                # skip_backward_post_hook, register the backward post hook for its module
                # instead of the param so that the wgrad accumulation and reduce will be performed
                # in backward_dw() method of the module instead of the hook of backward() method.
                # Otherwise, register the backward post hook for the param.
                if self.ddp_config.delay_wgrad_compute and getattr(
                    param, 'skip_backward_post_hook', False
                ):
                    for module in self.module.modules():
                        if hasattr(module, "register_wgrad_accumulation_and_reduce_hooks"):
                            for param_value in module.parameters():
                                if param is param_value:
                                    module.register_wgrad_accumulation_and_reduce_hooks(
                                        self._make_backward_post_hook(param)
                                    )
                                    break
                else:
                    # Expand so we get access to grad_fn.
                    param_tmp = param.expand_as(param)
                    # Get the gradient accumulator function.
                    grad_acc = param_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(self._make_backward_post_hook(param))
                    self.grad_accs.append(grad_acc)

        self.use_forward_hook = (
            self.ddp_config.use_distributed_optimizer and self.ddp_config.overlap_param_gather
        )
        self.remove_forward_pre_hook_handles = {}
        if self.use_forward_hook:
            self.enable_forward_pre_hook()
        self.overlap_param_gather_with_optimizer_step = False

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

    def _make_forward_pre_hook(self):
        """
        Create a forward pre-hook to wait on all-gather handles when necessary (i.e.,
        when a module uses a parameter in a bucket with a still incomplete all-gather).
        """

        def hook(module, *unused):
            assert (
                self.use_forward_hook
            ), "Should use pre-hook only when overlap_param_gather is True"

            if is_graph_capturing():
                return

            # Make sure all parameters in this module have been all-gathered as necessary.
            for param in module.parameters(recurse=False):
                # Skip parameters without an associated buffer (such parameters have a
                # .requires_grad field equal to False).
                if param not in self.param_to_bucket_group:
                    continue
                assert param.requires_grad

                # If aligning param all-gather across pipeline stages, all-gather is dispatched
                # by start_param_sync calls in core/pipeline_parallelism/schedules.py.
                # If overlapping param all-gather with optimizer step, then all-gather has
                # already been dispatched in optimizer step.
                skip_next_bucket_dispatch = (
                    self.ddp_config.align_param_gather
                    or self.overlap_param_gather_with_optimizer_step
                )
                self.param_to_bucket_group[param].finish_param_sync(
                    skip_next_bucket_dispatch=skip_next_bucket_dispatch
                )

        return hook

    def _make_backward_post_hook(self, param: torch.nn.Parameter):
        """
        Creates a backward post-hook to dispatch an all-reduce / reduce-scatter when
        ready (i.e., when all grads in a bucket have been computed in all microbatches
        in a batch).
        """

        def hook(*unused):
            if is_graph_capturing():
                return

            if param in self.param_to_bucket_group:
                assert param.requires_grad
                if self.ddp_config.overlap_grad_reduce:
                    assert (
                        param.grad is not None
                    ), 'param.grad being None is not safe when overlap_grad_reduce is True'
                if param.grad is not None and (
                    not param.grad_added_to_main_grad or getattr(param, 'zero_out_wgrad', False)
                ):
                    param.main_grad.add_(param.grad.data)
                param.grad = None

                if self.ddp_config.overlap_grad_reduce:
                    self.param_to_bucket_group[param].register_grad_ready(param)

        return hook

    @contextmanager
    def no_sync(self):
        """
        Context manager that turns off gradient synchronization.
        """
        for bucket_group in self.bucket_groups + self.expert_parallel_bucket_groups:
            bucket_group.is_last_microbatch = False
        try:
            yield
        finally:
            for bucket_group in self.bucket_groups + self.expert_parallel_bucket_groups:
                bucket_group.is_last_microbatch = True

    def start_param_sync(self, *unused, force_sync: bool = False, force_dispatch: bool = False):
        """
        Initiates param sync (all-gather) communication operations for all model parameters.

        By default, when overlap_param_gather is set to True, dispatches asynchronous communication
        calls; when overlap_param_gather is set to False, calls synchronous communication
        ops. Can override this default behavior using flags below.

        Args:
            force_sync (bool, optional): force synchronous collective regardless of
                other settings.
            force_dispatch (bool, optional): force dispatch regardless of other settings.
        """
        if not force_sync:
            # If overlapping param AG with optimizer step, AG should not be dispatched again
            # in forward_backward_step.
            if self.overlap_param_gather_with_optimizer_step and not force_dispatch:
                return

        for bucket_group in self.bucket_groups + self.expert_parallel_bucket_groups:
            bucket_group.start_param_sync(force_sync=force_sync)

            if not self.ddp_config.overlap_param_gather:
                # For MXFP8 params, we need to copy the all-gathered param data from the buffer to
                # the param.data, since param buffer is not mapped to model params for MXFP8 case.
                # The paramaters are cast from bf16 to MXFP8 during copy.
                # In the case of "overlap_param_gather=True", the param copy is done
                # in "finish_param_sync" stage after zeroing the shared gardient buffers.
                if self.ddp_config.reuse_grad_buf_for_mxfp8_param_ag:
                    for bucket in bucket_group.buckets:
                        for param in bucket.params:
                            param_start, param_end = bucket.param_to_index[param]
                            param_slice = bucket.param_data.view(-1)[param_start:param_end]
                            param.data.copy_(param_slice.view(param.data.shape))
                        # All-gathered params are not needed after being copied to param.data.
                        # Zero out the param buffer (shared with grad buffer) for gradient
                        # accumulation. We cannot zero out the entire grad buffer because one grad
                        # buffer may correspond to multiple param buffers. If we zero out the entire
                        # grad buffer, it would clear the data of those param buffers that have not
                        # yet completed AG.
                        bucket.param_data.zero_()
                else:
                    fp8_params = []
                    for bucket in bucket_group.buckets:
                        for param in bucket.params:
                            if is_float8tensor(param):
                                fp8_params.append(param)
                    if len(fp8_params) > 0:
                        post_all_gather_processing(fp8_params)

    def start_grad_sync(self, *unused):
        """
        Initiates grad sync (all-reduce or reduce-scatter) communication operations
        for all model gradients.

        When overlap_grad_reduce is set to True, dispatches asynchronous communication
        calls. When overlap_grad_reduce is set to False, calls synchronous
        communication ops.
        """
        for bucket_group in self.bucket_groups + self.expert_parallel_bucket_groups:
            bucket_group.start_grad_sync()

    def finish_grad_sync(self):
        """
        Finishes grad sync (all-reduce or reduce-scatter) communication operations
        for all model gradients.

        When overlap_grad_reduce is set to True, waits for asynchronous communication
        calls to complete. When overlap_grad_reduce is set to False, calls synchronous
        communication ops.
        """
        for bucket_group in self.bucket_groups + self.expert_parallel_bucket_groups:
            bucket_group.finish_grad_sync()

    def scale_gradients(self, scaling_factor: float):
        """Scale all gradients inside the buffers by `scaling_factor`."""
        for buffer in self.buffers + self.expert_parallel_buffers:
            buffer.scale_gradients(scaling_factor)

    def zero_grad_buffer(self):
        """
        Zeros out all grad buffers. Needs to be called at the beginning of each
        training iteration.
        """
        if getattr(self.config, 'cuda_graph_impl', 'none') != 'transformer_engine':
            # Don't reset grad_added_to_main_grad when CUDA Graph is used.
            # Because in CUDA Graph it no longer has the opportunity to set it back
            # to True, and there will be a double-GA.
            for param in self.params_with_grad:
                param.grad_added_to_main_grad = False
        for buffer in self.buffers + self.expert_parallel_buffers:
            buffer.reset()
        for bucket_group in self.bucket_groups + self.expert_parallel_bucket_groups:
            bucket_group.reset()

    def broadcast_params(self):
        """
        Syncs parameters across all DP ranks.
        """
        for param in self.module.parameters():
            is_expert_parallel = not getattr(param, 'allreduce', True)

            if is_expert_parallel:
                data_parallel_group = self.expt_dp_group
            else:
                data_parallel_group = self.dp_cp_group
            torch.distributed.broadcast(
                param.data,
                src=torch.distributed.get_global_rank(data_parallel_group, 0),
                group=data_parallel_group,
            )

```

The fix has been applied. The only lint warning is a pre-existing environment issue (`torch` import not resolved by the linter).

Here's the updated code:

```388:402:/lustre/fsw/portfolios/coreai/users/shanmugamr/RL/3rdparty/Megatron-LM-workspace/Megatron-LM/megatron/core/distributed/distributed_data_parallel.py
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

---

**User**

Actually can you implement option 2. 

---

**Cursor**

Let me look at the locations where `toggle_cuda_graphs(set_to="none")` is called and add `disable_forward_pre_hook` before them:

```python
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
import gc
import os
import re
import time
import warnings
from collections import defaultdict
from contextlib import AbstractContextManager, contextmanager, nullcontext
from functools import partial
from typing import Any, Iterator, Optional, TypeVar, cast

import ray
import torch
from megatron.bridge import AutoBridge
from megatron.bridge.models.model_provider import get_model
from megatron.bridge.training import fault_tolerance
from megatron.bridge.training.checkpointing import (
    checkpoint_exists,
    init_checkpointing_context,
    load_checkpoint,
    maybe_finalize_async_save,
    save_checkpoint,
)
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    LoggerConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainingConfig,
)
from megatron.bridge.training.initialize import (
    initialize_megatron,
    set_jit_fusion_options,
)
from megatron.bridge.training.optim import setup_optimizer
from megatron.bridge.training.setup import (
    _update_model_config_funcs,
)
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer
from megatron.bridge.training.utils.train_utils import (
    logical_and_across_model_parallel_group,
    reduce_max_stat_across_model_parallel_group,
)
from megatron.bridge.utils.common_utils import get_rank_safe
from megatron.bridge.utils.instantiate_utils import InstantiationMode
from megatron.bridge.utils.vocab_utils import calculate_padded_vocab_size
from megatron.core import parallel_state
from megatron.core.transformer.utils import toggle_cuda_graphs
from megatron.core.distributed import DistributedDataParallel
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import (
    FullyShardedDataParallel as custom_FSDP,
)
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import (
    InferenceWrapperConfig,
)
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.models.gpt import GPTModel
from megatron.core.optimizer import ChainedOptimizer
from megatron.core.parallel_state import (
    get_context_parallel_group,
    get_context_parallel_rank,
    get_pipeline_model_parallel_group,
    get_pipeline_model_parallel_last_rank,
    get_pipeline_model_parallel_world_size,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    is_pipeline_last_stage,
)
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.utils import get_ltor_masks_and_position_ids
from ray.util.queue import Queue
from transformers import PreTrainedTokenizerBase

from nemo_rl.algorithms.interfaces import LossFunction, LossType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.model_utils import (
    allgather_cp_sharded_tensor,
    distributed_vocab_topk,
    from_parallel_logits_to_logprobs,
    from_parallel_logits_to_logprobs_packed_sequences,
)
from nemo_rl.distributed.named_sharding import NamedSharding
from nemo_rl.models.generation.fp8 import (
    convert_calibration_to_vllm_format,
    get_vllm_qkv_scale_names,
)
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationOutputSpec,
    verify_right_padding,
)
from nemo_rl.models.generation.vllm.config import VllmConfig
from nemo_rl.models.megatron.common import (
    _get_pack_sequence_parameters_for_megatron,
    _pack_sequences_for_megatron,
    broadcast_tensor,
    forward_step_arbitrary_loss,
    get_moe_metrics,
)
from nemo_rl.models.megatron.community_import import import_model_from_hf_name
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.interfaces import (
    ColocatablePolicyInterface,
    LogprobOutputSpec,
)
from nemo_rl.models.policy.utils import (
    configure_dynamo_cache,
    get_megatron_checkpoint_dir,
    get_runtime_env_for_policy_worker,
)
from nemo_rl.models.policy.workers.base_policy_worker import AbstractPolicyWorker
from nemo_rl.models.policy.workers.patches import apply_transformer_engine_patch
from nemo_rl.utils.nsys import wrap_with_nvtx_name
from nemo_rl.utils.packed_tensor import packed_broadcast_producer

try:
    from megatron.core.distributed import (
        TorchFullyShardedDataParallel as torch_FSDP,  # noqa: F401 unused-import
    )

    HAVE_FSDP2 = True
except ImportError:
    HAVE_FSDP2 = False

TokenizerType = TypeVar("TokenizerType", bound=PreTrainedTokenizerBase)


def broadcast_object_across_pp_ranks(obj):
    """Broadcast an object across pipeline parallel ranks.

    This utility function handles broadcasting an object from the rank that owns it
    to all other pipeline parallel ranks. If only one rank has the object (non-None),
    it will be broadcast to all other ranks.

    Args:
        obj: The object to broadcast. Can be None on ranks that don't own it.

    Returns:
        The object on all ranks (either the original or the broadcast copy).

    Raises:
        ValueError: If the object doesn't exist on any pipeline parallel rank.
    """
    pp_size = get_pipeline_model_parallel_world_size()
    pp_group = get_pipeline_model_parallel_group()

    if pp_size == 1:
        return obj

    # ------------------------------------------------------------------
    # 1. Gather presence flags from all PP ranks to find the source rank
    # ------------------------------------------------------------------
    has_obj = obj is not None
    obj_flags = [None] * pp_size
    torch.distributed.all_gather_object(obj_flags, has_obj, group=pp_group)

    # ------------------------------------------------------------------
    # 2. Identify the owning rank (the only rank with True flag)
    # ------------------------------------------------------------------
    src_rank = None  # Rank *inside* the PP group
    for rank, flag in enumerate(obj_flags):
        if flag:
            src_rank = rank
            break

    if src_rank is None:
        raise ValueError("Object must exist on at least one PP rank")

    # ------------------------------------------------------------------
    # 3. Broadcast the object from the source rank to all ranks
    # ------------------------------------------------------------------
    # Use broadcast_object_list which is more robust than all_gather_object
    obj_list = [obj]
    pp_ranks = torch.distributed.get_process_group_ranks(pp_group)
    global_src = pp_ranks[src_rank]
    torch.distributed.broadcast_object_list(obj_list, src=global_src, group=pp_group)

    return obj_list[0]


def setup_megatron_model(
    policy_cfg: PolicyConfig,
    cfg: ConfigContainer,
    load_optimizer: bool = True,
    get_embedding_ranks=None,  # TODO @sahilj: What is this?
    get_position_embedding_ranks=None,
):
    state = GlobalState()
    state.cfg = cfg
    # TODO: Freeze state.cfg

    cfg.dist.external_gpu_device_mapping = True
    initialize_megatron(
        cfg=cfg,
        get_embedding_ranks=get_embedding_ranks,
        get_position_embedding_ranks=get_position_embedding_ranks,
    )

    if cfg.ft and cfg.ft.enable_ft_package:
        fault_tolerance.setup(cfg, state)
        fault_tolerance.maybe_setup_simulated_fault(cfg.ft)

    # Set pytorch JIT layer fusion options and warmup JIT functions.
    set_jit_fusion_options(cfg.model, cfg.train.micro_batch_size)

    # Adjust the startup time so it reflects the largest value.
    # This will be closer to what scheduler will see (outside of
    # image ... launches.
    start_time_tensor = torch.tensor(
        [state.start_time], dtype=torch.double, device="cuda"
    )
    torch.distributed.all_reduce(start_time_tensor, op=torch.distributed.ReduceOp.MIN)
    state.start_time = start_time_tensor.item()

    print(
        "time to initialize megatron (seconds): {:.3f}".format(
            time.time() - state.start_time
        )
    )
    torch.distributed.barrier()

    # Context used for persisting some state between checkpoint saves.
    checkpointing_context = init_checkpointing_context(cfg.checkpoint)

    # Tokenizer
    build_tokenizer(
        cfg.tokenizer,
        make_vocab_size_divisible_by=cfg.model.make_vocab_size_divisible_by
        // cfg.model.tensor_model_parallel_size,
        tensor_model_parallel_size=cfg.model.tensor_model_parallel_size,
        trust_remote_code=True,
    )
    assert cfg.model.vocab_size, "vocab size must be specified in model config"

    torch.distributed.barrier()

    pre_wrap_hook = []
    mixed_precision_wrapper = Float16Module
    if policy_cfg["megatron_cfg"]["freeze_moe_router"]:

        def freeze_moe_router(megatron_model):
            if not isinstance(megatron_model, list):
                megatron_model = [megatron_model]
            for model_module in megatron_model:
                # Handle both wrapped (Float16Module) and unwrapped models
                if isinstance(model_module, Float16Module):
                    model_module = model_module.module
                # Handle VLM models
                if hasattr(model_module, "language_model"):
                    model_module = model_module.language_model
                for layer in model_module.decoder.layers:
                    if hasattr(layer, "mlp") and hasattr(layer.mlp, "router"):
                        layer.mlp.router.weight.requires_grad = False

        mixed_precision_wrapper = CustomFloat16Module
        pre_wrap_hook.extend([freeze_moe_router])

    # Model, optimizer, and learning rate.
    model = get_model(
        cfg.model,
        cfg.ddp,
        use_torch_fsdp2=cfg.dist.use_torch_fsdp2,
        overlap_param_gather_with_optimizer_step=cfg.optimizer.overlap_param_gather_with_optimizer_step,
        data_parallel_random_init=cfg.rng.data_parallel_random_init,
        pre_wrap_hook=pre_wrap_hook,
        mixed_precision_wrapper=mixed_precision_wrapper,
    )
    if load_optimizer:
        optimizer, scheduler = setup_optimizer(
            optimizer_config=cfg.optimizer,
            scheduler_config=cfg.scheduler,
            model=model,
            use_gloo_process_groups=cfg.dist.use_gloo_process_groups,
        )
    else:
        optimizer = None
        scheduler = None

    print("Model, optimizer, and learning rate scheduler built")
    torch.distributed.barrier()

    # Load checkpoint if applicable
    if (
        cfg.checkpoint.load is not None
        or cfg.checkpoint.pretrained_checkpoint is not None
    ) and (
        checkpoint_exists(cfg.checkpoint.load)
        or checkpoint_exists(cfg.checkpoint.pretrained_checkpoint)
    ):
        load_checkpoint(
            state,
            model,
            optimizer,
            scheduler,
            checkpointing_context=checkpointing_context,
            skip_load_to_model_and_opt=HAVE_FSDP2 and cfg.dist.use_torch_fsdp2,
        )
        print("Checkpoint loaded")
    torch.distributed.barrier()

    return state, model, optimizer, scheduler, checkpointing_context


def destroy_parallel_state():
    """Safely destroy parallel state and reset async call tracking.

    This function is called during initialization to clean up temporary distributed
    state from model import operations. Resetting async call tracking ensures that
    when the main Megatron distributed context is created, all ranks start with
    consistent call_idx values for async checkpointing.
    """
    if torch.distributed.is_initialized():
        try:
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()
        except:
            pass  # Ignore errors if already destroyed
    if hasattr(parallel_state, "destroy_model_parallel"):
        try:
            parallel_state.destroy_model_parallel()
        except:
            pass  # Ignore errors if already destroyed

    # Reset async calls queue to prevent call_idx mismatches after distributed context recreation
    try:
        import nemo.tron.utils.async_utils as nemo_async_utils
        from megatron.core.dist_checkpointing.strategies.async_utils import (
            AsyncCallsQueue,
        )

        # Clean up any existing async callers first
        old_call_idx = getattr(nemo_async_utils._async_calls_queue, "call_idx", None)
        num_unfinalized = (
            nemo_async_utils._async_calls_queue.get_num_unfinalized_calls()
        )
        if num_unfinalized > 0:
            print(
                f"[WARNING] Resetting async calls queue with {num_unfinalized} unfinalized calls"
            )
        try:
            nemo_async_utils._async_calls_queue.close()
        except:
            pass  # Ignore errors during cleanup
        # Reset the global async calls queue by creating a new instance
        nemo_async_utils._async_calls_queue = AsyncCallsQueue()
        print(f"[DEBUG] Reset NeMo async calls queue (old call_idx: {old_call_idx})")
    except ImportError:
        pass

    # Also reset the Megatron async calls queue if it exists
    try:
        import megatron.training.async_utils as megatron_async_utils
        from megatron.core.dist_checkpointing.strategies.async_utils import (
            AsyncCallsQueue,
        )

        # Clean up any existing async callers first
        old_call_idx = getattr(
            megatron_async_utils._async_calls_queue, "call_idx", None
        )
        num_unfinalized = (
            megatron_async_utils._async_calls_queue.get_num_unfinalized_calls()
        )
        if num_unfinalized > 0:
            print(
                f"[WARNING] Resetting Megatron async calls queue with {num_unfinalized} unfinalized calls"
            )
        try:
            megatron_async_utils._async_calls_queue.close()
        except:
            pass  # Ignore errors during cleanup
        # Reset the Megatron global async calls queue as well
        megatron_async_utils._async_calls_queue = AsyncCallsQueue()
        print(
            f"[DEBUG] Reset Megatron async calls queue (old call_idx: {old_call_idx})"
        )
    except ImportError:
        pass

    # Reset the third global async_calls instance in base strategy module
    try:
        import megatron.core.dist_checkpointing.strategies.base as base_strategy
        from megatron.core.dist_checkpointing.strategies.async_utils import (
            AsyncCallsQueue,
        )

        # Clean up and reset the global async_calls in base strategy
        old_call_idx = getattr(base_strategy.async_calls, "call_idx", None)
        num_unfinalized = base_strategy.async_calls.get_num_unfinalized_calls()
        if num_unfinalized > 0:
            print(
                f"[WARNING] Resetting base strategy async_calls with {num_unfinalized} unfinalized calls"
            )
        try:
            base_strategy.async_calls.close()
        except:
            pass
        base_strategy.async_calls = AsyncCallsQueue()
        print(f"[DEBUG] Reset base strategy async_calls (old call_idx: {old_call_idx})")
    except ImportError:
        pass


@ray.remote(
    runtime_env=get_runtime_env_for_policy_worker("megatron_policy_worker")
)  # pragma: no cover
class MegatronPolicyWorker(AbstractPolicyWorker, ColocatablePolicyInterface):
    def __repr__(self):
        """Customizes the actor's prefix in the Ray logs.

        This makes it easier to identify which worker is producing specific log messages.
        """
        if torch.distributed.is_initialized():
            return f"{self.__class__.__qualname__}[rank={torch.distributed.get_rank()}]"
        else:
            return f"{self.__class__.__qualname__}"

    def __init__(
        self,
        config: PolicyConfig,
        tokenizer: TokenizerType,
        weights_path: Optional[str] = None,
        optimizer_path: Optional[str] = None,
        init_optimizer: bool = True,
        init_reference_model: bool = True,
        *,
        worker_sharding_annotations: NamedSharding,
        pre_init_communication_queue: Queue,
        **kwargs: Any,
    ):
        apply_transformer_engine_patch()

        self.is_generation_colocated = None
        if "generation" in config and config["generation"] is not None:
            self.is_generation_colocated = config["generation"]["colocated"]["enabled"]

        # Explicitly set NCCL_CUMEM_ENABLE to 1 to avoid the P2P initialization error for PyNCCLCommunicator.
        # See https://github.com/NVIDIA-NeMo/RL/issues/564 for more details.
        if not self.is_generation_colocated:
            os.environ["NCCL_CUMEM_ENABLE"] = "1"

        self.cfg = config
        dtype_map = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }
        self.dtype = dtype_map[self.cfg["precision"]]

        self.optimizer_cpu_offload = self.cfg["megatron_cfg"]["optimizer"][
            "optimizer_cpu_offload"
        ]
        self.offload_optimizer_for_logprob = self.cfg["offload_optimizer_for_logprob"]

        # Reward models are not yet supported with Megatron.
        if "reward_model_cfg" in self.cfg and self.cfg["reward_model_cfg"]["enabled"]:
            raise NotImplementedError(
                "Reward models are not yet supported with the Megatron backend, this issue is "
                "tracked in https://github.com/NVIDIA-NeMo/RL/issues/720"
            )

        # Disable dynamo autotune_local_cache to avoid crash when there's already a cache
        # with different order of node_bundles
        configure_dynamo_cache()

        # cfg["model_name"] is allowed to be either an HF model name or a path to an HF checkpoint
        # check if hf_model_name is a path
        hf_model_name = self.cfg["model_name"]
        # Check if the checkpoint already exists
        hf_model_subdir = hf_model_name
        if os.path.exists(hf_model_name):
            hf_model_subdir = f"model_{hf_model_subdir.replace('/', '_')}"

        pretrained_path = f"{get_megatron_checkpoint_dir()}/{hf_model_subdir}"
        pt_checkpoint_exists = os.path.exists(pretrained_path) and os.path.exists(
            os.path.join(pretrained_path, "iter_0000000")
        )

        # Ensure clean slate before import
        destroy_parallel_state()

        # Set for rank for non-collocated to check which ranks to broadcast from
        self.rank = get_rank_safe()
        # Need to initialize the process group before calling into Megatron-Bridge, otherwise Megatron-Bridge will try to set an incorrect device
        torch.distributed.init_process_group("nccl")
        if pt_checkpoint_exists:
            print(f"Checkpoint already exists at {pretrained_path}. Skipping import.")
        else:
            hf_config_overrides = self.cfg.get("hf_config_overrides", {}) or {}
            import_model_from_hf_name(
                hf_model_name,
                pretrained_path,
                self.cfg["megatron_cfg"],
                **hf_config_overrides,
            )

            if parallel_state.model_parallel_is_initialized():
                print("Reinitializing model parallel after loading model state.")
                parallel_state.destroy_model_parallel()

        pretrained_run_config = os.path.join(
            pretrained_path, "iter_0000000/run_config.yaml"
        )

        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if not os.path.exists(pretrained_run_config):
            raise FileNotFoundError(
                f"Pretrained run config not found at {pretrained_run_config} on rank={get_rank_safe()}. This usually means that the one-time HF->mcore conversion on rank=0 saved to a directory not being mounted on this node. Please check "
            )

        try:
            cfg_from_pretrained = ConfigContainer.from_yaml(
                pretrained_run_config, mode=InstantiationMode.STRICT
            )
        except Exception as e:
            # Add helpful context as a note to the exception
            e.add_note(
                f"\n{'=' * 80}\n"
                f"NOTE: A common cause of this error is when the HF->mcore converted checkpoint is\n"
                f"created with an older version of megatron-bridge.\n"
                f"If this checkpoint is old or was generated by a different code version,\n"
                f"try deleting it and rerunning the code.\n"
                f"The checkpoint will be automatically regenerated with the current version.\n\n"
                f"Checkpoint location: {pretrained_path}\n"
                f"{'=' * 80}"
            )
            raise
        model_cfg = cfg_from_pretrained.model

        cfg_from_pretrained.logger = LoggerConfig()

        model_cfg.tensor_model_parallel_size = self.cfg["megatron_cfg"][
            "tensor_model_parallel_size"
        ]
        model_cfg.pipeline_model_parallel_size = self.cfg["megatron_cfg"][
            "pipeline_model_parallel_size"
        ]
        model_cfg.num_layers_in_first_pipeline_stage = self.cfg["megatron_cfg"][
            "num_layers_in_first_pipeline_stage"
        ]
        model_cfg.num_layers_in_last_pipeline_stage = self.cfg["megatron_cfg"][
            "num_layers_in_last_pipeline_stage"
        ]
        model_cfg.sequence_parallel = self.cfg["megatron_cfg"]["sequence_parallel"]
        model_cfg.context_parallel_size = self.cfg["megatron_cfg"][
            "context_parallel_size"
        ]
        if model_cfg.context_parallel_size > 1:
            assert self.cfg["sequence_packing"]["enabled"], (
                "Sequence Packing must be enabled to use Context Parallelism with MCore"
            )
        model_cfg.expert_tensor_parallel_size = self.cfg["megatron_cfg"][
            "expert_tensor_parallel_size"
        ]
        model_cfg.expert_model_parallel_size = self.cfg["megatron_cfg"][
            "expert_model_parallel_size"
        ]

        # Setting moe_router_dtype to higher precision (e.g. fp64) can improve numerical stability,
        # especially when using many experts.
        model_cfg.moe_router_dtype = self.cfg["megatron_cfg"]["moe_router_dtype"]

        # The below two configs (and "freeze_moe_router") are used to stabilize moe training
        # by preventing updates to the moe router. We found that this is helpful in reducing
        # logprob error during training.

        # Set this to "none" to disable load balancing loss.
        model_cfg.moe_router_load_balancing_type = self.cfg["megatron_cfg"][
            "moe_router_load_balancing_type"
        ]
        # Set this to 0.0 to disable updates to the moe router expert bias
        model_cfg.moe_router_bias_update_rate = self.cfg["megatron_cfg"][
            "moe_router_bias_update_rate"
        ]

        model_cfg.moe_permute_fusion = self.cfg["megatron_cfg"]["moe_permute_fusion"]
        if "layernorm_epsilon" in self.cfg["megatron_cfg"]:
            model_cfg.layernorm_epsilon = self.cfg["megatron_cfg"]["layernorm_epsilon"]

        model_cfg.sequence_parallel = self.cfg["megatron_cfg"]["sequence_parallel"]
        model_cfg.bf16 = self.dtype == torch.bfloat16
        model_cfg.fp16 = self.dtype == torch.float16
        if model_cfg.fp16:
            assert not model_cfg.bf16, "fp16 and bf16 cannot be used together"
            model_cfg.params_dtype = torch.float16
        elif model_cfg.bf16:
            assert not model_cfg.fp16, "fp16 and bf16 cannot be used together"
            model_cfg.params_dtype = torch.bfloat16
        else:
            model_cfg.params_dtype = torch.float32
        model_cfg.pipeline_dtype = dtype_map[self.cfg["megatron_cfg"]["pipeline_dtype"]]
        model_cfg.parallel_output = True
        if self.cfg["megatron_cfg"]["activation_checkpointing"]:
            model_cfg.recompute_granularity = "full"
            model_cfg.recompute_method = "uniform"
            model_cfg.recompute_num_layers = 1
        if not model_cfg.gated_linear_unit:
            assert model_cfg.activation_func is not None, (
                "activation_func must be set if not using gated_linear_unit. This likely "
                "indicates an issue in configuration conversion (e.g. activation func was "
                "a lambda and couldn't be serialized). This is based on this check "
                "https://github.com/NVIDIA/Megatron-LM/blob/1ab876ddc4c1893c76f26d775226a8d1dcdfb3d2/megatron/core/transformer/mlp.py#L174."
            )
        model_cfg.apply_rope_fusion = self.cfg["megatron_cfg"]["apply_rope_fusion"]
        model_cfg.bias_activation_fusion = self.cfg["megatron_cfg"][
            "bias_activation_fusion"
        ]
        fp8_cfg = self.cfg["megatron_cfg"].get("fp8_cfg", None)
        self.fp8_cfg = fp8_cfg
        if fp8_cfg is not None and fp8_cfg.get("enabled", False):
            try:
                model_cfg.fp8 = fp8_cfg["fp8"]
                model_cfg.fp8_recipe = fp8_cfg["fp8_recipe"]
                model_cfg.fp8_param = fp8_cfg["fp8_param"]
            except KeyError as e:
                raise KeyError(f"Missing key in fp8_cfg: {e}")
            if model_cfg.fp8_param:
                warnings.warn(
                    "Setting fp8_param=True sometimes causes NaN token_mult_prob_error, please use with caution. "
                    "Refer to https://github.com/NVIDIA-NeMo/RL/issues/1164 for latest updates with this issue."
                )

        optimizer_cpu_offload = self.cfg["megatron_cfg"]["optimizer"][
            "optimizer_cpu_offload"
        ]
        optimizer_offload_fraction = self.cfg["megatron_cfg"]["optimizer"][
            "optimizer_offload_fraction"
        ]
        if optimizer_cpu_offload:
            # Currently, hybrid optimizer (partly on GPU and partly on CPU) is not supported because it conflicts with the way
            # Nemo-rl handles the optimizer offload/onload between generation and training. So if using CPU optimizer the offload_fraction should be 1.0.
            assert optimizer_offload_fraction == 1.0, (
                "Currently for optimizer offloading, only optimizer_offload_fraction=1.0 is supported"
            )
        if (
            "logprob_chunk_size" in self.cfg
            and self.cfg["logprob_chunk_size"] is not None
            and self.cfg["logprob_chunk_size"] > 0
        ):
            assert self.cfg["megatron_cfg"]["defer_fp32_logits"], (
                "defer_fp32_logits must be True if logprob_chunk_size is set"
            )
        self.defer_fp32_logits = self.cfg["megatron_cfg"].get(
            "defer_fp32_logits", None
        ) and (model_cfg.fp16 or model_cfg.bf16)

        checkpoint_config = CheckpointConfig(
            save_interval=100,
            save=weights_path,
            load=weights_path,
            pretrained_checkpoint=pretrained_path,  # This is the path to the pretrained ckpt for the SFT case
            async_save=False,  # This doesn't work right now.
            fully_parallel_save=True,
            fully_parallel_load=True,  # Enable fully parallel load
            load_rng=False,
        )
        ref_checkpoint_config = CheckpointConfig(
            pretrained_checkpoint=pretrained_path,  # This is the path to the pretrained ckpt for the SFT case
            save=None,
            load=None,
            fully_parallel_load=True,  # Enable fully parallel load
            load_rng=False,
        )

        assert "train_iters" in self.cfg["megatron_cfg"], (
            "train_iters must be set in megatron_cfg. For an example, see "
            "https://github.com/NVIDIA-NeMo/RL/blob/bccbc377705a81a1f4b3c31ad9767bcc15f735a8/nemo_rl/algorithms/sft.py#L175-L179."
        )

        ## These settings are required for correct gradient computations in mcore
        ## when calculate_per_token_loss is True, there is no scaling of the gradient in mcore,
        ## so we handle the scaling in nemo-rl.
        ## perform_initialization = True is a workaround to ensure the correct tensor parallel attributes are set
        ## on the TP-sharded parameters.
        model_cfg.calculate_per_token_loss = True
        model_cfg.perform_initialization = True
        # SHAN CHECK:
        model_cfg.cuda_graph_impl = "local"
        model_cfg.cuda_graph_scope = None
        model_cfg.use_te_rng_tracker = True

        assert (
            "aux_loss" not in model_cfg.moe_router_load_balancing_type
            or model_cfg.moe_aux_loss_coeff == 0
        ), (
            "MoE aux loss is currently not supported due to a known bug in Megatron-LM. "
            "See https://github.com/NVIDIA/Megatron-LM/issues/1984 for more details."
        )

        self.megatron_cfg = ConfigContainer(
            model=model_cfg,
            checkpoint=checkpoint_config,
            logger=LoggerConfig(logging_level=0),
            train=TrainingConfig(
                micro_batch_size=1,  # ignored
                global_batch_size=self.cfg["train_global_batch_size"],  # ignored
                train_iters=self.cfg["megatron_cfg"][
                    "train_iters"
                ],  # Set by algorithm setup
            ),
            optimizer=OptimizerConfig(
                **self.cfg["megatron_cfg"]["optimizer"],
            ),
            ddp=DistributedDataParallelConfig(
                check_for_nan_in_grad=True,
                grad_reduce_in_fp32=self.cfg["megatron_cfg"][
                    "distributed_data_parallel_config"
                ]["grad_reduce_in_fp32"],
                overlap_grad_reduce=self.cfg["megatron_cfg"][
                    "distributed_data_parallel_config"
                ]["overlap_grad_reduce"],
                overlap_param_gather=self.cfg["megatron_cfg"][
                    "distributed_data_parallel_config"
                ]["overlap_param_gather"],
                # we need to set average_in_collective=False with calculate_per_token_loss=True.
                # otherwise, mcore throws an assertion error.
                average_in_collective=False,
                use_distributed_optimizer=self.cfg["megatron_cfg"]["optimizer"][
                    "use_distributed_optimizer"
                ],
                data_parallel_sharding_strategy=self.cfg["megatron_cfg"][
                    "distributed_data_parallel_config"
                ]["data_parallel_sharding_strategy"],
            ),
            scheduler=SchedulerConfig(
                **self.cfg["megatron_cfg"]["scheduler"],
            ),
            dataset=None,
            tokenizer=TokenizerConfig(
                tokenizer_type="HuggingFaceTokenizer",
                tokenizer_model=hf_model_name,
            ),
        )
        # TODO: this validation should happen inside mbridge: https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/1665
        if self.dtype == torch.bfloat16:
            assert self.megatron_cfg.model.bf16 == True, (
                "policy.megatron_cfg.model.bf16=True must be set if policy.precision=bfloat16. This is handled by nemo-rl so this indicates something is misconfigured."
            )
            assert (
                self.megatron_cfg.optimizer.use_precision_aware_optimizer == False
                or self.megatron_cfg.optimizer.bf16 == True
            ), (
                "policy.megatron_cfg.optimizer.bf16=True must be set if policy.precision=bfloat16 when using use_precision_aware_optimizer=True"
            )
        elif self.dtype == torch.float16:
            assert self.megatron_cfg.model.fp16 == True, (
                "policy.megatron_cfg.model.fp16=True must be set if policy.precision=float16. This is handled by nemo-rl so this indicates something is misconfigured."
            )
            assert (
                self.megatron_cfg.optimizer.use_precision_aware_optimizer == False
                or self.megatron_cfg.optimizer.fp16 == True
            ), (
                "policy.megatron_cfg.optimizer.fp16=True must be set if policy.precision=float16 when using use_precision_aware_optimizer=True"
            )
        elif self.dtype == torch.float32:
            assert (
                self.megatron_cfg.model.bf16 == False
                and self.megatron_cfg.model.fp16 == False
            ), (
                "policy.megatron_cfg.model.bf16=False and policy.megatron_cfg.model.fp16=False must be set if policy.precision=float32. This is handled by nemo-rl so this indicates something is misconfigured."
            )
            assert (
                self.megatron_cfg.optimizer.bf16 == False
                and self.megatron_cfg.optimizer.fp16 == False
            ), (
                "policy.megatron_cfg.optimizer.bf16=False and policy.megatron_cfg.optimizer.fp16=False must be set if policy.precision=float32"
            )
        self.megatron_cfg.validate()
        (
            self.mcore_state,
            self.model,
            self.optimizer,
            self.scheduler,
            self.checkpointing_context,
        ) = setup_megatron_model(
            policy_cfg=self.cfg, cfg=self.megatron_cfg, load_optimizer=init_optimizer
        )

        # Set the param sync function for the model
        if (
            self.megatron_cfg.ddp.overlap_param_gather
            and self.megatron_cfg.ddp.align_param_gather
        ):
            self.megatron_cfg.param_sync_func = [
                model_chunk.start_param_sync for model_chunk in self.model
            ]
            if len(self.model) == 1:
                self.megatron_cfg.param_sync_func = self.megatron_cfg.param_sync_func[0]

        self.model = self.model[0]  # Get the first model from the list

        if init_reference_model:
            self.model = self.move_model(self.model, "cpu")
            ref_ckpt_context = init_checkpointing_context(ref_checkpoint_config)

            # Create a separate megatron config for the reference model with the correct checkpoint config
            self.megatron_cfg.model.cuda_graph_impl = "none"
            self.megatron_cfg.model.use_te_rng_tracker = True
            ref_megatron_cfg = ConfigContainer(
                model=self.megatron_cfg.model,
                checkpoint=ref_checkpoint_config,  # Use the reference checkpoint config
                logger=self.megatron_cfg.logger,
                train=self.megatron_cfg.train,
                optimizer=self.megatron_cfg.optimizer,
                ddp=self.megatron_cfg.ddp,
                scheduler=self.megatron_cfg.scheduler,
                dataset=self.megatron_cfg.dataset,
                tokenizer=self.megatron_cfg.tokenizer,
            )

            # Create a separate state object for the reference model
            ref_state = GlobalState()
            ref_state.cfg = ref_megatron_cfg

            # Configure mixed precision wrapper for reference model
            ref_mixed_precision_wrapper = Float16Module
            if self.cfg["megatron_cfg"].get("freeze_moe_router", False):
                ref_mixed_precision_wrapper = CustomFloat16Module

            reference_model = get_model(
                self.megatron_cfg.model,
                self.megatron_cfg.ddp,
                use_torch_fsdp2=self.megatron_cfg.dist.use_torch_fsdp2,
                overlap_param_gather_with_optimizer_step=self.megatron_cfg.optimizer.overlap_param_gather_with_optimizer_step,
                pre_wrap_hook=self.megatron_cfg.rng.data_parallel_random_init,
                mixed_precision_wrapper=ref_mixed_precision_wrapper,
            )
            print("Loading the Reference Model")
            if (
                ref_checkpoint_config.pretrained_checkpoint is not None
                and checkpoint_exists(ref_checkpoint_config.pretrained_checkpoint)
            ):
                load_checkpoint(
                    ref_state,  # Use the separate state object with ref checkpoint config
                    reference_model,
                    None,  # no optimizer
                    None,  # no scheduler
                    checkpointing_context=ref_ckpt_context,
                    skip_load_to_model_and_opt=HAVE_FSDP2
                    and self.megatron_cfg.dist.use_torch_fsdp2,
                )
                reference_model = reference_model[0]
                reference_model.eval()
                self.reference_state_dict = {}
                for name, item in reference_model.state_dict().items():
                    if isinstance(item, torch.Tensor):
                        cpu_item = item.detach().to(
                            device="cpu", non_blocking=True, copy=True
                        )
                        del item
                    else:
                        cpu_item = item
                    self.reference_state_dict[name] = cpu_item
                print("Reference model loaded")
            else:
                print("Reference model not loaded")

            self.model = self.move_model(self.model, "cuda")

        _update_model_config_funcs(
            [self.model],
            self.megatron_cfg.model,
            self.megatron_cfg.ddp,
            self.optimizer,
            align_grad_reduce=self.megatron_cfg.dist.align_grad_reduce,
        )

        tokenizer_config = TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model=hf_model_name,
        )

        self.megatron_tokenizer = build_tokenizer(
            tokenizer_config,
            make_vocab_size_divisible_by=self.megatron_cfg.model.make_vocab_size_divisible_by
            // self.cfg["megatron_cfg"]["tensor_model_parallel_size"],
            tensor_model_parallel_size=self.cfg["megatron_cfg"][
                "tensor_model_parallel_size"
            ],
            trust_remote_code=True,
        )
        self.final_padded_vocab_size = calculate_padded_vocab_size(
            self.megatron_cfg.model.vocab_size,
            self.megatron_cfg.model.make_vocab_size_divisible_by,
            self.cfg["megatron_cfg"]["tensor_model_parallel_size"],
        )
        self.dp_size = worker_sharding_annotations.get_axis_size("data_parallel")
        self.megatron_bridge = AutoBridge.from_hf_pretrained(
            hf_model_name, trust_remote_code=True
        )

        self.should_disable_forward_pre_hook = (
            self.cfg["megatron_cfg"]["optimizer"]["use_distributed_optimizer"]
            and self.cfg["megatron_cfg"]["distributed_data_parallel_config"][
                "overlap_param_gather"
            ]
        )

        # vars used for refit
        ## will be initialized in prepare_refit_info
        # refit_param_info_mcore combines the conversion tasks with the param memory
        # [(mcore_param_name, estimated_memory), ...]
        # Note: here param name is local param name, with local layer number and
        # local expert id etc.
        self.refit_conversion_tasks = (
            None  # Meta data for conversion params from megatron bridge
        )
        self.refit_conversion_tasks_current_index = None
        self.refit_param_info_mcore = None

        ## used for streaming update inference engine weights
        self._held_gather_buffer = None

    def enable_forward_pre_hook(self):
        assert isinstance(self.model, DistributedDataParallel)
        self.model.enable_forward_pre_hook()

    def disable_forward_pre_hook(self, param_sync=True):
        assert isinstance(self.model, DistributedDataParallel)
        self.model.disable_forward_pre_hook(param_sync=param_sync)

    @wrap_with_nvtx_name("megatron_policy_worker/train")
    def train(
        self,
        data: BatchedDataDict,
        loss_fn: LossFunction,
        eval_mode: bool = False,
        gbs: Optional[int] = None,
        mbs: Optional[int] = None,
    ) -> dict[str, Any]:
        """Train the policy on a batch of data with a given loss function."""
        # Disable CUDA graphs for training to avoid conflicts with inference tensors.
        # This prevents "Inplace update to inference tensor outside InferenceMode" errors
        # when switching from generation (which uses CUDA graphs) to training.
        # Use toggle_cuda_graphs on the underlying module (not the DDP wrapper) to properly
        # reset CUDA graph state, following the pattern from megatron/rl/rl_utils.py.
        lang_module = (
            self.model.module.module
            if hasattr(self.model.module, "module")
            else self.model.module
        )

        # Disable forward pre-hooks BEFORE toggle_cuda_graphs to avoid stale hook handles
        # for CudaGraphManager modules that will be removed by toggle_cuda_graphs.
        if self.should_disable_forward_pre_hook:
            self.disable_forward_pre_hook()

        toggle_cuda_graphs(lang_module, set_to="none", reset_cuda_graphs=True)

        self.model.zero_grad_buffer()
        if hasattr(self.model, "inference_params"):
            self.model.inference_params = None

        # Reset any cached attention states
        for module in self.model.modules():
            if hasattr(module, "reset_inference_cache"):
                module.reset_inference_cache()
            if hasattr(module, "_inference_key_value_memory"):
                module._inference_key_value_memory = None

        if gbs is None:
            gbs = self.cfg["train_global_batch_size"]
        if mbs is None:
            mbs = self.cfg["train_micro_batch_size"]
        local_gbs = gbs // self.dp_size
        total_dataset_size = torch.tensor(data.size, device="cuda")
        torch.distributed.all_reduce(
            total_dataset_size,
            op=torch.distributed.ReduceOp.SUM,
            group=parallel_state.get_data_parallel_group(),
        )
        num_global_batches = int(total_dataset_size.item()) // gbs

        if eval_mode:
            ctx: AbstractContextManager[Any] = torch.no_grad()
            self.model.eval()
        else:
            ctx = nullcontext()
            # Ensure model is in training mode
            self.model.train()

        with ctx:
            # dim 1 is always assumed to be the sequence dim, sanity check this here
            sequence_dim = 1
            seq_dim_size = data["input_ids"].shape[sequence_dim]
            for k, v in data.items():
                if torch.is_tensor(v) and len(v.shape) > 1:
                    assert v.shape[sequence_dim] == seq_dim_size, (
                        f"Dim 1 must be the sequence dim, expected dim 1={seq_dim_size} but got shape {v.shape}"
                    )

            forward_step = partial(
                forward_step_arbitrary_loss, loss_fn=loss_fn, policy_cfg=self.cfg
            )
            all_mb_metrics = []
            losses = []
            total_num_microbatches = 0
            for gb_idx in range(num_global_batches):
                global_batch = data.get_batch(batch_idx=gb_idx, batch_size=local_gbs)

                assert "sample_mask" in global_batch, (
                    "sample_mask must be present in the data!"
                )
                ## get the normalization factor for the loss
                local_valid_seqs = torch.sum(global_batch["sample_mask"])

                if not "token_mask" in global_batch:
                    local_valid_toks = (
                        local_valid_seqs * global_batch["input_ids"].shape[1]
                    )
                else:
                    local_valid_toks = torch.sum(
                        global_batch["token_mask"][:, 1:]
                        * global_batch["sample_mask"].unsqueeze(-1)
                    )

                to_reduce = torch.tensor([local_valid_seqs, local_valid_toks]).cuda()
                torch.distributed.all_reduce(
                    to_reduce, group=parallel_state.get_data_parallel_group()
                )
                global_valid_seqs, global_valid_toks = to_reduce[0], to_reduce[1]

                if (
                    hasattr(loss_fn, "loss_type")
                    and loss_fn.loss_type == LossType.TOKEN_LEVEL
                ):
                    assert "token_mask" in global_batch, (
                        "token_mask must be present in the data when using token-level loss"
                    )

                batch = data.get_batch(batch_idx=gb_idx, batch_size=local_gbs)
                pack_seqs = False
                seqlen_key = None
                pad_factor = 1
                pad_full_seq_to = None
                pad_packed_seq_to_multiple_of = 1
                if self.cfg["dynamic_batching"]["enabled"]:
                    data_iterator = batch.make_microbatch_iterator_with_dynamic_shapes()
                    data_iterator_len = (
                        batch.get_microbatch_iterator_dynamic_shapes_len()
                    )
                elif self.cfg["sequence_packing"]["enabled"]:
                    data_iterator = (
                        batch.make_microbatch_iterator_for_packable_sequences()
                    )
                    data_iterator_len, seq_dim_size = (
                        batch.get_microbatch_iterator_for_packable_sequences_len()
                    )
                    mbs = 1
                    pack_seqs = True
                    seqlen_key = "input_lengths"
                    (
                        pad_factor,
                        pad_packed_seq_to_multiple_of,
                        pad_full_seq_to,
                    ) = _get_pack_sequence_parameters_for_megatron(
                        self.cfg["megatron_cfg"],
                        seq_dim_size,
                    )
                    # if pad_full_seq_to is not None, we need to use it as the sequence length
                    seq_dim_size = pad_full_seq_to or seq_dim_size
                else:
                    data_iterator = batch.make_microbatch_iterator(mbs)
                    data_iterator_len = local_gbs // mbs

                # Track total microbatches for MoE aux-loss averaging
                total_num_microbatches += int(data_iterator_len)

                rerun_state_machine = get_rerun_state_machine()
                while rerun_state_machine.should_run_forward_backward(data_iterator):
                    # Set grad to zero.
                    self.model.zero_grad_buffer()
                    self.optimizer.zero_grad()

                    # Forward pass.
                    forward_backward_func = get_forward_backward_func()
                    losses_reduced = forward_backward_func(
                        forward_step_func=partial(
                            forward_step,
                            self.mcore_state,
                            global_valid_seqs,
                            global_valid_toks,
                            pack_sequences=pack_seqs,
                            seq_length_key=seqlen_key,
                            pad_individual_seqs_to_multiple_of=pad_factor,
                            pad_packed_seq_to_multiple_of=pad_packed_seq_to_multiple_of,
                            pad_full_seq_to=pad_full_seq_to,
                            defer_fp32_logits=self.defer_fp32_logits,
                        ),
                        data_iterator=data_iterator,
                        model=self.model,
                        num_microbatches=data_iterator_len,
                        seq_length=seq_dim_size,
                        micro_batch_size=mbs,
                        decoder_seq_length=seq_dim_size,
                        forward_only=eval_mode,
                        do_not_average_loss=True,
                    )

                # Empty unused memory.
                if self.cfg["megatron_cfg"]["empty_unused_memory_level"] >= 1:
                    torch.cuda.empty_cache()

                # Update parameters.
                if not eval_mode:
                    update_successful, grad_norm, num_zeros_in_grad = (
                        self.optimizer.step()
                    )
                else:
                    update_successful, grad_norm, num_zeros_in_grad = (True, 0.0, 0.0)

                # when freezing sub-models we may have a mixture of successful and unsucessful ranks,
                # so we must gather across mp ranks
                update_successful = logical_and_across_model_parallel_group(
                    update_successful
                )
                # grad_norm and num_zeros_in_grad will be None on ranks without trainable params,
                # so we must gather across mp ranks
                grad_norm: float = reduce_max_stat_across_model_parallel_group(
                    grad_norm
                )
                num_zeros_in_grad: float = reduce_max_stat_across_model_parallel_group(
                    num_zeros_in_grad
                )

                if update_successful:
                    skipped_iter = 0
                else:
                    skipped_iter = 1

                # Empty unused memory.
                if self.cfg["megatron_cfg"]["empty_unused_memory_level"] >= 2:
                    torch.cuda.empty_cache()

                if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    # keep all microbatch metrics to be normalized later
                    gb_loss_metrics = []
                    mb_losses = []
                    for x in losses_reduced:
                        loss_metrics = {}
                        for k in x.keys():
                            loss_metrics[k] = x[k] / num_global_batches
                        gb_loss_metrics.append(loss_metrics)
                        curr_lr = self.scheduler.get_lr(self.optimizer.param_groups[0])
                        curr_wd = self.scheduler.get_wd()
                        loss_metrics["lr"] = curr_lr
                        loss_metrics["wd"] = curr_wd
                        loss_metrics["global_valid_seqs"] = global_valid_seqs.item()
                        loss_metrics["global_valid_toks"] = global_valid_toks.item()
                        mb_losses.append(loss_metrics["loss"])

                    torch.distributed.broadcast_object_list(
                        [gb_loss_metrics],
                        src=get_pipeline_model_parallel_last_rank(),
                        group=get_pipeline_model_parallel_group(),
                    )
                else:
                    loss_metrics = [None]  # type: ignore
                    torch.distributed.broadcast_object_list(
                        loss_metrics,
                        src=get_pipeline_model_parallel_last_rank(),
                        group=get_pipeline_model_parallel_group(),
                    )
                    gb_loss_metrics = loss_metrics[0]
                    mb_losses = [x["loss"] for x in gb_loss_metrics]

                all_mb_metrics.extend(gb_loss_metrics)
                losses.append(torch.tensor(mb_losses).sum().item())

        if not eval_mode:
            # take one LR step every rollout batch
            # we need to scale the step by gbs to counteract the fact that NeMo automatically
            # scales lr_warmup_steps by gbs during init
            self.scheduler.step(increment=gbs)

        # Aggregate metrics across all microbatches
        mb_metrics = defaultdict(list)
        for m in all_mb_metrics:
            for k, v in m.items():
                mb_metrics[k].append(v)

        with torch.no_grad():
            global_loss = torch.tensor(losses, device="cuda")
            torch.distributed.all_reduce(
                global_loss,
                op=torch.distributed.ReduceOp.SUM,
                group=parallel_state.get_data_parallel_group(),
            )

        metrics = {
            "global_loss": global_loss.cpu(),
            "rank": torch.distributed.get_rank(),
            "gpu_name": torch.cuda.get_device_name(),
            "model_dtype": self.dtype,
            "all_mb_metrics": dict(mb_metrics),
            "grad_norm": torch.tensor([grad_norm]),
        }
        # Collect MoE aux metrics averaged across microbatches
        num_moe_experts = getattr(self.model.config, "num_moe_experts", None)
        if num_moe_experts is not None and num_moe_experts > 1:
            moe_loss_scale = 1.0 / max(1, total_num_microbatches)
            moe_metrics = get_moe_metrics(
                loss_scale=moe_loss_scale,
                per_layer_logging=self.cfg["megatron_cfg"]["moe_per_layer_logging"],
            )
            if moe_metrics:
                metrics["moe_metrics"] = moe_metrics
        return metrics

    @wrap_with_nvtx_name("megatron_policy_worker/get_logprobs")
    def get_logprobs(
        self, *, data: BatchedDataDict[Any], micro_batch_size: Optional[int] = None
    ) -> BatchedDataDict[LogprobOutputSpec]:
        """Get the logprobs of the model for a batch of data.

        Uses the configured logprob_batch_size to do microbatching.
        Input data is assumed to be right-padded. The method internally converts to
        left-padded format for computation, and returns outputs in right-padded format.
        If micro_batch_size is provided, it will be used instead of the configured
        logprob_batch_size.

        Returns:
          a BatchedDataDict with key "logprobs" and shape [batch_size, sequence_length].
          We use the convention that the logprob of the first token is 0 so that the sequence length is maintained.
          The logprob of input token i is specified at position i in the output logprobs tensor.
        """
        no_grad = torch.no_grad()
        no_grad.__enter__()
        logprob_batch_size = (
            micro_batch_size
            if micro_batch_size is not None
            else self.cfg["logprob_batch_size"]
        )

        # dim 1 is always assumed to be the sequence dim, sanity check this here
        sequence_dim = 1
        input_seq_dim_size = data["input_ids"].shape[sequence_dim]
        for k, v in data.items():
            if torch.is_tensor(v) and len(v.shape) > 1:
                assert v.shape[sequence_dim] == input_seq_dim_size, (
                    f"Dim 1 must be the sequence dim, expected dim 1={input_seq_dim_size} but got shape {v.shape}"
                )

        self.model.eval()

        pp_seq_dim_size = input_seq_dim_size
        pp_grp = get_pipeline_model_parallel_group()
        pad_factor = 1
        pad_packed_seq_to_multiple_of = 1
        pad_full_seq_to = None
        if self.cfg["sequence_packing"]["enabled"]:
            _, seq_dim_size = data.get_microbatch_iterator_for_packable_sequences_len()
            (
                pad_factor,
                pad_packed_seq_to_multiple_of,
                pad_full_seq_to,
            ) = _get_pack_sequence_parameters_for_megatron(
                self.cfg["megatron_cfg"],
                seq_dim_size,
            )
            pp_seq_dim_size = pad_full_seq_to or pp_seq_dim_size

        def forward_step_fn(
            data_iterator: Iterator[BatchedDataDict[Any]], model: GPTModel
        ):
            nonlocal pad_full_seq_to, pad_packed_seq_to_multiple_of, pad_factor
            data_dict = next(data_iterator).to("cuda")
            if self.cfg["sequence_packing"]["enabled"]:
                original_seq_length = data_dict["input_ids"].shape[1]
                cp_size = self.cfg["megatron_cfg"]["context_parallel_size"]
                cp_rank = get_context_parallel_rank()
                (
                    input_ids,
                    input_ids_cp_sharded,
                    packed_seq_params,
                    cu_seqlens,
                    cu_seqlens_padded,
                ) = _pack_sequences_for_megatron(
                    data_dict["input_ids"].clone(),
                    data_dict["input_lengths"],
                    pad_individual_seqs_to_multiple_of=pad_factor,
                    pad_packed_seq_to_multiple_of=pad_packed_seq_to_multiple_of,
                    pad_packed_seq_to=pad_full_seq_to,
                    cp_rank=cp_rank,
                    cp_size=cp_size,
                )
                attention_mask, position_ids = None, None
                unpacked_input_ids = data_dict["input_ids"]
            else:
                input_ids = data_dict["input_ids"]
                input_ids_cp_sharded = input_ids
                attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
                    data=input_ids,
                    eod_token=0,  # used for loss_mask, which we don't use
                    pad_token=0,  # used for loss_mask, which we don't use
                    reset_position_ids=False,
                    reset_attention_mask=False,
                    eod_mask_loss=False,
                    pad_mask_loss=False,
                )
                packed_seq_params = None
                unpacked_input_ids = input_ids

            multimodal_data = data_dict.get_multimodal_dict(
                as_tensors=True, device=input_ids.device
            )
            if len(multimodal_data) > 0:
                position_ids = None

            additional_kwargs = {}
            # Mamba models currently do not support packed_seq_params
            if packed_seq_params is not None:
                additional_kwargs["packed_seq_params"] = packed_seq_params

            if self.defer_fp32_logits:
                additional_kwargs["fp32_output"] = False

            output_tensor = model(
                input_ids=input_ids_cp_sharded,
                position_ids=position_ids,
                attention_mask=attention_mask,
                **multimodal_data,
                **additional_kwargs,
            )

            # Apply temperature scaling to logits for training
            # This matches the dtensor worker's _apply_temperature_scaling in the train method
            if "generation" in self.cfg and self.cfg["generation"] is not None:
                output_tensor.div_(self.cfg["generation"]["temperature"])

            def collection_fn(output_tensor):
                stc = time.time()
                tp_grp = get_tensor_model_parallel_group()
                tp_rank = get_tensor_model_parallel_rank()
                logprob_chunk_size = self.cfg.get("logprob_chunk_size", None)
                if self.cfg["sequence_packing"]["enabled"]:
                    token_logprobs = from_parallel_logits_to_logprobs_packed_sequences(
                        output_tensor,
                        target=input_ids,
                        cu_seqlens_padded=cu_seqlens_padded,
                        unpacked_seqlen=original_seq_length,
                        vocab_start_index=tp_rank * output_tensor.shape[-1],
                        vocab_end_index=(tp_rank + 1) * output_tensor.shape[-1],
                        group=tp_grp,
                        inference_only=True,
                        cp_group=get_context_parallel_group(),
                        chunk_size=logprob_chunk_size,
                    )
                else:
                    token_logprobs = from_parallel_logits_to_logprobs(
                        output_tensor,
                        target=unpacked_input_ids,
                        vocab_start_index=tp_rank * output_tensor.shape[-1],
                        vocab_end_index=(tp_rank + 1) * output_tensor.shape[-1],
                        tp_group=tp_grp,
                        inference_only=True,
                        chunk_size=logprob_chunk_size,
                    )

                # Prepend 0 logprob for first token to maintain same sequence length as input
                token_logprobs = torch.cat(
                    [torch.zeros_like(token_logprobs[:, :1]), token_logprobs], dim=1
                )
                return torch.tensor(0.0, device=token_logprobs.device), {
                    "logprobs": token_logprobs
                }

            return output_tensor, collection_fn

        if self.cfg["dynamic_batching"]["enabled"]:
            mb_iterator = data.make_microbatch_iterator_with_dynamic_shapes()
            data_iterator_len = data.get_microbatch_iterator_dynamic_shapes_len()
            micro_batch_size = logprob_batch_size
        elif self.cfg["sequence_packing"]["enabled"]:
            mb_iterator = data.make_microbatch_iterator_for_packable_sequences()
            data_iterator_len, _ = (
                data.get_microbatch_iterator_for_packable_sequences_len()
            )
            micro_batch_size = 1
        else:
            mb_iterator = data.make_microbatch_iterator(logprob_batch_size)
            data_iterator_len = max(1, data.size // logprob_batch_size)
            micro_batch_size = logprob_batch_size

        forward_backward_func = get_forward_backward_func()
        list_of_logprobs = forward_backward_func(
            forward_step_func=forward_step_fn,
            data_iterator=mb_iterator,
            model=self.model,
            num_microbatches=data_iterator_len,
            seq_length=pp_seq_dim_size,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=pp_seq_dim_size,
            forward_only=True,
        )
        if is_pipeline_last_stage(ignore_virtual=True):
            all_log_probs_padded = []
            all_logprobs = [l["logprobs"] for l in list_of_logprobs]
            for lp in all_logprobs:
                padding_needed = input_seq_dim_size - lp.shape[1]
                if padding_needed > 0:
                    lp = torch.nn.functional.pad(
                        lp, (0, padding_needed), mode="constant", value=0.0
                    )
                all_log_probs_padded.append(lp)

            logprobs = torch.cat(all_log_probs_padded, dim=0)
            # broadcast logprobs to first pp rank
            broadcast_tensor(logprobs, torch.distributed.get_rank(), pp_grp)
        else:
            logprobs = broadcast_tensor(
                None, get_pipeline_model_parallel_last_rank(), pp_grp
            )

        no_grad.__exit__(None, None, None)
        return BatchedDataDict[LogprobOutputSpec](logprobs=logprobs).to("cpu")

    @contextmanager
    def use_reference_model(self):
        """Context manager that temporarily swaps the reference model and active model.

        On entry: Moves model to CPU, moves reference_model to CUDA. Swaps the references
        On exit: Restores original references and re-flips cuda/cpu
        """
        ## disable overlap param gather when swapping weights
        if self.should_disable_forward_pre_hook:
            self.disable_forward_pre_hook()

        with torch.no_grad():
            try:
                # Save original references
                model_state_dict = {}
                for name, item in self.model.state_dict().items():
                    if isinstance(item, torch.Tensor):
                        item = item.detach().to(
                            device="cpu", non_blocking=True, copy=True
                        )
                    model_state_dict[name] = item

                self.model.load_state_dict(self.reference_state_dict, strict=True)
                # for name, item in self.reference_state_dict.items():
                # if isinstance(item, torch.Tensor):
                # self.model.state_dict()[name] = item.detach().to(device="cuda", non_blocking=True, copy=True)

                if self.cfg["megatron_cfg"]["empty_unused_memory_level"] >= 1:
                    gc.collect()
                    torch.cuda.empty_cache()

                # - self.model is the original reference_model, now on CUDA
                # - self.reference_model is the original model, now on CPU
                yield

            finally:
                # Restore original references and device placement
                self.model.load_state_dict(model_state_dict, strict=True)
                # for name, item in model_state_dict.items():
                # if isinstance(item, torch.Tensor):
                # item = item.detach().to(device="cuda", non_blocking=True, copy=True)
                # self.model.state_dict()[name] = item

                if self.cfg["megatron_cfg"]["empty_unused_memory_level"] >= 1:
                    gc.collect()
                    torch.cuda.empty_cache()

                ## re-enable overlap param gather after weight swap
                if self.should_disable_forward_pre_hook:
                    self.enable_forward_pre_hook()

    @wrap_with_nvtx_name("megatron_policy_worker/get_topk_logits")
    def get_topk_logits(
        self,
        *,
        data: BatchedDataDict[GenerationDatumSpec],
        k: int,
        micro_batch_size: Optional[int] = None,
    ):
        """Get the top-k logits and indices for a batch of data.

        The major difference from get_logprobs is that we compute top-k logits and indices for each position in the sequence.

        Returns:
            BatchedDataDict containing:
                - topk_logits: Tensor of top-k logits for each position in the sequence
                - topk_indices: Tensor of top-k indices for each position in the sequence
        """
        no_grad = torch.no_grad()
        no_grad.__enter__()

        logprob_batch_size = (
            micro_batch_size
            if micro_batch_size is not None
            else self.cfg["logprob_batch_size"]
        )

        sequence_dim = 1
        input_seq_dim_size = data["input_ids"].shape[sequence_dim]
        # Avoid shadowing the function argument `k` by using a distinct variable name
        for tensor_name, v in data.items():
            if torch.is_tensor(v) and len(v.shape) > 1:
                assert v.shape[sequence_dim] == input_seq_dim_size, (
                    f"Tensor {tensor_name} must have sequence dimension {sequence_dim} of size {input_seq_dim_size}, but got shape {v.shape}"
                )

        self.model.eval()

        pp_seq_dim_size = input_seq_dim_size
        pp_grp = get_pipeline_model_parallel_group()

        pad_factor = 1
        pad_packed_seq_to_multiple_of = 1
        pad_full_seq_to = None
        if self.cfg["sequence_packing"]["enabled"]:
            _, seq_dim_size = data.get_microbatch_iterator_for_packable_sequences_len()
            (
                pad_factor,
                pad_packed_seq_to_multiple_of,
                pad_full_seq_to,
            ) = _get_pack_sequence_parameters_for_megatron(
                self.cfg["megatron_cfg"],
                seq_dim_size,
            )
            pp_seq_dim_size = pad_full_seq_to or pp_seq_dim_size

        def forward_step_fn(
            data_iterator: Iterator[BatchedDataDict[Any]], model: GPTModel
        ):
            nonlocal pad_full_seq_to, pad_packed_seq_to_multiple_of, pad_factor
            data_dict = next(data_iterator).to("cuda")

            pack = self.cfg["sequence_packing"]["enabled"]
            if pack:
                original_seq_length = data_dict["input_ids"].shape[1]
                cp_size = self.cfg["megatron_cfg"]["context_parallel_size"]
                cp_rank = get_context_parallel_rank()

                (
                    input_ids_unpacked,
                    input_ids_cp_sharded,
                    packed_seq_params,
                    cu_seqlens,
                    cu_seqlens_padded,
                ) = _pack_sequences_for_megatron(
                    data_dict["input_ids"].clone(),
                    data_dict["input_lengths"],
                    pad_individual_seqs_to_multiple_of=pad_factor,
                    pad_packed_seq_to_multiple_of=pad_packed_seq_to_multiple_of,
                    pad_packed_seq_to=pad_full_seq_to,
                    cp_rank=cp_rank,
                    cp_size=cp_size,
                )
                attention_mask, position_ids = None, None
                seq_lengths = data_dict["input_lengths"]
                unpacked_seqlen = original_seq_length
            else:
                input_ids_cp_sharded = data_dict["input_ids"]
                attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
                    data=input_ids_cp_sharded,
                    eod_token=0,
                    pad_token=0,
                    reset_position_ids=False,
                    reset_attention_mask=False,
                    eod_mask_loss=False,
                    pad_mask_loss=False,
                )
                packed_seq_params = None

            multimodal_data = data_dict.get_multimodal_dict(
                as_tensors=True, device=input_ids_cp_sharded.device
            )
            if len(multimodal_data) > 0:
                position_ids = None

            additional_kwargs = {}
            if packed_seq_params is not None:
                additional_kwargs["packed_seq_params"] = packed_seq_params

            output_tensor = model(
                input_ids=input_ids_cp_sharded,
                position_ids=position_ids,
                attention_mask=attention_mask,
                **additional_kwargs,
                **multimodal_data,
            )

            if "generation" in self.cfg and self.cfg["generation"] is not None:
                output_tensor.div_(self.cfg["generation"]["temperature"])

            def collection_fn(_):
                # Only the last PP stage produces final logits/top-k; earlier stages return empty
                # if not is_pipeline_last_stage(ignore_virtual=True):
                # return output_tensor.new_zeros(()), {}

                tp_grp = get_tensor_model_parallel_group()
                tp_rank = get_tensor_model_parallel_rank()
                vocab_shard_size = output_tensor.shape[-1]
                vocab_start_index = tp_rank * vocab_shard_size

                chunk_size = None
                if "logprob_chunk_size" in self.cfg:
                    chunk_size = self.cfg["logprob_chunk_size"]

                topk_vals_local, topk_idx_local = distributed_vocab_topk(
                    output_tensor,
                    k,
                    tp_grp,
                    vocab_start_index=vocab_start_index,
                    vocab_end_index=vocab_start_index + vocab_shard_size,
                    chunk_size=chunk_size,
                )

                if self.cfg["megatron_cfg"]["context_parallel_size"] > 1:
                    cp_grp = get_context_parallel_group()
                    if pack:
                        # Per-sequence CP allgather following packed-sequence logic
                        batch_size = data_dict["input_ids"].shape[0]
                        total_packed_len = int(cu_seqlens_padded[-1].item())

                        topk_vals_full = torch.zeros(
                            (1, total_packed_len, k),
                            dtype=topk_vals_local.dtype,
                            device=topk_vals_local.device,
                        )
                        topk_idx_full = torch.zeros(
                            (1, total_packed_len, k),
                            dtype=topk_idx_local.dtype,
                            device=topk_idx_local.device,
                        )

                        for i in range(batch_size):
                            start_idx = int(cu_seqlens_padded[i].item())
                            end_idx = int(cu_seqlens_padded[i + 1].item())
                            if end_idx > start_idx:
                                local_vals_slice = topk_vals_local[
                                    :, start_idx // cp_size : end_idx // cp_size, :
                                ]
                                local_idx_slice = topk_idx_local[
                                    :, start_idx // cp_size : end_idx // cp_size, :
                                ]
                                gathered_vals = allgather_cp_sharded_tensor(
                                    local_vals_slice, cp_grp, seq_dim=1
                                )
                                gathered_idx = allgather_cp_sharded_tensor(
                                    local_idx_slice, cp_grp, seq_dim=1
                                )
                                # Some kernels may return [X, Y, k] where X*Y = (end_idx - start_idx).
                                # Flatten leading dims and reshape to [1, expected_len, k] to match target.
                                expected_len = end_idx - start_idx
                                if (
                                    gathered_vals.dim() == 3
                                    and gathered_vals.shape[1] != expected_len
                                ):
                                    gathered_vals = gathered_vals.reshape(
                                        1, expected_len, gathered_vals.shape[-1]
                                    )
                                if (
                                    gathered_idx.dim() == 3
                                    and gathered_idx.shape[1] != expected_len
                                ):
                                    gathered_idx = gathered_idx.reshape(
                                        1, expected_len, gathered_idx.shape[-1]
                                    )
                                topk_vals_full[:, start_idx:end_idx, :] = gathered_vals
                                topk_idx_full[:, start_idx:end_idx, :] = gathered_idx
                    else:
                        # Sequence packing must be enabled when CP > 1
                        raise RuntimeError(
                            "Context Parallelism (CP>1) requires sequence packing to be enabled."
                        )
                else:
                    topk_vals_full = topk_vals_local
                    topk_idx_full = topk_idx_local

                if pack:
                    batch_size = data_dict["input_ids"].shape[0]
                    out_vals = torch.zeros(
                        (batch_size, unpacked_seqlen, k),
                        dtype=topk_vals_full.dtype,
                        device=topk_vals_full.device,
                    )
                    out_idx = torch.zeros(
                        (batch_size, unpacked_seqlen, k),
                        dtype=topk_idx_full.dtype,
                        device=topk_idx_full.device,
                    )
                    for i in range(batch_size):
                        seq_len = int(seq_lengths[i].item())
                        start_idx = int(cu_seqlens_padded[i].item())
                        if seq_len > 0:
                            out_vals[i, :seq_len, :] = topk_vals_full[
                                0, start_idx : start_idx + seq_len, :
                            ]
                            out_idx[i, :seq_len, :] = topk_idx_full[
                                0, start_idx : start_idx + seq_len, :
                            ]
                    return output_tensor.new_zeros(()), {
                        "topk_logits": out_vals,
                        "topk_indices": out_idx,
                    }
                else:
                    return output_tensor.new_zeros(()), {
                        "topk_logits": topk_vals_full,
                        "topk_indices": topk_idx_full,
                    }

            return output_tensor, collection_fn

        if self.cfg["dynamic_batching"]["enabled"]:
            mb_iterator = data.make_microbatch_iterator_with_dynamic_shapes()
            data_iterator_len = data.get_microbatch_iterator_dynamic_shapes_len()
            micro_batch = logprob_batch_size
        elif self.cfg["sequence_packing"]["enabled"]:
            mb_iterator = data.make_microbatch_iterator_for_packable_sequences()
            data_iterator_len, _ = (
                data.get_microbatch_iterator_for_packable_sequences_len()
            )
            micro_batch = 1
        else:
            mb_iterator = data.make_microbatch_iterator(logprob_batch_size)
            data_iterator_len = max(1, data.size // logprob_batch_size)
            micro_batch = logprob_batch_size

        forward_backward_func = get_forward_backward_func()
        list_of_outputs = forward_backward_func(
            forward_step_func=forward_step_fn,
            data_iterator=mb_iterator,
            model=self.model,
            num_microbatches=data_iterator_len,
            seq_length=pp_seq_dim_size,
            micro_batch_size=micro_batch,
            decoder_seq_length=pp_seq_dim_size,
            forward_only=True,
        )

        if is_pipeline_last_stage(ignore_virtual=True):
            logits_chunks = []
            indices_chunks = []
            for out in list_of_outputs:
                tk = out["topk_logits"]
                ti = out["topk_indices"]
                pad_len = input_seq_dim_size - tk.shape[1]
                if pad_len > 0:
                    tk = torch.nn.functional.pad(tk, (0, 0, 0, pad_len), value=0.0)
                    ti = torch.nn.functional.pad(ti, (0, 0, 0, pad_len), value=0)
                logits_chunks.append(tk)
                indices_chunks.append(ti)

            topk_logits = torch.cat(logits_chunks, dim=0)
            topk_indices = torch.cat(indices_chunks, dim=0)

            topk_logits = broadcast_tensor(
                topk_logits, torch.distributed.get_rank(), pp_grp
            )
            topk_indices = broadcast_tensor(
                topk_indices, torch.distributed.get_rank(), pp_grp
            )
        else:
            last_pp_rank = get_pipeline_model_parallel_last_rank()
            topk_logits = broadcast_tensor(None, last_pp_rank, pp_grp)
            topk_indices = broadcast_tensor(None, last_pp_rank, pp_grp)

        no_grad.__exit__(None, None, None)
        return BatchedDataDict.from_batches(
            [{"topk_logits": topk_logits.cpu(), "topk_indices": topk_indices.cpu()}]
        )

    @torch.no_grad()
    @wrap_with_nvtx_name("megatron_policy_worker/generate")
    def generate(
        self, *, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> BatchedDataDict[GenerationOutputSpec]:
        """Generate a batch of data using huggingface framework generation.

        Args:
            data: BatchedDataDict containing input_ids and input_lengths tensors
        Returns:
            BatchedDataDict conforming to GenerationOutputSpec:
                - output_ids: input + generated token IDs
                - logprobs: Log probabilities for each token
                - generation_lengths: Lengths of each response
        """
        # 512 bATCH SIZE (200 tokens)
        no_grad = torch.no_grad()
        no_grad.__enter__()
        self.model.config.flash_decode = False
        if self.should_disable_forward_pre_hook:
            self.model = self.move_model(
                self.model, "cuda", move_params=True, move_grads=False
            )
        # Verify input is right padded
        assert isinstance(data, BatchedDataDict), (
            f"data must be a BatchedDataDict, got type: {type(data)}"
        )
        assert "input_ids" in data and "input_lengths" in data, (
            f"input_ids and input_lengths must be present in the BatchedDataDict, got keys: {data.keys()}"
        )
        is_right_padded, error_msg = verify_right_padding(
            data, pad_value=self.tokenizer.pad_token_id
        )
        if not is_right_padded:
            warnings.warn(
                f"Input to Megatron Generation worker is not properly right-padded: {error_msg}"
            )

        model_cfg = self.megatron_cfg.model
        inference_wrapper_config = InferenceWrapperConfig(
            hidden_size=model_cfg.hidden_size,
            inference_batch_times_seqlen_threshold=1000000,
            fp32_residual_connection=model_cfg.fp32_residual_connection,
            params_dtype=model_cfg.params_dtype,
            padded_vocab_size=self.final_padded_vocab_size,  # Use the potentially updated value
            inference_max_seq_length=self.cfg["generation"]["max_new_tokens"],  # type: ignore
            inference_max_requests=self.cfg["generation_batch_size"],
        )

        from megatron.core.inference.contexts.dynamic_context import (
            DynamicInferenceContext,
        )
        from megatron.core.inference.engines.dynamic_engine import (
            DynamicInferenceEngine,
        )
        from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
            GPTInferenceWrapper,
        )
        from megatron.core.inference.sampling_params import SamplingParams

        mcore_generation_config = self.cfg["generation"]["mcore_generation_config"]
        buffer_size_gb = mcore_generation_config.get("buffer_size_gb", 20)

        num_cuda_graphs = mcore_generation_config.get("num_cuda_graphs", 16)
        block_size_tokens = mcore_generation_config.get("block_size_tokens", 256)
        use_cuda_graphs_for_non_decode_steps = mcore_generation_config.get(
            "use_cuda_graphs_for_non_decode_steps", True
        )
        enable_chunked_prefill = mcore_generation_config.get(
            "enable_chunked_prefill", True
        )
        unified_memory_level = mcore_generation_config.get("unified_memory_level", 0)
        buffer_guaranteed_fraction = mcore_generation_config.get(
            "buffer_guaranteed_fraction", 0.1
        )
        max_tokens = mcore_generation_config.get("max_tokens", 16384)

        # Get the underlying language module (not the DDP wrapper) for toggle_cuda_graphs.
        # This follows the pattern from megatron/rl/rl_utils.py megatron_rl_inference_mode().
        # The DDP wrapper can interfere with CUDA graph state management.
        lang_module = (
            self.model.module.module
            if hasattr(self.model.module, "module")
            else self.model.module
        )

        # Enable CUDA graphs for inference using toggle_cuda_graphs on the underlying module.
        # reset_cuda_graphs=True ensures fresh CUDA graphs are created for each generate() call.
        

        toggle_cuda_graphs(lang_module, set_to="local", reset_cuda_graphs=True)

        model_config = self.model.config
        model_config.cuda_graph_impl = "local"

        dynamic_context = DynamicInferenceContext(
            params_dtype=inference_wrapper_config.params_dtype,
            num_layers=model_config.num_layers,
            kv_channels=model_config.kv_channels,
            num_attention_heads=model_config.num_query_groups,
            max_sequence_length=self.cfg["generation"]["max_new_tokens"],
            buffer_guaranteed_fraction=buffer_guaranteed_fraction,
            buffer_size_gb=buffer_size_gb,
            materialize_only_last_token_logits=False,
            num_cuda_graphs=num_cuda_graphs,
            block_size_tokens=block_size_tokens,
            tensor_model_parallel_size=self.cfg["megatron_cfg"][
                "tensor_model_parallel_size"
            ],
            use_cuda_graphs_for_non_decode_steps=use_cuda_graphs_for_non_decode_steps,
            use_flashinfer_fused_rope=False,
            unified_memory_level=unified_memory_level,
            max_tokens_override=max_tokens,
        )
        inference_wrapped_model = GPTInferenceWrapper(
            self.model, inference_wrapper_config, dynamic_context
        )

        inference_wrapped_model.prep_model_for_inference()
        # Set pipeline parallel flag
        inference_wrapped_model.model_is_pipeline_parallel = (
            self.cfg["megatron_cfg"]["pipeline_model_parallel_size"] > 1
        )

        text_generation_controller = TextGenerationController(
            inference_wrapped_model=inference_wrapped_model,
            tokenizer=self.megatron_tokenizer,
        )

        # Calculate seed based on node and rank to ensure reproducibility across workers
        local_rank = torch.cuda.current_device()  # Local GPU index on the node
        num_gpus_per_node = torch.cuda.device_count()
        node_idx = self.rank // num_gpus_per_node if num_gpus_per_node > 0 else 0
        seed = (node_idx * 1024) + local_rank

        # New API: DynamicInferenceEngine has additional parameters
        dynamic_engine = DynamicInferenceEngine(
            text_generation_controller,
            dynamic_context,
            enable_cuda_graph=True,
            random_seed=seed,
            track_paused_request_events=False,
            enable_chunked_prefill=enable_chunked_prefill,
            inference_logging_step_interval=0,
        )

        # Handle None values for top_k - convert to integer as required by Megatron
        top_k_cfg = self.cfg["generation"]["top_k"]
        top_k_val = 1 if greedy else (int(top_k_cfg) if top_k_cfg is not None else 0)

        top_p_cfg = self.cfg["generation"]["top_p"]
        top_p_val = (
            0.0 if greedy else (float(top_p_cfg) if top_p_cfg is not None else 0.0)
        )

        # New API: SamplingParams now includes termination_id and uses num_tokens_total
        sampling_params = SamplingParams(
            temperature=self.cfg["generation"]["temperature"] if not greedy else 0,
            top_k=top_k_val,
            top_p=top_p_val,
            skip_prompt_log_probs=False,
            return_log_probs=True,
            num_tokens_total=self.cfg["generation"]["max_new_tokens"],
            num_tokens_to_generate=None,
            termination_id=self.megatron_tokenizer.eod,
        )

        input_ids = data["input_ids"]
        prompt_tokens_tensor = input_ids.cuda()
        prompt_lengths_tensor = data["input_lengths"]
        request_id = 0

        # New API: add_request now takes sampling_params as a parameter
        for p, prompt_len in zip(
            prompt_tokens_tensor, prompt_lengths_tensor, strict=True
        ):
            dynamic_engine.add_request(
                request_id,
                p[:prompt_len],
                sampling_params=sampling_params,
            )
            request_id += 1

        result = []
        import time
        start = time.time()
        while dynamic_engine.has_unfinished_requests():
            result_step = dynamic_engine.step_modern(verbose=False)
            finished_requests = result_step.get("finished_requests", [])
            for finished_request in finished_requests:
                result.append(finished_request)
        end = time.time()
        print(f"SHAN : Time taken: {end - start} seconds")

        # Sort results by request_id to maintain original batch order
        result.sort(key=lambda x: x.request_id)

        out = {
            "tokens": [x.prompt_tokens.tolist() + x.generated_tokens for x in result],
            "logprobs": [x.prompt_log_probs + x.generated_log_probs for x in result],
        }

        input_lengths = data["input_lengths"]
        # pad the out "tokens" and "logprobs" and make them into tensors from lists
        batch_size = data["input_ids"].size(0)
        max_gen_seq_len = max([len(x.generated_tokens) for x in result])
        padded_input_length = input_ids.size(1)

        max_seq_len = padded_input_length + max_gen_seq_len
        # Create padded tensors for tokens and logprobs
        output_ids_padded = torch.full(
            (batch_size, max_seq_len),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=data["input_ids"].device,
        )

        logprobs_padded = torch.zeros(
            (batch_size, max_seq_len),
            dtype=torch.float,
            device=data["input_ids"].device,
        )

        # Fill in the padded tensors with actual values
        generation_lengths = torch.zeros(
            batch_size, dtype=torch.long, device=data["input_ids"].device
        )
        unpadded_sequence_lengths = torch.zeros(
            batch_size, dtype=torch.long, device=data["input_ids"].device
        )
        for i in range(batch_size):
            seq_len = len(out["tokens"][i])
            output_ids_padded[i, :seq_len] = torch.tensor(
                out["tokens"][i], dtype=torch.long, device=data["input_ids"].device
            )
            generation_lengths[i] = seq_len - input_lengths[i].item()
            unpadded_sequence_lengths[i] = seq_len
            logprob_len = len(out["logprobs"][i])
            logprobs_padded[i, 1 : logprob_len + 1] = torch.tensor(
                out["logprobs"][i],
                dtype=torch.float,
                device=data["input_ids"].device,
            )

        out_dict = {
            "output_ids": output_ids_padded,
            "logprobs": logprobs_padded,
            "generation_lengths": generation_lengths,
            "unpadded_sequence_lengths": unpadded_sequence_lengths,
        }

        self.model.config.flash_decode = False

        # Disable CUDA graphs after inference using toggle_cuda_graphs on the underlying module.
        # This follows the pattern from megatron/rl/rl_utils.py megatron_rl_inference_mode().
        # reset_cuda_graphs=True ensures:
        # 1. cudagraph_manager attributes are deleted from modules
        # 2. _CudagraphGlobalRecord state is reset via delete_cuda_graphs()
        # 3. Model is ready for training (avoids "inference tensor" errors)
        # 4. Next generate() call will create fresh CUDA graphs with new DynamicInferenceContext
        toggle_cuda_graphs(lang_module, set_to="none", reset_cuda_graphs=True)

        no_grad.__exit__(None, None, None)

        return BatchedDataDict.from_batches([out_dict]).to("cpu")

    @torch.no_grad()
    @wrap_with_nvtx_name("megatron_policy_worker/prepare_refit_info")
    def prepare_refit_info(self) -> None:
        """Prepare state dict metadata for weight refitting and IPC streaming."""
        self.refit_param_info_mcore = self._calculate_refit_param_info()

        # Collect tensor metadata for refit / hf side info
        refit_param_info_hf = {}
        # Reuse shared iterator that appends FP8 KV/Q scales when enabled
        for name, tensor in self._iter_params_with_optional_kv_scales():
            refit_param_info_hf[name] = (tensor.shape, tensor.dtype)

        return refit_param_info_hf

    def _calculate_refit_param_info(self) -> list[tuple[str, int]]:
        """Calculate parameter information for refit.

        Each task contains:
        - param_name: Local parameter name without module prefixes
        - mapping: MegatronParamMapping instance for weight transformation
        - pp_rank: Pipeline-parallel rank owning the parameter
        - vp_stage: Virtual-pipeline stage index
        - megatron_module: Reference to Megatron model/submodule
        - param_weight: Target parameter tensor for converted weight

        Returns:
            List of (parameter_name, size_in_bytes) tuples.
        """
        self.refit_conversion_tasks = self.megatron_bridge.get_conversion_tasks(
            [self.model]
        )
        param_info = []

        def calculate_size_in_bytes(param, tp_size, ep_size):
            if param is None:
                # need to broadcast for other pp ranks
                size_in_bytes = None
            else:
                # Calculate size for this parameter
                prec_to_bytes = {
                    torch.bfloat16: 2,
                    torch.float16: 2,
                    torch.float32: 4,
                }
                scale = prec_to_bytes[self.dtype] / prec_to_bytes[param.dtype]
                size_in_bytes = (
                    param.element_size() * param.numel() * tp_size * ep_size * scale
                )

            # Broadcast size_in_bytes across pipeline parallel ranks
            return broadcast_object_across_pp_ranks(size_in_bytes)

        for task in self.refit_conversion_tasks:
            param_info.append(
                (
                    task.param_name,
                    calculate_size_in_bytes(
                        task.param_weight,
                        task.mapping.tp_size,
                        task.mapping.ep_size if task.mapping.is_expert else 1,
                    ),
                )
            )
        return param_info

    def _iter_params_with_optional_kv_scales(
        self,
        kv_scales: Optional[dict[str, float]] = None,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield exported HF parameters and optionally append FP8 KV/Q scale tensors.

        This helper is used by both IPC-based streaming and collective broadcast
        so that the logic for adding KV scales stays consistent in one place.
        """
        base_iter = self.megatron_bridge.export_hf_weights(
            [self.model],
            show_progress=False,
            conversion_tasks=self.refit_conversion_tasks,  # used for metadata caching
        )

        # Yield the original parameters first.
        for name, tensor in base_iter:
            yield name, tensor

        # Check whether FP8 KV cache is enabled.
        use_fp8_kv_cache = False
        if (
            "generation" in self.cfg
            and self.cfg["generation"] is not None
            and self.cfg["generation"]["backend"] == "vllm"
        ):
            generation_cfg = cast(VllmConfig, self.cfg["generation"])
            use_fp8_kv_cache = (
                "vllm_cfg" in generation_cfg
                and "kv_cache_dtype" in generation_cfg["vllm_cfg"]
                and generation_cfg["vllm_cfg"]["kv_cache_dtype"].startswith("fp8")
            )

        if not use_fp8_kv_cache:
            return

        # Append KV (and potentially Q) scale entries to match metadata.
        num_layers = self.megatron_bridge.transformer_config.num_layers
        keys: list[str] = []
        for layer_idx in range(num_layers):
            scale_names = get_vllm_qkv_scale_names(layer_idx)
            keys.extend(scale_names.values())

        for param_name in keys:
            if kv_scales and param_name in kv_scales:
                scale_value = kv_scales[param_name]
            else:
                scale_value = 1.0
            scale_tensor = torch.tensor(
                scale_value, dtype=torch.float32, device="cuda"
            ).reshape(1)
            yield param_name, scale_tensor

    @torch.no_grad()
    @wrap_with_nvtx_name("megatron_policy_worker/stream_weights_via_ipc_zmq")
    def stream_weights_via_ipc_zmq(
        self, buffer_size_bytes: int = 0, kv_scales: Optional[dict[str, float]] = None
    ) -> None:
        """Stream model weights to peer process via ZMQ IPC socket."""
        self.maybe_init_zmq()

        from nemo_rl.models.policy.utils import stream_weights_via_ipc_zmq_impl

        # Use the shared implementation to append optional KV scales.
        stream_weights_via_ipc_zmq_impl(
            params_generator=self._iter_params_with_optional_kv_scales(
                kv_scales=kv_scales
            ),
            buffer_size_bytes=buffer_size_bytes,
            zmq_socket=self.zmq_socket,
            rank=self.rank,
            worker_name=str(self),
        )

    @torch.no_grad()
    def broadcast_weights_for_collective(
        self, kv_scales: Optional[dict[str, float]] = None
    ) -> None:
        """Broadcast the weights for collective communication."""
        # param_iterator will return (name, tensor), we only need tensor.
        packed_broadcast_producer(
            iterator=self._iter_params_with_optional_kv_scales(kv_scales=kv_scales),
            group=self.model_update_group,
            src=0,
            post_iter_func=lambda x: x[1],
        )

    def prepare_for_lp_inference(self):
        self.model = self.move_model(self.model, "cuda", move_grads=False)
        self.model.eval()

        # offload grads to cpu
        self.model = self.move_model(
            self.model, "cpu", move_params=False, move_grads=True
        )  # get rid of grad buffers

        # offload optimizer to cpu
        torch.randn(1).cuda()  # wake up torch allocator
        if (
            hasattr(self, "optimizer")
            and self.optimizer is not None
            and not self.optimizer_cpu_offload
            and self.offload_optimizer_for_logprob
        ):
            self.move_optimizer("cpu")

        gc.collect()
        torch.cuda.empty_cache()

    def prepare_for_training(self, *args, **kwargs):
        # onload models and optimizer state to cuda
        self.model = self.move_model(
            self.model, "cuda", move_grads=True, move_params=True
        )
        self.model.train()

        # Move optimizer state to CUDA if it exists
        # colocated generation will always offload optimizer to cuda before refit
        if (
            hasattr(self, "optimizer")
            and self.optimizer is not None
            and not self.optimizer_cpu_offload
            and (self.offload_optimizer_for_logprob or self.is_generation_colocated)
        ):
            self.move_optimizer("cuda")

        if self.cfg["megatron_cfg"]["empty_unused_memory_level"] >= 1:
            torch.cuda.empty_cache()

    @wrap_with_nvtx_name("megatron_policy_worker/offload_before_refit")
    def offload_before_refit(self):
        """Offload the optimizer and buffers to the CPU."""
        no_grad = torch.no_grad()
        no_grad.__enter__()
        allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
        reserved = torch.cuda.memory_reserved() / (1024**3)  # Convert to GB
        print(
            f"GPU Memory before optimizer offload: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )
        self.model = self.move_model(
            self.model, "cpu", move_params=False, move_grads=True
        )  # get rid of grad buffers
        torch.randn(1).cuda()  # wake up torch allocator
        if (
            hasattr(self, "optimizer")
            and self.optimizer is not None
            and not self.optimizer_cpu_offload
        ):
            self.move_optimizer("cpu")

        gc.collect()
        torch.cuda.empty_cache()

        # Print memory stats after offloading
        allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
        reserved = torch.cuda.memory_reserved() / (1024**3)  # Convert to GB
        print(
            f"GPU Memory after optimizer offload: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )
        no_grad.__exit__(None, None, None)

    @wrap_with_nvtx_name("megatron_policy_worker/offload_after_refit")
    def offload_after_refit(self):
        """Offload as much as possible on the CPU."""
        no_grad = torch.no_grad()
        no_grad.__enter__()
        self.model = self.move_model(self.model, "cpu")
        self.model.eval()
        torch.randn(1).cuda()  # wake up torch allocator
        self.offload_before_refit()  # rerun the old offload function

        allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
        reserved = torch.cuda.memory_reserved() / (1024**3)  # Convert to GB
        print(
            f"GPU Memory after refit complete: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )
        no_grad.__exit__(None, None, None)

    @torch.no_grad()
    def move_model(
        self,
        model: torch.nn.Module,
        device: str,
        move_params: bool = True,
        move_grads: bool = True,
    ) -> torch.nn.Module:
        # move all param and grad buffers to the device
        if isinstance(model, DistributedDataParallel):
            # DDP case
            for buffers in [model.buffers, model.expert_parallel_buffers]:
                for buffer_idx in range(len(buffers)):
                    if device == "cpu":
                        buffers[buffer_idx].offload_to_cpu(
                            move_params=move_params, move_grads=move_grads
                        )
                    elif device == "cuda":
                        buffers[buffer_idx].reload_from_cpu(
                            move_params=move_params, move_grads=move_grads
                        )
                    else:
                        raise ValueError(
                            f"Invalid device: {device}. Only strings 'cpu' and 'cuda' are supported."
                        )
        elif isinstance(model, custom_FSDP):
            if device == "cpu":
                model.param_and_grad_buffer.offload_to_cpu(move_params, move_grads)
            elif device == "cuda":
                model.param_and_grad_buffer.reload_from_cpu(
                    move_params=move_params, move_grads=move_grads
                )
            else:
                raise ValueError(
                    f"Invalid device: {device}. Only strings 'cpu' and 'cuda' are supported."
                )
        else:
            # Ordinary offload case
            if move_params:
                for name, param in model.state_dict().items():
                    new_state_dict = {}
                    for name, item in model.state_dict().items():
                        if isinstance(item, torch.Tensor):
                            item = item.detach().to(
                                device=device, non_blocking=True, copy=True
                            )
                        new_state_dict[name] = item
                    model.load_state_dict(new_state_dict)
        return model

    def move_optimizer(self, device: str):
        # Iterate through the state dictionaries for each parameter group
        if isinstance(self.optimizer, ChainedOptimizer):
            optimizer_state = self.optimizer.state
        else:
            optimizer_state = self.optimizer._get_state()
        for _, state in optimizer_state.items():
            # Iterate through the state items (e.g., momentum, variance) for a parameter
            for k, v in state.items():
                # Check if the item is a tensor
                if torch.is_tensor(v):
                    # Move the tensor to device and update the state dictionary
                    if device == "cpu":
                        if v.is_cuda:
                            state[k] = v.to("cpu")
                    elif device == "cuda":
                        if not v.is_cuda:
                            state[k] = v.to("cuda")
                    else:
                        raise ValueError(
                            f"Invalid device: {device}. Only strings 'cpu' and 'cuda' are supported."
                        )

    def save_checkpoint(
        self,
        weights_path: str,
        optimizer_path: Optional[str] = None,
        **kwargs,
    ):
        """Save a training checkpoint.

        Args:
            weights_path: The specific directory path where the checkpoint will be saved.
            optimizer_path: If not None, optimizer and scheduler states are saved if they exist.
        """
        if not torch.distributed.is_initialized():
            raise RuntimeError(
                "Distributed process group is not initialized. Cannot save checkpoint."
            )

        if self.mcore_state is None or self.model is None:
            raise RuntimeError(
                "Megatron core state or model is not initialized. Cannot save checkpoint."
            )

        original_save_path = self.mcore_state.cfg.checkpoint.save
        # save_dir = os.path.dirname(weights_path)
        release_name = os.path.basename(weights_path)

        try:
            maybe_finalize_async_save(
                self.mcore_state,
                ckpt_cfg=self.mcore_state.cfg.checkpoint,
                blocking=False,
            )
            self.mcore_state.cfg.checkpoint.save = weights_path

            optimizer_to_save = None
            scheduler_to_save = None

            if optimizer_path is not None:
                if self.optimizer is not None:
                    optimizer_to_save = self.optimizer
                if self.scheduler is not None:
                    scheduler_to_save = self.scheduler

            # Ensure model is in eval mode for consistent saving, unless actively training
            # This is a common practice, though NeMo's save might handle this.
            # For safety, if not in training loop, setting to eval.
            is_training = self.model.training
            if not is_training:
                self.model.eval()

            if self.should_disable_forward_pre_hook:
                self.disable_forward_pre_hook()
            save_checkpoint(
                state=self.mcore_state,
                model=[self.model],
                optimizer=optimizer_to_save,
                opt_param_scheduler=scheduler_to_save,
                num_floating_point_operations_so_far=self.mcore_state.train_state.floating_point_operations_so_far,
                checkpointing_context=self.checkpointing_context,
            )
            print(f"Saved checkpoint to {weights_path}")
            maybe_finalize_async_save(
                self.mcore_state,
                ckpt_cfg=self.mcore_state.cfg.checkpoint,
                blocking=True,
                terminate=True,
            )
            if self.should_disable_forward_pre_hook:
                self.enable_forward_pre_hook()

            if not is_training:  # Restore training state if it was changed
                self.model.train()

        except Exception as e:
            print(f"Failed to save checkpoint to {weights_path}: {e}")
            raise
        finally:
            self.mcore_state.cfg.checkpoint.save = original_save_path

    def load_checkpoint(self, weights_path: str, optimizer_path: Optional[str] = None):
        """Load a training checkpoint.

        Args:
            weights_path: The exact directory path from which to load the checkpoint.
            optimizer_path: If not None, attempts to load optimizer and scheduler states
                            if self.optimizer and self.scheduler are initialized.
        """
        raise NotImplementedError(
            "Loading checkpoints outside of the init function is not yet implemented for Megatron policy."
        )

    def check_tensor_parallel_attributes(self) -> dict[str, Any]:
        """Check tensor parallel attributes on model parameters.

        Returns:
            Dictionary containing information about tensor parallel parameters:
            - tp_params: List of parameter names that have tensor_model_parallel=True
            - non_tp_params: List of parameter names that have tensor_model_parallel=False
            - total_params: Total number of parameters checked
            - tp_size: Tensor parallel size from config
        """
        tp_params = []
        non_tp_params = []
        total_params = 0

        for name, param in self.model.named_parameters():
            total_params += 1
            tensor_model_parallel = getattr(param, "tensor_model_parallel", False)

            if tensor_model_parallel:
                tp_params.append(
                    {
                        "name": name,
                        "tensor_model_parallel": tensor_model_parallel,
                        "partition_dim": getattr(param, "partition_dim", None),
                        "partition_stride": getattr(param, "partition_stride", None),
                        "shape": list(param.shape),
                    }
                )
            else:
                non_tp_params.append(
                    {
                        "name": name,
                        "tensor_model_parallel": tensor_model_parallel,
                        "shape": list(param.shape),
                    }
                )

        return {
            "tp_params": tp_params,
            "non_tp_params": non_tp_params,
            "total_params": total_params,
            "tp_size": self.megatron_cfg.model.tensor_model_parallel_size,
        }

    @torch.no_grad()
    def calibrate_qkv_fp8_scales(
        self,
        *,
        data: BatchedDataDict[Any],
        micro_batch_size: Optional[int] = None,
        percentile: float = 99.9,
        margin: float = 1.05,
        include_q: bool = False,
    ) -> dict[str, Any]:
        """One-shot calibration of Q/K/V activation scales (for FP8 KV cache).

        - Captures each layer's `query_key_value` output through forward hooks, splits Q/K/V, and computes percentile amax.
        - In parallel (DP/TP/PP) environments, first computes local percentiles, then takes max across all ranks for conservativeness.
        - By default only returns and saves K/V scales, optionally returns Q.

        Args:
            data: Representative sample batch for calibration, following get_logprobs input conventions.
            micro_batch_size: Micro batch size during calibration; if None, reuses logprob_batch_size.
            percentile: Percentile for amax (e.g. 99.9).
            margin: Margin factor, e.g. 1.05.
            save_path: If provided, rank0 will save results as JSON.
            include_q: Whether to also return Q scale (usually only K/V needed).

        Returns:
            { "format": "fp8", "percentile": float, "margin": float,
              "layers": { layer_name: {"k_scale": float, "v_scale": float[, "q_scale": float] } } }
        """

        # Allow overriding FP8 max for Q, K, V via environment variables for ease of testing.
        # Defaults align with FP8 e4m3 max magnitude.
        # Use different defaults for Q, K, V to adapt to distribution diffefences
        def _get_env_float(name: str, default: float) -> float:
            try:
                val = os.getenv(name, None)
                return float(val) if val is not None and val != "" else default
            except Exception:
                return default

        FP8_MAX_Q = _get_env_float("FP8_MAX_Q", 448.0)
        FP8_MAX_K = _get_env_float("FP8_MAX_K", 448.0)
        FP8_MAX_V = _get_env_float("FP8_MAX_V", 448.0)

        self.model.eval()

        # Record local percentile amax for q/k/v of each layer
        layer_to_samples_q: dict[str, list[float]] = defaultdict(list)
        layer_to_samples_k: dict[str, list[float]] = defaultdict(list)
        layer_to_samples_v: dict[str, list[float]] = defaultdict(list)
        hook_handles = []

        def _extract_layer_key(module_name: str) -> str:
            # Expected format: "module.decoder.layers.<idx>.self_attention.query_key_value"
            m = re.search(r"module\.decoder\.layers\.(\d+)", module_name)
            if m is not None:
                return f"layer_{m.group(1)}"
            return module_name

        # Hook to capture q/k/v after q/k norm and RoPE
        def _pre_hook_builder_core_attention(module_name: str):
            layer_key = _extract_layer_key(module_name)

            def _pre_hook(module, inputs):
                args = inputs if isinstance(inputs, (tuple, list)) else (inputs,)
                if len(args) == 1 and isinstance(args[0], (tuple, list)):
                    args = args[0]
                # Expected first 3 args to be q, k, v (typical signature for Megatron CoreAttention)
                q = args[0]
                k = args[1]
                v = args[2]
                if include_q:
                    layer_to_samples_q[layer_key].append(
                        float(torch.amax(torch.abs(q)).item())
                    )
                layer_to_samples_k[layer_key].append(
                    float(torch.amax(torch.abs(k)).item())
                )
                layer_to_samples_v[layer_key].append(
                    float(torch.amax(torch.abs(v)).item())
                )

            return _pre_hook

        matched_modules = []
        # Try to register forward_pre_hook on core_attention first
        for name, module in self.model.named_modules():
            if "self_attention.core_attention" in name:
                try:
                    handle = module.register_forward_pre_hook(
                        _pre_hook_builder_core_attention(name)
                    )
                    hook_handles.append(handle)
                    matched_modules.append((name, module.__class__.__name__, "pre"))
                except Exception as e:
                    print(
                        f"Error registering pre-hook for qkv scale calibration on {name}: {e}"
                        " Please check if the model is compatible with the current calibration logic. "
                        "The expected module name is 'self_attention.core_attention'."
                    )
                    raise

        # Run a forward pass to trigger hooks (reuse get_logprobs forward path)
        try:
            _ = self.get_logprobs(data=data, micro_batch_size=micro_batch_size)
        finally:
            for h in hook_handles:
                try:
                    h.remove()
                except Exception as e:
                    print(f"Error removing hook for qkv scale calibration: {e}")
                    raise

        # Compute local percentile amax
        def _percentile(values: list[float], p: float) -> float:
            if not values:
                return 0.0
            t = torch.tensor(sorted(values), device="cuda", dtype=torch.float32)
            rank = max(
                0, min(len(values) - 1, int(round((p / 100.0) * (len(values) - 1))))
            )
            return float(t[rank].item())

        local_layer_to_pamax = {}
        for layer_key in set(
            list(layer_to_samples_k.keys())
            + list(layer_to_samples_v.keys())
            + (list(layer_to_samples_q.keys()) if include_q else [])
        ):
            entry = {}
            if include_q:
                entry["q_amax_p"] = _percentile(
                    layer_to_samples_q.get(layer_key, []), percentile
                )
            entry["k_amax_p"] = _percentile(
                layer_to_samples_k.get(layer_key, []), percentile
            )
            entry["v_amax_p"] = _percentile(
                layer_to_samples_v.get(layer_key, []), percentile
            )
            local_layer_to_pamax[layer_key] = entry

        # Merge across all ranks: take maximum of percentile amax (conservative approach)
        world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )
        gathered = [None for _ in range(world_size)] if world_size > 1 else None
        if world_size > 1:
            torch.distributed.all_gather_object(gathered, local_layer_to_pamax)
            merged = defaultdict(dict)
            for d in gathered:  # type: ignore
                if d is None:
                    continue
                for k, v in d.items():
                    dst = merged[k]
                    for kk, vv in v.items():
                        dst[kk] = max(dst.get(kk, 0.0), float(vv))
            layer_to_pamax = dict(merged)
        else:
            layer_to_pamax = local_layer_to_pamax

        # Compute scale (symmetric quantization): scale = pamax / fp8_max
        result_layers = {}
        for layer_key, vals in layer_to_pamax.items():
            out_entry = {}
            if include_q:
                q_scale = (vals.get("q_amax_p", 0.0) * margin) / FP8_MAX_Q
                out_entry["q_scale"] = float(q_scale)
            k_scale = (vals.get("k_amax_p", 0.0) * margin) / FP8_MAX_K
            v_scale = (vals.get("v_amax_p", 0.0) * margin) / FP8_MAX_V
            out_entry["k_scale"] = float(k_scale)
            out_entry["v_scale"] = float(v_scale)
            result_layers[layer_key] = out_entry

        vllm_format_scales = convert_calibration_to_vllm_format(result_layers)

        final_result = {
            "format": "fp8",
            "percentile": percentile,
            "margin": margin,
            "layers": vllm_format_scales,
        }

        # Sync results across all ranks (broadcast rank0's result)
        if world_size > 1:
            if torch.distributed.get_rank() == 0:
                obj_list = [final_result]
                torch.distributed.broadcast_object_list(obj_list, src=0)
                final_result = obj_list[0]
            else:
                obj_list = [None]
                torch.distributed.broadcast_object_list(obj_list, src=0)
                final_result = obj_list[0]  # type: ignore

        return final_result


class CustomFloat16Module(Float16Module):
    """Float 16 Module.

    Attributes:
        config (TransformerConfig): Transformer config
        fp16 (bool) : Specifies if the model runs in fp16 mode
        bf16 (bool) : Specifies if the model runs in bf16 mode

    Args:
        config (TransformerConfig): The transformer config used to initalize the model
    """

    def __init__(self, config: TransformerConfig, module: torch.nn.Module):
        super(CustomFloat16Module, self).__init__(config, module)
        self.re_enable_float32_expert_bias()

    def re_enable_float32_expert_bias(self) -> None:
        """Ensure MoE router expert bias stays in float32 for numerical stability.

        Walks the wrapped module to find MoE routers and invokes the
        `_maintain_float32_expert_bias()` helper which recreates or casts the
        expert bias tensors to float32 as required by Megatron-LM.
        """
        module = self.module
        # Handle VLM models where language model is nested
        if hasattr(module, "language_model"):
            module = module.language_model
        if hasattr(module, "decoder") and hasattr(module.decoder, "layers"):
            for layer in module.decoder.layers:
                mlp = getattr(layer, "mlp", None)
                router = getattr(mlp, "router", None) if mlp is not None else None
                if router is not None and hasattr(
                    router, "_maintain_float32_expert_bias"
                ):
                    router._maintain_float32_expert_bias()

```

```python
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
import gc
import os
import re
import time
import warnings
from collections import defaultdict
from contextlib import AbstractContextManager, contextmanager, nullcontext
from functools import partial
from typing import Any, Iterator, Optional, TypeVar, cast

import ray
import torch
from megatron.bridge import AutoBridge
from megatron.bridge.models.model_provider import get_model
from megatron.bridge.training import fault_tolerance
from megatron.bridge.training.checkpointing import (
    checkpoint_exists,
    init_checkpointing_context,
    load_checkpoint,
    maybe_finalize_async_save,
    save_checkpoint,
)
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    LoggerConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainingConfig,
)
from megatron.bridge.training.initialize import (
    initialize_megatron,
    set_jit_fusion_options,
)
from megatron.bridge.training.optim import setup_optimizer
from megatron.bridge.training.setup import (
    _update_model_config_funcs,
)
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer
from megatron.bridge.training.utils.train_utils import (
    logical_and_across_model_parallel_group,
    reduce_max_stat_across_model_parallel_group,
)
from megatron.bridge.utils.common_utils import get_rank_safe
from megatron.bridge.utils.instantiate_utils import InstantiationMode
from megatron.bridge.utils.vocab_utils import calculate_padded_vocab_size
from megatron.core import parallel_state
from megatron.core.transformer.utils import toggle_cuda_graphs
from megatron.core.distributed import DistributedDataParallel
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import (
    FullyShardedDataParallel as custom_FSDP,
)
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import (
    InferenceWrapperConfig,
)
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.models.gpt import GPTModel
from megatron.core.optimizer import ChainedOptimizer
from megatron.core.parallel_state import (
    get_context_parallel_group,
    get_context_parallel_rank,
    get_pipeline_model_parallel_group,
    get_pipeline_model_parallel_last_rank,
    get_pipeline_model_parallel_world_size,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    is_pipeline_last_stage,
)
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.utils import get_ltor_masks_and_position_ids
from ray.util.queue import Queue
from transformers import PreTrainedTokenizerBase

from nemo_rl.algorithms.interfaces import LossFunction, LossType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.model_utils import (
    allgather_cp_sharded_tensor,
    distributed_vocab_topk,
    from_parallel_logits_to_logprobs,
    from_parallel_logits_to_logprobs_packed_sequences,
)
from nemo_rl.distributed.named_sharding import NamedSharding
from nemo_rl.models.generation.fp8 import (
    convert_calibration_to_vllm_format,
    get_vllm_qkv_scale_names,
)
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationOutputSpec,
    verify_right_padding,
)
from nemo_rl.models.generation.vllm.config import VllmConfig
from nemo_rl.models.megatron.common import (
    _get_pack_sequence_parameters_for_megatron,
    _pack_sequences_for_megatron,
    broadcast_tensor,
    forward_step_arbitrary_loss,
    get_moe_metrics,
)
from nemo_rl.models.megatron.community_import import import_model_from_hf_name
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.interfaces import (
    ColocatablePolicyInterface,
    LogprobOutputSpec,
)
from nemo_rl.models.policy.utils import (
    configure_dynamo_cache,
    get_megatron_checkpoint_dir,
    get_runtime_env_for_policy_worker,
)
from nemo_rl.models.policy.workers.base_policy_worker import AbstractPolicyWorker
from nemo_rl.models.policy.workers.patches import apply_transformer_engine_patch
from nemo_rl.utils.nsys import wrap_with_nvtx_name
from nemo_rl.utils.packed_tensor import packed_broadcast_producer

try:
    from megatron.core.distributed import (
        TorchFullyShardedDataParallel as torch_FSDP,  # noqa: F401 unused-import
    )

    HAVE_FSDP2 = True
except ImportError:
    HAVE_FSDP2 = False

TokenizerType = TypeVar("TokenizerType", bound=PreTrainedTokenizerBase)


def broadcast_object_across_pp_ranks(obj):
    """Broadcast an object across pipeline parallel ranks.

    This utility function handles broadcasting an object from the rank that owns it
    to all other pipeline parallel ranks. If only one rank has the object (non-None),
    it will be broadcast to all other ranks.

    Args:
        obj: The object to broadcast. Can be None on ranks that don't own it.

    Returns:
        The object on all ranks (either the original or the broadcast copy).

    Raises:
        ValueError: If the object doesn't exist on any pipeline parallel rank.
    """
    pp_size = get_pipeline_model_parallel_world_size()
    pp_group = get_pipeline_model_parallel_group()

    if pp_size == 1:
        return obj

    # ------------------------------------------------------------------
    # 1. Gather presence flags from all PP ranks to find the source rank
    # ------------------------------------------------------------------
    has_obj = obj is not None
    obj_flags = [None] * pp_size
    torch.distributed.all_gather_object(obj_flags, has_obj, group=pp_group)

    # ------------------------------------------------------------------
    # 2. Identify the owning rank (the only rank with True flag)
    # ------------------------------------------------------------------
    src_rank = None  # Rank *inside* the PP group
    for rank, flag in enumerate(obj_flags):
        if flag:
            src_rank = rank
            break

    if src_rank is None:
        raise ValueError("Object must exist on at least one PP rank")

    # ------------------------------------------------------------------
    # 3. Broadcast the object from the source rank to all ranks
    # ------------------------------------------------------------------
    # Use broadcast_object_list which is more robust than all_gather_object
    obj_list = [obj]
    pp_ranks = torch.distributed.get_process_group_ranks(pp_group)
    global_src = pp_ranks[src_rank]
    torch.distributed.broadcast_object_list(obj_list, src=global_src, group=pp_group)

    return obj_list[0]


def setup_megatron_model(
    policy_cfg: PolicyConfig,
    cfg: ConfigContainer,
    load_optimizer: bool = True,
    get_embedding_ranks=None,  # TODO @sahilj: What is this?
    get_position_embedding_ranks=None,
):
    state = GlobalState()
    state.cfg = cfg
    # TODO: Freeze state.cfg

    cfg.dist.external_gpu_device_mapping = True
    initialize_megatron(
        cfg=cfg,
        get_embedding_ranks=get_embedding_ranks,
        get_position_embedding_ranks=get_position_embedding_ranks,
    )

    if cfg.ft and cfg.ft.enable_ft_package:
        fault_tolerance.setup(cfg, state)
        fault_tolerance.maybe_setup_simulated_fault(cfg.ft)

    # Set pytorch JIT layer fusion options and warmup JIT functions.
    set_jit_fusion_options(cfg.model, cfg.train.micro_batch_size)

    # Adjust the startup time so it reflects the largest value.
    # This will be closer to what scheduler will see (outside of
    # image ... launches.
    start_time_tensor = torch.tensor(
        [state.start_time], dtype=torch.double, device="cuda"
    )
    torch.distributed.all_reduce(start_time_tensor, op=torch.distributed.ReduceOp.MIN)
    state.start_time = start_time_tensor.item()

    print(
        "time to initialize megatron (seconds): {:.3f}".format(
            time.time() - state.start_time
        )
    )
    torch.distributed.barrier()

    # Context used for persisting some state between checkpoint saves.
    checkpointing_context = init_checkpointing_context(cfg.checkpoint)

    # Tokenizer
    build_tokenizer(
        cfg.tokenizer,
        make_vocab_size_divisible_by=cfg.model.make_vocab_size_divisible_by
        // cfg.model.tensor_model_parallel_size,
        tensor_model_parallel_size=cfg.model.tensor_model_parallel_size,
        trust_remote_code=True,
    )
    assert cfg.model.vocab_size, "vocab size must be specified in model config"

    torch.distributed.barrier()

    pre_wrap_hook = []
    mixed_precision_wrapper = Float16Module
    if policy_cfg["megatron_cfg"]["freeze_moe_router"]:

        def freeze_moe_router(megatron_model):
            if not isinstance(megatron_model, list):
                megatron_model = [megatron_model]
            for model_module in megatron_model:
                # Handle both wrapped (Float16Module) and unwrapped models
                if isinstance(model_module, Float16Module):
                    model_module = model_module.module
                # Handle VLM models
                if hasattr(model_module, "language_model"):
                    model_module = model_module.language_model
                for layer in model_module.decoder.layers:
                    if hasattr(layer, "mlp") and hasattr(layer.mlp, "router"):
                        layer.mlp.router.weight.requires_grad = False

        mixed_precision_wrapper = CustomFloat16Module
        pre_wrap_hook.extend([freeze_moe_router])

    # Model, optimizer, and learning rate.
    model = get_model(
        cfg.model,
        cfg.ddp,
        use_torch_fsdp2=cfg.dist.use_torch_fsdp2,
        overlap_param_gather_with_optimizer_step=cfg.optimizer.overlap_param_gather_with_optimizer_step,
        data_parallel_random_init=cfg.rng.data_parallel_random_init,
        pre_wrap_hook=pre_wrap_hook,
        mixed_precision_wrapper=mixed_precision_wrapper,
    )
    if load_optimizer:
        optimizer, scheduler = setup_optimizer(
            optimizer_config=cfg.optimizer,
            scheduler_config=cfg.scheduler,
            model=model,
            use_gloo_process_groups=cfg.dist.use_gloo_process_groups,
        )
    else:
        optimizer = None
        scheduler = None

    print("Model, optimizer, and learning rate scheduler built")
    torch.distributed.barrier()

    # Load checkpoint if applicable
    if (
        cfg.checkpoint.load is not None
        or cfg.checkpoint.pretrained_checkpoint is not None
    ) and (
        checkpoint_exists(cfg.checkpoint.load)
        or checkpoint_exists(cfg.checkpoint.pretrained_checkpoint)
    ):
        load_checkpoint(
            state,
            model,
            optimizer,
            scheduler,
            checkpointing_context=checkpointing_context,
            skip_load_to_model_and_opt=HAVE_FSDP2 and cfg.dist.use_torch_fsdp2,
        )
        print("Checkpoint loaded")
    torch.distributed.barrier()

    return state, model, optimizer, scheduler, checkpointing_context


def destroy_parallel_state():
    """Safely destroy parallel state and reset async call tracking.

    This function is called during initialization to clean up temporary distributed
    state from model import operations. Resetting async call tracking ensures that
    when the main Megatron distributed context is created, all ranks start with
    consistent call_idx values for async checkpointing.
    """
    if torch.distributed.is_initialized():
        try:
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()
        except:
            pass  # Ignore errors if already destroyed
    if hasattr(parallel_state, "destroy_model_parallel"):
        try:
            parallel_state.destroy_model_parallel()
        except:
            pass  # Ignore errors if already destroyed

    # Reset async calls queue to prevent call_idx mismatches after distributed context recreation
    try:
        import nemo.tron.utils.async_utils as nemo_async_utils
        from megatron.core.dist_checkpointing.strategies.async_utils import (
            AsyncCallsQueue,
        )

        # Clean up any existing async callers first
        old_call_idx = getattr(nemo_async_utils._async_calls_queue, "call_idx", None)
        num_unfinalized = (
            nemo_async_utils._async_calls_queue.get_num_unfinalized_calls()
        )
        if num_unfinalized > 0:
            print(
                f"[WARNING] Resetting async calls queue with {num_unfinalized} unfinalized calls"
            )
        try:
            nemo_async_utils._async_calls_queue.close()
        except:
            pass  # Ignore errors during cleanup
        # Reset the global async calls queue by creating a new instance
        nemo_async_utils._async_calls_queue = AsyncCallsQueue()
        print(f"[DEBUG] Reset NeMo async calls queue (old call_idx: {old_call_idx})")
    except ImportError:
        pass

    # Also reset the Megatron async calls queue if it exists
    try:
        import megatron.training.async_utils as megatron_async_utils
        from megatron.core.dist_checkpointing.strategies.async_utils import (
            AsyncCallsQueue,
        )

        # Clean up any existing async callers first
        old_call_idx = getattr(
            megatron_async_utils._async_calls_queue, "call_idx", None
        )
        num_unfinalized = (
            megatron_async_utils._async_calls_queue.get_num_unfinalized_calls()
        )
        if num_unfinalized > 0:
            print(
                f"[WARNING] Resetting Megatron async calls queue with {num_unfinalized} unfinalized calls"
            )
        try:
            megatron_async_utils._async_calls_queue.close()
        except:
            pass  # Ignore errors during cleanup
        # Reset the Megatron global async calls queue as well
        megatron_async_utils._async_calls_queue = AsyncCallsQueue()
        print(
            f"[DEBUG] Reset Megatron async calls queue (old call_idx: {old_call_idx})"
        )
    except ImportError:
        pass

    # Reset the third global async_calls instance in base strategy module
    try:
        import megatron.core.dist_checkpointing.strategies.base as base_strategy
        from megatron.core.dist_checkpointing.strategies.async_utils import (
            AsyncCallsQueue,
        )

        # Clean up and reset the global async_calls in base strategy
        old_call_idx = getattr(base_strategy.async_calls, "call_idx", None)
        num_unfinalized = base_strategy.async_calls.get_num_unfinalized_calls()
        if num_unfinalized > 0:
            print(
                f"[WARNING] Resetting base strategy async_calls with {num_unfinalized} unfinalized calls"
            )
        try:
            base_strategy.async_calls.close()
        except:
            pass
        base_strategy.async_calls = AsyncCallsQueue()
        print(f"[DEBUG] Reset base strategy async_calls (old call_idx: {old_call_idx})")
    except ImportError:
        pass


@ray.remote(
    runtime_env=get_runtime_env_for_policy_worker("megatron_policy_worker")
)  # pragma: no cover
class MegatronPolicyWorker(AbstractPolicyWorker, ColocatablePolicyInterface):
    def __repr__(self):
        """Customizes the actor's prefix in the Ray logs.

        This makes it easier to identify which worker is producing specific log messages.
        """
        if torch.distributed.is_initialized():
            return f"{self.__class__.__qualname__}[rank={torch.distributed.get_rank()}]"
        else:
            return f"{self.__class__.__qualname__}"

    def __init__(
        self,
        config: PolicyConfig,
        tokenizer: TokenizerType,
        weights_path: Optional[str] = None,
        optimizer_path: Optional[str] = None,
        init_optimizer: bool = True,
        init_reference_model: bool = True,
        *,
        worker_sharding_annotations: NamedSharding,
        pre_init_communication_queue: Queue,
        **kwargs: Any,
    ):
        apply_transformer_engine_patch()

        self.is_generation_colocated = None
        if "generation" in config and config["generation"] is not None:
            self.is_generation_colocated = config["generation"]["colocated"]["enabled"]

        # Explicitly set NCCL_CUMEM_ENABLE to 1 to avoid the P2P initialization error for PyNCCLCommunicator.
        # See https://github.com/NVIDIA-NeMo/RL/issues/564 for more details.
        if not self.is_generation_colocated:
            os.environ["NCCL_CUMEM_ENABLE"] = "1"

        self.cfg = config
        dtype_map = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }
        self.dtype = dtype_map[self.cfg["precision"]]

        self.optimizer_cpu_offload = self.cfg["megatron_cfg"]["optimizer"][
            "optimizer_cpu_offload"
        ]
        self.offload_optimizer_for_logprob = self.cfg["offload_optimizer_for_logprob"]

        # Reward models are not yet supported with Megatron.
        if "reward_model_cfg" in self.cfg and self.cfg["reward_model_cfg"]["enabled"]:
            raise NotImplementedError(
                "Reward models are not yet supported with the Megatron backend, this issue is "
                "tracked in https://github.com/NVIDIA-NeMo/RL/issues/720"
            )

        # Disable dynamo autotune_local_cache to avoid crash when there's already a cache
        # with different order of node_bundles
        configure_dynamo_cache()

        # cfg["model_name"] is allowed to be either an HF model name or a path to an HF checkpoint
        # check if hf_model_name is a path
        hf_model_name = self.cfg["model_name"]
        # Check if the checkpoint already exists
        hf_model_subdir = hf_model_name
        if os.path.exists(hf_model_name):
            hf_model_subdir = f"model_{hf_model_subdir.replace('/', '_')}"

        pretrained_path = f"{get_megatron_checkpoint_dir()}/{hf_model_subdir}"
        pt_checkpoint_exists = os.path.exists(pretrained_path) and os.path.exists(
            os.path.join(pretrained_path, "iter_0000000")
        )

        # Ensure clean slate before import
        destroy_parallel_state()

        # Set for rank for non-collocated to check which ranks to broadcast from
        self.rank = get_rank_safe()
        # Need to initialize the process group before calling into Megatron-Bridge, otherwise Megatron-Bridge will try to set an incorrect device
        torch.distributed.init_process_group("nccl")
        if pt_checkpoint_exists:
            print(f"Checkpoint already exists at {pretrained_path}. Skipping import.")
        else:
            hf_config_overrides = self.cfg.get("hf_config_overrides", {}) or {}
            import_model_from_hf_name(
                hf_model_name,
                pretrained_path,
                self.cfg["megatron_cfg"],
                **hf_config_overrides,
            )

            if parallel_state.model_parallel_is_initialized():
                print("Reinitializing model parallel after loading model state.")
                parallel_state.destroy_model_parallel()

        pretrained_run_config = os.path.join(
            pretrained_path, "iter_0000000/run_config.yaml"
        )

        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if not os.path.exists(pretrained_run_config):
            raise FileNotFoundError(
                f"Pretrained run config not found at {pretrained_run_config} on rank={get_rank_safe()}. This usually means that the one-time HF->mcore conversion on rank=0 saved to a directory not being mounted on this node. Please check "
            )

        try:
            cfg_from_pretrained = ConfigContainer.from_yaml(
                pretrained_run_config, mode=InstantiationMode.STRICT
            )
        except Exception as e:
            # Add helpful context as a note to the exception
            e.add_note(
                f"\n{'=' * 80}\n"
                f"NOTE: A common cause of this error is when the HF->mcore converted checkpoint is\n"
                f"created with an older version of megatron-bridge.\n"
                f"If this checkpoint is old or was generated by a different code version,\n"
                f"try deleting it and rerunning the code.\n"
                f"The checkpoint will be automatically regenerated with the current version.\n\n"
                f"Checkpoint location: {pretrained_path}\n"
                f"{'=' * 80}"
            )
            raise
        model_cfg = cfg_from_pretrained.model

        cfg_from_pretrained.logger = LoggerConfig()

        model_cfg.tensor_model_parallel_size = self.cfg["megatron_cfg"][
            "tensor_model_parallel_size"
        ]
        model_cfg.pipeline_model_parallel_size = self.cfg["megatron_cfg"][
            "pipeline_model_parallel_size"
        ]
        model_cfg.num_layers_in_first_pipeline_stage = self.cfg["megatron_cfg"][
            "num_layers_in_first_pipeline_stage"
        ]
        model_cfg.num_layers_in_last_pipeline_stage = self.cfg["megatron_cfg"][
            "num_layers_in_last_pipeline_stage"
        ]
        model_cfg.sequence_parallel = self.cfg["megatron_cfg"]["sequence_parallel"]
        model_cfg.context_parallel_size = self.cfg["megatron_cfg"][
            "context_parallel_size"
        ]
        if model_cfg.context_parallel_size > 1:
            assert self.cfg["sequence_packing"]["enabled"], (
                "Sequence Packing must be enabled to use Context Parallelism with MCore"
            )
        model_cfg.expert_tensor_parallel_size = self.cfg["megatron_cfg"][
            "expert_tensor_parallel_size"
        ]
        model_cfg.expert_model_parallel_size = self.cfg["megatron_cfg"][
            "expert_model_parallel_size"
        ]

        # Setting moe_router_dtype to higher precision (e.g. fp64) can improve numerical stability,
        # especially when using many experts.
        model_cfg.moe_router_dtype = self.cfg["megatron_cfg"]["moe_router_dtype"]

        # The below two configs (and "freeze_moe_router") are used to stabilize moe training
        # by preventing updates to the moe router. We found that this is helpful in reducing
        # logprob error during training.

        # Set this to "none" to disable load balancing loss.
        model_cfg.moe_router_load_balancing_type = self.cfg["megatron_cfg"][
            "moe_router_load_balancing_type"
        ]
        # Set this to 0.0 to disable updates to the moe router expert bias
        model_cfg.moe_router_bias_update_rate = self.cfg["megatron_cfg"][
            "moe_router_bias_update_rate"
        ]

        model_cfg.moe_permute_fusion = self.cfg["megatron_cfg"]["moe_permute_fusion"]
        if "layernorm_epsilon" in self.cfg["megatron_cfg"]:
            model_cfg.layernorm_epsilon = self.cfg["megatron_cfg"]["layernorm_epsilon"]

        model_cfg.sequence_parallel = self.cfg["megatron_cfg"]["sequence_parallel"]
        model_cfg.bf16 = self.dtype == torch.bfloat16
        model_cfg.fp16 = self.dtype == torch.float16
        if model_cfg.fp16:
            assert not model_cfg.bf16, "fp16 and bf16 cannot be used together"
            model_cfg.params_dtype = torch.float16
        elif model_cfg.bf16:
            assert not model_cfg.fp16, "fp16 and bf16 cannot be used together"
            model_cfg.params_dtype = torch.bfloat16
        else:
            model_cfg.params_dtype = torch.float32
        model_cfg.pipeline_dtype = dtype_map[self.cfg["megatron_cfg"]["pipeline_dtype"]]
        model_cfg.parallel_output = True
        if self.cfg["megatron_cfg"]["activation_checkpointing"]:
            model_cfg.recompute_granularity = "full"
            model_cfg.recompute_method = "uniform"
            model_cfg.recompute_num_layers = 1
        if not model_cfg.gated_linear_unit:
            assert model_cfg.activation_func is not None, (
                "activation_func must be set if not using gated_linear_unit. This likely "
                "indicates an issue in configuration conversion (e.g. activation func was "
                "a lambda and couldn't be serialized). This is based on this check "
                "https://github.com/NVIDIA/Megatron-LM/blob/1ab876ddc4c1893c76f26d775226a8d1dcdfb3d2/megatron/core/transformer/mlp.py#L174."
            )
        model_cfg.apply_rope_fusion = self.cfg["megatron_cfg"]["apply_rope_fusion"]
        model_cfg.bias_activation_fusion = self.cfg["megatron_cfg"][
            "bias_activation_fusion"
        ]
        fp8_cfg = self.cfg["megatron_cfg"].get("fp8_cfg", None)
        self.fp8_cfg = fp8_cfg
        if fp8_cfg is not None and fp8_cfg.get("enabled", False):
            try:
                model_cfg.fp8 = fp8_cfg["fp8"]
                model_cfg.fp8_recipe = fp8_cfg["fp8_recipe"]
                model_cfg.fp8_param = fp8_cfg["fp8_param"]
            except KeyError as e:
                raise KeyError(f"Missing key in fp8_cfg: {e}")
            if model_cfg.fp8_param:
                warnings.warn(
                    "Setting fp8_param=True sometimes causes NaN token_mult_prob_error, please use with caution. "
                    "Refer to https://github.com/NVIDIA-NeMo/RL/issues/1164 for latest updates with this issue."
                )

        optimizer_cpu_offload = self.cfg["megatron_cfg"]["optimizer"][
            "optimizer_cpu_offload"
        ]
        optimizer_offload_fraction = self.cfg["megatron_cfg"]["optimizer"][
            "optimizer_offload_fraction"
        ]
        if optimizer_cpu_offload:
            # Currently, hybrid optimizer (partly on GPU and partly on CPU) is not supported because it conflicts with the way
            # Nemo-rl handles the optimizer offload/onload between generation and training. So if using CPU optimizer the offload_fraction should be 1.0.
            assert optimizer_offload_fraction == 1.0, (
                "Currently for optimizer offloading, only optimizer_offload_fraction=1.0 is supported"
            )
        if (
            "logprob_chunk_size" in self.cfg
            and self.cfg["logprob_chunk_size"] is not None
            and self.cfg["logprob_chunk_size"] > 0
        ):
            assert self.cfg["megatron_cfg"]["defer_fp32_logits"], (
                "defer_fp32_logits must be True if logprob_chunk_size is set"
            )
        self.defer_fp32_logits = self.cfg["megatron_cfg"].get(
            "defer_fp32_logits", None
        ) and (model_cfg.fp16 or model_cfg.bf16)

        checkpoint_config = CheckpointConfig(
            save_interval=100,
            save=weights_path,
            load=weights_path,
            pretrained_checkpoint=pretrained_path,  # This is the path to the pretrained ckpt for the SFT case
            async_save=False,  # This doesn't work right now.
            fully_parallel_save=True,
            fully_parallel_load=True,  # Enable fully parallel load
            load_rng=False,
        )
        ref_checkpoint_config = CheckpointConfig(
            pretrained_checkpoint=pretrained_path,  # This is the path to the pretrained ckpt for the SFT case
            save=None,
            load=None,
            fully_parallel_load=True,  # Enable fully parallel load
            load_rng=False,
        )

        assert "train_iters" in self.cfg["megatron_cfg"], (
            "train_iters must be set in megatron_cfg. For an example, see "
            "https://github.com/NVIDIA-NeMo/RL/blob/bccbc377705a81a1f4b3c31ad9767bcc15f735a8/nemo_rl/algorithms/sft.py#L175-L179."
        )

        ## These settings are required for correct gradient computations in mcore
        ## when calculate_per_token_loss is True, there is no scaling of the gradient in mcore,
        ## so we handle the scaling in nemo-rl.
        ## perform_initialization = True is a workaround to ensure the correct tensor parallel attributes are set
        ## on the TP-sharded parameters.
        model_cfg.calculate_per_token_loss = True
        model_cfg.perform_initialization = True
        # SHAN CHECK:
        model_cfg.cuda_graph_impl = "local"
        model_cfg.cuda_graph_scope = None
        model_cfg.use_te_rng_tracker = True

        assert (
            "aux_loss" not in model_cfg.moe_router_load_balancing_type
            or model_cfg.moe_aux_loss_coeff == 0
        ), (
            "MoE aux loss is currently not supported due to a known bug in Megatron-LM. "
            "See https://github.com/NVIDIA/Megatron-LM/issues/1984 for more details."
        )

        self.megatron_cfg = ConfigContainer(
            model=model_cfg,
            checkpoint=checkpoint_config,
            logger=LoggerConfig(logging_level=0),
            train=TrainingConfig(
                micro_batch_size=1,  # ignored
                global_batch_size=self.cfg["train_global_batch_size"],  # ignored
                train_iters=self.cfg["megatron_cfg"][
                    "train_iters"
                ],  # Set by algorithm setup
            ),
            optimizer=OptimizerConfig(
                **self.cfg["megatron_cfg"]["optimizer"],
            ),
            ddp=DistributedDataParallelConfig(
                check_for_nan_in_grad=True,
                grad_reduce_in_fp32=self.cfg["megatron_cfg"][
                    "distributed_data_parallel_config"
                ]["grad_reduce_in_fp32"],
                overlap_grad_reduce=self.cfg["megatron_cfg"][
                    "distributed_data_parallel_config"
                ]["overlap_grad_reduce"],
                overlap_param_gather=self.cfg["megatron_cfg"][
                    "distributed_data_parallel_config"
                ]["overlap_param_gather"],
                # we need to set average_in_collective=False with calculate_per_token_loss=True.
                # otherwise, mcore throws an assertion error.
                average_in_collective=False,
                use_distributed_optimizer=self.cfg["megatron_cfg"]["optimizer"][
                    "use_distributed_optimizer"
                ],
                data_parallel_sharding_strategy=self.cfg["megatron_cfg"][
                    "distributed_data_parallel_config"
                ]["data_parallel_sharding_strategy"],
            ),
            scheduler=SchedulerConfig(
                **self.cfg["megatron_cfg"]["scheduler"],
            ),
            dataset=None,
            tokenizer=TokenizerConfig(
                tokenizer_type="HuggingFaceTokenizer",
                tokenizer_model=hf_model_name,
            ),
        )
        # TODO: this validation should happen inside mbridge: https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/1665
        if self.dtype == torch.bfloat16:
            assert self.megatron_cfg.model.bf16 == True, (
                "policy.megatron_cfg.model.bf16=True must be set if policy.precision=bfloat16. This is handled by nemo-rl so this indicates something is misconfigured."
            )
            assert (
                self.megatron_cfg.optimizer.use_precision_aware_optimizer == False
                or self.megatron_cfg.optimizer.bf16 == True
            ), (
                "policy.megatron_cfg.optimizer.bf16=True must be set if policy.precision=bfloat16 when using use_precision_aware_optimizer=True"
            )
        elif self.dtype == torch.float16:
            assert self.megatron_cfg.model.fp16 == True, (
                "policy.megatron_cfg.model.fp16=True must be set if policy.precision=float16. This is handled by nemo-rl so this indicates something is misconfigured."
            )
            assert (
                self.megatron_cfg.optimizer.use_precision_aware_optimizer == False
                or self.megatron_cfg.optimizer.fp16 == True
            ), (
                "policy.megatron_cfg.optimizer.fp16=True must be set if policy.precision=float16 when using use_precision_aware_optimizer=True"
            )
        elif self.dtype == torch.float32:
            assert (
                self.megatron_cfg.model.bf16 == False
                and self.megatron_cfg.model.fp16 == False
            ), (
                "policy.megatron_cfg.model.bf16=False and policy.megatron_cfg.model.fp16=False must be set if policy.precision=float32. This is handled by nemo-rl so this indicates something is misconfigured."
            )
            assert (
                self.megatron_cfg.optimizer.bf16 == False
                and self.megatron_cfg.optimizer.fp16 == False
            ), (
                "policy.megatron_cfg.optimizer.bf16=False and policy.megatron_cfg.optimizer.fp16=False must be set if policy.precision=float32"
            )
        self.megatron_cfg.validate()
        (
            self.mcore_state,
            self.model,
            self.optimizer,
            self.scheduler,
            self.checkpointing_context,
        ) = setup_megatron_model(
            policy_cfg=self.cfg, cfg=self.megatron_cfg, load_optimizer=init_optimizer
        )

        # Set the param sync function for the model
        if (
            self.megatron_cfg.ddp.overlap_param_gather
            and self.megatron_cfg.ddp.align_param_gather
        ):
            self.megatron_cfg.param_sync_func = [
                model_chunk.start_param_sync for model_chunk in self.model
            ]
            if len(self.model) == 1:
                self.megatron_cfg.param_sync_func = self.megatron_cfg.param_sync_func[0]

        self.model = self.model[0]  # Get the first model from the list

        if init_reference_model:
            self.model = self.move_model(self.model, "cpu")
            ref_ckpt_context = init_checkpointing_context(ref_checkpoint_config)

            # Create a separate megatron config for the reference model with the correct checkpoint config
            self.megatron_cfg.model.cuda_graph_impl = "none"
            self.megatron_cfg.model.use_te_rng_tracker = True
            ref_megatron_cfg = ConfigContainer(
                model=self.megatron_cfg.model,
                checkpoint=ref_checkpoint_config,  # Use the reference checkpoint config
                logger=self.megatron_cfg.logger,
                train=self.megatron_cfg.train,
                optimizer=self.megatron_cfg.optimizer,
                ddp=self.megatron_cfg.ddp,
                scheduler=self.megatron_cfg.scheduler,
                dataset=self.megatron_cfg.dataset,
                tokenizer=self.megatron_cfg.tokenizer,
            )

            # Create a separate state object for the reference model
            ref_state = GlobalState()
            ref_state.cfg = ref_megatron_cfg

            # Configure mixed precision wrapper for reference model
            ref_mixed_precision_wrapper = Float16Module
            if self.cfg["megatron_cfg"].get("freeze_moe_router", False):
                ref_mixed_precision_wrapper = CustomFloat16Module

            reference_model = get_model(
                self.megatron_cfg.model,
                self.megatron_cfg.ddp,
                use_torch_fsdp2=self.megatron_cfg.dist.use_torch_fsdp2,
                overlap_param_gather_with_optimizer_step=self.megatron_cfg.optimizer.overlap_param_gather_with_optimizer_step,
                pre_wrap_hook=self.megatron_cfg.rng.data_parallel_random_init,
                mixed_precision_wrapper=ref_mixed_precision_wrapper,
            )
            print("Loading the Reference Model")
            if (
                ref_checkpoint_config.pretrained_checkpoint is not None
                and checkpoint_exists(ref_checkpoint_config.pretrained_checkpoint)
            ):
                load_checkpoint(
                    ref_state,  # Use the separate state object with ref checkpoint config
                    reference_model,
                    None,  # no optimizer
                    None,  # no scheduler
                    checkpointing_context=ref_ckpt_context,
                    skip_load_to_model_and_opt=HAVE_FSDP2
                    and self.megatron_cfg.dist.use_torch_fsdp2,
                )
                reference_model = reference_model[0]
                reference_model.eval()
                self.reference_state_dict = {}
                for name, item in reference_model.state_dict().items():
                    if isinstance(item, torch.Tensor):
                        cpu_item = item.detach().to(
                            device="cpu", non_blocking=True, copy=True
                        )
                        del item
                    else:
                        cpu_item = item
                    self.reference_state_dict[name] = cpu_item
                print("Reference model loaded")
            else:
                print("Reference model not loaded")

            self.model = self.move_model(self.model, "cuda")

        _update_model_config_funcs(
            [self.model],
            self.megatron_cfg.model,
            self.megatron_cfg.ddp,
            self.optimizer,
            align_grad_reduce=self.megatron_cfg.dist.align_grad_reduce,
        )

        tokenizer_config = TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model=hf_model_name,
        )

        self.megatron_tokenizer = build_tokenizer(
            tokenizer_config,
            make_vocab_size_divisible_by=self.megatron_cfg.model.make_vocab_size_divisible_by
            // self.cfg["megatron_cfg"]["tensor_model_parallel_size"],
            tensor_model_parallel_size=self.cfg["megatron_cfg"][
                "tensor_model_parallel_size"
            ],
            trust_remote_code=True,
        )
        self.final_padded_vocab_size = calculate_padded_vocab_size(
            self.megatron_cfg.model.vocab_size,
            self.megatron_cfg.model.make_vocab_size_divisible_by,
            self.cfg["megatron_cfg"]["tensor_model_parallel_size"],
        )
        self.dp_size = worker_sharding_annotations.get_axis_size("data_parallel")
        self.megatron_bridge = AutoBridge.from_hf_pretrained(
            hf_model_name, trust_remote_code=True
        )

        self.should_disable_forward_pre_hook = (
            self.cfg["megatron_cfg"]["optimizer"]["use_distributed_optimizer"]
            and self.cfg["megatron_cfg"]["distributed_data_parallel_config"][
                "overlap_param_gather"
            ]
        )

        # vars used for refit
        ## will be initialized in prepare_refit_info
        # refit_param_info_mcore combines the conversion tasks with the param memory
        # [(mcore_param_name, estimated_memory), ...]
        # Note: here param name is local param name, with local layer number and
        # local expert id etc.
        self.refit_conversion_tasks = (
            None  # Meta data for conversion params from megatron bridge
        )
        self.refit_conversion_tasks_current_index = None
        self.refit_param_info_mcore = None

        ## used for streaming update inference engine weights
        self._held_gather_buffer = None

    def enable_forward_pre_hook(self):
        assert isinstance(self.model, DistributedDataParallel)
        self.model.enable_forward_pre_hook()

    def disable_forward_pre_hook(self, param_sync=True):
        assert isinstance(self.model, DistributedDataParallel)
        self.model.disable_forward_pre_hook(param_sync=param_sync)

    @wrap_with_nvtx_name("megatron_policy_worker/train")
    def train(
        self,
        data: BatchedDataDict,
        loss_fn: LossFunction,
        eval_mode: bool = False,
        gbs: Optional[int] = None,
        mbs: Optional[int] = None,
    ) -> dict[str, Any]:
        """Train the policy on a batch of data with a given loss function."""
        # Disable CUDA graphs for training to avoid conflicts with inference tensors.
        # This prevents "Inplace update to inference tensor outside InferenceMode" errors
        # when switching from generation (which uses CUDA graphs) to training.
        # Use toggle_cuda_graphs on the underlying module (not the DDP wrapper) to properly
        # reset CUDA graph state, following the pattern from megatron/rl/rl_utils.py.
        lang_module = (
            self.model.module.module
            if hasattr(self.model.module, "module")
            else self.model.module
        )

        # Disable forward pre-hooks BEFORE toggle_cuda_graphs to avoid stale hook handles
        # for CudaGraphManager modules that will be removed by toggle_cuda_graphs.
        if self.should_disable_forward_pre_hook:
            self.disable_forward_pre_hook()

        toggle_cuda_graphs(lang_module, set_to="none", reset_cuda_graphs=True)

        self.model.zero_grad_buffer()
        if hasattr(self.model, "inference_params"):
            self.model.inference_params = None

        # Reset any cached attention states
        for module in self.model.modules():
            if hasattr(module, "reset_inference_cache"):
                module.reset_inference_cache()
            if hasattr(module, "_inference_key_value_memory"):
                module._inference_key_value_memory = None

        if gbs is None:
            gbs = self.cfg["train_global_batch_size"]
        if mbs is None:
            mbs = self.cfg["train_micro_batch_size"]
        local_gbs = gbs // self.dp_size
        total_dataset_size = torch.tensor(data.size, device="cuda")
        torch.distributed.all_reduce(
            total_dataset_size,
            op=torch.distributed.ReduceOp.SUM,
            group=parallel_state.get_data_parallel_group(),
        )
        num_global_batches = int(total_dataset_size.item()) // gbs

        if eval_mode:
            ctx: AbstractContextManager[Any] = torch.no_grad()
            self.model.eval()
        else:
            ctx = nullcontext()
            # Ensure model is in training mode
            self.model.train()

        with ctx:
            # dim 1 is always assumed to be the sequence dim, sanity check this here
            sequence_dim = 1
            seq_dim_size = data["input_ids"].shape[sequence_dim]
            for k, v in data.items():
                if torch.is_tensor(v) and len(v.shape) > 1:
                    assert v.shape[sequence_dim] == seq_dim_size, (
                        f"Dim 1 must be the sequence dim, expected dim 1={seq_dim_size} but got shape {v.shape}"
                    )

            forward_step = partial(
                forward_step_arbitrary_loss, loss_fn=loss_fn, policy_cfg=self.cfg
            )
            all_mb_metrics = []
            losses = []
            total_num_microbatches = 0
            for gb_idx in range(num_global_batches):
                global_batch = data.get_batch(batch_idx=gb_idx, batch_size=local_gbs)

                assert "sample_mask" in global_batch, (
                    "sample_mask must be present in the data!"
                )
                ## get the normalization factor for the loss
                local_valid_seqs = torch.sum(global_batch["sample_mask"])

                if not "token_mask" in global_batch:
                    local_valid_toks = (
                        local_valid_seqs * global_batch["input_ids"].shape[1]
                    )
                else:
                    local_valid_toks = torch.sum(
                        global_batch["token_mask"][:, 1:]
                        * global_batch["sample_mask"].unsqueeze(-1)
                    )

                to_reduce = torch.tensor([local_valid_seqs, local_valid_toks]).cuda()
                torch.distributed.all_reduce(
                    to_reduce, group=parallel_state.get_data_parallel_group()
                )
                global_valid_seqs, global_valid_toks = to_reduce[0], to_reduce[1]

                if (
                    hasattr(loss_fn, "loss_type")
                    and loss_fn.loss_type == LossType.TOKEN_LEVEL
                ):
                    assert "token_mask" in global_batch, (
                        "token_mask must be present in the data when using token-level loss"
                    )

                batch = data.get_batch(batch_idx=gb_idx, batch_size=local_gbs)
                pack_seqs = False
                seqlen_key = None
                pad_factor = 1
                pad_full_seq_to = None
                pad_packed_seq_to_multiple_of = 1
                if self.cfg["dynamic_batching"]["enabled"]:
                    data_iterator = batch.make_microbatch_iterator_with_dynamic_shapes()
                    data_iterator_len = (
                        batch.get_microbatch_iterator_dynamic_shapes_len()
                    )
                elif self.cfg["sequence_packing"]["enabled"]:
                    data_iterator = (
                        batch.make_microbatch_iterator_for_packable_sequences()
                    )
                    data_iterator_len, seq_dim_size = (
                        batch.get_microbatch_iterator_for_packable_sequences_len()
                    )
                    mbs = 1
                    pack_seqs = True
                    seqlen_key = "input_lengths"
                    (
                        pad_factor,
                        pad_packed_seq_to_multiple_of,
                        pad_full_seq_to,
                    ) = _get_pack_sequence_parameters_for_megatron(
                        self.cfg["megatron_cfg"],
                        seq_dim_size,
                    )
                    # if pad_full_seq_to is not None, we need to use it as the sequence length
                    seq_dim_size = pad_full_seq_to or seq_dim_size
                else:
                    data_iterator = batch.make_microbatch_iterator(mbs)
                    data_iterator_len = local_gbs // mbs

                # Track total microbatches for MoE aux-loss averaging
                total_num_microbatches += int(data_iterator_len)

                rerun_state_machine = get_rerun_state_machine()
                while rerun_state_machine.should_run_forward_backward(data_iterator):
                    # Set grad to zero.
                    self.model.zero_grad_buffer()
                    self.optimizer.zero_grad()

                    # Forward pass.
                    forward_backward_func = get_forward_backward_func()
                    losses_reduced = forward_backward_func(
                        forward_step_func=partial(
                            forward_step,
                            self.mcore_state,
                            global_valid_seqs,
                            global_valid_toks,
                            pack_sequences=pack_seqs,
                            seq_length_key=seqlen_key,
                            pad_individual_seqs_to_multiple_of=pad_factor,
                            pad_packed_seq_to_multiple_of=pad_packed_seq_to_multiple_of,
                            pad_full_seq_to=pad_full_seq_to,
                            defer_fp32_logits=self.defer_fp32_logits,
                        ),
                        data_iterator=data_iterator,
                        model=self.model,
                        num_microbatches=data_iterator_len,
                        seq_length=seq_dim_size,
                        micro_batch_size=mbs,
                        decoder_seq_length=seq_dim_size,
                        forward_only=eval_mode,
                        do_not_average_loss=True,
                    )

                # Empty unused memory.
                if self.cfg["megatron_cfg"]["empty_unused_memory_level"] >= 1:
                    torch.cuda.empty_cache()

                # Update parameters.
                if not eval_mode:
                    update_successful, grad_norm, num_zeros_in_grad = (
                        self.optimizer.step()
                    )
                else:
                    update_successful, grad_norm, num_zeros_in_grad = (True, 0.0, 0.0)

                # when freezing sub-models we may have a mixture of successful and unsucessful ranks,
                # so we must gather across mp ranks
                update_successful = logical_and_across_model_parallel_group(
                    update_successful
                )
                # grad_norm and num_zeros_in_grad will be None on ranks without trainable params,
                # so we must gather across mp ranks
                grad_norm: float = reduce_max_stat_across_model_parallel_group(
                    grad_norm
                )
                num_zeros_in_grad: float = reduce_max_stat_across_model_parallel_group(
                    num_zeros_in_grad
                )

                if update_successful:
                    skipped_iter = 0
                else:
                    skipped_iter = 1

                # Empty unused memory.
                if self.cfg["megatron_cfg"]["empty_unused_memory_level"] >= 2:
                    torch.cuda.empty_cache()

                if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    # keep all microbatch metrics to be normalized later
                    gb_loss_metrics = []
                    mb_losses = []
                    for x in losses_reduced:
                        loss_metrics = {}
                        for k in x.keys():
                            loss_metrics[k] = x[k] / num_global_batches
                        gb_loss_metrics.append(loss_metrics)
                        curr_lr = self.scheduler.get_lr(self.optimizer.param_groups[0])
                        curr_wd = self.scheduler.get_wd()
                        loss_metrics["lr"] = curr_lr
                        loss_metrics["wd"] = curr_wd
                        loss_metrics["global_valid_seqs"] = global_valid_seqs.item()
                        loss_metrics["global_valid_toks"] = global_valid_toks.item()
                        mb_losses.append(loss_metrics["loss"])

                    torch.distributed.broadcast_object_list(
                        [gb_loss_metrics],
                        src=get_pipeline_model_parallel_last_rank(),
                        group=get_pipeline_model_parallel_group(),
                    )
                else:
                    loss_metrics = [None]  # type: ignore
                    torch.distributed.broadcast_object_list(
                        loss_metrics,
                        src=get_pipeline_model_parallel_last_rank(),
                        group=get_pipeline_model_parallel_group(),
                    )
                    gb_loss_metrics = loss_metrics[0]
                    mb_losses = [x["loss"] for x in gb_loss_metrics]

                all_mb_metrics.extend(gb_loss_metrics)
                losses.append(torch.tensor(mb_losses).sum().item())

        if not eval_mode:
            # take one LR step every rollout batch
            # we need to scale the step by gbs to counteract the fact that NeMo automatically
            # scales lr_warmup_steps by gbs during init
            self.scheduler.step(increment=gbs)

        # Aggregate metrics across all microbatches
        mb_metrics = defaultdict(list)
        for m in all_mb_metrics:
            for k, v in m.items():
                mb_metrics[k].append(v)

        with torch.no_grad():
            global_loss = torch.tensor(losses, device="cuda")
            torch.distributed.all_reduce(
                global_loss,
                op=torch.distributed.ReduceOp.SUM,
                group=parallel_state.get_data_parallel_group(),
            )

        metrics = {
            "global_loss": global_loss.cpu(),
            "rank": torch.distributed.get_rank(),
            "gpu_name": torch.cuda.get_device_name(),
            "model_dtype": self.dtype,
            "all_mb_metrics": dict(mb_metrics),
            "grad_norm": torch.tensor([grad_norm]),
        }
        # Collect MoE aux metrics averaged across microbatches
        num_moe_experts = getattr(self.model.config, "num_moe_experts", None)
        if num_moe_experts is not None and num_moe_experts > 1:
            moe_loss_scale = 1.0 / max(1, total_num_microbatches)
            moe_metrics = get_moe_metrics(
                loss_scale=moe_loss_scale,
                per_layer_logging=self.cfg["megatron_cfg"]["moe_per_layer_logging"],
            )
            if moe_metrics:
                metrics["moe_metrics"] = moe_metrics
        return metrics

    @wrap_with_nvtx_name("megatron_policy_worker/get_logprobs")
    def get_logprobs(
        self, *, data: BatchedDataDict[Any], micro_batch_size: Optional[int] = None
    ) -> BatchedDataDict[LogprobOutputSpec]:
        """Get the logprobs of the model for a batch of data.

        Uses the configured logprob_batch_size to do microbatching.
        Input data is assumed to be right-padded. The method internally converts to
        left-padded format for computation, and returns outputs in right-padded format.
        If micro_batch_size is provided, it will be used instead of the configured
        logprob_batch_size.

        Returns:
          a BatchedDataDict with key "logprobs" and shape [batch_size, sequence_length].
          We use the convention that the logprob of the first token is 0 so that the sequence length is maintained.
          The logprob of input token i is specified at position i in the output logprobs tensor.
        """
        no_grad = torch.no_grad()
        no_grad.__enter__()
        logprob_batch_size = (
            micro_batch_size
            if micro_batch_size is not None
            else self.cfg["logprob_batch_size"]
        )

        # dim 1 is always assumed to be the sequence dim, sanity check this here
        sequence_dim = 1
        input_seq_dim_size = data["input_ids"].shape[sequence_dim]
        for k, v in data.items():
            if torch.is_tensor(v) and len(v.shape) > 1:
                assert v.shape[sequence_dim] == input_seq_dim_size, (
                    f"Dim 1 must be the sequence dim, expected dim 1={input_seq_dim_size} but got shape {v.shape}"
                )

        self.model.eval()

        pp_seq_dim_size = input_seq_dim_size
        pp_grp = get_pipeline_model_parallel_group()
        pad_factor = 1
        pad_packed_seq_to_multiple_of = 1
        pad_full_seq_to = None
        if self.cfg["sequence_packing"]["enabled"]:
            _, seq_dim_size = data.get_microbatch_iterator_for_packable_sequences_len()
            (
                pad_factor,
                pad_packed_seq_to_multiple_of,
                pad_full_seq_to,
            ) = _get_pack_sequence_parameters_for_megatron(
                self.cfg["megatron_cfg"],
                seq_dim_size,
            )
            pp_seq_dim_size = pad_full_seq_to or pp_seq_dim_size

        def forward_step_fn(
            data_iterator: Iterator[BatchedDataDict[Any]], model: GPTModel
        ):
            nonlocal pad_full_seq_to, pad_packed_seq_to_multiple_of, pad_factor
            data_dict = next(data_iterator).to("cuda")
            if self.cfg["sequence_packing"]["enabled"]:
                original_seq_length = data_dict["input_ids"].shape[1]
                cp_size = self.cfg["megatron_cfg"]["context_parallel_size"]
                cp_rank = get_context_parallel_rank()
                (
                    input_ids,
                    input_ids_cp_sharded,
                    packed_seq_params,
                    cu_seqlens,
                    cu_seqlens_padded,
                ) = _pack_sequences_for_megatron(
                    data_dict["input_ids"].clone(),
                    data_dict["input_lengths"],
                    pad_individual_seqs_to_multiple_of=pad_factor,
                    pad_packed_seq_to_multiple_of=pad_packed_seq_to_multiple_of,
                    pad_packed_seq_to=pad_full_seq_to,
                    cp_rank=cp_rank,
                    cp_size=cp_size,
                )
                attention_mask, position_ids = None, None
                unpacked_input_ids = data_dict["input_ids"]
            else:
                input_ids = data_dict["input_ids"]
                input_ids_cp_sharded = input_ids
                attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
                    data=input_ids,
                    eod_token=0,  # used for loss_mask, which we don't use
                    pad_token=0,  # used for loss_mask, which we don't use
                    reset_position_ids=False,
                    reset_attention_mask=False,
                    eod_mask_loss=False,
                    pad_mask_loss=False,
                )
                packed_seq_params = None
                unpacked_input_ids = input_ids

            multimodal_data = data_dict.get_multimodal_dict(
                as_tensors=True, device=input_ids.device
            )
            if len(multimodal_data) > 0:
                position_ids = None

            additional_kwargs = {}
            # Mamba models currently do not support packed_seq_params
            if packed_seq_params is not None:
                additional_kwargs["packed_seq_params"] = packed_seq_params

            if self.defer_fp32_logits:
                additional_kwargs["fp32_output"] = False

            output_tensor = model(
                input_ids=input_ids_cp_sharded,
                position_ids=position_ids,
                attention_mask=attention_mask,
                **multimodal_data,
                **additional_kwargs,
            )

            # Apply temperature scaling to logits for training
            # This matches the dtensor worker's _apply_temperature_scaling in the train method
            if "generation" in self.cfg and self.cfg["generation"] is not None:
                output_tensor.div_(self.cfg["generation"]["temperature"])

            def collection_fn(output_tensor):
                stc = time.time()
                tp_grp = get_tensor_model_parallel_group()
                tp_rank = get_tensor_model_parallel_rank()
                logprob_chunk_size = self.cfg.get("logprob_chunk_size", None)
                if self.cfg["sequence_packing"]["enabled"]:
                    token_logprobs = from_parallel_logits_to_logprobs_packed_sequences(
                        output_tensor,
                        target=input_ids,
                        cu_seqlens_padded=cu_seqlens_padded,
                        unpacked_seqlen=original_seq_length,
                        vocab_start_index=tp_rank * output_tensor.shape[-1],
                        vocab_end_index=(tp_rank + 1) * output_tensor.shape[-1],
                        group=tp_grp,
                        inference_only=True,
                        cp_group=get_context_parallel_group(),
                        chunk_size=logprob_chunk_size,
                    )
                else:
                    token_logprobs = from_parallel_logits_to_logprobs(
                        output_tensor,
                        target=unpacked_input_ids,
                        vocab_start_index=tp_rank * output_tensor.shape[-1],
                        vocab_end_index=(tp_rank + 1) * output_tensor.shape[-1],
                        tp_group=tp_grp,
                        inference_only=True,
                        chunk_size=logprob_chunk_size,
                    )

                # Prepend 0 logprob for first token to maintain same sequence length as input
                token_logprobs = torch.cat(
                    [torch.zeros_like(token_logprobs[:, :1]), token_logprobs], dim=1
                )
                return torch.tensor(0.0, device=token_logprobs.device), {
                    "logprobs": token_logprobs
                }

            return output_tensor, collection_fn

        if self.cfg["dynamic_batching"]["enabled"]:
            mb_iterator = data.make_microbatch_iterator_with_dynamic_shapes()
            data_iterator_len = data.get_microbatch_iterator_dynamic_shapes_len()
            micro_batch_size = logprob_batch_size
        elif self.cfg["sequence_packing"]["enabled"]:
            mb_iterator = data.make_microbatch_iterator_for_packable_sequences()
            data_iterator_len, _ = (
                data.get_microbatch_iterator_for_packable_sequences_len()
            )
            micro_batch_size = 1
        else:
            mb_iterator = data.make_microbatch_iterator(logprob_batch_size)
            data_iterator_len = max(1, data.size // logprob_batch_size)
            micro_batch_size = logprob_batch_size

        forward_backward_func = get_forward_backward_func()
        list_of_logprobs = forward_backward_func(
            forward_step_func=forward_step_fn,
            data_iterator=mb_iterator,
            model=self.model,
            num_microbatches=data_iterator_len,
            seq_length=pp_seq_dim_size,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=pp_seq_dim_size,
            forward_only=True,
        )
        if is_pipeline_last_stage(ignore_virtual=True):
            all_log_probs_padded = []
            all_logprobs = [l["logprobs"] for l in list_of_logprobs]
            for lp in all_logprobs:
                padding_needed = input_seq_dim_size - lp.shape[1]
                if padding_needed > 0:
                    lp = torch.nn.functional.pad(
                        lp, (0, padding_needed), mode="constant", value=0.0
                    )
                all_log_probs_padded.append(lp)

            logprobs = torch.cat(all_log_probs_padded, dim=0)
            # broadcast logprobs to first pp rank
            broadcast_tensor(logprobs, torch.distributed.get_rank(), pp_grp)
        else:
            logprobs = broadcast_tensor(
                None, get_pipeline_model_parallel_last_rank(), pp_grp
            )

        no_grad.__exit__(None, None, None)
        return BatchedDataDict[LogprobOutputSpec](logprobs=logprobs).to("cpu")

    @contextmanager
    def use_reference_model(self):
        """Context manager that temporarily swaps the reference model and active model.

        On entry: Moves model to CPU, moves reference_model to CUDA. Swaps the references
        On exit: Restores original references and re-flips cuda/cpu
        """
        ## disable overlap param gather when swapping weights
        if self.should_disable_forward_pre_hook:
            self.disable_forward_pre_hook()

        with torch.no_grad():
            try:
                # Save original references
                model_state_dict = {}
                for name, item in self.model.state_dict().items():
                    if isinstance(item, torch.Tensor):
                        item = item.detach().to(
                            device="cpu", non_blocking=True, copy=True
                        )
                    model_state_dict[name] = item

                self.model.load_state_dict(self.reference_state_dict, strict=True)
                # for name, item in self.reference_state_dict.items():
                # if isinstance(item, torch.Tensor):
                # self.model.state_dict()[name] = item.detach().to(device="cuda", non_blocking=True, copy=True)

                if self.cfg["megatron_cfg"]["empty_unused_memory_level"] >= 1:
                    gc.collect()
                    torch.cuda.empty_cache()

                # - self.model is the original reference_model, now on CUDA
                # - self.reference_model is the original model, now on CPU
                yield

            finally:
                # Restore original references and device placement
                self.model.load_state_dict(model_state_dict, strict=True)
                # for name, item in model_state_dict.items():
                # if isinstance(item, torch.Tensor):
                # item = item.detach().to(device="cuda", non_blocking=True, copy=True)
                # self.model.state_dict()[name] = item

                if self.cfg["megatron_cfg"]["empty_unused_memory_level"] >= 1:
                    gc.collect()
                    torch.cuda.empty_cache()

                ## re-enable overlap param gather after weight swap
                if self.should_disable_forward_pre_hook:
                    self.enable_forward_pre_hook()

    @wrap_with_nvtx_name("megatron_policy_worker/get_topk_logits")
    def get_topk_logits(
        self,
        *,
        data: BatchedDataDict[GenerationDatumSpec],
        k: int,
        micro_batch_size: Optional[int] = None,
    ):
        """Get the top-k logits and indices for a batch of data.

        The major difference from get_logprobs is that we compute top-k logits and indices for each position in the sequence.

        Returns:
            BatchedDataDict containing:
                - topk_logits: Tensor of top-k logits for each position in the sequence
                - topk_indices: Tensor of top-k indices for each position in the sequence
        """
        no_grad = torch.no_grad()
        no_grad.__enter__()

        logprob_batch_size = (
            micro_batch_size
            if micro_batch_size is not None
            else self.cfg["logprob_batch_size"]
        )

        sequence_dim = 1
        input_seq_dim_size = data["input_ids"].shape[sequence_dim]
        # Avoid shadowing the function argument `k` by using a distinct variable name
        for tensor_name, v in data.items():
            if torch.is_tensor(v) and len(v.shape) > 1:
                assert v.shape[sequence_dim] == input_seq_dim_size, (
                    f"Tensor {tensor_name} must have sequence dimension {sequence_dim} of size {input_seq_dim_size}, but got shape {v.shape}"
                )

        self.model.eval()

        pp_seq_dim_size = input_seq_dim_size
        pp_grp = get_pipeline_model_parallel_group()

        pad_factor = 1
        pad_packed_seq_to_multiple_of = 1
        pad_full_seq_to = None
        if self.cfg["sequence_packing"]["enabled"]:
            _, seq_dim_size = data.get_microbatch_iterator_for_packable_sequences_len()
            (
                pad_factor,
                pad_packed_seq_to_multiple_of,
                pad_full_seq_to,
            ) = _get_pack_sequence_parameters_for_megatron(
                self.cfg["megatron_cfg"],
                seq_dim_size,
            )
            pp_seq_dim_size = pad_full_seq_to or pp_seq_dim_size

        def forward_step_fn(
            data_iterator: Iterator[BatchedDataDict[Any]], model: GPTModel
        ):
            nonlocal pad_full_seq_to, pad_packed_seq_to_multiple_of, pad_factor
            data_dict = next(data_iterator).to("cuda")

            pack = self.cfg["sequence_packing"]["enabled"]
            if pack:
                original_seq_length = data_dict["input_ids"].shape[1]
                cp_size = self.cfg["megatron_cfg"]["context_parallel_size"]
                cp_rank = get_context_parallel_rank()

                (
                    input_ids_unpacked,
                    input_ids_cp_sharded,
                    packed_seq_params,
                    cu_seqlens,
                    cu_seqlens_padded,
                ) = _pack_sequences_for_megatron(
                    data_dict["input_ids"].clone(),
                    data_dict["input_lengths"],
                    pad_individual_seqs_to_multiple_of=pad_factor,
                    pad_packed_seq_to_multiple_of=pad_packed_seq_to_multiple_of,
                    pad_packed_seq_to=pad_full_seq_to,
                    cp_rank=cp_rank,
                    cp_size=cp_size,
                )
                attention_mask, position_ids = None, None
                seq_lengths = data_dict["input_lengths"]
                unpacked_seqlen = original_seq_length
            else:
                input_ids_cp_sharded = data_dict["input_ids"]
                attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
                    data=input_ids_cp_sharded,
                    eod_token=0,
                    pad_token=0,
                    reset_position_ids=False,
                    reset_attention_mask=False,
                    eod_mask_loss=False,
                    pad_mask_loss=False,
                )
                packed_seq_params = None

            multimodal_data = data_dict.get_multimodal_dict(
                as_tensors=True, device=input_ids_cp_sharded.device
            )
            if len(multimodal_data) > 0:
                position_ids = None

            additional_kwargs = {}
            if packed_seq_params is not None:
                additional_kwargs["packed_seq_params"] = packed_seq_params

            output_tensor = model(
                input_ids=input_ids_cp_sharded,
                position_ids=position_ids,
                attention_mask=attention_mask,
                **additional_kwargs,
                **multimodal_data,
            )

            if "generation" in self.cfg and self.cfg["generation"] is not None:
                output_tensor.div_(self.cfg["generation"]["temperature"])

            def collection_fn(_):
                # Only the last PP stage produces final logits/top-k; earlier stages return empty
                # if not is_pipeline_last_stage(ignore_virtual=True):
                # return output_tensor.new_zeros(()), {}

                tp_grp = get_tensor_model_parallel_group()
                tp_rank = get_tensor_model_parallel_rank()
                vocab_shard_size = output_tensor.shape[-1]
                vocab_start_index = tp_rank * vocab_shard_size

                chunk_size = None
                if "logprob_chunk_size" in self.cfg:
                    chunk_size = self.cfg["logprob_chunk_size"]

                topk_vals_local, topk_idx_local = distributed_vocab_topk(
                    output_tensor,
                    k,
                    tp_grp,
                    vocab_start_index=vocab_start_index,
                    vocab_end_index=vocab_start_index + vocab_shard_size,
                    chunk_size=chunk_size,
                )

                if self.cfg["megatron_cfg"]["context_parallel_size"] > 1:
                    cp_grp = get_context_parallel_group()
                    if pack:
                        # Per-sequence CP allgather following packed-sequence logic
                        batch_size = data_dict["input_ids"].shape[0]
                        total_packed_len = int(cu_seqlens_padded[-1].item())

                        topk_vals_full = torch.zeros(
                            (1, total_packed_len, k),
                            dtype=topk_vals_local.dtype,
                            device=topk_vals_local.device,
                        )
                        topk_idx_full = torch.zeros(
                            (1, total_packed_len, k),
                            dtype=topk_idx_local.dtype,
                            device=topk_idx_local.device,
                        )

                        for i in range(batch_size):
                            start_idx = int(cu_seqlens_padded[i].item())
                            end_idx = int(cu_seqlens_padded[i + 1].item())
                            if end_idx > start_idx:
                                local_vals_slice = topk_vals_local[
                                    :, start_idx // cp_size : end_idx // cp_size, :
                                ]
                                local_idx_slice = topk_idx_local[
                                    :, start_idx // cp_size : end_idx // cp_size, :
                                ]
                                gathered_vals = allgather_cp_sharded_tensor(
                                    local_vals_slice, cp_grp, seq_dim=1
                                )
                                gathered_idx = allgather_cp_sharded_tensor(
                                    local_idx_slice, cp_grp, seq_dim=1
                                )
                                # Some kernels may return [X, Y, k] where X*Y = (end_idx - start_idx).
                                # Flatten leading dims and reshape to [1, expected_len, k] to match target.
                                expected_len = end_idx - start_idx
                                if (
                                    gathered_vals.dim() == 3
                                    and gathered_vals.shape[1] != expected_len
                                ):
                                    gathered_vals = gathered_vals.reshape(
                                        1, expected_len, gathered_vals.shape[-1]
                                    )
                                if (
                                    gathered_idx.dim() == 3
                                    and gathered_idx.shape[1] != expected_len
                                ):
                                    gathered_idx = gathered_idx.reshape(
                                        1, expected_len, gathered_idx.shape[-1]
                                    )
                                topk_vals_full[:, start_idx:end_idx, :] = gathered_vals
                                topk_idx_full[:, start_idx:end_idx, :] = gathered_idx
                    else:
                        # Sequence packing must be enabled when CP > 1
                        raise RuntimeError(
                            "Context Parallelism (CP>1) requires sequence packing to be enabled."
                        )
                else:
                    topk_vals_full = topk_vals_local
                    topk_idx_full = topk_idx_local

                if pack:
                    batch_size = data_dict["input_ids"].shape[0]
                    out_vals = torch.zeros(
                        (batch_size, unpacked_seqlen, k),
                        dtype=topk_vals_full.dtype,
                        device=topk_vals_full.device,
                    )
                    out_idx = torch.zeros(
                        (batch_size, unpacked_seqlen, k),
                        dtype=topk_idx_full.dtype,
                        device=topk_idx_full.device,
                    )
                    for i in range(batch_size):
                        seq_len = int(seq_lengths[i].item())
                        start_idx = int(cu_seqlens_padded[i].item())
                        if seq_len > 0:
                            out_vals[i, :seq_len, :] = topk_vals_full[
                                0, start_idx : start_idx + seq_len, :
                            ]
                            out_idx[i, :seq_len, :] = topk_idx_full[
                                0, start_idx : start_idx + seq_len, :
                            ]
                    return output_tensor.new_zeros(()), {
                        "topk_logits": out_vals,
                        "topk_indices": out_idx,
                    }
                else:
                    return output_tensor.new_zeros(()), {
                        "topk_logits": topk_vals_full,
                        "topk_indices": topk_idx_full,
                    }

            return output_tensor, collection_fn

        if self.cfg["dynamic_batching"]["enabled"]:
            mb_iterator = data.make_microbatch_iterator_with_dynamic_shapes()
            data_iterator_len = data.get_microbatch_iterator_dynamic_shapes_len()
            micro_batch = logprob_batch_size
        elif self.cfg["sequence_packing"]["enabled"]:
            mb_iterator = data.make_microbatch_iterator_for_packable_sequences()
            data_iterator_len, _ = (
                data.get_microbatch_iterator_for_packable_sequences_len()
            )
            micro_batch = 1
        else:
            mb_iterator = data.make_microbatch_iterator(logprob_batch_size)
            data_iterator_len = max(1, data.size // logprob_batch_size)
            micro_batch = logprob_batch_size

        forward_backward_func = get_forward_backward_func()
        list_of_outputs = forward_backward_func(
            forward_step_func=forward_step_fn,
            data_iterator=mb_iterator,
            model=self.model,
            num_microbatches=data_iterator_len,
            seq_length=pp_seq_dim_size,
            micro_batch_size=micro_batch,
            decoder_seq_length=pp_seq_dim_size,
            forward_only=True,
        )

        if is_pipeline_last_stage(ignore_virtual=True):
            logits_chunks = []
            indices_chunks = []
            for out in list_of_outputs:
                tk = out["topk_logits"]
                ti = out["topk_indices"]
                pad_len = input_seq_dim_size - tk.shape[1]
                if pad_len > 0:
                    tk = torch.nn.functional.pad(tk, (0, 0, 0, pad_len), value=0.0)
                    ti = torch.nn.functional.pad(ti, (0, 0, 0, pad_len), value=0)
                logits_chunks.append(tk)
                indices_chunks.append(ti)

            topk_logits = torch.cat(logits_chunks, dim=0)
            topk_indices = torch.cat(indices_chunks, dim=0)

            topk_logits = broadcast_tensor(
                topk_logits, torch.distributed.get_rank(), pp_grp
            )
            topk_indices = broadcast_tensor(
                topk_indices, torch.distributed.get_rank(), pp_grp
            )
        else:
            last_pp_rank = get_pipeline_model_parallel_last_rank()
            topk_logits = broadcast_tensor(None, last_pp_rank, pp_grp)
            topk_indices = broadcast_tensor(None, last_pp_rank, pp_grp)

        no_grad.__exit__(None, None, None)
        return BatchedDataDict.from_batches(
            [{"topk_logits": topk_logits.cpu(), "topk_indices": topk_indices.cpu()}]
        )

    @torch.no_grad()
    @wrap_with_nvtx_name("megatron_policy_worker/generate")
    def generate(
        self, *, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> BatchedDataDict[GenerationOutputSpec]:
        """Generate a batch of data using huggingface framework generation.

        Args:
            data: BatchedDataDict containing input_ids and input_lengths tensors
        Returns:
            BatchedDataDict conforming to GenerationOutputSpec:
                - output_ids: input + generated token IDs
                - logprobs: Log probabilities for each token
                - generation_lengths: Lengths of each response
        """
        # 512 bATCH SIZE (200 tokens)
        no_grad = torch.no_grad()
        no_grad.__enter__()
        self.model.config.flash_decode = False
        if self.should_disable_forward_pre_hook:
            self.model = self.move_model(
                self.model, "cuda", move_params=True, move_grads=False
            )
        # Verify input is right padded
        assert isinstance(data, BatchedDataDict), (
            f"data must be a BatchedDataDict, got type: {type(data)}"
        )
        assert "input_ids" in data and "input_lengths" in data, (
            f"input_ids and input_lengths must be present in the BatchedDataDict, got keys: {data.keys()}"
        )
        is_right_padded, error_msg = verify_right_padding(
            data, pad_value=self.tokenizer.pad_token_id
        )
        if not is_right_padded:
            warnings.warn(
                f"Input to Megatron Generation worker is not properly right-padded: {error_msg}"
            )

        model_cfg = self.megatron_cfg.model
        inference_wrapper_config = InferenceWrapperConfig(
            hidden_size=model_cfg.hidden_size,
            inference_batch_times_seqlen_threshold=1000000,
            fp32_residual_connection=model_cfg.fp32_residual_connection,
            params_dtype=model_cfg.params_dtype,
            padded_vocab_size=self.final_padded_vocab_size,  # Use the potentially updated value
            inference_max_seq_length=self.cfg["generation"]["max_new_tokens"],  # type: ignore
            inference_max_requests=self.cfg["generation_batch_size"],
        )

        from megatron.core.inference.contexts.dynamic_context import (
            DynamicInferenceContext,
        )
        from megatron.core.inference.engines.dynamic_engine import (
            DynamicInferenceEngine,
        )
        from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
            GPTInferenceWrapper,
        )
        from megatron.core.inference.sampling_params import SamplingParams

        mcore_generation_config = self.cfg["generation"]["mcore_generation_config"]
        buffer_size_gb = mcore_generation_config.get("buffer_size_gb", 20)

        num_cuda_graphs = mcore_generation_config.get("num_cuda_graphs", 16)
        block_size_tokens = mcore_generation_config.get("block_size_tokens", 256)
        use_cuda_graphs_for_non_decode_steps = mcore_generation_config.get(
            "use_cuda_graphs_for_non_decode_steps", True
        )
        enable_chunked_prefill = mcore_generation_config.get(
            "enable_chunked_prefill", True
        )
        unified_memory_level = mcore_generation_config.get("unified_memory_level", 0)
        buffer_guaranteed_fraction = mcore_generation_config.get(
            "buffer_guaranteed_fraction", 0.1
        )
        max_tokens = mcore_generation_config.get("max_tokens", 16384)

        # Get the underlying language module (not the DDP wrapper) for toggle_cuda_graphs.
        # This follows the pattern from megatron/rl/rl_utils.py megatron_rl_inference_mode().
        # The DDP wrapper can interfere with CUDA graph state management.
        lang_module = (
            self.model.module.module
            if hasattr(self.model.module, "module")
            else self.model.module
        )

        # Enable CUDA graphs for inference using toggle_cuda_graphs on the underlying module.
        # reset_cuda_graphs=True ensures fresh CUDA graphs are created for each generate() call.
        

        toggle_cuda_graphs(lang_module, set_to="local", reset_cuda_graphs=True)

        model_config = self.model.config
        model_config.cuda_graph_impl = "local"

        dynamic_context = DynamicInferenceContext(
            params_dtype=inference_wrapper_config.params_dtype,
            num_layers=model_config.num_layers,
            kv_channels=model_config.kv_channels,
            num_attention_heads=model_config.num_query_groups,
            max_sequence_length=self.cfg["generation"]["max_new_tokens"],
            buffer_guaranteed_fraction=buffer_guaranteed_fraction,
            buffer_size_gb=buffer_size_gb,
            materialize_only_last_token_logits=False,
            num_cuda_graphs=num_cuda_graphs,
            block_size_tokens=block_size_tokens,
            tensor_model_parallel_size=self.cfg["megatron_cfg"][
                "tensor_model_parallel_size"
            ],
            use_cuda_graphs_for_non_decode_steps=use_cuda_graphs_for_non_decode_steps,
            use_flashinfer_fused_rope=False,
            unified_memory_level=unified_memory_level,
            max_tokens_override=max_tokens,
        )
        inference_wrapped_model = GPTInferenceWrapper(
            self.model, inference_wrapper_config, dynamic_context
        )

        inference_wrapped_model.prep_model_for_inference()
        # Set pipeline parallel flag
        inference_wrapped_model.model_is_pipeline_parallel = (
            self.cfg["megatron_cfg"]["pipeline_model_parallel_size"] > 1
        )

        text_generation_controller = TextGenerationController(
            inference_wrapped_model=inference_wrapped_model,
            tokenizer=self.megatron_tokenizer,
        )

        # Calculate seed based on node and rank to ensure reproducibility across workers
        local_rank = torch.cuda.current_device()  # Local GPU index on the node
        num_gpus_per_node = torch.cuda.device_count()
        node_idx = self.rank // num_gpus_per_node if num_gpus_per_node > 0 else 0
        seed = (node_idx * 1024) + local_rank

        # New API: DynamicInferenceEngine has additional parameters
        dynamic_engine = DynamicInferenceEngine(
            text_generation_controller,
            dynamic_context,
            enable_cuda_graph=True,
            random_seed=seed,
            track_paused_request_events=False,
            enable_chunked_prefill=enable_chunked_prefill,
            inference_logging_step_interval=0,
        )

        # Handle None values for top_k - convert to integer as required by Megatron
        top_k_cfg = self.cfg["generation"]["top_k"]
        top_k_val = 1 if greedy else (int(top_k_cfg) if top_k_cfg is not None else 0)

        top_p_cfg = self.cfg["generation"]["top_p"]
        top_p_val = (
            0.0 if greedy else (float(top_p_cfg) if top_p_cfg is not None else 0.0)
        )

        # New API: SamplingParams now includes termination_id and uses num_tokens_total
        sampling_params = SamplingParams(
            temperature=self.cfg["generation"]["temperature"] if not greedy else 0,
            top_k=top_k_val,
            top_p=top_p_val,
            skip_prompt_log_probs=False,
            return_log_probs=True,
            num_tokens_total=self.cfg["generation"]["max_new_tokens"],
            num_tokens_to_generate=None,
            termination_id=self.megatron_tokenizer.eod,
        )

        input_ids = data["input_ids"]
        prompt_tokens_tensor = input_ids.cuda()
        prompt_lengths_tensor = data["input_lengths"]
        request_id = 0

        # New API: add_request now takes sampling_params as a parameter
        for p, prompt_len in zip(
            prompt_tokens_tensor, prompt_lengths_tensor, strict=True
        ):
            dynamic_engine.add_request(
                request_id,
                p[:prompt_len],
                sampling_params=sampling_params,
            )
            request_id += 1

        result = []
        import time
        start = time.time()
        while dynamic_engine.has_unfinished_requests():
            result_step = dynamic_engine.step_modern(verbose=False)
            finished_requests = result_step.get("finished_requests", [])
            for finished_request in finished_requests:
                result.append(finished_request)
        end = time.time()
        print(f"SHAN : Time taken: {end - start} seconds")

        # Sort results by request_id to maintain original batch order
        result.sort(key=lambda x: x.request_id)

        out = {
            "tokens": [x.prompt_tokens.tolist() + x.generated_tokens for x in result],
            "logprobs": [x.prompt_log_probs + x.generated_log_probs for x in result],
        }

        input_lengths = data["input_lengths"]
        # pad the out "tokens" and "logprobs" and make them into tensors from lists
        batch_size = data["input_ids"].size(0)
        max_gen_seq_len = max([len(x.generated_tokens) for x in result])
        padded_input_length = input_ids.size(1)

        max_seq_len = padded_input_length + max_gen_seq_len
        # Create padded tensors for tokens and logprobs
        output_ids_padded = torch.full(
            (batch_size, max_seq_len),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=data["input_ids"].device,
        )

        logprobs_padded = torch.zeros(
            (batch_size, max_seq_len),
            dtype=torch.float,
            device=data["input_ids"].device,
        )

        # Fill in the padded tensors with actual values
        generation_lengths = torch.zeros(
            batch_size, dtype=torch.long, device=data["input_ids"].device
        )
        unpadded_sequence_lengths = torch.zeros(
            batch_size, dtype=torch.long, device=data["input_ids"].device
        )
        for i in range(batch_size):
            seq_len = len(out["tokens"][i])
            output_ids_padded[i, :seq_len] = torch.tensor(
                out["tokens"][i], dtype=torch.long, device=data["input_ids"].device
            )
            generation_lengths[i] = seq_len - input_lengths[i].item()
            unpadded_sequence_lengths[i] = seq_len
            logprob_len = len(out["logprobs"][i])
            logprobs_padded[i, 1 : logprob_len + 1] = torch.tensor(
                out["logprobs"][i],
                dtype=torch.float,
                device=data["input_ids"].device,
            )

        out_dict = {
            "output_ids": output_ids_padded,
            "logprobs": logprobs_padded,
            "generation_lengths": generation_lengths,
            "unpadded_sequence_lengths": unpadded_sequence_lengths,
        }

        self.model.config.flash_decode = False

        # Disable forward pre-hooks BEFORE toggle_cuda_graphs to avoid stale hook handles
        # for CudaGraphManager modules that will be removed by toggle_cuda_graphs.
        if self.should_disable_forward_pre_hook:
            self.disable_forward_pre_hook()

        # Disable CUDA graphs after inference using toggle_cuda_graphs on the underlying module.
        # This follows the pattern from megatron/rl/rl_utils.py megatron_rl_inference_mode().
        # reset_cuda_graphs=True ensures:
        # 1. cudagraph_manager attributes are deleted from modules
        # 2. _CudagraphGlobalRecord state is reset via delete_cuda_graphs()
        # 3. Model is ready for training (avoids "inference tensor" errors)
        # 4. Next generate() call will create fresh CUDA graphs with new DynamicInferenceContext
        toggle_cuda_graphs(lang_module, set_to="none", reset_cuda_graphs=True)

        no_grad.__exit__(None, None, None)

        return BatchedDataDict.from_batches([out_dict]).to("cpu")

    @torch.no_grad()
    @wrap_with_nvtx_name("megatron_policy_worker/prepare_refit_info")
    def prepare_refit_info(self) -> None:
        """Prepare state dict metadata for weight refitting and IPC streaming."""
        self.refit_param_info_mcore = self._calculate_refit_param_info()

        # Collect tensor metadata for refit / hf side info
        refit_param_info_hf = {}
        # Reuse shared iterator that appends FP8 KV/Q scales when enabled
        for name, tensor in self._iter_params_with_optional_kv_scales():
            refit_param_info_hf[name] = (tensor.shape, tensor.dtype)

        return refit_param_info_hf

    def _calculate_refit_param_info(self) -> list[tuple[str, int]]:
        """Calculate parameter information for refit.

        Each task contains:
        - param_name: Local parameter name without module prefixes
        - mapping: MegatronParamMapping instance for weight transformation
        - pp_rank: Pipeline-parallel rank owning the parameter
        - vp_stage: Virtual-pipeline stage index
        - megatron_module: Reference to Megatron model/submodule
        - param_weight: Target parameter tensor for converted weight

        Returns:
            List of (parameter_name, size_in_bytes) tuples.
        """
        self.refit_conversion_tasks = self.megatron_bridge.get_conversion_tasks(
            [self.model]
        )
        param_info = []

        def calculate_size_in_bytes(param, tp_size, ep_size):
            if param is None:
                # need to broadcast for other pp ranks
                size_in_bytes = None
            else:
                # Calculate size for this parameter
                prec_to_bytes = {
                    torch.bfloat16: 2,
                    torch.float16: 2,
                    torch.float32: 4,
                }
                scale = prec_to_bytes[self.dtype] / prec_to_bytes[param.dtype]
                size_in_bytes = (
                    param.element_size() * param.numel() * tp_size * ep_size * scale
                )

            # Broadcast size_in_bytes across pipeline parallel ranks
            return broadcast_object_across_pp_ranks(size_in_bytes)

        for task in self.refit_conversion_tasks:
            param_info.append(
                (
                    task.param_name,
                    calculate_size_in_bytes(
                        task.param_weight,
                        task.mapping.tp_size,
                        task.mapping.ep_size if task.mapping.is_expert else 1,
                    ),
                )
            )
        return param_info

    def _iter_params_with_optional_kv_scales(
        self,
        kv_scales: Optional[dict[str, float]] = None,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield exported HF parameters and optionally append FP8 KV/Q scale tensors.

        This helper is used by both IPC-based streaming and collective broadcast
        so that the logic for adding KV scales stays consistent in one place.
        """
        base_iter = self.megatron_bridge.export_hf_weights(
            [self.model],
            show_progress=False,
            conversion_tasks=self.refit_conversion_tasks,  # used for metadata caching
        )

        # Yield the original parameters first.
        for name, tensor in base_iter:
            yield name, tensor

        # Check whether FP8 KV cache is enabled.
        use_fp8_kv_cache = False
        if (
            "generation" in self.cfg
            and self.cfg["generation"] is not None
            and self.cfg["generation"]["backend"] == "vllm"
        ):
            generation_cfg = cast(VllmConfig, self.cfg["generation"])
            use_fp8_kv_cache = (
                "vllm_cfg" in generation_cfg
                and "kv_cache_dtype" in generation_cfg["vllm_cfg"]
                and generation_cfg["vllm_cfg"]["kv_cache_dtype"].startswith("fp8")
            )

        if not use_fp8_kv_cache:
            return

        # Append KV (and potentially Q) scale entries to match metadata.
        num_layers = self.megatron_bridge.transformer_config.num_layers
        keys: list[str] = []
        for layer_idx in range(num_layers):
            scale_names = get_vllm_qkv_scale_names(layer_idx)
            keys.extend(scale_names.values())

        for param_name in keys:
            if kv_scales and param_name in kv_scales:
                scale_value = kv_scales[param_name]
            else:
                scale_value = 1.0
            scale_tensor = torch.tensor(
                scale_value, dtype=torch.float32, device="cuda"
            ).reshape(1)
            yield param_name, scale_tensor

    @torch.no_grad()
    @wrap_with_nvtx_name("megatron_policy_worker/stream_weights_via_ipc_zmq")
    def stream_weights_via_ipc_zmq(
        self, buffer_size_bytes: int = 0, kv_scales: Optional[dict[str, float]] = None
    ) -> None:
        """Stream model weights to peer process via ZMQ IPC socket."""
        self.maybe_init_zmq()

        from nemo_rl.models.policy.utils import stream_weights_via_ipc_zmq_impl

        # Use the shared implementation to append optional KV scales.
        stream_weights_via_ipc_zmq_impl(
            params_generator=self._iter_params_with_optional_kv_scales(
                kv_scales=kv_scales
            ),
            buffer_size_bytes=buffer_size_bytes,
            zmq_socket=self.zmq_socket,
            rank=self.rank,
            worker_name=str(self),
        )

    @torch.no_grad()
    def broadcast_weights_for_collective(
        self, kv_scales: Optional[dict[str, float]] = None
    ) -> None:
        """Broadcast the weights for collective communication."""
        # param_iterator will return (name, tensor), we only need tensor.
        packed_broadcast_producer(
            iterator=self._iter_params_with_optional_kv_scales(kv_scales=kv_scales),
            group=self.model_update_group,
            src=0,
            post_iter_func=lambda x: x[1],
        )

    def prepare_for_lp_inference(self):
        self.model = self.move_model(self.model, "cuda", move_grads=False)
        self.model.eval()

        # offload grads to cpu
        self.model = self.move_model(
            self.model, "cpu", move_params=False, move_grads=True
        )  # get rid of grad buffers

        # offload optimizer to cpu
        torch.randn(1).cuda()  # wake up torch allocator
        if (
            hasattr(self, "optimizer")
            and self.optimizer is not None
            and not self.optimizer_cpu_offload
            and self.offload_optimizer_for_logprob
        ):
            self.move_optimizer("cpu")

        gc.collect()
        torch.cuda.empty_cache()

    def prepare_for_training(self, *args, **kwargs):
        # onload models and optimizer state to cuda
        self.model = self.move_model(
            self.model, "cuda", move_grads=True, move_params=True
        )
        self.model.train()

        # Move optimizer state to CUDA if it exists
        # colocated generation will always offload optimizer to cuda before refit
        if (
            hasattr(self, "optimizer")
            and self.optimizer is not None
            and not self.optimizer_cpu_offload
            and (self.offload_optimizer_for_logprob or self.is_generation_colocated)
        ):
            self.move_optimizer("cuda")

        if self.cfg["megatron_cfg"]["empty_unused_memory_level"] >= 1:
            torch.cuda.empty_cache()

    @wrap_with_nvtx_name("megatron_policy_worker/offload_before_refit")
    def offload_before_refit(self):
        """Offload the optimizer and buffers to the CPU."""
        no_grad = torch.no_grad()
        no_grad.__enter__()
        allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
        reserved = torch.cuda.memory_reserved() / (1024**3)  # Convert to GB
        print(
            f"GPU Memory before optimizer offload: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )
        self.model = self.move_model(
            self.model, "cpu", move_params=False, move_grads=True
        )  # get rid of grad buffers
        torch.randn(1).cuda()  # wake up torch allocator
        if (
            hasattr(self, "optimizer")
            and self.optimizer is not None
            and not self.optimizer_cpu_offload
        ):
            self.move_optimizer("cpu")

        gc.collect()
        torch.cuda.empty_cache()

        # Print memory stats after offloading
        allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
        reserved = torch.cuda.memory_reserved() / (1024**3)  # Convert to GB
        print(
            f"GPU Memory after optimizer offload: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )
        no_grad.__exit__(None, None, None)

    @wrap_with_nvtx_name("megatron_policy_worker/offload_after_refit")
    def offload_after_refit(self):
        """Offload as much as possible on the CPU."""
        no_grad = torch.no_grad()
        no_grad.__enter__()
        self.model = self.move_model(self.model, "cpu")
        self.model.eval()
        torch.randn(1).cuda()  # wake up torch allocator
        self.offload_before_refit()  # rerun the old offload function

        allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
        reserved = torch.cuda.memory_reserved() / (1024**3)  # Convert to GB
        print(
            f"GPU Memory after refit complete: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )
        no_grad.__exit__(None, None, None)

    @torch.no_grad()
    def move_model(
        self,
        model: torch.nn.Module,
        device: str,
        move_params: bool = True,
        move_grads: bool = True,
    ) -> torch.nn.Module:
        # move all param and grad buffers to the device
        if isinstance(model, DistributedDataParallel):
            # DDP case
            for buffers in [model.buffers, model.expert_parallel_buffers]:
                for buffer_idx in range(len(buffers)):
                    if device == "cpu":
                        buffers[buffer_idx].offload_to_cpu(
                            move_params=move_params, move_grads=move_grads
                        )
                    elif device == "cuda":
                        buffers[buffer_idx].reload_from_cpu(
                            move_params=move_params, move_grads=move_grads
                        )
                    else:
                        raise ValueError(
                            f"Invalid device: {device}. Only strings 'cpu' and 'cuda' are supported."
                        )
        elif isinstance(model, custom_FSDP):
            if device == "cpu":
                model.param_and_grad_buffer.offload_to_cpu(move_params, move_grads)
            elif device == "cuda":
                model.param_and_grad_buffer.reload_from_cpu(
                    move_params=move_params, move_grads=move_grads
                )
            else:
                raise ValueError(
                    f"Invalid device: {device}. Only strings 'cpu' and 'cuda' are supported."
                )
        else:
            # Ordinary offload case
            if move_params:
                for name, param in model.state_dict().items():
                    new_state_dict = {}
                    for name, item in model.state_dict().items():
                        if isinstance(item, torch.Tensor):
                            item = item.detach().to(
                                device=device, non_blocking=True, copy=True
                            )
                        new_state_dict[name] = item
                    model.load_state_dict(new_state_dict)
        return model

    def move_optimizer(self, device: str):
        # Iterate through the state dictionaries for each parameter group
        if isinstance(self.optimizer, ChainedOptimizer):
            optimizer_state = self.optimizer.state
        else:
            optimizer_state = self.optimizer._get_state()
        for _, state in optimizer_state.items():
            # Iterate through the state items (e.g., momentum, variance) for a parameter
            for k, v in state.items():
                # Check if the item is a tensor
                if torch.is_tensor(v):
                    # Move the tensor to device and update the state dictionary
                    if device == "cpu":
                        if v.is_cuda:
                            state[k] = v.to("cpu")
                    elif device == "cuda":
                        if not v.is_cuda:
                            state[k] = v.to("cuda")
                    else:
                        raise ValueError(
                            f"Invalid device: {device}. Only strings 'cpu' and 'cuda' are supported."
                        )

    def save_checkpoint(
        self,
        weights_path: str,
        optimizer_path: Optional[str] = None,
        **kwargs,
    ):
        """Save a training checkpoint.

        Args:
            weights_path: The specific directory path where the checkpoint will be saved.
            optimizer_path: If not None, optimizer and scheduler states are saved if they exist.
        """
        if not torch.distributed.is_initialized():
            raise RuntimeError(
                "Distributed process group is not initialized. Cannot save checkpoint."
            )

        if self.mcore_state is None or self.model is None:
            raise RuntimeError(
                "Megatron core state or model is not initialized. Cannot save checkpoint."
            )

        original_save_path = self.mcore_state.cfg.checkpoint.save
        # save_dir = os.path.dirname(weights_path)
        release_name = os.path.basename(weights_path)

        try:
            maybe_finalize_async_save(
                self.mcore_state,
                ckpt_cfg=self.mcore_state.cfg.checkpoint,
                blocking=False,
            )
            self.mcore_state.cfg.checkpoint.save = weights_path

            optimizer_to_save = None
            scheduler_to_save = None

            if optimizer_path is not None:
                if self.optimizer is not None:
                    optimizer_to_save = self.optimizer
                if self.scheduler is not None:
                    scheduler_to_save = self.scheduler

            # Ensure model is in eval mode for consistent saving, unless actively training
            # This is a common practice, though NeMo's save might handle this.
            # For safety, if not in training loop, setting to eval.
            is_training = self.model.training
            if not is_training:
                self.model.eval()

            if self.should_disable_forward_pre_hook:
                self.disable_forward_pre_hook()
            save_checkpoint(
                state=self.mcore_state,
                model=[self.model],
                optimizer=optimizer_to_save,
                opt_param_scheduler=scheduler_to_save,
                num_floating_point_operations_so_far=self.mcore_state.train_state.floating_point_operations_so_far,
                checkpointing_context=self.checkpointing_context,
            )
            print(f"Saved checkpoint to {weights_path}")
            maybe_finalize_async_save(
                self.mcore_state,
                ckpt_cfg=self.mcore_state.cfg.checkpoint,
                blocking=True,
                terminate=True,
            )
            if self.should_disable_forward_pre_hook:
                self.enable_forward_pre_hook()

            if not is_training:  # Restore training state if it was changed
                self.model.train()

        except Exception as e:
            print(f"Failed to save checkpoint to {weights_path}: {e}")
            raise
        finally:
            self.mcore_state.cfg.checkpoint.save = original_save_path

    def load_checkpoint(self, weights_path: str, optimizer_path: Optional[str] = None):
        """Load a training checkpoint.

        Args:
            weights_path: The exact directory path from which to load the checkpoint.
            optimizer_path: If not None, attempts to load optimizer and scheduler states
                            if self.optimizer and self.scheduler are initialized.
        """
        raise NotImplementedError(
            "Loading checkpoints outside of the init function is not yet implemented for Megatron policy."
        )

    def check_tensor_parallel_attributes(self) -> dict[str, Any]:
        """Check tensor parallel attributes on model parameters.

        Returns:
            Dictionary containing information about tensor parallel parameters:
            - tp_params: List of parameter names that have tensor_model_parallel=True
            - non_tp_params: List of parameter names that have tensor_model_parallel=False
            - total_params: Total number of parameters checked
            - tp_size: Tensor parallel size from config
        """
        tp_params = []
        non_tp_params = []
        total_params = 0

        for name, param in self.model.named_parameters():
            total_params += 1
            tensor_model_parallel = getattr(param, "tensor_model_parallel", False)

            if tensor_model_parallel:
                tp_params.append(
                    {
                        "name": name,
                        "tensor_model_parallel": tensor_model_parallel,
                        "partition_dim": getattr(param, "partition_dim", None),
                        "partition_stride": getattr(param, "partition_stride", None),
                        "shape": list(param.shape),
                    }
                )
            else:
                non_tp_params.append(
                    {
                        "name": name,
                        "tensor_model_parallel": tensor_model_parallel,
                        "shape": list(param.shape),
                    }
                )

        return {
            "tp_params": tp_params,
            "non_tp_params": non_tp_params,
            "total_params": total_params,
            "tp_size": self.megatron_cfg.model.tensor_model_parallel_size,
        }

    @torch.no_grad()
    def calibrate_qkv_fp8_scales(
        self,
        *,
        data: BatchedDataDict[Any],
        micro_batch_size: Optional[int] = None,
        percentile: float = 99.9,
        margin: float = 1.05,
        include_q: bool = False,
    ) -> dict[str, Any]:
        """One-shot calibration of Q/K/V activation scales (for FP8 KV cache).

        - Captures each layer's `query_key_value` output through forward hooks, splits Q/K/V, and computes percentile amax.
        - In parallel (DP/TP/PP) environments, first computes local percentiles, then takes max across all ranks for conservativeness.
        - By default only returns and saves K/V scales, optionally returns Q.

        Args:
            data: Representative sample batch for calibration, following get_logprobs input conventions.
            micro_batch_size: Micro batch size during calibration; if None, reuses logprob_batch_size.
            percentile: Percentile for amax (e.g. 99.9).
            margin: Margin factor, e.g. 1.05.
            save_path: If provided, rank0 will save results as JSON.
            include_q: Whether to also return Q scale (usually only K/V needed).

        Returns:
            { "format": "fp8", "percentile": float, "margin": float,
              "layers": { layer_name: {"k_scale": float, "v_scale": float[, "q_scale": float] } } }
        """

        # Allow overriding FP8 max for Q, K, V via environment variables for ease of testing.
        # Defaults align with FP8 e4m3 max magnitude.
        # Use different defaults for Q, K, V to adapt to distribution diffefences
        def _get_env_float(name: str, default: float) -> float:
            try:
                val = os.getenv(name, None)
                return float(val) if val is not None and val != "" else default
            except Exception:
                return default

        FP8_MAX_Q = _get_env_float("FP8_MAX_Q", 448.0)
        FP8_MAX_K = _get_env_float("FP8_MAX_K", 448.0)
        FP8_MAX_V = _get_env_float("FP8_MAX_V", 448.0)

        self.model.eval()

        # Record local percentile amax for q/k/v of each layer
        layer_to_samples_q: dict[str, list[float]] = defaultdict(list)
        layer_to_samples_k: dict[str, list[float]] = defaultdict(list)
        layer_to_samples_v: dict[str, list[float]] = defaultdict(list)
        hook_handles = []

        def _extract_layer_key(module_name: str) -> str:
            # Expected format: "module.decoder.layers.<idx>.self_attention.query_key_value"
            m = re.search(r"module\.decoder\.layers\.(\d+)", module_name)
            if m is not None:
                return f"layer_{m.group(1)}"
            return module_name

        # Hook to capture q/k/v after q/k norm and RoPE
        def _pre_hook_builder_core_attention(module_name: str):
            layer_key = _extract_layer_key(module_name)

            def _pre_hook(module, inputs):
                args = inputs if isinstance(inputs, (tuple, list)) else (inputs,)
                if len(args) == 1 and isinstance(args[0], (tuple, list)):
                    args = args[0]
                # Expected first 3 args to be q, k, v (typical signature for Megatron CoreAttention)
                q = args[0]
                k = args[1]
                v = args[2]
                if include_q:
                    layer_to_samples_q[layer_key].append(
                        float(torch.amax(torch.abs(q)).item())
                    )
                layer_to_samples_k[layer_key].append(
                    float(torch.amax(torch.abs(k)).item())
                )
                layer_to_samples_v[layer_key].append(
                    float(torch.amax(torch.abs(v)).item())
                )

            return _pre_hook

        matched_modules = []
        # Try to register forward_pre_hook on core_attention first
        for name, module in self.model.named_modules():
            if "self_attention.core_attention" in name:
                try:
                    handle = module.register_forward_pre_hook(
                        _pre_hook_builder_core_attention(name)
                    )
                    hook_handles.append(handle)
                    matched_modules.append((name, module.__class__.__name__, "pre"))
                except Exception as e:
                    print(
                        f"Error registering pre-hook for qkv scale calibration on {name}: {e}"
                        " Please check if the model is compatible with the current calibration logic. "
                        "The expected module name is 'self_attention.core_attention'."
                    )
                    raise

        # Run a forward pass to trigger hooks (reuse get_logprobs forward path)
        try:
            _ = self.get_logprobs(data=data, micro_batch_size=micro_batch_size)
        finally:
            for h in hook_handles:
                try:
                    h.remove()
                except Exception as e:
                    print(f"Error removing hook for qkv scale calibration: {e}")
                    raise

        # Compute local percentile amax
        def _percentile(values: list[float], p: float) -> float:
            if not values:
                return 0.0
            t = torch.tensor(sorted(values), device="cuda", dtype=torch.float32)
            rank = max(
                0, min(len(values) - 1, int(round((p / 100.0) * (len(values) - 1))))
            )
            return float(t[rank].item())

        local_layer_to_pamax = {}
        for layer_key in set(
            list(layer_to_samples_k.keys())
            + list(layer_to_samples_v.keys())
            + (list(layer_to_samples_q.keys()) if include_q else [])
        ):
            entry = {}
            if include_q:
                entry["q_amax_p"] = _percentile(
                    layer_to_samples_q.get(layer_key, []), percentile
                )
            entry["k_amax_p"] = _percentile(
                layer_to_samples_k.get(layer_key, []), percentile
            )
            entry["v_amax_p"] = _percentile(
                layer_to_samples_v.get(layer_key, []), percentile
            )
            local_layer_to_pamax[layer_key] = entry

        # Merge across all ranks: take maximum of percentile amax (conservative approach)
        world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )
        gathered = [None for _ in range(world_size)] if world_size > 1 else None
        if world_size > 1:
            torch.distributed.all_gather_object(gathered, local_layer_to_pamax)
            merged = defaultdict(dict)
            for d in gathered:  # type: ignore
                if d is None:
                    continue
                for k, v in d.items():
                    dst = merged[k]
                    for kk, vv in v.items():
                        dst[kk] = max(dst.get(kk, 0.0), float(vv))
            layer_to_pamax = dict(merged)
        else:
            layer_to_pamax = local_layer_to_pamax

        # Compute scale (symmetric quantization): scale = pamax / fp8_max
        result_layers = {}
        for layer_key, vals in layer_to_pamax.items():
            out_entry = {}
            if include_q:
                q_scale = (vals.get("q_amax_p", 0.0) * margin) / FP8_MAX_Q
                out_entry["q_scale"] = float(q_scale)
            k_scale = (vals.get("k_amax_p", 0.0) * margin) / FP8_MAX_K
            v_scale = (vals.get("v_amax_p", 0.0) * margin) / FP8_MAX_V
            out_entry["k_scale"] = float(k_scale)
            out_entry["v_scale"] = float(v_scale)
            result_layers[layer_key] = out_entry

        vllm_format_scales = convert_calibration_to_vllm_format(result_layers)

        final_result = {
            "format": "fp8",
            "percentile": percentile,
            "margin": margin,
            "layers": vllm_format_scales,
        }

        # Sync results across all ranks (broadcast rank0's result)
        if world_size > 1:
            if torch.distributed.get_rank() == 0:
                obj_list = [final_result]
                torch.distributed.broadcast_object_list(obj_list, src=0)
                final_result = obj_list[0]
            else:
                obj_list = [None]
                torch.distributed.broadcast_object_list(obj_list, src=0)
                final_result = obj_list[0]  # type: ignore

        return final_result


class CustomFloat16Module(Float16Module):
    """Float 16 Module.

    Attributes:
        config (TransformerConfig): Transformer config
        fp16 (bool) : Specifies if the model runs in fp16 mode
        bf16 (bool) : Specifies if the model runs in bf16 mode

    Args:
        config (TransformerConfig): The transformer config used to initalize the model
    """

    def __init__(self, config: TransformerConfig, module: torch.nn.Module):
        super(CustomFloat16Module, self).__init__(config, module)
        self.re_enable_float32_expert_bias()

    def re_enable_float32_expert_bias(self) -> None:
        """Ensure MoE router expert bias stays in float32 for numerical stability.

        Walks the wrapped module to find MoE routers and invokes the
        `_maintain_float32_expert_bias()` helper which recreates or casts the
        expert bias tensors to float32 as required by Megatron-LM.
        """
        module = self.module
        # Handle VLM models where language model is nested
        if hasattr(module, "language_model"):
            module = module.language_model
        if hasattr(module, "decoder") and hasattr(module.decoder, "layers"):
            for layer in module.decoder.layers:
                mlp = getattr(layer, "mlp", None)
                router = getattr(mlp, "router", None) if mlp is not None else None
                if router is not None and hasattr(
                    router, "_maintain_float32_expert_bias"
                ):
                    router._maintain_float32_expert_bias()

```

All the linter warnings are pre-existing import resolution issues (the linter can't find `ray`, `torch`, `megatron` packages). My changes are correct.

## Summary of Option 2 Implementation

I've added `disable_forward_pre_hook()` calls **before** `toggle_cuda_graphs(set_to="none")` in two locations:

### Location 1: In `train()` method (around line 971)

```967:975:/lustre/fsw/portfolios/coreai/users/shanmugamr/RL/nemo_rl/models/policy/workers/megatron_policy_worker.py
        # Disable forward pre-hooks BEFORE toggle_cuda_graphs to avoid stale hook handles
        # for CudaGraphManager modules that will be removed by toggle_cuda_graphs.
        if self.should_disable_forward_pre_hook:
            self.disable_forward_pre_hook()

        toggle_cuda_graphs(lang_module, set_to="none", reset_cuda_graphs=True)
```

### Location 2: In `generate()` method (around line 2045)

```2044:2052:/lustre/fsw/portfolios/coreai/users/shanmugamr/RL/nemo_rl/models/policy/workers/megatron_policy_worker.py
        # Disable forward pre-hooks BEFORE toggle_cuda_graphs to avoid stale hook handles
        # for CudaGraphManager modules that will be removed by toggle_cuda_graphs.
        if self.should_disable_forward_pre_hook:
            self.disable_forward_pre_hook()

        toggle_cuda_graphs(lang_module, set_to="none", reset_cuda_graphs=True)
```

### Why This Works

By calling `disable_forward_pre_hook()` **before** `toggle_cuda_graphs(set_to="none")`:

1. The forward hooks are removed while `CudaGraphManager` modules are **still in the module tree**
2. `remove_forward_pre_hook_handles` gets cleared properly
3. Then `toggle_cuda_graphs` can safely delete the `CudaGraphManager` modules
4. No orphaned hook handles remain

You now have **both fixes** in place (Option 1 in DDP + Option 2 in the worker), which provides defense in depth.

```python
        # Disable forward pre-hooks BEFORE toggle_cuda_graphs to avoid stale hook handles
        # for CudaGraphManager modules that will be removed by toggle_cuda_graphs.
        if self.should_disable_forward_pre_hook:
            self.disable_forward_pre_hook()

        toggle_cuda_graphs(lang_module, set_to="none", reset_cuda_graphs=True)
```

```python
        # Disable forward pre-hooks BEFORE toggle_cuda_graphs to avoid stale hook handles
        # for CudaGraphManager modules that will be removed by toggle_cuda_graphs.
        if self.should_disable_forward_pre_hook:
            self.disable_forward_pre_hook()

        toggle_cuda_graphs(lang_module, set_to="none", reset_cuda_graphs=True)
```

---

**User**

I get this issue
```
Traceback (most recent call last):
  File "/opt/nemo-rl/examples/run_grpo_math.py", line 260, in <module>
    main()
  File "/opt/nemo-rl/examples/run_grpo_math.py", line 243, in main
    grpo_train(
  File "/opt/nemo-rl/nemo_rl/algorithms/grpo.py", line 1191, in grpo_train
    repeated_batch, rollout_metrics = run_multi_turn_rollout(
                                      ^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/nemo-rl/nemo_rl/experience/rollouts.py", line 420, in run_multi_turn_rollout
    active_batch, generated_ids, gen_metrics = generate_responses(
                                               ^^^^^^^^^^^^^^^^^^^
  File "/opt/nemo-rl/nemo_rl/experience/rollouts.py", line 74, in generate_responses
    generation_outputs = policy_generation.generate(
                         ^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/nemo-rl/nemo_rl/models/policy/lm_policy.py", line 588, in generate
    self.worker_group.get_all_worker_results(futures),
    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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
ray.exceptions.RayTaskError: ray::MegatronPolicyWorker.generate() (pid=785851, ip=10.65.29.29, actor_id=c135c9ce96c668bb65036ec201000000, repr=MegatronPolicyWorker[rank=0])
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/ray_venvs/nemo_rl.models.policy.workers.megatron_policy_worker.MegatronPolicyWorker/lib/python3.12/site-packages/torch/utils/_contextlib.py", line 120, in decorate_context
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/opt/nemo-rl/nemo_rl/utils/nsys.py", line 88, in wrapper
    ret = func(*args, **kwargs)
          ^^^^^^^^^^^^^^^^^^^^^
  File "/opt/nemo-rl/nemo_rl/models/policy/workers/megatron_policy_worker.py", line 2046, in generate
    self.disable_forward_pre_hook()
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/nemo-rl/nemo_rl/models/policy/workers/megatron_policy_worker.py", line 944, in disable_forward_pre_hook
    self.model.disable_forward_pre_hook(param_sync=param_sync)
  File "/opt/nemo-rl/3rdparty/Megatron-LM-workspace/Megatron-LM/megatron/core/distributed/distributed_data_parallel.py", line 396, in disable_forward_pre_hook
    assert self.remove_forward_pre_hook_handles[module] is not None
           ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^
KeyError: CudaGraphManager()
```

---

**Cursor**

```plaintext
File "/opt/nemo-rl/3rdparty/Megatron-LM-workspace/Megatron-LM/megatron/core/distributed/distributed_data_parallel.py", line 396, in disable_forward_pre_hook
    assert self.remove_forward_pre_hook_handles[module] is not None
```

Let me check the current state of the DDP file - it seems the Option 1 fix may not be in place:

I see - the Option 1 fix needs to be re-applied. The error is now a `KeyError` because `CudaGraphManager` was added **after** `enable_forward_pre_hook` was called, so it's not in the handles dict. Let me re-apply the fix:

```python
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.

import logging
from contextlib import contextmanager
from typing import Optional

import torch

from .. import parallel_state
from ..config_logger import has_config_logger_enabled, log_config_to_disk
from ..fp8_utils import is_float8tensor, post_all_gather_processing
from ..process_groups_config import ProcessGroupCollection
from ..transformer.cuda_graphs import is_graph_capturing
from ..transformer.transformer_config import TransformerConfig
from ..utils import log_single_rank
from .data_parallel_base import _BaseDataParallel
from .distributed_data_parallel_config import DistributedDataParallelConfig
from .param_and_grad_buffer import _ParamAndGradBuffer, partition_buckets

logger = logging.getLogger(__name__)


class DistributedDataParallel(_BaseDataParallel):
    """
    DDP wrapper which stores grads in contiguous buffers. Also has option of overlapping
    communication with backprop computation by breaking up full model's gradients into smaller
    buckets and running all-reduce / reduce-scatter on each bucket asynchronously. This class
    also provides the option to do the gradient accumulation in a type other than the param type
    (e.g., fp32 for a bf16 model).

    Args:
        config: Transformer config object.
        ddp_config: DistributedDataParallel config object.
        module: Underlying model.
        disable_bucketing: If true, force assign all parameters to a single bucket. If false,
            use standard bucketing policy: assign parameters to smaller buckets and all-reduce
            per bucket _if_ overlap_grad_reduce is True and pp_rank is 0.
        pg_collection: Optional unified process group for distributed training.

    """

    def __init__(
        self,
        config: TransformerConfig,
        ddp_config: DistributedDataParallelConfig,
        module: torch.nn.Module,
        disable_bucketing: bool = False,
        pg_collection: Optional[ProcessGroupCollection] = None,
    ):
        super().__init__(config=config, module=module)
        if has_config_logger_enabled(config):
            log_config_to_disk(config, locals(), prefix=type(self).__name__)

        # If bucket_size is not provided as an input, use sane default.
        # If using very large dp_sizes, make buckets larger to ensure that chunks used in NCCL
        # ring-reduce implementations are large enough to remain bandwidth-bound rather than
        # latency-bound.
        if ddp_config.bucket_size is None:
            ddp_config.bucket_size = max(
                40000000, 1000000 * parallel_state.get_data_parallel_world_size()
            )
        # Set bucket_size to infinity if overlap_grad_reduce is False.
        if not ddp_config.overlap_grad_reduce:
            ddp_config.bucket_size = None

        self.ddp_config = ddp_config
        log_single_rank(
            logger,
            logging.INFO,
            f'Setting up DistributedDataParallel with config {self.ddp_config}',
        )

        if pg_collection is None:
            self.dp_group = parallel_state.get_data_parallel_group(
                with_context_parallel=False, partial_data_parallel=False
            )
            self.dp_cp_group = parallel_state.get_data_parallel_group(
                with_context_parallel=True, partial_data_parallel=False
            )
            self.intra_dp_cp_group = parallel_state.get_data_parallel_group(
                with_context_parallel=True, partial_data_parallel=True
            )
            self.expt_dp_group = parallel_state.get_expert_data_parallel_group()
            self.intra_expt_dp_group = parallel_state.get_expert_data_parallel_group(
                partial_expert_data_parallel=True
            )
            if self.ddp_config.num_distributed_optimizer_instances > 1:
                self.inter_dist_opt_group = (
                    parallel_state.get_inter_distributed_optimizer_instance_group()
                )
            self.tp_group = parallel_state.get_tensor_model_parallel_group()
            self.pp_group = parallel_state.get_pipeline_model_parallel_group()
            self.ep_group = parallel_state.get_expert_model_parallel_group()
        else:
            # Setup process groups using DDP-specific helper method
            process_groups = ProcessGroupCollection.setup_process_groups_for_ddp(
                pg_collection, config, self.ddp_config
            )

            self.dp_group = process_groups['dp_group']
            self.dp_cp_group = process_groups['dp_cp_group']
            self.intra_dp_cp_group = process_groups['intra_dp_cp_group']
            self.expt_dp_group = process_groups['expt_dp_group']
            self.intra_expt_dp_group = process_groups['intra_expt_dp_group']
            self.tp_group = process_groups['tp_group']
            self.pp_group = process_groups['pp_group']
            self.ep_group = process_groups['ep_group']

            # Set inter_dist_opt_group if multiple optimizer instances
            if self.ddp_config.num_distributed_optimizer_instances > 1:
                self.inter_dist_opt_group = process_groups['inter_dist_opt_group']

        # Turn off bucketing if we are on a pipeline stage that is not the first (since
        # data-parallel communication on these stages is not on the critical path), or if
        # disable_bucketing is True (e.g., we might not want to break up model parameters
        # into buckets for model chunks after the first in the interleaved schedule).
        self.bucket_size = self.ddp_config.bucket_size
        if isinstance(self.pp_group, list):
            pp_rank = self.pp_group[0].rank()
        else:
            pp_rank = self.pp_group.rank()
        if disable_bucketing or pp_rank > 0:
            self.bucket_size = None

        self.param_to_bucket_group = {}

        # Group parameters by their gradient type.
        param_to_name = {}
        dense_params = []
        expert_parallel_params = []
        self.params_with_grad = []
        for name, param in self.module.named_parameters():
            if not param.requires_grad:
                continue

            # Track params with grad to enable direct setting
            # of param.grad_added_to_main_grad
            self.params_with_grad.append(param)

            param.grad_added_to_main_grad = False
            param_to_name[param] = name

            if getattr(param, 'allreduce', True):
                dense_params.append(param)
            else:
                expert_parallel_params.append(param)

        def _allocate_buffers_for_parameters(
            input_params, data_parallel_group, gradient_scaling_factor
        ):
            param_and_grad_dtype_to_params = {}
            param_and_grad_dtype_to_offsets = {}
            param_and_grad_dtype_to_indices = {}

            # Group parameters by their gradient type.
            for param in input_params:
                assert param.requires_grad

                param_dtype = param.dtype
                if is_float8tensor(param):
                    # Currently TE's Float8Tensor is a wrapper of torch.Tensor. It has a "fake"
                    # dtype (usually a higher precision dtype such as bfloat16), but its actual
                    # data is stored in the form of a torch uint8 tensor within the Float8Tensor's
                    # ".data" attribute. Therefore, when creating the param buffer for fp8 params,
                    # it is necessary to use torch.uint8, not the "fake" dtype got from
                    # "param.dtype".
                    param_dtype = torch.uint8
                grad_dtype = torch.float if self.ddp_config.grad_reduce_in_fp32 else param.dtype

                params = param_and_grad_dtype_to_params.get((param_dtype, grad_dtype), [])
                params.append(param)
                param_and_grad_dtype_to_params[(param_dtype, grad_dtype)] = params

                # Get the index of each param among the params with same dtype, if a param is fp8,
                # use its "fake" high precision dtype to find which params have same dtype with it.
                # For example:
                #     Case 1:
                #         params = [p1(bf16), p2(bf16), p3(bf16), p4(bf16)]
                #         param_and_grad_dtype_to_indices = {
                #             (torch.bfloat16, torch.float32): [0, 1, 2, 3],
                #         }
                #     Case 2:
                #         params = [p1(bf16), p2(fp8), p3(fp8), p4(bf16)]
                #         param_and_grad_dtype_to_indices = {
                #             (torch.bfloat16, torch.float32): [0, 3],
                #             (torch.uint8, torch.float32): [1, 2],
                #         }
                # We need these indices to load a non-native-fp8 checkpoint in native-fp8 mode.
                offset = param_and_grad_dtype_to_offsets.get((param.dtype, grad_dtype), 0)
                param_and_grad_dtype_to_offsets[(param.dtype, grad_dtype)] = offset + 1
                indices = param_and_grad_dtype_to_indices.get((param_dtype, grad_dtype), [])
                indices.append(offset)
                param_and_grad_dtype_to_indices[(param_dtype, grad_dtype)] = indices

            if not config.calculate_per_token_loss:
                target_gradient_scaling_factor = 1.0 / self.dp_cp_group.size()
                if self.ddp_config.average_in_collective:
                    if self.ddp_config.num_distributed_optimizer_instances == 1:
                        # Collective is averaging gradients in collective with data_parallel_group.
                        assert (
                            gradient_scaling_factor / data_parallel_group.size()
                            == target_gradient_scaling_factor
                        )
                    else:
                        # For non-expert parameters, gradient_scaling_factor is 1.
                        # For expert parameters, gradient_scaling_factor is edp_size/dp_size.
                        assert (gradient_scaling_factor == 1) or (
                            gradient_scaling_factor
                            == (self.expt_dp_group.size() / self.dp_cp_group.size())
                        )
                else:
                    assert gradient_scaling_factor == target_gradient_scaling_factor

            # Allocate the grad buffers and map the grads.
            buffers = []
            pg_collection = ProcessGroupCollection()
            pg_collection.tp = self.tp_group
            pg_collection.dp_cp = self.dp_cp_group
            for (param_dtype, grad_dtype), params in param_and_grad_dtype_to_params.items():
                buffers.append(
                    _ParamAndGradBuffer(
                        self.ddp_config,
                        param_dtype,
                        grad_dtype,
                        params,
                        data_parallel_group,
                        self.bucket_size,
                        param_to_name,
                        gradient_scaling_factor,
                        param_and_grad_dtype_to_indices[(param_dtype, grad_dtype)],
                        self.ddp_config.nccl_ub,
                        pg_collection,
                    )
                )

            # In some scenarios, we want to put buckets from different buffers into a group so that
            # their communication can be aggregated. For example, when there are both fp8 buffers
            # and bf16 buffers in the model and vpp is enabled, each model chunk will have an fp8
            # bucket and a bf16 bucket, which doubles the number of communication kernels, and
            # because of the use of CUDA_DEVICE_MAX_CONNECTIONS=1, having multiple back-to-back
            # communications will prevent the overlap of the communication kernels with computation
            # kernels.
            # If bucketing is explicitly disabled, then put all buckets in a buffer into a single
            # bucket group.
            bucket_groups = partition_buckets(buffers, force_single_bucket_group=disable_bucketing)

            if self.ddp_config.num_distributed_optimizer_instances > 1:
                assert (
                    self.ddp_config.use_distributed_optimizer
                ), 'Partial DistOpt cannot be used without DistOpt'
                communication_stream = torch.cuda.Stream(device=torch.cuda.current_device())
                for bucket_group in bucket_groups:
                    bucket_group.inter_distributed_optimizer_instance_group = (
                        self.inter_dist_opt_group
                    )
                    bucket_group.communication_stream = communication_stream

            # Set `next_param_gather_bucket_group` for different bucket groups by iterating through
            # buckets in reverse order (since all-gathers happen in reverse order of buckets).
            if self.ddp_config.use_distributed_optimizer and self.ddp_config.overlap_param_gather:
                num_bucket_groups = len(bucket_groups)
                for i in range(1, num_bucket_groups):
                    bucket_groups[num_bucket_groups - i].next_param_gather_bucket_group = (
                        bucket_groups[num_bucket_groups - i - 1]
                    )

            # Create map from param to bucket group, used in pre_hook.
            for bucket_group in bucket_groups:
                for bucket in bucket_group.buckets:
                    for param in bucket.params_list:
                        self.param_to_bucket_group[param] = bucket_group

            return buffers, bucket_groups

        if config.calculate_per_token_loss:
            assert (
                not self.ddp_config.average_in_collective
            ), "Cannot average in collective when calculating per-token loss!"
            gradient_scaling_factor = 1.0
            expert_gradient_scaling_factor = 1.0
        else:
            # The goal is to scale reduced gradients by 1/dp_size.
            # This can be achieved in two ways:
            #
            # Case 1: average_in_collective=True
            # - Non-expert parameters:
            #   1. No pre-scaling (gradient_scaling_factor=1.0)
            #   2. Do average reduction over dp group (equals to sum then divide by dp_size)
            #   3. Final result is scaled by 1/dp_size as desired
            #
            # - Expert parameters:
            #   1. Scale by edp_size/dp_size before reduction
            #   2. Do average reduction over edp group (equals to sum then divide by edp_size)
            #   3. Resulted scaling: (edp_size/dp_size) * (1/edp_size) = 1/dp_size as desired
            #   (edp_size = expert data parallel world size)
            #
            # Case 2: average_in_collective=False
            # - Both expert and non-expert parameters:
            #   1. Scale gradients by 1/dp_size before reduction
            #   2. Do sum reduction across data parallel ranks
            #   3. Final result is scaled by 1/dp_size as desired
            if self.ddp_config.average_in_collective:
                gradient_scaling_factor = 1.0
                expert_gradient_scaling_factor = self.expt_dp_group.size() / self.dp_cp_group.size()
            else:
                data_parallel_world_size = self.dp_cp_group.size()

                gradient_scaling_factor = 1.0 / data_parallel_world_size
                expert_gradient_scaling_factor = 1.0 / data_parallel_world_size

        # Allocate the param+grad buffers for dense params' grads.
        self.buffers, self.bucket_groups = _allocate_buffers_for_parameters(
            dense_params, self.intra_dp_cp_group, gradient_scaling_factor=gradient_scaling_factor
        )

        # Allocate separate param+grad buffers for expert parallel params' grads.
        self.expert_parallel_buffers, self.expert_parallel_bucket_groups = (
            _allocate_buffers_for_parameters(
                expert_parallel_params,
                self.intra_expt_dp_group,
                gradient_scaling_factor=expert_gradient_scaling_factor,
            )
        )

        # Delete references to weight_tensor if they exist since we don't want two parameter copies
        # if we re-mapped parameters (which happens when we use the distributed optimizer).
        # This is a temporary workaround around a TE bug that is fixed with
        # https://github.com/NVIDIA/TransformerEngine/pull/719.
        if self.ddp_config.use_distributed_optimizer:

            @torch.no_grad()
            def unmap_weight_tensor(m):
                if hasattr(m, 'weight_tensor'):
                    m.weight_tensor = None

            self.module.apply(unmap_weight_tensor)

        # Register backward hook.
        # Accumulation function for the gradients need to be stored so they
        # don't go out of scope.
        self.grad_accs = []
        for param in self.module.parameters():
            if param.requires_grad:
                # When delay_wgrad_compute is True and the param is marked with
                # skip_backward_post_hook, register the backward post hook for its module
                # instead of the param so that the wgrad accumulation and reduce will be performed
                # in backward_dw() method of the module instead of the hook of backward() method.
                # Otherwise, register the backward post hook for the param.
                if self.ddp_config.delay_wgrad_compute and getattr(
                    param, 'skip_backward_post_hook', False
                ):
                    for module in self.module.modules():
                        if hasattr(module, "register_wgrad_accumulation_and_reduce_hooks"):
                            for param_value in module.parameters():
                                if param is param_value:
                                    module.register_wgrad_accumulation_and_reduce_hooks(
                                        self._make_backward_post_hook(param)
                                    )
                                    break
                else:
                    # Expand so we get access to grad_fn.
                    param_tmp = param.expand_as(param)
                    # Get the gradient accumulator function.
                    grad_acc = param_tmp.grad_fn.next_functions[0][0]
                    grad_acc.register_hook(self._make_backward_post_hook(param))
                    self.grad_accs.append(grad_acc)

        self.use_forward_hook = (
            self.ddp_config.use_distributed_optimizer and self.ddp_config.overlap_param_gather
        )
        self.remove_forward_pre_hook_handles = {}
        if self.use_forward_hook:
            self.enable_forward_pre_hook()
        self.overlap_param_gather_with_optimizer_step = False

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

    def disable_forward_pre_hook(self, param_sync: bool = True):
        """
        Disable forward pre-hooks needed for param all-gather overlap with forward compute.
        Skip synchronous param all-gather if `param_sync` is False.
        """
        assert self.use_forward_hook
        # De-register forward pre-hook for all registered modules.
        # We iterate over the dict keys directly (instead of self.module.modules()) to handle
        # cases where modules may have been dynamically added or removed from the module tree
        # (e.g., CudaGraphManager objects added/removed by toggle_cuda_graphs).
        for module, handle in list(self.remove_forward_pre_hook_handles.items()):
            handle.remove()
        self.remove_forward_pre_hook_handles.clear()

        # Force synchronize parameters.
        if param_sync:
            self.start_param_sync(force_sync=True)

    def _make_forward_pre_hook(self):
        """
        Create a forward pre-hook to wait on all-gather handles when necessary (i.e.,
        when a module uses a parameter in a bucket with a still incomplete all-gather).
        """

        def hook(module, *unused):
            assert (
                self.use_forward_hook
            ), "Should use pre-hook only when overlap_param_gather is True"

            if is_graph_capturing():
                return

            # Make sure all parameters in this module have been all-gathered as necessary.
            for param in module.parameters(recurse=False):
                # Skip parameters without an associated buffer (such parameters have a
                # .requires_grad field equal to False).
                if param not in self.param_to_bucket_group:
                    continue
                assert param.requires_grad

                # If aligning param all-gather across pipeline stages, all-gather is dispatched
                # by start_param_sync calls in core/pipeline_parallelism/schedules.py.
                # If overlapping param all-gather with optimizer step, then all-gather has
                # already been dispatched in optimizer step.
                skip_next_bucket_dispatch = (
                    self.ddp_config.align_param_gather
                    or self.overlap_param_gather_with_optimizer_step
                )
                self.param_to_bucket_group[param].finish_param_sync(
                    skip_next_bucket_dispatch=skip_next_bucket_dispatch
                )

        return hook

    def _make_backward_post_hook(self, param: torch.nn.Parameter):
        """
        Creates a backward post-hook to dispatch an all-reduce / reduce-scatter when
        ready (i.e., when all grads in a bucket have been computed in all microbatches
        in a batch).
        """

        def hook(*unused):
            if is_graph_capturing():
                return

            if param in self.param_to_bucket_group:
                assert param.requires_grad
                if self.ddp_config.overlap_grad_reduce:
                    assert (
                        param.grad is not None
                    ), 'param.grad being None is not safe when overlap_grad_reduce is True'
                if param.grad is not None and (
                    not param.grad_added_to_main_grad or getattr(param, 'zero_out_wgrad', False)
                ):
                    param.main_grad.add_(param.grad.data)
                param.grad = None

                if self.ddp_config.overlap_grad_reduce:
                    self.param_to_bucket_group[param].register_grad_ready(param)

        return hook

    @contextmanager
    def no_sync(self):
        """
        Context manager that turns off gradient synchronization.
        """
        for bucket_group in self.bucket_groups + self.expert_parallel_bucket_groups:
            bucket_group.is_last_microbatch = False
        try:
            yield
        finally:
            for bucket_group in self.bucket_groups + self.expert_parallel_bucket_groups:
                bucket_group.is_last_microbatch = True

    def start_param_sync(self, *unused, force_sync: bool = False, force_dispatch: bool = False):
        """
        Initiates param sync (all-gather) communication operations for all model parameters.

        By default, when overlap_param_gather is set to True, dispatches asynchronous communication
        calls; when overlap_param_gather is set to False, calls synchronous communication
        ops. Can override this default behavior using flags below.

        Args:
            force_sync (bool, optional): force synchronous collective regardless of
                other settings.
            force_dispatch (bool, optional): force dispatch regardless of other settings.
        """
        if not force_sync:
            # If overlapping param AG with optimizer step, AG should not be dispatched again
            # in forward_backward_step.
            if self.overlap_param_gather_with_optimizer_step and not force_dispatch:
                return

        for bucket_group in self.bucket_groups + self.expert_parallel_bucket_groups:
            bucket_group.start_param_sync(force_sync=force_sync)

            if not self.ddp_config.overlap_param_gather:
                # For MXFP8 params, we need to copy the all-gathered param data from the buffer to
                # the param.data, since param buffer is not mapped to model params for MXFP8 case.
                # The paramaters are cast from bf16 to MXFP8 during copy.
                # In the case of "overlap_param_gather=True", the param copy is done
                # in "finish_param_sync" stage after zeroing the shared gardient buffers.
                if self.ddp_config.reuse_grad_buf_for_mxfp8_param_ag:
                    for bucket in bucket_group.buckets:
                        for param in bucket.params:
                            param_start, param_end = bucket.param_to_index[param]
                            param_slice = bucket.param_data.view(-1)[param_start:param_end]
                            param.data.copy_(param_slice.view(param.data.shape))
                        # All-gathered params are not needed after being copied to param.data.
                        # Zero out the param buffer (shared with grad buffer) for gradient
                        # accumulation. We cannot zero out the entire grad buffer because one grad
                        # buffer may correspond to multiple param buffers. If we zero out the entire
                        # grad buffer, it would clear the data of those param buffers that have not
                        # yet completed AG.
                        bucket.param_data.zero_()
                else:
                    fp8_params = []
                    for bucket in bucket_group.buckets:
                        for param in bucket.params:
                            if is_float8tensor(param):
                                fp8_params.append(param)
                    if len(fp8_params) > 0:
                        post_all_gather_processing(fp8_params)

    def start_grad_sync(self, *unused):
        """
        Initiates grad sync (all-reduce or reduce-scatter) communication operations
        for all model gradients.

        When overlap_grad_reduce is set to True, dispatches asynchronous communication
        calls. When overlap_grad_reduce is set to False, calls synchronous
        communication ops.
        """
        for bucket_group in self.bucket_groups + self.expert_parallel_bucket_groups:
            bucket_group.start_grad_sync()

    def finish_grad_sync(self):
        """
        Finishes grad sync (all-reduce or reduce-scatter) communication operations
        for all model gradients.

        When overlap_grad_reduce is set to True, waits for asynchronous communication
        calls to complete. When overlap_grad_reduce is set to False, calls synchronous
        communication ops.
        """
        for bucket_group in self.bucket_groups + self.expert_parallel_bucket_groups:
            bucket_group.finish_grad_sync()

    def scale_gradients(self, scaling_factor: float):
        """Scale all gradients inside the buffers by `scaling_factor`."""
        for buffer in self.buffers + self.expert_parallel_buffers:
            buffer.scale_gradients(scaling_factor)

    def zero_grad_buffer(self):
        """
        Zeros out all grad buffers. Needs to be called at the beginning of each
        training iteration.
        """
        if getattr(self.config, 'cuda_graph_impl', 'none') != 'transformer_engine':
            # Don't reset grad_added_to_main_grad when CUDA Graph is used.
            # Because in CUDA Graph it no longer has the opportunity to set it back
            # to True, and there will be a double-GA.
            for param in self.params_with_grad:
                param.grad_added_to_main_grad = False
        for buffer in self.buffers + self.expert_parallel_buffers:
            buffer.reset()
        for bucket_group in self.bucket_groups + self.expert_parallel_bucket_groups:
            bucket_group.reset()

    def broadcast_params(self):
        """
        Syncs parameters across all DP ranks.
        """
        for param in self.module.parameters():
            is_expert_parallel = not getattr(param, 'allreduce', True)

            if is_expert_parallel:
                data_parallel_group = self.expt_dp_group
            else:
                data_parallel_group = self.dp_cp_group
            torch.distributed.broadcast(
                param.data,
                src=torch.distributed.get_global_rank(data_parallel_group, 0),
                group=data_parallel_group,
            )

```

The fix is re-applied. The only lint warning is the pre-existing `torch` import issue.

## Explanation of the New Error

This was actually a **different manifestation** of the same problem, but in reverse:

| Original Error | New Error |
|----------------|-----------|
| Modules **removed** after `enable_forward_pre_hook` | Modules **added** after `enable_forward_pre_hook` |
| `CudaGraphManager` deleted by `toggle_cuda_graphs(set_to="none")` | `CudaGraphManager` added by `toggle_cuda_graphs(set_to="local")` |
| Leftover handles in dict → Assertion failed | Module not in dict → KeyError |

### The Flow That Caused This Error

```
┌────────────────────────────────────────────────────────────────────┐
│ enable_forward_pre_hook() called earlier                           │
│   └─ Registers hooks for modules (NO CudaGraphManager yet)         │
├────────────────────────────────────────────────────────────────────┤
│ generate() starts                                                   │
│   └─ toggle_cuda_graphs(set_to="local")                            │
│       └─ ADDS CudaGraphManager to module tree                      │
├────────────────────────────────────────────────────────────────────┤
│ ... generation happens ...                                          │
├────────────────────────────────────────────────────────────────────┤
│ generate() ending                                                   │
│   └─ disable_forward_pre_hook() [your Option 2 addition]           │
│       └─ for module in self.module.modules():                      │
│           └─ Includes CudaGraphManager (added in step 2)           │
│           └─ But CudaGraphManager NOT in handles dict!             │
│           └─ KeyError!                                             │
└────────────────────────────────────────────────────────────────────┘
```

### Why Option 1 Fixes Both Cases

By iterating over `list(self.remove_forward_pre_hook_handles.items())` instead of `self.module.modules()`:

- **Removed modules**: Still in the dict, so they get cleaned up ✓
- **Added modules**: Not in the dict, so we don't try to access them ✓

The fix is now in place. You can try running your application again.

```plaintext
┌────────────────────────────────────────────────────────────────────┐
│ enable_forward_pre_hook() called earlier                           │
│   └─ Registers hooks for modules (NO CudaGraphManager yet)         │
├────────────────────────────────────────────────────────────────────┤
│ generate() starts                                                   │
│   └─ toggle_cuda_graphs(set_to="local")                            │
│       └─ ADDS CudaGraphManager to module tree                      │
├────────────────────────────────────────────────────────────────────┤
│ ... generation happens ...                                          │
├────────────────────────────────────────────────────────────────────┤
│ generate() ending                                                   │
│   └─ disable_forward_pre_hook() [your Option 2 addition]           │
│       └─ for module in self.module.modules():                      │
│           └─ Includes CudaGraphManager (added in step 2)           │
│           └─ But CudaGraphManager NOT in handles dict!             │
│           └─ KeyError!                                             │
└────────────────────────────────────────────────────────────────────┘
```

---

**User**

Why does applying that fix cause a memory problem ?  It happens after the first step
```

========================= Step 2/500 =========================
▶ Preparing batch...
▶ Generating responses for batch of size 512...
(MegatronPolicyWorker[rank=0] pid=801433) GPU Memory before optimizer offload: 36.80GB allocated, 42.81GB reserved
(MegatronPolicyWorker[rank=0] pid=801433) /opt/nemo-rl/3rdparty/Megatron-LM-workspace/Megatron-LM/megatron/core/inference/engines/dynamic_engine.py:114: UserWarning: The `enable_cuda_graph` argument is deprecated and will be removed in `megatron-core 0.15`. `enable_cuda_graph` is now read directly from the transformer config object.
(MegatronPolicyWorker[rank=0] pid=801433)   warnings.warn(
cuda graph warmup - decode, d 1464:   0%|          | 0/16 [00:00<?, ?it/s]
(MegatronPolicyWorker[rank=0] pid=801433) GPU Memory after optimizer offload: 22.36GB allocated, 23.67GB reserved
(MegatronPolicyWorker[rank=0] pid=801433) GPU Memory after refit complete: 22.36GB allocated, 23.67GB reserved
cuda graph warmup - decode, d 1464:   6%|▋         | 1/16 [00:00<00:03,  3.76it/s]
cuda graph warmup - decode, d 1440:   6%|▋         | 1/16 [00:00<00:03,  3.76it/s]
cuda graph warmup - decode, d 1344:  12%|█▎        | 2/16 [00:00<00:03,  4.11it/s]
cuda graph warmup - decode, d 1248:  19%|█▉        | 3/16 [00:00<00:02,  4.72it/s]
cuda graph warmup - decode, d 1152:  25%|██▌       | 4/16 [00:00<00:02,  4.79it/s]
cuda graph warmup - decode, d 1056:  31%|███▏      | 5/16 [00:01<00:02,  4.73it/s]
cuda graph warmup - decode, d 960:  38%|███▊      | 6/16 [00:01<00:02,  4.62it/s]
cuda graph warmup - decode, d 864:  44%|████▍     | 7/16 [00:01<00:01,  4.79it/s]
cuda graph warmup - decode, d 768:  50%|█████     | 8/16 [00:01<00:01,  4.55it/s]
cuda graph warmup - decode, d 672:  56%|█████▋    | 9/16 [00:01<00:01,  4.58it/s]
cuda graph warmup - decode, d 576:  62%|██████▎   | 10/16 [00:02<00:01,  4.50it/s]
cuda graph warmup - decode, d 480:  69%|██████▉   | 11/16 [00:02<00:01,  4.61it/s]
cuda graph warmup - decode, d 384:  75%|███████▌  | 12/16 [00:02<00:00,  4.69it/s]
cuda graph warmup - decode, d 288:  81%|████████▏ | 13/16 [00:02<00:00,  4.61it/s]
cuda graph warmup - decode, d 192:  88%|████████▊ | 14/16 [00:03<00:00,  4.17it/s]
cuda graph warmup - decode, d 96:  94%|█████████▍| 15/16 [00:03<00:00,  4.01it/s]
cuda graph warmup - decode, d 96: 100%|██████████| 16/16 [00:03<00:00,  4.03it/s]
cuda graph warmup - non_decode, d 1464:   0%|          | 0/16 [00:00<?, ?it/s]
cuda graph warmup - non_decode, d 1440:   6%|▋         | 1/16 [00:00<00:03,  4.27it/s]
cuda graph warmup - non_decode, d 1344:  12%|█▎        | 2/16 [00:00<00:03,  3.60it/s]
cuda graph warmup - non_decode, d 1248:  19%|█▉        | 3/16 [00:00<00:03,  3.49it/s]
cuda graph warmup - non_decode, d 1152:  25%|██▌       | 4/16 [00:01<00:03,  3.84it/s]
cuda graph warmup - non_decode, d 1056:  31%|███▏      | 5/16 [00:01<00:02,  3.97it/s]
cuda graph warmup - non_decode, d 960:  38%|███▊      | 6/16 [00:01<00:02,  4.03it/s]
cuda graph warmup - non_decode, d 864:  44%|████▍     | 7/16 [00:01<00:02,  4.34it/s]
cuda graph warmup - non_decode, d 768:  50%|█████     | 8/16 [00:02<00:02,  3.98it/s]
cuda graph warmup - non_decode, d 672:  56%|█████▋    | 9/16 [00:02<00:01,  4.19it/s]
cuda graph warmup - non_decode, d 576:  62%|██████▎   | 10/16 [00:02<00:01,  4.36it/s]
cuda graph warmup - non_decode, d 480:  69%|██████▉   | 11/16 [00:02<00:01,  4.48it/s]
cuda graph warmup - non_decode, d 384:  75%|███████▌  | 12/16 [00:02<00:00,  4.69it/s]
cuda graph warmup - non_decode, d 288:  81%|████████▏ | 13/16 [00:03<00:00,  4.84it/s]
cuda graph warmup - non_decode, d 192:  88%|████████▊ | 14/16 [00:03<00:00,  4.77it/s]
cuda graph warmup - non_decode, d 96:  94%|█████████▍| 15/16 [00:03<00:00,  4.73it/s]
cuda graph warmup - non_decode, d 96: 100%|██████████| 16/16 [00:03<00:00,  4.34it/s]
(MegatronPolicyWorker[rank=0] pid=801433) SHAN : Time taken: 6.88466215133667 seconds
▶ Processing rewards...,
▶ Computing advantages...
▶ Preparing for logprob inference...
▶ Computing logprobs...
▶ Preparing for training...
▶ Training policy...
Traceback (most recent call last):
  File "/opt/nemo-rl/examples/run_grpo_math.py", line 260, in <module>
    main()
  File "/opt/nemo-rl/examples/run_grpo_math.py", line 243, in main
    grpo_train(
  File "/opt/nemo-rl/nemo_rl/algorithms/grpo.py", line 1352, in grpo_train
    train_results = policy.train(train_data, loss_fn)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/nemo-rl/nemo_rl/models/policy/lm_policy.py", line 532, in train
    results = self.worker_group.get_all_worker_results(futures)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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
ray.exceptions.RayTaskError(OutOfMemoryError): ray::MegatronPolicyWorker.train() (pid=801433, ip=10.65.29.29, actor_id=39ed80c77981b95c61c30e3001000000, repr=MegatronPolicyWorker[rank=0])
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/nemo-rl/nemo_rl/utils/nsys.py", line 88, in wrapper
    ret = func(*args, **kwargs)
          ^^^^^^^^^^^^^^^^^^^^^
  File "/opt/nemo-rl/nemo_rl/models/policy/workers/megatron_policy_worker.py", line 1101, in train
    losses_reduced = forward_backward_func(
                     ^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/nemo-rl/3rdparty/Megatron-LM-workspace/Megatron-LM/megatron/core/pipeline_parallel/schedules.py", line 629, in forward_backward_no_pipelining
    backward_step(
  File "/opt/nemo-rl/3rdparty/Megatron-LM-workspace/Megatron-LM/megatron/core/pipeline_parallel/schedules.py", line 480, in backward_step
    custom_backward(output_tensor[0], output_tensor_grad[0])
  File "/opt/nemo-rl/3rdparty/Megatron-LM-workspace/Megatron-LM/megatron/core/pipeline_parallel/schedules.py", line 173, in custom_backward
    Variable._execution_engine.run_backward(
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.16 GiB. GPU 0 has a total capacity of 79.11 GiB of which 780.88 MiB is free. Including non-PyTorch memory, this process has 78.32 GiB memory in use. Of the allocated memory 71.15 GiB is allocated by PyTorch, with 92.22 MiB allocated in private pools (e.g., CUDA Graphs), and 3.09 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
```

---

**Cursor**

I see the problem! The Option 2 fix is causing memory issues because:

1. `disable_forward_pre_hook()` is being called, which removes the hooks
2. **But the hooks are never re-enabled!**

In the original `use_reference_model`, there's symmetry - it disables then re-enables:
```python
if self.should_disable_forward_pre_hook:
    self.disable_forward_pre_hook()  # Disable
# ... work ...
finally:
    if self.should_disable_forward_pre_hook:
        self.enable_forward_pre_hook()  # Re-enable!
```

But in my Option 2 fix, I only added `disable_forward_pre_hook()` without re-enabling. This breaks the distributed optimizer's param gathering overlap, causing memory issues.

**Since Option 1 already handles the robustness**, let me **revert Option 2** - it's not needed and is causing problems:

```python
if self.should_disable_forward_pre_hook:
    self.disable_forward_pre_hook()  # Disable
# ... work ...
finally:
    if self.should_disable_forward_pre_hook:
        self.enable_forward_pre_hook()  # Re-enable!
```

```python
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
import gc
import os
import re
import time
import warnings
from collections import defaultdict
from contextlib import AbstractContextManager, contextmanager, nullcontext
from functools import partial
from typing import Any, Iterator, Optional, TypeVar, cast

import ray
import torch
from megatron.bridge import AutoBridge
from megatron.bridge.models.model_provider import get_model
from megatron.bridge.training import fault_tolerance
from megatron.bridge.training.checkpointing import (
    checkpoint_exists,
    init_checkpointing_context,
    load_checkpoint,
    maybe_finalize_async_save,
    save_checkpoint,
)
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    LoggerConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainingConfig,
)
from megatron.bridge.training.initialize import (
    initialize_megatron,
    set_jit_fusion_options,
)
from megatron.bridge.training.optim import setup_optimizer
from megatron.bridge.training.setup import (
    _update_model_config_funcs,
)
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer
from megatron.bridge.training.utils.train_utils import (
    logical_and_across_model_parallel_group,
    reduce_max_stat_across_model_parallel_group,
)
from megatron.bridge.utils.common_utils import get_rank_safe
from megatron.bridge.utils.instantiate_utils import InstantiationMode
from megatron.bridge.utils.vocab_utils import calculate_padded_vocab_size
from megatron.core import parallel_state
from megatron.core.transformer.utils import toggle_cuda_graphs
from megatron.core.distributed import DistributedDataParallel
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import (
    FullyShardedDataParallel as custom_FSDP,
)
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import (
    InferenceWrapperConfig,
)
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.models.gpt import GPTModel
from megatron.core.optimizer import ChainedOptimizer
from megatron.core.parallel_state import (
    get_context_parallel_group,
    get_context_parallel_rank,
    get_pipeline_model_parallel_group,
    get_pipeline_model_parallel_last_rank,
    get_pipeline_model_parallel_world_size,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    is_pipeline_last_stage,
)
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.utils import get_ltor_masks_and_position_ids
from ray.util.queue import Queue
from transformers import PreTrainedTokenizerBase

from nemo_rl.algorithms.interfaces import LossFunction, LossType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.model_utils import (
    allgather_cp_sharded_tensor,
    distributed_vocab_topk,
    from_parallel_logits_to_logprobs,
    from_parallel_logits_to_logprobs_packed_sequences,
)
from nemo_rl.distributed.named_sharding import NamedSharding
from nemo_rl.models.generation.fp8 import (
    convert_calibration_to_vllm_format,
    get_vllm_qkv_scale_names,
)
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationOutputSpec,
    verify_right_padding,
)
from nemo_rl.models.generation.vllm.config import VllmConfig
from nemo_rl.models.megatron.common import (
    _get_pack_sequence_parameters_for_megatron,
    _pack_sequences_for_megatron,
    broadcast_tensor,
    forward_step_arbitrary_loss,
    get_moe_metrics,
)
from nemo_rl.models.megatron.community_import import import_model_from_hf_name
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.interfaces import (
    ColocatablePolicyInterface,
    LogprobOutputSpec,
)
from nemo_rl.models.policy.utils import (
    configure_dynamo_cache,
    get_megatron_checkpoint_dir,
    get_runtime_env_for_policy_worker,
)
from nemo_rl.models.policy.workers.base_policy_worker import AbstractPolicyWorker
from nemo_rl.models.policy.workers.patches import apply_transformer_engine_patch
from nemo_rl.utils.nsys import wrap_with_nvtx_name
from nemo_rl.utils.packed_tensor import packed_broadcast_producer

try:
    from megatron.core.distributed import (
        TorchFullyShardedDataParallel as torch_FSDP,  # noqa: F401 unused-import
    )

    HAVE_FSDP2 = True
except ImportError:
    HAVE_FSDP2 = False

TokenizerType = TypeVar("TokenizerType", bound=PreTrainedTokenizerBase)


def broadcast_object_across_pp_ranks(obj):
    """Broadcast an object across pipeline parallel ranks.

    This utility function handles broadcasting an object from the rank that owns it
    to all other pipeline parallel ranks. If only one rank has the object (non-None),
    it will be broadcast to all other ranks.

    Args:
        obj: The object to broadcast. Can be None on ranks that don't own it.

    Returns:
        The object on all ranks (either the original or the broadcast copy).

    Raises:
        ValueError: If the object doesn't exist on any pipeline parallel rank.
    """
    pp_size = get_pipeline_model_parallel_world_size()
    pp_group = get_pipeline_model_parallel_group()

    if pp_size == 1:
        return obj

    # ------------------------------------------------------------------
    # 1. Gather presence flags from all PP ranks to find the source rank
    # ------------------------------------------------------------------
    has_obj = obj is not None
    obj_flags = [None] * pp_size
    torch.distributed.all_gather_object(obj_flags, has_obj, group=pp_group)

    # ------------------------------------------------------------------
    # 2. Identify the owning rank (the only rank with True flag)
    # ------------------------------------------------------------------
    src_rank = None  # Rank *inside* the PP group
    for rank, flag in enumerate(obj_flags):
        if flag:
            src_rank = rank
            break

    if src_rank is None:
        raise ValueError("Object must exist on at least one PP rank")

    # ------------------------------------------------------------------
    # 3. Broadcast the object from the source rank to all ranks
    # ------------------------------------------------------------------
    # Use broadcast_object_list which is more robust than all_gather_object
    obj_list = [obj]
    pp_ranks = torch.distributed.get_process_group_ranks(pp_group)
    global_src = pp_ranks[src_rank]
    torch.distributed.broadcast_object_list(obj_list, src=global_src, group=pp_group)

    return obj_list[0]


def setup_megatron_model(
    policy_cfg: PolicyConfig,
    cfg: ConfigContainer,
    load_optimizer: bool = True,
    get_embedding_ranks=None,  # TODO @sahilj: What is this?
    get_position_embedding_ranks=None,
):
    state = GlobalState()
    state.cfg = cfg
    # TODO: Freeze state.cfg

    cfg.dist.external_gpu_device_mapping = True
    initialize_megatron(
        cfg=cfg,
        get_embedding_ranks=get_embedding_ranks,
        get_position_embedding_ranks=get_position_embedding_ranks,
    )

    if cfg.ft and cfg.ft.enable_ft_package:
        fault_tolerance.setup(cfg, state)
        fault_tolerance.maybe_setup_simulated_fault(cfg.ft)

    # Set pytorch JIT layer fusion options and warmup JIT functions.
    set_jit_fusion_options(cfg.model, cfg.train.micro_batch_size)

    # Adjust the startup time so it reflects the largest value.
    # This will be closer to what scheduler will see (outside of
    # image ... launches.
    start_time_tensor = torch.tensor(
        [state.start_time], dtype=torch.double, device="cuda"
    )
    torch.distributed.all_reduce(start_time_tensor, op=torch.distributed.ReduceOp.MIN)
    state.start_time = start_time_tensor.item()

    print(
        "time to initialize megatron (seconds): {:.3f}".format(
            time.time() - state.start_time
        )
    )
    torch.distributed.barrier()

    # Context used for persisting some state between checkpoint saves.
    checkpointing_context = init_checkpointing_context(cfg.checkpoint)

    # Tokenizer
    build_tokenizer(
        cfg.tokenizer,
        make_vocab_size_divisible_by=cfg.model.make_vocab_size_divisible_by
        // cfg.model.tensor_model_parallel_size,
        tensor_model_parallel_size=cfg.model.tensor_model_parallel_size,
        trust_remote_code=True,
    )
    assert cfg.model.vocab_size, "vocab size must be specified in model config"

    torch.distributed.barrier()

    pre_wrap_hook = []
    mixed_precision_wrapper = Float16Module
    if policy_cfg["megatron_cfg"]["freeze_moe_router"]:

        def freeze_moe_router(megatron_model):
            if not isinstance(megatron_model, list):
                megatron_model = [megatron_model]
            for model_module in megatron_model:
                # Handle both wrapped (Float16Module) and unwrapped models
                if isinstance(model_module, Float16Module):
                    model_module = model_module.module
                # Handle VLM models
                if hasattr(model_module, "language_model"):
                    model_module = model_module.language_model
                for layer in model_module.decoder.layers:
                    if hasattr(layer, "mlp") and hasattr(layer.mlp, "router"):
                        layer.mlp.router.weight.requires_grad = False

        mixed_precision_wrapper = CustomFloat16Module
        pre_wrap_hook.extend([freeze_moe_router])

    # Model, optimizer, and learning rate.
    model = get_model(
        cfg.model,
        cfg.ddp,
        use_torch_fsdp2=cfg.dist.use_torch_fsdp2,
        overlap_param_gather_with_optimizer_step=cfg.optimizer.overlap_param_gather_with_optimizer_step,
        data_parallel_random_init=cfg.rng.data_parallel_random_init,
        pre_wrap_hook=pre_wrap_hook,
        mixed_precision_wrapper=mixed_precision_wrapper,
    )
    if load_optimizer:
        optimizer, scheduler = setup_optimizer(
            optimizer_config=cfg.optimizer,
            scheduler_config=cfg.scheduler,
            model=model,
            use_gloo_process_groups=cfg.dist.use_gloo_process_groups,
        )
    else:
        optimizer = None
        scheduler = None

    print("Model, optimizer, and learning rate scheduler built")
    torch.distributed.barrier()

    # Load checkpoint if applicable
    if (
        cfg.checkpoint.load is not None
        or cfg.checkpoint.pretrained_checkpoint is not None
    ) and (
        checkpoint_exists(cfg.checkpoint.load)
        or checkpoint_exists(cfg.checkpoint.pretrained_checkpoint)
    ):
        load_checkpoint(
            state,
            model,
            optimizer,
            scheduler,
            checkpointing_context=checkpointing_context,
            skip_load_to_model_and_opt=HAVE_FSDP2 and cfg.dist.use_torch_fsdp2,
        )
        print("Checkpoint loaded")
    torch.distributed.barrier()

    return state, model, optimizer, scheduler, checkpointing_context


def destroy_parallel_state():
    """Safely destroy parallel state and reset async call tracking.

    This function is called during initialization to clean up temporary distributed
    state from model import operations. Resetting async call tracking ensures that
    when the main Megatron distributed context is created, all ranks start with
    consistent call_idx values for async checkpointing.
    """
    if torch.distributed.is_initialized():
        try:
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()
        except:
            pass  # Ignore errors if already destroyed
    if hasattr(parallel_state, "destroy_model_parallel"):
        try:
            parallel_state.destroy_model_parallel()
        except:
            pass  # Ignore errors if already destroyed

    # Reset async calls queue to prevent call_idx mismatches after distributed context recreation
    try:
        import nemo.tron.utils.async_utils as nemo_async_utils
        from megatron.core.dist_checkpointing.strategies.async_utils import (
            AsyncCallsQueue,
        )

        # Clean up any existing async callers first
        old_call_idx = getattr(nemo_async_utils._async_calls_queue, "call_idx", None)
        num_unfinalized = (
            nemo_async_utils._async_calls_queue.get_num_unfinalized_calls()
        )
        if num_unfinalized > 0:
            print(
                f"[WARNING] Resetting async calls queue with {num_unfinalized} unfinalized calls"
            )
        try:
            nemo_async_utils._async_calls_queue.close()
        except:
            pass  # Ignore errors during cleanup
        # Reset the global async calls queue by creating a new instance
        nemo_async_utils._async_calls_queue = AsyncCallsQueue()
        print(f"[DEBUG] Reset NeMo async calls queue (old call_idx: {old_call_idx})")
    except ImportError:
        pass

    # Also reset the Megatron async calls queue if it exists
    try:
        import megatron.training.async_utils as megatron_async_utils
        from megatron.core.dist_checkpointing.strategies.async_utils import (
            AsyncCallsQueue,
        )

        # Clean up any existing async callers first
        old_call_idx = getattr(
            megatron_async_utils._async_calls_queue, "call_idx", None
        )
        num_unfinalized = (
            megatron_async_utils._async_calls_queue.get_num_unfinalized_calls()
        )
        if num_unfinalized > 0:
            print(
                f"[WARNING] Resetting Megatron async calls queue with {num_unfinalized} unfinalized calls"
            )
        try:
            megatron_async_utils._async_calls_queue.close()
        except:
            pass  # Ignore errors during cleanup
        # Reset the Megatron global async calls queue as well
        megatron_async_utils._async_calls_queue = AsyncCallsQueue()
        print(
            f"[DEBUG] Reset Megatron async calls queue (old call_idx: {old_call_idx})"
        )
    except ImportError:
        pass

    # Reset the third global async_calls instance in base strategy module
    try:
        import megatron.core.dist_checkpointing.strategies.base as base_strategy
        from megatron.core.dist_checkpointing.strategies.async_utils import (
            AsyncCallsQueue,
        )

        # Clean up and reset the global async_calls in base strategy
        old_call_idx = getattr(base_strategy.async_calls, "call_idx", None)
        num_unfinalized = base_strategy.async_calls.get_num_unfinalized_calls()
        if num_unfinalized > 0:
            print(
                f"[WARNING] Resetting base strategy async_calls with {num_unfinalized} unfinalized calls"
            )
        try:
            base_strategy.async_calls.close()
        except:
            pass
        base_strategy.async_calls = AsyncCallsQueue()
        print(f"[DEBUG] Reset base strategy async_calls (old call_idx: {old_call_idx})")
    except ImportError:
        pass


@ray.remote(
    runtime_env=get_runtime_env_for_policy_worker("megatron_policy_worker")
)  # pragma: no cover
class MegatronPolicyWorker(AbstractPolicyWorker, ColocatablePolicyInterface):
    def __repr__(self):
        """Customizes the actor's prefix in the Ray logs.

        This makes it easier to identify which worker is producing specific log messages.
        """
        if torch.distributed.is_initialized():
            return f"{self.__class__.__qualname__}[rank={torch.distributed.get_rank()}]"
        else:
            return f"{self.__class__.__qualname__}"

    def __init__(
        self,
        config: PolicyConfig,
        tokenizer: TokenizerType,
        weights_path: Optional[str] = None,
        optimizer_path: Optional[str] = None,
        init_optimizer: bool = True,
        init_reference_model: bool = True,
        *,
        worker_sharding_annotations: NamedSharding,
        pre_init_communication_queue: Queue,
        **kwargs: Any,
    ):
        apply_transformer_engine_patch()

        self.is_generation_colocated = None
        if "generation" in config and config["generation"] is not None:
            self.is_generation_colocated = config["generation"]["colocated"]["enabled"]

        # Explicitly set NCCL_CUMEM_ENABLE to 1 to avoid the P2P initialization error for PyNCCLCommunicator.
        # See https://github.com/NVIDIA-NeMo/RL/issues/564 for more details.
        if not self.is_generation_colocated:
            os.environ["NCCL_CUMEM_ENABLE"] = "1"

        self.cfg = config
        dtype_map = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }
        self.dtype = dtype_map[self.cfg["precision"]]

        self.optimizer_cpu_offload = self.cfg["megatron_cfg"]["optimizer"][
            "optimizer_cpu_offload"
        ]
        self.offload_optimizer_for_logprob = self.cfg["offload_optimizer_for_logprob"]

        # Reward models are not yet supported with Megatron.
        if "reward_model_cfg" in self.cfg and self.cfg["reward_model_cfg"]["enabled"]:
            raise NotImplementedError(
                "Reward models are not yet supported with the Megatron backend, this issue is "
                "tracked in https://github.com/NVIDIA-NeMo/RL/issues/720"
            )

        # Disable dynamo autotune_local_cache to avoid crash when there's already a cache
        # with different order of node_bundles
        configure_dynamo_cache()

        # cfg["model_name"] is allowed to be either an HF model name or a path to an HF checkpoint
        # check if hf_model_name is a path
        hf_model_name = self.cfg["model_name"]
        # Check if the checkpoint already exists
        hf_model_subdir = hf_model_name
        if os.path.exists(hf_model_name):
            hf_model_subdir = f"model_{hf_model_subdir.replace('/', '_')}"

        pretrained_path = f"{get_megatron_checkpoint_dir()}/{hf_model_subdir}"
        pt_checkpoint_exists = os.path.exists(pretrained_path) and os.path.exists(
            os.path.join(pretrained_path, "iter_0000000")
        )

        # Ensure clean slate before import
        destroy_parallel_state()

        # Set for rank for non-collocated to check which ranks to broadcast from
        self.rank = get_rank_safe()
        # Need to initialize the process group before calling into Megatron-Bridge, otherwise Megatron-Bridge will try to set an incorrect device
        torch.distributed.init_process_group("nccl")
        if pt_checkpoint_exists:
            print(f"Checkpoint already exists at {pretrained_path}. Skipping import.")
        else:
            hf_config_overrides = self.cfg.get("hf_config_overrides", {}) or {}
            import_model_from_hf_name(
                hf_model_name,
                pretrained_path,
                self.cfg["megatron_cfg"],
                **hf_config_overrides,
            )

            if parallel_state.model_parallel_is_initialized():
                print("Reinitializing model parallel after loading model state.")
                parallel_state.destroy_model_parallel()

        pretrained_run_config = os.path.join(
            pretrained_path, "iter_0000000/run_config.yaml"
        )

        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if not os.path.exists(pretrained_run_config):
            raise FileNotFoundError(
                f"Pretrained run config not found at {pretrained_run_config} on rank={get_rank_safe()}. This usually means that the one-time HF->mcore conversion on rank=0 saved to a directory not being mounted on this node. Please check "
            )

        try:
            cfg_from_pretrained = ConfigContainer.from_yaml(
                pretrained_run_config, mode=InstantiationMode.STRICT
            )
        except Exception as e:
            # Add helpful context as a note to the exception
            e.add_note(
                f"\n{'=' * 80}\n"
                f"NOTE: A common cause of this error is when the HF->mcore converted checkpoint is\n"
                f"created with an older version of megatron-bridge.\n"
                f"If this checkpoint is old or was generated by a different code version,\n"
                f"try deleting it and rerunning the code.\n"
                f"The checkpoint will be automatically regenerated with the current version.\n\n"
                f"Checkpoint location: {pretrained_path}\n"
                f"{'=' * 80}"
            )
            raise
        model_cfg = cfg_from_pretrained.model

        cfg_from_pretrained.logger = LoggerConfig()

        model_cfg.tensor_model_parallel_size = self.cfg["megatron_cfg"][
            "tensor_model_parallel_size"
        ]
        model_cfg.pipeline_model_parallel_size = self.cfg["megatron_cfg"][
            "pipeline_model_parallel_size"
        ]
        model_cfg.num_layers_in_first_pipeline_stage = self.cfg["megatron_cfg"][
            "num_layers_in_first_pipeline_stage"
        ]
        model_cfg.num_layers_in_last_pipeline_stage = self.cfg["megatron_cfg"][
            "num_layers_in_last_pipeline_stage"
        ]
        model_cfg.sequence_parallel = self.cfg["megatron_cfg"]["sequence_parallel"]
        model_cfg.context_parallel_size = self.cfg["megatron_cfg"][
            "context_parallel_size"
        ]
        if model_cfg.context_parallel_size > 1:
            assert self.cfg["sequence_packing"]["enabled"], (
                "Sequence Packing must be enabled to use Context Parallelism with MCore"
            )
        model_cfg.expert_tensor_parallel_size = self.cfg["megatron_cfg"][
            "expert_tensor_parallel_size"
        ]
        model_cfg.expert_model_parallel_size = self.cfg["megatron_cfg"][
            "expert_model_parallel_size"
        ]

        # Setting moe_router_dtype to higher precision (e.g. fp64) can improve numerical stability,
        # especially when using many experts.
        model_cfg.moe_router_dtype = self.cfg["megatron_cfg"]["moe_router_dtype"]

        # The below two configs (and "freeze_moe_router") are used to stabilize moe training
        # by preventing updates to the moe router. We found that this is helpful in reducing
        # logprob error during training.

        # Set this to "none" to disable load balancing loss.
        model_cfg.moe_router_load_balancing_type = self.cfg["megatron_cfg"][
            "moe_router_load_balancing_type"
        ]
        # Set this to 0.0 to disable updates to the moe router expert bias
        model_cfg.moe_router_bias_update_rate = self.cfg["megatron_cfg"][
            "moe_router_bias_update_rate"
        ]

        model_cfg.moe_permute_fusion = self.cfg["megatron_cfg"]["moe_permute_fusion"]
        if "layernorm_epsilon" in self.cfg["megatron_cfg"]:
            model_cfg.layernorm_epsilon = self.cfg["megatron_cfg"]["layernorm_epsilon"]

        model_cfg.sequence_parallel = self.cfg["megatron_cfg"]["sequence_parallel"]
        model_cfg.bf16 = self.dtype == torch.bfloat16
        model_cfg.fp16 = self.dtype == torch.float16
        if model_cfg.fp16:
            assert not model_cfg.bf16, "fp16 and bf16 cannot be used together"
            model_cfg.params_dtype = torch.float16
        elif model_cfg.bf16:
            assert not model_cfg.fp16, "fp16 and bf16 cannot be used together"
            model_cfg.params_dtype = torch.bfloat16
        else:
            model_cfg.params_dtype = torch.float32
        model_cfg.pipeline_dtype = dtype_map[self.cfg["megatron_cfg"]["pipeline_dtype"]]
        model_cfg.parallel_output = True
        if self.cfg["megatron_cfg"]["activation_checkpointing"]:
            model_cfg.recompute_granularity = "full"
            model_cfg.recompute_method = "uniform"
            model_cfg.recompute_num_layers = 1
        if not model_cfg.gated_linear_unit:
            assert model_cfg.activation_func is not None, (
                "activation_func must be set if not using gated_linear_unit. This likely "
                "indicates an issue in configuration conversion (e.g. activation func was "
                "a lambda and couldn't be serialized). This is based on this check "
                "https://github.com/NVIDIA/Megatron-LM/blob/1ab876ddc4c1893c76f26d775226a8d1dcdfb3d2/megatron/core/transformer/mlp.py#L174."
            )
        model_cfg.apply_rope_fusion = self.cfg["megatron_cfg"]["apply_rope_fusion"]
        model_cfg.bias_activation_fusion = self.cfg["megatron_cfg"][
            "bias_activation_fusion"
        ]
        fp8_cfg = self.cfg["megatron_cfg"].get("fp8_cfg", None)
        self.fp8_cfg = fp8_cfg
        if fp8_cfg is not None and fp8_cfg.get("enabled", False):
            try:
                model_cfg.fp8 = fp8_cfg["fp8"]
                model_cfg.fp8_recipe = fp8_cfg["fp8_recipe"]
                model_cfg.fp8_param = fp8_cfg["fp8_param"]
            except KeyError as e:
                raise KeyError(f"Missing key in fp8_cfg: {e}")
            if model_cfg.fp8_param:
                warnings.warn(
                    "Setting fp8_param=True sometimes causes NaN token_mult_prob_error, please use with caution. "
                    "Refer to https://github.com/NVIDIA-NeMo/RL/issues/1164 for latest updates with this issue."
                )

        optimizer_cpu_offload = self.cfg["megatron_cfg"]["optimizer"][
            "optimizer_cpu_offload"
        ]
        optimizer_offload_fraction = self.cfg["megatron_cfg"]["optimizer"][
            "optimizer_offload_fraction"
        ]
        if optimizer_cpu_offload:
            # Currently, hybrid optimizer (partly on GPU and partly on CPU) is not supported because it conflicts with the way
            # Nemo-rl handles the optimizer offload/onload between generation and training. So if using CPU optimizer the offload_fraction should be 1.0.
            assert optimizer_offload_fraction == 1.0, (
                "Currently for optimizer offloading, only optimizer_offload_fraction=1.0 is supported"
            )
        if (
            "logprob_chunk_size" in self.cfg
            and self.cfg["logprob_chunk_size"] is not None
            and self.cfg["logprob_chunk_size"] > 0
        ):
            assert self.cfg["megatron_cfg"]["defer_fp32_logits"], (
                "defer_fp32_logits must be True if logprob_chunk_size is set"
            )
        self.defer_fp32_logits = self.cfg["megatron_cfg"].get(
            "defer_fp32_logits", None
        ) and (model_cfg.fp16 or model_cfg.bf16)

        checkpoint_config = CheckpointConfig(
            save_interval=100,
            save=weights_path,
            load=weights_path,
            pretrained_checkpoint=pretrained_path,  # This is the path to the pretrained ckpt for the SFT case
            async_save=False,  # This doesn't work right now.
            fully_parallel_save=True,
            fully_parallel_load=True,  # Enable fully parallel load
            load_rng=False,
        )
        ref_checkpoint_config = CheckpointConfig(
            pretrained_checkpoint=pretrained_path,  # This is the path to the pretrained ckpt for the SFT case
            save=None,
            load=None,
            fully_parallel_load=True,  # Enable fully parallel load
            load_rng=False,
        )

        assert "train_iters" in self.cfg["megatron_cfg"], (
            "train_iters must be set in megatron_cfg. For an example, see "
            "https://github.com/NVIDIA-NeMo/RL/blob/bccbc377705a81a1f4b3c31ad9767bcc15f735a8/nemo_rl/algorithms/sft.py#L175-L179."
        )

        ## These settings are required for correct gradient computations in mcore
        ## when calculate_per_token_loss is True, there is no scaling of the gradient in mcore,
        ## so we handle the scaling in nemo-rl.
        ## perform_initialization = True is a workaround to ensure the correct tensor parallel attributes are set
        ## on the TP-sharded parameters.
        model_cfg.calculate_per_token_loss = True
        model_cfg.perform_initialization = True
        # SHAN CHECK:
        model_cfg.cuda_graph_impl = "local"
        model_cfg.cuda_graph_scope = None
        model_cfg.use_te_rng_tracker = True

        assert (
            "aux_loss" not in model_cfg.moe_router_load_balancing_type
            or model_cfg.moe_aux_loss_coeff == 0
        ), (
            "MoE aux loss is currently not supported due to a known bug in Megatron-LM. "
            "See https://github.com/NVIDIA/Megatron-LM/issues/1984 for more details."
        )

        self.megatron_cfg = ConfigContainer(
            model=model_cfg,
            checkpoint=checkpoint_config,
            logger=LoggerConfig(logging_level=0),
            train=TrainingConfig(
                micro_batch_size=1,  # ignored
                global_batch_size=self.cfg["train_global_batch_size"],  # ignored
                train_iters=self.cfg["megatron_cfg"][
                    "train_iters"
                ],  # Set by algorithm setup
            ),
            optimizer=OptimizerConfig(
                **self.cfg["megatron_cfg"]["optimizer"],
            ),
            ddp=DistributedDataParallelConfig(
                check_for_nan_in_grad=True,
                grad_reduce_in_fp32=self.cfg["megatron_cfg"][
                    "distributed_data_parallel_config"
                ]["grad_reduce_in_fp32"],
                overlap_grad_reduce=self.cfg["megatron_cfg"][
                    "distributed_data_parallel_config"
                ]["overlap_grad_reduce"],
                overlap_param_gather=self.cfg["megatron_cfg"][
                    "distributed_data_parallel_config"
                ]["overlap_param_gather"],
                # we need to set average_in_collective=False with calculate_per_token_loss=True.
                # otherwise, mcore throws an assertion error.
                average_in_collective=False,
                use_distributed_optimizer=self.cfg["megatron_cfg"]["optimizer"][
                    "use_distributed_optimizer"
                ],
                data_parallel_sharding_strategy=self.cfg["megatron_cfg"][
                    "distributed_data_parallel_config"
                ]["data_parallel_sharding_strategy"],
            ),
            scheduler=SchedulerConfig(
                **self.cfg["megatron_cfg"]["scheduler"],
            ),
            dataset=None,
            tokenizer=TokenizerConfig(
                tokenizer_type="HuggingFaceTokenizer",
                tokenizer_model=hf_model_name,
            ),
        )
        # TODO: this validation should happen inside mbridge: https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/1665
        if self.dtype == torch.bfloat16:
            assert self.megatron_cfg.model.bf16 == True, (
                "policy.megatron_cfg.model.bf16=True must be set if policy.precision=bfloat16. This is handled by nemo-rl so this indicates something is misconfigured."
            )
            assert (
                self.megatron_cfg.optimizer.use_precision_aware_optimizer == False
                or self.megatron_cfg.optimizer.bf16 == True
            ), (
                "policy.megatron_cfg.optimizer.bf16=True must be set if policy.precision=bfloat16 when using use_precision_aware_optimizer=True"
            )
        elif self.dtype == torch.float16:
            assert self.megatron_cfg.model.fp16 == True, (
                "policy.megatron_cfg.model.fp16=True must be set if policy.precision=float16. This is handled by nemo-rl so this indicates something is misconfigured."
            )
            assert (
                self.megatron_cfg.optimizer.use_precision_aware_optimizer == False
                or self.megatron_cfg.optimizer.fp16 == True
            ), (
                "policy.megatron_cfg.optimizer.fp16=True must be set if policy.precision=float16 when using use_precision_aware_optimizer=True"
            )
        elif self.dtype == torch.float32:
            assert (
                self.megatron_cfg.model.bf16 == False
                and self.megatron_cfg.model.fp16 == False
            ), (
                "policy.megatron_cfg.model.bf16=False and policy.megatron_cfg.model.fp16=False must be set if policy.precision=float32. This is handled by nemo-rl so this indicates something is misconfigured."
            )
            assert (
                self.megatron_cfg.optimizer.bf16 == False
                and self.megatron_cfg.optimizer.fp16 == False
            ), (
                "policy.megatron_cfg.optimizer.bf16=False and policy.megatron_cfg.optimizer.fp16=False must be set if policy.precision=float32"
            )
        self.megatron_cfg.validate()
        (
            self.mcore_state,
            self.model,
            self.optimizer,
            self.scheduler,
            self.checkpointing_context,
        ) = setup_megatron_model(
            policy_cfg=self.cfg, cfg=self.megatron_cfg, load_optimizer=init_optimizer
        )

        # Set the param sync function for the model
        if (
            self.megatron_cfg.ddp.overlap_param_gather
            and self.megatron_cfg.ddp.align_param_gather
        ):
            self.megatron_cfg.param_sync_func = [
                model_chunk.start_param_sync for model_chunk in self.model
            ]
            if len(self.model) == 1:
                self.megatron_cfg.param_sync_func = self.megatron_cfg.param_sync_func[0]

        self.model = self.model[0]  # Get the first model from the list

        if init_reference_model:
            self.model = self.move_model(self.model, "cpu")
            ref_ckpt_context = init_checkpointing_context(ref_checkpoint_config)

            # Create a separate megatron config for the reference model with the correct checkpoint config
            self.megatron_cfg.model.cuda_graph_impl = "none"
            self.megatron_cfg.model.use_te_rng_tracker = True
            ref_megatron_cfg = ConfigContainer(
                model=self.megatron_cfg.model,
                checkpoint=ref_checkpoint_config,  # Use the reference checkpoint config
                logger=self.megatron_cfg.logger,
                train=self.megatron_cfg.train,
                optimizer=self.megatron_cfg.optimizer,
                ddp=self.megatron_cfg.ddp,
                scheduler=self.megatron_cfg.scheduler,
                dataset=self.megatron_cfg.dataset,
                tokenizer=self.megatron_cfg.tokenizer,
            )

            # Create a separate state object for the reference model
            ref_state = GlobalState()
            ref_state.cfg = ref_megatron_cfg

            # Configure mixed precision wrapper for reference model
            ref_mixed_precision_wrapper = Float16Module
            if self.cfg["megatron_cfg"].get("freeze_moe_router", False):
                ref_mixed_precision_wrapper = CustomFloat16Module

            reference_model = get_model(
                self.megatron_cfg.model,
                self.megatron_cfg.ddp,
                use_torch_fsdp2=self.megatron_cfg.dist.use_torch_fsdp2,
                overlap_param_gather_with_optimizer_step=self.megatron_cfg.optimizer.overlap_param_gather_with_optimizer_step,
                pre_wrap_hook=self.megatron_cfg.rng.data_parallel_random_init,
                mixed_precision_wrapper=ref_mixed_precision_wrapper,
            )
            print("Loading the Reference Model")
            if (
                ref_checkpoint_config.pretrained_checkpoint is not None
                and checkpoint_exists(ref_checkpoint_config.pretrained_checkpoint)
            ):
                load_checkpoint(
                    ref_state,  # Use the separate state object with ref checkpoint config
                    reference_model,
                    None,  # no optimizer
                    None,  # no scheduler
                    checkpointing_context=ref_ckpt_context,
                    skip_load_to_model_and_opt=HAVE_FSDP2
                    and self.megatron_cfg.dist.use_torch_fsdp2,
                )
                reference_model = reference_model[0]
                reference_model.eval()
                self.reference_state_dict = {}
                for name, item in reference_model.state_dict().items():
                    if isinstance(item, torch.Tensor):
                        cpu_item = item.detach().to(
                            device="cpu", non_blocking=True, copy=True
                        )
                        del item
                    else:
                        cpu_item = item
                    self.reference_state_dict[name] = cpu_item
                print("Reference model loaded")
            else:
                print("Reference model not loaded")

            self.model = self.move_model(self.model, "cuda")

        _update_model_config_funcs(
            [self.model],
            self.megatron_cfg.model,
            self.megatron_cfg.ddp,
            self.optimizer,
            align_grad_reduce=self.megatron_cfg.dist.align_grad_reduce,
        )

        tokenizer_config = TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model=hf_model_name,
        )

        self.megatron_tokenizer = build_tokenizer(
            tokenizer_config,
            make_vocab_size_divisible_by=self.megatron_cfg.model.make_vocab_size_divisible_by
            // self.cfg["megatron_cfg"]["tensor_model_parallel_size"],
            tensor_model_parallel_size=self.cfg["megatron_cfg"][
                "tensor_model_parallel_size"
            ],
            trust_remote_code=True,
        )
        self.final_padded_vocab_size = calculate_padded_vocab_size(
            self.megatron_cfg.model.vocab_size,
            self.megatron_cfg.model.make_vocab_size_divisible_by,
            self.cfg["megatron_cfg"]["tensor_model_parallel_size"],
        )
        self.dp_size = worker_sharding_annotations.get_axis_size("data_parallel")
        self.megatron_bridge = AutoBridge.from_hf_pretrained(
            hf_model_name, trust_remote_code=True
        )

        self.should_disable_forward_pre_hook = (
            self.cfg["megatron_cfg"]["optimizer"]["use_distributed_optimizer"]
            and self.cfg["megatron_cfg"]["distributed_data_parallel_config"][
                "overlap_param_gather"
            ]
        )

        # vars used for refit
        ## will be initialized in prepare_refit_info
        # refit_param_info_mcore combines the conversion tasks with the param memory
        # [(mcore_param_name, estimated_memory), ...]
        # Note: here param name is local param name, with local layer number and
        # local expert id etc.
        self.refit_conversion_tasks = (
            None  # Meta data for conversion params from megatron bridge
        )
        self.refit_conversion_tasks_current_index = None
        self.refit_param_info_mcore = None

        ## used for streaming update inference engine weights
        self._held_gather_buffer = None

    def enable_forward_pre_hook(self):
        assert isinstance(self.model, DistributedDataParallel)
        self.model.enable_forward_pre_hook()

    def disable_forward_pre_hook(self, param_sync=True):
        assert isinstance(self.model, DistributedDataParallel)
        self.model.disable_forward_pre_hook(param_sync=param_sync)

    @wrap_with_nvtx_name("megatron_policy_worker/train")
    def train(
        self,
        data: BatchedDataDict,
        loss_fn: LossFunction,
        eval_mode: bool = False,
        gbs: Optional[int] = None,
        mbs: Optional[int] = None,
    ) -> dict[str, Any]:
        """Train the policy on a batch of data with a given loss function."""
        # Disable CUDA graphs for training to avoid conflicts with inference tensors.
        # This prevents "Inplace update to inference tensor outside InferenceMode" errors
        # when switching from generation (which uses CUDA graphs) to training.
        # Use toggle_cuda_graphs on the underlying module (not the DDP wrapper) to properly
        # reset CUDA graph state, following the pattern from megatron/rl/rl_utils.py.
        lang_module = (
            self.model.module.module
            if hasattr(self.model.module, "module")
            else self.model.module
        )

        toggle_cuda_graphs(lang_module, set_to="none", reset_cuda_graphs=True)

        self.model.zero_grad_buffer()
        if hasattr(self.model, "inference_params"):
            self.model.inference_params = None

        # Reset any cached attention states
        for module in self.model.modules():
            if hasattr(module, "reset_inference_cache"):
                module.reset_inference_cache()
            if hasattr(module, "_inference_key_value_memory"):
                module._inference_key_value_memory = None

        if gbs is None:
            gbs = self.cfg["train_global_batch_size"]
        if mbs is None:
            mbs = self.cfg["train_micro_batch_size"]
        local_gbs = gbs // self.dp_size
        total_dataset_size = torch.tensor(data.size, device="cuda")
        torch.distributed.all_reduce(
            total_dataset_size,
            op=torch.distributed.ReduceOp.SUM,
            group=parallel_state.get_data_parallel_group(),
        )
        num_global_batches = int(total_dataset_size.item()) // gbs

        if eval_mode:
            ctx: AbstractContextManager[Any] = torch.no_grad()
            self.model.eval()
        else:
            ctx = nullcontext()
            # Ensure model is in training mode
            self.model.train()

        with ctx:
            # dim 1 is always assumed to be the sequence dim, sanity check this here
            sequence_dim = 1
            seq_dim_size = data["input_ids"].shape[sequence_dim]
            for k, v in data.items():
                if torch.is_tensor(v) and len(v.shape) > 1:
                    assert v.shape[sequence_dim] == seq_dim_size, (
                        f"Dim 1 must be the sequence dim, expected dim 1={seq_dim_size} but got shape {v.shape}"
                    )

            forward_step = partial(
                forward_step_arbitrary_loss, loss_fn=loss_fn, policy_cfg=self.cfg
            )
            all_mb_metrics = []
            losses = []
            total_num_microbatches = 0
            for gb_idx in range(num_global_batches):
                global_batch = data.get_batch(batch_idx=gb_idx, batch_size=local_gbs)

                assert "sample_mask" in global_batch, (
                    "sample_mask must be present in the data!"
                )
                ## get the normalization factor for the loss
                local_valid_seqs = torch.sum(global_batch["sample_mask"])

                if not "token_mask" in global_batch:
                    local_valid_toks = (
                        local_valid_seqs * global_batch["input_ids"].shape[1]
                    )
                else:
                    local_valid_toks = torch.sum(
                        global_batch["token_mask"][:, 1:]
                        * global_batch["sample_mask"].unsqueeze(-1)
                    )

                to_reduce = torch.tensor([local_valid_seqs, local_valid_toks]).cuda()
                torch.distributed.all_reduce(
                    to_reduce, group=parallel_state.get_data_parallel_group()
                )
                global_valid_seqs, global_valid_toks = to_reduce[0], to_reduce[1]

                if (
                    hasattr(loss_fn, "loss_type")
                    and loss_fn.loss_type == LossType.TOKEN_LEVEL
                ):
                    assert "token_mask" in global_batch, (
                        "token_mask must be present in the data when using token-level loss"
                    )

                batch = data.get_batch(batch_idx=gb_idx, batch_size=local_gbs)
                pack_seqs = False
                seqlen_key = None
                pad_factor = 1
                pad_full_seq_to = None
                pad_packed_seq_to_multiple_of = 1
                if self.cfg["dynamic_batching"]["enabled"]:
                    data_iterator = batch.make_microbatch_iterator_with_dynamic_shapes()
                    data_iterator_len = (
                        batch.get_microbatch_iterator_dynamic_shapes_len()
                    )
                elif self.cfg["sequence_packing"]["enabled"]:
                    data_iterator = (
                        batch.make_microbatch_iterator_for_packable_sequences()
                    )
                    data_iterator_len, seq_dim_size = (
                        batch.get_microbatch_iterator_for_packable_sequences_len()
                    )
                    mbs = 1
                    pack_seqs = True
                    seqlen_key = "input_lengths"
                    (
                        pad_factor,
                        pad_packed_seq_to_multiple_of,
                        pad_full_seq_to,
                    ) = _get_pack_sequence_parameters_for_megatron(
                        self.cfg["megatron_cfg"],
                        seq_dim_size,
                    )
                    # if pad_full_seq_to is not None, we need to use it as the sequence length
                    seq_dim_size = pad_full_seq_to or seq_dim_size
                else:
                    data_iterator = batch.make_microbatch_iterator(mbs)
                    data_iterator_len = local_gbs // mbs

                # Track total microbatches for MoE aux-loss averaging
                total_num_microbatches += int(data_iterator_len)

                rerun_state_machine = get_rerun_state_machine()
                while rerun_state_machine.should_run_forward_backward(data_iterator):
                    # Set grad to zero.
                    self.model.zero_grad_buffer()
                    self.optimizer.zero_grad()

                    # Forward pass.
                    forward_backward_func = get_forward_backward_func()
                    losses_reduced = forward_backward_func(
                        forward_step_func=partial(
                            forward_step,
                            self.mcore_state,
                            global_valid_seqs,
                            global_valid_toks,
                            pack_sequences=pack_seqs,
                            seq_length_key=seqlen_key,
                            pad_individual_seqs_to_multiple_of=pad_factor,
                            pad_packed_seq_to_multiple_of=pad_packed_seq_to_multiple_of,
                            pad_full_seq_to=pad_full_seq_to,
                            defer_fp32_logits=self.defer_fp32_logits,
                        ),
                        data_iterator=data_iterator,
                        model=self.model,
                        num_microbatches=data_iterator_len,
                        seq_length=seq_dim_size,
                        micro_batch_size=mbs,
                        decoder_seq_length=seq_dim_size,
                        forward_only=eval_mode,
                        do_not_average_loss=True,
                    )

                # Empty unused memory.
                if self.cfg["megatron_cfg"]["empty_unused_memory_level"] >= 1:
                    torch.cuda.empty_cache()

                # Update parameters.
                if not eval_mode:
                    update_successful, grad_norm, num_zeros_in_grad = (
                        self.optimizer.step()
                    )
                else:
                    update_successful, grad_norm, num_zeros_in_grad = (True, 0.0, 0.0)

                # when freezing sub-models we may have a mixture of successful and unsucessful ranks,
                # so we must gather across mp ranks
                update_successful = logical_and_across_model_parallel_group(
                    update_successful
                )
                # grad_norm and num_zeros_in_grad will be None on ranks without trainable params,
                # so we must gather across mp ranks
                grad_norm: float = reduce_max_stat_across_model_parallel_group(
                    grad_norm
                )
                num_zeros_in_grad: float = reduce_max_stat_across_model_parallel_group(
                    num_zeros_in_grad
                )

                if update_successful:
                    skipped_iter = 0
                else:
                    skipped_iter = 1

                # Empty unused memory.
                if self.cfg["megatron_cfg"]["empty_unused_memory_level"] >= 2:
                    torch.cuda.empty_cache()

                if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    # keep all microbatch metrics to be normalized later
                    gb_loss_metrics = []
                    mb_losses = []
                    for x in losses_reduced:
                        loss_metrics = {}
                        for k in x.keys():
                            loss_metrics[k] = x[k] / num_global_batches
                        gb_loss_metrics.append(loss_metrics)
                        curr_lr = self.scheduler.get_lr(self.optimizer.param_groups[0])
                        curr_wd = self.scheduler.get_wd()
                        loss_metrics["lr"] = curr_lr
                        loss_metrics["wd"] = curr_wd
                        loss_metrics["global_valid_seqs"] = global_valid_seqs.item()
                        loss_metrics["global_valid_toks"] = global_valid_toks.item()
                        mb_losses.append(loss_metrics["loss"])

                    torch.distributed.broadcast_object_list(
                        [gb_loss_metrics],
                        src=get_pipeline_model_parallel_last_rank(),
                        group=get_pipeline_model_parallel_group(),
                    )
                else:
                    loss_metrics = [None]  # type: ignore
                    torch.distributed.broadcast_object_list(
                        loss_metrics,
                        src=get_pipeline_model_parallel_last_rank(),
                        group=get_pipeline_model_parallel_group(),
                    )
                    gb_loss_metrics = loss_metrics[0]
                    mb_losses = [x["loss"] for x in gb_loss_metrics]

                all_mb_metrics.extend(gb_loss_metrics)
                losses.append(torch.tensor(mb_losses).sum().item())

        if not eval_mode:
            # take one LR step every rollout batch
            # we need to scale the step by gbs to counteract the fact that NeMo automatically
            # scales lr_warmup_steps by gbs during init
            self.scheduler.step(increment=gbs)

        # Aggregate metrics across all microbatches
        mb_metrics = defaultdict(list)
        for m in all_mb_metrics:
            for k, v in m.items():
                mb_metrics[k].append(v)

        with torch.no_grad():
            global_loss = torch.tensor(losses, device="cuda")
            torch.distributed.all_reduce(
                global_loss,
                op=torch.distributed.ReduceOp.SUM,
                group=parallel_state.get_data_parallel_group(),
            )

        metrics = {
            "global_loss": global_loss.cpu(),
            "rank": torch.distributed.get_rank(),
            "gpu_name": torch.cuda.get_device_name(),
            "model_dtype": self.dtype,
            "all_mb_metrics": dict(mb_metrics),
            "grad_norm": torch.tensor([grad_norm]),
        }
        # Collect MoE aux metrics averaged across microbatches
        num_moe_experts = getattr(self.model.config, "num_moe_experts", None)
        if num_moe_experts is not None and num_moe_experts > 1:
            moe_loss_scale = 1.0 / max(1, total_num_microbatches)
            moe_metrics = get_moe_metrics(
                loss_scale=moe_loss_scale,
                per_layer_logging=self.cfg["megatron_cfg"]["moe_per_layer_logging"],
            )
            if moe_metrics:
                metrics["moe_metrics"] = moe_metrics
        return metrics

    @wrap_with_nvtx_name("megatron_policy_worker/get_logprobs")
    def get_logprobs(
        self, *, data: BatchedDataDict[Any], micro_batch_size: Optional[int] = None
    ) -> BatchedDataDict[LogprobOutputSpec]:
        """Get the logprobs of the model for a batch of data.

        Uses the configured logprob_batch_size to do microbatching.
        Input data is assumed to be right-padded. The method internally converts to
        left-padded format for computation, and returns outputs in right-padded format.
        If micro_batch_size is provided, it will be used instead of the configured
        logprob_batch_size.

        Returns:
          a BatchedDataDict with key "logprobs" and shape [batch_size, sequence_length].
          We use the convention that the logprob of the first token is 0 so that the sequence length is maintained.
          The logprob of input token i is specified at position i in the output logprobs tensor.
        """
        no_grad = torch.no_grad()
        no_grad.__enter__()
        logprob_batch_size = (
            micro_batch_size
            if micro_batch_size is not None
            else self.cfg["logprob_batch_size"]
        )

        # dim 1 is always assumed to be the sequence dim, sanity check this here
        sequence_dim = 1
        input_seq_dim_size = data["input_ids"].shape[sequence_dim]
        for k, v in data.items():
            if torch.is_tensor(v) and len(v.shape) > 1:
                assert v.shape[sequence_dim] == input_seq_dim_size, (
                    f"Dim 1 must be the sequence dim, expected dim 1={input_seq_dim_size} but got shape {v.shape}"
                )

        self.model.eval()

        pp_seq_dim_size = input_seq_dim_size
        pp_grp = get_pipeline_model_parallel_group()
        pad_factor = 1
        pad_packed_seq_to_multiple_of = 1
        pad_full_seq_to = None
        if self.cfg["sequence_packing"]["enabled"]:
            _, seq_dim_size = data.get_microbatch_iterator_for_packable_sequences_len()
            (
                pad_factor,
                pad_packed_seq_to_multiple_of,
                pad_full_seq_to,
            ) = _get_pack_sequence_parameters_for_megatron(
                self.cfg["megatron_cfg"],
                seq_dim_size,
            )
            pp_seq_dim_size = pad_full_seq_to or pp_seq_dim_size

        def forward_step_fn(
            data_iterator: Iterator[BatchedDataDict[Any]], model: GPTModel
        ):
            nonlocal pad_full_seq_to, pad_packed_seq_to_multiple_of, pad_factor
            data_dict = next(data_iterator).to("cuda")
            if self.cfg["sequence_packing"]["enabled"]:
                original_seq_length = data_dict["input_ids"].shape[1]
                cp_size = self.cfg["megatron_cfg"]["context_parallel_size"]
                cp_rank = get_context_parallel_rank()
                (
                    input_ids,
                    input_ids_cp_sharded,
                    packed_seq_params,
                    cu_seqlens,
                    cu_seqlens_padded,
                ) = _pack_sequences_for_megatron(
                    data_dict["input_ids"].clone(),
                    data_dict["input_lengths"],
                    pad_individual_seqs_to_multiple_of=pad_factor,
                    pad_packed_seq_to_multiple_of=pad_packed_seq_to_multiple_of,
                    pad_packed_seq_to=pad_full_seq_to,
                    cp_rank=cp_rank,
                    cp_size=cp_size,
                )
                attention_mask, position_ids = None, None
                unpacked_input_ids = data_dict["input_ids"]
            else:
                input_ids = data_dict["input_ids"]
                input_ids_cp_sharded = input_ids
                attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
                    data=input_ids,
                    eod_token=0,  # used for loss_mask, which we don't use
                    pad_token=0,  # used for loss_mask, which we don't use
                    reset_position_ids=False,
                    reset_attention_mask=False,
                    eod_mask_loss=False,
                    pad_mask_loss=False,
                )
                packed_seq_params = None
                unpacked_input_ids = input_ids

            multimodal_data = data_dict.get_multimodal_dict(
                as_tensors=True, device=input_ids.device
            )
            if len(multimodal_data) > 0:
                position_ids = None

            additional_kwargs = {}
            # Mamba models currently do not support packed_seq_params
            if packed_seq_params is not None:
                additional_kwargs["packed_seq_params"] = packed_seq_params

            if self.defer_fp32_logits:
                additional_kwargs["fp32_output"] = False

            output_tensor = model(
                input_ids=input_ids_cp_sharded,
                position_ids=position_ids,
                attention_mask=attention_mask,
                **multimodal_data,
                **additional_kwargs,
            )

            # Apply temperature scaling to logits for training
            # This matches the dtensor worker's _apply_temperature_scaling in the train method
            if "generation" in self.cfg and self.cfg["generation"] is not None:
                output_tensor.div_(self.cfg["generation"]["temperature"])

            def collection_fn(output_tensor):
                stc = time.time()
                tp_grp = get_tensor_model_parallel_group()
                tp_rank = get_tensor_model_parallel_rank()
                logprob_chunk_size = self.cfg.get("logprob_chunk_size", None)
                if self.cfg["sequence_packing"]["enabled"]:
                    token_logprobs = from_parallel_logits_to_logprobs_packed_sequences(
                        output_tensor,
                        target=input_ids,
                        cu_seqlens_padded=cu_seqlens_padded,
                        unpacked_seqlen=original_seq_length,
                        vocab_start_index=tp_rank * output_tensor.shape[-1],
                        vocab_end_index=(tp_rank + 1) * output_tensor.shape[-1],
                        group=tp_grp,
                        inference_only=True,
                        cp_group=get_context_parallel_group(),
                        chunk_size=logprob_chunk_size,
                    )
                else:
                    token_logprobs = from_parallel_logits_to_logprobs(
                        output_tensor,
                        target=unpacked_input_ids,
                        vocab_start_index=tp_rank * output_tensor.shape[-1],
                        vocab_end_index=(tp_rank + 1) * output_tensor.shape[-1],
                        tp_group=tp_grp,
                        inference_only=True,
                        chunk_size=logprob_chunk_size,
                    )

                # Prepend 0 logprob for first token to maintain same sequence length as input
                token_logprobs = torch.cat(
                    [torch.zeros_like(token_logprobs[:, :1]), token_logprobs], dim=1
                )
                return torch.tensor(0.0, device=token_logprobs.device), {
                    "logprobs": token_logprobs
                }

            return output_tensor, collection_fn

        if self.cfg["dynamic_batching"]["enabled"]:
            mb_iterator = data.make_microbatch_iterator_with_dynamic_shapes()
            data_iterator_len = data.get_microbatch_iterator_dynamic_shapes_len()
            micro_batch_size = logprob_batch_size
        elif self.cfg["sequence_packing"]["enabled"]:
            mb_iterator = data.make_microbatch_iterator_for_packable_sequences()
            data_iterator_len, _ = (
                data.get_microbatch_iterator_for_packable_sequences_len()
            )
            micro_batch_size = 1
        else:
            mb_iterator = data.make_microbatch_iterator(logprob_batch_size)
            data_iterator_len = max(1, data.size // logprob_batch_size)
            micro_batch_size = logprob_batch_size

        forward_backward_func = get_forward_backward_func()
        list_of_logprobs = forward_backward_func(
            forward_step_func=forward_step_fn,
            data_iterator=mb_iterator,
            model=self.model,
            num_microbatches=data_iterator_len,
            seq_length=pp_seq_dim_size,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=pp_seq_dim_size,
            forward_only=True,
        )
        if is_pipeline_last_stage(ignore_virtual=True):
            all_log_probs_padded = []
            all_logprobs = [l["logprobs"] for l in list_of_logprobs]
            for lp in all_logprobs:
                padding_needed = input_seq_dim_size - lp.shape[1]
                if padding_needed > 0:
                    lp = torch.nn.functional.pad(
                        lp, (0, padding_needed), mode="constant", value=0.0
                    )
                all_log_probs_padded.append(lp)

            logprobs = torch.cat(all_log_probs_padded, dim=0)
            # broadcast logprobs to first pp rank
            broadcast_tensor(logprobs, torch.distributed.get_rank(), pp_grp)
        else:
            logprobs = broadcast_tensor(
                None, get_pipeline_model_parallel_last_rank(), pp_grp
            )

        no_grad.__exit__(None, None, None)
        return BatchedDataDict[LogprobOutputSpec](logprobs=logprobs).to("cpu")

    @contextmanager
    def use_reference_model(self):
        """Context manager that temporarily swaps the reference model and active model.

        On entry: Moves model to CPU, moves reference_model to CUDA. Swaps the references
        On exit: Restores original references and re-flips cuda/cpu
        """
        ## disable overlap param gather when swapping weights
        if self.should_disable_forward_pre_hook:
            self.disable_forward_pre_hook()

        with torch.no_grad():
            try:
                # Save original references
                model_state_dict = {}
                for name, item in self.model.state_dict().items():
                    if isinstance(item, torch.Tensor):
                        item = item.detach().to(
                            device="cpu", non_blocking=True, copy=True
                        )
                    model_state_dict[name] = item

                self.model.load_state_dict(self.reference_state_dict, strict=True)
                # for name, item in self.reference_state_dict.items():
                # if isinstance(item, torch.Tensor):
                # self.model.state_dict()[name] = item.detach().to(device="cuda", non_blocking=True, copy=True)

                if self.cfg["megatron_cfg"]["empty_unused_memory_level"] >= 1:
                    gc.collect()
                    torch.cuda.empty_cache()

                # - self.model is the original reference_model, now on CUDA
                # - self.reference_model is the original model, now on CPU
                yield

            finally:
                # Restore original references and device placement
                self.model.load_state_dict(model_state_dict, strict=True)
                # for name, item in model_state_dict.items():
                # if isinstance(item, torch.Tensor):
                # item = item.detach().to(device="cuda", non_blocking=True, copy=True)
                # self.model.state_dict()[name] = item

                if self.cfg["megatron_cfg"]["empty_unused_memory_level"] >= 1:
                    gc.collect()
                    torch.cuda.empty_cache()

                ## re-enable overlap param gather after weight swap
                if self.should_disable_forward_pre_hook:
                    self.enable_forward_pre_hook()

    @wrap_with_nvtx_name("megatron_policy_worker/get_topk_logits")
    def get_topk_logits(
        self,
        *,
        data: BatchedDataDict[GenerationDatumSpec],
        k: int,
        micro_batch_size: Optional[int] = None,
    ):
        """Get the top-k logits and indices for a batch of data.

        The major difference from get_logprobs is that we compute top-k logits and indices for each position in the sequence.

        Returns:
            BatchedDataDict containing:
                - topk_logits: Tensor of top-k logits for each position in the sequence
                - topk_indices: Tensor of top-k indices for each position in the sequence
        """
        no_grad = torch.no_grad()
        no_grad.__enter__()

        logprob_batch_size = (
            micro_batch_size
            if micro_batch_size is not None
            else self.cfg["logprob_batch_size"]
        )

        sequence_dim = 1
        input_seq_dim_size = data["input_ids"].shape[sequence_dim]
        # Avoid shadowing the function argument `k` by using a distinct variable name
        for tensor_name, v in data.items():
            if torch.is_tensor(v) and len(v.shape) > 1:
                assert v.shape[sequence_dim] == input_seq_dim_size, (
                    f"Tensor {tensor_name} must have sequence dimension {sequence_dim} of size {input_seq_dim_size}, but got shape {v.shape}"
                )

        self.model.eval()

        pp_seq_dim_size = input_seq_dim_size
        pp_grp = get_pipeline_model_parallel_group()

        pad_factor = 1
        pad_packed_seq_to_multiple_of = 1
        pad_full_seq_to = None
        if self.cfg["sequence_packing"]["enabled"]:
            _, seq_dim_size = data.get_microbatch_iterator_for_packable_sequences_len()
            (
                pad_factor,
                pad_packed_seq_to_multiple_of,
                pad_full_seq_to,
            ) = _get_pack_sequence_parameters_for_megatron(
                self.cfg["megatron_cfg"],
                seq_dim_size,
            )
            pp_seq_dim_size = pad_full_seq_to or pp_seq_dim_size

        def forward_step_fn(
            data_iterator: Iterator[BatchedDataDict[Any]], model: GPTModel
        ):
            nonlocal pad_full_seq_to, pad_packed_seq_to_multiple_of, pad_factor
            data_dict = next(data_iterator).to("cuda")

            pack = self.cfg["sequence_packing"]["enabled"]
            if pack:
                original_seq_length = data_dict["input_ids"].shape[1]
                cp_size = self.cfg["megatron_cfg"]["context_parallel_size"]
                cp_rank = get_context_parallel_rank()

                (
                    input_ids_unpacked,
                    input_ids_cp_sharded,
                    packed_seq_params,
                    cu_seqlens,
                    cu_seqlens_padded,
                ) = _pack_sequences_for_megatron(
                    data_dict["input_ids"].clone(),
                    data_dict["input_lengths"],
                    pad_individual_seqs_to_multiple_of=pad_factor,
                    pad_packed_seq_to_multiple_of=pad_packed_seq_to_multiple_of,
                    pad_packed_seq_to=pad_full_seq_to,
                    cp_rank=cp_rank,
                    cp_size=cp_size,
                )
                attention_mask, position_ids = None, None
                seq_lengths = data_dict["input_lengths"]
                unpacked_seqlen = original_seq_length
            else:
                input_ids_cp_sharded = data_dict["input_ids"]
                attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
                    data=input_ids_cp_sharded,
                    eod_token=0,
                    pad_token=0,
                    reset_position_ids=False,
                    reset_attention_mask=False,
                    eod_mask_loss=False,
                    pad_mask_loss=False,
                )
                packed_seq_params = None

            multimodal_data = data_dict.get_multimodal_dict(
                as_tensors=True, device=input_ids_cp_sharded.device
            )
            if len(multimodal_data) > 0:
                position_ids = None

            additional_kwargs = {}
            if packed_seq_params is not None:
                additional_kwargs["packed_seq_params"] = packed_seq_params

            output_tensor = model(
                input_ids=input_ids_cp_sharded,
                position_ids=position_ids,
                attention_mask=attention_mask,
                **additional_kwargs,
                **multimodal_data,
            )

            if "generation" in self.cfg and self.cfg["generation"] is not None:
                output_tensor.div_(self.cfg["generation"]["temperature"])

            def collection_fn(_):
                # Only the last PP stage produces final logits/top-k; earlier stages return empty
                # if not is_pipeline_last_stage(ignore_virtual=True):
                # return output_tensor.new_zeros(()), {}

                tp_grp = get_tensor_model_parallel_group()
                tp_rank = get_tensor_model_parallel_rank()
                vocab_shard_size = output_tensor.shape[-1]
                vocab_start_index = tp_rank * vocab_shard_size

                chunk_size = None
                if "logprob_chunk_size" in self.cfg:
                    chunk_size = self.cfg["logprob_chunk_size"]

                topk_vals_local, topk_idx_local = distributed_vocab_topk(
                    output_tensor,
                    k,
                    tp_grp,
                    vocab_start_index=vocab_start_index,
                    vocab_end_index=vocab_start_index + vocab_shard_size,
                    chunk_size=chunk_size,
                )

                if self.cfg["megatron_cfg"]["context_parallel_size"] > 1:
                    cp_grp = get_context_parallel_group()
                    if pack:
                        # Per-sequence CP allgather following packed-sequence logic
                        batch_size = data_dict["input_ids"].shape[0]
                        total_packed_len = int(cu_seqlens_padded[-1].item())

                        topk_vals_full = torch.zeros(
                            (1, total_packed_len, k),
                            dtype=topk_vals_local.dtype,
                            device=topk_vals_local.device,
                        )
                        topk_idx_full = torch.zeros(
                            (1, total_packed_len, k),
                            dtype=topk_idx_local.dtype,
                            device=topk_idx_local.device,
                        )

                        for i in range(batch_size):
                            start_idx = int(cu_seqlens_padded[i].item())
                            end_idx = int(cu_seqlens_padded[i + 1].item())
                            if end_idx > start_idx:
                                local_vals_slice = topk_vals_local[
                                    :, start_idx // cp_size : end_idx // cp_size, :
                                ]
                                local_idx_slice = topk_idx_local[
                                    :, start_idx // cp_size : end_idx // cp_size, :
                                ]
                                gathered_vals = allgather_cp_sharded_tensor(
                                    local_vals_slice, cp_grp, seq_dim=1
                                )
                                gathered_idx = allgather_cp_sharded_tensor(
                                    local_idx_slice, cp_grp, seq_dim=1
                                )
                                # Some kernels may return [X, Y, k] where X*Y = (end_idx - start_idx).
                                # Flatten leading dims and reshape to [1, expected_len, k] to match target.
                                expected_len = end_idx - start_idx
                                if (
                                    gathered_vals.dim() == 3
                                    and gathered_vals.shape[1] != expected_len
                                ):
                                    gathered_vals = gathered_vals.reshape(
                                        1, expected_len, gathered_vals.shape[-1]
                                    )
                                if (
                                    gathered_idx.dim() == 3
                                    and gathered_idx.shape[1] != expected_len
                                ):
                                    gathered_idx = gathered_idx.reshape(
                                        1, expected_len, gathered_idx.shape[-1]
                                    )
                                topk_vals_full[:, start_idx:end_idx, :] = gathered_vals
                                topk_idx_full[:, start_idx:end_idx, :] = gathered_idx
                    else:
                        # Sequence packing must be enabled when CP > 1
                        raise RuntimeError(
                            "Context Parallelism (CP>1) requires sequence packing to be enabled."
                        )
                else:
                    topk_vals_full = topk_vals_local
                    topk_idx_full = topk_idx_local

                if pack:
                    batch_size = data_dict["input_ids"].shape[0]
                    out_vals = torch.zeros(
                        (batch_size, unpacked_seqlen, k),
                        dtype=topk_vals_full.dtype,
                        device=topk_vals_full.device,
                    )
                    out_idx = torch.zeros(
                        (batch_size, unpacked_seqlen, k),
                        dtype=topk_idx_full.dtype,
                        device=topk_idx_full.device,
                    )
                    for i in range(batch_size):
                        seq_len = int(seq_lengths[i].item())
                        start_idx = int(cu_seqlens_padded[i].item())
                        if seq_len > 0:
                            out_vals[i, :seq_len, :] = topk_vals_full[
                                0, start_idx : start_idx + seq_len, :
                            ]
                            out_idx[i, :seq_len, :] = topk_idx_full[
                                0, start_idx : start_idx + seq_len, :
                            ]
                    return output_tensor.new_zeros(()), {
                        "topk_logits": out_vals,
                        "topk_indices": out_idx,
                    }
                else:
                    return output_tensor.new_zeros(()), {
                        "topk_logits": topk_vals_full,
                        "topk_indices": topk_idx_full,
                    }

            return output_tensor, collection_fn

        if self.cfg["dynamic_batching"]["enabled"]:
            mb_iterator = data.make_microbatch_iterator_with_dynamic_shapes()
            data_iterator_len = data.get_microbatch_iterator_dynamic_shapes_len()
            micro_batch = logprob_batch_size
        elif self.cfg["sequence_packing"]["enabled"]:
            mb_iterator = data.make_microbatch_iterator_for_packable_sequences()
            data_iterator_len, _ = (
                data.get_microbatch_iterator_for_packable_sequences_len()
            )
            micro_batch = 1
        else:
            mb_iterator = data.make_microbatch_iterator(logprob_batch_size)
            data_iterator_len = max(1, data.size // logprob_batch_size)
            micro_batch = logprob_batch_size

        forward_backward_func = get_forward_backward_func()
        list_of_outputs = forward_backward_func(
            forward_step_func=forward_step_fn,
            data_iterator=mb_iterator,
            model=self.model,
            num_microbatches=data_iterator_len,
            seq_length=pp_seq_dim_size,
            micro_batch_size=micro_batch,
            decoder_seq_length=pp_seq_dim_size,
            forward_only=True,
        )

        if is_pipeline_last_stage(ignore_virtual=True):
            logits_chunks = []
            indices_chunks = []
            for out in list_of_outputs:
                tk = out["topk_logits"]
                ti = out["topk_indices"]
                pad_len = input_seq_dim_size - tk.shape[1]
                if pad_len > 0:
                    tk = torch.nn.functional.pad(tk, (0, 0, 0, pad_len), value=0.0)
                    ti = torch.nn.functional.pad(ti, (0, 0, 0, pad_len), value=0)
                logits_chunks.append(tk)
                indices_chunks.append(ti)

            topk_logits = torch.cat(logits_chunks, dim=0)
            topk_indices = torch.cat(indices_chunks, dim=0)

            topk_logits = broadcast_tensor(
                topk_logits, torch.distributed.get_rank(), pp_grp
            )
            topk_indices = broadcast_tensor(
                topk_indices, torch.distributed.get_rank(), pp_grp
            )
        else:
            last_pp_rank = get_pipeline_model_parallel_last_rank()
            topk_logits = broadcast_tensor(None, last_pp_rank, pp_grp)
            topk_indices = broadcast_tensor(None, last_pp_rank, pp_grp)

        no_grad.__exit__(None, None, None)
        return BatchedDataDict.from_batches(
            [{"topk_logits": topk_logits.cpu(), "topk_indices": topk_indices.cpu()}]
        )

    @torch.no_grad()
    @wrap_with_nvtx_name("megatron_policy_worker/generate")
    def generate(
        self, *, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> BatchedDataDict[GenerationOutputSpec]:
        """Generate a batch of data using huggingface framework generation.

        Args:
            data: BatchedDataDict containing input_ids and input_lengths tensors
        Returns:
            BatchedDataDict conforming to GenerationOutputSpec:
                - output_ids: input + generated token IDs
                - logprobs: Log probabilities for each token
                - generation_lengths: Lengths of each response
        """
        # 512 bATCH SIZE (200 tokens)
        no_grad = torch.no_grad()
        no_grad.__enter__()
        self.model.config.flash_decode = False
        if self.should_disable_forward_pre_hook:
            self.model = self.move_model(
                self.model, "cuda", move_params=True, move_grads=False
            )
        # Verify input is right padded
        assert isinstance(data, BatchedDataDict), (
            f"data must be a BatchedDataDict, got type: {type(data)}"
        )
        assert "input_ids" in data and "input_lengths" in data, (
            f"input_ids and input_lengths must be present in the BatchedDataDict, got keys: {data.keys()}"
        )
        is_right_padded, error_msg = verify_right_padding(
            data, pad_value=self.tokenizer.pad_token_id
        )
        if not is_right_padded:
            warnings.warn(
                f"Input to Megatron Generation worker is not properly right-padded: {error_msg}"
            )

        model_cfg = self.megatron_cfg.model
        inference_wrapper_config = InferenceWrapperConfig(
            hidden_size=model_cfg.hidden_size,
            inference_batch_times_seqlen_threshold=1000000,
            fp32_residual_connection=model_cfg.fp32_residual_connection,
            params_dtype=model_cfg.params_dtype,
            padded_vocab_size=self.final_padded_vocab_size,  # Use the potentially updated value
            inference_max_seq_length=self.cfg["generation"]["max_new_tokens"],  # type: ignore
            inference_max_requests=self.cfg["generation_batch_size"],
        )

        from megatron.core.inference.contexts.dynamic_context import (
            DynamicInferenceContext,
        )
        from megatron.core.inference.engines.dynamic_engine import (
            DynamicInferenceEngine,
        )
        from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
            GPTInferenceWrapper,
        )
        from megatron.core.inference.sampling_params import SamplingParams

        mcore_generation_config = self.cfg["generation"]["mcore_generation_config"]
        buffer_size_gb = mcore_generation_config.get("buffer_size_gb", 20)

        num_cuda_graphs = mcore_generation_config.get("num_cuda_graphs", 16)
        block_size_tokens = mcore_generation_config.get("block_size_tokens", 256)
        use_cuda_graphs_for_non_decode_steps = mcore_generation_config.get(
            "use_cuda_graphs_for_non_decode_steps", True
        )
        enable_chunked_prefill = mcore_generation_config.get(
            "enable_chunked_prefill", True
        )
        unified_memory_level = mcore_generation_config.get("unified_memory_level", 0)
        buffer_guaranteed_fraction = mcore_generation_config.get(
            "buffer_guaranteed_fraction", 0.1
        )
        max_tokens = mcore_generation_config.get("max_tokens", 16384)

        # Get the underlying language module (not the DDP wrapper) for toggle_cuda_graphs.
        # This follows the pattern from megatron/rl/rl_utils.py megatron_rl_inference_mode().
        # The DDP wrapper can interfere with CUDA graph state management.
        lang_module = (
            self.model.module.module
            if hasattr(self.model.module, "module")
            else self.model.module
        )

        # Enable CUDA graphs for inference using toggle_cuda_graphs on the underlying module.
        # reset_cuda_graphs=True ensures fresh CUDA graphs are created for each generate() call.
        

        toggle_cuda_graphs(lang_module, set_to="local", reset_cuda_graphs=True)

        model_config = self.model.config
        model_config.cuda_graph_impl = "local"

        dynamic_context = DynamicInferenceContext(
            params_dtype=inference_wrapper_config.params_dtype,
            num_layers=model_config.num_layers,
            kv_channels=model_config.kv_channels,
            num_attention_heads=model_config.num_query_groups,
            max_sequence_length=self.cfg["generation"]["max_new_tokens"],
            buffer_guaranteed_fraction=buffer_guaranteed_fraction,
            buffer_size_gb=buffer_size_gb,
            materialize_only_last_token_logits=False,
            num_cuda_graphs=num_cuda_graphs,
            block_size_tokens=block_size_tokens,
            tensor_model_parallel_size=self.cfg["megatron_cfg"][
                "tensor_model_parallel_size"
            ],
            use_cuda_graphs_for_non_decode_steps=use_cuda_graphs_for_non_decode_steps,
            use_flashinfer_fused_rope=False,
            unified_memory_level=unified_memory_level,
            max_tokens_override=max_tokens,
        )
        inference_wrapped_model = GPTInferenceWrapper(
            self.model, inference_wrapper_config, dynamic_context
        )

        inference_wrapped_model.prep_model_for_inference()
        # Set pipeline parallel flag
        inference_wrapped_model.model_is_pipeline_parallel = (
            self.cfg["megatron_cfg"]["pipeline_model_parallel_size"] > 1
        )

        text_generation_controller = TextGenerationController(
            inference_wrapped_model=inference_wrapped_model,
            tokenizer=self.megatron_tokenizer,
        )

        # Calculate seed based on node and rank to ensure reproducibility across workers
        local_rank = torch.cuda.current_device()  # Local GPU index on the node
        num_gpus_per_node = torch.cuda.device_count()
        node_idx = self.rank // num_gpus_per_node if num_gpus_per_node > 0 else 0
        seed = (node_idx * 1024) + local_rank

        # New API: DynamicInferenceEngine has additional parameters
        dynamic_engine = DynamicInferenceEngine(
            text_generation_controller,
            dynamic_context,
            enable_cuda_graph=True,
            random_seed=seed,
            track_paused_request_events=False,
            enable_chunked_prefill=enable_chunked_prefill,
            inference_logging_step_interval=0,
        )

        # Handle None values for top_k - convert to integer as required by Megatron
        top_k_cfg = self.cfg["generation"]["top_k"]
        top_k_val = 1 if greedy else (int(top_k_cfg) if top_k_cfg is not None else 0)

        top_p_cfg = self.cfg["generation"]["top_p"]
        top_p_val = (
            0.0 if greedy else (float(top_p_cfg) if top_p_cfg is not None else 0.0)
        )

        # New API: SamplingParams now includes termination_id and uses num_tokens_total
        sampling_params = SamplingParams(
            temperature=self.cfg["generation"]["temperature"] if not greedy else 0,
            top_k=top_k_val,
            top_p=top_p_val,
            skip_prompt_log_probs=False,
            return_log_probs=True,
            num_tokens_total=self.cfg["generation"]["max_new_tokens"],
            num_tokens_to_generate=None,
            termination_id=self.megatron_tokenizer.eod,
        )

        input_ids = data["input_ids"]
        prompt_tokens_tensor = input_ids.cuda()
        prompt_lengths_tensor = data["input_lengths"]
        request_id = 0

        # New API: add_request now takes sampling_params as a parameter
        for p, prompt_len in zip(
            prompt_tokens_tensor, prompt_lengths_tensor, strict=True
        ):
            dynamic_engine.add_request(
                request_id,
                p[:prompt_len],
                sampling_params=sampling_params,
            )
            request_id += 1

        result = []
        import time
        start = time.time()
        while dynamic_engine.has_unfinished_requests():
            result_step = dynamic_engine.step_modern(verbose=False)
            finished_requests = result_step.get("finished_requests", [])
            for finished_request in finished_requests:
                result.append(finished_request)
        end = time.time()
        print(f"SHAN : Time taken: {end - start} seconds")

        # Sort results by request_id to maintain original batch order
        result.sort(key=lambda x: x.request_id)

        out = {
            "tokens": [x.prompt_tokens.tolist() + x.generated_tokens for x in result],
            "logprobs": [x.prompt_log_probs + x.generated_log_probs for x in result],
        }

        input_lengths = data["input_lengths"]
        # pad the out "tokens" and "logprobs" and make them into tensors from lists
        batch_size = data["input_ids"].size(0)
        max_gen_seq_len = max([len(x.generated_tokens) for x in result])
        padded_input_length = input_ids.size(1)

        max_seq_len = padded_input_length + max_gen_seq_len
        # Create padded tensors for tokens and logprobs
        output_ids_padded = torch.full(
            (batch_size, max_seq_len),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=data["input_ids"].device,
        )

        logprobs_padded = torch.zeros(
            (batch_size, max_seq_len),
            dtype=torch.float,
            device=data["input_ids"].device,
        )

        # Fill in the padded tensors with actual values
        generation_lengths = torch.zeros(
            batch_size, dtype=torch.long, device=data["input_ids"].device
        )
        unpadded_sequence_lengths = torch.zeros(
            batch_size, dtype=torch.long, device=data["input_ids"].device
        )
        for i in range(batch_size):
            seq_len = len(out["tokens"][i])
            output_ids_padded[i, :seq_len] = torch.tensor(
                out["tokens"][i], dtype=torch.long, device=data["input_ids"].device
            )
            generation_lengths[i] = seq_len - input_lengths[i].item()
            unpadded_sequence_lengths[i] = seq_len
            logprob_len = len(out["logprobs"][i])
            logprobs_padded[i, 1 : logprob_len + 1] = torch.tensor(
                out["logprobs"][i],
                dtype=torch.float,
                device=data["input_ids"].device,
            )

        out_dict = {
            "output_ids": output_ids_padded,
            "logprobs": logprobs_padded,
            "generation_lengths": generation_lengths,
            "unpadded_sequence_lengths": unpadded_sequence_lengths,
        }

        self.model.config.flash_decode = False

        # Disable forward pre-hooks BEFORE toggle_cuda_graphs to avoid stale hook handles
        # for CudaGraphManager modules that will be removed by toggle_cuda_graphs.
        if self.should_disable_forward_pre_hook:
            self.disable_forward_pre_hook()

        # Disable CUDA graphs after inference using toggle_cuda_graphs on the underlying module.
        # This follows the pattern from megatron/rl/rl_utils.py megatron_rl_inference_mode().
        # reset_cuda_graphs=True ensures:
        # 1. cudagraph_manager attributes are deleted from modules
        # 2. _CudagraphGlobalRecord state is reset via delete_cuda_graphs()
        # 3. Model is ready for training (avoids "inference tensor" errors)
        # 4. Next generate() call will create fresh CUDA graphs with new DynamicInferenceContext
        toggle_cuda_graphs(lang_module, set_to="none", reset_cuda_graphs=True)

        no_grad.__exit__(None, None, None)

        return BatchedDataDict.from_batches([out_dict]).to("cpu")

    @torch.no_grad()
    @wrap_with_nvtx_name("megatron_policy_worker/prepare_refit_info")
    def prepare_refit_info(self) -> None:
        """Prepare state dict metadata for weight refitting and IPC streaming."""
        self.refit_param_info_mcore = self._calculate_refit_param_info()

        # Collect tensor metadata for refit / hf side info
        refit_param_info_hf = {}
        # Reuse shared iterator that appends FP8 KV/Q scales when enabled
        for name, tensor in self._iter_params_with_optional_kv_scales():
            refit_param_info_hf[name] = (tensor.shape, tensor.dtype)

        return refit_param_info_hf

    def _calculate_refit_param_info(self) -> list[tuple[str, int]]:
        """Calculate parameter information for refit.

        Each task contains:
        - param_name: Local parameter name without module prefixes
        - mapping: MegatronParamMapping instance for weight transformation
        - pp_rank: Pipeline-parallel rank owning the parameter
        - vp_stage: Virtual-pipeline stage index
        - megatron_module: Reference to Megatron model/submodule
        - param_weight: Target parameter tensor for converted weight

        Returns:
            List of (parameter_name, size_in_bytes) tuples.
        """
        self.refit_conversion_tasks = self.megatron_bridge.get_conversion_tasks(
            [self.model]
        )
        param_info = []

        def calculate_size_in_bytes(param, tp_size, ep_size):
            if param is None:
                # need to broadcast for other pp ranks
                size_in_bytes = None
            else:
                # Calculate size for this parameter
                prec_to_bytes = {
                    torch.bfloat16: 2,
                    torch.float16: 2,
                    torch.float32: 4,
                }
                scale = prec_to_bytes[self.dtype] / prec_to_bytes[param.dtype]
                size_in_bytes = (
                    param.element_size() * param.numel() * tp_size * ep_size * scale
                )

            # Broadcast size_in_bytes across pipeline parallel ranks
            return broadcast_object_across_pp_ranks(size_in_bytes)

        for task in self.refit_conversion_tasks:
            param_info.append(
                (
                    task.param_name,
                    calculate_size_in_bytes(
                        task.param_weight,
                        task.mapping.tp_size,
                        task.mapping.ep_size if task.mapping.is_expert else 1,
                    ),
                )
            )
        return param_info

    def _iter_params_with_optional_kv_scales(
        self,
        kv_scales: Optional[dict[str, float]] = None,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield exported HF parameters and optionally append FP8 KV/Q scale tensors.

        This helper is used by both IPC-based streaming and collective broadcast
        so that the logic for adding KV scales stays consistent in one place.
        """
        base_iter = self.megatron_bridge.export_hf_weights(
            [self.model],
            show_progress=False,
            conversion_tasks=self.refit_conversion_tasks,  # used for metadata caching
        )

        # Yield the original parameters first.
        for name, tensor in base_iter:
            yield name, tensor

        # Check whether FP8 KV cache is enabled.
        use_fp8_kv_cache = False
        if (
            "generation" in self.cfg
            and self.cfg["generation"] is not None
            and self.cfg["generation"]["backend"] == "vllm"
        ):
            generation_cfg = cast(VllmConfig, self.cfg["generation"])
            use_fp8_kv_cache = (
                "vllm_cfg" in generation_cfg
                and "kv_cache_dtype" in generation_cfg["vllm_cfg"]
                and generation_cfg["vllm_cfg"]["kv_cache_dtype"].startswith("fp8")
            )

        if not use_fp8_kv_cache:
            return

        # Append KV (and potentially Q) scale entries to match metadata.
        num_layers = self.megatron_bridge.transformer_config.num_layers
        keys: list[str] = []
        for layer_idx in range(num_layers):
            scale_names = get_vllm_qkv_scale_names(layer_idx)
            keys.extend(scale_names.values())

        for param_name in keys:
            if kv_scales and param_name in kv_scales:
                scale_value = kv_scales[param_name]
            else:
                scale_value = 1.0
            scale_tensor = torch.tensor(
                scale_value, dtype=torch.float32, device="cuda"
            ).reshape(1)
            yield param_name, scale_tensor

    @torch.no_grad()
    @wrap_with_nvtx_name("megatron_policy_worker/stream_weights_via_ipc_zmq")
    def stream_weights_via_ipc_zmq(
        self, buffer_size_bytes: int = 0, kv_scales: Optional[dict[str, float]] = None
    ) -> None:
        """Stream model weights to peer process via ZMQ IPC socket."""
        self.maybe_init_zmq()

        from nemo_rl.models.policy.utils import stream_weights_via_ipc_zmq_impl

        # Use the shared implementation to append optional KV scales.
        stream_weights_via_ipc_zmq_impl(
            params_generator=self._iter_params_with_optional_kv_scales(
                kv_scales=kv_scales
            ),
            buffer_size_bytes=buffer_size_bytes,
            zmq_socket=self.zmq_socket,
            rank=self.rank,
            worker_name=str(self),
        )

    @torch.no_grad()
    def broadcast_weights_for_collective(
        self, kv_scales: Optional[dict[str, float]] = None
    ) -> None:
        """Broadcast the weights for collective communication."""
        # param_iterator will return (name, tensor), we only need tensor.
        packed_broadcast_producer(
            iterator=self._iter_params_with_optional_kv_scales(kv_scales=kv_scales),
            group=self.model_update_group,
            src=0,
            post_iter_func=lambda x: x[1],
        )

    def prepare_for_lp_inference(self):
        self.model = self.move_model(self.model, "cuda", move_grads=False)
        self.model.eval()

        # offload grads to cpu
        self.model = self.move_model(
            self.model, "cpu", move_params=False, move_grads=True
        )  # get rid of grad buffers

        # offload optimizer to cpu
        torch.randn(1).cuda()  # wake up torch allocator
        if (
            hasattr(self, "optimizer")
            and self.optimizer is not None
            and not self.optimizer_cpu_offload
            and self.offload_optimizer_for_logprob
        ):
            self.move_optimizer("cpu")

        gc.collect()
        torch.cuda.empty_cache()

    def prepare_for_training(self, *args, **kwargs):
        # onload models and optimizer state to cuda
        self.model = self.move_model(
            self.model, "cuda", move_grads=True, move_params=True
        )
        self.model.train()

        # Move optimizer state to CUDA if it exists
        # colocated generation will always offload optimizer to cuda before refit
        if (
            hasattr(self, "optimizer")
            and self.optimizer is not None
            and not self.optimizer_cpu_offload
            and (self.offload_optimizer_for_logprob or self.is_generation_colocated)
        ):
            self.move_optimizer("cuda")

        if self.cfg["megatron_cfg"]["empty_unused_memory_level"] >= 1:
            torch.cuda.empty_cache()

    @wrap_with_nvtx_name("megatron_policy_worker/offload_before_refit")
    def offload_before_refit(self):
        """Offload the optimizer and buffers to the CPU."""
        no_grad = torch.no_grad()
        no_grad.__enter__()
        allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
        reserved = torch.cuda.memory_reserved() / (1024**3)  # Convert to GB
        print(
            f"GPU Memory before optimizer offload: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )
        self.model = self.move_model(
            self.model, "cpu", move_params=False, move_grads=True
        )  # get rid of grad buffers
        torch.randn(1).cuda()  # wake up torch allocator
        if (
            hasattr(self, "optimizer")
            and self.optimizer is not None
            and not self.optimizer_cpu_offload
        ):
            self.move_optimizer("cpu")

        gc.collect()
        torch.cuda.empty_cache()

        # Print memory stats after offloading
        allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
        reserved = torch.cuda.memory_reserved() / (1024**3)  # Convert to GB
        print(
            f"GPU Memory after optimizer offload: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )
        no_grad.__exit__(None, None, None)

    @wrap_with_nvtx_name("megatron_policy_worker/offload_after_refit")
    def offload_after_refit(self):
        """Offload as much as possible on the CPU."""
        no_grad = torch.no_grad()
        no_grad.__enter__()
        self.model = self.move_model(self.model, "cpu")
        self.model.eval()
        torch.randn(1).cuda()  # wake up torch allocator
        self.offload_before_refit()  # rerun the old offload function

        allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
        reserved = torch.cuda.memory_reserved() / (1024**3)  # Convert to GB
        print(
            f"GPU Memory after refit complete: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )
        no_grad.__exit__(None, None, None)

    @torch.no_grad()
    def move_model(
        self,
        model: torch.nn.Module,
        device: str,
        move_params: bool = True,
        move_grads: bool = True,
    ) -> torch.nn.Module:
        # move all param and grad buffers to the device
        if isinstance(model, DistributedDataParallel):
            # DDP case
            for buffers in [model.buffers, model.expert_parallel_buffers]:
                for buffer_idx in range(len(buffers)):
                    if device == "cpu":
                        buffers[buffer_idx].offload_to_cpu(
                            move_params=move_params, move_grads=move_grads
                        )
                    elif device == "cuda":
                        buffers[buffer_idx].reload_from_cpu(
                            move_params=move_params, move_grads=move_grads
                        )
                    else:
                        raise ValueError(
                            f"Invalid device: {device}. Only strings 'cpu' and 'cuda' are supported."
                        )
        elif isinstance(model, custom_FSDP):
            if device == "cpu":
                model.param_and_grad_buffer.offload_to_cpu(move_params, move_grads)
            elif device == "cuda":
                model.param_and_grad_buffer.reload_from_cpu(
                    move_params=move_params, move_grads=move_grads
                )
            else:
                raise ValueError(
                    f"Invalid device: {device}. Only strings 'cpu' and 'cuda' are supported."
                )
        else:
            # Ordinary offload case
            if move_params:
                for name, param in model.state_dict().items():
                    new_state_dict = {}
                    for name, item in model.state_dict().items():
                        if isinstance(item, torch.Tensor):
                            item = item.detach().to(
                                device=device, non_blocking=True, copy=True
                            )
                        new_state_dict[name] = item
                    model.load_state_dict(new_state_dict)
        return model

    def move_optimizer(self, device: str):
        # Iterate through the state dictionaries for each parameter group
        if isinstance(self.optimizer, ChainedOptimizer):
            optimizer_state = self.optimizer.state
        else:
            optimizer_state = self.optimizer._get_state()
        for _, state in optimizer_state.items():
            # Iterate through the state items (e.g., momentum, variance) for a parameter
            for k, v in state.items():
                # Check if the item is a tensor
                if torch.is_tensor(v):
                    # Move the tensor to device and update the state dictionary
                    if device == "cpu":
                        if v.is_cuda:
                            state[k] = v.to("cpu")
                    elif device == "cuda":
                        if not v.is_cuda:
                            state[k] = v.to("cuda")
                    else:
                        raise ValueError(
                            f"Invalid device: {device}. Only strings 'cpu' and 'cuda' are supported."
                        )

    def save_checkpoint(
        self,
        weights_path: str,
        optimizer_path: Optional[str] = None,
        **kwargs,
    ):
        """Save a training checkpoint.

        Args:
            weights_path: The specific directory path where the checkpoint will be saved.
            optimizer_path: If not None, optimizer and scheduler states are saved if they exist.
        """
        if not torch.distributed.is_initialized():
            raise RuntimeError(
                "Distributed process group is not initialized. Cannot save checkpoint."
            )

        if self.mcore_state is None or self.model is None:
            raise RuntimeError(
                "Megatron core state or model is not initialized. Cannot save checkpoint."
            )

        original_save_path = self.mcore_state.cfg.checkpoint.save
        # save_dir = os.path.dirname(weights_path)
        release_name = os.path.basename(weights_path)

        try:
            maybe_finalize_async_save(
                self.mcore_state,
                ckpt_cfg=self.mcore_state.cfg.checkpoint,
                blocking=False,
            )
            self.mcore_state.cfg.checkpoint.save = weights_path

            optimizer_to_save = None
            scheduler_to_save = None

            if optimizer_path is not None:
                if self.optimizer is not None:
                    optimizer_to_save = self.optimizer
                if self.scheduler is not None:
                    scheduler_to_save = self.scheduler

            # Ensure model is in eval mode for consistent saving, unless actively training
            # This is a common practice, though NeMo's save might handle this.
            # For safety, if not in training loop, setting to eval.
            is_training = self.model.training
            if not is_training:
                self.model.eval()

            if self.should_disable_forward_pre_hook:
                self.disable_forward_pre_hook()
            save_checkpoint(
                state=self.mcore_state,
                model=[self.model],
                optimizer=optimizer_to_save,
                opt_param_scheduler=scheduler_to_save,
                num_floating_point_operations_so_far=self.mcore_state.train_state.floating_point_operations_so_far,
                checkpointing_context=self.checkpointing_context,
            )
            print(f"Saved checkpoint to {weights_path}")
            maybe_finalize_async_save(
                self.mcore_state,
                ckpt_cfg=self.mcore_state.cfg.checkpoint,
                blocking=True,
                terminate=True,
            )
            if self.should_disable_forward_pre_hook:
                self.enable_forward_pre_hook()

            if not is_training:  # Restore training state if it was changed
                self.model.train()

        except Exception as e:
            print(f"Failed to save checkpoint to {weights_path}: {e}")
            raise
        finally:
            self.mcore_state.cfg.checkpoint.save = original_save_path

    def load_checkpoint(self, weights_path: str, optimizer_path: Optional[str] = None):
        """Load a training checkpoint.

        Args:
            weights_path: The exact directory path from which to load the checkpoint.
            optimizer_path: If not None, attempts to load optimizer and scheduler states
                            if self.optimizer and self.scheduler are initialized.
        """
        raise NotImplementedError(
            "Loading checkpoints outside of the init function is not yet implemented for Megatron policy."
        )

    def check_tensor_parallel_attributes(self) -> dict[str, Any]:
        """Check tensor parallel attributes on model parameters.

        Returns:
            Dictionary containing information about tensor parallel parameters:
            - tp_params: List of parameter names that have tensor_model_parallel=True
            - non_tp_params: List of parameter names that have tensor_model_parallel=False
            - total_params: Total number of parameters checked
            - tp_size: Tensor parallel size from config
        """
        tp_params = []
        non_tp_params = []
        total_params = 0

        for name, param in self.model.named_parameters():
            total_params += 1
            tensor_model_parallel = getattr(param, "tensor_model_parallel", False)

            if tensor_model_parallel:
                tp_params.append(
                    {
                        "name": name,
                        "tensor_model_parallel": tensor_model_parallel,
                        "partition_dim": getattr(param, "partition_dim", None),
                        "partition_stride": getattr(param, "partition_stride", None),
                        "shape": list(param.shape),
                    }
                )
            else:
                non_tp_params.append(
                    {
                        "name": name,
                        "tensor_model_parallel": tensor_model_parallel,
                        "shape": list(param.shape),
                    }
                )

        return {
            "tp_params": tp_params,
            "non_tp_params": non_tp_params,
            "total_params": total_params,
            "tp_size": self.megatron_cfg.model.tensor_model_parallel_size,
        }

    @torch.no_grad()
    def calibrate_qkv_fp8_scales(
        self,
        *,
        data: BatchedDataDict[Any],
        micro_batch_size: Optional[int] = None,
        percentile: float = 99.9,
        margin: float = 1.05,
        include_q: bool = False,
    ) -> dict[str, Any]:
        """One-shot calibration of Q/K/V activation scales (for FP8 KV cache).

        - Captures each layer's `query_key_value` output through forward hooks, splits Q/K/V, and computes percentile amax.
        - In parallel (DP/TP/PP) environments, first computes local percentiles, then takes max across all ranks for conservativeness.
        - By default only returns and saves K/V scales, optionally returns Q.

        Args:
            data: Representative sample batch for calibration, following get_logprobs input conventions.
            micro_batch_size: Micro batch size during calibration; if None, reuses logprob_batch_size.
            percentile: Percentile for amax (e.g. 99.9).
            margin: Margin factor, e.g. 1.05.
            save_path: If provided, rank0 will save results as JSON.
            include_q: Whether to also return Q scale (usually only K/V needed).

        Returns:
            { "format": "fp8", "percentile": float, "margin": float,
              "layers": { layer_name: {"k_scale": float, "v_scale": float[, "q_scale": float] } } }
        """

        # Allow overriding FP8 max for Q, K, V via environment variables for ease of testing.
        # Defaults align with FP8 e4m3 max magnitude.
        # Use different defaults for Q, K, V to adapt to distribution diffefences
        def _get_env_float(name: str, default: float) -> float:
            try:
                val = os.getenv(name, None)
                return float(val) if val is not None and val != "" else default
            except Exception:
                return default

        FP8_MAX_Q = _get_env_float("FP8_MAX_Q", 448.0)
        FP8_MAX_K = _get_env_float("FP8_MAX_K", 448.0)
        FP8_MAX_V = _get_env_float("FP8_MAX_V", 448.0)

        self.model.eval()

        # Record local percentile amax for q/k/v of each layer
        layer_to_samples_q: dict[str, list[float]] = defaultdict(list)
        layer_to_samples_k: dict[str, list[float]] = defaultdict(list)
        layer_to_samples_v: dict[str, list[float]] = defaultdict(list)
        hook_handles = []

        def _extract_layer_key(module_name: str) -> str:
            # Expected format: "module.decoder.layers.<idx>.self_attention.query_key_value"
            m = re.search(r"module\.decoder\.layers\.(\d+)", module_name)
            if m is not None:
                return f"layer_{m.group(1)}"
            return module_name

        # Hook to capture q/k/v after q/k norm and RoPE
        def _pre_hook_builder_core_attention(module_name: str):
            layer_key = _extract_layer_key(module_name)

            def _pre_hook(module, inputs):
                args = inputs if isinstance(inputs, (tuple, list)) else (inputs,)
                if len(args) == 1 and isinstance(args[0], (tuple, list)):
                    args = args[0]
                # Expected first 3 args to be q, k, v (typical signature for Megatron CoreAttention)
                q = args[0]
                k = args[1]
                v = args[2]
                if include_q:
                    layer_to_samples_q[layer_key].append(
                        float(torch.amax(torch.abs(q)).item())
                    )
                layer_to_samples_k[layer_key].append(
                    float(torch.amax(torch.abs(k)).item())
                )
                layer_to_samples_v[layer_key].append(
                    float(torch.amax(torch.abs(v)).item())
                )

            return _pre_hook

        matched_modules = []
        # Try to register forward_pre_hook on core_attention first
        for name, module in self.model.named_modules():
            if "self_attention.core_attention" in name:
                try:
                    handle = module.register_forward_pre_hook(
                        _pre_hook_builder_core_attention(name)
                    )
                    hook_handles.append(handle)
                    matched_modules.append((name, module.__class__.__name__, "pre"))
                except Exception as e:
                    print(
                        f"Error registering pre-hook for qkv scale calibration on {name}: {e}"
                        " Please check if the model is compatible with the current calibration logic. "
                        "The expected module name is 'self_attention.core_attention'."
                    )
                    raise

        # Run a forward pass to trigger hooks (reuse get_logprobs forward path)
        try:
            _ = self.get_logprobs(data=data, micro_batch_size=micro_batch_size)
        finally:
            for h in hook_handles:
                try:
                    h.remove()
                except Exception as e:
                    print(f"Error removing hook for qkv scale calibration: {e}")
                    raise

        # Compute local percentile amax
        def _percentile(values: list[float], p: float) -> float:
            if not values:
                return 0.0
            t = torch.tensor(sorted(values), device="cuda", dtype=torch.float32)
            rank = max(
                0, min(len(values) - 1, int(round((p / 100.0) * (len(values) - 1))))
            )
            return float(t[rank].item())

        local_layer_to_pamax = {}
        for layer_key in set(
            list(layer_to_samples_k.keys())
            + list(layer_to_samples_v.keys())
            + (list(layer_to_samples_q.keys()) if include_q else [])
        ):
            entry = {}
            if include_q:
                entry["q_amax_p"] = _percentile(
                    layer_to_samples_q.get(layer_key, []), percentile
                )
            entry["k_amax_p"] = _percentile(
                layer_to_samples_k.get(layer_key, []), percentile
            )
            entry["v_amax_p"] = _percentile(
                layer_to_samples_v.get(layer_key, []), percentile
            )
            local_layer_to_pamax[layer_key] = entry

        # Merge across all ranks: take maximum of percentile amax (conservative approach)
        world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )
        gathered = [None for _ in range(world_size)] if world_size > 1 else None
        if world_size > 1:
            torch.distributed.all_gather_object(gathered, local_layer_to_pamax)
            merged = defaultdict(dict)
            for d in gathered:  # type: ignore
                if d is None:
                    continue
                for k, v in d.items():
                    dst = merged[k]
                    for kk, vv in v.items():
                        dst[kk] = max(dst.get(kk, 0.0), float(vv))
            layer_to_pamax = dict(merged)
        else:
            layer_to_pamax = local_layer_to_pamax

        # Compute scale (symmetric quantization): scale = pamax / fp8_max
        result_layers = {}
        for layer_key, vals in layer_to_pamax.items():
            out_entry = {}
            if include_q:
                q_scale = (vals.get("q_amax_p", 0.0) * margin) / FP8_MAX_Q
                out_entry["q_scale"] = float(q_scale)
            k_scale = (vals.get("k_amax_p", 0.0) * margin) / FP8_MAX_K
            v_scale = (vals.get("v_amax_p", 0.0) * margin) / FP8_MAX_V
            out_entry["k_scale"] = float(k_scale)
            out_entry["v_scale"] = float(v_scale)
            result_layers[layer_key] = out_entry

        vllm_format_scales = convert_calibration_to_vllm_format(result_layers)

        final_result = {
            "format": "fp8",
            "percentile": percentile,
            "margin": margin,
            "layers": vllm_format_scales,
        }

        # Sync results across all ranks (broadcast rank0's result)
        if world_size > 1:
            if torch.distributed.get_rank() == 0:
                obj_list = [final_result]
                torch.distributed.broadcast_object_list(obj_list, src=0)
                final_result = obj_list[0]
            else:
                obj_list = [None]
                torch.distributed.broadcast_object_list(obj_list, src=0)
                final_result = obj_list[0]  # type: ignore

        return final_result


class CustomFloat16Module(Float16Module):
    """Float 16 Module.

    Attributes:
        config (TransformerConfig): Transformer config
        fp16 (bool) : Specifies if the model runs in fp16 mode
        bf16 (bool) : Specifies if the model runs in bf16 mode

    Args:
        config (TransformerConfig): The transformer config used to initalize the model
    """

    def __init__(self, config: TransformerConfig, module: torch.nn.Module):
        super(CustomFloat16Module, self).__init__(config, module)
        self.re_enable_float32_expert_bias()

    def re_enable_float32_expert_bias(self) -> None:
        """Ensure MoE router expert bias stays in float32 for numerical stability.

        Walks the wrapped module to find MoE routers and invokes the
        `_maintain_float32_expert_bias()` helper which recreates or casts the
        expert bias tensors to float32 as required by Megatron-LM.
        """
        module = self.module
        # Handle VLM models where language model is nested
        if hasattr(module, "language_model"):
            module = module.language_model
        if hasattr(module, "decoder") and hasattr(module.decoder, "layers"):
            for layer in module.decoder.layers:
                mlp = getattr(layer, "mlp", None)
                router = getattr(mlp, "router", None) if mlp is not None else None
                if router is not None and hasattr(
                    router, "_maintain_float32_expert_bias"
                ):
                    router._maintain_float32_expert_bias()

```

```python
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
import gc
import os
import re
import time
import warnings
from collections import defaultdict
from contextlib import AbstractContextManager, contextmanager, nullcontext
from functools import partial
from typing import Any, Iterator, Optional, TypeVar, cast

import ray
import torch
from megatron.bridge import AutoBridge
from megatron.bridge.models.model_provider import get_model
from megatron.bridge.training import fault_tolerance
from megatron.bridge.training.checkpointing import (
    checkpoint_exists,
    init_checkpointing_context,
    load_checkpoint,
    maybe_finalize_async_save,
    save_checkpoint,
)
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    LoggerConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainingConfig,
)
from megatron.bridge.training.initialize import (
    initialize_megatron,
    set_jit_fusion_options,
)
from megatron.bridge.training.optim import setup_optimizer
from megatron.bridge.training.setup import (
    _update_model_config_funcs,
)
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer
from megatron.bridge.training.utils.train_utils import (
    logical_and_across_model_parallel_group,
    reduce_max_stat_across_model_parallel_group,
)
from megatron.bridge.utils.common_utils import get_rank_safe
from megatron.bridge.utils.instantiate_utils import InstantiationMode
from megatron.bridge.utils.vocab_utils import calculate_padded_vocab_size
from megatron.core import parallel_state
from megatron.core.transformer.utils import toggle_cuda_graphs
from megatron.core.distributed import DistributedDataParallel
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import (
    FullyShardedDataParallel as custom_FSDP,
)
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import (
    InferenceWrapperConfig,
)
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.models.gpt import GPTModel
from megatron.core.optimizer import ChainedOptimizer
from megatron.core.parallel_state import (
    get_context_parallel_group,
    get_context_parallel_rank,
    get_pipeline_model_parallel_group,
    get_pipeline_model_parallel_last_rank,
    get_pipeline_model_parallel_world_size,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    is_pipeline_last_stage,
)
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.utils import get_ltor_masks_and_position_ids
from ray.util.queue import Queue
from transformers import PreTrainedTokenizerBase

from nemo_rl.algorithms.interfaces import LossFunction, LossType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.model_utils import (
    allgather_cp_sharded_tensor,
    distributed_vocab_topk,
    from_parallel_logits_to_logprobs,
    from_parallel_logits_to_logprobs_packed_sequences,
)
from nemo_rl.distributed.named_sharding import NamedSharding
from nemo_rl.models.generation.fp8 import (
    convert_calibration_to_vllm_format,
    get_vllm_qkv_scale_names,
)
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationOutputSpec,
    verify_right_padding,
)
from nemo_rl.models.generation.vllm.config import VllmConfig
from nemo_rl.models.megatron.common import (
    _get_pack_sequence_parameters_for_megatron,
    _pack_sequences_for_megatron,
    broadcast_tensor,
    forward_step_arbitrary_loss,
    get_moe_metrics,
)
from nemo_rl.models.megatron.community_import import import_model_from_hf_name
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.interfaces import (
    ColocatablePolicyInterface,
    LogprobOutputSpec,
)
from nemo_rl.models.policy.utils import (
    configure_dynamo_cache,
    get_megatron_checkpoint_dir,
    get_runtime_env_for_policy_worker,
)
from nemo_rl.models.policy.workers.base_policy_worker import AbstractPolicyWorker
from nemo_rl.models.policy.workers.patches import apply_transformer_engine_patch
from nemo_rl.utils.nsys import wrap_with_nvtx_name
from nemo_rl.utils.packed_tensor import packed_broadcast_producer

try:
    from megatron.core.distributed import (
        TorchFullyShardedDataParallel as torch_FSDP,  # noqa: F401 unused-import
    )

    HAVE_FSDP2 = True
except ImportError:
    HAVE_FSDP2 = False

TokenizerType = TypeVar("TokenizerType", bound=PreTrainedTokenizerBase)


def broadcast_object_across_pp_ranks(obj):
    """Broadcast an object across pipeline parallel ranks.

    This utility function handles broadcasting an object from the rank that owns it
    to all other pipeline parallel ranks. If only one rank has the object (non-None),
    it will be broadcast to all other ranks.

    Args:
        obj: The object to broadcast. Can be None on ranks that don't own it.

    Returns:
        The object on all ranks (either the original or the broadcast copy).

    Raises:
        ValueError: If the object doesn't exist on any pipeline parallel rank.
    """
    pp_size = get_pipeline_model_parallel_world_size()
    pp_group = get_pipeline_model_parallel_group()

    if pp_size == 1:
        return obj

    # ------------------------------------------------------------------
    # 1. Gather presence flags from all PP ranks to find the source rank
    # ------------------------------------------------------------------
    has_obj = obj is not None
    obj_flags = [None] * pp_size
    torch.distributed.all_gather_object(obj_flags, has_obj, group=pp_group)

    # ------------------------------------------------------------------
    # 2. Identify the owning rank (the only rank with True flag)
    # ------------------------------------------------------------------
    src_rank = None  # Rank *inside* the PP group
    for rank, flag in enumerate(obj_flags):
        if flag:
            src_rank = rank
            break

    if src_rank is None:
        raise ValueError("Object must exist on at least one PP rank")

    # ------------------------------------------------------------------
    # 3. Broadcast the object from the source rank to all ranks
    # ------------------------------------------------------------------
    # Use broadcast_object_list which is more robust than all_gather_object
    obj_list = [obj]
    pp_ranks = torch.distributed.get_process_group_ranks(pp_group)
    global_src = pp_ranks[src_rank]
    torch.distributed.broadcast_object_list(obj_list, src=global_src, group=pp_group)

    return obj_list[0]


def setup_megatron_model(
    policy_cfg: PolicyConfig,
    cfg: ConfigContainer,
    load_optimizer: bool = True,
    get_embedding_ranks=None,  # TODO @sahilj: What is this?
    get_position_embedding_ranks=None,
):
    state = GlobalState()
    state.cfg = cfg
    # TODO: Freeze state.cfg

    cfg.dist.external_gpu_device_mapping = True
    initialize_megatron(
        cfg=cfg,
        get_embedding_ranks=get_embedding_ranks,
        get_position_embedding_ranks=get_position_embedding_ranks,
    )

    if cfg.ft and cfg.ft.enable_ft_package:
        fault_tolerance.setup(cfg, state)
        fault_tolerance.maybe_setup_simulated_fault(cfg.ft)

    # Set pytorch JIT layer fusion options and warmup JIT functions.
    set_jit_fusion_options(cfg.model, cfg.train.micro_batch_size)

    # Adjust the startup time so it reflects the largest value.
    # This will be closer to what scheduler will see (outside of
    # image ... launches.
    start_time_tensor = torch.tensor(
        [state.start_time], dtype=torch.double, device="cuda"
    )
    torch.distributed.all_reduce(start_time_tensor, op=torch.distributed.ReduceOp.MIN)
    state.start_time = start_time_tensor.item()

    print(
        "time to initialize megatron (seconds): {:.3f}".format(
            time.time() - state.start_time
        )
    )
    torch.distributed.barrier()

    # Context used for persisting some state between checkpoint saves.
    checkpointing_context = init_checkpointing_context(cfg.checkpoint)

    # Tokenizer
    build_tokenizer(
        cfg.tokenizer,
        make_vocab_size_divisible_by=cfg.model.make_vocab_size_divisible_by
        // cfg.model.tensor_model_parallel_size,
        tensor_model_parallel_size=cfg.model.tensor_model_parallel_size,
        trust_remote_code=True,
    )
    assert cfg.model.vocab_size, "vocab size must be specified in model config"

    torch.distributed.barrier()

    pre_wrap_hook = []
    mixed_precision_wrapper = Float16Module
    if policy_cfg["megatron_cfg"]["freeze_moe_router"]:

        def freeze_moe_router(megatron_model):
            if not isinstance(megatron_model, list):
                megatron_model = [megatron_model]
            for model_module in megatron_model:
                # Handle both wrapped (Float16Module) and unwrapped models
                if isinstance(model_module, Float16Module):
                    model_module = model_module.module
                # Handle VLM models
                if hasattr(model_module, "language_model"):
                    model_module = model_module.language_model
                for layer in model_module.decoder.layers:
                    if hasattr(layer, "mlp") and hasattr(layer.mlp, "router"):
                        layer.mlp.router.weight.requires_grad = False

        mixed_precision_wrapper = CustomFloat16Module
        pre_wrap_hook.extend([freeze_moe_router])

    # Model, optimizer, and learning rate.
    model = get_model(
        cfg.model,
        cfg.ddp,
        use_torch_fsdp2=cfg.dist.use_torch_fsdp2,
        overlap_param_gather_with_optimizer_step=cfg.optimizer.overlap_param_gather_with_optimizer_step,
        data_parallel_random_init=cfg.rng.data_parallel_random_init,
        pre_wrap_hook=pre_wrap_hook,
        mixed_precision_wrapper=mixed_precision_wrapper,
    )
    if load_optimizer:
        optimizer, scheduler = setup_optimizer(
            optimizer_config=cfg.optimizer,
            scheduler_config=cfg.scheduler,
            model=model,
            use_gloo_process_groups=cfg.dist.use_gloo_process_groups,
        )
    else:
        optimizer = None
        scheduler = None

    print("Model, optimizer, and learning rate scheduler built")
    torch.distributed.barrier()

    # Load checkpoint if applicable
    if (
        cfg.checkpoint.load is not None
        or cfg.checkpoint.pretrained_checkpoint is not None
    ) and (
        checkpoint_exists(cfg.checkpoint.load)
        or checkpoint_exists(cfg.checkpoint.pretrained_checkpoint)
    ):
        load_checkpoint(
            state,
            model,
            optimizer,
            scheduler,
            checkpointing_context=checkpointing_context,
            skip_load_to_model_and_opt=HAVE_FSDP2 and cfg.dist.use_torch_fsdp2,
        )
        print("Checkpoint loaded")
    torch.distributed.barrier()

    return state, model, optimizer, scheduler, checkpointing_context


def destroy_parallel_state():
    """Safely destroy parallel state and reset async call tracking.

    This function is called during initialization to clean up temporary distributed
    state from model import operations. Resetting async call tracking ensures that
    when the main Megatron distributed context is created, all ranks start with
    consistent call_idx values for async checkpointing.
    """
    if torch.distributed.is_initialized():
        try:
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()
        except:
            pass  # Ignore errors if already destroyed
    if hasattr(parallel_state, "destroy_model_parallel"):
        try:
            parallel_state.destroy_model_parallel()
        except:
            pass  # Ignore errors if already destroyed

    # Reset async calls queue to prevent call_idx mismatches after distributed context recreation
    try:
        import nemo.tron.utils.async_utils as nemo_async_utils
        from megatron.core.dist_checkpointing.strategies.async_utils import (
            AsyncCallsQueue,
        )

        # Clean up any existing async callers first
        old_call_idx = getattr(nemo_async_utils._async_calls_queue, "call_idx", None)
        num_unfinalized = (
            nemo_async_utils._async_calls_queue.get_num_unfinalized_calls()
        )
        if num_unfinalized > 0:
            print(
                f"[WARNING] Resetting async calls queue with {num_unfinalized} unfinalized calls"
            )
        try:
            nemo_async_utils._async_calls_queue.close()
        except:
            pass  # Ignore errors during cleanup
        # Reset the global async calls queue by creating a new instance
        nemo_async_utils._async_calls_queue = AsyncCallsQueue()
        print(f"[DEBUG] Reset NeMo async calls queue (old call_idx: {old_call_idx})")
    except ImportError:
        pass

    # Also reset the Megatron async calls queue if it exists
    try:
        import megatron.training.async_utils as megatron_async_utils
        from megatron.core.dist_checkpointing.strategies.async_utils import (
            AsyncCallsQueue,
        )

        # Clean up any existing async callers first
        old_call_idx = getattr(
            megatron_async_utils._async_calls_queue, "call_idx", None
        )
        num_unfinalized = (
            megatron_async_utils._async_calls_queue.get_num_unfinalized_calls()
        )
        if num_unfinalized > 0:
            print(
                f"[WARNING] Resetting Megatron async calls queue with {num_unfinalized} unfinalized calls"
            )
        try:
            megatron_async_utils._async_calls_queue.close()
        except:
            pass  # Ignore errors during cleanup
        # Reset the Megatron global async calls queue as well
        megatron_async_utils._async_calls_queue = AsyncCallsQueue()
        print(
            f"[DEBUG] Reset Megatron async calls queue (old call_idx: {old_call_idx})"
        )
    except ImportError:
        pass

    # Reset the third global async_calls instance in base strategy module
    try:
        import megatron.core.dist_checkpointing.strategies.base as base_strategy
        from megatron.core.dist_checkpointing.strategies.async_utils import (
            AsyncCallsQueue,
        )

        # Clean up and reset the global async_calls in base strategy
        old_call_idx = getattr(base_strategy.async_calls, "call_idx", None)
        num_unfinalized = base_strategy.async_calls.get_num_unfinalized_calls()
        if num_unfinalized > 0:
            print(
                f"[WARNING] Resetting base strategy async_calls with {num_unfinalized} unfinalized calls"
            )
        try:
            base_strategy.async_calls.close()
        except:
            pass
        base_strategy.async_calls = AsyncCallsQueue()
        print(f"[DEBUG] Reset base strategy async_calls (old call_idx: {old_call_idx})")
    except ImportError:
        pass


@ray.remote(
    runtime_env=get_runtime_env_for_policy_worker("megatron_policy_worker")
)  # pragma: no cover
class MegatronPolicyWorker(AbstractPolicyWorker, ColocatablePolicyInterface):
    def __repr__(self):
        """Customizes the actor's prefix in the Ray logs.

        This makes it easier to identify which worker is producing specific log messages.
        """
        if torch.distributed.is_initialized():
            return f"{self.__class__.__qualname__}[rank={torch.distributed.get_rank()}]"
        else:
            return f"{self.__class__.__qualname__}"

    def __init__(
        self,
        config: PolicyConfig,
        tokenizer: TokenizerType,
        weights_path: Optional[str] = None,
        optimizer_path: Optional[str] = None,
        init_optimizer: bool = True,
        init_reference_model: bool = True,
        *,
        worker_sharding_annotations: NamedSharding,
        pre_init_communication_queue: Queue,
        **kwargs: Any,
    ):
        apply_transformer_engine_patch()

        self.is_generation_colocated = None
        if "generation" in config and config["generation"] is not None:
            self.is_generation_colocated = config["generation"]["colocated"]["enabled"]

        # Explicitly set NCCL_CUMEM_ENABLE to 1 to avoid the P2P initialization error for PyNCCLCommunicator.
        # See https://github.com/NVIDIA-NeMo/RL/issues/564 for more details.
        if not self.is_generation_colocated:
            os.environ["NCCL_CUMEM_ENABLE"] = "1"

        self.cfg = config
        dtype_map = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }
        self.dtype = dtype_map[self.cfg["precision"]]

        self.optimizer_cpu_offload = self.cfg["megatron_cfg"]["optimizer"][
            "optimizer_cpu_offload"
        ]
        self.offload_optimizer_for_logprob = self.cfg["offload_optimizer_for_logprob"]

        # Reward models are not yet supported with Megatron.
        if "reward_model_cfg" in self.cfg and self.cfg["reward_model_cfg"]["enabled"]:
            raise NotImplementedError(
                "Reward models are not yet supported with the Megatron backend, this issue is "
                "tracked in https://github.com/NVIDIA-NeMo/RL/issues/720"
            )

        # Disable dynamo autotune_local_cache to avoid crash when there's already a cache
        # with different order of node_bundles
        configure_dynamo_cache()

        # cfg["model_name"] is allowed to be either an HF model name or a path to an HF checkpoint
        # check if hf_model_name is a path
        hf_model_name = self.cfg["model_name"]
        # Check if the checkpoint already exists
        hf_model_subdir = hf_model_name
        if os.path.exists(hf_model_name):
            hf_model_subdir = f"model_{hf_model_subdir.replace('/', '_')}"

        pretrained_path = f"{get_megatron_checkpoint_dir()}/{hf_model_subdir}"
        pt_checkpoint_exists = os.path.exists(pretrained_path) and os.path.exists(
            os.path.join(pretrained_path, "iter_0000000")
        )

        # Ensure clean slate before import
        destroy_parallel_state()

        # Set for rank for non-collocated to check which ranks to broadcast from
        self.rank = get_rank_safe()
        # Need to initialize the process group before calling into Megatron-Bridge, otherwise Megatron-Bridge will try to set an incorrect device
        torch.distributed.init_process_group("nccl")
        if pt_checkpoint_exists:
            print(f"Checkpoint already exists at {pretrained_path}. Skipping import.")
        else:
            hf_config_overrides = self.cfg.get("hf_config_overrides", {}) or {}
            import_model_from_hf_name(
                hf_model_name,
                pretrained_path,
                self.cfg["megatron_cfg"],
                **hf_config_overrides,
            )

            if parallel_state.model_parallel_is_initialized():
                print("Reinitializing model parallel after loading model state.")
                parallel_state.destroy_model_parallel()

        pretrained_run_config = os.path.join(
            pretrained_path, "iter_0000000/run_config.yaml"
        )

        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if not os.path.exists(pretrained_run_config):
            raise FileNotFoundError(
                f"Pretrained run config not found at {pretrained_run_config} on rank={get_rank_safe()}. This usually means that the one-time HF->mcore conversion on rank=0 saved to a directory not being mounted on this node. Please check "
            )

        try:
            cfg_from_pretrained = ConfigContainer.from_yaml(
                pretrained_run_config, mode=InstantiationMode.STRICT
            )
        except Exception as e:
            # Add helpful context as a note to the exception
            e.add_note(
                f"\n{'=' * 80}\n"
                f"NOTE: A common cause of this error is when the HF->mcore converted checkpoint is\n"
                f"created with an older version of megatron-bridge.\n"
                f"If this checkpoint is old or was generated by a different code version,\n"
                f"try deleting it and rerunning the code.\n"
                f"The checkpoint will be automatically regenerated with the current version.\n\n"
                f"Checkpoint location: {pretrained_path}\n"
                f"{'=' * 80}"
            )
            raise
        model_cfg = cfg_from_pretrained.model

        cfg_from_pretrained.logger = LoggerConfig()

        model_cfg.tensor_model_parallel_size = self.cfg["megatron_cfg"][
            "tensor_model_parallel_size"
        ]
        model_cfg.pipeline_model_parallel_size = self.cfg["megatron_cfg"][
            "pipeline_model_parallel_size"
        ]
        model_cfg.num_layers_in_first_pipeline_stage = self.cfg["megatron_cfg"][
            "num_layers_in_first_pipeline_stage"
        ]
        model_cfg.num_layers_in_last_pipeline_stage = self.cfg["megatron_cfg"][
            "num_layers_in_last_pipeline_stage"
        ]
        model_cfg.sequence_parallel = self.cfg["megatron_cfg"]["sequence_parallel"]
        model_cfg.context_parallel_size = self.cfg["megatron_cfg"][
            "context_parallel_size"
        ]
        if model_cfg.context_parallel_size > 1:
            assert self.cfg["sequence_packing"]["enabled"], (
                "Sequence Packing must be enabled to use Context Parallelism with MCore"
            )
        model_cfg.expert_tensor_parallel_size = self.cfg["megatron_cfg"][
            "expert_tensor_parallel_size"
        ]
        model_cfg.expert_model_parallel_size = self.cfg["megatron_cfg"][
            "expert_model_parallel_size"
        ]

        # Setting moe_router_dtype to higher precision (e.g. fp64) can improve numerical stability,
        # especially when using many experts.
        model_cfg.moe_router_dtype = self.cfg["megatron_cfg"]["moe_router_dtype"]

        # The below two configs (and "freeze_moe_router") are used to stabilize moe training
        # by preventing updates to the moe router. We found that this is helpful in reducing
        # logprob error during training.

        # Set this to "none" to disable load balancing loss.
        model_cfg.moe_router_load_balancing_type = self.cfg["megatron_cfg"][
            "moe_router_load_balancing_type"
        ]
        # Set this to 0.0 to disable updates to the moe router expert bias
        model_cfg.moe_router_bias_update_rate = self.cfg["megatron_cfg"][
            "moe_router_bias_update_rate"
        ]

        model_cfg.moe_permute_fusion = self.cfg["megatron_cfg"]["moe_permute_fusion"]
        if "layernorm_epsilon" in self.cfg["megatron_cfg"]:
            model_cfg.layernorm_epsilon = self.cfg["megatron_cfg"]["layernorm_epsilon"]

        model_cfg.sequence_parallel = self.cfg["megatron_cfg"]["sequence_parallel"]
        model_cfg.bf16 = self.dtype == torch.bfloat16
        model_cfg.fp16 = self.dtype == torch.float16
        if model_cfg.fp16:
            assert not model_cfg.bf16, "fp16 and bf16 cannot be used together"
            model_cfg.params_dtype = torch.float16
        elif model_cfg.bf16:
            assert not model_cfg.fp16, "fp16 and bf16 cannot be used together"
            model_cfg.params_dtype = torch.bfloat16
        else:
            model_cfg.params_dtype = torch.float32
        model_cfg.pipeline_dtype = dtype_map[self.cfg["megatron_cfg"]["pipeline_dtype"]]
        model_cfg.parallel_output = True
        if self.cfg["megatron_cfg"]["activation_checkpointing"]:
            model_cfg.recompute_granularity = "full"
            model_cfg.recompute_method = "uniform"
            model_cfg.recompute_num_layers = 1
        if not model_cfg.gated_linear_unit:
            assert model_cfg.activation_func is not None, (
                "activation_func must be set if not using gated_linear_unit. This likely "
                "indicates an issue in configuration conversion (e.g. activation func was "
                "a lambda and couldn't be serialized). This is based on this check "
                "https://github.com/NVIDIA/Megatron-LM/blob/1ab876ddc4c1893c76f26d775226a8d1dcdfb3d2/megatron/core/transformer/mlp.py#L174."
            )
        model_cfg.apply_rope_fusion = self.cfg["megatron_cfg"]["apply_rope_fusion"]
        model_cfg.bias_activation_fusion = self.cfg["megatron_cfg"][
            "bias_activation_fusion"
        ]
        fp8_cfg = self.cfg["megatron_cfg"].get("fp8_cfg", None)
        self.fp8_cfg = fp8_cfg
        if fp8_cfg is not None and fp8_cfg.get("enabled", False):
            try:
                model_cfg.fp8 = fp8_cfg["fp8"]
                model_cfg.fp8_recipe = fp8_cfg["fp8_recipe"]
                model_cfg.fp8_param = fp8_cfg["fp8_param"]
            except KeyError as e:
                raise KeyError(f"Missing key in fp8_cfg: {e}")
            if model_cfg.fp8_param:
                warnings.warn(
                    "Setting fp8_param=True sometimes causes NaN token_mult_prob_error, please use with caution. "
                    "Refer to https://github.com/NVIDIA-NeMo/RL/issues/1164 for latest updates with this issue."
                )

        optimizer_cpu_offload = self.cfg["megatron_cfg"]["optimizer"][
            "optimizer_cpu_offload"
        ]
        optimizer_offload_fraction = self.cfg["megatron_cfg"]["optimizer"][
            "optimizer_offload_fraction"
        ]
        if optimizer_cpu_offload:
            # Currently, hybrid optimizer (partly on GPU and partly on CPU) is not supported because it conflicts with the way
            # Nemo-rl handles the optimizer offload/onload between generation and training. So if using CPU optimizer the offload_fraction should be 1.0.
            assert optimizer_offload_fraction == 1.0, (
                "Currently for optimizer offloading, only optimizer_offload_fraction=1.0 is supported"
            )
        if (
            "logprob_chunk_size" in self.cfg
            and self.cfg["logprob_chunk_size"] is not None
            and self.cfg["logprob_chunk_size"] > 0
        ):
            assert self.cfg["megatron_cfg"]["defer_fp32_logits"], (
                "defer_fp32_logits must be True if logprob_chunk_size is set"
            )
        self.defer_fp32_logits = self.cfg["megatron_cfg"].get(
            "defer_fp32_logits", None
        ) and (model_cfg.fp16 or model_cfg.bf16)

        checkpoint_config = CheckpointConfig(
            save_interval=100,
            save=weights_path,
            load=weights_path,
            pretrained_checkpoint=pretrained_path,  # This is the path to the pretrained ckpt for the SFT case
            async_save=False,  # This doesn't work right now.
            fully_parallel_save=True,
            fully_parallel_load=True,  # Enable fully parallel load
            load_rng=False,
        )
        ref_checkpoint_config = CheckpointConfig(
            pretrained_checkpoint=pretrained_path,  # This is the path to the pretrained ckpt for the SFT case
            save=None,
            load=None,
            fully_parallel_load=True,  # Enable fully parallel load
            load_rng=False,
        )

        assert "train_iters" in self.cfg["megatron_cfg"], (
            "train_iters must be set in megatron_cfg. For an example, see "
            "https://github.com/NVIDIA-NeMo/RL/blob/bccbc377705a81a1f4b3c31ad9767bcc15f735a8/nemo_rl/algorithms/sft.py#L175-L179."
        )

        ## These settings are required for correct gradient computations in mcore
        ## when calculate_per_token_loss is True, there is no scaling of the gradient in mcore,
        ## so we handle the scaling in nemo-rl.
        ## perform_initialization = True is a workaround to ensure the correct tensor parallel attributes are set
        ## on the TP-sharded parameters.
        model_cfg.calculate_per_token_loss = True
        model_cfg.perform_initialization = True
        # SHAN CHECK:
        model_cfg.cuda_graph_impl = "local"
        model_cfg.cuda_graph_scope = None
        model_cfg.use_te_rng_tracker = True

        assert (
            "aux_loss" not in model_cfg.moe_router_load_balancing_type
            or model_cfg.moe_aux_loss_coeff == 0
        ), (
            "MoE aux loss is currently not supported due to a known bug in Megatron-LM. "
            "See https://github.com/NVIDIA/Megatron-LM/issues/1984 for more details."
        )

        self.megatron_cfg = ConfigContainer(
            model=model_cfg,
            checkpoint=checkpoint_config,
            logger=LoggerConfig(logging_level=0),
            train=TrainingConfig(
                micro_batch_size=1,  # ignored
                global_batch_size=self.cfg["train_global_batch_size"],  # ignored
                train_iters=self.cfg["megatron_cfg"][
                    "train_iters"
                ],  # Set by algorithm setup
            ),
            optimizer=OptimizerConfig(
                **self.cfg["megatron_cfg"]["optimizer"],
            ),
            ddp=DistributedDataParallelConfig(
                check_for_nan_in_grad=True,
                grad_reduce_in_fp32=self.cfg["megatron_cfg"][
                    "distributed_data_parallel_config"
                ]["grad_reduce_in_fp32"],
                overlap_grad_reduce=self.cfg["megatron_cfg"][
                    "distributed_data_parallel_config"
                ]["overlap_grad_reduce"],
                overlap_param_gather=self.cfg["megatron_cfg"][
                    "distributed_data_parallel_config"
                ]["overlap_param_gather"],
                # we need to set average_in_collective=False with calculate_per_token_loss=True.
                # otherwise, mcore throws an assertion error.
                average_in_collective=False,
                use_distributed_optimizer=self.cfg["megatron_cfg"]["optimizer"][
                    "use_distributed_optimizer"
                ],
                data_parallel_sharding_strategy=self.cfg["megatron_cfg"][
                    "distributed_data_parallel_config"
                ]["data_parallel_sharding_strategy"],
            ),
            scheduler=SchedulerConfig(
                **self.cfg["megatron_cfg"]["scheduler"],
            ),
            dataset=None,
            tokenizer=TokenizerConfig(
                tokenizer_type="HuggingFaceTokenizer",
                tokenizer_model=hf_model_name,
            ),
        )
        # TODO: this validation should happen inside mbridge: https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/1665
        if self.dtype == torch.bfloat16:
            assert self.megatron_cfg.model.bf16 == True, (
                "policy.megatron_cfg.model.bf16=True must be set if policy.precision=bfloat16. This is handled by nemo-rl so this indicates something is misconfigured."
            )
            assert (
                self.megatron_cfg.optimizer.use_precision_aware_optimizer == False
                or self.megatron_cfg.optimizer.bf16 == True
            ), (
                "policy.megatron_cfg.optimizer.bf16=True must be set if policy.precision=bfloat16 when using use_precision_aware_optimizer=True"
            )
        elif self.dtype == torch.float16:
            assert self.megatron_cfg.model.fp16 == True, (
                "policy.megatron_cfg.model.fp16=True must be set if policy.precision=float16. This is handled by nemo-rl so this indicates something is misconfigured."
            )
            assert (
                self.megatron_cfg.optimizer.use_precision_aware_optimizer == False
                or self.megatron_cfg.optimizer.fp16 == True
            ), (
                "policy.megatron_cfg.optimizer.fp16=True must be set if policy.precision=float16 when using use_precision_aware_optimizer=True"
            )
        elif self.dtype == torch.float32:
            assert (
                self.megatron_cfg.model.bf16 == False
                and self.megatron_cfg.model.fp16 == False
            ), (
                "policy.megatron_cfg.model.bf16=False and policy.megatron_cfg.model.fp16=False must be set if policy.precision=float32. This is handled by nemo-rl so this indicates something is misconfigured."
            )
            assert (
                self.megatron_cfg.optimizer.bf16 == False
                and self.megatron_cfg.optimizer.fp16 == False
            ), (
                "policy.megatron_cfg.optimizer.bf16=False and policy.megatron_cfg.optimizer.fp16=False must be set if policy.precision=float32"
            )
        self.megatron_cfg.validate()
        (
            self.mcore_state,
            self.model,
            self.optimizer,
            self.scheduler,
            self.checkpointing_context,
        ) = setup_megatron_model(
            policy_cfg=self.cfg, cfg=self.megatron_cfg, load_optimizer=init_optimizer
        )

        # Set the param sync function for the model
        if (
            self.megatron_cfg.ddp.overlap_param_gather
            and self.megatron_cfg.ddp.align_param_gather
        ):
            self.megatron_cfg.param_sync_func = [
                model_chunk.start_param_sync for model_chunk in self.model
            ]
            if len(self.model) == 1:
                self.megatron_cfg.param_sync_func = self.megatron_cfg.param_sync_func[0]

        self.model = self.model[0]  # Get the first model from the list

        if init_reference_model:
            self.model = self.move_model(self.model, "cpu")
            ref_ckpt_context = init_checkpointing_context(ref_checkpoint_config)

            # Create a separate megatron config for the reference model with the correct checkpoint config
            self.megatron_cfg.model.cuda_graph_impl = "none"
            self.megatron_cfg.model.use_te_rng_tracker = True
            ref_megatron_cfg = ConfigContainer(
                model=self.megatron_cfg.model,
                checkpoint=ref_checkpoint_config,  # Use the reference checkpoint config
                logger=self.megatron_cfg.logger,
                train=self.megatron_cfg.train,
                optimizer=self.megatron_cfg.optimizer,
                ddp=self.megatron_cfg.ddp,
                scheduler=self.megatron_cfg.scheduler,
                dataset=self.megatron_cfg.dataset,
                tokenizer=self.megatron_cfg.tokenizer,
            )

            # Create a separate state object for the reference model
            ref_state = GlobalState()
            ref_state.cfg = ref_megatron_cfg

            # Configure mixed precision wrapper for reference model
            ref_mixed_precision_wrapper = Float16Module
            if self.cfg["megatron_cfg"].get("freeze_moe_router", False):
                ref_mixed_precision_wrapper = CustomFloat16Module

            reference_model = get_model(
                self.megatron_cfg.model,
                self.megatron_cfg.ddp,
                use_torch_fsdp2=self.megatron_cfg.dist.use_torch_fsdp2,
                overlap_param_gather_with_optimizer_step=self.megatron_cfg.optimizer.overlap_param_gather_with_optimizer_step,
                pre_wrap_hook=self.megatron_cfg.rng.data_parallel_random_init,
                mixed_precision_wrapper=ref_mixed_precision_wrapper,
            )
            print("Loading the Reference Model")
            if (
                ref_checkpoint_config.pretrained_checkpoint is not None
                and checkpoint_exists(ref_checkpoint_config.pretrained_checkpoint)
            ):
                load_checkpoint(
                    ref_state,  # Use the separate state object with ref checkpoint config
                    reference_model,
                    None,  # no optimizer
                    None,  # no scheduler
                    checkpointing_context=ref_ckpt_context,
                    skip_load_to_model_and_opt=HAVE_FSDP2
                    and self.megatron_cfg.dist.use_torch_fsdp2,
                )
                reference_model = reference_model[0]
                reference_model.eval()
                self.reference_state_dict = {}
                for name, item in reference_model.state_dict().items():
                    if isinstance(item, torch.Tensor):
                        cpu_item = item.detach().to(
                            device="cpu", non_blocking=True, copy=True
                        )
                        del item
                    else:
                        cpu_item = item
                    self.reference_state_dict[name] = cpu_item
                print("Reference model loaded")
            else:
                print("Reference model not loaded")

            self.model = self.move_model(self.model, "cuda")

        _update_model_config_funcs(
            [self.model],
            self.megatron_cfg.model,
            self.megatron_cfg.ddp,
            self.optimizer,
            align_grad_reduce=self.megatron_cfg.dist.align_grad_reduce,
        )

        tokenizer_config = TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model=hf_model_name,
        )

        self.megatron_tokenizer = build_tokenizer(
            tokenizer_config,
            make_vocab_size_divisible_by=self.megatron_cfg.model.make_vocab_size_divisible_by
            // self.cfg["megatron_cfg"]["tensor_model_parallel_size"],
            tensor_model_parallel_size=self.cfg["megatron_cfg"][
                "tensor_model_parallel_size"
            ],
            trust_remote_code=True,
        )
        self.final_padded_vocab_size = calculate_padded_vocab_size(
            self.megatron_cfg.model.vocab_size,
            self.megatron_cfg.model.make_vocab_size_divisible_by,
            self.cfg["megatron_cfg"]["tensor_model_parallel_size"],
        )
        self.dp_size = worker_sharding_annotations.get_axis_size("data_parallel")
        self.megatron_bridge = AutoBridge.from_hf_pretrained(
            hf_model_name, trust_remote_code=True
        )

        self.should_disable_forward_pre_hook = (
            self.cfg["megatron_cfg"]["optimizer"]["use_distributed_optimizer"]
            and self.cfg["megatron_cfg"]["distributed_data_parallel_config"][
                "overlap_param_gather"
            ]
        )

        # vars used for refit
        ## will be initialized in prepare_refit_info
        # refit_param_info_mcore combines the conversion tasks with the param memory
        # [(mcore_param_name, estimated_memory), ...]
        # Note: here param name is local param name, with local layer number and
        # local expert id etc.
        self.refit_conversion_tasks = (
            None  # Meta data for conversion params from megatron bridge
        )
        self.refit_conversion_tasks_current_index = None
        self.refit_param_info_mcore = None

        ## used for streaming update inference engine weights
        self._held_gather_buffer = None

    def enable_forward_pre_hook(self):
        assert isinstance(self.model, DistributedDataParallel)
        self.model.enable_forward_pre_hook()

    def disable_forward_pre_hook(self, param_sync=True):
        assert isinstance(self.model, DistributedDataParallel)
        self.model.disable_forward_pre_hook(param_sync=param_sync)

    @wrap_with_nvtx_name("megatron_policy_worker/train")
    def train(
        self,
        data: BatchedDataDict,
        loss_fn: LossFunction,
        eval_mode: bool = False,
        gbs: Optional[int] = None,
        mbs: Optional[int] = None,
    ) -> dict[str, Any]:
        """Train the policy on a batch of data with a given loss function."""
        # Disable CUDA graphs for training to avoid conflicts with inference tensors.
        # This prevents "Inplace update to inference tensor outside InferenceMode" errors
        # when switching from generation (which uses CUDA graphs) to training.
        # Use toggle_cuda_graphs on the underlying module (not the DDP wrapper) to properly
        # reset CUDA graph state, following the pattern from megatron/rl/rl_utils.py.
        lang_module = (
            self.model.module.module
            if hasattr(self.model.module, "module")
            else self.model.module
        )

        toggle_cuda_graphs(lang_module, set_to="none", reset_cuda_graphs=True)

        self.model.zero_grad_buffer()
        if hasattr(self.model, "inference_params"):
            self.model.inference_params = None

        # Reset any cached attention states
        for module in self.model.modules():
            if hasattr(module, "reset_inference_cache"):
                module.reset_inference_cache()
            if hasattr(module, "_inference_key_value_memory"):
                module._inference_key_value_memory = None

        if gbs is None:
            gbs = self.cfg["train_global_batch_size"]
        if mbs is None:
            mbs = self.cfg["train_micro_batch_size"]
        local_gbs = gbs // self.dp_size
        total_dataset_size = torch.tensor(data.size, device="cuda")
        torch.distributed.all_reduce(
            total_dataset_size,
            op=torch.distributed.ReduceOp.SUM,
            group=parallel_state.get_data_parallel_group(),
        )
        num_global_batches = int(total_dataset_size.item()) // gbs

        if eval_mode:
            ctx: AbstractContextManager[Any] = torch.no_grad()
            self.model.eval()
        else:
            ctx = nullcontext()
            # Ensure model is in training mode
            self.model.train()

        with ctx:
            # dim 1 is always assumed to be the sequence dim, sanity check this here
            sequence_dim = 1
            seq_dim_size = data["input_ids"].shape[sequence_dim]
            for k, v in data.items():
                if torch.is_tensor(v) and len(v.shape) > 1:
                    assert v.shape[sequence_dim] == seq_dim_size, (
                        f"Dim 1 must be the sequence dim, expected dim 1={seq_dim_size} but got shape {v.shape}"
                    )

            forward_step = partial(
                forward_step_arbitrary_loss, loss_fn=loss_fn, policy_cfg=self.cfg
            )
            all_mb_metrics = []
            losses = []
            total_num_microbatches = 0
            for gb_idx in range(num_global_batches):
                global_batch = data.get_batch(batch_idx=gb_idx, batch_size=local_gbs)

                assert "sample_mask" in global_batch, (
                    "sample_mask must be present in the data!"
                )
                ## get the normalization factor for the loss
                local_valid_seqs = torch.sum(global_batch["sample_mask"])

                if not "token_mask" in global_batch:
                    local_valid_toks = (
                        local_valid_seqs * global_batch["input_ids"].shape[1]
                    )
                else:
                    local_valid_toks = torch.sum(
                        global_batch["token_mask"][:, 1:]
                        * global_batch["sample_mask"].unsqueeze(-1)
                    )

                to_reduce = torch.tensor([local_valid_seqs, local_valid_toks]).cuda()
                torch.distributed.all_reduce(
                    to_reduce, group=parallel_state.get_data_parallel_group()
                )
                global_valid_seqs, global_valid_toks = to_reduce[0], to_reduce[1]

                if (
                    hasattr(loss_fn, "loss_type")
                    and loss_fn.loss_type == LossType.TOKEN_LEVEL
                ):
                    assert "token_mask" in global_batch, (
                        "token_mask must be present in the data when using token-level loss"
                    )

                batch = data.get_batch(batch_idx=gb_idx, batch_size=local_gbs)
                pack_seqs = False
                seqlen_key = None
                pad_factor = 1
                pad_full_seq_to = None
                pad_packed_seq_to_multiple_of = 1
                if self.cfg["dynamic_batching"]["enabled"]:
                    data_iterator = batch.make_microbatch_iterator_with_dynamic_shapes()
                    data_iterator_len = (
                        batch.get_microbatch_iterator_dynamic_shapes_len()
                    )
                elif self.cfg["sequence_packing"]["enabled"]:
                    data_iterator = (
                        batch.make_microbatch_iterator_for_packable_sequences()
                    )
                    data_iterator_len, seq_dim_size = (
                        batch.get_microbatch_iterator_for_packable_sequences_len()
                    )
                    mbs = 1
                    pack_seqs = True
                    seqlen_key = "input_lengths"
                    (
                        pad_factor,
                        pad_packed_seq_to_multiple_of,
                        pad_full_seq_to,
                    ) = _get_pack_sequence_parameters_for_megatron(
                        self.cfg["megatron_cfg"],
                        seq_dim_size,
                    )
                    # if pad_full_seq_to is not None, we need to use it as the sequence length
                    seq_dim_size = pad_full_seq_to or seq_dim_size
                else:
                    data_iterator = batch.make_microbatch_iterator(mbs)
                    data_iterator_len = local_gbs // mbs

                # Track total microbatches for MoE aux-loss averaging
                total_num_microbatches += int(data_iterator_len)

                rerun_state_machine = get_rerun_state_machine()
                while rerun_state_machine.should_run_forward_backward(data_iterator):
                    # Set grad to zero.
                    self.model.zero_grad_buffer()
                    self.optimizer.zero_grad()

                    # Forward pass.
                    forward_backward_func = get_forward_backward_func()
                    losses_reduced = forward_backward_func(
                        forward_step_func=partial(
                            forward_step,
                            self.mcore_state,
                            global_valid_seqs,
                            global_valid_toks,
                            pack_sequences=pack_seqs,
                            seq_length_key=seqlen_key,
                            pad_individual_seqs_to_multiple_of=pad_factor,
                            pad_packed_seq_to_multiple_of=pad_packed_seq_to_multiple_of,
                            pad_full_seq_to=pad_full_seq_to,
                            defer_fp32_logits=self.defer_fp32_logits,
                        ),
                        data_iterator=data_iterator,
                        model=self.model,
                        num_microbatches=data_iterator_len,
                        seq_length=seq_dim_size,
                        micro_batch_size=mbs,
                        decoder_seq_length=seq_dim_size,
                        forward_only=eval_mode,
                        do_not_average_loss=True,
                    )

                # Empty unused memory.
                if self.cfg["megatron_cfg"]["empty_unused_memory_level"] >= 1:
                    torch.cuda.empty_cache()

                # Update parameters.
                if not eval_mode:
                    update_successful, grad_norm, num_zeros_in_grad = (
                        self.optimizer.step()
                    )
                else:
                    update_successful, grad_norm, num_zeros_in_grad = (True, 0.0, 0.0)

                # when freezing sub-models we may have a mixture of successful and unsucessful ranks,
                # so we must gather across mp ranks
                update_successful = logical_and_across_model_parallel_group(
                    update_successful
                )
                # grad_norm and num_zeros_in_grad will be None on ranks without trainable params,
                # so we must gather across mp ranks
                grad_norm: float = reduce_max_stat_across_model_parallel_group(
                    grad_norm
                )
                num_zeros_in_grad: float = reduce_max_stat_across_model_parallel_group(
                    num_zeros_in_grad
                )

                if update_successful:
                    skipped_iter = 0
                else:
                    skipped_iter = 1

                # Empty unused memory.
                if self.cfg["megatron_cfg"]["empty_unused_memory_level"] >= 2:
                    torch.cuda.empty_cache()

                if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    # keep all microbatch metrics to be normalized later
                    gb_loss_metrics = []
                    mb_losses = []
                    for x in losses_reduced:
                        loss_metrics = {}
                        for k in x.keys():
                            loss_metrics[k] = x[k] / num_global_batches
                        gb_loss_metrics.append(loss_metrics)
                        curr_lr = self.scheduler.get_lr(self.optimizer.param_groups[0])
                        curr_wd = self.scheduler.get_wd()
                        loss_metrics["lr"] = curr_lr
                        loss_metrics["wd"] = curr_wd
                        loss_metrics["global_valid_seqs"] = global_valid_seqs.item()
                        loss_metrics["global_valid_toks"] = global_valid_toks.item()
                        mb_losses.append(loss_metrics["loss"])

                    torch.distributed.broadcast_object_list(
                        [gb_loss_metrics],
                        src=get_pipeline_model_parallel_last_rank(),
                        group=get_pipeline_model_parallel_group(),
                    )
                else:
                    loss_metrics = [None]  # type: ignore
                    torch.distributed.broadcast_object_list(
                        loss_metrics,
                        src=get_pipeline_model_parallel_last_rank(),
                        group=get_pipeline_model_parallel_group(),
                    )
                    gb_loss_metrics = loss_metrics[0]
                    mb_losses = [x["loss"] for x in gb_loss_metrics]

                all_mb_metrics.extend(gb_loss_metrics)
                losses.append(torch.tensor(mb_losses).sum().item())

        if not eval_mode:
            # take one LR step every rollout batch
            # we need to scale the step by gbs to counteract the fact that NeMo automatically
            # scales lr_warmup_steps by gbs during init
            self.scheduler.step(increment=gbs)

        # Aggregate metrics across all microbatches
        mb_metrics = defaultdict(list)
        for m in all_mb_metrics:
            for k, v in m.items():
                mb_metrics[k].append(v)

        with torch.no_grad():
            global_loss = torch.tensor(losses, device="cuda")
            torch.distributed.all_reduce(
                global_loss,
                op=torch.distributed.ReduceOp.SUM,
                group=parallel_state.get_data_parallel_group(),
            )

        metrics = {
            "global_loss": global_loss.cpu(),
            "rank": torch.distributed.get_rank(),
            "gpu_name": torch.cuda.get_device_name(),
            "model_dtype": self.dtype,
            "all_mb_metrics": dict(mb_metrics),
            "grad_norm": torch.tensor([grad_norm]),
        }
        # Collect MoE aux metrics averaged across microbatches
        num_moe_experts = getattr(self.model.config, "num_moe_experts", None)
        if num_moe_experts is not None and num_moe_experts > 1:
            moe_loss_scale = 1.0 / max(1, total_num_microbatches)
            moe_metrics = get_moe_metrics(
                loss_scale=moe_loss_scale,
                per_layer_logging=self.cfg["megatron_cfg"]["moe_per_layer_logging"],
            )
            if moe_metrics:
                metrics["moe_metrics"] = moe_metrics
        return metrics

    @wrap_with_nvtx_name("megatron_policy_worker/get_logprobs")
    def get_logprobs(
        self, *, data: BatchedDataDict[Any], micro_batch_size: Optional[int] = None
    ) -> BatchedDataDict[LogprobOutputSpec]:
        """Get the logprobs of the model for a batch of data.

        Uses the configured logprob_batch_size to do microbatching.
        Input data is assumed to be right-padded. The method internally converts to
        left-padded format for computation, and returns outputs in right-padded format.
        If micro_batch_size is provided, it will be used instead of the configured
        logprob_batch_size.

        Returns:
          a BatchedDataDict with key "logprobs" and shape [batch_size, sequence_length].
          We use the convention that the logprob of the first token is 0 so that the sequence length is maintained.
          The logprob of input token i is specified at position i in the output logprobs tensor.
        """
        no_grad = torch.no_grad()
        no_grad.__enter__()
        logprob_batch_size = (
            micro_batch_size
            if micro_batch_size is not None
            else self.cfg["logprob_batch_size"]
        )

        # dim 1 is always assumed to be the sequence dim, sanity check this here
        sequence_dim = 1
        input_seq_dim_size = data["input_ids"].shape[sequence_dim]
        for k, v in data.items():
            if torch.is_tensor(v) and len(v.shape) > 1:
                assert v.shape[sequence_dim] == input_seq_dim_size, (
                    f"Dim 1 must be the sequence dim, expected dim 1={input_seq_dim_size} but got shape {v.shape}"
                )

        self.model.eval()

        pp_seq_dim_size = input_seq_dim_size
        pp_grp = get_pipeline_model_parallel_group()
        pad_factor = 1
        pad_packed_seq_to_multiple_of = 1
        pad_full_seq_to = None
        if self.cfg["sequence_packing"]["enabled"]:
            _, seq_dim_size = data.get_microbatch_iterator_for_packable_sequences_len()
            (
                pad_factor,
                pad_packed_seq_to_multiple_of,
                pad_full_seq_to,
            ) = _get_pack_sequence_parameters_for_megatron(
                self.cfg["megatron_cfg"],
                seq_dim_size,
            )
            pp_seq_dim_size = pad_full_seq_to or pp_seq_dim_size

        def forward_step_fn(
            data_iterator: Iterator[BatchedDataDict[Any]], model: GPTModel
        ):
            nonlocal pad_full_seq_to, pad_packed_seq_to_multiple_of, pad_factor
            data_dict = next(data_iterator).to("cuda")
            if self.cfg["sequence_packing"]["enabled"]:
                original_seq_length = data_dict["input_ids"].shape[1]
                cp_size = self.cfg["megatron_cfg"]["context_parallel_size"]
                cp_rank = get_context_parallel_rank()
                (
                    input_ids,
                    input_ids_cp_sharded,
                    packed_seq_params,
                    cu_seqlens,
                    cu_seqlens_padded,
                ) = _pack_sequences_for_megatron(
                    data_dict["input_ids"].clone(),
                    data_dict["input_lengths"],
                    pad_individual_seqs_to_multiple_of=pad_factor,
                    pad_packed_seq_to_multiple_of=pad_packed_seq_to_multiple_of,
                    pad_packed_seq_to=pad_full_seq_to,
                    cp_rank=cp_rank,
                    cp_size=cp_size,
                )
                attention_mask, position_ids = None, None
                unpacked_input_ids = data_dict["input_ids"]
            else:
                input_ids = data_dict["input_ids"]
                input_ids_cp_sharded = input_ids
                attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
                    data=input_ids,
                    eod_token=0,  # used for loss_mask, which we don't use
                    pad_token=0,  # used for loss_mask, which we don't use
                    reset_position_ids=False,
                    reset_attention_mask=False,
                    eod_mask_loss=False,
                    pad_mask_loss=False,
                )
                packed_seq_params = None
                unpacked_input_ids = input_ids

            multimodal_data = data_dict.get_multimodal_dict(
                as_tensors=True, device=input_ids.device
            )
            if len(multimodal_data) > 0:
                position_ids = None

            additional_kwargs = {}
            # Mamba models currently do not support packed_seq_params
            if packed_seq_params is not None:
                additional_kwargs["packed_seq_params"] = packed_seq_params

            if self.defer_fp32_logits:
                additional_kwargs["fp32_output"] = False

            output_tensor = model(
                input_ids=input_ids_cp_sharded,
                position_ids=position_ids,
                attention_mask=attention_mask,
                **multimodal_data,
                **additional_kwargs,
            )

            # Apply temperature scaling to logits for training
            # This matches the dtensor worker's _apply_temperature_scaling in the train method
            if "generation" in self.cfg and self.cfg["generation"] is not None:
                output_tensor.div_(self.cfg["generation"]["temperature"])

            def collection_fn(output_tensor):
                stc = time.time()
                tp_grp = get_tensor_model_parallel_group()
                tp_rank = get_tensor_model_parallel_rank()
                logprob_chunk_size = self.cfg.get("logprob_chunk_size", None)
                if self.cfg["sequence_packing"]["enabled"]:
                    token_logprobs = from_parallel_logits_to_logprobs_packed_sequences(
                        output_tensor,
                        target=input_ids,
                        cu_seqlens_padded=cu_seqlens_padded,
                        unpacked_seqlen=original_seq_length,
                        vocab_start_index=tp_rank * output_tensor.shape[-1],
                        vocab_end_index=(tp_rank + 1) * output_tensor.shape[-1],
                        group=tp_grp,
                        inference_only=True,
                        cp_group=get_context_parallel_group(),
                        chunk_size=logprob_chunk_size,
                    )
                else:
                    token_logprobs = from_parallel_logits_to_logprobs(
                        output_tensor,
                        target=unpacked_input_ids,
                        vocab_start_index=tp_rank * output_tensor.shape[-1],
                        vocab_end_index=(tp_rank + 1) * output_tensor.shape[-1],
                        tp_group=tp_grp,
                        inference_only=True,
                        chunk_size=logprob_chunk_size,
                    )

                # Prepend 0 logprob for first token to maintain same sequence length as input
                token_logprobs = torch.cat(
                    [torch.zeros_like(token_logprobs[:, :1]), token_logprobs], dim=1
                )
                return torch.tensor(0.0, device=token_logprobs.device), {
                    "logprobs": token_logprobs
                }

            return output_tensor, collection_fn

        if self.cfg["dynamic_batching"]["enabled"]:
            mb_iterator = data.make_microbatch_iterator_with_dynamic_shapes()
            data_iterator_len = data.get_microbatch_iterator_dynamic_shapes_len()
            micro_batch_size = logprob_batch_size
        elif self.cfg["sequence_packing"]["enabled"]:
            mb_iterator = data.make_microbatch_iterator_for_packable_sequences()
            data_iterator_len, _ = (
                data.get_microbatch_iterator_for_packable_sequences_len()
            )
            micro_batch_size = 1
        else:
            mb_iterator = data.make_microbatch_iterator(logprob_batch_size)
            data_iterator_len = max(1, data.size // logprob_batch_size)
            micro_batch_size = logprob_batch_size

        forward_backward_func = get_forward_backward_func()
        list_of_logprobs = forward_backward_func(
            forward_step_func=forward_step_fn,
            data_iterator=mb_iterator,
            model=self.model,
            num_microbatches=data_iterator_len,
            seq_length=pp_seq_dim_size,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=pp_seq_dim_size,
            forward_only=True,
        )
        if is_pipeline_last_stage(ignore_virtual=True):
            all_log_probs_padded = []
            all_logprobs = [l["logprobs"] for l in list_of_logprobs]
            for lp in all_logprobs:
                padding_needed = input_seq_dim_size - lp.shape[1]
                if padding_needed > 0:
                    lp = torch.nn.functional.pad(
                        lp, (0, padding_needed), mode="constant", value=0.0
                    )
                all_log_probs_padded.append(lp)

            logprobs = torch.cat(all_log_probs_padded, dim=0)
            # broadcast logprobs to first pp rank
            broadcast_tensor(logprobs, torch.distributed.get_rank(), pp_grp)
        else:
            logprobs = broadcast_tensor(
                None, get_pipeline_model_parallel_last_rank(), pp_grp
            )

        no_grad.__exit__(None, None, None)
        return BatchedDataDict[LogprobOutputSpec](logprobs=logprobs).to("cpu")

    @contextmanager
    def use_reference_model(self):
        """Context manager that temporarily swaps the reference model and active model.

        On entry: Moves model to CPU, moves reference_model to CUDA. Swaps the references
        On exit: Restores original references and re-flips cuda/cpu
        """
        ## disable overlap param gather when swapping weights
        if self.should_disable_forward_pre_hook:
            self.disable_forward_pre_hook()

        with torch.no_grad():
            try:
                # Save original references
                model_state_dict = {}
                for name, item in self.model.state_dict().items():
                    if isinstance(item, torch.Tensor):
                        item = item.detach().to(
                            device="cpu", non_blocking=True, copy=True
                        )
                    model_state_dict[name] = item

                self.model.load_state_dict(self.reference_state_dict, strict=True)
                # for name, item in self.reference_state_dict.items():
                # if isinstance(item, torch.Tensor):
                # self.model.state_dict()[name] = item.detach().to(device="cuda", non_blocking=True, copy=True)

                if self.cfg["megatron_cfg"]["empty_unused_memory_level"] >= 1:
                    gc.collect()
                    torch.cuda.empty_cache()

                # - self.model is the original reference_model, now on CUDA
                # - self.reference_model is the original model, now on CPU
                yield

            finally:
                # Restore original references and device placement
                self.model.load_state_dict(model_state_dict, strict=True)
                # for name, item in model_state_dict.items():
                # if isinstance(item, torch.Tensor):
                # item = item.detach().to(device="cuda", non_blocking=True, copy=True)
                # self.model.state_dict()[name] = item

                if self.cfg["megatron_cfg"]["empty_unused_memory_level"] >= 1:
                    gc.collect()
                    torch.cuda.empty_cache()

                ## re-enable overlap param gather after weight swap
                if self.should_disable_forward_pre_hook:
                    self.enable_forward_pre_hook()

    @wrap_with_nvtx_name("megatron_policy_worker/get_topk_logits")
    def get_topk_logits(
        self,
        *,
        data: BatchedDataDict[GenerationDatumSpec],
        k: int,
        micro_batch_size: Optional[int] = None,
    ):
        """Get the top-k logits and indices for a batch of data.

        The major difference from get_logprobs is that we compute top-k logits and indices for each position in the sequence.

        Returns:
            BatchedDataDict containing:
                - topk_logits: Tensor of top-k logits for each position in the sequence
                - topk_indices: Tensor of top-k indices for each position in the sequence
        """
        no_grad = torch.no_grad()
        no_grad.__enter__()

        logprob_batch_size = (
            micro_batch_size
            if micro_batch_size is not None
            else self.cfg["logprob_batch_size"]
        )

        sequence_dim = 1
        input_seq_dim_size = data["input_ids"].shape[sequence_dim]
        # Avoid shadowing the function argument `k` by using a distinct variable name
        for tensor_name, v in data.items():
            if torch.is_tensor(v) and len(v.shape) > 1:
                assert v.shape[sequence_dim] == input_seq_dim_size, (
                    f"Tensor {tensor_name} must have sequence dimension {sequence_dim} of size {input_seq_dim_size}, but got shape {v.shape}"
                )

        self.model.eval()

        pp_seq_dim_size = input_seq_dim_size
        pp_grp = get_pipeline_model_parallel_group()

        pad_factor = 1
        pad_packed_seq_to_multiple_of = 1
        pad_full_seq_to = None
        if self.cfg["sequence_packing"]["enabled"]:
            _, seq_dim_size = data.get_microbatch_iterator_for_packable_sequences_len()
            (
                pad_factor,
                pad_packed_seq_to_multiple_of,
                pad_full_seq_to,
            ) = _get_pack_sequence_parameters_for_megatron(
                self.cfg["megatron_cfg"],
                seq_dim_size,
            )
            pp_seq_dim_size = pad_full_seq_to or pp_seq_dim_size

        def forward_step_fn(
            data_iterator: Iterator[BatchedDataDict[Any]], model: GPTModel
        ):
            nonlocal pad_full_seq_to, pad_packed_seq_to_multiple_of, pad_factor
            data_dict = next(data_iterator).to("cuda")

            pack = self.cfg["sequence_packing"]["enabled"]
            if pack:
                original_seq_length = data_dict["input_ids"].shape[1]
                cp_size = self.cfg["megatron_cfg"]["context_parallel_size"]
                cp_rank = get_context_parallel_rank()

                (
                    input_ids_unpacked,
                    input_ids_cp_sharded,
                    packed_seq_params,
                    cu_seqlens,
                    cu_seqlens_padded,
                ) = _pack_sequences_for_megatron(
                    data_dict["input_ids"].clone(),
                    data_dict["input_lengths"],
                    pad_individual_seqs_to_multiple_of=pad_factor,
                    pad_packed_seq_to_multiple_of=pad_packed_seq_to_multiple_of,
                    pad_packed_seq_to=pad_full_seq_to,
                    cp_rank=cp_rank,
                    cp_size=cp_size,
                )
                attention_mask, position_ids = None, None
                seq_lengths = data_dict["input_lengths"]
                unpacked_seqlen = original_seq_length
            else:
                input_ids_cp_sharded = data_dict["input_ids"]
                attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
                    data=input_ids_cp_sharded,
                    eod_token=0,
                    pad_token=0,
                    reset_position_ids=False,
                    reset_attention_mask=False,
                    eod_mask_loss=False,
                    pad_mask_loss=False,
                )
                packed_seq_params = None

            multimodal_data = data_dict.get_multimodal_dict(
                as_tensors=True, device=input_ids_cp_sharded.device
            )
            if len(multimodal_data) > 0:
                position_ids = None

            additional_kwargs = {}
            if packed_seq_params is not None:
                additional_kwargs["packed_seq_params"] = packed_seq_params

            output_tensor = model(
                input_ids=input_ids_cp_sharded,
                position_ids=position_ids,
                attention_mask=attention_mask,
                **additional_kwargs,
                **multimodal_data,
            )

            if "generation" in self.cfg and self.cfg["generation"] is not None:
                output_tensor.div_(self.cfg["generation"]["temperature"])

            def collection_fn(_):
                # Only the last PP stage produces final logits/top-k; earlier stages return empty
                # if not is_pipeline_last_stage(ignore_virtual=True):
                # return output_tensor.new_zeros(()), {}

                tp_grp = get_tensor_model_parallel_group()
                tp_rank = get_tensor_model_parallel_rank()
                vocab_shard_size = output_tensor.shape[-1]
                vocab_start_index = tp_rank * vocab_shard_size

                chunk_size = None
                if "logprob_chunk_size" in self.cfg:
                    chunk_size = self.cfg["logprob_chunk_size"]

                topk_vals_local, topk_idx_local = distributed_vocab_topk(
                    output_tensor,
                    k,
                    tp_grp,
                    vocab_start_index=vocab_start_index,
                    vocab_end_index=vocab_start_index + vocab_shard_size,
                    chunk_size=chunk_size,
                )

                if self.cfg["megatron_cfg"]["context_parallel_size"] > 1:
                    cp_grp = get_context_parallel_group()
                    if pack:
                        # Per-sequence CP allgather following packed-sequence logic
                        batch_size = data_dict["input_ids"].shape[0]
                        total_packed_len = int(cu_seqlens_padded[-1].item())

                        topk_vals_full = torch.zeros(
                            (1, total_packed_len, k),
                            dtype=topk_vals_local.dtype,
                            device=topk_vals_local.device,
                        )
                        topk_idx_full = torch.zeros(
                            (1, total_packed_len, k),
                            dtype=topk_idx_local.dtype,
                            device=topk_idx_local.device,
                        )

                        for i in range(batch_size):
                            start_idx = int(cu_seqlens_padded[i].item())
                            end_idx = int(cu_seqlens_padded[i + 1].item())
                            if end_idx > start_idx:
                                local_vals_slice = topk_vals_local[
                                    :, start_idx // cp_size : end_idx // cp_size, :
                                ]
                                local_idx_slice = topk_idx_local[
                                    :, start_idx // cp_size : end_idx // cp_size, :
                                ]
                                gathered_vals = allgather_cp_sharded_tensor(
                                    local_vals_slice, cp_grp, seq_dim=1
                                )
                                gathered_idx = allgather_cp_sharded_tensor(
                                    local_idx_slice, cp_grp, seq_dim=1
                                )
                                # Some kernels may return [X, Y, k] where X*Y = (end_idx - start_idx).
                                # Flatten leading dims and reshape to [1, expected_len, k] to match target.
                                expected_len = end_idx - start_idx
                                if (
                                    gathered_vals.dim() == 3
                                    and gathered_vals.shape[1] != expected_len
                                ):
                                    gathered_vals = gathered_vals.reshape(
                                        1, expected_len, gathered_vals.shape[-1]
                                    )
                                if (
                                    gathered_idx.dim() == 3
                                    and gathered_idx.shape[1] != expected_len
                                ):
                                    gathered_idx = gathered_idx.reshape(
                                        1, expected_len, gathered_idx.shape[-1]
                                    )
                                topk_vals_full[:, start_idx:end_idx, :] = gathered_vals
                                topk_idx_full[:, start_idx:end_idx, :] = gathered_idx
                    else:
                        # Sequence packing must be enabled when CP > 1
                        raise RuntimeError(
                            "Context Parallelism (CP>1) requires sequence packing to be enabled."
                        )
                else:
                    topk_vals_full = topk_vals_local
                    topk_idx_full = topk_idx_local

                if pack:
                    batch_size = data_dict["input_ids"].shape[0]
                    out_vals = torch.zeros(
                        (batch_size, unpacked_seqlen, k),
                        dtype=topk_vals_full.dtype,
                        device=topk_vals_full.device,
                    )
                    out_idx = torch.zeros(
                        (batch_size, unpacked_seqlen, k),
                        dtype=topk_idx_full.dtype,
                        device=topk_idx_full.device,
                    )
                    for i in range(batch_size):
                        seq_len = int(seq_lengths[i].item())
                        start_idx = int(cu_seqlens_padded[i].item())
                        if seq_len > 0:
                            out_vals[i, :seq_len, :] = topk_vals_full[
                                0, start_idx : start_idx + seq_len, :
                            ]
                            out_idx[i, :seq_len, :] = topk_idx_full[
                                0, start_idx : start_idx + seq_len, :
                            ]
                    return output_tensor.new_zeros(()), {
                        "topk_logits": out_vals,
                        "topk_indices": out_idx,
                    }
                else:
                    return output_tensor.new_zeros(()), {
                        "topk_logits": topk_vals_full,
                        "topk_indices": topk_idx_full,
                    }

            return output_tensor, collection_fn

        if self.cfg["dynamic_batching"]["enabled"]:
            mb_iterator = data.make_microbatch_iterator_with_dynamic_shapes()
            data_iterator_len = data.get_microbatch_iterator_dynamic_shapes_len()
            micro_batch = logprob_batch_size
        elif self.cfg["sequence_packing"]["enabled"]:
            mb_iterator = data.make_microbatch_iterator_for_packable_sequences()
            data_iterator_len, _ = (
                data.get_microbatch_iterator_for_packable_sequences_len()
            )
            micro_batch = 1
        else:
            mb_iterator = data.make_microbatch_iterator(logprob_batch_size)
            data_iterator_len = max(1, data.size // logprob_batch_size)
            micro_batch = logprob_batch_size

        forward_backward_func = get_forward_backward_func()
        list_of_outputs = forward_backward_func(
            forward_step_func=forward_step_fn,
            data_iterator=mb_iterator,
            model=self.model,
            num_microbatches=data_iterator_len,
            seq_length=pp_seq_dim_size,
            micro_batch_size=micro_batch,
            decoder_seq_length=pp_seq_dim_size,
            forward_only=True,
        )

        if is_pipeline_last_stage(ignore_virtual=True):
            logits_chunks = []
            indices_chunks = []
            for out in list_of_outputs:
                tk = out["topk_logits"]
                ti = out["topk_indices"]
                pad_len = input_seq_dim_size - tk.shape[1]
                if pad_len > 0:
                    tk = torch.nn.functional.pad(tk, (0, 0, 0, pad_len), value=0.0)
                    ti = torch.nn.functional.pad(ti, (0, 0, 0, pad_len), value=0)
                logits_chunks.append(tk)
                indices_chunks.append(ti)

            topk_logits = torch.cat(logits_chunks, dim=0)
            topk_indices = torch.cat(indices_chunks, dim=0)

            topk_logits = broadcast_tensor(
                topk_logits, torch.distributed.get_rank(), pp_grp
            )
            topk_indices = broadcast_tensor(
                topk_indices, torch.distributed.get_rank(), pp_grp
            )
        else:
            last_pp_rank = get_pipeline_model_parallel_last_rank()
            topk_logits = broadcast_tensor(None, last_pp_rank, pp_grp)
            topk_indices = broadcast_tensor(None, last_pp_rank, pp_grp)

        no_grad.__exit__(None, None, None)
        return BatchedDataDict.from_batches(
            [{"topk_logits": topk_logits.cpu(), "topk_indices": topk_indices.cpu()}]
        )

    @torch.no_grad()
    @wrap_with_nvtx_name("megatron_policy_worker/generate")
    def generate(
        self, *, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> BatchedDataDict[GenerationOutputSpec]:
        """Generate a batch of data using huggingface framework generation.

        Args:
            data: BatchedDataDict containing input_ids and input_lengths tensors
        Returns:
            BatchedDataDict conforming to GenerationOutputSpec:
                - output_ids: input + generated token IDs
                - logprobs: Log probabilities for each token
                - generation_lengths: Lengths of each response
        """
        # 512 bATCH SIZE (200 tokens)
        no_grad = torch.no_grad()
        no_grad.__enter__()
        self.model.config.flash_decode = False
        if self.should_disable_forward_pre_hook:
            self.model = self.move_model(
                self.model, "cuda", move_params=True, move_grads=False
            )
        # Verify input is right padded
        assert isinstance(data, BatchedDataDict), (
            f"data must be a BatchedDataDict, got type: {type(data)}"
        )
        assert "input_ids" in data and "input_lengths" in data, (
            f"input_ids and input_lengths must be present in the BatchedDataDict, got keys: {data.keys()}"
        )
        is_right_padded, error_msg = verify_right_padding(
            data, pad_value=self.tokenizer.pad_token_id
        )
        if not is_right_padded:
            warnings.warn(
                f"Input to Megatron Generation worker is not properly right-padded: {error_msg}"
            )

        model_cfg = self.megatron_cfg.model
        inference_wrapper_config = InferenceWrapperConfig(
            hidden_size=model_cfg.hidden_size,
            inference_batch_times_seqlen_threshold=1000000,
            fp32_residual_connection=model_cfg.fp32_residual_connection,
            params_dtype=model_cfg.params_dtype,
            padded_vocab_size=self.final_padded_vocab_size,  # Use the potentially updated value
            inference_max_seq_length=self.cfg["generation"]["max_new_tokens"],  # type: ignore
            inference_max_requests=self.cfg["generation_batch_size"],
        )

        from megatron.core.inference.contexts.dynamic_context import (
            DynamicInferenceContext,
        )
        from megatron.core.inference.engines.dynamic_engine import (
            DynamicInferenceEngine,
        )
        from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
            GPTInferenceWrapper,
        )
        from megatron.core.inference.sampling_params import SamplingParams

        mcore_generation_config = self.cfg["generation"]["mcore_generation_config"]
        buffer_size_gb = mcore_generation_config.get("buffer_size_gb", 20)

        num_cuda_graphs = mcore_generation_config.get("num_cuda_graphs", 16)
        block_size_tokens = mcore_generation_config.get("block_size_tokens", 256)
        use_cuda_graphs_for_non_decode_steps = mcore_generation_config.get(
            "use_cuda_graphs_for_non_decode_steps", True
        )
        enable_chunked_prefill = mcore_generation_config.get(
            "enable_chunked_prefill", True
        )
        unified_memory_level = mcore_generation_config.get("unified_memory_level", 0)
        buffer_guaranteed_fraction = mcore_generation_config.get(
            "buffer_guaranteed_fraction", 0.1
        )
        max_tokens = mcore_generation_config.get("max_tokens", 16384)

        # Get the underlying language module (not the DDP wrapper) for toggle_cuda_graphs.
        # This follows the pattern from megatron/rl/rl_utils.py megatron_rl_inference_mode().
        # The DDP wrapper can interfere with CUDA graph state management.
        lang_module = (
            self.model.module.module
            if hasattr(self.model.module, "module")
            else self.model.module
        )

        # Enable CUDA graphs for inference using toggle_cuda_graphs on the underlying module.
        # reset_cuda_graphs=True ensures fresh CUDA graphs are created for each generate() call.
        

        toggle_cuda_graphs(lang_module, set_to="local", reset_cuda_graphs=True)

        model_config = self.model.config
        model_config.cuda_graph_impl = "local"

        dynamic_context = DynamicInferenceContext(
            params_dtype=inference_wrapper_config.params_dtype,
            num_layers=model_config.num_layers,
            kv_channels=model_config.kv_channels,
            num_attention_heads=model_config.num_query_groups,
            max_sequence_length=self.cfg["generation"]["max_new_tokens"],
            buffer_guaranteed_fraction=buffer_guaranteed_fraction,
            buffer_size_gb=buffer_size_gb,
            materialize_only_last_token_logits=False,
            num_cuda_graphs=num_cuda_graphs,
            block_size_tokens=block_size_tokens,
            tensor_model_parallel_size=self.cfg["megatron_cfg"][
                "tensor_model_parallel_size"
            ],
            use_cuda_graphs_for_non_decode_steps=use_cuda_graphs_for_non_decode_steps,
            use_flashinfer_fused_rope=False,
            unified_memory_level=unified_memory_level,
            max_tokens_override=max_tokens,
        )
        inference_wrapped_model = GPTInferenceWrapper(
            self.model, inference_wrapper_config, dynamic_context
        )

        inference_wrapped_model.prep_model_for_inference()
        # Set pipeline parallel flag
        inference_wrapped_model.model_is_pipeline_parallel = (
            self.cfg["megatron_cfg"]["pipeline_model_parallel_size"] > 1
        )

        text_generation_controller = TextGenerationController(
            inference_wrapped_model=inference_wrapped_model,
            tokenizer=self.megatron_tokenizer,
        )

        # Calculate seed based on node and rank to ensure reproducibility across workers
        local_rank = torch.cuda.current_device()  # Local GPU index on the node
        num_gpus_per_node = torch.cuda.device_count()
        node_idx = self.rank // num_gpus_per_node if num_gpus_per_node > 0 else 0
        seed = (node_idx * 1024) + local_rank

        # New API: DynamicInferenceEngine has additional parameters
        dynamic_engine = DynamicInferenceEngine(
            text_generation_controller,
            dynamic_context,
            enable_cuda_graph=True,
            random_seed=seed,
            track_paused_request_events=False,
            enable_chunked_prefill=enable_chunked_prefill,
            inference_logging_step_interval=0,
        )

        # Handle None values for top_k - convert to integer as required by Megatron
        top_k_cfg = self.cfg["generation"]["top_k"]
        top_k_val = 1 if greedy else (int(top_k_cfg) if top_k_cfg is not None else 0)

        top_p_cfg = self.cfg["generation"]["top_p"]
        top_p_val = (
            0.0 if greedy else (float(top_p_cfg) if top_p_cfg is not None else 0.0)
        )

        # New API: SamplingParams now includes termination_id and uses num_tokens_total
        sampling_params = SamplingParams(
            temperature=self.cfg["generation"]["temperature"] if not greedy else 0,
            top_k=top_k_val,
            top_p=top_p_val,
            skip_prompt_log_probs=False,
            return_log_probs=True,
            num_tokens_total=self.cfg["generation"]["max_new_tokens"],
            num_tokens_to_generate=None,
            termination_id=self.megatron_tokenizer.eod,
        )

        input_ids = data["input_ids"]
        prompt_tokens_tensor = input_ids.cuda()
        prompt_lengths_tensor = data["input_lengths"]
        request_id = 0

        # New API: add_request now takes sampling_params as a parameter
        for p, prompt_len in zip(
            prompt_tokens_tensor, prompt_lengths_tensor, strict=True
        ):
            dynamic_engine.add_request(
                request_id,
                p[:prompt_len],
                sampling_params=sampling_params,
            )
            request_id += 1

        result = []
        import time
        start = time.time()
        while dynamic_engine.has_unfinished_requests():
            result_step = dynamic_engine.step_modern(verbose=False)
            finished_requests = result_step.get("finished_requests", [])
            for finished_request in finished_requests:
                result.append(finished_request)
        end = time.time()
        print(f"SHAN : Time taken: {end - start} seconds")

        # Sort results by request_id to maintain original batch order
        result.sort(key=lambda x: x.request_id)

        out = {
            "tokens": [x.prompt_tokens.tolist() + x.generated_tokens for x in result],
            "logprobs": [x.prompt_log_probs + x.generated_log_probs for x in result],
        }

        input_lengths = data["input_lengths"]
        # pad the out "tokens" and "logprobs" and make them into tensors from lists
        batch_size = data["input_ids"].size(0)
        max_gen_seq_len = max([len(x.generated_tokens) for x in result])
        padded_input_length = input_ids.size(1)

        max_seq_len = padded_input_length + max_gen_seq_len
        # Create padded tensors for tokens and logprobs
        output_ids_padded = torch.full(
            (batch_size, max_seq_len),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=data["input_ids"].device,
        )

        logprobs_padded = torch.zeros(
            (batch_size, max_seq_len),
            dtype=torch.float,
            device=data["input_ids"].device,
        )

        # Fill in the padded tensors with actual values
        generation_lengths = torch.zeros(
            batch_size, dtype=torch.long, device=data["input_ids"].device
        )
        unpadded_sequence_lengths = torch.zeros(
            batch_size, dtype=torch.long, device=data["input_ids"].device
        )
        for i in range(batch_size):
            seq_len = len(out["tokens"][i])
            output_ids_padded[i, :seq_len] = torch.tensor(
                out["tokens"][i], dtype=torch.long, device=data["input_ids"].device
            )
            generation_lengths[i] = seq_len - input_lengths[i].item()
            unpadded_sequence_lengths[i] = seq_len
            logprob_len = len(out["logprobs"][i])
            logprobs_padded[i, 1 : logprob_len + 1] = torch.tensor(
                out["logprobs"][i],
                dtype=torch.float,
                device=data["input_ids"].device,
            )

        out_dict = {
            "output_ids": output_ids_padded,
            "logprobs": logprobs_padded,
            "generation_lengths": generation_lengths,
            "unpadded_sequence_lengths": unpadded_sequence_lengths,
        }

        self.model.config.flash_decode = False

        # Disable CUDA graphs after inference using toggle_cuda_graphs on the underlying module.
        # This follows the pattern from megatron/rl/rl_utils.py megatron_rl_inference_mode().
        # reset_cuda_graphs=True ensures:
        # 1. cudagraph_manager attributes are deleted from modules
        # 2. _CudagraphGlobalRecord state is reset via delete_cuda_graphs()
        # 3. Model is ready for training (avoids "inference tensor" errors)
        # 4. Next generate() call will create fresh CUDA graphs with new DynamicInferenceContext
        toggle_cuda_graphs(lang_module, set_to="none", reset_cuda_graphs=True)

        no_grad.__exit__(None, None, None)

        return BatchedDataDict.from_batches([out_dict]).to("cpu")

    @torch.no_grad()
    @wrap_with_nvtx_name("megatron_policy_worker/prepare_refit_info")
    def prepare_refit_info(self) -> None:
        """Prepare state dict metadata for weight refitting and IPC streaming."""
        self.refit_param_info_mcore = self._calculate_refit_param_info()

        # Collect tensor metadata for refit / hf side info
        refit_param_info_hf = {}
        # Reuse shared iterator that appends FP8 KV/Q scales when enabled
        for name, tensor in self._iter_params_with_optional_kv_scales():
            refit_param_info_hf[name] = (tensor.shape, tensor.dtype)

        return refit_param_info_hf

    def _calculate_refit_param_info(self) -> list[tuple[str, int]]:
        """Calculate parameter information for refit.

        Each task contains:
        - param_name: Local parameter name without module prefixes
        - mapping: MegatronParamMapping instance for weight transformation
        - pp_rank: Pipeline-parallel rank owning the parameter
        - vp_stage: Virtual-pipeline stage index
        - megatron_module: Reference to Megatron model/submodule
        - param_weight: Target parameter tensor for converted weight

        Returns:
            List of (parameter_name, size_in_bytes) tuples.
        """
        self.refit_conversion_tasks = self.megatron_bridge.get_conversion_tasks(
            [self.model]
        )
        param_info = []

        def calculate_size_in_bytes(param, tp_size, ep_size):
            if param is None:
                # need to broadcast for other pp ranks
                size_in_bytes = None
            else:
                # Calculate size for this parameter
                prec_to_bytes = {
                    torch.bfloat16: 2,
                    torch.float16: 2,
                    torch.float32: 4,
                }
                scale = prec_to_bytes[self.dtype] / prec_to_bytes[param.dtype]
                size_in_bytes = (
                    param.element_size() * param.numel() * tp_size * ep_size * scale
                )

            # Broadcast size_in_bytes across pipeline parallel ranks
            return broadcast_object_across_pp_ranks(size_in_bytes)

        for task in self.refit_conversion_tasks:
            param_info.append(
                (
                    task.param_name,
                    calculate_size_in_bytes(
                        task.param_weight,
                        task.mapping.tp_size,
                        task.mapping.ep_size if task.mapping.is_expert else 1,
                    ),
                )
            )
        return param_info

    def _iter_params_with_optional_kv_scales(
        self,
        kv_scales: Optional[dict[str, float]] = None,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield exported HF parameters and optionally append FP8 KV/Q scale tensors.

        This helper is used by both IPC-based streaming and collective broadcast
        so that the logic for adding KV scales stays consistent in one place.
        """
        base_iter = self.megatron_bridge.export_hf_weights(
            [self.model],
            show_progress=False,
            conversion_tasks=self.refit_conversion_tasks,  # used for metadata caching
        )

        # Yield the original parameters first.
        for name, tensor in base_iter:
            yield name, tensor

        # Check whether FP8 KV cache is enabled.
        use_fp8_kv_cache = False
        if (
            "generation" in self.cfg
            and self.cfg["generation"] is not None
            and self.cfg["generation"]["backend"] == "vllm"
        ):
            generation_cfg = cast(VllmConfig, self.cfg["generation"])
            use_fp8_kv_cache = (
                "vllm_cfg" in generation_cfg
                and "kv_cache_dtype" in generation_cfg["vllm_cfg"]
                and generation_cfg["vllm_cfg"]["kv_cache_dtype"].startswith("fp8")
            )

        if not use_fp8_kv_cache:
            return

        # Append KV (and potentially Q) scale entries to match metadata.
        num_layers = self.megatron_bridge.transformer_config.num_layers
        keys: list[str] = []
        for layer_idx in range(num_layers):
            scale_names = get_vllm_qkv_scale_names(layer_idx)
            keys.extend(scale_names.values())

        for param_name in keys:
            if kv_scales and param_name in kv_scales:
                scale_value = kv_scales[param_name]
            else:
                scale_value = 1.0
            scale_tensor = torch.tensor(
                scale_value, dtype=torch.float32, device="cuda"
            ).reshape(1)
            yield param_name, scale_tensor

    @torch.no_grad()
    @wrap_with_nvtx_name("megatron_policy_worker/stream_weights_via_ipc_zmq")
    def stream_weights_via_ipc_zmq(
        self, buffer_size_bytes: int = 0, kv_scales: Optional[dict[str, float]] = None
    ) -> None:
        """Stream model weights to peer process via ZMQ IPC socket."""
        self.maybe_init_zmq()

        from nemo_rl.models.policy.utils import stream_weights_via_ipc_zmq_impl

        # Use the shared implementation to append optional KV scales.
        stream_weights_via_ipc_zmq_impl(
            params_generator=self._iter_params_with_optional_kv_scales(
                kv_scales=kv_scales
            ),
            buffer_size_bytes=buffer_size_bytes,
            zmq_socket=self.zmq_socket,
            rank=self.rank,
            worker_name=str(self),
        )

    @torch.no_grad()
    def broadcast_weights_for_collective(
        self, kv_scales: Optional[dict[str, float]] = None
    ) -> None:
        """Broadcast the weights for collective communication."""
        # param_iterator will return (name, tensor), we only need tensor.
        packed_broadcast_producer(
            iterator=self._iter_params_with_optional_kv_scales(kv_scales=kv_scales),
            group=self.model_update_group,
            src=0,
            post_iter_func=lambda x: x[1],
        )

    def prepare_for_lp_inference(self):
        self.model = self.move_model(self.model, "cuda", move_grads=False)
        self.model.eval()

        # offload grads to cpu
        self.model = self.move_model(
            self.model, "cpu", move_params=False, move_grads=True
        )  # get rid of grad buffers

        # offload optimizer to cpu
        torch.randn(1).cuda()  # wake up torch allocator
        if (
            hasattr(self, "optimizer")
            and self.optimizer is not None
            and not self.optimizer_cpu_offload
            and self.offload_optimizer_for_logprob
        ):
            self.move_optimizer("cpu")

        gc.collect()
        torch.cuda.empty_cache()

    def prepare_for_training(self, *args, **kwargs):
        # onload models and optimizer state to cuda
        self.model = self.move_model(
            self.model, "cuda", move_grads=True, move_params=True
        )
        self.model.train()

        # Move optimizer state to CUDA if it exists
        # colocated generation will always offload optimizer to cuda before refit
        if (
            hasattr(self, "optimizer")
            and self.optimizer is not None
            and not self.optimizer_cpu_offload
            and (self.offload_optimizer_for_logprob or self.is_generation_colocated)
        ):
            self.move_optimizer("cuda")

        if self.cfg["megatron_cfg"]["empty_unused_memory_level"] >= 1:
            torch.cuda.empty_cache()

    @wrap_with_nvtx_name("megatron_policy_worker/offload_before_refit")
    def offload_before_refit(self):
        """Offload the optimizer and buffers to the CPU."""
        no_grad = torch.no_grad()
        no_grad.__enter__()
        allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
        reserved = torch.cuda.memory_reserved() / (1024**3)  # Convert to GB
        print(
            f"GPU Memory before optimizer offload: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )
        self.model = self.move_model(
            self.model, "cpu", move_params=False, move_grads=True
        )  # get rid of grad buffers
        torch.randn(1).cuda()  # wake up torch allocator
        if (
            hasattr(self, "optimizer")
            and self.optimizer is not None
            and not self.optimizer_cpu_offload
        ):
            self.move_optimizer("cpu")

        gc.collect()
        torch.cuda.empty_cache()

        # Print memory stats after offloading
        allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
        reserved = torch.cuda.memory_reserved() / (1024**3)  # Convert to GB
        print(
            f"GPU Memory after optimizer offload: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )
        no_grad.__exit__(None, None, None)

    @wrap_with_nvtx_name("megatron_policy_worker/offload_after_refit")
    def offload_after_refit(self):
        """Offload as much as possible on the CPU."""
        no_grad = torch.no_grad()
        no_grad.__enter__()
        self.model = self.move_model(self.model, "cpu")
        self.model.eval()
        torch.randn(1).cuda()  # wake up torch allocator
        self.offload_before_refit()  # rerun the old offload function

        allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
        reserved = torch.cuda.memory_reserved() / (1024**3)  # Convert to GB
        print(
            f"GPU Memory after refit complete: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )
        no_grad.__exit__(None, None, None)

    @torch.no_grad()
    def move_model(
        self,
        model: torch.nn.Module,
        device: str,
        move_params: bool = True,
        move_grads: bool = True,
    ) -> torch.nn.Module:
        # move all param and grad buffers to the device
        if isinstance(model, DistributedDataParallel):
            # DDP case
            for buffers in [model.buffers, model.expert_parallel_buffers]:
                for buffer_idx in range(len(buffers)):
                    if device == "cpu":
                        buffers[buffer_idx].offload_to_cpu(
                            move_params=move_params, move_grads=move_grads
                        )
                    elif device == "cuda":
                        buffers[buffer_idx].reload_from_cpu(
                            move_params=move_params, move_grads=move_grads
                        )
                    else:
                        raise ValueError(
                            f"Invalid device: {device}. Only strings 'cpu' and 'cuda' are supported."
                        )
        elif isinstance(model, custom_FSDP):
            if device == "cpu":
                model.param_and_grad_buffer.offload_to_cpu(move_params, move_grads)
            elif device == "cuda":
                model.param_and_grad_buffer.reload_from_cpu(
                    move_params=move_params, move_grads=move_grads
                )
            else:
                raise ValueError(
                    f"Invalid device: {device}. Only strings 'cpu' and 'cuda' are supported."
                )
        else:
            # Ordinary offload case
            if move_params:
                for name, param in model.state_dict().items():
                    new_state_dict = {}
                    for name, item in model.state_dict().items():
                        if isinstance(item, torch.Tensor):
                            item = item.detach().to(
                                device=device, non_blocking=True, copy=True
                            )
                        new_state_dict[name] = item
                    model.load_state_dict(new_state_dict)
        return model

    def move_optimizer(self, device: str):
        # Iterate through the state dictionaries for each parameter group
        if isinstance(self.optimizer, ChainedOptimizer):
            optimizer_state = self.optimizer.state
        else:
            optimizer_state = self.optimizer._get_state()
        for _, state in optimizer_state.items():
            # Iterate through the state items (e.g., momentum, variance) for a parameter
            for k, v in state.items():
                # Check if the item is a tensor
                if torch.is_tensor(v):
                    # Move the tensor to device and update the state dictionary
                    if device == "cpu":
                        if v.is_cuda:
                            state[k] = v.to("cpu")
                    elif device == "cuda":
                        if not v.is_cuda:
                            state[k] = v.to("cuda")
                    else:
                        raise ValueError(
                            f"Invalid device: {device}. Only strings 'cpu' and 'cuda' are supported."
                        )

    def save_checkpoint(
        self,
        weights_path: str,
        optimizer_path: Optional[str] = None,
        **kwargs,
    ):
        """Save a training checkpoint.

        Args:
            weights_path: The specific directory path where the checkpoint will be saved.
            optimizer_path: If not None, optimizer and scheduler states are saved if they exist.
        """
        if not torch.distributed.is_initialized():
            raise RuntimeError(
                "Distributed process group is not initialized. Cannot save checkpoint."
            )

        if self.mcore_state is None or self.model is None:
            raise RuntimeError(
                "Megatron core state or model is not initialized. Cannot save checkpoint."
            )

        original_save_path = self.mcore_state.cfg.checkpoint.save
        # save_dir = os.path.dirname(weights_path)
        release_name = os.path.basename(weights_path)

        try:
            maybe_finalize_async_save(
                self.mcore_state,
                ckpt_cfg=self.mcore_state.cfg.checkpoint,
                blocking=False,
            )
            self.mcore_state.cfg.checkpoint.save = weights_path

            optimizer_to_save = None
            scheduler_to_save = None

            if optimizer_path is not None:
                if self.optimizer is not None:
                    optimizer_to_save = self.optimizer
                if self.scheduler is not None:
                    scheduler_to_save = self.scheduler

            # Ensure model is in eval mode for consistent saving, unless actively training
            # This is a common practice, though NeMo's save might handle this.
            # For safety, if not in training loop, setting to eval.
            is_training = self.model.training
            if not is_training:
                self.model.eval()

            if self.should_disable_forward_pre_hook:
                self.disable_forward_pre_hook()
            save_checkpoint(
                state=self.mcore_state,
                model=[self.model],
                optimizer=optimizer_to_save,
                opt_param_scheduler=scheduler_to_save,
                num_floating_point_operations_so_far=self.mcore_state.train_state.floating_point_operations_so_far,
                checkpointing_context=self.checkpointing_context,
            )
            print(f"Saved checkpoint to {weights_path}")
            maybe_finalize_async_save(
                self.mcore_state,
                ckpt_cfg=self.mcore_state.cfg.checkpoint,
                blocking=True,
                terminate=True,
            )
            if self.should_disable_forward_pre_hook:
                self.enable_forward_pre_hook()

            if not is_training:  # Restore training state if it was changed
                self.model.train()

        except Exception as e:
            print(f"Failed to save checkpoint to {weights_path}: {e}")
            raise
        finally:
            self.mcore_state.cfg.checkpoint.save = original_save_path

    def load_checkpoint(self, weights_path: str, optimizer_path: Optional[str] = None):
        """Load a training checkpoint.

        Args:
            weights_path: The exact directory path from which to load the checkpoint.
            optimizer_path: If not None, attempts to load optimizer and scheduler states
                            if self.optimizer and self.scheduler are initialized.
        """
        raise NotImplementedError(
            "Loading checkpoints outside of the init function is not yet implemented for Megatron policy."
        )

    def check_tensor_parallel_attributes(self) -> dict[str, Any]:
        """Check tensor parallel attributes on model parameters.

        Returns:
            Dictionary containing information about tensor parallel parameters:
            - tp_params: List of parameter names that have tensor_model_parallel=True
            - non_tp_params: List of parameter names that have tensor_model_parallel=False
            - total_params: Total number of parameters checked
            - tp_size: Tensor parallel size from config
        """
        tp_params = []
        non_tp_params = []
        total_params = 0

        for name, param in self.model.named_parameters():
            total_params += 1
            tensor_model_parallel = getattr(param, "tensor_model_parallel", False)

            if tensor_model_parallel:
                tp_params.append(
                    {
                        "name": name,
                        "tensor_model_parallel": tensor_model_parallel,
                        "partition_dim": getattr(param, "partition_dim", None),
                        "partition_stride": getattr(param, "partition_stride", None),
                        "shape": list(param.shape),
                    }
                )
            else:
                non_tp_params.append(
                    {
                        "name": name,
                        "tensor_model_parallel": tensor_model_parallel,
                        "shape": list(param.shape),
                    }
                )

        return {
            "tp_params": tp_params,
            "non_tp_params": non_tp_params,
            "total_params": total_params,
            "tp_size": self.megatron_cfg.model.tensor_model_parallel_size,
        }

    @torch.no_grad()
    def calibrate_qkv_fp8_scales(
        self,
        *,
        data: BatchedDataDict[Any],
        micro_batch_size: Optional[int] = None,
        percentile: float = 99.9,
        margin: float = 1.05,
        include_q: bool = False,
    ) -> dict[str, Any]:
        """One-shot calibration of Q/K/V activation scales (for FP8 KV cache).

        - Captures each layer's `query_key_value` output through forward hooks, splits Q/K/V, and computes percentile amax.
        - In parallel (DP/TP/PP) environments, first computes local percentiles, then takes max across all ranks for conservativeness.
        - By default only returns and saves K/V scales, optionally returns Q.

        Args:
            data: Representative sample batch for calibration, following get_logprobs input conventions.
            micro_batch_size: Micro batch size during calibration; if None, reuses logprob_batch_size.
            percentile: Percentile for amax (e.g. 99.9).
            margin: Margin factor, e.g. 1.05.
            save_path: If provided, rank0 will save results as JSON.
            include_q: Whether to also return Q scale (usually only K/V needed).

        Returns:
            { "format": "fp8", "percentile": float, "margin": float,
              "layers": { layer_name: {"k_scale": float, "v_scale": float[, "q_scale": float] } } }
        """

        # Allow overriding FP8 max for Q, K, V via environment variables for ease of testing.
        # Defaults align with FP8 e4m3 max magnitude.
        # Use different defaults for Q, K, V to adapt to distribution diffefences
        def _get_env_float(name: str, default: float) -> float:
            try:
                val = os.getenv(name, None)
                return float(val) if val is not None and val != "" else default
            except Exception:
                return default

        FP8_MAX_Q = _get_env_float("FP8_MAX_Q", 448.0)
        FP8_MAX_K = _get_env_float("FP8_MAX_K", 448.0)
        FP8_MAX_V = _get_env_float("FP8_MAX_V", 448.0)

        self.model.eval()

        # Record local percentile amax for q/k/v of each layer
        layer_to_samples_q: dict[str, list[float]] = defaultdict(list)
        layer_to_samples_k: dict[str, list[float]] = defaultdict(list)
        layer_to_samples_v: dict[str, list[float]] = defaultdict(list)
        hook_handles = []

        def _extract_layer_key(module_name: str) -> str:
            # Expected format: "module.decoder.layers.<idx>.self_attention.query_key_value"
            m = re.search(r"module\.decoder\.layers\.(\d+)", module_name)
            if m is not None:
                return f"layer_{m.group(1)}"
            return module_name

        # Hook to capture q/k/v after q/k norm and RoPE
        def _pre_hook_builder_core_attention(module_name: str):
            layer_key = _extract_layer_key(module_name)

            def _pre_hook(module, inputs):
                args = inputs if isinstance(inputs, (tuple, list)) else (inputs,)
                if len(args) == 1 and isinstance(args[0], (tuple, list)):
                    args = args[0]
                # Expected first 3 args to be q, k, v (typical signature for Megatron CoreAttention)
                q = args[0]
                k = args[1]
                v = args[2]
                if include_q:
                    layer_to_samples_q[layer_key].append(
                        float(torch.amax(torch.abs(q)).item())
                    )
                layer_to_samples_k[layer_key].append(
                    float(torch.amax(torch.abs(k)).item())
                )
                layer_to_samples_v[layer_key].append(
                    float(torch.amax(torch.abs(v)).item())
                )

            return _pre_hook

        matched_modules = []
        # Try to register forward_pre_hook on core_attention first
        for name, module in self.model.named_modules():
            if "self_attention.core_attention" in name:
                try:
                    handle = module.register_forward_pre_hook(
                        _pre_hook_builder_core_attention(name)
                    )
                    hook_handles.append(handle)
                    matched_modules.append((name, module.__class__.__name__, "pre"))
                except Exception as e:
                    print(
                        f"Error registering pre-hook for qkv scale calibration on {name}: {e}"
                        " Please check if the model is compatible with the current calibration logic. "
                        "The expected module name is 'self_attention.core_attention'."
                    )
                    raise

        # Run a forward pass to trigger hooks (reuse get_logprobs forward path)
        try:
            _ = self.get_logprobs(data=data, micro_batch_size=micro_batch_size)
        finally:
            for h in hook_handles:
                try:
                    h.remove()
                except Exception as e:
                    print(f"Error removing hook for qkv scale calibration: {e}")
                    raise

        # Compute local percentile amax
        def _percentile(values: list[float], p: float) -> float:
            if not values:
                return 0.0
            t = torch.tensor(sorted(values), device="cuda", dtype=torch.float32)
            rank = max(
                0, min(len(values) - 1, int(round((p / 100.0) * (len(values) - 1))))
            )
            return float(t[rank].item())

        local_layer_to_pamax = {}
        for layer_key in set(
            list(layer_to_samples_k.keys())
            + list(layer_to_samples_v.keys())
            + (list(layer_to_samples_q.keys()) if include_q else [])
        ):
            entry = {}
            if include_q:
                entry["q_amax_p"] = _percentile(
                    layer_to_samples_q.get(layer_key, []), percentile
                )
            entry["k_amax_p"] = _percentile(
                layer_to_samples_k.get(layer_key, []), percentile
            )
            entry["v_amax_p"] = _percentile(
                layer_to_samples_v.get(layer_key, []), percentile
            )
            local_layer_to_pamax[layer_key] = entry

        # Merge across all ranks: take maximum of percentile amax (conservative approach)
        world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )
        gathered = [None for _ in range(world_size)] if world_size > 1 else None
        if world_size > 1:
            torch.distributed.all_gather_object(gathered, local_layer_to_pamax)
            merged = defaultdict(dict)
            for d in gathered:  # type: ignore
                if d is None:
                    continue
                for k, v in d.items():
                    dst = merged[k]
                    for kk, vv in v.items():
                        dst[kk] = max(dst.get(kk, 0.0), float(vv))
            layer_to_pamax = dict(merged)
        else:
            layer_to_pamax = local_layer_to_pamax

        # Compute scale (symmetric quantization): scale = pamax / fp8_max
        result_layers = {}
        for layer_key, vals in layer_to_pamax.items():
            out_entry = {}
            if include_q:
                q_scale = (vals.get("q_amax_p", 0.0) * margin) / FP8_MAX_Q
                out_entry["q_scale"] = float(q_scale)
            k_scale = (vals.get("k_amax_p", 0.0) * margin) / FP8_MAX_K
            v_scale = (vals.get("v_amax_p", 0.0) * margin) / FP8_MAX_V
            out_entry["k_scale"] = float(k_scale)
            out_entry["v_scale"] = float(v_scale)
            result_layers[layer_key] = out_entry

        vllm_format_scales = convert_calibration_to_vllm_format(result_layers)

        final_result = {
            "format": "fp8",
            "percentile": percentile,
            "margin": margin,
            "layers": vllm_format_scales,
        }

        # Sync results across all ranks (broadcast rank0's result)
        if world_size > 1:
            if torch.distributed.get_rank() == 0:
                obj_list = [final_result]
                torch.distributed.broadcast_object_list(obj_list, src=0)
                final_result = obj_list[0]
            else:
                obj_list = [None]
                torch.distributed.broadcast_object_list(obj_list, src=0)
                final_result = obj_list[0]  # type: ignore

        return final_result


class CustomFloat16Module(Float16Module):
    """Float 16 Module.

    Attributes:
        config (TransformerConfig): Transformer config
        fp16 (bool) : Specifies if the model runs in fp16 mode
        bf16 (bool) : Specifies if the model runs in bf16 mode

    Args:
        config (TransformerConfig): The transformer config used to initalize the model
    """

    def __init__(self, config: TransformerConfig, module: torch.nn.Module):
        super(CustomFloat16Module, self).__init__(config, module)
        self.re_enable_float32_expert_bias()

    def re_enable_float32_expert_bias(self) -> None:
        """Ensure MoE router expert bias stays in float32 for numerical stability.

        Walks the wrapped module to find MoE routers and invokes the
        `_maintain_float32_expert_bias()` helper which recreates or casts the
        expert bias tensors to float32 as required by Megatron-LM.
        """
        module = self.module
        # Handle VLM models where language model is nested
        if hasattr(module, "language_model"):
            module = module.language_model
        if hasattr(module, "decoder") and hasattr(module.decoder, "layers"):
            for layer in module.decoder.layers:
                mlp = getattr(layer, "mlp", None)
                router = getattr(mlp, "router", None) if mlp is not None else None
                if router is not None and hasattr(
                    router, "_maintain_float32_expert_bias"
                ):
                    router._maintain_float32_expert_bias()

```

I've reverted the Option 2 changes. All remaining lint warnings are pre-existing import resolution issues.

## Summary

**Option 2 has been reverted** because it was causing memory issues. The problem was:

1. `disable_forward_pre_hook()` was called but never re-enabled
2. This broke the distributed optimizer's param gathering overlap
3. Parameters that should be sharded were being fully materialized, consuming extra GPU memory
4. OOM during backward pass

## Current State

**Only Option 1 remains** (in `distributed_data_parallel.py`):

```python
def disable_forward_pre_hook(self, param_sync: bool = True):
    assert self.use_forward_hook
    # De-register forward pre-hook for all registered modules.
    # We iterate over the dict keys directly (instead of self.module.modules()) to handle
    # cases where modules may have been dynamically added or removed from the module tree
    # (e.g., CudaGraphManager objects added/removed by toggle_cuda_graphs).
    for module, handle in list(self.remove_forward_pre_hook_handles.items()):
        handle.remove()
    self.remove_forward_pre_hook_handles.clear()

    if param_sync:
        self.start_param_sync(force_sync=True)
```

This fix makes `disable_forward_pre_hook` robust to:
- Modules **removed** from the tree after registration
- Modules **added** to the tree after registration

**Option 1 alone should be sufficient** to fix your original error. Try running your application again.

```python
def disable_forward_pre_hook(self, param_sync: bool = True):
    assert self.use_forward_hook
    # De-register forward pre-hook for all registered modules.
    # We iterate over the dict keys directly (instead of self.module.modules()) to handle
    # cases where modules may have been dynamically added or removed from the module tree
    # (e.g., CudaGraphManager objects added/removed by toggle_cuda_graphs).
    for module, handle in list(self.remove_forward_pre_hook_handles.items()):
        handle.remove()
    self.remove_forward_pre_hook_handles.clear()

    if param_sync:
        self.start_param_sync(force_sync=True)
```

```python
def disable_forward_pre_hook(self, param_sync: bool = True):
    assert self.use_forward_hook
    # De-register forward pre-hook for all registered modules.
    # We iterate over the dict keys directly (instead of self.module.modules()) to handle
    # cases where modules may have been dynamically added or removed from the module tree
    # (e.g., CudaGraphManager objects added/removed by toggle_cuda_graphs).
    for module, handle in list(self.remove_forward_pre_hook_handles.items()):
        handle.remove()
    self.remove_forward_pre_hook_handles.clear()

    if param_sync:
        self.start_param_sync(force_sync=True)
```

---

**User**

Still get this issue in step 2
```
========================= Step 2/500 =========================
▶ Preparing batch...
▶ Generating responses for batch of size 512...
(MegatronPolicyWorker[rank=0] pid=817281) GPU Memory before optimizer offload: 36.80GB allocated, 42.81GB reserved
(MegatronPolicyWorker[rank=0] pid=817281) /opt/nemo-rl/3rdparty/Megatron-LM-workspace/Megatron-LM/megatron/core/inference/engines/dynamic_engine.py:114: UserWarning: The `enable_cuda_graph` argument is deprecated and will be removed in `megatron-core 0.15`. `enable_cuda_graph` is now read directly from the transformer config object.
(MegatronPolicyWorker[rank=0] pid=817281)   warnings.warn(
cuda graph warmup - decode, d 1464:   0%|          | 0/16 [00:00<?, ?it/s]
(MegatronPolicyWorker[rank=0] pid=817281) GPU Memory after optimizer offload: 22.36GB allocated, 23.67GB reserved
(MegatronPolicyWorker[rank=0] pid=817281) GPU Memory after refit complete: 22.36GB allocated, 23.67GB reserved
cuda graph warmup - decode, d 1440:   6%|▋         | 1/16 [00:00<00:04,  3.63it/s]
cuda graph warmup - decode, d 1344:  12%|█▎        | 2/16 [00:00<00:03,  3.95it/s]
cuda graph warmup - decode, d 1248:  19%|█▉        | 3/16 [00:00<00:02,  4.53it/s]
cuda graph warmup - decode, d 1152:  25%|██▌       | 4/16 [00:00<00:02,  4.16it/s]
cuda graph warmup - decode, d 1056:  31%|███▏      | 5/16 [00:01<00:03,  3.65it/s]
cuda graph warmup - decode, d 960:  38%|███▊      | 6/16 [00:01<00:02,  3.84it/s]
cuda graph warmup - decode, d 864:  44%|████▍     | 7/16 [00:01<00:02,  4.17it/s]
cuda graph warmup - decode, d 768:  50%|█████     | 8/16 [00:01<00:01,  4.22it/s]
cuda graph warmup - decode, d 672:  56%|█████▋    | 9/16 [00:02<00:01,  4.29it/s]
cuda graph warmup - decode, d 576:  62%|██████▎   | 10/16 [00:02<00:01,  4.27it/s]
cuda graph warmup - decode, d 480:  69%|██████▉   | 11/16 [00:02<00:01,  4.40it/s]
cuda graph warmup - decode, d 384:  75%|███████▌  | 12/16 [00:02<00:00,  4.50it/s]
cuda graph warmup - decode, d 288:  81%|████████▏ | 13/16 [00:03<00:00,  4.53it/s]
cuda graph warmup - decode, d 192:  88%|████████▊ | 14/16 [00:03<00:00,  4.44it/s]
cuda graph warmup - decode, d 96:  94%|█████████▍| 15/16 [00:03<00:00,  4.55it/s]
cuda graph warmup - decode, d 96: 100%|██████████| 16/16 [00:03<00:00,  4.30it/s]
cuda graph warmup - non_decode, d 1464:   0%|          | 0/16 [00:00<?, ?it/s]
cuda graph warmup - non_decode, d 1440:   6%|▋         | 1/16 [00:00<00:03,  3.88it/s]
cuda graph warmup - non_decode, d 1344:  12%|█▎        | 2/16 [00:00<00:03,  4.02it/s]
cuda graph warmup - non_decode, d 1248:  19%|█▉        | 3/16 [00:00<00:03,  4.08it/s]
cuda graph warmup - non_decode, d 1152:  25%|██▌       | 4/16 [00:00<00:02,  4.20it/s]
cuda graph warmup - non_decode, d 1056:  31%|███▏      | 5/16 [00:01<00:02,  4.18it/s]
cuda graph warmup - non_decode, d 960:  38%|███▊      | 6/16 [00:01<00:02,  4.42it/s]
cuda graph warmup - non_decode, d 864:  44%|████▍     | 7/16 [00:01<00:01,  4.60it/s]
cuda graph warmup - non_decode, d 768:  50%|█████     | 8/16 [00:01<00:01,  4.59it/s]
cuda graph warmup - non_decode, d 672:  56%|█████▋    | 9/16 [00:02<00:01,  4.60it/s]
cuda graph warmup - non_decode, d 576:  62%|██████▎   | 10/16 [00:02<00:01,  4.20it/s]
cuda graph warmup - non_decode, d 480:  69%|██████▉   | 11/16 [00:02<00:01,  3.97it/s]
cuda graph warmup - non_decode, d 384:  75%|███████▌  | 12/16 [00:02<00:00,  4.21it/s]
cuda graph warmup - non_decode, d 288:  81%|████████▏ | 13/16 [00:03<00:00,  4.40it/s]
cuda graph warmup - non_decode, d 192:  88%|████████▊ | 14/16 [00:03<00:00,  4.42it/s]
cuda graph warmup - non_decode, d 96:  94%|█████████▍| 15/16 [00:03<00:00,  4.45it/s]
cuda graph warmup - non_decode, d 96: 100%|██████████| 16/16 [00:03<00:00,  4.34it/s]
(MegatronPolicyWorker[rank=0] pid=817281) SHAN : Time taken: 6.7844390869140625 seconds
▶ Processing rewards...,
▶ Computing advantages...
▶ Preparing for logprob inference...
▶ Computing logprobs...
▶ Preparing for training...
▶ Training policy...
Traceback (most recent call last):
  File "/opt/nemo-rl/examples/run_grpo_math.py", line 260, in <module>
    main()
  File "/opt/nemo-rl/examples/run_grpo_math.py", line 243, in main
    grpo_train(
  File "/opt/nemo-rl/nemo_rl/algorithms/grpo.py", line 1352, in grpo_train
    train_results = policy.train(train_data, loss_fn)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/nemo-rl/nemo_rl/models/policy/lm_policy.py", line 532, in train
    results = self.worker_group.get_all_worker_results(futures)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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
ray.exceptions.RayTaskError(OutOfMemoryError): ray::MegatronPolicyWorker.train() (pid=817281, ip=10.65.29.29, actor_id=8f06c22898daa49fa88bdfbe01000000, repr=MegatronPolicyWorker[rank=0])
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/nemo-rl/nemo_rl/utils/nsys.py", line 88, in wrapper
    ret = func(*args, **kwargs)
          ^^^^^^^^^^^^^^^^^^^^^
  File "/opt/nemo-rl/nemo_rl/models/policy/workers/megatron_policy_worker.py", line 1096, in train
    losses_reduced = forward_backward_func(
                     ^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/nemo-rl/3rdparty/Megatron-LM-workspace/Megatron-LM/megatron/core/pipeline_parallel/schedules.py", line 629, in forward_backward_no_pipelining
    backward_step(
  File "/opt/nemo-rl/3rdparty/Megatron-LM-workspace/Megatron-LM/megatron/core/pipeline_parallel/schedules.py", line 480, in backward_step
    custom_backward(output_tensor[0], output_tensor_grad[0])
  File "/opt/nemo-rl/3rdparty/Megatron-LM-workspace/Megatron-LM/megatron/core/pipeline_parallel/schedules.py", line 173, in custom_backward
    Variable._execution_engine.run_backward(
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.15 GiB. GPU 0 has a total capacity of 79.11 GiB of which 780.88 MiB is free. Including non-PyTorch memory, this process has 78.32 GiB memory in use. Of the allocated memory 71.13 GiB is allocated by PyTorch, with 92.22 MiB allocated in private pools (e.g., CUDA Graphs), and 3.11 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
```

---

**Cursor**

This OOM issue might be **unrelated to the Option 1 fix**. Let me ask to clarify:

**Was this OOM error happening before we made any changes?** Or did it only start after we applied the Option 1 fix to `disable_forward_pre_hook`?

The OOM happens during training (backward pass), not during the hook disable/enable operations. The key observations:

1. **CUDA graph warmup is happening at Step 2** - this suggests graphs are being recreated each step (which consumes memory)
2. **Memory jumps from 22.36GB to 71.13GB** during training - a ~49GB increase

If this OOM was happening **before** our changes, then it's a separate issue related to:
- CUDA graph memory not being properly freed between steps
- Memory fragmentation
- The `reset_cuda_graphs=True` flag forcing recreation every step

---

**To verify if Option 1 caused this**, you could temporarily revert Option 1 and see if the OOM still happens:

Looking at the conversation history, I realize something important:

**The original error was an AssertionError** that happened during `get_reference_policy_logprobs` (before training started). Now that we've fixed that assertion, the code proceeds further and reaches training, where it then hits OOM.

**Question**: Was this training workflow (with reference model + CUDA graphs) working successfully before today? Or is this a new configuration you're trying to enable?

If this is a new configuration, the OOM might have always been a potential issue - it just wasn't reached before because the code was failing earlier with the assertion error.

---

**To help diagnose**, could you tell me:

1. Did Step 1 complete successfully with training? (I notice the logs show Step 2, so Step 1 must have worked)

2. Was this exact same configuration (with `init_reference_model=True` and CUDA graphs enabled) working before without any code changes?

3. If you disable the reference model (`init_reference_model=False` in your config), does the OOM still happen at Step 2?

This will help determine if the OOM is:
- A pre-existing issue with the reference model + training memory usage
- Something caused by the forward hook state changes
- A CUDA graph memory fragmentation issue

---

**User**

1. Yes it completed. 
2. Only now I am implementing cuda graphs. (init_reference_model was working)
3. I havent tried. 

---

**Cursor**

Thank you for clarifying! Now I understand:

1. ✅ **The original assertion error is fixed** (Option 1 is working)
2. ❌ **The OOM is a separate, new issue** - it's caused by CUDA graphs, not our forward pre-hook fix

The OOM is happening because you're implementing CUDA graphs for the first time, and there's a memory management issue with how CUDA graphs interact with the training workflow.

## The Problem

Looking at your logs, I see **"cuda graph warmup"** happening at **Step 2**. This means:
- CUDA graphs are being **recreated every step** (due to `reset_cuda_graphs=True`)
- The old CUDA graph memory may not be fully freed before new ones are allocated
- This causes memory to accumulate

Let me check the CUDA graph toggle logic:

```python
if reset_cuda_graphs and set_to == "none":
    delete_cuda_graphs()
```

I see the issue. Looking at your code:

1. **At end of generate()**: `toggle_cuda_graphs(set_to="none", reset_cuda_graphs=True)` - should free CUDA graphs
2. **At step 2 generate()**: `toggle_cuda_graphs(set_to="local", reset_cuda_graphs=True)` - creates new CUDA graphs

The problem is that **CUDA graph memory may not be fully released** even after `delete_cuda_graphs()` is called. PyTorch caches GPU memory, so the old graphs might still be occupying space when new ones are allocated.

## Quick Fix: Add explicit cache clearing

Try adding `torch.cuda.empty_cache()` after disabling CUDA graphs:

```python
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
import gc
import os
import re
import time
import warnings
from collections import defaultdict
from contextlib import AbstractContextManager, contextmanager, nullcontext
from functools import partial
from typing import Any, Iterator, Optional, TypeVar, cast

import ray
import torch
from megatron.bridge import AutoBridge
from megatron.bridge.models.model_provider import get_model
from megatron.bridge.training import fault_tolerance
from megatron.bridge.training.checkpointing import (
    checkpoint_exists,
    init_checkpointing_context,
    load_checkpoint,
    maybe_finalize_async_save,
    save_checkpoint,
)
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    LoggerConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainingConfig,
)
from megatron.bridge.training.initialize import (
    initialize_megatron,
    set_jit_fusion_options,
)
from megatron.bridge.training.optim import setup_optimizer
from megatron.bridge.training.setup import (
    _update_model_config_funcs,
)
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer
from megatron.bridge.training.utils.train_utils import (
    logical_and_across_model_parallel_group,
    reduce_max_stat_across_model_parallel_group,
)
from megatron.bridge.utils.common_utils import get_rank_safe
from megatron.bridge.utils.instantiate_utils import InstantiationMode
from megatron.bridge.utils.vocab_utils import calculate_padded_vocab_size
from megatron.core import parallel_state
from megatron.core.transformer.utils import toggle_cuda_graphs
from megatron.core.distributed import DistributedDataParallel
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import (
    FullyShardedDataParallel as custom_FSDP,
)
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import (
    InferenceWrapperConfig,
)
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.models.gpt import GPTModel
from megatron.core.optimizer import ChainedOptimizer
from megatron.core.parallel_state import (
    get_context_parallel_group,
    get_context_parallel_rank,
    get_pipeline_model_parallel_group,
    get_pipeline_model_parallel_last_rank,
    get_pipeline_model_parallel_world_size,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    is_pipeline_last_stage,
)
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.utils import get_ltor_masks_and_position_ids
from ray.util.queue import Queue
from transformers import PreTrainedTokenizerBase

from nemo_rl.algorithms.interfaces import LossFunction, LossType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.model_utils import (
    allgather_cp_sharded_tensor,
    distributed_vocab_topk,
    from_parallel_logits_to_logprobs,
    from_parallel_logits_to_logprobs_packed_sequences,
)
from nemo_rl.distributed.named_sharding import NamedSharding
from nemo_rl.models.generation.fp8 import (
    convert_calibration_to_vllm_format,
    get_vllm_qkv_scale_names,
)
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationOutputSpec,
    verify_right_padding,
)
from nemo_rl.models.generation.vllm.config import VllmConfig
from nemo_rl.models.megatron.common import (
    _get_pack_sequence_parameters_for_megatron,
    _pack_sequences_for_megatron,
    broadcast_tensor,
    forward_step_arbitrary_loss,
    get_moe_metrics,
)
from nemo_rl.models.megatron.community_import import import_model_from_hf_name
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.interfaces import (
    ColocatablePolicyInterface,
    LogprobOutputSpec,
)
from nemo_rl.models.policy.utils import (
    configure_dynamo_cache,
    get_megatron_checkpoint_dir,
    get_runtime_env_for_policy_worker,
)
from nemo_rl.models.policy.workers.base_policy_worker import AbstractPolicyWorker
from nemo_rl.models.policy.workers.patches import apply_transformer_engine_patch
from nemo_rl.utils.nsys import wrap_with_nvtx_name
from nemo_rl.utils.packed_tensor import packed_broadcast_producer

try:
    from megatron.core.distributed import (
        TorchFullyShardedDataParallel as torch_FSDP,  # noqa: F401 unused-import
    )

    HAVE_FSDP2 = True
except ImportError:
    HAVE_FSDP2 = False

TokenizerType = TypeVar("TokenizerType", bound=PreTrainedTokenizerBase)


def broadcast_object_across_pp_ranks(obj):
    """Broadcast an object across pipeline parallel ranks.

    This utility function handles broadcasting an object from the rank that owns it
    to all other pipeline parallel ranks. If only one rank has the object (non-None),
    it will be broadcast to all other ranks.

    Args:
        obj: The object to broadcast. Can be None on ranks that don't own it.

    Returns:
        The object on all ranks (either the original or the broadcast copy).

    Raises:
        ValueError: If the object doesn't exist on any pipeline parallel rank.
    """
    pp_size = get_pipeline_model_parallel_world_size()
    pp_group = get_pipeline_model_parallel_group()

    if pp_size == 1:
        return obj

    # ------------------------------------------------------------------
    # 1. Gather presence flags from all PP ranks to find the source rank
    # ------------------------------------------------------------------
    has_obj = obj is not None
    obj_flags = [None] * pp_size
    torch.distributed.all_gather_object(obj_flags, has_obj, group=pp_group)

    # ------------------------------------------------------------------
    # 2. Identify the owning rank (the only rank with True flag)
    # ------------------------------------------------------------------
    src_rank = None  # Rank *inside* the PP group
    for rank, flag in enumerate(obj_flags):
        if flag:
            src_rank = rank
            break

    if src_rank is None:
        raise ValueError("Object must exist on at least one PP rank")

    # ------------------------------------------------------------------
    # 3. Broadcast the object from the source rank to all ranks
    # ------------------------------------------------------------------
    # Use broadcast_object_list which is more robust than all_gather_object
    obj_list = [obj]
    pp_ranks = torch.distributed.get_process_group_ranks(pp_group)
    global_src = pp_ranks[src_rank]
    torch.distributed.broadcast_object_list(obj_list, src=global_src, group=pp_group)

    return obj_list[0]


def setup_megatron_model(
    policy_cfg: PolicyConfig,
    cfg: ConfigContainer,
    load_optimizer: bool = True,
    get_embedding_ranks=None,  # TODO @sahilj: What is this?
    get_position_embedding_ranks=None,
):
    state = GlobalState()
    state.cfg = cfg
    # TODO: Freeze state.cfg

    cfg.dist.external_gpu_device_mapping = True
    initialize_megatron(
        cfg=cfg,
        get_embedding_ranks=get_embedding_ranks,
        get_position_embedding_ranks=get_position_embedding_ranks,
    )

    if cfg.ft and cfg.ft.enable_ft_package:
        fault_tolerance.setup(cfg, state)
        fault_tolerance.maybe_setup_simulated_fault(cfg.ft)

    # Set pytorch JIT layer fusion options and warmup JIT functions.
    set_jit_fusion_options(cfg.model, cfg.train.micro_batch_size)

    # Adjust the startup time so it reflects the largest value.
    # This will be closer to what scheduler will see (outside of
    # image ... launches.
    start_time_tensor = torch.tensor(
        [state.start_time], dtype=torch.double, device="cuda"
    )
    torch.distributed.all_reduce(start_time_tensor, op=torch.distributed.ReduceOp.MIN)
    state.start_time = start_time_tensor.item()

    print(
        "time to initialize megatron (seconds): {:.3f}".format(
            time.time() - state.start_time
        )
    )
    torch.distributed.barrier()

    # Context used for persisting some state between checkpoint saves.
    checkpointing_context = init_checkpointing_context(cfg.checkpoint)

    # Tokenizer
    build_tokenizer(
        cfg.tokenizer,
        make_vocab_size_divisible_by=cfg.model.make_vocab_size_divisible_by
        // cfg.model.tensor_model_parallel_size,
        tensor_model_parallel_size=cfg.model.tensor_model_parallel_size,
        trust_remote_code=True,
    )
    assert cfg.model.vocab_size, "vocab size must be specified in model config"

    torch.distributed.barrier()

    pre_wrap_hook = []
    mixed_precision_wrapper = Float16Module
    if policy_cfg["megatron_cfg"]["freeze_moe_router"]:

        def freeze_moe_router(megatron_model):
            if not isinstance(megatron_model, list):
                megatron_model = [megatron_model]
            for model_module in megatron_model:
                # Handle both wrapped (Float16Module) and unwrapped models
                if isinstance(model_module, Float16Module):
                    model_module = model_module.module
                # Handle VLM models
                if hasattr(model_module, "language_model"):
                    model_module = model_module.language_model
                for layer in model_module.decoder.layers:
                    if hasattr(layer, "mlp") and hasattr(layer.mlp, "router"):
                        layer.mlp.router.weight.requires_grad = False

        mixed_precision_wrapper = CustomFloat16Module
        pre_wrap_hook.extend([freeze_moe_router])

    # Model, optimizer, and learning rate.
    model = get_model(
        cfg.model,
        cfg.ddp,
        use_torch_fsdp2=cfg.dist.use_torch_fsdp2,
        overlap_param_gather_with_optimizer_step=cfg.optimizer.overlap_param_gather_with_optimizer_step,
        data_parallel_random_init=cfg.rng.data_parallel_random_init,
        pre_wrap_hook=pre_wrap_hook,
        mixed_precision_wrapper=mixed_precision_wrapper,
    )
    if load_optimizer:
        optimizer, scheduler = setup_optimizer(
            optimizer_config=cfg.optimizer,
            scheduler_config=cfg.scheduler,
            model=model,
            use_gloo_process_groups=cfg.dist.use_gloo_process_groups,
        )
    else:
        optimizer = None
        scheduler = None

    print("Model, optimizer, and learning rate scheduler built")
    torch.distributed.barrier()

    # Load checkpoint if applicable
    if (
        cfg.checkpoint.load is not None
        or cfg.checkpoint.pretrained_checkpoint is not None
    ) and (
        checkpoint_exists(cfg.checkpoint.load)
        or checkpoint_exists(cfg.checkpoint.pretrained_checkpoint)
    ):
        load_checkpoint(
            state,
            model,
            optimizer,
            scheduler,
            checkpointing_context=checkpointing_context,
            skip_load_to_model_and_opt=HAVE_FSDP2 and cfg.dist.use_torch_fsdp2,
        )
        print("Checkpoint loaded")
    torch.distributed.barrier()

    return state, model, optimizer, scheduler, checkpointing_context


def destroy_parallel_state():
    """Safely destroy parallel state and reset async call tracking.

    This function is called during initialization to clean up temporary distributed
    state from model import operations. Resetting async call tracking ensures that
    when the main Megatron distributed context is created, all ranks start with
    consistent call_idx values for async checkpointing.
    """
    if torch.distributed.is_initialized():
        try:
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()
        except:
            pass  # Ignore errors if already destroyed
    if hasattr(parallel_state, "destroy_model_parallel"):
        try:
            parallel_state.destroy_model_parallel()
        except:
            pass  # Ignore errors if already destroyed

    # Reset async calls queue to prevent call_idx mismatches after distributed context recreation
    try:
        import nemo.tron.utils.async_utils as nemo_async_utils
        from megatron.core.dist_checkpointing.strategies.async_utils import (
            AsyncCallsQueue,
        )

        # Clean up any existing async callers first
        old_call_idx = getattr(nemo_async_utils._async_calls_queue, "call_idx", None)
        num_unfinalized = (
            nemo_async_utils._async_calls_queue.get_num_unfinalized_calls()
        )
        if num_unfinalized > 0:
            print(
                f"[WARNING] Resetting async calls queue with {num_unfinalized} unfinalized calls"
            )
        try:
            nemo_async_utils._async_calls_queue.close()
        except:
            pass  # Ignore errors during cleanup
        # Reset the global async calls queue by creating a new instance
        nemo_async_utils._async_calls_queue = AsyncCallsQueue()
        print(f"[DEBUG] Reset NeMo async calls queue (old call_idx: {old_call_idx})")
    except ImportError:
        pass

    # Also reset the Megatron async calls queue if it exists
    try:
        import megatron.training.async_utils as megatron_async_utils
        from megatron.core.dist_checkpointing.strategies.async_utils import (
            AsyncCallsQueue,
        )

        # Clean up any existing async callers first
        old_call_idx = getattr(
            megatron_async_utils._async_calls_queue, "call_idx", None
        )
        num_unfinalized = (
            megatron_async_utils._async_calls_queue.get_num_unfinalized_calls()
        )
        if num_unfinalized > 0:
            print(
                f"[WARNING] Resetting Megatron async calls queue with {num_unfinalized} unfinalized calls"
            )
        try:
            megatron_async_utils._async_calls_queue.close()
        except:
            pass  # Ignore errors during cleanup
        # Reset the Megatron global async calls queue as well
        megatron_async_utils._async_calls_queue = AsyncCallsQueue()
        print(
            f"[DEBUG] Reset Megatron async calls queue (old call_idx: {old_call_idx})"
        )
    except ImportError:
        pass

    # Reset the third global async_calls instance in base strategy module
    try:
        import megatron.core.dist_checkpointing.strategies.base as base_strategy
        from megatron.core.dist_checkpointing.strategies.async_utils import (
            AsyncCallsQueue,
        )

        # Clean up and reset the global async_calls in base strategy
        old_call_idx = getattr(base_strategy.async_calls, "call_idx", None)
        num_unfinalized = base_strategy.async_calls.get_num_unfinalized_calls()
        if num_unfinalized > 0:
            print(
                f"[WARNING] Resetting base strategy async_calls with {num_unfinalized} unfinalized calls"
            )
        try:
            base_strategy.async_calls.close()
        except:
            pass
        base_strategy.async_calls = AsyncCallsQueue()
        print(f"[DEBUG] Reset base strategy async_calls (old call_idx: {old_call_idx})")
    except ImportError:
        pass


@ray.remote(
    runtime_env=get_runtime_env_for_policy_worker("megatron_policy_worker")
)  # pragma: no cover
class MegatronPolicyWorker(AbstractPolicyWorker, ColocatablePolicyInterface):
    def __repr__(self):
        """Customizes the actor's prefix in the Ray logs.

        This makes it easier to identify which worker is producing specific log messages.
        """
        if torch.distributed.is_initialized():
            return f"{self.__class__.__qualname__}[rank={torch.distributed.get_rank()}]"
        else:
            return f"{self.__class__.__qualname__}"

    def __init__(
        self,
        config: PolicyConfig,
        tokenizer: TokenizerType,
        weights_path: Optional[str] = None,
        optimizer_path: Optional[str] = None,
        init_optimizer: bool = True,
        init_reference_model: bool = True,
        *,
        worker_sharding_annotations: NamedSharding,
        pre_init_communication_queue: Queue,
        **kwargs: Any,
    ):
        apply_transformer_engine_patch()

        self.is_generation_colocated = None
        if "generation" in config and config["generation"] is not None:
            self.is_generation_colocated = config["generation"]["colocated"]["enabled"]

        # Explicitly set NCCL_CUMEM_ENABLE to 1 to avoid the P2P initialization error for PyNCCLCommunicator.
        # See https://github.com/NVIDIA-NeMo/RL/issues/564 for more details.
        if not self.is_generation_colocated:
            os.environ["NCCL_CUMEM_ENABLE"] = "1"

        self.cfg = config
        dtype_map = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }
        self.dtype = dtype_map[self.cfg["precision"]]

        self.optimizer_cpu_offload = self.cfg["megatron_cfg"]["optimizer"][
            "optimizer_cpu_offload"
        ]
        self.offload_optimizer_for_logprob = self.cfg["offload_optimizer_for_logprob"]

        # Reward models are not yet supported with Megatron.
        if "reward_model_cfg" in self.cfg and self.cfg["reward_model_cfg"]["enabled"]:
            raise NotImplementedError(
                "Reward models are not yet supported with the Megatron backend, this issue is "
                "tracked in https://github.com/NVIDIA-NeMo/RL/issues/720"
            )

        # Disable dynamo autotune_local_cache to avoid crash when there's already a cache
        # with different order of node_bundles
        configure_dynamo_cache()

        # cfg["model_name"] is allowed to be either an HF model name or a path to an HF checkpoint
        # check if hf_model_name is a path
        hf_model_name = self.cfg["model_name"]
        # Check if the checkpoint already exists
        hf_model_subdir = hf_model_name
        if os.path.exists(hf_model_name):
            hf_model_subdir = f"model_{hf_model_subdir.replace('/', '_')}"

        pretrained_path = f"{get_megatron_checkpoint_dir()}/{hf_model_subdir}"
        pt_checkpoint_exists = os.path.exists(pretrained_path) and os.path.exists(
            os.path.join(pretrained_path, "iter_0000000")
        )

        # Ensure clean slate before import
        destroy_parallel_state()

        # Set for rank for non-collocated to check which ranks to broadcast from
        self.rank = get_rank_safe()
        # Need to initialize the process group before calling into Megatron-Bridge, otherwise Megatron-Bridge will try to set an incorrect device
        torch.distributed.init_process_group("nccl")
        if pt_checkpoint_exists:
            print(f"Checkpoint already exists at {pretrained_path}. Skipping import.")
        else:
            hf_config_overrides = self.cfg.get("hf_config_overrides", {}) or {}
            import_model_from_hf_name(
                hf_model_name,
                pretrained_path,
                self.cfg["megatron_cfg"],
                **hf_config_overrides,
            )

            if parallel_state.model_parallel_is_initialized():
                print("Reinitializing model parallel after loading model state.")
                parallel_state.destroy_model_parallel()

        pretrained_run_config = os.path.join(
            pretrained_path, "iter_0000000/run_config.yaml"
        )

        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if not os.path.exists(pretrained_run_config):
            raise FileNotFoundError(
                f"Pretrained run config not found at {pretrained_run_config} on rank={get_rank_safe()}. This usually means that the one-time HF->mcore conversion on rank=0 saved to a directory not being mounted on this node. Please check "
            )

        try:
            cfg_from_pretrained = ConfigContainer.from_yaml(
                pretrained_run_config, mode=InstantiationMode.STRICT
            )
        except Exception as e:
            # Add helpful context as a note to the exception
            e.add_note(
                f"\n{'=' * 80}\n"
                f"NOTE: A common cause of this error is when the HF->mcore converted checkpoint is\n"
                f"created with an older version of megatron-bridge.\n"
                f"If this checkpoint is old or was generated by a different code version,\n"
                f"try deleting it and rerunning the code.\n"
                f"The checkpoint will be automatically regenerated with the current version.\n\n"
                f"Checkpoint location: {pretrained_path}\n"
                f"{'=' * 80}"
            )
            raise
        model_cfg = cfg_from_pretrained.model

        cfg_from_pretrained.logger = LoggerConfig()

        model_cfg.tensor_model_parallel_size = self.cfg["megatron_cfg"][
            "tensor_model_parallel_size"
        ]
        model_cfg.pipeline_model_parallel_size = self.cfg["megatron_cfg"][
            "pipeline_model_parallel_size"
        ]
        model_cfg.num_layers_in_first_pipeline_stage = self.cfg["megatron_cfg"][
            "num_layers_in_first_pipeline_stage"
        ]
        model_cfg.num_layers_in_last_pipeline_stage = self.cfg["megatron_cfg"][
            "num_layers_in_last_pipeline_stage"
        ]
        model_cfg.sequence_parallel = self.cfg["megatron_cfg"]["sequence_parallel"]
        model_cfg.context_parallel_size = self.cfg["megatron_cfg"][
            "context_parallel_size"
        ]
        if model_cfg.context_parallel_size > 1:
            assert self.cfg["sequence_packing"]["enabled"], (
                "Sequence Packing must be enabled to use Context Parallelism with MCore"
            )
        model_cfg.expert_tensor_parallel_size = self.cfg["megatron_cfg"][
            "expert_tensor_parallel_size"
        ]
        model_cfg.expert_model_parallel_size = self.cfg["megatron_cfg"][
            "expert_model_parallel_size"
        ]

        # Setting moe_router_dtype to higher precision (e.g. fp64) can improve numerical stability,
        # especially when using many experts.
        model_cfg.moe_router_dtype = self.cfg["megatron_cfg"]["moe_router_dtype"]

        # The below two configs (and "freeze_moe_router") are used to stabilize moe training
        # by preventing updates to the moe router. We found that this is helpful in reducing
        # logprob error during training.

        # Set this to "none" to disable load balancing loss.
        model_cfg.moe_router_load_balancing_type = self.cfg["megatron_cfg"][
            "moe_router_load_balancing_type"
        ]
        # Set this to 0.0 to disable updates to the moe router expert bias
        model_cfg.moe_router_bias_update_rate = self.cfg["megatron_cfg"][
            "moe_router_bias_update_rate"
        ]

        model_cfg.moe_permute_fusion = self.cfg["megatron_cfg"]["moe_permute_fusion"]
        if "layernorm_epsilon" in self.cfg["megatron_cfg"]:
            model_cfg.layernorm_epsilon = self.cfg["megatron_cfg"]["layernorm_epsilon"]

        model_cfg.sequence_parallel = self.cfg["megatron_cfg"]["sequence_parallel"]
        model_cfg.bf16 = self.dtype == torch.bfloat16
        model_cfg.fp16 = self.dtype == torch.float16
        if model_cfg.fp16:
            assert not model_cfg.bf16, "fp16 and bf16 cannot be used together"
            model_cfg.params_dtype = torch.float16
        elif model_cfg.bf16:
            assert not model_cfg.fp16, "fp16 and bf16 cannot be used together"
            model_cfg.params_dtype = torch.bfloat16
        else:
            model_cfg.params_dtype = torch.float32
        model_cfg.pipeline_dtype = dtype_map[self.cfg["megatron_cfg"]["pipeline_dtype"]]
        model_cfg.parallel_output = True
        if self.cfg["megatron_cfg"]["activation_checkpointing"]:
            model_cfg.recompute_granularity = "full"
            model_cfg.recompute_method = "uniform"
            model_cfg.recompute_num_layers = 1
        if not model_cfg.gated_linear_unit:
            assert model_cfg.activation_func is not None, (
                "activation_func must be set if not using gated_linear_unit. This likely "
                "indicates an issue in configuration conversion (e.g. activation func was "
                "a lambda and couldn't be serialized). This is based on this check "
                "https://github.com/NVIDIA/Megatron-LM/blob/1ab876ddc4c1893c76f26d775226a8d1dcdfb3d2/megatron/core/transformer/mlp.py#L174."
            )
        model_cfg.apply_rope_fusion = self.cfg["megatron_cfg"]["apply_rope_fusion"]
        model_cfg.bias_activation_fusion = self.cfg["megatron_cfg"][
            "bias_activation_fusion"
        ]
        fp8_cfg = self.cfg["megatron_cfg"].get("fp8_cfg", None)
        self.fp8_cfg = fp8_cfg
        if fp8_cfg is not None and fp8_cfg.get("enabled", False):
            try:
                model_cfg.fp8 = fp8_cfg["fp8"]
                model_cfg.fp8_recipe = fp8_cfg["fp8_recipe"]
                model_cfg.fp8_param = fp8_cfg["fp8_param"]
            except KeyError as e:
                raise KeyError(f"Missing key in fp8_cfg: {e}")
            if model_cfg.fp8_param:
                warnings.warn(
                    "Setting fp8_param=True sometimes causes NaN token_mult_prob_error, please use with caution. "
                    "Refer to https://github.com/NVIDIA-NeMo/RL/issues/1164 for latest updates with this issue."
                )

        optimizer_cpu_offload = self.cfg["megatron_cfg"]["optimizer"][
            "optimizer_cpu_offload"
        ]
        optimizer_offload_fraction = self.cfg["megatron_cfg"]["optimizer"][
            "optimizer_offload_fraction"
        ]
        if optimizer_cpu_offload:
            # Currently, hybrid optimizer (partly on GPU and partly on CPU) is not supported because it conflicts with the way
            # Nemo-rl handles the optimizer offload/onload between generation and training. So if using CPU optimizer the offload_fraction should be 1.0.
            assert optimizer_offload_fraction == 1.0, (
                "Currently for optimizer offloading, only optimizer_offload_fraction=1.0 is supported"
            )
        if (
            "logprob_chunk_size" in self.cfg
            and self.cfg["logprob_chunk_size"] is not None
            and self.cfg["logprob_chunk_size"] > 0
        ):
            assert self.cfg["megatron_cfg"]["defer_fp32_logits"], (
                "defer_fp32_logits must be True if logprob_chunk_size is set"
            )
        self.defer_fp32_logits = self.cfg["megatron_cfg"].get(
            "defer_fp32_logits", None
        ) and (model_cfg.fp16 or model_cfg.bf16)

        checkpoint_config = CheckpointConfig(
            save_interval=100,
            save=weights_path,
            load=weights_path,
            pretrained_checkpoint=pretrained_path,  # This is the path to the pretrained ckpt for the SFT case
            async_save=False,  # This doesn't work right now.
            fully_parallel_save=True,
            fully_parallel_load=True,  # Enable fully parallel load
            load_rng=False,
        )
        ref_checkpoint_config = CheckpointConfig(
            pretrained_checkpoint=pretrained_path,  # This is the path to the pretrained ckpt for the SFT case
            save=None,
            load=None,
            fully_parallel_load=True,  # Enable fully parallel load
            load_rng=False,
        )

        assert "train_iters" in self.cfg["megatron_cfg"], (
            "train_iters must be set in megatron_cfg. For an example, see "
            "https://github.com/NVIDIA-NeMo/RL/blob/bccbc377705a81a1f4b3c31ad9767bcc15f735a8/nemo_rl/algorithms/sft.py#L175-L179."
        )

        ## These settings are required for correct gradient computations in mcore
        ## when calculate_per_token_loss is True, there is no scaling of the gradient in mcore,
        ## so we handle the scaling in nemo-rl.
        ## perform_initialization = True is a workaround to ensure the correct tensor parallel attributes are set
        ## on the TP-sharded parameters.
        model_cfg.calculate_per_token_loss = True
        model_cfg.perform_initialization = True
        # SHAN CHECK:
        model_cfg.cuda_graph_impl = "local"
        model_cfg.cuda_graph_scope = None
        model_cfg.use_te_rng_tracker = True

        assert (
            "aux_loss" not in model_cfg.moe_router_load_balancing_type
            or model_cfg.moe_aux_loss_coeff == 0
        ), (
            "MoE aux loss is currently not supported due to a known bug in Megatron-LM. "
            "See https://github.com/NVIDIA/Megatron-LM/issues/1984 for more details."
        )

        self.megatron_cfg = ConfigContainer(
            model=model_cfg,
            checkpoint=checkpoint_config,
            logger=LoggerConfig(logging_level=0),
            train=TrainingConfig(
                micro_batch_size=1,  # ignored
                global_batch_size=self.cfg["train_global_batch_size"],  # ignored
                train_iters=self.cfg["megatron_cfg"][
                    "train_iters"
                ],  # Set by algorithm setup
            ),
            optimizer=OptimizerConfig(
                **self.cfg["megatron_cfg"]["optimizer"],
            ),
            ddp=DistributedDataParallelConfig(
                check_for_nan_in_grad=True,
                grad_reduce_in_fp32=self.cfg["megatron_cfg"][
                    "distributed_data_parallel_config"
                ]["grad_reduce_in_fp32"],
                overlap_grad_reduce=self.cfg["megatron_cfg"][
                    "distributed_data_parallel_config"
                ]["overlap_grad_reduce"],
                overlap_param_gather=self.cfg["megatron_cfg"][
                    "distributed_data_parallel_config"
                ]["overlap_param_gather"],
                # we need to set average_in_collective=False with calculate_per_token_loss=True.
                # otherwise, mcore throws an assertion error.
                average_in_collective=False,
                use_distributed_optimizer=self.cfg["megatron_cfg"]["optimizer"][
                    "use_distributed_optimizer"
                ],
                data_parallel_sharding_strategy=self.cfg["megatron_cfg"][
                    "distributed_data_parallel_config"
                ]["data_parallel_sharding_strategy"],
            ),
            scheduler=SchedulerConfig(
                **self.cfg["megatron_cfg"]["scheduler"],
            ),
            dataset=None,
            tokenizer=TokenizerConfig(
                tokenizer_type="HuggingFaceTokenizer",
                tokenizer_model=hf_model_name,
            ),
        )
        # TODO: this validation should happen inside mbridge: https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/1665
        if self.dtype == torch.bfloat16:
            assert self.megatron_cfg.model.bf16 == True, (
                "policy.megatron_cfg.model.bf16=True must be set if policy.precision=bfloat16. This is handled by nemo-rl so this indicates something is misconfigured."
            )
            assert (
                self.megatron_cfg.optimizer.use_precision_aware_optimizer == False
                or self.megatron_cfg.optimizer.bf16 == True
            ), (
                "policy.megatron_cfg.optimizer.bf16=True must be set if policy.precision=bfloat16 when using use_precision_aware_optimizer=True"
            )
        elif self.dtype == torch.float16:
            assert self.megatron_cfg.model.fp16 == True, (
                "policy.megatron_cfg.model.fp16=True must be set if policy.precision=float16. This is handled by nemo-rl so this indicates something is misconfigured."
            )
            assert (
                self.megatron_cfg.optimizer.use_precision_aware_optimizer == False
                or self.megatron_cfg.optimizer.fp16 == True
            ), (
                "policy.megatron_cfg.optimizer.fp16=True must be set if policy.precision=float16 when using use_precision_aware_optimizer=True"
            )
        elif self.dtype == torch.float32:
            assert (
                self.megatron_cfg.model.bf16 == False
                and self.megatron_cfg.model.fp16 == False
            ), (
                "policy.megatron_cfg.model.bf16=False and policy.megatron_cfg.model.fp16=False must be set if policy.precision=float32. This is handled by nemo-rl so this indicates something is misconfigured."
            )
            assert (
                self.megatron_cfg.optimizer.bf16 == False
                and self.megatron_cfg.optimizer.fp16 == False
            ), (
                "policy.megatron_cfg.optimizer.bf16=False and policy.megatron_cfg.optimizer.fp16=False must be set if policy.precision=float32"
            )
        self.megatron_cfg.validate()
        (
            self.mcore_state,
            self.model,
            self.optimizer,
            self.scheduler,
            self.checkpointing_context,
        ) = setup_megatron_model(
            policy_cfg=self.cfg, cfg=self.megatron_cfg, load_optimizer=init_optimizer
        )

        # Set the param sync function for the model
        if (
            self.megatron_cfg.ddp.overlap_param_gather
            and self.megatron_cfg.ddp.align_param_gather
        ):
            self.megatron_cfg.param_sync_func = [
                model_chunk.start_param_sync for model_chunk in self.model
            ]
            if len(self.model) == 1:
                self.megatron_cfg.param_sync_func = self.megatron_cfg.param_sync_func[0]

        self.model = self.model[0]  # Get the first model from the list

        if init_reference_model:
            self.model = self.move_model(self.model, "cpu")
            ref_ckpt_context = init_checkpointing_context(ref_checkpoint_config)

            # Create a separate megatron config for the reference model with the correct checkpoint config
            self.megatron_cfg.model.cuda_graph_impl = "none"
            self.megatron_cfg.model.use_te_rng_tracker = True
            ref_megatron_cfg = ConfigContainer(
                model=self.megatron_cfg.model,
                checkpoint=ref_checkpoint_config,  # Use the reference checkpoint config
                logger=self.megatron_cfg.logger,
                train=self.megatron_cfg.train,
                optimizer=self.megatron_cfg.optimizer,
                ddp=self.megatron_cfg.ddp,
                scheduler=self.megatron_cfg.scheduler,
                dataset=self.megatron_cfg.dataset,
                tokenizer=self.megatron_cfg.tokenizer,
            )

            # Create a separate state object for the reference model
            ref_state = GlobalState()
            ref_state.cfg = ref_megatron_cfg

            # Configure mixed precision wrapper for reference model
            ref_mixed_precision_wrapper = Float16Module
            if self.cfg["megatron_cfg"].get("freeze_moe_router", False):
                ref_mixed_precision_wrapper = CustomFloat16Module

            reference_model = get_model(
                self.megatron_cfg.model,
                self.megatron_cfg.ddp,
                use_torch_fsdp2=self.megatron_cfg.dist.use_torch_fsdp2,
                overlap_param_gather_with_optimizer_step=self.megatron_cfg.optimizer.overlap_param_gather_with_optimizer_step,
                pre_wrap_hook=self.megatron_cfg.rng.data_parallel_random_init,
                mixed_precision_wrapper=ref_mixed_precision_wrapper,
            )
            print("Loading the Reference Model")
            if (
                ref_checkpoint_config.pretrained_checkpoint is not None
                and checkpoint_exists(ref_checkpoint_config.pretrained_checkpoint)
            ):
                load_checkpoint(
                    ref_state,  # Use the separate state object with ref checkpoint config
                    reference_model,
                    None,  # no optimizer
                    None,  # no scheduler
                    checkpointing_context=ref_ckpt_context,
                    skip_load_to_model_and_opt=HAVE_FSDP2
                    and self.megatron_cfg.dist.use_torch_fsdp2,
                )
                reference_model = reference_model[0]
                reference_model.eval()
                self.reference_state_dict = {}
                for name, item in reference_model.state_dict().items():
                    if isinstance(item, torch.Tensor):
                        cpu_item = item.detach().to(
                            device="cpu", non_blocking=True, copy=True
                        )
                        del item
                    else:
                        cpu_item = item
                    self.reference_state_dict[name] = cpu_item
                print("Reference model loaded")
            else:
                print("Reference model not loaded")

            self.model = self.move_model(self.model, "cuda")

        _update_model_config_funcs(
            [self.model],
            self.megatron_cfg.model,
            self.megatron_cfg.ddp,
            self.optimizer,
            align_grad_reduce=self.megatron_cfg.dist.align_grad_reduce,
        )

        tokenizer_config = TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model=hf_model_name,
        )

        self.megatron_tokenizer = build_tokenizer(
            tokenizer_config,
            make_vocab_size_divisible_by=self.megatron_cfg.model.make_vocab_size_divisible_by
            // self.cfg["megatron_cfg"]["tensor_model_parallel_size"],
            tensor_model_parallel_size=self.cfg["megatron_cfg"][
                "tensor_model_parallel_size"
            ],
            trust_remote_code=True,
        )
        self.final_padded_vocab_size = calculate_padded_vocab_size(
            self.megatron_cfg.model.vocab_size,
            self.megatron_cfg.model.make_vocab_size_divisible_by,
            self.cfg["megatron_cfg"]["tensor_model_parallel_size"],
        )
        self.dp_size = worker_sharding_annotations.get_axis_size("data_parallel")
        self.megatron_bridge = AutoBridge.from_hf_pretrained(
            hf_model_name, trust_remote_code=True
        )

        self.should_disable_forward_pre_hook = (
            self.cfg["megatron_cfg"]["optimizer"]["use_distributed_optimizer"]
            and self.cfg["megatron_cfg"]["distributed_data_parallel_config"][
                "overlap_param_gather"
            ]
        )

        # vars used for refit
        ## will be initialized in prepare_refit_info
        # refit_param_info_mcore combines the conversion tasks with the param memory
        # [(mcore_param_name, estimated_memory), ...]
        # Note: here param name is local param name, with local layer number and
        # local expert id etc.
        self.refit_conversion_tasks = (
            None  # Meta data for conversion params from megatron bridge
        )
        self.refit_conversion_tasks_current_index = None
        self.refit_param_info_mcore = None

        ## used for streaming update inference engine weights
        self._held_gather_buffer = None

    def enable_forward_pre_hook(self):
        assert isinstance(self.model, DistributedDataParallel)
        self.model.enable_forward_pre_hook()

    def disable_forward_pre_hook(self, param_sync=True):
        assert isinstance(self.model, DistributedDataParallel)
        self.model.disable_forward_pre_hook(param_sync=param_sync)

    @wrap_with_nvtx_name("megatron_policy_worker/train")
    def train(
        self,
        data: BatchedDataDict,
        loss_fn: LossFunction,
        eval_mode: bool = False,
        gbs: Optional[int] = None,
        mbs: Optional[int] = None,
    ) -> dict[str, Any]:
        """Train the policy on a batch of data with a given loss function."""
        # Disable CUDA graphs for training to avoid conflicts with inference tensors.
        # This prevents "Inplace update to inference tensor outside InferenceMode" errors
        # when switching from generation (which uses CUDA graphs) to training.
        # Use toggle_cuda_graphs on the underlying module (not the DDP wrapper) to properly
        # reset CUDA graph state, following the pattern from megatron/rl/rl_utils.py.
        lang_module = (
            self.model.module.module
            if hasattr(self.model.module, "module")
            else self.model.module
        )

        toggle_cuda_graphs(lang_module, set_to="none", reset_cuda_graphs=True)

        self.model.zero_grad_buffer()
        if hasattr(self.model, "inference_params"):
            self.model.inference_params = None

        # Reset any cached attention states
        for module in self.model.modules():
            if hasattr(module, "reset_inference_cache"):
                module.reset_inference_cache()
            if hasattr(module, "_inference_key_value_memory"):
                module._inference_key_value_memory = None

        if gbs is None:
            gbs = self.cfg["train_global_batch_size"]
        if mbs is None:
            mbs = self.cfg["train_micro_batch_size"]
        local_gbs = gbs // self.dp_size
        total_dataset_size = torch.tensor(data.size, device="cuda")
        torch.distributed.all_reduce(
            total_dataset_size,
            op=torch.distributed.ReduceOp.SUM,
            group=parallel_state.get_data_parallel_group(),
        )
        num_global_batches = int(total_dataset_size.item()) // gbs

        if eval_mode:
            ctx: AbstractContextManager[Any] = torch.no_grad()
            self.model.eval()
        else:
            ctx = nullcontext()
            # Ensure model is in training mode
            self.model.train()

        with ctx:
            # dim 1 is always assumed to be the sequence dim, sanity check this here
            sequence_dim = 1
            seq_dim_size = data["input_ids"].shape[sequence_dim]
            for k, v in data.items():
                if torch.is_tensor(v) and len(v.shape) > 1:
                    assert v.shape[sequence_dim] == seq_dim_size, (
                        f"Dim 1 must be the sequence dim, expected dim 1={seq_dim_size} but got shape {v.shape}"
                    )

            forward_step = partial(
                forward_step_arbitrary_loss, loss_fn=loss_fn, policy_cfg=self.cfg
            )
            all_mb_metrics = []
            losses = []
            total_num_microbatches = 0
            for gb_idx in range(num_global_batches):
                global_batch = data.get_batch(batch_idx=gb_idx, batch_size=local_gbs)

                assert "sample_mask" in global_batch, (
                    "sample_mask must be present in the data!"
                )
                ## get the normalization factor for the loss
                local_valid_seqs = torch.sum(global_batch["sample_mask"])

                if not "token_mask" in global_batch:
                    local_valid_toks = (
                        local_valid_seqs * global_batch["input_ids"].shape[1]
                    )
                else:
                    local_valid_toks = torch.sum(
                        global_batch["token_mask"][:, 1:]
                        * global_batch["sample_mask"].unsqueeze(-1)
                    )

                to_reduce = torch.tensor([local_valid_seqs, local_valid_toks]).cuda()
                torch.distributed.all_reduce(
                    to_reduce, group=parallel_state.get_data_parallel_group()
                )
                global_valid_seqs, global_valid_toks = to_reduce[0], to_reduce[1]

                if (
                    hasattr(loss_fn, "loss_type")
                    and loss_fn.loss_type == LossType.TOKEN_LEVEL
                ):
                    assert "token_mask" in global_batch, (
                        "token_mask must be present in the data when using token-level loss"
                    )

                batch = data.get_batch(batch_idx=gb_idx, batch_size=local_gbs)
                pack_seqs = False
                seqlen_key = None
                pad_factor = 1
                pad_full_seq_to = None
                pad_packed_seq_to_multiple_of = 1
                if self.cfg["dynamic_batching"]["enabled"]:
                    data_iterator = batch.make_microbatch_iterator_with_dynamic_shapes()
                    data_iterator_len = (
                        batch.get_microbatch_iterator_dynamic_shapes_len()
                    )
                elif self.cfg["sequence_packing"]["enabled"]:
                    data_iterator = (
                        batch.make_microbatch_iterator_for_packable_sequences()
                    )
                    data_iterator_len, seq_dim_size = (
                        batch.get_microbatch_iterator_for_packable_sequences_len()
                    )
                    mbs = 1
                    pack_seqs = True
                    seqlen_key = "input_lengths"
                    (
                        pad_factor,
                        pad_packed_seq_to_multiple_of,
                        pad_full_seq_to,
                    ) = _get_pack_sequence_parameters_for_megatron(
                        self.cfg["megatron_cfg"],
                        seq_dim_size,
                    )
                    # if pad_full_seq_to is not None, we need to use it as the sequence length
                    seq_dim_size = pad_full_seq_to or seq_dim_size
                else:
                    data_iterator = batch.make_microbatch_iterator(mbs)
                    data_iterator_len = local_gbs // mbs

                # Track total microbatches for MoE aux-loss averaging
                total_num_microbatches += int(data_iterator_len)

                rerun_state_machine = get_rerun_state_machine()
                while rerun_state_machine.should_run_forward_backward(data_iterator):
                    # Set grad to zero.
                    self.model.zero_grad_buffer()
                    self.optimizer.zero_grad()

                    # Forward pass.
                    forward_backward_func = get_forward_backward_func()
                    losses_reduced = forward_backward_func(
                        forward_step_func=partial(
                            forward_step,
                            self.mcore_state,
                            global_valid_seqs,
                            global_valid_toks,
                            pack_sequences=pack_seqs,
                            seq_length_key=seqlen_key,
                            pad_individual_seqs_to_multiple_of=pad_factor,
                            pad_packed_seq_to_multiple_of=pad_packed_seq_to_multiple_of,
                            pad_full_seq_to=pad_full_seq_to,
                            defer_fp32_logits=self.defer_fp32_logits,
                        ),
                        data_iterator=data_iterator,
                        model=self.model,
                        num_microbatches=data_iterator_len,
                        seq_length=seq_dim_size,
                        micro_batch_size=mbs,
                        decoder_seq_length=seq_dim_size,
                        forward_only=eval_mode,
                        do_not_average_loss=True,
                    )

                # Empty unused memory.
                if self.cfg["megatron_cfg"]["empty_unused_memory_level"] >= 1:
                    torch.cuda.empty_cache()

                # Update parameters.
                if not eval_mode:
                    update_successful, grad_norm, num_zeros_in_grad = (
                        self.optimizer.step()
                    )
                else:
                    update_successful, grad_norm, num_zeros_in_grad = (True, 0.0, 0.0)

                # when freezing sub-models we may have a mixture of successful and unsucessful ranks,
                # so we must gather across mp ranks
                update_successful = logical_and_across_model_parallel_group(
                    update_successful
                )
                # grad_norm and num_zeros_in_grad will be None on ranks without trainable params,
                # so we must gather across mp ranks
                grad_norm: float = reduce_max_stat_across_model_parallel_group(
                    grad_norm
                )
                num_zeros_in_grad: float = reduce_max_stat_across_model_parallel_group(
                    num_zeros_in_grad
                )

                if update_successful:
                    skipped_iter = 0
                else:
                    skipped_iter = 1

                # Empty unused memory.
                if self.cfg["megatron_cfg"]["empty_unused_memory_level"] >= 2:
                    torch.cuda.empty_cache()

                if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    # keep all microbatch metrics to be normalized later
                    gb_loss_metrics = []
                    mb_losses = []
                    for x in losses_reduced:
                        loss_metrics = {}
                        for k in x.keys():
                            loss_metrics[k] = x[k] / num_global_batches
                        gb_loss_metrics.append(loss_metrics)
                        curr_lr = self.scheduler.get_lr(self.optimizer.param_groups[0])
                        curr_wd = self.scheduler.get_wd()
                        loss_metrics["lr"] = curr_lr
                        loss_metrics["wd"] = curr_wd
                        loss_metrics["global_valid_seqs"] = global_valid_seqs.item()
                        loss_metrics["global_valid_toks"] = global_valid_toks.item()
                        mb_losses.append(loss_metrics["loss"])

                    torch.distributed.broadcast_object_list(
                        [gb_loss_metrics],
                        src=get_pipeline_model_parallel_last_rank(),
                        group=get_pipeline_model_parallel_group(),
                    )
                else:
                    loss_metrics = [None]  # type: ignore
                    torch.distributed.broadcast_object_list(
                        loss_metrics,
                        src=get_pipeline_model_parallel_last_rank(),
                        group=get_pipeline_model_parallel_group(),
                    )
                    gb_loss_metrics = loss_metrics[0]
                    mb_losses = [x["loss"] for x in gb_loss_metrics]

                all_mb_metrics.extend(gb_loss_metrics)
                losses.append(torch.tensor(mb_losses).sum().item())

        if not eval_mode:
            # take one LR step every rollout batch
            # we need to scale the step by gbs to counteract the fact that NeMo automatically
            # scales lr_warmup_steps by gbs during init
            self.scheduler.step(increment=gbs)

        # Aggregate metrics across all microbatches
        mb_metrics = defaultdict(list)
        for m in all_mb_metrics:
            for k, v in m.items():
                mb_metrics[k].append(v)

        with torch.no_grad():
            global_loss = torch.tensor(losses, device="cuda")
            torch.distributed.all_reduce(
                global_loss,
                op=torch.distributed.ReduceOp.SUM,
                group=parallel_state.get_data_parallel_group(),
            )

        metrics = {
            "global_loss": global_loss.cpu(),
            "rank": torch.distributed.get_rank(),
            "gpu_name": torch.cuda.get_device_name(),
            "model_dtype": self.dtype,
            "all_mb_metrics": dict(mb_metrics),
            "grad_norm": torch.tensor([grad_norm]),
        }
        # Collect MoE aux metrics averaged across microbatches
        num_moe_experts = getattr(self.model.config, "num_moe_experts", None)
        if num_moe_experts is not None and num_moe_experts > 1:
            moe_loss_scale = 1.0 / max(1, total_num_microbatches)
            moe_metrics = get_moe_metrics(
                loss_scale=moe_loss_scale,
                per_layer_logging=self.cfg["megatron_cfg"]["moe_per_layer_logging"],
            )
            if moe_metrics:
                metrics["moe_metrics"] = moe_metrics
        return metrics

    @wrap_with_nvtx_name("megatron_policy_worker/get_logprobs")
    def get_logprobs(
        self, *, data: BatchedDataDict[Any], micro_batch_size: Optional[int] = None
    ) -> BatchedDataDict[LogprobOutputSpec]:
        """Get the logprobs of the model for a batch of data.

        Uses the configured logprob_batch_size to do microbatching.
        Input data is assumed to be right-padded. The method internally converts to
        left-padded format for computation, and returns outputs in right-padded format.
        If micro_batch_size is provided, it will be used instead of the configured
        logprob_batch_size.

        Returns:
          a BatchedDataDict with key "logprobs" and shape [batch_size, sequence_length].
          We use the convention that the logprob of the first token is 0 so that the sequence length is maintained.
          The logprob of input token i is specified at position i in the output logprobs tensor.
        """
        no_grad = torch.no_grad()
        no_grad.__enter__()
        logprob_batch_size = (
            micro_batch_size
            if micro_batch_size is not None
            else self.cfg["logprob_batch_size"]
        )

        # dim 1 is always assumed to be the sequence dim, sanity check this here
        sequence_dim = 1
        input_seq_dim_size = data["input_ids"].shape[sequence_dim]
        for k, v in data.items():
            if torch.is_tensor(v) and len(v.shape) > 1:
                assert v.shape[sequence_dim] == input_seq_dim_size, (
                    f"Dim 1 must be the sequence dim, expected dim 1={input_seq_dim_size} but got shape {v.shape}"
                )

        self.model.eval()

        pp_seq_dim_size = input_seq_dim_size
        pp_grp = get_pipeline_model_parallel_group()
        pad_factor = 1
        pad_packed_seq_to_multiple_of = 1
        pad_full_seq_to = None
        if self.cfg["sequence_packing"]["enabled"]:
            _, seq_dim_size = data.get_microbatch_iterator_for_packable_sequences_len()
            (
                pad_factor,
                pad_packed_seq_to_multiple_of,
                pad_full_seq_to,
            ) = _get_pack_sequence_parameters_for_megatron(
                self.cfg["megatron_cfg"],
                seq_dim_size,
            )
            pp_seq_dim_size = pad_full_seq_to or pp_seq_dim_size

        def forward_step_fn(
            data_iterator: Iterator[BatchedDataDict[Any]], model: GPTModel
        ):
            nonlocal pad_full_seq_to, pad_packed_seq_to_multiple_of, pad_factor
            data_dict = next(data_iterator).to("cuda")
            if self.cfg["sequence_packing"]["enabled"]:
                original_seq_length = data_dict["input_ids"].shape[1]
                cp_size = self.cfg["megatron_cfg"]["context_parallel_size"]
                cp_rank = get_context_parallel_rank()
                (
                    input_ids,
                    input_ids_cp_sharded,
                    packed_seq_params,
                    cu_seqlens,
                    cu_seqlens_padded,
                ) = _pack_sequences_for_megatron(
                    data_dict["input_ids"].clone(),
                    data_dict["input_lengths"],
                    pad_individual_seqs_to_multiple_of=pad_factor,
                    pad_packed_seq_to_multiple_of=pad_packed_seq_to_multiple_of,
                    pad_packed_seq_to=pad_full_seq_to,
                    cp_rank=cp_rank,
                    cp_size=cp_size,
                )
                attention_mask, position_ids = None, None
                unpacked_input_ids = data_dict["input_ids"]
            else:
                input_ids = data_dict["input_ids"]
                input_ids_cp_sharded = input_ids
                attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
                    data=input_ids,
                    eod_token=0,  # used for loss_mask, which we don't use
                    pad_token=0,  # used for loss_mask, which we don't use
                    reset_position_ids=False,
                    reset_attention_mask=False,
                    eod_mask_loss=False,
                    pad_mask_loss=False,
                )
                packed_seq_params = None
                unpacked_input_ids = input_ids

            multimodal_data = data_dict.get_multimodal_dict(
                as_tensors=True, device=input_ids.device
            )
            if len(multimodal_data) > 0:
                position_ids = None

            additional_kwargs = {}
            # Mamba models currently do not support packed_seq_params
            if packed_seq_params is not None:
                additional_kwargs["packed_seq_params"] = packed_seq_params

            if self.defer_fp32_logits:
                additional_kwargs["fp32_output"] = False

            output_tensor = model(
                input_ids=input_ids_cp_sharded,
                position_ids=position_ids,
                attention_mask=attention_mask,
                **multimodal_data,
                **additional_kwargs,
            )

            # Apply temperature scaling to logits for training
            # This matches the dtensor worker's _apply_temperature_scaling in the train method
            if "generation" in self.cfg and self.cfg["generation"] is not None:
                output_tensor.div_(self.cfg["generation"]["temperature"])

            def collection_fn(output_tensor):
                stc = time.time()
                tp_grp = get_tensor_model_parallel_group()
                tp_rank = get_tensor_model_parallel_rank()
                logprob_chunk_size = self.cfg.get("logprob_chunk_size", None)
                if self.cfg["sequence_packing"]["enabled"]:
                    token_logprobs = from_parallel_logits_to_logprobs_packed_sequences(
                        output_tensor,
                        target=input_ids,
                        cu_seqlens_padded=cu_seqlens_padded,
                        unpacked_seqlen=original_seq_length,
                        vocab_start_index=tp_rank * output_tensor.shape[-1],
                        vocab_end_index=(tp_rank + 1) * output_tensor.shape[-1],
                        group=tp_grp,
                        inference_only=True,
                        cp_group=get_context_parallel_group(),
                        chunk_size=logprob_chunk_size,
                    )
                else:
                    token_logprobs = from_parallel_logits_to_logprobs(
                        output_tensor,
                        target=unpacked_input_ids,
                        vocab_start_index=tp_rank * output_tensor.shape[-1],
                        vocab_end_index=(tp_rank + 1) * output_tensor.shape[-1],
                        tp_group=tp_grp,
                        inference_only=True,
                        chunk_size=logprob_chunk_size,
                    )

                # Prepend 0 logprob for first token to maintain same sequence length as input
                token_logprobs = torch.cat(
                    [torch.zeros_like(token_logprobs[:, :1]), token_logprobs], dim=1
                )
                return torch.tensor(0.0, device=token_logprobs.device), {
                    "logprobs": token_logprobs
                }

            return output_tensor, collection_fn

        if self.cfg["dynamic_batching"]["enabled"]:
            mb_iterator = data.make_microbatch_iterator_with_dynamic_shapes()
            data_iterator_len = data.get_microbatch_iterator_dynamic_shapes_len()
            micro_batch_size = logprob_batch_size
        elif self.cfg["sequence_packing"]["enabled"]:
            mb_iterator = data.make_microbatch_iterator_for_packable_sequences()
            data_iterator_len, _ = (
                data.get_microbatch_iterator_for_packable_sequences_len()
            )
            micro_batch_size = 1
        else:
            mb_iterator = data.make_microbatch_iterator(logprob_batch_size)
            data_iterator_len = max(1, data.size // logprob_batch_size)
            micro_batch_size = logprob_batch_size

        forward_backward_func = get_forward_backward_func()
        list_of_logprobs = forward_backward_func(
            forward_step_func=forward_step_fn,
            data_iterator=mb_iterator,
            model=self.model,
            num_microbatches=data_iterator_len,
            seq_length=pp_seq_dim_size,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=pp_seq_dim_size,
            forward_only=True,
        )
        if is_pipeline_last_stage(ignore_virtual=True):
            all_log_probs_padded = []
            all_logprobs = [l["logprobs"] for l in list_of_logprobs]
            for lp in all_logprobs:
                padding_needed = input_seq_dim_size - lp.shape[1]
                if padding_needed > 0:
                    lp = torch.nn.functional.pad(
                        lp, (0, padding_needed), mode="constant", value=0.0
                    )
                all_log_probs_padded.append(lp)

            logprobs = torch.cat(all_log_probs_padded, dim=0)
            # broadcast logprobs to first pp rank
            broadcast_tensor(logprobs, torch.distributed.get_rank(), pp_grp)
        else:
            logprobs = broadcast_tensor(
                None, get_pipeline_model_parallel_last_rank(), pp_grp
            )

        no_grad.__exit__(None, None, None)
        return BatchedDataDict[LogprobOutputSpec](logprobs=logprobs).to("cpu")

    @contextmanager
    def use_reference_model(self):
        """Context manager that temporarily swaps the reference model and active model.

        On entry: Moves model to CPU, moves reference_model to CUDA. Swaps the references
        On exit: Restores original references and re-flips cuda/cpu
        """
        ## disable overlap param gather when swapping weights
        if self.should_disable_forward_pre_hook:
            self.disable_forward_pre_hook()

        with torch.no_grad():
            try:
                # Save original references
                model_state_dict = {}
                for name, item in self.model.state_dict().items():
                    if isinstance(item, torch.Tensor):
                        item = item.detach().to(
                            device="cpu", non_blocking=True, copy=True
                        )
                    model_state_dict[name] = item

                self.model.load_state_dict(self.reference_state_dict, strict=True)
                # for name, item in self.reference_state_dict.items():
                # if isinstance(item, torch.Tensor):
                # self.model.state_dict()[name] = item.detach().to(device="cuda", non_blocking=True, copy=True)

                if self.cfg["megatron_cfg"]["empty_unused_memory_level"] >= 1:
                    gc.collect()
                    torch.cuda.empty_cache()

                # - self.model is the original reference_model, now on CUDA
                # - self.reference_model is the original model, now on CPU
                yield

            finally:
                # Restore original references and device placement
                self.model.load_state_dict(model_state_dict, strict=True)
                # for name, item in model_state_dict.items():
                # if isinstance(item, torch.Tensor):
                # item = item.detach().to(device="cuda", non_blocking=True, copy=True)
                # self.model.state_dict()[name] = item

                if self.cfg["megatron_cfg"]["empty_unused_memory_level"] >= 1:
                    gc.collect()
                    torch.cuda.empty_cache()

                ## re-enable overlap param gather after weight swap
                if self.should_disable_forward_pre_hook:
                    self.enable_forward_pre_hook()

    @wrap_with_nvtx_name("megatron_policy_worker/get_topk_logits")
    def get_topk_logits(
        self,
        *,
        data: BatchedDataDict[GenerationDatumSpec],
        k: int,
        micro_batch_size: Optional[int] = None,
    ):
        """Get the top-k logits and indices for a batch of data.

        The major difference from get_logprobs is that we compute top-k logits and indices for each position in the sequence.

        Returns:
            BatchedDataDict containing:
                - topk_logits: Tensor of top-k logits for each position in the sequence
                - topk_indices: Tensor of top-k indices for each position in the sequence
        """
        no_grad = torch.no_grad()
        no_grad.__enter__()

        logprob_batch_size = (
            micro_batch_size
            if micro_batch_size is not None
            else self.cfg["logprob_batch_size"]
        )

        sequence_dim = 1
        input_seq_dim_size = data["input_ids"].shape[sequence_dim]
        # Avoid shadowing the function argument `k` by using a distinct variable name
        for tensor_name, v in data.items():
            if torch.is_tensor(v) and len(v.shape) > 1:
                assert v.shape[sequence_dim] == input_seq_dim_size, (
                    f"Tensor {tensor_name} must have sequence dimension {sequence_dim} of size {input_seq_dim_size}, but got shape {v.shape}"
                )

        self.model.eval()

        pp_seq_dim_size = input_seq_dim_size
        pp_grp = get_pipeline_model_parallel_group()

        pad_factor = 1
        pad_packed_seq_to_multiple_of = 1
        pad_full_seq_to = None
        if self.cfg["sequence_packing"]["enabled"]:
            _, seq_dim_size = data.get_microbatch_iterator_for_packable_sequences_len()
            (
                pad_factor,
                pad_packed_seq_to_multiple_of,
                pad_full_seq_to,
            ) = _get_pack_sequence_parameters_for_megatron(
                self.cfg["megatron_cfg"],
                seq_dim_size,
            )
            pp_seq_dim_size = pad_full_seq_to or pp_seq_dim_size

        def forward_step_fn(
            data_iterator: Iterator[BatchedDataDict[Any]], model: GPTModel
        ):
            nonlocal pad_full_seq_to, pad_packed_seq_to_multiple_of, pad_factor
            data_dict = next(data_iterator).to("cuda")

            pack = self.cfg["sequence_packing"]["enabled"]
            if pack:
                original_seq_length = data_dict["input_ids"].shape[1]
                cp_size = self.cfg["megatron_cfg"]["context_parallel_size"]
                cp_rank = get_context_parallel_rank()

                (
                    input_ids_unpacked,
                    input_ids_cp_sharded,
                    packed_seq_params,
                    cu_seqlens,
                    cu_seqlens_padded,
                ) = _pack_sequences_for_megatron(
                    data_dict["input_ids"].clone(),
                    data_dict["input_lengths"],
                    pad_individual_seqs_to_multiple_of=pad_factor,
                    pad_packed_seq_to_multiple_of=pad_packed_seq_to_multiple_of,
                    pad_packed_seq_to=pad_full_seq_to,
                    cp_rank=cp_rank,
                    cp_size=cp_size,
                )
                attention_mask, position_ids = None, None
                seq_lengths = data_dict["input_lengths"]
                unpacked_seqlen = original_seq_length
            else:
                input_ids_cp_sharded = data_dict["input_ids"]
                attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
                    data=input_ids_cp_sharded,
                    eod_token=0,
                    pad_token=0,
                    reset_position_ids=False,
                    reset_attention_mask=False,
                    eod_mask_loss=False,
                    pad_mask_loss=False,
                )
                packed_seq_params = None

            multimodal_data = data_dict.get_multimodal_dict(
                as_tensors=True, device=input_ids_cp_sharded.device
            )
            if len(multimodal_data) > 0:
                position_ids = None

            additional_kwargs = {}
            if packed_seq_params is not None:
                additional_kwargs["packed_seq_params"] = packed_seq_params

            output_tensor = model(
                input_ids=input_ids_cp_sharded,
                position_ids=position_ids,
                attention_mask=attention_mask,
                **additional_kwargs,
                **multimodal_data,
            )

            if "generation" in self.cfg and self.cfg["generation"] is not None:
                output_tensor.div_(self.cfg["generation"]["temperature"])

            def collection_fn(_):
                # Only the last PP stage produces final logits/top-k; earlier stages return empty
                # if not is_pipeline_last_stage(ignore_virtual=True):
                # return output_tensor.new_zeros(()), {}

                tp_grp = get_tensor_model_parallel_group()
                tp_rank = get_tensor_model_parallel_rank()
                vocab_shard_size = output_tensor.shape[-1]
                vocab_start_index = tp_rank * vocab_shard_size

                chunk_size = None
                if "logprob_chunk_size" in self.cfg:
                    chunk_size = self.cfg["logprob_chunk_size"]

                topk_vals_local, topk_idx_local = distributed_vocab_topk(
                    output_tensor,
                    k,
                    tp_grp,
                    vocab_start_index=vocab_start_index,
                    vocab_end_index=vocab_start_index + vocab_shard_size,
                    chunk_size=chunk_size,
                )

                if self.cfg["megatron_cfg"]["context_parallel_size"] > 1:
                    cp_grp = get_context_parallel_group()
                    if pack:
                        # Per-sequence CP allgather following packed-sequence logic
                        batch_size = data_dict["input_ids"].shape[0]
                        total_packed_len = int(cu_seqlens_padded[-1].item())

                        topk_vals_full = torch.zeros(
                            (1, total_packed_len, k),
                            dtype=topk_vals_local.dtype,
                            device=topk_vals_local.device,
                        )
                        topk_idx_full = torch.zeros(
                            (1, total_packed_len, k),
                            dtype=topk_idx_local.dtype,
                            device=topk_idx_local.device,
                        )

                        for i in range(batch_size):
                            start_idx = int(cu_seqlens_padded[i].item())
                            end_idx = int(cu_seqlens_padded[i + 1].item())
                            if end_idx > start_idx:
                                local_vals_slice = topk_vals_local[
                                    :, start_idx // cp_size : end_idx // cp_size, :
                                ]
                                local_idx_slice = topk_idx_local[
                                    :, start_idx // cp_size : end_idx // cp_size, :
                                ]
                                gathered_vals = allgather_cp_sharded_tensor(
                                    local_vals_slice, cp_grp, seq_dim=1
                                )
                                gathered_idx = allgather_cp_sharded_tensor(
                                    local_idx_slice, cp_grp, seq_dim=1
                                )
                                # Some kernels may return [X, Y, k] where X*Y = (end_idx - start_idx).
                                # Flatten leading dims and reshape to [1, expected_len, k] to match target.
                                expected_len = end_idx - start_idx
                                if (
                                    gathered_vals.dim() == 3
                                    and gathered_vals.shape[1] != expected_len
                                ):
                                    gathered_vals = gathered_vals.reshape(
                                        1, expected_len, gathered_vals.shape[-1]
                                    )
                                if (
                                    gathered_idx.dim() == 3
                                    and gathered_idx.shape[1] != expected_len
                                ):
                                    gathered_idx = gathered_idx.reshape(
                                        1, expected_len, gathered_idx.shape[-1]
                                    )
                                topk_vals_full[:, start_idx:end_idx, :] = gathered_vals
                                topk_idx_full[:, start_idx:end_idx, :] = gathered_idx
                    else:
                        # Sequence packing must be enabled when CP > 1
                        raise RuntimeError(
                            "Context Parallelism (CP>1) requires sequence packing to be enabled."
                        )
                else:
                    topk_vals_full = topk_vals_local
                    topk_idx_full = topk_idx_local

                if pack:
                    batch_size = data_dict["input_ids"].shape[0]
                    out_vals = torch.zeros(
                        (batch_size, unpacked_seqlen, k),
                        dtype=topk_vals_full.dtype,
                        device=topk_vals_full.device,
                    )
                    out_idx = torch.zeros(
                        (batch_size, unpacked_seqlen, k),
                        dtype=topk_idx_full.dtype,
                        device=topk_idx_full.device,
                    )
                    for i in range(batch_size):
                        seq_len = int(seq_lengths[i].item())
                        start_idx = int(cu_seqlens_padded[i].item())
                        if seq_len > 0:
                            out_vals[i, :seq_len, :] = topk_vals_full[
                                0, start_idx : start_idx + seq_len, :
                            ]
                            out_idx[i, :seq_len, :] = topk_idx_full[
                                0, start_idx : start_idx + seq_len, :
                            ]
                    return output_tensor.new_zeros(()), {
                        "topk_logits": out_vals,
                        "topk_indices": out_idx,
                    }
                else:
                    return output_tensor.new_zeros(()), {
                        "topk_logits": topk_vals_full,
                        "topk_indices": topk_idx_full,
                    }

            return output_tensor, collection_fn

        if self.cfg["dynamic_batching"]["enabled"]:
            mb_iterator = data.make_microbatch_iterator_with_dynamic_shapes()
            data_iterator_len = data.get_microbatch_iterator_dynamic_shapes_len()
            micro_batch = logprob_batch_size
        elif self.cfg["sequence_packing"]["enabled"]:
            mb_iterator = data.make_microbatch_iterator_for_packable_sequences()
            data_iterator_len, _ = (
                data.get_microbatch_iterator_for_packable_sequences_len()
            )
            micro_batch = 1
        else:
            mb_iterator = data.make_microbatch_iterator(logprob_batch_size)
            data_iterator_len = max(1, data.size // logprob_batch_size)
            micro_batch = logprob_batch_size

        forward_backward_func = get_forward_backward_func()
        list_of_outputs = forward_backward_func(
            forward_step_func=forward_step_fn,
            data_iterator=mb_iterator,
            model=self.model,
            num_microbatches=data_iterator_len,
            seq_length=pp_seq_dim_size,
            micro_batch_size=micro_batch,
            decoder_seq_length=pp_seq_dim_size,
            forward_only=True,
        )

        if is_pipeline_last_stage(ignore_virtual=True):
            logits_chunks = []
            indices_chunks = []
            for out in list_of_outputs:
                tk = out["topk_logits"]
                ti = out["topk_indices"]
                pad_len = input_seq_dim_size - tk.shape[1]
                if pad_len > 0:
                    tk = torch.nn.functional.pad(tk, (0, 0, 0, pad_len), value=0.0)
                    ti = torch.nn.functional.pad(ti, (0, 0, 0, pad_len), value=0)
                logits_chunks.append(tk)
                indices_chunks.append(ti)

            topk_logits = torch.cat(logits_chunks, dim=0)
            topk_indices = torch.cat(indices_chunks, dim=0)

            topk_logits = broadcast_tensor(
                topk_logits, torch.distributed.get_rank(), pp_grp
            )
            topk_indices = broadcast_tensor(
                topk_indices, torch.distributed.get_rank(), pp_grp
            )
        else:
            last_pp_rank = get_pipeline_model_parallel_last_rank()
            topk_logits = broadcast_tensor(None, last_pp_rank, pp_grp)
            topk_indices = broadcast_tensor(None, last_pp_rank, pp_grp)

        no_grad.__exit__(None, None, None)
        return BatchedDataDict.from_batches(
            [{"topk_logits": topk_logits.cpu(), "topk_indices": topk_indices.cpu()}]
        )

    @torch.no_grad()
    @wrap_with_nvtx_name("megatron_policy_worker/generate")
    def generate(
        self, *, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> BatchedDataDict[GenerationOutputSpec]:
        """Generate a batch of data using huggingface framework generation.

        Args:
            data: BatchedDataDict containing input_ids and input_lengths tensors
        Returns:
            BatchedDataDict conforming to GenerationOutputSpec:
                - output_ids: input + generated token IDs
                - logprobs: Log probabilities for each token
                - generation_lengths: Lengths of each response
        """
        # 512 bATCH SIZE (200 tokens)
        no_grad = torch.no_grad()
        no_grad.__enter__()
        self.model.config.flash_decode = False
        if self.should_disable_forward_pre_hook:
            self.model = self.move_model(
                self.model, "cuda", move_params=True, move_grads=False
            )
        # Verify input is right padded
        assert isinstance(data, BatchedDataDict), (
            f"data must be a BatchedDataDict, got type: {type(data)}"
        )
        assert "input_ids" in data and "input_lengths" in data, (
            f"input_ids and input_lengths must be present in the BatchedDataDict, got keys: {data.keys()}"
        )
        is_right_padded, error_msg = verify_right_padding(
            data, pad_value=self.tokenizer.pad_token_id
        )
        if not is_right_padded:
            warnings.warn(
                f"Input to Megatron Generation worker is not properly right-padded: {error_msg}"
            )

        model_cfg = self.megatron_cfg.model
        inference_wrapper_config = InferenceWrapperConfig(
            hidden_size=model_cfg.hidden_size,
            inference_batch_times_seqlen_threshold=1000000,
            fp32_residual_connection=model_cfg.fp32_residual_connection,
            params_dtype=model_cfg.params_dtype,
            padded_vocab_size=self.final_padded_vocab_size,  # Use the potentially updated value
            inference_max_seq_length=self.cfg["generation"]["max_new_tokens"],  # type: ignore
            inference_max_requests=self.cfg["generation_batch_size"],
        )

        from megatron.core.inference.contexts.dynamic_context import (
            DynamicInferenceContext,
        )
        from megatron.core.inference.engines.dynamic_engine import (
            DynamicInferenceEngine,
        )
        from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
            GPTInferenceWrapper,
        )
        from megatron.core.inference.sampling_params import SamplingParams

        mcore_generation_config = self.cfg["generation"]["mcore_generation_config"]
        buffer_size_gb = mcore_generation_config.get("buffer_size_gb", 20)

        num_cuda_graphs = mcore_generation_config.get("num_cuda_graphs", 16)
        block_size_tokens = mcore_generation_config.get("block_size_tokens", 256)
        use_cuda_graphs_for_non_decode_steps = mcore_generation_config.get(
            "use_cuda_graphs_for_non_decode_steps", True
        )
        enable_chunked_prefill = mcore_generation_config.get(
            "enable_chunked_prefill", True
        )
        unified_memory_level = mcore_generation_config.get("unified_memory_level", 0)
        buffer_guaranteed_fraction = mcore_generation_config.get(
            "buffer_guaranteed_fraction", 0.1
        )
        max_tokens = mcore_generation_config.get("max_tokens", 16384)

        # Get the underlying language module (not the DDP wrapper) for toggle_cuda_graphs.
        # This follows the pattern from megatron/rl/rl_utils.py megatron_rl_inference_mode().
        # The DDP wrapper can interfere with CUDA graph state management.
        lang_module = (
            self.model.module.module
            if hasattr(self.model.module, "module")
            else self.model.module
        )

        # Enable CUDA graphs for inference using toggle_cuda_graphs on the underlying module.
        # reset_cuda_graphs=True ensures fresh CUDA graphs are created for each generate() call.
        

        toggle_cuda_graphs(lang_module, set_to="local", reset_cuda_graphs=True)

        model_config = self.model.config
        model_config.cuda_graph_impl = "local"

        dynamic_context = DynamicInferenceContext(
            params_dtype=inference_wrapper_config.params_dtype,
            num_layers=model_config.num_layers,
            kv_channels=model_config.kv_channels,
            num_attention_heads=model_config.num_query_groups,
            max_sequence_length=self.cfg["generation"]["max_new_tokens"],
            buffer_guaranteed_fraction=buffer_guaranteed_fraction,
            buffer_size_gb=buffer_size_gb,
            materialize_only_last_token_logits=False,
            num_cuda_graphs=num_cuda_graphs,
            block_size_tokens=block_size_tokens,
            tensor_model_parallel_size=self.cfg["megatron_cfg"][
                "tensor_model_parallel_size"
            ],
            use_cuda_graphs_for_non_decode_steps=use_cuda_graphs_for_non_decode_steps,
            use_flashinfer_fused_rope=False,
            unified_memory_level=unified_memory_level,
            max_tokens_override=max_tokens,
        )
        inference_wrapped_model = GPTInferenceWrapper(
            self.model, inference_wrapper_config, dynamic_context
        )

        inference_wrapped_model.prep_model_for_inference()
        # Set pipeline parallel flag
        inference_wrapped_model.model_is_pipeline_parallel = (
            self.cfg["megatron_cfg"]["pipeline_model_parallel_size"] > 1
        )

        text_generation_controller = TextGenerationController(
            inference_wrapped_model=inference_wrapped_model,
            tokenizer=self.megatron_tokenizer,
        )

        # Calculate seed based on node and rank to ensure reproducibility across workers
        local_rank = torch.cuda.current_device()  # Local GPU index on the node
        num_gpus_per_node = torch.cuda.device_count()
        node_idx = self.rank // num_gpus_per_node if num_gpus_per_node > 0 else 0
        seed = (node_idx * 1024) + local_rank

        # New API: DynamicInferenceEngine has additional parameters
        dynamic_engine = DynamicInferenceEngine(
            text_generation_controller,
            dynamic_context,
            enable_cuda_graph=True,
            random_seed=seed,
            track_paused_request_events=False,
            enable_chunked_prefill=enable_chunked_prefill,
            inference_logging_step_interval=0,
        )

        # Handle None values for top_k - convert to integer as required by Megatron
        top_k_cfg = self.cfg["generation"]["top_k"]
        top_k_val = 1 if greedy else (int(top_k_cfg) if top_k_cfg is not None else 0)

        top_p_cfg = self.cfg["generation"]["top_p"]
        top_p_val = (
            0.0 if greedy else (float(top_p_cfg) if top_p_cfg is not None else 0.0)
        )

        # New API: SamplingParams now includes termination_id and uses num_tokens_total
        sampling_params = SamplingParams(
            temperature=self.cfg["generation"]["temperature"] if not greedy else 0,
            top_k=top_k_val,
            top_p=top_p_val,
            skip_prompt_log_probs=False,
            return_log_probs=True,
            num_tokens_total=self.cfg["generation"]["max_new_tokens"],
            num_tokens_to_generate=None,
            termination_id=self.megatron_tokenizer.eod,
        )

        input_ids = data["input_ids"]
        prompt_tokens_tensor = input_ids.cuda()
        prompt_lengths_tensor = data["input_lengths"]
        request_id = 0

        # New API: add_request now takes sampling_params as a parameter
        for p, prompt_len in zip(
            prompt_tokens_tensor, prompt_lengths_tensor, strict=True
        ):
            dynamic_engine.add_request(
                request_id,
                p[:prompt_len],
                sampling_params=sampling_params,
            )
            request_id += 1

        result = []
        import time
        start = time.time()
        while dynamic_engine.has_unfinished_requests():
            result_step = dynamic_engine.step_modern(verbose=False)
            finished_requests = result_step.get("finished_requests", [])
            for finished_request in finished_requests:
                result.append(finished_request)
        end = time.time()
        print(f"SHAN : Time taken: {end - start} seconds")

        # Sort results by request_id to maintain original batch order
        result.sort(key=lambda x: x.request_id)

        out = {
            "tokens": [x.prompt_tokens.tolist() + x.generated_tokens for x in result],
            "logprobs": [x.prompt_log_probs + x.generated_log_probs for x in result],
        }

        input_lengths = data["input_lengths"]
        # pad the out "tokens" and "logprobs" and make them into tensors from lists
        batch_size = data["input_ids"].size(0)
        max_gen_seq_len = max([len(x.generated_tokens) for x in result])
        padded_input_length = input_ids.size(1)

        max_seq_len = padded_input_length + max_gen_seq_len
        # Create padded tensors for tokens and logprobs
        output_ids_padded = torch.full(
            (batch_size, max_seq_len),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=data["input_ids"].device,
        )

        logprobs_padded = torch.zeros(
            (batch_size, max_seq_len),
            dtype=torch.float,
            device=data["input_ids"].device,
        )

        # Fill in the padded tensors with actual values
        generation_lengths = torch.zeros(
            batch_size, dtype=torch.long, device=data["input_ids"].device
        )
        unpadded_sequence_lengths = torch.zeros(
            batch_size, dtype=torch.long, device=data["input_ids"].device
        )
        for i in range(batch_size):
            seq_len = len(out["tokens"][i])
            output_ids_padded[i, :seq_len] = torch.tensor(
                out["tokens"][i], dtype=torch.long, device=data["input_ids"].device
            )
            generation_lengths[i] = seq_len - input_lengths[i].item()
            unpadded_sequence_lengths[i] = seq_len
            logprob_len = len(out["logprobs"][i])
            logprobs_padded[i, 1 : logprob_len + 1] = torch.tensor(
                out["logprobs"][i],
                dtype=torch.float,
                device=data["input_ids"].device,
            )

        out_dict = {
            "output_ids": output_ids_padded,
            "logprobs": logprobs_padded,
            "generation_lengths": generation_lengths,
            "unpadded_sequence_lengths": unpadded_sequence_lengths,
        }

        self.model.config.flash_decode = False

        # Disable CUDA graphs after inference using toggle_cuda_graphs on the underlying module.
        # This follows the pattern from megatron/rl/rl_utils.py megatron_rl_inference_mode().
        # reset_cuda_graphs=True ensures:
        # 1. cudagraph_manager attributes are deleted from modules
        # 2. _CudagraphGlobalRecord state is reset via delete_cuda_graphs()
        # 3. Model is ready for training (avoids "inference tensor" errors)
        # 4. Next generate() call will create fresh CUDA graphs with new DynamicInferenceContext
        toggle_cuda_graphs(lang_module, set_to="none", reset_cuda_graphs=True)

        # Force release CUDA graph memory to prevent OOM at next step
        torch.cuda.empty_cache()

        no_grad.__exit__(None, None, None)

        return BatchedDataDict.from_batches([out_dict]).to("cpu")

    @torch.no_grad()
    @wrap_with_nvtx_name("megatron_policy_worker/prepare_refit_info")
    def prepare_refit_info(self) -> None:
        """Prepare state dict metadata for weight refitting and IPC streaming."""
        self.refit_param_info_mcore = self._calculate_refit_param_info()

        # Collect tensor metadata for refit / hf side info
        refit_param_info_hf = {}
        # Reuse shared iterator that appends FP8 KV/Q scales when enabled
        for name, tensor in self._iter_params_with_optional_kv_scales():
            refit_param_info_hf[name] = (tensor.shape, tensor.dtype)

        return refit_param_info_hf

    def _calculate_refit_param_info(self) -> list[tuple[str, int]]:
        """Calculate parameter information for refit.

        Each task contains:
        - param_name: Local parameter name without module prefixes
        - mapping: MegatronParamMapping instance for weight transformation
        - pp_rank: Pipeline-parallel rank owning the parameter
        - vp_stage: Virtual-pipeline stage index
        - megatron_module: Reference to Megatron model/submodule
        - param_weight: Target parameter tensor for converted weight

        Returns:
            List of (parameter_name, size_in_bytes) tuples.
        """
        self.refit_conversion_tasks = self.megatron_bridge.get_conversion_tasks(
            [self.model]
        )
        param_info = []

        def calculate_size_in_bytes(param, tp_size, ep_size):
            if param is None:
                # need to broadcast for other pp ranks
                size_in_bytes = None
            else:
                # Calculate size for this parameter
                prec_to_bytes = {
                    torch.bfloat16: 2,
                    torch.float16: 2,
                    torch.float32: 4,
                }
                scale = prec_to_bytes[self.dtype] / prec_to_bytes[param.dtype]
                size_in_bytes = (
                    param.element_size() * param.numel() * tp_size * ep_size * scale
                )

            # Broadcast size_in_bytes across pipeline parallel ranks
            return broadcast_object_across_pp_ranks(size_in_bytes)

        for task in self.refit_conversion_tasks:
            param_info.append(
                (
                    task.param_name,
                    calculate_size_in_bytes(
                        task.param_weight,
                        task.mapping.tp_size,
                        task.mapping.ep_size if task.mapping.is_expert else 1,
                    ),
                )
            )
        return param_info

    def _iter_params_with_optional_kv_scales(
        self,
        kv_scales: Optional[dict[str, float]] = None,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield exported HF parameters and optionally append FP8 KV/Q scale tensors.

        This helper is used by both IPC-based streaming and collective broadcast
        so that the logic for adding KV scales stays consistent in one place.
        """
        base_iter = self.megatron_bridge.export_hf_weights(
            [self.model],
            show_progress=False,
            conversion_tasks=self.refit_conversion_tasks,  # used for metadata caching
        )

        # Yield the original parameters first.
        for name, tensor in base_iter:
            yield name, tensor

        # Check whether FP8 KV cache is enabled.
        use_fp8_kv_cache = False
        if (
            "generation" in self.cfg
            and self.cfg["generation"] is not None
            and self.cfg["generation"]["backend"] == "vllm"
        ):
            generation_cfg = cast(VllmConfig, self.cfg["generation"])
            use_fp8_kv_cache = (
                "vllm_cfg" in generation_cfg
                and "kv_cache_dtype" in generation_cfg["vllm_cfg"]
                and generation_cfg["vllm_cfg"]["kv_cache_dtype"].startswith("fp8")
            )

        if not use_fp8_kv_cache:
            return

        # Append KV (and potentially Q) scale entries to match metadata.
        num_layers = self.megatron_bridge.transformer_config.num_layers
        keys: list[str] = []
        for layer_idx in range(num_layers):
            scale_names = get_vllm_qkv_scale_names(layer_idx)
            keys.extend(scale_names.values())

        for param_name in keys:
            if kv_scales and param_name in kv_scales:
                scale_value = kv_scales[param_name]
            else:
                scale_value = 1.0
            scale_tensor = torch.tensor(
                scale_value, dtype=torch.float32, device="cuda"
            ).reshape(1)
            yield param_name, scale_tensor

    @torch.no_grad()
    @wrap_with_nvtx_name("megatron_policy_worker/stream_weights_via_ipc_zmq")
    def stream_weights_via_ipc_zmq(
        self, buffer_size_bytes: int = 0, kv_scales: Optional[dict[str, float]] = None
    ) -> None:
        """Stream model weights to peer process via ZMQ IPC socket."""
        self.maybe_init_zmq()

        from nemo_rl.models.policy.utils import stream_weights_via_ipc_zmq_impl

        # Use the shared implementation to append optional KV scales.
        stream_weights_via_ipc_zmq_impl(
            params_generator=self._iter_params_with_optional_kv_scales(
                kv_scales=kv_scales
            ),
            buffer_size_bytes=buffer_size_bytes,
            zmq_socket=self.zmq_socket,
            rank=self.rank,
            worker_name=str(self),
        )

    @torch.no_grad()
    def broadcast_weights_for_collective(
        self, kv_scales: Optional[dict[str, float]] = None
    ) -> None:
        """Broadcast the weights for collective communication."""
        # param_iterator will return (name, tensor), we only need tensor.
        packed_broadcast_producer(
            iterator=self._iter_params_with_optional_kv_scales(kv_scales=kv_scales),
            group=self.model_update_group,
            src=0,
            post_iter_func=lambda x: x[1],
        )

    def prepare_for_lp_inference(self):
        self.model = self.move_model(self.model, "cuda", move_grads=False)
        self.model.eval()

        # offload grads to cpu
        self.model = self.move_model(
            self.model, "cpu", move_params=False, move_grads=True
        )  # get rid of grad buffers

        # offload optimizer to cpu
        torch.randn(1).cuda()  # wake up torch allocator
        if (
            hasattr(self, "optimizer")
            and self.optimizer is not None
            and not self.optimizer_cpu_offload
            and self.offload_optimizer_for_logprob
        ):
            self.move_optimizer("cpu")

        gc.collect()
        torch.cuda.empty_cache()

    def prepare_for_training(self, *args, **kwargs):
        # onload models and optimizer state to cuda
        self.model = self.move_model(
            self.model, "cuda", move_grads=True, move_params=True
        )
        self.model.train()

        # Move optimizer state to CUDA if it exists
        # colocated generation will always offload optimizer to cuda before refit
        if (
            hasattr(self, "optimizer")
            and self.optimizer is not None
            and not self.optimizer_cpu_offload
            and (self.offload_optimizer_for_logprob or self.is_generation_colocated)
        ):
            self.move_optimizer("cuda")

        if self.cfg["megatron_cfg"]["empty_unused_memory_level"] >= 1:
            torch.cuda.empty_cache()

    @wrap_with_nvtx_name("megatron_policy_worker/offload_before_refit")
    def offload_before_refit(self):
        """Offload the optimizer and buffers to the CPU."""
        no_grad = torch.no_grad()
        no_grad.__enter__()
        allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
        reserved = torch.cuda.memory_reserved() / (1024**3)  # Convert to GB
        print(
            f"GPU Memory before optimizer offload: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )
        self.model = self.move_model(
            self.model, "cpu", move_params=False, move_grads=True
        )  # get rid of grad buffers
        torch.randn(1).cuda()  # wake up torch allocator
        if (
            hasattr(self, "optimizer")
            and self.optimizer is not None
            and not self.optimizer_cpu_offload
        ):
            self.move_optimizer("cpu")

        gc.collect()
        torch.cuda.empty_cache()

        # Print memory stats after offloading
        allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
        reserved = torch.cuda.memory_reserved() / (1024**3)  # Convert to GB
        print(
            f"GPU Memory after optimizer offload: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )
        no_grad.__exit__(None, None, None)

    @wrap_with_nvtx_name("megatron_policy_worker/offload_after_refit")
    def offload_after_refit(self):
        """Offload as much as possible on the CPU."""
        no_grad = torch.no_grad()
        no_grad.__enter__()
        self.model = self.move_model(self.model, "cpu")
        self.model.eval()
        torch.randn(1).cuda()  # wake up torch allocator
        self.offload_before_refit()  # rerun the old offload function

        allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
        reserved = torch.cuda.memory_reserved() / (1024**3)  # Convert to GB
        print(
            f"GPU Memory after refit complete: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )
        no_grad.__exit__(None, None, None)

    @torch.no_grad()
    def move_model(
        self,
        model: torch.nn.Module,
        device: str,
        move_params: bool = True,
        move_grads: bool = True,
    ) -> torch.nn.Module:
        # move all param and grad buffers to the device
        if isinstance(model, DistributedDataParallel):
            # DDP case
            for buffers in [model.buffers, model.expert_parallel_buffers]:
                for buffer_idx in range(len(buffers)):
                    if device == "cpu":
                        buffers[buffer_idx].offload_to_cpu(
                            move_params=move_params, move_grads=move_grads
                        )
                    elif device == "cuda":
                        buffers[buffer_idx].reload_from_cpu(
                            move_params=move_params, move_grads=move_grads
                        )
                    else:
                        raise ValueError(
                            f"Invalid device: {device}. Only strings 'cpu' and 'cuda' are supported."
                        )
        elif isinstance(model, custom_FSDP):
            if device == "cpu":
                model.param_and_grad_buffer.offload_to_cpu(move_params, move_grads)
            elif device == "cuda":
                model.param_and_grad_buffer.reload_from_cpu(
                    move_params=move_params, move_grads=move_grads
                )
            else:
                raise ValueError(
                    f"Invalid device: {device}. Only strings 'cpu' and 'cuda' are supported."
                )
        else:
            # Ordinary offload case
            if move_params:
                for name, param in model.state_dict().items():
                    new_state_dict = {}
                    for name, item in model.state_dict().items():
                        if isinstance(item, torch.Tensor):
                            item = item.detach().to(
                                device=device, non_blocking=True, copy=True
                            )
                        new_state_dict[name] = item
                    model.load_state_dict(new_state_dict)
        return model

    def move_optimizer(self, device: str):
        # Iterate through the state dictionaries for each parameter group
        if isinstance(self.optimizer, ChainedOptimizer):
            optimizer_state = self.optimizer.state
        else:
            optimizer_state = self.optimizer._get_state()
        for _, state in optimizer_state.items():
            # Iterate through the state items (e.g., momentum, variance) for a parameter
            for k, v in state.items():
                # Check if the item is a tensor
                if torch.is_tensor(v):
                    # Move the tensor to device and update the state dictionary
                    if device == "cpu":
                        if v.is_cuda:
                            state[k] = v.to("cpu")
                    elif device == "cuda":
                        if not v.is_cuda:
                            state[k] = v.to("cuda")
                    else:
                        raise ValueError(
                            f"Invalid device: {device}. Only strings 'cpu' and 'cuda' are supported."
                        )

    def save_checkpoint(
        self,
        weights_path: str,
        optimizer_path: Optional[str] = None,
        **kwargs,
    ):
        """Save a training checkpoint.

        Args:
            weights_path: The specific directory path where the checkpoint will be saved.
            optimizer_path: If not None, optimizer and scheduler states are saved if they exist.
        """
        if not torch.distributed.is_initialized():
            raise RuntimeError(
                "Distributed process group is not initialized. Cannot save checkpoint."
            )

        if self.mcore_state is None or self.model is None:
            raise RuntimeError(
                "Megatron core state or model is not initialized. Cannot save checkpoint."
            )

        original_save_path = self.mcore_state.cfg.checkpoint.save
        # save_dir = os.path.dirname(weights_path)
        release_name = os.path.basename(weights_path)

        try:
            maybe_finalize_async_save(
                self.mcore_state,
                ckpt_cfg=self.mcore_state.cfg.checkpoint,
                blocking=False,
            )
            self.mcore_state.cfg.checkpoint.save = weights_path

            optimizer_to_save = None
            scheduler_to_save = None

            if optimizer_path is not None:
                if self.optimizer is not None:
                    optimizer_to_save = self.optimizer
                if self.scheduler is not None:
                    scheduler_to_save = self.scheduler

            # Ensure model is in eval mode for consistent saving, unless actively training
            # This is a common practice, though NeMo's save might handle this.
            # For safety, if not in training loop, setting to eval.
            is_training = self.model.training
            if not is_training:
                self.model.eval()

            if self.should_disable_forward_pre_hook:
                self.disable_forward_pre_hook()
            save_checkpoint(
                state=self.mcore_state,
                model=[self.model],
                optimizer=optimizer_to_save,
                opt_param_scheduler=scheduler_to_save,
                num_floating_point_operations_so_far=self.mcore_state.train_state.floating_point_operations_so_far,
                checkpointing_context=self.checkpointing_context,
            )
            print(f"Saved checkpoint to {weights_path}")
            maybe_finalize_async_save(
                self.mcore_state,
                ckpt_cfg=self.mcore_state.cfg.checkpoint,
                blocking=True,
                terminate=True,
            )
            if self.should_disable_forward_pre_hook:
                self.enable_forward_pre_hook()

            if not is_training:  # Restore training state if it was changed
                self.model.train()

        except Exception as e:
            print(f"Failed to save checkpoint to {weights_path}: {e}")
            raise
        finally:
            self.mcore_state.cfg.checkpoint.save = original_save_path

    def load_checkpoint(self, weights_path: str, optimizer_path: Optional[str] = None):
        """Load a training checkpoint.

        Args:
            weights_path: The exact directory path from which to load the checkpoint.
            optimizer_path: If not None, attempts to load optimizer and scheduler states
                            if self.optimizer and self.scheduler are initialized.
        """
        raise NotImplementedError(
            "Loading checkpoints outside of the init function is not yet implemented for Megatron policy."
        )

    def check_tensor_parallel_attributes(self) -> dict[str, Any]:
        """Check tensor parallel attributes on model parameters.

        Returns:
            Dictionary containing information about tensor parallel parameters:
            - tp_params: List of parameter names that have tensor_model_parallel=True
            - non_tp_params: List of parameter names that have tensor_model_parallel=False
            - total_params: Total number of parameters checked
            - tp_size: Tensor parallel size from config
        """
        tp_params = []
        non_tp_params = []
        total_params = 0

        for name, param in self.model.named_parameters():
            total_params += 1
            tensor_model_parallel = getattr(param, "tensor_model_parallel", False)

            if tensor_model_parallel:
                tp_params.append(
                    {
                        "name": name,
                        "tensor_model_parallel": tensor_model_parallel,
                        "partition_dim": getattr(param, "partition_dim", None),
                        "partition_stride": getattr(param, "partition_stride", None),
                        "shape": list(param.shape),
                    }
                )
            else:
                non_tp_params.append(
                    {
                        "name": name,
                        "tensor_model_parallel": tensor_model_parallel,
                        "shape": list(param.shape),
                    }
                )

        return {
            "tp_params": tp_params,
            "non_tp_params": non_tp_params,
            "total_params": total_params,
            "tp_size": self.megatron_cfg.model.tensor_model_parallel_size,
        }

    @torch.no_grad()
    def calibrate_qkv_fp8_scales(
        self,
        *,
        data: BatchedDataDict[Any],
        micro_batch_size: Optional[int] = None,
        percentile: float = 99.9,
        margin: float = 1.05,
        include_q: bool = False,
    ) -> dict[str, Any]:
        """One-shot calibration of Q/K/V activation scales (for FP8 KV cache).

        - Captures each layer's `query_key_value` output through forward hooks, splits Q/K/V, and computes percentile amax.
        - In parallel (DP/TP/PP) environments, first computes local percentiles, then takes max across all ranks for conservativeness.
        - By default only returns and saves K/V scales, optionally returns Q.

        Args:
            data: Representative sample batch for calibration, following get_logprobs input conventions.
            micro_batch_size: Micro batch size during calibration; if None, reuses logprob_batch_size.
            percentile: Percentile for amax (e.g. 99.9).
            margin: Margin factor, e.g. 1.05.
            save_path: If provided, rank0 will save results as JSON.
            include_q: Whether to also return Q scale (usually only K/V needed).

        Returns:
            { "format": "fp8", "percentile": float, "margin": float,
              "layers": { layer_name: {"k_scale": float, "v_scale": float[, "q_scale": float] } } }
        """

        # Allow overriding FP8 max for Q, K, V via environment variables for ease of testing.
        # Defaults align with FP8 e4m3 max magnitude.
        # Use different defaults for Q, K, V to adapt to distribution diffefences
        def _get_env_float(name: str, default: float) -> float:
            try:
                val = os.getenv(name, None)
                return float(val) if val is not None and val != "" else default
            except Exception:
                return default

        FP8_MAX_Q = _get_env_float("FP8_MAX_Q", 448.0)
        FP8_MAX_K = _get_env_float("FP8_MAX_K", 448.0)
        FP8_MAX_V = _get_env_float("FP8_MAX_V", 448.0)

        self.model.eval()

        # Record local percentile amax for q/k/v of each layer
        layer_to_samples_q: dict[str, list[float]] = defaultdict(list)
        layer_to_samples_k: dict[str, list[float]] = defaultdict(list)
        layer_to_samples_v: dict[str, list[float]] = defaultdict(list)
        hook_handles = []

        def _extract_layer_key(module_name: str) -> str:
            # Expected format: "module.decoder.layers.<idx>.self_attention.query_key_value"
            m = re.search(r"module\.decoder\.layers\.(\d+)", module_name)
            if m is not None:
                return f"layer_{m.group(1)}"
            return module_name

        # Hook to capture q/k/v after q/k norm and RoPE
        def _pre_hook_builder_core_attention(module_name: str):
            layer_key = _extract_layer_key(module_name)

            def _pre_hook(module, inputs):
                args = inputs if isinstance(inputs, (tuple, list)) else (inputs,)
                if len(args) == 1 and isinstance(args[0], (tuple, list)):
                    args = args[0]
                # Expected first 3 args to be q, k, v (typical signature for Megatron CoreAttention)
                q = args[0]
                k = args[1]
                v = args[2]
                if include_q:
                    layer_to_samples_q[layer_key].append(
                        float(torch.amax(torch.abs(q)).item())
                    )
                layer_to_samples_k[layer_key].append(
                    float(torch.amax(torch.abs(k)).item())
                )
                layer_to_samples_v[layer_key].append(
                    float(torch.amax(torch.abs(v)).item())
                )

            return _pre_hook

        matched_modules = []
        # Try to register forward_pre_hook on core_attention first
        for name, module in self.model.named_modules():
            if "self_attention.core_attention" in name:
                try:
                    handle = module.register_forward_pre_hook(
                        _pre_hook_builder_core_attention(name)
                    )
                    hook_handles.append(handle)
                    matched_modules.append((name, module.__class__.__name__, "pre"))
                except Exception as e:
                    print(
                        f"Error registering pre-hook for qkv scale calibration on {name}: {e}"
                        " Please check if the model is compatible with the current calibration logic. "
                        "The expected module name is 'self_attention.core_attention'."
                    )
                    raise

        # Run a forward pass to trigger hooks (reuse get_logprobs forward path)
        try:
            _ = self.get_logprobs(data=data, micro_batch_size=micro_batch_size)
        finally:
            for h in hook_handles:
                try:
                    h.remove()
                except Exception as e:
                    print(f"Error removing hook for qkv scale calibration: {e}")
                    raise

        # Compute local percentile amax
        def _percentile(values: list[float], p: float) -> float:
            if not values:
                return 0.0
            t = torch.tensor(sorted(values), device="cuda", dtype=torch.float32)
            rank = max(
                0, min(len(values) - 1, int(round((p / 100.0) * (len(values) - 1))))
            )
            return float(t[rank].item())

        local_layer_to_pamax = {}
        for layer_key in set(
            list(layer_to_samples_k.keys())
            + list(layer_to_samples_v.keys())
            + (list(layer_to_samples_q.keys()) if include_q else [])
        ):
            entry = {}
            if include_q:
                entry["q_amax_p"] = _percentile(
                    layer_to_samples_q.get(layer_key, []), percentile
                )
            entry["k_amax_p"] = _percentile(
                layer_to_samples_k.get(layer_key, []), percentile
            )
            entry["v_amax_p"] = _percentile(
                layer_to_samples_v.get(layer_key, []), percentile
            )
            local_layer_to_pamax[layer_key] = entry

        # Merge across all ranks: take maximum of percentile amax (conservative approach)
        world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )
        gathered = [None for _ in range(world_size)] if world_size > 1 else None
        if world_size > 1:
            torch.distributed.all_gather_object(gathered, local_layer_to_pamax)
            merged = defaultdict(dict)
            for d in gathered:  # type: ignore
                if d is None:
                    continue
                for k, v in d.items():
                    dst = merged[k]
                    for kk, vv in v.items():
                        dst[kk] = max(dst.get(kk, 0.0), float(vv))
            layer_to_pamax = dict(merged)
        else:
            layer_to_pamax = local_layer_to_pamax

        # Compute scale (symmetric quantization): scale = pamax / fp8_max
        result_layers = {}
        for layer_key, vals in layer_to_pamax.items():
            out_entry = {}
            if include_q:
                q_scale = (vals.get("q_amax_p", 0.0) * margin) / FP8_MAX_Q
                out_entry["q_scale"] = float(q_scale)
            k_scale = (vals.get("k_amax_p", 0.0) * margin) / FP8_MAX_K
            v_scale = (vals.get("v_amax_p", 0.0) * margin) / FP8_MAX_V
            out_entry["k_scale"] = float(k_scale)
            out_entry["v_scale"] = float(v_scale)
            result_layers[layer_key] = out_entry

        vllm_format_scales = convert_calibration_to_vllm_format(result_layers)

        final_result = {
            "format": "fp8",
            "percentile": percentile,
            "margin": margin,
            "layers": vllm_format_scales,
        }

        # Sync results across all ranks (broadcast rank0's result)
        if world_size > 1:
            if torch.distributed.get_rank() == 0:
                obj_list = [final_result]
                torch.distributed.broadcast_object_list(obj_list, src=0)
                final_result = obj_list[0]
            else:
                obj_list = [None]
                torch.distributed.broadcast_object_list(obj_list, src=0)
                final_result = obj_list[0]  # type: ignore

        return final_result


class CustomFloat16Module(Float16Module):
    """Float 16 Module.

    Attributes:
        config (TransformerConfig): Transformer config
        fp16 (bool) : Specifies if the model runs in fp16 mode
        bf16 (bool) : Specifies if the model runs in bf16 mode

    Args:
        config (TransformerConfig): The transformer config used to initalize the model
    """

    def __init__(self, config: TransformerConfig, module: torch.nn.Module):
        super(CustomFloat16Module, self).__init__(config, module)
        self.re_enable_float32_expert_bias()

    def re_enable_float32_expert_bias(self) -> None:
        """Ensure MoE router expert bias stays in float32 for numerical stability.

        Walks the wrapped module to find MoE routers and invokes the
        `_maintain_float32_expert_bias()` helper which recreates or casts the
        expert bias tensors to float32 as required by Megatron-LM.
        """
        module = self.module
        # Handle VLM models where language model is nested
        if hasattr(module, "language_model"):
            module = module.language_model
        if hasattr(module, "decoder") and hasattr(module.decoder, "layers"):
            for layer in module.decoder.layers:
                mlp = getattr(layer, "mlp", None)
                router = getattr(mlp, "router", None) if mlp is not None else None
                if router is not None and hasattr(
                    router, "_maintain_float32_expert_bias"
                ):
                    router._maintain_float32_expert_bias()

```

```python
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
import gc
import os
import re
import time
import warnings
from collections import defaultdict
from contextlib import AbstractContextManager, contextmanager, nullcontext
from functools import partial
from typing import Any, Iterator, Optional, TypeVar, cast

import ray
import torch
from megatron.bridge import AutoBridge
from megatron.bridge.models.model_provider import get_model
from megatron.bridge.training import fault_tolerance
from megatron.bridge.training.checkpointing import (
    checkpoint_exists,
    init_checkpointing_context,
    load_checkpoint,
    maybe_finalize_async_save,
    save_checkpoint,
)
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    LoggerConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainingConfig,
)
from megatron.bridge.training.initialize import (
    initialize_megatron,
    set_jit_fusion_options,
)
from megatron.bridge.training.optim import setup_optimizer
from megatron.bridge.training.setup import (
    _update_model_config_funcs,
)
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer
from megatron.bridge.training.utils.train_utils import (
    logical_and_across_model_parallel_group,
    reduce_max_stat_across_model_parallel_group,
)
from megatron.bridge.utils.common_utils import get_rank_safe
from megatron.bridge.utils.instantiate_utils import InstantiationMode
from megatron.bridge.utils.vocab_utils import calculate_padded_vocab_size
from megatron.core import parallel_state
from megatron.core.transformer.utils import toggle_cuda_graphs
from megatron.core.distributed import DistributedDataParallel
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import (
    FullyShardedDataParallel as custom_FSDP,
)
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import (
    InferenceWrapperConfig,
)
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.models.gpt import GPTModel
from megatron.core.optimizer import ChainedOptimizer
from megatron.core.parallel_state import (
    get_context_parallel_group,
    get_context_parallel_rank,
    get_pipeline_model_parallel_group,
    get_pipeline_model_parallel_last_rank,
    get_pipeline_model_parallel_world_size,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    is_pipeline_last_stage,
)
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.utils import get_ltor_masks_and_position_ids
from ray.util.queue import Queue
from transformers import PreTrainedTokenizerBase

from nemo_rl.algorithms.interfaces import LossFunction, LossType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.model_utils import (
    allgather_cp_sharded_tensor,
    distributed_vocab_topk,
    from_parallel_logits_to_logprobs,
    from_parallel_logits_to_logprobs_packed_sequences,
)
from nemo_rl.distributed.named_sharding import NamedSharding
from nemo_rl.models.generation.fp8 import (
    convert_calibration_to_vllm_format,
    get_vllm_qkv_scale_names,
)
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationOutputSpec,
    verify_right_padding,
)
from nemo_rl.models.generation.vllm.config import VllmConfig
from nemo_rl.models.megatron.common import (
    _get_pack_sequence_parameters_for_megatron,
    _pack_sequences_for_megatron,
    broadcast_tensor,
    forward_step_arbitrary_loss,
    get_moe_metrics,
)
from nemo_rl.models.megatron.community_import import import_model_from_hf_name
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.interfaces import (
    ColocatablePolicyInterface,
    LogprobOutputSpec,
)
from nemo_rl.models.policy.utils import (
    configure_dynamo_cache,
    get_megatron_checkpoint_dir,
    get_runtime_env_for_policy_worker,
)
from nemo_rl.models.policy.workers.base_policy_worker import AbstractPolicyWorker
from nemo_rl.models.policy.workers.patches import apply_transformer_engine_patch
from nemo_rl.utils.nsys import wrap_with_nvtx_name
from nemo_rl.utils.packed_tensor import packed_broadcast_producer

try:
    from megatron.core.distributed import (
        TorchFullyShardedDataParallel as torch_FSDP,  # noqa: F401 unused-import
    )

    HAVE_FSDP2 = True
except ImportError:
    HAVE_FSDP2 = False

TokenizerType = TypeVar("TokenizerType", bound=PreTrainedTokenizerBase)


def broadcast_object_across_pp_ranks(obj):
    """Broadcast an object across pipeline parallel ranks.

    This utility function handles broadcasting an object from the rank that owns it
    to all other pipeline parallel ranks. If only one rank has the object (non-None),
    it will be broadcast to all other ranks.

    Args:
        obj: The object to broadcast. Can be None on ranks that don't own it.

    Returns:
        The object on all ranks (either the original or the broadcast copy).

    Raises:
        ValueError: If the object doesn't exist on any pipeline parallel rank.
    """
    pp_size = get_pipeline_model_parallel_world_size()
    pp_group = get_pipeline_model_parallel_group()

    if pp_size == 1:
        return obj

    # ------------------------------------------------------------------
    # 1. Gather presence flags from all PP ranks to find the source rank
    # ------------------------------------------------------------------
    has_obj = obj is not None
    obj_flags = [None] * pp_size
    torch.distributed.all_gather_object(obj_flags, has_obj, group=pp_group)

    # ------------------------------------------------------------------
    # 2. Identify the owning rank (the only rank with True flag)
    # ------------------------------------------------------------------
    src_rank = None  # Rank *inside* the PP group
    for rank, flag in enumerate(obj_flags):
        if flag:
            src_rank = rank
            break

    if src_rank is None:
        raise ValueError("Object must exist on at least one PP rank")

    # ------------------------------------------------------------------
    # 3. Broadcast the object from the source rank to all ranks
    # ------------------------------------------------------------------
    # Use broadcast_object_list which is more robust than all_gather_object
    obj_list = [obj]
    pp_ranks = torch.distributed.get_process_group_ranks(pp_group)
    global_src = pp_ranks[src_rank]
    torch.distributed.broadcast_object_list(obj_list, src=global_src, group=pp_group)

    return obj_list[0]


def setup_megatron_model(
    policy_cfg: PolicyConfig,
    cfg: ConfigContainer,
    load_optimizer: bool = True,
    get_embedding_ranks=None,  # TODO @sahilj: What is this?
    get_position_embedding_ranks=None,
):
    state = GlobalState()
    state.cfg = cfg
    # TODO: Freeze state.cfg

    cfg.dist.external_gpu_device_mapping = True
    initialize_megatron(
        cfg=cfg,
        get_embedding_ranks=get_embedding_ranks,
        get_position_embedding_ranks=get_position_embedding_ranks,
    )

    if cfg.ft and cfg.ft.enable_ft_package:
        fault_tolerance.setup(cfg, state)
        fault_tolerance.maybe_setup_simulated_fault(cfg.ft)

    # Set pytorch JIT layer fusion options and warmup JIT functions.
    set_jit_fusion_options(cfg.model, cfg.train.micro_batch_size)

    # Adjust the startup time so it reflects the largest value.
    # This will be closer to what scheduler will see (outside of
    # image ... launches.
    start_time_tensor = torch.tensor(
        [state.start_time], dtype=torch.double, device="cuda"
    )
    torch.distributed.all_reduce(start_time_tensor, op=torch.distributed.ReduceOp.MIN)
    state.start_time = start_time_tensor.item()

    print(
        "time to initialize megatron (seconds): {:.3f}".format(
            time.time() - state.start_time
        )
    )
    torch.distributed.barrier()

    # Context used for persisting some state between checkpoint saves.
    checkpointing_context = init_checkpointing_context(cfg.checkpoint)

    # Tokenizer
    build_tokenizer(
        cfg.tokenizer,
        make_vocab_size_divisible_by=cfg.model.make_vocab_size_divisible_by
        // cfg.model.tensor_model_parallel_size,
        tensor_model_parallel_size=cfg.model.tensor_model_parallel_size,
        trust_remote_code=True,
    )
    assert cfg.model.vocab_size, "vocab size must be specified in model config"

    torch.distributed.barrier()

    pre_wrap_hook = []
    mixed_precision_wrapper = Float16Module
    if policy_cfg["megatron_cfg"]["freeze_moe_router"]:

        def freeze_moe_router(megatron_model):
            if not isinstance(megatron_model, list):
                megatron_model = [megatron_model]
            for model_module in megatron_model:
                # Handle both wrapped (Float16Module) and unwrapped models
                if isinstance(model_module, Float16Module):
                    model_module = model_module.module
                # Handle VLM models
                if hasattr(model_module, "language_model"):
                    model_module = model_module.language_model
                for layer in model_module.decoder.layers:
                    if hasattr(layer, "mlp") and hasattr(layer.mlp, "router"):
                        layer.mlp.router.weight.requires_grad = False

        mixed_precision_wrapper = CustomFloat16Module
        pre_wrap_hook.extend([freeze_moe_router])

    # Model, optimizer, and learning rate.
    model = get_model(
        cfg.model,
        cfg.ddp,
        use_torch_fsdp2=cfg.dist.use_torch_fsdp2,
        overlap_param_gather_with_optimizer_step=cfg.optimizer.overlap_param_gather_with_optimizer_step,
        data_parallel_random_init=cfg.rng.data_parallel_random_init,
        pre_wrap_hook=pre_wrap_hook,
        mixed_precision_wrapper=mixed_precision_wrapper,
    )
    if load_optimizer:
        optimizer, scheduler = setup_optimizer(
            optimizer_config=cfg.optimizer,
            scheduler_config=cfg.scheduler,
            model=model,
            use_gloo_process_groups=cfg.dist.use_gloo_process_groups,
        )
    else:
        optimizer = None
        scheduler = None

    print("Model, optimizer, and learning rate scheduler built")
    torch.distributed.barrier()

    # Load checkpoint if applicable
    if (
        cfg.checkpoint.load is not None
        or cfg.checkpoint.pretrained_checkpoint is not None
    ) and (
        checkpoint_exists(cfg.checkpoint.load)
        or checkpoint_exists(cfg.checkpoint.pretrained_checkpoint)
    ):
        load_checkpoint(
            state,
            model,
            optimizer,
            scheduler,
            checkpointing_context=checkpointing_context,
            skip_load_to_model_and_opt=HAVE_FSDP2 and cfg.dist.use_torch_fsdp2,
        )
        print("Checkpoint loaded")
    torch.distributed.barrier()

    return state, model, optimizer, scheduler, checkpointing_context


def destroy_parallel_state():
    """Safely destroy parallel state and reset async call tracking.

    This function is called during initialization to clean up temporary distributed
    state from model import operations. Resetting async call tracking ensures that
    when the main Megatron distributed context is created, all ranks start with
    consistent call_idx values for async checkpointing.
    """
    if torch.distributed.is_initialized():
        try:
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()
        except:
            pass  # Ignore errors if already destroyed
    if hasattr(parallel_state, "destroy_model_parallel"):
        try:
            parallel_state.destroy_model_parallel()
        except:
            pass  # Ignore errors if already destroyed

    # Reset async calls queue to prevent call_idx mismatches after distributed context recreation
    try:
        import nemo.tron.utils.async_utils as nemo_async_utils
        from megatron.core.dist_checkpointing.strategies.async_utils import (
            AsyncCallsQueue,
        )

        # Clean up any existing async callers first
        old_call_idx = getattr(nemo_async_utils._async_calls_queue, "call_idx", None)
        num_unfinalized = (
            nemo_async_utils._async_calls_queue.get_num_unfinalized_calls()
        )
        if num_unfinalized > 0:
            print(
                f"[WARNING] Resetting async calls queue with {num_unfinalized} unfinalized calls"
            )
        try:
            nemo_async_utils._async_calls_queue.close()
        except:
            pass  # Ignore errors during cleanup
        # Reset the global async calls queue by creating a new instance
        nemo_async_utils._async_calls_queue = AsyncCallsQueue()
        print(f"[DEBUG] Reset NeMo async calls queue (old call_idx: {old_call_idx})")
    except ImportError:
        pass

    # Also reset the Megatron async calls queue if it exists
    try:
        import megatron.training.async_utils as megatron_async_utils
        from megatron.core.dist_checkpointing.strategies.async_utils import (
            AsyncCallsQueue,
        )

        # Clean up any existing async callers first
        old_call_idx = getattr(
            megatron_async_utils._async_calls_queue, "call_idx", None
        )
        num_unfinalized = (
            megatron_async_utils._async_calls_queue.get_num_unfinalized_calls()
        )
        if num_unfinalized > 0:
            print(
                f"[WARNING] Resetting Megatron async calls queue with {num_unfinalized} unfinalized calls"
            )
        try:
            megatron_async_utils._async_calls_queue.close()
        except:
            pass  # Ignore errors during cleanup
        # Reset the Megatron global async calls queue as well
        megatron_async_utils._async_calls_queue = AsyncCallsQueue()
        print(
            f"[DEBUG] Reset Megatron async calls queue (old call_idx: {old_call_idx})"
        )
    except ImportError:
        pass

    # Reset the third global async_calls instance in base strategy module
    try:
        import megatron.core.dist_checkpointing.strategies.base as base_strategy
        from megatron.core.dist_checkpointing.strategies.async_utils import (
            AsyncCallsQueue,
        )

        # Clean up and reset the global async_calls in base strategy
        old_call_idx = getattr(base_strategy.async_calls, "call_idx", None)
        num_unfinalized = base_strategy.async_calls.get_num_unfinalized_calls()
        if num_unfinalized > 0:
            print(
                f"[WARNING] Resetting base strategy async_calls with {num_unfinalized} unfinalized calls"
            )
        try:
            base_strategy.async_calls.close()
        except:
            pass
        base_strategy.async_calls = AsyncCallsQueue()
        print(f"[DEBUG] Reset base strategy async_calls (old call_idx: {old_call_idx})")
    except ImportError:
        pass


@ray.remote(
    runtime_env=get_runtime_env_for_policy_worker("megatron_policy_worker")
)  # pragma: no cover
class MegatronPolicyWorker(AbstractPolicyWorker, ColocatablePolicyInterface):
    def __repr__(self):
        """Customizes the actor's prefix in the Ray logs.

        This makes it easier to identify which worker is producing specific log messages.
        """
        if torch.distributed.is_initialized():
            return f"{self.__class__.__qualname__}[rank={torch.distributed.get_rank()}]"
        else:
            return f"{self.__class__.__qualname__}"

    def __init__(
        self,
        config: PolicyConfig,
        tokenizer: TokenizerType,
        weights_path: Optional[str] = None,
        optimizer_path: Optional[str] = None,
        init_optimizer: bool = True,
        init_reference_model: bool = True,
        *,
        worker_sharding_annotations: NamedSharding,
        pre_init_communication_queue: Queue,
        **kwargs: Any,
    ):
        apply_transformer_engine_patch()

        self.is_generation_colocated = None
        if "generation" in config and config["generation"] is not None:
            self.is_generation_colocated = config["generation"]["colocated"]["enabled"]

        # Explicitly set NCCL_CUMEM_ENABLE to 1 to avoid the P2P initialization error for PyNCCLCommunicator.
        # See https://github.com/NVIDIA-NeMo/RL/issues/564 for more details.
        if not self.is_generation_colocated:
            os.environ["NCCL_CUMEM_ENABLE"] = "1"

        self.cfg = config
        dtype_map = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }
        self.dtype = dtype_map[self.cfg["precision"]]

        self.optimizer_cpu_offload = self.cfg["megatron_cfg"]["optimizer"][
            "optimizer_cpu_offload"
        ]
        self.offload_optimizer_for_logprob = self.cfg["offload_optimizer_for_logprob"]

        # Reward models are not yet supported with Megatron.
        if "reward_model_cfg" in self.cfg and self.cfg["reward_model_cfg"]["enabled"]:
            raise NotImplementedError(
                "Reward models are not yet supported with the Megatron backend, this issue is "
                "tracked in https://github.com/NVIDIA-NeMo/RL/issues/720"
            )

        # Disable dynamo autotune_local_cache to avoid crash when there's already a cache
        # with different order of node_bundles
        configure_dynamo_cache()

        # cfg["model_name"] is allowed to be either an HF model name or a path to an HF checkpoint
        # check if hf_model_name is a path
        hf_model_name = self.cfg["model_name"]
        # Check if the checkpoint already exists
        hf_model_subdir = hf_model_name
        if os.path.exists(hf_model_name):
            hf_model_subdir = f"model_{hf_model_subdir.replace('/', '_')}"

        pretrained_path = f"{get_megatron_checkpoint_dir()}/{hf_model_subdir}"
        pt_checkpoint_exists = os.path.exists(pretrained_path) and os.path.exists(
            os.path.join(pretrained_path, "iter_0000000")
        )

        # Ensure clean slate before import
        destroy_parallel_state()

        # Set for rank for non-collocated to check which ranks to broadcast from
        self.rank = get_rank_safe()
        # Need to initialize the process group before calling into Megatron-Bridge, otherwise Megatron-Bridge will try to set an incorrect device
        torch.distributed.init_process_group("nccl")
        if pt_checkpoint_exists:
            print(f"Checkpoint already exists at {pretrained_path}. Skipping import.")
        else:
            hf_config_overrides = self.cfg.get("hf_config_overrides", {}) or {}
            import_model_from_hf_name(
                hf_model_name,
                pretrained_path,
                self.cfg["megatron_cfg"],
                **hf_config_overrides,
            )

            if parallel_state.model_parallel_is_initialized():
                print("Reinitializing model parallel after loading model state.")
                parallel_state.destroy_model_parallel()

        pretrained_run_config = os.path.join(
            pretrained_path, "iter_0000000/run_config.yaml"
        )

        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if not os.path.exists(pretrained_run_config):
            raise FileNotFoundError(
                f"Pretrained run config not found at {pretrained_run_config} on rank={get_rank_safe()}. This usually means that the one-time HF->mcore conversion on rank=0 saved to a directory not being mounted on this node. Please check "
            )

        try:
            cfg_from_pretrained = ConfigContainer.from_yaml(
                pretrained_run_config, mode=InstantiationMode.STRICT
            )
        except Exception as e:
            # Add helpful context as a note to the exception
            e.add_note(
                f"\n{'=' * 80}\n"
                f"NOTE: A common cause of this error is when the HF->mcore converted checkpoint is\n"
                f"created with an older version of megatron-bridge.\n"
                f"If this checkpoint is old or was generated by a different code version,\n"
                f"try deleting it and rerunning the code.\n"
                f"The checkpoint will be automatically regenerated with the current version.\n\n"
                f"Checkpoint location: {pretrained_path}\n"
                f"{'=' * 80}"
            )
            raise
        model_cfg = cfg_from_pretrained.model

        cfg_from_pretrained.logger = LoggerConfig()

        model_cfg.tensor_model_parallel_size = self.cfg["megatron_cfg"][
            "tensor_model_parallel_size"
        ]
        model_cfg.pipeline_model_parallel_size = self.cfg["megatron_cfg"][
            "pipeline_model_parallel_size"
        ]
        model_cfg.num_layers_in_first_pipeline_stage = self.cfg["megatron_cfg"][
            "num_layers_in_first_pipeline_stage"
        ]
        model_cfg.num_layers_in_last_pipeline_stage = self.cfg["megatron_cfg"][
            "num_layers_in_last_pipeline_stage"
        ]
        model_cfg.sequence_parallel = self.cfg["megatron_cfg"]["sequence_parallel"]
        model_cfg.context_parallel_size = self.cfg["megatron_cfg"][
            "context_parallel_size"
        ]
        if model_cfg.context_parallel_size > 1:
            assert self.cfg["sequence_packing"]["enabled"], (
                "Sequence Packing must be enabled to use Context Parallelism with MCore"
            )
        model_cfg.expert_tensor_parallel_size = self.cfg["megatron_cfg"][
            "expert_tensor_parallel_size"
        ]
        model_cfg.expert_model_parallel_size = self.cfg["megatron_cfg"][
            "expert_model_parallel_size"
        ]

        # Setting moe_router_dtype to higher precision (e.g. fp64) can improve numerical stability,
        # especially when using many experts.
        model_cfg.moe_router_dtype = self.cfg["megatron_cfg"]["moe_router_dtype"]

        # The below two configs (and "freeze_moe_router") are used to stabilize moe training
        # by preventing updates to the moe router. We found that this is helpful in reducing
        # logprob error during training.

        # Set this to "none" to disable load balancing loss.
        model_cfg.moe_router_load_balancing_type = self.cfg["megatron_cfg"][
            "moe_router_load_balancing_type"
        ]
        # Set this to 0.0 to disable updates to the moe router expert bias
        model_cfg.moe_router_bias_update_rate = self.cfg["megatron_cfg"][
            "moe_router_bias_update_rate"
        ]

        model_cfg.moe_permute_fusion = self.cfg["megatron_cfg"]["moe_permute_fusion"]
        if "layernorm_epsilon" in self.cfg["megatron_cfg"]:
            model_cfg.layernorm_epsilon = self.cfg["megatron_cfg"]["layernorm_epsilon"]

        model_cfg.sequence_parallel = self.cfg["megatron_cfg"]["sequence_parallel"]
        model_cfg.bf16 = self.dtype == torch.bfloat16
        model_cfg.fp16 = self.dtype == torch.float16
        if model_cfg.fp16:
            assert not model_cfg.bf16, "fp16 and bf16 cannot be used together"
            model_cfg.params_dtype = torch.float16
        elif model_cfg.bf16:
            assert not model_cfg.fp16, "fp16 and bf16 cannot be used together"
            model_cfg.params_dtype = torch.bfloat16
        else:
            model_cfg.params_dtype = torch.float32
        model_cfg.pipeline_dtype = dtype_map[self.cfg["megatron_cfg"]["pipeline_dtype"]]
        model_cfg.parallel_output = True
        if self.cfg["megatron_cfg"]["activation_checkpointing"]:
            model_cfg.recompute_granularity = "full"
            model_cfg.recompute_method = "uniform"
            model_cfg.recompute_num_layers = 1
        if not model_cfg.gated_linear_unit:
            assert model_cfg.activation_func is not None, (
                "activation_func must be set if not using gated_linear_unit. This likely "
                "indicates an issue in configuration conversion (e.g. activation func was "
                "a lambda and couldn't be serialized). This is based on this check "
                "https://github.com/NVIDIA/Megatron-LM/blob/1ab876ddc4c1893c76f26d775226a8d1dcdfb3d2/megatron/core/transformer/mlp.py#L174."
            )
        model_cfg.apply_rope_fusion = self.cfg["megatron_cfg"]["apply_rope_fusion"]
        model_cfg.bias_activation_fusion = self.cfg["megatron_cfg"][
            "bias_activation_fusion"
        ]
        fp8_cfg = self.cfg["megatron_cfg"].get("fp8_cfg", None)
        self.fp8_cfg = fp8_cfg
        if fp8_cfg is not None and fp8_cfg.get("enabled", False):
            try:
                model_cfg.fp8 = fp8_cfg["fp8"]
                model_cfg.fp8_recipe = fp8_cfg["fp8_recipe"]
                model_cfg.fp8_param = fp8_cfg["fp8_param"]
            except KeyError as e:
                raise KeyError(f"Missing key in fp8_cfg: {e}")
            if model_cfg.fp8_param:
                warnings.warn(
                    "Setting fp8_param=True sometimes causes NaN token_mult_prob_error, please use with caution. "
                    "Refer to https://github.com/NVIDIA-NeMo/RL/issues/1164 for latest updates with this issue."
                )

        optimizer_cpu_offload = self.cfg["megatron_cfg"]["optimizer"][
            "optimizer_cpu_offload"
        ]
        optimizer_offload_fraction = self.cfg["megatron_cfg"]["optimizer"][
            "optimizer_offload_fraction"
        ]
        if optimizer_cpu_offload:
            # Currently, hybrid optimizer (partly on GPU and partly on CPU) is not supported because it conflicts with the way
            # Nemo-rl handles the optimizer offload/onload between generation and training. So if using CPU optimizer the offload_fraction should be 1.0.
            assert optimizer_offload_fraction == 1.0, (
                "Currently for optimizer offloading, only optimizer_offload_fraction=1.0 is supported"
            )
        if (
            "logprob_chunk_size" in self.cfg
            and self.cfg["logprob_chunk_size"] is not None
            and self.cfg["logprob_chunk_size"] > 0
        ):
            assert self.cfg["megatron_cfg"]["defer_fp32_logits"], (
                "defer_fp32_logits must be True if logprob_chunk_size is set"
            )
        self.defer_fp32_logits = self.cfg["megatron_cfg"].get(
            "defer_fp32_logits", None
        ) and (model_cfg.fp16 or model_cfg.bf16)

        checkpoint_config = CheckpointConfig(
            save_interval=100,
            save=weights_path,
            load=weights_path,
            pretrained_checkpoint=pretrained_path,  # This is the path to the pretrained ckpt for the SFT case
            async_save=False,  # This doesn't work right now.
            fully_parallel_save=True,
            fully_parallel_load=True,  # Enable fully parallel load
            load_rng=False,
        )
        ref_checkpoint_config = CheckpointConfig(
            pretrained_checkpoint=pretrained_path,  # This is the path to the pretrained ckpt for the SFT case
            save=None,
            load=None,
            fully_parallel_load=True,  # Enable fully parallel load
            load_rng=False,
        )

        assert "train_iters" in self.cfg["megatron_cfg"], (
            "train_iters must be set in megatron_cfg. For an example, see "
            "https://github.com/NVIDIA-NeMo/RL/blob/bccbc377705a81a1f4b3c31ad9767bcc15f735a8/nemo_rl/algorithms/sft.py#L175-L179."
        )

        ## These settings are required for correct gradient computations in mcore
        ## when calculate_per_token_loss is True, there is no scaling of the gradient in mcore,
        ## so we handle the scaling in nemo-rl.
        ## perform_initialization = True is a workaround to ensure the correct tensor parallel attributes are set
        ## on the TP-sharded parameters.
        model_cfg.calculate_per_token_loss = True
        model_cfg.perform_initialization = True
        # SHAN CHECK:
        model_cfg.cuda_graph_impl = "local"
        model_cfg.cuda_graph_scope = None
        model_cfg.use_te_rng_tracker = True

        assert (
            "aux_loss" not in model_cfg.moe_router_load_balancing_type
            or model_cfg.moe_aux_loss_coeff == 0
        ), (
            "MoE aux loss is currently not supported due to a known bug in Megatron-LM. "
            "See https://github.com/NVIDIA/Megatron-LM/issues/1984 for more details."
        )

        self.megatron_cfg = ConfigContainer(
            model=model_cfg,
            checkpoint=checkpoint_config,
            logger=LoggerConfig(logging_level=0),
            train=TrainingConfig(
                micro_batch_size=1,  # ignored
                global_batch_size=self.cfg["train_global_batch_size"],  # ignored
                train_iters=self.cfg["megatron_cfg"][
                    "train_iters"
                ],  # Set by algorithm setup
            ),
            optimizer=OptimizerConfig(
                **self.cfg["megatron_cfg"]["optimizer"],
            ),
            ddp=DistributedDataParallelConfig(
                check_for_nan_in_grad=True,
                grad_reduce_in_fp32=self.cfg["megatron_cfg"][
                    "distributed_data_parallel_config"
                ]["grad_reduce_in_fp32"],
                overlap_grad_reduce=self.cfg["megatron_cfg"][
                    "distributed_data_parallel_config"
                ]["overlap_grad_reduce"],
                overlap_param_gather=self.cfg["megatron_cfg"][
                    "distributed_data_parallel_config"
                ]["overlap_param_gather"],
                # we need to set average_in_collective=False with calculate_per_token_loss=True.
                # otherwise, mcore throws an assertion error.
                average_in_collective=False,
                use_distributed_optimizer=self.cfg["megatron_cfg"]["optimizer"][
                    "use_distributed_optimizer"
                ],
                data_parallel_sharding_strategy=self.cfg["megatron_cfg"][
                    "distributed_data_parallel_config"
                ]["data_parallel_sharding_strategy"],
            ),
            scheduler=SchedulerConfig(
                **self.cfg["megatron_cfg"]["scheduler"],
            ),
            dataset=None,
            tokenizer=TokenizerConfig(
                tokenizer_type="HuggingFaceTokenizer",
                tokenizer_model=hf_model_name,
            ),
        )
        # TODO: this validation should happen inside mbridge: https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/1665
        if self.dtype == torch.bfloat16:
            assert self.megatron_cfg.model.bf16 == True, (
                "policy.megatron_cfg.model.bf16=True must be set if policy.precision=bfloat16. This is handled by nemo-rl so this indicates something is misconfigured."
            )
            assert (
                self.megatron_cfg.optimizer.use_precision_aware_optimizer == False
                or self.megatron_cfg.optimizer.bf16 == True
            ), (
                "policy.megatron_cfg.optimizer.bf16=True must be set if policy.precision=bfloat16 when using use_precision_aware_optimizer=True"
            )
        elif self.dtype == torch.float16:
            assert self.megatron_cfg.model.fp16 == True, (
                "policy.megatron_cfg.model.fp16=True must be set if policy.precision=float16. This is handled by nemo-rl so this indicates something is misconfigured."
            )
            assert (
                self.megatron_cfg.optimizer.use_precision_aware_optimizer == False
                or self.megatron_cfg.optimizer.fp16 == True
            ), (
                "policy.megatron_cfg.optimizer.fp16=True must be set if policy.precision=float16 when using use_precision_aware_optimizer=True"
            )
        elif self.dtype == torch.float32:
            assert (
                self.megatron_cfg.model.bf16 == False
                and self.megatron_cfg.model.fp16 == False
            ), (
                "policy.megatron_cfg.model.bf16=False and policy.megatron_cfg.model.fp16=False must be set if policy.precision=float32. This is handled by nemo-rl so this indicates something is misconfigured."
            )
            assert (
                self.megatron_cfg.optimizer.bf16 == False
                and self.megatron_cfg.optimizer.fp16 == False
            ), (
                "policy.megatron_cfg.optimizer.bf16=False and policy.megatron_cfg.optimizer.fp16=False must be set if policy.precision=float32"
            )
        self.megatron_cfg.validate()
        (
            self.mcore_state,
            self.model,
            self.optimizer,
            self.scheduler,
            self.checkpointing_context,
        ) = setup_megatron_model(
            policy_cfg=self.cfg, cfg=self.megatron_cfg, load_optimizer=init_optimizer
        )

        # Set the param sync function for the model
        if (
            self.megatron_cfg.ddp.overlap_param_gather
            and self.megatron_cfg.ddp.align_param_gather
        ):
            self.megatron_cfg.param_sync_func = [
                model_chunk.start_param_sync for model_chunk in self.model
            ]
            if len(self.model) == 1:
                self.megatron_cfg.param_sync_func = self.megatron_cfg.param_sync_func[0]

        self.model = self.model[0]  # Get the first model from the list

        if init_reference_model:
            self.model = self.move_model(self.model, "cpu")
            ref_ckpt_context = init_checkpointing_context(ref_checkpoint_config)

            # Create a separate megatron config for the reference model with the correct checkpoint config
            self.megatron_cfg.model.cuda_graph_impl = "none"
            self.megatron_cfg.model.use_te_rng_tracker = True
            ref_megatron_cfg = ConfigContainer(
                model=self.megatron_cfg.model,
                checkpoint=ref_checkpoint_config,  # Use the reference checkpoint config
                logger=self.megatron_cfg.logger,
                train=self.megatron_cfg.train,
                optimizer=self.megatron_cfg.optimizer,
                ddp=self.megatron_cfg.ddp,
                scheduler=self.megatron_cfg.scheduler,
                dataset=self.megatron_cfg.dataset,
                tokenizer=self.megatron_cfg.tokenizer,
            )

            # Create a separate state object for the reference model
            ref_state = GlobalState()
            ref_state.cfg = ref_megatron_cfg

            # Configure mixed precision wrapper for reference model
            ref_mixed_precision_wrapper = Float16Module
            if self.cfg["megatron_cfg"].get("freeze_moe_router", False):
                ref_mixed_precision_wrapper = CustomFloat16Module

            reference_model = get_model(
                self.megatron_cfg.model,
                self.megatron_cfg.ddp,
                use_torch_fsdp2=self.megatron_cfg.dist.use_torch_fsdp2,
                overlap_param_gather_with_optimizer_step=self.megatron_cfg.optimizer.overlap_param_gather_with_optimizer_step,
                pre_wrap_hook=self.megatron_cfg.rng.data_parallel_random_init,
                mixed_precision_wrapper=ref_mixed_precision_wrapper,
            )
            print("Loading the Reference Model")
            if (
                ref_checkpoint_config.pretrained_checkpoint is not None
                and checkpoint_exists(ref_checkpoint_config.pretrained_checkpoint)
            ):
                load_checkpoint(
                    ref_state,  # Use the separate state object with ref checkpoint config
                    reference_model,
                    None,  # no optimizer
                    None,  # no scheduler
                    checkpointing_context=ref_ckpt_context,
                    skip_load_to_model_and_opt=HAVE_FSDP2
                    and self.megatron_cfg.dist.use_torch_fsdp2,
                )
                reference_model = reference_model[0]
                reference_model.eval()
                self.reference_state_dict = {}
                for name, item in reference_model.state_dict().items():
                    if isinstance(item, torch.Tensor):
                        cpu_item = item.detach().to(
                            device="cpu", non_blocking=True, copy=True
                        )
                        del item
                    else:
                        cpu_item = item
                    self.reference_state_dict[name] = cpu_item
                print("Reference model loaded")
            else:
                print("Reference model not loaded")

            self.model = self.move_model(self.model, "cuda")

        _update_model_config_funcs(
            [self.model],
            self.megatron_cfg.model,
            self.megatron_cfg.ddp,
            self.optimizer,
            align_grad_reduce=self.megatron_cfg.dist.align_grad_reduce,
        )

        tokenizer_config = TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model=hf_model_name,
        )

        self.megatron_tokenizer = build_tokenizer(
            tokenizer_config,
            make_vocab_size_divisible_by=self.megatron_cfg.model.make_vocab_size_divisible_by
            // self.cfg["megatron_cfg"]["tensor_model_parallel_size"],
            tensor_model_parallel_size=self.cfg["megatron_cfg"][
                "tensor_model_parallel_size"
            ],
            trust_remote_code=True,
        )
        self.final_padded_vocab_size = calculate_padded_vocab_size(
            self.megatron_cfg.model.vocab_size,
            self.megatron_cfg.model.make_vocab_size_divisible_by,
            self.cfg["megatron_cfg"]["tensor_model_parallel_size"],
        )
        self.dp_size = worker_sharding_annotations.get_axis_size("data_parallel")
        self.megatron_bridge = AutoBridge.from_hf_pretrained(
            hf_model_name, trust_remote_code=True
        )

        self.should_disable_forward_pre_hook = (
            self.cfg["megatron_cfg"]["optimizer"]["use_distributed_optimizer"]
            and self.cfg["megatron_cfg"]["distributed_data_parallel_config"][
                "overlap_param_gather"
            ]
        )

        # vars used for refit
        ## will be initialized in prepare_refit_info
        # refit_param_info_mcore combines the conversion tasks with the param memory
        # [(mcore_param_name, estimated_memory), ...]
        # Note: here param name is local param name, with local layer number and
        # local expert id etc.
        self.refit_conversion_tasks = (
            None  # Meta data for conversion params from megatron bridge
        )
        self.refit_conversion_tasks_current_index = None
        self.refit_param_info_mcore = None

        ## used for streaming update inference engine weights
        self._held_gather_buffer = None

    def enable_forward_pre_hook(self):
        assert isinstance(self.model, DistributedDataParallel)
        self.model.enable_forward_pre_hook()

    def disable_forward_pre_hook(self, param_sync=True):
        assert isinstance(self.model, DistributedDataParallel)
        self.model.disable_forward_pre_hook(param_sync=param_sync)

    @wrap_with_nvtx_name("megatron_policy_worker/train")
    def train(
        self,
        data: BatchedDataDict,
        loss_fn: LossFunction,
        eval_mode: bool = False,
        gbs: Optional[int] = None,
        mbs: Optional[int] = None,
    ) -> dict[str, Any]:
        """Train the policy on a batch of data with a given loss function."""
        # Disable CUDA graphs for training to avoid conflicts with inference tensors.
        # This prevents "Inplace update to inference tensor outside InferenceMode" errors
        # when switching from generation (which uses CUDA graphs) to training.
        # Use toggle_cuda_graphs on the underlying module (not the DDP wrapper) to properly
        # reset CUDA graph state, following the pattern from megatron/rl/rl_utils.py.
        lang_module = (
            self.model.module.module
            if hasattr(self.model.module, "module")
            else self.model.module
        )

        toggle_cuda_graphs(lang_module, set_to="none", reset_cuda_graphs=True)

        # Force release any cached CUDA graph memory before training
        torch.cuda.empty_cache()

        self.model.zero_grad_buffer()
        if hasattr(self.model, "inference_params"):
            self.model.inference_params = None

        # Reset any cached attention states
        for module in self.model.modules():
            if hasattr(module, "reset_inference_cache"):
                module.reset_inference_cache()
            if hasattr(module, "_inference_key_value_memory"):
                module._inference_key_value_memory = None

        if gbs is None:
            gbs = self.cfg["train_global_batch_size"]
        if mbs is None:
            mbs = self.cfg["train_micro_batch_size"]
        local_gbs = gbs // self.dp_size
        total_dataset_size = torch.tensor(data.size, device="cuda")
        torch.distributed.all_reduce(
            total_dataset_size,
            op=torch.distributed.ReduceOp.SUM,
            group=parallel_state.get_data_parallel_group(),
        )
        num_global_batches = int(total_dataset_size.item()) // gbs

        if eval_mode:
            ctx: AbstractContextManager[Any] = torch.no_grad()
            self.model.eval()
        else:
            ctx = nullcontext()
            # Ensure model is in training mode
            self.model.train()

        with ctx:
            # dim 1 is always assumed to be the sequence dim, sanity check this here
            sequence_dim = 1
            seq_dim_size = data["input_ids"].shape[sequence_dim]
            for k, v in data.items():
                if torch.is_tensor(v) and len(v.shape) > 1:
                    assert v.shape[sequence_dim] == seq_dim_size, (
                        f"Dim 1 must be the sequence dim, expected dim 1={seq_dim_size} but got shape {v.shape}"
                    )

            forward_step = partial(
                forward_step_arbitrary_loss, loss_fn=loss_fn, policy_cfg=self.cfg
            )
            all_mb_metrics = []
            losses = []
            total_num_microbatches = 0
            for gb_idx in range(num_global_batches):
                global_batch = data.get_batch(batch_idx=gb_idx, batch_size=local_gbs)

                assert "sample_mask" in global_batch, (
                    "sample_mask must be present in the data!"
                )
                ## get the normalization factor for the loss
                local_valid_seqs = torch.sum(global_batch["sample_mask"])

                if not "token_mask" in global_batch:
                    local_valid_toks = (
                        local_valid_seqs * global_batch["input_ids"].shape[1]
                    )
                else:
                    local_valid_toks = torch.sum(
                        global_batch["token_mask"][:, 1:]
                        * global_batch["sample_mask"].unsqueeze(-1)
                    )

                to_reduce = torch.tensor([local_valid_seqs, local_valid_toks]).cuda()
                torch.distributed.all_reduce(
                    to_reduce, group=parallel_state.get_data_parallel_group()
                )
                global_valid_seqs, global_valid_toks = to_reduce[0], to_reduce[1]

                if (
                    hasattr(loss_fn, "loss_type")
                    and loss_fn.loss_type == LossType.TOKEN_LEVEL
                ):
                    assert "token_mask" in global_batch, (
                        "token_mask must be present in the data when using token-level loss"
                    )

                batch = data.get_batch(batch_idx=gb_idx, batch_size=local_gbs)
                pack_seqs = False
                seqlen_key = None
                pad_factor = 1
                pad_full_seq_to = None
                pad_packed_seq_to_multiple_of = 1
                if self.cfg["dynamic_batching"]["enabled"]:
                    data_iterator = batch.make_microbatch_iterator_with_dynamic_shapes()
                    data_iterator_len = (
                        batch.get_microbatch_iterator_dynamic_shapes_len()
                    )
                elif self.cfg["sequence_packing"]["enabled"]:
                    data_iterator = (
                        batch.make_microbatch_iterator_for_packable_sequences()
                    )
                    data_iterator_len, seq_dim_size = (
                        batch.get_microbatch_iterator_for_packable_sequences_len()
                    )
                    mbs = 1
                    pack_seqs = True
                    seqlen_key = "input_lengths"
                    (
                        pad_factor,
                        pad_packed_seq_to_multiple_of,
                        pad_full_seq_to,
                    ) = _get_pack_sequence_parameters_for_megatron(
                        self.cfg["megatron_cfg"],
                        seq_dim_size,
                    )
                    # if pad_full_seq_to is not None, we need to use it as the sequence length
                    seq_dim_size = pad_full_seq_to or seq_dim_size
                else:
                    data_iterator = batch.make_microbatch_iterator(mbs)
                    data_iterator_len = local_gbs // mbs

                # Track total microbatches for MoE aux-loss averaging
                total_num_microbatches += int(data_iterator_len)

                rerun_state_machine = get_rerun_state_machine()
                while rerun_state_machine.should_run_forward_backward(data_iterator):
                    # Set grad to zero.
                    self.model.zero_grad_buffer()
                    self.optimizer.zero_grad()

                    # Forward pass.
                    forward_backward_func = get_forward_backward_func()
                    losses_reduced = forward_backward_func(
                        forward_step_func=partial(
                            forward_step,
                            self.mcore_state,
                            global_valid_seqs,
                            global_valid_toks,
                            pack_sequences=pack_seqs,
                            seq_length_key=seqlen_key,
                            pad_individual_seqs_to_multiple_of=pad_factor,
                            pad_packed_seq_to_multiple_of=pad_packed_seq_to_multiple_of,
                            pad_full_seq_to=pad_full_seq_to,
                            defer_fp32_logits=self.defer_fp32_logits,
                        ),
                        data_iterator=data_iterator,
                        model=self.model,
                        num_microbatches=data_iterator_len,
                        seq_length=seq_dim_size,
                        micro_batch_size=mbs,
                        decoder_seq_length=seq_dim_size,
                        forward_only=eval_mode,
                        do_not_average_loss=True,
                    )

                # Empty unused memory.
                if self.cfg["megatron_cfg"]["empty_unused_memory_level"] >= 1:
                    torch.cuda.empty_cache()

                # Update parameters.
                if not eval_mode:
                    update_successful, grad_norm, num_zeros_in_grad = (
                        self.optimizer.step()
                    )
                else:
                    update_successful, grad_norm, num_zeros_in_grad = (True, 0.0, 0.0)

                # when freezing sub-models we may have a mixture of successful and unsucessful ranks,
                # so we must gather across mp ranks
                update_successful = logical_and_across_model_parallel_group(
                    update_successful
                )
                # grad_norm and num_zeros_in_grad will be None on ranks without trainable params,
                # so we must gather across mp ranks
                grad_norm: float = reduce_max_stat_across_model_parallel_group(
                    grad_norm
                )
                num_zeros_in_grad: float = reduce_max_stat_across_model_parallel_group(
                    num_zeros_in_grad
                )

                if update_successful:
                    skipped_iter = 0
                else:
                    skipped_iter = 1

                # Empty unused memory.
                if self.cfg["megatron_cfg"]["empty_unused_memory_level"] >= 2:
                    torch.cuda.empty_cache()

                if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    # keep all microbatch metrics to be normalized later
                    gb_loss_metrics = []
                    mb_losses = []
                    for x in losses_reduced:
                        loss_metrics = {}
                        for k in x.keys():
                            loss_metrics[k] = x[k] / num_global_batches
                        gb_loss_metrics.append(loss_metrics)
                        curr_lr = self.scheduler.get_lr(self.optimizer.param_groups[0])
                        curr_wd = self.scheduler.get_wd()
                        loss_metrics["lr"] = curr_lr
                        loss_metrics["wd"] = curr_wd
                        loss_metrics["global_valid_seqs"] = global_valid_seqs.item()
                        loss_metrics["global_valid_toks"] = global_valid_toks.item()
                        mb_losses.append(loss_metrics["loss"])

                    torch.distributed.broadcast_object_list(
                        [gb_loss_metrics],
                        src=get_pipeline_model_parallel_last_rank(),
                        group=get_pipeline_model_parallel_group(),
                    )
                else:
                    loss_metrics = [None]  # type: ignore
                    torch.distributed.broadcast_object_list(
                        loss_metrics,
                        src=get_pipeline_model_parallel_last_rank(),
                        group=get_pipeline_model_parallel_group(),
                    )
                    gb_loss_metrics = loss_metrics[0]
                    mb_losses = [x["loss"] for x in gb_loss_metrics]

                all_mb_metrics.extend(gb_loss_metrics)
                losses.append(torch.tensor(mb_losses).sum().item())

        if not eval_mode:
            # take one LR step every rollout batch
            # we need to scale the step by gbs to counteract the fact that NeMo automatically
            # scales lr_warmup_steps by gbs during init
            self.scheduler.step(increment=gbs)

        # Aggregate metrics across all microbatches
        mb_metrics = defaultdict(list)
        for m in all_mb_metrics:
            for k, v in m.items():
                mb_metrics[k].append(v)

        with torch.no_grad():
            global_loss = torch.tensor(losses, device="cuda")
            torch.distributed.all_reduce(
                global_loss,
                op=torch.distributed.ReduceOp.SUM,
                group=parallel_state.get_data_parallel_group(),
            )

        metrics = {
            "global_loss": global_loss.cpu(),
            "rank": torch.distributed.get_rank(),
            "gpu_name": torch.cuda.get_device_name(),
            "model_dtype": self.dtype,
            "all_mb_metrics": dict(mb_metrics),
            "grad_norm": torch.tensor([grad_norm]),
        }
        # Collect MoE aux metrics averaged across microbatches
        num_moe_experts = getattr(self.model.config, "num_moe_experts", None)
        if num_moe_experts is not None and num_moe_experts > 1:
            moe_loss_scale = 1.0 / max(1, total_num_microbatches)
            moe_metrics = get_moe_metrics(
                loss_scale=moe_loss_scale,
                per_layer_logging=self.cfg["megatron_cfg"]["moe_per_layer_logging"],
            )
            if moe_metrics:
                metrics["moe_metrics"] = moe_metrics
        return metrics

    @wrap_with_nvtx_name("megatron_policy_worker/get_logprobs")
    def get_logprobs(
        self, *, data: BatchedDataDict[Any], micro_batch_size: Optional[int] = None
    ) -> BatchedDataDict[LogprobOutputSpec]:
        """Get the logprobs of the model for a batch of data.

        Uses the configured logprob_batch_size to do microbatching.
        Input data is assumed to be right-padded. The method internally converts to
        left-padded format for computation, and returns outputs in right-padded format.
        If micro_batch_size is provided, it will be used instead of the configured
        logprob_batch_size.

        Returns:
          a BatchedDataDict with key "logprobs" and shape [batch_size, sequence_length].
          We use the convention that the logprob of the first token is 0 so that the sequence length is maintained.
          The logprob of input token i is specified at position i in the output logprobs tensor.
        """
        no_grad = torch.no_grad()
        no_grad.__enter__()
        logprob_batch_size = (
            micro_batch_size
            if micro_batch_size is not None
            else self.cfg["logprob_batch_size"]
        )

        # dim 1 is always assumed to be the sequence dim, sanity check this here
        sequence_dim = 1
        input_seq_dim_size = data["input_ids"].shape[sequence_dim]
        for k, v in data.items():
            if torch.is_tensor(v) and len(v.shape) > 1:
                assert v.shape[sequence_dim] == input_seq_dim_size, (
                    f"Dim 1 must be the sequence dim, expected dim 1={input_seq_dim_size} but got shape {v.shape}"
                )

        self.model.eval()

        pp_seq_dim_size = input_seq_dim_size
        pp_grp = get_pipeline_model_parallel_group()
        pad_factor = 1
        pad_packed_seq_to_multiple_of = 1
        pad_full_seq_to = None
        if self.cfg["sequence_packing"]["enabled"]:
            _, seq_dim_size = data.get_microbatch_iterator_for_packable_sequences_len()
            (
                pad_factor,
                pad_packed_seq_to_multiple_of,
                pad_full_seq_to,
            ) = _get_pack_sequence_parameters_for_megatron(
                self.cfg["megatron_cfg"],
                seq_dim_size,
            )
            pp_seq_dim_size = pad_full_seq_to or pp_seq_dim_size

        def forward_step_fn(
            data_iterator: Iterator[BatchedDataDict[Any]], model: GPTModel
        ):
            nonlocal pad_full_seq_to, pad_packed_seq_to_multiple_of, pad_factor
            data_dict = next(data_iterator).to("cuda")
            if self.cfg["sequence_packing"]["enabled"]:
                original_seq_length = data_dict["input_ids"].shape[1]
                cp_size = self.cfg["megatron_cfg"]["context_parallel_size"]
                cp_rank = get_context_parallel_rank()
                (
                    input_ids,
                    input_ids_cp_sharded,
                    packed_seq_params,
                    cu_seqlens,
                    cu_seqlens_padded,
                ) = _pack_sequences_for_megatron(
                    data_dict["input_ids"].clone(),
                    data_dict["input_lengths"],
                    pad_individual_seqs_to_multiple_of=pad_factor,
                    pad_packed_seq_to_multiple_of=pad_packed_seq_to_multiple_of,
                    pad_packed_seq_to=pad_full_seq_to,
                    cp_rank=cp_rank,
                    cp_size=cp_size,
                )
                attention_mask, position_ids = None, None
                unpacked_input_ids = data_dict["input_ids"]
            else:
                input_ids = data_dict["input_ids"]
                input_ids_cp_sharded = input_ids
                attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
                    data=input_ids,
                    eod_token=0,  # used for loss_mask, which we don't use
                    pad_token=0,  # used for loss_mask, which we don't use
                    reset_position_ids=False,
                    reset_attention_mask=False,
                    eod_mask_loss=False,
                    pad_mask_loss=False,
                )
                packed_seq_params = None
                unpacked_input_ids = input_ids

            multimodal_data = data_dict.get_multimodal_dict(
                as_tensors=True, device=input_ids.device
            )
            if len(multimodal_data) > 0:
                position_ids = None

            additional_kwargs = {}
            # Mamba models currently do not support packed_seq_params
            if packed_seq_params is not None:
                additional_kwargs["packed_seq_params"] = packed_seq_params

            if self.defer_fp32_logits:
                additional_kwargs["fp32_output"] = False

            output_tensor = model(
                input_ids=input_ids_cp_sharded,
                position_ids=position_ids,
                attention_mask=attention_mask,
                **multimodal_data,
                **additional_kwargs,
            )

            # Apply temperature scaling to logits for training
            # This matches the dtensor worker's _apply_temperature_scaling in the train method
            if "generation" in self.cfg and self.cfg["generation"] is not None:
                output_tensor.div_(self.cfg["generation"]["temperature"])

            def collection_fn(output_tensor):
                stc = time.time()
                tp_grp = get_tensor_model_parallel_group()
                tp_rank = get_tensor_model_parallel_rank()
                logprob_chunk_size = self.cfg.get("logprob_chunk_size", None)
                if self.cfg["sequence_packing"]["enabled"]:
                    token_logprobs = from_parallel_logits_to_logprobs_packed_sequences(
                        output_tensor,
                        target=input_ids,
                        cu_seqlens_padded=cu_seqlens_padded,
                        unpacked_seqlen=original_seq_length,
                        vocab_start_index=tp_rank * output_tensor.shape[-1],
                        vocab_end_index=(tp_rank + 1) * output_tensor.shape[-1],
                        group=tp_grp,
                        inference_only=True,
                        cp_group=get_context_parallel_group(),
                        chunk_size=logprob_chunk_size,
                    )
                else:
                    token_logprobs = from_parallel_logits_to_logprobs(
                        output_tensor,
                        target=unpacked_input_ids,
                        vocab_start_index=tp_rank * output_tensor.shape[-1],
                        vocab_end_index=(tp_rank + 1) * output_tensor.shape[-1],
                        tp_group=tp_grp,
                        inference_only=True,
                        chunk_size=logprob_chunk_size,
                    )

                # Prepend 0 logprob for first token to maintain same sequence length as input
                token_logprobs = torch.cat(
                    [torch.zeros_like(token_logprobs[:, :1]), token_logprobs], dim=1
                )
                return torch.tensor(0.0, device=token_logprobs.device), {
                    "logprobs": token_logprobs
                }

            return output_tensor, collection_fn

        if self.cfg["dynamic_batching"]["enabled"]:
            mb_iterator = data.make_microbatch_iterator_with_dynamic_shapes()
            data_iterator_len = data.get_microbatch_iterator_dynamic_shapes_len()
            micro_batch_size = logprob_batch_size
        elif self.cfg["sequence_packing"]["enabled"]:
            mb_iterator = data.make_microbatch_iterator_for_packable_sequences()
            data_iterator_len, _ = (
                data.get_microbatch_iterator_for_packable_sequences_len()
            )
            micro_batch_size = 1
        else:
            mb_iterator = data.make_microbatch_iterator(logprob_batch_size)
            data_iterator_len = max(1, data.size // logprob_batch_size)
            micro_batch_size = logprob_batch_size

        forward_backward_func = get_forward_backward_func()
        list_of_logprobs = forward_backward_func(
            forward_step_func=forward_step_fn,
            data_iterator=mb_iterator,
            model=self.model,
            num_microbatches=data_iterator_len,
            seq_length=pp_seq_dim_size,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=pp_seq_dim_size,
            forward_only=True,
        )
        if is_pipeline_last_stage(ignore_virtual=True):
            all_log_probs_padded = []
            all_logprobs = [l["logprobs"] for l in list_of_logprobs]
            for lp in all_logprobs:
                padding_needed = input_seq_dim_size - lp.shape[1]
                if padding_needed > 0:
                    lp = torch.nn.functional.pad(
                        lp, (0, padding_needed), mode="constant", value=0.0
                    )
                all_log_probs_padded.append(lp)

            logprobs = torch.cat(all_log_probs_padded, dim=0)
            # broadcast logprobs to first pp rank
            broadcast_tensor(logprobs, torch.distributed.get_rank(), pp_grp)
        else:
            logprobs = broadcast_tensor(
                None, get_pipeline_model_parallel_last_rank(), pp_grp
            )

        no_grad.__exit__(None, None, None)
        return BatchedDataDict[LogprobOutputSpec](logprobs=logprobs).to("cpu")

    @contextmanager
    def use_reference_model(self):
        """Context manager that temporarily swaps the reference model and active model.

        On entry: Moves model to CPU, moves reference_model to CUDA. Swaps the references
        On exit: Restores original references and re-flips cuda/cpu
        """
        ## disable overlap param gather when swapping weights
        if self.should_disable_forward_pre_hook:
            self.disable_forward_pre_hook()

        with torch.no_grad():
            try:
                # Save original references
                model_state_dict = {}
                for name, item in self.model.state_dict().items():
                    if isinstance(item, torch.Tensor):
                        item = item.detach().to(
                            device="cpu", non_blocking=True, copy=True
                        )
                    model_state_dict[name] = item

                self.model.load_state_dict(self.reference_state_dict, strict=True)
                # for name, item in self.reference_state_dict.items():
                # if isinstance(item, torch.Tensor):
                # self.model.state_dict()[name] = item.detach().to(device="cuda", non_blocking=True, copy=True)

                if self.cfg["megatron_cfg"]["empty_unused_memory_level"] >= 1:
                    gc.collect()
                    torch.cuda.empty_cache()

                # - self.model is the original reference_model, now on CUDA
                # - self.reference_model is the original model, now on CPU
                yield

            finally:
                # Restore original references and device placement
                self.model.load_state_dict(model_state_dict, strict=True)
                # for name, item in model_state_dict.items():
                # if isinstance(item, torch.Tensor):
                # item = item.detach().to(device="cuda", non_blocking=True, copy=True)
                # self.model.state_dict()[name] = item

                if self.cfg["megatron_cfg"]["empty_unused_memory_level"] >= 1:
                    gc.collect()
                    torch.cuda.empty_cache()

                ## re-enable overlap param gather after weight swap
                if self.should_disable_forward_pre_hook:
                    self.enable_forward_pre_hook()

    @wrap_with_nvtx_name("megatron_policy_worker/get_topk_logits")
    def get_topk_logits(
        self,
        *,
        data: BatchedDataDict[GenerationDatumSpec],
        k: int,
        micro_batch_size: Optional[int] = None,
    ):
        """Get the top-k logits and indices for a batch of data.

        The major difference from get_logprobs is that we compute top-k logits and indices for each position in the sequence.

        Returns:
            BatchedDataDict containing:
                - topk_logits: Tensor of top-k logits for each position in the sequence
                - topk_indices: Tensor of top-k indices for each position in the sequence
        """
        no_grad = torch.no_grad()
        no_grad.__enter__()

        logprob_batch_size = (
            micro_batch_size
            if micro_batch_size is not None
            else self.cfg["logprob_batch_size"]
        )

        sequence_dim = 1
        input_seq_dim_size = data["input_ids"].shape[sequence_dim]
        # Avoid shadowing the function argument `k` by using a distinct variable name
        for tensor_name, v in data.items():
            if torch.is_tensor(v) and len(v.shape) > 1:
                assert v.shape[sequence_dim] == input_seq_dim_size, (
                    f"Tensor {tensor_name} must have sequence dimension {sequence_dim} of size {input_seq_dim_size}, but got shape {v.shape}"
                )

        self.model.eval()

        pp_seq_dim_size = input_seq_dim_size
        pp_grp = get_pipeline_model_parallel_group()

        pad_factor = 1
        pad_packed_seq_to_multiple_of = 1
        pad_full_seq_to = None
        if self.cfg["sequence_packing"]["enabled"]:
            _, seq_dim_size = data.get_microbatch_iterator_for_packable_sequences_len()
            (
                pad_factor,
                pad_packed_seq_to_multiple_of,
                pad_full_seq_to,
            ) = _get_pack_sequence_parameters_for_megatron(
                self.cfg["megatron_cfg"],
                seq_dim_size,
            )
            pp_seq_dim_size = pad_full_seq_to or pp_seq_dim_size

        def forward_step_fn(
            data_iterator: Iterator[BatchedDataDict[Any]], model: GPTModel
        ):
            nonlocal pad_full_seq_to, pad_packed_seq_to_multiple_of, pad_factor
            data_dict = next(data_iterator).to("cuda")

            pack = self.cfg["sequence_packing"]["enabled"]
            if pack:
                original_seq_length = data_dict["input_ids"].shape[1]
                cp_size = self.cfg["megatron_cfg"]["context_parallel_size"]
                cp_rank = get_context_parallel_rank()

                (
                    input_ids_unpacked,
                    input_ids_cp_sharded,
                    packed_seq_params,
                    cu_seqlens,
                    cu_seqlens_padded,
                ) = _pack_sequences_for_megatron(
                    data_dict["input_ids"].clone(),
                    data_dict["input_lengths"],
                    pad_individual_seqs_to_multiple_of=pad_factor,
                    pad_packed_seq_to_multiple_of=pad_packed_seq_to_multiple_of,
                    pad_packed_seq_to=pad_full_seq_to,
                    cp_rank=cp_rank,
                    cp_size=cp_size,
                )
                attention_mask, position_ids = None, None
                seq_lengths = data_dict["input_lengths"]
                unpacked_seqlen = original_seq_length
            else:
                input_ids_cp_sharded = data_dict["input_ids"]
                attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
                    data=input_ids_cp_sharded,
                    eod_token=0,
                    pad_token=0,
                    reset_position_ids=False,
                    reset_attention_mask=False,
                    eod_mask_loss=False,
                    pad_mask_loss=False,
                )
                packed_seq_params = None

            multimodal_data = data_dict.get_multimodal_dict(
                as_tensors=True, device=input_ids_cp_sharded.device
            )
            if len(multimodal_data) > 0:
                position_ids = None

            additional_kwargs = {}
            if packed_seq_params is not None:
                additional_kwargs["packed_seq_params"] = packed_seq_params

            output_tensor = model(
                input_ids=input_ids_cp_sharded,
                position_ids=position_ids,
                attention_mask=attention_mask,
                **additional_kwargs,
                **multimodal_data,
            )

            if "generation" in self.cfg and self.cfg["generation"] is not None:
                output_tensor.div_(self.cfg["generation"]["temperature"])

            def collection_fn(_):
                # Only the last PP stage produces final logits/top-k; earlier stages return empty
                # if not is_pipeline_last_stage(ignore_virtual=True):
                # return output_tensor.new_zeros(()), {}

                tp_grp = get_tensor_model_parallel_group()
                tp_rank = get_tensor_model_parallel_rank()
                vocab_shard_size = output_tensor.shape[-1]
                vocab_start_index = tp_rank * vocab_shard_size

                chunk_size = None
                if "logprob_chunk_size" in self.cfg:
                    chunk_size = self.cfg["logprob_chunk_size"]

                topk_vals_local, topk_idx_local = distributed_vocab_topk(
                    output_tensor,
                    k,
                    tp_grp,
                    vocab_start_index=vocab_start_index,
                    vocab_end_index=vocab_start_index + vocab_shard_size,
                    chunk_size=chunk_size,
                )

                if self.cfg["megatron_cfg"]["context_parallel_size"] > 1:
                    cp_grp = get_context_parallel_group()
                    if pack:
                        # Per-sequence CP allgather following packed-sequence logic
                        batch_size = data_dict["input_ids"].shape[0]
                        total_packed_len = int(cu_seqlens_padded[-1].item())

                        topk_vals_full = torch.zeros(
                            (1, total_packed_len, k),
                            dtype=topk_vals_local.dtype,
                            device=topk_vals_local.device,
                        )
                        topk_idx_full = torch.zeros(
                            (1, total_packed_len, k),
                            dtype=topk_idx_local.dtype,
                            device=topk_idx_local.device,
                        )

                        for i in range(batch_size):
                            start_idx = int(cu_seqlens_padded[i].item())
                            end_idx = int(cu_seqlens_padded[i + 1].item())
                            if end_idx > start_idx:
                                local_vals_slice = topk_vals_local[
                                    :, start_idx // cp_size : end_idx // cp_size, :
                                ]
                                local_idx_slice = topk_idx_local[
                                    :, start_idx // cp_size : end_idx // cp_size, :
                                ]
                                gathered_vals = allgather_cp_sharded_tensor(
                                    local_vals_slice, cp_grp, seq_dim=1
                                )
                                gathered_idx = allgather_cp_sharded_tensor(
                                    local_idx_slice, cp_grp, seq_dim=1
                                )
                                # Some kernels may return [X, Y, k] where X*Y = (end_idx - start_idx).
                                # Flatten leading dims and reshape to [1, expected_len, k] to match target.
                                expected_len = end_idx - start_idx
                                if (
                                    gathered_vals.dim() == 3
                                    and gathered_vals.shape[1] != expected_len
                                ):
                                    gathered_vals = gathered_vals.reshape(
                                        1, expected_len, gathered_vals.shape[-1]
                                    )
                                if (
                                    gathered_idx.dim() == 3
                                    and gathered_idx.shape[1] != expected_len
                                ):
                                    gathered_idx = gathered_idx.reshape(
                                        1, expected_len, gathered_idx.shape[-1]
                                    )
                                topk_vals_full[:, start_idx:end_idx, :] = gathered_vals
                                topk_idx_full[:, start_idx:end_idx, :] = gathered_idx
                    else:
                        # Sequence packing must be enabled when CP > 1
                        raise RuntimeError(
                            "Context Parallelism (CP>1) requires sequence packing to be enabled."
                        )
                else:
                    topk_vals_full = topk_vals_local
                    topk_idx_full = topk_idx_local

                if pack:
                    batch_size = data_dict["input_ids"].shape[0]
                    out_vals = torch.zeros(
                        (batch_size, unpacked_seqlen, k),
                        dtype=topk_vals_full.dtype,
                        device=topk_vals_full.device,
                    )
                    out_idx = torch.zeros(
                        (batch_size, unpacked_seqlen, k),
                        dtype=topk_idx_full.dtype,
                        device=topk_idx_full.device,
                    )
                    for i in range(batch_size):
                        seq_len = int(seq_lengths[i].item())
                        start_idx = int(cu_seqlens_padded[i].item())
                        if seq_len > 0:
                            out_vals[i, :seq_len, :] = topk_vals_full[
                                0, start_idx : start_idx + seq_len, :
                            ]
                            out_idx[i, :seq_len, :] = topk_idx_full[
                                0, start_idx : start_idx + seq_len, :
                            ]
                    return output_tensor.new_zeros(()), {
                        "topk_logits": out_vals,
                        "topk_indices": out_idx,
                    }
                else:
                    return output_tensor.new_zeros(()), {
                        "topk_logits": topk_vals_full,
                        "topk_indices": topk_idx_full,
                    }

            return output_tensor, collection_fn

        if self.cfg["dynamic_batching"]["enabled"]:
            mb_iterator = data.make_microbatch_iterator_with_dynamic_shapes()
            data_iterator_len = data.get_microbatch_iterator_dynamic_shapes_len()
            micro_batch = logprob_batch_size
        elif self.cfg["sequence_packing"]["enabled"]:
            mb_iterator = data.make_microbatch_iterator_for_packable_sequences()
            data_iterator_len, _ = (
                data.get_microbatch_iterator_for_packable_sequences_len()
            )
            micro_batch = 1
        else:
            mb_iterator = data.make_microbatch_iterator(logprob_batch_size)
            data_iterator_len = max(1, data.size // logprob_batch_size)
            micro_batch = logprob_batch_size

        forward_backward_func = get_forward_backward_func()
        list_of_outputs = forward_backward_func(
            forward_step_func=forward_step_fn,
            data_iterator=mb_iterator,
            model=self.model,
            num_microbatches=data_iterator_len,
            seq_length=pp_seq_dim_size,
            micro_batch_size=micro_batch,
            decoder_seq_length=pp_seq_dim_size,
            forward_only=True,
        )

        if is_pipeline_last_stage(ignore_virtual=True):
            logits_chunks = []
            indices_chunks = []
            for out in list_of_outputs:
                tk = out["topk_logits"]
                ti = out["topk_indices"]
                pad_len = input_seq_dim_size - tk.shape[1]
                if pad_len > 0:
                    tk = torch.nn.functional.pad(tk, (0, 0, 0, pad_len), value=0.0)
                    ti = torch.nn.functional.pad(ti, (0, 0, 0, pad_len), value=0)
                logits_chunks.append(tk)
                indices_chunks.append(ti)

            topk_logits = torch.cat(logits_chunks, dim=0)
            topk_indices = torch.cat(indices_chunks, dim=0)

            topk_logits = broadcast_tensor(
                topk_logits, torch.distributed.get_rank(), pp_grp
            )
            topk_indices = broadcast_tensor(
                topk_indices, torch.distributed.get_rank(), pp_grp
            )
        else:
            last_pp_rank = get_pipeline_model_parallel_last_rank()
            topk_logits = broadcast_tensor(None, last_pp_rank, pp_grp)
            topk_indices = broadcast_tensor(None, last_pp_rank, pp_grp)

        no_grad.__exit__(None, None, None)
        return BatchedDataDict.from_batches(
            [{"topk_logits": topk_logits.cpu(), "topk_indices": topk_indices.cpu()}]
        )

    @torch.no_grad()
    @wrap_with_nvtx_name("megatron_policy_worker/generate")
    def generate(
        self, *, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> BatchedDataDict[GenerationOutputSpec]:
        """Generate a batch of data using huggingface framework generation.

        Args:
            data: BatchedDataDict containing input_ids and input_lengths tensors
        Returns:
            BatchedDataDict conforming to GenerationOutputSpec:
                - output_ids: input + generated token IDs
                - logprobs: Log probabilities for each token
                - generation_lengths: Lengths of each response
        """
        # 512 bATCH SIZE (200 tokens)
        no_grad = torch.no_grad()
        no_grad.__enter__()
        self.model.config.flash_decode = False
        if self.should_disable_forward_pre_hook:
            self.model = self.move_model(
                self.model, "cuda", move_params=True, move_grads=False
            )
        # Verify input is right padded
        assert isinstance(data, BatchedDataDict), (
            f"data must be a BatchedDataDict, got type: {type(data)}"
        )
        assert "input_ids" in data and "input_lengths" in data, (
            f"input_ids and input_lengths must be present in the BatchedDataDict, got keys: {data.keys()}"
        )
        is_right_padded, error_msg = verify_right_padding(
            data, pad_value=self.tokenizer.pad_token_id
        )
        if not is_right_padded:
            warnings.warn(
                f"Input to Megatron Generation worker is not properly right-padded: {error_msg}"
            )

        model_cfg = self.megatron_cfg.model
        inference_wrapper_config = InferenceWrapperConfig(
            hidden_size=model_cfg.hidden_size,
            inference_batch_times_seqlen_threshold=1000000,
            fp32_residual_connection=model_cfg.fp32_residual_connection,
            params_dtype=model_cfg.params_dtype,
            padded_vocab_size=self.final_padded_vocab_size,  # Use the potentially updated value
            inference_max_seq_length=self.cfg["generation"]["max_new_tokens"],  # type: ignore
            inference_max_requests=self.cfg["generation_batch_size"],
        )

        from megatron.core.inference.contexts.dynamic_context import (
            DynamicInferenceContext,
        )
        from megatron.core.inference.engines.dynamic_engine import (
            DynamicInferenceEngine,
        )
        from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
            GPTInferenceWrapper,
        )
        from megatron.core.inference.sampling_params import SamplingParams

        mcore_generation_config = self.cfg["generation"]["mcore_generation_config"]
        buffer_size_gb = mcore_generation_config.get("buffer_size_gb", 20)

        num_cuda_graphs = mcore_generation_config.get("num_cuda_graphs", 16)
        block_size_tokens = mcore_generation_config.get("block_size_tokens", 256)
        use_cuda_graphs_for_non_decode_steps = mcore_generation_config.get(
            "use_cuda_graphs_for_non_decode_steps", True
        )
        enable_chunked_prefill = mcore_generation_config.get(
            "enable_chunked_prefill", True
        )
        unified_memory_level = mcore_generation_config.get("unified_memory_level", 0)
        buffer_guaranteed_fraction = mcore_generation_config.get(
            "buffer_guaranteed_fraction", 0.1
        )
        max_tokens = mcore_generation_config.get("max_tokens", 16384)

        # Get the underlying language module (not the DDP wrapper) for toggle_cuda_graphs.
        # This follows the pattern from megatron/rl/rl_utils.py megatron_rl_inference_mode().
        # The DDP wrapper can interfere with CUDA graph state management.
        lang_module = (
            self.model.module.module
            if hasattr(self.model.module, "module")
            else self.model.module
        )

        # Enable CUDA graphs for inference using toggle_cuda_graphs on the underlying module.
        # reset_cuda_graphs=True ensures fresh CUDA graphs are created for each generate() call.
        

        toggle_cuda_graphs(lang_module, set_to="local", reset_cuda_graphs=True)

        model_config = self.model.config
        model_config.cuda_graph_impl = "local"

        dynamic_context = DynamicInferenceContext(
            params_dtype=inference_wrapper_config.params_dtype,
            num_layers=model_config.num_layers,
            kv_channels=model_config.kv_channels,
            num_attention_heads=model_config.num_query_groups,
            max_sequence_length=self.cfg["generation"]["max_new_tokens"],
            buffer_guaranteed_fraction=buffer_guaranteed_fraction,
            buffer_size_gb=buffer_size_gb,
            materialize_only_last_token_logits=False,
            num_cuda_graphs=num_cuda_graphs,
            block_size_tokens=block_size_tokens,
            tensor_model_parallel_size=self.cfg["megatron_cfg"][
                "tensor_model_parallel_size"
            ],
            use_cuda_graphs_for_non_decode_steps=use_cuda_graphs_for_non_decode_steps,
            use_flashinfer_fused_rope=False,
            unified_memory_level=unified_memory_level,
            max_tokens_override=max_tokens,
        )
        inference_wrapped_model = GPTInferenceWrapper(
            self.model, inference_wrapper_config, dynamic_context
        )

        inference_wrapped_model.prep_model_for_inference()
        # Set pipeline parallel flag
        inference_wrapped_model.model_is_pipeline_parallel = (
            self.cfg["megatron_cfg"]["pipeline_model_parallel_size"] > 1
        )

        text_generation_controller = TextGenerationController(
            inference_wrapped_model=inference_wrapped_model,
            tokenizer=self.megatron_tokenizer,
        )

        # Calculate seed based on node and rank to ensure reproducibility across workers
        local_rank = torch.cuda.current_device()  # Local GPU index on the node
        num_gpus_per_node = torch.cuda.device_count()
        node_idx = self.rank // num_gpus_per_node if num_gpus_per_node > 0 else 0
        seed = (node_idx * 1024) + local_rank

        # New API: DynamicInferenceEngine has additional parameters
        dynamic_engine = DynamicInferenceEngine(
            text_generation_controller,
            dynamic_context,
            enable_cuda_graph=True,
            random_seed=seed,
            track_paused_request_events=False,
            enable_chunked_prefill=enable_chunked_prefill,
            inference_logging_step_interval=0,
        )

        # Handle None values for top_k - convert to integer as required by Megatron
        top_k_cfg = self.cfg["generation"]["top_k"]
        top_k_val = 1 if greedy else (int(top_k_cfg) if top_k_cfg is not None else 0)

        top_p_cfg = self.cfg["generation"]["top_p"]
        top_p_val = (
            0.0 if greedy else (float(top_p_cfg) if top_p_cfg is not None else 0.0)
        )

        # New API: SamplingParams now includes termination_id and uses num_tokens_total
        sampling_params = SamplingParams(
            temperature=self.cfg["generation"]["temperature"] if not greedy else 0,
            top_k=top_k_val,
            top_p=top_p_val,
            skip_prompt_log_probs=False,
            return_log_probs=True,
            num_tokens_total=self.cfg["generation"]["max_new_tokens"],
            num_tokens_to_generate=None,
            termination_id=self.megatron_tokenizer.eod,
        )

        input_ids = data["input_ids"]
        prompt_tokens_tensor = input_ids.cuda()
        prompt_lengths_tensor = data["input_lengths"]
        request_id = 0

        # New API: add_request now takes sampling_params as a parameter
        for p, prompt_len in zip(
            prompt_tokens_tensor, prompt_lengths_tensor, strict=True
        ):
            dynamic_engine.add_request(
                request_id,
                p[:prompt_len],
                sampling_params=sampling_params,
            )
            request_id += 1

        result = []
        import time
        start = time.time()
        while dynamic_engine.has_unfinished_requests():
            result_step = dynamic_engine.step_modern(verbose=False)
            finished_requests = result_step.get("finished_requests", [])
            for finished_request in finished_requests:
                result.append(finished_request)
        end = time.time()
        print(f"SHAN : Time taken: {end - start} seconds")

        # Sort results by request_id to maintain original batch order
        result.sort(key=lambda x: x.request_id)

        out = {
            "tokens": [x.prompt_tokens.tolist() + x.generated_tokens for x in result],
            "logprobs": [x.prompt_log_probs + x.generated_log_probs for x in result],
        }

        input_lengths = data["input_lengths"]
        # pad the out "tokens" and "logprobs" and make them into tensors from lists
        batch_size = data["input_ids"].size(0)
        max_gen_seq_len = max([len(x.generated_tokens) for x in result])
        padded_input_length = input_ids.size(1)

        max_seq_len = padded_input_length + max_gen_seq_len
        # Create padded tensors for tokens and logprobs
        output_ids_padded = torch.full(
            (batch_size, max_seq_len),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=data["input_ids"].device,
        )

        logprobs_padded = torch.zeros(
            (batch_size, max_seq_len),
            dtype=torch.float,
            device=data["input_ids"].device,
        )

        # Fill in the padded tensors with actual values
        generation_lengths = torch.zeros(
            batch_size, dtype=torch.long, device=data["input_ids"].device
        )
        unpadded_sequence_lengths = torch.zeros(
            batch_size, dtype=torch.long, device=data["input_ids"].device
        )
        for i in range(batch_size):
            seq_len = len(out["tokens"][i])
            output_ids_padded[i, :seq_len] = torch.tensor(
                out["tokens"][i], dtype=torch.long, device=data["input_ids"].device
            )
            generation_lengths[i] = seq_len - input_lengths[i].item()
            unpadded_sequence_lengths[i] = seq_len
            logprob_len = len(out["logprobs"][i])
            logprobs_padded[i, 1 : logprob_len + 1] = torch.tensor(
                out["logprobs"][i],
                dtype=torch.float,
                device=data["input_ids"].device,
            )

        out_dict = {
            "output_ids": output_ids_padded,
            "logprobs": logprobs_padded,
            "generation_lengths": generation_lengths,
            "unpadded_sequence_lengths": unpadded_sequence_lengths,
        }

        self.model.config.flash_decode = False

        # Disable CUDA graphs after inference using toggle_cuda_graphs on the underlying module.
        # This follows the pattern from megatron/rl/rl_utils.py megatron_rl_inference_mode().
        # reset_cuda_graphs=True ensures:
        # 1. cudagraph_manager attributes are deleted from modules
        # 2. _CudagraphGlobalRecord state is reset via delete_cuda_graphs()
        # 3. Model is ready for training (avoids "inference tensor" errors)
        # 4. Next generate() call will create fresh CUDA graphs with new DynamicInferenceContext
        toggle_cuda_graphs(lang_module, set_to="none", reset_cuda_graphs=True)

        # Force release CUDA graph memory to prevent OOM at next step
        torch.cuda.empty_cache()

        no_grad.__exit__(None, None, None)

        return BatchedDataDict.from_batches([out_dict]).to("cpu")

    @torch.no_grad()
    @wrap_with_nvtx_name("megatron_policy_worker/prepare_refit_info")
    def prepare_refit_info(self) -> None:
        """Prepare state dict metadata for weight refitting and IPC streaming."""
        self.refit_param_info_mcore = self._calculate_refit_param_info()

        # Collect tensor metadata for refit / hf side info
        refit_param_info_hf = {}
        # Reuse shared iterator that appends FP8 KV/Q scales when enabled
        for name, tensor in self._iter_params_with_optional_kv_scales():
            refit_param_info_hf[name] = (tensor.shape, tensor.dtype)

        return refit_param_info_hf

    def _calculate_refit_param_info(self) -> list[tuple[str, int]]:
        """Calculate parameter information for refit.

        Each task contains:
        - param_name: Local parameter name without module prefixes
        - mapping: MegatronParamMapping instance for weight transformation
        - pp_rank: Pipeline-parallel rank owning the parameter
        - vp_stage: Virtual-pipeline stage index
        - megatron_module: Reference to Megatron model/submodule
        - param_weight: Target parameter tensor for converted weight

        Returns:
            List of (parameter_name, size_in_bytes) tuples.
        """
        self.refit_conversion_tasks = self.megatron_bridge.get_conversion_tasks(
            [self.model]
        )
        param_info = []

        def calculate_size_in_bytes(param, tp_size, ep_size):
            if param is None:
                # need to broadcast for other pp ranks
                size_in_bytes = None
            else:
                # Calculate size for this parameter
                prec_to_bytes = {
                    torch.bfloat16: 2,
                    torch.float16: 2,
                    torch.float32: 4,
                }
                scale = prec_to_bytes[self.dtype] / prec_to_bytes[param.dtype]
                size_in_bytes = (
                    param.element_size() * param.numel() * tp_size * ep_size * scale
                )

            # Broadcast size_in_bytes across pipeline parallel ranks
            return broadcast_object_across_pp_ranks(size_in_bytes)

        for task in self.refit_conversion_tasks:
            param_info.append(
                (
                    task.param_name,
                    calculate_size_in_bytes(
                        task.param_weight,
                        task.mapping.tp_size,
                        task.mapping.ep_size if task.mapping.is_expert else 1,
                    ),
                )
            )
        return param_info

    def _iter_params_with_optional_kv_scales(
        self,
        kv_scales: Optional[dict[str, float]] = None,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield exported HF parameters and optionally append FP8 KV/Q scale tensors.

        This helper is used by both IPC-based streaming and collective broadcast
        so that the logic for adding KV scales stays consistent in one place.
        """
        base_iter = self.megatron_bridge.export_hf_weights(
            [self.model],
            show_progress=False,
            conversion_tasks=self.refit_conversion_tasks,  # used for metadata caching
        )

        # Yield the original parameters first.
        for name, tensor in base_iter:
            yield name, tensor

        # Check whether FP8 KV cache is enabled.
        use_fp8_kv_cache = False
        if (
            "generation" in self.cfg
            and self.cfg["generation"] is not None
            and self.cfg["generation"]["backend"] == "vllm"
        ):
            generation_cfg = cast(VllmConfig, self.cfg["generation"])
            use_fp8_kv_cache = (
                "vllm_cfg" in generation_cfg
                and "kv_cache_dtype" in generation_cfg["vllm_cfg"]
                and generation_cfg["vllm_cfg"]["kv_cache_dtype"].startswith("fp8")
            )

        if not use_fp8_kv_cache:
            return

        # Append KV (and potentially Q) scale entries to match metadata.
        num_layers = self.megatron_bridge.transformer_config.num_layers
        keys: list[str] = []
        for layer_idx in range(num_layers):
            scale_names = get_vllm_qkv_scale_names(layer_idx)
            keys.extend(scale_names.values())

        for param_name in keys:
            if kv_scales and param_name in kv_scales:
                scale_value = kv_scales[param_name]
            else:
                scale_value = 1.0
            scale_tensor = torch.tensor(
                scale_value, dtype=torch.float32, device="cuda"
            ).reshape(1)
            yield param_name, scale_tensor

    @torch.no_grad()
    @wrap_with_nvtx_name("megatron_policy_worker/stream_weights_via_ipc_zmq")
    def stream_weights_via_ipc_zmq(
        self, buffer_size_bytes: int = 0, kv_scales: Optional[dict[str, float]] = None
    ) -> None:
        """Stream model weights to peer process via ZMQ IPC socket."""
        self.maybe_init_zmq()

        from nemo_rl.models.policy.utils import stream_weights_via_ipc_zmq_impl

        # Use the shared implementation to append optional KV scales.
        stream_weights_via_ipc_zmq_impl(
            params_generator=self._iter_params_with_optional_kv_scales(
                kv_scales=kv_scales
            ),
            buffer_size_bytes=buffer_size_bytes,
            zmq_socket=self.zmq_socket,
            rank=self.rank,
            worker_name=str(self),
        )

    @torch.no_grad()
    def broadcast_weights_for_collective(
        self, kv_scales: Optional[dict[str, float]] = None
    ) -> None:
        """Broadcast the weights for collective communication."""
        # param_iterator will return (name, tensor), we only need tensor.
        packed_broadcast_producer(
            iterator=self._iter_params_with_optional_kv_scales(kv_scales=kv_scales),
            group=self.model_update_group,
            src=0,
            post_iter_func=lambda x: x[1],
        )

    def prepare_for_lp_inference(self):
        self.model = self.move_model(self.model, "cuda", move_grads=False)
        self.model.eval()

        # offload grads to cpu
        self.model = self.move_model(
            self.model, "cpu", move_params=False, move_grads=True
        )  # get rid of grad buffers

        # offload optimizer to cpu
        torch.randn(1).cuda()  # wake up torch allocator
        if (
            hasattr(self, "optimizer")
            and self.optimizer is not None
            and not self.optimizer_cpu_offload
            and self.offload_optimizer_for_logprob
        ):
            self.move_optimizer("cpu")

        gc.collect()
        torch.cuda.empty_cache()

    def prepare_for_training(self, *args, **kwargs):
        # onload models and optimizer state to cuda
        self.model = self.move_model(
            self.model, "cuda", move_grads=True, move_params=True
        )
        self.model.train()

        # Move optimizer state to CUDA if it exists
        # colocated generation will always offload optimizer to cuda before refit
        if (
            hasattr(self, "optimizer")
            and self.optimizer is not None
            and not self.optimizer_cpu_offload
            and (self.offload_optimizer_for_logprob or self.is_generation_colocated)
        ):
            self.move_optimizer("cuda")

        if self.cfg["megatron_cfg"]["empty_unused_memory_level"] >= 1:
            torch.cuda.empty_cache()

    @wrap_with_nvtx_name("megatron_policy_worker/offload_before_refit")
    def offload_before_refit(self):
        """Offload the optimizer and buffers to the CPU."""
        no_grad = torch.no_grad()
        no_grad.__enter__()
        allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
        reserved = torch.cuda.memory_reserved() / (1024**3)  # Convert to GB
        print(
            f"GPU Memory before optimizer offload: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )
        self.model = self.move_model(
            self.model, "cpu", move_params=False, move_grads=True
        )  # get rid of grad buffers
        torch.randn(1).cuda()  # wake up torch allocator
        if (
            hasattr(self, "optimizer")
            and self.optimizer is not None
            and not self.optimizer_cpu_offload
        ):
            self.move_optimizer("cpu")

        gc.collect()
        torch.cuda.empty_cache()

        # Print memory stats after offloading
        allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
        reserved = torch.cuda.memory_reserved() / (1024**3)  # Convert to GB
        print(
            f"GPU Memory after optimizer offload: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )
        no_grad.__exit__(None, None, None)

    @wrap_with_nvtx_name("megatron_policy_worker/offload_after_refit")
    def offload_after_refit(self):
        """Offload as much as possible on the CPU."""
        no_grad = torch.no_grad()
        no_grad.__enter__()
        self.model = self.move_model(self.model, "cpu")
        self.model.eval()
        torch.randn(1).cuda()  # wake up torch allocator
        self.offload_before_refit()  # rerun the old offload function

        allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
        reserved = torch.cuda.memory_reserved() / (1024**3)  # Convert to GB
        print(
            f"GPU Memory after refit complete: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )
        no_grad.__exit__(None, None, None)

    @torch.no_grad()
    def move_model(
        self,
        model: torch.nn.Module,
        device: str,
        move_params: bool = True,
        move_grads: bool = True,
    ) -> torch.nn.Module:
        # move all param and grad buffers to the device
        if isinstance(model, DistributedDataParallel):
            # DDP case
            for buffers in [model.buffers, model.expert_parallel_buffers]:
                for buffer_idx in range(len(buffers)):
                    if device == "cpu":
                        buffers[buffer_idx].offload_to_cpu(
                            move_params=move_params, move_grads=move_grads
                        )
                    elif device == "cuda":
                        buffers[buffer_idx].reload_from_cpu(
                            move_params=move_params, move_grads=move_grads
                        )
                    else:
                        raise ValueError(
                            f"Invalid device: {device}. Only strings 'cpu' and 'cuda' are supported."
                        )
        elif isinstance(model, custom_FSDP):
            if device == "cpu":
                model.param_and_grad_buffer.offload_to_cpu(move_params, move_grads)
            elif device == "cuda":
                model.param_and_grad_buffer.reload_from_cpu(
                    move_params=move_params, move_grads=move_grads
                )
            else:
                raise ValueError(
                    f"Invalid device: {device}. Only strings 'cpu' and 'cuda' are supported."
                )
        else:
            # Ordinary offload case
            if move_params:
                for name, param in model.state_dict().items():
                    new_state_dict = {}
                    for name, item in model.state_dict().items():
                        if isinstance(item, torch.Tensor):
                            item = item.detach().to(
                                device=device, non_blocking=True, copy=True
                            )
                        new_state_dict[name] = item
                    model.load_state_dict(new_state_dict)
        return model

    def move_optimizer(self, device: str):
        # Iterate through the state dictionaries for each parameter group
        if isinstance(self.optimizer, ChainedOptimizer):
            optimizer_state = self.optimizer.state
        else:
            optimizer_state = self.optimizer._get_state()
        for _, state in optimizer_state.items():
            # Iterate through the state items (e.g., momentum, variance) for a parameter
            for k, v in state.items():
                # Check if the item is a tensor
                if torch.is_tensor(v):
                    # Move the tensor to device and update the state dictionary
                    if device == "cpu":
                        if v.is_cuda:
                            state[k] = v.to("cpu")
                    elif device == "cuda":
                        if not v.is_cuda:
                            state[k] = v.to("cuda")
                    else:
                        raise ValueError(
                            f"Invalid device: {device}. Only strings 'cpu' and 'cuda' are supported."
                        )

    def save_checkpoint(
        self,
        weights_path: str,
        optimizer_path: Optional[str] = None,
        **kwargs,
    ):
        """Save a training checkpoint.

        Args:
            weights_path: The specific directory path where the checkpoint will be saved.
            optimizer_path: If not None, optimizer and scheduler states are saved if they exist.
        """
        if not torch.distributed.is_initialized():
            raise RuntimeError(
                "Distributed process group is not initialized. Cannot save checkpoint."
            )

        if self.mcore_state is None or self.model is None:
            raise RuntimeError(
                "Megatron core state or model is not initialized. Cannot save checkpoint."
            )

        original_save_path = self.mcore_state.cfg.checkpoint.save
        # save_dir = os.path.dirname(weights_path)
        release_name = os.path.basename(weights_path)

        try:
            maybe_finalize_async_save(
                self.mcore_state,
                ckpt_cfg=self.mcore_state.cfg.checkpoint,
                blocking=False,
            )
            self.mcore_state.cfg.checkpoint.save = weights_path

            optimizer_to_save = None
            scheduler_to_save = None

            if optimizer_path is not None:
                if self.optimizer is not None:
                    optimizer_to_save = self.optimizer
                if self.scheduler is not None:
                    scheduler_to_save = self.scheduler

            # Ensure model is in eval mode for consistent saving, unless actively training
            # This is a common practice, though NeMo's save might handle this.
            # For safety, if not in training loop, setting to eval.
            is_training = self.model.training
            if not is_training:
                self.model.eval()

            if self.should_disable_forward_pre_hook:
                self.disable_forward_pre_hook()
            save_checkpoint(
                state=self.mcore_state,
                model=[self.model],
                optimizer=optimizer_to_save,
                opt_param_scheduler=scheduler_to_save,
                num_floating_point_operations_so_far=self.mcore_state.train_state.floating_point_operations_so_far,
                checkpointing_context=self.checkpointing_context,
            )
            print(f"Saved checkpoint to {weights_path}")
            maybe_finalize_async_save(
                self.mcore_state,
                ckpt_cfg=self.mcore_state.cfg.checkpoint,
                blocking=True,
                terminate=True,
            )
            if self.should_disable_forward_pre_hook:
                self.enable_forward_pre_hook()

            if not is_training:  # Restore training state if it was changed
                self.model.train()

        except Exception as e:
            print(f"Failed to save checkpoint to {weights_path}: {e}")
            raise
        finally:
            self.mcore_state.cfg.checkpoint.save = original_save_path

    def load_checkpoint(self, weights_path: str, optimizer_path: Optional[str] = None):
        """Load a training checkpoint.

        Args:
            weights_path: The exact directory path from which to load the checkpoint.
            optimizer_path: If not None, attempts to load optimizer and scheduler states
                            if self.optimizer and self.scheduler are initialized.
        """
        raise NotImplementedError(
            "Loading checkpoints outside of the init function is not yet implemented for Megatron policy."
        )

    def check_tensor_parallel_attributes(self) -> dict[str, Any]:
        """Check tensor parallel attributes on model parameters.

        Returns:
            Dictionary containing information about tensor parallel parameters:
            - tp_params: List of parameter names that have tensor_model_parallel=True
            - non_tp_params: List of parameter names that have tensor_model_parallel=False
            - total_params: Total number of parameters checked
            - tp_size: Tensor parallel size from config
        """
        tp_params = []
        non_tp_params = []
        total_params = 0

        for name, param in self.model.named_parameters():
            total_params += 1
            tensor_model_parallel = getattr(param, "tensor_model_parallel", False)

            if tensor_model_parallel:
                tp_params.append(
                    {
                        "name": name,
                        "tensor_model_parallel": tensor_model_parallel,
                        "partition_dim": getattr(param, "partition_dim", None),
                        "partition_stride": getattr(param, "partition_stride", None),
                        "shape": list(param.shape),
                    }
                )
            else:
                non_tp_params.append(
                    {
                        "name": name,
                        "tensor_model_parallel": tensor_model_parallel,
                        "shape": list(param.shape),
                    }
                )

        return {
            "tp_params": tp_params,
            "non_tp_params": non_tp_params,
            "total_params": total_params,
            "tp_size": self.megatron_cfg.model.tensor_model_parallel_size,
        }

    @torch.no_grad()
    def calibrate_qkv_fp8_scales(
        self,
        *,
        data: BatchedDataDict[Any],
        micro_batch_size: Optional[int] = None,
        percentile: float = 99.9,
        margin: float = 1.05,
        include_q: bool = False,
    ) -> dict[str, Any]:
        """One-shot calibration of Q/K/V activation scales (for FP8 KV cache).

        - Captures each layer's `query_key_value` output through forward hooks, splits Q/K/V, and computes percentile amax.
        - In parallel (DP/TP/PP) environments, first computes local percentiles, then takes max across all ranks for conservativeness.
        - By default only returns and saves K/V scales, optionally returns Q.

        Args:
            data: Representative sample batch for calibration, following get_logprobs input conventions.
            micro_batch_size: Micro batch size during calibration; if None, reuses logprob_batch_size.
            percentile: Percentile for amax (e.g. 99.9).
            margin: Margin factor, e.g. 1.05.
            save_path: If provided, rank0 will save results as JSON.
            include_q: Whether to also return Q scale (usually only K/V needed).

        Returns:
            { "format": "fp8", "percentile": float, "margin": float,
              "layers": { layer_name: {"k_scale": float, "v_scale": float[, "q_scale": float] } } }
        """

        # Allow overriding FP8 max for Q, K, V via environment variables for ease of testing.
        # Defaults align with FP8 e4m3 max magnitude.
        # Use different defaults for Q, K, V to adapt to distribution diffefences
        def _get_env_float(name: str, default: float) -> float:
            try:
                val = os.getenv(name, None)
                return float(val) if val is not None and val != "" else default
            except Exception:
                return default

        FP8_MAX_Q = _get_env_float("FP8_MAX_Q", 448.0)
        FP8_MAX_K = _get_env_float("FP8_MAX_K", 448.0)
        FP8_MAX_V = _get_env_float("FP8_MAX_V", 448.0)

        self.model.eval()

        # Record local percentile amax for q/k/v of each layer
        layer_to_samples_q: dict[str, list[float]] = defaultdict(list)
        layer_to_samples_k: dict[str, list[float]] = defaultdict(list)
        layer_to_samples_v: dict[str, list[float]] = defaultdict(list)
        hook_handles = []

        def _extract_layer_key(module_name: str) -> str:
            # Expected format: "module.decoder.layers.<idx>.self_attention.query_key_value"
            m = re.search(r"module\.decoder\.layers\.(\d+)", module_name)
            if m is not None:
                return f"layer_{m.group(1)}"
            return module_name

        # Hook to capture q/k/v after q/k norm and RoPE
        def _pre_hook_builder_core_attention(module_name: str):
            layer_key = _extract_layer_key(module_name)

            def _pre_hook(module, inputs):
                args = inputs if isinstance(inputs, (tuple, list)) else (inputs,)
                if len(args) == 1 and isinstance(args[0], (tuple, list)):
                    args = args[0]
                # Expected first 3 args to be q, k, v (typical signature for Megatron CoreAttention)
                q = args[0]
                k = args[1]
                v = args[2]
                if include_q:
                    layer_to_samples_q[layer_key].append(
                        float(torch.amax(torch.abs(q)).item())
                    )
                layer_to_samples_k[layer_key].append(
                    float(torch.amax(torch.abs(k)).item())
                )
                layer_to_samples_v[layer_key].append(
                    float(torch.amax(torch.abs(v)).item())
                )

            return _pre_hook

        matched_modules = []
        # Try to register forward_pre_hook on core_attention first
        for name, module in self.model.named_modules():
            if "self_attention.core_attention" in name:
                try:
                    handle = module.register_forward_pre_hook(
                        _pre_hook_builder_core_attention(name)
                    )
                    hook_handles.append(handle)
                    matched_modules.append((name, module.__class__.__name__, "pre"))
                except Exception as e:
                    print(
                        f"Error registering pre-hook for qkv scale calibration on {name}: {e}"
                        " Please check if the model is compatible with the current calibration logic. "
                        "The expected module name is 'self_attention.core_attention'."
                    )
                    raise

        # Run a forward pass to trigger hooks (reuse get_logprobs forward path)
        try:
            _ = self.get_logprobs(data=data, micro_batch_size=micro_batch_size)
        finally:
            for h in hook_handles:
                try:
                    h.remove()
                except Exception as e:
                    print(f"Error removing hook for qkv scale calibration: {e}")
                    raise

        # Compute local percentile amax
        def _percentile(values: list[float], p: float) -> float:
            if not values:
                return 0.0
            t = torch.tensor(sorted(values), device="cuda", dtype=torch.float32)
            rank = max(
                0, min(len(values) - 1, int(round((p / 100.0) * (len(values) - 1))))
            )
            return float(t[rank].item())

        local_layer_to_pamax = {}
        for layer_key in set(
            list(layer_to_samples_k.keys())
            + list(layer_to_samples_v.keys())
            + (list(layer_to_samples_q.keys()) if include_q else [])
        ):
            entry = {}
            if include_q:
                entry["q_amax_p"] = _percentile(
                    layer_to_samples_q.get(layer_key, []), percentile
                )
            entry["k_amax_p"] = _percentile(
                layer_to_samples_k.get(layer_key, []), percentile
            )
            entry["v_amax_p"] = _percentile(
                layer_to_samples_v.get(layer_key, []), percentile
            )
            local_layer_to_pamax[layer_key] = entry

        # Merge across all ranks: take maximum of percentile amax (conservative approach)
        world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )
        gathered = [None for _ in range(world_size)] if world_size > 1 else None
        if world_size > 1:
            torch.distributed.all_gather_object(gathered, local_layer_to_pamax)
            merged = defaultdict(dict)
            for d in gathered:  # type: ignore
                if d is None:
                    continue
                for k, v in d.items():
                    dst = merged[k]
                    for kk, vv in v.items():
                        dst[kk] = max(dst.get(kk, 0.0), float(vv))
            layer_to_pamax = dict(merged)
        else:
            layer_to_pamax = local_layer_to_pamax

        # Compute scale (symmetric quantization): scale = pamax / fp8_max
        result_layers = {}
        for layer_key, vals in layer_to_pamax.items():
            out_entry = {}
            if include_q:
                q_scale = (vals.get("q_amax_p", 0.0) * margin) / FP8_MAX_Q
                out_entry["q_scale"] = float(q_scale)
            k_scale = (vals.get("k_amax_p", 0.0) * margin) / FP8_MAX_K
            v_scale = (vals.get("v_amax_p", 0.0) * margin) / FP8_MAX_V
            out_entry["k_scale"] = float(k_scale)
            out_entry["v_scale"] = float(v_scale)
            result_layers[layer_key] = out_entry

        vllm_format_scales = convert_calibration_to_vllm_format(result_layers)

        final_result = {
            "format": "fp8",
            "percentile": percentile,
            "margin": margin,
            "layers": vllm_format_scales,
        }

        # Sync results across all ranks (broadcast rank0's result)
        if world_size > 1:
            if torch.distributed.get_rank() == 0:
                obj_list = [final_result]
                torch.distributed.broadcast_object_list(obj_list, src=0)
                final_result = obj_list[0]
            else:
                obj_list = [None]
                torch.distributed.broadcast_object_list(obj_list, src=0)
                final_result = obj_list[0]  # type: ignore

        return final_result


class CustomFloat16Module(Float16Module):
    """Float 16 Module.

    Attributes:
        config (TransformerConfig): Transformer config
        fp16 (bool) : Specifies if the model runs in fp16 mode
        bf16 (bool) : Specifies if the model runs in bf16 mode

    Args:
        config (TransformerConfig): The transformer config used to initalize the model
    """

    def __init__(self, config: TransformerConfig, module: torch.nn.Module):
        super(CustomFloat16Module, self).__init__(config, module)
        self.re_enable_float32_expert_bias()

    def re_enable_float32_expert_bias(self) -> None:
        """Ensure MoE router expert bias stays in float32 for numerical stability.

        Walks the wrapped module to find MoE routers and invokes the
        `_maintain_float32_expert_bias()` helper which recreates or casts the
        expert bias tensors to float32 as required by Megatron-LM.
        """
        module = self.module
        # Handle VLM models where language model is nested
        if hasattr(module, "language_model"):
            module = module.language_model
        if hasattr(module, "decoder") and hasattr(module.decoder, "layers"):
            for layer in module.decoder.layers:
                mlp = getattr(layer, "mlp", None)
                router = getattr(mlp, "router", None) if mlp is not None else None
                if router is not None and hasattr(
                    router, "_maintain_float32_expert_bias"
                ):
                    router._maintain_float32_expert_bias()

```

All linter warnings are pre-existing import resolution issues.

## Summary of Changes

I've added `torch.cuda.empty_cache()` calls to force release CUDA graph memory:

### 1. In `generate()` after disabling CUDA graphs:
```python
toggle_cuda_graphs(lang_module, set_to="none", reset_cuda_graphs=True)

# Force release CUDA graph memory to prevent OOM at next step
torch.cuda.empty_cache()
```

### 2. In `train()` before training starts:
```python
toggle_cuda_graphs(lang_module, set_to="none", reset_cuda_graphs=True)

# Force release any cached CUDA graph memory before training
torch.cuda.empty_cache()
```

## Why This Should Help

The OOM was likely caused by:
1. **Step 1**: CUDA graphs allocated (~X GB)
2. **Step 1 end**: `delete_cuda_graphs()` called, but PyTorch keeps memory cached
3. **Step 2**: New CUDA graphs allocated (~X GB), but old memory still cached
4. **Training**: Not enough free memory → OOM

With `torch.cuda.empty_cache()`:
- Forces PyTorch to release cached (but unused) GPU memory
- Ensures memory from Step 1's CUDA graphs is truly freed before Step 2

Try running your application again. If OOM still occurs, we may need to look at other memory optimizations (like reducing the number of CUDA graph warmup sizes).

```python
toggle_cuda_graphs(lang_module, set_to="none", reset_cuda_graphs=True)

# Force release CUDA graph memory to prevent OOM at next step
torch.cuda.empty_cache()
```

```python
toggle_cuda_graphs(lang_module, set_to="none", reset_cuda_graphs=True)

# Force release any cached CUDA graph memory before training
torch.cuda.empty_cache()
```

---

**User**

sTILL GETTING THIS ISSUE
```

========================= Step 2/500 =========================
▶ Preparing batch...
▶ Generating responses for batch of size 512...
(MegatronPolicyWorker[rank=0] pid=833730) GPU Memory before optimizer offload: 36.80GB allocated, 42.80GB reserved
(MegatronPolicyWorker[rank=0] pid=833730) /opt/nemo-rl/3rdparty/Megatron-LM-workspace/Megatron-LM/megatron/core/inference/engines/dynamic_engine.py:114: UserWarning: The `enable_cuda_graph` argument is deprecated and will be removed in `megatron-core 0.15`. `enable_cuda_graph` is now read directly from the transformer config object.
(MegatronPolicyWorker[rank=0] pid=833730)   warnings.warn(
cuda graph warmup - decode, d 1464:   0%|          | 0/16 [00:00<?, ?it/s]
(MegatronPolicyWorker[rank=0] pid=833730) GPU Memory after optimizer offload: 22.36GB allocated, 23.67GB reserved
(MegatronPolicyWorker[rank=0] pid=833730) GPU Memory after refit complete: 22.36GB allocated, 23.67GB reserved
cuda graph warmup - decode, d 1440:   6%|▋         | 1/16 [00:00<00:04,  3.66it/s]
cuda graph warmup - decode, d 1344:  12%|█▎        | 2/16 [00:00<00:03,  3.99it/s]
cuda graph warmup - decode, d 1248:  19%|█▉        | 3/16 [00:00<00:02,  4.55it/s]
cuda graph warmup - decode, d 1152:  25%|██▌       | 4/16 [00:00<00:02,  4.66it/s]
cuda graph warmup - decode, d 1056:  31%|███▏      | 5/16 [00:01<00:02,  4.61it/s]
cuda graph warmup - decode, d 960:  38%|███▊      | 6/16 [00:01<00:02,  4.37it/s]
cuda graph warmup - decode, d 864:  44%|████▍     | 7/16 [00:01<00:01,  4.57it/s]
cuda graph warmup - decode, d 768:  50%|█████     | 8/16 [00:01<00:01,  4.57it/s]
cuda graph warmup - decode, d 672:  56%|█████▋    | 9/16 [00:02<00:01,  4.57it/s]
cuda graph warmup - decode, d 576:  62%|██████▎   | 10/16 [00:02<00:01,  4.48it/s]
cuda graph warmup - decode, d 480:  69%|██████▉   | 11/16 [00:02<00:01,  4.58it/s]
cuda graph warmup - decode, d 384:  75%|███████▌  | 12/16 [00:02<00:00,  4.65it/s]
cuda graph warmup - decode, d 288:  81%|████████▏ | 13/16 [00:02<00:00,  4.69it/s]
cuda graph warmup - decode, d 192:  88%|████████▊ | 14/16 [00:03<00:00,  4.67it/s]
cuda graph warmup - decode, d 96:  94%|█████████▍| 15/16 [00:03<00:00,  4.76it/s]
cuda graph warmup - decode, d 96: 100%|██████████| 16/16 [00:03<00:00,  4.58it/s]
cuda graph warmup - non_decode, d 1464:   0%|          | 0/16 [00:00<?, ?it/s]
cuda graph warmup - non_decode, d 1440:   6%|▋         | 1/16 [00:00<00:03,  4.29it/s]
cuda graph warmup - non_decode, d 1344:  12%|█▎        | 2/16 [00:00<00:03,  4.27it/s]
cuda graph warmup - non_decode, d 1248:  19%|█▉        | 3/16 [00:00<00:03,  4.23it/s]
cuda graph warmup - non_decode, d 1152:  25%|██▌       | 4/16 [00:00<00:02,  4.33it/s]
cuda graph warmup - non_decode, d 1056:  31%|███▏      | 5/16 [00:01<00:02,  4.31it/s]
cuda graph warmup - non_decode, d 1056:  38%|███▊      | 6/16 [00:01<00:02,  4.55it/s]
cuda graph warmup - non_decode, d 960:  38%|███▊      | 6/16 [00:01<00:02,  4.55it/s]
cuda graph warmup - non_decode, d 864:  44%|████▍     | 7/16 [00:01<00:01,  4.72it/s]
cuda graph warmup - non_decode, d 768:  50%|█████     | 8/16 [00:01<00:01,  4.73it/s]
cuda graph warmup - non_decode, d 672:  56%|█████▋    | 9/16 [00:01<00:01,  4.71it/s]
cuda graph warmup - non_decode, d 576:  62%|██████▎   | 10/16 [00:02<00:01,  4.31it/s]
cuda graph warmup - non_decode, d 480:  69%|██████▉   | 11/16 [00:02<00:01,  4.31it/s]
cuda graph warmup - non_decode, d 384:  75%|███████▌  | 12/16 [00:02<00:00,  4.13it/s]
cuda graph warmup - non_decode, d 288:  81%|████████▏ | 13/16 [00:02<00:00,  4.37it/s]
cuda graph warmup - non_decode, d 192:  88%|████████▊ | 14/16 [00:03<00:00,  4.42it/s]
cuda graph warmup - non_decode, d 96:  94%|█████████▍| 15/16 [00:03<00:00,  4.47it/s]
cuda graph warmup - non_decode, d 96: 100%|██████████| 16/16 [00:03<00:00,  4.44it/s]
(MegatronPolicyWorker[rank=0] pid=833730) SHAN : Time taken: 6.756725788116455 seconds
▶ Processing rewards...,
▶ Computing advantages...
▶ Preparing for logprob inference...
▶ Computing logprobs...
▶ Preparing for training...
▶ Training policy...
Traceback (most recent call last):
  File "/opt/nemo-rl/examples/run_grpo_math.py", line 260, in <module>
    main()
  File "/opt/nemo-rl/examples/run_grpo_math.py", line 243, in main
    grpo_train(
  File "/opt/nemo-rl/nemo_rl/algorithms/grpo.py", line 1352, in grpo_train
    train_results = policy.train(train_data, loss_fn)
                    ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/nemo-rl/nemo_rl/models/policy/lm_policy.py", line 532, in train
    results = self.worker_group.get_all_worker_results(futures)
              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
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
ray.exceptions.RayTaskError(OutOfMemoryError): ray::MegatronPolicyWorker.train() (pid=833730, ip=10.65.29.29, actor_id=c6f7f0a8d248670a2fc3be2b01000000, repr=MegatronPolicyWorker[rank=0])
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/nemo-rl/nemo_rl/utils/nsys.py", line 88, in wrapper
    ret = func(*args, **kwargs)
          ^^^^^^^^^^^^^^^^^^^^^
  File "/opt/nemo-rl/nemo_rl/models/policy/workers/megatron_policy_worker.py", line 1099, in train
    losses_reduced = forward_backward_func(
                     ^^^^^^^^^^^^^^^^^^^^^^
  File "/opt/nemo-rl/3rdparty/Megatron-LM-workspace/Megatron-LM/megatron/core/pipeline_parallel/schedules.py", line 629, in forward_backward_no_pipelining
    backward_step(
  File "/opt/nemo-rl/3rdparty/Megatron-LM-workspace/Megatron-LM/megatron/core/pipeline_parallel/schedules.py", line 480, in backward_step
    custom_backward(output_tensor[0], output_tensor_grad[0])
  File "/opt/nemo-rl/3rdparty/Megatron-LM-workspace/Megatron-LM/megatron/core/pipeline_parallel/schedules.py", line 173, in custom_backward
    Variable._execution_engine.run_backward(
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.15 GiB. GPU 0 has a total capacity of 79.11 GiB of which 782.88 MiB is free. Including non-PyTorch memory, this process has 78.31 GiB memory in use. Of the allocated memory 71.14 GiB is allocated by PyTorch, with 92.22 MiB allocated in private pools (e.g., CUDA Graphs), and 3.10 GiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
```

---

**Cursor**

Looking at the memory numbers more carefully:

```
71.14 GiB allocated by PyTorch
92.22 MiB allocated in private pools (e.g., CUDA Graphs)  ← CUDA graphs are tiny!
3.10 GiB is reserved by PyTorch but unallocated ← FRAGMENTATION
```

**CUDA graphs only use 92MB** - they're not the problem. The issue is **memory fragmentation** - there's 3.1 GB reserved but unusable because it's fragmented.

## Key Insight

- **Step 1 works** because GPU memory is "clean"
- **Step 2 fails** because after Step 1's generate→train cycle, memory becomes fragmented
- Training needs a contiguous 1.15 GB block, but only 782 MB contiguous is available

## Solutions

### Option 1: Try PyTorch's expandable segments (quick fix)

Set this environment variable before running:
```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

### Option 2: Force garbage collection and cache clearing between steps

Let me add more aggressive memory cleanup. First, let me check what happens between logprob computation and training:

```plaintext
71.14 GiB allocated by PyTorch
92.22 MiB allocated in private pools (e.g., CUDA Graphs)  ← CUDA graphs are tiny!
3.10 GiB is reserved by PyTorch but unallocated ← FRAGMENTATION
```

```shellscript
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

Let me add more aggressive memory cleanup at the start of `train()`:

```python
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
import gc
import os
import re
import time
import warnings
from collections import defaultdict
from contextlib import AbstractContextManager, contextmanager, nullcontext
from functools import partial
from typing import Any, Iterator, Optional, TypeVar, cast

import ray
import torch
from megatron.bridge import AutoBridge
from megatron.bridge.models.model_provider import get_model
from megatron.bridge.training import fault_tolerance
from megatron.bridge.training.checkpointing import (
    checkpoint_exists,
    init_checkpointing_context,
    load_checkpoint,
    maybe_finalize_async_save,
    save_checkpoint,
)
from megatron.bridge.training.config import (
    CheckpointConfig,
    ConfigContainer,
    DistributedDataParallelConfig,
    LoggerConfig,
    OptimizerConfig,
    SchedulerConfig,
    TokenizerConfig,
    TrainingConfig,
)
from megatron.bridge.training.initialize import (
    initialize_megatron,
    set_jit_fusion_options,
)
from megatron.bridge.training.optim import setup_optimizer
from megatron.bridge.training.setup import (
    _update_model_config_funcs,
)
from megatron.bridge.training.state import GlobalState
from megatron.bridge.training.tokenizers.tokenizer import build_tokenizer
from megatron.bridge.training.utils.train_utils import (
    logical_and_across_model_parallel_group,
    reduce_max_stat_across_model_parallel_group,
)
from megatron.bridge.utils.common_utils import get_rank_safe
from megatron.bridge.utils.instantiate_utils import InstantiationMode
from megatron.bridge.utils.vocab_utils import calculate_padded_vocab_size
from megatron.core import parallel_state
from megatron.core.transformer.utils import toggle_cuda_graphs
from megatron.core.distributed import DistributedDataParallel
from megatron.core.distributed.fsdp.mcore_fsdp_adapter import (
    FullyShardedDataParallel as custom_FSDP,
)
from megatron.core.inference.model_inference_wrappers.inference_wrapper_config import (
    InferenceWrapperConfig,
)
from megatron.core.inference.text_generation_controllers.text_generation_controller import (
    TextGenerationController,
)
from megatron.core.models.gpt import GPTModel
from megatron.core.optimizer import ChainedOptimizer
from megatron.core.parallel_state import (
    get_context_parallel_group,
    get_context_parallel_rank,
    get_pipeline_model_parallel_group,
    get_pipeline_model_parallel_last_rank,
    get_pipeline_model_parallel_world_size,
    get_tensor_model_parallel_group,
    get_tensor_model_parallel_rank,
    is_pipeline_last_stage,
)
from megatron.core.pipeline_parallel import get_forward_backward_func
from megatron.core.rerun_state_machine import get_rerun_state_machine
from megatron.core.transformer.module import Float16Module
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.training.utils import get_ltor_masks_and_position_ids
from ray.util.queue import Queue
from transformers import PreTrainedTokenizerBase

from nemo_rl.algorithms.interfaces import LossFunction, LossType
from nemo_rl.distributed.batched_data_dict import BatchedDataDict
from nemo_rl.distributed.model_utils import (
    allgather_cp_sharded_tensor,
    distributed_vocab_topk,
    from_parallel_logits_to_logprobs,
    from_parallel_logits_to_logprobs_packed_sequences,
)
from nemo_rl.distributed.named_sharding import NamedSharding
from nemo_rl.models.generation.fp8 import (
    convert_calibration_to_vllm_format,
    get_vllm_qkv_scale_names,
)
from nemo_rl.models.generation.interfaces import (
    GenerationDatumSpec,
    GenerationOutputSpec,
    verify_right_padding,
)
from nemo_rl.models.generation.vllm.config import VllmConfig
from nemo_rl.models.megatron.common import (
    _get_pack_sequence_parameters_for_megatron,
    _pack_sequences_for_megatron,
    broadcast_tensor,
    forward_step_arbitrary_loss,
    get_moe_metrics,
)
from nemo_rl.models.megatron.community_import import import_model_from_hf_name
from nemo_rl.models.policy import PolicyConfig
from nemo_rl.models.policy.interfaces import (
    ColocatablePolicyInterface,
    LogprobOutputSpec,
)
from nemo_rl.models.policy.utils import (
    configure_dynamo_cache,
    get_megatron_checkpoint_dir,
    get_runtime_env_for_policy_worker,
)
from nemo_rl.models.policy.workers.base_policy_worker import AbstractPolicyWorker
from nemo_rl.models.policy.workers.patches import apply_transformer_engine_patch
from nemo_rl.utils.nsys import wrap_with_nvtx_name
from nemo_rl.utils.packed_tensor import packed_broadcast_producer

try:
    from megatron.core.distributed import (
        TorchFullyShardedDataParallel as torch_FSDP,  # noqa: F401 unused-import
    )

    HAVE_FSDP2 = True
except ImportError:
    HAVE_FSDP2 = False

TokenizerType = TypeVar("TokenizerType", bound=PreTrainedTokenizerBase)


def broadcast_object_across_pp_ranks(obj):
    """Broadcast an object across pipeline parallel ranks.

    This utility function handles broadcasting an object from the rank that owns it
    to all other pipeline parallel ranks. If only one rank has the object (non-None),
    it will be broadcast to all other ranks.

    Args:
        obj: The object to broadcast. Can be None on ranks that don't own it.

    Returns:
        The object on all ranks (either the original or the broadcast copy).

    Raises:
        ValueError: If the object doesn't exist on any pipeline parallel rank.
    """
    pp_size = get_pipeline_model_parallel_world_size()
    pp_group = get_pipeline_model_parallel_group()

    if pp_size == 1:
        return obj

    # ------------------------------------------------------------------
    # 1. Gather presence flags from all PP ranks to find the source rank
    # ------------------------------------------------------------------
    has_obj = obj is not None
    obj_flags = [None] * pp_size
    torch.distributed.all_gather_object(obj_flags, has_obj, group=pp_group)

    # ------------------------------------------------------------------
    # 2. Identify the owning rank (the only rank with True flag)
    # ------------------------------------------------------------------
    src_rank = None  # Rank *inside* the PP group
    for rank, flag in enumerate(obj_flags):
        if flag:
            src_rank = rank
            break

    if src_rank is None:
        raise ValueError("Object must exist on at least one PP rank")

    # ------------------------------------------------------------------
    # 3. Broadcast the object from the source rank to all ranks
    # ------------------------------------------------------------------
    # Use broadcast_object_list which is more robust than all_gather_object
    obj_list = [obj]
    pp_ranks = torch.distributed.get_process_group_ranks(pp_group)
    global_src = pp_ranks[src_rank]
    torch.distributed.broadcast_object_list(obj_list, src=global_src, group=pp_group)

    return obj_list[0]


def setup_megatron_model(
    policy_cfg: PolicyConfig,
    cfg: ConfigContainer,
    load_optimizer: bool = True,
    get_embedding_ranks=None,  # TODO @sahilj: What is this?
    get_position_embedding_ranks=None,
):
    state = GlobalState()
    state.cfg = cfg
    # TODO: Freeze state.cfg

    cfg.dist.external_gpu_device_mapping = True
    initialize_megatron(
        cfg=cfg,
        get_embedding_ranks=get_embedding_ranks,
        get_position_embedding_ranks=get_position_embedding_ranks,
    )

    if cfg.ft and cfg.ft.enable_ft_package:
        fault_tolerance.setup(cfg, state)
        fault_tolerance.maybe_setup_simulated_fault(cfg.ft)

    # Set pytorch JIT layer fusion options and warmup JIT functions.
    set_jit_fusion_options(cfg.model, cfg.train.micro_batch_size)

    # Adjust the startup time so it reflects the largest value.
    # This will be closer to what scheduler will see (outside of
    # image ... launches.
    start_time_tensor = torch.tensor(
        [state.start_time], dtype=torch.double, device="cuda"
    )
    torch.distributed.all_reduce(start_time_tensor, op=torch.distributed.ReduceOp.MIN)
    state.start_time = start_time_tensor.item()

    print(
        "time to initialize megatron (seconds): {:.3f}".format(
            time.time() - state.start_time
        )
    )
    torch.distributed.barrier()

    # Context used for persisting some state between checkpoint saves.
    checkpointing_context = init_checkpointing_context(cfg.checkpoint)

    # Tokenizer
    build_tokenizer(
        cfg.tokenizer,
        make_vocab_size_divisible_by=cfg.model.make_vocab_size_divisible_by
        // cfg.model.tensor_model_parallel_size,
        tensor_model_parallel_size=cfg.model.tensor_model_parallel_size,
        trust_remote_code=True,
    )
    assert cfg.model.vocab_size, "vocab size must be specified in model config"

    torch.distributed.barrier()

    pre_wrap_hook = []
    mixed_precision_wrapper = Float16Module
    if policy_cfg["megatron_cfg"]["freeze_moe_router"]:

        def freeze_moe_router(megatron_model):
            if not isinstance(megatron_model, list):
                megatron_model = [megatron_model]
            for model_module in megatron_model:
                # Handle both wrapped (Float16Module) and unwrapped models
                if isinstance(model_module, Float16Module):
                    model_module = model_module.module
                # Handle VLM models
                if hasattr(model_module, "language_model"):
                    model_module = model_module.language_model
                for layer in model_module.decoder.layers:
                    if hasattr(layer, "mlp") and hasattr(layer.mlp, "router"):
                        layer.mlp.router.weight.requires_grad = False

        mixed_precision_wrapper = CustomFloat16Module
        pre_wrap_hook.extend([freeze_moe_router])

    # Model, optimizer, and learning rate.
    model = get_model(
        cfg.model,
        cfg.ddp,
        use_torch_fsdp2=cfg.dist.use_torch_fsdp2,
        overlap_param_gather_with_optimizer_step=cfg.optimizer.overlap_param_gather_with_optimizer_step,
        data_parallel_random_init=cfg.rng.data_parallel_random_init,
        pre_wrap_hook=pre_wrap_hook,
        mixed_precision_wrapper=mixed_precision_wrapper,
    )
    if load_optimizer:
        optimizer, scheduler = setup_optimizer(
            optimizer_config=cfg.optimizer,
            scheduler_config=cfg.scheduler,
            model=model,
            use_gloo_process_groups=cfg.dist.use_gloo_process_groups,
        )
    else:
        optimizer = None
        scheduler = None

    print("Model, optimizer, and learning rate scheduler built")
    torch.distributed.barrier()

    # Load checkpoint if applicable
    if (
        cfg.checkpoint.load is not None
        or cfg.checkpoint.pretrained_checkpoint is not None
    ) and (
        checkpoint_exists(cfg.checkpoint.load)
        or checkpoint_exists(cfg.checkpoint.pretrained_checkpoint)
    ):
        load_checkpoint(
            state,
            model,
            optimizer,
            scheduler,
            checkpointing_context=checkpointing_context,
            skip_load_to_model_and_opt=HAVE_FSDP2 and cfg.dist.use_torch_fsdp2,
        )
        print("Checkpoint loaded")
    torch.distributed.barrier()

    return state, model, optimizer, scheduler, checkpointing_context


def destroy_parallel_state():
    """Safely destroy parallel state and reset async call tracking.

    This function is called during initialization to clean up temporary distributed
    state from model import operations. Resetting async call tracking ensures that
    when the main Megatron distributed context is created, all ranks start with
    consistent call_idx values for async checkpointing.
    """
    if torch.distributed.is_initialized():
        try:
            torch.distributed.barrier()
            torch.distributed.destroy_process_group()
        except:
            pass  # Ignore errors if already destroyed
    if hasattr(parallel_state, "destroy_model_parallel"):
        try:
            parallel_state.destroy_model_parallel()
        except:
            pass  # Ignore errors if already destroyed

    # Reset async calls queue to prevent call_idx mismatches after distributed context recreation
    try:
        import nemo.tron.utils.async_utils as nemo_async_utils
        from megatron.core.dist_checkpointing.strategies.async_utils import (
            AsyncCallsQueue,
        )

        # Clean up any existing async callers first
        old_call_idx = getattr(nemo_async_utils._async_calls_queue, "call_idx", None)
        num_unfinalized = (
            nemo_async_utils._async_calls_queue.get_num_unfinalized_calls()
        )
        if num_unfinalized > 0:
            print(
                f"[WARNING] Resetting async calls queue with {num_unfinalized} unfinalized calls"
            )
        try:
            nemo_async_utils._async_calls_queue.close()
        except:
            pass  # Ignore errors during cleanup
        # Reset the global async calls queue by creating a new instance
        nemo_async_utils._async_calls_queue = AsyncCallsQueue()
        print(f"[DEBUG] Reset NeMo async calls queue (old call_idx: {old_call_idx})")
    except ImportError:
        pass

    # Also reset the Megatron async calls queue if it exists
    try:
        import megatron.training.async_utils as megatron_async_utils
        from megatron.core.dist_checkpointing.strategies.async_utils import (
            AsyncCallsQueue,
        )

        # Clean up any existing async callers first
        old_call_idx = getattr(
            megatron_async_utils._async_calls_queue, "call_idx", None
        )
        num_unfinalized = (
            megatron_async_utils._async_calls_queue.get_num_unfinalized_calls()
        )
        if num_unfinalized > 0:
            print(
                f"[WARNING] Resetting Megatron async calls queue with {num_unfinalized} unfinalized calls"
            )
        try:
            megatron_async_utils._async_calls_queue.close()
        except:
            pass  # Ignore errors during cleanup
        # Reset the Megatron global async calls queue as well
        megatron_async_utils._async_calls_queue = AsyncCallsQueue()
        print(
            f"[DEBUG] Reset Megatron async calls queue (old call_idx: {old_call_idx})"
        )
    except ImportError:
        pass

    # Reset the third global async_calls instance in base strategy module
    try:
        import megatron.core.dist_checkpointing.strategies.base as base_strategy
        from megatron.core.dist_checkpointing.strategies.async_utils import (
            AsyncCallsQueue,
        )

        # Clean up and reset the global async_calls in base strategy
        old_call_idx = getattr(base_strategy.async_calls, "call_idx", None)
        num_unfinalized = base_strategy.async_calls.get_num_unfinalized_calls()
        if num_unfinalized > 0:
            print(
                f"[WARNING] Resetting base strategy async_calls with {num_unfinalized} unfinalized calls"
            )
        try:
            base_strategy.async_calls.close()
        except:
            pass
        base_strategy.async_calls = AsyncCallsQueue()
        print(f"[DEBUG] Reset base strategy async_calls (old call_idx: {old_call_idx})")
    except ImportError:
        pass


@ray.remote(
    runtime_env=get_runtime_env_for_policy_worker("megatron_policy_worker")
)  # pragma: no cover
class MegatronPolicyWorker(AbstractPolicyWorker, ColocatablePolicyInterface):
    def __repr__(self):
        """Customizes the actor's prefix in the Ray logs.

        This makes it easier to identify which worker is producing specific log messages.
        """
        if torch.distributed.is_initialized():
            return f"{self.__class__.__qualname__}[rank={torch.distributed.get_rank()}]"
        else:
            return f"{self.__class__.__qualname__}"

    def __init__(
        self,
        config: PolicyConfig,
        tokenizer: TokenizerType,
        weights_path: Optional[str] = None,
        optimizer_path: Optional[str] = None,
        init_optimizer: bool = True,
        init_reference_model: bool = True,
        *,
        worker_sharding_annotations: NamedSharding,
        pre_init_communication_queue: Queue,
        **kwargs: Any,
    ):
        apply_transformer_engine_patch()

        self.is_generation_colocated = None
        if "generation" in config and config["generation"] is not None:
            self.is_generation_colocated = config["generation"]["colocated"]["enabled"]

        # Explicitly set NCCL_CUMEM_ENABLE to 1 to avoid the P2P initialization error for PyNCCLCommunicator.
        # See https://github.com/NVIDIA-NeMo/RL/issues/564 for more details.
        if not self.is_generation_colocated:
            os.environ["NCCL_CUMEM_ENABLE"] = "1"

        self.cfg = config
        dtype_map = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }
        self.dtype = dtype_map[self.cfg["precision"]]

        self.optimizer_cpu_offload = self.cfg["megatron_cfg"]["optimizer"][
            "optimizer_cpu_offload"
        ]
        self.offload_optimizer_for_logprob = self.cfg["offload_optimizer_for_logprob"]

        # Reward models are not yet supported with Megatron.
        if "reward_model_cfg" in self.cfg and self.cfg["reward_model_cfg"]["enabled"]:
            raise NotImplementedError(
                "Reward models are not yet supported with the Megatron backend, this issue is "
                "tracked in https://github.com/NVIDIA-NeMo/RL/issues/720"
            )

        # Disable dynamo autotune_local_cache to avoid crash when there's already a cache
        # with different order of node_bundles
        configure_dynamo_cache()

        # cfg["model_name"] is allowed to be either an HF model name or a path to an HF checkpoint
        # check if hf_model_name is a path
        hf_model_name = self.cfg["model_name"]
        # Check if the checkpoint already exists
        hf_model_subdir = hf_model_name
        if os.path.exists(hf_model_name):
            hf_model_subdir = f"model_{hf_model_subdir.replace('/', '_')}"

        pretrained_path = f"{get_megatron_checkpoint_dir()}/{hf_model_subdir}"
        pt_checkpoint_exists = os.path.exists(pretrained_path) and os.path.exists(
            os.path.join(pretrained_path, "iter_0000000")
        )

        # Ensure clean slate before import
        destroy_parallel_state()

        # Set for rank for non-collocated to check which ranks to broadcast from
        self.rank = get_rank_safe()
        # Need to initialize the process group before calling into Megatron-Bridge, otherwise Megatron-Bridge will try to set an incorrect device
        torch.distributed.init_process_group("nccl")
        if pt_checkpoint_exists:
            print(f"Checkpoint already exists at {pretrained_path}. Skipping import.")
        else:
            hf_config_overrides = self.cfg.get("hf_config_overrides", {}) or {}
            import_model_from_hf_name(
                hf_model_name,
                pretrained_path,
                self.cfg["megatron_cfg"],
                **hf_config_overrides,
            )

            if parallel_state.model_parallel_is_initialized():
                print("Reinitializing model parallel after loading model state.")
                parallel_state.destroy_model_parallel()

        pretrained_run_config = os.path.join(
            pretrained_path, "iter_0000000/run_config.yaml"
        )

        self.tokenizer = tokenizer
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if not os.path.exists(pretrained_run_config):
            raise FileNotFoundError(
                f"Pretrained run config not found at {pretrained_run_config} on rank={get_rank_safe()}. This usually means that the one-time HF->mcore conversion on rank=0 saved to a directory not being mounted on this node. Please check "
            )

        try:
            cfg_from_pretrained = ConfigContainer.from_yaml(
                pretrained_run_config, mode=InstantiationMode.STRICT
            )
        except Exception as e:
            # Add helpful context as a note to the exception
            e.add_note(
                f"\n{'=' * 80}\n"
                f"NOTE: A common cause of this error is when the HF->mcore converted checkpoint is\n"
                f"created with an older version of megatron-bridge.\n"
                f"If this checkpoint is old or was generated by a different code version,\n"
                f"try deleting it and rerunning the code.\n"
                f"The checkpoint will be automatically regenerated with the current version.\n\n"
                f"Checkpoint location: {pretrained_path}\n"
                f"{'=' * 80}"
            )
            raise
        model_cfg = cfg_from_pretrained.model

        cfg_from_pretrained.logger = LoggerConfig()

        model_cfg.tensor_model_parallel_size = self.cfg["megatron_cfg"][
            "tensor_model_parallel_size"
        ]
        model_cfg.pipeline_model_parallel_size = self.cfg["megatron_cfg"][
            "pipeline_model_parallel_size"
        ]
        model_cfg.num_layers_in_first_pipeline_stage = self.cfg["megatron_cfg"][
            "num_layers_in_first_pipeline_stage"
        ]
        model_cfg.num_layers_in_last_pipeline_stage = self.cfg["megatron_cfg"][
            "num_layers_in_last_pipeline_stage"
        ]
        model_cfg.sequence_parallel = self.cfg["megatron_cfg"]["sequence_parallel"]
        model_cfg.context_parallel_size = self.cfg["megatron_cfg"][
            "context_parallel_size"
        ]
        if model_cfg.context_parallel_size > 1:
            assert self.cfg["sequence_packing"]["enabled"], (
                "Sequence Packing must be enabled to use Context Parallelism with MCore"
            )
        model_cfg.expert_tensor_parallel_size = self.cfg["megatron_cfg"][
            "expert_tensor_parallel_size"
        ]
        model_cfg.expert_model_parallel_size = self.cfg["megatron_cfg"][
            "expert_model_parallel_size"
        ]

        # Setting moe_router_dtype to higher precision (e.g. fp64) can improve numerical stability,
        # especially when using many experts.
        model_cfg.moe_router_dtype = self.cfg["megatron_cfg"]["moe_router_dtype"]

        # The below two configs (and "freeze_moe_router") are used to stabilize moe training
        # by preventing updates to the moe router. We found that this is helpful in reducing
        # logprob error during training.

        # Set this to "none" to disable load balancing loss.
        model_cfg.moe_router_load_balancing_type = self.cfg["megatron_cfg"][
            "moe_router_load_balancing_type"
        ]
        # Set this to 0.0 to disable updates to the moe router expert bias
        model_cfg.moe_router_bias_update_rate = self.cfg["megatron_cfg"][
            "moe_router_bias_update_rate"
        ]

        model_cfg.moe_permute_fusion = self.cfg["megatron_cfg"]["moe_permute_fusion"]
        if "layernorm_epsilon" in self.cfg["megatron_cfg"]:
            model_cfg.layernorm_epsilon = self.cfg["megatron_cfg"]["layernorm_epsilon"]

        model_cfg.sequence_parallel = self.cfg["megatron_cfg"]["sequence_parallel"]
        model_cfg.bf16 = self.dtype == torch.bfloat16
        model_cfg.fp16 = self.dtype == torch.float16
        if model_cfg.fp16:
            assert not model_cfg.bf16, "fp16 and bf16 cannot be used together"
            model_cfg.params_dtype = torch.float16
        elif model_cfg.bf16:
            assert not model_cfg.fp16, "fp16 and bf16 cannot be used together"
            model_cfg.params_dtype = torch.bfloat16
        else:
            model_cfg.params_dtype = torch.float32
        model_cfg.pipeline_dtype = dtype_map[self.cfg["megatron_cfg"]["pipeline_dtype"]]
        model_cfg.parallel_output = True
        if self.cfg["megatron_cfg"]["activation_checkpointing"]:
            model_cfg.recompute_granularity = "full"
            model_cfg.recompute_method = "uniform"
            model_cfg.recompute_num_layers = 1
        if not model_cfg.gated_linear_unit:
            assert model_cfg.activation_func is not None, (
                "activation_func must be set if not using gated_linear_unit. This likely "
                "indicates an issue in configuration conversion (e.g. activation func was "
                "a lambda and couldn't be serialized). This is based on this check "
                "https://github.com/NVIDIA/Megatron-LM/blob/1ab876ddc4c1893c76f26d775226a8d1dcdfb3d2/megatron/core/transformer/mlp.py#L174."
            )
        model_cfg.apply_rope_fusion = self.cfg["megatron_cfg"]["apply_rope_fusion"]
        model_cfg.bias_activation_fusion = self.cfg["megatron_cfg"][
            "bias_activation_fusion"
        ]
        fp8_cfg = self.cfg["megatron_cfg"].get("fp8_cfg", None)
        self.fp8_cfg = fp8_cfg
        if fp8_cfg is not None and fp8_cfg.get("enabled", False):
            try:
                model_cfg.fp8 = fp8_cfg["fp8"]
                model_cfg.fp8_recipe = fp8_cfg["fp8_recipe"]
                model_cfg.fp8_param = fp8_cfg["fp8_param"]
            except KeyError as e:
                raise KeyError(f"Missing key in fp8_cfg: {e}")
            if model_cfg.fp8_param:
                warnings.warn(
                    "Setting fp8_param=True sometimes causes NaN token_mult_prob_error, please use with caution. "
                    "Refer to https://github.com/NVIDIA-NeMo/RL/issues/1164 for latest updates with this issue."
                )

        optimizer_cpu_offload = self.cfg["megatron_cfg"]["optimizer"][
            "optimizer_cpu_offload"
        ]
        optimizer_offload_fraction = self.cfg["megatron_cfg"]["optimizer"][
            "optimizer_offload_fraction"
        ]
        if optimizer_cpu_offload:
            # Currently, hybrid optimizer (partly on GPU and partly on CPU) is not supported because it conflicts with the way
            # Nemo-rl handles the optimizer offload/onload between generation and training. So if using CPU optimizer the offload_fraction should be 1.0.
            assert optimizer_offload_fraction == 1.0, (
                "Currently for optimizer offloading, only optimizer_offload_fraction=1.0 is supported"
            )
        if (
            "logprob_chunk_size" in self.cfg
            and self.cfg["logprob_chunk_size"] is not None
            and self.cfg["logprob_chunk_size"] > 0
        ):
            assert self.cfg["megatron_cfg"]["defer_fp32_logits"], (
                "defer_fp32_logits must be True if logprob_chunk_size is set"
            )
        self.defer_fp32_logits = self.cfg["megatron_cfg"].get(
            "defer_fp32_logits", None
        ) and (model_cfg.fp16 or model_cfg.bf16)

        checkpoint_config = CheckpointConfig(
            save_interval=100,
            save=weights_path,
            load=weights_path,
            pretrained_checkpoint=pretrained_path,  # This is the path to the pretrained ckpt for the SFT case
            async_save=False,  # This doesn't work right now.
            fully_parallel_save=True,
            fully_parallel_load=True,  # Enable fully parallel load
            load_rng=False,
        )
        ref_checkpoint_config = CheckpointConfig(
            pretrained_checkpoint=pretrained_path,  # This is the path to the pretrained ckpt for the SFT case
            save=None,
            load=None,
            fully_parallel_load=True,  # Enable fully parallel load
            load_rng=False,
        )

        assert "train_iters" in self.cfg["megatron_cfg"], (
            "train_iters must be set in megatron_cfg. For an example, see "
            "https://github.com/NVIDIA-NeMo/RL/blob/bccbc377705a81a1f4b3c31ad9767bcc15f735a8/nemo_rl/algorithms/sft.py#L175-L179."
        )

        ## These settings are required for correct gradient computations in mcore
        ## when calculate_per_token_loss is True, there is no scaling of the gradient in mcore,
        ## so we handle the scaling in nemo-rl.
        ## perform_initialization = True is a workaround to ensure the correct tensor parallel attributes are set
        ## on the TP-sharded parameters.
        model_cfg.calculate_per_token_loss = True
        model_cfg.perform_initialization = True
        # SHAN CHECK:
        model_cfg.cuda_graph_impl = "local"
        model_cfg.cuda_graph_scope = None
        model_cfg.use_te_rng_tracker = True

        assert (
            "aux_loss" not in model_cfg.moe_router_load_balancing_type
            or model_cfg.moe_aux_loss_coeff == 0
        ), (
            "MoE aux loss is currently not supported due to a known bug in Megatron-LM. "
            "See https://github.com/NVIDIA/Megatron-LM/issues/1984 for more details."
        )

        self.megatron_cfg = ConfigContainer(
            model=model_cfg,
            checkpoint=checkpoint_config,
            logger=LoggerConfig(logging_level=0),
            train=TrainingConfig(
                micro_batch_size=1,  # ignored
                global_batch_size=self.cfg["train_global_batch_size"],  # ignored
                train_iters=self.cfg["megatron_cfg"][
                    "train_iters"
                ],  # Set by algorithm setup
            ),
            optimizer=OptimizerConfig(
                **self.cfg["megatron_cfg"]["optimizer"],
            ),
            ddp=DistributedDataParallelConfig(
                check_for_nan_in_grad=True,
                grad_reduce_in_fp32=self.cfg["megatron_cfg"][
                    "distributed_data_parallel_config"
                ]["grad_reduce_in_fp32"],
                overlap_grad_reduce=self.cfg["megatron_cfg"][
                    "distributed_data_parallel_config"
                ]["overlap_grad_reduce"],
                overlap_param_gather=self.cfg["megatron_cfg"][
                    "distributed_data_parallel_config"
                ]["overlap_param_gather"],
                # we need to set average_in_collective=False with calculate_per_token_loss=True.
                # otherwise, mcore throws an assertion error.
                average_in_collective=False,
                use_distributed_optimizer=self.cfg["megatron_cfg"]["optimizer"][
                    "use_distributed_optimizer"
                ],
                data_parallel_sharding_strategy=self.cfg["megatron_cfg"][
                    "distributed_data_parallel_config"
                ]["data_parallel_sharding_strategy"],
            ),
            scheduler=SchedulerConfig(
                **self.cfg["megatron_cfg"]["scheduler"],
            ),
            dataset=None,
            tokenizer=TokenizerConfig(
                tokenizer_type="HuggingFaceTokenizer",
                tokenizer_model=hf_model_name,
            ),
        )
        # TODO: this validation should happen inside mbridge: https://github.com/NVIDIA-NeMo/Megatron-Bridge/issues/1665
        if self.dtype == torch.bfloat16:
            assert self.megatron_cfg.model.bf16 == True, (
                "policy.megatron_cfg.model.bf16=True must be set if policy.precision=bfloat16. This is handled by nemo-rl so this indicates something is misconfigured."
            )
            assert (
                self.megatron_cfg.optimizer.use_precision_aware_optimizer == False
                or self.megatron_cfg.optimizer.bf16 == True
            ), (
                "policy.megatron_cfg.optimizer.bf16=True must be set if policy.precision=bfloat16 when using use_precision_aware_optimizer=True"
            )
        elif self.dtype == torch.float16:
            assert self.megatron_cfg.model.fp16 == True, (
                "policy.megatron_cfg.model.fp16=True must be set if policy.precision=float16. This is handled by nemo-rl so this indicates something is misconfigured."
            )
            assert (
                self.megatron_cfg.optimizer.use_precision_aware_optimizer == False
                or self.megatron_cfg.optimizer.fp16 == True
            ), (
                "policy.megatron_cfg.optimizer.fp16=True must be set if policy.precision=float16 when using use_precision_aware_optimizer=True"
            )
        elif self.dtype == torch.float32:
            assert (
                self.megatron_cfg.model.bf16 == False
                and self.megatron_cfg.model.fp16 == False
            ), (
                "policy.megatron_cfg.model.bf16=False and policy.megatron_cfg.model.fp16=False must be set if policy.precision=float32. This is handled by nemo-rl so this indicates something is misconfigured."
            )
            assert (
                self.megatron_cfg.optimizer.bf16 == False
                and self.megatron_cfg.optimizer.fp16 == False
            ), (
                "policy.megatron_cfg.optimizer.bf16=False and policy.megatron_cfg.optimizer.fp16=False must be set if policy.precision=float32"
            )
        self.megatron_cfg.validate()
        (
            self.mcore_state,
            self.model,
            self.optimizer,
            self.scheduler,
            self.checkpointing_context,
        ) = setup_megatron_model(
            policy_cfg=self.cfg, cfg=self.megatron_cfg, load_optimizer=init_optimizer
        )

        # Set the param sync function for the model
        if (
            self.megatron_cfg.ddp.overlap_param_gather
            and self.megatron_cfg.ddp.align_param_gather
        ):
            self.megatron_cfg.param_sync_func = [
                model_chunk.start_param_sync for model_chunk in self.model
            ]
            if len(self.model) == 1:
                self.megatron_cfg.param_sync_func = self.megatron_cfg.param_sync_func[0]

        self.model = self.model[0]  # Get the first model from the list

        if init_reference_model:
            self.model = self.move_model(self.model, "cpu")
            ref_ckpt_context = init_checkpointing_context(ref_checkpoint_config)

            # Create a separate megatron config for the reference model with the correct checkpoint config
            self.megatron_cfg.model.cuda_graph_impl = "none"
            self.megatron_cfg.model.use_te_rng_tracker = True
            ref_megatron_cfg = ConfigContainer(
                model=self.megatron_cfg.model,
                checkpoint=ref_checkpoint_config,  # Use the reference checkpoint config
                logger=self.megatron_cfg.logger,
                train=self.megatron_cfg.train,
                optimizer=self.megatron_cfg.optimizer,
                ddp=self.megatron_cfg.ddp,
                scheduler=self.megatron_cfg.scheduler,
                dataset=self.megatron_cfg.dataset,
                tokenizer=self.megatron_cfg.tokenizer,
            )

            # Create a separate state object for the reference model
            ref_state = GlobalState()
            ref_state.cfg = ref_megatron_cfg

            # Configure mixed precision wrapper for reference model
            ref_mixed_precision_wrapper = Float16Module
            if self.cfg["megatron_cfg"].get("freeze_moe_router", False):
                ref_mixed_precision_wrapper = CustomFloat16Module

            reference_model = get_model(
                self.megatron_cfg.model,
                self.megatron_cfg.ddp,
                use_torch_fsdp2=self.megatron_cfg.dist.use_torch_fsdp2,
                overlap_param_gather_with_optimizer_step=self.megatron_cfg.optimizer.overlap_param_gather_with_optimizer_step,
                pre_wrap_hook=self.megatron_cfg.rng.data_parallel_random_init,
                mixed_precision_wrapper=ref_mixed_precision_wrapper,
            )
            print("Loading the Reference Model")
            if (
                ref_checkpoint_config.pretrained_checkpoint is not None
                and checkpoint_exists(ref_checkpoint_config.pretrained_checkpoint)
            ):
                load_checkpoint(
                    ref_state,  # Use the separate state object with ref checkpoint config
                    reference_model,
                    None,  # no optimizer
                    None,  # no scheduler
                    checkpointing_context=ref_ckpt_context,
                    skip_load_to_model_and_opt=HAVE_FSDP2
                    and self.megatron_cfg.dist.use_torch_fsdp2,
                )
                reference_model = reference_model[0]
                reference_model.eval()
                self.reference_state_dict = {}
                for name, item in reference_model.state_dict().items():
                    if isinstance(item, torch.Tensor):
                        cpu_item = item.detach().to(
                            device="cpu", non_blocking=True, copy=True
                        )
                        del item
                    else:
                        cpu_item = item
                    self.reference_state_dict[name] = cpu_item
                print("Reference model loaded")
            else:
                print("Reference model not loaded")

            self.model = self.move_model(self.model, "cuda")

        _update_model_config_funcs(
            [self.model],
            self.megatron_cfg.model,
            self.megatron_cfg.ddp,
            self.optimizer,
            align_grad_reduce=self.megatron_cfg.dist.align_grad_reduce,
        )

        tokenizer_config = TokenizerConfig(
            tokenizer_type="HuggingFaceTokenizer",
            tokenizer_model=hf_model_name,
        )

        self.megatron_tokenizer = build_tokenizer(
            tokenizer_config,
            make_vocab_size_divisible_by=self.megatron_cfg.model.make_vocab_size_divisible_by
            // self.cfg["megatron_cfg"]["tensor_model_parallel_size"],
            tensor_model_parallel_size=self.cfg["megatron_cfg"][
                "tensor_model_parallel_size"
            ],
            trust_remote_code=True,
        )
        self.final_padded_vocab_size = calculate_padded_vocab_size(
            self.megatron_cfg.model.vocab_size,
            self.megatron_cfg.model.make_vocab_size_divisible_by,
            self.cfg["megatron_cfg"]["tensor_model_parallel_size"],
        )
        self.dp_size = worker_sharding_annotations.get_axis_size("data_parallel")
        self.megatron_bridge = AutoBridge.from_hf_pretrained(
            hf_model_name, trust_remote_code=True
        )

        self.should_disable_forward_pre_hook = (
            self.cfg["megatron_cfg"]["optimizer"]["use_distributed_optimizer"]
            and self.cfg["megatron_cfg"]["distributed_data_parallel_config"][
                "overlap_param_gather"
            ]
        )

        # vars used for refit
        ## will be initialized in prepare_refit_info
        # refit_param_info_mcore combines the conversion tasks with the param memory
        # [(mcore_param_name, estimated_memory), ...]
        # Note: here param name is local param name, with local layer number and
        # local expert id etc.
        self.refit_conversion_tasks = (
            None  # Meta data for conversion params from megatron bridge
        )
        self.refit_conversion_tasks_current_index = None
        self.refit_param_info_mcore = None

        ## used for streaming update inference engine weights
        self._held_gather_buffer = None

    def enable_forward_pre_hook(self):
        assert isinstance(self.model, DistributedDataParallel)
        self.model.enable_forward_pre_hook()

    def disable_forward_pre_hook(self, param_sync=True):
        assert isinstance(self.model, DistributedDataParallel)
        self.model.disable_forward_pre_hook(param_sync=param_sync)

    @wrap_with_nvtx_name("megatron_policy_worker/train")
    def train(
        self,
        data: BatchedDataDict,
        loss_fn: LossFunction,
        eval_mode: bool = False,
        gbs: Optional[int] = None,
        mbs: Optional[int] = None,
    ) -> dict[str, Any]:
        """Train the policy on a batch of data with a given loss function."""
        # Disable CUDA graphs for training to avoid conflicts with inference tensors.
        # This prevents "Inplace update to inference tensor outside InferenceMode" errors
        # when switching from generation (which uses CUDA graphs) to training.
        # Use toggle_cuda_graphs on the underlying module (not the DDP wrapper) to properly
        # reset CUDA graph state, following the pattern from megatron/rl/rl_utils.py.
        lang_module = (
            self.model.module.module
            if hasattr(self.model.module, "module")
            else self.model.module
        )

        toggle_cuda_graphs(lang_module, set_to="none", reset_cuda_graphs=True)

        self.model.zero_grad_buffer()
        if hasattr(self.model, "inference_params"):
            self.model.inference_params = None

        # Reset any cached attention states
        for module in self.model.modules():
            if hasattr(module, "reset_inference_cache"):
                module.reset_inference_cache()
            if hasattr(module, "_inference_key_value_memory"):
                module._inference_key_value_memory = None

        # Aggressive memory cleanup before training to avoid fragmentation OOM
        # This is especially important after CUDA graph operations
        gc.collect()
        torch.cuda.empty_cache()

        if gbs is None:
            gbs = self.cfg["train_global_batch_size"]
        if mbs is None:
            mbs = self.cfg["train_micro_batch_size"]
        local_gbs = gbs // self.dp_size
        total_dataset_size = torch.tensor(data.size, device="cuda")
        torch.distributed.all_reduce(
            total_dataset_size,
            op=torch.distributed.ReduceOp.SUM,
            group=parallel_state.get_data_parallel_group(),
        )
        num_global_batches = int(total_dataset_size.item()) // gbs

        if eval_mode:
            ctx: AbstractContextManager[Any] = torch.no_grad()
            self.model.eval()
        else:
            ctx = nullcontext()
            # Ensure model is in training mode
            self.model.train()

        with ctx:
            # dim 1 is always assumed to be the sequence dim, sanity check this here
            sequence_dim = 1
            seq_dim_size = data["input_ids"].shape[sequence_dim]
            for k, v in data.items():
                if torch.is_tensor(v) and len(v.shape) > 1:
                    assert v.shape[sequence_dim] == seq_dim_size, (
                        f"Dim 1 must be the sequence dim, expected dim 1={seq_dim_size} but got shape {v.shape}"
                    )

            forward_step = partial(
                forward_step_arbitrary_loss, loss_fn=loss_fn, policy_cfg=self.cfg
            )
            all_mb_metrics = []
            losses = []
            total_num_microbatches = 0
            for gb_idx in range(num_global_batches):
                global_batch = data.get_batch(batch_idx=gb_idx, batch_size=local_gbs)

                assert "sample_mask" in global_batch, (
                    "sample_mask must be present in the data!"
                )
                ## get the normalization factor for the loss
                local_valid_seqs = torch.sum(global_batch["sample_mask"])

                if not "token_mask" in global_batch:
                    local_valid_toks = (
                        local_valid_seqs * global_batch["input_ids"].shape[1]
                    )
                else:
                    local_valid_toks = torch.sum(
                        global_batch["token_mask"][:, 1:]
                        * global_batch["sample_mask"].unsqueeze(-1)
                    )

                to_reduce = torch.tensor([local_valid_seqs, local_valid_toks]).cuda()
                torch.distributed.all_reduce(
                    to_reduce, group=parallel_state.get_data_parallel_group()
                )
                global_valid_seqs, global_valid_toks = to_reduce[0], to_reduce[1]

                if (
                    hasattr(loss_fn, "loss_type")
                    and loss_fn.loss_type == LossType.TOKEN_LEVEL
                ):
                    assert "token_mask" in global_batch, (
                        "token_mask must be present in the data when using token-level loss"
                    )

                batch = data.get_batch(batch_idx=gb_idx, batch_size=local_gbs)
                pack_seqs = False
                seqlen_key = None
                pad_factor = 1
                pad_full_seq_to = None
                pad_packed_seq_to_multiple_of = 1
                if self.cfg["dynamic_batching"]["enabled"]:
                    data_iterator = batch.make_microbatch_iterator_with_dynamic_shapes()
                    data_iterator_len = (
                        batch.get_microbatch_iterator_dynamic_shapes_len()
                    )
                elif self.cfg["sequence_packing"]["enabled"]:
                    data_iterator = (
                        batch.make_microbatch_iterator_for_packable_sequences()
                    )
                    data_iterator_len, seq_dim_size = (
                        batch.get_microbatch_iterator_for_packable_sequences_len()
                    )
                    mbs = 1
                    pack_seqs = True
                    seqlen_key = "input_lengths"
                    (
                        pad_factor,
                        pad_packed_seq_to_multiple_of,
                        pad_full_seq_to,
                    ) = _get_pack_sequence_parameters_for_megatron(
                        self.cfg["megatron_cfg"],
                        seq_dim_size,
                    )
                    # if pad_full_seq_to is not None, we need to use it as the sequence length
                    seq_dim_size = pad_full_seq_to or seq_dim_size
                else:
                    data_iterator = batch.make_microbatch_iterator(mbs)
                    data_iterator_len = local_gbs // mbs

                # Track total microbatches for MoE aux-loss averaging
                total_num_microbatches += int(data_iterator_len)

                rerun_state_machine = get_rerun_state_machine()
                while rerun_state_machine.should_run_forward_backward(data_iterator):
                    # Set grad to zero.
                    self.model.zero_grad_buffer()
                    self.optimizer.zero_grad()

                    # Forward pass.
                    forward_backward_func = get_forward_backward_func()
                    losses_reduced = forward_backward_func(
                        forward_step_func=partial(
                            forward_step,
                            self.mcore_state,
                            global_valid_seqs,
                            global_valid_toks,
                            pack_sequences=pack_seqs,
                            seq_length_key=seqlen_key,
                            pad_individual_seqs_to_multiple_of=pad_factor,
                            pad_packed_seq_to_multiple_of=pad_packed_seq_to_multiple_of,
                            pad_full_seq_to=pad_full_seq_to,
                            defer_fp32_logits=self.defer_fp32_logits,
                        ),
                        data_iterator=data_iterator,
                        model=self.model,
                        num_microbatches=data_iterator_len,
                        seq_length=seq_dim_size,
                        micro_batch_size=mbs,
                        decoder_seq_length=seq_dim_size,
                        forward_only=eval_mode,
                        do_not_average_loss=True,
                    )

                # Empty unused memory.
                if self.cfg["megatron_cfg"]["empty_unused_memory_level"] >= 1:
                    torch.cuda.empty_cache()

                # Update parameters.
                if not eval_mode:
                    update_successful, grad_norm, num_zeros_in_grad = (
                        self.optimizer.step()
                    )
                else:
                    update_successful, grad_norm, num_zeros_in_grad = (True, 0.0, 0.0)

                # when freezing sub-models we may have a mixture of successful and unsucessful ranks,
                # so we must gather across mp ranks
                update_successful = logical_and_across_model_parallel_group(
                    update_successful
                )
                # grad_norm and num_zeros_in_grad will be None on ranks without trainable params,
                # so we must gather across mp ranks
                grad_norm: float = reduce_max_stat_across_model_parallel_group(
                    grad_norm
                )
                num_zeros_in_grad: float = reduce_max_stat_across_model_parallel_group(
                    num_zeros_in_grad
                )

                if update_successful:
                    skipped_iter = 0
                else:
                    skipped_iter = 1

                # Empty unused memory.
                if self.cfg["megatron_cfg"]["empty_unused_memory_level"] >= 2:
                    torch.cuda.empty_cache()

                if parallel_state.is_pipeline_last_stage(ignore_virtual=True):
                    # keep all microbatch metrics to be normalized later
                    gb_loss_metrics = []
                    mb_losses = []
                    for x in losses_reduced:
                        loss_metrics = {}
                        for k in x.keys():
                            loss_metrics[k] = x[k] / num_global_batches
                        gb_loss_metrics.append(loss_metrics)
                        curr_lr = self.scheduler.get_lr(self.optimizer.param_groups[0])
                        curr_wd = self.scheduler.get_wd()
                        loss_metrics["lr"] = curr_lr
                        loss_metrics["wd"] = curr_wd
                        loss_metrics["global_valid_seqs"] = global_valid_seqs.item()
                        loss_metrics["global_valid_toks"] = global_valid_toks.item()
                        mb_losses.append(loss_metrics["loss"])

                    torch.distributed.broadcast_object_list(
                        [gb_loss_metrics],
                        src=get_pipeline_model_parallel_last_rank(),
                        group=get_pipeline_model_parallel_group(),
                    )
                else:
                    loss_metrics = [None]  # type: ignore
                    torch.distributed.broadcast_object_list(
                        loss_metrics,
                        src=get_pipeline_model_parallel_last_rank(),
                        group=get_pipeline_model_parallel_group(),
                    )
                    gb_loss_metrics = loss_metrics[0]
                    mb_losses = [x["loss"] for x in gb_loss_metrics]

                all_mb_metrics.extend(gb_loss_metrics)
                losses.append(torch.tensor(mb_losses).sum().item())

        if not eval_mode:
            # take one LR step every rollout batch
            # we need to scale the step by gbs to counteract the fact that NeMo automatically
            # scales lr_warmup_steps by gbs during init
            self.scheduler.step(increment=gbs)

        # Aggregate metrics across all microbatches
        mb_metrics = defaultdict(list)
        for m in all_mb_metrics:
            for k, v in m.items():
                mb_metrics[k].append(v)

        with torch.no_grad():
            global_loss = torch.tensor(losses, device="cuda")
            torch.distributed.all_reduce(
                global_loss,
                op=torch.distributed.ReduceOp.SUM,
                group=parallel_state.get_data_parallel_group(),
            )

        metrics = {
            "global_loss": global_loss.cpu(),
            "rank": torch.distributed.get_rank(),
            "gpu_name": torch.cuda.get_device_name(),
            "model_dtype": self.dtype,
            "all_mb_metrics": dict(mb_metrics),
            "grad_norm": torch.tensor([grad_norm]),
        }
        # Collect MoE aux metrics averaged across microbatches
        num_moe_experts = getattr(self.model.config, "num_moe_experts", None)
        if num_moe_experts is not None and num_moe_experts > 1:
            moe_loss_scale = 1.0 / max(1, total_num_microbatches)
            moe_metrics = get_moe_metrics(
                loss_scale=moe_loss_scale,
                per_layer_logging=self.cfg["megatron_cfg"]["moe_per_layer_logging"],
            )
            if moe_metrics:
                metrics["moe_metrics"] = moe_metrics
        return metrics

    @wrap_with_nvtx_name("megatron_policy_worker/get_logprobs")
    def get_logprobs(
        self, *, data: BatchedDataDict[Any], micro_batch_size: Optional[int] = None
    ) -> BatchedDataDict[LogprobOutputSpec]:
        """Get the logprobs of the model for a batch of data.

        Uses the configured logprob_batch_size to do microbatching.
        Input data is assumed to be right-padded. The method internally converts to
        left-padded format for computation, and returns outputs in right-padded format.
        If micro_batch_size is provided, it will be used instead of the configured
        logprob_batch_size.

        Returns:
          a BatchedDataDict with key "logprobs" and shape [batch_size, sequence_length].
          We use the convention that the logprob of the first token is 0 so that the sequence length is maintained.
          The logprob of input token i is specified at position i in the output logprobs tensor.
        """
        no_grad = torch.no_grad()
        no_grad.__enter__()
        logprob_batch_size = (
            micro_batch_size
            if micro_batch_size is not None
            else self.cfg["logprob_batch_size"]
        )

        # dim 1 is always assumed to be the sequence dim, sanity check this here
        sequence_dim = 1
        input_seq_dim_size = data["input_ids"].shape[sequence_dim]
        for k, v in data.items():
            if torch.is_tensor(v) and len(v.shape) > 1:
                assert v.shape[sequence_dim] == input_seq_dim_size, (
                    f"Dim 1 must be the sequence dim, expected dim 1={input_seq_dim_size} but got shape {v.shape}"
                )

        self.model.eval()

        pp_seq_dim_size = input_seq_dim_size
        pp_grp = get_pipeline_model_parallel_group()
        pad_factor = 1
        pad_packed_seq_to_multiple_of = 1
        pad_full_seq_to = None
        if self.cfg["sequence_packing"]["enabled"]:
            _, seq_dim_size = data.get_microbatch_iterator_for_packable_sequences_len()
            (
                pad_factor,
                pad_packed_seq_to_multiple_of,
                pad_full_seq_to,
            ) = _get_pack_sequence_parameters_for_megatron(
                self.cfg["megatron_cfg"],
                seq_dim_size,
            )
            pp_seq_dim_size = pad_full_seq_to or pp_seq_dim_size

        def forward_step_fn(
            data_iterator: Iterator[BatchedDataDict[Any]], model: GPTModel
        ):
            nonlocal pad_full_seq_to, pad_packed_seq_to_multiple_of, pad_factor
            data_dict = next(data_iterator).to("cuda")
            if self.cfg["sequence_packing"]["enabled"]:
                original_seq_length = data_dict["input_ids"].shape[1]
                cp_size = self.cfg["megatron_cfg"]["context_parallel_size"]
                cp_rank = get_context_parallel_rank()
                (
                    input_ids,
                    input_ids_cp_sharded,
                    packed_seq_params,
                    cu_seqlens,
                    cu_seqlens_padded,
                ) = _pack_sequences_for_megatron(
                    data_dict["input_ids"].clone(),
                    data_dict["input_lengths"],
                    pad_individual_seqs_to_multiple_of=pad_factor,
                    pad_packed_seq_to_multiple_of=pad_packed_seq_to_multiple_of,
                    pad_packed_seq_to=pad_full_seq_to,
                    cp_rank=cp_rank,
                    cp_size=cp_size,
                )
                attention_mask, position_ids = None, None
                unpacked_input_ids = data_dict["input_ids"]
            else:
                input_ids = data_dict["input_ids"]
                input_ids_cp_sharded = input_ids
                attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
                    data=input_ids,
                    eod_token=0,  # used for loss_mask, which we don't use
                    pad_token=0,  # used for loss_mask, which we don't use
                    reset_position_ids=False,
                    reset_attention_mask=False,
                    eod_mask_loss=False,
                    pad_mask_loss=False,
                )
                packed_seq_params = None
                unpacked_input_ids = input_ids

            multimodal_data = data_dict.get_multimodal_dict(
                as_tensors=True, device=input_ids.device
            )
            if len(multimodal_data) > 0:
                position_ids = None

            additional_kwargs = {}
            # Mamba models currently do not support packed_seq_params
            if packed_seq_params is not None:
                additional_kwargs["packed_seq_params"] = packed_seq_params

            if self.defer_fp32_logits:
                additional_kwargs["fp32_output"] = False

            output_tensor = model(
                input_ids=input_ids_cp_sharded,
                position_ids=position_ids,
                attention_mask=attention_mask,
                **multimodal_data,
                **additional_kwargs,
            )

            # Apply temperature scaling to logits for training
            # This matches the dtensor worker's _apply_temperature_scaling in the train method
            if "generation" in self.cfg and self.cfg["generation"] is not None:
                output_tensor.div_(self.cfg["generation"]["temperature"])

            def collection_fn(output_tensor):
                stc = time.time()
                tp_grp = get_tensor_model_parallel_group()
                tp_rank = get_tensor_model_parallel_rank()
                logprob_chunk_size = self.cfg.get("logprob_chunk_size", None)
                if self.cfg["sequence_packing"]["enabled"]:
                    token_logprobs = from_parallel_logits_to_logprobs_packed_sequences(
                        output_tensor,
                        target=input_ids,
                        cu_seqlens_padded=cu_seqlens_padded,
                        unpacked_seqlen=original_seq_length,
                        vocab_start_index=tp_rank * output_tensor.shape[-1],
                        vocab_end_index=(tp_rank + 1) * output_tensor.shape[-1],
                        group=tp_grp,
                        inference_only=True,
                        cp_group=get_context_parallel_group(),
                        chunk_size=logprob_chunk_size,
                    )
                else:
                    token_logprobs = from_parallel_logits_to_logprobs(
                        output_tensor,
                        target=unpacked_input_ids,
                        vocab_start_index=tp_rank * output_tensor.shape[-1],
                        vocab_end_index=(tp_rank + 1) * output_tensor.shape[-1],
                        tp_group=tp_grp,
                        inference_only=True,
                        chunk_size=logprob_chunk_size,
                    )

                # Prepend 0 logprob for first token to maintain same sequence length as input
                token_logprobs = torch.cat(
                    [torch.zeros_like(token_logprobs[:, :1]), token_logprobs], dim=1
                )
                return torch.tensor(0.0, device=token_logprobs.device), {
                    "logprobs": token_logprobs
                }

            return output_tensor, collection_fn

        if self.cfg["dynamic_batching"]["enabled"]:
            mb_iterator = data.make_microbatch_iterator_with_dynamic_shapes()
            data_iterator_len = data.get_microbatch_iterator_dynamic_shapes_len()
            micro_batch_size = logprob_batch_size
        elif self.cfg["sequence_packing"]["enabled"]:
            mb_iterator = data.make_microbatch_iterator_for_packable_sequences()
            data_iterator_len, _ = (
                data.get_microbatch_iterator_for_packable_sequences_len()
            )
            micro_batch_size = 1
        else:
            mb_iterator = data.make_microbatch_iterator(logprob_batch_size)
            data_iterator_len = max(1, data.size // logprob_batch_size)
            micro_batch_size = logprob_batch_size

        forward_backward_func = get_forward_backward_func()
        list_of_logprobs = forward_backward_func(
            forward_step_func=forward_step_fn,
            data_iterator=mb_iterator,
            model=self.model,
            num_microbatches=data_iterator_len,
            seq_length=pp_seq_dim_size,
            micro_batch_size=micro_batch_size,
            decoder_seq_length=pp_seq_dim_size,
            forward_only=True,
        )
        if is_pipeline_last_stage(ignore_virtual=True):
            all_log_probs_padded = []
            all_logprobs = [l["logprobs"] for l in list_of_logprobs]
            for lp in all_logprobs:
                padding_needed = input_seq_dim_size - lp.shape[1]
                if padding_needed > 0:
                    lp = torch.nn.functional.pad(
                        lp, (0, padding_needed), mode="constant", value=0.0
                    )
                all_log_probs_padded.append(lp)

            logprobs = torch.cat(all_log_probs_padded, dim=0)
            # broadcast logprobs to first pp rank
            broadcast_tensor(logprobs, torch.distributed.get_rank(), pp_grp)
        else:
            logprobs = broadcast_tensor(
                None, get_pipeline_model_parallel_last_rank(), pp_grp
            )

        no_grad.__exit__(None, None, None)
        return BatchedDataDict[LogprobOutputSpec](logprobs=logprobs).to("cpu")

    @contextmanager
    def use_reference_model(self):
        """Context manager that temporarily swaps the reference model and active model.

        On entry: Moves model to CPU, moves reference_model to CUDA. Swaps the references
        On exit: Restores original references and re-flips cuda/cpu
        """
        ## disable overlap param gather when swapping weights
        if self.should_disable_forward_pre_hook:
            self.disable_forward_pre_hook()

        with torch.no_grad():
            try:
                # Save original references
                model_state_dict = {}
                for name, item in self.model.state_dict().items():
                    if isinstance(item, torch.Tensor):
                        item = item.detach().to(
                            device="cpu", non_blocking=True, copy=True
                        )
                    model_state_dict[name] = item

                self.model.load_state_dict(self.reference_state_dict, strict=True)
                # for name, item in self.reference_state_dict.items():
                # if isinstance(item, torch.Tensor):
                # self.model.state_dict()[name] = item.detach().to(device="cuda", non_blocking=True, copy=True)

                if self.cfg["megatron_cfg"]["empty_unused_memory_level"] >= 1:
                    gc.collect()
                    torch.cuda.empty_cache()

                # - self.model is the original reference_model, now on CUDA
                # - self.reference_model is the original model, now on CPU
                yield

            finally:
                # Restore original references and device placement
                self.model.load_state_dict(model_state_dict, strict=True)
                # for name, item in model_state_dict.items():
                # if isinstance(item, torch.Tensor):
                # item = item.detach().to(device="cuda", non_blocking=True, copy=True)
                # self.model.state_dict()[name] = item

                if self.cfg["megatron_cfg"]["empty_unused_memory_level"] >= 1:
                    gc.collect()
                    torch.cuda.empty_cache()

                ## re-enable overlap param gather after weight swap
                if self.should_disable_forward_pre_hook:
                    self.enable_forward_pre_hook()

    @wrap_with_nvtx_name("megatron_policy_worker/get_topk_logits")
    def get_topk_logits(
        self,
        *,
        data: BatchedDataDict[GenerationDatumSpec],
        k: int,
        micro_batch_size: Optional[int] = None,
    ):
        """Get the top-k logits and indices for a batch of data.

        The major difference from get_logprobs is that we compute top-k logits and indices for each position in the sequence.

        Returns:
            BatchedDataDict containing:
                - topk_logits: Tensor of top-k logits for each position in the sequence
                - topk_indices: Tensor of top-k indices for each position in the sequence
        """
        no_grad = torch.no_grad()
        no_grad.__enter__()

        logprob_batch_size = (
            micro_batch_size
            if micro_batch_size is not None
            else self.cfg["logprob_batch_size"]
        )

        sequence_dim = 1
        input_seq_dim_size = data["input_ids"].shape[sequence_dim]
        # Avoid shadowing the function argument `k` by using a distinct variable name
        for tensor_name, v in data.items():
            if torch.is_tensor(v) and len(v.shape) > 1:
                assert v.shape[sequence_dim] == input_seq_dim_size, (
                    f"Tensor {tensor_name} must have sequence dimension {sequence_dim} of size {input_seq_dim_size}, but got shape {v.shape}"
                )

        self.model.eval()

        pp_seq_dim_size = input_seq_dim_size
        pp_grp = get_pipeline_model_parallel_group()

        pad_factor = 1
        pad_packed_seq_to_multiple_of = 1
        pad_full_seq_to = None
        if self.cfg["sequence_packing"]["enabled"]:
            _, seq_dim_size = data.get_microbatch_iterator_for_packable_sequences_len()
            (
                pad_factor,
                pad_packed_seq_to_multiple_of,
                pad_full_seq_to,
            ) = _get_pack_sequence_parameters_for_megatron(
                self.cfg["megatron_cfg"],
                seq_dim_size,
            )
            pp_seq_dim_size = pad_full_seq_to or pp_seq_dim_size

        def forward_step_fn(
            data_iterator: Iterator[BatchedDataDict[Any]], model: GPTModel
        ):
            nonlocal pad_full_seq_to, pad_packed_seq_to_multiple_of, pad_factor
            data_dict = next(data_iterator).to("cuda")

            pack = self.cfg["sequence_packing"]["enabled"]
            if pack:
                original_seq_length = data_dict["input_ids"].shape[1]
                cp_size = self.cfg["megatron_cfg"]["context_parallel_size"]
                cp_rank = get_context_parallel_rank()

                (
                    input_ids_unpacked,
                    input_ids_cp_sharded,
                    packed_seq_params,
                    cu_seqlens,
                    cu_seqlens_padded,
                ) = _pack_sequences_for_megatron(
                    data_dict["input_ids"].clone(),
                    data_dict["input_lengths"],
                    pad_individual_seqs_to_multiple_of=pad_factor,
                    pad_packed_seq_to_multiple_of=pad_packed_seq_to_multiple_of,
                    pad_packed_seq_to=pad_full_seq_to,
                    cp_rank=cp_rank,
                    cp_size=cp_size,
                )
                attention_mask, position_ids = None, None
                seq_lengths = data_dict["input_lengths"]
                unpacked_seqlen = original_seq_length
            else:
                input_ids_cp_sharded = data_dict["input_ids"]
                attention_mask, _, position_ids = get_ltor_masks_and_position_ids(
                    data=input_ids_cp_sharded,
                    eod_token=0,
                    pad_token=0,
                    reset_position_ids=False,
                    reset_attention_mask=False,
                    eod_mask_loss=False,
                    pad_mask_loss=False,
                )
                packed_seq_params = None

            multimodal_data = data_dict.get_multimodal_dict(
                as_tensors=True, device=input_ids_cp_sharded.device
            )
            if len(multimodal_data) > 0:
                position_ids = None

            additional_kwargs = {}
            if packed_seq_params is not None:
                additional_kwargs["packed_seq_params"] = packed_seq_params

            output_tensor = model(
                input_ids=input_ids_cp_sharded,
                position_ids=position_ids,
                attention_mask=attention_mask,
                **additional_kwargs,
                **multimodal_data,
            )

            if "generation" in self.cfg and self.cfg["generation"] is not None:
                output_tensor.div_(self.cfg["generation"]["temperature"])

            def collection_fn(_):
                # Only the last PP stage produces final logits/top-k; earlier stages return empty
                # if not is_pipeline_last_stage(ignore_virtual=True):
                # return output_tensor.new_zeros(()), {}

                tp_grp = get_tensor_model_parallel_group()
                tp_rank = get_tensor_model_parallel_rank()
                vocab_shard_size = output_tensor.shape[-1]
                vocab_start_index = tp_rank * vocab_shard_size

                chunk_size = None
                if "logprob_chunk_size" in self.cfg:
                    chunk_size = self.cfg["logprob_chunk_size"]

                topk_vals_local, topk_idx_local = distributed_vocab_topk(
                    output_tensor,
                    k,
                    tp_grp,
                    vocab_start_index=vocab_start_index,
                    vocab_end_index=vocab_start_index + vocab_shard_size,
                    chunk_size=chunk_size,
                )

                if self.cfg["megatron_cfg"]["context_parallel_size"] > 1:
                    cp_grp = get_context_parallel_group()
                    if pack:
                        # Per-sequence CP allgather following packed-sequence logic
                        batch_size = data_dict["input_ids"].shape[0]
                        total_packed_len = int(cu_seqlens_padded[-1].item())

                        topk_vals_full = torch.zeros(
                            (1, total_packed_len, k),
                            dtype=topk_vals_local.dtype,
                            device=topk_vals_local.device,
                        )
                        topk_idx_full = torch.zeros(
                            (1, total_packed_len, k),
                            dtype=topk_idx_local.dtype,
                            device=topk_idx_local.device,
                        )

                        for i in range(batch_size):
                            start_idx = int(cu_seqlens_padded[i].item())
                            end_idx = int(cu_seqlens_padded[i + 1].item())
                            if end_idx > start_idx:
                                local_vals_slice = topk_vals_local[
                                    :, start_idx // cp_size : end_idx // cp_size, :
                                ]
                                local_idx_slice = topk_idx_local[
                                    :, start_idx // cp_size : end_idx // cp_size, :
                                ]
                                gathered_vals = allgather_cp_sharded_tensor(
                                    local_vals_slice, cp_grp, seq_dim=1
                                )
                                gathered_idx = allgather_cp_sharded_tensor(
                                    local_idx_slice, cp_grp, seq_dim=1
                                )
                                # Some kernels may return [X, Y, k] where X*Y = (end_idx - start_idx).
                                # Flatten leading dims and reshape to [1, expected_len, k] to match target.
                                expected_len = end_idx - start_idx
                                if (
                                    gathered_vals.dim() == 3
                                    and gathered_vals.shape[1] != expected_len
                                ):
                                    gathered_vals = gathered_vals.reshape(
                                        1, expected_len, gathered_vals.shape[-1]
                                    )
                                if (
                                    gathered_idx.dim() == 3
                                    and gathered_idx.shape[1] != expected_len
                                ):
                                    gathered_idx = gathered_idx.reshape(
                                        1, expected_len, gathered_idx.shape[-1]
                                    )
                                topk_vals_full[:, start_idx:end_idx, :] = gathered_vals
                                topk_idx_full[:, start_idx:end_idx, :] = gathered_idx
                    else:
                        # Sequence packing must be enabled when CP > 1
                        raise RuntimeError(
                            "Context Parallelism (CP>1) requires sequence packing to be enabled."
                        )
                else:
                    topk_vals_full = topk_vals_local
                    topk_idx_full = topk_idx_local

                if pack:
                    batch_size = data_dict["input_ids"].shape[0]
                    out_vals = torch.zeros(
                        (batch_size, unpacked_seqlen, k),
                        dtype=topk_vals_full.dtype,
                        device=topk_vals_full.device,
                    )
                    out_idx = torch.zeros(
                        (batch_size, unpacked_seqlen, k),
                        dtype=topk_idx_full.dtype,
                        device=topk_idx_full.device,
                    )
                    for i in range(batch_size):
                        seq_len = int(seq_lengths[i].item())
                        start_idx = int(cu_seqlens_padded[i].item())
                        if seq_len > 0:
                            out_vals[i, :seq_len, :] = topk_vals_full[
                                0, start_idx : start_idx + seq_len, :
                            ]
                            out_idx[i, :seq_len, :] = topk_idx_full[
                                0, start_idx : start_idx + seq_len, :
                            ]
                    return output_tensor.new_zeros(()), {
                        "topk_logits": out_vals,
                        "topk_indices": out_idx,
                    }
                else:
                    return output_tensor.new_zeros(()), {
                        "topk_logits": topk_vals_full,
                        "topk_indices": topk_idx_full,
                    }

            return output_tensor, collection_fn

        if self.cfg["dynamic_batching"]["enabled"]:
            mb_iterator = data.make_microbatch_iterator_with_dynamic_shapes()
            data_iterator_len = data.get_microbatch_iterator_dynamic_shapes_len()
            micro_batch = logprob_batch_size
        elif self.cfg["sequence_packing"]["enabled"]:
            mb_iterator = data.make_microbatch_iterator_for_packable_sequences()
            data_iterator_len, _ = (
                data.get_microbatch_iterator_for_packable_sequences_len()
            )
            micro_batch = 1
        else:
            mb_iterator = data.make_microbatch_iterator(logprob_batch_size)
            data_iterator_len = max(1, data.size // logprob_batch_size)
            micro_batch = logprob_batch_size

        forward_backward_func = get_forward_backward_func()
        list_of_outputs = forward_backward_func(
            forward_step_func=forward_step_fn,
            data_iterator=mb_iterator,
            model=self.model,
            num_microbatches=data_iterator_len,
            seq_length=pp_seq_dim_size,
            micro_batch_size=micro_batch,
            decoder_seq_length=pp_seq_dim_size,
            forward_only=True,
        )

        if is_pipeline_last_stage(ignore_virtual=True):
            logits_chunks = []
            indices_chunks = []
            for out in list_of_outputs:
                tk = out["topk_logits"]
                ti = out["topk_indices"]
                pad_len = input_seq_dim_size - tk.shape[1]
                if pad_len > 0:
                    tk = torch.nn.functional.pad(tk, (0, 0, 0, pad_len), value=0.0)
                    ti = torch.nn.functional.pad(ti, (0, 0, 0, pad_len), value=0)
                logits_chunks.append(tk)
                indices_chunks.append(ti)

            topk_logits = torch.cat(logits_chunks, dim=0)
            topk_indices = torch.cat(indices_chunks, dim=0)

            topk_logits = broadcast_tensor(
                topk_logits, torch.distributed.get_rank(), pp_grp
            )
            topk_indices = broadcast_tensor(
                topk_indices, torch.distributed.get_rank(), pp_grp
            )
        else:
            last_pp_rank = get_pipeline_model_parallel_last_rank()
            topk_logits = broadcast_tensor(None, last_pp_rank, pp_grp)
            topk_indices = broadcast_tensor(None, last_pp_rank, pp_grp)

        no_grad.__exit__(None, None, None)
        return BatchedDataDict.from_batches(
            [{"topk_logits": topk_logits.cpu(), "topk_indices": topk_indices.cpu()}]
        )

    @torch.no_grad()
    @wrap_with_nvtx_name("megatron_policy_worker/generate")
    def generate(
        self, *, data: BatchedDataDict[GenerationDatumSpec], greedy: bool = False
    ) -> BatchedDataDict[GenerationOutputSpec]:
        """Generate a batch of data using huggingface framework generation.

        Args:
            data: BatchedDataDict containing input_ids and input_lengths tensors
        Returns:
            BatchedDataDict conforming to GenerationOutputSpec:
                - output_ids: input + generated token IDs
                - logprobs: Log probabilities for each token
                - generation_lengths: Lengths of each response
        """
        # 512 bATCH SIZE (200 tokens)
        no_grad = torch.no_grad()
        no_grad.__enter__()
        self.model.config.flash_decode = False
        if self.should_disable_forward_pre_hook:
            self.model = self.move_model(
                self.model, "cuda", move_params=True, move_grads=False
            )
        # Verify input is right padded
        assert isinstance(data, BatchedDataDict), (
            f"data must be a BatchedDataDict, got type: {type(data)}"
        )
        assert "input_ids" in data and "input_lengths" in data, (
            f"input_ids and input_lengths must be present in the BatchedDataDict, got keys: {data.keys()}"
        )
        is_right_padded, error_msg = verify_right_padding(
            data, pad_value=self.tokenizer.pad_token_id
        )
        if not is_right_padded:
            warnings.warn(
                f"Input to Megatron Generation worker is not properly right-padded: {error_msg}"
            )

        model_cfg = self.megatron_cfg.model
        inference_wrapper_config = InferenceWrapperConfig(
            hidden_size=model_cfg.hidden_size,
            inference_batch_times_seqlen_threshold=1000000,
            fp32_residual_connection=model_cfg.fp32_residual_connection,
            params_dtype=model_cfg.params_dtype,
            padded_vocab_size=self.final_padded_vocab_size,  # Use the potentially updated value
            inference_max_seq_length=self.cfg["generation"]["max_new_tokens"],  # type: ignore
            inference_max_requests=self.cfg["generation_batch_size"],
        )

        from megatron.core.inference.contexts.dynamic_context import (
            DynamicInferenceContext,
        )
        from megatron.core.inference.engines.dynamic_engine import (
            DynamicInferenceEngine,
        )
        from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
            GPTInferenceWrapper,
        )
        from megatron.core.inference.sampling_params import SamplingParams

        mcore_generation_config = self.cfg["generation"]["mcore_generation_config"]
        buffer_size_gb = mcore_generation_config.get("buffer_size_gb", 20)

        num_cuda_graphs = mcore_generation_config.get("num_cuda_graphs", 16)
        block_size_tokens = mcore_generation_config.get("block_size_tokens", 256)
        use_cuda_graphs_for_non_decode_steps = mcore_generation_config.get(
            "use_cuda_graphs_for_non_decode_steps", True
        )
        enable_chunked_prefill = mcore_generation_config.get(
            "enable_chunked_prefill", True
        )
        unified_memory_level = mcore_generation_config.get("unified_memory_level", 0)
        buffer_guaranteed_fraction = mcore_generation_config.get(
            "buffer_guaranteed_fraction", 0.1
        )
        max_tokens = mcore_generation_config.get("max_tokens", 16384)

        # Get the underlying language module (not the DDP wrapper) for toggle_cuda_graphs.
        # This follows the pattern from megatron/rl/rl_utils.py megatron_rl_inference_mode().
        # The DDP wrapper can interfere with CUDA graph state management.
        lang_module = (
            self.model.module.module
            if hasattr(self.model.module, "module")
            else self.model.module
        )

        # Enable CUDA graphs for inference using toggle_cuda_graphs on the underlying module.
        # reset_cuda_graphs=True ensures fresh CUDA graphs are created for each generate() call.
        

        toggle_cuda_graphs(lang_module, set_to="local", reset_cuda_graphs=True)

        model_config = self.model.config
        model_config.cuda_graph_impl = "local"

        dynamic_context = DynamicInferenceContext(
            params_dtype=inference_wrapper_config.params_dtype,
            num_layers=model_config.num_layers,
            kv_channels=model_config.kv_channels,
            num_attention_heads=model_config.num_query_groups,
            max_sequence_length=self.cfg["generation"]["max_new_tokens"],
            buffer_guaranteed_fraction=buffer_guaranteed_fraction,
            buffer_size_gb=buffer_size_gb,
            materialize_only_last_token_logits=False,
            num_cuda_graphs=num_cuda_graphs,
            block_size_tokens=block_size_tokens,
            tensor_model_parallel_size=self.cfg["megatron_cfg"][
                "tensor_model_parallel_size"
            ],
            use_cuda_graphs_for_non_decode_steps=use_cuda_graphs_for_non_decode_steps,
            use_flashinfer_fused_rope=False,
            unified_memory_level=unified_memory_level,
            max_tokens_override=max_tokens,
        )
        inference_wrapped_model = GPTInferenceWrapper(
            self.model, inference_wrapper_config, dynamic_context
        )

        inference_wrapped_model.prep_model_for_inference()
        # Set pipeline parallel flag
        inference_wrapped_model.model_is_pipeline_parallel = (
            self.cfg["megatron_cfg"]["pipeline_model_parallel_size"] > 1
        )

        text_generation_controller = TextGenerationController(
            inference_wrapped_model=inference_wrapped_model,
            tokenizer=self.megatron_tokenizer,
        )

        # Calculate seed based on node and rank to ensure reproducibility across workers
        local_rank = torch.cuda.current_device()  # Local GPU index on the node
        num_gpus_per_node = torch.cuda.device_count()
        node_idx = self.rank // num_gpus_per_node if num_gpus_per_node > 0 else 0
        seed = (node_idx * 1024) + local_rank

        # New API: DynamicInferenceEngine has additional parameters
        dynamic_engine = DynamicInferenceEngine(
            text_generation_controller,
            dynamic_context,
            enable_cuda_graph=True,
            random_seed=seed,
            track_paused_request_events=False,
            enable_chunked_prefill=enable_chunked_prefill,
            inference_logging_step_interval=0,
        )

        # Handle None values for top_k - convert to integer as required by Megatron
        top_k_cfg = self.cfg["generation"]["top_k"]
        top_k_val = 1 if greedy else (int(top_k_cfg) if top_k_cfg is not None else 0)

        top_p_cfg = self.cfg["generation"]["top_p"]
        top_p_val = (
            0.0 if greedy else (float(top_p_cfg) if top_p_cfg is not None else 0.0)
        )

        # New API: SamplingParams now includes termination_id and uses num_tokens_total
        sampling_params = SamplingParams(
            temperature=self.cfg["generation"]["temperature"] if not greedy else 0,
            top_k=top_k_val,
            top_p=top_p_val,
            skip_prompt_log_probs=False,
            return_log_probs=True,
            num_tokens_total=self.cfg["generation"]["max_new_tokens"],
            num_tokens_to_generate=None,
            termination_id=self.megatron_tokenizer.eod,
        )

        input_ids = data["input_ids"]
        prompt_tokens_tensor = input_ids.cuda()
        prompt_lengths_tensor = data["input_lengths"]
        request_id = 0

        # New API: add_request now takes sampling_params as a parameter
        for p, prompt_len in zip(
            prompt_tokens_tensor, prompt_lengths_tensor, strict=True
        ):
            dynamic_engine.add_request(
                request_id,
                p[:prompt_len],
                sampling_params=sampling_params,
            )
            request_id += 1

        result = []
        import time
        start = time.time()
        while dynamic_engine.has_unfinished_requests():
            result_step = dynamic_engine.step_modern(verbose=False)
            finished_requests = result_step.get("finished_requests", [])
            for finished_request in finished_requests:
                result.append(finished_request)
        end = time.time()
        print(f"SHAN : Time taken: {end - start} seconds")

        # Sort results by request_id to maintain original batch order
        result.sort(key=lambda x: x.request_id)

        out = {
            "tokens": [x.prompt_tokens.tolist() + x.generated_tokens for x in result],
            "logprobs": [x.prompt_log_probs + x.generated_log_probs for x in result],
        }

        input_lengths = data["input_lengths"]
        # pad the out "tokens" and "logprobs" and make them into tensors from lists
        batch_size = data["input_ids"].size(0)
        max_gen_seq_len = max([len(x.generated_tokens) for x in result])
        padded_input_length = input_ids.size(1)

        max_seq_len = padded_input_length + max_gen_seq_len
        # Create padded tensors for tokens and logprobs
        output_ids_padded = torch.full(
            (batch_size, max_seq_len),
            self.tokenizer.pad_token_id,
            dtype=torch.long,
            device=data["input_ids"].device,
        )

        logprobs_padded = torch.zeros(
            (batch_size, max_seq_len),
            dtype=torch.float,
            device=data["input_ids"].device,
        )

        # Fill in the padded tensors with actual values
        generation_lengths = torch.zeros(
            batch_size, dtype=torch.long, device=data["input_ids"].device
        )
        unpadded_sequence_lengths = torch.zeros(
            batch_size, dtype=torch.long, device=data["input_ids"].device
        )
        for i in range(batch_size):
            seq_len = len(out["tokens"][i])
            output_ids_padded[i, :seq_len] = torch.tensor(
                out["tokens"][i], dtype=torch.long, device=data["input_ids"].device
            )
            generation_lengths[i] = seq_len - input_lengths[i].item()
            unpadded_sequence_lengths[i] = seq_len
            logprob_len = len(out["logprobs"][i])
            logprobs_padded[i, 1 : logprob_len + 1] = torch.tensor(
                out["logprobs"][i],
                dtype=torch.float,
                device=data["input_ids"].device,
            )

        out_dict = {
            "output_ids": output_ids_padded,
            "logprobs": logprobs_padded,
            "generation_lengths": generation_lengths,
            "unpadded_sequence_lengths": unpadded_sequence_lengths,
        }

        self.model.config.flash_decode = False

        # Disable CUDA graphs after inference using toggle_cuda_graphs on the underlying module.
        # This follows the pattern from megatron/rl/rl_utils.py megatron_rl_inference_mode().
        # reset_cuda_graphs=True ensures:
        # 1. cudagraph_manager attributes are deleted from modules
        # 2. _CudagraphGlobalRecord state is reset via delete_cuda_graphs()
        # 3. Model is ready for training (avoids "inference tensor" errors)
        # 4. Next generate() call will create fresh CUDA graphs with new DynamicInferenceContext
        toggle_cuda_graphs(lang_module, set_to="none", reset_cuda_graphs=True)

        # Force release CUDA graph memory to prevent OOM at next step
        torch.cuda.empty_cache()

        no_grad.__exit__(None, None, None)

        return BatchedDataDict.from_batches([out_dict]).to("cpu")

    @torch.no_grad()
    @wrap_with_nvtx_name("megatron_policy_worker/prepare_refit_info")
    def prepare_refit_info(self) -> None:
        """Prepare state dict metadata for weight refitting and IPC streaming."""
        self.refit_param_info_mcore = self._calculate_refit_param_info()

        # Collect tensor metadata for refit / hf side info
        refit_param_info_hf = {}
        # Reuse shared iterator that appends FP8 KV/Q scales when enabled
        for name, tensor in self._iter_params_with_optional_kv_scales():
            refit_param_info_hf[name] = (tensor.shape, tensor.dtype)

        return refit_param_info_hf

    def _calculate_refit_param_info(self) -> list[tuple[str, int]]:
        """Calculate parameter information for refit.

        Each task contains:
        - param_name: Local parameter name without module prefixes
        - mapping: MegatronParamMapping instance for weight transformation
        - pp_rank: Pipeline-parallel rank owning the parameter
        - vp_stage: Virtual-pipeline stage index
        - megatron_module: Reference to Megatron model/submodule
        - param_weight: Target parameter tensor for converted weight

        Returns:
            List of (parameter_name, size_in_bytes) tuples.
        """
        self.refit_conversion_tasks = self.megatron_bridge.get_conversion_tasks(
            [self.model]
        )
        param_info = []

        def calculate_size_in_bytes(param, tp_size, ep_size):
            if param is None:
                # need to broadcast for other pp ranks
                size_in_bytes = None
            else:
                # Calculate size for this parameter
                prec_to_bytes = {
                    torch.bfloat16: 2,
                    torch.float16: 2,
                    torch.float32: 4,
                }
                scale = prec_to_bytes[self.dtype] / prec_to_bytes[param.dtype]
                size_in_bytes = (
                    param.element_size() * param.numel() * tp_size * ep_size * scale
                )

            # Broadcast size_in_bytes across pipeline parallel ranks
            return broadcast_object_across_pp_ranks(size_in_bytes)

        for task in self.refit_conversion_tasks:
            param_info.append(
                (
                    task.param_name,
                    calculate_size_in_bytes(
                        task.param_weight,
                        task.mapping.tp_size,
                        task.mapping.ep_size if task.mapping.is_expert else 1,
                    ),
                )
            )
        return param_info

    def _iter_params_with_optional_kv_scales(
        self,
        kv_scales: Optional[dict[str, float]] = None,
    ) -> Iterator[tuple[str, torch.Tensor]]:
        """Yield exported HF parameters and optionally append FP8 KV/Q scale tensors.

        This helper is used by both IPC-based streaming and collective broadcast
        so that the logic for adding KV scales stays consistent in one place.
        """
        base_iter = self.megatron_bridge.export_hf_weights(
            [self.model],
            show_progress=False,
            conversion_tasks=self.refit_conversion_tasks,  # used for metadata caching
        )

        # Yield the original parameters first.
        for name, tensor in base_iter:
            yield name, tensor

        # Check whether FP8 KV cache is enabled.
        use_fp8_kv_cache = False
        if (
            "generation" in self.cfg
            and self.cfg["generation"] is not None
            and self.cfg["generation"]["backend"] == "vllm"
        ):
            generation_cfg = cast(VllmConfig, self.cfg["generation"])
            use_fp8_kv_cache = (
                "vllm_cfg" in generation_cfg
                and "kv_cache_dtype" in generation_cfg["vllm_cfg"]
                and generation_cfg["vllm_cfg"]["kv_cache_dtype"].startswith("fp8")
            )

        if not use_fp8_kv_cache:
            return

        # Append KV (and potentially Q) scale entries to match metadata.
        num_layers = self.megatron_bridge.transformer_config.num_layers
        keys: list[str] = []
        for layer_idx in range(num_layers):
            scale_names = get_vllm_qkv_scale_names(layer_idx)
            keys.extend(scale_names.values())

        for param_name in keys:
            if kv_scales and param_name in kv_scales:
                scale_value = kv_scales[param_name]
            else:
                scale_value = 1.0
            scale_tensor = torch.tensor(
                scale_value, dtype=torch.float32, device="cuda"
            ).reshape(1)
            yield param_name, scale_tensor

    @torch.no_grad()
    @wrap_with_nvtx_name("megatron_policy_worker/stream_weights_via_ipc_zmq")
    def stream_weights_via_ipc_zmq(
        self, buffer_size_bytes: int = 0, kv_scales: Optional[dict[str, float]] = None
    ) -> None:
        """Stream model weights to peer process via ZMQ IPC socket."""
        self.maybe_init_zmq()

        from nemo_rl.models.policy.utils import stream_weights_via_ipc_zmq_impl

        # Use the shared implementation to append optional KV scales.
        stream_weights_via_ipc_zmq_impl(
            params_generator=self._iter_params_with_optional_kv_scales(
                kv_scales=kv_scales
            ),
            buffer_size_bytes=buffer_size_bytes,
            zmq_socket=self.zmq_socket,
            rank=self.rank,
            worker_name=str(self),
        )

    @torch.no_grad()
    def broadcast_weights_for_collective(
        self, kv_scales: Optional[dict[str, float]] = None
    ) -> None:
        """Broadcast the weights for collective communication."""
        # param_iterator will return (name, tensor), we only need tensor.
        packed_broadcast_producer(
            iterator=self._iter_params_with_optional_kv_scales(kv_scales=kv_scales),
            group=self.model_update_group,
            src=0,
            post_iter_func=lambda x: x[1],
        )

    def prepare_for_lp_inference(self):
        self.model = self.move_model(self.model, "cuda", move_grads=False)
        self.model.eval()

        # offload grads to cpu
        self.model = self.move_model(
            self.model, "cpu", move_params=False, move_grads=True
        )  # get rid of grad buffers

        # offload optimizer to cpu
        torch.randn(1).cuda()  # wake up torch allocator
        if (
            hasattr(self, "optimizer")
            and self.optimizer is not None
            and not self.optimizer_cpu_offload
            and self.offload_optimizer_for_logprob
        ):
            self.move_optimizer("cpu")

        gc.collect()
        torch.cuda.empty_cache()

    def prepare_for_training(self, *args, **kwargs):
        # onload models and optimizer state to cuda
        self.model = self.move_model(
            self.model, "cuda", move_grads=True, move_params=True
        )
        self.model.train()

        # Move optimizer state to CUDA if it exists
        # colocated generation will always offload optimizer to cuda before refit
        if (
            hasattr(self, "optimizer")
            and self.optimizer is not None
            and not self.optimizer_cpu_offload
            and (self.offload_optimizer_for_logprob or self.is_generation_colocated)
        ):
            self.move_optimizer("cuda")

        if self.cfg["megatron_cfg"]["empty_unused_memory_level"] >= 1:
            torch.cuda.empty_cache()

    @wrap_with_nvtx_name("megatron_policy_worker/offload_before_refit")
    def offload_before_refit(self):
        """Offload the optimizer and buffers to the CPU."""
        no_grad = torch.no_grad()
        no_grad.__enter__()
        allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
        reserved = torch.cuda.memory_reserved() / (1024**3)  # Convert to GB
        print(
            f"GPU Memory before optimizer offload: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )
        self.model = self.move_model(
            self.model, "cpu", move_params=False, move_grads=True
        )  # get rid of grad buffers
        torch.randn(1).cuda()  # wake up torch allocator
        if (
            hasattr(self, "optimizer")
            and self.optimizer is not None
            and not self.optimizer_cpu_offload
        ):
            self.move_optimizer("cpu")

        gc.collect()
        torch.cuda.empty_cache()

        # Print memory stats after offloading
        allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
        reserved = torch.cuda.memory_reserved() / (1024**3)  # Convert to GB
        print(
            f"GPU Memory after optimizer offload: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )
        no_grad.__exit__(None, None, None)

    @wrap_with_nvtx_name("megatron_policy_worker/offload_after_refit")
    def offload_after_refit(self):
        """Offload as much as possible on the CPU."""
        no_grad = torch.no_grad()
        no_grad.__enter__()
        self.model = self.move_model(self.model, "cpu")
        self.model.eval()
        torch.randn(1).cuda()  # wake up torch allocator
        self.offload_before_refit()  # rerun the old offload function

        allocated = torch.cuda.memory_allocated() / (1024**3)  # Convert to GB
        reserved = torch.cuda.memory_reserved() / (1024**3)  # Convert to GB
        print(
            f"GPU Memory after refit complete: {allocated:.2f}GB allocated, {reserved:.2f}GB reserved"
        )
        no_grad.__exit__(None, None, None)

    @torch.no_grad()
    def move_model(
        self,
        model: torch.nn.Module,
        device: str,
        move_params: bool = True,
        move_grads: bool = True,
    ) -> torch.nn.Module:
        # move all param and grad buffers to the device
        if isinstance(model, DistributedDataParallel):
            # DDP case
            for buffers in [model.buffers, model.expert_parallel_buffers]:
                for buffer_idx in range(len(buffers)):
                    if device == "cpu":
                        buffers[buffer_idx].offload_to_cpu(
                            move_params=move_params, move_grads=move_grads
                        )
                    elif device == "cuda":
                        buffers[buffer_idx].reload_from_cpu(
                            move_params=move_params, move_grads=move_grads
                        )
                    else:
                        raise ValueError(
                            f"Invalid device: {device}. Only strings 'cpu' and 'cuda' are supported."
                        )
        elif isinstance(model, custom_FSDP):
            if device == "cpu":
                model.param_and_grad_buffer.offload_to_cpu(move_params, move_grads)
            elif device == "cuda":
                model.param_and_grad_buffer.reload_from_cpu(
                    move_params=move_params, move_grads=move_grads
                )
            else:
                raise ValueError(
                    f"Invalid device: {device}. Only strings 'cpu' and 'cuda' are supported."
                )
        else:
            # Ordinary offload case
            if move_params:
                for name, param in model.state_dict().items():
                    new_state_dict = {}
                    for name, item in model.state_dict().items():
                        if isinstance(item, torch.Tensor):
                            item = item.detach().to(
                                device=device, non_blocking=True, copy=True
                            )
                        new_state_dict[name] = item
                    model.load_state_dict(new_state_dict)
        return model

    def move_optimizer(self, device: str):
        # Iterate through the state dictionaries for each parameter group
        if isinstance(self.optimizer, ChainedOptimizer):
            optimizer_state = self.optimizer.state
        else:
            optimizer_state = self.optimizer._get_state()
        for _, state in optimizer_state.items():
            # Iterate through the state items (e.g., momentum, variance) for a parameter
            for k, v in state.items():
                # Check if the item is a tensor
                if torch.is_tensor(v):
                    # Move the tensor to device and update the state dictionary
                    if device == "cpu":
                        if v.is_cuda:
                            state[k] = v.to("cpu")
                    elif device == "cuda":
                        if not v.is_cuda:
                            state[k] = v.to("cuda")
                    else:
                        raise ValueError(
                            f"Invalid device: {device}. Only strings 'cpu' and 'cuda' are supported."
                        )

    def save_checkpoint(
        self,
        weights_path: str,
        optimizer_path: Optional[str] = None,
        **kwargs,
    ):
        """Save a training checkpoint.

        Args:
            weights_path: The specific directory path where the checkpoint will be saved.
            optimizer_path: If not None, optimizer and scheduler states are saved if they exist.
        """
        if not torch.distributed.is_initialized():
            raise RuntimeError(
                "Distributed process group is not initialized. Cannot save checkpoint."
            )

        if self.mcore_state is None or self.model is None:
            raise RuntimeError(
                "Megatron core state or model is not initialized. Cannot save checkpoint."
            )

        original_save_path = self.mcore_state.cfg.checkpoint.save
        # save_dir = os.path.dirname(weights_path)
        release_name = os.path.basename(weights_path)

        try:
            maybe_finalize_async_save(
                self.mcore_state,
                ckpt_cfg=self.mcore_state.cfg.checkpoint,
                blocking=False,
            )
            self.mcore_state.cfg.checkpoint.save = weights_path

            optimizer_to_save = None
            scheduler_to_save = None

            if optimizer_path is not None:
                if self.optimizer is not None:
                    optimizer_to_save = self.optimizer
                if self.scheduler is not None:
                    scheduler_to_save = self.scheduler

            # Ensure model is in eval mode for consistent saving, unless actively training
            # This is a common practice, though NeMo's save might handle this.
            # For safety, if not in training loop, setting to eval.
            is_training = self.model.training
            if not is_training:
                self.model.eval()

            if self.should_disable_forward_pre_hook:
                self.disable_forward_pre_hook()
            save_checkpoint(
                state=self.mcore_state,
                model=[self.model],
                optimizer=optimizer_to_save,
                opt_param_scheduler=scheduler_to_save,
                num_floating_point_operations_so_far=self.mcore_state.train_state.floating_point_operations_so_far,
                checkpointing_context=self.checkpointing_context,
            )
            print(f"Saved checkpoint to {weights_path}")
            maybe_finalize_async_save(
                self.mcore_state,
                ckpt_cfg=self.mcore_state.cfg.checkpoint,
                blocking=True,
                terminate=True,
            )
            if self.should_disable_forward_pre_hook:
                self.enable_forward_pre_hook()

            if not is_training:  # Restore training state if it was changed
                self.model.train()

        except Exception as e:
            print(f"Failed to save checkpoint to {weights_path}: {e}")
            raise
        finally:
            self.mcore_state.cfg.checkpoint.save = original_save_path

    def load_checkpoint(self, weights_path: str, optimizer_path: Optional[str] = None):
        """Load a training checkpoint.

        Args:
            weights_path: The exact directory path from which to load the checkpoint.
            optimizer_path: If not None, attempts to load optimizer and scheduler states
                            if self.optimizer and self.scheduler are initialized.
        """
        raise NotImplementedError(
            "Loading checkpoints outside of the init function is not yet implemented for Megatron policy."
        )

    def check_tensor_parallel_attributes(self) -> dict[str, Any]:
        """Check tensor parallel attributes on model parameters.

        Returns:
            Dictionary containing information about tensor parallel parameters:
            - tp_params: List of parameter names that have tensor_model_parallel=True
            - non_tp_params: List of parameter names that have tensor_model_parallel=False
            - total_params: Total number of parameters checked
            - tp_size: Tensor parallel size from config
        """
        tp_params = []
        non_tp_params = []
        total_params = 0

        for name, param in self.model.named_parameters():
            total_params += 1
            tensor_model_parallel = getattr(param, "tensor_model_parallel", False)

            if tensor_model_parallel:
                tp_params.append(
                    {
                        "name": name,
                        "tensor_model_parallel": tensor_model_parallel,
                        "partition_dim": getattr(param, "partition_dim", None),
                        "partition_stride": getattr(param, "partition_stride", None),
                        "shape": list(param.shape),
                    }
                )
            else:
                non_tp_params.append(
                    {
                        "name": name,
                        "tensor_model_parallel": tensor_model_parallel,
                        "shape": list(param.shape),
                    }
                )

        return {
            "tp_params": tp_params,
            "non_tp_params": non_tp_params,
            "total_params": total_params,
            "tp_size": self.megatron_cfg.model.tensor_model_parallel_size,
        }

    @torch.no_grad()
    def calibrate_qkv_fp8_scales(
        self,
        *,
        data: BatchedDataDict[Any],
        micro_batch_size: Optional[int] = None,
        percentile: float = 99.9,
        margin: float = 1.05,
        include_q: bool = False,
    ) -> dict[str, Any]:
        """One-shot calibration of Q/K/V activation scales (for FP8 KV cache).

        - Captures each layer's `query_key_value` output through forward hooks, splits Q/K/V, and computes percentile amax.
        - In parallel (DP/TP/PP) environments, first computes local percentiles, then takes max across all ranks for conservativeness.
        - By default only returns and saves K/V scales, optionally returns Q.

        Args:
            data: Representative sample batch for calibration, following get_logprobs input conventions.
            micro_batch_size: Micro batch size during calibration; if None, reuses logprob_batch_size.
            percentile: Percentile for amax (e.g. 99.9).
            margin: Margin factor, e.g. 1.05.
            save_path: If provided, rank0 will save results as JSON.
            include_q: Whether to also return Q scale (usually only K/V needed).

        Returns:
            { "format": "fp8", "percentile": float, "margin": float,
              "layers": { layer_name: {"k_scale": float, "v_scale": float[, "q_scale": float] } } }
        """

        # Allow overriding FP8 max for Q, K, V via environment variables for ease of testing.
        # Defaults align with FP8 e4m3 max magnitude.
        # Use different defaults for Q, K, V to adapt to distribution diffefences
        def _get_env_float(name: str, default: float) -> float:
            try:
                val = os.getenv(name, None)
                return float(val) if val is not None and val != "" else default
            except Exception:
                return default

        FP8_MAX_Q = _get_env_float("FP8_MAX_Q", 448.0)
        FP8_MAX_K = _get_env_float("FP8_MAX_K", 448.0)
        FP8_MAX_V = _get_env_float("FP8_MAX_V", 448.0)

        self.model.eval()

        # Record local percentile amax for q/k/v of each layer
        layer_to_samples_q: dict[str, list[float]] = defaultdict(list)
        layer_to_samples_k: dict[str, list[float]] = defaultdict(list)
        layer_to_samples_v: dict[str, list[float]] = defaultdict(list)
        hook_handles = []

        def _extract_layer_key(module_name: str) -> str:
            # Expected format: "module.decoder.layers.<idx>.self_attention.query_key_value"
            m = re.search(r"module\.decoder\.layers\.(\d+)", module_name)
            if m is not None:
                return f"layer_{m.group(1)}"
            return module_name

        # Hook to capture q/k/v after q/k norm and RoPE
        def _pre_hook_builder_core_attention(module_name: str):
            layer_key = _extract_layer_key(module_name)

            def _pre_hook(module, inputs):
                args = inputs if isinstance(inputs, (tuple, list)) else (inputs,)
                if len(args) == 1 and isinstance(args[0], (tuple, list)):
                    args = args[0]
                # Expected first 3 args to be q, k, v (typical signature for Megatron CoreAttention)
                q = args[0]
                k = args[1]
                v = args[2]
                if include_q:
                    layer_to_samples_q[layer_key].append(
                        float(torch.amax(torch.abs(q)).item())
                    )
                layer_to_samples_k[layer_key].append(
                    float(torch.amax(torch.abs(k)).item())
                )
                layer_to_samples_v[layer_key].append(
                    float(torch.amax(torch.abs(v)).item())
                )

            return _pre_hook

        matched_modules = []
        # Try to register forward_pre_hook on core_attention first
        for name, module in self.model.named_modules():
            if "self_attention.core_attention" in name:
                try:
                    handle = module.register_forward_pre_hook(
                        _pre_hook_builder_core_attention(name)
                    )
                    hook_handles.append(handle)
                    matched_modules.append((name, module.__class__.__name__, "pre"))
                except Exception as e:
                    print(
                        f"Error registering pre-hook for qkv scale calibration on {name}: {e}"
                        " Please check if the model is compatible with the current calibration logic. "
                        "The expected module name is 'self_attention.core_attention'."
                    )
                    raise

        # Run a forward pass to trigger hooks (reuse get_logprobs forward path)
        try:
            _ = self.get_logprobs(data=data, micro_batch_size=micro_batch_size)
        finally:
            for h in hook_handles:
                try:
                    h.remove()
                except Exception as e:
                    print(f"Error removing hook for qkv scale calibration: {e}")
                    raise

        # Compute local percentile amax
        def _percentile(values: list[float], p: float) -> float:
            if not values:
                return 0.0
            t = torch.tensor(sorted(values), device="cuda", dtype=torch.float32)
            rank = max(
                0, min(len(values) - 1, int(round((p / 100.0) * (len(values) - 1))))
            )
            return float(t[rank].item())

        local_layer_to_pamax = {}
        for layer_key in set(
            list(layer_to_samples_k.keys())
            + list(layer_to_samples_v.keys())
            + (list(layer_to_samples_q.keys()) if include_q else [])
        ):
            entry = {}
            if include_q:
                entry["q_amax_p"] = _percentile(
                    layer_to_samples_q.get(layer_key, []), percentile
                )
            entry["k_amax_p"] = _percentile(
                layer_to_samples_k.get(layer_key, []), percentile
            )
            entry["v_amax_p"] = _percentile(
                layer_to_samples_v.get(layer_key, []), percentile
            )
            local_layer_to_pamax[layer_key] = entry

        # Merge across all ranks: take maximum of percentile amax (conservative approach)
        world_size = (
            torch.distributed.get_world_size()
            if torch.distributed.is_initialized()
            else 1
        )
        gathered = [None for _ in range(world_size)] if world_size > 1 else None
        if world_size > 1:
            torch.distributed.all_gather_object(gathered, local_layer_to_pamax)
            merged = defaultdict(dict)
            for d in gathered:  # type: ignore
                if d is None:
                    continue
                for k, v in d.items():
                    dst = merged[k]
                    for kk, vv in v.items():
                        dst[kk] = max(dst.get(kk, 0.0), float(vv))
            layer_to_pamax = dict(merged)
        else:
            layer_to_pamax = local_layer_to_pamax

        # Compute scale (symmetric quantization): scale = pamax / fp8_max
        result_layers = {}
        for layer_key, vals in layer_to_pamax.items():
            out_entry = {}
            if include_q:
                q_scale = (vals.get("q_amax_p", 0.0) * margin) / FP8_MAX_Q
                out_entry["q_scale"] = float(q_scale)
            k_scale = (vals.get("k_amax_p", 0.0) * margin) / FP8_MAX_K
            v_scale = (vals.get("v_amax_p", 0.0) * margin) / FP8_MAX_V
            out_entry["k_scale"] = float(k_scale)
            out_entry["v_scale"] = float(v_scale)
            result_layers[layer_key] = out_entry

        vllm_format_scales = convert_calibration_to_vllm_format(result_layers)

        final_result = {
            "format": "fp8",
            "percentile": percentile,
            "margin": margin,
            "layers": vllm_format_scales,
        }

        # Sync results across all ranks (broadcast rank0's result)
        if world_size > 1:
            if torch.distributed.get_rank() == 0:
                obj_list = [final_result]
                torch.distributed.broadcast_object_list(obj_list, src=0)
                final_result = obj_list[0]
            else:
                obj_list = [None]
                torch.distributed.broadcast_object_list(obj_list, src=0)
                final_result = obj_list[0]  # type: ignore

        return final_result


class CustomFloat16Module(Float16Module):
    """Float 16 Module.

    Attributes:
        config (TransformerConfig): Transformer config
        fp16 (bool) : Specifies if the model runs in fp16 mode
        bf16 (bool) : Specifies if the model runs in bf16 mode

    Args:
        config (TransformerConfig): The transformer config used to initalize the model
    """

    def __init__(self, config: TransformerConfig, module: torch.nn.Module):
        super(CustomFloat16Module, self).__init__(config, module)
        self.re_enable_float32_expert_bias()

    def re_enable_float32_expert_bias(self) -> None:
        """Ensure MoE router expert bias stays in float32 for numerical stability.

        Walks the wrapped module to find MoE routers and invokes the
        `_maintain_float32_expert_bias()` helper which recreates or casts the
        expert bias tensors to float32 as required by Megatron-LM.
        """
        module = self.module
        # Handle VLM models where language model is nested
        if hasattr(module, "language_model"):
            module = module.language_model
        if hasattr(module, "decoder") and hasattr(module.decoder, "layers"):
            for layer in module.decoder.layers:
                mlp = getattr(layer, "mlp", None)
                router = getattr(mlp, "router", None) if mlp is not None else None
                if router is not None and hasattr(
                    router, "_maintain_float32_expert_bias"
                ):
                    router._maintain_float32_expert_bias()

```

Only pre-existing import warnings. 

## Summary

I've moved the memory cleanup to **after** all the inference state resets but **before** training starts. This should help with fragmentation.

```python
# Reset inference states first
self.model.zero_grad_buffer()
# ... reset caches ...

# Then aggressive cleanup
gc.collect()
torch.cuda.empty_cache()
```

## But More Importantly...

The error message tells us exactly what to do. Please also try running with this environment variable:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

This tells PyTorch to use expandable memory segments, which helps avoid fragmentation issues where you have enough total free memory but not enough **contiguous** memory.

## Why Step 2 Fails But Step 1 Works

| Step | GPU Memory State |
|------|------------------|
| Step 1 start | Clean, defragmented |
| Step 1 generate | Allocates for CUDA graphs, inference |
| Step 1 train | Allocates for gradients, works fine |
| Step 1 end | Memory freed but **fragmented** |
| Step 2 generate | New allocations in fragmented memory |
| Step 2 train | Can't find contiguous 1.15GB block → **OOM** |

The `expandable_segments:True` option is PyTorch's solution for exactly this problem.

**Try both:**
1. The code change (already applied)
2. Set `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` in your environment

```python
# Reset inference states first
self.model.zero_grad_buffer()
# ... reset caches ...

# Then aggressive cleanup
gc.collect()
torch.cuda.empty_cache()
```

```shellscript
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

```python
# Reset inference states first
self.model.zero_grad_buffer()
# ... reset caches ...

# Then aggressive cleanup
gc.collect()
torch.cuda.empty_cache()
```

```shellscript
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

