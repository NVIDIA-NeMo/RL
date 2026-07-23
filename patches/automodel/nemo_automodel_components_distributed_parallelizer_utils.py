# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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

from typing import Callable, Iterator, List, Optional, Set, Tuple, Union

import torch
import torch.nn as nn
from torch.distributed.fsdp import (
    fully_shard,
)
from torch.distributed.tensor import Shard

UniformSubtreeItem = Union[Tuple[nn.Module, torch.dtype], Tuple[str, nn.Module, torch.dtype]]


def attr_shard_placement_fn(param: nn.Parameter) -> Optional[Shard]:
    """``fully_shard`` placement hook honoring a per-parameter shard-dim hint.

    Parameters may carry an integer ``_fsdp_shard_dim`` attribute set by the
    module that constructed them (e.g. stacked multi-adapter LoRA parameters
    of shape ``[n_adapters, r, in_features]`` that must shard on dim 1 so each
    adapter slot's local shard matches the layout of an independently-sharded
    single-adapter ``[r, in_features]`` parameter).

    Returns ``None`` for parameters without the attribute, which keeps FSDP's
    default dim-0 sharding — stock models are byte-identically unaffected.
    """
    dim = getattr(param, "_fsdp_shard_dim", None)
    if dim is None:
        return None
    return Shard(int(dim))


def iter_maximal_uniform_dtype_subtrees(
    module: nn.Module,
    *,
    include_buffers: bool = True,
    tensor_pred: Optional[Callable[[torch.Tensor], bool]] = None,
    return_paths: bool = False,
) -> Iterator[UniformSubtreeItem]:
    """
    Traverse `module` and yield maximal submodules whose entire subtree has a unified dtype.

    - include_buffers: include buffers in dtype unification checks.
    - tensor_pred: predicate to choose which tensors to consider (default: all).
                   Example: tensor_pred=torch.is_floating_point  (to consider only FP tensors)
    - return_paths: if True, yields (qualified_name, module, dtype); else (module, dtype).

    Notes:
    - If a module subtree has no tensors passing `tensor_pred`, it is ignored.
    - Maximality ensures no yielded module is a strict child of another yielded module.
    """
    if tensor_pred is None:
        tensor_pred = lambda t: True

    def _local_dtype_set(m: nn.Module) -> Set[torch.dtype]:
        ds: Set[torch.dtype] = set()
        for p in m.parameters(recurse=False):
            if tensor_pred(p):
                ds.add(p.dtype)
        if include_buffers:
            for b in m.buffers(recurse=False):
                if tensor_pred(b):
                    ds.add(b.dtype)
        return ds

    def _visit(m: nn.Module, path: Tuple[str, ...]) -> Tuple[Set[torch.dtype], List[UniformSubtreeItem]]:
        local = _local_dtype_set(m)
        subtree_dtypes: Set[torch.dtype] = set(local)
        collected: List[UniformSubtreeItem] = []

        # Recurse into children
        for name, child in m.named_children():
            child_set, child_yields = _visit(child, path + (name,))
            subtree_dtypes |= child_set
            collected.extend(child_yields)

        # If entire subtree has exactly one dtype (and not empty), this node is maximal: override children yields
        if len(subtree_dtypes) == 1:
            if subtree_dtypes:
                dtype = next(iter(subtree_dtypes))
                if return_paths:
                    qname = ".".join(path)  # empty string at root
                    return subtree_dtypes, [(qname, m, dtype)]
                else:
                    return subtree_dtypes, [(m, dtype)]
            # else: no tensors in subtree -> ignore entirely
        # Not uniform -> keep whatever maximal sets children produced
        return subtree_dtypes, collected

    _, items = _visit(module, ())
    # Stream results
    for it in items:
        yield it


def _group_params_by_dtype(layer):
    ans = {}
    for name, param in layer.named_parameters():
        if param.dtype not in ans:
            ans[param.dtype] = []
        ans[param.dtype].append(param)
    return ans


def _get_module_from_path(layer, path):
    for name in path.split("."):
        layer = getattr(layer, name)
    return layer


def _fully_shard(module, mesh, mp_policy, offload_policy):
    if isinstance(module, nn.ModuleList):
        for layer in module:
            _fully_shard(layer, mesh, mp_policy, offload_policy)
    else:
        fully_shard(
            module,
            mesh=mesh,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
            shard_placement_fn=attr_shard_placement_fn,
        )


def fully_shard_by_dtype(module, mesh, mp_policy, offload_policy):
    # calling _group_params_by_dtype is not optimal here, because we may
    # end up with two traversals over the module, but this code is not in the hot path.
    grouped_params = _group_params_by_dtype(module)
    if len(grouped_params) == 0:
        return
    elif len(grouped_params) == 1:
        fully_shard(
            module,
            mesh=mesh,
            mp_policy=mp_policy,
            offload_policy=offload_policy,
            shard_placement_fn=attr_shard_placement_fn,
        )
    else:
        least_items_dtype = min(grouped_params.items(), key=lambda x: len(x[1]))[0]
        for path, mod, dtype in iter_maximal_uniform_dtype_subtrees(
            module,
            tensor_pred=torch.is_floating_point,
            return_paths=True,
        ):
            if (len(grouped_params) == 2 and dtype == least_items_dtype) or len(grouped_params) > 2:
                _fully_shard(
                    _get_module_from_path(module, path),
                    mesh=mesh,
                    mp_policy=mp_policy,
                    offload_policy=offload_policy,
                )
        if len(grouped_params) == 2:
            fully_shard(
                module,
                mesh=mesh,
                mp_policy=mp_policy,
                offload_policy=offload_policy,
                shard_placement_fn=attr_shard_placement_fn,
            )
