# Native MXFP8 NCCL Reshard Refit Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Transfer Megatron `fp8_param=true` MXFP8 weights and E8M0 scales directly into an MXFP8 vLLM rollout model through NCCL Reshard, without a BF16 staging round trip.

**Architecture:** Keep PR #3477's TP/EP-aware `xferdtensor` path as the data plane. Add ordered checkpoint-format components beneath each logical HF parameter and bind them through a vLLM 0.25.1 adapter that owns native layerwise reload setup and finalization. The core plan and transport remain version-neutral; later vLLM changes stay inside the adapter.

**Tech Stack:** Python 3.13, PyTorch and DTensor placements, Transformer Engine MXFP8 metadata, Megatron-Core and Megatron-Bridge, vLLM 0.25.1 ModelOpt MXFP8, Ray, NCCL, pytest, Pyrefly, pre-commit.

**Spec:** `docs/superpowers/specs/2026-08-27-native-mxfp8-nccl-reshard-refit-design.md`

## Global Constraints

- Start implementation in a fresh worktree from `origin/main` commit `4c6e5c84e9676c2f4e178210ad0171f0c1cfddcd`; do not continue from the 4,500-line prototype branch.
- Current `origin/main` already has the native layerwise lifecycle entrypoints from PR #3545. Implement the PR #3651 architecture against those entrypoints rather than stacking PR #3651's stale head `7300feb586e7105eb934c79186182f9eb6a88024`.
- Target NeMo-RL's pinned vLLM 0.25.1. vLLM 0.28 is a non-blocking API-contract probe until NeMo-RL bumps its dependency.
- Initial runtime support is Megatron policy to vLLM generation, non-colocated NCCL Reshard, ETP=1, and dense FFN plus routed-MoE FC1/FC2 parameters.
- QKVO, Mamba, embeddings, shared experts, routers, output heads, MTP heads, and ignored layers remain on the existing BF16 misc path.
- The wire format is canonical checkpoint storage: E4M3 `weight` plus uint8 E8M0 `weight_scale`. Never transfer CuTeDSL-transposed weights, FlashInfer-swizzled scales, or TRTLLM-packed experts.
- Keep BF16, BF16-to-MXFP8 receiver quantization, and matching blockwise-FP8 behavior unchanged.
- Reject native MXFP8-to-BF16, format-mismatched FP8, ETP>1, and GEMM-swizzled Transformer Engine source scales before communicator setup.
- The NCCL Reshard core must not call vLLM `process_weights_after_loading()` directly. The vLLM adapter is the only owner of native reload initialization, finalization, and failure state.
- Use dataclasses for internal metadata, type hints on every new function, specific exceptions, signed commits, and focused tests before each implementation step.
- GPU jobs use `/home` for source, `/raid/scratch` for environments and caches, and `/lustre` only for containers, checkpoints, and durable logs. Run `--test-only`, commit and push, then submit and monitor once per minute for five minutes.

---

### Task 1: Pass the MXFP8 Recipe to Megatron's Optimizer

**Files:**
- Modify: `nemo_rl/models/megatron/setup.py:1392-1396`
- Test: `tests/unit/models/megatron/test_megatron_setup.py`

**Interfaces:**
- Consumes: `config["megatron_cfg"]["fp8_cfg"]` with `enabled`, `fp8_recipe`, and `fp8_param`.
- Produces: `ConfigContainer.optimizer.fp8_recipe == "mxfp8"` whenever FP8 is enabled with that recipe.

- [ ] **Step 1: Write the failing optimizer-plumbing test**

Add a focused `@pytest.mark.mcore` test beside `TestCreateMegatronConfigOptimizerOffload`. Patch the sibling config builders exactly as that class does and inspect the real call arguments passed to `OptimizerConfig` and `DistributedDataParallelConfig`:

```python
@pytest.mark.mcore
def test_create_megatron_config_passes_mxfp8_recipe_to_optimizer() -> None:
    from nemo_rl.models.megatron.setup import _create_megatron_config

    config = {
        "megatron_cfg": {
            "optimizer": {"use_distributed_optimizer": True},
            "scheduler": {},
            "distributed_data_parallel_config": {
                "overlap_param_gather": True,
                "grad_reduce_in_fp32": True,
                "overlap_grad_reduce": True,
                "data_parallel_sharding_strategy": "optim_grads_params",
            },
            "fp8_cfg": {
                "enabled": True,
                "fp8": "e4m3",
                "fp8_recipe": "mxfp8",
                "fp8_param": True,
            },
            "train_iters": 10,
        },
        "train_global_batch_size": 8,
    }
    with (
        patch("nemo_rl.models.megatron.setup.ConfigContainer"),
        patch("nemo_rl.models.megatron.setup.TrainingConfig"),
        patch("nemo_rl.models.megatron.setup.OptimizerConfig") as optimizer,
        patch("nemo_rl.models.megatron.setup.DistributedDataParallelConfig") as ddp,
        patch("nemo_rl.models.megatron.setup.SchedulerConfig"),
        patch("nemo_rl.models.megatron.setup.TokenizerConfig"),
        patch("nemo_rl.models.megatron.setup.LoggerConfig"),
    ):
        _create_megatron_config(
            model_cfg=MagicMock(),
            checkpoint_config=MagicMock(),
            config=config,
            hf_model_name="test-model",
            dtype=torch.bfloat16,
            fp8_param_enabled=True,
        )

    assert optimizer.call_args.kwargs["fp8_recipe"] == "mxfp8"
    assert ddp.call_args.kwargs["fp8_param_gather"] is True
    assert ddp.call_args.kwargs["reuse_grad_buf_for_mxfp8_param_ag"] is True
```

- [ ] **Step 2: Run the focused test and verify RED**

Run:

```bash
uv run pytest -q tests/unit/models/megatron/test_megatron_setup.py -k passes_mxfp8_recipe
```

Expected: FAIL because the `OptimizerConfig` call has no `fp8_recipe` keyword.

- [ ] **Step 3: Add the minimal optimizer argument**

Extend `optimizer_kwargs` only when FP8 is enabled:

```python
if fp8_cfg is not None and fp8_cfg.get("enabled", False):
    optimizer_kwargs["fp8_recipe"] = fp8_cfg.get("fp8_recipe")
```

Do not add a YAML key. The value already has one source of truth in `fp8_cfg`.

- [ ] **Step 4: Run the setup tests and verify GREEN**

Run:

```bash
uv run pytest -q tests/unit/models/megatron/test_megatron_setup.py -k 'fp8 or optimizer'
```

Expected: PASS.

- [ ] **Step 5: Commit the optimizer fix**

```bash
git add nemo_rl/models/megatron/setup.py tests/unit/models/megatron/test_megatron_setup.py
git commit -s -m "fix(megatron): pass MXFP8 recipe to optimizer"
```

### Task 2: Add the Version-Neutral Refit Component Contract

**Files:**
- Create: `nemo_rl/weight_sync/refit_components.py`
- Modify: `pyrefly.toml`
- Modify: `nemo_rl/weight_sync/nccl_reshard_utils.py:67-118`
- Modify: `nemo_rl/weight_sync/nccl_reshard_utils.py:260-330`
- Modify: `nemo_rl/weight_sync/nccl_reshard_utils.py:757-900`
- Test: `tests/unit/weight_sync/test_refit_components.py`
- Test: `tests/unit/weight_sync/test_nccl_reshard_utils.py`

**Interfaces:**
- Produces: `RefitComponentMeta`, `normalize_refit_components()`, `component_plan_digest()`, and role-aware `HFToLocalParamMap.get()`.
- Preserves: string-keyed `HFToLocalParamMap` callers as implicit `(name, "weight")` entries.

- [ ] **Step 1: Write failing component normalization tests**

Create `tests/unit/weight_sync/test_refit_components.py` with these cases:

```python
def test_legacy_weight_becomes_one_component() -> None:
    components = normalize_refit_components(
        "model.layers.0.mlp.down_proj.weight",
        {"shape": [64, 256], "dtype": "torch.bfloat16"},
    )
    assert [(c.role, c.global_shape, c.dtype) for c in components] == [
        ("weight", (64, 256), "torch.bfloat16")
    ]


def test_native_mxfp8_requires_ordered_value_and_scale() -> None:
    components = normalize_refit_components(
        "model.layers.0.mlp.down_proj.weight",
        {
            "shape": [64, 256],
            "dtype": "torch.float8_e4m3fn",
            "components": [
                {"role": "weight", "shape": [64, 256], "dtype": "torch.float8_e4m3fn"},
                {"role": "weight_scale", "shape": [64, 8], "dtype": "torch.uint8"},
            ],
        },
    )
    assert [c.role for c in components] == ["weight", "weight_scale"]
```

Also test duplicate roles, missing `weight`, non-uint8 scales, `K % 32 != 0`, and a scale shape other than `[..., K / 32]`.

- [ ] **Step 2: Run the new test and verify RED**

Run:

```bash
uv run pytest -q tests/unit/weight_sync/test_refit_components.py
```

Expected: collection fails because `refit_components.py` does not exist.

- [ ] **Step 3: Implement immutable component metadata**

Create the internal dataclass and normalization entrypoint:

```python
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

RefitComponentRole = Literal["weight", "weight_scale"]


@dataclass(frozen=True)
class RefitComponentMeta:
    logical_name: str
    checkpoint_name: str
    role: RefitComponentRole
    dtype: str
    global_shape: tuple[int, ...]
    src_placements: tuple[Any, ...] = ()
    dst_placements: tuple[Any, ...] = ()

    def to_wire(self) -> dict[str, Any]:
        return {
            "logical_name": self.logical_name,
            "checkpoint_name": self.checkpoint_name,
            "role": self.role,
            "dtype": self.dtype,
            "global_shape": list(self.global_shape),
            "src_placements": list(self.src_placements),
            "dst_placements": list(self.dst_placements),
        }


def normalize_refit_components(
    logical_name: str,
    metadata: Mapping[str, Any],
) -> tuple[RefitComponentMeta, ...]:
    logical_shape = _positive_shape(metadata["shape"], logical_name)
    serialized = metadata.get("components")
    if serialized is None:
        serialized = [
            {"role": "weight", "shape": logical_shape, "dtype": metadata["dtype"]}
        ]
    if not isinstance(serialized, Sequence) or not serialized:
        raise ValueError(f"{logical_name} components must not be empty")

    result: list[RefitComponentMeta] = []
    roles: set[str] = set()
    for item in serialized:
        role = item["role"]
        if role not in ("weight", "weight_scale"):
            raise ValueError(f"{logical_name} has unsupported component role {role!r}")
        if role in roles:
            raise ValueError(f"{logical_name} has duplicate component role {role!r}")
        roles.add(role)
        shape = _positive_shape(item["shape"], f"{logical_name} {role}")
        dtype = str(item["dtype"])
        result.append(
            RefitComponentMeta(
                logical_name=logical_name,
                checkpoint_name=(
                    logical_name if role == "weight" else f"{logical_name}_scale"
                ),
                role=role,
                dtype=dtype,
                global_shape=shape,
            )
        )

    if "weight" not in roles:
        raise ValueError(f"{logical_name} components must include 'weight'")
    _validate_weight_scale_pair(logical_name, logical_shape, result)
    return tuple(result)
```

Implement `_positive_shape()` to reject booleans, zero, negative, and non-integer dimensions. Implement `_validate_weight_scale_pair()` to require the `weight` shape to equal the logical shape and, when a scale is present, require uint8 dtype and exact `[..., K / 32]` shape. The implementation accepts no component roles except `weight` and `weight_scale`. The scale checkpoint name is a canonical wire name, not a vLLM parameter suffix.

- [ ] **Step 4: Add failing role-aware map compatibility tests**

Add to `tests/unit/weight_sync/test_nccl_reshard_utils.py`:

```python
def test_hf_to_local_param_map_normalizes_legacy_keys() -> None:
    spec = LocalParamSpec(base=torch.empty(1))
    mapping = HFToLocalParamMap(specs={"x.weight": spec})
    assert mapping.get("x.weight") is spec
    assert mapping.get("x.weight", role="weight") is spec
    assert mapping.get("x.weight", role="weight_scale") is None
```

- [ ] **Step 5: Implement backward-compatible role lookup**

```python
@dataclass
class HFToLocalParamMap:
    specs: dict[str | tuple[str, str], LocalParamSpec] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.specs = {
            (key, "weight") if isinstance(key, str) else key: value
            for key, value in self.specs.items()
        }

    def get(
        self,
        hf_name: str,
        default: LocalParamSpec | None = None,
        *,
        role: str = "weight",
    ) -> LocalParamSpec | None:
        return self.specs.get((hf_name, role), default)
```

- [ ] **Step 6: Write a failing deterministic digest test**

```python
def test_component_plan_digest_is_stable_and_order_sensitive() -> None:
    first = _native_refit_info()
    second = copy.deepcopy(first)
    assert component_plan_digest(first) == component_plan_digest(second)
    second["per_layer_params"]["model.layers.0"][0]["components"].reverse()
    assert component_plan_digest(first) != component_plan_digest(second)
```

- [ ] **Step 7: Implement canonical digest serialization**

`component_plan_digest(refit_info)` must serialize only logical names, component order, roles, dtypes, global shapes, normalized placement kinds/dimensions, PP stage, and expert grouping. Build a `canonical_plan` list in `layer_names` and parameter order, then use `json.dumps(canonical_plan, sort_keys=True, separators=(",", ":"))` and SHA-256. Do not include tensor values, object ids, device ids, or runtime pointers.

- [ ] **Step 8: Attach component placement metadata to every bulk parameter**

Update `build_nccl_reshard_refit_info()` so legacy entries receive one implicit component and native entries preserve two ordered components. Compute placements from the parent logical weight name, validate every `Shard.dim` against the component rank, and prepend the grouped expert dimension to both value and scale shapes.

- [ ] **Step 9: Run contract tests and commit**

```bash
uv run pytest -q \
  tests/unit/weight_sync/test_refit_components.py \
  tests/unit/weight_sync/test_nccl_reshard_utils.py
git add \
  nemo_rl/weight_sync/refit_components.py \
  nemo_rl/weight_sync/nccl_reshard_utils.py \
  pyrefly.toml \
  tests/unit/weight_sync/test_refit_components.py \
  tests/unit/weight_sync/test_nccl_reshard_utils.py
git commit -s -m "refactor(refit): add ordered weight components"
```

### Task 3: Validate the Plan Before Creating NCCL Communicators

**Files:**
- Modify: `nemo_rl/weight_sync/nccl_reshard_utils.py:260-330`
- Modify: `nemo_rl/weight_sync/nccl_reshard_weight_synchronizer.py:138-218`
- Test: `tests/unit/weight_sync/test_weight_synchronizer.py`
- Test: `tests/unit/weight_sync/test_nccl_reshard_utils.py`

**Interfaces:**
- Consumes: `component_plan_digest(refit_info)` from Task 2.
- Produces: `refit_info["plan_digest"]` generated on the policy side and recomputed on the generation side before any communicator is initialized.

- [ ] **Step 1: Write a failing initialization-order test**

Use the existing `TestNcclReshardWeightSynchronizer` fakes and record calls:

```python
def test_init_validates_refit_plan_before_collectives(self) -> None:
    events: list[str] = []
    sync = self._make_sync(events=events)

    sync.init_communicator()

    assert events.index("policy.prepare_refit_info") < events.index("policy.init_collective")
    assert events.index("generation.prepare_refit_info") < events.index("generation.init_collective")
```

- [ ] **Step 2: Run the ordering test and verify RED**

Run:

```bash
uv run pytest -q tests/unit/weight_sync/test_weight_synchronizer.py -k validates_refit_plan_before_collectives
```

Expected: FAIL because current initialization creates both communicator families before preparing metadata.

- [ ] **Step 3: Reorder synchronizer initialization**

Make `init_communicator()` perform these phases in order:

```text
1. policy.prepare_nccl_reshard_refit_info(train_parallelism, gen_parallelism, train_world_size, inference_world_size)
2. compute and attach plan_digest
3. generation.prepare_nccl_reshard_refit_info(wire_info)
4. initialize model_update_group
5. initialize per-PP-stage NCCL Reshard groups
```

No rank may enter `init_collective()` or `init_nccl_reshard_comm_group()` before both metadata calls return successfully.

- [ ] **Step 4: Write failing digest-tamper tests**

```python
def test_restore_refit_info_rejects_plan_digest_mismatch() -> None:
    info = _native_refit_info()
    info["plan_digest"] = component_plan_digest(info)
    info["per_layer_params"]["model.layers.0"][0]["components"][1]["dtype"] = "torch.float16"
    with pytest.raises(ValueError, match="refit plan digest mismatch"):
        restore_refit_info_placements(info)
```

- [ ] **Step 5: Implement generation-side recomputation**

`restore_refit_info_placements()` must retain the received digest, restore placements, recompute the canonical digest, and raise a `ValueError` containing both digests when they differ.

- [ ] **Step 6: Run synchronizer tests and commit**

```bash
uv run pytest -q \
  tests/unit/weight_sync/test_weight_synchronizer.py -k NcclReshard \
  tests/unit/weight_sync/test_nccl_reshard_utils.py
git add \
  nemo_rl/weight_sync/nccl_reshard_utils.py \
  nemo_rl/weight_sync/nccl_reshard_weight_synchronizer.py \
  tests/unit/weight_sync/test_weight_synchronizer.py \
  tests/unit/weight_sync/test_nccl_reshard_utils.py
git commit -s -m "fix(refit): validate plans before NCCL setup"
```

### Task 4: Extract Canonical Native MXFP8 Storage

**Files:**
- Create: `nemo_rl/models/policy/workers/mxfp8_refit_source.py`
- Modify: `pyrefly.toml`
- Test: `tests/unit/models/policy/test_mxfp8_refit_source.py`

**Interfaces:**
- Consumes: a Transformer Engine MXFP8 tensor with `shape` and `get_metadata()`.
- Produces: `NativeMXFP8Components(weight: torch.Tensor, weight_scale: torch.Tensor)` through `extract_native_mxfp8_components(tensor: Any)`.

- [ ] **Step 1: Write failing value, scale, and padding tests**

```python
def test_extract_native_mxfp8_components_crops_scale_padding() -> None:
    source = FakeMXFP8Tensor(
        shape=(3, 64),
        rowwise_data=torch.arange(3 * 64, dtype=torch.uint8).reshape(3, 64),
        rowwise_scale_inv=torch.arange(5 * 4, dtype=torch.uint8).reshape(5, 4),
        with_gemm_swizzled_scales=False,
    )

    result = extract_native_mxfp8_components(source)

    assert result.weight.dtype == torch.float8_e4m3fn
    assert torch.equal(result.weight.view(torch.uint8), source.rowwise_data)
    assert result.weight_scale.shape == (3, 2)
    assert torch.equal(result.weight_scale, source.rowwise_scale_inv[:3, :2])
```

Add cases for grouped shapes `(2, 3, 64)`, missing metadata, non-tensor values, wrong dtypes, undersized storage, unaligned K, and `with_gemm_swizzled_scales=True`.

- [ ] **Step 2: Run the extraction tests and verify RED**

Run:

```bash
uv run pytest -q tests/unit/models/policy/test_mxfp8_refit_source.py
```

Expected: collection fails because the module does not exist.

- [ ] **Step 3: Implement byte-preserving extraction**

```python
@dataclass(frozen=True)
class NativeMXFP8Components:
    weight: torch.Tensor
    weight_scale: torch.Tensor


def extract_native_mxfp8_components(tensor: Any) -> NativeMXFP8Components:
    metadata = tensor.get_metadata()
    if metadata.get("with_gemm_swizzled_scales"):
        raise ValueError("Native MXFP8 refit requires compact rowwise scales")
    shape = tuple(int(size) for size in tensor.shape)
    if not shape or shape[-1] % 32:
        raise ValueError(f"Native MXFP8 refit requires K divisible by 32; got {shape}")
    rows = math.prod(shape[:-1])
    columns = shape[-1] // 32
    data = metadata.get("rowwise_data")
    scale = metadata.get("rowwise_scale_inv")
    _validate_native_storage(data, scale, rows, shape[-1], columns)
    return NativeMXFP8Components(
        weight=data.reshape(shape).view(torch.float8_e4m3fn),
        weight_scale=scale[:rows, :columns].reshape(*shape[:-1], columns),
    )
```

`_validate_native_storage()` must require exact value-byte count and at least the logical scale rectangle. It must not call `.contiguous()`, transpose, quantize, or dequantize.

- [ ] **Step 4: Run tests and commit**

```bash
uv run pytest -q tests/unit/models/policy/test_mxfp8_refit_source.py
git add \
  nemo_rl/models/policy/workers/mxfp8_refit_source.py \
  pyrefly.toml \
  tests/unit/models/policy/test_mxfp8_refit_source.py
git commit -s -m "feat(refit): extract native MXFP8 components"
```

### Task 5: Build Megatron Native MXFP8 Source Components

**Files:**
- Modify: `nemo_rl/models/policy/workers/megatron_policy_worker.py:2285-2315`
- Modify: `nemo_rl/models/policy/workers/megatron_policy_worker.py:2806-3105`
- Test: `tests/unit/models/policy/test_megatron_worker.py`
- Test: `tests/unit/models/megatron/test_group_experts.py`

**Interfaces:**
- Consumes: Tasks 2 and 4.
- Produces: `_is_native_mxfp8_export()`, `_build_native_mxfp8_shape_metadata()`, `_iter_local_native_mxfp8_param_components()`, and source specs keyed by `(logical_name, role)`.

- [ ] **Step 1: Write failing native-path selection tests**

```python
@pytest.mark.parametrize(
    ("fp8_param", "recipe", "expected"),
    [(True, "mxfp8", True), (False, "mxfp8", False), (True, "blockwise", False)],
)
def test_native_mxfp8_export_selection(fp8_param, recipe, expected) -> None:
    worker = _worker_with_fp8_cfg(fp8_param=fp8_param, fp8_recipe=recipe)
    assert worker._is_native_mxfp8_export() is expected
```

- [ ] **Step 2: Run the selection test and verify RED**

Run:

```bash
uv run pytest -q tests/unit/models/policy/test_megatron_worker.py -k native_mxfp8_export_selection
```

Expected: FAIL because `_is_native_mxfp8_export()` does not exist.

- [ ] **Step 3: Implement exact source-path selection**

```python
def _is_native_mxfp8_export(self) -> bool:
    return bool(
        self.fp8_cfg is not None
        and self.fp8_cfg.get("fp8_param", False)
        and self.fp8_cfg.get("fp8_recipe") == "mxfp8"
        and self.cfg["generation"]["vllm_cfg"]["precision"] == "fp8"
        and self.cfg["generation"]["vllm_cfg"].get("is_mx") is True
    )
```

Keep `_is_fp8_export()` blockwise-only.

- [ ] **Step 4: Write failing fused FC1/FC2 component tests**

Use fake `GatedMLPMapping`, `FusedGatedExpertMapping`, and `FusedExpertMapping` tasks. Assert that FC1 gate/up split the value and scale on the same output dimension, FC2 stays direct, and each logical name exposes both roles.

```python
source_map = worker.build_hf_to_local_param_map(refit_info)
assert source_map.get(gate_name, role="weight").base.shape == (intermediate, hidden)
assert source_map.get(gate_name, role="weight_scale").base.shape == (
    intermediate,
    hidden // 32,
)
```

- [ ] **Step 5: Implement component iteration and metadata**

Add `_iter_local_native_mxfp8_param_components(self) -> Iterator[tuple[str, str, torch.Tensor]]`. Its control flow is fixed:

```text
for each local conversion task in bridge order:
  skip PP placeholders whose param_weight is None
  extract canonical value and scale from param_weight
  GatedMLPMapping or FusedGatedExpertMapping:
    split value on output dimension -2 into gate and up
    split scale on the same dimension into gate and up
    emit (gate_name, weight), (gate_name, weight_scale),
         (up_name, weight), and (up_name, weight_scale)
  FusedExpertMapping:
    emit (down_name, weight) and (down_name, weight_scale)
  reject any bulk FFN mapping type outside those three classes
```

Add `_build_native_mxfp8_shape_metadata(self, train_parallelism: dict[str, int]) -> OrderedDict[str, dict[str, Any]]`. For each emitted logical projection, reconstruct its global shape from the local task shape and TP/EP sizes, then attach exactly these descriptors:

```python
{
    "shape": global_shape,
    "dtype": "torch.float8_e4m3fn",
    "components": [
        {"role": "weight", "shape": global_shape, "dtype": "torch.float8_e4m3fn"},
        {
            "role": "weight_scale",
            "shape": [*global_shape[:-1], global_shape[-1] // 32],
            "dtype": "torch.uint8",
        },
    ],
}
```

The metadata function must not materialize full tensors. It emits only FFN/routed-expert entries.

- [ ] **Step 6: Write failing repeated grouped-expert tests**

For `moe_single_grouped_weight=True`, fake `get_grouped_quantized_members(grouped_param, create_if_missing=False)`. Call `spec.pre()` twice after replacing member metadata and assert that the second result uses the new bytes and a new temporary tensor.

```python
first = gate_spec.pre(gate_spec.base).buf
members[0]._metadata["rowwise_data"] = replacement
second = gate_spec.pre(gate_spec.base).buf
assert first.data_ptr() != second.data_ptr()
assert torch.equal(second[0].view(torch.uint8), replacement[:half])
```

- [ ] **Step 7: Implement grouped source refresh**

Add `_build_native_grouped_mxfp8_tasks(self) -> list[Any]`. It must use Bridge's mapping registry to create a `WeightConversionTask` for each globally named grouped FC1/FC2 parameter, retaining `param_weight=None` for parameters owned by another PP stage.

Add `_materialize_native_grouped_component(self, task: Any, projection: str, role: str) -> torch.Tensor` with this flow:

```text
members = get_grouped_quantized_members(task.param_weight, create_if_missing=False)
for member in numeric local-expert order:
  components = extract_native_mxfp8_components(member)
  choose components.weight or components.weight_scale from role
  for FC1, split the chosen tensor on output dimension -2 and select gate or up
  for FC2, keep the chosen tensor whole
stack the selected tensors on expert dimension 0
```

Never pass the grouped container to `extract_native_mxfp8_components()`.

- [ ] **Step 8: Transfer every source component**

In policy `nccl_reshard_refit()`, replace the one-transfer-per-parameter loop with ordered component iteration:

```python
for component in param_info["components"]:
    role = component["role"]
    spec = self.hf_to_local_param_map.get(param_info["name"], role=role)
    if spec is None:
        raise RuntimeError(f"missing {role!r} source for {param_info['name']!r}")
    ctx = spec.pre(spec.base) if spec.pre is not None else RefitCtx(buf=spec.base)
    src_tensor = DTensorRef(ctx.buf, component["global_shape"])
    xferdtensor(
        src_tensor,
        param_info["src_mesh_info"],
        component["src_placements"],
        None,
        param_info["dst_mesh_info"],
        component["dst_placements"],
        group,
        nccl_reshard_stream,
    )
```

- [ ] **Step 9: Run source tests and commit**

```bash
uv run pytest -q \
  tests/unit/models/policy/test_mxfp8_refit_source.py \
  tests/unit/models/policy/test_megatron_worker.py -k 'native_mxfp8 or nccl_reshard' \
  tests/unit/models/megatron/test_group_experts.py -k native_mxfp8 \
  tests/unit/weight_sync/test_nccl_reshard_utils.py
git add \
  nemo_rl/models/policy/workers/megatron_policy_worker.py \
  tests/unit/models/policy/test_megatron_worker.py \
  tests/unit/models/megatron/test_group_experts.py
git commit -s -m "feat(refit): expose native MXFP8 source components"
```

### Task 6: Add a vLLM 0.25.1 Refit Adapter

**Files:**
- Create: `nemo_rl/models/generation/vllm/refit_adapter.py`
- Modify: `pyrefly.toml`
- Test: `tests/unit/models/generation/test_vllm_refit_adapter.py`

**Interfaces:**
- Consumes: a vLLM model runner, model config, device, and component-aware refit plan.
- Produces: `VllmRefitAdapter`, `Vllm0251RefitAdapter`, `VllmRefitCapabilities`, and `create_vllm_refit_adapter()`.

- [ ] **Step 1: Write failing capability and lifecycle tests**

```python
def test_0251_adapter_owns_one_reload_lifecycle(monkeypatch) -> None:
    events: list[str] = []
    adapter = _make_adapter(monkeypatch, events)

    adapter.prepare(_native_refit_info())
    adapter.begin_update()
    adapter.finish_update()

    assert events == ["enter_config", "initialize", "finalize", "exit_config"]


def test_adapter_fails_closed_after_abort(monkeypatch) -> None:
    adapter = _make_adapter(monkeypatch, [])
    error = RuntimeError("receive failed")
    adapter.begin_update()
    adapter.abort_update(error)
    with pytest.raises(RuntimeError, match="worker is unusable"):
        adapter.begin_update()
```

- [ ] **Step 2: Run adapter tests and verify RED**

Run:

```bash
uv run pytest -q tests/unit/models/generation/test_vllm_refit_adapter.py
```

Expected: collection fails because `refit_adapter.py` does not exist.

- [ ] **Step 3: Define the adapter protocol and capability record**

```python
@dataclass(frozen=True)
class VllmRefitCapabilities:
    layerwise_reload: bool
    weight_transfer_engine_registry: bool
    trainer_weight_transfer: bool


@runtime_checkable
class VllmRefitAdapter(Protocol):
    def validate_plan(self, refit_info: Mapping[str, Any]) -> None:
        raise NotImplementedError

    def prepare(self, refit_info: Mapping[str, Any]) -> None:
        raise NotImplementedError

    def begin_update(self) -> None:
        raise NotImplementedError

    def resolve_destination(
        self,
        logical_name: str,
        role: str,
        component: Mapping[str, Any],
        value_param: torch.Tensor,
        merged_slice: tuple[slice, ...] | None,
    ) -> LocalParamSpec:
        raise NotImplementedError

    def finish_update(self) -> None:
        raise NotImplementedError

    def abort_update(self, error: BaseException) -> None:
        raise NotImplementedError
```

- [ ] **Step 4: Implement lazy 0.25.1 lifecycle ownership**

`Vllm0251RefitAdapter.begin_update()` must lazily import and enter `set_current_vllm_config`, then call `initialize_layerwise_reload(model)`. `finish_update()` must call `finalize_layerwise_reload(model, model_config)` exactly once and exit the config context. `abort_update()` must exit the context, record the original exception, and make later updates fail immediately.

No vLLM reload symbol may be imported at module import time.

- [ ] **Step 5: Add a non-blocking vLLM 0.28 capability probe test**

Mock the v0.28 modules and assert capability detection records the newer engine API without selecting it for the pinned runtime:

```python
assert capabilities.layerwise_reload is True
assert capabilities.weight_transfer_engine_registry is True
assert capabilities.trainer_weight_transfer is True
assert isinstance(
    create_vllm_refit_adapter(
        model_runner=fake_runner,
        model_config=fake_model_config,
        device=torch.device("cuda"),
    ),
    Vllm0251RefitAdapter,
)
```

Selection must use required attributes and call signatures. Do not compare parsed version strings inside production code.

- [ ] **Step 6: Run adapter tests and commit**

```bash
uv run pytest -q tests/unit/models/generation/test_vllm_refit_adapter.py
git add \
  nemo_rl/models/generation/vllm/refit_adapter.py \
  pyrefly.toml \
  tests/unit/models/generation/test_vllm_refit_adapter.py
git commit -s -m "refactor(vllm): isolate native refit lifecycle"
```

### Task 7: Bind Canonical MXFP8 Destinations Inside the Adapter

**Files:**
- Modify: `nemo_rl/models/generation/vllm/refit_adapter.py`
- Modify: `nemo_rl/models/generation/vllm/vllm_backend.py:1011-1409`
- Test: `tests/unit/models/generation/test_vllm_refit_adapter.py`
- Test: `tests/unit/models/generation/test_nccl_reshard_backend.py`

**Interfaces:**
- Consumes: Task 2 component metadata and Task 6 lifecycle.
- Produces: direct checkpoint-format value and scale destinations, with the full generation refit enclosed by one adapter lifecycle.

- [ ] **Step 1: Write failing dense and routed-MoE destination tests**

Cover dense `gate_up_proj`/`down_proj` and grouped `w13`/`w2` at TP2. Assert exact pointers, shapes, and dtypes for both roles:

```python
mapping = backend.build_hf_to_local_param_map(_native_refit_info())
value_spec = mapping.get(hf_name, role="weight")
scale_spec = mapping.get(hf_name, role="weight_scale")
assert value_spec is not None and scale_spec is not None
assert value_spec.base.dtype == torch.float8_e4m3fn
assert scale_spec.base.dtype == torch.uint8
```

Monkeypatch `quantize_mxfp8_weight` to raise so the test proves the native path never receiver-quantizes. Also inspect the native transfer specs and assert that no BF16 staging tensor or BF16 transfer component is allocated.

- [ ] **Step 2: Run destination tests and verify RED**

Run:

```bash
uv run pytest -q tests/unit/models/generation/test_nccl_reshard_backend.py -k native_mxfp8
```

Expected: FAIL because generation has no role-specific destination specs.

- [ ] **Step 3: Implement vLLM 0.25.1 checkpoint-name resolution**

Inside `Vllm0251RefitAdapter.resolve_destination()`:

```python
if role == "weight":
    target = value_param
elif role == "weight_scale":
    value_name = self._parameter_names_by_id[id(value_param)]
    target_name = f"{value_name}_scale_from_checkpoint"
    target = self._parameters.get(target_name)
    if target is None:
        raise ValueError(
            f"MXFP8 checkpoint scale {target_name!r} for {logical_name!r} is missing"
        )
else:
    raise ValueError(f"unsupported refit component role {role!r}")
```

For fused gate/up and w13 targets, apply the same output-axis region to value and scale. Validate local shape from the component destination placements before constructing `LocalParamSpec`.

- [ ] **Step 4: Write a failing lifecycle-order test around the complete receive**

```python
def test_native_nccl_reshard_begins_reload_before_resolving_destinations() -> None:
    events: list[str] = []
    backend = _backend_with_recording_adapter(events)
    backend.nccl_reshard_refit()
    assert events[0] == "begin_update"
    assert events.index("begin_update") < events.index("resolve_destination")
    assert events.index("bulk_receive") < events.index("misc_receive")
    assert events[-1] == "finish_update"
    assert events.count("finish_update") == 1
```

Record the runtime kernel tensor pointers before the update, execute two native updates with different source bytes, and assert that finalization keeps both runtime pointers stable while changing their values:

```python
value_ptr = runtime_value.data_ptr()
scale_ptr = runtime_scale.data_ptr()
backend.nccl_reshard_refit()
first_value = runtime_value.clone()
backend.nccl_reshard_refit()
assert runtime_value.data_ptr() == value_ptr
assert runtime_scale.data_ptr() == scale_ptr
assert not torch.equal(runtime_value, first_value)
```

This is the CUDA Graph safety contract. A native update may restore temporary checkpoint-format views, but `finish_update()` must leave the tensors used by captured kernels at their original addresses.

- [ ] **Step 5: Enclose bulk and misc transfer in one native lifecycle**

Restructure generation `nccl_reshard_refit()` as:

```python
adapter = self._get_nccl_reshard_refit_adapter()
adapter.begin_update()
try:
    self.hf_to_local_param_map = self.build_hf_to_local_param_map(
        self.nccl_reshard_refit_info
    )
    self._receive_nccl_reshard_bulk_components()
    self._receive_and_load_misc_params()
    adapter.finish_update()
except BaseException as error:
    adapter.abort_update(error)
    raise
```

The destination map must be rebuilt after `begin_update()` because native initialization restores checkpoint-format storage and can replace runtime tensor views. Existing non-native paths may keep their cached map.

- [ ] **Step 6: Receive every ordered destination component**

Use `component["global_shape"]`, component placements, and `mapping.get(name, role=role)` for each transfer. Raise `RuntimeError` for a missing spec; never skip it.

- [ ] **Step 7: Preserve legacy paths with focused regression tests**

Run and retain assertions for:

```text
BF16 -> BF16 direct reshard
BF16 -> MXFP8 receiver quantization
blockwise FP8 -> matching blockwise FP8
merged dense gate/up
grouped routed-expert W13/W2
misc parameter loading
MTP exclusion
```

Add one mixed-scope test whose refit plan contains native routed-expert FC1/FC2 components and BF16 attention, embedding, shared-expert, router, output-head, and MTP entries. Assert that the adapter receives only the native value/scale pairs, the misc loader receives each BF16 entry exactly once, and no parameter appears in both paths.

- [ ] **Step 8: Run generation tests and commit**

```bash
uv run pytest -q \
  tests/unit/models/generation/test_vllm_refit_adapter.py \
  tests/unit/models/generation/test_nccl_reshard_backend.py \
  tests/unit/models/generation/test_vllm_backend.py -k 'refit or lifecycle' \
  tests/unit/models/generation/test_vllm_fp8_quantization.py
git add \
  nemo_rl/models/generation/vllm/refit_adapter.py \
  nemo_rl/models/generation/vllm/vllm_backend.py \
  tests/unit/models/generation/test_vllm_refit_adapter.py \
  tests/unit/models/generation/test_nccl_reshard_backend.py \
  tests/unit/models/generation/test_vllm_backend.py
git commit -s -m "feat(refit): load native MXFP8 components in vLLM"
```

### Task 8: Enable Only the Validated Native MXFP8 Configuration

**Files:**
- Modify: `nemo_rl/weight_sync/nccl_reshard_utils.py:562-755`
- Modify: `tests/unit/weight_sync/test_nccl_reshard_utils.py`
- Create: `experiments/native_mxfp8_source_refit/submit_oci_hsg.sh`
- Create: `experiments/native_mxfp8_source_refit/README.md`

**Interfaces:**
- Consumes: complete native source and destination path.
- Produces: setup-time acceptance for `fp8_param=true`, `fp8_recipe=mxfp8`, vLLM `precision=fp8`, and `is_mx=true`; every other unsupported pairing fails before job startup.

- [ ] **Step 1: Replace the old rejection test with an exact acceptance test**

```python
def test_check_nccl_reshard_accepts_native_mxfp8_pair() -> None:
    config = _valid_config()
    config["policy"]["megatron_cfg"]["fp8_cfg"].update(
        {"enabled": True, "fp8_param": True, "fp8_recipe": "mxfp8"}
    )
    config["policy"]["generation"]["vllm_cfg"].update(
        {"precision": "fp8", "is_mx": True}
    )
    check_nccl_reshard_refit_support(config)
```

Add rejection tests for native MXFP8-to-BF16, native MXFP8-to-blockwise, blockwise-to-MXFP8, ETP2, SGLang, and missing `is_mx`.

- [ ] **Step 2: Run validator tests and verify RED**

Run:

```bash
uv run pytest -q tests/unit/weight_sync/test_nccl_reshard_utils.py -k 'support and mxfp8'
```

Expected: the valid native pair is rejected by the current blockwise-only gate.

- [ ] **Step 3: Implement exact precision-pair validation**

Accept only this new branch:

```python
native_mxfp8 = bool(
    fp8_param
    and fp8_recipe == "mxfp8"
    and gen_precision == "fp8"
    and vllm_cfg.get("is_mx") is True
)
```

Do not broaden blockwise or BF16 routes.

- [ ] **Step 4: Add an explicit experiment switch**

Create `experiments/native_mxfp8_source_refit/submit_oci_hsg.sh` by adapting the validated launcher at `/Users/sna/MXFP8_generation/.worktrees/nemorl-mxfp8-e2e-fp8param-false/research/mxfp8_training_rl/submit_oci_hsg.sh`. Add `FP8_PARAM=true|false`. For `true`, set all of:

```text
policy.megatron_cfg.fp8_cfg.enabled=true
policy.megatron_cfg.fp8_cfg.fp8=e4m3
policy.megatron_cfg.fp8_cfg.fp8_recipe=mxfp8
policy.megatron_cfg.fp8_cfg.fp8_param=true
policy.megatron_cfg.distributed_data_parallel_config.overlap_param_gather=true
policy.megatron_cfg.distributed_data_parallel_config.overlap_grad_reduce=true
policy.generation.refit_transport=nccl_reshard
policy.generation.vllm_cfg.precision=fp8
policy.generation.vllm_cfg.is_mx=true
policy.generation.vllm_kwargs.moe_backend=flashinfer_trtllm
```

The launcher must key node-local environments by source commit, not run suffix, and must not create a cache tree on `/lustre`.

Do not expose `reuse_grad_buf_for_mxfp8_param_ag` as a launcher option. Task 1 derives it from the single `fp8_cfg` source of truth.

- [ ] **Step 5: Run validator tests and commit**

```bash
uv run pytest -q tests/unit/weight_sync/test_nccl_reshard_utils.py
git add \
  nemo_rl/weight_sync/nccl_reshard_utils.py \
  tests/unit/weight_sync/test_nccl_reshard_utils.py \
  experiments/native_mxfp8_source_refit/submit_oci_hsg.sh \
  experiments/native_mxfp8_source_refit/README.md
git commit -s -m "feat(refit): enable native MXFP8 NCCL Reshard"
```

### Task 9: Run the Full Local Validation Gate

**Files:**
- Modify only files required by failures attributable to Tasks 1-8.

**Interfaces:**
- Consumes: all implementation commits.
- Produces: a clean unit, type, lint, and compatibility gate before GPU submission.

- [ ] **Step 1: Run the complete focused suite**

```bash
uv run pytest -q \
  tests/unit/models/megatron/test_megatron_setup.py \
  tests/unit/models/policy/test_mxfp8_refit_source.py \
  tests/unit/models/policy/test_megatron_worker.py \
  tests/unit/models/megatron/test_group_experts.py \
  tests/unit/models/generation/test_vllm_refit_adapter.py \
  tests/unit/models/generation/test_nccl_reshard_backend.py \
  tests/unit/models/generation/test_vllm_backend.py \
  tests/unit/models/generation/test_vllm_fp8_quantization.py \
  tests/unit/weight_sync/test_refit_components.py \
  tests/unit/weight_sync/test_nccl_reshard_utils.py \
  tests/unit/weight_sync/test_weight_synchronizer.py
```

Expected: PASS.

- [ ] **Step 2: Run type and formatting checks on changed files**

```bash
uv run --group dev pyrefly check \
  nemo_rl/models/megatron/setup.py \
  nemo_rl/models/policy/workers/mxfp8_refit_source.py \
  nemo_rl/models/policy/workers/megatron_policy_worker.py \
  nemo_rl/models/generation/vllm/refit_adapter.py \
  nemo_rl/models/generation/vllm/vllm_backend.py \
  nemo_rl/weight_sync/refit_components.py \
  nemo_rl/weight_sync/nccl_reshard_utils.py \
  nemo_rl/weight_sync/nccl_reshard_weight_synchronizer.py
uv run pre-commit run --all-files
```

Expected: PASS with no skipped changed source file.

Verify that the version-neutral core does not call vLLM reload internals directly:

```bash
if rg -n 'process_weights_after_loading|initialize_layerwise_reload|finalize_layerwise_reload' \
  nemo_rl/weight_sync/refit_components.py \
  nemo_rl/models/policy/workers/mxfp8_refit_source.py; then
  echo "vLLM reload internals leaked into the version-neutral refit core" >&2
  exit 1
fi
```

- [ ] **Step 3: Run the pinned vLLM contract tests**

```bash
uv run python -c 'import vllm; assert vllm.__version__ == "0.25.1"'
uv run pytest -q tests/unit/models/generation/test_vllm_refit_adapter.py -k 0251
```

Expected: vLLM reports `0.25.1`; tests pass.

- [ ] **Step 4: Run the non-blocking vLLM 0.28 source contract probe**

Against the local vLLM checkout at `/Users/sna/MXFP8_generation/vllm-v0251-adaptive`:

```bash
git -C /Users/sna/MXFP8_generation/vllm-v0251-adaptive show \
  v0.28.0:vllm/model_executor/model_loader/reload/layerwise.py \
  | rg 'initialize_layerwise_reload|finalize_layerwise_reload'
git -C /Users/sna/MXFP8_generation/vllm-v0251-adaptive show \
  v0.28.0:vllm/distributed/weight_transfer/factory.py \
  | rg 'register_engine|TrainerWeightTransferEngine'
```

Expected: all four symbols are found. This records source compatibility only; it is not an end-to-end NeMo-RL v0.28 claim.

- [ ] **Step 5: Commit any test-only corrections**

If the gate required changes, stage only the affected implementation and test paths from this explicit set, then inspect the staged diff before committing:

```bash
git add -u -- \
  nemo_rl/models/megatron/setup.py \
  nemo_rl/models/policy/workers/mxfp8_refit_source.py \
  nemo_rl/models/policy/workers/megatron_policy_worker.py \
  nemo_rl/models/generation/vllm/refit_adapter.py \
  nemo_rl/models/generation/vllm/vllm_backend.py \
  nemo_rl/weight_sync/refit_components.py \
  nemo_rl/weight_sync/nccl_reshard_utils.py \
  nemo_rl/weight_sync/nccl_reshard_weight_synchronizer.py \
  pyrefly.toml \
  tests/unit/models/megatron/test_megatron_setup.py \
  tests/unit/models/megatron/test_group_experts.py \
  tests/unit/models/policy/test_mxfp8_refit_source.py \
  tests/unit/models/policy/test_megatron_worker.py \
  tests/unit/models/generation/test_vllm_refit_adapter.py \
  tests/unit/models/generation/test_nccl_reshard_backend.py \
  tests/unit/models/generation/test_vllm_backend.py \
  tests/unit/models/generation/test_vllm_fp8_quantization.py \
  tests/unit/weight_sync/test_refit_components.py \
  tests/unit/weight_sync/test_nccl_reshard_utils.py \
  tests/unit/weight_sync/test_weight_synchronizer.py
git diff --cached --check
git commit -s -m "test(refit): cover native MXFP8 lifecycle"
```

If no file changed, do not create an empty commit.

### Task 10: Validate Correctness and Performance on GB200

**Files:**
- Modify: `experiments/native_mxfp8_source_refit/README.md`
- Create: `experiments/native_mxfp8_source_refit/native_mxfp8_results.json`

**Interfaces:**
- Consumes: a committed and pushed implementation branch plus the Task 8 launcher.
- Produces: two-step correctness evidence and matched 20-step metrics for Qwen3-30B-A3B and Nemotron-3 Nano.

- [ ] **Step 1: Push the exact tested commit**

```bash
git status --short
git push -u fork HEAD
git rev-parse HEAD
```

Expected: clean status and one full commit SHA recorded in the experiment README.

- [ ] **Step 2: Check scheduling without allocating GPUs**

Run for both models:

```bash
MODEL=qwen30 FP8_PARAM=true MAX_STEPS=2 ACTION=test-only \
  ./experiments/native_mxfp8_source_refit/submit_oci_hsg.sh
MODEL=nano FP8_PARAM=true MAX_STEPS=2 ACTION=test-only \
  ./experiments/native_mxfp8_source_refit/submit_oci_hsg.sh
```

Expected: SLURM accepts the selected node/GPU/account shape.

- [ ] **Step 3: Submit two-step correctness smokes**

```bash
MODEL=qwen30 FP8_PARAM=true MAX_STEPS=2 ACTION=submit \
  ./experiments/native_mxfp8_source_refit/submit_oci_hsg.sh
MODEL=nano FP8_PARAM=true MAX_STEPS=2 ACTION=submit \
  ./experiments/native_mxfp8_source_refit/submit_oci_hsg.sh
```

Monitor with one `squeue --me` query per minute for five minutes. Inspect only the submitted jobs' logs. Do not run recursive scans on shared storage.

- [ ] **Step 4: Verify the native path from logs**

Each smoke must show:

```text
fp8_param=true
fp8_recipe=mxfp8
native components: weight=torch.float8_e4m3fn, weight_scale=torch.uint8
refit_transport=nccl_reshard
vLLM lifecycle: initialize once, finalize once per refit
no BF16 receiver quantization on native components
```

Both steps must finish with finite loss, reward, entropy, and generation KL error. A NaN or a refit-plan mismatch fails the gate.

- [ ] **Step 5: Run a value-changing repeated-refit check**

For Qwen, save the checksum of one routed-expert value and scale destination after refit 1, complete one optimizer update, refit again, and assert both checksums changed. Also compare one fixed prompt against a cold-load worker initialized from the same step-2 checkpoint.

- [ ] **Step 6: Submit matched 20-step runs**

Run `fp8_param=false` and `fp8_param=true` for each model from the same source commit and recipe:

```bash
MODEL=qwen30 FP8_PARAM=false MAX_STEPS=20 ACTION=submit ./experiments/native_mxfp8_source_refit/submit_oci_hsg.sh
MODEL=qwen30 FP8_PARAM=true  MAX_STEPS=20 ACTION=submit ./experiments/native_mxfp8_source_refit/submit_oci_hsg.sh
MODEL=nano   FP8_PARAM=false MAX_STEPS=20 ACTION=submit ./experiments/native_mxfp8_source_refit/submit_oci_hsg.sh
MODEL=nano   FP8_PARAM=true  MAX_STEPS=20 ACTION=submit ./experiments/native_mxfp8_source_refit/submit_oci_hsg.sh
```

- [ ] **Step 7: Record steady-state metrics**

Average steps 2 through 19 and write machine-readable results to `native_mxfp8_results.json`. The root object must contain `source_commit: str`, `gpu: "GB200"`, `step_window: [2, 19]`, and `runs: list[run]`. Every run object must contain these measured fields and no sentinel values:

| Field | Type |
|---|---|
| `model` | `str` |
| `fp8_param` | `bool` |
| `wandb_url` | non-empty `str` |
| `e2e_step_s` | positive `float` |
| `e2e_tokens_per_s_per_gpu` | positive `float` |
| `generation_s` | positive `float` |
| `generation_tokens_per_s_per_gpu` | positive `float` |
| `policy_training_s` | positive `float` |
| `policy_logprob_s` | positive `float` |
| `reference_logprob_s` | positive `float` |
| `refit_s` | positive `float` |
| `loss_mean` | finite `float` |
| `reward_mean` | finite `float` |
| `entropy_mean` | finite `float` |
| `generation_kl_error_mean` | finite `float` |

Add a small validation command to the experiment README that rejects missing fields, non-finite metrics, empty URLs, and a source commit that differs from the tested branch head.

- [ ] **Step 8: Run checkpoint-resume validation**

Enable checkpointing for a two-step Qwen run, resume for two more steps, and require one successful native refit after resume. Verify that optimizer masters load and that the model does not silently revert to BF16 parameter storage.

- [ ] **Step 9: Update the experiment report and commit evidence**

Add job IDs, W&B links, exact commit, config differences, correctness ranges, pointer-preservation evidence, and the step-2-through-19 table to `experiments/native_mxfp8_source_refit/README.md`.

```bash
git add \
  experiments/native_mxfp8_source_refit/README.md \
  experiments/native_mxfp8_source_refit/native_mxfp8_results.json
git commit -s -m "docs(refit): report native MXFP8 GB200 results"
git push
```

### Task 11: Final Review Gate

**Files:**
- Modify only files required by review findings.

**Interfaces:**
- Consumes: implementation, local validation, and GB200 evidence.
- Produces: a reviewable branch with no known correctness or compatibility blocker.

- [ ] **Step 1: Run NeMo-RL self-review**

Invoke the repository's `nemo-rl-pr-review` workflow against the complete branch. Review the diff from the pinned base, not the old prototype branch.

- [ ] **Step 2: Resolve every high-confidence finding**

For each accepted finding, add a regression test first, verify it fails, implement the fix, and rerun the focused suite. Record rejected findings with a concrete code or test reason.

- [ ] **Step 3: Re-run the final verification set**

```bash
uv run pytest -q \
  tests/unit/models/megatron/test_megatron_setup.py \
  tests/unit/models/policy/test_mxfp8_refit_source.py \
  tests/unit/models/policy/test_megatron_worker.py \
  tests/unit/models/megatron/test_group_experts.py \
  tests/unit/models/generation/test_vllm_refit_adapter.py \
  tests/unit/models/generation/test_nccl_reshard_backend.py \
  tests/unit/models/generation/test_vllm_backend.py \
  tests/unit/weight_sync/test_refit_components.py \
  tests/unit/weight_sync/test_nccl_reshard_utils.py \
  tests/unit/weight_sync/test_weight_synchronizer.py
uv run pre-commit run --all-files
git diff --check origin/main...HEAD
git status --short
```

Expected: all commands pass and the worktree is clean.

- [ ] **Step 4: Commit and push review corrections**

Use `git diff --name-only origin/main...HEAD` and `git status --short` to identify reviewed paths. Stage each accepted review fix by its exact path, run `git diff --cached --check`, then commit with `git commit -s -m "fix(refit): address native MXFP8 self-review"` and push.

Skip the commit when review produces no code changes.
