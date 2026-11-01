# Issue #2774 Megatron-Bridge Patch (NemotronH MTP Conversion)

This patch addresses the root cause behind:
- https://github.com/NVIDIA-NeMo/RL/issues/2774

## Why this file exists

The code fix lives in the `Megatron-Bridge` submodule, but this PR is intentionally scoped to the `nemo-rl` repository only.

To avoid publishing a submodule gitlink that points to an inaccessible commit, this PR provides a patch artifact plus apply instructions for maintainers with `Megatron-Bridge` write access.

## What is fixed

1. `mapping_registry.py` (`MegatronMappingRegistry._add_separate_layernorm_mappings`)
   - Preserve wrapper mappings (e.g. `_MTPFlatteningMapping`) by shallow cloning instead of `print+break`.
   - This prevents silent drops of MTP separate-LN aliases.

2. `nemotron_h_bridge.py` (`_MTPFlatteningQKVMapping.resolve`)
   - Support both capture shapes:
     - Megatron side: `(outer, inner)`
     - HF reverse-lookup side: `(flat)`
   - This fixes reverse lookup failures for `mtp.layers.*.mixer.{q,k,v}_proj.weight`.

## How to apply (in Megatron-Bridge repo)

From the `Megatron-Bridge` repository root:

```bash
git apply /path/to/nemo-rl/3rdparty/Megatron-Bridge-workspace/patches/issue-2774-mtp-conversion.patch
```

Or from a checked out `nemo-rl` repo where submodule is present:

```bash
cd RL/3rdparty/Megatron-Bridge-workspace/Megatron-Bridge
git apply ../patches/issue-2774-mtp-conversion.patch
```

## Validation summary

- Baseline (without fix): `Bug #1 'Unrecognized mapping type' hits = 48`
- With fix: `Bug #1 ... hits = 0`
- Bug #2 capture mismatch signal is eliminated in registry-level diagnostics and remains 0 in with-fix roundtrip logs.
