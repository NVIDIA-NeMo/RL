# CUDA Driver Compatibility

NeMo-RL containers ship with a specific CUDA toolkit version. Whether the container runs correctly on your cluster depends on the relationship between that toolkit version and the host's installed GPU driver. This guide explains how to verify compatibility and what to do when there's a mismatch.

## Quick Reference

| NeMo-RL Release | Container Base Image | CUDA Version | Minimum Host Driver |
|---|---|---|---|
| v0.5.0+ | `cuda-dl-base:25.05-cuda12.9-devel-ubuntu24.04` | 12.9 | ≥ 525.60.13 |

> **Rule of thumb:** If your host driver is **newer** than what the container expects, it will work. If it's **older**, read the [Forward Compatibility](#forward-compatibility-older-driver-newer-cuda) section carefully.

## How to Check Your Versions

```sh
# Host driver version and CUDA version it supports
nvidia-smi

# CUDA toolkit version inside the container
nvcc --version
```

## Key Concepts: KMD vs UMD

A CUDA installation has two layers:

| Layer | What It Is | Where It Lives |
|---|---|---|
| **Kernel Mode Driver (KMD)** | The GPU kernel module (`nvidia.ko`). Manages GPU hardware, memory (HMM), PCIe, NVLink, etc. | Host OS — shared by all containers |
| **User Mode Driver (UMD)** | Libraries like `libcuda.so`, the PTX JIT compiler, etc. These implement the CUDA runtime/driver API. | Inside the container (or host `/usr/lib`) |

When you run a container, the **KMD always comes from the host**. The **UMD comes from the container** (or from a forward-compat shim — see below). This split is central to understanding compatibility.

## Backward Compatibility (Newer Driver, Older CUDA)

**This is the common, safe case.** A newer host driver always supports older CUDA toolkit versions within the same or earlier major release family.

**Example:** Your cluster has driver 580 (CUDA 13 era), but the NeMo-RL container uses CUDA 12.9. This works out of the box — no extra packages or configuration needed.

NVIDIA guarantees backward compatibility across these ranges:

| CUDA Toolkit | Minimum Driver | Native Driver Range |
|---|---|---|
| CUDA 11.x | ≥ 450 | 450 – 524 |
| CUDA 12.x | ≥ 525 | 525 – 579 |
| CUDA 13.x | ≥ 580 | 580+ |

A driver from a later range (e.g., 580+ running CUDA 12.x) works via backward compatibility. **This is always the preferred setup.**

## Forward Compatibility (Older Driver, Newer CUDA)

Forward compatibility is the opposite: running a **newer** CUDA toolkit on an **older** host driver. This requires the `cuda-compat` package, which replaces the UMD libraries so the container's newer CUDA code can talk to the older KMD.

**The NeMo-RL container already includes `cuda-compat`.** The official NVIDIA CUDA base images install the corresponding `cuda-compat-<major>-<minor>` package (e.g., `cuda-compat-12-9` for CUDA 12.9) and configure the library path automatically. You do not need to install it yourself when using the container.

For bare-metal installations, the package can be installed manually:

```sh
# Example: install the CUDA 12.9 forward-compat package (bare metal only)
apt install cuda-compat-12-9
```

The package installs replacement UMD libraries into `/usr/local/cuda-X.Y/compat/`. On bare metal, you must ensure these are on `LD_LIBRARY_PATH` before the default system libraries.

### Limitations

Forward compatibility has significant constraints:

1. **Datacenter GPUs only** — forward compat packages only work on NVIDIA Data Center GPUs (e.g., A100, H100, H200, B200) and select NGC Server Ready RTX SKUs.
2. **UMD only — the KMD is not replaced.** The cuda-compat package shims the user-mode libraries, but your host's kernel-mode driver is still the old version. This means:
   - CUDA API calls work through the shimmed UMD.
   - **Kernel-level features** (HMM, GPU-Direct RDMA, NCCL's low-level transport paths) still depend on the host KMD.
   - **KMD bugs in the old driver cannot be worked around with cuda-compat.**
3. **No CUDA-OpenGL/Vulkan interop** when using forward compat.

### Forward Compatibility Matrix

| cuda-compat Package | Driver 535+ | 550+ | 570+ | 580+ |
|---|---|---|---|---|
| cuda-compat-13-0 | Compatible | Compatible | Compatible | N/A (native) |
| cuda-compat-12-8 | Compatible | Compatible | N/A (native) | N/A |
| cuda-compat-12-4 | Compatible | N/A (native) | N/A | N/A |

"N/A (native)" means the driver natively supports that CUDA version — no compat package needed.

## Driver Lifecycle: LTS vs Production vs New Feature Branches

NVIDIA datacenter (Tesla) drivers follow a branching model. This matters when deciding which driver to install on your cluster:

| Branch | Release Cadence | Support Duration | Use Case |
|---|---|---|---|
| **New Feature Branch (NFB)** | Quarterly | Short | Early access to new features |
| **Production Branch (PB)** | Semi-annually | 1 year | General production use |
| **Long Term Support Branch (LTSB)** | At least once per GPU architecture | 3 years | Clusters that cannot update drivers frequently |

The LTS distinction primarily matters for **forward compatibility**: if your cluster is pinned to an LTS driver and you need to run a newer CUDA toolkit, you'll rely on the cuda-compat package. The older your LTS driver, the more likely you'll hit KMD-level gaps (as described above).

See the [NVIDIA Driver Lifecycle documentation](https://docs.nvidia.com/datacenter/tesla/drivers/driver-lifecycle.html) for the current list of supported branches.

## Recommendations for NeMo-RL Users

1. **Prefer backward compatibility.** Upgrade your host driver to be at or above the container's native driver version. For NeMo-RL v0.5.0+ (CUDA 12.9), that means driver ≥ 575.51.03.
2. **Use forward compat only as a temporary bridge** — for example, while waiting for a cluster-wide driver upgrade to be validated.
3. **Check `nvidia-smi` before launching training.** Confirm your driver version meets the minimum for your container's CUDA version.
4. **If you see errors 803 or 804**, you have a forward-compat configuration issue:
   - `CUDA_ERROR_SYSTEM_DRIVER_MISMATCH (803)` — the cuda-compat UMD doesn't match the host KMD.
   - `CUDA_ERROR_COMPAT_NOT_SUPPORTED_ON_DEVICE (804)` — your GPU isn't supported by the forward-compat package (e.g., non-datacenter GPU).

## Further Reading

- [CUDA Compatibility — NVIDIA Docs](https://docs.nvidia.com/deploy/cuda-compatibility/)
- [CUDA Forward Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/forward-compatibility.html)
- [NVIDIA Driver Lifecycle](https://docs.nvidia.com/datacenter/tesla/drivers/driver-lifecycle.html)
