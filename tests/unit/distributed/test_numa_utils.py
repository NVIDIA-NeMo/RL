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
"""Tests for NUMA-aware CPU affinity and memory binding.

Unit tests run anywhere (no GPU needed). The benchmark test
(TestNUMABindingBenchmark) requires a GPU and libnuma and is
skipped otherwise — run it inside the RL container on a DGX/GB200
to validate that binding actually improves D2H performance.
"""

import os
import subprocess
import tempfile

import pytest

from nemo_rl.distributed.numa_utils import (
    _get_numa_node,
    _load_libnuma,
    _parse_cpulist,
    _set_numa_membind,
    bind_to_gpu_numa,
)

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _write_affinity_file_from_topo() -> str | None:
    """Parse nvidia-smi topo and write a GPU→cpulist affinity file.

    Mirrors the production probe in ray.sub: the "CPU Affinity" column is
    unreliable on GB200 (empty for GPUs not directly attached to a socket), so
    ray.sub reads the "NUMA Affinity" column and looks up the node-local CPU
    list from sysfs. We do the same here so the benchmark can't drift from
    production:

      nvidia-smi topo -m | awk '/^GPU[0-9]/ { numa=$(NF-1); ... }'
        → cat /sys/devices/system/node/node<numa>/cpulist

    nvidia-smi topo -m output format (GB200 NVL72 example):

            GPU0  ... NIC5  CPU Affinity  NUMA Affinity  GPU NUMA ID
        GPU0  X   ... SYS   0-71          0              N/A
        GPU2 ...      NODE   72-143        1              N/A
        NIC0 ...       X
        ...

    The header row is indented (starts with whitespace) and NIC rows start with
    "NIC", so startswith("GPU") skips both. On a GPU row the trailing
    whitespace-split columns are:
        parts[-3] = CPU Affinity  (unreliable on GB200 — NOT used)
        parts[-2] = NUMA Affinity (e.g. "0"/"1") ← used; matches ray.sub's $(NF-1)
        parts[-1] = GPU NUMA ID   (e.g. "N/A")

    The resulting affinity file has one line per GPU, e.g. "2:72-143".

    Returns the path to the temp file, or None if parsing fails.
    """
    try:
        output = subprocess.run(
            ["nvidia-smi", "topo", "-m"],
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return None

    entries: list[str] = []
    for line in output.splitlines():
        if not line.startswith("GPU"):
            continue
        parts = line.split()
        gpu_idx = parts[0].replace("GPU", "")
        # NUMA Affinity column (ray.sub parses $(NF-1)). On GB200 this is a list
        # like "0,2-17" (the GPU-local CPU NUMA node plus the GPU's HBM NUMA
        # nodes); take the first entry, which is the local CPU NUMA node.
        numa = parts[-2].split(",")[0].split("-")[0]
        if not numa.isdigit():
            continue
        try:
            with open(f"/sys/devices/system/node/node{numa}/cpulist") as nf:
                cpulist = nf.read().strip()
        except OSError:
            continue
        if cpulist:
            entries.append(f"{gpu_idx}:{cpulist}")

    if not entries:
        return None

    f = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    f.write("\n".join(entries) + "\n")
    f.flush()
    f.close()
    return f.name


def _reset_membind_to_all() -> None:
    """Reset NUMA memory policy to allow allocations from all nodes.

    Used to establish the "unbound" baseline in benchmarks. In production
    code we only bind (never unbind), so this helper is test-only.
    """
    libnuma = _load_libnuma()
    if libnuma is None:
        return

    import ctypes

    libnuma.numa_set_membind.argtypes = [ctypes.c_void_p]
    libnuma.numa_allocate_nodemask.restype = ctypes.c_void_p
    libnuma.numa_bitmask_setbit.argtypes = [ctypes.c_void_p, ctypes.c_uint]
    libnuma.numa_bitmask_setbit.restype = ctypes.c_void_p
    libnuma.numa_bitmask_free.argtypes = [ctypes.c_void_p]
    libnuma.numa_max_node.restype = ctypes.c_int

    max_node = libnuma.numa_max_node()
    nodemask = libnuma.numa_allocate_nodemask()
    for n in range(max_node + 1):
        libnuma.numa_bitmask_setbit(nodemask, n)
    libnuma.numa_set_membind(nodemask)
    libnuma.numa_bitmask_free(nodemask)


def _reset_all_bindings() -> None:
    """Reset both CPU affinity and memory policy to unbound defaults."""
    os.sched_setaffinity(0, set(range(os.cpu_count() or 256)))
    _reset_membind_to_all()


def _patch_affinity_path(path: str):
    """Temporarily override GPU_CPU_AFFINITY_PATH in the numa_utils module."""
    import nemo_rl.distributed.numa_utils as mod

    old = mod.GPU_CPU_AFFINITY_PATH
    mod.GPU_CPU_AFFINITY_PATH = path
    return old


def _split_available_cpus() -> tuple[list[int], list[int]] | tuple[None, None]:
    """Split the CPUs available to this process into two non-empty groups.

    Lets the CPU-binding tests exercise a real ``os.sched_setaffinity`` on any
    host instead of hard-coding a 144-CPU (GB200) layout that fails on smaller
    machines. Returns ``(group0, group1)`` as sorted lists, or ``(None, None)``
    when there are fewer than 2 available CPUs to split.
    """
    avail = sorted(os.sched_getaffinity(0))
    if len(avail) < 2:
        return None, None
    mid = len(avail) // 2
    return avail[:mid], avail[mid:]


# ---------------------------------------------------------------------------
# Pure unit tests (no GPU or libnuma required)
# ---------------------------------------------------------------------------


class TestParseCpulist:
    def test_single_range(self):
        assert _parse_cpulist("0-71") == set(range(72))

    def test_multiple_ranges(self):
        assert _parse_cpulist("0-3,8-11") == {0, 1, 2, 3, 8, 9, 10, 11}

    def test_single_values(self):
        assert _parse_cpulist("0,4,8") == {0, 4, 8}

    def test_mixed(self):
        assert _parse_cpulist("0-2,5,10-12") == {0, 1, 2, 5, 10, 11, 12}

    def test_whitespace(self):
        assert _parse_cpulist(" 0-3 , 8 ") == {0, 1, 2, 3, 8}

    def test_single_cpu(self):
        assert _parse_cpulist("42") == {42}


class TestBindToGpuNuma:
    """Test bind_to_gpu_numa logic with mock affinity files."""

    def test_disabled_via_env(self, monkeypatch):
        monkeypatch.setenv("NRL_DISABLE_NUMA_BINDING", "1")
        assert bind_to_gpu_numa() is False

    def test_no_cuda_visible_devices(self, monkeypatch):
        monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
        monkeypatch.delenv("NRL_DISABLE_NUMA_BINDING", raising=False)
        assert bind_to_gpu_numa() is False

    def test_empty_cuda_visible_devices(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")
        monkeypatch.delenv("NRL_DISABLE_NUMA_BINDING", raising=False)
        assert bind_to_gpu_numa() is False

    def test_missing_affinity_file(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
        monkeypatch.delenv("NRL_DISABLE_NUMA_BINDING", raising=False)

        old_path = _patch_affinity_path("/nonexistent/path")
        try:
            assert bind_to_gpu_numa() is False
        finally:
            _patch_affinity_path(old_path)

    def test_gpu_not_in_file(self, monkeypatch):
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
        monkeypatch.delenv("NRL_DISABLE_NUMA_BINDING", raising=False)
        monkeypatch.setenv("NRL_DISABLE_NUMA_MEMBIND", "1")

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("0:0-71\n1:0-71\n2:72-143\n3:72-143\n")
            f.flush()

            old_path = _patch_affinity_path(f.name)
            try:
                assert bind_to_gpu_numa() is False
            finally:
                _patch_affinity_path(old_path)
                os.unlink(f.name)

    def test_successful_cpu_binding(self, monkeypatch):
        """Verify sched_setaffinity is called with the correct CPU set."""
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
        monkeypatch.delenv("NRL_DISABLE_NUMA_BINDING", raising=False)
        monkeypatch.setenv("NRL_DISABLE_NUMA_MEMBIND", "1")

        # Derive the cpulist from the CPUs actually available to this process so
        # the test is host-portable. GPUs 0/1 map to the first CPU group, GPUs
        # 2/3 to the second; binding GPU 2 should land us on the second group.
        group0, group1 = _split_available_cpus()
        if group1 is None:
            pytest.skip("need >= 2 available CPUs to exercise NUMA CPU binding")
        cpus0 = ",".join(map(str, group0))
        cpus1 = ",".join(map(str, group1))

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(f"0:{cpus0}\n1:{cpus0}\n2:{cpus1}\n3:{cpus1}\n")
            f.flush()

            old_path = _patch_affinity_path(f.name)
            try:
                result = bind_to_gpu_numa()
                assert result is True
                bound_cpus = os.sched_getaffinity(0)
                assert bound_cpus == set(group1)
            finally:
                _patch_affinity_path(old_path)
                os.unlink(f.name)
                _reset_all_bindings()

    def test_multi_gpu_cvd_uses_first(self, monkeypatch):
        """When CUDA_VISIBLE_DEVICES=2,3 we bind to GPU 2's cpulist."""
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
        monkeypatch.delenv("NRL_DISABLE_NUMA_BINDING", raising=False)
        monkeypatch.setenv("NRL_DISABLE_NUMA_MEMBIND", "1")

        # Host-portable cpulist (see test_successful_cpu_binding).
        group0, group1 = _split_available_cpus()
        if group1 is None:
            pytest.skip("need >= 2 available CPUs to exercise NUMA CPU binding")
        cpus0 = ",".join(map(str, group0))
        cpus1 = ",".join(map(str, group1))

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(f"0:{cpus0}\n1:{cpus0}\n2:{cpus1}\n3:{cpus1}\n")
            f.flush()

            old_path = _patch_affinity_path(f.name)
            try:
                result = bind_to_gpu_numa()
                assert result is True
                bound_cpus = os.sched_getaffinity(0)
                assert bound_cpus == set(group1)
            finally:
                _patch_affinity_path(old_path)
                os.unlink(f.name)
                _reset_all_bindings()


class TestSetNumaMembind:
    """Test membind with libnuma (skipped if libnuma unavailable)."""

    @pytest.fixture(autouse=True)
    def _check_libnuma(self):
        if _load_libnuma() is None:
            pytest.skip("libnuma.so.1 not available")

    def test_membind_disabled(self, monkeypatch):
        monkeypatch.setenv("NRL_DISABLE_NUMA_MEMBIND", "1")
        assert _set_numa_membind({0, 1, 2}) is False

    def test_membind_succeeds(self, monkeypatch):
        monkeypatch.delenv("NRL_DISABLE_NUMA_MEMBIND", raising=False)
        cpus = os.sched_getaffinity(0)
        assert _set_numa_membind(cpus) is True

    def test_get_numa_node_valid(self):
        libnuma = _load_libnuma()
        node = _get_numa_node(libnuma, {0})
        assert node >= 0


# ---------------------------------------------------------------------------
# GPU benchmark test — validates binding has measurable impact on D2H
# ---------------------------------------------------------------------------

HAS_CUDA = False
try:
    import torch

    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    pass

HAS_LIBNUMA = _load_libnuma() is not None


@pytest.mark.skipif(not HAS_CUDA, reason="No CUDA GPU available")
@pytest.mark.skipif(not HAS_LIBNUMA, reason="libnuma.so.1 not available")
class TestNUMABindingBenchmark:
    """D2H bandwidth benchmark to validate NUMA binding impact.

    Run inside the RL container on a multi-socket node (DGX or GB200):

        CUDA_VISIBLE_DEVICES=0 pytest tests/unit/distributed/test_numa_utils.py::TestNUMABindingBenchmark -v -s

    The test measures D2H copy time with and without NUMA binding and
    reports the speedup. It also verifies the process is correctly bound
    to the expected NUMA node for the assigned GPU.
    """

    TENSOR_ELEMENTS = 64 * 1024 * 1024  # 256 MB at float32
    WARMUP_ITERS = 10
    BENCH_ITERS = 50

    def _d2h_bandwidth_ms(self) -> float:
        """Measure average D2H copy time in milliseconds."""
        t_gpu = torch.randn(self.TENSOR_ELEMENTS, device="cuda", dtype=torch.float32)

        for _ in range(self.WARMUP_ITERS):
            _ = t_gpu.to("cpu", non_blocking=True)
            torch.cuda.synchronize()

        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        for _ in range(self.BENCH_ITERS):
            _ = t_gpu.to("cpu", non_blocking=True)
        end.record()
        torch.cuda.synchronize()

        return start.elapsed_time(end) / self.BENCH_ITERS

    def test_d2h_with_numa_binding(self):
        """Measure D2H bandwidth before and after NUMA binding.

        This test does NOT assert a specific speedup (it varies by
        platform) — it reports the numbers so you can compare.
        The key assertion is that binding succeeds and the process
        ends up on the expected NUMA node.
        """
        cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        if not cvd:
            pytest.skip(
                "CUDA_VISIBLE_DEVICES not set — run with CUDA_VISIBLE_DEVICES=N"
            )

        # Baseline: reset to fully unbound (all CPUs, all NUMA nodes)
        _reset_all_bindings()
        unbound_ms = self._d2h_bandwidth_ms()

        # Generate the affinity file from live nvidia-smi topo
        affinity_file = _write_affinity_file_from_topo()
        if affinity_file is None:
            pytest.skip("Could not parse nvidia-smi topo output")

        # Apply NUMA binding (CPU affinity + membind)
        old_path = _patch_affinity_path(affinity_file)
        gpu_str = cvd.split(",")[0]
        try:
            result = bind_to_gpu_numa()
            assert result is True, f"bind_to_gpu_numa() failed for GPU {gpu_str}"
        finally:
            _patch_affinity_path(old_path)

        bound_cpus = os.sched_getaffinity(0)
        libnuma = _load_libnuma()
        numa_node = _get_numa_node(libnuma, bound_cpus)
        assert numa_node >= 0, (
            f"Could not determine NUMA node for CPU {min(bound_cpus)}"
        )

        bound_ms = self._d2h_bandwidth_ms()
        speedup = unbound_ms / bound_ms if bound_ms > 0 else float("inf")

        print(f"\n{'=' * 60}")
        print(f"NUMA Binding D2H Benchmark (GPU {gpu_str})")
        print(f"{'=' * 60}")
        print(
            f"  Tensor size:    {self.TENSOR_ELEMENTS * 4 / 1024 / 1024:.0f} MB (float32)"
        )
        print(f"  Iterations:     {self.BENCH_ITERS}")
        print(f"  Unbound D2H:    {unbound_ms:.3f} ms/iter")
        print(f"  Bound D2H:      {bound_ms:.3f} ms/iter")
        print(f"  Speedup:        {speedup:.3f}x")
        print(f"  Bound CPUs:     {min(bound_cpus)}-{max(bound_cpus)}")
        print(f"  NUMA node:      {numa_node}")
        print(f"{'=' * 60}")

        os.unlink(affinity_file)
        _reset_all_bindings()

    def test_actor_mapping_correctness(self):
        """Verify that bind_to_gpu_numa places the process on the correct NUMA node.

        For each GPU on the node, sets CUDA_VISIBLE_DEVICES, calls
        bind_to_gpu_numa, and checks that the resulting CPU affinity
        and NUMA node match nvidia-smi topo.
        """
        affinity_file = _write_affinity_file_from_topo()
        if affinity_file is None:
            pytest.skip("Could not parse nvidia-smi topo output")

        # Read back the affinity file to get the expected mapping
        gpu_map: dict[str, str] = {}
        with open(affinity_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    idx, cpulist = line.split(":", 1)
                    gpu_map[idx] = cpulist

        libnuma = _load_libnuma()
        old_path = _patch_affinity_path(affinity_file)
        old_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")

        results = []
        try:
            for gpu_idx, expected_cpulist in gpu_map.items():
                _reset_all_bindings()
                os.environ["CUDA_VISIBLE_DEVICES"] = gpu_idx

                result = bind_to_gpu_numa()
                assert result is True, f"bind_to_gpu_numa() failed for GPU {gpu_idx}"

                bound_cpus = os.sched_getaffinity(0)
                expected_cpus = _parse_cpulist(expected_cpulist)
                assert bound_cpus == expected_cpus, (
                    f"GPU {gpu_idx}: expected CPUs {expected_cpulist}, "
                    f"got {min(bound_cpus)}-{max(bound_cpus)}"
                )

                numa_node = _get_numa_node(libnuma, bound_cpus)
                results.append((gpu_idx, expected_cpulist, numa_node))
        finally:
            _patch_affinity_path(old_path)
            if old_cvd:
                os.environ["CUDA_VISIBLE_DEVICES"] = old_cvd
            else:
                os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            _reset_all_bindings()
            os.unlink(affinity_file)

        print(f"\n{'=' * 60}")
        print("Actor → GPU → NUMA Mapping Verification")
        print(f"{'=' * 60}")
        for gpu_idx, cpulist, numa_node in results:
            print(f"  GPU {gpu_idx} → CPUs {cpulist} → NUMA node {numa_node}")
        print(f"{'=' * 60}")
