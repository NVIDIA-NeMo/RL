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
"""What does register mode's ``put`` actually pay, and is there a cheaper shape?

Register mode registers the producer's own allocation, and PyTorch hands it a
fresh one every step, so the pin-table dedupe never hits and every put pays a
full ``ibv_reg_mr``. The alternative is the one TransferQueue's GDR path already
uses for staging: keep a persistent *registered* arena and copy into it. That
gives up "no copy at put" but may be far cheaper, because registration scales
with pages pinned while a copy scales with bytes at HBM bandwidth.

This measures both at the payload size a real GRPO step produces, plus the
scaling either side of it, so the crossover is visible rather than assumed.

    python tools/bench_pin_vs_copy.py [--mb 5.86]
"""

from __future__ import annotations

import argparse
import statistics
from time import perf_counter

from nemo_rl.data_plane.factory import maybe_configure_data_plane_env


def _bench(label: str, fn, reps: int = 10) -> float:
    """Run ``fn`` ``reps`` times; report and return the median in microseconds."""
    samples = []
    for _ in range(reps):
        started = perf_counter()
        fn()
        samples.append((perf_counter() - started) * 1e6)
    median = statistics.median(samples)
    print(f"  {label:42s} median={median:10.1f}us  min={min(samples):10.1f}us")
    return median


def main() -> int:
    """Compare fresh registration against a copy into a pre-registered arena."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mb",
        type=float,
        nargs="+",
        default=[0.5, 5.86, 64.0],
        help="payload sizes; 5.86 is the measured per-put size of a GRPO step",
    )
    args = parser.parse_args()

    maybe_configure_data_plane_env(
        {
            "enabled": True,
            "impl": "transfer_queue",
            "backend": "transfer_engine",
            "claim_meta_poll_interval_s": 0.5,
        }
    )
    import torch
    from mooncake.engine import TransferEngine

    from nemo_rl.data_plane.adapters.transfer_queue import rdma_devices

    torch.cuda.init()
    engine = TransferEngine()
    if engine.initialize("", "P2PHANDSHAKE", "rdma", rdma_devices()) != 0:
        print("FAIL: engine init")
        return 1

    for mb in args.mb:
        n_el = int(mb * 1048576 // 4)
        print(f"\n=== {mb} MB ===")

        def register_fresh(device: str) -> None:
            t = torch.empty(n_el, dtype=torch.float32, device=device)
            storage = t.untyped_storage()
            engine.register_memory(storage.data_ptr(), storage.nbytes())
            engine.unregister_memory(storage.data_ptr())

        reg_cuda = _bench("register+unregister fresh CUDA", lambda: register_fresh("cuda"))
        _bench("register+unregister fresh host", lambda: register_fresh("cpu"))

        arena = torch.empty(n_el, dtype=torch.float32, device="cuda")
        arena_storage = arena.untyped_storage()
        engine.register_memory(arena_storage.data_ptr(), arena_storage.nbytes())
        source = torch.empty(n_el, dtype=torch.float32, device="cuda")

        def copy_into_arena() -> None:
            arena.copy_(source)
            torch.cuda.synchronize()

        copy = _bench("D2D copy into pre-registered arena", copy_into_arena)
        engine.unregister_memory(arena_storage.data_ptr())
        print(f"  -> registering fresh memory costs {reg_cuda / max(copy, 1e-9):.0f}x the copy")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
