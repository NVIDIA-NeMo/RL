#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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
"""Gate the training driver on the exact live Slurm GPU allocation."""

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path


def main():
    import ray

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", type=int, required=True)
    parser.add_argument("--gpus-per-node", type=int, required=True)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    deadline = time.monotonic() + args.timeout
    while True:
        try:
            ray.init(address=os.environ["RAY_ADDRESS"], logging_level="ERROR")
            nodes = [node for node in ray.nodes() if node.get("Alive")]
            if len(nodes) == args.nodes:
                counts = [
                    int(node.get("Resources", {}).get("GPU", 0)) for node in nodes
                ]
                if counts != [args.gpus_per_node] * args.nodes:
                    raise RuntimeError(f"Unexpected per-node Ray GPU counts: {counts}")
                resources = ray.cluster_resources()
                expected = args.nodes * args.gpus_per_node
                if int(resources.get("worker_units", 0)) != expected:
                    raise RuntimeError(
                        "Ray worker_units differ from the Slurm GPU allocation"
                    )
                receipt = {
                    "status": "pass",
                    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
                    "expected_nodes": args.nodes,
                    "expected_gpus": expected,
                    "cluster_resources": resources,
                    "nodes": [
                        {
                            "node_id": node["NodeID"],
                            "address": node["NodeManagerAddress"],
                            "resources": node["Resources"],
                        }
                        for node in nodes
                    ],
                }
                args.output.write_text(json.dumps(receipt, indent=2) + "\n")
                print(
                    json.dumps(
                        {"status": "pass", "nodes": args.nodes, "gpus": expected}
                    ),
                    flush=True,
                )
                return
            if len(nodes) > args.nodes:
                raise RuntimeError("Ray contains nodes outside this allocation")
            print(f"Waiting for Ray nodes: {len(nodes)}/{args.nodes}", flush=True)
        except ConnectionError:
            print("Waiting for Ray head", flush=True)
        finally:
            ray.shutdown()
        if time.monotonic() >= deadline:
            raise TimeoutError("Not all allocated Ray workers became ready")
        time.sleep(5)


if __name__ == "__main__":
    main()
