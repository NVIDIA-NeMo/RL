#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.  All rights reserved.
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
"""Sidecar that monitors a peer RayCluster and tears down both clusters on failure.

Used for disaggregated RL-Gym setups where both clusters should fail fast together.
Runs as a sidecar container in the head pod of each RayCluster.

Monitors:
  1. Peer RayCluster status (deleted / failed / suspended)
  2. ConfigMap "error" key (set by the application via K8sEndpointRegistry.signal_error())

Environment variables:
  SELF_CLUSTER_NAME   - Name of this RayCluster (to self-delete)
  PEER_CLUSTER_NAME   - Name of the peer RayCluster to watch
  JOB_ID              - Job ID for the ConfigMap endpoint registry (optional)
  POLL_INTERVAL       - Seconds between status checks (default: 10)
  MAX_PEER_FAILURES   - Consecutive failures before teardown (default: 3)
"""

import json
import os
import ssl
import sys
import time
import urllib.request
from pathlib import Path

SELF_CLUSTER_NAME = os.environ["SELF_CLUSTER_NAME"]
PEER_CLUSTER_NAME = os.environ["PEER_CLUSTER_NAME"]
JOB_ID = os.environ.get("JOB_ID", "")
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "10"))
MAX_PEER_FAILURES = int(os.environ.get("MAX_PEER_FAILURES", "3"))

SA_PATH = Path("/var/run/secrets/kubernetes.io/serviceaccount")
NAMESPACE = (
    (SA_PATH / "namespace").read_text().strip()
    if (SA_PATH / "namespace").exists()
    else "default"
)
TOKEN = (SA_PATH / "token").read_text().strip() if (SA_PATH / "token").exists() else ""
APISERVER = "https://kubernetes.default.svc"

# Trust the in-cluster CA.
SSL_CTX = (
    ssl.create_default_context(cafile=str(SA_PATH / "ca.crt"))
    if (SA_PATH / "ca.crt").exists()
    else ssl._create_unverified_context()
)


def kube_request(path: str, method: str = "GET") -> dict:
    req = urllib.request.Request(f"{APISERVER}{path}", method=method)
    req.add_header("Authorization", f"Bearer {TOKEN}")
    try:
        with urllib.request.urlopen(req, context=SSL_CTX, timeout=10) as resp:
            return json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return {"code": e.code, "message": e.reason}
    except Exception as e:
        return {"code": 0, "message": str(e)}


def teardown(reason: str):
    print(f"[peer-watcher] TEARING DOWN: {reason}", flush=True)
    path_prefix = f"/apis/ray.io/v1/namespaces/{NAMESPACE}/rayclusters"
    for name in (SELF_CLUSTER_NAME, PEER_CLUSTER_NAME):
        print(f"[peer-watcher] Deleting RayCluster: {name}", flush=True)
        kube_request(f"{path_prefix}/{name}", method="DELETE")
    if JOB_ID:
        print(
            f"[peer-watcher] Deleting ConfigMap: nemo-rl-endpoints-{JOB_ID}", flush=True
        )
        kube_request(
            f"/api/v1/namespaces/{NAMESPACE}/configmaps/nemo-rl-endpoints-{JOB_ID}",
            method="DELETE",
        )
    sys.exit(1)


def main():
    print(
        f"[peer-watcher] Watching peer={PEER_CLUSTER_NAME}, self={SELF_CLUSTER_NAME}",
        flush=True,
    )
    print(
        f"[peer-watcher] namespace={NAMESPACE}, poll={POLL_INTERVAL}s, max_failures={MAX_PEER_FAILURES}",
        flush=True,
    )

    consecutive_failures = 0

    while True:
        time.sleep(POLL_INTERVAL)

        # Check peer RayCluster.
        resp = kube_request(
            f"/apis/ray.io/v1/namespaces/{NAMESPACE}/rayclusters/{PEER_CLUSTER_NAME}"
        )
        code = resp.get("code", 0)
        if code == 404:
            teardown(f"Peer {PEER_CLUSTER_NAME} not found (deleted)")

        status = resp.get("status", {}).get("state", "")
        if status in ("failed", "suspended"):
            consecutive_failures += 1
            print(
                f"[peer-watcher] Peer {PEER_CLUSTER_NAME} is {status} ({consecutive_failures}/{MAX_PEER_FAILURES})",
                flush=True,
            )
            if consecutive_failures >= MAX_PEER_FAILURES:
                teardown(
                    f"Peer {PEER_CLUSTER_NAME} failed {MAX_PEER_FAILURES} consecutive checks"
                )
            continue

        # Check ConfigMap error signal.
        if JOB_ID:
            cm = kube_request(
                f"/api/v1/namespaces/{NAMESPACE}/configmaps/nemo-rl-endpoints-{JOB_ID}"
            )
            error = cm.get("data", {}).get("error", "")
            if error:
                teardown(f"Error signaled via ConfigMap: {error}")

        # Unknown state (e.g., K8s API error, transient network issue) — treat as failure.
        if code != 0 and status == "":
            consecutive_failures += 1
            print(
                f"[peer-watcher] Could not determine peer status (code={code}), treating as failure ({consecutive_failures}/{MAX_PEER_FAILURES})",
                flush=True,
            )
            if consecutive_failures >= MAX_PEER_FAILURES:
                teardown(
                    f"Peer {PEER_CLUSTER_NAME} unreachable for {MAX_PEER_FAILURES} consecutive checks"
                )
            continue

        # Peer healthy.
        if consecutive_failures > 0:
            print(
                f"[peer-watcher] Peer {PEER_CLUSTER_NAME} recovered (status={status})",
                flush=True,
            )
        consecutive_failures = 0


if __name__ == "__main__":
    main()
