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
"""Watchdog for RayCluster teardown. Two modes, picked by env:

1. **Peer-cluster watch** (sidecar, original mode):
   Monitors a peer RayCluster and tears down both clusters on failure.
   Used for disaggregated RL-Gym setups where both clusters should fail
   fast together. Runs as a sidecar container on the head pod.

   Env (peer mode is active when ``SELF_CLUSTER_NAME`` is set):
     SELF_CLUSTER_NAME   - Name of this RayCluster (to self-delete)
     PEER_CLUSTER_NAME   - Name of the peer RayCluster to watch
     JOB_ID              - ConfigMap endpoint-registry job id (optional)
     POLL_INTERVAL       - Seconds between status checks (default: 10)
     MAX_PEER_FAILURES   - Consecutive failures before teardown (default: 3)

2. **Ray-job completion watch** (standalone k8s Job, new mode):
   Polls a Ray Job's status via the RayCluster dashboard HTTP API and
   deletes the declared RayClusters when the job reaches a terminal state
   (SUCCEEDED, FAILED, or STOPPED). Used by ``nrl-k8s launch/run
   --teardown-on-exit``.

   Env (ray-job mode is active when ``RAY_JOB_ID`` is set):
     RAY_JOB_ID          - Ray Job submission id to watch
     DASHBOARD_URL       - Cluster-internal Ray dashboard URL,
                           e.g. http://<cluster>-head-svc.<ns>.svc.cluster.local:8265
     CLUSTERS_TO_DELETE  - Comma-separated RayCluster names to delete on
                           terminal state
     JOB_ID              - ConfigMap endpoint-registry job id (optional)
     POLL_INTERVAL       - Seconds between status checks (default: 30)
"""

import json
import os
import ssl
import sys
import time
import urllib.request
from pathlib import Path

SELF_CLUSTER_NAME = os.environ.get("SELF_CLUSTER_NAME", "")
PEER_CLUSTER_NAME = os.environ.get("PEER_CLUSTER_NAME", "")
JOB_ID = os.environ.get("JOB_ID", "")
POLL_INTERVAL = int(os.environ.get("POLL_INTERVAL", "10"))
MAX_PEER_FAILURES = int(os.environ.get("MAX_PEER_FAILURES", "3"))

RAY_JOB_ID = os.environ.get("RAY_JOB_ID", "")
DASHBOARD_URL = os.environ.get("DASHBOARD_URL", "")
CLUSTERS_TO_DELETE = [
    c for c in os.environ.get("CLUSTERS_TO_DELETE", "").split(",") if c
]
TERMINAL_JOB_STATES = {"SUCCEEDED", "FAILED", "STOPPED"}

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


def teardown(reason: str, clusters=None, exit_code=1):
    """Delete the given RayClusters (+ the endpoint-registry ConfigMap, if any)
    then exit. Caller passes an explicit cluster list for the ray-job mode; the
    peer-cluster mode defaults to [self, peer]."""
    if clusters is None:
        clusters = [c for c in (SELF_CLUSTER_NAME, PEER_CLUSTER_NAME) if c]
    print(f"[peer-watcher] TEARING DOWN: {reason}", flush=True)
    path_prefix = f"/apis/ray.io/v1/namespaces/{NAMESPACE}/rayclusters"
    for name in clusters:
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
    sys.exit(exit_code)


def watch_ray_job():
    """Block until the Ray Job at DASHBOARD_URL reaches a terminal state,
    then delete every cluster in CLUSTERS_TO_DELETE. Stdlib-only HTTP so
    this stays runnable in a tiny image."""
    print(
        f"[peer-watcher] Watching Ray job {RAY_JOB_ID} via {DASHBOARD_URL}; "
        f"will delete {CLUSTERS_TO_DELETE} on terminal state",
        flush=True,
    )
    poll = int(os.environ.get("POLL_INTERVAL", "30"))
    while True:
        time.sleep(poll)
        try:
            with urllib.request.urlopen(
                f"{DASHBOARD_URL}/api/jobs/{RAY_JOB_ID}", timeout=15
            ) as resp:
                data = json.loads(resp.read())
        except Exception as exc:
            print(f"[peer-watcher] poll error: {exc}", flush=True)
            continue
        status = (data.get("status") or "").upper()
        if status in TERMINAL_JOB_STATES:
            teardown(
                f"Ray job {RAY_JOB_ID} reached {status}",
                clusters=CLUSTERS_TO_DELETE,
                exit_code=0 if status == "SUCCEEDED" else 1,
            )


def main():
    if RAY_JOB_ID:
        if not DASHBOARD_URL or not CLUSTERS_TO_DELETE:
            raise RuntimeError(
                "RAY_JOB_ID set but DASHBOARD_URL or CLUSTERS_TO_DELETE missing"
            )
        watch_ray_job()
        return

    if not SELF_CLUSTER_NAME or not PEER_CLUSTER_NAME:
        raise RuntimeError(
            "peer mode requires SELF_CLUSTER_NAME and PEER_CLUSTER_NAME"
        )
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
