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
"""Helpers for code that needs to know it's running inside a Kubernetes pod."""

import os

# Path the kubelet projects into every pod's filesystem with the namespace
# the pod was scheduled into. Reading this avoids having to thread the
# namespace through env vars at deployment time.
POD_NAMESPACE_FILE = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"


def is_in_kubernetes() -> bool:
    """True when the current process runs inside a Kubernetes pod.

    The kubelet always sets ``KUBERNETES_SERVICE_HOST`` on every container
    it launches; we use its presence as the canonical signal.
    """
    return "KUBERNETES_SERVICE_HOST" in os.environ


def read_pod_namespace() -> str | None:
    """Return the namespace this pod is running in, or ``None`` if unavailable.

    The serviceaccount projection is only mounted when ``automountServiceAccountToken``
    is enabled (the default). On unusual setups the file may be absent — callers
    should treat ``None`` as "namespace unknown" rather than as an error.
    """
    try:
        with open(POD_NAMESPACE_FILE) as f:
            ns = f.read().strip()
            return ns or None
    except OSError:
        return None
