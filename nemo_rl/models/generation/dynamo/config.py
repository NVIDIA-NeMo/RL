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

from typing import Any, NotRequired

from pydantic import BaseModel, PositiveInt

from nemo_rl.models.generation.interfaces import GenerationConfig


class DynamoCfg(BaseModel, extra="allow"):
    """Pointer to the DynamoGraphDeployment that serves this run's rollouts.

    Two ways to specify the frontend:

    * ``dgd_name`` — friendly path. The class derives
      ``http://{dgd_name}-frontend.{namespace}.svc.cluster.local:{frontend_port}/v1``
      from the dynamo operator's stable Service naming convention. Requires
      running inside the Kubernetes pod that has cluster-DNS access. nrl-k8s
      stamps this field automatically when it brings up the DGD.

    * ``frontend_url`` — escape hatch. Any reachable HTTP URL. Use this when
      pointing at a hand-rolled DGD with a non-default Service name, an
      external cluster reached via NodePort/Ingress, or running outside
      Kubernetes entirely. Setting this disables the K8s in-pod check.

    At least one of the two must be set. ``frontend_url`` wins if both are.

    ``engine_world_size`` is the number of vLLM tensor/pipeline ranks behind
    each discovered Dynamo worker endpoint. NCCL refit requires ``dgd_name``
    so the fixed worker fleet can be discovered through the frontend.
    """

    engine_world_size: PositiveInt
    dgd_name: str | None = None
    frontend_url: str | None = None
    namespace: str | None = None
    frontend_port: int = 8000
    dyn_system_port: int = 9090
    # HTTP timeout, in seconds, for direct generate()/generate_async() calls to
    # the DGD frontend. Required only when using direct generation.
    request_timeout_s: float | None = None


class DynamoConfig(GenerationConfig):
    """GenerationConfig for the Dynamo k8s backend.

    The DGD owns the actual inference engine (vLLM / sglang / trtllm) and all of
    its arguments — nemo-rl never sees them. The ``vllm_cfg`` / ``vllm_kwargs``
    fields are kept as compatibility shims because nemo-gym today reads
    ``cfg["vllm_cfg"]["max_model_len"]`` directly. They are *not* authoritative
    for the DGD's behaviour.
    """

    dynamo_cfg: DynamoCfg
    vllm_cfg: NotRequired[dict[str, Any]]
    vllm_kwargs: NotRequired[dict[str, Any]]
