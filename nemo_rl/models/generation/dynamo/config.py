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

from typing import Any, Literal, NotRequired

from pydantic import BaseModel, Field, PositiveInt, model_validator

from nemo_rl.models.generation.interfaces import GenerationConfig


class DynamoWorkerArgs(BaseModel, extra="forbid"):
    """Structured ``dynamo.vllm`` runtime arguments owned by Dynamo."""

    tool_call_parser: str | None = None
    reasoning_parser: str | None = None
    exclude_tools_when_tool_choice_none: bool = True
    enable_structural_tag: bool = False
    structural_tag_scope: Literal["auto", "always"] = "auto"
    structural_tag_schema: Literal["auto", "strict"] = "auto"
    custom_jinja_template: str | None = None
    endpoint_types: list[Literal["chat", "completions"]] = Field(
        default_factory=lambda: ["chat", "completions"]
    )
    extra_cli_args: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_endpoint_types(self) -> "DynamoWorkerArgs":
        if not self.endpoint_types:
            raise ValueError("endpoint_types must contain at least one endpoint.")
        if len(self.endpoint_types) != len(set(self.endpoint_types)):
            raise ValueError("endpoint_types must not contain duplicates.")
        return self


class DynamoFrontendArgs(BaseModel, extra="forbid"):
    """Arguments for the Ray-managed Dynamo frontend."""

    router_mode: Literal[
        "round-robin",
        "random",
        "power-of-two",
        "kv",
        "direct",
        "least-loaded",
        "device-aware-weighted",
    ] = "round-robin"
    router_reset_states: bool = True
    extra_cli_args: list[str] = Field(default_factory=list)


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

    External deployments require at least one of the two; ``frontend_url`` wins
    if both are. ``deployment='ray'`` instead creates a fixed local deployment
    and rejects both external frontend fields.

    ``engine_world_size`` is the number of vLLM tensor/pipeline ranks behind
    each discovered Dynamo worker endpoint. External NCCL refit requires
    ``dgd_name``; Ray-managed refit uses the worker actors it owns.
    """

    deployment: Literal["external", "ray"] = "external"
    engine_world_size: PositiveInt
    dgd_name: str | None = None
    frontend_url: str | None = None
    namespace: str | None = None
    frontend_port: int = 8000
    dyn_system_port: int = 9090
    dynamo_python: str = "python"
    startup_timeout_s: float = 300.0
    etcd_port: int = 0
    etcd_peer_port: int = 0
    nats_port: int = 0
    system_port_base: int = 29000
    worker_args: DynamoWorkerArgs = Field(default_factory=DynamoWorkerArgs)
    frontend_args: DynamoFrontendArgs = Field(default_factory=DynamoFrontendArgs)
    # HTTP timeout, in seconds, for direct generate()/generate_async() calls to
    # the DGD frontend. Required only when using direct generation.
    request_timeout_s: float | None = None

    @model_validator(mode="before")
    @classmethod
    def _set_ray_defaults(cls, data: Any) -> Any:
        if isinstance(data, dict) and data.get("deployment") == "ray":
            data = dict(data)
            data.setdefault("dynamo_python", "/opt/dynamo_venv/bin/python")
            data.setdefault("frontend_port", 0)
        return data

    @model_validator(mode="after")
    def _validate_deployment(self) -> "DynamoCfg":
        if self.deployment == "ray" and (
            self.dgd_name is not None or self.frontend_url is not None
        ):
            raise ValueError(
                "Ray-managed Dynamo owns its frontend; dgd_name and frontend_url "
                "must not be set when deployment='ray'."
            )
        if self.startup_timeout_s <= 0:
            raise ValueError("startup_timeout_s must be positive.")
        for field_name in ("frontend_port", "etcd_port", "etcd_peer_port", "nats_port"):
            value = getattr(self, field_name)
            if not (0 <= value <= 65535):
                raise ValueError(
                    f"{field_name} must be 0 (automatic) or between 1 and 65535."
                )
        if not (1 <= self.system_port_base <= 65535):
            raise ValueError("system_port_base must be between 1 and 65535.")
        if not self.dynamo_python:
            raise ValueError("dynamo_python must not be empty.")
        if self.request_timeout_s is not None and self.request_timeout_s <= 0:
            raise ValueError("request_timeout_s must be positive when configured.")
        return self


class DynamoConfig(GenerationConfig):
    """GenerationConfig for the Dynamo k8s backend.

    In ``deployment='external'`` mode the DGD owns the actual inference engine,
    and ``vllm_cfg`` / ``vllm_kwargs`` remain compatibility shims. In
    ``deployment='ray'`` mode NeMo-RL launches ``dynamo.vllm`` workers and these
    fields are authoritative for engine construction.
    """

    dynamo_cfg: DynamoCfg
    vllm_cfg: NotRequired[dict[str, Any]]
    vllm_kwargs: NotRequired[dict[str, Any]]
