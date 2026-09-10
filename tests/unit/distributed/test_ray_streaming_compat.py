"""Guard Ray upgrades against streaming metadata loss across actor environments.

Megatron workers inherit an async generator from a backend-only module. When a
single controller cannot import that module, Ray builds a placeholder class while
deserializing the actor handle. Ray 2.58 rejects the explicit streaming option on
that placeholder, even though the remote worker's method really is a generator.

Exercise Ray's actual deserialization, metadata, and options implementations
without starting a cluster. Only the GCS transport and module availability are
faked. Keep this test when updating the resolved Ray version in uv.lock.
"""

import json
import pickle
import sys
from collections.abc import AsyncIterator
from unittest.mock import Mock

import pytest
import ray
from ray import cloudpickle
from ray._private.function_manager import FunctionActorManager
from ray._raylet import PythonFunctionDescriptor
from ray.actor import ActorMethod, _ActorClassMethodMetadata


class BackendGenerationMixin:
    async def generate_async(self) -> AsyncIterator[int]:
        yield 1


@ray.remote
class BackendWorker(BackendGenerationMixin):
    pass


@pytest.mark.parametrize("backend_available", [True, False])
def test_streaming_options_after_actor_class_deserialization(
    monkeypatch: pytest.MonkeyPatch, backend_available: bool
) -> None:
    """An isolated controller must still be able to request streaming returns."""
    # Model separate worker/controller metadata caches without touching other tests.
    monkeypatch.setattr(_ActorClassMethodMetadata, "_cache", {})
    descriptor = PythonFunctionDescriptor(__name__, "__init__", "BackendWorker")
    worker_class = BackendWorker.__ray_metadata__.modified_class
    worker_metadata = _ActorClassMethodMetadata.create(worker_class, descriptor)
    assert worker_metadata.method_is_generator["generate_async"]

    serialized_class = cloudpickle.dumps(worker_class)
    job_id = ray.JobID.from_int(1)
    actor_record = pickle.dumps(
        {
            "job_id": job_id.binary(),
            "class_name": "BackendWorker",
            "module": __name__,
            "class": serialized_class,
            "actor_method_names": json.dumps(list(worker_metadata.signatures)),
        }
    )
    gcs_client = Mock()
    gcs_client.internal_kv_get.return_value = actor_record
    manager = FunctionActorManager(Mock(gcs_client=gcs_client))

    if not backend_available:
        # A None entry makes imports fail even though this test module is on disk.
        # The inherited mixin is serialized by reference, just like Megatron's.
        monkeypatch.setitem(sys.modules, __name__, None)
        with pytest.raises(ModuleNotFoundError):
            pickle.loads(serialized_class)

    controller_class = manager._load_actor_class_from_gcs(job_id, descriptor)
    monkeypatch.setattr(_ActorClassMethodMetadata, "_cache", {})
    controller_metadata = _ActorClassMethodMetadata.create(controller_class, descriptor)
    # Only options() is exercised, before any task submission. Avoid constructing
    # a live actor handle or depending on version-specific constructor arguments;
    # use the real metadata that Ray would install on the method handle.
    method = ActorMethod.__new__(ActorMethod)
    method._is_generator = controller_metadata.method_is_generator["generate_async"]

    # Match MegatronGeneration.generate_async, including its explicit override.
    # Do not xfail: a Ray upgrade must preserve this cross-environment contract.
    method.options(num_returns="streaming")
