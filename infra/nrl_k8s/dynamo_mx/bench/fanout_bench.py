"""MX direct-vs-tree fan-out producer using the real model tensor layout.

Roles:
  trainer             publish source metadata and hold
  seed <seed_id>      pull trainer, then republish local registered buffers
  receiver <id>       pull PARENT=trainer or PARENT=seed:<seed_id>

Every process writes a JSON timeline under RESULT_DIR. Run direct and tree trials
in separate result directories, then summarize makespan and compare them with
``differentiator_suite.py fanout``.
"""

from __future__ import annotations

import base64
import json
import os
import pickle
import sys
import time
from pathlib import Path

import torch
from modelexpress.nixl_transfer import NixlTransferManager, is_nixl_available
from safetensors import safe_open


ROLE = sys.argv[1]
IDENT = sys.argv[2] if len(sys.argv) > 2 else "0"
PARENT = os.environ.get("PARENT", "trainer")
RESULT_DIR = Path(os.environ.get("RESULT_DIR", "/mnt/rl-workspace/fanout_bench"))
EXPECTED_RECEIVERS = int(os.environ.get("EXPECTED_RECEIVERS", "1"))
MODEL_ID = "Qwen/Qwen3-30B-A3B-Instruct-2507"
CHECKPOINT_BYTES = 61_064_245_248
STAGE_NAMES = (
    "control_discovery", "source_preparation", "setup_registration",
    "transfer_planning", "wire_transfer", "receive_sync", "transformation",
    "installation", "post_install", "rollout_readiness",
)
HF_SNAPSHOT = os.environ.get("HF_SNAPSHOT")
if not HF_SNAPSHOT:
    raise RuntimeError("HF_SNAPSHOT must name an immutable Qwen3-30B-A3B snapshot")
SNAPSHOT = Path(HF_SNAPSHOT)


def resolve_snapshot():
    snapshot = SNAPSHOT.resolve()
    index_path = snapshot / "model.safetensors.index.json"
    if not index_path.is_file():
        raise RuntimeError(
            f"HF_SNAPSHOT must identify a local {MODEL_ID} snapshot: {SNAPSHOT}"
        )
    if "Qwen3-30B-A3B" not in str(snapshot):
        raise RuntimeError(f"Refusing non-30B snapshot: {snapshot}")
    return snapshot, json.loads(index_path.read_text())


def specs(snapshot, index):
    dtype_map = {
        "BF16": torch.bfloat16,
        "F16": torch.float16,
        "F32": torch.float32,
        "F8_E4M3": torch.float8_e4m3fn,
        "I64": torch.int64,
        "I32": torch.int32,
        "U8": torch.uint8,
        "BOOL": torch.bool,
    }
    output = {}
    for shard in sorted(set(index["weight_map"].values())):
        with safe_open(str(snapshot / shard), framework="pt") as handle:
            for name in handle.keys():
                tensor_slice = handle.get_slice(name)
                dtype_name = tensor_slice.get_dtype()
                if dtype_name not in dtype_map:
                    raise RuntimeError(f"Unsupported safetensors dtype {dtype_name}: {name}")
                output[name] = (
                    list(tensor_slice.get_shape()),
                    dtype_map[dtype_name],
                )
    return output


def load_checkpoint(snapshot, index):
    output = {}
    for shard in sorted(set(index["weight_map"].values())):
        with safe_open(str(snapshot / shard), framework="pt", device="cpu") as handle:
            for name in handle.keys():
                output[name] = handle.get_tensor(name).to("cuda:0")
    return output


def artifact(tensor_source):
    return {
        "schema_version": "refit-stage-v1",
        "model": MODEL_ID,
        "checkpoint": snapshot.name,
        "checkpoint_bytes": CHECKPOINT_BYTES,
        "tensor_source": tensor_source,
    }


def stages(wire_seconds=None):
    output = {
        name: {"status": "unavailable", "seconds": None} for name in STAGE_NAMES
    }
    if wire_seconds is not None:
        output["wire_transfer"] = {
            "status": "available",
            "seconds": wire_seconds,
            "source": "NixlTransferManager.receive_from_source",
        }
    return output


def metadata_path(parent):
    if parent == "trainer":
        return RESULT_DIR / "trainer.pkl"
    return RESULT_DIR / f"{parent.replace(':', '_')}.pkl"


def write_metadata(path, manager):
    path.write_bytes(
        pickle.dumps(
            {
                "agent_metadata": base64.b64encode(manager.nixl_metadata).decode(),
                "descriptors": manager.tensor_descriptors,
            }
        )
    )


def wait_metadata(path):
    while not path.exists():
        time.sleep(0.1)
    payload = pickle.loads(path.read_bytes())
    return (
        base64.b64decode(payload["agent_metadata"]),
        payload["descriptors"],
    )


assert is_nixl_available()
RESULT_DIR.mkdir(parents=True, exist_ok=True)
snapshot, checkpoint_index = resolve_snapshot()
layout = specs(snapshot, checkpoint_index)
if ROLE == "trainer":
    buffers = load_checkpoint(snapshot, checkpoint_index)
else:
    buffers = {
        name: torch.empty(shape, dtype=dtype, device="cuda:0")
        for name, (shape, dtype) in layout.items()
    }
manager = NixlTransferManager(
    agent_name=f"fanout-{ROLE}-{IDENT}",
    device_id=0,
    listen_port=0,
)
manager.initialize()
manager.register_tensors(buffers)
total_bytes = sum(t.numel() * t.element_size() for t in buffers.values())
if total_bytes != CHECKPOINT_BYTES:
    raise RuntimeError(
        f"{MODEL_ID} checkpoint is {total_bytes} bytes; expected {CHECKPOINT_BYTES}"
    )

if ROLE == "trainer":
    torch.cuda.synchronize()
    write_metadata(metadata_path("trainer"), manager)
    start = time.time()
    while len(list(RESULT_DIR.glob("receiver_*.json"))) < EXPECTED_RECEIVERS:
        time.sleep(0.2)
    result = {
        "role": ROLE,
        "bytes": total_bytes,
        "start_epoch": start,
        "end_epoch": time.time(),
        "stages": stages(),
        **artifact("safetensors"),
    }
    Path(RESULT_DIR, "trainer_result.json").write_text(json.dumps(result, indent=2))
elif ROLE == "seed":
    source_metadata, descriptors = wait_metadata(metadata_path("trainer"))
    start = time.time()
    transferred, tensors, duration = manager.receive_from_source(
        source_metadata,
        descriptors,
        timeout_seconds=900,
    )
    torch.cuda.synchronize()
    write_metadata(metadata_path(f"seed:{IDENT}"), manager)
    ready = time.time()
    while len(list(RESULT_DIR.glob(f"receiver_*_seed_{IDENT}.json"))) < EXPECTED_RECEIVERS:
        time.sleep(0.2)
    result = {
        "role": ROLE,
        "id": IDENT,
        "parent": "trainer",
        "bytes": transferred,
        "tensors": tensors,
        "pull_seconds": duration,
        "stages": stages(duration),
        "start_epoch": start,
        "ready_epoch": ready,
        **artifact("received_safetensors"),
    }
    Path(RESULT_DIR, f"seed_{IDENT}.json").write_text(json.dumps(result, indent=2))
elif ROLE == "receiver":
    source_metadata, descriptors = wait_metadata(metadata_path(PARENT))
    start = time.time()
    transferred, tensors, duration = manager.receive_from_source(
        source_metadata,
        descriptors,
        timeout_seconds=900,
    )
    torch.cuda.synchronize()
    end = time.time()
    result = {
        "role": ROLE,
        "id": IDENT,
        "parent": PARENT,
        "bytes": transferred,
        "tensors": tensors,
        "pull_seconds": duration,
        "gbps": transferred * 8 / duration / 1e9,
        "stages": stages(duration),
        "start_epoch": start,
        "end_epoch": end,
        **artifact("received_safetensors"),
    }
    suffix = PARENT.replace(":", "_")
    Path(RESULT_DIR, f"receiver_{IDENT}_{suffix}.json").write_text(
        json.dumps(result, indent=2)
    )
    print("FANOUT_RESULT", json.dumps(result), flush=True)
else:
    raise ValueError(f"Unknown role: {ROLE}")
