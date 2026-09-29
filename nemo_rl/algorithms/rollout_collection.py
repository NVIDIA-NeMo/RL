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
"""Standalone rollout collection (stage A of the decoupled PPO pipeline).

Generates frozen-policy (pi_0) rollouts with a generation-only job — no policy
or value workers, no weight refits, no replay buffer — and banks each finished
prompt group as one file on disk:

    <out_dir>/shard_<k>/group_<dataset_idx>.pt

Each group file is the same trajectory dict the async PPO collector pushes to
the replay buffer (``{"batch": final_batch, "rollout_metrics": ...}``), plus
input-side fields the NeMo-Gym rollout path drops (``extra_env_info``, ``idx``,
``task_name`` are grafted back into ``batch``) and provenance metadata. The
critic-pretraining stage (:mod:`nemo_rl.algorithms.critic_pretrain`) consumes
these files.

Sharding is coordination-free: shard ``k`` of ``num_shards`` owns dataset
indices with ``idx % num_shards == k``; resume simply skips indices whose group
file already exists, so a killed job loses only its in-flight episodes.

Concurrency is admission-controlled at the SAMPLE level: each of a group's
``gens_per_prompt`` episodes is its own rollout call, bounded by
``collection.max_inflight_samples``, so when one episode finishes the freed
slot is immediately backfilled by the next pending sample (from any group).
Completed samples are reassembled into whole-group files; a group is banked
all-or-nothing, exactly as before.
"""

import hashlib
import json
import math
import os
import subprocess
import threading
import time
from pathlib import Path
from typing import Any, Optional

import torch

from nemo_rl.data.collate_fn import rl_collate_fn
from nemo_rl.distributed.batched_data_dict import BatchedDataDict

SHARD_FORMAT_VERSION = 1
_GROUP_PREFIX = "group_"
_GROUP_SUFFIX = ".pt"


# ===============================================================================
# Pure helpers (unit-tested, no heavy deps)
# ===============================================================================
def assigned_indices(
    dataset_len: int,
    shard_id: int,
    num_shards: int,
    index_start: int = 0,
    index_end: Optional[int] = None,
    max_groups: Optional[int] = None,
) -> list[int]:
    """Dataset indices owned by this shard (strided split over [start, end)).

    Strided (``idx % num_shards == shard_id``) rather than contiguous so every
    shard samples the full curriculum mix.
    """
    assert 0 <= shard_id < num_shards, (
        f"shard_id must be in [0, {num_shards}), got {shard_id}"
    )
    end = dataset_len if index_end is None else min(index_end, dataset_len)
    assert index_start >= 0 and index_start <= end, (
        f"invalid index range [{index_start}, {end})"
    )
    indices = [i for i in range(index_start, end) if i % num_shards == shard_id]
    if max_groups is not None:
        indices = indices[:max_groups]
    return indices


def group_filename(dataset_idx: int) -> str:
    """Group file name for a dataset index (fixed-width so listings sort)."""
    return f"{_GROUP_PREFIX}{dataset_idx:08d}{_GROUP_SUFFIX}"


def parse_group_index(filename: str) -> Optional[int]:
    """Inverse of :func:`group_filename`; None for non-group files."""
    name = os.path.basename(filename)
    if not (name.startswith(_GROUP_PREFIX) and name.endswith(_GROUP_SUFFIX)):
        return None
    stem = name[len(_GROUP_PREFIX) : -len(_GROUP_SUFFIX)]
    if not stem.isdigit():
        return None
    return int(stem)


def existing_group_indices(shard_dir: str | Path) -> set[int]:
    """Dataset indices with a completed group file in ``shard_dir``.

    Only finalized files count: in-progress writes use a ``.tmp*`` suffix and
    are atomically renamed on completion, so a crash never leaves a partial
    file that would be skipped on resume.
    """
    shard_dir = Path(shard_dir)
    if not shard_dir.is_dir():
        return set()
    out = set()
    for name in os.listdir(shard_dir):
        idx = parse_group_index(name)
        if idx is not None:
            out.add(idx)
    return out


def existing_group_indices_all(out_dir: str | Path) -> set[int]:
    """Union of completed group indices across ALL shard dirs under ``out_dir``.

    Resume scans the whole output tree rather than just this task's shard dir,
    so ``num_shards`` can be changed freely between submissions: a group banked
    under any previous sharding layout is never regenerated (``dataset_idx`` is
    globally unique and encoded in the filename, regardless of which shard dir
    holds it).
    """
    out_dir = Path(out_dir)
    done: set[int] = set()
    if not out_dir.is_dir():
        return done
    done |= existing_group_indices(out_dir)
    for sub in out_dir.glob("shard_*"):
        if sub.is_dir():
            done |= existing_group_indices(sub)
    return done


def write_group_atomic(shard_dir: str | Path, dataset_idx: int, payload: dict) -> Path:
    """torch.save ``payload`` to a tmp file, then atomically rename into place.

    The tmp file is fsynced before the rename so a node/kernel crash cannot
    leave a truncated file under the FINAL name (which resume would treat as
    done and stage B would fail to load).
    """
    shard_dir = Path(shard_dir)
    final_path = shard_dir / group_filename(dataset_idx)
    tmp_path = shard_dir / f"{group_filename(dataset_idx)}.tmp.{os.getpid()}"
    with open(tmp_path, "wb") as f:
        torch.save(payload, f)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, final_path)
    return final_path


def load_group(path: str | Path) -> dict:
    """Load a group file written by :func:`write_group_atomic`.

    weights_only=False: trajectories are pickled BatchedDataDict/dicts written
    by this same pipeline (trusted local artifact), not plain tensors.
    """
    return torch.load(path, weights_only=False, map_location="cpu")


def build_group_payload(
    dataset_idx: int,
    input_batch: BatchedDataDict,
    final_batch: BatchedDataDict,
    rollout_metrics: dict[str, Any],
) -> dict[str, Any]:
    """Assemble the on-disk group payload.

    The NeMo-Gym rollout path returns a ``final_batch`` WITHOUT
    ``extra_env_info`` / ``idx`` / ``task_name`` (unlike the generic multi-turn
    path); graft them back from the input batch so downstream consumers see the
    same batch shape the PPO driver loops build, and so answer-conditioned
    (privileged) critics can reach ``extra_env_info`` at train time.
    """
    for key in ("extra_env_info", "idx", "task_name"):
        if key not in final_batch and key in input_batch:
            final_batch[key] = input_batch[key]
    # _rowidx is a per-call scratch field run_async_nemo_gym_rollout writes into
    # extra_env_info rows (always 0 for single-sample calls); drop it so stored
    # rows match the pre-rollout inputs regardless of call batching.
    for row in final_batch.get("extra_env_info") or []:
        if isinstance(row, dict):
            row.pop("_rowidx", None)
    return {
        "format_version": SHARD_FORMAT_VERSION,
        "dataset_idx": dataset_idx,
        "batch": final_batch,
        "rollout_metrics": dict(rollout_metrics),
        "timestamp": time.time(),
    }


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _best_effort_git_commit() -> str:
    try:
        return (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=os.path.dirname(os.path.abspath(__file__)),
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
            or "unknown"
        )
    except Exception:
        return "unknown"


def build_shard_meta(
    master_config: Any,
    collection: dict[str, Any],
    tokenizer: Any,
    dataset_len: int,
) -> dict[str, Any]:
    """Provenance metadata written once per shard dir (``meta.json``).

    Downstream stages assert on the model/tokenizer identity recorded here —
    shards are token-id level and are invalidated by any tokenizer, chat
    template, max-length, or dataset change.
    """
    data_train = master_config.data.get("train")
    if isinstance(data_train, list):
        data_train = data_train[0] if data_train else {}
    chat_template = getattr(tokenizer, "chat_template", None) or ""
    dataset_path = (data_train or {}).get("data_path")
    return {
        "format_version": SHARD_FORMAT_VERSION,
        "model_name": master_config.policy["model_name"],
        "tokenizer_name_or_path": getattr(tokenizer, "name_or_path", "unknown"),
        "chat_template_sha256": _sha256(chat_template),
        "max_total_sequence_length": master_config.policy[
            "max_total_sequence_length"
        ],
        "dataset_path": dataset_path,
        "dataset_sha256": _file_sha256(dataset_path) if dataset_path else None,
        "dataset_len": dataset_len,
        "gens_per_prompt": collection["gens_per_prompt"],
        "shard_id": collection["shard_id"],
        "num_shards": collection["num_shards"],
        "index_start": collection["index_start"],
        "index_end": collection["index_end"],
        "git_commit": _best_effort_git_commit(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
    }


def resolve_collection_config(
    raw: Optional[dict[str, Any]], ppo_config: dict[str, Any]
) -> dict[str, Any]:
    """Fill defaults for the ``collection:`` config block."""
    cfg = dict(raw or {})
    assert cfg.get("out_dir"), (
        "collection.out_dir is required (pass ++collection.out_dir=<path>)"
    )
    cfg.setdefault("shard_id", 0)
    cfg.setdefault("num_shards", 1)
    cfg.setdefault("gens_per_prompt", ppo_config["num_generations_per_prompt"])
    cfg.setdefault("index_start", 0)
    cfg.setdefault("index_end", None)
    cfg.setdefault("max_groups", None)
    cfg.setdefault("log_every", 5)
    cfg.setdefault("max_consecutive_failures", 10)
    for key in ("shard_id", "num_shards", "gens_per_prompt"):
        cfg[key] = int(cfg[key])
    # Sample-level admission: concurrency is bounded per SAMPLE so a group's
    # stragglers never idle the engine. The legacy group-level knob converts.
    if cfg.get("max_inflight_samples") is None:
        legacy_groups = cfg.get("max_inflight_groups")
        cfg["max_inflight_samples"] = (
            int(legacy_groups) if legacy_groups is not None else 3
        ) * cfg["gens_per_prompt"]
    cfg["max_inflight_samples"] = int(cfg["max_inflight_samples"])
    assert cfg["max_inflight_samples"] >= 1
    return cfg


# ===============================================================================
# NeMo-Gym spinup (mirrors the inline _spinup_nemo_gym in ppo.setup(), which is
# not importable standalone; kept behaviorally identical)
# ===============================================================================
def spinup_nemo_gym(master_config: Any, base_urls: list[str], model_name: str):
    """Spin up the NeMo-Gym actor against the given vLLM server URLs."""
    from nemo_rl.environments.nemo_gym import spinup_nemo_gym_actor

    return spinup_nemo_gym_actor(
        env_configs=master_config.env,
        base_urls=base_urls,
        model_name=model_name,
        enable_router_replay=False,
        routed_experts_dtype="int16",
        use_fastokens=bool(master_config.policy["tokenizer"].get("use_fastokens")),
    )


# ===============================================================================
# Collection loop
# ===============================================================================
def _run_rollout_batch(
    policy_generation: Any,
    input_batch: BatchedDataDict,
    tokenizer: Any,
    task_to_env: dict[str, Any],
    master_config: Any,
) -> tuple[BatchedDataDict, dict[str, Any], dict[str, Any]]:
    """Run one batch (here: a single sample) through the NeMo-Gym rollout path.

    Mirrors AsyncTrajectoryCollector._run_prompt_group_worker's gym branch:
    stop tokens are cleared on a copied generation config so this path is safe
    by construction (run_async_nemo_gym_rollout asserts they are unset).
    """
    from nemo_rl.experience.rollouts import (
        get_nemo_gym_thinking_tags,
        run_nemo_gym_rollout_sync,
    )

    generation_config = {
        **master_config.policy["generation"],
        "stop_token_ids": None,
        "stop_strings": None,
    }
    result = run_nemo_gym_rollout_sync(
        policy_generation=policy_generation,
        input_batch=input_batch,
        tokenizer=tokenizer,
        task_to_env=task_to_env,
        max_seq_len=master_config.policy["max_total_sequence_length"],
        generation_config=generation_config,
        log_full_result_tables=True,
        max_rollout_turns=None,
        greedy=False,
        reward_penalty_config=getattr(master_config, "reward_penalties", None),
        thinking_tags=get_nemo_gym_thinking_tags(master_config.env),
    )
    agent_name = input_batch["extra_env_info"][0]["agent_ref"]["name"]
    table_key = f"{agent_name}/full_result"
    table = result.rollout_metrics.pop(table_key, None)
    if table is None or len(table.data) != 1:
        raise ValueError(f"Expected one raw Gym result in {table_key}")
    full_result = json.loads(table.data[0][0])
    return result.final_batch.to("cpu"), result.rollout_metrics, full_result


def _aggregate_sample_metrics(metrics_list: list[dict[str, Any]]) -> dict[str, Any]:
    """Mean-aggregate numeric per-sample rollout metrics into one group dict.

    Sample-level admission produces one rollout_metrics dict per episode; the
    group payload keeps a single dict (as a whole-group call would), so numeric
    fields are averaged and non-numeric fields dropped. Non-finite values are
    skipped too: single-sample calls report NaN for */stddev-style statistics,
    which would otherwise poison the group mean.
    """
    merged: dict[str, list[float]] = {}
    for m in metrics_list:
        for k, v in (m or {}).items():
            if isinstance(v, (bool, int, float)) and math.isfinite(float(v)):
                merged.setdefault(k, []).append(float(v))
    out: dict[str, Any] = {k: sum(v) / len(v) for k, v in merged.items()}
    out["aggregated_from_samples"] = float(len(metrics_list))
    return out


def assemble_group_payload(
    dataset_idx: int,
    input_batch: BatchedDataDict,
    sample_batches: list[BatchedDataDict],
    sample_metrics: list[dict[str, Any]],
    full_results: list[dict[str, Any]],
) -> dict[str, Any]:
    """Reassemble per-sample rollout results into one group payload.

    ``sample_batches`` must be in generation order; concatenating them yields
    the same batch content a single whole-group rollout call produces
    (semantically identical for all consumers; byte layout may differ, e.g.
    ``from_batches`` sorts keys).
    """
    final_batch = BatchedDataDict.from_batches(sample_batches)
    resolved = []
    for full_result in full_results:
        value = full_result.get("resolved")
        if value not in (0, 1, False, True):
            raise ValueError(
                "Gym result is missing binary resolved; refusing to bank "
                "a group without verifier pass labels"
            )
        resolved.append(bool(value))
    final_batch["verifier_resolved"] = torch.tensor(resolved, dtype=torch.bool)
    if len(full_results) != len(sample_batches):
        raise ValueError("Full Gym result count does not match trajectory count")
    payload = build_group_payload(
        dataset_idx, input_batch, final_batch, _aggregate_sample_metrics(sample_metrics)
    )
    payload["full_results"] = full_results
    return payload


def _vllm_engine_stats_line(policy_generation: Any, interval_s: float) -> str:
    """One-line engine summary from the in-process vLLM metrics logger.

    The async worker samples cumulative ``vllm:generation_tokens`` plus
    running/waiting/KV gauges every ``interval_s`` (see
    ``enable_vllm_metrics_logger``); tokens/s over the window is the counter
    delta divided by the window span. Metrics are cleared after reading so each
    heartbeat reports a fresh window. Best-effort: any failure returns "".
    """
    try:
        m = policy_generation.get_vllm_logger_metrics()
        parts = []
        tputs = []
        for series in (m.get("generation_tokens") or {}).values():
            if len(series) >= 2:
                tputs.append((series[-1] - series[0]) / ((len(series) - 1) * interval_s))
        if tputs:
            parts.append(f"{sum(tputs):.0f} gen tok/s")
        running = [v for s in (m.get("inflight_batch_sizes") or {}).values() for v in s]
        if running:
            parts.append(f"running {sum(running) / len(running):.0f} reqs")
        kv = [v[-1] for v in (m.get("kv_cache_usage_perc") or {}).values() if v]
        if kv:
            parts.append(f"kv {100 * max(kv):.1f}%")
        policy_generation.clear_vllm_logger_metrics()
        return " | ".join(parts)
    except Exception:
        return ""


def collect_rollouts(
    policy_generation: Any,
    tokenizer: Any,
    task_to_env: dict[str, Any],
    master_config: Any,
    dataset: Any,
    collection: dict[str, Any],
) -> dict[str, Any]:
    """Generate this shard's assigned prompt groups and write them to disk.

    Admission control is per SAMPLE: one worker thread per in-flight episode
    (bounded by ``collection.max_inflight_samples``), each blocking in its own
    single-sample ``run_async_nemo_gym_rollout`` call. When an episode
    finishes, the freed slot is immediately backfilled by the next pending
    sample from any group — a slow straggler holds one slot, not its whole
    group's worth. Completed samples are reassembled (in generation order)
    into the atomic per-group file; a group is banked all-or-nothing, and any
    sample failure fails the whole group (its remaining samples are skipped
    and partials discarded; resume regenerates it).

    Returns a stats dict (also written to ``shard_summary.json``).
    """
    out_dir = Path(collection["out_dir"])
    shard_dir = out_dir / f"shard_{collection['shard_id']:03d}"
    shard_dir.mkdir(parents=True, exist_ok=True)

    meta_path = shard_dir / "meta.json"
    meta = build_shard_meta(master_config, collection, tokenizer, len(dataset))
    if meta_path.exists():
        saved_meta = json.loads(meta_path.read_text())
        identity_keys = (
            "model_name",
            "tokenizer_name_or_path",
            "chat_template_sha256",
            "max_total_sequence_length",
            "dataset_path",
            "dataset_sha256",
            "dataset_len",
            "gens_per_prompt",
        )
        mismatched = [
            key for key in identity_keys if saved_meta.get(key) != meta.get(key)
        ]
        if mismatched:
            raise ValueError(
                f"Refusing to mix incompatible rollouts in {shard_dir}: "
                f"metadata changed for {mismatched}"
            )
    else:
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
    print(f"📁 Shard dir: {shard_dir}")

    todo = assigned_indices(
        len(dataset),
        collection["shard_id"],
        collection["num_shards"],
        index_start=collection["index_start"],
        index_end=collection["index_end"],
        max_groups=collection["max_groups"],
    )
    # Scan ALL shard dirs (not just ours): num_shards may differ from a prior
    # submission, and a group banked under any layout must not be regenerated.
    existing = existing_group_indices_all(out_dir)
    skipped = [i for i in todo if i in existing]
    pending = [i for i in todo if i not in existing]
    print(
        f"🎯 Assigned {len(todo)} groups "
        f"(resume: {len(skipped)} already done, {len(pending)} to generate) "
        f"x {collection['gens_per_prompt']} gens/prompt"
    )

    gens_per_prompt = collection["gens_per_prompt"]
    max_inflight_samples = collection["max_inflight_samples"]
    log_every = max(1, int(collection["log_every"]))
    inflight = threading.Semaphore(max_inflight_samples)
    state_lock = threading.Lock()
    manifest_path = shard_dir / "manifest.jsonl"
    # Per-group assembly state, keyed by dataset_idx. Bounded: samples are
    # submitted group-by-group, so at most ~max_inflight_samples/gens + 1
    # groups are open at once. Failed groups keep only a tombstone in
    # `failed_groups` (partials are dropped immediately).
    groups_state: dict[int, dict[str, Any]] = {}
    failed_groups: set[int] = set()
    stats = {
        "completed": 0,
        "failed": 0,
        "samples": 0,
        "reward_sum": 0.0,
        "consecutive_failures": 0,
        "inflight_samples": 0,
    }
    start_time = time.perf_counter()
    abort = threading.Event()

    def _fail_group(dataset_idx: int, err: Exception, where: str) -> None:
        """First failure fails the whole group; partials are discarded."""
        with state_lock:
            if dataset_idx in failed_groups:
                return
            failed_groups.add(dataset_idx)
            groups_state.pop(dataset_idx, None)
            stats["failed"] += 1
            stats["consecutive_failures"] += 1
            consecutive = stats["consecutive_failures"]
        print(f"❌ group {dataset_idx} failed ({where}): {err}", flush=True)
        if consecutive >= collection["max_consecutive_failures"]:
            print(
                f"🛑 {consecutive} consecutive group failures — aborting "
                "collection (systemic problem, e.g. dead engine or gym).",
                flush=True,
            )
            abort.set()

    def _sample_worker(
        dataset_idx: int, sample_idx: int, sample_batch: BatchedDataDict
    ) -> None:
        try:
            final_one, metrics_one, full_result = _run_rollout_batch(
                policy_generation, sample_batch, tokenizer, task_to_env, master_config
            )
            complete_group = None
            with state_lock:
                group = groups_state.get(dataset_idx)
                if group is None:
                    return  # group already failed; discard this sample
                group["parts"][sample_idx] = final_one
                group["metrics"][sample_idx] = metrics_one
                group["full_results"][sample_idx] = full_result
                if len(group["parts"]) == group["expected"]:
                    groups_state.pop(dataset_idx)
                    complete_group = group
            if complete_group is None:
                return
            ordered = [
                complete_group["parts"][j] for j in range(complete_group["expected"])
            ]
            ordered_metrics = [
                complete_group["metrics"][j] for j in range(complete_group["expected"])
            ]
            ordered_full_results = [
                complete_group["full_results"][j]
                for j in range(complete_group["expected"])
            ]
            payload = assemble_group_payload(
                dataset_idx,
                complete_group["input_batch"],
                ordered,
                ordered_metrics,
                ordered_full_results,
            )
            write_group_atomic(shard_dir, dataset_idx, payload)
            # From here the group is banked on disk; a bookkeeping error must
            # not mark it failed (that would double-count it and bump the
            # consecutive-failure abort counter for a successful group).
            try:
                rewards = payload["batch"]["total_reward"]
                record = {
                    "dataset_idx": dataset_idx,
                    "num_samples": int(payload["batch"].size),
                    "pass_count": int(payload["batch"]["verifier_resolved"].sum()),
                    "reward_mean": float(rewards.float().mean()),
                    "truncated_frac": float(
                        payload["batch"]["truncated"].float().mean()
                    ),
                    "seconds": round(
                        time.perf_counter() - complete_group["start"], 1
                    ),
                }
                with state_lock:
                    stats["completed"] += 1
                    stats["samples"] += record["num_samples"]
                    stats["reward_sum"] += (
                        record["reward_mean"] * record["num_samples"]
                    )
                    stats["consecutive_failures"] = 0
                    done = stats["completed"]
                    inflight_now = stats["inflight_samples"]
                    with open(manifest_path, "a") as f:
                        f.write(json.dumps(record) + "\n")
                if done % log_every == 0 or done == len(pending):
                    elapsed = time.perf_counter() - start_time
                    rate = stats["samples"] / max(elapsed, 1e-6) * 3600
                    engine_line = _vllm_engine_stats_line(
                        policy_generation,
                        master_config.policy["generation"]["vllm_cfg"].get(
                            "vllm_metrics_logger_interval", 0.5
                        ),
                    )
                    print(
                        f"✅ [{done}/{len(pending)}] group {dataset_idx}: "
                        f"reward={record['reward_mean']:.3f} "
                        f"({record['seconds']}s) | {rate:.0f} samples/h | "
                        f"{inflight_now} samples in flight"
                        + (f" | vLLM: {engine_line}" if engine_line else ""),
                        flush=True,
                    )
            except Exception as e:
                print(
                    f"⚠️ post-write bookkeeping failed for group {dataset_idx} "
                    f"(group file IS banked): {e}",
                    flush=True,
                )
        except Exception as e:
            import traceback

            traceback.print_exc()
            _fail_group(dataset_idx, e, "rollout")
        finally:
            with state_lock:
                stats["inflight_samples"] -= 1
            inflight.release()

    threads: set[threading.Thread] = set()
    for dataset_idx in pending:
        if abort.is_set():
            break
        # Build the group's input batch in the submit loop (cheap CPU work);
        # repeat_interleave deep-copies list fields, so each sample slice owns
        # its rows (run_async_nemo_gym_rollout mutates extra_env_info in place).
        # A deterministically-bad dataset row must not wedge the shard: count it
        # as a group failure and move on instead of crashing the whole job.
        try:
            datum = dataset[dataset_idx]
            single = rl_collate_fn([datum])
            repeated_batch = single.repeat_interleave(gens_per_prompt)
        except Exception as e:
            _fail_group(dataset_idx, e, "input batch")
            continue
        with state_lock:
            groups_state[dataset_idx] = {
                "parts": {},
                "metrics": {},
                "full_results": {},
                "expected": gens_per_prompt,
                "input_batch": repeated_batch,
                "start": time.perf_counter(),
            }
        for sample_idx in range(gens_per_prompt):
            if abort.is_set():
                break
            with state_lock:
                if dataset_idx in failed_groups:
                    break  # a sibling sample failed; skip the rest of the group
            sample_batch = repeated_batch.slice(sample_idx, sample_idx + 1)
            inflight.acquire()
            with state_lock:
                dead = abort.is_set() or dataset_idx in failed_groups
                if not dead:
                    stats["inflight_samples"] += 1
            if dead:
                inflight.release()
                break
            t = threading.Thread(
                target=_sample_worker,
                args=(dataset_idx, sample_idx, sample_batch),
                daemon=True,
            )
            threads.add(t)
            t.start()
        # Prune finished thread objects so the set stays bounded on long runs.
        if len(threads) > 4 * max_inflight_samples:
            for t in [t for t in threads if not t.is_alive()]:
                threads.discard(t)

    for t in list(threads):
        t.join()

    elapsed = time.perf_counter() - start_time
    summary = {
        "assigned": len(todo),
        "skipped_existing": len(skipped),
        "completed": stats["completed"],
        "failed": stats["failed"],
        "samples": stats["samples"],
        "mean_reward": (
            stats["reward_sum"] / stats["samples"] if stats["samples"] else None
        ),
        "elapsed_s": round(elapsed, 1),
        "samples_per_hour": (
            round(stats["samples"] / elapsed * 3600, 1) if elapsed > 0 else None
        ),
        "aborted": abort.is_set(),
        "remaining": len(pending) - stats["completed"],
    }
    with open(shard_dir / "shard_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"🏁 Collection summary: {json.dumps(summary)}")
    return summary
