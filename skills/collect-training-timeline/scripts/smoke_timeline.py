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

"""Exercise timeline writers in a revision-specific source overlay."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import time
from pathlib import Path
from types import ModuleType


def load_module(name: str, path: Path) -> ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Overlay root containing nemo_rl/ and 3rdparty/.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    timer_module = load_module(
        "timeline_overlay_timer",
        args.source_root / "nemo_rl/utils/timer.py",
    )
    helper_module = load_module(
        "timeline_overlay_openhands_helper",
        args.source_root
        / (
            "3rdparty/Gym-workspace/Gym/responses_api_agents/swe_agents/"
            "swe_openhands_setup/OpenHands/openhands/utils/nemo_gym_timeline.py"
        ),
    )

    rollout_dir = args.output_dir / "nemo_gym" / "smoke_rollout"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    training_path = args.output_dir / "training_timeline.jsonl"
    lifecycle_path = args.output_dir / "rollout_lifecycle.jsonl"
    rollout_path = rollout_dir / "timeline.jsonl"
    training_path.write_text("", encoding="utf-8")
    rollout_path.write_text("", encoding="utf-8")

    now_us = time.time_ns() // 1_000
    lifecycle_records = (
        {
            "event": "rollout_buffered",
            "ts": now_us,
            "rollout_id": "smoke_rollout",
            "generation_weight_version": 0,
            "target_weight_version": 1,
            "target_display_step": 2,
        },
        {
            "event": "rollout_consumed",
            "ts": now_us + 1,
            "rollout_id": "smoke_rollout",
            "generation_weight_version": 0,
            "target_weight_version": 1,
            "target_display_step": 2,
            "consumed_weight_version": 0,
            "consumed_display_step": 1,
            "disposition": "consumed_early",
        },
    )
    lifecycle_path.write_text(
        "".join(json.dumps(record) + "\n" for record in lifecycle_records),
        encoding="utf-8",
    )

    os.environ["NRL_TRAINING_TIMELINE"] = "1"
    os.environ["NRL_TRAINING_TIMELINE_FPATH"] = str(training_path)
    timer = timer_module.Timer()
    timer.start("total_step_time")
    with timer.time("exposed_generation"):
        time.sleep(0.002)
    with timer.time("policy_training"):
        time.sleep(0.002)

    os.environ["NEMO_GYM_TIMELINE_FPATH"] = str(rollout_path)
    os.environ["NEMO_GYM_ROLLOUT_ID"] = "smoke_rollout"
    span = helper_module.nemo_gym_timeline_span
    cache_metrics = helper_module.nemo_gym_cache_metrics(
        {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "prompt_tokens_details": {"cached_tokens": 75},
        }
    )
    assert cache_metrics == {
        "prompt_tokens": 100,
        "decode_tokens": 20,
        "cached_prompt_tokens": 75,
        "uncached_prompt_tokens": 25,
        "kv_cache_hit_rate": 0.75,
    }
    zero_cache_metrics = helper_module.nemo_gym_cache_metrics(
        {
            "prompt_tokens": 100,
            "completion_tokens": 0,
            "prompt_tokens_details": {"cached_tokens": 0},
        }
    )
    assert zero_cache_metrics == {
        "prompt_tokens": 100,
        "decode_tokens": 0,
        "cached_prompt_tokens": 0,
        "uncached_prompt_tokens": 100,
        "kv_cache_hit_rate": 0.0,
    }
    no_cache_details_metrics = helper_module.nemo_gym_cache_metrics(
        {"prompt_tokens": 80, "completion_tokens": 12}
    )
    assert no_cache_details_metrics == {
        "prompt_tokens": 80,
        "decode_tokens": 12,
    }
    llm_timing_metrics = helper_module.nemo_gym_llm_timing_metrics(
        {
            "nemo_llm_arrival_ts_us": now_us + 100,
            "nemo_llm_queued_ts_us": now_us + 200,
            "nemo_llm_first_scheduled_ts_us": now_us + 300,
            "nemo_llm_first_token_ts_us": now_us + 400,
            "nemo_llm_last_token_ts_us": now_us + 500,
            "nemo_llm_response_ready_ts_us": now_us + 600,
            "nemo_gym_llm_response_received_ts_us": now_us + 700,
            "nemo_gym_tokenize_duration_us": 0,
            "nemo_gym_response_ready_ts_us": now_us + 800,
        }
    )
    assert llm_timing_metrics == {
        "llm_arrival_ts_us": now_us + 100,
        "queued_ts_us": now_us + 200,
        "first_scheduled_ts_us": now_us + 300,
        "first_token_ts_us": now_us + 400,
        "last_token_ts_us": now_us + 500,
        "llm_response_ready_ts_us": now_us + 600,
        "gym_llm_response_received_ts_us": now_us + 700,
        "gym_tokenize_duration_us": 0,
        "gym_response_ready_ts_us": now_us + 800,
    }
    detailed_client_timing_metrics = {
        "openhands_response_received_ts_us": now_us + 900,
        "openhands_parse_validate_duration_us": 0,
    }
    # PD-disaggregation legs; older helpers predate them.
    disagg_metrics: dict = {}
    if hasattr(helper_module, "nemo_gym_disagg_metrics"):
        disagg_metrics = helper_module.nemo_gym_disagg_metrics(
            {
                "nemo_fe_recv_ts_us": now_us + 10,
                "nemo_fe_fwd_ts_us": now_us + 20,
                "nemo_ctx_arrival_ts_us": now_us + 30,
                "nemo_ctx_queued_ts_us": now_us + 40,
                "nemo_ctx_first_scheduled_ts_us": now_us + 50,
                "nemo_ctx_done_ts_us": now_us + 90,
                # A full miss with prefill from position 0 must survive.
                "nemo_ctx_cached_tokens": 0,
                "nemo_ctx_computed_tokens": 100,
                "nemo_ctx_first_begin": 0,
                "nemo_ctx_num_chunks": 1,
                "nemo_ctx_num_chunks_bogus": True,
            }
        )
        assert disagg_metrics == {
            "frontend_recv_ts_us": now_us + 10,
            "frontend_fwd_ts_us": now_us + 20,
            "ctx_arrival_ts_us": now_us + 30,
            "ctx_queued_ts_us": now_us + 40,
            "ctx_first_scheduled_ts_us": now_us + 50,
            "ctx_done_ts_us": now_us + 90,
            "ctx_cached_prompt_tokens": 0,
            "ctx_computed_tokens": 100,
            "ctx_first_begin": 0,
            "ctx_num_chunks": 1,
        }, disagg_metrics

    for name, category in (
        ("ray_queue", "orchestration"),
        ("agent_container_spinup", "orchestration"),
        ("openhands_run", "orchestration"),
        ("create_runtime", "runtime"),
        ("connect_runtime", "runtime"),
        ("initialize_runtime", "runtime"),
        ("prepare_instruction", "agent"),
        ("CmdRunAction", "tool"),
        ("run_controller", "agent"),
        ("extract_patch", "agent"),
        ("runtime_cleanup", "runtime"),
        ("eval_container_spinup", "orchestration"),
        ("final_eval", "orchestration"),
    ):
        with span(name, category):
            time.sleep(0.001)

    with span("model_call", "model", turn=1) as model_call_args:
        time.sleep(0.001)
        model_call_args["generation_replica_idx"] = 0
        model_call_args.update(cache_metrics)
        model_call_args.update(llm_timing_metrics)
        model_call_args.update(detailed_client_timing_metrics)
        model_call_args.update(disagg_metrics)

    with span("model_call", "model", turn=2) as model_call_args:
        time.sleep(0.001)
        model_call_args["generation_replica_idx"] = 0
        model_call_args.update(zero_cache_metrics)
        model_call_args.update(llm_timing_metrics)
        model_call_args.update(detailed_client_timing_metrics)
        model_call_args.update(disagg_metrics)

    with span("model_call", "model", turn=3) as model_call_args:
        time.sleep(0.001)
        model_call_args["generation_replica_idx"] = 0
        model_call_args.update(no_cache_details_metrics)
        model_call_args.update(llm_timing_metrics)
        model_call_args.update(detailed_client_timing_metrics)
        model_call_args.update(disagg_metrics)

    timer.stop("total_step_time")
    print(f"training_timeline={training_path}")
    print(f"rollout_lifecycle={lifecycle_path}")
    print(f"rollout_root={args.output_dir / 'nemo_gym'}")


if __name__ == "__main__":
    main()
