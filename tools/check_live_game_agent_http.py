# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run corrected agents over HTTP against an existing allocation's model/resources.

Creates separate loopback agent servers; does not modify the running services.
Only scalar summaries are persisted. Run inside the owning allocation/container.
"""

import argparse
import asyncio
import json
import math
import os
from pathlib import Path
import socket
from types import SimpleNamespace


def check_real_mixed_collation(raw_results, model_config):
    """Process actual HTTP results with the real tokenizer/image processor."""
    import torch
    from transformers import AutoProcessor, AutoTokenizer
    from nemo_rl.data.llm_message_utils import (
        batched_message_log_to_flat_message,
        message_log_to_flat_messages,
    )
    from nemo_rl.data.multimodal_utils import PackedTensor
    from nemo_rl.distributed.batched_data_dict import (
        BatchedDataDict,
        _prepare_multimodal_sharing,
    )
    from nemo_rl.environments.nemo_gym import NemoGym
    from nemo_rl.experience.rollouts import (
        attach_initial_nemo_gym_image_payloads,
        _reattach_original_multimodal_payloads,
    )

    torch.set_num_threads(2)
    model = model_config.get("tokenizer") or model_config["model"]
    processor = AutoProcessor.from_pretrained(model, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)
    adapter = SimpleNamespace(
        _processor=processor, _pad_dynamic_image_shapes=True, cfg={}
    )
    postprocess = (
        NemoGym.__ray_metadata__.modified_class._postprocess_nemo_gym_to_nemo_rl_result
    )
    originals, logs = {}, []
    for row, raw in raw_results:
        game = row["env_id"]
        if game not in originals:
            batch = BatchedDataDict(
                {
                    "message_log": [
                        [
                            {
                                "role": "user",
                                "token_ids": torch.tensor([], dtype=torch.long),
                            }
                        ]
                    ],
                    "extra_env_info": [row],
                }
            )
            attach_initial_nemo_gym_image_payloads(
                batch,
                processor,
                env_config={"nemo_gym": {"pad_dynamic_image_shapes": True}},
            )
            _prepare_multimodal_sharing(batch["message_log"])
            originals[game] = batch["message_log"][0]
        result = postprocess(
            adapter, row, raw, tokenizer, include_initial_multimodal_data=False
        )
        _reattach_original_multimodal_payloads([result], [originals[game]])
        logs.append(result["message_log"])
    flat_rows = [message_log_to_flat_messages(log) for log in logs]
    values = [row["pixel_values"] for row in flat_rows]
    assert any(p.deduplication_enabled for p in values), "No shared-image row exercised"
    assert any(not p.deduplication_enabled and len(p) > 1 for p in values), (
        "No legacy multiturn row exercised"
    )
    flat, lengths = batched_message_log_to_flat_message(
        logs, pad_value_dict={"token_ids": 0}
    )
    for key, packed in flat.items():
        if not isinstance(packed, PackedTensor):
            continue
        assert len(packed) == len(logs), key
        for index, row in enumerate(flat_rows):
            expected = row[key].as_tensor() if key in row else None
            actual = packed.slice([index]).as_tensor()
            if expected is None:
                assert actual is None
            else:
                torch.testing.assert_close(actual, expected)
    assert lengths.tolist() == [sum(len(m["token_ids"]) for m in log) for log in logs]
    print(
        f"REAL_MIXED_POSTPROCESS_COLLATION_PASS rows={len(logs)} shared_and_legacy_multiturn=true",
        flush=True,
    )


async def check(args):
    # Optional cluster dependencies live in the selected Gym service interpreter.
    import aiohttp
    import psutil
    import uvicorn
    from omegaconf import OmegaConf

    source_env = psutil.Process(args.config_pid).environ()
    os.environ["NEMO_GYM_CONFIG_DICT"] = source_env["NEMO_GYM_CONFIG_DICT"]
    from nemo_gym import server_utils
    from nemo_gym.global_config import (
        get_first_server_config_dict,
        get_global_config_dict,
    )
    from nemo_gym.server_utils import ServerClient
    from responses_api_agents.gymv_agent.app import GymVAgent, GymVAgentConfig
    from responses_api_agents.visgym_agent.app import (
        TextActionAgent,
        TextActionAgentConfig,
    )

    root = Path(__file__).resolve().parents[1]
    assert Path(server_utils.__file__).resolve().is_relative_to(root)
    global_config = get_global_config_dict()
    client = ServerClient.load_from_global_config()
    selected = {}
    agent_names = {"gym_v_agent", "visgym_agent"}
    if args.collate:
        agent_names.add("image_tools_simple_agent")
    with args.manifest.open() as stream:
        for line in stream:
            row = json.loads(line)
            if row.get("agent_ref", {}).get("name") in agent_names and (
                not args.game or row["env_id"] in args.game
            ):
                selected.setdefault(row["env_id"], row)
    assert selected, "No selected game rows"
    if args.collate:
        # Match the real collector's wire representation for local image paths.
        from nemo_rl.data.multimodal_utils import encode_images_in_examples

        encode_images_in_examples(list(selected.values()))
    if not args.game and not args.collate:
        assert len(selected) == 34, len(selected)
    raw_results = []
    servers, tasks, sockets, urls = [], [], [], {}
    try:
        for name, agent_cls, config_cls in (
            ("gym_v_agent", GymVAgent, GymVAgentConfig),
            ("visgym_agent", TextActionAgent, TextActionAgentConfig),
        ):
            cfg = OmegaConf.to_container(
                get_first_server_config_dict(global_config, name), resolve=True
            )
            agent = agent_cls(
                config=config_cls.model_validate({**cfg, "name": name}),
                server_client=client,
            )
            app = agent.setup_webserver()
            sock = socket.socket()
            sock.bind(("127.0.0.1", 0))
            sock.listen(128)
            sockets.append(sock)
            urls[name] = f"http://127.0.0.1:{sock.getsockname()[1]}/run"
            server = uvicorn.Server(
                uvicorn.Config(app, log_level="warning", access_log=False)
            )
            servers.append(server)
            tasks.append(asyncio.create_task(server.serve(sockets=[sock])))
        async with asyncio.timeout(30):
            while not all(server.started for server in servers):
                if any(task.done() for task in tasks):
                    await asyncio.gather(*tasks)
                    raise RuntimeError("Probe agent server exited during startup")
                await asyncio.sleep(0.05)
        semaphore = asyncio.Semaphore(8)
        if args.collate:
            image_cfg = get_first_server_config_dict(
                global_config, "image_tools_simple_agent"
            )
            urls["image_tools_simple_agent"] = (
                f"http://{image_cfg.host}:{image_cfg.port}/run"
            )
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=600)
        ) as session:

            async def rollout(index, row):
                async with semaphore:
                    name = row["agent_ref"]["name"]
                    body = {
                        **row,
                        "_ng_task_index": 1000000 + index,
                        "_ng_rollout_index": 0,
                    }
                    # Match collect_nemo_gym_rollouts' explicit sampling stamp;
                    # raw manifest rows intentionally omit these runtime values.
                    body["responses_create_params"] = {
                        **row["responses_create_params"],
                        "temperature": args.temperature,
                        "top_p": args.top_p,
                    }
                    started = asyncio.get_running_loop().time()
                    async with session.post(urls[name], json=body) as response:
                        if response.status != 200:
                            raise RuntimeError(
                                f"{row['env_id']} HTTP {response.status}"
                            )
                        result = await response.json()
                    reward = result["reward"]
                    assert math.isfinite(reward)
                    output = result["response"]
                    assert output["output"], "No real model output"
                    record = {
                        "game": row["env_id"],
                        "agent": name,
                        "reward": reward,
                        "duration_seconds": asyncio.get_running_loop().time() - started,
                        "metadata": output.get("metadata"),
                        "status": "PASS",
                    }
                    print(json.dumps(record), flush=True)
                    if args.collate:
                        raw_results.append((body, result))
                    return record

            records = await asyncio.gather(
                *(
                    rollout(i, row)
                    for i, row in enumerate(list(selected.values()) * args.repeats)
                )
            )
        if args.collate:
            assert {row["agent_ref"]["name"] for row, _ in raw_results} == {
                "gym_v_agent",
                "visgym_agent",
                "image_tools_simple_agent",
            }
            check_real_mixed_collation(
                raw_results, get_first_server_config_dict(global_config, "policy_model")
            )
        args.result.write_text(
            json.dumps({"source": str(root), "results": records}, indent=2) + "\n"
        )
        print(f"LIVE_GAME_AGENT_HTTP_PASS games={len(records)}", flush=True)
    finally:
        for server in servers:
            server.should_exit = True
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        for sock in sockets:
            sock.close()
        if server_utils._GLOBAL_AIOHTTP_CLIENT is not None:
            await server_utils._GLOBAL_AIOHTTP_CLIENT.close()
            server_utils._GLOBAL_AIOHTTP_CLIENT = None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-pid", type=int, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--game", action="append")
    parser.add_argument("--temperature", type=float, required=True)
    parser.add_argument("--top-p", type=float, required=True)
    parser.add_argument("--collate", action="store_true")
    parser.add_argument("--repeats", type=int, default=1)
    asyncio.run(check(parser.parse_args()))
