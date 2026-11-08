# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 18:49:53 2026

@author: degert
"""

import argparse
import concurrent.futures
import copy
#import functools
#import itertools
import json
import math
#import multiprocessing
import os
#import random
import re
#from typing import Any, Optional, TypedDict

from openai import OpenAI
from tqdm import tqdm

#from nemo_rl.data.interfaces import DatumSpec, TaskDataSpec
#from nemo_rl.data.datasets import AllTaskProcessedDataset
from nemo_rl.data.datasets.preference_datasets.reward_benchmarks import (
    JudgeBenchDataset,
    RMBenchDataset,
    RewardBench2Dataset,
    HelpSteer3LocalDataset,
)
from nemo_rl.data.genrm_prompts import (
    DWRL_POINTWISE_RUBRIC_TEMPLATE,
    GENRM_POINTWISE_DEFAULT_PRINCIPLE,
    build_openai_judge_content,
)


#SCORE_PROMPT = 'Based on your evaluation, is the response helpful and harmless? Answer with only a single word: yes or no'
SCORE_PROMPT = 'Based on your evaluation, is the original response good? Answer with only a single word: yes or no'
YES_TOKEN = "yes"
def get_json_response(response):
    if response is None:
        return None
    
    resp_no_thinking1 = re.sub(r"(?i)(?s)(\<think\>)(.*?)(\<\/think\>)", "", response).strip()
    resp_no_thinking2 = re.sub(r"(?i)(?s)(.*?)(\<\/think\>)", "", resp_no_thinking1).strip()
    
    try:
        jp = json.loads(re.search(r"(?i)(?s)(\{\n)(.*?)(\n\})", resp_no_thinking2).group(0))
        return jp["overall"] if isinstance(jp.get("overall"), dict) else jp
    except:
        return resp_no_thinking2


def get_score_from_vllm(samp, response, principle):
    try:
        prompt_content = build_openai_judge_content(DWRL_POINTWISE_RUBRIC_TEMPLATE, samp['context'], response=response, principle=principle)
        completion = client.chat.completions.create(
          model="model",
          messages=[{"role": "user", "content": prompt_content}],
          temperature=args.temperature,
          top_p=args.top_p,
          max_tokens=args.max_tokens,
          #extra_body={"top_k": 20},
          extra_body={"chat_template_kwargs": {"enable_thinking": True}},
          stream=False
        )
        
        thought = completion.choices[0].message.content
        if thought is None:
            return -127, None
        # The BT call must show the model its generation as training does (dwrl.py stage 2), so it needs the raw
        # generation: a reasoning parser on the server would split the reasoning off.
        message = completion.choices[0].message
        if getattr(message, "reasoning", None) or getattr(message, "reasoning_content", None):
            raise ValueError("the vLLM server returns the reasoning separately: serve the judge without --reasoning-parser")
        # A template whose generation prompt opens the reasoning (Super 3.5: "<think>\n") returns the generation without
        # that opening: put it back so the turn reads as the model produced it.
        assistant_turn = thought if thought.lstrip().startswith("<think>") else "<think>\n" + thought

        completion = client.chat.completions.create(
          model="model",
          messages=[{"role": "user", "content": prompt_content}] + [{"role": "assistant", "content": assistant_turn}] + [{"role": "user", "content": SCORE_PROMPT}],
          temperature=1.0,
          top_p=1.0,
          max_tokens=1,
          #extra_body={"top_k": 20},
          # truncate_history_thinking would cut the reasoning out of the earlier assistant turn (Super 3.5's template)
          extra_body={"chat_template_kwargs": {"enable_thinking": False, "truncate_history_thinking": False}},
          logprobs=True,
          top_logprobs=20,
          stream=False
        )
        
        score = -999
        for possible_tokens in completion.choices[0].logprobs.content[-1].top_logprobs:
            if possible_tokens.token == YES_TOKEN:
                score = math.exp(possible_tokens.logprob)
                break
        if score == -999:
            for possible_tokens in completion.choices[0].logprobs.content[-1].top_logprobs:
                if possible_tokens.token.lower().strip() == YES_TOKEN:
                    score = math.exp(possible_tokens.logprob)
                    break
        
        if score == -999:
            print("##################")
            print("raw response: ", completion.choices[0].message.content)
            for possible_tokens in completion.choices[0].logprobs.content[-1].top_logprobs:
                print(possible_tokens.token, math.exp(possible_tokens.logprob))
            print("##################")
        
        return score, thought
    except Exception as e:
        print("ERROR calling vllm: ", e, flush=True)
        return -255, None


def benchmark_single(samp, idx):
    response_1 = samp["response_1"]
    response_2 = samp["response_2"]
    principle = samp.get("principle", GENRM_POINTWISE_DEFAULT_PRINCIPLE)
    # As in training (run_dwrl_vlm.py): the data's principles are written for pairwise judging; fix the header line for a single response.
    principle = principle.replace("Evaluate both responses", "Evaluate the Response")

    score_1, thought_1 = get_score_from_vllm(samp, response_1, principle)
    score_2, thought_2 = get_score_from_vllm(samp, response_2, principle)
    
    #gt = 0 if overall_preference < 0 else 1
    
    #samp_copy = copy.deepcopy(samp)
    #samp_copy['score_1'] = score_1
    #samp_copy['score_2'] = score_2
    #samp_copy['gt'] = gt
    
    payload = {}
    payload['idx'] = idx
    payload['prediction_1'] = get_json_response(thought_1)
    payload['prediction_2'] = get_json_response(thought_2)
    payload['metadata'] = copy.deepcopy(samp)
    payload['score_1'] = score_1
    payload['score_2'] = score_2
    
    return payload


def parse_args():
    parser = argparse.ArgumentParser()
    
    parser.register('type', 'bool', (lambda x: x.lower() in ("true", "1")))
    optional = parser._action_groups.pop()
    required = parser.add_argument_group('required arguments')
    
    required.add_argument('-d', '--dataset',
                        required=True,
                        type=str,
                        choices=["rewardbench2", "judgebench", "rmbench", "hs3local", "hs4local"],
                        help='which benchmark to run')
    required.add_argument('-s', '--server',
                        required=True,
                        type=str,
                        help='hostname where the model is served')
    required.add_argument('-o', '--output',
                        required=True,
                        type=str,
                        default=None,
                        help='path to output file to save')
    
    optional.add_argument('--temperature',
                        default=1.0,
                        type=float,
                        help='temperature for generation')
    optional.add_argument('--top_p',
                        default=0.95,
                        type=float,
                        help='top_p for generation')
    optional.add_argument('--top_k',
                        default=20,
                        type=int,
                        help='top_k for generation')
    optional.add_argument('--max_tokens',
                        default=16384,
                        type=int,
                        help='max_tokens for generation')
    optional.add_argument('--num_workers',
                        type=int,
                        default=64,
                        help="No of simultaneous queries to send to the API. Defaults to 64.",
                        required=False)
    optional.add_argument('--port',
                        type=int,
                        default=8000,
                        help="Port for the server where the model is hosted",
                        required=False)
    optional.add_argument('--hs_eval_file',
                        type=str,
                        default=None,
                        help="Location of HS3 eval file to use",
                        required=False)
    optional.add_argument('--media_path',
                        type=str,
                        default=None,
                        help="Location of media folder for multimodal",
                        required=False)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    if os.path.isfile(args.output):
        print("Output file already exists, aborting", flush=True)
        exit(0)
        
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    output_file_tmp = args.output + ".tmp"
    if os.path.exists(output_file_tmp):
        with open(output_file_tmp, "r", encoding="utf_8") as fr:
            existing_file = [json.loads(line) for line in fr]
        existing_file = sorted(existing_file, key=lambda x: x['idx'])
        existing_ids = {x['idx'] for x in existing_file}
    else:
        existing_ids = {}

    client = OpenAI(
      base_url = f"http://{args.server}:{args.port}/v1",
      api_key = "xxx"
    )
    
    if args.dataset == "judgebench":
        dataset_loader = JudgeBenchDataset()
    elif args.dataset == "rmbench":
        dataset_loader = RMBenchDataset()
    elif args.dataset == "rewardbench2":
        dataset_loader = RewardBench2Dataset()
    elif args.dataset in ["hs3local", "hs4local"]:
        dataset_loader = HelpSteer3LocalDataset(args.hs_eval_file, media_path=args.media_path, split='validation', max_number=10000, double_swap=False)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    test_dataset = dataset_loader.formatted_ds
    if test_dataset is None or len(test_dataset) == 0:
        raise RuntimeError(f"⚠️ Warning: {args.dataset} dataset is empty or failed to load.")
    
    dataset_remaining = [(idx,x) for idx,x in enumerate(test_dataset) if idx not in existing_ids]
    
    #model_responses = []
    with open(output_file_tmp, "a", encoding="utf_8", newline="\n", buffering=1, errors='replace') as fw:
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.num_workers) as executor:
            # Submit all tasks to the executor and get Future objects
            future_to_prompt = [executor.submit(benchmark_single, x, idx) for idx,x in dataset_remaining]
    
            # Iterate over completed futures as they finish
            for future in tqdm(concurrent.futures.as_completed(future_to_prompt), total=len(future_to_prompt)):
                # The result() method blocks until the task is complete and returns the value
                # or raises any exception that occurred during the call
                result = future.result()
                
                fw.write(json.dumps(result, ensure_ascii=False) + "\n")
                
                #model_responses.append(result)
    
    #model_responses = sorted(model_responses, key=lambda x: x['idx'])
    
    with open(output_file_tmp, "r", encoding="utf_8") as fr:
        model_responses = [json.loads(line) for line in fr]
    
    model_responses = sorted(model_responses, key=lambda x: x['idx'])
    model_responses = list({x['idx']:x for x in model_responses}.values())
    
    with open(args.output, "w", encoding="utf_8", newline="\n") as fw:
        json.dump(model_responses, fw, ensure_ascii=False, indent=2)
    os.remove(output_file_tmp)
    
    print("All done!", flush=True)
    