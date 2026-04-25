# -*- coding: utf-8 -*-
"""
Created on Wed Apr 22 18:49:53 2026

@author: degert
"""

"""
hostname -I

vllm serve /lustre/fsw/portfolios/llmservice/users/jiaqiz/models/qwen235b_principle_comparison_genrm_step1230 \
--served-model-name model \
--gpu-memory-utilization 0.8 \
--tensor-parallel-size 8 \
--max-num-seqs 64 \
--max-model-len 32768 \
--model-loader-extra-config "{\"enable_multithread_load\": true, \"num_threads\": 64}" \
--host 0.0.0.0
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
)


DWRL_PROMPT_TEMPLATE = """You are an expert evaluation judge specializing in the assessment of LLM responses. You are impartial, rigorous, and consistent. Given the conversation context and a single response to the user's latest query, you will follow the evaluation plan and scoring guidelines exactly as written below.

#### Conversation Context ####
{context}

#### Response to be Scored ####
[The Start of Response]
{response}
[The End of Response]

Please act as an impartial judge and evaluate the quality of the response provided by an AI assistant to the user prompt. Begin your evaluation by generating your own answer to the prompt. You must provide your answer before judging any answers. When evaluating the assistant's response, compare the assistant's response with your answer. You must identify and correct any mistakes or inaccurate information. Then consider if the assistant's response is helpful, relevant, and concise. Helpful means the answer correctly responds to the prompt or follows the instructions. Note when the user prompt has any ambiguity or more than one interpretation, it is more helpful and appropriate to ask for clarifications or more information from the user rather than providing an answer based on assumptions. Relevant means all parts of the response closely connect or are appropriate to what is being asked. Concise means the response is clear and not verbose or excessive. Then consider the creativity and novelty of the assistant's response when needed. Finally, identify any missing important information in the assistant's response which would be beneficial to include when responding to the user prompt.

#### Scoring Guidelines ####
Based on the evaluation plan above, assign scores using these scales:
**Individual Helpfulness Scores (1-5):**
- 5: Extremely Helpful - Completely aligned with what the user was asking for
- 4: Mostly Helpful - Generally useful with minor room for improvement
- 3: Partially Helpful - Misses the overall goal in some way
- 2: Borderline Unhelpful - Mostly doesn't capture what the user wanted
- 1: Not Helpful - Completely missed the essence of the request

#### Output Format ####
Analyze step by step following the evaluation plan, then provide your judgment as JSON:
```json
{{
    "response_analysis": "Your detailed analysis of the Response based on the evaluation plan",
    "score": <1-5>
}}
```"""

#SCORE_PROMPT = 'Based on your evaluation, is the response helpful and harmless? Answer with only a single word: yes or no'
SCORE_PROMPT = 'Based on your evaluation, is the original response good? Answer with only a single word: yes or no'


def flatten_to_single_turn(message_log):
    ret = ""
    for idx, message in enumerate(message_log):
        if message["role"] == "system":
            ret += "System: " + message["content"].strip() + "\n\n"
        elif message["role"] == "user":
            ret += "User: " + message["content"].strip() + "\n\n"
        elif message["role"] == "assistant":
            resp_no_thinking1 = re.sub(r"(?i)(?s)(\<think\>)(.*?)(\<\/think\>)", "", message["content"]).strip()
            resp_no_thinking2 = re.sub(r"(?i)(?s)(.*?)(\<\/think\>)", "", resp_no_thinking1).strip()
            ret += "Assistant: " + resp_no_thinking2 + "\n\n"
    
    return ret.strip()

def get_json_response(response):
    if response is None:
        return None
    
    resp_no_thinking1 = re.sub(r"(?i)(?s)(\<think\>)(.*?)(\<\/think\>)", "", response).strip()
    resp_no_thinking2 = re.sub(r"(?i)(?s)(.*?)(\<\/think\>)", "", resp_no_thinking1).strip()
    
    try:
        jp = json.loads(re.search(r"(?i)(?s)(\{\n)(.*?)(\n\})", resp_no_thinking2).group(0))
        return jp
    except:
        return resp_no_thinking2


def get_score_from_vllm(samp, response):
    try:
        completion = client.chat.completions.create(
          model="model",
          messages=[{"role": "user", "content": DWRL_PROMPT_TEMPLATE.format(context=flatten_to_single_turn(samp['context']) if isinstance(samp['context'], list) else samp['context'], response=response)}],
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
        
        completion = client.chat.completions.create(
          model="model",
          messages=[{"role": "user", "content": DWRL_PROMPT_TEMPLATE.format(context=flatten_to_single_turn(samp['context']) if isinstance(samp['context'], list) else samp['context'], response=response)}] + [{"role": "assistant", "content": thought}] + [{"role": "user", "content": SCORE_PROMPT}],
          temperature=1.0,
          top_p=1.0,
          max_tokens=1,
          #extra_body={"top_k": 20},
          extra_body={"chat_template_kwargs": {"enable_thinking": False}},
          logprobs=True,
          top_logprobs=20,
          stream=False
        )
        
        score = -999
        for possible_tokens in completion.choices[0].logprobs.content[-1].top_logprobs:
            if possible_tokens.token == "yes":
                score = math.exp(possible_tokens.logprob)
                break
        if score == -999:
            for possible_tokens in completion.choices[0].logprobs.content[-1].top_logprobs:
                if possible_tokens.token.lower().strip() == "yes":
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
    response_1 = samp["response1"]
    response_2 = samp["response2"]
    #preference = samp["preference"]
    
    score_1, thought_1 = get_score_from_vllm(samp, response_1)
    score_2, thought_2 = get_score_from_vllm(samp, response_2)
    
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
                        choices=["rewardbench2", "judgebench", "rmbench"],
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
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    test_dataset = dataset_loader.formatted_ds
    if test_dataset is None or len(test_dataset) == 0:
        raise RuntimeError(f"⚠️ Warning: {args.dataset} dataset is empty or failed to load.")
    
    dataset_remaining = [(idx,x) for idx,x in enumerate(test_dataset) if idx not in existing_ids]
    
    #model_responses = []
    with open(output_file_tmp, "a", encoding="utf_8", newline="\n", buffering=1) as fw:
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
    
    with open(args.output, "w", encoding="utf_8", newline="\n") as fw:
        json.dump(model_responses, fw, ensure_ascii=False, indent=2)
    os.remove(output_file_tmp)
    
    print("All done!", flush=True)
    