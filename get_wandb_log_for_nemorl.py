
import wandb
import argparse
import os
import pandas as pd
#import pprint

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--org", type=str, default="nvidia")
    parser.add_argument("--project", type=str, default="nemo-rl")
    parser.add_argument("--uid", type=str, required=True)
    parser.add_argument("--at-step", type=int, default=5)
    parser.add_argument("--average-steps", type=int, default=5)
    return parser.parse_args()

def main():
    args = parse_args()
    assert os.environ.get("WANDB_API_KEY") is not None, "WANDB_API_KEY is not set"

    keys = [
        # "train/mean_total_tokens_per_sample",
        "timing/train/total_step_time", # 
        # "timing/train/exposed_generation", # for off-policy
        "timing/train/generation", # on-policy
        "timing/train/policy_and_reference_logprobs",
        "timing/train/policy_training",
        # "timing/train/weight_sync", 
        "timing/train/data_processing",
        "timing/train/reward_calculation",
        "timing/train/training_prep",


        # "timing/train/prepare_for_generation/total",
        "timing/train/prepare_for_generation/transfer_and_update_weights",
        "performance/tokens_per_sec_per_gpu",
        "performance/generation_tokens_per_sec_per_gpu",
        "performance/training_worker_group_tokens_per_sec_per_gpu",
        "performance/policy_training_tokens_per_sec_per_gpu",
        "performance/policy_and_reference_logprobs_tokens_per_sec_per_gpu",
        "performance/train_flops_per_gpu",
        "performance/train_fp_utilization",
    ]

    api = wandb.Api()
    run = api.run(f"{args.org}/{args.project}/{args.uid}")
    min_step = args.at_step - args.average_steps // 2
    max_step = args.at_step + args.average_steps // 2 - 1 + (args.average_steps % 2)
    history = run.history().loc[min_step:max_step, keys]

    # add MFU
    history['performance/train_mfu'] = history['performance/train_fp_utilization'] * 100
    # get average of the history
    average_history = history.mean(axis=0)

    
    history_at_step = history.loc[args.at_step]
    # print("Average history:")
    # #pprint.pprint(average_history)
    # print(average_history)
    # print("History at step:")
    # #pprint.pprint(history_at_step)
    # print(history_at_step)

    # with pd.option_context('display.float_format', '{:.2f}'.format): 블록 내에서만 설정이 유지됩니다.
    with pd.option_context('display.float_format', '{:.2f}'.format):
        print("---")
        print("Average history (Averaged across steps):")
        print(average_history)
        print("---")
        print("History at step (Step {}):".format(args.at_step))
        print(history_at_step)
        print("---")

if __name__ == "__main__":
    main()