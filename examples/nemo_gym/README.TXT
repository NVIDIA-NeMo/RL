
The five doublerollout_*.yaml config files reproduce the experimental results in the paper.

The chat template of the starting checkpoint needs to be replaced: use the one in folder nanov3release_modified_chat_template for nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16; use the one in folder qwen330ba3bthink_modified_chat_template for Qwen/Qwen3-30B-A3B-Thinking-2507. This is only needed during training. When serving the trained models, feel free to switch back to the original.
