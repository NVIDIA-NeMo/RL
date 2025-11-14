from typing import Iterator, Optional

import torch
from torch.utils.data import Sampler


class RLSampler(Sampler[int]):
    def __init__(
        self,
        data_source,
        tokenizer,
        max_seq_len: Optional[int] = None,
        shuffle: bool = False,
    ) -> None:
        self.data_source = data_source
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[int]:
        seed = int(torch.empty((), dtype=torch.int64).random_().item())
        generator = torch.Generator()
        generator.manual_seed(seed)
        n = len(self.data_source)
        if self.shuffle:
            data_row_idxs = torch.randperm(n, generator=generator).tolist()
        else:
            data_row_idxs = range(n)
        if self.max_seq_len is not None:
            for row_idx in data_row_idxs:
                datum = self.data_source[row_idx]
                input_ids = self.tokenizer.apply_chat_template(
                    datum["message_log"],
                    add_generation_prompt=True,
                    tokenize=True,
                )
                input_len = len(input_ids)
                if input_len >= self.max_seq_len:
                    print(
                        f"⚠️ WARNING: RLSampler: skipping prompt at row index {row_idx} with length {input_len} tokens greater than max sequence length {self.max_seq_len}.",
                        flush=True,
                    )
                    continue
                yield datum
        else:
            for row_idx in data_row_idxs:
                yield self.data_source[row_idx]
