from typing import Iterator, Optional

from torch.utils.data import RandomSampler, Sampler, SequentialSampler


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
        n = len(self.data_source)
        data_idx_set = list(range(n))
        if self.shuffle:
            self.idx_sampler = RandomSampler(data_idx_set)
        else:
            self.idx_sampler = SequentialSampler(data_idx_set)

    def __iter__(self) -> Iterator[int]:
        if self.max_seq_len is not None:
            for idx in self.idx_sampler:
                datum = self.data_source[idx]
                input_ids = self.tokenizer.apply_chat_template(
                    datum["message_log"],
                    add_generation_prompt=True,
                    tokenize=True,
                )
                input_len = len(input_ids)
                if input_len >= self.max_seq_len:
                    print(
                        f"⚠️ WARNING: RLSampler: skipping source index {idx} with length {input_len} tokens greater than max sequence length {self.max_seq_len}.",
                        flush=True,
                    )
                    continue
                yield idx
        else:
            yield from self.idx_sampler
