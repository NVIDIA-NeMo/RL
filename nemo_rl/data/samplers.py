import concurrent.futures
from typing import Iterator, Optional

from torch.utils.data import Sampler
from torchdata import RandomSampler, SequentialSampler


class RLSampler(Sampler[int]):
    def __init__(
        self,
        data_source,
        tokenizer,
        shuffle: Optional[bool] = None,
        max_seq_len: Optional[int] = None,
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


_WORKER_TOKENIZER = None


def _init_worker(tokenizer) -> None:
    global _WORKER_TOKENIZER
    _WORKER_TOKENIZER = tokenizer


def _rank_preserving_token_length(
    seq_rank: int, idx: int, input_str: str
) -> tuple[int, int]:
    global _WORKER_TOKENIZER
    assert _WORKER_TOKENIZER is not None
    input_ids = _WORKER_TOKENIZER.encode(
        input_str,
        add_special_tokens=False,
        padding=False,
        truncation=False,
    )
    return seq_rank, idx, len(input_ids)


def _drain_futures_batch(
    futures_rem, futures_done, batch: list, max_seq_len: int
) -> None:
    done, _ = concurrent.futures.wait(
        futures_rem,
        return_when=concurrent.futures.FIRST_COMPLETED,
    )
    for future in done:
        seq_rank, idx, input_len = future.result()
        if input_len >= max_seq_len:
            print(
                f"⚠️ WARNING: RLBatchSampler: skipping source index {idx} with length {input_len} tokens greater than max sequence length {max_seq_len}.",
                flush=True,
            )
            continue
        batch.append((seq_rank, idx))
    futures_done |= done
    futures_rem -= done


class RLBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        data_source,
        tokenizer,
        batch_size: Optional[int] = 1,
        shuffle: Optional[bool] = None,
        max_seq_len: Optional[int] = None,
        num_workers: int = 1,
    ) -> None:
        self.data_source = data_source
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.max_seq_len = max_seq_len
        self.executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=num_workers,
            initializer=_init_worker,
            initargs=(tokenizer,),
        )
        n = len(data_source)
        data_idx_set = list(range(n))
        if shuffle:
            self.idx_sampler = RandomSampler(data_idx_set)
        else:
            self.idx_sampler = SequentialSampler(data_idx_set)

    def __iter__(self) -> Iterator[list[int]]:
        batch = []
        if self.max_seq_len is not None:
            futures_rem = set()
            futures_done = set()
            for seq_rank, idx in enumerate(self.idx_sampler):
                if len(futures_rem) + len(futures_done) >= self.batch_size:
                    _drain_futures_batch(
                        futures_rem, futures_done, batch, self.max_seq_len
                    )
                if len(batch) == self.batch_size:
                    batch.sort()
                    yield [idx for _, idx in batch]
                    batch.clear()
                    futures_done.clear()
                datum = self.data_source[idx]
                input_str = self.tokenizer.apply_chat_template(
                    datum["message_log"],
                    add_generation_prompt=True,
                    tokenize=False,
                )
                future = self.executor.submit(
                    _rank_preserving_token_length,
                    seq_rank,
                    idx,
                    input_str,
                    self.tokenizer,
                    self.max_seq_len,
                )
                futures_rem.add(future)
        else:
            for seq_rank, idx in enumerate(self.idx_sampler):
                if len(batch) == self.batch_size:
                    batch.sort()
                    yield [idx for _, idx in batch]
                    batch.clear()
                batch.append((seq_rank, idx))
        if futures_rem:
            _drain_futures_batch(futures_rem, futures_done, batch, self.max_seq_len)
        if batch:
            batch.sort()
            yield [idx for _, idx in batch]

    # TODO: state dict.
