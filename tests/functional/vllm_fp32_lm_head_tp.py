"""Run with torchrun --standalone --nproc-per-node=2 to test native TP/refit."""

import os

import torch
import torch.distributed as dist
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed import (
    destroy_distributed_environment,
    destroy_model_parallel,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead


@torch.inference_mode()
def main() -> None:
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    init_distributed_environment(
        world_size=world_size,
        rank=rank,
        distributed_init_method="env://",
        local_rank=rank,
    )
    with set_current_vllm_config(VllmConfig()):
        initialize_model_parallel(tensor_model_parallel_size=world_size)
    cases = 0
    for dtype in (torch.bfloat16, torch.float16):
        for vocab in (256, 259):
            torch.manual_seed(42)
            weight = torch.randn(vocab, 64, device="cuda", dtype=dtype)
            hidden = torch.randn(7, 64, device="cuda", dtype=dtype)
            with set_current_vllm_config(VllmConfig()), torch.device("cuda"):
                head = ParallelLMHead(vocab, 64, params_dtype=dtype)
                processor = LogitsProcessor(vocab)
            head.weight_loader(head.weight, weight)
            pointer = head.weight.data_ptr()
            for offset in (0, 0.125):
                weight.add_(offset)
                head.weight_loader(head.weight, weight)
                for head_dtype in (dtype, torch.float32):
                    processor.head_dtype = head_dtype
                    logits = processor(head, hidden)
                    if logits is not None:
                        reference = hidden.double() @ weight.double().t()
                        assert logits.shape == (7, vocab)
                        assert logits.dtype == head_dtype
                        torch.testing.assert_close(
                            logits.double(),
                            reference,
                            atol=2e-4 if head_dtype == torch.float32 else 0.15,
                            rtol=1e-4 if head_dtype == torch.float32 else 0.01,
                        )
                        if head_dtype == torch.float32:
                            assert not torch.equal(logits, logits.to(dtype).float())
                    assert head.weight.dtype == dtype
                    assert head.weight.data_ptr() == pointer
                    cases += 1
    dist.barrier()
    if rank == 0:
        print(
            f"PASS: {cases} native LM-head TP={world_size} dtype/padding/refit cases",
            flush=True,
        )
    destroy_model_parallel()
    destroy_distributed_environment()


if __name__ == "__main__":
    main()
