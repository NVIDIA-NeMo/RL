import torch

from nemo_rl.distributed.mx_megatron_helpers import collect_megatron_publish_set


class ReplicatedOnlyModule(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(2))


def _published_names(*, tp_rank: int) -> list[str]:
    model = ReplicatedOnlyModule()
    published = collect_megatron_publish_set(
        model,
        tp_size=2,
        pp_size=1,
        pp_rank=0,
        ep_size=1,
        ep_rank=0,
        tp_rank=tp_rank,
    )
    return [name for name, _, _, _ in published]


def test_collect_megatron_publish_set_skips_replicated_on_nonzero_tp_rank():
    assert _published_names(tp_rank=1) == []


def test_collect_megatron_publish_set_publishes_replicated_on_zero_tp_rank():
    assert _published_names(tp_rank=0) == ["weight"]
