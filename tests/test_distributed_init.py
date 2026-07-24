from datetime import timedelta

import pytest

from wan_va.distributed import util


def test_init_distributed_uses_configured_timeout(monkeypatch):
    calls = {}
    monkeypatch.setenv("TORCH_DISTRIBUTED_TIMEOUT", "2400")
    monkeypatch.setattr(
        util.torch.cuda,
        "set_device",
        lambda local_rank: calls.setdefault("local_rank", local_rank),
    )
    monkeypatch.setattr(
        util.dist,
        "init_process_group",
        lambda **kwargs: calls.setdefault("process_group", kwargs),
    )

    util.init_distributed(world_size=128, local_rank=7, rank=127)

    assert calls["local_rank"] == 7
    assert calls["process_group"] == {
        "backend": "nccl",
        "init_method": "env://",
        "rank": 127,
        "world_size": 128,
        "timeout": timedelta(seconds=2400),
    }


def test_init_distributed_rejects_nonpositive_timeout(monkeypatch):
    monkeypatch.setenv("TORCH_DISTRIBUTED_TIMEOUT", "0")

    with pytest.raises(ValueError, match="must be greater than zero"):
        util.init_distributed(world_size=1, local_rank=0, rank=0)
