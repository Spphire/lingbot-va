from __future__ import annotations

import torch

from wan_va.modules.model import WanAttention


def test_denoise_query_keeps_full_committed_context() -> None:
    attention = WanAttention(
        dim=8,
        heads=2,
        dim_head=4,
        attn_mode="torch",
    )
    attention.init_kv_cache(
        "test",
        total_tolen=4,
        num_head=2,
        head_dim=4,
        device=torch.device("cpu"),
        dtype=torch.float32,
        batch_size=1,
    )

    seen_key_lengths: list[int] = []

    def capture_attention(query, key, value):
        seen_key_lengths.append(int(key.shape[1]))
        return torch.zeros_like(query)

    attention.attn_op = capture_attention
    context = torch.randn(1, 4, 8)
    attention(context, context, context, rotary_emb=None, update_cache=2, cache_name="test")

    cache = attention.attn_caches["test"]
    committed_key = cache["k"].clone()
    committed_mask = cache["mask"].clone()
    current = torch.randn(1, 3, 8)
    attention(current, current, current, rotary_emb=None, update_cache=0, cache_name="test")

    assert seen_key_lengths == [4, 7]
    torch.testing.assert_close(cache["k"], committed_key)
    torch.testing.assert_close(cache["mask"], committed_mask)
    assert int(cache["mask"].sum()) == 4
