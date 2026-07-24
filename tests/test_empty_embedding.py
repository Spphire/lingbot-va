from types import SimpleNamespace

import torch

from wan_va.dataset.empty_embedding import encode_empty_prompt


class FakeTokenizer:
    def __call__(self, *_args, **_kwargs):
        return SimpleNamespace(
            input_ids=torch.tensor([[7, 0, 0, 0]]),
            attention_mask=torch.tensor([[1, 0, 0, 0]]),
        )


class FakeTextEncoder:
    def __call__(self, input_ids, attention_mask):
        assert input_ids.device == attention_mask.device
        hidden = torch.arange(24, dtype=torch.float32).reshape(1, 4, 6)
        return SimpleNamespace(last_hidden_state=hidden.to(input_ids.device))


def test_encode_empty_prompt_keeps_tokens_and_zero_pads():
    embedding = encode_empty_prompt(
        FakeTokenizer(),
        FakeTextEncoder(),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
        max_sequence_length=4,
    )

    assert embedding.shape == (4, 6)
    assert embedding.dtype == torch.bfloat16
    torch.testing.assert_close(
        embedding[0],
        torch.arange(6, dtype=torch.bfloat16),
    )
    assert torch.count_nonzero(embedding[1:]) == 0
