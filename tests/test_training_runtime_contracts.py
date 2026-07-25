import json

import pytest
import torch

from wan_va.train import (
    append_metrics_jsonl,
    apply_action_history_condition_dropout,
)


def _action_dict():
    latent = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    return {
        "latent": latent.clone(),
        "noisy_latents": latent.clone() + 100,
        "targets": latent.clone() + 200,
        "actions_mask": torch.ones_like(latent, dtype=torch.bool),
    }


def test_action_history_dropout_one_zeros_only_clean_condition():
    action_dict = _action_dict()
    preserved = {
        key: value.clone()
        for key, value in action_dict.items()
        if key != "latent"
    }

    dropped = apply_action_history_condition_dropout(action_dict, 1.0)

    assert dropped.item() == 2
    assert not torch.count_nonzero(action_dict["latent"])
    for key, expected in preserved.items():
        torch.testing.assert_close(action_dict[key], expected)


def test_action_history_dropout_zero_is_noop_without_rng_draw():
    action_dict = _action_dict()
    original = action_dict["latent"].clone()
    torch.manual_seed(1234)
    rng_state = torch.random.get_rng_state()

    dropped = apply_action_history_condition_dropout(action_dict, 0.0)

    assert dropped.item() == 0
    torch.testing.assert_close(action_dict["latent"], original)
    assert torch.equal(torch.random.get_rng_state(), rng_state)


def test_action_history_dropout_rejects_invalid_probability():
    with pytest.raises(ValueError, match=r"within \[0, 1\]"):
        apply_action_history_condition_dropout(_action_dict(), 1.01)


def test_metrics_jsonl_appends_one_json_object_per_step(tmp_path):
    path = tmp_path / "metrics.jsonl"
    append_metrics_jsonl(path, {"step": 0, "lr": 1e-6})
    append_metrics_jsonl(path, {"step": 1, "lr": 2e-6})

    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    assert rows == [
        {"step": 0, "lr": 1e-6},
        {"step": 1, "lr": 2e-6},
    ]
