from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from script.evaluate_nmx_offline import (
    compute_metrics,
    denormalize_training_action,
    deployment_native_first_chunk_target,
    flatten_server_action,
    has_valid_evaluation_target,
    physical_action_bounds,
    resolve_action_history_mode,
    split_deployment_native_windows,
    split_episode_chunks,
)


def test_flatten_server_action_preserves_frame_step_order() -> None:
    action = np.arange(16 * 2 * 3, dtype=np.float64).reshape(16, 2, 3)
    flattened = flatten_server_action(action)

    assert flattened.shape == (6, 16)
    np.testing.assert_array_equal(flattened[0], action[:, 0, 0])
    np.testing.assert_array_equal(flattened[1], action[:, 0, 1])
    np.testing.assert_array_equal(flattened[3], action[:, 1, 0])


def test_compute_metrics_ignores_masked_condition_steps() -> None:
    target = np.zeros((4, 16), dtype=np.float64)
    prediction = target.copy()
    prediction[0] = 100.0
    prediction[2, 0] = 3.0
    mask = np.ones_like(target, dtype=bool)
    mask[0] = False

    metrics = compute_metrics(prediction, target, mask)

    assert metrics["valid_values"] == 48
    assert metrics["mae"] == pytest.approx(3.0 / 48.0)
    assert metrics["rmse"] == pytest.approx(3.0 / np.sqrt(48.0))


def test_compute_metrics_reports_quaternion_geodesic_error() -> None:
    target = np.zeros((1, 16), dtype=np.float64)
    prediction = np.zeros((1, 16), dtype=np.float64)
    target[:, 6] = 1.0
    target[:, 14] = 1.0
    prediction[:, 3] = 1.0
    prediction[:, 11] = 1.0
    mask = np.ones_like(target, dtype=bool)

    metrics = compute_metrics(prediction, target, mask)

    assert metrics["left_rotation_mae_rad"] == pytest.approx(np.pi)
    assert metrics["right_rotation_mae_rad"] == pytest.approx(np.pi)


def test_valid_evaluation_target_requires_at_least_one_unmasked_value() -> None:
    assert not has_valid_evaluation_target(np.zeros((48, 16), dtype=bool))
    mask = np.zeros((48, 16), dtype=bool)
    mask[12, 0] = True
    assert has_valid_evaluation_target(mask)


def test_split_episode_chunks_excludes_condition_once_and_keeps_tail() -> None:
    sample = {
        "latents": torch.arange(10).reshape(1, 10, 1, 1),
        "actions": torch.arange(10).reshape(1, 10, 1, 1).float(),
        "actions_mask": torch.ones(1, 10, 1, 1, dtype=torch.bool),
    }

    condition_latent, condition_action, chunks = split_episode_chunks(
        sample,
        frame_chunk_size=4,
        grouping_start_from_one=True,
    )

    assert condition_latent.flatten().tolist() == [0]
    assert condition_action.flatten().tolist() == [0]
    assert [chunk.start_latent for chunk in chunks] == [1, 5, 9]
    assert [chunk.latent.shape[1] for chunk in chunks] == [4, 4, 1]
    assert torch.cat([chunk.latent for chunk in chunks], dim=1).flatten().tolist() == list(
        range(1, 10)
    )


def test_deployment_native_first_chunk_masks_only_condition_action_latent() -> None:
    sample = {
        "latents": torch.arange(10).reshape(1, 10, 1, 1),
        "actions": torch.arange(120).reshape(1, 10, 12, 1).float(),
        "actions_mask": torch.ones(1, 10, 12, 1, dtype=torch.bool),
    }

    condition, target, mask = deployment_native_first_chunk_target(
        sample,
        frame_chunk_size=4,
    )

    assert condition.flatten().tolist() == [0]
    assert target.shape == (1, 4, 12, 1)
    assert not mask[:, 0].any()
    assert mask[:, 1:].all()


def test_deployment_native_windows_cover_every_post_condition_latent_once() -> None:
    sample = {
        "latents": torch.arange(10).reshape(1, 10, 1, 1),
        "actions": torch.arange(120).reshape(1, 10, 12, 1).float(),
        "actions_mask": torch.ones(1, 10, 12, 1, dtype=torch.bool),
    }

    windows = split_deployment_native_windows(sample, frame_chunk_size=4)

    assert [window.start_latent for window in windows] == [0, 3, 6]
    assert [window.action.shape[1] for window in windows] == [4, 4, 4]
    covered = torch.cat([window.action[:, 1:] for window in windows], dim=1)
    torch.testing.assert_close(covered, sample["actions"][:, 1:])
    assert all(not window.mask[:, 0].any() for window in windows)
    assert all(window.mask[:, 1:].all() for window in windows)


def test_denormalize_action_and_plot_bounds_use_physical_channel_scale() -> None:
    normalized = torch.zeros(4, 2, 1, 1)
    normalized[1] = -1
    normalized[3] = 1
    mask = torch.ones_like(normalized, dtype=torch.bool)
    q01 = torch.tensor([-2.0, 10.0, 100.0, -0.5])
    q99 = torch.tensor([2.0, 20.0, 200.0, 0.5])

    action, action_mask = denormalize_training_action(
        normalized,
        mask,
        q01,
        q99,
        [1, 3],
    )
    lower, upper = physical_action_bounds(q01, q99, [1, 3])

    np.testing.assert_allclose(action[:, 0], 10.0)
    np.testing.assert_allclose(action[:, 1], 0.5, atol=1e-6)
    np.testing.assert_array_equal(action_mask, np.ones((2, 2), dtype=bool))
    np.testing.assert_array_equal(lower, np.asarray([10.0, -0.5]))
    np.testing.assert_array_equal(upper, np.asarray([20.0, 0.5]))


def test_action_history_auto_follows_training_dropout_contract() -> None:
    assert (
        resolve_action_history_mode(
            SimpleNamespace(action_history_condition_dropout_prob=1.0), "auto"
        )
        == "zero"
    )
    assert (
        resolve_action_history_mode(
            SimpleNamespace(action_history_condition_dropout_prob=0.0), "auto"
        )
        == "ground-truth"
    )
    with pytest.raises(ValueError, match="no unique evaluation contract"):
        resolve_action_history_mode(
            SimpleNamespace(action_history_condition_dropout_prob=0.5), "auto"
        )
