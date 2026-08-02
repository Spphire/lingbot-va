from __future__ import annotations

import numpy as np
import pytest

from script.evaluate_nmx_offline import (
    compute_metrics,
    flatten_server_action,
    has_valid_evaluation_target,
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
