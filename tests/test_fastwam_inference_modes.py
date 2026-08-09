import pytest

from wan_va.inference_modes import resolve_use_cfg, resolve_video_run_mode


def test_idm_always_runs_future_video_denoising() -> None:
    assert resolve_video_run_mode("inverse_dynamics", False) is True
    assert resolve_video_run_mode("inverse_dynamics", True) is True


def test_fastwam_defaults_to_action_only_but_can_generate_debug_video() -> None:
    assert resolve_video_run_mode("fastwam", False) is False
    assert resolve_video_run_mode("fastwam", True) is True


def test_inference_mode_rejects_unknown_contract_and_non_boolean_override() -> None:
    with pytest.raises(ValueError, match="action_condition_mode"):
        resolve_video_run_mode("shared", False)
    with pytest.raises(TypeError, match="boolean"):
        resolve_video_run_mode("fastwam", 1)  # type: ignore[arg-type]


def test_action_only_fastwam_ignores_unused_video_cfg() -> None:
    assert resolve_use_cfg("fastwam", False, 5.0, 1.0) is False
    assert resolve_use_cfg("fastwam", False, 5.0, 2.0) is True
    assert resolve_use_cfg("fastwam", True, 5.0, 1.0) is True
    assert resolve_use_cfg("inverse_dynamics", False, 5.0, 1.0) is True
