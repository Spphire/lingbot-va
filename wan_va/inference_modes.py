"""Inference branch selection derived from the trained action contract."""


def resolve_video_run_mode(
    action_condition_mode: str,
    fastwam_video_run_mode: bool,
) -> bool:
    """Return whether future-video denoising is required for this checkpoint."""

    mode = str(action_condition_mode)
    if mode not in {"inverse_dynamics", "fastwam"}:
        raise ValueError(
            "action_condition_mode must be 'inverse_dynamics' or 'fastwam', "
            f"got {mode!r}"
        )
    if not isinstance(fastwam_video_run_mode, bool):
        raise TypeError("fastwam_video_run_mode must be a boolean")
    return mode != "fastwam" or fastwam_video_run_mode


def resolve_use_cfg(
    action_condition_mode: str,
    fastwam_video_run_mode: bool,
    guidance_scale: float,
    action_guidance_scale: float,
) -> bool:
    """Return whether any branch that actually runs needs CFG batch doubling."""

    video_run_mode = resolve_video_run_mode(
        action_condition_mode,
        fastwam_video_run_mode,
    )
    return (video_run_mode and float(guidance_scale) > 1.0) or (
        float(action_guidance_scale) > 1.0
    )
