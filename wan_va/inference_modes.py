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
