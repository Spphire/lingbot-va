"""Opt-in adapter for NMX LingBot-VA next-state action datasets.

This module intentionally implements only the contract used by the recorded
NEDF chip training run. The upstream RoboTwin and LIBERO paths remain separate.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from einops import rearrange

from wan_va.chunking import iter_chunk_slices


NMX_ACTION_CONTRACT = "nmx_chunk_relative_v10"


def _get(config, key, default=None):
    if hasattr(config, "get"):
        return config.get(key, default)
    return getattr(config, key, default)


def uses_nmx_action_contract(config) -> bool:
    return _get(config, "action_contract", "legacy") == NMX_ACTION_CONTRACT


def apply_episode_action_validity(
    sample: dict,
    config,
    *,
    action_valid: bool,
    width_valid: bool,
) -> None:
    """Apply the episode-level pose and gripper validity recorded by NMX."""

    if not action_valid:
        for key in (
            "actions",
            "actions_mask",
            "raw_actions",
            "raw_actions_step_mask",
        ):
            sample[key] = torch.zeros_like(sample[key])
        return
    if width_valid:
        return

    canonical_ids = [int(value) for value in _get(config, "gripper_canonical_dims", [])]
    raw_ids = [int(value) for value in _get(config, "gripper_raw_dims", [])]
    if not canonical_ids or not raw_ids:
        raise ValueError(
            "Width-invalid NMX episodes require gripper_canonical_dims and "
            "gripper_raw_dims"
        )
    if any(index < 0 or index >= sample["actions"].shape[0] for index in canonical_ids):
        raise ValueError("gripper_canonical_dims contains an out-of-range channel")
    if any(index < 0 or index >= sample["raw_actions"].shape[-1] for index in raw_ids):
        raise ValueError("gripper_raw_dims contains an out-of-range channel")

    sample["actions"][canonical_ids] = 0
    sample["actions_mask"][canonical_ids] = False
    sample["raw_actions"][..., raw_ids] = 0


def load_action_norm_stats(dataset_root: str | Path, config) -> dict:
    """Load and validate the normalizer baked by nmx_vla for this dataset group."""

    configured_path = _get(config, "action_norm_stats_path")
    if configured_path:
        stats_path = Path(str(configured_path).format(dataset_root=dataset_root))
    else:
        filename = _get(
            config,
            "action_norm_stats_filename",
            "lingbot_action_norm_stats.json",
        )
        stats_path = Path(dataset_root) / filename
    if not stats_path.is_file():
        raise FileNotFoundError(f"Missing NMX action normalizer: {stats_path}")

    with stats_path.open("r", encoding="utf-8") as handle:
        stats = json.load(handle)
    action_dim = int(_get(config, "action_dim", 30))
    for key in ("q01", "q99"):
        values = stats.get(key)
        if not isinstance(values, list) or len(values) != action_dim:
            raise ValueError(
                f"{stats_path}: {key} must contain {action_dim} values, "
                f"got {None if values is None else len(values)}"
            )

    meta = stats.get("_meta", {})
    expected_version = _get(
        config,
        "action_norm_stats_version",
        "chunk_relative_v10_velocity_symmetric_scale",
    )
    checks = {
        "version": expected_version,
        "action_chunk_size_max": int(_get(config, "action_chunk_size_max", 4)),
        "chunk_grouping_start_from_one": bool(
            _get(config, "chunk_grouping_start_from_one", False)
        ),
        "relative_pose_frame": _get(config, "relative_pose_frame", "local_frame"),
        "quaternion_order": _get(config, "quaternion_order", "wxyz"),
        "action_quaternion_order": "xyzw",
    }
    for key, expected in checks.items():
        if meta.get(key) != expected:
            raise ValueError(
                f"{stats_path}: incompatible _meta.{key}: "
                f"expected {expected!r}, got {meta.get(key)!r}"
            )

    expected_groups = list(_get(config, "relative_pose_groups", []))
    recorded_groups = list(meta.get("relative_pose_groups", []))
    if recorded_groups[: len(expected_groups)] != expected_groups:
        raise ValueError(
            f"{stats_path}: configured relative_pose_groups are not a prefix "
            "of the recorded group layout"
        )
    used_ids = list(_get(config, "used_action_channel_ids", []))
    recorded_ids = list(meta.get("used_action_channel_ids", []))
    if recorded_ids[: len(used_ids)] != used_ids:
        raise ValueError(
            f"{stats_path}: configured used_action_channel_ids are not a prefix "
            "of the recorded canonical mapping"
        )
    return stats


def infer_frame_stride(frame_ids) -> int:
    values = np.asarray(frame_ids)
    if values.size > 1:
        return max(1, int(values[1] - values[0]))
    return 1


def infer_sampled_video_frames_per_latent(
    video_num_frames: int,
    latent_num_frames: int,
) -> int:
    if video_num_frames <= 0 or latent_num_frames <= 0:
        raise ValueError(
            "video_num_frames and latent_num_frames must both be positive, got "
            f"{video_num_frames} and {latent_num_frames}"
        )
    return max(1, int(math.ceil(video_num_frames / latent_num_frames)))


def prepare_raw_action_tensor(
    local_start_frame: int,
    latent_frame_ids,
    latent_frame_num: int,
    video_num_frames: int,
    action,
    config,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align absolute source actions to latent frames and mark all padding invalid."""

    frame_ids = np.asarray(latent_frame_ids)
    if frame_ids.size == 0:
        raise ValueError("latent_frame_ids is empty")
    action = torch.as_tensor(action, dtype=torch.float32)
    if action.ndim != 2:
        raise ValueError(f"action must have shape [T,D], got {tuple(action.shape)}")

    actions_per_frame = int(_get(config, "actions_per_frame", 1))
    frame_stride = infer_frame_stride(frame_ids)
    sampled_frames_per_latent = infer_sampled_video_frames_per_latent(
        int(video_num_frames), int(latent_frame_num)
    )
    actions_per_latent = frame_stride * sampled_frames_per_latent * actions_per_frame
    act_shift = max(0, int(frame_ids[0] - local_start_frame))
    action = action[act_shift * actions_per_frame :]
    valid = torch.ones(action.shape[0], dtype=torch.bool, device=action.device)

    head = torch.zeros(
        (actions_per_latent, action.shape[-1]),
        dtype=action.dtype,
        device=action.device,
    )
    action = torch.cat([head, action], dim=0)
    valid = torch.cat(
        [torch.zeros(actions_per_latent, dtype=torch.bool, device=valid.device), valid]
    )

    required = int(latent_frame_num) * actions_per_latent
    if action.shape[0] < required:
        tail = required - action.shape[0]
        action = torch.cat(
            [
                action,
                torch.zeros(
                    (tail, action.shape[-1]),
                    dtype=action.dtype,
                    device=action.device,
                ),
            ],
            dim=0,
        )
        valid = torch.cat(
            [valid, torch.zeros(tail, dtype=torch.bool, device=valid.device)]
        )
    else:
        action = action[:required]
        valid = valid[:required]

    return (
        rearrange(action, "(f n) d -> f n d", f=int(latent_frame_num)),
        rearrange(valid, "(f n) -> f n", f=int(latent_frame_num)),
    )


def _quat_to_xyzw(quat: torch.Tensor, order: str) -> torch.Tensor:
    if order == "xyzw":
        return quat
    if order == "wxyz":
        return torch.cat([quat[..., 1:4], quat[..., 0:1]], dim=-1)
    raise ValueError(f"Unsupported quaternion_order={order!r}")


def _quat_inverse(quat: torch.Tensor) -> torch.Tensor:
    return torch.cat([-quat[..., :3], quat[..., 3:4]], dim=-1)


def _quat_multiply(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    left_xyz, left_w = left[..., :3], left[..., 3:4]
    right_xyz, right_w = right[..., :3], right[..., 3:4]
    xyz = (
        left_w * right_xyz
        + right_w * left_xyz
        + torch.cross(left_xyz, right_xyz, dim=-1)
    )
    w = left_w * right_w - (left_xyz * right_xyz).sum(dim=-1, keepdim=True)
    return torch.cat([xyz, w], dim=-1)


def _quat_apply(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    quat_xyz, quat_w = quat[..., :3], quat[..., 3:4]
    uv = torch.cross(quat_xyz, vector, dim=-1)
    uuv = torch.cross(quat_xyz, uv, dim=-1)
    return vector + 2 * (quat_w * uv + uuv)


def _relative_pose(
    target_pose: torch.Tensor,
    anchor_pose: torch.Tensor,
    *,
    quaternion_order: str,
    relative_pose_frame: str,
) -> torch.Tensor:
    work = target_pose.to(dtype=torch.float64)
    anchor = anchor_pose.to(dtype=torch.float64)
    target_quat = _quat_to_xyzw(work[..., 3:7], quaternion_order)
    anchor_quat = _quat_to_xyzw(anchor[..., 3:7], quaternion_order)
    target_quat = target_quat / target_quat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    anchor_quat = anchor_quat / anchor_quat.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    anchor_inverse = _quat_inverse(anchor_quat)

    translation = work[..., :3] - anchor[..., :3]
    if relative_pose_frame == "local_frame":
        translation = _quat_apply(anchor_inverse.expand_as(target_quat), translation)
    elif relative_pose_frame != "world_frame":
        raise ValueError(f"Unsupported relative_pose_frame={relative_pose_frame!r}")

    relative_quat = _quat_multiply(anchor_inverse.expand_as(target_quat), target_quat)
    relative_quat = relative_quat / relative_quat.norm(
        dim=-1, keepdim=True
    ).clamp_min(1e-12)
    relative_quat = torch.where(
        relative_quat[..., 3:4] < 0,
        -relative_quat,
        relative_quat,
    )
    return torch.cat([translation, relative_quat], dim=-1).to(torch.float32)


def _transform_chunk(
    action: torch.Tensor,
    state: torch.Tensor,
    valid: torch.Tensor,
    config,
) -> torch.Tensor:
    width = sum(
        int(group["pose_slice"][1]) - int(group["pose_slice"][0])
        + (
            int(group["gripper_slice"][1]) - int(group["gripper_slice"][0])
            if group.get("gripper_slice") is not None
            else 0
        )
        for group in _get(config, "relative_pose_groups", [])
    )
    output = torch.zeros((action.shape[0], width), dtype=torch.float32, device=action.device)
    valid_indices = torch.nonzero(valid, as_tuple=False).flatten()
    if valid_indices.numel() == 0:
        return output

    first_valid = int(valid_indices[0].item())
    parts = []
    for group in _get(config, "relative_pose_groups", []):
        pose_start, pose_end = (int(v) for v in group["pose_slice"])
        relative = _relative_pose(
            action[first_valid:, pose_start:pose_end],
            state[first_valid, pose_start:pose_end],
            quaternion_order=_get(config, "quaternion_order", "wxyz"),
            relative_pose_frame=_get(config, "relative_pose_frame", "local_frame"),
        )
        group_parts = [relative]
        gripper_slice = group.get("gripper_slice")
        if gripper_slice is not None:
            grip_start, grip_end = (int(v) for v in gripper_slice)
            group_parts.append(action[first_valid:, grip_start:grip_end].float())
        parts.append(torch.cat(group_parts, dim=-1))
    output[first_valid:] = torch.cat(parts, dim=-1)
    return output


def build_model_actions_from_raw(
    raw_action: torch.Tensor,
    raw_state: torch.Tensor,
    step_valid_mask: torch.Tensor,
    config,
    *,
    q01,
    q99,
    chunk_size_frames: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build normalized canonical actions using the sampled attention chunk size."""

    if raw_action.ndim != 3:
        raise ValueError(f"raw_action must have shape [F,N,D], got {raw_action.shape}")
    if raw_state.shape != raw_action.shape:
        raise ValueError(
            f"raw_state must match raw_action, got {raw_state.shape} and {raw_action.shape}"
        )
    if step_valid_mask.shape != raw_action.shape[:2]:
        raise ValueError("step_valid_mask must match raw_action's [F,N] dimensions")

    frames, actions_per_latent, source_dim = raw_action.shape
    flat_action = raw_action.reshape(frames * actions_per_latent, source_dim).float()
    flat_state = raw_state.reshape(frames * actions_per_latent, source_dim).float()
    flat_valid = step_valid_mask.reshape(-1).bool()
    chunk_steps = int(chunk_size_frames) * actions_per_latent
    transformed = []
    for chunk in iter_chunk_slices(
        len(flat_action),
        chunk_steps,
        grouping_start_from_one=bool(
            _get(config, "chunk_grouping_start_from_one", False)
        ),
        prefix_step_size=actions_per_latent,
    ):
        transformed.append(
            _transform_chunk(
                flat_action[chunk],
                flat_state[chunk],
                flat_valid[chunk],
                config,
            )
        )
    source_action = torch.cat(transformed, dim=0)
    used_ids = [int(value) for value in _get(config, "used_action_channel_ids", [])]
    if source_action.shape[-1] != len(used_ids):
        raise ValueError(
            "Transformed action width does not match used_action_channel_ids: "
            f"{source_action.shape[-1]} != {len(used_ids)}"
        )

    action_dim = int(_get(config, "action_dim", 30))
    aligned = torch.zeros(
        (source_action.shape[0], action_dim),
        dtype=source_action.dtype,
        device=source_action.device,
    )
    aligned_mask = torch.zeros_like(aligned, dtype=torch.bool)
    for source_index, canonical_index in enumerate(used_ids):
        aligned[:, canonical_index] = source_action[:, source_index]
        aligned_mask[:, canonical_index] = flat_valid

    q01 = torch.as_tensor(q01, dtype=aligned.dtype, device=aligned.device).reshape(1, -1)
    q99 = torch.as_tensor(q99, dtype=aligned.dtype, device=aligned.device).reshape(1, -1)
    if q01.shape[-1] != action_dim or q99.shape[-1] != action_dim:
        raise ValueError("q01/q99 width must match action_dim")
    aligned = ((aligned - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0).clamp(
        -1.5, 1.5
    )
    aligned = aligned * aligned_mask.to(dtype=aligned.dtype)
    return (
        rearrange(aligned, "(f n) c -> c f n 1", f=frames),
        rearrange(aligned_mask, "(f n) c -> c f n 1", f=frames),
    )


def rebuild_batch_actions(batch_dict: dict, config, chunk_size_frames: int):
    actions = []
    masks = []
    for raw_action, raw_state, step_mask, q01, q99 in zip(
        batch_dict["raw_actions"],
        batch_dict["raw_states"],
        batch_dict["raw_actions_step_mask"],
        batch_dict["action_q01"],
        batch_dict["action_q99"],
    ):
        action, mask = build_model_actions_from_raw(
            raw_action,
            raw_state,
            step_mask,
            config,
            q01=q01,
            q99=q99,
            chunk_size_frames=chunk_size_frames,
        )
        actions.append(action)
        masks.append(mask)
    actions = torch.stack(actions)
    masks = torch.stack(masks)
    dataset_mask = batch_dict.get("actions_mask")
    if dataset_mask is None:
        raise ValueError(
            "NMX action rebuilding requires dataset actions_mask for channel validity"
        )
    if dataset_mask.shape != masks.shape:
        raise ValueError(
            f"Rebuilt action mask {tuple(masks.shape)} does not match dataset mask "
            f"{tuple(dataset_mask.shape)}"
        )
    masks = masks & dataset_mask.bool()
    actions = actions * masks.to(dtype=actions.dtype)
    return actions, masks
