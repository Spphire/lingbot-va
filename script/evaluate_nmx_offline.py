"""Evaluate a LingBot-VA checkpoint against cached training episodes."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from wan_va.configs import VA_CONFIGS
from wan_va.dataset.lerobot_latent_dataset import MultiLatentLeRobotDataset
from wan_va.dataset.nmx_action_adapter import load_action_norm_stats
from wan_va.distributed.fsdp import shard_model
from wan_va.distributed.util import _configure_model
from wan_va.modules.utils import load_transformer
from wan_va.utils import FlowMatchScheduler
import wan_va.wan_va_server as server_module
from wan_va.wan_va_server import VA_Server


CHANNEL_NAMES = (
    "left_x",
    "left_y",
    "left_z",
    "left_qx",
    "left_qy",
    "left_qz",
    "left_qw",
    "left_gripper",
    "right_x",
    "right_y",
    "right_z",
    "right_qx",
    "right_qy",
    "right_qz",
    "right_qw",
    "right_gripper",
)


class _NoopStreamingVAE:
    def clear_cache(self) -> None:
        pass


class OfflineActionServer(VA_Server):
    """Action-generation subset of the official server using cached latents."""

    def __init__(self, job_config: Any) -> None:
        self.cache_name = "pos"
        self.job_config = job_config
        self.save_root = str(job_config.save_root)
        self.dtype = job_config.param_dtype
        self.device = torch.device(f"cuda:{job_config.local_rank}")
        torch.cuda.set_device(self.device)
        self.enable_offload = False

        self.scheduler = FlowMatchScheduler(
            shift=job_config.snr_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )
        self.action_scheduler = FlowMatchScheduler(
            shift=job_config.action_snr_shift,
            sigma_min=0.0,
            extra_one_step=True,
        )
        self.scheduler.set_timesteps(1000, training=True)
        self.action_scheduler.set_timesteps(1000, training=True)
        self.transformer = load_transformer(
            str(job_config.transformer_path),
            torch_dtype=self.dtype,
            torch_device=self.device,
            attn_mode=str(job_config.attn_mode),
        )
        self.transformer = _configure_model(
            model=self.transformer,
            shard_fn=shard_model,
            param_dtype=self.dtype,
            device=self.device,
            eval_mode=True,
        )
        self.env_type = "none"
        self.streaming_vae = _NoopStreamingVAE()
        self.streaming_vae_half = None
        self._offline_init_latent: torch.Tensor | None = None
        self._offline_latent_hw: tuple[int, int] | None = None

    def _encode_obs(self, obs: dict[str, Any]) -> torch.Tensor:
        if "video_latent" in obs:
            latent = torch.as_tensor(obs["video_latent"])
            if latent.ndim == 4:
                latent = latent.unsqueeze(0)
            if latent.ndim != 5:
                raise ValueError(
                    "Cached video latent must have shape [C,F,H,W] or [1,C,F,H,W], "
                    f"got {tuple(latent.shape)}"
                )
            return latent.to(device=self.device, dtype=self.dtype)
        if self._offline_init_latent is None:
            raise RuntimeError("Offline initial latent has not been set")
        return self._offline_init_latent

    def _reset(self, prompt: str | None = None) -> None:
        super()._reset(prompt=prompt)
        self.action_mask = self.action_mask.to(self.device)
        if self._offline_latent_hw is not None and self._offline_latent_hw != (
            self.latent_height,
            self.latent_width,
        ):
            self.latent_height, self.latent_width = self._offline_latent_hw
            self.transformer.clear_cache(self.cache_name)
            patch_t, patch_h, patch_w = self.job_config.patch_size
            latent_tokens = (
                int(self.job_config.frame_chunk_size)
                * self.latent_height
                * self.latent_width
            ) // (int(patch_t) * int(patch_h) * int(patch_w))
            action_tokens = int(self.job_config.frame_chunk_size) * int(
                self.action_per_frame
            )
            self.transformer.create_empty_cache(
                self.cache_name,
                self.job_config.attn_window,
                latent_tokens,
                action_tokens,
                dtype=self.dtype,
                device=self.device,
                batch_size=2 if self.use_cfg else 1,
            )

    def set_episode_latent_shape(self, latent: torch.Tensor) -> None:
        latent = torch.as_tensor(latent)
        if latent.ndim != 4:
            raise ValueError(f"Expected episode latent [C,F,H,W], got {tuple(latent.shape)}")
        self._offline_latent_hw = (int(latent.shape[-2]), int(latent.shape[-1]))

    @torch.no_grad()
    def cache_training_context(
        self,
        video_latent: torch.Tensor,
        normalized_action: torch.Tensor,
    ) -> None:
        """Replace sampled predictions with cached dataset video/action context."""

        latent = torch.as_tensor(video_latent, device=self.device, dtype=self.dtype)
        action = torch.as_tensor(normalized_action, device=self.device, dtype=self.dtype)
        if latent.ndim == 4:
            latent = latent.unsqueeze(0)
        if action.ndim == 4:
            action = action.unsqueeze(0)
        if latent.ndim != 5 or action.ndim != 5:
            raise ValueError(
                "Training context expects latent/action tensors with optional batch axes; "
                f"got {tuple(latent.shape)} and {tuple(action.shape)}"
            )
        if latent.shape[0] != 1 or action.shape[0] != 1:
            raise ValueError("Offline evaluation only supports batch size 1")
        if latent.shape[2] != action.shape[2]:
            raise ValueError(
                "Video/action context frame counts differ: "
                f"{latent.shape[2]} != {action.shape[2]}"
            )

        self.transformer.clear_pred_cache(self.cache_name)
        input_dict = self._prepare_latent_input(
            latent,
            action,
            frame_st_id=self.frame_st_id,
        )
        self.transformer(
            self._repeat_input_for_cfg(input_dict["latent_res_lst"]),
            update_cache=2,
            cache_name=self.cache_name,
            action_mode=False,
        )
        self.transformer(
            self._repeat_input_for_cfg(input_dict["action_res_lst"]),
            update_cache=2,
            cache_name=self.cache_name,
            action_mode=True,
        )
        self.frame_st_id += int(latent.shape[2])
        torch.cuda.empty_cache()

    def set_action_norm_stats(self, q01: torch.Tensor, q99: torch.Tensor) -> None:
        self.actions_q01 = torch.as_tensor(q01, dtype=torch.float32).reshape(-1, 1, 1)
        self.actions_q99 = torch.as_tensor(q99, dtype=torch.float32).reshape(-1, 1, 1)


@dataclass(frozen=True)
class EpisodeChunk:
    start_latent: int
    latent: torch.Tensor
    action: torch.Tensor
    mask: torch.Tensor


def _checkpoint_transformer(path: Path) -> Path:
    path = path.expanduser().resolve()
    transformer = path if path.name == "transformer" else path / "transformer"
    if not (transformer / "config.json").is_file():
        raise FileNotFoundError(
            "--checkpoint must name an exact checkpoint_step_N directory or its "
            f"transformer directory: {path}"
        )
    return transformer


def _sample_metadata(
    dataset: MultiLatentLeRobotDataset,
    global_index: int,
) -> dict[str, Any]:
    dataset_id = int(dataset.item_id_to_dataset_id[int(global_index)])
    local_index = int(global_index) - int(dataset.acc_dset_num[dataset_id])
    child = dataset._datasets[dataset_id]
    meta = dict(child.new_metas[local_index])
    meta["dataset_root"] = str(child.root)
    meta["global_sample_index"] = int(global_index)
    return meta


def shuffled_sample_candidates(
    dataset: MultiLatentLeRobotDataset,
    seed: int,
):
    candidates = list(range(len(dataset)))
    random.Random(seed).shuffle(candidates)
    yield from candidates


def _as_embedding(value: torch.Tensor, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    tensor = value
    if tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    if tensor.ndim != 3 or tuple(tensor.shape[-2:]) != (512, 4096):
        raise ValueError(f"Expected text embedding [1,512,4096], got {tuple(tensor.shape)}")
    return tensor.to(device=device, dtype=dtype)


def flatten_server_action(action: np.ndarray) -> np.ndarray:
    if action.ndim != 3 or action.shape[0] != len(CHANNEL_NAMES):
        raise ValueError(f"Expected action [16,F,H], got {action.shape}")
    return np.moveaxis(action, 0, -1).reshape(-1, action.shape[0])


def split_episode_chunks(
    sample: dict[str, torch.Tensor],
    *,
    frame_chunk_size: int,
    grouping_start_from_one: bool,
    max_chunks: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[EpisodeChunk]]:
    """Split one complete sample while isolating its one-frame causal condition."""

    latents = torch.as_tensor(sample["latents"])
    actions = torch.as_tensor(sample["actions"])
    masks = torch.as_tensor(sample["actions_mask"])
    if latents.ndim != 4:
        raise ValueError(f"Expected latents [C,F,H,W], got {tuple(latents.shape)}")
    if actions.ndim != 4 or masks.shape != actions.shape:
        raise ValueError(
            "Expected matching actions/actions_mask [C,F,N,1], got "
            f"{tuple(actions.shape)} and {tuple(masks.shape)}"
        )
    if int(latents.shape[1]) != int(actions.shape[1]):
        raise ValueError(
            f"Latent/action frame counts differ: {latents.shape[1]} != {actions.shape[1]}"
        )
    if not grouping_start_from_one:
        raise ValueError(
            "Complete NMX episode evaluation requires chunk_grouping_start_from_one=True"
        )
    if frame_chunk_size <= 0:
        raise ValueError(f"frame_chunk_size must be positive, got {frame_chunk_size}")
    if max_chunks is not None and max_chunks <= 0:
        raise ValueError(f"max_chunks must be positive, got {max_chunks}")

    condition_latent = latents[:, :1].contiguous()
    condition_action = torch.zeros_like(actions[:, :1])
    chunks: list[EpisodeChunk] = []
    for start in range(1, int(latents.shape[1]), int(frame_chunk_size)):
        if max_chunks is not None and len(chunks) >= max_chunks:
            break
        end = min(start + int(frame_chunk_size), int(latents.shape[1]))
        chunks.append(
            EpisodeChunk(
                start_latent=start,
                latent=latents[:, start:end].contiguous(),
                action=actions[:, start:end].contiguous(),
                mask=masks[:, start:end].contiguous(),
            )
        )
    return condition_latent, condition_action, chunks


def deployment_native_first_chunk_target(
    sample: dict[str, torch.Tensor],
    *,
    frame_chunk_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return the frame-zero server input and its aligned action target."""

    latents = torch.as_tensor(sample["latents"])
    actions = torch.as_tensor(sample["actions"])
    masks = torch.as_tensor(sample["actions_mask"]).clone()
    if latents.ndim != 4:
        raise ValueError(f"Expected latents [C,F,H,W], got {tuple(latents.shape)}")
    if actions.ndim != 4 or masks.shape != actions.shape:
        raise ValueError(
            "Expected matching actions/actions_mask [C,F,N,1], got "
            f"{tuple(actions.shape)} and {tuple(masks.shape)}"
        )
    if int(latents.shape[1]) < frame_chunk_size:
        raise ValueError(
            f"Sample has {latents.shape[1]} latent frames, fewer than "
            f"frame_chunk_size={frame_chunk_size}"
        )
    masks[:, :1] = False
    return (
        latents[:, :1].contiguous(),
        actions[:, :frame_chunk_size].contiguous(),
        masks[:, :frame_chunk_size].contiguous(),
    )


def denormalize_training_action(
    normalized: torch.Tensor,
    mask: torch.Tensor,
    q01: torch.Tensor,
    q99: torch.Tensor,
    used_action_channel_ids: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    if normalized.ndim != 4 or mask.shape != normalized.shape:
        raise ValueError(
            "Expected matching normalized action/mask [C,F,N,1], got "
            f"{tuple(normalized.shape)} and {tuple(mask.shape)}"
        )
    used_ids = [int(value) for value in used_action_channel_ids]
    if normalized.shape[0] != q01.numel() or normalized.shape[0] != q99.numel():
        raise ValueError("Normalizer width must match canonical action width")
    normalized = normalized[..., 0].float()
    mask = mask[..., 0].bool()
    q01 = q01.float().reshape(-1, 1, 1)
    q99 = q99.float().reshape(-1, 1, 1)
    action = (normalized + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
    selected = action[used_ids].permute(1, 2, 0).reshape(-1, len(used_ids))
    selected_mask = mask[used_ids].permute(1, 2, 0).reshape(-1, len(used_ids))
    return selected.numpy(), selected_mask.numpy()


def physical_action_bounds(
    q01: torch.Tensor,
    q99: torch.Tensor,
    used_action_channel_ids: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    used_ids = [int(value) for value in used_action_channel_ids]
    lower = torch.as_tensor(q01).float().flatten()[used_ids].numpy()
    upper = torch.as_tensor(q99).float().flatten()[used_ids].numpy()
    if np.any(~np.isfinite(lower)) or np.any(~np.isfinite(upper)):
        raise ValueError("Action plot bounds must be finite")
    if np.any(upper <= lower):
        raise ValueError("Every action q99 bound must be greater than q01")
    return lower, upper


def resolve_action_history_mode(config: Any, requested: str) -> str:
    if requested in {"zero", "ground-truth"}:
        return requested
    if requested != "auto":
        raise ValueError(f"Unsupported action history mode: {requested}")
    probability = float(getattr(config, "action_history_condition_dropout_prob", 0.0))
    if probability == 1.0:
        return "zero"
    if probability == 0.0:
        return "ground-truth"
    raise ValueError(
        "A fractional action-history dropout probability has no unique evaluation "
        "contract; pass --action-history-mode explicitly"
    )


def has_valid_evaluation_target(mask: np.ndarray) -> bool:
    return bool(np.any(mask))


def _quat_angle_rad(prediction: np.ndarray, target: np.ndarray) -> np.ndarray:
    pred_norm = prediction / np.clip(np.linalg.norm(prediction, axis=-1, keepdims=True), 1e-12, None)
    target_norm = target / np.clip(np.linalg.norm(target, axis=-1, keepdims=True), 1e-12, None)
    dot = np.abs(np.sum(pred_norm * target_norm, axis=-1))
    return 2.0 * np.arccos(np.clip(dot, 0.0, 1.0))


def compute_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
) -> dict[str, Any]:
    if prediction.shape != target.shape or mask.shape != target.shape:
        raise ValueError("Prediction, target, and mask must have identical shapes")
    valid = mask.astype(bool)
    error = prediction - target
    values = error[valid]
    if values.size == 0:
        raise ValueError("Evaluation window has no valid action targets")
    metrics: dict[str, Any] = {
        "valid_values": int(values.size),
        "mae": float(np.mean(np.abs(values))),
        "rmse": float(np.sqrt(np.mean(np.square(values)))),
        "per_channel_mae": {
            name: float(np.mean(np.abs(error[:, index][valid[:, index]])))
            if np.any(valid[:, index])
            else None
            for index, name in enumerate(CHANNEL_NAMES)
        },
    }
    for side, base in (("left", 0), ("right", 8)):
        step_valid = np.all(valid[:, base : base + 7], axis=1)
        if not np.any(step_valid):
            continue
        pos_error = error[step_valid, base : base + 3]
        quat_error = _quat_angle_rad(
            prediction[step_valid, base + 3 : base + 7],
            target[step_valid, base + 3 : base + 7],
        )
        grip_valid = valid[:, base + 7]
        metrics[f"{side}_position_rmse_m"] = float(
            np.sqrt(np.mean(np.square(pos_error)))
        )
        metrics[f"{side}_rotation_mae_rad"] = float(np.mean(quat_error))
        metrics[f"{side}_gripper_mae"] = (
            float(np.mean(np.abs(error[grip_valid, base + 7])))
            if np.any(grip_valid)
            else None
        )
    return metrics


def _plot_curves(
    path: Path,
    prediction: np.ndarray,
    target: np.ndarray,
    mask: np.ndarray,
    control_fps: float,
    lower_bounds: np.ndarray,
    upper_bounds: np.ndarray,
    chunk_boundaries: list[int],
) -> None:
    time_s = np.arange(prediction.shape[0], dtype=np.float64) / float(control_fps)
    figure, axes = plt.subplots(4, 4, figsize=(18, 12), sharex=True)
    for index, (axis, name) in enumerate(zip(axes.flat, CHANNEL_NAMES)):
        valid = mask[:, index].astype(bool)
        visible_target = np.where(valid, target[:, index], np.nan)
        visible_prediction = np.where(valid, prediction[:, index], np.nan)
        axis.plot(time_s, visible_target, color="#222222", linewidth=1.5, label="ground truth")
        axis.plot(
            time_s,
            visible_prediction,
            color="#d1495b",
            linewidth=1.2,
            label="prediction",
        )
        for boundary in chunk_boundaries[:-1]:
            axis.axvline(
                float(boundary) / float(control_fps),
                color="#777777",
                linewidth=0.8,
                alpha=0.3,
            )
        axis.set_ylim(float(lower_bounds[index]), float(upper_bounds[index]))
        axis.set_title(name)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(loc="best", fontsize=8)
    for axis in axes[-1, :]:
        axis.set_xlabel("time (s)")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _load_empty_embedding(path: Path) -> torch.Tensor:
    value = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(value, dict):
        for key in ("text_emb", "prompt_embeds", "embedding"):
            if key in value:
                value = value[key]
                break
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"Empty embedding is not a tensor: {path}")
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-model", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, action="append", required=True)
    parser.add_argument("--config-name", default="nmx_chip_train", choices=tuple(VA_CONFIGS))
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--sample-index", type=int, action="append")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--num-init-worker", type=int, default=1)
    parser.add_argument(
        "--evaluation-mode",
        choices=(
            "deployment_native_first_chunk",
            "teacher_forced_video_full_episode",
        ),
        default="deployment_native_first_chunk",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        help="Limit chunks per episode for a smoke test; default evaluates the full sample",
    )
    parser.add_argument(
        "--action-history-mode",
        choices=("auto", "zero", "ground-truth"),
        default="auto",
        help="auto follows the training action-history dropout contract",
    )
    args = parser.parse_args()

    if args.episodes <= 0:
        parser.error("--episodes must be positive")
    transformer_path = _checkpoint_transformer(args.checkpoint)
    config = deepcopy(VA_CONFIGS[args.config_name])
    config.dataset_path = [str(path.expanduser().resolve()) for path in args.dataset_root]
    config.wan22_pretrained_model_name_or_path = str(args.base_model.expanduser().resolve())
    config.transformer_path = str(transformer_path)
    config.local_rank = int(args.gpu)
    config.rank = 0
    config.world_size = 1
    config.attn_mode = "torch"
    config.cfg_prob = 0.0
    config.max_latent_frames = None
    config.save_root = str(args.output_dir.expanduser().resolve())
    config.norm_stat = load_action_norm_stats(config.dataset_path[0], config)
    action_history_mode = resolve_action_history_mode(config, args.action_history_mode)
    output_dir = Path(config.save_root)
    output_dir.mkdir(parents=True, exist_ok=True)
    server_module.save_async = lambda *_args, **_kwargs: None

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    dataset = MultiLatentLeRobotDataset(
        config=config,
        num_init_worker=int(args.num_init_worker),
    )
    explicit_indices = [int(value) for value in args.sample_index] if args.sample_index else None
    if explicit_indices and any(index < 0 or index >= len(dataset) for index in explicit_indices):
        raise IndexError(
            f"Sample indices out of range for dataset length {len(dataset)}: "
            f"{explicit_indices}"
        )

    server = OfflineActionServer(config)
    empty_embedding = _load_empty_embedding(Path(config.empty_emb_path))
    records: list[dict[str, Any]] = []
    all_prediction: list[np.ndarray] = []
    all_target: list[np.ndarray] = []
    all_mask: list[np.ndarray] = []
    skipped_without_targets = 0
    selected_episodes: set[tuple[str, int]] = set()

    if explicit_indices is not None:
        candidates = iter(explicit_indices)
        requested_samples = len(explicit_indices)
    else:
        candidates = shuffled_sample_candidates(dataset, args.seed)
        requested_samples = int(args.episodes)

    for sample_index in candidates:
        metadata = _sample_metadata(dataset, sample_index)
        episode_key = (
            str(metadata["dataset_root"]),
            int(metadata["episode_index"]),
        )
        if explicit_indices is None and episode_key in selected_episodes:
            continue
        sample = dataset[sample_index]
        used_ids = [int(value) for value in config.used_action_channel_ids]
        q01 = torch.as_tensor(sample["action_q01"]).float()
        q99 = torch.as_tensor(sample["action_q99"]).float()
        lower_bounds, upper_bounds = physical_action_bounds(q01, q99, used_ids)
        if args.evaluation_mode == "deployment_native_first_chunk":
            condition_latent, target_action, target_mask = (
                deployment_native_first_chunk_target(
                    sample,
                    frame_chunk_size=int(config.frame_chunk_size),
                )
            )
            chunk_targets = [
                denormalize_training_action(
                    target_action,
                    target_mask,
                    q01,
                    q99,
                    used_ids,
                )
            ]
            chunks = []
        else:
            condition_latent, condition_action, chunks = split_episode_chunks(
                sample,
                frame_chunk_size=int(config.frame_chunk_size),
                grouping_start_from_one=bool(config.chunk_grouping_start_from_one),
                max_chunks=args.max_chunks,
            )
            chunk_targets = [
                denormalize_training_action(chunk.action, chunk.mask, q01, q99, used_ids)
                for chunk in chunks
            ]
        if not chunk_targets or not any(
            has_valid_evaluation_target(mask) for _, mask in chunk_targets
        ):
            if explicit_indices is not None:
                raise ValueError(
                    f"Sample index {sample_index} has no valid action targets in the "
                    "complete episode"
                )
            skipped_without_targets += 1
            continue
        if explicit_indices is None:
            selected_episodes.add(episode_key)

        ordinal = len(records)
        server.set_episode_latent_shape(sample["latents"])
        server._reset(prompt=None)
        server.prompt_embeds = _as_embedding(sample["text_emb"], server.device, server.dtype)
        server.negative_prompt_embeds = _as_embedding(empty_embedding, server.device, server.dtype)
        server.set_action_norm_stats(q01, q99)

        predictions: list[np.ndarray] = []
        targets: list[np.ndarray] = []
        masks: list[np.ndarray] = []
        chunk_boundaries: list[int] = []
        if args.evaluation_mode == "deployment_native_first_chunk":
            torch.manual_seed(args.seed + ordinal * 100_000)
            action, _latents = server._infer(
                {"video_latent": condition_latent},
                frame_st_id=0,
            )
            chunk_prediction = flatten_server_action(np.asarray(action))
            chunk_target, chunk_mask = chunk_targets[0]
            if chunk_prediction.shape != chunk_target.shape:
                raise ValueError(
                    "Deployment-native prediction/target shape mismatch: "
                    f"{chunk_prediction.shape} != {chunk_target.shape}"
                )
            predictions.append(chunk_prediction)
            targets.append(chunk_target)
            masks.append(chunk_mask)
            chunk_boundaries.append(int(chunk_prediction.shape[0]))
        else:
            server.cache_training_context(condition_latent, condition_action)
            for chunk_index, (chunk, (chunk_target, chunk_mask)) in enumerate(
                zip(chunks, chunk_targets, strict=True)
            ):
                if server.frame_st_id != chunk.start_latent:
                    raise RuntimeError(
                        "KV-cache timeline drifted before chunk "
                        f"{chunk_index}: {server.frame_st_id} != {chunk.start_latent}"
                    )
                torch.manual_seed(args.seed + ordinal * 100_000 + chunk_index)
                action, _latents = server._infer({}, frame_st_id=server.frame_st_id)
                chunk_prediction = flatten_server_action(np.asarray(action))
                expected_steps = int(chunk.action.shape[1] * chunk.action.shape[2])
                chunk_prediction = chunk_prediction[:expected_steps]
                if chunk_prediction.shape != chunk_target.shape:
                    raise ValueError(
                        "Prediction/target shape mismatch in chunk "
                        f"{chunk_index}: {chunk_prediction.shape} != {chunk_target.shape}"
                    )
                predictions.append(chunk_prediction)
                targets.append(chunk_target)
                masks.append(chunk_mask)
                chunk_boundaries.append(sum(value.shape[0] for value in predictions))

                if chunk_index + 1 < len(chunks):
                    history_action = (
                        torch.zeros_like(chunk.action)
                        if action_history_mode == "zero"
                        else chunk.action
                    )
                    server.cache_training_context(chunk.latent, history_action)

        prediction = np.concatenate(predictions)
        target = np.concatenate(targets)
        mask = np.concatenate(masks)
        metrics = compute_metrics(prediction, target, mask)
        stem = f"sample_{sample_index:06d}_episode_{int(metadata['episode_index']):06d}"
        np.savez_compressed(
            output_dir / f"{stem}.npz",
            prediction=prediction,
            target=target,
            mask=mask,
            channel_names=np.asarray(CHANNEL_NAMES),
            lower_bounds=lower_bounds,
            upper_bounds=upper_bounds,
            chunk_boundaries=np.asarray(chunk_boundaries, dtype=np.int64),
            condition_action_steps=np.asarray(int(config.action_per_frame)),
        )
        _plot_curves(
            output_dir / f"{stem}.png",
            prediction,
            target,
            mask,
            float(config.control_fps),
            lower_bounds,
            upper_bounds,
            chunk_boundaries,
        )
        valid_steps = np.any(mask, axis=1)
        records.append(
            {
                "sample": metadata,
                "mode": args.evaluation_mode,
                "action_history_mode": action_history_mode,
                "total_latent_frames": int(sample["latents"].shape[1]),
                "condition_latent_frames": 1,
                "predicted_latent_frames": (
                    int(config.frame_chunk_size) - 1
                    if args.evaluation_mode == "deployment_native_first_chunk"
                    else int(sum(chunk.latent.shape[1] for chunk in chunks))
                ),
                "chunks": (
                    1
                    if args.evaluation_mode == "deployment_native_first_chunk"
                    else len(chunks)
                ),
                "predicted_action_steps": int(prediction.shape[0]),
                "valid_action_steps": int(np.count_nonzero(valid_steps)),
                "covered_seconds": float(prediction.shape[0] / config.control_fps),
                "chunk_boundaries": chunk_boundaries,
                "plot_lower_bounds_q01": lower_bounds.tolist(),
                "plot_upper_bounds_q99": upper_bounds.tolist(),
                "metrics": metrics,
            }
        )
        all_prediction.append(prediction)
        all_target.append(target)
        all_mask.append(mask)
        if len(records) == requested_samples:
            break

    if len(records) != requested_samples:
        raise ValueError(
            f"Requested {requested_samples} samples with valid action targets but found "
            f"only {len(records)}; skipped {skipped_without_targets} invalid windows"
        )

    aggregate = compute_metrics(
        np.concatenate(all_prediction),
        np.concatenate(all_target),
        np.concatenate(all_mask),
    )
    summary = {
        "schema_version": 2,
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "transformer": str(transformer_path),
        "config_name": args.config_name,
        "visual_contract": str(config.visual_contract),
        "evaluation_mode": args.evaluation_mode,
        "action_history_mode": action_history_mode,
        "chunk_grouping_start_from_one": bool(config.chunk_grouping_start_from_one),
        "max_chunks": args.max_chunks,
        "dataset_roots": list(config.dataset_path),
        "seed": int(args.seed),
        "evaluated_episodes": len(records),
        "skipped_episodes_without_valid_targets": skipped_without_targets,
        "condition_action_steps_excluded_once_per_episode": int(config.action_per_frame),
        "control_fps": float(config.control_fps),
        "plot_scale": "denormalized_physical_units",
        "plot_y_limits": "per_channel_q01_q99",
        "aggregate": aggregate,
        "episodes": records,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
