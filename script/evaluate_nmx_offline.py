"""Evaluate a LingBot-VA checkpoint against cached training episodes."""

from __future__ import annotations

import argparse
from copy import deepcopy
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

    def _encode_obs(self, obs: dict[str, Any]) -> torch.Tensor:
        del obs
        if self._offline_init_latent is None:
            raise RuntimeError("Offline initial latent has not been set")
        return self._offline_init_latent


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


def denormalize_ground_truth(
    sample: dict[str, torch.Tensor],
    config: Any,
) -> tuple[np.ndarray, np.ndarray]:
    frames = int(config.frame_chunk_size)
    used_ids = [int(value) for value in config.used_action_channel_ids]
    normalized = sample["actions"][:, :frames, :, 0].float()
    mask = sample["actions_mask"][:, :frames, :, 0].bool()
    q01 = sample["action_q01"].float().reshape(-1, 1, 1)
    q99 = sample["action_q99"].float().reshape(-1, 1, 1)
    action = (normalized + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
    selected = action[used_ids].permute(1, 2, 0).reshape(-1, len(used_ids))
    selected_mask = mask[used_ids].permute(1, 2, 0).reshape(-1, len(used_ids))
    return selected.numpy(), selected_mask.numpy()


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
) -> None:
    time_s = np.arange(prediction.shape[0], dtype=np.float64) / float(control_fps)
    figure, axes = plt.subplots(4, 4, figsize=(18, 12), sharex=True)
    for index, (axis, name) in enumerate(zip(axes.flat, CHANNEL_NAMES)):
        valid = mask[:, index].astype(bool)
        axis.plot(time_s, target[:, index], color="#222222", linewidth=1.5, label="ground truth")
        axis.plot(time_s, prediction[:, index], color="#d1495b", linewidth=1.2, label="prediction")
        if np.any(~valid):
            axis.scatter(time_s[~valid], target[~valid, index], color="#aaaaaa", s=8, label="masked")
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
    config.save_root = str(args.output_dir.expanduser().resolve())
    config.norm_stat = load_action_norm_stats(config.dataset_path[0], config)
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
        target, mask = denormalize_ground_truth(sample, config)
        if not has_valid_evaluation_target(mask):
            if explicit_indices is not None:
                raise ValueError(
                    f"Sample index {sample_index} has no valid action targets in the "
                    "evaluation window"
                )
            skipped_without_targets += 1
            continue
        if explicit_indices is None:
            selected_episodes.add(episode_key)

        ordinal = len(records)
        server._reset(prompt=None)
        server.prompt_embeds = _as_embedding(sample["text_emb"], server.device, server.dtype)
        server.negative_prompt_embeds = _as_embedding(empty_embedding, server.device, server.dtype)
        server._offline_init_latent = sample["latents"][None, :, :1].to(
            device=server.device,
            dtype=server.dtype,
        )
        torch.manual_seed(args.seed + ordinal)
        action, _latents = server._infer({}, frame_st_id=0)
        prediction = flatten_server_action(np.asarray(action))
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction/target shape mismatch: {prediction.shape} != {target.shape}"
            )
        metrics = compute_metrics(prediction, target, mask)
        stem = f"sample_{sample_index:06d}_episode_{int(metadata['episode_index']):06d}"
        np.savez_compressed(
            output_dir / f"{stem}.npz",
            prediction=prediction,
            target=target,
            mask=mask,
            channel_names=np.asarray(CHANNEL_NAMES),
        )
        _plot_curves(
            output_dir / f"{stem}.png",
            prediction,
            target,
            mask,
            float(config.control_fps),
        )
        records.append({"sample": metadata, "metrics": metrics})
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
        "schema_version": 1,
        "checkpoint": str(args.checkpoint.expanduser().resolve()),
        "transformer": str(transformer_path),
        "config_name": args.config_name,
        "visual_contract": str(config.visual_contract),
        "dataset_roots": list(config.dataset_path),
        "seed": int(args.seed),
        "evaluated_episodes": len(records),
        "skipped_windows_without_valid_targets": skipped_without_targets,
        "ignored_condition_steps_per_window": int(config.action_per_frame),
        "control_fps": float(config.control_fps),
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
