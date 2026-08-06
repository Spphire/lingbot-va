"""Training config for the recorded NMX chip dataset group."""

import os
from copy import deepcopy

from easydict import EasyDict

from .shared_config import va_shared_cfg


_RECORDED_DATASET_PATHS = [
    "/mnt/workspace/shenyibo/datasets/chip_0709_1952episodes",
    "/mnt/workspace/shenyibo/datasets/chip_0711_199episodes",
    "/mnt/workspace/shenyibo/datasets/chip_paralle_0710_742episodes",
    "/mnt/workspace/shenyibo/datasets/chip_paralle_0712_1048episodes",
    "/mnt/workspace/shenyibo/datasets/chip_recovery_0709_449episodes",
    "/mnt/workspace/shenyibo/datasets/chip_recovery_0711_121episodes",
    "/mnt/workspace/shenyibo/datasets/chip_recovery_clamp_0712_367episodes",
    "/mnt/workspace/shenyibo/datasets/chip_recovery_clamp_561episodes",
    "/mnt/workspace/shenyibo/datasets/chip_recovery_place_700episodes",
    "/mnt/workspace/shenyibo/datasets/chip_recovery_slide_0712_389episodes",
    "/mnt/workspace/shenyibo/datasets/chip_recovery_slide_672episodes",
    "/mnt/workspace/shenyibo/datasets/chip_reset_0709_454episodes",
    "/mnt/workspace/shenyibo/datasets/waic_chip_0715",
]


va_nmx_chip_train_cfg = EasyDict(__name__="Config: NMX chip train")
va_nmx_chip_train_cfg.update(va_shared_cfg)

configured_paths = os.getenv("NMX_CHIP_DATASET_PATHS")
va_nmx_chip_train_cfg.dataset_path = (
    configured_paths.split(os.pathsep) if configured_paths else _RECORDED_DATASET_PATHS
)
va_nmx_chip_train_cfg.empty_emb_path = os.getenv(
    "NMX_CHIP_EMPTY_EMB_PATH",
    "/mnt/workspace/shenyibo/datasets/chip_0711_199episodes/empty_emb.pt",
)
va_nmx_chip_train_cfg.text_emb_override_path = os.getenv(
    "NMX_CHIP_TEXT_EMB_OVERRIDE_PATH",
    "/mnt/workspace/shenyibo/datasets/chip_paralle_0712_1048episodes/"
    "meta/latent_generation_native_10fps/prompt_text_emb.pt",
)
va_nmx_chip_train_cfg.text_emb_override_shape = (512, 4096)
va_nmx_chip_train_cfg.text_emb_override_dtype = "torch.bfloat16"
va_nmx_chip_train_cfg.text_emb_override_sha256 = (
    "04dd65d97b83e80e65594db57d6ddd5f58f0ec72d2734ab7768493ed745da5c5"
)
va_nmx_chip_train_cfg.task_prompt_override = """Hardware: 310
Dataset: chip
FPS: 10
Action Space: state; 23d total; 7d left arm eef pose; 1d left gripper; 7d right arm eef pose; 1d right gripper; 7d head eef pose
Task: Place the chip into the wooden box based on the soccer ball's position"""
va_nmx_chip_train_cfg.wan22_pretrained_model_name_or_path = os.getenv(
    "LINGBOT_VA_MODEL_PATH",
    "/path/to/pretrained/model",
)

# The recorded run dropped the head view and head action with probability 1.
# This config keeps that supervision choice while preserving the upstream model's
# single-canvas input instead of porting nmx_vla's independent multiview slots.
va_nmx_chip_train_cfg.obs_cam_keys = [
    "observation.images.wrist_image_1",
    "observation.images.wrist_image_2",
]
va_nmx_chip_train_cfg.env_type = "none"
va_nmx_chip_train_cfg.height = 320
va_nmx_chip_train_cfg.width = 240
va_nmx_chip_train_cfg.visual_contract = "upstream_single_canvas_v1"
va_nmx_chip_train_cfg.expected_latent_view_shapes = [(20, 15), (20, 15)]
va_nmx_chip_train_cfg.expected_latent_channels = 48
va_nmx_chip_train_cfg.action_dim = 30
va_nmx_chip_train_cfg.action_per_frame = 12
va_nmx_chip_train_cfg.actions_per_frame = 1
va_nmx_chip_train_cfg.action_condition_mode = "inverse_dynamics"

va_nmx_chip_train_cfg.action_contract = "nmx_chunk_relative_v10"
va_nmx_chip_train_cfg.action_chunk_size_min = 1
va_nmx_chip_train_cfg.action_chunk_size_max = 4
va_nmx_chip_train_cfg.window_size_min = 3
va_nmx_chip_train_cfg.window_size_max = 3
va_nmx_chip_train_cfg.chunk_grouping_start_from_one = True
va_nmx_chip_train_cfg.max_latent_frames = 116
va_nmx_chip_train_cfg.action_history_condition_dropout_prob = 1.0
va_nmx_chip_train_cfg.action_history_condition_dropout_mode = (
    "zero_normalized_clean_condition"
)
va_nmx_chip_train_cfg.relative_pose_frame = "local_frame"
va_nmx_chip_train_cfg.quaternion_order = "wxyz"
va_nmx_chip_train_cfg.relative_pose_groups = [
    {"pose_slice": [0, 7], "gripper_slice": [7, 8]},
    {"pose_slice": [8, 15], "gripper_slice": [15, 16]},
]
va_nmx_chip_train_cfg.used_action_channel_ids = (
    list(range(0, 7)) + [28] + list(range(7, 14)) + [29]
)
va_nmx_chip_train_cfg.gripper_canonical_dims = [28, 29]
va_nmx_chip_train_cfg.gripper_raw_dims = [7, 15]
inverse_ids = [len(va_nmx_chip_train_cfg.used_action_channel_ids)] * 30
for source_index, canonical_index in enumerate(
    va_nmx_chip_train_cfg.used_action_channel_ids
):
    inverse_ids[canonical_index] = source_index
va_nmx_chip_train_cfg.inverse_used_action_channel_ids = inverse_ids
va_nmx_chip_train_cfg.action_norm_stats_filename = "lingbot_action_norm_stats.json"
va_nmx_chip_train_cfg.action_norm_stats_version = (
    "chunk_relative_v10_velocity_symmetric_scale"
)
va_nmx_chip_train_cfg.action_norm_method = "quantiles"
va_nmx_chip_train_cfg.action_quaternion_order = "xyzw"
va_nmx_chip_train_cfg.action_history_adds_parameters = False

# Persist the raw-camera and streaming contract used by the training runs so
# deployment and offline evaluation can reproduce the same model inputs.
va_nmx_chip_train_cfg.raw_image_hw = [360, 640]
va_nmx_chip_train_cfg.reshape_mode = "crop_resize"
va_nmx_chip_train_cfg.center_crop_before_resize = False
va_nmx_chip_train_cfg.crop_before_resize = [185, 185, 0, 0]
va_nmx_chip_train_cfg.crop_resize_size = [320, 240]
va_nmx_chip_train_cfg.pad_after_resize = None
va_nmx_chip_train_cfg.camera_rotation_degrees = [0, 0]
va_nmx_chip_train_cfg.control_action_dim = 16
va_nmx_chip_train_cfg.control_chunk_length = 48
va_nmx_chip_train_cfg.source_action_per_frame = 1
va_nmx_chip_train_cfg.source_video_frame_stride = 3
va_nmx_chip_train_cfg.video_frames_per_latent = 4
va_nmx_chip_train_cfg.sampled_video_fps = 10.0
va_nmx_chip_train_cfg.control_fps = 30.0
va_nmx_chip_train_cfg.first_prediction_skip_action_latents = 1

va_nmx_chip_train_cfg.attn_window = 3
va_nmx_chip_train_cfg.frame_chunk_size = 4
va_nmx_chip_train_cfg.guidance_scale = 5
va_nmx_chip_train_cfg.action_guidance_scale = 1
va_nmx_chip_train_cfg.num_inference_steps = 25
va_nmx_chip_train_cfg.video_exec_step = -1
va_nmx_chip_train_cfg.action_num_inference_steps = 50
va_nmx_chip_train_cfg.snr_shift = 5.0
va_nmx_chip_train_cfg.action_snr_shift = 1.0

va_nmx_chip_train_cfg.enable_wandb = False
va_nmx_chip_train_cfg.load_worker = 4
va_nmx_chip_train_cfg.num_init_worker = 8
va_nmx_chip_train_cfg.save_interval = 4000
va_nmx_chip_train_cfg.gc_interval = 50
va_nmx_chip_train_cfg.cfg_prob = 0.1
va_nmx_chip_train_cfg.learning_rate = 1e-5
va_nmx_chip_train_cfg.beta1 = 0.9
va_nmx_chip_train_cfg.beta2 = 0.95
va_nmx_chip_train_cfg.weight_decay = 0.1
va_nmx_chip_train_cfg.warmup_steps = 10
va_nmx_chip_train_cfg.batch_size = 1
va_nmx_chip_train_cfg.gradient_accumulation_steps = 1
va_nmx_chip_train_cfg.gradient_clipping = 2.0
va_nmx_chip_train_cfg.num_steps = 20000
va_nmx_chip_train_cfg.seed = 42


# A visual-only A/B control. All data, action, optimizer, and model settings are
# deep-copied from the upstream single-canvas baseline.
va_nmx_chip_train_per_view_pad_cfg = deepcopy(va_nmx_chip_train_cfg)
va_nmx_chip_train_per_view_pad_cfg.__name__ = (
    "Config: NMX chip train, per-view padding"
)
va_nmx_chip_train_per_view_pad_cfg.visual_contract = (
    "per_view_zero_pad_then_concat_v1"
)


# FastWAM keeps the original shared LingBot-VA backbone and changes only the
# action-conditioning attention contract. The A/B pair remains visual-only.
va_nmx_chip_train_fastwam_cfg = deepcopy(va_nmx_chip_train_cfg)
va_nmx_chip_train_fastwam_cfg.__name__ = (
    "Config: NMX chip train, FastWAM action conditioning"
)
va_nmx_chip_train_fastwam_cfg.action_condition_mode = "fastwam"

# Optional MoT structure with the same upstream 30D/FastWAM data contract.
va_nmx_chip_train_mot_cfg = deepcopy(va_nmx_chip_train_fastwam_cfg)
va_nmx_chip_train_mot_cfg.__name__ = "Config: NMX chip train, MoT FastWAM"
va_nmx_chip_train_mot_cfg.model_structure = "mot"
va_nmx_chip_train_mot_cfg.mot_config = {"action_hidden_dim": 768, "action_mlp_hidden_dim": 256, "init_mode": "video_interp_alpha"}


# Fold-towel task config. Only task-owned data assets differ from the existing
# MoT + FastWAM upstream single-canvas baseline.
va_nmx_fold_towel_train_mot_fastwam_cfg = deepcopy(va_nmx_chip_train_mot_cfg)
va_nmx_fold_towel_train_mot_fastwam_cfg.__name__ = (
    "Config: NMX fold towel train, MoT FastWAM, upstream single-canvas"
)
va_nmx_fold_towel_train_mot_fastwam_cfg.dataset_path = [
    "/mnt/dataset/fold_towel_correction_20260804",
    "/mnt/dataset/fold_towel_overall_20260804",
]
va_nmx_fold_towel_train_mot_fastwam_cfg.empty_emb_path = (
    "/mnt/dataset/fold_towel_overall_20260804/empty_emb.pt"
)
va_nmx_fold_towel_train_mot_fastwam_cfg.text_emb_override_path = (
    "/mnt/workspace/shenyibo/lingbot-va-assets/fold_towel_20260804/prompt_text_emb.pt"
)
va_nmx_fold_towel_train_mot_fastwam_cfg.text_emb_override_sha256 = (
    "31bf55eb231ce3a5b4d0b0b3a6b8e262d7dc79ea51bfc9381e6dd8947b12727b"
)
va_nmx_fold_towel_train_mot_fastwam_cfg.task_prompt_override = (
    "Fold the towel in half twice"
)
va_nmx_fold_towel_train_mot_fastwam_cfg.action_norm_stats_path = (
    "/mnt/workspace/shenyibo/lingbot-va-assets/fold_towel_20260804/lingbot_action_norm_stats.json"
)


va_nmx_chip_train_per_view_pad_fastwam_cfg = deepcopy(
    va_nmx_chip_train_per_view_pad_cfg
)
va_nmx_chip_train_per_view_pad_fastwam_cfg.__name__ = (
    "Config: NMX chip train, per-view padding, FastWAM action conditioning"
)
va_nmx_chip_train_per_view_pad_fastwam_cfg.action_condition_mode = "fastwam"


va_nmx_chip_episode109_overfit_cfg = deepcopy(va_nmx_chip_train_cfg)
va_nmx_chip_episode109_overfit_cfg.__name__ = (
    "Config: NMX chip episode 109 overfit audit"
)
va_nmx_chip_episode109_overfit_cfg.dataset_path = [
    "/mnt/workspace/shenyibo/datasets/chip_0711_199episodes"
]
va_nmx_chip_episode109_overfit_cfg.episode_index_filter = [109]
va_nmx_chip_episode109_overfit_cfg.max_latent_frames = None
va_nmx_chip_episode109_overfit_cfg.cfg_prob = 0.0
va_nmx_chip_episode109_overfit_cfg.load_worker = 1
va_nmx_chip_episode109_overfit_cfg.num_init_worker = 1
va_nmx_chip_episode109_overfit_cfg.num_steps = 1000
va_nmx_chip_episode109_overfit_cfg.save_interval = 100
