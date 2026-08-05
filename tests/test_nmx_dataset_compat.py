import json
import hashlib
from types import SimpleNamespace

import numpy as np
import torch

from wan_va.chunking import build_chunk_ids, iter_chunk_slices
from wan_va.configs import VA_CONFIGS
from wan_va.dataset.lerobot_latent_dataset import (
    PER_VIEW_ZERO_PAD_VISUAL_CONTRACT,
    UPSTREAM_SINGLE_CANVAS_VISUAL_CONTRACT,
    LatentLeRobotDataset,
    _load_text_emb_override,
    compose_nmx_latent_views,
    discover_dataset_roots,
)
from wan_va.dataset.nmx_action_adapter import (
    apply_episode_action_validity,
    build_model_actions_from_raw,
    load_action_norm_stats,
    prepare_raw_action_tensor,
    rebuild_batch_actions,
)


def _config(**overrides):
    values = {
        "action_dim": 30,
        "actions_per_frame": 1,
        "action_chunk_size_max": 4,
        "chunk_grouping_start_from_one": True,
        "relative_pose_frame": "local_frame",
        "quaternion_order": "wxyz",
        "relative_pose_groups": [
            {"pose_slice": [0, 7], "gripper_slice": [7, 8]},
            {"pose_slice": [8, 15], "gripper_slice": [15, 16]},
        ],
        "used_action_channel_ids": (
            list(range(0, 7)) + [28] + list(range(7, 14)) + [29]
        ),
        "gripper_canonical_dims": [28, 29],
        "gripper_raw_dims": [7, 15],
        "action_norm_stats_version": "chunk_relative_v10_velocity_symmetric_scale",
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _pose(x=0.0, y=0.0, z=0.0, quat=None):
    # Source order is wxyz.
    quat = [1.0, 0.0, 0.0, 0.0] if quat is None else quat
    return [x, y, z, *quat]


def test_offset_chunk_grouping_matches_attention_ids():
    slices = list(
        iter_chunk_slices(
            8,
            3,
            grouping_start_from_one=True,
            prefix_step_size=1,
        )
    )
    assert [(item.start, item.stop) for item in slices] == [
        (0, 1),
        (1, 4),
        (4, 7),
        (7, 8),
    ]
    ids = build_chunk_ids(
        torch.arange(8),
        3,
        grouping_start_from_one=True,
    )
    assert ids.tolist() == [0, 1, 1, 1, 2, 2, 2, 3]


def test_prepare_raw_actions_masks_head_and_tail_padding():
    raw, mask = prepare_raw_action_tensor(
        local_start_frame=0,
        latent_frame_ids=[0, 3, 6, 9, 12],
        latent_frame_num=2,
        video_num_frames=5,
        action=torch.ones(1, 23),
        config=_config(),
    )

    # stride=3, sampled video frames per latent=ceil(5/2)=3, so N=9.
    assert raw.shape == (2, 9, 23)
    assert mask.shape == (2, 9)
    assert torch.count_nonzero(mask[0]) == 0
    assert torch.count_nonzero(mask[1]) == 1


def test_dataset_root_discovery_accepts_an_ordered_path_list(tmp_path):
    direct_root = tmp_path / "direct"
    nested_root = tmp_path / "collection" / "nested"
    for root in (direct_root, nested_root):
        (root / "meta").mkdir(parents=True)
        (root / "meta" / "info.json").write_text("{}", encoding="utf-8")

    roots = discover_dataset_roots(
        [direct_root, nested_root.parent, direct_root]
    )

    assert roots == [str(direct_root), str(nested_root)]


def _visual_config(visual_contract):
    return _config(
        env_type="none",
        patch_size=(1, 2, 2),
        visual_contract=visual_contract,
        expected_latent_view_shapes=[(20, 15), (20, 15)],
        expected_latent_channels=1,
    )


def test_upstream_single_canvas_keeps_raw_20x30_geometry():
    left = torch.ones(2, 20, 15, 1)
    right = torch.full_like(left, 2)

    composed = compose_nmx_latent_views(
        [left, right],
        _visual_config(UPSTREAM_SINGLE_CANVAS_VISUAL_CONTRACT),
    )

    assert composed.shape == (2, 20, 30, 1)
    assert (composed.shape[1] // 2) * (composed.shape[2] // 2) == 150
    # This cross-view pair is intentionally part of the upstream baseline.
    torch.testing.assert_close(composed[0, 0, 14:16, 0], torch.tensor([1.0, 2.0]))


def test_per_view_padding_builds_20x32_without_cross_view_patch():
    left = torch.ones(2, 20, 15, 1)
    right = torch.full_like(left, 2)

    composed = compose_nmx_latent_views(
        [left, right],
        _visual_config(PER_VIEW_ZERO_PAD_VISUAL_CONTRACT),
    )

    assert composed.shape == (2, 20, 32, 1)
    assert (composed.shape[1] // 2) * (composed.shape[2] // 2) == 160
    torch.testing.assert_close(
        composed[0, 0, 14:18, 0],
        torch.tensor([1.0, 0.0, 2.0, 2.0]),
    )
    assert composed[0, 0, 31, 0] == 0


def test_visual_contract_rejects_unexpected_view_geometry():
    config = _visual_config(UPSTREAM_SINGLE_CANVAS_VISUAL_CONTRACT)

    try:
        compose_nmx_latent_views(
            [torch.zeros(1, 20, 15, 1), torch.zeros(1, 20, 16, 1)],
            config,
        )
    except ValueError as exc:
        assert "expected [(20, 15), (20, 15)]" in str(exc)
    else:
        raise AssertionError("unexpected latent geometry must fail fast")


def test_visual_ab_configs_differ_only_in_visual_contract():
    baseline = VA_CONFIGS["nmx_chip_train"]
    padded = VA_CONFIGS["nmx_chip_train_per_view_pad"]

    assert baseline.visual_contract == UPSTREAM_SINGLE_CANVAS_VISUAL_CONTRACT
    assert padded.visual_contract == PER_VIEW_ZERO_PAD_VISUAL_CONTRACT
    assert len(baseline.dataset_path) == 13
    assert all(path.startswith("/mnt/workspace/shenyibo/datasets/") for path in baseline.dataset_path)
    assert baseline.gradient_accumulation_steps == 1
    assert baseline.batch_size == 1
    assert baseline.num_steps == 20000
    assert baseline.save_interval == 4000
    assert baseline.action_history_condition_dropout_prob == 1.0
    assert baseline.action_history_condition_dropout_mode == (
        "zero_normalized_clean_condition"
    )
    baseline_values = dict(baseline)
    padded_values = dict(padded)
    for key in ("__name__", "visual_contract"):
        baseline_values.pop(key, None)
        padded_values.pop(key, None)
    assert baseline_values == padded_values

    for key in (
        "dataset_path",
        "expected_latent_view_shapes",
        "relative_pose_groups",
        "used_action_channel_ids",
        "inverse_used_action_channel_ids",
    ):
        assert baseline[key] is not padded[key]
    assert baseline.relative_pose_groups[0] is not padded.relative_pose_groups[0]


def test_fastwam_visual_ab_configs_change_only_action_mode_and_visual_contract():
    idm_baseline = VA_CONFIGS["nmx_chip_train"]
    fastwam_baseline = VA_CONFIGS["nmx_chip_train_fastwam"]
    fastwam_padded = VA_CONFIGS["nmx_chip_train_per_view_pad_fastwam"]

    assert idm_baseline.action_condition_mode == "inverse_dynamics"
    assert fastwam_baseline.action_condition_mode == "fastwam"
    assert fastwam_padded.action_condition_mode == "fastwam"
    assert idm_baseline.action_norm_method == "quantiles"
    assert fastwam_baseline.action_norm_method == idm_baseline.action_norm_method
    assert idm_baseline.raw_image_hw == [360, 640]
    assert idm_baseline.crop_before_resize == [185, 185, 0, 0]
    assert idm_baseline.crop_resize_size == [320, 240]
    assert idm_baseline.control_chunk_length == 48
    assert idm_baseline.control_fps == 30.0
    assert fastwam_baseline.visual_contract == UPSTREAM_SINGLE_CANVAS_VISUAL_CONTRACT
    assert fastwam_padded.visual_contract == PER_VIEW_ZERO_PAD_VISUAL_CONTRACT

    idm_values = dict(idm_baseline)
    fastwam_values = dict(fastwam_baseline)
    for key in ("__name__", "action_condition_mode"):
        idm_values.pop(key, None)
        fastwam_values.pop(key, None)
    assert idm_values == fastwam_values

    baseline_values = dict(fastwam_baseline)
    padded_values = dict(fastwam_padded)
    for key in ("__name__", "visual_contract"):
        baseline_values.pop(key, None)
        padded_values.pop(key, None)
    assert baseline_values == padded_values


def test_text_embedding_override_is_loaded_once_and_replaces_cached_text(tmp_path):
    override = torch.arange(12, dtype=torch.bfloat16).reshape(3, 4)
    override_path = tmp_path / "prompt_text_emb.pt"
    torch.save(override, override_path)
    digest = hashlib.sha256(
        override.contiguous().view(torch.uint8).numpy().tobytes()
    ).hexdigest()
    config = _visual_config(UPSTREAM_SINGLE_CANVAS_VISUAL_CONTRACT)
    config.text_emb_override_path = str(override_path)
    config.text_emb_override_shape = (3, 4)
    config.text_emb_override_dtype = "torch.bfloat16"
    config.text_emb_override_sha256 = digest

    loaded = _load_text_emb_override(config)
    assert _load_text_emb_override(config) is loaded

    dataset = object.__new__(LatentLeRobotDataset)
    dataset.config = config
    dataset.nmx_action_contract = True
    dataset.used_video_keys = ["left", "right"]
    dataset.text_emb_override = loaded
    dataset.empty_emb = None
    dataset.cfg_prob = 0.0
    data = {"left.text_emb": torch.full((3, 4), -1, dtype=torch.bfloat16)}
    for key in dataset.used_video_keys:
        data[f"{key}.latent"] = torch.ones(20 * 15, 1)
        data[f"{key}.latent_num_frames"] = 1
        data[f"{key}.latent_height"] = 20
        data[f"{key}.latent_width"] = 15

    sample = dataset._cat_video_latents(data)
    assert sample["text_emb"] is loaded
    torch.testing.assert_close(sample["text_emb"], override)


def test_temporal_clip_keeps_latents_and_frame_ids_aligned(monkeypatch):
    dataset = object.__new__(LatentLeRobotDataset)
    dataset.config = _config(max_latent_frames=4)
    dataset.nmx_action_contract = True
    dataset.used_video_keys = ["left", "right"]

    source_frames = 8
    latent_height = 2
    latent_width = 3
    channels = 2
    frame_ids = list(range(0, 87, 3))
    data = {}
    originals = {}
    for view_index, key in enumerate(dataset.used_video_keys):
        latent = torch.arange(
            source_frames * latent_height * latent_width * channels,
            dtype=torch.float32,
        ).reshape(-1, channels)
        latent = latent + view_index * 1000
        originals[key] = latent.clone()
        data.update(
            {
                f"{key}.latent": latent,
                f"{key}.latent_num_frames": source_frames,
                f"{key}.latent_height": latent_height,
                f"{key}.latent_width": latent_width,
                f"{key}.video_num_frames": len(frame_ids),
                f"{key}.frame_ids": frame_ids.copy(),
            }
        )

    monkeypatch.setattr(np.random, "randint", lambda *args: 2)
    clipped = dataset._clip_temporal_latents(data)

    tokens_per_frame = latent_height * latent_width
    expected_ids = frame_ids[8:21]
    for key in dataset.used_video_keys:
        torch.testing.assert_close(
            clipped[f"{key}.latent"],
            originals[key][2 * tokens_per_frame : 6 * tokens_per_frame],
        )
        assert clipped[f"{key}.latent_num_frames"] == 4
        assert clipped[f"{key}.video_num_frames"] == len(expected_ids)
        assert clipped[f"{key}.frame_ids"] == expected_ids


def test_episode_metadata_applies_pose_and_gripper_validity():
    dataset = object.__new__(LatentLeRobotDataset)
    dataset.nmx_action_contract = True
    dataset.meta = SimpleNamespace(
        episodes={
            0: {
                "episode_index": 0,
                "tasks": ["task"],
                "action_config": [
                    {"start_frame": 0, "end_frame": 12, "action_text": "task"}
                ],
                "slam_validity_all": True,
                "width_detection": False,
                "width_calibration": True,
            }
        }
    )
    dataset._check_meta = lambda *args: True
    dataset.parse_meta()

    assert dataset.new_metas[0]["slam_validity_all"] is True
    assert dataset.new_metas[0]["width_validity"] is False
    sample = {
        "actions": torch.ones(30, 2, 3, 1),
        "actions_mask": torch.ones(30, 2, 3, 1, dtype=torch.bool),
        "raw_actions": torch.ones(2, 3, 23),
        "raw_actions_step_mask": torch.ones(2, 3, dtype=torch.bool),
    }
    apply_episode_action_validity(
        sample,
        _config(),
        action_valid=True,
        width_valid=False,
    )

    assert torch.all(sample["actions_mask"][:28])
    assert not sample["actions_mask"][28:].any()
    assert torch.all(sample["raw_actions"][..., :7])
    assert not sample["raw_actions"][..., [7, 15]].any()
    assert torch.all(sample["raw_actions_step_mask"])

    apply_episode_action_validity(
        sample,
        _config(),
        action_valid=False,
        width_valid=False,
    )

    assert all(torch.count_nonzero(value) == 0 for value in sample.values())


def test_rebuilt_actions_preserve_dataset_channel_mask(monkeypatch):
    rebuilt_actions = torch.ones(30, 2, 1, 1)
    rebuilt_mask = torch.ones_like(rebuilt_actions, dtype=torch.bool)
    monkeypatch.setattr(
        "wan_va.dataset.nmx_action_adapter.build_model_actions_from_raw",
        lambda *args, **kwargs: (rebuilt_actions.clone(), rebuilt_mask.clone()),
    )
    dataset_mask = torch.ones(1, 30, 2, 1, 1, dtype=torch.bool)
    dataset_mask[:, [28, 29]] = False
    batch = {
        "raw_actions": torch.zeros(1, 2, 1, 23),
        "raw_states": torch.zeros(1, 2, 1, 23),
        "raw_actions_step_mask": torch.ones(1, 2, 1, dtype=torch.bool),
        "action_q01": torch.zeros(1, 30),
        "action_q99": torch.ones(1, 30),
        "actions_mask": dataset_mask,
    }

    actions, mask = rebuild_batch_actions(batch, _config(), chunk_size_frames=2)

    assert torch.all(mask[:, :28])
    assert not mask[:, 28:].any()
    assert not actions[:, 28:].any()


def test_state_anchored_chunk_relative_actions_map_to_30d():
    config = _config()
    raw_action = torch.tensor(
        [
            [[0.0] * 23],
            [[*_pose(0.2), 0.2, *_pose(10.2), 0.3, *([0.0] * 7)]],
            [[*_pose(0.4), 0.4, *_pose(10.4), 0.5, *([0.0] * 7)]],
            [[*_pose(0.8), 0.6, *_pose(10.8), 0.7, *([0.0] * 7)]],
        ],
        dtype=torch.float32,
    )
    raw_state = torch.tensor(
        [
            [[0.0] * 23],
            [[*_pose(0.0), 0.0, *_pose(10.0), 0.0, *([0.0] * 7)]],
            [[*_pose(0.2), 0.0, *_pose(10.2), 0.0, *([0.0] * 7)]],
            [[*_pose(0.6), 0.0, *_pose(10.6), 0.0, *([0.0] * 7)]],
        ],
        dtype=torch.float32,
    )
    step_mask = torch.tensor([[False], [True], [True], [True]])
    q01 = torch.full((30,), -1.0)
    q99 = torch.full((30,), 1.0)

    actions, mask = build_model_actions_from_raw(
        raw_action,
        raw_state,
        step_mask,
        config,
        q01=q01,
        q99=q99,
        chunk_size_frames=2,
    )

    assert actions.shape == (30, 4, 1, 1)
    assert torch.count_nonzero(mask[:, 0]) == 0
    # Frames 1-2 share state(frame 1) as their anchor; frame 3 starts a new chunk.
    torch.testing.assert_close(actions[0, 1:, 0, 0], torch.tensor([0.2, 0.4, 0.2]))
    torch.testing.assert_close(actions[7, 1:, 0, 0], torch.tensor([0.2, 0.4, 0.2]))
    torch.testing.assert_close(actions[6, 1:, 0, 0], torch.ones(3))
    torch.testing.assert_close(actions[13, 1:, 0, 0], torch.ones(3))
    torch.testing.assert_close(actions[28, 1:, 0, 0], torch.tensor([0.2, 0.4, 0.6]))
    torch.testing.assert_close(actions[29, 1:, 0, 0], torch.tensor([0.3, 0.5, 0.7]))
    assert torch.count_nonzero(mask[14:28]) == 0


def test_local_frame_rotation_and_quaternion_hemisphere():
    root_half = 2**-0.5
    config = _config()
    # Anchor is +90 degrees around Z. A world +X displacement is local -Y.
    anchor_pose = _pose(quat=[root_half, 0.0, 0.0, root_half])
    target_pose = _pose(1.0, quat=[-root_half, 0.0, 0.0, -root_half])
    action_row = [*target_pose, 0.0, *target_pose, 0.0, *([0.0] * 7)]
    state_row = [*anchor_pose, 0.0, *anchor_pose, 0.0, *([0.0] * 7)]
    raw_action = torch.tensor([[[0.0] * 23], [action_row]])
    raw_state = torch.tensor([[[0.0] * 23], [state_row]])
    mask = torch.tensor([[False], [True]])

    actions, _ = build_model_actions_from_raw(
        raw_action,
        raw_state,
        mask,
        config,
        q01=torch.full((30,), -1.0),
        q99=torch.full((30,), 1.0),
        chunk_size_frames=1,
    )

    torch.testing.assert_close(actions[0:3, 1, 0, 0], torch.tensor([0.0, -1.0, 0.0]), atol=1e-5, rtol=0)
    torch.testing.assert_close(actions[3:7, 1, 0, 0], torch.tensor([0.0, 0.0, 0.0, 1.0]), atol=1e-5, rtol=0)


def test_normalizer_metadata_is_validated(tmp_path):
    config = _config()
    stats = {
        "q01": [0.0] * 30,
        "q99": [1.0] * 30,
        "_meta": {
            "version": config.action_norm_stats_version,
            "action_chunk_size_max": 4,
            "chunk_grouping_start_from_one": True,
            "relative_pose_frame": "local_frame",
            "quaternion_order": "wxyz",
            "action_quaternion_order": "xyzw",
            "relative_pose_groups": config.relative_pose_groups
            + [{"pose_slice": [16, 23]}],
            "used_action_channel_ids": config.used_action_channel_ids
            + list(range(14, 21)),
        },
    }
    path = tmp_path / "lingbot_action_norm_stats.json"
    path.write_text(json.dumps(stats), encoding="utf-8")
    loaded = load_action_norm_stats(tmp_path, config)
    assert loaded["q99"] == [1.0] * 30

    stats["_meta"]["quaternion_order"] = "xyzw"
    path.write_text(json.dumps(stats), encoding="utf-8")
    try:
        load_action_norm_stats(tmp_path, config)
    except ValueError as exc:
        assert "quaternion_order" in str(exc)
    else:
        raise AssertionError("incompatible normalizer metadata must fail")
