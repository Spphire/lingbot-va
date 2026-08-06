import json
from pathlib import Path

from wan_va.run_contract import persist_run_contracts, sha256_file, write_checkpoint_manifest


def _config(tmp_path: Path) -> dict:
    dataset = tmp_path / "dataset"
    dataset.mkdir()
    norm = dataset / "lingbot_action_norm_stats.json"
    norm.write_text(json.dumps({"q01": [0.0] * 30, "q99": [1.0] * 30}), encoding="utf-8")
    prompt = tmp_path / "prompt.pt"
    prompt.write_bytes(b"prompt")
    empty = tmp_path / "empty.pt"
    empty.write_bytes(b"empty")
    return {
        "save_root": str(tmp_path / "run"),
        "dataset_path": [str(dataset)],
        "wan22_pretrained_model_name_or_path": str(tmp_path / "base"),
        "param_dtype": "torch.bfloat16",
        "patch_size": [1, 2, 2],
        "action_condition_mode": "fastwam",
        "model_structure": "mot",
        "mot_config": {"action_hidden_dim": 768, "action_mlp_hidden_dim": 256},
        "visual_contract": "upstream_single_canvas_v1",
        "expected_latent_view_shapes": [[20, 15], [20, 15]],
        "expected_latent_channels": 48,
        "raw_image_hw": [360, 640],
        "reshape_mode": "crop_resize",
        "center_crop_before_resize": False,
        "crop_before_resize": [185, 185, 0, 0],
        "crop_resize_size": [320, 240],
        "pad_after_resize": None,
        "camera_rotation_degrees": [0, 0],
        "obs_cam_keys": ["left", "right"],
        "action_contract": "nmx_chunk_relative_v10",
        "relative_pose_frame": "local_frame",
        "quaternion_order": "wxyz",
        "action_quaternion_order": "xyzw",
        "used_action_channel_ids": list(range(16)),
        "action_dim": 30,
        "control_action_dim": 16,
        "control_chunk_length": 48,
        "action_norm_method": "quantiles",
        "action_norm_stats_filename": norm.name,
        "action_history_condition_dropout_prob": 1.0,
        "action_history_condition_dropout_mode": "zero_normalized_clean_condition",
        "action_history_adds_parameters": False,
        "source_action_per_frame": 1,
        "source_video_frame_stride": 3,
        "video_frames_per_latent": 4,
        "action_per_frame": 12,
        "sampled_video_fps": 10.0,
        "control_fps": 30.0,
        "first_prediction_skip_action_latents": 1,
        "frame_chunk_size": 4,
        "text_emb_override_path": str(prompt),
        "text_emb_override_sha256": "tensor-sha",
        "text_emb_override_dtype": "torch.bfloat16",
        "empty_emb_path": str(empty),
        "task_prompt_override": "task",
    }


def test_run_and_checkpoint_contracts_are_self_describing(tmp_path, monkeypatch):
    config = _config(tmp_path)
    norm_sha = sha256_file(Path(config["dataset_path"][0]) / config["action_norm_stats_filename"])
    dataset_manifest = tmp_path / "dataset_manifest.json"
    dataset_manifest.write_text(
        json.dumps({"action_contract": {"normalizer_sha256": norm_sha}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("LINGBOT_VA_GIT_COMMIT", "training-commit")

    contract = persist_run_contracts(
        config,
        config_name="nmx_chip_train",
        launch_args={"num_steps": 20},
        dataset_manifest_path=dataset_manifest,
        repo_root=tmp_path,
    )

    run = Path(config["save_root"])
    resolved = json.loads((run / "resolved_train_config.json").read_text())
    deployment = json.loads((run / "deployment_manifest.json").read_text())
    assert resolved["config"]["action_norm_method"] == "quantiles"
    assert deployment["policy_contract"]["action_norm_method"] == "quantiles"
    assert deployment["policy_contract"]["action_condition_mode"] == "fastwam"
    assert deployment["policy_contract"]["model_structure"] == "mot"
    assert deployment["policy_contract"]["mot_config"]["action_hidden_dim"] == 768
    assert deployment["policy_contract"]["latent_canvas_hw"] == [20, 30]
    assert deployment["policy_contract"]["visual_tokens_per_frame"] == 150
    assert deployment["resolved_train_config"]["sha256"] == sha256_file(
        run / "resolved_train_config.json"
    )
    run_manifest = json.loads((run / "run_manifest.json").read_text())
    assert run_manifest["provenance"] == "native"
    assert run_manifest["action_condition_mode"] == "fastwam"
    assert run_manifest["model_structure"] == "mot"
    assert run_manifest["deployment_manifest"]["sha256"] == contract[
        "deployment_manifest_sha256"
    ]

    checkpoint = run / "checkpoints" / "checkpoint_step_20"
    transformer = checkpoint / "transformer"
    transformer.mkdir(parents=True)
    (transformer / "config.json").write_text("{}\n", encoding="utf-8")
    manifest_path = write_checkpoint_manifest(checkpoint, step=20, run_contract=contract)
    manifest = json.loads(manifest_path.read_text())
    assert manifest["step"] == 20
    assert manifest["deployment_manifest"]["sha256"] == contract[
        "deployment_manifest_sha256"
    ]
    assert manifest["run_manifest"]["sha256"] == contract["run_manifest_sha256"]
