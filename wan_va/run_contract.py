"""Persist the resolved training and deployment contracts for a run."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


RUN_CONFIG_SCHEMA_VERSION = 1
DEPLOYMENT_MANIFEST_SCHEMA_VERSION = 1
CHECKPOINT_MANIFEST_SCHEMA_VERSION = 1


def sha256_file(path: str | os.PathLike[str]) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    text = str(value)
    if text.startswith("torch."):
        return text
    raise TypeError(f"Training config value is not JSON serializable: {value!r}")


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, ensure_ascii=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _load_json_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Unable to read {label} {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must contain a JSON object: {path}")
    return dict(payload)


def _git_commit(repo_root: Path) -> str:
    configured = os.getenv("LINGBOT_VA_GIT_COMMIT")
    if configured:
        return configured
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise ValueError("Unable to resolve the LingBot-VA Git commit") from exc


def _required(config: Mapping[str, Any], key: str) -> Any:
    if key not in config:
        raise ValueError(f"Resolved training config is missing {key!r}")
    return config[key]


def _normalizer_contract(config: Mapping[str, Any]) -> dict[str, Any]:
    configured_path = config.get("action_norm_stats_path")
    if configured_path:
        candidates = [Path(str(configured_path))]
    else:
        filename = str(config.get("action_norm_stats_filename") or "")
        candidates = [Path(str(root)) / filename for root in config.get("dataset_path", [])]
    candidates = [path.resolve() for path in candidates if path.is_file()]
    if not candidates:
        raise ValueError("No action normalization statistics file is available")
    hashes = {sha256_file(path) for path in candidates}
    if len(hashes) != 1:
        raise ValueError("Dataset roots do not share identical action normalization statistics")
    return {
        "filename": candidates[0].name,
        "source_path": str(candidates[0]),
        "sha256": hashes.pop(),
    }


def _asset_contract(path_value: Any, *, filename: str | None = None) -> dict[str, Any]:
    path = Path(str(path_value)).expanduser().resolve()
    if not path.is_file():
        raise ValueError(f"Deployment asset is missing: {path}")
    return {
        "filename": filename or path.name,
        "source_path": str(path),
        "sha256": sha256_file(path),
    }


def _visual_layout(config: Mapping[str, Any]) -> tuple[list[int], int]:
    patch = [int(value) for value in _required(config, "patch_size")]
    views = [list(map(int, shape)) for shape in _required(config, "expected_latent_view_shapes")]
    contract = str(_required(config, "visual_contract"))
    height = views[0][0]
    if any(shape[0] != height for shape in views):
        raise ValueError("All latent views must have the same height")
    if contract == "upstream_single_canvas_v1":
        width = sum(shape[1] for shape in views)
    elif contract == "per_view_zero_pad_then_concat_v1":
        patch_width = patch[2]
        width = sum(((shape[1] + patch_width - 1) // patch_width) * patch_width for shape in views)
    else:
        raise ValueError(f"Unsupported visual contract: {contract}")
    tokens = (int(_required(config, "frame_chunk_size")) * height * width) // (
        patch[0] * patch[1] * patch[2]
    )
    return [height, width], tokens // int(_required(config, "frame_chunk_size"))


def persist_run_contracts(
    config: Mapping[str, Any],
    *,
    config_name: str,
    launch_args: Mapping[str, Any],
    dataset_manifest_path: str | os.PathLike[str],
    repo_root: str | os.PathLike[str],
    provenance: str = "native",
    git_commit: str | None = None,
) -> dict[str, Any]:
    """Write self-contained run metadata after all runtime overrides are applied."""

    resolved = _json_safe(dict(config))
    save_root = Path(str(_required(resolved, "save_root"))).expanduser().resolve()
    dataset_path = Path(dataset_manifest_path).expanduser().resolve()
    dataset_manifest = _load_json_mapping(dataset_path, "dataset manifest")
    dataset_manifest_sha256 = sha256_file(dataset_path)
    git_commit = git_commit or _git_commit(Path(repo_root).resolve())

    if str(_required(resolved, "action_norm_method")) != "quantiles":
        raise ValueError("NMX LingBot-VA deployment requires action_norm_method='quantiles'")
    latent_canvas_hw, visual_tokens = _visual_layout(resolved)
    normalizer = _normalizer_contract(resolved)
    expected_normalizer = (
        dataset_manifest.get("action_contract", {}).get("normalizer_sha256")
        if isinstance(dataset_manifest.get("action_contract"), Mapping)
        else None
    )
    if expected_normalizer and normalizer["sha256"] != expected_normalizer:
        raise ValueError(
            "Action normalizer does not match dataset_manifest.json: "
            f"{normalizer['sha256']} != {expected_normalizer}"
        )

    prompt = _asset_contract(_required(resolved, "text_emb_override_path"), filename="prompt_text_emb.pt")
    prompt["tensor_sha256"] = str(_required(resolved, "text_emb_override_sha256"))
    empty = _asset_contract(_required(resolved, "empty_emb_path"), filename="empty_emb.pt")

    resolved_payload = {
        "schema_version": RUN_CONFIG_SCHEMA_VERSION,
        "provenance": provenance,
        "config_name": config_name,
        "git_commit": git_commit,
        "launch_args": _json_safe(dict(launch_args)),
        "config": resolved,
    }
    resolved_path = save_root / "resolved_train_config.json"
    _write_json_atomic(resolved_path, resolved_payload)
    resolved_sha256 = sha256_file(resolved_path)

    policy_contract_keys = (
        "action_condition_mode",
        "model_structure",
        "mot_config",
        "visual_contract",
        "raw_image_hw",
        "reshape_mode",
        "center_crop_before_resize",
        "crop_before_resize",
        "crop_resize_size",
        "pad_after_resize",
        "camera_rotation_degrees",
        "obs_cam_keys",
        "action_contract",
        "relative_pose_frame",
        "quaternion_order",
        "action_quaternion_order",
        "used_action_channel_ids",
        "action_dim",
        "control_action_dim",
        "control_chunk_length",
        "action_norm_method",
        "action_history_condition_dropout_prob",
        "action_history_condition_dropout_mode",
        "action_history_adds_parameters",
        "source_action_per_frame",
        "source_video_frame_stride",
        "video_frames_per_latent",
        "action_per_frame",
        "sampled_video_fps",
        "control_fps",
        "first_prediction_skip_action_latents",
        "frame_chunk_size",
    )
    policy_contract = {key: _required(resolved, key) for key in policy_contract_keys}
    policy_contract.update(
        {
            "upstream_config_name": config_name,
            "upstream_git_commit": git_commit,
            "training_dataset_manifest_sha256": dataset_manifest_sha256,
            "training_base_model_path": str(
                Path(str(_required(resolved, "wan22_pretrained_model_name_or_path"))).resolve()
            ),
            "latent_canvas_hw": latent_canvas_hw,
            "visual_tokens_per_frame": visual_tokens,
            "zero_action_history": float(
                _required(resolved, "action_history_condition_dropout_prob")
            )
            == 1.0,
            "embedding_dtype": str(_required(resolved, "text_emb_override_dtype")).removeprefix("torch."),
            "task": str(_required(resolved, "task_prompt_override")),
            "norm_stats_sha256": normalizer["sha256"],
            "prompt_embedding_tensor_sha256": prompt["tensor_sha256"],
            "empty_embedding_sha256": empty["sha256"],
        }
    )
    deployment_payload = {
        "schema_version": DEPLOYMENT_MANIFEST_SCHEMA_VERSION,
        "provenance": provenance,
        "resolved_train_config": {
            "path": resolved_path.name,
            "sha256": resolved_sha256,
        },
        "policy_contract": policy_contract,
        "assets": {
            "norm_stats": normalizer,
            "prompt_embedding": prompt,
            "empty_embedding": empty,
        },
    }
    deployment_path = save_root / "deployment_manifest.json"
    _write_json_atomic(deployment_path, deployment_payload)
    deployment_sha256 = sha256_file(deployment_path)

    run_manifest_path = save_root / "run_manifest.json"
    if run_manifest_path.exists():
        run_manifest = _load_json_mapping(run_manifest_path, "run manifest")
        expected_existing = {
            "config_name": config_name,
            "git_commit": git_commit,
            "visual_contract": policy_contract["visual_contract"],
            "dataset_manifest_sha256": dataset_manifest_sha256,
        }
        for key, expected in expected_existing.items():
            actual = run_manifest.get(key)
            if actual is not None and actual != expected:
                raise ValueError(
                    f"Existing run_manifest.json {key} mismatch: "
                    f"expected={expected!r}, actual={actual!r}"
                )
    else:
        run_manifest = {}
    run_manifest.update(
        {
            "schema_version": 2,
            "run_id": str(os.getenv("RUN_ID") or save_root.name),
            "provenance": provenance,
            "git_commit": git_commit,
            "config_name": config_name,
            "visual_contract": policy_contract["visual_contract"],
            "action_condition_mode": policy_contract["action_condition_mode"],
            "model_structure": policy_contract["model_structure"],
            "mot_config": policy_contract["mot_config"],
            "latent_canvas_hwc": [*latent_canvas_hw, int(_required(resolved, "expected_latent_channels"))],
            "visual_tokens_per_frame": visual_tokens,
            "dataset_manifest_sha256": dataset_manifest_sha256,
            "model_path": policy_contract["training_base_model_path"],
            "action_history_condition": {
                "dropout_probability": policy_contract[
                    "action_history_condition_dropout_prob"
                ],
                "mode": policy_contract["action_history_condition_dropout_mode"],
                "adds_parameters": policy_contract[
                    "action_history_adds_parameters"
                ],
            },
            "resolved_train_config": {
                "path": resolved_path.name,
                "sha256": resolved_sha256,
            },
            "deployment_manifest": {
                "path": deployment_path.name,
                "sha256": deployment_sha256,
            },
        }
    )
    _write_json_atomic(run_manifest_path, run_manifest)
    run_manifest_sha256 = sha256_file(run_manifest_path)
    return {
        "resolved_train_config_path": str(resolved_path),
        "resolved_train_config_sha256": resolved_sha256,
        "deployment_manifest_path": str(deployment_path),
        "deployment_manifest_sha256": deployment_sha256,
        "run_manifest_path": str(run_manifest_path),
        "run_manifest_sha256": run_manifest_sha256,
    }


def write_checkpoint_manifest(
    checkpoint_dir: str | os.PathLike[str],
    *,
    step: int,
    run_contract: Mapping[str, Any],
) -> Path:
    checkpoint = Path(checkpoint_dir)
    transformer_config = checkpoint / "transformer" / "config.json"
    if not transformer_config.is_file():
        raise ValueError(f"Transformer config is missing: {transformer_config}")
    payload = {
        "schema_version": CHECKPOINT_MANIFEST_SCHEMA_VERSION,
        "step": int(step),
        "resolved_train_config": {
            "path": "../../resolved_train_config.json",
            "sha256": str(run_contract["resolved_train_config_sha256"]),
        },
        "deployment_manifest": {
            "path": "../../deployment_manifest.json",
            "sha256": str(run_contract["deployment_manifest_sha256"]),
        },
        "run_manifest": {
            "path": "../../run_manifest.json",
            "sha256": str(run_contract["run_manifest_sha256"]),
        },
        "transformer_config_sha256": sha256_file(transformer_config),
    }
    destination = checkpoint / "checkpoint_manifest.json"
    _write_json_atomic(destination, payload)
    return destination


__all__ = [
    "persist_run_contracts",
    "sha256_file",
    "write_checkpoint_manifest",
]
