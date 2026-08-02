#!/usr/bin/env python3
"""Reconstruct versioned run contracts for a completed NMX LingBot-VA run."""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from wan_va.configs import VA_CONFIGS
from wan_va.run_contract import persist_run_contracts, write_checkpoint_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--dataset-manifest", required=True)
    parser.add_argument("--norm-stats", required=True)
    parser.add_argument("--prompt-embedding", required=True)
    parser.add_argument("--empty-embedding", required=True)
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).expanduser().resolve()
    run_manifest_path = run_dir / "run_manifest.json"
    run_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
    config_name = str(run_manifest["config_name"])
    if config_name not in VA_CONFIGS:
        raise ValueError(f"Unknown recorded config_name: {config_name}")
    for name in ("resolved_train_config.json", "deployment_manifest.json"):
        if (run_dir / name).exists() and not args.force:
            raise FileExistsError(f"Refusing to replace {run_dir / name}; pass --force")

    config = copy.deepcopy(VA_CONFIGS[config_name])
    config.save_root = str(run_dir)
    config.wan22_pretrained_model_name_or_path = str(Path(args.base_model).resolve())
    config.dataset_manifest_path = str(Path(args.dataset_manifest).resolve())
    config.action_norm_stats_path = str(Path(args.norm_stats).resolve())
    config.text_emb_override_path = str(Path(args.prompt_embedding).resolve())
    config.empty_emb_path = str(Path(args.empty_embedding).resolve())
    config.rank = 0
    config.local_rank = 0
    distributed = run_manifest.get("distributed", {})
    config.world_size = int(distributed.get("world_size", 1))
    config.distributed_backend = str(distributed.get("backend", "deepspeed"))
    training = run_manifest.get("training", {})
    batching = run_manifest.get("batching", {})
    optimizer = run_manifest.get("optimizer", {})
    config.num_steps = int(training.get("max_optimizer_steps", config.num_steps))
    config.save_interval = int(training.get("save_interval", config.save_interval))
    config.max_latent_frames = int(training.get("max_latent_frames", config.max_latent_frames))
    config.cfg_prob = float(training.get("cfg_dropout_probability", config.cfg_prob))
    config.batch_size = int(batching.get("micro_batch_size", config.batch_size))
    config.gradient_accumulation_steps = int(
        batching.get("gradient_accumulation_steps", config.gradient_accumulation_steps)
    )
    config.learning_rate = float(optimizer.get("learning_rate", config.learning_rate))
    config.weight_decay = float(optimizer.get("weight_decay", config.weight_decay))

    contract = persist_run_contracts(
        config,
        config_name=config_name,
        launch_args={"reconstructed_from": str(run_dir / "launch.sh")},
        dataset_manifest_path=args.dataset_manifest,
        repo_root=Path(__file__).resolve().parents[1],
        provenance="reconstructed",
        git_commit=str(run_manifest["git_commit"]),
    )
    for checkpoint in sorted((run_dir / "checkpoints").glob("checkpoint_step_*")):
        if not (checkpoint / "transformer" / "config.json").is_file():
            continue
        try:
            step = int(checkpoint.name.rsplit("_", 1)[1])
        except ValueError:
            continue
        write_checkpoint_manifest(checkpoint, step=step, run_contract=contract)

    print(json.dumps(contract, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
