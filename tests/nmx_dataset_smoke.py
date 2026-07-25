"""Read-only smoke test for an NMX LingBot-VA dataset root."""

import argparse
from copy import deepcopy

import torch
from torch.utils.data import DataLoader

from wan_va.configs import VA_CONFIGS
from wan_va.dataset.lerobot_latent_dataset import MultiLatentLeRobotDataset
from wan_va.dataset.nmx_action_adapter import rebuild_batch_actions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        default="/mnt/workspace/shenyibo/datasets/chip_0709_1952episodes",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--all-recorded", action="store_true")
    args = parser.parse_args()

    config = deepcopy(VA_CONFIGS["nmx_chip_train"])
    if not args.all_recorded:
        config.dataset_path = args.dataset_root
    config.cfg_prob = 0.0
    dataset = MultiLatentLeRobotDataset(config=config, num_init_worker=1)
    sample = dataset[args.sample_index]
    batch = next(iter(DataLoader(dataset, batch_size=1, num_workers=0)))

    assert sample["latents"].shape[1] <= config.max_latent_frames
    assert sample["raw_actions"].shape == sample["raw_states"].shape
    assert sample["raw_actions"].shape[-1] == 23
    assert not sample["raw_actions_step_mask"][0].any()
    for chunk_size in (1, 4):
        actions, masks = rebuild_batch_actions(batch, config, chunk_size)
        assert actions.shape == masks.shape
        assert actions.shape[1] == config.action_dim
        assert not masks[:, :, 0].any()
        assert torch.isfinite(actions).all()

    print(
        {
            "dataset_samples": len(dataset),
            "latents": tuple(sample["latents"].shape),
            "raw_actions": tuple(sample["raw_actions"].shape),
            "actions": tuple(sample["actions"].shape),
            "valid_action_values": int(sample["actions_mask"].sum()),
        }
    )


if __name__ == "__main__":
    main()
