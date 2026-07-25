"""Compare this branch's adapter with nmx_vla on one real dataset sample."""

from copy import deepcopy

import torch

from nmx_vla.dataloader.lingbot_va_datasets.lerobot_action_processing import (
    build_model_actions_from_raw as build_nmx_actions,
)
from wan_va.configs import VA_CONFIGS
from wan_va.dataset.lerobot_latent_dataset import MultiLatentLeRobotDataset
from wan_va.dataset.nmx_action_adapter import build_model_actions_from_raw


def main():
    config = deepcopy(VA_CONFIGS["nmx_chip_train"])
    config.dataset_path = "/mnt/workspace/shenyibo/datasets/chip_0709_1952episodes"
    config.cfg_prob = 0.0
    dataset = MultiLatentLeRobotDataset(config=config, num_init_worker=1)
    sample = dataset[0]

    for chunk_size in range(1, 5):
        ours = build_model_actions_from_raw(
            sample["raw_actions"],
            sample["raw_states"],
            sample["raw_actions_step_mask"],
            config,
            q01=sample["action_q01"],
            q99=sample["action_q99"],
            chunk_size_frames=chunk_size,
        )
        reference = build_nmx_actions(
            sample["raw_actions"],
            sample["raw_actions_step_mask"],
            config,
            q01=sample["action_q01"],
            q99=sample["action_q99"],
            chunk_size_frames=chunk_size,
            chunk_grouping_start_from_one=True,
            raw_state=sample["raw_states"],
        )
        torch.testing.assert_close(ours[0], reference[0], atol=1e-6, rtol=1e-6)
        assert torch.equal(ours[1], reference[1])
        print(
            f"K={chunk_size}: max_abs="
            f"{float((ours[0] - reference[0]).abs().max()):.3e}"
        )


if __name__ == "__main__":
    main()
