"""Verify that the real-data A/B configs differ only in visual composition."""

import argparse
from copy import deepcopy

import numpy as np
import torch

from wan_va.configs import VA_CONFIGS
from wan_va.dataset.lerobot_latent_dataset import (
    PER_VIEW_ZERO_PAD_VISUAL_CONTRACT,
    UPSTREAM_SINGLE_CANVAS_VISUAL_CONTRACT,
    MultiLatentLeRobotDataset,
)


def _read_sample(dataset, visual_contract, sample_index, seed):
    for child_dataset in dataset._datasets:
        child_dataset.config.visual_contract = visual_contract
    np.random.seed(seed)
    torch.manual_seed(seed)
    return dataset[sample_index]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-root",
        default="/mnt/workspace/shenyibo/datasets/chip_0709_1952episodes",
    )
    parser.add_argument("--sample-index", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = deepcopy(VA_CONFIGS["nmx_chip_train"])
    config.dataset_path = args.dataset_root
    config.cfg_prob = 0.0
    dataset = MultiLatentLeRobotDataset(config=config, num_init_worker=1)

    baseline = _read_sample(
        dataset,
        UPSTREAM_SINGLE_CANVAS_VISUAL_CONTRACT,
        args.sample_index,
        args.seed,
    )
    padded = _read_sample(
        dataset,
        PER_VIEW_ZERO_PAD_VISUAL_CONTRACT,
        args.sample_index,
        args.seed,
    )

    assert baseline["latents"].shape[-2:] == (20, 30)
    assert padded["latents"].shape[-2:] == (20, 32)
    torch.testing.assert_close(
        padded["latents"][..., :15], baseline["latents"][..., :15]
    )
    assert not torch.count_nonzero(padded["latents"][..., 15])
    torch.testing.assert_close(
        padded["latents"][..., 16:31], baseline["latents"][..., 15:30]
    )
    assert not torch.count_nonzero(padded["latents"][..., 31])

    parity_keys = (
        "text_emb",
        "actions",
        "actions_mask",
        "raw_actions",
        "raw_actions_step_mask",
        "raw_states",
        "action_q01",
        "action_q99",
    )
    for key in parity_keys:
        torch.testing.assert_close(padded[key], baseline[key])

    print(
        {
            "sample_index": args.sample_index,
            "baseline_latents": tuple(baseline["latents"].shape),
            "baseline_tokens_per_frame": 150,
            "padded_latents": tuple(padded["latents"].shape),
            "padded_tokens_per_frame": 160,
            "non_visual_parity_keys": parity_keys,
        }
    )


if __name__ == "__main__":
    main()
