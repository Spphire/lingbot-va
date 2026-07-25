"""Validate NMX episode-level action masks on representative real samples."""

from copy import deepcopy

from torch.utils.data._utils.collate import default_collate

from wan_va.configs import VA_CONFIGS
from wan_va.dataset.lerobot_latent_dataset import MultiLatentLeRobotDataset
from wan_va.dataset.nmx_action_adapter import rebuild_batch_actions


DATASET_ROOT = "/mnt/workspace/shenyibo/datasets/chip_0709_1952episodes"
CASES = {
    0: (True, True),
    1: (True, False),
    9: (False, True),
    443: (False, False),
}


def main():
    config = deepcopy(VA_CONFIGS["nmx_chip_train"])
    config.dataset_path = DATASET_ROOT
    config.cfg_prob = 0.0
    dataset = MultiLatentLeRobotDataset(config=config, num_init_worker=1)
    results = {}

    for sample_index, (action_valid, width_valid) in CASES.items():
        sample = dataset[sample_index]
        batch = default_collate([sample])
        valid_steps = int(sample["raw_actions_step_mask"].sum())
        if not action_valid:
            assert valid_steps == 0

        for chunk_size in range(1, 5):
            _, mask = rebuild_batch_actions(batch, config, chunk_size)
            pose_values = int(mask[:, :14].sum())
            gripper_values = int(mask[:, [28, 29]].sum())
            if action_valid:
                assert pose_values == valid_steps * 14
                assert gripper_values == (valid_steps * 2 if width_valid else 0)
            else:
                assert int(mask.sum()) == 0

        results[sample_index] = {
            "action_valid": action_valid,
            "width_valid": width_valid,
            "valid_steps": valid_steps,
            "action_mask_values": int(sample["actions_mask"].sum()),
        }

    print(results)


if __name__ == "__main__":
    main()
