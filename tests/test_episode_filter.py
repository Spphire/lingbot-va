from types import SimpleNamespace

import pytest

from wan_va.configs.va_nmx_chip_train_cfg import (
    va_nmx_chip_episode109_overfit_cfg,
    va_nmx_chip_train_cfg,
)
from wan_va.dataset.lerobot_latent_dataset import LatentLeRobotDataset


def _dataset(episode_index_filter):
    dataset = LatentLeRobotDataset.__new__(LatentLeRobotDataset)
    dataset.config = SimpleNamespace(episode_index_filter=episode_index_filter)
    dataset.nmx_action_contract = False
    dataset.meta = SimpleNamespace(
        episodes={
            108: {
                "episode_index": 108,
                "tasks": ["task"],
                "action_config": [{"start_frame": 0, "end_frame": 64}],
            },
            109: {
                "episode_index": 109,
                "tasks": ["task"],
                "action_config": [{"start_frame": 0, "end_frame": 64}],
            },
        }
    )
    dataset._check_meta = lambda *_args: True
    return dataset


def test_parse_meta_keeps_only_configured_episode():
    dataset = _dataset([109])

    dataset.parse_meta()

    assert [meta["episode_index"] for meta in dataset.new_metas] == [109]


def test_parse_meta_rejects_episode_without_usable_latent_segment():
    dataset = _dataset([110])

    with pytest.raises(ValueError, match="episode indices \\[110\\]"):
        dataset.parse_meta()


def test_overfit_config_changes_only_experiment_controls():
    assert va_nmx_chip_episode109_overfit_cfg.dataset_path == [
        "/mnt/workspace/shenyibo/datasets/chip_0711_199episodes"
    ]
    assert va_nmx_chip_episode109_overfit_cfg.episode_index_filter == [109]
    assert va_nmx_chip_episode109_overfit_cfg.max_latent_frames is None
    assert va_nmx_chip_episode109_overfit_cfg.cfg_prob == 0.0
    assert va_nmx_chip_episode109_overfit_cfg.num_steps == 1000
    assert va_nmx_chip_episode109_overfit_cfg.save_interval == 100

    for field in (
        "action_contract",
        "action_history_condition_dropout_prob",
        "chunk_grouping_start_from_one",
        "expected_latent_view_shapes",
        "visual_contract",
    ):
        assert va_nmx_chip_episode109_overfit_cfg[field] == va_nmx_chip_train_cfg[field]
