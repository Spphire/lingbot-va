import torch

from wan_va.modules.model import FlexAttnFunc, WanTransformer3DModel


def _mask(action_condition_mode):
    # Token order: noisy video, clean video, noisy action, clean action.
    return FlexAttnFunc._get_mask_mod(
        seq_ids=torch.zeros(4, dtype=torch.long),
        frame_ids=torch.tensor([0, 0, 1, 1]),
        noise_ids=torch.tensor([0, 1, 0, 1]),
        chunk_ids=torch.zeros(4, dtype=torch.long),
        modality_ids=torch.tensor([0, 0, 1, 1]),
        window_size=3,
        action_condition_mode=action_condition_mode,
    )


def _allows(mask, query_index, key_index):
    zero = torch.tensor(0, dtype=torch.long)
    return bool(mask(zero, zero, torch.tensor(query_index), torch.tensor(key_index)))


def test_fastwam_blocks_current_clean_video_from_noisy_action():
    inverse_dynamics = _mask("inverse_dynamics")
    fastwam = _mask("fastwam")

    assert _allows(inverse_dynamics, 2, 1)
    assert not _allows(fastwam, 2, 1)


def test_fastwam_preserves_other_joint_attention_edges():
    inverse_dynamics = _mask("inverse_dynamics")
    fastwam = _mask("fastwam")

    for query_index, key_index in ((0, 0), (1, 1), (2, 2), (3, 3), (3, 1)):
        assert _allows(fastwam, query_index, key_index) == _allows(
            inverse_dynamics, query_index, key_index
        )


def test_transformer_rejects_unknown_action_condition_mode_before_allocation():
    try:
        WanTransformer3DModel(action_condition_mode="unknown")
    except ValueError as exc:
        assert "action_condition_mode" in str(exc)
    else:
        raise AssertionError("unknown action conditioning must fail closed")
