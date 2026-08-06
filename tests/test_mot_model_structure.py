import torch

from wan_va.modules.model import WanTransformer3DModel
from wan_va.modules.mot_support.model import WanTransformer3DModel as MotSharedModel
from wan_va.modules.mot_support.model_mot import WanTransformer3DMoTModel
from wan_va.modules.utils import load_transformer


def _tiny_kwargs():
    return {
        "patch_size": (1, 2, 2),
        "num_attention_heads": 2,
        "attention_head_dim": 8,
        "in_channels": 4,
        "out_channels": 4,
        "action_dim": 30,
        "text_dim": 16,
        "freq_dim": 8,
        "ffn_dim": 32,
        "num_layers": 1,
        "cross_attn_norm": True,
        "eps": 1e-6,
        "rope_max_seq_len": 64,
        "attn_mode": "torch",
        "action_condition_mode": "fastwam",
    }


def _tiny_input():
    batch, frames, height, width = 1, 4, 4, 4
    latent = torch.randn(batch, 4, frames, height, width)
    action = torch.randn(batch, 30, frames, 1, 1)
    return {
        "latent_dict": {
            "noisy_latents": latent,
            "latent": latent.clone(),
            "timesteps": torch.ones(batch, frames),
            "cond_timesteps": torch.zeros(batch, frames),
            "grid_id": torch.zeros(
                batch, 4, frames * (height // 2) * (width // 2), dtype=torch.long
            ),
            "text_emb": torch.randn(batch, 4, 16),
        },
        "action_dict": {
            "noisy_latents": action,
            "latent": action.clone(),
            "timesteps": torch.ones(batch, frames),
            "cond_timesteps": torch.zeros(batch, frames),
            "grid_id": torch.zeros(batch, 4, frames, dtype=torch.long),
        },
        "chunk_size": 1,
        "window_size": 3,
        "chunk_grouping_start_from_one": True,
    }


def test_mot_forward_backward_preserves_30d_action_contract():
    model = WanTransformer3DMoTModel(
        **_tiny_kwargs(),
        model_structure="mot",
        action_hidden_dim=8,
        action_ffn_dim=16,
        action_mlp_hidden_dim=4,
    )
    video, action = model(_tiny_input(), train_mode=True)
    assert video.shape == (1, 64, 4)
    assert action.shape == (1, 4, 30)
    (video.square().mean() + action.square().mean()).backward()
    assert model.action_decoder[-1].weight.grad is not None


def test_shared_checkpoint_converts_and_mot_checkpoint_auto_loads(tmp_path):
    shared_dir = tmp_path / "shared"
    mot_dir = tmp_path / "mot"
    MotSharedModel(**_tiny_kwargs()).save_pretrained(shared_dir)

    model = load_transformer(
        shared_dir,
        torch.float32,
        "cpu",
        model_structure="mot",
        mot_config={
            "action_hidden_dim": 8,
            "action_ffn_dim": 16,
            "action_mlp_hidden_dim": 4,
        },
        attn_mode="torch",
        action_condition_mode="fastwam",
    )
    assert isinstance(model, WanTransformer3DMoTModel)
    assert model.config.action_dim == 30
    model.save_pretrained(mot_dir)

    reloaded = load_transformer(
        mot_dir,
        torch.float32,
        "cpu",
        attn_mode="torch",
        action_condition_mode="fastwam",
    )
    assert isinstance(reloaded, WanTransformer3DMoTModel)
    assert reloaded.config.model_structure == "mot"
    assert reloaded.config.action_dim == 30
    assert reloaded.config.action_condition_mode == "fastwam"

def test_fsdp_sharding_discovers_mot_stream_modules(monkeypatch):
    from wan_va.distributed import fsdp

    model = WanTransformer3DMoTModel(
        **_tiny_kwargs(),
        model_structure="mot",
        action_hidden_dim=8,
        action_ffn_dim=16,
        action_mlp_hidden_dim=4,
    )
    seen = []
    monkeypatch.setattr(fsdp, "fully_shard", lambda module, **kwargs: seen.append(module))
    fsdp.shard_model(model)
    assert len(seen) == 2 + 6 * len(model.blocks)


def test_legacy_loader_defaults_to_shared(tmp_path):
    shared_dir = tmp_path / "legacy-shared"
    WanTransformer3DModel(**_tiny_kwargs()).save_pretrained(shared_dir)
    loaded = load_transformer(
        shared_dir, torch.float32, "cpu", attn_mode="torch", action_condition_mode="fastwam"
    )
    assert type(loaded) is WanTransformer3DModel
    assert loaded.config.action_dim == 30
