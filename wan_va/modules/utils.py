# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import json
from pathlib import Path

import torch
from diffusers import AutoencoderKLWan
from transformers import (
    T5TokenizerFast,
    UMT5EncoderModel,
)

from .model import WanTransformer3DModel


def _read_transformer_config(transformer_path):
    """Read local metadata; legacy checkpoints default to the shared backbone."""
    path = Path(str(transformer_path)).expanduser()
    config_path = path / "config.json"
    if not config_path.is_file():
        return {}
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def load_vae(
    vae_path,
    torch_dtype,
    torch_device,
):
    vae = AutoencoderKLWan.from_pretrained(
        vae_path,
        torch_dtype=torch_dtype,
    )
    return vae.to(torch_device)


def load_text_encoder(
    text_encoder_path,
    torch_dtype,
    torch_device,
):
    text_encoder = UMT5EncoderModel.from_pretrained(
        text_encoder_path,
        torch_dtype=torch_dtype,
    )
    return text_encoder.to(torch_device)


def load_tokenizer(tokenizer_path, ):
    tokenizer = T5TokenizerFast.from_pretrained(tokenizer_path, )
    return tokenizer


def load_transformer(
    transformer_path,
    torch_dtype,
    torch_device,
    **kwargs
):
    requested_structure = kwargs.pop("model_structure", None)
    mot_config = dict(kwargs.pop("mot_config", {}) or {})
    checkpoint_config = _read_transformer_config(transformer_path)
    model_structure = str(
        requested_structure or checkpoint_config.get("model_structure", "shared")
    )
    if model_structure not in {"shared", "mot"}:
        raise ValueError(
            "model_structure must be 'shared' or 'mot', "
            f"got {model_structure!r}"
        )
    if model_structure == "mot":
        from .mot_support.model import WanTransformer3DModel as MotSharedModel
        from .mot_support.model_mot import WanTransformer3DMoTModel
        source_structure = str(checkpoint_config.get("model_structure", "shared"))
        if source_structure == "mot":
            model = WanTransformer3DMoTModel.from_pretrained(
                transformer_path, torch_dtype=torch_dtype, **kwargs
            )
        else:
            shared_model = MotSharedModel.from_pretrained(
                transformer_path, torch_dtype=torch_dtype, **kwargs
            )
            model = WanTransformer3DMoTModel.from_shared_model(
                shared_model,
                action_hidden_dim=int(
                    mot_config.get("action_hidden_dim", 768)
                ),
                action_ffn_dim=mot_config.get("action_ffn_dim"),
                action_mlp_hidden_dim=int(
                    mot_config.get("action_mlp_hidden_dim", 256)
                ),
                init_mode=str(mot_config.get("init_mode", "video_interp_alpha")),
                alpha_scale=mot_config.get("alpha_scale"),
            )
    else:
        model = WanTransformer3DModel.from_pretrained(
            transformer_path, torch_dtype=torch_dtype, **kwargs
        )
    return model.to(torch_device)


def patchify(x, patch_size):
    if patch_size is None or patch_size == 1:
        return x
    batch_size, channels, frames, height, width = x.shape
    x = x.view(batch_size, channels, frames, height // patch_size, patch_size,
               width // patch_size, patch_size)
    x = x.permute(0, 1, 6, 4, 2, 3, 5).contiguous()
    x = x.view(batch_size, channels * patch_size * patch_size, frames,
               height // patch_size, width // patch_size)
    return x


class WanVAEStreamingWrapper:

    def __init__(self, vae_model):
        self.vae = vae_model
        self.encoder = vae_model.encoder
        self.quant_conv = vae_model.quant_conv

        if hasattr(self.vae, "_cached_conv_counts"):
            self.enc_conv_num = self.vae._cached_conv_counts["encoder"]
        else:
            count = 0
            for m in self.encoder.modules():
                if m.__class__.__name__ == "WanCausalConv3d":
                    count += 1
            self.enc_conv_num = count

        self.clear_cache()

    def clear_cache(self):
        self.feat_cache = [None] * self.enc_conv_num

    def encode_chunk(self, x_chunk):
        if hasattr(self.vae.config,
                   "patch_size") and self.vae.config.patch_size is not None:
            x_chunk = patchify(x_chunk, self.vae.config.patch_size)
        feat_idx = [0]
        out = self.encoder(x_chunk,
                           feat_cache=self.feat_cache,
                           feat_idx=feat_idx)
        enc = self.quant_conv(out)
        return enc
