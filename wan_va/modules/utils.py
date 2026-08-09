# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
import logging

import torch
from diffusers import AutoencoderKLWan
from transformers import (
    T5TokenizerFast,
    UMT5EncoderModel,
)

from .model import WanTransformer3DModel


logger = logging.getLogger(__name__)
_WAN_VAE_CACHE_T = 2


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
    model = WanTransformer3DModel.from_pretrained(
        transformer_path,
        torch_dtype=torch_dtype,
        **kwargs
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
        self._eager_encoder = vae_model.encoder
        self._eager_quant_conv = vae_model.quant_conv
        self._compiled = False
        self._cudagraphs_active = False

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

    def compile_encoder(
        self,
        *,
        mode: str = "default",
        fullgraph: bool = False,
        dynamic: bool = False,
    ) -> None:
        """Compile VAE tensor regions while leaving streaming state eager."""
        if self._compiled:
            return
        try:
            self.encoder = torch.compile(
                self._eager_encoder,
                mode=mode,
                fullgraph=fullgraph,
                dynamic=dynamic,
            )
            self.quant_conv = torch.compile(
                self._eager_quant_conv,
                mode=mode,
                fullgraph=fullgraph,
                dynamic=dynamic,
            )
        except Exception as exc:
            self.encoder = self._eager_encoder
            self.quant_conv = self._eager_quant_conv
            logger.warning(
                "VAE encoder compile failed; using eager path: %s", exc
            )
            return
        self._compiled = True
        self._cudagraphs_active = mode in {"reduce-overhead", "max-autotune"}
        logger.info(
            "Compiled VAE encoder mode=%s dynamic=%s cudagraphs=%s",
            mode,
            dynamic,
            self._cudagraphs_active,
        )

    def encode_chunk(self, x_chunk, *, force_eager: bool = False):
        if hasattr(self.vae.config,
                   "patch_size") and self.vae.config.patch_size is not None:
            x_chunk = patchify(x_chunk, self.vae.config.patch_size)
        use_compiled = self._compiled and not force_eager
        first = self.feat_cache[0] if self.feat_cache else None
        if use_compiled and (
            first is None
            or (isinstance(first, torch.Tensor) and first.shape[2] < _WAN_VAE_CACHE_T)
        ):
            use_compiled = False
        encoder = self.encoder if use_compiled else self._eager_encoder
        quant_conv = self.quant_conv if use_compiled else self._eager_quant_conv
        cudagraphs_active = self._cudagraphs_active and use_compiled
        if cudagraphs_active:
            mark_step = getattr(torch.compiler, "cudagraph_mark_step_begin", None)
            if mark_step is not None:
                mark_step()
        feat_idx = [0]
        out = encoder(x_chunk, feat_cache=self.feat_cache, feat_idx=feat_idx)
        if cudagraphs_active:
            out = out.clone()
            self.feat_cache = [
                entry.clone() if isinstance(entry, torch.Tensor) else entry
                for entry in self.feat_cache
            ]
        enc = quant_conv(out)
        if cudagraphs_active:
            enc = enc.clone()
        return enc
