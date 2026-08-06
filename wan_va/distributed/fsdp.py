# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import gc

import torch
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper as ptd_checkpoint_wrapper,
)

def apply_ac(model):
    """Apply activation checkpointing to the model."""
    for layer_id, transformer_block in enumerate(model.blocks):
        transformer_block = ptd_checkpoint_wrapper(transformer_block, preserve_rng_state=False)
        model.blocks[layer_id] = transformer_block


def shard_model(model,
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32):
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype,
        cast_forward_inputs=False,
    )
    fsdp_config = {"mp_policy": mp_policy, "reshard_after_forward": True}

    for block in model.blocks:
        # Shared blocks expose attn1/attn2/ffn; MoT blocks expose separate
        # video/action streams. Shard every stream-specific projection while
        # keeping the enclosing block as the FSDP unit.
        submodules = (
            (block.attn1, block.attn2, block.ffn)
            if hasattr(block, "attn1")
            else (
                block.video_attn1,
                block.video_attn2,
                block.video_ffn,
                block.action_attn1,
                block.action_attn2,
                block.action_ffn,
            )
        )
        for submodule in submodules:
            fully_shard(submodule, **fsdp_config)
        fully_shard(block, **fsdp_config)

    fully_shard(model, **fsdp_config)
    return model


def free_model(model):
    del model
    gc.collect()
    torch.cuda.empty_cache()
