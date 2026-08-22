# Copyright 2024-2025 The Robbyant Team Authors. All rights reserved.
# Integrated into nmx_vla by Wendi Chen.
import math
from collections import namedtuple
from copy import deepcopy
from functools import partial
from typing import Callable, ClassVar

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import FeedForward
from diffusers.models.embeddings import (
    PixArtAlphaTextProjection,
    TimestepEmbedding,
    Timesteps,
)
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm
from einops import rearrange
from torch.nn.attention.flex_attention import (
    BlockMask,
    and_masks,
    create_block_mask,
    flex_attention,
    or_masks,
)

try:
    from flash_attn_interface import flash_attn_func
except ImportError:
    from flash_attn import flash_attn_func

from .chunking import build_chunk_ids
from logging import getLogger

from .distributed.sequence_parallel import (
    all_to_all_sequence_parallel,
    gather_sequence_parallel_region,
    get_sequence_parallel_state,
    sequence_parallel_is_enabled,
    split_sequence_parallel_region,
)

logger = getLogger(__name__)


# missing_action_filling ("missing action filling") helpers live in a dedicated module now so this file
# can stay focused on the core transformer class. The helpers are model-class-free (they take
# a ``module`` / raw weights as args), so this import is one-directional: model.py imports them,
# missing_action_filling.py never imports model.py (no circular import). model_mot.py imports
# the SAME names from the SAME module, so shared + MoT share ONE implementation.
from .missing_action_filling import (
    _assert_missing_action_filling_configured_for_per_frame,
    _check_missing_action_filling_config_buf,
    _embed_action_with_missing_filling,
    _make_action_condition_all_missing_inputs,
    _register_missing_action_filling,
)

__all__ = [
    "WanTransformer3DModel",
    "WanRotaryPosEmbed",
    "WanTimeTextImageEmbedding",
    "apply_wan_modulation",
    "apply_wan_rotary_emb",
    "build_joint_attention_mask",
    "build_action_query_shape_groups",
    "build_joint_piece_ids",
    "build_joint_token_metadata",
    "build_latent_query_shape_groups",
    "build_action_time_offsets",
    "build_piece_local_grid_ids",
    "compile_transformer_blocks",
    "embed_lingbot_inputs",
    "embed_lingbot_timesteps",
    "finalize_lingbot_outputs",
    "split_wan_scale_shift",
]

from logging import getLogger

logger = getLogger(__name__)


def compile_transformer_blocks(
    blocks: nn.ModuleList,
    *,
    mode: str = "default",
    dynamic: bool | None = None,
    method_name: str | None = None,
) -> bool:
    """Wrap every transformer block's compute hot loop with ``torch.compile``.

    The per-block compute (projection + norm + modulation + SDPA + feed-forward
    chains) is the launch-bound hot loop of cached inference: each denoise step
    runs all blocks, and every block dispatches dozens of small pointwise
    kernels between the matmuls. Compiling at the block granularity lets
    Inductor fuse those chains while the stateful KV-cache bookkeeping (dict
    tensor mutation, ``valid_len`` slicing) stays in eager code behind explicit
    ``@torch.compiler.disable`` boundaries -- so its Python side effects persist
    across calls -- the same region-compile strategy the VAE encoder uses in
    ``WanVAEStreamingWrapper.compile_encoder``.

    Two compile granularities are supported, picked by ``method_name``:

    * ``method_name is None`` (default): replace each block with an
      ``OptimizedModule`` wrapper. This only accelerates call sites that invoke
      ``block(...)`` (routing through ``OptimizedModule.__call__``), e.g.
      ``WanTransformer3DModel`` whose cached forward calls ``block(...)``.
    * ``method_name="..."``: compile that bound method *in place*, shadowing the
      eager class method with a compiled instance attribute. Required when the
      inference loop calls a custom method directly -- e.g.
      ``WanTransformer3DMoTModel`` runs ``block.forward_single_stream(...)``.
      Accessing a non-``forward`` method on an ``OptimizedModule`` returns the
      *eager* original (``__getattr__`` delegates to ``_orig_mod``), so a
      whole-module compile silently no-ops for that path. Leaving the block
      object itself unwrapped also keeps the model's cache helpers
      (``cache_cross_kv`` / ``init_kv_cache`` / ...) reaching ``block.*``
      directly.

    Idempotent and defensive: already-compiled blocks are skipped, and any
    compile error restores the eager blocks/methods so a partial compile can
    never leave the model half-wrapped.

    Args:
        blocks: ``nn.ModuleList`` of transformer blocks, compiled in place.
        mode: Forwarded to ``torch.compile``. ``"default"`` applies Inductor
            fusion only (no CUDA Graphs) and is the safe production setting.
            ``"reduce-overhead"`` additionally replays each compiled region
            through a CUDA Graph for further launch-overhead savings; enable it
            only after validating numerics on-device, because it reuses output
            buffers across replays and interacts with the hand-rolled KV cache.
        dynamic: Forwarded to ``torch.compile``. ``None`` lets Dynamo
            specialize statically first and switch to dynamic shapes when the
            KV window length changes, avoiding a recompile per window size as
            the cache fills during warmup.
        method_name: When given, compile this bound method on each block instead
            of wrapping the whole module (see above).

    Returns:
        ``True`` if anything was compiled, ``False`` on no-op or fallback.
    """
    originals = list(blocks)
    if len(originals) == 0:
        return False

    if method_name is None:
        if all(hasattr(b, "_orig_mod") for b in originals):
            return False
        try:
            for i, block in enumerate(originals):
                if hasattr(block, "_orig_mod"):
                    continue
                blocks[i] = torch.compile(block, mode=mode, dynamic=dynamic)
        except Exception as exc:  # pragma: no cover - defensive fallback
            for i, block in enumerate(originals):
                blocks[i] = block
            logger.warning(
                "torch.compile failed for transformer blocks; falling back to "
                "eager. mode=%s, dynamic=%s, error=%s",
                mode,
                dynamic,
                exc,
            )
            return False
        logger.info(
            "Compiled %d transformer blocks with torch.compile(mode=%s, dynamic=%s).",
            len(originals),
            mode,
            dynamic,
        )
        return True

    # Method-level compile: ``OptimizedModule`` only routes ``__call__`` /
    # ``forward`` through the compiled graph, so a custom hot-path method must be
    # compiled directly or it silently runs eager.
    marker = f"_nmx_compiled_method_{method_name}"
    if all(getattr(b, marker, False) for b in originals):
        return False
    try:
        for block in originals:
            if getattr(block, marker, False):
                continue
            if not hasattr(block, method_name):
                raise AttributeError(
                    f"{type(block).__name__} has no method {method_name!r} to compile"
                )
            setattr(
                block,
                method_name,
                torch.compile(getattr(block, method_name), mode=mode, dynamic=dynamic),
            )
            setattr(block, marker, True)
    except Exception as exc:  # pragma: no cover - defensive fallback
        for block in originals:
            if getattr(block, marker, False):
                # Drop the compiled instance attribute so the eager class method
                # is used again, leaving no block half-compiled.
                try:
                    delattr(block, method_name)
                except AttributeError:
                    pass
                setattr(block, marker, False)
        logger.warning(
            "torch.compile failed for transformer block %s(); falling back to "
            "eager. mode=%s, dynamic=%s, error=%s",
            method_name,
            mode,
            dynamic,
            exc,
        )
        return False
    logger.info(
        "Compiled %d transformer block %s() methods with "
        "torch.compile(mode=%s, dynamic=%s).",
        len(originals),
        method_name,
        mode,
        dynamic,
    )
    return True


def custom_sdpa(q, k, v):
    out = F.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    )
    return out.transpose(1, 2)


def sequence_parallel_self_attention(attn_op, query, key, value):
    """Run self-attention on an already sequence-sharded local tensor.

    Inputs are [batch, local_seq, all_heads, head_dim]. The all-to-all exchange
    turns them into [batch, full_seq, local_heads, head_dim] for attention, then
    returns [batch, local_seq, all_heads, head_dim].
    """
    state = get_sequence_parallel_state()
    sp_size = int(state.size)
    if query.shape[2] % sp_size != 0:
        raise ValueError(
            "Sequence-parallel self-attention requires attention heads to be "
            f"divisible by sequence_parallel_size: heads={query.shape[2]}, "
            f"sequence_parallel_size={sp_size}."
        )

    query = all_to_all_sequence_parallel(query, scatter_dim=2, gather_dim=1)
    key = all_to_all_sequence_parallel(key, scatter_dim=2, gather_dim=1)
    value = all_to_all_sequence_parallel(value, scatter_dim=2, gather_dim=1)

    hidden_states = attn_op(query, key, value)
    hidden_states = all_to_all_sequence_parallel(
        hidden_states,
        scatter_dim=1,
        gather_dim=2,
    )
    return hidden_states


def apply_wan_modulation(
    hidden_states: torch.Tensor,
    shift: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return hidden_states * (1.0 + scale) + shift


def apply_wan_rotary_emb(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    # Real-valued RoPE: rotate consecutive (even, odd) channel pairs by the
    # per-position angles in ``freqs``. This is algebraically identical to the
    # complex form (it is the cos/sin of the same fp32 angle), but uses only
    # real ops -- no fp64, no complex view -- so Inductor can codegen and fuse
    # it instead of forcing a graph break. The rotation runs in fp32 and casts
    # back to ``x.dtype`` (bf16), matching eager to within bf16 rounding noise.
    cos = freqs.cos()
    sin = freqs.sin()
    x_pairs = x.float().unflatten(-1, (-1, 2))
    x_even = x_pairs[..., 0]
    x_odd = x_pairs[..., 1]
    rotated = torch.stack(
        [x_even * cos - x_odd * sin, x_even * sin + x_odd * cos], dim=-1
    )
    return rotated.flatten(-2).to(x.dtype)


def split_wan_scale_shift(
    scale_shift_table: torch.Tensor,
    temb: torch.Tensor,
) -> tuple[torch.Tensor, ...]:
    temb_scale_shift_table = (
        scale_shift_table[None].to(dtype=temb.dtype, device=temb.device) + temb.float()
    )
    return tuple(
        tensor.squeeze(1)
        for tensor in rearrange(temb_scale_shift_table, "b l n c -> b n l c").chunk(
            scale_shift_table.shape[1], dim=1
        )
    )


def finalize_lingbot_outputs(
    hidden_states: torch.Tensor,
    temb: torch.Tensor,
    norm_out: nn.Module,
    scale_shift_table: torch.Tensor,
) -> torch.Tensor:
    shift, scale = split_wan_scale_shift(scale_shift_table, temb[:, :, None, ...])
    shift = shift.to(hidden_states.device)
    scale = scale.to(hidden_states.device)
    return apply_wan_modulation(
        norm_out(hidden_states.float()),
        shift,
        scale,
    ).type_as(hidden_states)


def embed_lingbot_inputs(
    latents: torch.Tensor,
    *,
    input_type: str,
    patch_size,
    patch_embedding_mlp: nn.Module | None = None,
    action_embedder: nn.Module | None = None,
    text_embedder: nn.Module | None = None,
) -> torch.Tensor:
    if input_type == "latent":
        if patch_embedding_mlp is None:
            raise ValueError("patch_embedding_mlp is required for latent inputs")
        hidden_states = rearrange(
            latents,
            "b c (f p1) (h p2) (w p3) -> b (f h w) (c p1 p2 p3)",
            p1=patch_size[0],
            p2=patch_size[1],
            p3=patch_size[2],
        )
        hidden_states = hidden_states.to(patch_embedding_mlp.weight.dtype)
        return patch_embedding_mlp(hidden_states)
    if input_type == "action":
        if action_embedder is None:
            raise ValueError("action_embedder is required for action inputs")
        # missing_action_filling is a TRAIN-only mechanism (see `_embed_action_with_missing`): at
        # inference the action stream is the denoise TARGET, generated from noise rather
        # than a conditioning input, so "missing action" does not apply and missing_action_filling's
        # columns are intentionally never used here. If a future inference path feeds
        # partial ground-truth actions as conditioning, it must route through
        # `_embed_action_with_missing` with explicit validity instead of this plain embed.
        hidden_states = rearrange(latents, "b c f h w -> b (f h w) c")
        hidden_states = hidden_states.to(next(action_embedder.parameters()).dtype)
        return action_embedder(hidden_states)
    if input_type == "text":
        if text_embedder is None:
            raise ValueError("text_embedder is required for text inputs")
        latents = latents.to(next(text_embedder.parameters()).dtype)
        return text_embedder(latents)
    raise ValueError(f"Unsupported input type: {input_type}")


def embed_lingbot_timesteps(
    timesteps: torch.Tensor,
    height: int,
    width: int,
    dtype,
    *,
    patch_size,
    condition_embedder: nn.Module,
    condition_embedder_action: nn.Module,
    action_mode: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    patch_scale_h, patch_scale_w = (
        (1, 1) if action_mode else (patch_size[1], patch_size[2])
    )
    latent_time_steps = torch.repeat_interleave(
        timesteps, (height // patch_scale_h) * (width // patch_scale_w), dim=1
    )
    current_condition_embedder = (
        condition_embedder_action if action_mode else condition_embedder
    )
    temb, timestep_proj = current_condition_embedder(latent_time_steps, dtype=dtype)
    timestep_proj = timestep_proj.unflatten(2, (6, -1))
    return temb, timestep_proj


def _build_latent_token_ids(latent_shape, patch_size):
    batch_size, _, latent_frames, latent_height, latent_width = latent_shape
    patch_frames = latent_frames // patch_size[0]
    patch_height = latent_height // patch_size[1]
    patch_width = latent_width // patch_size[2]
    latent_seq_id = (
        torch.arange(batch_size)[:, None, None, None]
        .expand(-1, patch_frames, patch_height, patch_width)
        .flatten()
    )
    latent_frame_id = (
        torch.arange(patch_frames)[None, :, None, None]
        .expand(batch_size, -1, patch_height, patch_width)[None]
        .flatten()
    )
    return latent_seq_id, latent_frame_id


def _build_multiview_latent_token_ids(latent_view_shapes, patch_size):
    seq_ids = []
    frame_ids = []
    batch_size = latent_view_shapes[0][0]
    for latent_shape in latent_view_shapes:
        latent_seq_id, latent_frame_id = _build_latent_token_ids(
            latent_shape,
            patch_size,
        )
        seq_ids.append(latent_seq_id.reshape(batch_size, -1))
        frame_ids.append(latent_frame_id.reshape(batch_size, -1))
    return (
        torch.cat(seq_ids, dim=1).flatten(),
        torch.cat(frame_ids, dim=1).flatten(),
    )


def build_latent_query_shape_groups(
    latent_shape,
    patch_size,
    latent_view_shapes=None,
) -> list[tuple[int, int]]:
    source_shapes = (
        latent_view_shapes if latent_view_shapes is not None else [latent_shape]
    )
    return [
        (
            math.ceil(int(shape[2]) / int(patch_size[0])),
            (int(shape[3]) // int(patch_size[1]))
            * (int(shape[4]) // int(patch_size[2])),
        )
        for shape in source_shapes
    ]


def build_action_query_shape_groups(action_shape) -> list[tuple[int, int]]:
    return [(int(action_shape[2]), int(action_shape[3]) * int(action_shape[4]))]


def build_action_time_offsets(action_shape, *, device, dtype) -> list[torch.Tensor]:
    _, _, _, action_height, action_width = action_shape
    offsets = (torch.arange(action_height, device=device, dtype=dtype) + 1) / float(
        action_height + 1
    )
    offsets = offsets[:, None].expand(action_height, action_width).reshape(-1)
    return [offsets]


def build_joint_token_metadata(
    latent_shape,
    action_shape,
    padded_length,
    chunk_size,
    patch_size,
    device,
    latent_view_shapes=None,
    piece_frame_ranges=None,
    chunk_grouping_start_from_one=False,
    include_video_prediction=True,
):
    batch_size = latent_shape[0]
    _, _, action_frames, action_height, action_width = action_shape

    if latent_view_shapes is None:
        latent_seq_id, latent_frame_id = _build_latent_token_ids(
            latent_shape,
            patch_size,
        )
    else:
        latent_seq_id, latent_frame_id = _build_multiview_latent_token_ids(
            latent_view_shapes,
            patch_size,
        )
    action_seq_id = (
        torch.arange(batch_size)[:, None, None, None]
        .expand(-1, action_frames, action_height, action_width)
        .flatten()
    )
    seq_id_parts = []
    modality_id_parts = []
    if include_video_prediction:
        seq_id_parts.append(latent_seq_id)
        modality_id_parts.append(torch.zeros_like(latent_seq_id))
    seq_id_parts.extend([latent_seq_id, action_seq_id, action_seq_id])
    modality_id_parts.extend(
        [
            torch.zeros_like(latent_seq_id),
            torch.ones_like(action_seq_id),
            torch.ones_like(action_seq_id),
        ]
    )
    seq_ids = torch.cat(seq_id_parts)
    modality_ids = torch.cat(modality_id_parts)

    action_frame_id = (
        torch.arange(action_frames)[None, :, None, None]
        .expand(batch_size, -1, action_height, action_width)[None]
        .flatten()
    )
    latent_chunk_frame_id = _piece_local_frame_ids(
        seq_ids=latent_seq_id,
        frame_ids=latent_frame_id,
        piece_frame_ranges=piece_frame_ranges,
        temporal_patch_size=int(patch_size[0]),
        batch_size=batch_size,
        device=device,
    )
    action_chunk_frame_id = _piece_local_frame_ids(
        seq_ids=action_seq_id,
        frame_ids=action_frame_id,
        piece_frame_ranges=piece_frame_ranges,
        temporal_patch_size=1,
        batch_size=batch_size,
        device=device,
    )
    latent_chunk_id = build_chunk_ids(
        latent_chunk_frame_id,
        chunk_size,
        chunk_grouping_start_from_one=chunk_grouping_start_from_one,
    )
    action_chunk_id = build_chunk_ids(
        action_chunk_frame_id,
        chunk_size,
        chunk_grouping_start_from_one=chunk_grouping_start_from_one,
    )
    interleaved_chunk_id_parts = []
    chunk_id_parts = []
    if include_video_prediction:
        interleaved_chunk_id_parts.append(latent_chunk_id * 2)
        chunk_id_parts.append(latent_chunk_id)
    interleaved_chunk_id_parts.extend(
        [latent_chunk_id * 2, action_chunk_id * 2 + 1, action_chunk_id * 2 + 1]
    )
    chunk_id_parts.extend([latent_chunk_id, action_chunk_id, action_chunk_id])
    interleaved_chunk_ids = torch.cat(interleaved_chunk_id_parts)
    chunk_ids = torch.cat(chunk_id_parts)

    noise_id_parts = []
    if include_video_prediction:
        noise_id_parts.append(torch.zeros_like(latent_frame_id))
    noise_id_parts.extend(
        [
            torch.ones_like(latent_frame_id),
            torch.zeros_like(action_frame_id),
            torch.ones_like(action_frame_id),
        ]
    )
    noise_ids = torch.cat(noise_id_parts)

    seq_ids = F.pad(seq_ids, (0, padded_length), value=-1)
    interleaved_chunk_ids = F.pad(interleaved_chunk_ids, (0, padded_length), value=-1)
    chunk_ids = F.pad(chunk_ids, (0, padded_length), value=-1)
    modality_ids = F.pad(modality_ids, (0, padded_length), value=-1)
    noise_ids = F.pad(noise_ids, (0, padded_length), value=-1)

    return (
        seq_ids.long().to(device),
        interleaved_chunk_ids.long().to(device),
        chunk_ids.long().to(device),
        modality_ids.long().to(device),
        noise_ids.long().to(device),
    )


def _normalize_piece_frame_ranges(
    piece_frame_ranges,
    *,
    batch_size: int,
    device,
) -> torch.Tensor | None:
    if piece_frame_ranges is None:
        return None
    piece_frame_ranges = torch.as_tensor(
        piece_frame_ranges,
        dtype=torch.long,
        device=device,
    )
    if piece_frame_ranges.ndim == 2:
        piece_frame_ranges = piece_frame_ranges.unsqueeze(0)
    if piece_frame_ranges.ndim != 3 or piece_frame_ranges.shape[-1] != 2:
        raise ValueError(
            "`piece_frame_ranges` must have shape [B, piece_count, 2], "
            f"got {tuple(piece_frame_ranges.shape)}."
        )
    if piece_frame_ranges.shape[0] == 1 and batch_size > 1:
        piece_frame_ranges = piece_frame_ranges.expand(batch_size, -1, -1)
    if piece_frame_ranges.shape[0] != batch_size:
        raise ValueError(
            "`piece_frame_ranges` batch size does not match latent/action batch size: "
            f"{piece_frame_ranges.shape[0]} vs {batch_size}."
        )
    return piece_frame_ranges


def _piece_local_frame_ids(
    *,
    seq_ids: torch.Tensor,
    frame_ids: torch.Tensor,
    piece_frame_ranges,
    temporal_patch_size: int,
    batch_size: int,
    device,
) -> torch.Tensor:
    if piece_frame_ranges is None:
        return frame_ids
    piece_frame_ranges = _normalize_piece_frame_ranges(
        piece_frame_ranges,
        batch_size=batch_size,
        device=device,
    )
    frame_ids = frame_ids.long().to(device)
    local_frame_ids = frame_ids.clone()
    seq_ids = seq_ids.long().to(device)
    temporal_patch_size = max(1, int(temporal_patch_size))
    _, piece_count, _ = piece_frame_ranges.shape
    for batch_idx in range(batch_size):
        batch_mask = seq_ids == batch_idx
        for piece_idx in range(piece_count):
            start_frame = int(piece_frame_ranges[batch_idx, piece_idx, 0].item())
            end_frame = int(piece_frame_ranges[batch_idx, piece_idx, 1].item())
            if end_frame <= start_frame:
                continue
            start_patch = start_frame // temporal_patch_size
            end_patch = math.ceil(end_frame / temporal_patch_size)
            mask = batch_mask & (frame_ids >= start_patch) & (frame_ids < end_patch)
            local_frame_ids[mask] = frame_ids[mask] - start_patch
    return local_frame_ids


def build_piece_local_grid_ids(
    grid_id: torch.Tensor,
    *,
    piece_frame_ranges,
    query_shape_groups: list[tuple[int, int]],
    temporal_patch_size: int,
    local_time_offsets_by_group: list[torch.Tensor] | None = None,
) -> torch.Tensor:
    if piece_frame_ranges is None:
        return grid_id
    if grid_id.ndim != 3 or grid_id.shape[1] < 1:
        raise ValueError(
            f"`grid_id` must have shape [B, C, S], got {tuple(grid_id.shape)}"
        )
    batch_size = int(grid_id.shape[0])
    piece_frame_ranges = _normalize_piece_frame_ranges(
        piece_frame_ranges,
        batch_size=batch_size,
        device=grid_id.device,
    )
    out = grid_id.clone()
    temporal_patch_size = max(1, int(temporal_patch_size))
    _, piece_count, _ = piece_frame_ranges.shape
    expected_tokens = sum(
        int(frames) * int(tokens) for frames, tokens in query_shape_groups
    )
    if expected_tokens != int(grid_id.shape[2]):
        raise ValueError(
            "`query_shape_groups` token count does not match grid_id length: "
            f"{expected_tokens} vs {int(grid_id.shape[2])}"
        )
    if local_time_offsets_by_group is not None and len(
        local_time_offsets_by_group
    ) != len(query_shape_groups):
        raise ValueError(
            "`local_time_offsets_by_group` length must match `query_shape_groups`: "
            f"{len(local_time_offsets_by_group)} vs {len(query_shape_groups)}"
        )
    for batch_idx in range(batch_size):
        query_offset = 0
        for group_idx, (frame_count, tokens_per_frame) in enumerate(query_shape_groups):
            frame_count = int(frame_count)
            tokens_per_frame = int(tokens_per_frame)
            local_offsets = None
            if local_time_offsets_by_group is not None:
                local_offsets = local_time_offsets_by_group[group_idx].to(
                    device=grid_id.device,
                    dtype=out.dtype,
                )
                if local_offsets.numel() != tokens_per_frame:
                    raise ValueError(
                        "local time offset count must match tokens_per_frame: "
                        f"{local_offsets.numel()} vs {tokens_per_frame}"
                    )
                local_offsets = local_offsets.reshape(1, tokens_per_frame)
            for piece_idx in range(piece_count):
                start_frame = int(piece_frame_ranges[batch_idx, piece_idx, 0].item())
                end_frame = int(piece_frame_ranges[batch_idx, piece_idx, 1].item())
                if end_frame <= start_frame:
                    continue
                start_patch = max(0, start_frame // temporal_patch_size)
                end_patch = min(frame_count, math.ceil(end_frame / temporal_patch_size))
                if end_patch <= start_patch:
                    continue
                token_start = query_offset + start_patch * tokens_per_frame
                token_end = query_offset + end_patch * tokens_per_frame
                if local_offsets is None:
                    out[batch_idx, 0, token_start:token_end] -= start_patch
                    continue

                local_frames = torch.arange(
                    end_patch - start_patch,
                    device=grid_id.device,
                    dtype=out.dtype,
                )
                local_times = (local_frames[:, None] + local_offsets).reshape(-1)
                out[batch_idx, 0, token_start:token_end] = local_times
            query_offset += frame_count * tokens_per_frame
    return out


def _assign_piece_ids_for_frame_tokens(
    *,
    seq_ids: torch.Tensor,
    frame_ids: torch.Tensor,
    piece_frame_ranges: torch.Tensor,
    temporal_patch_size: int,
) -> torch.Tensor:
    piece_ids = torch.full_like(seq_ids, -1)
    temporal_patch_size = max(1, int(temporal_patch_size))
    batch_size, piece_count, _ = piece_frame_ranges.shape
    for batch_idx in range(batch_size):
        batch_mask = seq_ids == batch_idx
        for piece_idx in range(piece_count):
            start_frame = int(piece_frame_ranges[batch_idx, piece_idx, 0].item())
            end_frame = int(piece_frame_ranges[batch_idx, piece_idx, 1].item())
            if end_frame <= start_frame:
                continue
            start_patch = start_frame // temporal_patch_size
            end_patch = math.ceil(end_frame / temporal_patch_size)
            piece_mask = (
                batch_mask & (frame_ids >= start_patch) & (frame_ids < end_patch)
            )
            piece_ids[piece_mask] = piece_idx
    return piece_ids


def build_joint_piece_ids(
    latent_shape,
    action_shape,
    padded_length,
    patch_size,
    device,
    *,
    piece_frame_ranges=None,
    latent_view_shapes=None,
    include_video_prediction=True,
) -> torch.Tensor | None:
    batch_size = latent_shape[0]
    piece_frame_ranges = _normalize_piece_frame_ranges(
        piece_frame_ranges,
        batch_size=batch_size,
        device=device,
    )
    if piece_frame_ranges is None:
        return None

    _, _, action_frames, action_height, action_width = action_shape
    if latent_view_shapes is None:
        latent_seq_id, latent_frame_id = _build_latent_token_ids(
            latent_shape,
            patch_size,
        )
    else:
        latent_seq_id, latent_frame_id = _build_multiview_latent_token_ids(
            latent_view_shapes,
            patch_size,
        )
    action_seq_id = (
        torch.arange(batch_size)[:, None, None, None]
        .expand(-1, action_frames, action_height, action_width)
        .flatten()
    )
    action_frame_id = (
        torch.arange(action_frames)[None, :, None, None]
        .expand(batch_size, -1, action_height, action_width)[None]
        .flatten()
    )

    latent_seq_id = latent_seq_id.long().to(device)
    latent_frame_id = latent_frame_id.long().to(device)
    action_seq_id = action_seq_id.long().to(device)
    action_frame_id = action_frame_id.long().to(device)
    latent_piece_id = _assign_piece_ids_for_frame_tokens(
        seq_ids=latent_seq_id,
        frame_ids=latent_frame_id,
        piece_frame_ranges=piece_frame_ranges,
        temporal_patch_size=int(patch_size[0]),
    )
    action_piece_id = _assign_piece_ids_for_frame_tokens(
        seq_ids=action_seq_id,
        frame_ids=action_frame_id,
        piece_frame_ranges=piece_frame_ranges,
        temporal_patch_size=1,
    )
    piece_id_parts = []
    if include_video_prediction:
        piece_id_parts.append(latent_piece_id)
    piece_id_parts.extend([latent_piece_id, action_piece_id, action_piece_id])
    piece_ids = torch.cat(piece_id_parts)
    return F.pad(piece_ids, (0, padded_length), value=-1).long().to(device)


def build_joint_attention_mask(
    latent_shape,
    action_shape,
    chunk_size,
    window_size,
    patch_size,
    device,
    action_condition_mode="inverse_dynamics",
    latent_view_shapes=None,
    piece_frame_ranges=None,
    attention_piece_frame_ranges=None,
    chunk_grouping_start_from_one=False,
    include_video_prediction=True,
) -> torch.Tensor:
    (
        seq_ids,
        interleaved_chunk_ids,
        chunk_ids,
        modality_ids,
        noise_ids,
    ) = build_joint_token_metadata(
        latent_shape=latent_shape,
        action_shape=action_shape,
        padded_length=0,
        chunk_size=chunk_size,
        patch_size=patch_size,
        device=device,
        latent_view_shapes=latent_view_shapes,
        piece_frame_ranges=piece_frame_ranges,
        chunk_grouping_start_from_one=chunk_grouping_start_from_one,
        include_video_prediction=include_video_prediction,
    )
    attention_piece_ids = None
    if attention_piece_frame_ranges is not None:
        attention_piece_ids = build_joint_piece_ids(
            latent_shape=latent_shape,
            action_shape=action_shape,
            padded_length=0,
            patch_size=patch_size,
            device=device,
            piece_frame_ranges=attention_piece_frame_ranges,
            latent_view_shapes=latent_view_shapes,
            include_video_prediction=include_video_prediction,
        )
    mask_mod = FlexAttnFunc._get_mask_mod(
        seq_ids,
        interleaved_chunk_ids,
        noise_ids,
        chunk_ids,
        modality_ids,
        window_size,
        action_condition_mode=action_condition_mode,
        attention_piece_ids=attention_piece_ids,
    )
    q_idx = torch.arange(len(seq_ids), device=device)[:, None]
    kv_idx = torch.arange(len(seq_ids), device=device)[None, :]
    dummy = torch.zeros((), dtype=torch.long, device=device)
    return mask_mod(dummy, dummy, q_idx, kv_idx).unsqueeze(0).unsqueeze(0)


class FlexAttnFunc(nn.Module):
    flex_attn: ClassVar[Callable] = torch.compile(
        flex_attention,
        dynamic=True,
    )
    # The compiled mask builder can select an invalid Triton XBLOCK for the
    # long packed sequences used by the full NOE mixture. Mask construction is
    # a one-time startup operation, so keep it eager and reserve compilation
    # for the attention kernel itself.
    create_block_mask_func: ClassVar[Callable] = create_block_mask
    mask_block_size: ClassVar[int] = 64
    attention_mask: ClassVar[BlockMask] = None
    cross_attention_mask: ClassVar[BlockMask] = None
    cross_attention_seq_ids: ClassVar[torch.Tensor | None] = None
    cross_attention_piece_ids: ClassVar[torch.Tensor | None] = None
    cross_attention_text_seq_ids: ClassVar[torch.Tensor | None] = None
    cross_attention_text_piece_ids: ClassVar[torch.Tensor | None] = None

    @classmethod
    def set_mask_block_size(cls, block_size: int) -> None:
        block_size = int(block_size)
        if block_size < 64 or block_size % 64 != 0:
            raise ValueError(
                "flex_mask_block_size must be >= 64 and divisible by the "
                f"FlexAttention compute tile size 64, got {block_size}."
            )
        cls.mask_block_size = block_size
        cls._patch_inductor_default_config_for_mask_block_size(block_size)

    @staticmethod
    def _patch_inductor_default_config_for_mask_block_size(block_size: int) -> None:
        if block_size >= 128:
            return
        try:
            import torch._inductor.kernel.flex_attention as inductor_flex_attention
        except Exception:
            return

        if not hasattr(inductor_flex_attention, "_nmx_vla_original_get_nv_config"):
            inductor_flex_attention._nmx_vla_original_get_nv_config = (
                inductor_flex_attention._get_nv_config
            )

            def _nmx_vla_get_nv_config(query, mode):
                config = inductor_flex_attention._nmx_vla_original_get_nv_config(
                    query, mode
                )
                mask_block_size = getattr(
                    inductor_flex_attention,
                    "_nmx_vla_flex_mask_block_size",
                    128,
                )
                if mask_block_size >= 128:
                    return config
                block_m, block_n, num_warps, num_stages = config
                return (
                    min(block_m, mask_block_size),
                    min(block_n, mask_block_size),
                    num_warps,
                    num_stages,
                )

            inductor_flex_attention._get_nv_config = _nmx_vla_get_nv_config

        inductor_flex_attention._nmx_vla_flex_mask_block_size = block_size

    def __init__(
        self,
        is_cross=False,
    ) -> None:
        super().__init__()
        self.is_cross = is_cross

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        dtype=torch.bfloat16,
    ) -> torch.Tensor:
        q_varlen = rearrange(query[0], "s n d -> 1 n s d")
        k_varlen = rearrange(key[0], "s n d -> 1 n s d")
        v_varlen = rearrange(value[0], "s n d -> 1 n s d")

        half_dtypes = (torch.float16, torch.bfloat16)
        assert dtype in half_dtypes

        def half(x):
            return x if x.dtype in half_dtypes else x.to(dtype)

        q_varlen = half(q_varlen)
        k_varlen = half(k_varlen)
        v_varlen = half(v_varlen)
        q_varlen = q_varlen.to(v_varlen.dtype)
        k_varlen = k_varlen.to(v_varlen.dtype)

        block_mask = (
            FlexAttnFunc.cross_attention_mask
            if self.is_cross
            else FlexAttnFunc.attention_mask
        )

        x_out = FlexAttnFunc.flex_attn(
            q_varlen,
            k_varlen,
            v_varlen,
            block_mask=block_mask,
            kernel_options={
                "BLOCK_M": 64,
                "BLOCK_N": 64,
                "BLOCK_M1": 32,
                "BLOCK_N1": 64,
                "BLOCK_M2": 64,
                "BLOCK_N2": 32,
            },
        )

        x_out = rearrange(x_out, "b n s d -> b s n d")
        return x_out

    @staticmethod
    @torch.no_grad()
    def init_mask(
        latent_shape,
        action_shape,
        padded_length,
        chunk_size,
        window_size,
        patch_size,
        device,
        action_condition_mode="inverse_dynamics",
        latent_view_shapes=None,
        piece_frame_ranges=None,
        attention_piece_frame_ranges=None,
        text_tokens_per_piece=512,
        chunk_grouping_start_from_one=False,
        include_video_prediction=True,
    ):
        torch._inductor.config.realize_opcount_threshold = 100
        if action_condition_mode not in {"inverse_dynamics", "fastwam"}:
            raise ValueError(
                "action_condition_mode must be 'inverse_dynamics' or 'fastwam', "
                f"got {action_condition_mode!r}"
            )
        (
            seq_ids,
            interleaved_chunk_ids,
            chunk_ids,
            modality_ids,
            noise_ids,
        ) = build_joint_token_metadata(
            latent_shape=latent_shape,
            action_shape=action_shape,
            padded_length=padded_length,
            chunk_size=chunk_size,
            patch_size=patch_size,
            device=device,
            latent_view_shapes=latent_view_shapes,
            piece_frame_ranges=piece_frame_ranges,
            chunk_grouping_start_from_one=chunk_grouping_start_from_one,
            include_video_prediction=include_video_prediction,
        )
        piece_ids = build_joint_piece_ids(
            latent_shape=latent_shape,
            action_shape=action_shape,
            padded_length=padded_length,
            patch_size=patch_size,
            device=device,
            piece_frame_ranges=piece_frame_ranges,
            latent_view_shapes=latent_view_shapes,
            include_video_prediction=include_video_prediction,
        )

        # Self-attention isolation uses packing physical boundaries
        # (attention_piece_frame_ranges) so that different source episodes
        # packed into one unit cannot attend each other, while subtask
        # segments within the same episode remain fully connected.
        attn_piece_ids = None
        if attention_piece_frame_ranges is not None:
            attn_piece_ids = build_joint_piece_ids(
                latent_shape=latent_shape,
                action_shape=action_shape,
                padded_length=padded_length,
                patch_size=patch_size,
                device=device,
                piece_frame_ranges=attention_piece_frame_ranges,
                latent_view_shapes=latent_view_shapes,
                include_video_prediction=include_video_prediction,
            )

        mask_mod = FlexAttnFunc._get_mask_mod(
            seq_ids,
            interleaved_chunk_ids,
            noise_ids,
            chunk_ids,
            modality_ids,
            window_size,
            action_condition_mode=action_condition_mode,
            attention_piece_ids=attn_piece_ids,
        )
        block_mask = FlexAttnFunc.create_block_mask_func(
            mask_mod,
            1,
            1,
            len(seq_ids),
            len(seq_ids),
            device=device,
            BLOCK_SIZE=FlexAttnFunc.mask_block_size,
        )
        FlexAttnFunc.attention_mask = block_mask

        text_seq_ids, text_piece_ids = FlexAttnFunc._build_cross_text_ids(
            batch_size=int(latent_shape[0]),
            device=device,
            piece_frame_ranges=piece_frame_ranges,
            text_tokens_per_piece=text_tokens_per_piece,
        )
        mask_mod_cross = FlexAttnFunc._get_cross_mask_mod(
            seq_ids.long().to(device),
            text_seq_ids.long().to(device),
            piece_ids=piece_ids,
            text_piece_ids=text_piece_ids,
        )
        block_mask_cross = FlexAttnFunc.create_block_mask_func(
            mask_mod_cross,
            1,
            1,
            len(seq_ids),
            len(text_seq_ids),
            device=device,
            BLOCK_SIZE=FlexAttnFunc.mask_block_size,
        )
        FlexAttnFunc.cross_attention_mask = block_mask_cross
        FlexAttnFunc.cross_attention_seq_ids = seq_ids.long().to(device)
        FlexAttnFunc.cross_attention_piece_ids = (
            piece_ids.long().to(device) if piece_ids is not None else None
        )
        FlexAttnFunc.cross_attention_text_seq_ids = text_seq_ids.long().to(device)
        FlexAttnFunc.cross_attention_text_piece_ids = (
            text_piece_ids.long().to(device) if text_piece_ids is not None else None
        )

    @staticmethod
    @torch.no_grad()
    def init_cross_attention_query_slice_mask(query_start, query_length, device):
        seq_ids = FlexAttnFunc.cross_attention_seq_ids
        text_seq_ids = FlexAttnFunc.cross_attention_text_seq_ids
        if seq_ids is None or text_seq_ids is None:
            raise RuntimeError(
                "Cross-attention metadata is not initialized; call init_mask first."
            )

        query_start = int(query_start)
        query_length = int(query_length)
        query_end = query_start + query_length
        if query_start < 0 or query_end > int(seq_ids.shape[0]):
            raise ValueError(
                "Invalid cross-attention query slice for sequence-parallel mask: "
                f"start={query_start}, length={query_length}, total={seq_ids.shape[0]}."
            )

        local_seq_ids = seq_ids[query_start:query_end].long().to(device)
        text_seq_ids = text_seq_ids.long().to(device)
        text_piece_ids = FlexAttnFunc.cross_attention_text_piece_ids
        if text_piece_ids is not None:
            text_piece_ids = text_piece_ids.long().to(device)
            piece_ids = FlexAttnFunc.cross_attention_piece_ids
            if piece_ids is None:
                raise RuntimeError(
                    "Cross-attention text piece ids are initialized without query piece ids."
                )
            piece_ids = piece_ids[query_start:query_end].long().to(device)
        else:
            piece_ids = None

        mask_mod_cross = FlexAttnFunc._get_cross_mask_mod(
            local_seq_ids,
            text_seq_ids,
            piece_ids=piece_ids,
            text_piece_ids=text_piece_ids,
        )
        FlexAttnFunc.cross_attention_mask = FlexAttnFunc.create_block_mask_func(
            mask_mod_cross,
            1,
            1,
            query_length,
            int(text_seq_ids.shape[0]),
            device=device,
            BLOCK_SIZE=FlexAttnFunc.mask_block_size,
        )

    @staticmethod
    @torch.no_grad()
    def _build_cross_text_ids(
        batch_size,
        device,
        piece_frame_ranges=None,
        text_tokens_per_piece=512,
    ):
        text_tokens_per_piece = int(text_tokens_per_piece)
        if piece_frame_ranges is None:
            text_seq_ids = (
                torch.arange(batch_size, device=device)[:, None]
                .expand(-1, text_tokens_per_piece)
                .flatten()
            )
            return text_seq_ids, None

        piece_count = int(piece_frame_ranges.shape[1])
        seq_chunks = []
        piece_chunks = []
        for batch_idx in range(batch_size):
            seq_chunks.append(
                torch.full(
                    (piece_count * text_tokens_per_piece,),
                    batch_idx,
                    dtype=torch.long,
                    device=device,
                )
            )
            piece_chunks.append(
                torch.arange(
                    piece_count, device=device, dtype=torch.long
                ).repeat_interleave(text_tokens_per_piece)
            )
        return torch.cat(seq_chunks), torch.cat(piece_chunks)

    @staticmethod
    @torch.no_grad()
    def _get_cross_mask_mod(seq_ids, text_seq_ids, piece_ids=None, text_piece_ids=None):
        def seq_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            valid = (
                (seq_ids[q_idx] == text_seq_ids[kv_idx])
                & (seq_ids[q_idx] >= 0)
                & (text_seq_ids[kv_idx] >= 0)
            )
            if piece_ids is not None and text_piece_ids is not None:
                valid = (
                    valid
                    & (piece_ids[q_idx] == text_piece_ids[kv_idx])
                    & (piece_ids[q_idx] >= 0)
                    & (text_piece_ids[kv_idx] >= 0)
                )
            return valid

        return seq_mask

    @staticmethod
    @torch.no_grad()
    def _get_mask_mod(
        seq_ids,
        interleaved_chunk_ids,
        noise_ids,
        chunk_ids,
        modality_ids,
        window_size,
        action_condition_mode="inverse_dynamics",
        attention_piece_ids=None,
    ):
        def seq_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (
                (seq_ids[q_idx] == seq_ids[kv_idx])
                & (seq_ids[q_idx] >= 0)
                & (seq_ids[kv_idx] >= 0)
            )

        def block_causal_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return interleaved_chunk_ids[kv_idx] <= interleaved_chunk_ids[q_idx]

        def block_causal_mask_exclude_self(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return interleaved_chunk_ids[kv_idx] < interleaved_chunk_ids[q_idx]

        def block_self_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return interleaved_chunk_ids[kv_idx] == interleaved_chunk_ids[q_idx]

        def clean2clean_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (noise_ids[q_idx] == 1) & (noise_ids[kv_idx] == 1)

        def noise2clean_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (noise_ids[q_idx] == 0) & (noise_ids[kv_idx] == 1)

        def noise2noise_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (noise_ids[q_idx] == 0) & (noise_ids[kv_idx] == 0)

        def block_window_mask(
            b: torch.Tensor,
            h: torch.Tensor,
            q_idx: torch.Tensor,
            kv_idx: torch.Tensor,
            window_size: int,
        ):
            return (
                interleaved_chunk_ids[q_idx] - interleaved_chunk_ids[kv_idx]
            ).abs() <= window_size

        def action_no_current_clean_video_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            blocked = (
                (noise_ids[q_idx] == 0)
                & (modality_ids[q_idx] == 1)
                & (noise_ids[kv_idx] == 1)
                & (modality_ids[kv_idx] == 0)
                & (chunk_ids[q_idx] == chunk_ids[kv_idx])
            )
            return ~blocked

        def piece_mask(
            b: torch.Tensor, h: torch.Tensor, q_idx: torch.Tensor, kv_idx: torch.Tensor
        ):
            return (
                (attention_piece_ids[q_idx] == attention_piece_ids[kv_idx])
                & (attention_piece_ids[q_idx] >= 0)
                & (attention_piece_ids[kv_idx] >= 0)
            )

        mask_list = []
        mask_list.append(and_masks(clean2clean_mask, block_causal_mask))
        mask_list.append(and_masks(noise2clean_mask, block_causal_mask_exclude_self))
        mask_list.append(and_masks(noise2noise_mask, block_self_mask))
        mask = or_masks(*mask_list)
        if action_condition_mode == "fastwam":
            mask = and_masks(mask, action_no_current_clean_video_mask)
        mask = and_masks(mask, seq_mask)
        if attention_piece_ids is not None:
            mask = and_masks(mask, piece_mask)
        mask = and_masks(mask, partial(block_window_mask, window_size=window_size))
        return mask


class WanTimeTextImageEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        time_freq_dim,
        time_proj_dim,
        text_embed_dim,
        pos_embed_seq_len,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(
            num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0
        )
        self.time_embedder = TimestepEmbedding(
            in_channels=time_freq_dim, time_embed_dim=dim
        )
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(
            text_embed_dim, dim, act_fn="gelu_tanh"
        )

    def forward(
        self,
        timestep: torch.Tensor,
        dtype=None,
    ):
        B, L = timestep.shape
        timestep = timestep.reshape(-1)
        timestep = self.timesteps_proj(timestep)
        # time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        time_embedder_dtype = self.time_embedder.linear_1.weight.dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).to(dtype=dtype)
        timestep_proj = self.time_proj(self.act_fn(temb))
        return temb.reshape(B, L, -1), timestep_proj.reshape(B, L, -1)


class WanRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        attention_head_dim: int,
        patch_size,
        max_seq_len: int,
        theta: float = 10000.0,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len
        self.theta = theta

        self.f_dim = self.attention_head_dim - 2 * (self.attention_head_dim // 3)
        self.h_dim = self.attention_head_dim // 3
        self.w_dim = self.attention_head_dim // 3

        # Precompute and register buffers
        f_freqs_base, h_freqs_base, w_freqs_base = self._precompute_freqs_base()
        self.f_freqs_base = f_freqs_base
        self.h_freqs_base = h_freqs_base
        self.w_freqs_base = w_freqs_base

    def _precompute_freqs_base(self):
        # freqs_base = 1.0 / (theta ** (2k / dim))
        f_freqs_base = 1.0 / (
            self.theta
            ** (
                torch.arange(0, self.f_dim, 2)[: (self.f_dim // 2)].double()
                / self.f_dim
            )
        )
        h_freqs_base = 1.0 / (
            self.theta
            ** (
                torch.arange(0, self.h_dim, 2)[: (self.h_dim // 2)].double()
                / self.h_dim
            )
        )
        w_freqs_base = 1.0 / (
            self.theta
            ** (
                torch.arange(0, self.w_dim, 2)[: (self.w_dim // 2)].double()
                / self.w_dim
            )
        )
        return f_freqs_base, h_freqs_base, w_freqs_base

    def forward(self, grid_ids):
        with torch.no_grad():
            # Move the fp64 base frequencies onto the compute device once and
            # cache them there. Re-running ``.to(device)`` every call issues a
            # fresh CPU->CUDA copy, which is illegal inside a CUDA-graph capture
            # (the steady-state denoise hot path). Caching keeps the copy out of
            # the graph while preserving fp64 precision -- registering a buffer
            # instead would risk a later ``model.to(dtype)`` downcasting it.
            device = grid_ids.device
            if self.f_freqs_base.device != device:
                self.f_freqs_base = self.f_freqs_base.to(device)
                self.h_freqs_base = self.h_freqs_base.to(device)
                self.w_freqs_base = self.w_freqs_base.to(device)
            f_freqs = grid_ids[:, 0, :].unsqueeze(-1) * self.f_freqs_base
            h_freqs = grid_ids[:, 1, :].unsqueeze(-1) * self.h_freqs_base
            w_freqs = grid_ids[:, 2, :].unsqueeze(-1) * self.w_freqs_base
            # Return the real rotation angles (fp32). Consumers turn these into
            # cos/sin inside ``apply_wan_rotary_emb``; keeping the representation
            # real avoids complex/fp64 ops that block torch.compile fusion.
            freqs = torch.cat([f_freqs, h_freqs, w_freqs], dim=-1).float()

        return freqs


# Cached cross-attention key/value for a fixed text context. ``context`` is the
# exact text tensor the (key, value) were projected from; its identity is
# checked on lookup so a prompt change can never serve stale projections.
_CrossKVCache = namedtuple("_CrossKVCache", ("context", "key", "value"))


class WanAttention(torch.nn.Module):
    def __init__(
        self,
        dim,
        heads=8,
        dim_head=64,
        eps=1e-5,
        dropout=0.0,
        cross_attention_dim_head=None,
        attn_mode="torch",
    ):
        super().__init__()
        if attn_mode == "torch":
            self.attn_op = custom_sdpa
        elif attn_mode == "flashattn":
            self.attn_op = flash_attn_func
        elif attn_mode == "flex":
            self.attn_op = FlexAttnFunc(cross_attention_dim_head is not None)
        else:
            raise ValueError(
                f"Unsupported attention mode: {attn_mode}, only support torch and flashattn"
            )

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.cross_attention_dim_head = cross_attention_dim_head
        self.kv_inner_dim = (
            self.inner_dim
            if cross_attention_dim_head is None
            else cross_attention_dim_head * heads
        )

        self.to_q = torch.nn.Linear(dim, self.inner_dim, bias=True)
        self.to_k = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_v = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_out = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.inner_dim, dim, bias=True),
                torch.nn.Dropout(dropout),
            ]
        )
        self.norm_q = torch.nn.RMSNorm(
            dim_head * heads, eps=eps, elementwise_affine=True
        )
        self.norm_k = torch.nn.RMSNorm(
            dim_head * heads, eps=eps, elementwise_affine=True
        )
        self.attn_caches = {} if cross_attention_dim_head is None else None
        # Cross-attention K/V depend only on the per-episode fixed text context.
        # Pre-projected once via ``cache_cross_kv`` and reused by ``forward``.
        self._cross_kv_cache: dict[str, _CrossKVCache] = {}

    def clear_pred_cache(self, cache_name):
        if self.attn_caches is None:
            return
        cache = self.attn_caches.get(cache_name)
        if cache is None:
            return
        pred_len = int(cache.get("pred_len", 0))
        if pred_len <= 0:
            return
        valid_len = int(cache["valid_len"])
        new_valid = max(0, valid_len - pred_len)
        cache["mask"][new_valid:valid_len] = False
        cache["id"][new_valid:valid_len] = -1
        cache["is_pred"][new_valid:valid_len] = False
        cache["valid_len"] = new_valid
        cache["write_pos"] = new_valid
        cache["pred_len"] = 0

    def clear_cache(self, cache_name):
        if self.attn_caches is None:
            return
        self.attn_caches[cache_name] = None

    @torch.no_grad()
    def cache_cross_kv(
        self,
        encoder_hidden_states,
        cache_name="pos",
        action_encoder_hidden_states: torch.Tensor | None = None,
    ):
        """Pre-project the fixed text context into cross-attention K/V.

        No-op for self-attention. Overwrites any existing entry, so it is safe
        to call on every episode reset.
        """
        if self.cross_attention_dim_head is None:
            return
        key = self.norm_k(self.to_k(encoder_hidden_states))
        key = key.unflatten(2, (self.heads, -1))
        value = self.to_v(encoder_hidden_states).unflatten(2, (self.heads, -1))
        self._cross_kv_cache[cache_name] = _CrossKVCache(
            encoder_hidden_states, key, value
        )

    def clear_cross_kv(self, cache_name=None):
        """Drop cached cross-attention K/V (one ``cache_name``, or all)."""
        if cache_name is None:
            self._cross_kv_cache.clear()
        else:
            self._cross_kv_cache.pop(cache_name, None)

    def _cross_kv(self, k, v, cache_name):
        """Cross-attention (key, value): reuse the cached projection when it
        matches the current context, otherwise project on the fly."""
        entry = self._cross_kv_cache.get(cache_name)
        if entry is not None and entry.context is k:
            return entry.key, entry.value
        key = self.norm_k(self.to_k(k)).unflatten(2, (self.heads, -1))
        value = self.to_v(v).unflatten(2, (self.heads, -1))
        return key, value

    @torch.compiler.disable
    def _get_cached_kv(self, cache_name):
        # Eager-only: reads the Python ``valid_len`` and slices the cache buffer.
        # Running this under torch.compile would bake ``valid_len`` into the
        # graph; keeping it eager hands the compiled region a concrete history
        # tensor (or None) every call, with no in-graph cache state.
        cache = self.attn_caches.get(cache_name)
        if cache is None:
            return None, None
        valid_len = int(cache.get("valid_len", 0))
        if valid_len == 0:
            return None, None
        return cache["k"][:, :valid_len], cache["v"][:, :valid_len]

    def init_kv_cache(
        self,
        cache_name,
        total_tolen,
        num_head,
        head_dim,
        device,
        dtype,
        batch_size,
        headroom: int = 0,
    ):
        """Allocate a linear-addressing KV cache for ``cache_name``.

        Valid tokens always live at slots ``[0, valid_len)`` and ``valid_len``
        never exceeds ``capacity`` (older tokens are evicted in
        ``update_cache``). The physical buffer is sized ``capacity + headroom``
        so the denoise hot path can stage the current step's K/V into the
        headroom region (``_stage_and_view``) and hand SDPA a single contiguous
        slice with no ``torch.cat`` copy of the history.
        """
        if self.attn_caches is None:
            return
        capacity = int(total_tolen)
        physical_len = capacity + int(headroom)
        self.attn_caches[cache_name] = {
            "k": torch.empty(
                [batch_size, physical_len, num_head, head_dim],
                device=device,
                dtype=dtype,
            ),
            "v": torch.empty(
                [batch_size, physical_len, num_head, head_dim],
                device=device,
                dtype=dtype,
            ),
            "id": torch.full((physical_len,), -1, device=device),
            "mask": torch.zeros((physical_len,), dtype=torch.bool, device=device),
            "is_pred": torch.zeros((physical_len,), dtype=torch.bool, device=device),
            "capacity": capacity,
            "headroom": int(headroom),
            "write_pos": 0,
            "valid_len": 0,
            "pred_len": 0,
            "next_id": 0,
        }

    def _next_cache_id(self, cache_name):
        cache = self.attn_caches[cache_name]
        cache_id = int(cache["next_id"])
        cache["next_id"] = cache_id + 1
        return cache_id

    def _evict_oldest(self, cache: dict, evict_len: int) -> None:
        """In-place left shift to drop the oldest ``evict_len`` valid tokens.

        Source and destination ranges overlap so we materialise the keep
        region via ``.clone()`` before the write-back. This is invoked at most
        a handful of times per chunk (last denoise steps + compute_kv_cache
        refresh) so the O(capacity * H * D) cost is amortised away from the
        per-layer / per-denoise-step hot path.

        Pred tokens live at the *tail* of the valid range ``[valid_len -
        pred_len, valid_len)``; head eviction only shortens ``pred_len`` when
        it reaches into that tail region.
        """
        if evict_len <= 0:
            return
        valid_len = int(cache["valid_len"])
        pred_len = int(cache["pred_len"])
        if evict_len >= valid_len:
            cache["mask"][:valid_len] = False
            cache["id"][:valid_len] = -1
            cache["is_pred"][:valid_len] = False
            cache["valid_len"] = 0
            cache["write_pos"] = 0
            cache["pred_len"] = 0
            return
        new_valid = valid_len - evict_len
        cache["k"][:, :new_valid] = cache["k"][:, evict_len:valid_len].clone()
        cache["v"][:, :new_valid] = cache["v"][:, evict_len:valid_len].clone()
        cache["id"][:new_valid] = cache["id"][evict_len:valid_len].clone()
        cache["mask"][:new_valid] = cache["mask"][evict_len:valid_len].clone()
        cache["is_pred"][:new_valid] = cache["is_pred"][evict_len:valid_len].clone()
        cache["mask"][new_valid:valid_len] = False
        cache["id"][new_valid:valid_len] = -1
        cache["is_pred"][new_valid:valid_len] = False
        cache["valid_len"] = new_valid
        cache["write_pos"] = new_valid
        non_pred = valid_len - pred_len
        if evict_len > non_pred:
            cache["pred_len"] = max(0, pred_len - (evict_len - non_pred))

    @torch.compiler.disable
    def update_cache(self, cache_name, key, value, is_pred):
        # Eager-only: the commit path mutates plain-dict tensors in place and
        # advances Python bookkeeping (``valid_len`` / ``next_id`` / eviction).
        # Under torch.compile these side effects are traced once and never
        # replayed, so ``valid_len`` freezes and the cache silently stops
        # growing. Forcing eager makes the mutations persist across calls.
        cache = self.attn_caches[cache_name]
        if not is_pred and int(cache.get("pred_len", 0)) > 0:
            self.clear_pred_cache(cache_name)
        capacity = int(cache["capacity"])
        key_size = int(key.shape[1])
        if key_size >= capacity:
            # Single write already saturates the window; tail-clip and drop
            # everything we currently hold so the new K/V occupies [0, capacity).
            key = key[:, -capacity:]
            value = value[:, -capacity:]
            key_size = capacity
            self._evict_oldest(cache, int(cache["valid_len"]))
        valid_len = int(cache["valid_len"])
        overflow = valid_len + key_size - capacity
        if overflow > 0:
            self._evict_oldest(cache, overflow)
            valid_len = int(cache["valid_len"])

        new_id = self._next_cache_id(cache_name)
        end = valid_len + key_size
        cache["k"][:, valid_len:end] = key
        cache["v"][:, valid_len:end] = value
        cache["id"][valid_len:end] = new_id
        cache["mask"][valid_len:end] = True
        cache["is_pred"][valid_len:end] = is_pred

        cache["valid_len"] = end
        cache["write_pos"] = end
        if is_pred:
            cache["pred_len"] = min(end, int(cache["pred_len"]) + key_size)
        else:
            cache["pred_len"] = 0
        return None

    @torch.compiler.disable
    def _stage_and_view(self, cache_name, key, value):
        """Eager-only zero-copy staging for the denoise hot path.

        Writes the current step's K/V into the pre-allocated headroom region
        ``[valid_len, valid_len + n)`` and returns a contiguous view spanning
        the committed history plus that staged step. ``valid_len`` is left
        untouched (the staged slot is transient and overwritten by the next
        denoise step / promoted by ``update_cache`` on commit), so no roll-back
        is needed. Keeping the in-place write in eager hands the compiled region
        a plain tensor view -- the same thing the commit path already feeds
        SDPA -- so it stays compile-safe while avoiding the O(history) copy a
        per-step ``torch.cat`` would incur.
        """
        cache = self.attn_caches[cache_name]
        valid_len = int(cache["valid_len"])
        key_size = int(key.shape[1])
        end = valid_len + key_size
        if end > cache["k"].shape[1]:
            # Headroom undersized (should not happen with create_empty_cache):
            # fall back to a functional concat so correctness never depends on
            # buffer sizing.
            if valid_len == 0:
                return key, value
            return (
                torch.cat([cache["k"][:, :valid_len], key], dim=1),
                torch.cat([cache["v"][:, :valid_len], value], dim=1),
            )
        cache["k"][:, valid_len:end] = key
        cache["v"][:, valid_len:end] = value
        return cache["k"][:, :end], cache["v"][:, :end]

    def forward(
        self,
        q,
        k,
        v,
        rotary_emb,
        update_cache=0,
        cache_name="pos",
    ):
        kv_cache = (
            self.attn_caches[cache_name]
            if (self.attn_caches is not None) and (cache_name in self.attn_caches)
            else None
        )

        query = self.norm_q(self.to_q(q)).unflatten(2, (self.heads, -1))
        if self.cross_attention_dim_head is not None:
            # Cross-attention: K/V come from the fixed text context (no rope,
            # no KV-cache), so reuse the per-episode projection cached by
            # ``cache_cross_kv`` and skip rope entirely.
            key, value = self._cross_kv(k, v, cache_name)
        else:
            key = self.norm_k(self.to_k(k)).unflatten(2, (self.heads, -1))
            value = self.to_v(v).unflatten(2, (self.heads, -1))
            if rotary_emb is not None:
                query = apply_wan_rotary_emb(query, rotary_emb)
                key = apply_wan_rotary_emb(key, rotary_emb)
        if kv_cache is not None and kv_cache["k"] is not None:
            if update_cache == 0:
                # Hot path (denoise steps): stage the current K/V into the cache
                # headroom in eager and feed SDPA a single contiguous slice --
                # zero torch.cat copy of the history, zero in-graph cache
                # mutation. This is the dominant cost in the action denoise loop
                # (30 blocks x ~11 steps), so avoiding the per-step history copy
                # is what keeps the compiled path faster than eager.
                key, value = self._stage_and_view(cache_name, key, value)
            else:
                # Commit path: ``update_cache`` is compile-disabled, so its
                # dict/int mutations run eager and survive across calls.
                self.update_cache(cache_name, key, value, is_pred=(update_cache == 1))
                key, value = self._get_cached_kv(cache_name)
                if key is None or value is None:
                    raise RuntimeError("KV cache is empty after cache update.")

        if sequence_parallel_is_enabled() and self.cross_attention_dim_head is None:
            if kv_cache is not None:
                raise RuntimeError(
                    "Sequence parallel is supported only for train-time self-attention "
                    "without KV cache."
                )
            hidden_states = sequence_parallel_self_attention(
                self.attn_op,
                query,
                key,
                value,
            )
        else:
            hidden_states = self.attn_op(query, key, value)

        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)
        hidden_states = self.to_out[0](hidden_states)
        hidden_states = self.to_out[1](hidden_states)
        return hidden_states


class WanTransformerBlock(nn.Module):
    def __init__(
        self,
        dim,
        ffn_dim,
        num_heads,
        cross_attn_norm=False,
        eps=1e-6,
        attn_mode: str = "flashattn",
    ):
        super().__init__()
        self.attn_mode = attn_mode

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=None,
            attn_mode=attn_mode,
        )

        # 2. Cross-attention
        self.attn2 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=dim // num_heads,
            attn_mode=attn_mode,
        )
        self.norm2 = (
            FP32LayerNorm(dim, eps, elementwise_affine=True)
            if cross_attn_norm
            else nn.Identity()
        )

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def forward(
        self,
        hidden_states,
        encoder_hidden_states,
        temb,
        rotary_emb,
        update_cache=0,
        cache_name="pos",
    ) -> torch.Tensor:
        (
            shift_msa,
            scale_msa,
            gate_msa,
            c_shift_msa,
            c_scale_msa,
            c_gate_msa,
        ) = split_wan_scale_shift(self.scale_shift_table, temb)
        # 1. Self-attention
        norm_hidden_states = apply_wan_modulation(
            self.norm1(hidden_states.float()),
            shift_msa,
            scale_msa,
        ).type_as(hidden_states)
        attn_output = self.attn1(
            norm_hidden_states,
            norm_hidden_states,
            norm_hidden_states,
            rotary_emb,
            update_cache=update_cache,
            cache_name=cache_name,
        )
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(
            hidden_states
        )

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(
            norm_hidden_states,
            encoder_hidden_states,
            encoder_hidden_states,
            None,
            update_cache=0,
            cache_name=cache_name,
        )
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = apply_wan_modulation(
            self.norm3(hidden_states.float()),
            c_shift_msa,
            c_scale_msa,
        ).type_as(hidden_states)

        ff_output = self.ffn(norm_hidden_states)

        hidden_states = (
            hidden_states.float() + ff_output.float() * c_gate_msa
        ).type_as(hidden_states)
        return hidden_states


class WanTransformer3DModel(ModelMixin, ConfigMixin):
    r"""
    TODO
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = [
        # "patch_embedding",
        "patch_embedding_mlp",
        "condition_embedder",
        "condition_embedder_action",
        "norm",
    ]
    _no_split_modules = ["WanTransformerBlock"]
    _keep_in_fp32_modules = [
        "time_embedder",
        "scale_shift_table",
        "scale_shift_table_action",
        "norm1",
        "action_norm1",
        "text_norm1",
        "norm2",
        "action_norm2",
        "text_norm2",
        "norm3",
        "action_norm3",
        "text_norm3",
        "action_embedder_missing",
    ]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]
    _repeated_blocks = ["WanTransformerBlock"]

    @register_to_config
    def __init__(
        self,
        patch_size=[1, 2, 2],
        num_attention_heads=24,
        attention_head_dim=128,
        in_channels=48,
        out_channels=48,
        action_dim=30,
        text_dim=4096,
        freq_dim=256,
        ffn_dim=14336,
        num_layers=30,
        cross_attn_norm=True,
        eps=1e-06,
        rope_max_seq_len=1024,
        pos_embed_seq_len=None,
        attn_mode="torch",
        flex_mask_block_size=64,
        action_condition_mode="inverse_dynamics",
        missing_action_filling_pad_unused=None,
        missing_action_filling_used_channels=None,
        missing_action_filling_gripper_channels=None,
    ):
        r"""
        TODO
        """
        super().__init__()
        if action_condition_mode not in {"inverse_dynamics", "fastwam"}:
            raise ValueError(
                "action_condition_mode must be 'inverse_dynamics' or 'fastwam', "
                f"got {action_condition_mode!r}"
            )
        FlexAttnFunc.set_mask_block_size(flex_mask_block_size)
        self.patch_size = patch_size
        self.num_attention_heads = num_attention_heads
        self.attention_head_dim = attention_head_dim
        inner_dim = num_attention_heads * attention_head_dim
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding_mlp = nn.Linear(
            in_channels * patch_size[0] * patch_size[1] * patch_size[2], inner_dim
        )
        self.action_embedder = nn.Linear(action_dim, inner_dim)
        # missing_action_filling 注册 (cols / used / gripper / pad_unused + param + 4 buffers)。
        # 见 missing_action_filling.py；config 来自 YAML (datasets.vla_data.learnable_action_embedding.*)。
        _register_missing_action_filling(
            self,
            used_channels_arg=missing_action_filling_used_channels,
            gripper_channels_arg=missing_action_filling_gripper_channels,
            pad_unused_arg=missing_action_filling_pad_unused,
            action_dim=self.action_embedder.weight.shape[1],
            hidden_dim=self.action_embedder.weight.shape[0],
            param_name="action_embedder_missing",
        )
        # ===== /missing_action_filling =====
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            pos_embed_seq_len=pos_embed_seq_len,
        )
        self.condition_embedder_action = deepcopy(self.condition_embedder)

        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(
                    inner_dim,
                    ffn_dim,
                    num_attention_heads,
                    cross_attn_norm,
                    eps,
                    attn_mode=attn_mode,
                )
                for _ in range(num_layers)
            ]
        )

        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.action_proj_out = nn.Linear(inner_dim, action_dim)
        self.scale_shift_table = nn.Parameter(
            torch.randn(1, 2, inner_dim) / inner_dim**0.5
        )

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        """Delegate missing_action_filling config check, then run super()."""
        _check_missing_action_filling_config_buf(self, state_dict, prefix)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def clear_cache(self, cache_name):
        for block in self.blocks:
            block.attn1.clear_cache(cache_name)
            block.attn2.clear_cross_kv(cache_name)

    def compile_blocks(
        self, *, mode: str = "default", dynamic: bool | None = None
    ) -> None:
        """Compile the per-block compute hot loop in place.

        Thin wrapper over :func:`compile_transformer_blocks`; see it for the
        rationale, mode trade-offs, and fallback behavior.
        """
        compile_transformer_blocks(self.blocks, mode=mode, dynamic=dynamic)

    def cache_cross_kv(self, encoder_hidden_states, cache_name="pos"):
        """Pre-project the fixed text context into every block's cross-attention
        K/V so the denoise loop reuses them instead of recomputing to_k/to_v on
        the unchanging prompt every step."""
        for block in self.blocks:
            block.attn2.cache_cross_kv(encoder_hidden_states, cache_name)

    def clear_pred_cache(self, cache_name):
        for block in self.blocks:
            block.attn1.clear_pred_cache(cache_name)

    def create_empty_cache(
        self,
        cache_name,
        attn_window,
        latent_token_per_chunk,
        action_token_per_chunk,
        device,
        dtype,
        batch_size,
    ):
        total_tolen = (attn_window // 2) * latent_token_per_chunk + (
            attn_window // 2
        ) * action_token_per_chunk
        # Headroom holds at most one stream's per-call K/V staged on top of a
        # full cache by ``_stage_and_view`` so the SDPA hot path reads a single
        # contiguous slice without copying the history via ``torch.cat``.
        headroom = max(int(latent_token_per_chunk), int(action_token_per_chunk))
        for block in self.blocks:
            block.attn1.init_kv_cache(
                cache_name,
                total_tolen,
                self.num_attention_heads,
                self.attention_head_dim,
                device,
                dtype,
                batch_size,
                headroom=headroom,
            )

    def _input_embed(self, latents, input_type="latent"):
        return embed_lingbot_inputs(
            latents,
            input_type=input_type,
            patch_size=self.patch_size,
            patch_embedding_mlp=self.patch_embedding_mlp,
            action_embedder=self.action_embedder,
            text_embedder=self.condition_embedder.text_embedder,
        )

    def _input_embed_multiview_latents(self, latents_by_view):
        return torch.cat(
            [
                self._input_embed(latent, input_type="latent")
                for latent in latents_by_view
            ],
            dim=1,
        )

    def _time_embed(self, timesteps, H, W, dtype, action_mode=False):
        return embed_lingbot_timesteps(
            timesteps,
            H,
            W,
            dtype,
            patch_size=self.patch_size,
            condition_embedder=self.condition_embedder,
            condition_embedder_action=self.condition_embedder_action,
            action_mode=action_mode,
        )

    def _time_embed_multiview_latents(
        self,
        timesteps: torch.Tensor,
        cond_timesteps: torch.Tensor,
        view_shapes: list[tuple],
        dtype,
    ):
        token_timestep_groups = []
        for current_timesteps in (timesteps, cond_timesteps):
            current_view_timesteps = []
            for view_shape in view_shapes:
                _, _, _, height, width = view_shape
                repeats = (height // self.patch_size[1]) * (width // self.patch_size[2])
                current_view_timesteps.append(
                    torch.repeat_interleave(current_timesteps, repeats, dim=1)
                )
            token_timestep_groups.append(
                torch.cat(current_view_timesteps, dim=1).flatten(0, 1)
            )
        latent_time_steps = torch.cat(token_timestep_groups)[None]
        temb, timestep_proj = self.condition_embedder(latent_time_steps, dtype=dtype)
        timestep_proj = timestep_proj.unflatten(2, (6, -1))
        return temb, timestep_proj

    def _embed_action_with_missing(
        self,
        action_input,
        action_validity,
        width_validity,
        action_validity_per_frame=None,
        width_validity_per_frame=None,
        channel_validity_per_frame=None,
        missing_action_filling_enabled=None,
        used_channel_mask=None,
        gripper_channel_mask=None,
        pad_unused_mask=None,
    ):
        """Shared-backbone wrapper over `_embed_action_with_missing_filling`，用 action_embedder
        这套 linear 权重。公式 / v-split / Args / Returns 见 `_embed_action_with_missing_filling`
        (missing_action_filling.py) 的 docstring，这里不重复。"""
        # Thin wrapper over the shared core: pass THIS backbone's normal linear
        # (action_embedder) weight/bias + missing_action_filling param (action_embedder_missing) + the
        # channel sets resolved once in __init__. The shared backbone has no further encoder
        # layers, so the core's output IS the embedded action — return it directly.
        # WHY this is a wrapper now: issue #6 dedup — the validity-grid build + v/(1-v) split
        # + matmul used to be duplicated verbatim here and in model_mot.py; both now call the
        # one module-level `_embed_action_with_missing_filling`, differing only in which linear
        # they feed in and (MoT only) the post-linear encoder stack.
        return _embed_action_with_missing_filling(
            action_input,
            action_validity,
            width_validity,
            action_validity_per_frame,
            width_validity_per_frame,
            channel_validity_per_frame,
            missing_action_filling_enabled,
            w_normal_weight=self.action_embedder.weight,
            w_normal_bias=self.action_embedder.bias,
            w_missing_param=self.action_embedder_missing,
            used_channels=self._missing_action_filling_used_channels,
            gripper_channels=self._missing_action_filling_gripper_channels,
            pad_unused=getattr(self, "_missing_action_filling_pad_unused", False),
            used_channel_mask=used_channel_mask,
            gripper_channel_mask=gripper_channel_mask,
            pad_unused_mask=pad_unused_mask,
        )

    def forward_train(self, input_dict):
        latent_dict = input_dict["latent_dict"]
        action_dict = input_dict["action_dict"]
        include_video_prediction = not bool(
            input_dict.get("skip_video_prediction", False)
        )

        is_multiview_latent = "noisy_latents_by_view" in latent_dict
        if is_multiview_latent:
            latent_noisy_by_view = [
                latent.to(torch.bfloat16)
                for latent in latent_dict["noisy_latents_by_view"]
            ]
            latent_clean_by_view = [
                latent.to(torch.bfloat16) for latent in latent_dict["latent_by_view"]
            ]
            latent_shape = latent_noisy_by_view[0].shape
            latent_view_shapes = [
                tuple(latent.shape) for latent in latent_noisy_by_view
            ]
        else:
            latent_noisy = latent_dict["noisy_latents"].to(torch.bfloat16)
            latent_clean = latent_dict["latent"].to(torch.bfloat16)
            latent_shape = latent_noisy.shape
            latent_view_shapes = None

        action_noisy = action_dict["noisy_latents"].to(torch.bfloat16)
        action_clean = action_dict["latent"].to(torch.bfloat16)
        batch_size = latent_shape[0]

        if is_multiview_latent:
            latent_hidden_states = self._input_embed_multiview_latents(
                latent_noisy_by_view
            ).flatten(0, 1)[None]
            condition_latent_hidden_states = self._input_embed_multiview_latents(
                latent_clean_by_view
            ).flatten(0, 1)[None]
        else:
            latent_hidden_states = self._input_embed(
                latent_noisy, input_type="latent"
            ).flatten(0, 1)[None]
            condition_latent_hidden_states = self._input_embed(
                latent_clean, input_type="latent"
            ).flatten(0, 1)[None]
        _action_validity = action_dict.get("action_validity")
        _width_validity = action_dict.get("width_validity")
        _action_validity_pf = action_dict.get("action_validity_per_frame")
        _width_validity_pf = action_dict.get("width_validity_per_frame")
        _channel_validity_pf = action_dict.get("channel_validity_per_frame")
        _missing_action_filling_enabled = action_dict.get(
            "learnable_action_embedding_enabled"
        )
        _missing_action_filling_used_mask = action_dict.get(
            "learnable_action_embedding_used_mask"
        )
        _missing_action_filling_gripper_mask = action_dict.get(
            "learnable_action_embedding_gripper_mask"
        )
        _missing_action_filling_pad_unused = action_dict.get(
            "learnable_action_embedding_pad_unused"
        )
        _cond_action_validity = action_dict.get(
            "condition_action_validity",
            _action_validity,
        )
        _cond_width_validity = action_dict.get(
            "condition_width_validity",
            _width_validity,
        )
        _cond_action_validity_pf = action_dict.get(
            "condition_action_validity_per_frame",
            _action_validity_pf,
        )
        _cond_width_validity_pf = action_dict.get(
            "condition_width_validity_per_frame",
            _width_validity_pf,
        )
        _cond_channel_validity_pf = action_dict.get(
            "condition_channel_validity_per_frame",
            _channel_validity_pf,
        )
        _cond_missing_action_filling_enabled = action_dict.get(
            "condition_learnable_action_embedding_enabled",
            _missing_action_filling_enabled,
        )
        # Symmetric fail-loud guard for older/non-resident model variants. Current
        # models keep action_embedder_missing resident and use per-sample enabled
        # to decide whether validity streams produce a non-trivial mask.
        _assert_missing_action_filling_configured_for_per_frame(
            self.action_embedder_missing,
            _action_validity_pf,
            _width_validity_pf,
            _channel_validity_pf,
        )
        _assert_missing_action_filling_configured_for_per_frame(
            self.action_embedder_missing,
            _cond_action_validity_pf,
            _cond_width_validity_pf,
            _cond_channel_validity_pf,
        )

        _action_h = self._embed_action_with_missing(
            action_noisy,
            _action_validity,
            _width_validity,
            action_validity_per_frame=_action_validity_pf,
            width_validity_per_frame=_width_validity_pf,
            channel_validity_per_frame=_channel_validity_pf,
            missing_action_filling_enabled=_missing_action_filling_enabled,
            used_channel_mask=_missing_action_filling_used_mask,
            gripper_channel_mask=_missing_action_filling_gripper_mask,
            pad_unused_mask=_missing_action_filling_pad_unused,
        )
        # Add a zero-valued term that still touches missing_action_filling, so it stays in the
        # autograd graph even on batches where nothing was missing. This keeps
        # DeepSpeed's gradient communication consistent across steps.
        if self.action_embedder_missing is not None:
            _action_h = _action_h + (
                0.0 * self.action_embedder_missing.float().sum()
            ).to(_action_h.dtype)
        action_hidden_states = _action_h.flatten(0, 1)[None]
        text_emb = latent_dict.get("text_emb_by_piece", latent_dict["text_emb"])
        text_hidden_states = self._input_embed(text_emb, input_type="text")
        if text_hidden_states.ndim == 4:
            batch_text, piece_count, text_len, text_dim = text_hidden_states.shape
            text_hidden_states = text_hidden_states.reshape(
                1, batch_text * piece_count * text_len, text_dim
            )
            text_tokens_per_piece = int(text_len)
        else:
            text_hidden_states = text_hidden_states.flatten(0, 1)[None]
            text_tokens_per_piece = int(latent_dict["text_emb"].shape[-2])

        _cond_action_h = self._embed_action_with_missing(
            action_clean,
            _cond_action_validity,
            _cond_width_validity,
            action_validity_per_frame=_cond_action_validity_pf,
            width_validity_per_frame=_cond_width_validity_pf,
            channel_validity_per_frame=_cond_channel_validity_pf,
            missing_action_filling_enabled=_cond_missing_action_filling_enabled,
            used_channel_mask=_missing_action_filling_used_mask,
            gripper_channel_mask=_missing_action_filling_gripper_mask,
            pad_unused_mask=_missing_action_filling_pad_unused,
        )
        condition_action_hidden_states = _cond_action_h.flatten(0, 1)[None]

        hidden_state_parts = []
        if include_video_prediction:
            hidden_state_parts.append(latent_hidden_states)
        hidden_state_parts.extend(
            [
                condition_latent_hidden_states,
                action_hidden_states,
                condition_action_hidden_states,
            ]
        )
        hidden_states = torch.cat(hidden_state_parts, dim=1)

        piece_frame_ranges = latent_dict.get("text_emb_latent_frame_ranges")
        latent_base_grid_id = build_piece_local_grid_ids(
            latent_dict["grid_id"],
            piece_frame_ranges=piece_frame_ranges,
            query_shape_groups=build_latent_query_shape_groups(
                latent_shape,
                self.patch_size,
                latent_view_shapes=latent_view_shapes,
            ),
            temporal_patch_size=int(self.patch_size[0]),
        )
        action_base_grid_id = build_piece_local_grid_ids(
            action_dict["grid_id"],
            piece_frame_ranges=piece_frame_ranges,
            query_shape_groups=build_action_query_shape_groups(action_noisy.shape),
            temporal_patch_size=1,
            local_time_offsets_by_group=build_action_time_offsets(
                action_noisy.shape,
                device=action_dict["grid_id"].device,
                dtype=action_dict["grid_id"].dtype,
            ),
        )
        latent_grid_id = latent_base_grid_id.permute(1, 0, 2).flatten(1)[None]
        action_grid_id = action_base_grid_id.permute(1, 0, 2).flatten(1)[None]
        full_grid_id_parts = []
        if include_video_prediction:
            full_grid_id_parts.append(latent_grid_id)
        full_grid_id_parts.extend([latent_grid_id, action_grid_id, action_grid_id])
        full_grid_id = torch.cat(full_grid_id_parts, dim=2)

        rotary_emb = self.rope(full_grid_id)[:, :, None]

        if is_multiview_latent:
            latent_temb, latent_timestep_proj = self._time_embed_multiview_latents(
                latent_dict["timesteps"],
                latent_dict["cond_timesteps"],
                latent_view_shapes,
                dtype=hidden_states.dtype,
            )
        else:
            latent_time_steps = torch.cat(
                [
                    latent_dict["timesteps"].flatten(0, 1),
                    latent_dict["cond_timesteps"].flatten(0, 1),
                ]
            )[None]
            latent_temb, latent_timestep_proj = self._time_embed(
                latent_time_steps,
                latent_noisy.shape[-2],
                latent_noisy.shape[-1],
                dtype=hidden_states.dtype,
                action_mode=False,
            )
        action_time_steps = torch.cat(
            [
                action_dict["timesteps"].flatten(0, 1),
                action_dict["cond_timesteps"].flatten(0, 1),
            ]
        )[None]
        action_temb, action_timestep_proj = self._time_embed(
            action_time_steps,
            action_noisy.shape[-2],
            action_noisy.shape[-1],
            dtype=hidden_states.dtype,
            action_mode=True,
        )
        temb_parts = []
        timestep_proj_parts = []
        if include_video_prediction:
            temb_parts.append(latent_temb[:, : latent_hidden_states.shape[1]])
            timestep_proj_parts.append(
                latent_timestep_proj[:, : latent_hidden_states.shape[1]]
            )
        temb_parts.extend(
            [latent_temb[:, latent_hidden_states.shape[1] :], action_temb]
        )
        timestep_proj_parts.extend(
            [
                latent_timestep_proj[:, latent_hidden_states.shape[1] :],
                action_timestep_proj,
            ]
        )
        temb = torch.cat(temb_parts, dim=1)
        timestep_proj = torch.cat(timestep_proj_parts, dim=1)

        sp_enabled = sequence_parallel_is_enabled()
        sp_state = get_sequence_parallel_state() if sp_enabled else None
        sp_size = int(sp_state.size) if sp_state is not None else 1
        pad_alignment = math.lcm(128, sp_size)
        total_length = hidden_states.shape[1]
        padded_length = (pad_alignment - total_length % pad_alignment) % pad_alignment
        hidden_states = F.pad(hidden_states, (0, 0, 0, padded_length))
        rotary_emb = F.pad(rotary_emb, (0, 0, 0, 0, 0, padded_length))
        temb = F.pad(temb, (0, 0, 0, padded_length))
        timestep_proj = F.pad(timestep_proj, (0, 0, 0, 0, 0, padded_length))
        if sp_enabled and hidden_states.shape[1] % sp_size != 0:
            raise ValueError(
                "Sequence-parallel train forward requires padded sequence length "
                f"to be divisible by sequence_parallel_size: seq_len={hidden_states.shape[1]}, "
                f"sequence_parallel_size={sp_size}."
            )

        split_list = []
        if include_video_prediction:
            split_list.append(latent_hidden_states.shape[1])
        split_list.extend(
            [
                condition_latent_hidden_states.shape[1],
                action_hidden_states.shape[1],
                condition_action_hidden_states.shape[1],
                padded_length,
            ]
        )

        FlexAttnFunc.init_mask(
            latent_shape,
            action_noisy.shape,
            padded_length,
            input_dict["chunk_size"],
            window_size=input_dict["window_size"],
            patch_size=self.patch_size,
            device=hidden_states.device,
            action_condition_mode=self.config.action_condition_mode,
            latent_view_shapes=latent_view_shapes,
            piece_frame_ranges=latent_dict.get("text_emb_latent_frame_ranges"),
            attention_piece_frame_ranges=latent_dict.get(
                "attention_piece_frame_ranges"
            ),
            text_tokens_per_piece=text_tokens_per_piece,
            chunk_grouping_start_from_one=input_dict.get(
                "chunk_grouping_start_from_one", False
            ),
            include_video_prediction=include_video_prediction,
        )

        if sp_enabled:
            local_sequence_length = hidden_states.shape[1] // sp_size
            local_sequence_start = int(sp_state.rank) * local_sequence_length
            FlexAttnFunc.init_cross_attention_query_slice_mask(
                local_sequence_start,
                local_sequence_length,
                hidden_states.device,
            )
            hidden_states = split_sequence_parallel_region(hidden_states, dim=1)
            rotary_emb = split_sequence_parallel_region(rotary_emb, dim=1)
            temb = split_sequence_parallel_region(temb, dim=1)
            timestep_proj = split_sequence_parallel_region(timestep_proj, dim=1)

        for block in self.blocks:
            hidden_states = block(
                hidden_states,
                text_hidden_states,
                timestep_proj,
                rotary_emb,
                update_cache=False,
            )
        hidden_states = finalize_lingbot_outputs(
            hidden_states,
            temb,
            self.norm_out,
            self.scale_shift_table,
        )
        if sp_enabled:
            hidden_states = gather_sequence_parallel_region(hidden_states, dim=1)

        split_states = torch.split(hidden_states, split_list, dim=1)
        if include_video_prediction:
            latent_hidden_states = split_states[0]
            action_hidden_states = split_states[2]
            latent_hidden_states = self.proj_out(latent_hidden_states)
            latent_hidden_states = rearrange(
                latent_hidden_states,
                "1 (b l) (n c) -> b (l n) c",
                n=math.prod(self.patch_size),
                b=batch_size,
            )  #
        else:
            latent_hidden_states = None
            action_hidden_states = split_states[1]
        action_hidden_states = self.action_proj_out(action_hidden_states)
        action_hidden_states = rearrange(
            action_hidden_states, "1 (b l) c -> b l c", b=batch_size
        )  #

        return latent_hidden_states, action_hidden_states

    def forward(
        self,
        input_dict,
        update_cache=0,
        cache_name="pos",
        action_mode=False,
        train_mode=False,
    ):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """
        if train_mode:
            return self.forward_train(input_dict)

        pach_scale_h, pach_scale_w = (
            (1, 1) if action_mode else (self.patch_size[1], self.patch_size[2])
        )
        if action_mode:  # action input emb
            if input_dict.get("action_condition_all_missing", False):
                # Train/infer parity for a fully-dropped action-history condition.
                # Training's action_history_condition_dropout embeds the condition
                # action through the missing_action_filling path with an all-zero
                # channel-validity grid (v=0 everywhere), producing the learned
                # "all channels missing" embedding rather than the raw action. The
                # plain action_embedder can only ever yield W_normal @ x + bias and
                # cannot reproduce that learned missing term, so the zeroed condition
                # (e.g. zero_action_condition) must be routed through the same
                # missing path to match training instead of leaving a train/infer gap.
                if self.action_embedder_missing is None:
                    raise RuntimeError(
                        "action_condition_all_missing requires missing_action_filling "
                        "(action_embedder_missing) to be configured; the plain "
                        "action_embedder cannot reproduce the learned missing "
                        "embedding used during action_history_condition_dropout "
                        "training."
                    )
                noisy_action = input_dict["noisy_latents"]
                (
                    channel_validity_all_missing,
                    missing_action_filling_enabled,
                ) = _make_action_condition_all_missing_inputs(noisy_action)
                latent_hidden_states = self._embed_action_with_missing(
                    noisy_action,
                    None,
                    None,
                    channel_validity_per_frame=channel_validity_all_missing,
                    missing_action_filling_enabled=missing_action_filling_enabled,
                )  # B L1 C
            else:
                latent_hidden_states = rearrange(
                    input_dict["noisy_latents"], "b c f h w -> b (f h w) c"
                )
                latent_hidden_states = self.action_embedder(
                    latent_hidden_states
                )  # B L1 C
            latent_time_steps = torch.repeat_interleave(
                input_dict["timesteps"],
                (input_dict["noisy_latents"].shape[-2] // pach_scale_h)
                * (input_dict["noisy_latents"].shape[-1] // pach_scale_w),
                dim=1,
            )  # L
        else:  # latent input emb
            latent_hidden_states_list = []
            latent_time_steps_view_list = []
            noisy_latents_by_view = input_dict["noisy_latents"]
            if not isinstance(noisy_latents_by_view, list):
                noisy_latents_by_view = [noisy_latents_by_view]
            for noisy_latents in noisy_latents_by_view:
                latent_hidden_states_list.append(
                    self._input_embed(noisy_latents, input_type="latent")
                )
                _, _, _, latent_height, latent_width = noisy_latents.shape
                patch_height = latent_height // self.patch_size[1]
                patch_width = latent_width // self.patch_size[2]
                latent_time_steps_view = torch.repeat_interleave(
                    input_dict["timesteps"],
                    patch_height * patch_width,
                    dim=1,
                )
                latent_time_steps_view_list.append(latent_time_steps_view)
            latent_time_steps = torch.cat(latent_time_steps_view_list, dim=1)
            latent_hidden_states = torch.cat(latent_hidden_states_list, dim=1)
        text_hidden_states = input_dict.get("text_hidden_states")
        if text_hidden_states is None:
            text_hidden_states = self.condition_embedder.text_embedder(
                input_dict["text_emb"]
            )  # B L2 C

        latent_grid_id = input_dict["grid_id"]
        rotary_emb = self.rope(latent_grid_id)[:, :, None]  # 1 L 1 C
        current_condition_embedder = (
            self.condition_embedder_action if action_mode else self.condition_embedder
        )
        temb, timestep_proj = current_condition_embedder(
            latent_time_steps, dtype=latent_hidden_states.dtype
        )
        timestep_proj = timestep_proj.unflatten(2, (6, -1))  # B L 6 C

        for block in self.blocks:
            latent_hidden_states = block(
                latent_hidden_states,
                text_hidden_states,
                timestep_proj,
                rotary_emb,
                update_cache=update_cache,
                cache_name=cache_name,
            )
        latent_hidden_states = finalize_lingbot_outputs(
            latent_hidden_states,
            temb,
            self.norm_out,
            self.scale_shift_table,
        )

        if action_mode:
            latent_hidden_states = self.action_proj_out(latent_hidden_states)
        else:
            latent_hidden_states = self.proj_out(latent_hidden_states)
            latent_hidden_states = rearrange(
                latent_hidden_states,
                "b l (n c) -> b (l n) c",
                n=math.prod(self.patch_size),
            )  #

        return latent_hidden_states


if __name__ == "__main__":
    model = WanTransformer3DModel(
        patch_size=[1, 2, 2],
        num_attention_heads=24,
        attention_head_dim=128,
        in_channels=48,
        out_channels=48,
        action_dim=30,
        text_dim=4096,
        freq_dim=256,
        ffn_dim=14336,
        num_layers=30,
        cross_attn_norm=True,
        eps=1e-6,
        rope_max_seq_len=1024,
        pos_embed_seq_len=None,
        attn_mode="torch",
    )
    logger.info("%s", model)
