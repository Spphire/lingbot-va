from __future__ import annotations

import math
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.attention import FeedForward
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm
from einops import rearrange

from .distributed.sequence_parallel import (
    all_to_all_sequence_parallel,
    gather_sequence_parallel_region,
    get_sequence_parallel_state,
    sequence_parallel_is_enabled,
    split_sequence_parallel_region,
)

# missing_action_filling helpers 见 missing_action_filling.py（shared + MoT 共用同一套实现，各自喂
# 自己那套 linear 权重 + replace param）。
from .missing_action_filling import (
    _assert_missing_action_filling_configured_for_per_frame,
    _check_missing_action_filling_config_buf,
    _embed_action_with_missing_filling,
    _make_action_condition_all_missing_inputs,
    _register_missing_action_filling,
)
from .model import (
    FlexAttnFunc,
    WanRotaryPosEmbed,
    WanTimeTextImageEmbedding,
    _CrossKVCache,
    apply_wan_modulation,
    apply_wan_rotary_emb,
    build_action_query_shape_groups,
    build_action_time_offsets,
    build_joint_attention_mask,
    build_latent_query_shape_groups,
    build_piece_local_grid_ids,
    compile_transformer_blocks,
    embed_lingbot_inputs,
    embed_lingbot_timesteps,
    finalize_lingbot_outputs,
    split_wan_scale_shift,
)

__all__ = ["WanTransformer3DMoTModel"]
from logging import getLogger

logger = getLogger(__name__)


def _split_sequence_parallel_stream(
    tensor: torch.Tensor,
    *,
    stream_name: str,
) -> torch.Tensor:
    state = get_sequence_parallel_state()
    sp_size = int(state.size)
    if tensor.shape[1] % sp_size != 0:
        raise ValueError(
            "MoT sequence parallel requires each stream sequence length to be "
            f"divisible by sequence_parallel size: {stream_name} length={tensor.shape[1]}, "
            f"sequence_parallel_size={sp_size}."
        )
    return split_sequence_parallel_region(tensor, dim=1)


def _split_sequence_parallel_query_mask(
    mask: torch.Tensor | None,
    *,
    stream_name: str,
) -> torch.Tensor | None:
    if mask is None or mask.ndim != 3:
        return mask
    return _split_sequence_parallel_stream(mask, stream_name=stream_name)


def _interpolate_axis(tensor: torch.Tensor, target_size: int, dim: int) -> torch.Tensor:
    current_size = tensor.shape[dim]
    if current_size == target_size:
        return tensor

    perm = [idx for idx in range(tensor.ndim) if idx != dim] + [dim]
    inv_perm = [0] * tensor.ndim
    for idx, value in enumerate(perm):
        inv_perm[value] = idx

    x = tensor.permute(*perm).contiguous()
    flat = x.reshape(-1, current_size).unsqueeze(1)
    flat = F.interpolate(
        flat.float(),
        size=target_size,
        mode="linear",
        align_corners=True,
    )
    x = flat.squeeze(1).reshape(*x.shape[:-1], target_size)
    return x.permute(*inv_perm).contiguous().to(dtype=tensor.dtype)


def _resize_tensor_like(source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    out = source.detach().to(dtype=torch.float32)
    for dim, target_size in enumerate(target.shape):
        out = _interpolate_axis(out, int(target_size), dim)
    return out.to(dtype=target.dtype)


def _copy_linear(
    source: nn.Linear,
    target: nn.Linear,
    *,
    alpha: float = 1.0,
) -> None:
    with torch.no_grad():
        target.weight.copy_(_resize_tensor_like(source.weight, target.weight) * alpha)
        if source.bias is not None and target.bias is not None:
            target.bias.copy_(_resize_tensor_like(source.bias, target.bias))


def _copy_norm(source: nn.Module, target: nn.Module) -> None:
    source_state = source.state_dict()
    target_state = target.state_dict()
    merged = {}
    for key, target_value in target_state.items():
        source_value = source_state[key]
        merged[key] = _resize_tensor_like(source_value, target_value)
    target.load_state_dict(merged, strict=True)


def _copy_module_with_resize(
    source: nn.Module,
    target: nn.Module,
    *,
    alpha: float = 1.0,
    scale_weight_like_params: bool = True,
) -> None:
    source_state = source.state_dict()
    target_state = target.state_dict()
    merged = {}
    for key, target_value in target_state.items():
        source_value = source_state[key]
        resized = _resize_tensor_like(source_value, target_value)
        if (
            scale_weight_like_params
            and target_value.ndim >= 2
            and key.endswith("weight")
        ):
            resized = resized * alpha
        if scale_weight_like_params and key.endswith("modulation"):
            resized = resized * alpha
        merged[key] = resized
    target.load_state_dict(merged, strict=True)


def _init_linear_xavier(linear: nn.Linear) -> None:
    nn.init.xavier_uniform_(linear.weight)
    if linear.bias is not None:
        nn.init.zeros_(linear.bias)


def build_piece_text_context_mask(
    *,
    piece_count: torch.Tensor | int,
    latent_frame_ranges: torch.Tensor,
    query_shape_groups: list[tuple[int, int]],
    text_len: int,
    context_len: int,
    temporal_patch_size: int,
    device,
) -> torch.Tensor:
    if isinstance(piece_count, int):
        batch_size = int(latent_frame_ranges.shape[0])
        piece_count = torch.full(
            (batch_size,),
            int(piece_count),
            dtype=torch.long,
            device=latent_frame_ranges.device,
        )
    else:
        piece_count = piece_count.to(dtype=torch.long)
        if piece_count.ndim == 0:
            piece_count = piece_count.reshape(1)

    latent_frame_ranges = latent_frame_ranges.to(dtype=torch.long)
    if latent_frame_ranges.ndim == 2:
        latent_frame_ranges = latent_frame_ranges.unsqueeze(0)
    if latent_frame_ranges.ndim != 3 or latent_frame_ranges.shape[-1] != 2:
        raise ValueError(
            "`text_emb_latent_frame_ranges` must have shape [B, piece_count, 2], "
            f"got {tuple(latent_frame_ranges.shape)}."
        )

    batch_size, max_piece_count, _ = latent_frame_ranges.shape
    if piece_count.shape[0] == 1 and batch_size > 1:
        piece_count = piece_count.expand(batch_size)
    if piece_count.shape[0] != batch_size:
        raise ValueError(
            "`text_emb_piece_count` batch size does not match latent frame ranges: "
            f"{piece_count.shape[0]} vs {batch_size}."
        )

    query_len = sum(
        frame_count * tokens_per_frame
        for frame_count, tokens_per_frame in query_shape_groups
    )
    mask = torch.zeros(
        (batch_size, query_len, context_len),
        dtype=torch.bool,
        device=device,
    )
    temporal_patch_size = max(1, int(temporal_patch_size))
    text_len = int(text_len)

    for batch_idx in range(batch_size):
        valid_piece_count = min(int(piece_count[batch_idx].item()), max_piece_count)
        for piece_idx in range(valid_piece_count):
            start_frame = int(latent_frame_ranges[batch_idx, piece_idx, 0].item())
            end_frame = int(latent_frame_ranges[batch_idx, piece_idx, 1].item())
            text_start = piece_idx * text_len
            text_end = min(text_start + text_len, context_len)
            if end_frame <= start_frame or text_start >= context_len:
                continue

            query_offset = 0
            for frame_count, tokens_per_frame in query_shape_groups:
                patched_frame_count = int(frame_count)
                start_patch = max(0, start_frame // temporal_patch_size)
                end_patch = min(
                    patched_frame_count,
                    math.ceil(end_frame / temporal_patch_size),
                )
                if end_patch > start_patch:
                    query_start = query_offset + start_patch * int(tokens_per_frame)
                    query_end = query_offset + end_patch * int(tokens_per_frame)
                    mask[batch_idx, query_start:query_end, text_start:text_end] = True
                query_offset += patched_frame_count * int(tokens_per_frame)

    return mask


class LingBotVAAttention(nn.Module):
    def __init__(
        self,
        in_dim: int,
        attn_dim: int,
        out_dim: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.to_q = nn.Linear(in_dim, attn_dim, bias=True)
        self.to_k = nn.Linear(in_dim, attn_dim, bias=True)
        self.to_v = nn.Linear(in_dim, attn_dim, bias=True)
        self.to_out = nn.ModuleList(
            [nn.Linear(attn_dim, out_dim, bias=True), nn.Dropout(0.0)]
        )
        self.norm_q = torch.nn.RMSNorm(attn_dim, eps=eps, elementwise_affine=True)
        self.norm_k = torch.nn.RMSNorm(attn_dim, eps=eps, elementwise_affine=True)

    def project_qkv(
        self,
        hidden_states: torch.Tensor,
        rotary_emb: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        query = self.norm_q(self.to_q(hidden_states))
        key = self.norm_k(self.to_k(hidden_states))
        value = self.to_v(hidden_states)
        if rotary_emb is not None:
            query = apply_wan_rotary_emb(
                query.unflatten(
                    2, (query.shape[2] // rotary_emb.shape[-1] // 2 * 2, -1)
                ),
                rotary_emb,
            ).flatten(2)
            key = apply_wan_rotary_emb(
                key.unflatten(2, (key.shape[2] // rotary_emb.shape[-1] // 2 * 2, -1)),
                rotary_emb,
            ).flatten(2)
        return query, key, value

    def out_proj(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.to_out[0](hidden_states)
        return self.to_out[1](hidden_states)


class LingBotVACrossAttention(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        context_dim: int,
        num_heads: int,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.attn_dim = hidden_dim
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"`hidden_dim` must be divisible by `num_heads`, got {hidden_dim} and {num_heads}"
            )
        self.head_dim = hidden_dim // num_heads

        self.to_q = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.to_k = nn.Linear(context_dim, hidden_dim, bias=True)
        self.to_v = nn.Linear(context_dim, hidden_dim, bias=True)
        self.to_out = nn.ModuleList(
            [nn.Linear(hidden_dim, hidden_dim, bias=True), nn.Dropout(0.0)]
        )
        self.norm_q = torch.nn.RMSNorm(hidden_dim, eps=eps, elementwise_affine=True)
        self.norm_k = torch.nn.RMSNorm(hidden_dim, eps=eps, elementwise_affine=True)
        # Cross-attention K/V depend only on the per-episode fixed text context.
        # Pre-projected once via ``cache_cross_kv`` and reused by ``forward``.
        self._cross_kv_cache: dict[str, _CrossKVCache] = {}

    def _project_cross_kv(
        self,
        context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        key = rearrange(
            self.norm_k(self.to_k(context)), "b s (h d) -> b h s d", h=self.num_heads
        )
        value = rearrange(self.to_v(context), "b s (h d) -> b h s d", h=self.num_heads)
        return key, value

    @torch.no_grad()
    def cache_cross_kv(self, context: torch.Tensor, cache_name: str = "pos") -> None:
        """Pre-project the fixed text context into cross-attention K/V.

        Overwrites any existing entry, so it is safe to call on every episode
        reset.
        """
        key, value = self._project_cross_kv(context)
        self._cross_kv_cache[cache_name] = _CrossKVCache(context, key, value)

    def clear_cross_kv(self, cache_name: str | None = None) -> None:
        """Drop cached cross-attention K/V (one ``cache_name``, or all)."""
        if cache_name is None:
            self._cross_kv_cache.clear()
        else:
            self._cross_kv_cache.pop(cache_name, None)

    def _cross_kv(
        self,
        context: torch.Tensor,
        cache_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Reuse the cached projection when it matches the current context,
        otherwise project on the fly."""
        entry = self._cross_kv_cache.get(cache_name)
        if entry is not None and entry.context is context:
            return entry.key, entry.value
        return self._project_cross_kv(context)

    def forward(
        self,
        hidden_states: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None = None,
        *,
        cache_name: str = "pos",
    ) -> torch.Tensor:
        query = self.norm_q(self.to_q(hidden_states))
        query = rearrange(query, "b s (h d) -> b h s d", h=self.num_heads)
        key, value = self._cross_kv(context, cache_name)

        attn_mask = None
        if context_mask is not None:
            if context_mask.ndim == 2:
                attn_mask = context_mask[:, None, None, :].to(dtype=torch.bool)
            elif context_mask.ndim == 3:
                attn_mask = context_mask[:, None, :, :].to(dtype=torch.bool)
            else:
                raise ValueError(
                    "`context_mask` must have shape [B, context_len] or "
                    f"[B, query_len, context_len], got {tuple(context_mask.shape)}."
                )

        hidden_states = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
        )
        hidden_states = rearrange(hidden_states, "b h s d -> b s (h d)")
        hidden_states = self.to_out[0](hidden_states)
        return self.to_out[1](hidden_states)


class LingBotVAMoTBlock(nn.Module):
    def __init__(
        self,
        video_dim: int,
        action_dim: int,
        mixed_attn_dim: int,
        num_heads: int,
        video_ffn_dim: int,
        action_ffn_dim: int,
        text_dim: int,
        eps: float,
        attn_mode: str,
    ):
        super().__init__()
        self.num_heads = int(num_heads)
        self.attn_mode = str(attn_mode)
        if mixed_attn_dim % self.num_heads != 0:
            raise ValueError(
                f"`mixed_attn_dim` must be divisible by `num_heads`, got {mixed_attn_dim} and {num_heads}"
            )
        self.mixed_head_dim = mixed_attn_dim // self.num_heads
        if self.attn_mode == "flex":
            self.mixed_attn = FlexAttnFunc(is_cross=False)
        elif self.attn_mode in {"torch", "flashattn"}:
            # MoT mixed attention needs an arbitrary joint mask; fall back to SDPA for
            # non-flex modes because flash-attn does not support this masking pattern.
            self.mixed_attn = None
        else:
            raise ValueError(f"Unsupported MoT attention mode: {self.attn_mode!r}")

        self.video_norm1 = FP32LayerNorm(video_dim, eps, elementwise_affine=False)
        self.video_attn1 = LingBotVAAttention(video_dim, mixed_attn_dim, video_dim, eps)
        self.video_norm2 = FP32LayerNorm(video_dim, eps, elementwise_affine=True)
        self.video_attn2 = LingBotVACrossAttention(video_dim, video_dim, num_heads, eps)
        self.video_norm3 = FP32LayerNorm(video_dim, eps, elementwise_affine=False)
        self.video_ffn = FeedForward(
            video_dim, inner_dim=video_ffn_dim, activation_fn="gelu-approximate"
        )
        self.video_scale_shift_table = nn.Parameter(
            torch.randn(1, 6, video_dim) / video_dim**0.5
        )

        self.action_norm1 = FP32LayerNorm(action_dim, eps, elementwise_affine=False)
        self.action_attn1 = LingBotVAAttention(
            action_dim, mixed_attn_dim, action_dim, eps
        )
        self.action_norm2 = FP32LayerNorm(action_dim, eps, elementwise_affine=True)
        self.action_attn2 = LingBotVACrossAttention(
            action_dim, action_dim, num_heads, eps
        )
        self.action_norm3 = FP32LayerNorm(action_dim, eps, elementwise_affine=False)
        self.action_ffn = FeedForward(
            action_dim, inner_dim=action_ffn_dim, activation_fn="gelu-approximate"
        )
        self.action_scale_shift_table = nn.Parameter(
            torch.randn(1, 6, action_dim) / action_dim**0.5
        )
        self.attn_caches: dict[str, dict | None] = {}

    def init_kv_cache(
        self,
        cache_name: str,
        total_tokens: int,
        mixed_attn_dim: int,
        device,
        dtype,
        batch_size: int,
        headroom: int = 0,
    ) -> None:
        """Allocate a linear-addressing KV cache for ``cache_name``.

        Valid tokens always live at slots ``[0, valid_len)`` and ``valid_len``
        never exceeds ``capacity`` (older tokens are evicted in
        ``update_cache``). The physical buffer is sized ``capacity + headroom``
        so the denoise hot path can stage the current step's K/V into the
        headroom region (``_stage_and_view``) and hand SDPA a single contiguous
        slice with no ``torch.cat`` copy of the history.
        """
        capacity = int(total_tokens)
        physical_len = capacity + int(headroom)
        self.attn_caches[cache_name] = {
            "k": torch.empty(
                [batch_size, physical_len, mixed_attn_dim],
                device=device,
                dtype=dtype,
            ),
            "v": torch.empty(
                [batch_size, physical_len, mixed_attn_dim],
                device=device,
                dtype=dtype,
            ),
            "id": torch.full((physical_len,), -1, device=device, dtype=torch.long),
            "mask": torch.zeros((physical_len,), dtype=torch.bool, device=device),
            "is_pred": torch.zeros((physical_len,), dtype=torch.bool, device=device),
            "capacity": capacity,
            "headroom": int(headroom),
            "write_pos": 0,
            "valid_len": 0,
            "pred_len": 0,
            "next_id": 0,
        }

    def clear_cache(self, cache_name: str) -> None:
        self.attn_caches[cache_name] = None
        self.video_attn2.clear_cross_kv(cache_name)
        self.action_attn2.clear_cross_kv(cache_name)

    @torch.no_grad()
    def cache_cross_kv(
        self,
        video_context: torch.Tensor,
        action_context: torch.Tensor | None = None,
        cache_name: str = "pos",
    ) -> None:
        """Pre-project the fixed text context into the cross-attentions.

        The video and action streams attend to separately-embedded text (with
        different feature dims), so each caches its own context. The action
        cross-attention is cached only when ``action_context`` is supplied;
        otherwise that stream falls back to on-the-fly projection.
        """
        self.video_attn2.cache_cross_kv(video_context, cache_name)
        if action_context is not None:
            self.action_attn2.cache_cross_kv(action_context, cache_name)

    def clear_pred_cache(self, cache_name: str) -> None:
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

    @torch.compiler.disable
    def _get_cached_kv(
        self,
        cache_name: str,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        # Eager-only: reads the Python ``valid_len`` and slices the cache buffer
        # so the compiled region receives a concrete history tensor (or None)
        # each call, with no in-graph cache state. See model.py for rationale.
        cache = self.attn_caches.get(cache_name)
        if cache is None:
            return None, None
        valid_len = int(cache.get("valid_len", 0))
        if valid_len == 0:
            return None, None
        return cache["k"][:, :valid_len], cache["v"][:, :valid_len]

    def _next_cache_id(self, cache_name: str) -> int:
        cache = self.attn_caches[cache_name]
        cache_id = int(cache["next_id"])
        cache["next_id"] = cache_id + 1
        return cache_id

    def _evict_oldest(self, cache: dict, evict_len: int) -> None:
        """In-place left shift to drop the oldest ``evict_len`` valid tokens.

        Source and destination ranges overlap so we materialise the keep
        region via ``.clone()`` before the write-back. This is invoked at most
        a handful of times per chunk (last denoise steps + compute_kv_cache
        refresh) so the O(capacity * D) cost is amortised away from the
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
    def update_cache(
        self,
        cache_name: str,
        key: torch.Tensor,
        value: torch.Tensor,
        *,
        is_pred: bool,
    ) -> None:
        # Eager-only: the commit path mutates plain-dict tensors in place and
        # advances Python bookkeeping (``valid_len`` / ``next_id`` / eviction).
        # Under torch.compile these side effects are traced once and never
        # replayed, freezing ``valid_len``; forcing eager makes them persist.
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

    @torch.compiler.disable
    def _stage_and_view(
        self,
        cache_name: str,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Eager-only zero-copy staging for the denoise hot path.

        Writes the current step's K/V into the pre-allocated headroom region
        ``[valid_len, valid_len + n)`` and returns a contiguous view spanning
        the committed history plus that staged step. ``valid_len`` is left
        untouched (the staged slot is transient and overwritten by the next
        denoise step / promoted by ``update_cache`` on commit), so no roll-back
        is needed. Keeping the in-place write in eager hands the compiled region
        a plain tensor view -- the same thing the commit path already feeds
        SDPA -- so it stays compile-safe while avoiding the O(history) copy a
        per-step ``torch.cat`` would incur. See model.py for the full rationale.
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

    def _mixed_attention_single(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
    ) -> torch.Tensor:
        query = rearrange(
            query, "b s (h d) -> b h s d", h=self.num_heads, d=self.mixed_head_dim
        )
        key = rearrange(
            key, "b s (h d) -> b h s d", h=self.num_heads, d=self.mixed_head_dim
        )
        value = rearrange(
            value, "b s (h d) -> b h s d", h=self.num_heads, d=self.mixed_head_dim
        )
        mixed = F.scaled_dot_product_attention(
            query,
            key,
            value,
        )
        return rearrange(mixed, "b h s d -> b s (h d)")

    def forward_single_stream(
        self,
        hidden_states: torch.Tensor,
        context: torch.Tensor,
        context_mask: torch.Tensor | None,
        t_mod: torch.Tensor,
        rotary_emb: torch.Tensor,
        *,
        stream_name: str,
        update_cache: int = 0,
        cache_name: str = "pos",
    ) -> torch.Tensor:
        if stream_name == "video":
            (
                shift_msa,
                scale_msa,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
            ) = split_wan_scale_shift(self.video_scale_shift_table, t_mod)
            attn_input = apply_wan_modulation(
                self.video_norm1(hidden_states.float()),
                shift_msa,
                scale_msa,
            ).type_as(hidden_states)
            attn_proj = self.video_attn1
            cross_norm = self.video_norm2
            cross_attn = self.video_attn2
            ffn_norm = self.video_norm3
            ffn = self.video_ffn
        elif stream_name == "action":
            (
                shift_msa,
                scale_msa,
                gate_msa,
                shift_mlp,
                scale_mlp,
                gate_mlp,
            ) = split_wan_scale_shift(self.action_scale_shift_table, t_mod)
            attn_input = apply_wan_modulation(
                self.action_norm1(hidden_states.float()),
                shift_msa,
                scale_msa,
            ).type_as(hidden_states)
            attn_proj = self.action_attn1
            cross_norm = self.action_norm2
            cross_attn = self.action_attn2
            ffn_norm = self.action_norm3
            ffn = self.action_ffn
        else:
            raise ValueError(f"Unsupported stream_name: {stream_name!r}")

        query = attn_proj.norm_q(attn_proj.to_q(attn_input))
        key = attn_proj.norm_k(attn_proj.to_k(attn_input))
        value = attn_proj.to_v(attn_input)
        query = apply_wan_rotary_emb(
            query.unflatten(2, (self.num_heads, self.mixed_head_dim)),
            rotary_emb,
        ).flatten(2)
        key = apply_wan_rotary_emb(
            key.unflatten(2, (self.num_heads, self.mixed_head_dim)),
            rotary_emb,
        ).flatten(2)

        kv_cache = self.attn_caches.get(cache_name)
        if kv_cache is None:
            key_pool, value_pool = key, value
        elif update_cache == 0:
            # Hot path (denoise steps): stage the current K/V into the cache
            # headroom in eager and feed SDPA a single contiguous slice -- zero
            # torch.cat copy of the history, zero in-graph cache mutation. This
            # is the dominant cost in the action denoise loop, so avoiding the
            # per-step history copy is what keeps the compiled path fast.
            key_pool, value_pool = self._stage_and_view(cache_name, key, value)
        else:
            # Commit path: ``update_cache`` is compile-disabled, so its dict/int
            # mutations run eager and survive across calls.
            self.update_cache(cache_name, key, value, is_pred=(update_cache == 1))
            key_pool, value_pool = self._get_cached_kv(cache_name)
            if key_pool is None or value_pool is None:
                raise ValueError("MoT cache is empty after writing current key/value.")

        mixed = self._mixed_attention_single(
            query,
            key_pool,
            value_pool,
        )

        hidden_states = (
            hidden_states.float() + attn_proj.out_proj(mixed).float() * gate_msa.float()
        ).type_as(hidden_states)

        cross_input = cross_norm(hidden_states.float()).type_as(hidden_states)
        hidden_states = hidden_states + cross_attn(
            cross_input, context, context_mask, cache_name=cache_name
        )

        mlp_input = apply_wan_modulation(
            ffn_norm(hidden_states.float()),
            shift_mlp,
            scale_mlp,
        ).type_as(hidden_states)
        hidden_states = (
            hidden_states.float() + ffn(mlp_input).float() * gate_mlp.float()
        ).type_as(hidden_states)

        return hidden_states

    def _mixed_attention(
        self,
        video_q: torch.Tensor,
        video_k: torch.Tensor,
        video_v: torch.Tensor,
        action_q: torch.Tensor,
        action_k: torch.Tensor,
        action_v: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if sequence_parallel_is_enabled():
            return self._sequence_parallel_mixed_attention(
                video_q,
                video_k,
                video_v,
                action_q,
                action_k,
                action_v,
                attention_mask,
            )

        query = torch.cat([video_q, action_q], dim=1)
        key = torch.cat([video_k, action_k], dim=1)
        value = torch.cat([video_v, action_v], dim=1)

        query = query.unflatten(2, (self.num_heads, self.mixed_head_dim))
        key = key.unflatten(2, (self.num_heads, self.mixed_head_dim))
        value = value.unflatten(2, (self.num_heads, self.mixed_head_dim))
        mixed = self._run_mixed_attention(query, key, value, attention_mask).flatten(2)
        return torch.split(mixed, [video_q.shape[1], action_q.shape[1]], dim=1)

    def _run_mixed_attention(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if self.attn_mode == "flex":
            return self.mixed_attn(query, key, value)

        if attention_mask is None:
            raise ValueError(
                "MoT dense attention requires a precomputed attention_mask."
            )
        query = rearrange(query, "b s h d -> b h s d")
        key = rearrange(key, "b s h d -> b h s d")
        value = rearrange(value, "b s h d -> b h s d")
        mixed = F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attention_mask,
        )
        return rearrange(mixed, "b h s d -> b s h d")

    def _sequence_parallel_mixed_attention(
        self,
        video_q: torch.Tensor,
        video_k: torch.Tensor,
        video_v: torch.Tensor,
        action_q: torch.Tensor,
        action_k: torch.Tensor,
        action_v: torch.Tensor,
        attention_mask: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        state = get_sequence_parallel_state()
        sp_size = int(state.size)
        if self.num_heads % sp_size != 0:
            raise ValueError(
                "MoT sequence parallel requires num_attention_heads to be divisible "
                f"by sequence_parallel size: heads={self.num_heads}, "
                f"sequence_parallel_size={sp_size}."
            )

        def to_full_sequence_local_heads(tensor: torch.Tensor) -> torch.Tensor:
            tensor = tensor.unflatten(2, (self.num_heads, self.mixed_head_dim))
            return all_to_all_sequence_parallel(tensor, scatter_dim=2, gather_dim=1)

        video_q = to_full_sequence_local_heads(video_q)
        video_k = to_full_sequence_local_heads(video_k)
        video_v = to_full_sequence_local_heads(video_v)
        action_q = to_full_sequence_local_heads(action_q)
        action_k = to_full_sequence_local_heads(action_k)
        action_v = to_full_sequence_local_heads(action_v)

        video_length = int(video_q.shape[1])
        action_length = int(action_q.shape[1])
        query = torch.cat([video_q, action_q], dim=1)
        key = torch.cat([video_k, action_k], dim=1)
        value = torch.cat([video_v, action_v], dim=1)

        mixed = self._run_mixed_attention(query, key, value, attention_mask)
        video_mixed, action_mixed = torch.split(
            mixed,
            [video_length, action_length],
            dim=1,
        )

        video_mixed = all_to_all_sequence_parallel(
            video_mixed,
            scatter_dim=1,
            gather_dim=2,
        ).flatten(2)
        action_mixed = all_to_all_sequence_parallel(
            action_mixed,
            scatter_dim=1,
            gather_dim=2,
        ).flatten(2)
        return video_mixed, action_mixed

    def forward(
        self,
        video_states: torch.Tensor,
        action_states: torch.Tensor,
        video_context: torch.Tensor,
        action_context: torch.Tensor,
        context_mask: torch.Tensor | None,
        video_t_mod: torch.Tensor,
        action_t_mod: torch.Tensor,
        video_rotary_emb: torch.Tensor,
        action_rotary_emb: torch.Tensor,
        attention_mask: torch.Tensor | None,
        video_context_mask: torch.Tensor | None = None,
        action_context_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        (
            v_shift_msa,
            v_scale_msa,
            v_gate_msa,
            v_shift_mlp,
            v_scale_mlp,
            v_gate_mlp,
        ) = split_wan_scale_shift(self.video_scale_shift_table, video_t_mod)
        (
            a_shift_msa,
            a_scale_msa,
            a_gate_msa,
            a_shift_mlp,
            a_scale_mlp,
            a_gate_mlp,
        ) = split_wan_scale_shift(self.action_scale_shift_table, action_t_mod)

        video_attn_input = apply_wan_modulation(
            self.video_norm1(video_states.float()),
            v_shift_msa,
            v_scale_msa,
        ).type_as(video_states)
        action_attn_input = apply_wan_modulation(
            self.action_norm1(action_states.float()),
            a_shift_msa,
            a_scale_msa,
        ).type_as(action_states)

        video_q = self.video_attn1.norm_q(self.video_attn1.to_q(video_attn_input))
        video_k = self.video_attn1.norm_k(self.video_attn1.to_k(video_attn_input))
        video_v = self.video_attn1.to_v(video_attn_input)
        action_q = self.action_attn1.norm_q(self.action_attn1.to_q(action_attn_input))
        action_k = self.action_attn1.norm_k(self.action_attn1.to_k(action_attn_input))
        action_v = self.action_attn1.to_v(action_attn_input)

        video_q = apply_wan_rotary_emb(
            video_q.unflatten(2, (self.num_heads, self.mixed_head_dim)),
            video_rotary_emb,
        ).flatten(2)
        video_k = apply_wan_rotary_emb(
            video_k.unflatten(2, (self.num_heads, self.mixed_head_dim)),
            video_rotary_emb,
        ).flatten(2)
        action_q = apply_wan_rotary_emb(
            action_q.unflatten(2, (self.num_heads, self.mixed_head_dim)),
            action_rotary_emb,
        ).flatten(2)
        action_k = apply_wan_rotary_emb(
            action_k.unflatten(2, (self.num_heads, self.mixed_head_dim)),
            action_rotary_emb,
        ).flatten(2)

        video_mixed, action_mixed = self._mixed_attention(
            video_q,
            video_k,
            video_v,
            action_q,
            action_k,
            action_v,
            attention_mask,
        )

        video_states = (
            video_states.float()
            + self.video_attn1.out_proj(video_mixed).float() * v_gate_msa.float()
        ).type_as(video_states)
        action_states = (
            action_states.float()
            + self.action_attn1.out_proj(action_mixed).float() * a_gate_msa.float()
        ).type_as(action_states)

        video_cross_input = self.video_norm2(video_states.float()).type_as(video_states)
        video_states = video_states + self.video_attn2(
            video_cross_input,
            video_context,
            video_context_mask if video_context_mask is not None else context_mask,
        )
        action_cross_input = self.action_norm2(action_states.float()).type_as(
            action_states
        )
        action_states = action_states + self.action_attn2(
            action_cross_input,
            action_context,
            action_context_mask if action_context_mask is not None else context_mask,
        )

        video_mlp_input = apply_wan_modulation(
            self.video_norm3(video_states.float()),
            v_shift_mlp,
            v_scale_mlp,
        ).type_as(video_states)
        action_mlp_input = apply_wan_modulation(
            self.action_norm3(action_states.float()),
            a_shift_mlp,
            a_scale_mlp,
        ).type_as(action_states)

        video_states = (
            video_states.float()
            + self.video_ffn(video_mlp_input).float() * v_gate_mlp.float()
        ).type_as(video_states)
        action_states = (
            action_states.float()
            + self.action_ffn(action_mlp_input).float() * a_gate_mlp.float()
        ).type_as(action_states)
        return video_states, action_states


class WanTransformer3DMoTModel(ModelMixin, ConfigMixin):
    _supports_gradient_checkpointing = True
    # Keep het null params in fp32 (diffusers honors this on dtype cast). Under MoT+FSDP
    # they are otherwise bf16 leaves whose grad is bf16, mismatching the fp32 grads of the
    # rest of the FSDP root reduce-scatter unit -> "uniform gradient dtype" assertion. shared
    # model.py already lists scale_shift/norms here; MoT had no list at all.
    _keep_in_fp32_modules = ["null_action_tokens", "null_width_action_correction"]

    @register_to_config
    def __init__(
        self,
        patch_size=(1, 2, 2),
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
        flex_mask_block_size=64,
        action_condition_mode="inverse_dynamics",
        model_structure="mot",
        action_hidden_dim=768,
        action_ffn_dim=None,
        action_mlp_hidden_dim=256,
        init_mode="video_interp_alpha",
        alpha_scale=None,
        missing_action_filling_pad_unused=None,
        missing_action_filling_used_channels=None,
        missing_action_filling_gripper_channels=None,
    ):
        super().__init__()
        if model_structure != "mot":
            raise ValueError(
                f"WanTransformer3DMoTModel requires model_structure='mot', got {model_structure!r}"
            )
        if action_condition_mode not in {"inverse_dynamics", "fastwam"}:
            raise ValueError(
                "action_condition_mode must be 'inverse_dynamics' or 'fastwam', "
                f"got {action_condition_mode!r}"
            )

        FlexAttnFunc.set_mask_block_size(flex_mask_block_size)
        self.patch_size = tuple(patch_size)
        self.attn_mode = str(attn_mode)
        self.num_attention_heads = int(num_attention_heads)
        self.attention_head_dim = int(attention_head_dim)
        self.video_dim = self.num_attention_heads * self.attention_head_dim
        self.action_hidden_dim = int(action_hidden_dim)
        self.action_ffn_dim = int(
            action_ffn_dim
            if action_ffn_dim is not None
            else round(
                float(ffn_dim) * float(self.action_hidden_dim) / float(self.video_dim)
            )
        )
        self.action_mlp_hidden_dim = int(action_mlp_hidden_dim)
        self.alpha_scale = (
            float(alpha_scale)
            if alpha_scale is not None
            else math.sqrt(float(self.video_dim) / float(self.action_hidden_dim))
        )

        self.rope = WanRotaryPosEmbed(
            self.attention_head_dim, self.patch_size, rope_max_seq_len
        )

        self.patch_embedding_mlp = nn.Linear(
            in_channels * math.prod(self.patch_size), self.video_dim
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, self.action_mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.action_mlp_hidden_dim, self.action_hidden_dim),
        )
        self.video_condition_embedder = WanTimeTextImageEmbedding(
            dim=self.video_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=self.video_dim * 6,
            text_embed_dim=text_dim,
            pos_embed_seq_len=pos_embed_seq_len,
        )
        self.action_condition_embedder = WanTimeTextImageEmbedding(
            dim=self.action_hidden_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=self.action_hidden_dim * 6,
            text_embed_dim=text_dim,
            pos_embed_seq_len=pos_embed_seq_len,
        )
        self.video_condition_embedder_action = deepcopy(self.video_condition_embedder)
        self.action_condition_embedder_action = deepcopy(self.action_condition_embedder)

        self.blocks = nn.ModuleList(
            [
                LingBotVAMoTBlock(
                    video_dim=self.video_dim,
                    action_dim=self.action_hidden_dim,
                    mixed_attn_dim=self.video_dim,
                    num_heads=self.num_attention_heads,
                    video_ffn_dim=int(ffn_dim),
                    action_ffn_dim=self.action_ffn_dim,
                    text_dim=text_dim,
                    eps=float(eps),
                    attn_mode=self.attn_mode,
                )
                for _ in range(int(num_layers))
            ]
        )

        self.video_norm_out = FP32LayerNorm(
            self.video_dim, eps, elementwise_affine=False
        )
        self.action_norm_out = FP32LayerNorm(
            self.action_hidden_dim, eps, elementwise_affine=False
        )
        self.video_proj_out = nn.Linear(
            self.video_dim, out_channels * math.prod(self.patch_size)
        )
        self.action_decoder = nn.Sequential(
            nn.Linear(self.action_hidden_dim, self.action_mlp_hidden_dim),
            nn.GELU(approximate="tanh"),
            nn.Linear(self.action_mlp_hidden_dim, action_dim),
        )
        self.video_scale_shift_table = nn.Parameter(
            torch.randn(1, 2, self.video_dim) / self.video_dim**0.5
        )
        self.action_scale_shift_table = nn.Parameter(
            torch.randn(1, 2, self.action_hidden_dim) / self.action_hidden_dim**0.5
        )

        # Heterogeneous-training null token bank — mirrors the shared backbone:
        # one learnable vector per action sequence position, used to overwrite
        # the action embedding when action labels are absent
        # (slam_validity_all=False), plus a zero-initialized additive
        # correction applied when only the gripper-width sensor is unreliable.
        # Sized in action_hidden_dim because action tokens flow through the
        # action stream of the MoT, not the video stream.
        # Shape is (action_chunk_size, action_hidden_dim) with NO leading singleton
        # dim: a size-1 dim 0 makes FSDP2 "fake-shard" (cannot split across 8 ranks)
        # and deadlocks the optimizer-state AllGather during checkpoint save. The
        # forward path adds the batch dim at use site via .unsqueeze(0).expand.
        # Sized to action_chunk_size (4), not the full L_action: null tokens are
        # tiled per-position by modulo, so all 4 rows fire on each invalid sample.
        self._action_chunk_size = 4
        self.null_action_tokens = nn.Parameter(
            torch.randn(self._action_chunk_size, self.action_hidden_dim) * 0.02
        )
        self.null_width_action_correction = nn.Parameter(
            torch.zeros(self._action_chunk_size, self.action_hidden_dim)
        )

        self._init_action_mlps()

        # missing_action_filling (MoT): applied at action_encoder[0] (the first linear).
        # action_encoder_missing is resident; per-sample enabled controls whether
        # validity streams produce a non-trivial mask.
        # Mechanism + config: see missing_action_filling.py.
        _register_missing_action_filling(
            self,
            used_channels_arg=missing_action_filling_used_channels,
            gripper_channels_arg=missing_action_filling_gripper_channels,
            pad_unused_arg=missing_action_filling_pad_unused,
            action_dim=self.action_encoder[0].weight.shape[1],
            hidden_dim=self.action_encoder[0].weight.shape[0],
            param_name="action_encoder_missing",
        )

        logger.info("init WanTransformer3DMoTModel success")

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

    def _init_action_mlps(self) -> None:
        for module in self.action_encoder:
            if isinstance(module, nn.Linear):
                _init_linear_xavier(module)
        for module in self.action_decoder:
            if isinstance(module, nn.Linear):
                _init_linear_xavier(module)

    def clear_cache(self, cache_name: str) -> None:
        for block in self.blocks:
            block.clear_cache(cache_name)

    def compile_blocks(
        self, *, mode: str = "default", dynamic: bool | None = None
    ) -> None:
        """Compile the per-block compute hot loop in place.

        Thin wrapper over :func:`compile_transformer_blocks`; see it for the
        rationale, mode trade-offs, and fallback behavior.

        The MoT cached forward calls ``block.forward_single_stream(...)``
        directly rather than ``block(...)``, so the block must be compiled at
        the *method* level: an ``OptimizedModule`` wrapper would delegate the
        custom method back to the eager original and the compile would silently
        no-op.
        """
        compile_transformer_blocks(
            self.blocks,
            mode=mode,
            dynamic=dynamic,
            method_name="forward_single_stream",
        )

    def cache_cross_kv(
        self,
        encoder_hidden_states: torch.Tensor,
        cache_name: str = "pos",
        action_encoder_hidden_states: torch.Tensor | None = None,
    ) -> None:
        """Pre-project the fixed text context into every block's cross-attention
        K/V so the denoise loop reuses them instead of recomputing to_k/to_v on
        the unchanging prompt every step.

        The video and action streams use separately-embedded text contexts with
        different feature dims. When ``action_encoder_hidden_states`` is omitted
        only the video stream is cached; the action stream then falls back to
        on-the-fly projection.
        """
        for block in self.blocks:
            block.cache_cross_kv(
                encoder_hidden_states,
                action_encoder_hidden_states,
                cache_name,
            )

    def clear_pred_cache(self, cache_name: str) -> None:
        for block in self.blocks:
            block.clear_pred_cache(cache_name)

    def create_empty_cache(
        self,
        cache_name,
        attn_window,
        latent_token_per_chunk,
        action_token_per_chunk,
        device,
        dtype,
        batch_size,
    ) -> None:
        total_tokens = (attn_window // 2) * latent_token_per_chunk + (
            attn_window // 2
        ) * action_token_per_chunk
        # Headroom holds at most one stream's per-call K/V staged on top of a
        # full cache by ``_stage_and_view`` so the SDPA hot path reads a single
        # contiguous slice without copying the history via ``torch.cat``.
        headroom = max(int(latent_token_per_chunk), int(action_token_per_chunk))
        for block in self.blocks:
            block.init_kv_cache(
                cache_name,
                total_tokens=total_tokens,
                mixed_attn_dim=self.video_dim,
                device=device,
                dtype=dtype,
                batch_size=batch_size,
                headroom=headroom,
            )

    def _input_embed(self, latents, input_type="latent"):
        return embed_lingbot_inputs(
            latents,
            input_type=input_type,
            patch_size=self.patch_size,
            patch_embedding_mlp=self.patch_embedding_mlp,
            action_embedder=self.action_encoder,
        )

    def _input_embed_multiview_latents(self, latents_by_view):
        return torch.cat(
            [
                self._input_embed(latent, input_type="latent")
                for latent in latents_by_view
            ],
            dim=1,
        )

    def _text_embed(self, latents, action_mode: bool = False):
        embedder = (
            self.action_condition_embedder
            if action_mode
            else self.video_condition_embedder
        )
        return embed_lingbot_inputs(
            latents,
            input_type="text",
            patch_size=self.patch_size,
            text_embedder=embedder.text_embedder,
        )

    def _text_context(self, latent_dict: dict, *, action_mode: bool = False):
        text_emb_by_piece = latent_dict.get("text_emb_by_piece")
        if text_emb_by_piece is None:
            text_hidden_states = self._text_embed(
                latent_dict["text_emb"],
                action_mode=action_mode,
            )
            context_mask = torch.ones(
                text_hidden_states.shape[:2],
                dtype=torch.bool,
                device=text_hidden_states.device,
            )
            return text_hidden_states, context_mask, None

        if text_emb_by_piece.ndim == 3:
            text_emb_by_piece = text_emb_by_piece.unsqueeze(0)
        if text_emb_by_piece.ndim != 4:
            raise ValueError(
                "`text_emb_by_piece` must have shape [B, piece_count, text_len, dim], "
                f"got {tuple(text_emb_by_piece.shape)}."
            )

        batch_size, piece_count, text_len, text_dim = text_emb_by_piece.shape
        flattened_text_emb = text_emb_by_piece.reshape(
            batch_size,
            piece_count * text_len,
            text_dim,
        )
        text_hidden_states = self._text_embed(
            flattened_text_emb,
            action_mode=action_mode,
        )

        metadata = {
            "piece_count": piece_count,
            "latent_frame_ranges": latent_dict["text_emb_latent_frame_ranges"],
            "text_len": int(text_len),
            "context_len": int(piece_count * text_len),
        }
        return text_hidden_states, None, metadata

    def _video_query_shape_groups(
        self,
        *,
        is_multiview_latent: bool,
        latent_shape: tuple,
        latent_view_shapes: list[tuple] | None,
        include_video_prediction: bool = True,
    ) -> list[tuple[int, int]]:
        shape_groups = []
        source_shapes = latent_view_shapes if is_multiview_latent else [latent_shape]
        stream_count = 2 if include_video_prediction else 1
        for _stream_idx in range(stream_count):
            for shape in source_shapes:
                _, _, frame_count, height, width = shape
                temporal_patches = math.ceil(int(frame_count) / int(self.patch_size[0]))
                tokens_per_frame = (int(height) // int(self.patch_size[1])) * (
                    int(width) // int(self.patch_size[2])
                )
                shape_groups.append((temporal_patches, tokens_per_frame))
        return shape_groups

    def _action_query_shape_groups(self, action_shape: tuple) -> list[tuple[int, int]]:
        _, _, frame_count, height, width = action_shape
        tokens_per_frame = int(height) * int(width)
        return [
            (int(frame_count), tokens_per_frame),
            (int(frame_count), tokens_per_frame),
        ]

    def _piece_context_mask(
        self,
        metadata: dict | None,
        *,
        query_shape_groups: list[tuple[int, int]],
        device,
    ) -> torch.Tensor | None:
        if metadata is None:
            return None
        return build_piece_text_context_mask(
            piece_count=metadata["piece_count"],
            latent_frame_ranges=metadata["latent_frame_ranges"],
            query_shape_groups=query_shape_groups,
            text_len=metadata["text_len"],
            context_len=metadata["context_len"],
            temporal_patch_size=int(self.patch_size[0]),
            device=device,
        )

    def _time_embed(
        self,
        timesteps,
        height: int,
        width: int,
        dtype,
        *,
        action_mode: bool = False,
    ):
        return embed_lingbot_timesteps(
            timesteps,
            height,
            width,
            dtype,
            patch_size=self.patch_size,
            condition_embedder=self.video_condition_embedder_action,
            condition_embedder_action=self.action_condition_embedder_action,
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
            if current_timesteps is None:
                continue
            for view_shape in view_shapes:
                _, _, _, height, width = view_shape
                repeats = (height // self.patch_size[1]) * (width // self.patch_size[2])
                token_timestep_groups.append(
                    torch.repeat_interleave(current_timesteps, repeats, dim=1)
                )
        latent_time_steps = torch.cat(token_timestep_groups, dim=1)
        temb, timestep_proj = self.video_condition_embedder_action(
            latent_time_steps,
            dtype=dtype,
        )
        timestep_proj = timestep_proj.unflatten(2, (6, -1))
        return temb, timestep_proj

    def _finalize_stream(
        self,
        hidden_states: torch.Tensor,
        temb: torch.Tensor,
        norm_out: nn.Module,
        scale_shift_table: torch.Tensor,
    ) -> torch.Tensor:
        return finalize_lingbot_outputs(
            hidden_states,
            temb,
            norm_out,
            scale_shift_table,
        )

    def _forward_cached(
        self,
        input_dict,
        *,
        update_cache: int,
        cache_name: str,
        action_mode: bool,
    ):
        if sequence_parallel_is_enabled():
            raise RuntimeError(
                "MoT sequence parallel currently supports only train_mode=True; "
                "cached inference/KV cache is not supported."
            )
        if self.attn_mode == "flex":
            raise NotImplementedError(
                "MoT cached inference requires attn_mode='torch' or 'flashattn'."
            )

        if action_mode:
            if input_dict.get("action_condition_all_missing", False):
                # Train/infer parity for a fully-dropped action-history condition.
                # Training's action_history_condition_dropout embeds the condition
                # action through the missing_action_filling path with an all-zero
                # channel-validity grid (v=0 everywhere), producing the learned
                # "all channels missing" embedding rather than the raw action. The
                # plain action_encoder can only ever yield W_normal @ x + bias and
                # cannot reproduce that learned missing term, so the zeroed condition
                # (e.g. zero_action_condition) must be routed through the same
                # missing path to match training instead of leaving a train/infer gap.
                if self.action_encoder_missing is None:
                    raise RuntimeError(
                        "action_condition_all_missing requires missing_action_filling "
                        "(action_encoder_missing) to be configured; the plain "
                        "action_encoder cannot reproduce the learned missing "
                        "embedding used during action_history_condition_dropout "
                        "training."
                    )
                noisy_action = input_dict["noisy_latents"]
                (
                    channel_validity_all_missing,
                    missing_action_filling_enabled,
                ) = _make_action_condition_all_missing_inputs(noisy_action)
                hidden_states = self._embed_action_with_missing(
                    noisy_action,
                    None,
                    None,
                    channel_validity_per_frame=channel_validity_all_missing,
                    missing_action_filling_enabled=missing_action_filling_enabled,
                )
            else:
                hidden_states = self._input_embed(
                    input_dict["noisy_latents"],
                    input_type="action",
                )
            timestep_height = input_dict["noisy_latents"].shape[-2]
            timestep_width = input_dict["noisy_latents"].shape[-1]
            temb, timestep_proj = self._time_embed(
                input_dict["timesteps"],
                timestep_height,
                timestep_width,
                dtype=hidden_states.dtype,
                action_mode=True,
            )
        else:
            noisy_latents_by_view = input_dict["noisy_latents"]
            if not isinstance(noisy_latents_by_view, list):
                noisy_latents_by_view = [noisy_latents_by_view]
            hidden_states = self._input_embed_multiview_latents(noisy_latents_by_view)
            temb, timestep_proj = self._time_embed_multiview_latents(
                timesteps=input_dict["timesteps"],
                cond_timesteps=None,
                view_shapes=[tuple(latent.shape) for latent in noisy_latents_by_view],
                dtype=hidden_states.dtype,
            )
        text_hidden_states = input_dict.get("text_hidden_states")
        if text_hidden_states is None:
            text_hidden_states = self._text_embed(
                input_dict["text_emb"],
                action_mode=action_mode,
            )
        context_mask = torch.ones(
            text_hidden_states.shape[:2],
            dtype=torch.bool,
            device=text_hidden_states.device,
        )
        rotary_emb = self.rope(input_dict["grid_id"]).unsqueeze(2)
        stream_name = "action" if action_mode else "video"
        for block in self.blocks:
            hidden_states = block.forward_single_stream(
                hidden_states,
                text_hidden_states,
                context_mask,
                timestep_proj,
                rotary_emb,
                stream_name=stream_name,
                update_cache=update_cache,
                cache_name=cache_name,
            )

        if action_mode:
            hidden_states = self._finalize_stream(
                hidden_states,
                temb,
                self.action_norm_out,
                self.action_scale_shift_table,
            )
            return self.action_decoder(hidden_states)

        hidden_states = self._finalize_stream(
            hidden_states,
            temb,
            self.video_norm_out,
            self.video_scale_shift_table,
        )
        hidden_states = self.video_proj_out(hidden_states)
        return rearrange(
            hidden_states,
            "b l (n c) -> b (l n) c",
            n=math.prod(self.patch_size),
        )

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
        """MoT wrapper over `_embed_action_with_missing_filling`：core 走 action_encoder[0]
        这个 first linear，core 出来后再过 encoder[1:] 的 GELU + 2nd linear。公式 / v-split /
        Args / Returns 见 `_embed_action_with_missing_filling` (missing_action_filling.py)。"""
        # 函数体 wrapper-WHY：core 喂 action_encoder[0] 的 weight/bias + action_encoder_missing
        # param，返回 first-linear 输出；MoT 再跑 REMAINING encoder layers (GELU + 2nd linear) —
        # 这段 post-linear stack 是与 shared backbone 唯一的结构差异，所以留在 wrapper 里。
        first_linear = self.action_encoder[0]
        hidden = _embed_action_with_missing_filling(
            action_input,
            action_validity,
            width_validity,
            action_validity_per_frame,
            width_validity_per_frame,
            channel_validity_per_frame,
            missing_action_filling_enabled,
            w_normal_weight=first_linear.weight,
            w_normal_bias=first_linear.bias,
            w_missing_param=self.action_encoder_missing,
            used_channels=self._missing_action_filling_used_channels,
            gripper_channels=self._missing_action_filling_gripper_channels,
            pad_unused=self._missing_action_filling_pad_unused,
            used_channel_mask=used_channel_mask,
            gripper_channel_mask=gripper_channel_mask,
            pad_unused_mask=pad_unused_mask,
        )
        # Remaining encoder stack: GELU (index 1) + second linear (index 2).
        for module in list(self.action_encoder)[1:]:
            hidden = module(hidden)
        return hidden

    def _maybe_replace_with_null_action(self, action_h, action_validity):
        """Replace action embeddings with learnable null tokens where the action is missing.

        Used in heterogeneous training: samples whose action stream is invalid get a
        per-position learnable null vector instead of their embedding.

        We never skip this even when every sample is valid: null_action_tokens has to
        appear in the autograd graph every step, otherwise under ZeRO-2 + bf16 its gradient
        goes missing on some ranks and training diverges to NaN within ~30 steps.
        """
        if action_validity is None:
            # Treat as all-VALID (no action missing): a None validity means the batch carries no
            # invalid-action info, so nothing should be replaced with null tokens. The previous
            # all-invalid (torch.zeros) default replaced EVERY embedding with the null token on
            # any old batch that omitted action_validity, contradicting both the shared backbone
            # (model.py:1896 — None action_validity leaves v at all-ones / unmasked) and this
            # file's own width-correction None default (:1504 torch.ones = all-valid).
            validity = torch.ones(
                action_h.shape[0], dtype=torch.bool, device=action_h.device
            )
        else:
            validity = action_validity.to(action_h.device).bool().view(-1)
        invalid = (~validity).view(-1, 1, 1)
        B, L, _ = action_h.shape
        # Each frame has one null token. Position i uses null_action_tokens[i %
        # chunk_size], so every action chunk reuses the same per-frame null token.
        indices = torch.arange(L, device=action_h.device) % self._action_chunk_size
        # Keep null_tok in fp32 through where(): bf16 storage quantizes the small
        # Adam updates to zero (update magnitude ~6e-5 sits at the bf16 quantum at
        # value ~0.016), so the fp32 path is required to let the null token train.
        null_tok = (
            self.null_action_tokens[indices].float().unsqueeze(0).expand(B, -1, -1)
        )
        result_fp32 = torch.where(invalid, null_tok, action_h.float())
        return result_fp32.to(action_h.dtype)

    def _maybe_apply_width_correction(self, action_h, action_validity, width_validity):
        """Heterogeneous-training: additive per-position correction applied
        when action labels are valid but the gripper-width sensor is unreliable.
        Mirror of the shared-backbone helper.

        No fast-path early return: null_width_action_correction must appear in the
        autograd graph every step (same ZeRO-2 + bf16 grad-consistency reason as
        _maybe_replace_with_null_action). When no correction is needed the mask is
        0 → mathematically a no-op but the param still touches the graph.
        """
        if width_validity is None:
            # Treat as all-True (no width missing) — synthesize True mask
            width = torch.ones(
                action_h.shape[0], dtype=torch.bool, device=action_h.device
            )
        else:
            width = width_validity.to(action_h.device).bool().view(-1)
        if action_validity is not None:
            action_v = action_validity.to(action_h.device).bool().view(-1)
            needs_correction = (~width) & action_v
        else:
            needs_correction = ~width
        B, L, _ = action_h.shape
        # Same tile-by-modulo pattern as null_action_tokens.
        indices = torch.arange(L, device=action_h.device) % self._action_chunk_size
        # fp32 path required for the same bf16-quantization reason as null_action.
        correction = (
            self.null_width_action_correction[indices]
            .float()
            .unsqueeze(0)
            .expand(B, -1, -1)
        )
        mask = needs_correction.view(-1, 1, 1).float()
        result_fp32 = action_h.float() + mask * correction
        return result_fp32.to(action_h.dtype)

    def forward_train(self, input_dict):
        latent_dict = input_dict["latent_dict"]
        action_dict = input_dict["action_dict"]
        include_video_prediction = not bool(
            input_dict.get("skip_video_prediction", False)
        )
        video_dtype = self.patch_embedding_mlp.weight.dtype
        action_dtype = self.action_encoder[0].weight.dtype
        is_multiview_latent = "noisy_latents_by_view" in latent_dict
        if is_multiview_latent:
            latent_noisy_by_view = [
                latent.to(video_dtype)
                for latent in latent_dict["noisy_latents_by_view"]
            ]
            latent_clean_by_view = [
                latent.to(video_dtype) for latent in latent_dict["latent_by_view"]
            ]
            latent_shape = latent_noisy_by_view[0].shape
            latent_view_shapes = [
                tuple(latent.shape) for latent in latent_noisy_by_view
            ]
        else:
            latent_noisy = latent_dict["noisy_latents"].to(video_dtype)
            latent_clean = latent_dict["latent"].to(video_dtype)
            latent_shape = latent_noisy.shape
            latent_view_shapes = None
        action_noisy = action_dict["noisy_latents"].to(action_dtype)
        action_clean = action_dict["latent"].to(action_dtype)

        if is_multiview_latent:
            latent_hidden_states = self._input_embed_multiview_latents(
                latent_noisy_by_view
            )
            condition_latent_hidden_states = self._input_embed_multiview_latents(
                latent_clean_by_view
            )
        else:
            latent_hidden_states = self._input_embed(latent_noisy, input_type="latent")
            condition_latent_hidden_states = self._input_embed(
                latent_clean, input_type="latent"
            )
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
        # Kept for older/non-resident model variants. Current MoT uses the
        # resident channel-wise missing_action_filling path below, matching the
        # shared backbone.
        _assert_missing_action_filling_configured_for_per_frame(
            self.action_encoder_missing,
            _action_validity_pf,
            _width_validity_pf,
            _channel_validity_pf,
        )
        _assert_missing_action_filling_configured_for_per_frame(
            self.action_encoder_missing,
            _cond_action_validity_pf,
            _cond_width_validity_pf,
            _cond_channel_validity_pf,
        )
        # Channel-wise missing_action_filling path is always resident. Per-sample
        # `learnable_action_embedding_enabled` gates whether validity streams affect
        # the mask, so disabled logical datasets behave like all-valid samples.
        action_hidden_states = self._embed_action_with_missing(
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
        condition_action_hidden_states = self._embed_action_with_missing(
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
        # Add a zero-valued term that touches the heterogeneity params, so they stay in the
        # autograd graph every step. A given micro-batch may only use one of them, but if a
        # param gets no gradient FSDP synthesizes a bf16 one that mismatches the fp32
        # gradients and breaks training. This zero touch gives them a real fp32 gradient.
        _hetero_touch = (
            self.null_action_tokens.float().sum()
            + self.null_width_action_correction.float().sum()
        )
        if self.action_encoder_missing is not None:
            _hetero_touch = _hetero_touch + self.action_encoder_missing.float().sum()
        action_hidden_states = action_hidden_states + (0.0 * _hetero_touch).to(
            action_hidden_states.dtype
        )

        video_state_parts = []
        if include_video_prediction:
            video_state_parts.append(latent_hidden_states)
        video_state_parts.append(condition_latent_hidden_states)
        video_states = torch.cat(video_state_parts, dim=1)
        action_states = torch.cat(
            [action_hidden_states, condition_action_hidden_states], dim=1
        )

        (
            video_text_hidden_states,
            context_mask,
            text_piece_metadata,
        ) = self._text_context(latent_dict, action_mode=False)
        action_text_hidden_states, _, _ = self._text_context(
            latent_dict,
            action_mode=True,
        )
        piece_frame_ranges = (
            text_piece_metadata["latent_frame_ranges"]
            if text_piece_metadata is not None
            else None
        )

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
        latent_grid_id_parts = []
        if include_video_prediction:
            latent_grid_id_parts.append(latent_base_grid_id)
        latent_grid_id_parts.append(latent_base_grid_id)
        latent_grid_id = torch.cat(latent_grid_id_parts, dim=2)
        action_grid_id = torch.cat(
            [
                action_base_grid_id,
                action_base_grid_id,
            ],
            dim=2,
        )
        video_rotary_emb = self.rope(latent_grid_id).unsqueeze(2)
        action_rotary_emb = self.rope(action_grid_id).unsqueeze(2)

        if is_multiview_latent:
            video_temb, video_timestep_proj = self._time_embed_multiview_latents(
                latent_dict["timesteps"],
                latent_dict["cond_timesteps"],
                latent_view_shapes,
                dtype=video_states.dtype,
            )
        else:
            latent_time_steps = torch.cat(
                [
                    latent_dict["timesteps"],
                    latent_dict["cond_timesteps"],
                ],
                dim=1,
            )
            video_temb, video_timestep_proj = self._time_embed(
                latent_time_steps,
                latent_noisy.shape[-2],
                latent_noisy.shape[-1],
                dtype=video_states.dtype,
                action_mode=False,
            )
        action_time_steps = torch.cat(
            [
                action_dict["timesteps"],
                action_dict["cond_timesteps"],
            ],
            dim=1,
        )
        action_temb, action_timestep_proj = self._time_embed(
            action_time_steps,
            action_noisy.shape[-2],
            action_noisy.shape[-1],
            dtype=action_states.dtype,
            action_mode=True,
        )
        if not include_video_prediction:
            video_temb = video_temb[:, latent_hidden_states.shape[1] :]
            video_timestep_proj = video_timestep_proj[
                :, latent_hidden_states.shape[1] :
            ]

        attention_mask = (
            build_joint_attention_mask(
                latent_shape=latent_shape,
                action_shape=action_noisy.shape,
                chunk_size=input_dict["chunk_size"],
                window_size=input_dict["window_size"],
                patch_size=self.patch_size,
                device=video_states.device,
                action_condition_mode=self.config.action_condition_mode,
                latent_view_shapes=latent_view_shapes,
                piece_frame_ranges=piece_frame_ranges,
                attention_piece_frame_ranges=latent_dict.get(
                    "attention_piece_frame_ranges"
                ),
                chunk_grouping_start_from_one=input_dict.get(
                    "chunk_grouping_start_from_one", False
                ),
                include_video_prediction=include_video_prediction,
            )
            if self.attn_mode != "flex"
            else None
        )

        if self.attn_mode == "flex":
            FlexAttnFunc.init_mask(
                latent_shape=latent_shape,
                action_shape=action_noisy.shape,
                padded_length=0,
                chunk_size=input_dict["chunk_size"],
                window_size=input_dict["window_size"],
                patch_size=self.patch_size,
                device=video_states.device,
                action_condition_mode=self.config.action_condition_mode,
                latent_view_shapes=latent_view_shapes,
                piece_frame_ranges=piece_frame_ranges,
                attention_piece_frame_ranges=latent_dict.get(
                    "attention_piece_frame_ranges"
                ),
                chunk_grouping_start_from_one=input_dict.get(
                    "chunk_grouping_start_from_one", False
                ),
                include_video_prediction=include_video_prediction,
            )
        elif attention_mask is not None:
            attention_mask = attention_mask.to(
                dtype=torch.bool, device=video_states.device
            )

        video_context_mask = self._piece_context_mask(
            text_piece_metadata,
            query_shape_groups=self._video_query_shape_groups(
                is_multiview_latent=is_multiview_latent,
                latent_shape=latent_shape,
                latent_view_shapes=latent_view_shapes,
                include_video_prediction=include_video_prediction,
            ),
            device=video_states.device,
        )
        action_context_mask = self._piece_context_mask(
            text_piece_metadata,
            query_shape_groups=self._action_query_shape_groups(action_noisy.shape),
            device=action_states.device,
        )

        if sequence_parallel_is_enabled():
            video_states = _split_sequence_parallel_stream(
                video_states,
                stream_name="video_states",
            )
            action_states = _split_sequence_parallel_stream(
                action_states,
                stream_name="action_states",
            )
            video_rotary_emb = _split_sequence_parallel_stream(
                video_rotary_emb,
                stream_name="video_rotary_emb",
            )
            action_rotary_emb = _split_sequence_parallel_stream(
                action_rotary_emb,
                stream_name="action_rotary_emb",
            )
            video_temb = _split_sequence_parallel_stream(
                video_temb,
                stream_name="video_temb",
            )
            action_temb = _split_sequence_parallel_stream(
                action_temb,
                stream_name="action_temb",
            )
            video_timestep_proj = _split_sequence_parallel_stream(
                video_timestep_proj,
                stream_name="video_timestep_proj",
            )
            action_timestep_proj = _split_sequence_parallel_stream(
                action_timestep_proj,
                stream_name="action_timestep_proj",
            )
            video_context_mask = _split_sequence_parallel_query_mask(
                video_context_mask,
                stream_name="video_context_mask",
            )
            action_context_mask = _split_sequence_parallel_query_mask(
                action_context_mask,
                stream_name="action_context_mask",
            )

        for block in self.blocks:
            video_states, action_states = block(
                video_states=video_states,
                action_states=action_states,
                video_context=video_text_hidden_states,
                action_context=action_text_hidden_states,
                context_mask=context_mask,
                video_context_mask=video_context_mask,
                action_context_mask=action_context_mask,
                video_t_mod=video_timestep_proj,
                action_t_mod=action_timestep_proj,
                video_rotary_emb=video_rotary_emb,
                action_rotary_emb=action_rotary_emb,
                attention_mask=attention_mask,
            )

        video_states = self._finalize_stream(
            video_states,
            video_temb,
            self.video_norm_out,
            self.video_scale_shift_table,
        )
        action_states = self._finalize_stream(
            action_states,
            action_temb,
            self.action_norm_out,
            self.action_scale_shift_table,
        )

        if sequence_parallel_is_enabled():
            video_states = gather_sequence_parallel_region(video_states, dim=1)
            action_states = gather_sequence_parallel_region(action_states, dim=1)

        video_split_list = []
        if include_video_prediction:
            video_split_list.append(latent_hidden_states.shape[1])
        video_split_list.append(condition_latent_hidden_states.shape[1])
        video_split_states = torch.split(video_states, video_split_list, dim=1)
        if include_video_prediction:
            latent_hidden_states = video_split_states[0]
        else:
            latent_hidden_states = None
        action_hidden_states, _ = torch.split(
            action_states,
            [
                action_hidden_states.shape[1],
                condition_action_hidden_states.shape[1],
            ],
            dim=1,
        )

        if include_video_prediction:
            latent_hidden_states = self.video_proj_out(latent_hidden_states)
            latent_hidden_states = rearrange(
                latent_hidden_states,
                "b l (n c) -> b (l n) c",
                n=math.prod(self.patch_size),
            )
        action_hidden_states = self.action_decoder(action_hidden_states)
        return latent_hidden_states, action_hidden_states

    def forward(
        self,
        input_dict,
        update_cache=0,
        cache_name="pos",
        action_mode=False,
        train_mode=False,
    ):
        if train_mode:
            return self.forward_train(input_dict)
        return self._forward_cached(
            input_dict,
            update_cache=update_cache,
            cache_name=cache_name,
            action_mode=action_mode,
        )

    @classmethod
    def from_shared_model(
        cls,
        shared_model,
        *,
        action_hidden_dim: int = 768,
        action_ffn_dim: int | None = None,
        action_mlp_hidden_dim: int = 256,
        init_mode: str = "video_interp_alpha",
        alpha_scale: float | None = None,
    ) -> "WanTransformer3DMoTModel":
        shared_cfg = dict(shared_model.config)
        shared_cfg.pop("_name_or_path", None)
        shared_cfg["model_structure"] = "mot"
        shared_cfg["action_hidden_dim"] = int(action_hidden_dim)
        shared_cfg["action_ffn_dim"] = action_ffn_dim
        shared_cfg["action_mlp_hidden_dim"] = int(action_mlp_hidden_dim)
        shared_cfg["init_mode"] = str(init_mode)
        shared_cfg["alpha_scale"] = alpha_scale
        model = cls(**shared_cfg)

        if init_mode != "video_interp_alpha":
            raise ValueError(f"Unsupported mot init_mode: {init_mode}")

        alpha = (
            float(alpha_scale) if alpha_scale is not None else float(model.alpha_scale)
        )

        with torch.no_grad():
            model.patch_embedding_mlp.load_state_dict(
                shared_model.patch_embedding_mlp.state_dict(), strict=True
            )
            model.video_condition_embedder.load_state_dict(
                shared_model.condition_embedder.state_dict(), strict=True
            )
            model.video_condition_embedder_action.load_state_dict(
                shared_model.condition_embedder_action.state_dict(), strict=True
            )
            model.video_norm_out.load_state_dict(
                shared_model.norm_out.state_dict(), strict=True
            )
            model.video_proj_out.load_state_dict(
                shared_model.proj_out.state_dict(), strict=True
            )
            model.video_scale_shift_table.copy_(shared_model.scale_shift_table)

            _copy_module_with_resize(
                shared_model.condition_embedder,
                model.action_condition_embedder,
                alpha=alpha,
            )
            _copy_module_with_resize(
                shared_model.condition_embedder_action,
                model.action_condition_embedder_action,
                alpha=alpha,
            )
            _copy_norm(shared_model.norm_out, model.action_norm_out)
            model.action_scale_shift_table.copy_(
                _resize_tensor_like(
                    shared_model.scale_shift_table,
                    model.action_scale_shift_table,
                )
            )

            for source_block, target_block in zip(shared_model.blocks, model.blocks):
                target_block.video_norm1.load_state_dict(
                    source_block.norm1.state_dict(), strict=True
                )
                target_block.video_attn1.to_q.load_state_dict(
                    source_block.attn1.to_q.state_dict(), strict=True
                )
                target_block.video_attn1.to_k.load_state_dict(
                    source_block.attn1.to_k.state_dict(), strict=True
                )
                target_block.video_attn1.to_v.load_state_dict(
                    source_block.attn1.to_v.state_dict(), strict=True
                )
                target_block.video_attn1.to_out[0].load_state_dict(
                    source_block.attn1.to_out[0].state_dict(), strict=True
                )
                target_block.video_attn1.norm_q.load_state_dict(
                    source_block.attn1.norm_q.state_dict(), strict=True
                )
                target_block.video_attn1.norm_k.load_state_dict(
                    source_block.attn1.norm_k.state_dict(), strict=True
                )
                target_block.video_norm2.load_state_dict(
                    source_block.norm2.state_dict(), strict=True
                )
                target_block.video_attn2.to_q.load_state_dict(
                    source_block.attn2.to_q.state_dict(), strict=True
                )
                target_block.video_attn2.to_k.load_state_dict(
                    source_block.attn2.to_k.state_dict(), strict=True
                )
                target_block.video_attn2.to_v.load_state_dict(
                    source_block.attn2.to_v.state_dict(), strict=True
                )
                target_block.video_attn2.to_out[0].load_state_dict(
                    source_block.attn2.to_out[0].state_dict(), strict=True
                )
                target_block.video_attn2.norm_q.load_state_dict(
                    source_block.attn2.norm_q.state_dict(), strict=True
                )
                target_block.video_attn2.norm_k.load_state_dict(
                    source_block.attn2.norm_k.state_dict(), strict=True
                )
                target_block.video_norm3.load_state_dict(
                    source_block.norm3.state_dict(), strict=True
                )
                target_block.video_ffn.load_state_dict(
                    source_block.ffn.state_dict(), strict=True
                )
                target_block.video_scale_shift_table.copy_(
                    source_block.scale_shift_table
                )

                _copy_norm(source_block.norm1, target_block.action_norm1)
                _copy_linear(
                    source_block.attn1.to_q,
                    target_block.action_attn1.to_q,
                    alpha=alpha,
                )
                _copy_linear(
                    source_block.attn1.to_k,
                    target_block.action_attn1.to_k,
                    alpha=alpha,
                )
                _copy_linear(
                    source_block.attn1.to_v,
                    target_block.action_attn1.to_v,
                    alpha=alpha,
                )
                _copy_linear(
                    source_block.attn1.to_out[0],
                    target_block.action_attn1.to_out[0],
                    alpha=alpha,
                )
                _copy_norm(source_block.attn1.norm_q, target_block.action_attn1.norm_q)
                _copy_norm(source_block.attn1.norm_k, target_block.action_attn1.norm_k)
                _copy_norm(source_block.norm2, target_block.action_norm2)
                _copy_linear(
                    source_block.attn2.to_q,
                    target_block.action_attn2.to_q,
                    alpha=alpha,
                )
                _copy_linear(
                    source_block.attn2.to_k,
                    target_block.action_attn2.to_k,
                    alpha=alpha,
                )
                _copy_linear(
                    source_block.attn2.to_v,
                    target_block.action_attn2.to_v,
                    alpha=alpha,
                )
                _copy_linear(
                    source_block.attn2.to_out[0],
                    target_block.action_attn2.to_out[0],
                    alpha=alpha,
                )
                _copy_norm(source_block.attn2.norm_q, target_block.action_attn2.norm_q)
                _copy_norm(source_block.attn2.norm_k, target_block.action_attn2.norm_k)
                _copy_norm(source_block.norm3, target_block.action_norm3)
                _copy_module_with_resize(
                    source_block.ffn,
                    target_block.action_ffn,
                    alpha=alpha,
                )
                target_block.action_scale_shift_table.copy_(
                    _resize_tensor_like(
                        source_block.scale_shift_table,
                        target_block.action_scale_shift_table,
                    )
                )
        return model
