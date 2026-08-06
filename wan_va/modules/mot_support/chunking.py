"""Shared chunk-grouping helpers for LingBot-VA."""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch


def resolve_chunk_grouping_start_from_one(config: Any) -> bool:
    """Return whether frame 0 should be isolated before chunk grouping."""

    if config is None:
        return False
    if hasattr(config, "get"):
        return bool(config.get("chunk_grouping_start_from_one", False))
    return bool(getattr(config, "chunk_grouping_start_from_one", False))


def iter_chunk_slices(
    length: int,
    chunk_step_size: int | None,
    *,
    chunk_grouping_start_from_one: bool = False,
    prefix_step_size: int = 1,
) -> Iterator[slice]:
    """Yield contiguous slices for the configured LingBot-VA chunk grouping."""

    length = int(length)
    if length <= 0:
        return
    if chunk_step_size is None or chunk_step_size <= 0:
        yield slice(0, length)
        return

    chunk_step_size = int(chunk_step_size)
    if not chunk_grouping_start_from_one:
        if chunk_step_size >= length:
            yield slice(0, length)
            return
        for start in range(0, length, chunk_step_size):
            yield slice(start, min(start + chunk_step_size, length))
        return

    prefix_step_size = max(1, int(prefix_step_size))
    prefix_end = min(prefix_step_size, length)
    yield slice(0, prefix_end)
    for start in range(prefix_end, length, chunk_step_size):
        yield slice(start, min(start + chunk_step_size, length))


def build_chunk_ids(
    frame_ids: torch.Tensor,
    chunk_size: int,
    *,
    chunk_grouping_start_from_one: bool = False,
) -> torch.Tensor:
    """Map local frame ids to chunk ids under old or frame-0-prefix grouping."""

    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if not chunk_grouping_start_from_one:
        return frame_ids // chunk_size

    shifted = torch.clamp(frame_ids - 1, min=0)
    offset_chunk_ids = 1 + torch.div(shifted, chunk_size, rounding_mode="floor")
    return torch.where(
        frame_ids == 0, torch.zeros_like(offset_chunk_ids), offset_chunk_ids
    )
