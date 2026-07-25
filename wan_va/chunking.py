"""Chunk grouping shared by NMX-compatible actions and attention masks."""

from collections.abc import Iterator

import torch


def iter_chunk_slices(
    length: int,
    chunk_step_size: int,
    *,
    grouping_start_from_one: bool = False,
    prefix_step_size: int = 1,
) -> Iterator[slice]:
    """Yield contiguous chunks, optionally isolating the leading condition frame."""

    length = int(length)
    if length <= 0:
        return
    chunk_step_size = int(chunk_step_size)
    if chunk_step_size <= 0:
        raise ValueError(f"chunk_step_size must be positive, got {chunk_step_size}")

    if not grouping_start_from_one:
        for start in range(0, length, chunk_step_size):
            yield slice(start, min(start + chunk_step_size, length))
        return

    prefix_end = min(max(1, int(prefix_step_size)), length)
    yield slice(0, prefix_end)
    for start in range(prefix_end, length, chunk_step_size):
        yield slice(start, min(start + chunk_step_size, length))


def build_chunk_ids(
    frame_ids: torch.Tensor,
    chunk_size: int,
    *,
    grouping_start_from_one: bool = False,
) -> torch.Tensor:
    """Map local frame ids to the same chunks used by action preprocessing."""

    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")
    if not grouping_start_from_one:
        return torch.div(frame_ids, chunk_size, rounding_mode="floor")

    shifted = torch.clamp(frame_ids - 1, min=0)
    offset_ids = 1 + torch.div(shifted, chunk_size, rounding_mode="floor")
    return torch.where(frame_ids == 0, torch.zeros_like(offset_ids), offset_ids)
