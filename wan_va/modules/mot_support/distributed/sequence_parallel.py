from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

import torch
import torch.distributed as dist


@dataclass(frozen=True)
class SequenceParallelState:
    enabled: bool = False
    mode: str = "ulysses"
    scope: str = "node"
    size: int = 1
    rank: int = 0
    world_size: int = 1
    local_rank: int = 0
    local_world_size: int = 1
    group: Any | None = None
    group_ranks: tuple[int, ...] = ()
    data_parallel_rank: int = 0
    data_parallel_world_size: int = 1


_STATE = SequenceParallelState()

_TENSOR_BROADCAST_DTYPE_TO_CODE = {
    torch.float64: 0,
    torch.float32: 1,
    torch.float16: 2,
    torch.bfloat16: 3,
    torch.int64: 4,
    torch.int32: 5,
    torch.int16: 6,
    torch.int8: 7,
    torch.uint8: 8,
    torch.bool: 9,
}
_TENSOR_BROADCAST_CODE_TO_DTYPE = {
    code: dtype for dtype, code in _TENSOR_BROADCAST_DTYPE_TO_CODE.items()
}


def get_sequence_parallel_state() -> SequenceParallelState:
    return _STATE


def sequence_parallel_is_enabled() -> bool:
    return bool(_STATE.enabled and _STATE.size > 1)


@contextmanager
def temporarily_disable_sequence_parallel() -> Iterator[None]:
    """Run local-replica work without treating the process as SP-sharded."""

    global _STATE

    previous = _STATE
    if not sequence_parallel_is_enabled():
        yield
        return

    if dist.is_available() and dist.is_initialized():
        global_rank = int(dist.get_rank())
    elif previous.group_ranks and 0 <= int(previous.rank) < len(previous.group_ranks):
        global_rank = int(previous.group_ranks[int(previous.rank)])
    else:
        global_rank = int(previous.data_parallel_rank) * int(previous.size) + int(
            previous.rank
        )

    _STATE = SequenceParallelState(
        enabled=False,
        mode=previous.mode,
        scope=previous.scope,
        size=1,
        rank=global_rank,
        world_size=int(previous.world_size),
        local_rank=int(previous.local_rank),
        local_world_size=int(previous.local_world_size),
        group=None,
        group_ranks=(),
        data_parallel_rank=global_rank,
        data_parallel_world_size=int(previous.world_size),
    )
    try:
        yield
    finally:
        _STATE = previous


def sequence_parallel_source_global_rank(source_sp_rank: int = 0) -> int:
    state = _require_enabled()
    source_sp_rank = int(source_sp_rank)
    if not 0 <= source_sp_rank < len(state.group_ranks):
        raise ValueError(
            "source_sp_rank must be within the sequence-parallel group, "
            f"got {source_sp_rank} for group_ranks={state.group_ranks}."
        )
    return int(state.group_ranks[source_sp_rank])


def sequence_parallel_is_source_rank(source_sp_rank: int = 0) -> bool:
    source_rank = sequence_parallel_source_global_rank(source_sp_rank)
    return int(dist.get_rank()) == source_rank


def sequence_parallel_is_logical_source_rank(source_sp_rank: int = 0) -> bool:
    if not sequence_parallel_is_enabled():
        return True
    return sequence_parallel_is_source_rank(source_sp_rank)


def sequence_parallel_logical_data_parallel_world_size() -> int:
    state = get_sequence_parallel_state()
    if sequence_parallel_is_enabled():
        return max(1, int(state.data_parallel_world_size))
    if dist.is_available() and dist.is_initialized():
        return max(1, int(dist.get_world_size()))
    return 1


def _parse_sp_size(raw_size, *, local_world_size: int) -> int:
    if raw_size is None:
        raw_size = "auto"
    if isinstance(raw_size, str):
        normalized = raw_size.strip().lower()
        if normalized == "auto":
            return int(local_world_size)
        return int(normalized)
    return int(raw_size)


def initialize_sequence_parallel(
    *,
    enabled: bool,
    mode: str = "ulysses",
    scope: str = "node",
    size="auto",
    rank: int,
    local_rank: int,
    world_size: int,
    local_world_size: int,
) -> SequenceParallelState:
    global _STATE

    rank = int(rank)
    local_rank = int(local_rank)
    world_size = int(world_size)
    local_world_size = int(local_world_size)
    if not enabled:
        _STATE = SequenceParallelState(
            enabled=False,
            rank=rank,
            world_size=world_size,
            local_rank=local_rank,
            local_world_size=local_world_size,
            data_parallel_rank=rank,
            data_parallel_world_size=world_size,
        )
        return _STATE

    if mode != "ulysses":
        raise ValueError(
            "trainer.deepspeed.sequence_parallel.mode currently supports only "
            f"'ulysses', got {mode!r}."
        )
    if scope != "node":
        raise ValueError(
            "trainer.deepspeed.sequence_parallel.scope currently supports only "
            f"'node', got {scope!r}."
        )
    if not (dist.is_available() and dist.is_initialized()):
        raise RuntimeError(
            "Sequence parallel requires torch.distributed to be initialized."
        )
    if world_size <= 1:
        raise ValueError("Sequence parallel requires WORLD_SIZE > 1.")
    if local_world_size <= 1:
        raise ValueError("Node-local sequence parallel requires LOCAL_WORLD_SIZE > 1.")

    sp_size = _parse_sp_size(size, local_world_size=local_world_size)
    if sp_size <= 1:
        raise ValueError(
            f"Sequence parallel size must be greater than 1, got {sp_size}."
        )
    if sp_size > local_world_size or local_world_size % sp_size != 0:
        raise ValueError(
            "Node-local sequence parallel size must divide LOCAL_WORLD_SIZE: "
            f"size={sp_size}, local_world_size={local_world_size}."
        )
    if world_size % sp_size != 0:
        raise ValueError(
            f"WORLD_SIZE ({world_size}) must be divisible by sequence_parallel size ({sp_size})."
        )
    if world_size % local_world_size != 0:
        raise ValueError(
            f"WORLD_SIZE ({world_size}) must be divisible by LOCAL_WORLD_SIZE "
            f"({local_world_size}) for node-local sequence parallel."
        )
    if rank % local_world_size != local_rank:
        raise ValueError(
            "Node-local sequence parallel assumes contiguous ranks per node: "
            f"rank={rank}, local_rank={local_rank}, "
            f"local_world_size={local_world_size}."
        )

    selected_group = None
    selected_ranks: tuple[int, ...] = ()
    for group_start in range(0, world_size, sp_size):
        group_ranks = tuple(range(group_start, group_start + sp_size))
        group = dist.new_group(ranks=list(group_ranks))
        if rank in group_ranks:
            selected_group = group
            selected_ranks = group_ranks

    if selected_group is None:
        raise RuntimeError(f"Failed to create sequence-parallel group for rank {rank}.")

    _STATE = SequenceParallelState(
        enabled=True,
        mode=mode,
        scope=scope,
        size=sp_size,
        rank=local_rank % sp_size,
        world_size=world_size,
        local_rank=local_rank,
        local_world_size=local_world_size,
        group=selected_group,
        group_ranks=selected_ranks,
        data_parallel_rank=rank // sp_size,
        data_parallel_world_size=world_size // sp_size,
    )
    return _STATE


def _require_enabled() -> SequenceParallelState:
    state = get_sequence_parallel_state()
    if not sequence_parallel_is_enabled():
        raise RuntimeError("Sequence parallel is not enabled.")
    return state


def split_sequence_parallel_region(
    tensor: torch.Tensor, *, dim: int = 1
) -> torch.Tensor:
    state = _require_enabled()
    if tensor.shape[dim] % state.size != 0:
        raise ValueError(
            f"Cannot split tensor dimension {dim} with length {tensor.shape[dim]} "
            f"across sequence_parallel size {state.size}."
        )
    return tensor.chunk(state.size, dim=dim)[state.rank].contiguous()


class _AllToAll(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_: torch.Tensor, scatter_dim: int, gather_dim: int):
        state = _require_enabled()
        ctx.scatter_dim = int(scatter_dim)
        ctx.gather_dim = int(gather_dim)
        ctx.group = state.group
        ctx.size = state.size

        if input_.shape[scatter_dim] % state.size != 0:
            raise ValueError(
                f"Cannot all-to-all tensor dimension {scatter_dim} with length "
                f"{input_.shape[scatter_dim]} across sequence_parallel size {state.size}."
            )
        input_chunks = [
            chunk.contiguous() for chunk in input_.chunk(state.size, dim=scatter_dim)
        ]
        output_chunks = [torch.empty_like(input_chunks[0]) for _ in range(state.size)]
        dist.all_to_all(output_chunks, input_chunks, group=state.group)
        return torch.cat(output_chunks, dim=gather_dim).contiguous()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_chunks = [
            chunk.contiguous()
            for chunk in grad_output.chunk(ctx.size, dim=ctx.gather_dim)
        ]
        output_chunks = [torch.empty_like(grad_chunks[0]) for _ in range(ctx.size)]
        dist.all_to_all(output_chunks, grad_chunks, group=ctx.group)
        grad_input = torch.cat(output_chunks, dim=ctx.scatter_dim).contiguous()
        return grad_input, None, None


def all_to_all_sequence_parallel(
    tensor: torch.Tensor,
    *,
    scatter_dim: int,
    gather_dim: int,
) -> torch.Tensor:
    return _AllToAll.apply(tensor, int(scatter_dim), int(gather_dim))


class _GatherSequence(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_: torch.Tensor, dim: int):
        state = _require_enabled()
        ctx.dim = int(dim)
        ctx.rank = state.rank
        ctx.size = state.size
        ctx.group = state.group

        chunks = [torch.empty_like(input_) for _ in range(state.size)]
        dist.all_gather(chunks, input_.contiguous(), group=state.group)
        return torch.cat(chunks, dim=dim).contiguous()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        grad_output = grad_output.contiguous().clone()
        dist.all_reduce(grad_output, group=ctx.group)
        return grad_output.chunk(ctx.size, dim=ctx.dim)[ctx.rank].contiguous(), None


def gather_sequence_parallel_region(
    tensor: torch.Tensor, *, dim: int = 1
) -> torch.Tensor:
    return _GatherSequence.apply(tensor, int(dim))


def broadcast_sequence_parallel_tensor(
    tensor: torch.Tensor | None,
    *,
    source_sp_rank: int = 0,
    device: torch.device | str | None = None,
    required: bool = True,
    max_ndim: int = 8,
) -> torch.Tensor | None:
    """Broadcast a full tensor from one rank inside the SP group.

    This is for non-autograd payloads such as batch-side conditioning tensors.
    The source rank sends tensor dtype/shape metadata first so non-source ranks
    can allocate the right output shape even when the batch has variable
    piece counts.
    """
    state = _require_enabled()
    source_rank = sequence_parallel_source_global_rank(source_sp_rank)
    is_source = int(dist.get_rank()) == source_rank
    max_ndim = int(max_ndim)

    if device is None:
        if torch.is_tensor(tensor):
            device = tensor.device
        elif torch.cuda.is_available():
            device = torch.device("cuda", torch.cuda.current_device())
        else:
            device = torch.device("cpu")
    device = torch.device(device)

    meta = torch.empty(2 + max_ndim, dtype=torch.long, device=device)
    out = None
    if is_source:
        if torch.is_tensor(tensor):
            out = tensor.to(device=device).contiguous()
            dtype_code = _TENSOR_BROADCAST_DTYPE_TO_CODE.get(out.dtype)
            if dtype_code is None:
                raise TypeError(
                    f"Unsupported dtype for sequence-parallel broadcast: {out.dtype}"
                )
            if out.ndim > max_ndim:
                raise ValueError(
                    "Cannot sequence-parallel broadcast tensor with "
                    f"ndim={out.ndim}; max supported is {max_ndim}."
                )
            meta.fill_(-1)
            meta[0] = int(dtype_code)
            meta[1] = int(out.ndim)
            if out.ndim > 0:
                meta[2 : 2 + out.ndim] = torch.tensor(
                    list(out.shape), dtype=torch.long, device=device
                )
        else:
            meta.fill_(-1)

    dist.broadcast(meta, src=source_rank, group=state.group)
    dtype_code = int(meta[0].item())
    if dtype_code < 0:
        if required:
            raise RuntimeError(
                "sequence-parallel broadcast source rank "
                f"{source_rank} did not provide a tensor."
            )
        return None

    dtype = _TENSOR_BROADCAST_CODE_TO_DTYPE.get(dtype_code)
    if dtype is None:
        raise RuntimeError(
            f"Received unsupported dtype code for sequence-parallel broadcast: {dtype_code}."
        )
    ndim = int(meta[1].item())
    if ndim < 0 or ndim > max_ndim:
        raise RuntimeError(
            f"Received invalid ndim for sequence-parallel broadcast: {ndim}."
        )
    shape = tuple(int(v) for v in meta[2 : 2 + ndim].detach().cpu().tolist())

    if not is_source:
        out = torch.empty(shape, dtype=dtype, device=device)
    assert out is not None
    dist.broadcast(out, src=source_rank, group=state.group)
    return out


def broadcast_sequence_parallel_tensor_dict(
    tensor_dict: dict[str, Any],
    keys,
    *,
    source_sp_rank: int = 0,
    device: torch.device | str | None = None,
    required: bool = True,
) -> None:
    """Broadcast selected tensor values in-place within the SP group."""
    for key in keys:
        tensor_dict[key] = broadcast_sequence_parallel_tensor(
            tensor_dict.get(key),
            source_sp_rank=source_sp_rank,
            device=device,
            required=required,
        )
