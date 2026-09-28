"""Collective communication primitives with NCCL/HCCL dual backends.

Uses torch.distributed as the backend for collective operations
(all_reduce, all_gather) used in tensor parallelism, covering both NCCL
(GPU) and HCCL (NPU) backends.
"""

__all__ = ["all_gather", "all_reduce", "broadcast", "barrier"]
import torch
import torch.distributed as dist

from minisgl.utils.device import get_tp_group, get_tp_size, is_distributed


def all_reduce(tensor: torch.Tensor, op: str = "sum") -> torch.Tensor:
    """All-reduce across the explicit TP group."""
    if op not in ("sum", "min"):
        raise ValueError(f"Unsupported all_reduce op: {op!r} (use 'sum' or 'min')")
    if not is_distributed():
        return tensor
    reduce_op = dist.ReduceOp.SUM if op == "sum" else dist.ReduceOp.MIN
    dist.all_reduce(tensor, op=reduce_op, group=get_tp_group())
    return tensor


def all_gather(tensor: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """All-gather across TP ranks along given dimension."""
    if not is_distributed():
        return tensor
    tp_size = get_tp_size()
    chunks = [torch.empty_like(tensor) for _ in range(tp_size)]
    dist.all_gather(chunks, tensor, group=get_tp_group())
    return torch.cat(chunks, dim=dim)


def broadcast(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Broadcast a tensor from one TP rank."""
    if is_distributed():
        dist.broadcast(tensor, src=src, group=get_tp_group())
    return tensor


def barrier() -> None:
    """Synchronize all TP ranks."""
    if is_distributed():
        dist.barrier(group=get_tp_group())
