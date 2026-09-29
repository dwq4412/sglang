"""Collective communication primitives with NCCL/HCCL dual backends.

Uses torch.distributed as the backend for collective operations
(all_reduce, all_gather) used in tensor parallelism, covering both NCCL
(GPU) and HCCL (NPU) backends.
"""

__all__ = [
    "all_gather",
    "all_reduce",
    "all_reduce_chunked",
    "broadcast",
    "barrier",
    "configure",
    "get_stats",
    "reset_stats",
]
import time

import torch
import torch.distributed as dist

from minisgl.utils.device import get_tp_group, get_tp_size, is_distributed

_mode = "legacy"
_stats_enabled = False
_stats: dict[str, dict[str, float]] = {}
_gather_buffer: tuple[tuple, torch.Tensor] | None = None
_gather_workspaces: dict[tuple, torch.Tensor] = {}
_ar_chunk_bytes = 0


def configure(mode: str = "legacy", stats: bool = False, ar_chunk_bytes: int = 0) -> None:
    """Select the collective implementation for the current process."""
    global _mode, _stats_enabled, _gather_buffer, _gather_workspaces, _ar_chunk_bytes
    if mode not in ("legacy", "tensor", "buffered", "adaptive"):
        raise ValueError(f"Unsupported collective mode: {mode!r}")
    _mode = mode
    _stats_enabled = stats
    _gather_buffer = None
    _gather_workspaces = {}
    _ar_chunk_bytes = max(0, ar_chunk_bytes)


def reset_stats() -> None:
    _stats.clear()


def get_stats() -> dict[str, dict[str, float]]:
    return {name: values.copy() for name, values in _stats.items()}


def _record(name: str, payload_bytes: int, elapsed: float) -> None:
    if not _stats_enabled:
        return
    values = _stats.setdefault(name, {"calls": 0, "bytes": 0, "elapsed_ms": 0})
    values["calls"] += 1
    values["bytes"] += payload_bytes
    values["elapsed_ms"] += elapsed * 1000


def all_reduce(tensor: torch.Tensor, op: str = "sum", group=None) -> torch.Tensor:
    """All-reduce across the explicit TP group."""
    if op not in ("sum", "min", "max"):
        raise ValueError(f"Unsupported all_reduce op: {op!r} (use sum, min, or max)")
    if not is_distributed():
        return tensor
    reduce_op = {
        "sum": dist.ReduceOp.SUM,
        "min": dist.ReduceOp.MIN,
        "max": dist.ReduceOp.MAX,
    }[op]
    start = time.perf_counter() if _stats_enabled else 0
    dist.all_reduce(
        tensor,
        op=reduce_op,
        group=group if group is not None else get_tp_group(),
    )
    _record(
        "all_reduce",
        tensor.numel() * tensor.element_size(),
        time.perf_counter() - start if start else 0,
    )
    return tensor


def all_reduce_chunked(tensor: torch.Tensor, group=None) -> torch.Tensor:
    """Reduce a large tensor in a few async chunks, otherwise use the baseline."""
    if not is_distributed() or _mode != "adaptive" or _ar_chunk_bytes <= 0:
        return all_reduce(tensor, group=group)
    flat = tensor.reshape(-1)
    chunk_elems = max(1, _ar_chunk_bytes // tensor.element_size())
    chunks = min(4, (flat.numel() + chunk_elems - 1) // chunk_elems)
    if chunks <= 1 or not hasattr(dist, "all_reduce"):
        return all_reduce(tensor, group=group)
    group = group if group is not None else get_tp_group()
    works = []
    original = tensor.clone()
    start = time.perf_counter() if _stats_enabled else 0
    for chunk in flat.tensor_split(chunks):
        try:
            works.append(
                dist.all_reduce(
                    chunk, op=dist.ReduceOp.SUM, group=group, async_op=True
                )
            )
        except (TypeError, RuntimeError):
            for work in works:
                if hasattr(work, "wait"):
                    work.wait()
            tensor.copy_(original)
            return all_reduce(tensor, group=group)
    try:
        for work in works:
            work.wait()
    except (AttributeError, RuntimeError):
        tensor.copy_(original)
        return all_reduce(tensor, group=group)
    _record(
        "all_reduce",
        tensor.numel() * tensor.element_size(),
        time.perf_counter() - start if start else 0,
    )
    return tensor


def all_gather(tensor: torch.Tensor, dim: int = -1, group=None) -> torch.Tensor:
    """All-gather across TP ranks along given dimension."""
    if not is_distributed():
        return tensor
    if not isinstance(tensor, torch.Tensor):
        raise TypeError("all_gather expects a torch.Tensor")
    if tensor.dim() == 0 or tensor.numel() == 0:
        raise ValueError("all_gather expects a non-empty tensor with at least one dimension")
    dim = dim if dim >= 0 else tensor.dim() + dim
    if not 0 <= dim < tensor.dim():
        raise ValueError(f"Invalid gather dimension {dim} for shape {tuple(tensor.shape)}")
    tp_size = get_tp_size()
    start = time.perf_counter() if _stats_enabled else 0
    if _mode == "legacy":
        chunks = [torch.empty_like(tensor) for _ in range(tp_size)]
        dist.all_gather(
            chunks,
            tensor.contiguous(),
            group=group if group is not None else get_tp_group(),
        )
        result = torch.cat(chunks, dim=dim)
    else:
        local = tensor.movedim(dim, 0).contiguous()
        key = (tuple(local.shape), local.dtype, local.device, tp_size)
        global _gather_buffer
        if _mode == "buffered" and _gather_buffer is not None and _gather_buffer[0] == key:
            output = _gather_buffer[1]
        elif _mode == "adaptive" and key in _gather_workspaces:
            output = _gather_workspaces[key]
        else:
            output = torch.empty(
                (tp_size * local.shape[0], *local.shape[1:]),
                dtype=local.dtype,
                device=local.device,
            )
            if _mode == "buffered":
                _gather_buffer = (key, output)
            elif _mode == "adaptive":
                _gather_workspaces[key] = output
        dist.all_gather_into_tensor(
            output,
            local,
            group=group if group is not None else get_tp_group(),
        )
        result = output.movedim(0, dim)
    _record(
        "all_gather",
        tensor.numel() * tensor.element_size(),
        time.perf_counter() - start if start else 0,
    )
    return result


def broadcast(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    """Broadcast a tensor from one TP rank."""
    if is_distributed():
        start = time.perf_counter() if _stats_enabled else 0
        dist.broadcast(tensor, src=src, group=get_tp_group())
        _record(
            "broadcast",
            tensor.numel() * tensor.element_size(),
            time.perf_counter() - start if start else 0,
        )
    return tensor


def barrier() -> None:
    """Synchronize all TP ranks."""
    if is_distributed():
        start = time.perf_counter() if _stats_enabled else 0
        dist.barrier(group=get_tp_group())
        _record("barrier", 0, time.perf_counter() - start if start else 0)
