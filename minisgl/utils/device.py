"""Device and distributed initialization utilities.

Supports: CUDA (NVIDIA), NPU (Ascend/torch_npu), and CPU fallback.
Device priority: NPU > CUDA > CPU (configurable via set_device).
"""

__all__ = [
    "DeviceState",
    "get_device",
    "get_device_type",
    "get_tp_rank",
    "get_tp_size",
    "get_tp_group",
    "init_distributed",
    "is_accelerator_available",
    "is_distributed",
    "is_npu_available",
    "mem_get_info",
    "reset_device_state",
    "set_device",
    "synchronize",
    "barrier",
]
import os
from dataclasses import dataclass

import torch
import torch.distributed as dist

_NPU_AVAILABLE: bool | None = None


def is_npu_available() -> bool:
    """Check if Ascend NPU is available via torch_npu."""
    global _NPU_AVAILABLE
    if _NPU_AVAILABLE is None:
        try:
            import torch_npu  # noqa: F401

            _NPU_AVAILABLE = torch.npu.is_available()
        except (ImportError, RuntimeError, AttributeError):
            _NPU_AVAILABLE = False
    return _NPU_AVAILABLE


def is_accelerator_available() -> bool:
    """Check if any accelerator (NPU or CUDA) is available."""
    return is_npu_available() or torch.cuda.is_available()


def get_device_type() -> str:
    """Return the best available device type string: 'npu', 'cuda', or 'cpu'."""
    if is_npu_available():
        return "npu"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


@dataclass
class DeviceState:
    """Holds tensor-parallel state to avoid module-level globals.

    Use `reset_device_state()` to clear state between sessions.
    """

    tp_rank: int = 0
    tp_size: int = 1
    device: torch.device | None = None
    tp_group: object | None = None


_state = DeviceState()


def reset_device_state() -> None:
    """Reset device state for clean multi-instance support."""
    global _state
    if dist.is_initialized():
        dist.destroy_process_group()
    _state = DeviceState()


def get_tp_rank() -> int:
    return _state.tp_rank


def get_tp_size() -> int:
    return _state.tp_size


def get_tp_group():
    """Return the explicit tensor-parallel process group, if initialized."""
    return _state.tp_group


def get_device() -> torch.device:
    if _state.device is None:
        device_type = get_device_type()
        if device_type == "cuda":
            # Several CUDA APIs, including mem_get_info, require an explicit
            # device index.  ``torch.device("cuda")`` is valid for tensor
            # placement but is not accepted by all runtime APIs.
            _state.device = torch.device("cuda", torch.cuda.current_device())
        elif device_type == "npu":
            # Keep NPU behavior consistent with CUDA when the backend exposes
            # current_device().  Older torch_npu versions may not, so retain
            # the type-only device as a fallback.
            current_device = getattr(torch.npu, "current_device", None)
            if current_device is None:
                _state.device = torch.device("npu")
            else:
                _state.device = torch.device("npu", current_device())
        else:
            _state.device = torch.device(device_type)
    return _state.device


def set_device(device: torch.device) -> None:
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    elif device.type == "npu" and device.index is None:
        current_device = getattr(torch.npu, "current_device", None)
        if current_device is not None:
            device = torch.device("npu", current_device())
    _state.device = device
    if device.type == "cuda":
        torch.cuda.set_device(device)
    elif device.type == "npu":
        torch.npu.set_device(device)


def synchronize() -> None:
    """Synchronize the current accelerator device."""
    if _state.device is None:
        return
    if _state.device.type == "cuda":
        torch.cuda.synchronize()
    elif _state.device.type == "npu":
        torch.npu.synchronize()


def mem_get_info(device: torch.device | None = None) -> tuple[int, int]:
    """Get (free, total) memory in bytes for the current accelerator.

    Returns (0, 0) for CPU.
    """
    dev = device or get_device()
    if dev.type == "cuda":
        return torch.cuda.mem_get_info(dev)
    elif dev.type == "npu":
        free = torch.npu.mem_get_info(dev)[0]
        total = torch.npu.mem_get_info(dev)[1]
        return free, total
    return 0, 0


def init_distributed(
    tp_rank: int = 0,
    tp_size: int = 1,
    backend: str | None = None,
    master_addr: str = "127.0.0.1",
    master_port: int = 29500,
) -> None:
    """Initialize distributed process group.

    Auto-selects backend: 'hccl' for NPU, 'nccl' for CUDA, 'gloo' for CPU.
    """
    if tp_size < 1:
        raise ValueError("tp_size must be positive")
    if not 0 <= tp_rank < tp_size:
        raise ValueError(f"tp_rank={tp_rank} outside tp_size={tp_size}")
    _state.tp_rank = tp_rank
    _state.tp_size = tp_size

    if tp_size > 1:
        if backend is None:
            dev_type = get_device_type()
            if dev_type == "npu":
                backend = "hccl"
            elif dev_type == "cuda":
                backend = "nccl"
            else:
                backend = "gloo"

        os.environ.setdefault("MASTER_ADDR", master_addr)
        os.environ.setdefault("MASTER_PORT", str(master_port))
        local_rank = int(os.environ.get("LOCAL_RANK", tp_rank))
        dev_type = get_device_type()
        if dev_type == "cuda":
            set_device(torch.device("cuda", local_rank))
        elif dev_type == "npu":
            set_device(torch.device("npu", local_rank))
        dist.init_process_group(
            backend=backend,
            init_method="env://",
            rank=int(os.environ.get("RANK", tp_rank)),
            world_size=int(os.environ.get("WORLD_SIZE", tp_size)),
        )
        _state.tp_group = dist.new_group(ranks=list(range(tp_size)))


def barrier() -> None:
    """Synchronize TP workers; a no-op in single-process mode."""
    if is_distributed():
        dist.barrier(group=_state.tp_group)


def is_distributed() -> bool:
    return _state.tp_size > 1 and dist.is_initialized()
