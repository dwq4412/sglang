"""Synchronous shared-memory slots; control ordering is carried by mp.Queue."""

from __future__ import annotations

import ctypes
import uuid
from contextlib import suppress
from multiprocessing import shared_memory


class ShmRing:
    def __init__(
        self, size: int, slots: int, name: str | None = None, create: bool = True
    ):
        if size <= 0 or slots <= 0:
            raise ValueError("shared-memory size and slot count must be positive")
        self.size, self.slots = size, slots
        self.name = name or f"minisgl_pd_{uuid.uuid4().hex}"
        total = size * slots
        self.shm = shared_memory.SharedMemory(name=self.name, create=create, size=total)
        self._registered = False

    def view(self, slot: int) -> memoryview:
        if not 0 <= slot < self.slots:
            raise IndexError(slot)
        start = slot * self.size
        return self.shm.buf[start : start + self.size]

    def put(self, slot: int, data: bytes) -> None:
        if len(data) > self.size:
            raise ValueError(f"transfer is {len(data)} bytes, slot holds {self.size}")
        view = self.view(slot)
        view[: len(data)] = data

    def get(self, slot: int, length: int) -> bytes:
        if not 0 <= length <= self.size:
            raise ValueError(length)
        return bytes(self.view(slot)[:length])

    def close(self, unlink: bool = False) -> None:
        self.unregister_cuda()
        self.shm.close()
        if unlink:
            with suppress(FileNotFoundError):
                self.shm.unlink()

    def register_cuda(self) -> None:
        """Register the mapping with CUDA so transfers can use pinned host RAM."""
        import torch

        if self._registered:
            return
        if not torch.cuda.is_available():
            raise RuntimeError("hssm requires CUDA")
        address = ctypes.addressof(ctypes.c_char.from_buffer(self.shm.buf))
        result = torch.cuda.cudart().cudaHostRegister(address, self.shm.size, 0)
        if result != 0:
            raise RuntimeError(f"cudaHostRegister failed with code {result}")
        self._registered = True

    def unregister_cuda(self) -> None:
        if not self._registered:
            return
        import torch

        address = ctypes.addressof(ctypes.c_char.from_buffer(self.shm.buf))
        torch.cuda.cudart().cudaHostUnregister(address)
        self._registered = False
