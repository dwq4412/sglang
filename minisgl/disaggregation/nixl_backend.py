"""Small NIXL/UCX adapter used by the 1P1D runtime.

NIXL is intentionally optional.  Importing this module is safe on machines
without CUDA or NIXL; constructing :class:`NixlEndpoint` reports the missing
dependency with an actionable error.
"""

from __future__ import annotations

import base64
import json
import sys
import time
import uuid
from dataclasses import dataclass


def _load_api():
    try:
        from nixl._api import nixl_agent, nixl_agent_config
    except ImportError as exc:
        raise ImportError(
            "The 'nixl' package is required for the nixl backend; "
            f"install it in the active interpreter ({sys.executable}) "
            "with NIXL/UCX support, then verify with "
            "python -c \"import nixl._api\"."
        ) from exc
    return nixl_agent, nixl_agent_config


def _read_operation():
    try:
        from nixl._bindings import nixl_xfer_op_t
    except ImportError as exc:
        raise RuntimeError(
            "NIXL exposes create_xfer_req but its operation enum is unavailable"
        ) from exc
    return nixl_xfer_op_t.NIXL_READ


@dataclass(frozen=True)
class NixlPage:
    """A page-sized registered memory range."""

    address: int
    length: int
    device: int

    def as_tuple(self) -> tuple[int, int, int]:
        """Return the descriptor shape expected by NIXL transfer APIs."""
        return self.address, self.length, self.device


class NixlEndpoint:
    """One NIXL agent and one registered KV pool.

    The public methods deliberately use plain bytes and small dataclasses so
    they can be passed through the existing multiprocessing control queues.
    """

    def __init__(self, pool, gpu_id: int, timeout: float, *, agent_name=None):
        nixl_agent, nixl_agent_config = _load_api()
        self.timeout = timeout
        self.gpu_id = gpu_id
        self.agent = nixl_agent(agent_name or f"mini-sglang-{uuid.uuid4()}",
                                nixl_agent_config(backends=[]))
        self.agent.create_backend("UCX", {})
        self._pool = pool
        self._memory = None
        self._closed = False
        self._peer_names: set[str] = set()
        self._remote_pools: dict[str, tuple[int, int, int]] = {}
        self._register_pool()

    def _register_pool(self) -> None:
        buffer = self._pool.buffer
        length = buffer.numel() * buffer.element_size()
        self._memory = self.agent.register_memory(
            [(buffer.data_ptr(), length, self.gpu_id, "")], "VRAM"
        )
        if not self._memory:
            raise RuntimeError("NIXL failed to register the GPU KV pool")

    @property
    def name(self) -> str:
        return self.agent.name

    def metadata(self) -> bytes:
        raw = self.agent.get_agent_metadata()
        if isinstance(raw, str):
            raw = raw.encode()
        return json.dumps({
            "v": 1,
            "agent": self.name,
            "metadata": base64.b64encode(raw).decode("ascii"),
            "gpu": self.gpu_id,
            "shape": list(self._pool.buffer.shape),
            "base": self._pool.buffer.data_ptr(),
            "page_bytes": self._page_bytes,
            "strides": list(self._pool.buffer.stride()),
        }, separators=(",", ":")).encode()

    def add_peer(self, metadata: bytes) -> str:
        obj = json.loads(metadata)
        if obj.get("v") != 1 or not obj.get("agent") or "metadata" not in obj:
            raise ValueError("invalid NIXL peer metadata")
        peer_name = str(obj["agent"])
        peer_metadata = base64.b64decode(obj["metadata"])
        self.agent.add_remote_agent(peer_metadata)
        self._peer_names.add(peer_name)
        self._remote_pools[peer_name] = (
            int(obj["base"]),
            int(obj["page_bytes"]),
            int(obj["gpu"]),
            tuple(int(value) for value in obj["strides"]),
        )
        return peer_name

    @property
    def _page_bytes(self) -> int:
        buffer = self._pool.buffer
        return buffer.stride(2) * buffer.element_size()

    def pages(self, page_ids: list[int]) -> list[NixlPage]:
        buffer = self._pool.buffer
        page_bytes = self._page_bytes
        return [
            NixlPage(
                buffer[plane, layer, page_id].data_ptr(), page_bytes, self.gpu_id
            )
            for page_id in page_ids
            for plane in range(buffer.shape[0])
            for layer in range(buffer.shape[1])
        ]

    def remote_pages(self, peer_name: str, page_ids: list[int]) -> list[NixlPage]:
        try:
            base, page_bytes, gpu_id, strides = self._remote_pools[peer_name]
        except KeyError as exc:
            raise ValueError(f"NIXL peer is not registered: {peer_name}") from exc
        element_size = self._pool.buffer.element_size()
        return [
            NixlPage(
                base
                + plane * strides[0] * element_size
                + layer * strides[1] * element_size
                + page_id * strides[2] * element_size,
                page_bytes,
                gpu_id,
            )
            for page_id in page_ids
            for plane in range(2)
            for layer in range(self._pool.num_layers)
        ]

    def write(self, peer_name: str, source: list[NixlPage], destination: list[NixlPage]):
        if self._closed:
            raise RuntimeError("NIXL endpoint is closed")
        if peer_name not in self._peer_names:
            raise ValueError(f"NIXL peer is not registered: {peer_name}")
        if len(source) != len(destination):
            raise ValueError("source and destination page counts differ")
        src = self.agent.get_xfer_descs([page.as_tuple() for page in source], "VRAM")
        dst = self.agent.get_xfer_descs([page.as_tuple() for page in destination], "VRAM")
        # This endpoint runs on decode and pulls data from the prefill peer.
        # NIXL_READ takes the local destination first and the remote source
        # second; using WRITE here makes UCX look for the peer's address range
        # in the local registration table.
        handle = self._create_xfer(dst, src, peer_name)
        if not handle or self.agent.transfer(handle) == "ERR":
            raise RuntimeError("NIXL failed to post UCX READ")
        deadline = time.monotonic() + self.timeout
        try:
            while True:
                state = self.agent.check_xfer_state(handle)
                if state == "DONE":
                    return
                if state == "ERR":
                    raise RuntimeError("NIXL UCX READ failed")
                if time.monotonic() >= deadline:
                    raise TimeoutError("timed out waiting for NIXL UCX READ")
                time.sleep(0.001)
        finally:
            self.agent.release_xfer_handle(handle)

    def _create_xfer(self, source, destination, peer_name: str):
        """Create a transfer across both supported NIXL Python APIs.

        Recent bindings expose ``create_xfer_req`` and require the enum
        operation plus descriptor lists. Older bindings exposed
        ``initialize_xfer`` and accepted the string ``"READ"`` instead.
        """
        create = getattr(self.agent, "create_xfer_req", None)
        if create is None:
            create = getattr(self.agent, "createXferReq", None)
        if create is not None:
            return create(
                _read_operation(),
                source,
                destination,
                peer_name,
            )

        initialize = getattr(self.agent, "initialize_xfer", None)
        if initialize is None:
            raise RuntimeError("NIXL agent has no supported transfer creation method")
        return initialize("READ", source, destination, peer_name)

    def cleanup(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._memory is not None:
            deregister = getattr(self.agent, "deregister_memory", None)
            if deregister is not None:
                deregister(self._memory)
        cleanup = getattr(self.agent, "cleanup", None)
        if cleanup is not None:
            cleanup()
