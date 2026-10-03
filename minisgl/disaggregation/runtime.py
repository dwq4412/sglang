"""Minimal synchronous 1P1D runtime for the opt-in transfer backends."""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import queue
import time
from dataclasses import dataclass
from typing import Any

import torch

from minisgl.config import ModelArgs, SamplingParams, ServerArgs
from minisgl.disaggregation.protocol import TransferKey, TransferMeta
from minisgl.disaggregation.shm_ring import ShmRing
from minisgl.scheduler.batch import OutputToken


@dataclass
class _Command:
    name: str
    payload: tuple[Any, ...] = ()


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    """Serialize tensor storage without requiring NumPy to support its dtype."""
    contiguous = tensor.detach().contiguous()
    return contiguous.view(torch.uint8).numpy().tobytes()


def _worker(role, args, model_args, commands, results, errors, ring_name, ring_size):
    engine = ring = nixl = None
    nixl_peer = None
    pending = {}
    try:
        device_index = args.pd_devices[role == "decode"]
        if torch.cuda.is_available():
            # Each PD worker sees the original physical device namespace. Do
            # not change CUDA_VISIBLE_DEVICES here: by the time a spawned
            # worker imports this module, CUDA may already have been queried.
            # Explicitly selecting the physical device is unambiguous and
            # prevents both workers from defaulting to device 0.
            torch.cuda.set_device(device_index)
            # logger.info(
            #     "PD worker role=%s pid=%d using cuda:%d (current=%d)",
            #     role,
            #     os.getpid(),
            #     device_index,
            #     torch.cuda.current_device(),
            # )
        from minisgl.engine.engine import Engine
        from minisgl.scheduler.scheduler import Scheduler

        worker_args = ServerArgs(
            **{
                **args.__dict__,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "cuda_graph_bs": 0,
            }
        )
        engine = Engine(worker_args, model_args, tp_rank=0)
        scheduler = Scheduler(worker_args, engine, cache_strategy="naive")
        if args.disaggregation_transfer_backend == "nixl":
            from minisgl.disaggregation.nixl_backend import NixlEndpoint

            nixl = NixlEndpoint(
                engine.kv_cache_pool,
                device_index,
                args.pd_timeout,
                agent_name=f"mini-sglang-{role}-{device_index}",
            )
        if args.disaggregation_transfer_backend in ("shm", "hssm"):
            ring = ShmRing(
                ring_size, args.disaggregation_shm_slot_count, ring_name, False
            )
        if args.disaggregation_transfer_backend == "hssm":
            ring.register_cuda()
        results.put((role, "READY", nixl.metadata() if nixl else None))
        # logger.info("PD worker role=%s READY", role)
        while True:
            cmd = commands.get()
            if cmd.name == "ADD":
                ids, sampling, uid = cmd.payload
                scheduler.add_request(ids, sampling, uid=uid)
            elif cmd.name == "ABORT":
                scheduler.abort_request(cmd.payload[0])
            elif cmd.name == "PEER":
                if nixl is None:
                    raise RuntimeError("PEER is only valid for the NIXL backend")
                nixl_peer = nixl.add_peer(cmd.payload[0])
                results.put((role, "PEER", nixl_peer))
            elif cmd.name == "RELEASE":
                scheduler.release_request(cmd.payload[0])
            elif cmd.name == "TRANSFER":
                ids, sampling, uid, payload, meta_bytes, offset, total_pages = (
                    cmd.payload
                )
                meta = TransferMeta.unpack(meta_bytes)
                if args.disaggregation_transfer_backend == "nixl":
                    if role != "decode" or nixl is None:
                        raise RuntimeError("NIXL transfer must be received by decode")
                    source_pages = list(payload)
                    handle = pending.get(uid)
                    if handle is None:
                        handle = engine.kv_cache_pool.alloc(total_pages)
                        pending[uid] = handle
                    destination_pages = handle.page_ids[
                        offset : offset + len(meta.page_ids)
                    ]
                    nixl.write(
                        nixl_peer,
                        nixl.remote_pages(nixl_peer, source_pages),
                        nixl.pages(destination_pages),
                    )
                elif args.disaggregation_transfer_backend == "queue":
                    dtype = getattr(torch, meta.dtype.removeprefix("torch."))
                    raw = payload
                else:
                    dtype = getattr(torch, meta.dtype.removeprefix("torch."))
                    slot, length = payload
                if args.disaggregation_transfer_backend != "nixl":
                    source = torch.frombuffer(
                        bytearray(raw), dtype=dtype
                    ).reshape(meta.shape)
                    handle = pending.get(uid)
                    if handle is None:
                        handle = engine.kv_cache_pool.alloc(total_pages)
                        pending[uid] = handle
                    target = engine.kv_cache_pool.buffer[
                        :, :, handle.page_ids[offset : offset + len(meta.page_ids)]
                    ]
                    target.copy_(source.to(device=target.device), non_blocking=False)
                if offset + len(meta.page_ids) == total_pages:
                    scheduler.add_transferred_request(
                        ids, sampling, uid, handle.page_ids
                    )
                    del pending[uid]
                results.put((role, "TRANSFERRED", uid))
                # logger.info(
                #     "PD worker role=%s transferred uid=%s pages=%d/%d",
                #     role,
                #     uid,
                #     offset,
                #     len(pages),
                # )
            elif cmd.name == "STEP":
                # logger.info("PD worker role=%s starting STEP", role)
                outputs = scheduler.step(prefill_only=role == "prefill")
                transfers = []
                if role == "prefill":
                    # Prefill produces the first output token. That token may
                    # also finish the request (max_tokens=1 or EOS), so using
                    # ``not out.finished`` to find transfer candidates loses
                    # the KV cache for exactly those valid requests. Requests
                    # that still need decode are the ones left in running.
                    for req in list(scheduler.prefill_manager.running):
                        uid = req.uid
                        pages = list(req.cache_handle.page_ids)
                        per_chunk = max(
                            1, args.disaggregation_shm_chunk_tokens // args.page_size
                        )
                        for chunk_index, offset in enumerate(
                            range(0, len(pages), per_chunk)
                        ):
                            slot = chunk_index % args.disaggregation_shm_slot_count
                            chunk_pages = pages[offset : offset + per_chunk]
                            tensor = engine.kv_cache_pool.buffer[
                                :, :, chunk_pages
                            ].detach()
                            if tensor.device.type != "cpu":
                                tensor = tensor.to("cpu")
                            # NumPy does not support a BF16 view in some
                            # PyTorch/NumPy combinations. Serialize the
                            # tensor as raw bytes instead; the metadata keeps
                            # the original dtype for the decode worker.
                            raw = _tensor_bytes(tensor)
                            meta = TransferMeta(
                                TransferKey("session", uid),
                                tuple(tensor.shape),
                                str(tensor.dtype),
                                tuple(chunk_pages),
                                len(req.input_ids),
                            )
                            if args.disaggregation_transfer_backend == "nixl":
                                payload = tuple(chunk_pages)
                            elif args.disaggregation_transfer_backend == "queue":
                                payload = raw
                            else:
                                ring.put(slot, raw)
                                payload = (slot, len(raw))
                            transfers.append(
                                (
                                    uid,
                                    payload,
                                    meta.pack(),
                                    req.input_ids,
                                    req.sampling_params,
                                    offset,
                                    len(pages),
                                )
                            )
                        if args.disaggregation_transfer_backend != "nixl":
                            scheduler.release_request(uid)
                        break
                results.put((role, "STEP", outputs, transfers))
                # logger.info(
                #     "PD worker role=%s finished STEP outputs=%d transfers=%d",
                #     role,
                #     len(outputs),
                #     len(transfers),
                # )
            elif cmd.name == "SHUTDOWN":
                break
    except BaseException as exc:
        errors.put((role, repr(exc)))
    finally:
        if nixl is not None:
            with contextlib.suppress(Exception):
                nixl.cleanup()
        if ring is not None:
            ring.close()
        if engine is not None:
            engine.cleanup()


class PDSchedulerProxy:
    """Scheduler-compatible coordinator; the original runtime is untouched."""

    def __init__(self, args: ServerArgs, model_args: ModelArgs):
        if args.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("PD CUDA backend requires CUDA")
        self.args = args
        self._ctx = mp.get_context("spawn")
        self._commands = [self._ctx.Queue(), self._ctx.Queue()]
        self._results, self._errors = self._ctx.Queue(), self._ctx.Queue()
        self._ready_metadata = {}
        self._active, self._next_uid, self._closed = set(), 0, False
        element_size = 2 if args.dtype in ("float16", "bfloat16") else 4
        chunk_pages = max(
            1,
            (args.disaggregation_shm_chunk_tokens + args.page_size - 1)
            // args.page_size,
        )
        ring_size = (
            2
            * model_args.num_layers
            * chunk_pages
            * args.page_size
            * model_args.num_kv_heads
            * model_args.head_dim
            * element_size
        )
        self._ring = (
            ShmRing(ring_size, args.disaggregation_shm_slot_count)
            if args.disaggregation_transfer_backend in ("shm", "hssm")
            else None
        )
        self._workers = [
            self._ctx.Process(
                target=_worker,
                args=(
                    role,
                    args,
                    model_args,
                    self._commands[i],
                    self._results,
                    self._errors,
                    self._ring.name if self._ring is not None else None,
                    ring_size,
                ),
            )
            for i, role in enumerate(("prefill", "decode"))
        ]
        for worker in self._workers:
            worker.start()
        self._wait_ready()

    def _wait_ready(self):
        ready = set()
        deadline = time.monotonic() + self.args.pd_timeout
        while len(ready) < 2:
            if time.monotonic() >= deadline:
                self.cleanup()
                raise TimeoutError("timed out starting PD workers")
            try:
                message = self._results.get(timeout=0.1)
                role, state = message[:2]
                if state == "READY":
                    ready.add(role)
                    if self.args.disaggregation_transfer_backend == "nixl":
                        self._ready_metadata[role] = message[2]
                        if len(ready) == 2:
                            for index, peer_role in enumerate(("prefill", "decode")):
                                other = "decode" if peer_role == "prefill" else "prefill"
                                self._commands[index].put(
                                    _Command("PEER", (self._ready_metadata[other],))
                                )
                            for _ in range(2):
                                peer_message = self._results.get(timeout=0.1)
                                if peer_message[1] != "PEER":
                                    raise RuntimeError("unexpected NIXL peer handshake")
            except queue.Empty:
                self._check_workers()

    def _check_workers(self):
        try:
            role, error = self._errors.get_nowait()
        except queue.Empty:
            error = None
        if error:
            self.cleanup()
            raise RuntimeError(f"PD {role} worker failed: {error}")
        if not all(worker.is_alive() for worker in self._workers):
            self.cleanup()
            raise RuntimeError("PD worker exited unexpectedly")

    def _get_result(self, timeout: float):
        """Wait for a protocol result while surfacing worker failures promptly."""
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._check_workers()
                raise TimeoutError("timed out waiting for PD worker result")
            try:
                return self._results.get(timeout=min(0.1, remaining))
            except queue.Empty:
                self._check_workers()

    def add_request(self, input_ids: list[int], sampling_params: SamplingParams) -> int:
        uid = self._next_uid
        self._next_uid += 1
        self._active.add(uid)
        self._commands[0].put(_Command("ADD", (list(input_ids), sampling_params, uid)))
        return uid

    def abort_request(self, uid: int) -> bool:
        if uid not in self._active:
            return False
        for commands in self._commands:
            commands.put(_Command("ABORT", (uid,)))
        self._active.discard(uid)
        return True

    def step(self) -> list[OutputToken]:
        self._commands[0].put(_Command("STEP"))
        _, state, outputs, transfers = self._get_result(self.args.pd_timeout)
        if state != "STEP":
            raise RuntimeError(f"unexpected PD message: {state}")
        # A request can finish during prefill (for example with
        # max_tokens=1). It has no decode KV transfer to wait for, so retire it
        # from the coordinator immediately.
        for output in outputs:
            if output.finished:
                self._active.discard(output.uid)
        for transfer in transfers:
            uid, payload, meta, ids, sampling, offset, total_pages = transfer
            self._commands[1].put(
                _Command(
                    "TRANSFER",
                    (ids, sampling, uid, payload, meta, offset, total_pages),
                )
            )
            role, state, transferred_uid = self._get_result(self.args.pd_timeout)
            if role != "decode" or state != "TRANSFERRED" or transferred_uid != uid:
                raise RuntimeError(
                    "unexpected PD transfer result: "
                    f"role={role} state={state} uid={transferred_uid} expected={uid}"
                )
            if self.args.disaggregation_transfer_backend == "nixl":
                self._commands[0].put(_Command("RELEASE", (uid,)))
        if self._active and not transfers:
            self._commands[1].put(_Command("STEP"))
            role, state, outputs, _ = self._get_result(self.args.pd_timeout)
            if role != "decode" or state != "STEP":
                raise RuntimeError(
                    f"unexpected PD decode result: role={role} state={state}"
                )
        for output in outputs:
            if output.finished:
                self._active.discard(output.uid)
        return outputs

    def is_idle(self) -> bool:
        return not self._active

    def cleanup(self):
        if self._closed:
            return
        self._closed = True
        for commands in self._commands:
            with contextlib.suppress(Exception):
                commands.put_nowait(_Command("SHUTDOWN"))
        for worker in self._workers:
            # A worker can be blocked in model loading or CUDA allocation and
            # therefore cannot consume SHUTDOWN. Keep failure cleanup short so
            # the sibling worker cannot retain its model after the other one
            # has failed during startup.
            worker.join(timeout=1)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=5)
            if worker.is_alive():
                # terminate() is cooperative on some platforms. A worker
                # stuck inside CUDA allocation must not survive a failed PD
                # startup and retain its model allocation.
                worker.kill()
                worker.join(timeout=5)
        if self._ring is not None:
            self._ring.close(unlink=True)
