"""Minimal synchronous 1P1D runtime for the opt-in SHM backends."""

from __future__ import annotations

import multiprocessing as mp
import os
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


def _worker(role, args, model_args, commands, results, errors, ring_name, ring_size):
    engine = ring = None
    pending = {}
    try:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.pd_devices[role == "decode"])
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
        ring = ShmRing(ring_size, args.disaggregation_shm_slot_count, ring_name, False)
        if args.disaggregation_transfer_backend == "shm_pinned":
            ring.register_cuda()
        results.put((role, "READY"))
        while True:
            cmd = commands.get()
            if cmd.name == "ADD":
                ids, sampling, uid = cmd.payload
                scheduler.add_request(ids, sampling, uid=uid)
            elif cmd.name == "ABORT":
                scheduler.abort_request(cmd.payload[0])
            elif cmd.name == "TRANSFER":
                ids, sampling, uid, slot, length, meta_bytes, offset, total_pages = (
                    cmd.payload
                )
                meta = TransferMeta.unpack(meta_bytes)
                dtype = getattr(torch, meta.dtype.removeprefix("torch."))
                source = torch.frombuffer(
                    bytearray(ring.get(slot, length)), dtype=dtype
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
            elif cmd.name == "STEP":
                outputs = scheduler.step(prefill_only=role == "prefill")
                transfers = []
                if role == "prefill":
                    for uid in {out.uid for out in outputs if not out.finished}:
                        req = next(
                            r for r in scheduler.prefill_manager.running if r.uid == uid
                        )
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
                            raw = tensor.contiguous().numpy().tobytes()
                            meta = TransferMeta(
                                TransferKey("session", uid),
                                tuple(tensor.shape),
                                str(tensor.dtype),
                                tuple(chunk_pages),
                                len(req.input_ids),
                            )
                            ring.put(slot, raw)
                            transfers.append(
                                (
                                    uid,
                                    slot,
                                    len(raw),
                                    meta.pack(),
                                    req.input_ids,
                                    req.sampling_params,
                                    offset,
                                    len(pages),
                                )
                            )
                        scheduler.release_request(uid)
                        break
                results.put((role, "STEP", outputs, transfers))
            elif cmd.name == "SHUTDOWN":
                break
    except BaseException as exc:
        errors.put((role, repr(exc)))
    finally:
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
        self._ring = ShmRing(ring_size, args.disaggregation_shm_slot_count)
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
                    self._ring.name,
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
                role, state = self._results.get(timeout=0.1)
                if state == "READY":
                    ready.add(role)
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
        _, state, outputs, transfers = self._results.get(timeout=self.args.pd_timeout)
        if state != "STEP":
            raise RuntimeError(f"unexpected PD message: {state}")
        for transfer in transfers:
            uid, slot, length, meta, ids, sampling, offset, total_pages = transfer
            self._commands[1].put(
                _Command(
                    "TRANSFER",
                    (ids, sampling, uid, slot, length, meta, offset, total_pages),
                )
            )
            self._results.get(timeout=self.args.pd_timeout)
        if self._active and not transfers:
            self._commands[1].put(_Command("STEP"))
            _, state, outputs, _ = self._results.get(timeout=self.args.pd_timeout)
            if state != "STEP":
                raise RuntimeError(f"unexpected PD message: {state}")
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
            commands.put(_Command("SHUTDOWN"))
        for worker in self._workers:
            worker.join(timeout=5)
            if worker.is_alive():
                worker.terminate()
        self._ring.close(unlink=True)
