"""Single-node tensor-parallel worker runtime.

The public process owns tokenization and request queues.  One spawned worker
owns each GPU and runs the normal Engine/Scheduler pair.  Commands are copied
to every worker so all schedulers make identical allocation decisions.
"""

from __future__ import annotations

import multiprocessing as mp
import os
import queue
import socket
from dataclasses import dataclass
from typing import Any

from minisgl.config import ModelArgs, SamplingParams, ServerArgs
from minisgl.scheduler.batch import OutputToken

__all__ = ["DistributedSchedulerProxy", "create_scheduler_runtime"]


@dataclass
class _Command:
    name: str
    payload: tuple[Any, ...] = ()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _worker(
    rank: int,
    args: ServerArgs,
    model_args: ModelArgs,
    commands,
    results,
    errors,
    ready,
    master_port: int,
) -> None:
    """Worker entry point; kept top-level so Windows spawn can pickle it."""
    try:
        os.environ.update(
            {
                "RANK": str(rank),
                "LOCAL_RANK": str(rank),
                "WORLD_SIZE": str(args.tp_size),
                "MASTER_ADDR": "127.0.0.1",
                "MASTER_PORT": str(master_port),
            }
        )
        from minisgl.engine.engine import Engine
        from minisgl.scheduler.scheduler import Scheduler

        engine = Engine(args, model_args, tp_rank=rank)
        scheduler = Scheduler(args, engine)
        ready.put(rank)
        while True:
            command: _Command = commands.get()
            if command.name == "ADD":
                input_ids, sampling, uid = command.payload
                scheduler.add_request(input_ids, sampling, uid=uid)
            elif command.name == "ABORT":
                scheduler.abort_request(command.payload[0])
            elif command.name == "STEP":
                step_results = scheduler.step()
                if rank == 0:
                    results.put(step_results)
            elif command.name == "SHUTDOWN":
                break
    except BaseException as exc:  # report startup and collective failures
        errors.put((rank, repr(exc)))
    finally:
        try:
            engine.cleanup()
        except (UnboundLocalError, NameError):
            pass
        try:
            import torch.distributed as dist

            if dist.is_initialized():
                dist.destroy_process_group()
        except Exception:
            pass


class DistributedSchedulerProxy:
    """Scheduler-compatible proxy backed by one worker per GPU."""

    def __init__(self, args: ServerArgs, model_args: ModelArgs) -> None:
        if not 2 <= args.tp_size <= 8:
            raise ValueError("tp_size must be between 2 and 8 for distributed mode")
        self.args = args
        self._ctx = mp.get_context("spawn")
        self._commands = [self._ctx.Queue() for _ in range(args.tp_size)]
        self._results = self._ctx.Queue()
        self._errors = self._ctx.Queue()
        self._ready = self._ctx.Queue()
        self._workers: list[mp.Process] = []
        self._active: set[int] = set()
        self._next_uid = 0
        self._closed = False
        port = _free_port()
        for rank in range(args.tp_size):
            process = self._ctx.Process(
                target=_worker,
                args=(
                    rank,
                    args,
                    model_args,
                    self._commands[rank],
                    self._results,
                    self._errors,
                    self._ready,
                    port,
                ),
                daemon=True,
            )
            process.start()
            self._workers.append(process)
        self._wait_until_ready()

    def _send(self, command: _Command) -> None:
        for commands in self._commands:
            commands.put(command)

    def _wait_until_ready(self) -> None:
        import time

        ready: set[int] = set()
        deadline = time.monotonic() + 600.0
        while time.monotonic() < deadline and len(ready) < len(self._workers):
            error = self._read_error()
            if error is not None:
                self.cleanup()
                raise RuntimeError(f"TP worker {error[0]} failed: {error[1]}")
            try:
                ready.add(self._ready.get(timeout=0.05))
            except queue.Empty:
                if not all(process.is_alive() for process in self._workers):
                    error = self._read_error()
                    self.cleanup()
                    raise RuntimeError(
                        f"TP worker failed: {error[1] if error else 'exited'}"
                    )
        if len(ready) == len(self._workers):
            return
        self.cleanup()
        raise TimeoutError("Timed out starting tensor-parallel workers")

    def _read_error(self):
        try:
            return self._errors.get_nowait()
        except queue.Empty:
            return None

    def add_request(
        self,
        input_ids: list[int],
        sampling_params: SamplingParams,
    ) -> int:
        uid = self._next_uid
        self._next_uid += 1
        self._active.add(uid)
        self._send(_Command("ADD", (list(input_ids), sampling_params, uid)))
        return uid

    def abort_request(self, uid: int) -> bool:
        if uid not in self._active:
            return False
        self._send(_Command("ABORT", (uid,)))
        self._active.discard(uid)
        return True

    def step(self) -> list[OutputToken]:
        self._send(_Command("STEP"))
        import time

        deadline = time.monotonic() + 600.0
        while True:
            error = self._read_error()
            if error is not None:
                self._fail_workers()
                raise RuntimeError(f"TP worker {error[0]} failed: {error[1]}")
            if not all(process.is_alive() for process in self._workers):
                self._fail_workers()
                raise RuntimeError("A tensor-parallel worker exited unexpectedly")
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                self._fail_workers()
                raise TimeoutError("Timed out waiting for tensor-parallel workers")
            try:
                results = self._results.get(timeout=min(remaining, 0.1))
                break
            except queue.Empty:
                continue
        for result in results:
            if result.finished:
                self._active.discard(result.uid)
        return results

    def is_idle(self) -> bool:
        return not self._active

    def cleanup(self) -> None:
        if self._closed:
            return
        self._closed = True
        for commands in self._commands:
            commands.put(_Command("SHUTDOWN"))
        for process in self._workers:
            process.join(timeout=10)
            if process.is_alive():
                process.terminate()
        self._workers.clear()
        for commands in self._commands:
            commands.close()
        self._results.close()
        self._errors.close()
        self._ready.close()

    def _fail_workers(self) -> None:
        """Stop every rank after a command/collective failure."""
        for process in self._workers:
            if process.is_alive():
                process.terminate()
        for process in self._workers:
            process.join(timeout=5)
        self._workers.clear()
        self._closed = True


def create_scheduler_runtime(args: ServerArgs, model_args: ModelArgs):
    """Return the existing local runtime or an automatic TP proxy."""
    if args.tp_size == 1:
        from minisgl.engine.engine import Engine
        from minisgl.scheduler.scheduler import Scheduler

        engine = Engine(args, model_args, tp_rank=0)
        return Scheduler(args, engine), engine
    # ServerArgs defaults graph capture to max_running_req for the local path;
    # TP workers must explicitly use eager execution until synchronized graph
    # capture is implemented.
    args.cuda_graph_bs = 0
    proxy = DistributedSchedulerProxy(args, model_args)
    return proxy, proxy