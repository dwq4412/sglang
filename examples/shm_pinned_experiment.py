"""Notebook helpers for measuring the mini-sglang SHM/SHM-pinned backend.

This module intentionally keeps the notebook thin.  The CPU checks and the
copy microbenchmark run without a model or CUDA.  The end-to-end benchmark
    uses the existing Scheduler for the baseline and PDSchedulerProxy for the
    ``queue``/``shm``/``shm_pinned`` paths.
"""

from __future__ import annotations

import gc
import statistics
import time
from dataclasses import dataclass
from typing import Any


@dataclass
class Sample:
    backend: str
    input_len: int
    output_len: int
    batch_size: int
    ttft_ms: float
    tpot_ms: float
    total_ms: float
    output_tokens: int
    success: bool
    error: str = ""


def percentile(values: list[float], quantile: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, int(len(ordered) * quantile)))
    return ordered[index]


def summarize_samples(samples: list[Sample]) -> dict[str, Any]:
    successful = [sample for sample in samples if sample.success]
    ttft = [sample.ttft_ms for sample in successful]
    tpot = [sample.tpot_ms for sample in successful]
    total = [sample.total_ms for sample in successful]
    output_tokens = sum(sample.output_tokens for sample in successful)
    elapsed_s = sum(sample.total_ms for sample in successful) / 1000.0
    return {
        "backend": samples[0].backend if samples else "",
        "input_len": samples[0].input_len if samples else 0,
        "output_len": samples[0].output_len if samples else 0,
        "batch_size": samples[0].batch_size if samples else 0,
        "requests": len(samples),
        "successful_requests": len(successful),
        "success_rate": len(successful) / len(samples) if samples else 0.0,
        "ttft_p50_ms": percentile(ttft, 0.50),
        "ttft_p90_ms": percentile(ttft, 0.90),
        "ttft_p99_ms": percentile(ttft, 0.99),
        "tpot_p50_ms": percentile(tpot, 0.50),
        "tpot_p90_ms": percentile(tpot, 0.90),
        "tpot_p99_ms": percentile(tpot, 0.99),
        "total_p50_ms": percentile(total, 0.50),
        "total_p90_ms": percentile(total, 0.90),
        "total_p99_ms": percentile(total, 0.99),
        "output_tokens": output_tokens,
        "output_tok_per_s": output_tokens / elapsed_s if elapsed_s else 0.0,
        "transfer_failures": sum(
            1 for sample in samples if not sample.success and "transfer" in sample.error
        ),
    }


def verify_protocol() -> dict[str, Any]:
    """Run the version/key/meta checks from the design document."""
    from minisgl.disaggregation.protocol import TransferKey, TransferMeta

    key = TransferKey("session-a", 17, 3)
    meta = TransferMeta(key, (2, 4, 8, 16, 32, 64), "torch.float16", (2, 9), 37)
    assert TransferKey.unpack(key.pack()) == key
    assert TransferMeta.unpack(meta.pack()) == meta
    try:
        TransferKey.unpack("[999,\"old\",1,0]")
    except ValueError:
        version_rejected = True
    else:
        version_rejected = False
    assert version_rejected
    return {
        "transfer_key_round_trip": True,
        "transfer_meta_round_trip": True,
        "version_rejected": version_rejected,
    }


def verify_ring(payload_size: int = 4096, slots: int = 3) -> dict[str, Any]:
    """Exercise slot bounds, wraparound, and cleanup using CPU shared memory."""
    from minisgl.disaggregation.shm_ring import ShmRing

    ring = ShmRing(payload_size, slots)
    try:
        expected = {}
        for index in range(slots * 3):
            slot = index % slots
            payload = bytes((index + value) % 251 for value in range(payload_size))
            ring.put(slot, payload)
            expected[slot] = payload
            assert ring.get(slot, payload_size) == expected[slot]
        try:
            ring.get(slots, 1)
        except IndexError:
            bounds_rejected = True
        else:
            bounds_rejected = False
        assert bounds_rejected
        return {"wraparound": True, "bounds_rejected": True, "slots": slots}
    finally:
        ring.close(unlink=True)


def benchmark_copy(
    payload_sizes: tuple[int, ...] = (1 << 20, 4 << 20, 16 << 20),
    warmup: int = 3,
    repeat: int = 20,
    pinned: bool = False,
) -> list[dict[str, Any]]:
    """Compare an ordinary host copy with the project's shared-memory copy.

    This measures host-side movement only.  It is not a GPU D2H/H2D result;
    the latter is included in the end-to-end PD benchmark below.
    """
    from minisgl.disaggregation.shm_ring import ShmRing

    rows = []
    for size in payload_sizes:
        payload = bytes(size)
        ring = ShmRing(size, 2)
        try:
            if pinned:
                ring.register_cuda()
            for _ in range(warmup):
                ring.put(0, payload)
                ring.get(0, size)
            shm_ms = []
            plain_ms = []
            for _ in range(repeat):
                start = time.perf_counter()
                ring.put(0, payload)
                ring.get(0, size)
                shm_ms.append((time.perf_counter() - start) * 1000)
                start = time.perf_counter()
                copied = bytes(bytearray(payload))
                assert len(copied) == size
                plain_ms.append((time.perf_counter() - start) * 1000)
            shm_p50 = statistics.median(shm_ms)
            plain_p50 = statistics.median(plain_ms)
            rows.append(
                {
                    "bytes": size,
                    "pinned": pinned,
                    "shm_p50_ms": shm_p50,
                    "plain_p50_ms": plain_p50,
                    "shm_gib_per_s": size / shm_p50 / 1e6,
                    "plain_gib_per_s": size / plain_p50 / 1e6,
                }
            )
        finally:
            ring.close(unlink=True)
    return rows


def benchmark_backend(
    model_path: str,
    backend: str,
    *,
    input_len: int = 128,
    output_len: int = 32,
    batch_size: int = 1,
    warmup: int = 1,
    repeat: int = 5,
    dtype: str = "float16",
    page_size: int = 16,
    chunk_tokens: int = 512,
    slot_count: int = 4,
    devices: tuple[int, int] = (0, 1),
) -> dict[str, Any]:
    """Run one fixed 1P1D comparison point and return percentile metrics."""
    import torch

    from minisgl.config import ModelArgs, SamplingParams, ServerArgs
    from minisgl.engine.engine import Engine
    from minisgl.scheduler.scheduler import Scheduler

    if backend not in ("none", "queue", "shm", "shm_pinned", "nixl"):
        raise ValueError(f"unsupported backend: {backend}")
    if backend in ("shm_pinned", "nixl") and not torch.cuda.is_available():
        raise RuntimeError(f"{backend} requires CUDA")
    if backend == "nixl":
        import sys

        try:
            import nixl._api  # noqa: F401
        except ImportError:
            raise RuntimeError(
                "NIXL is not installed in the notebook interpreter "
                f"{sys.executable}. Install the NIXL package with UCX support "
                "in this environment, then verify with: "
                "python -c \"import nixl._api\""
            ) from None
    model_args = ModelArgs.from_pretrained(model_path)
    common = {
        "model_path": model_path,
        "dtype": dtype,
        "page_size": page_size,
        "max_seq_len": input_len + output_len + 32,
        "max_running_req": batch_size,
        "cuda_graph_bs": 0,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "tp_size": 1,
        "disaggregation_transfer_backend": backend,
        "pd_devices": devices,
        "disaggregation_shm_slot_count": slot_count,
        "disaggregation_shm_chunk_tokens": chunk_tokens,
    }
    args = ServerArgs(**common)
    sampling = SamplingParams(temperature=0.0, max_tokens=output_len)
    input_ids = list(range(1, input_len + 1))
    engine = None
    runtime = None
    try:
        if backend == "none":
            engine = Engine(args, model_args, tp_rank=0)
        else:
            from minisgl.disaggregation.runtime import PDSchedulerProxy

            runtime = PDSchedulerProxy(args, model_args)

        active_engine = engine
        active_runtime = runtime

        def run_once() -> list[Sample]:
            scheduler = (
                Scheduler(args, active_engine, cache_strategy="naive")
                if active_runtime is None
                else active_runtime
            )
            uids = [
                scheduler.add_request(list(input_ids), sampling)
                for _ in range(batch_size)
            ]
            start = time.perf_counter()
            first: dict[int, float] = {}
            tokens: dict[int, int] = dict.fromkeys(uids, 0)
            finished: dict[int, float] = {}
            errors: dict[int, str] = {}
            limit = max(100, output_len * 8 + 20)
            for _ in range(limit):
                for output in scheduler.step():
                    now = time.perf_counter()
                    first.setdefault(output.uid, now)
                    if output.finish_reason == "error":
                        errors[output.uid] = "transfer or decode error"
                    elif output.finish_reason != "abort":
                        tokens[output.uid] = tokens.get(output.uid, 0) + 1
                    if output.finished:
                        finished[output.uid] = now
                if scheduler.is_idle():
                    break
            if not scheduler.is_idle():
                raise TimeoutError("benchmark request did not become idle")
            samples = []
            for uid in uids:
                ok = uid in finished and uid not in errors and tokens[uid] > 0
                ttft = (first[uid] - start) * 1000 if uid in first else float("nan")
                total = (finished[uid] - start) * 1000 if uid in finished else float("nan")
                tpot = (
                    (finished[uid] - first[uid]) * 1000 / max(tokens[uid] - 1, 1)
                    if ok
                    else float("nan")
                )
                samples.append(
                    Sample(
                        backend,
                        input_len,
                        output_len,
                        batch_size,
                        ttft,
                        tpot,
                        total,
                        tokens[uid],
                        ok,
                        errors.get(uid, ""),
                    )
                )
            return samples

        for _ in range(warmup):
            run_once()
        samples = [sample for _ in range(repeat) for sample in run_once()]
        return summarize_samples(samples)
    finally:
        if runtime is not None:
            runtime.cleanup()
        if engine is not None:
            engine.cleanup()
        # run_once closes over these aliases. Clear them before collecting so
        # the closure cannot keep the previous Engine/model alive between
        # backend iterations in the notebook.
        active_runtime = None
        active_engine = None
        # The notebook runs several backends in one parent process.  Engine's
        # cleanup drops KV-cache references, but PyTorch may retain freed CUDA
        # blocks in its allocator, which can starve the next backend startup.
        del runtime, engine
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            torch.cuda.empty_cache()


def to_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return a stable subset suitable for pandas.DataFrame."""
    fields = (
        "backend",
        "input_len",
        "output_len",
        "batch_size",
        "success_rate",
        "ttft_p50_ms",
        "ttft_p90_ms",
        "ttft_p99_ms",
        "tpot_p50_ms",
        "tpot_p90_ms",
        "tpot_p99_ms",
        "output_tok_per_s",
        "transfer_failures",
    )
    return [{field: result.get(field) for field in fields} for result in results]
