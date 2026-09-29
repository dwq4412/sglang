#!/usr/bin/env python3
"""Synchronized tensor-parallel benchmark.

Run with ``torchrun --standalone --nproc-per-node=N``.  Unlike the regular
benchmark, every rank creates its own engine so barriers, device events, and
peak-memory measurements happen in the worker that owns the device.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import statistics
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from minisgl.config import ModelArgs, SamplingParams, ServerArgs  # noqa: E402
from minisgl.engine.collectives import get_stats, reset_stats  # noqa: E402
from minisgl.engine.engine import Engine  # noqa: E402
from minisgl.scheduler.scheduler import Scheduler  # noqa: E402
from minisgl.utils.device import get_device, get_tp_group, is_distributed, synchronize  # noqa: E402


def _barrier() -> None:
    if is_distributed():
        torch.distributed.barrier(group=get_tp_group())


def _device_event() -> tuple[object, object] | None:
    device = get_device()
    if device.type == "cuda":
        return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    if device.type == "npu" and hasattr(torch.npu, "Event"):
        return torch.npu.Event(enable_timing=True), torch.npu.Event(enable_timing=True)
    return None


def _measure(run) -> tuple[float, float]:
    """Return wall-clock and accelerator-event milliseconds for one run."""
    _barrier()
    synchronize()
    events = _device_event()
    wall_start = time.perf_counter()
    if events is not None:
        events[0].record()
    run()
    if events is not None:
        events[1].record()
    synchronize()
    wall_ms = (time.perf_counter() - wall_start) * 1000
    kernel_ms = events[0].elapsed_time(events[1]) if events is not None else wall_ms
    _barrier()
    return wall_ms, kernel_ms


def _max_across_ranks(value: float) -> float:
    if not is_distributed():
        return value
    tensor = torch.tensor(value, dtype=torch.float64, device=get_device())
    torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.MAX, group=get_tp_group())
    return float(tensor.item())


def _percentiles(values: list[float]) -> tuple[float, float, float]:
    ordered = sorted(values)
    return (
        statistics.median(ordered),
        ordered[min(len(ordered) - 1, int(len(ordered) * 0.90))],
        ordered[min(len(ordered) - 1, int(len(ordered) * 0.99))],
    )


def _run_case(
    engine: Engine,
    args: ServerArgs,
    input_len: int,
    batch_size: int,
    output_len: int,
    warmup: int,
    repeat: int,
) -> dict[str, object]:
    input_ids = list(range(1, input_len + 1))
    sampling = SamplingParams(temperature=0.0, max_tokens=output_len)

    def run() -> None:
        scheduler = Scheduler(args, engine)
        for _ in range(batch_size):
            scheduler.add_request(list(input_ids), sampling)
        while not scheduler.is_idle():
            for _ in scheduler.step():
                pass

    for _ in range(warmup):
        _measure(run)
    reset_stats()
    wall_times: list[float] = []
    kernel_times: list[float] = []
    for _ in range(repeat):
        wall_ms, kernel_ms = _measure(run)
        wall_times.append(_max_across_ranks(wall_ms))
        kernel_times.append(_max_across_ranks(kernel_ms))

    wall_p50, wall_p90, wall_p99 = _percentiles(wall_times)
    kernel_p50, kernel_p90, kernel_p99 = _percentiles(kernel_times)
    stats = get_stats()
    peak_memory = 0
    device = get_device()
    if device.type == "cuda":
        peak_memory = torch.cuda.max_memory_allocated(device)
    return {
        "scenario": "generation",
        "input_len": input_len,
        "batch_size": batch_size,
        "output_len": output_len,
        "warmup": warmup,
        "repeat": repeat,
        "latency_median_ms": wall_p50,
        "latency_p90_ms": wall_p90,
        "latency_p99_ms": wall_p99,
        "kernel_median_ms": kernel_p50,
        "kernel_p90_ms": kernel_p90,
        "kernel_p99_ms": kernel_p99,
        "output_tok_per_s": batch_size * output_len / (wall_p50 / 1000),
        "peak_memory_bytes": peak_memory,
        "collective_stats": stats,
    }


def _write_results(path: str, rows: list[dict[str, object]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".csv":
        fields = sorted({key for row in rows for key in row})
        with output.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
    else:
        output.write_text(json.dumps(rows, indent=2, default=str), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mini-SGLang distributed benchmark")
    parser.add_argument("--model-path", required=True)
    parser.add_argument("--tp-size", type=int, default=int(os.environ.get("WORLD_SIZE", "1")))
    parser.add_argument("--dtype", choices=["auto", "float16", "bfloat16", "float32"], default="auto")
    parser.add_argument("--attention-backend", choices=["fa", "pt"], default="fa")
    parser.add_argument("--tp-collective-mode", choices=["legacy", "tensor", "buffered"], default="legacy")
    parser.add_argument("--tp-sampling-mode", choices=["gather", "distributed", "distributed_packed"], default="gather")
    parser.add_argument("--collective-stats", action="store_true")
    parser.add_argument("--page-size", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--input-len", type=int, default=128)
    parser.add_argument("--output-len", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--output", default="benchmark_distributed.json")
    cli = parser.parse_args()

    rank = int(os.environ.get("RANK", "0"))
    server_args = ServerArgs(
        model_path=cli.model_path,
        tp_size=cli.tp_size,
        dtype=cli.dtype,
        attention_backend=cli.attention_backend,
        page_size=cli.page_size,
        max_seq_len=cli.max_seq_len,
        max_running_req=max(cli.batch_size, 1),
        cuda_graph_bs=0,
        tp_collective_mode=cli.tp_collective_mode,
        tp_sampling_mode=cli.tp_sampling_mode,
        collective_stats=cli.collective_stats,
    )
    model_args = ModelArgs.from_pretrained(cli.model_path)
    engine = Engine(server_args, model_args, tp_rank=rank)
    try:
        row = _run_case(
            engine,
            server_args,
            cli.input_len,
            cli.batch_size,
            cli.output_len,
            cli.warmup,
            cli.repeat,
        )
        row.update(
            {
                "rank": rank,
                "tp_size": cli.tp_size,
                "dtype": cli.dtype,
                "attention_backend": cli.attention_backend,
                "collective_mode": cli.tp_collective_mode,
                "sampling_mode": cli.tp_sampling_mode,
                "page_size": cli.page_size,
                "torch_version": torch.__version__,
            }
        )
        if rank == 0:
            _write_results(cli.output, [row])
            print(json.dumps(row, indent=2, default=str))
    finally:
        engine.cleanup()


if __name__ == "__main__":
    main()