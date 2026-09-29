"""Jupyter-friendly wrappers for the distributed benchmark.

The functions in this module only orchestrate the existing benchmark process;
model loading and distributed initialization still happen inside torchrun
workers.  This keeps notebook state out of the inference process.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
BENCHMARK = ROOT / "examples" / "benchmark_distributed.py"


def run_benchmark(
    model_path: str,
    *,
    tp_size: int = 1,
    collective_mode: str = "legacy",
    sampling_mode: str = "gather",
    attention_backend: str = "fa",
    dtype: str = "auto",
    page_size: int = 16,
    batch_size: int = 1,
    input_len: int = 128,
    output_len: int = 20,
    warmup: int = 5,
    repeat: int = 20,
    output_path: str | Path | None = None,
    collective_stats: bool = True,
) -> dict[str, Any]:
    """Run one benchmark configuration and return its JSON result.

    In a notebook, call this function from a cell.  ``output_path`` is useful
    when results should be retained for later analysis; otherwise a temporary
    file is created by the benchmark process and removed afterwards.
    """
    import tempfile

    keep_output = output_path is not None
    if output_path is None:
        temporary = tempfile.NamedTemporaryFile(suffix=".json", delete=False)
        output = Path(temporary.name)
        temporary.close()
    else:
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)

    command = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        f"--nproc-per-node={tp_size}",
        str(BENCHMARK),
        "--model-path",
        model_path,
        "--tp-size",
        str(tp_size),
        "--dtype",
        dtype,
        "--attention-backend",
        attention_backend,
        "--tp-collective-mode",
        collective_mode,
        "--tp-sampling-mode",
        sampling_mode,
        "--page-size",
        str(page_size),
        "--batch-size",
        str(batch_size),
        "--input-len",
        str(input_len),
        "--output-len",
        str(output_len),
        "--warmup",
        str(warmup),
        "--repeat",
        str(repeat),
        "--output",
        str(output),
    ]
    if collective_stats:
        command.append("--collective-stats")

    try:
        subprocess.run(command, cwd=ROOT, check=True)
        data = json.loads(output.read_text(encoding="utf-8"))
        if isinstance(data, list):
            if len(data) != 1:
                raise ValueError(f"Expected one benchmark row, got {len(data)}")
            return data[0]
        return data
    finally:
        if not keep_output:
            output.unlink(missing_ok=True)


def run_ablation(
    model_path: str,
    *,
    tp_size: int = 2,
    collective_modes: tuple[str, ...] = ("legacy", "tensor", "buffered"),
    sampling_modes: tuple[str, ...] = ("gather", "distributed", "distributed_packed"),
    **kwargs: Any,
) -> list[dict[str, Any]]:
    """Run a small collective/sampling ablation matrix."""
    rows = []
    for collective_mode in collective_modes:
        for sampling_mode in sampling_modes:
            row = run_benchmark(
                model_path,
                tp_size=tp_size,
                collective_mode=collective_mode,
                sampling_mode=sampling_mode,
                **kwargs,
            )
            rows.append(row)
    return rows


def summarize(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Select notebook-friendly latency, throughput, and communication fields."""
    summary = []
    for row in rows:
        stats = row.get("collective_stats", {})
        summary.append(
            {
                "collective_mode": row.get("collective_mode"),
                "sampling_mode": row.get("sampling_mode"),
                "tp_size": row.get("tp_size"),
                "latency_median_ms": row.get("latency_median_ms"),
                "latency_p90_ms": row.get("latency_p90_ms"),
                "kernel_median_ms": row.get("kernel_median_ms"),
                "output_tok_per_s": row.get("output_tok_per_s"),
                "all_gather_calls": stats.get("all_gather", {}).get("calls", 0),
                "all_gather_bytes": stats.get("all_gather", {}).get("bytes", 0),
                "all_reduce_calls": stats.get("all_reduce", {}).get("calls", 0),
                "broadcast_calls": stats.get("broadcast", {}).get("calls", 0),
            }
        )
    return summary