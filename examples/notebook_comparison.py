"""Jupyter-friendly experiment matrix for TP and collective comparisons.

Usage from a notebook::

    from examples.notebook_comparison import *
    rows = run_all_cases()
    raw_df = make_raw_table(rows)
    comparison_df = make_comparison_table(raw_df)
    display(raw_df)
    display(comparison_df)

Each benchmark is executed in a subprocess through ``run_benchmark``. Results
are written individually, so an interrupted notebook can resume safely.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

import pandas as pd

from examples.notebook_experiments import run_benchmark

RESULT_DIR = Path("results/notebook_comparison")
RESULT_DIR.mkdir(parents=True, exist_ok=True)

MODELS = {
    "4B": "Qwen/Qwen3-4B",
    "8B": "Qwen/Qwen3-8B",
}

COMMON = {
    "dtype": "float16",
    "attention_backend": "pt",
    "cache_strategy": "naive",
    "page_size": 16,
    "max_seq_len": 2048,
    "warmup": 5,
    "repeat": 20,
    "collective_stats": True,
}

# These defaults are deliberately more demanding than the original smoke-test
# matrix.  They can be overridden by passing an explicit case list or by using
# the ``run_experiment_*`` arguments below.
TARGETED_COMMON = {
    **COMMON,
    "tp_size": 2,
    "input_len": 512,
    "output_len": 32,
    "batch_size": 4,
    "max_seq_len": 1024,
    "warmup": 10,
    "repeat": 50,
}


def _case(case_id: str, display_id: str, stage: str, name: str,
          model_label: str, compare_to: str | None = None,
          secondary_compare_to: str | None = None, **config: Any) -> dict[str, Any]:
    return {
        "id": case_id,
        "display_id": display_id,
        "stage": stage,
        "name": name,
        "model_label": model_label,
        "compare_to": compare_to,
        "secondary_compare_to": secondary_compare_to,
        **config,
    }


def model_cases(prefix: str, model_label: str) -> list[dict[str, Any]]:
    """Return the 12 runs required for one model.

    The page scan expands ``input_len=128/512`` into separate runs. The
    additional page16/512 reference is required for a fair page32/512
    comparison because the long-context experiment uses output_len=1.
    """
    common = {
        "tp_size": 2,
        "collective_mode": "buffered",
        "sampling_mode": "distributed_packed",
        "input_len": 128,
        "output_len": 32,
        "batch_size": 1,
    }
    gather_base = {
        **common,
        "collective_mode": "legacy",
        "sampling_mode": "gather",
    }
    return [
        _case(f"{prefix}1", "1" if prefix == "q04_" else "11", "阶段1 基线", "无通信绝对基线", model_label,
              **{**gather_base, "tp_size": 1}),
        _case(f"{prefix}2", "2" if prefix == "q04_" else "12", "阶段1 基线", "TP 通信基线", model_label,
              **gather_base),
        _case(f"{prefix}3", "3" if prefix == "q04_" else "13", "阶段2 集合通信后端", "tensor 后端", model_label,
              f"{prefix}2", **{**gather_base, "collective_mode": "tensor"}),
        _case(f"{prefix}4", "4" if prefix == "q04_" else "14", "阶段2 集合通信后端", "buffered 后端", model_label,
              f"{prefix}2", **{**gather_base, "collective_mode": "buffered"}),
        _case(f"{prefix}5", "5" if prefix == "q04_" else "15", "阶段3 采样通信", "distributed 采样", model_label,
              f"{prefix}2", **{**common, "collective_mode": "legacy", "sampling_mode": "distributed"}),
        _case(f"{prefix}6", "6" if prefix == "q04_" else "16", "阶段3 采样通信", "distributed_packed 采样", model_label,
              f"{prefix}2", f"{prefix}5", **{**common, "collective_mode": "legacy"}),
        _case(f"{prefix}7", "7" if prefix == "q04_" else "17", "阶段4 优化组合", "buffered + distributed_packed", model_label,
              f"{prefix}2", **common),
        _case(f"{prefix}8", "8" if prefix == "q04_" else "18", "阶段5 负载扫描", "长上下文 prefill 负载", model_label,
              f"{prefix}7", **{**common, "input_len": 512, "output_len": 1}),
        _case(f"{prefix}9", "9" if prefix == "q04_" else "19", "阶段5 负载扫描", "扩大 batch 负载", model_label,
              f"{prefix}7", **{**common, "batch_size": 4}),
        _case(f"{prefix}page512_ref", "10-ref" if prefix == "q04_" else "20-ref", "阶段6 KV-Cache page 扫描", "page16 长上下文严格对照", model_label,
              **{**common, "input_len": 512}),
        _case(f"{prefix}10_128", "10a" if prefix == "q04_" else "20a", "阶段6 KV-Cache page 扫描", "page32，input_len=128", model_label,
              f"{prefix}7", **{**common, "page_size": 32}),
        _case(f"{prefix}10_512", "10b" if prefix == "q04_" else "20b", "阶段6 KV-Cache page 扫描", "page32，input_len=512", model_label,
              f"{prefix}page512_ref", **{**common, "input_len": 512, "page_size": 32}),
    ]


def targeted_experiment_a_cases(model_label: str = "8B") -> list[dict[str, Any]]:
    """Build Experiment A: isolate collective backend improvements.

    Sampling is held at ``gather`` so a row differs from its baseline only in
    the all-gather implementation.  The default workload is input=512,
    output=32, batch=4, which increases the amount of TP work without mixing
    in a sampling optimization.
    """
    prefix = "expA_"
    base = {**TARGETED_COMMON, "sampling_mode": "gather"}
    modes = ("legacy", "tensor", "buffered")
    return [
        _case(
            f"{prefix}{mode}",
            f"A-{mode}",
            "实验A 集合通信后端",
            f"Experiment A: {mode} + gather",
            model_label,
            "expA_legacy" if mode != "legacy" else None,
            **{**base, "collective_mode": mode},
        )
        for mode in modes
    ]


def targeted_experiment_b_cases(model_label: str = "8B") -> list[dict[str, Any]]:
    """Build Experiment B: isolate distributed sampling communication."""
    prefix = "expB_"
    base = {
        **TARGETED_COMMON,
        "collective_mode": "buffered",
        "input_len": 128,
        "output_len": 128,
        "batch_size": 4,
        "max_seq_len": 512,
    }
    modes = ("gather", "distributed", "distributed_packed")
    return [
        _case(
            f"{prefix}{mode}",
            f"B-{mode}",
            "实验B packed sampling",
            f"Experiment B: buffered + {mode}",
            model_label,
            "expB_gather" if mode != "gather" else None,
            "expB_distributed" if mode == "distributed_packed" else None,
            **{**base, "sampling_mode": mode},
        )
        for mode in modes
    ]


CASES = model_cases("q04_", "4B") + model_cases("q08_", "8B")

# The three cases most users want first: a no-communication reference, the
# former TP baseline, and the current buffered/packed optimization.
MAIN_CASES = [
    case
    for case in CASES
    if case["id"].endswith(("_1", "_2", "_7"))
]


def run_main_comparison(
    skip_existing: bool = True,
    models: dict[str, str] | None = None,
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame]:
    """Run the three primary configurations for both built-in models.

    Returns ``(rows, raw_table, comparison_table)``.  Use
    :func:`run_all_cases` when the full 24-case ablation matrix is required.
    """
    rows = run_all_cases(
        MAIN_CASES,
        skip_existing=skip_existing,
        models=models,
    )
    raw_table = make_raw_table(rows)
    comparison_table = make_comparison_table(raw_table)
    save_tables(raw_table, comparison_table, pd.DataFrame())
    return rows, raw_table, comparison_table


def run_targeted_experiment(
    experiment: str,
    *,
    model_label: str = "8B",
    models: dict[str, str] | None = None,
    skip_existing: bool = True,
    collective_stats: bool | None = None,
) -> tuple[list[dict[str, Any]], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Run one of the focused workloads proposed for the TP experiments.

    ``experiment`` is ``"A"`` or ``"B"``.  By default the targeted runs keep
    collective statistics enabled because the experiment is intended to
    explain communication changes.  For final latency numbers, pass
    ``collective_stats=False`` and run the same matrix again in a separate
    output directory if desired.
    """
    normalized = experiment.upper()
    if normalized == "A":
        cases = targeted_experiment_a_cases(model_label)
    elif normalized == "B":
        cases = targeted_experiment_b_cases(model_label)
    else:
        raise ValueError("experiment must be 'A' or 'B'")

    if collective_stats is not None:
        cases = [
            {**case, "collective_stats": collective_stats}
            for case in cases
        ]
    rows = run_all_cases(cases, skip_existing=skip_existing, models=models)
    raw_table = make_raw_table(rows)
    comparison_table = make_comparison_table(raw_table)
    secondary_table = make_secondary_comparison_table(raw_table)
    save_tables(raw_table, comparison_table, secondary_table)
    return rows, raw_table, comparison_table, secondary_table


def run_experiment_a(**kwargs: Any):
    """Convenience wrapper for ``run_targeted_experiment("A", ...)``."""
    return run_targeted_experiment("A", **kwargs)


def run_experiment_b(**kwargs: Any):
    """Convenience wrapper for ``run_targeted_experiment("B", ...)``."""
    return run_targeted_experiment("B", **kwargs)


def _result_path(case: dict[str, Any], config: dict[str, Any]) -> Path:
    name = (
        f"{case['id']}_{case['model_label']}_tp{config['tp_size']}_"
        f"{config['collective_mode']}_{config['sampling_mode']}_"
        f"in{config['input_len']}_out{config['output_len']}_"
        f"bs{config['batch_size']}_max{config['max_seq_len']}_"
        f"page{config['page_size']}_w{config['warmup']}_r{config['repeat']}_"
        f"stats{int(config['collective_stats'])}.json"
    )
    return RESULT_DIR / re.sub(r"[^a-zA-Z0-9_.-]", "_", name)


def _read_result(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        if len(data) != 1:
            raise ValueError(f"Expected one result in {path}, got {len(data)}")
        return data[0]
    return data


def run_all_cases(
    cases: list[dict[str, Any]] | None = None,
    skip_existing: bool = True,
    models: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Run the full matrix and return rows enriched with experiment metadata."""
    cases = CASES if cases is None else cases
    model_paths = MODELS if models is None else models
    rows: list[dict[str, Any]] = []

    for index, case in enumerate(cases, 1):
        config = {**COMMON, **{
            key: case[key]
            for key in ("tp_size", "collective_mode", "sampling_mode",
                        "input_len", "output_len", "batch_size", "page_size",
                        "max_seq_len", "warmup", "repeat", "collective_stats")
            if key in case
        }}
        output_path = _result_path(case, config)
        print(f"[{index}/{len(cases)}] {case['display_id']} {case['model_label']} {case['name']}")

        if skip_existing and output_path.exists():
            benchmark_row = _read_result(output_path)
        else:
            started = time.time()
            model_value = str(model_paths[case["model_label"]])
            model_path = Path(model_value).expanduser()
            # A Windows-style relative path such as ``Qwen\\Qwen3-0.6B``
            # is not interpreted as a directory on Linux.
            if os.name != "nt" and "\\" in model_value and not model_path.exists():
                model_path = Path(model_value.replace("\\", "/")).expanduser()
            config_file = model_path / "config.json"
            if not config_file.is_file():
                raise FileNotFoundError(
                    f"Model config not found: {config_file}. "
                    "Set comparison.MODELS to local Linux model directories."
                )
            benchmark_row = run_benchmark(
                str(model_path),
                tp_size=config["tp_size"],
                collective_mode=config["collective_mode"],
                sampling_mode=config["sampling_mode"],
                attention_backend=config["attention_backend"],
                dtype=config["dtype"],
                cache_strategy=config["cache_strategy"],
                page_size=config["page_size"],
                max_seq_len=config["max_seq_len"],
                batch_size=config["batch_size"],
                input_len=config["input_len"],
                output_len=config["output_len"],
                warmup=config["warmup"],
                repeat=config["repeat"],
                collective_stats=config["collective_stats"],
                output_path=output_path,
            )
            print(f"  P50={benchmark_row['latency_median_ms']:.3f} ms, "
                  f"P99={benchmark_row['latency_p99_ms']:.3f} ms, "
                  f"throughput={benchmark_row['output_tok_per_s']:.2f} tok/s, "
                  f"elapsed={time.time() - started:.1f}s")

        rows.append({
            "experiment_id": case["id"],
            "display_id": case["display_id"],
            "stage": case["stage"],
            "experiment_name": case["name"],
            "model_label": case["model_label"],
            "compare_to": case.get("compare_to"),
            "secondary_compare_to": case.get("secondary_compare_to"),
            **config,
            **benchmark_row,
        })
        (RESULT_DIR / "all_results.json").write_text(
            json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    return rows


def _stat(row: dict[str, Any], name: str, field: str) -> float:
    return row.get("collective_stats", {}).get(name, {}).get(field, 0)


def make_raw_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    records = []
    for row in rows:
        ar_b, ar_ms = _stat(row, "all_reduce", "bytes"), _stat(row, "all_reduce", "elapsed_ms")
        ag_b, ag_ms = _stat(row, "all_gather", "bytes"), _stat(row, "all_gather", "elapsed_ms")
        bc_b, bc_ms = _stat(row, "broadcast", "bytes"), _stat(row, "broadcast", "elapsed_ms")
        total_b, total_ms = ar_b + ag_b + bc_b, ar_ms + ag_ms + bc_ms
        repeat = max(int(row.get("repeat", 1)), 1)
        records.append({
            "ID": row["display_id"], "内部ID": row["experiment_id"],
            "阶段": row["stage"], "实验": row["experiment_name"],
            "模型": row["model_label"], "TP": row["tp_size"],
            "collective": row["collective_mode"], "sampling": row["sampling_mode"],
            "input": row["input_len"], "output": row["output_len"],
            "batch": row["batch_size"], "page": row["page_size"],
            "P50(ms)": row["latency_median_ms"], "P90(ms)": row["latency_p90_ms"],
            "P99(ms)": row["latency_p99_ms"], "吞吐(tok/s)": row["output_tok_per_s"],
            "P50/token(ms)": row["latency_median_ms"] / max(row["output_len"], 1),
            "峰值显存(GiB)": row["peak_memory_bytes"] / 1024**3,
            "AR次数": _stat(row, "all_reduce", "calls"), "AR总量(MiB)": ar_b / 1024**2,
            "AR耗时(ms)": ar_ms, "AG次数": _stat(row, "all_gather", "calls"),
            "AG总量(MiB)": ag_b / 1024**2, "AG耗时(ms)": ag_ms,
            "广播次数": _stat(row, "broadcast", "calls"), "通信总量(MiB)": total_b / 1024**2,
            "通信总耗时(ms)": total_ms, "通信量/次实验(MiB)": total_b / repeat / 1024**2,
            "通信耗时/次实验(ms)": total_ms / repeat,
            "通信耗时/生成token(ms)": total_ms / repeat / max(row["output_len"], 1),
            "估算通信带宽(GB/s)": total_b / 1e9 / (total_ms / 1000) if total_ms else 0,
            "对比基线": row.get("compare_to"), "第二基线": row.get("secondary_compare_to"),
        })
    return pd.DataFrame(records)


def make_comparison_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    by_id = raw_df.set_index("内部ID")
    records = []
    for _, current in raw_df.iterrows():
        baseline_id = current["对比基线"]
        if not baseline_id or baseline_id not in by_id.index:
            continue
        baseline = by_id.loc[baseline_id]
        def pct(current_value: float, baseline_value: float):
            return (current_value / baseline_value - 1) * 100 if baseline_value else None

        records.append({
            "ID": current["ID"], "实验": current["实验"], "模型": current["模型"],
            "对比基线": baseline["ID"], "当前P50(ms)": current["P50(ms)"],
            "基线P50(ms)": baseline["P50(ms)"],
            "P50加速比": baseline["P50(ms)"] / current["P50(ms)"],
            "P50变化(%)": pct(current["P50(ms)"], baseline["P50(ms)"]),
            "P99变化(%)": pct(current["P99(ms)"], baseline["P99(ms)"]),
            "吞吐变化(%)": pct(current["吞吐(tok/s)"], baseline["吞吐(tok/s)"]),
            "峰值显存变化(MiB)": (current["峰值显存(GiB)"] - baseline["峰值显存(GiB)"]) * 1024,
            "AG次数变化": current["AG次数"] - baseline["AG次数"],
            "AG payload压缩比": baseline["AG总量(MiB)"] / current["AG总量(MiB)"] if current["AG总量(MiB)"] else None,
            "AG耗时变化(%)": pct(current["AG耗时(ms)"], baseline["AG耗时(ms)"]),
            "通信总量变化(%)": pct(current["通信总量(MiB)"], baseline["通信总量(MiB)"]),
            "通信耗时变化(%)": pct(current["通信总耗时(ms)"], baseline["通信总耗时(ms)"]),
        })
    return pd.DataFrame(records)


def make_secondary_comparison_table(raw_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for _, current in raw_df.iterrows():
        baseline_id = current["第二基线"]
        if not baseline_id:
            continue
        baseline = raw_df.loc[raw_df["内部ID"] == baseline_id].iloc[0]
        records.append({
            "ID": current["ID"], "模型": current["模型"],
            "当前配置": current["sampling"], "对比配置": baseline["sampling"],
            "P50加速比": baseline["P50(ms)"] / current["P50(ms)"],
            "AG次数变化": current["AG次数"] - baseline["AG次数"],
            "AG payload压缩比": baseline["AG总量(MiB)"] / current["AG总量(MiB)"] if current["AG总量(MiB)"] else None,
            "AG耗时变化(%)": ((current["AG耗时(ms)"] / baseline["AG耗时(ms)"]) - 1) * 100 if baseline["AG耗时(ms)"] else None,
        })
    return pd.DataFrame(records)


def save_tables(raw_df: pd.DataFrame, comparison_df: pd.DataFrame,
                secondary_df: pd.DataFrame) -> None:
    raw_df.to_csv(RESULT_DIR / "raw_results.csv", index=False, encoding="utf-8-sig")
    comparison_df.to_csv(RESULT_DIR / "comparison_results.csv", index=False, encoding="utf-8-sig")
    secondary_df.to_csv(RESULT_DIR / "packed_vs_distributed.csv", index=False, encoding="utf-8-sig")
