# Mini-SGLang

<p align="center">
  <a href="README.zh-CN.md">中文</a>
</p>

Mini-SGLang is a lightweight LLM inference project inspired by
[SGLang](https://github.com/sgl-project/sglang). This project focuses on
inference scheduling, KV cache management, and process-to-process data
transfer.

## HSSM Communication Backend

This project implements **HSSM**, a host shared-memory communication backend
for the prefill/decode disaggregated inference path.

HSSM provides:

- Shared-memory ring-buffer transport between prefill and decode workers.
- CUDA page-locked host memory registration through `cudaHostRegister`.
- A selectable backend alongside the regular SHM and queue paths.
- Protocol checks, copy benchmarks, and end-to-end comparison experiments.

Enable HSSM with:

```bash
python -m minisgl \
  --model-path Qwen/Qwen3-0.6B \
  --disaggregation-transfer-backend hssm \
  --pd-devices 0 1
```

HSSM requires CUDA for page-locked host memory registration and is intended for
prefill/decode workers running on two CUDA devices.

## Installation

```bash
pip install -e .
```

## Basic Usage

Start the regular inference server:

```bash
python -m minisgl --model-path Qwen/Qwen3-0.6B --port 8000
```

Run the HSSM tests:

```bash
python -m pytest tests/test_notebook_experiments.py
```

Related files:

- `minisgl/disaggregation/shm_ring.py` — shared-memory ring transport and
  CUDA host-memory registration.
- `minisgl/disaggregation/runtime.py` — prefill/decode runtime integration.
- `examples/hssm_experiment.py` — HSSM protocol and benchmark helpers.
- `examples/hssm_experiment.ipynb` — interactive HSSM experiments.

## Experiment Setup and Results

The HSSM experiment was run under the following conditions:

- Hardware: 2 × NVIDIA RTX 3090 (24 GB), connected through PCIe.
- Topology: 1P1D prefill/decode disaggregation across two GPUs.
- Model: `Qwen3-8B-FP8`.
- Dtype: `auto`.
- Input length: `1024` tokens.
- Output length: `1024` tokens.
- Batch size: `4`.
- Warmup iterations: `5`.
- Measured iterations: `20`.
- KV-cache page size: `16` tokens.
- Transfer chunk size: `16` tokens.
- Shared-memory slot count: `4`.

The experiment compares the `queue`, `shm`, `hssm`, and `nixl` transfer
backends. It includes protocol and ring-buffer checks, host-copy benchmarks,
GPU host-copy measurements, end-to-end latency/throughput measurements, and
chunk-size experiments. The complete configuration and recorded results are
available in [`results.ipynb`](results.ipynb).

Under the same SLO, HSSM was compared with NIXL in the 2 × RTX 3090 PCIe
environment. HSSM reduced TTFT P90 by **49.9%** and increased output
throughput by **4.7%**. These results validate the effectiveness of HSSM for
the PCIe dual-GPU prefill/decode disaggregation scenario. Detailed values and
measurement output should be read from `results.ipynb`.

## Acknowledgements

Many ideas and architectural references in this project come from the open
source [SGLang](https://github.com/sgl-project/sglang) project. Thanks to the
SGLang maintainers and contributors for providing an excellent reference for
high-performance LLM serving systems.

## License

This project is released under the Apache License 2.0.