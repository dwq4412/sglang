# Mini-SGLang

<p align="center">
  <a href="README.md">English</a>
</p>

Mini-SGLang 是一个受 [SGLang](https://github.com/sgl-project/sglang) 启发的
轻量级 LLM 推理项目，主要用于学习和实验推理调度、KV Cache 管理以及进程间
数据传输。

## HSSM 通信后端

本项目实现了 **HSSM**，用于 Prefill/Decode 分离推理路径的主机共享内存通信。

HSSM 提供：

- 基于共享内存环形缓冲区的 Prefill/Decode worker 通信。
- 通过 `cudaHostRegister` 注册 CUDA page-locked host memory。
- 可独立选择的通信后端，与普通 SHM 和 queue 路径并存。
- 协议检查、内存拷贝基准和端到端对比实验。

可以通过下面的参数启用 HSSM：

```bash
python -m minisgl \
  --model-path Qwen/Qwen3-0.6B \
  --disaggregation-transfer-backend hssm \
  --pd-devices 0 1
```

HSSM 的 page-locked host memory 注册需要 CUDA，主要用于运行在两张 CUDA
设备上的 Prefill/Decode worker。

## 安装

```bash
pip install -e .
```

## 基本使用

启动普通推理服务：

```bash
python -m minisgl --model-path Qwen/Qwen3-0.6B --port 8000
```

运行 HSSM 相关测试：

```bash
python -m pytest tests/test_notebook_experiments.py
```

相关文件：

- `minisgl/disaggregation/shm_ring.py`：共享内存环形传输和 CUDA host memory
  注册。
- `minisgl/disaggregation/runtime.py`：Prefill/Decode runtime 集成。
- `examples/hssm_experiment.py`：HSSM 协议检查和 benchmark 辅助代码。
- `examples/hssm_experiment.ipynb`：交互式 HSSM 实验 Notebook。

## 实验条件、参数与结果

HSSM 实验环境如下：

- 硬件：2 张 NVIDIA RTX 3090（24GB），通过 PCIe 连接。
- 拓扑：1P1D Prefill/Decode 分离，两个 worker 分布在两张 GPU 上。
- 模型：`Qwen3-8B-FP8`。
- 数据类型：`auto`。
- 输入长度：`1024` tokens。
- 输出长度：`1024` tokens。
- Batch size：`4`。
- Warmup 次数：`5`。
- 正式测试次数：`20`。
- KV Cache page size：`16` tokens。
- 传输 chunk size：`16` tokens。
- 共享内存 slot 数量：`4`。

实验对比了 `queue`、`shm`、`hssm` 和 `nixl` 四种通信后端，并包含协议与
环形缓冲区检查、host copy 微基准、GPU host copy 测试、端到端延迟/吞吐测试
以及 chunk size 扫描。完整的参数配置和 Notebook 中记录的具体结果请参考
[`results.ipynb`](results.ipynb)。

在相同 SLO 下，HSSM 与 NIXL 在 2 张 RTX 3090 PCIe 双卡环境中进行对比：

- TTFT P90 降低 **49.9%**；
- 输出吞吐提高 **4.7%**。

该结果验证了 HSSM 在 PCIe 双卡 1P1D Prefill/Decode 分离场景下的有效性。
具体测量值和实验输出以 `results.ipynb` 中的记录为准。

## 致谢

本项目中的许多设计思路和架构参考来自开源项目
[SGLang](https://github.com/sgl-project/sglang)。感谢 SGLang 的维护者和贡献者，
感谢这个优秀的开源项目为高性能 LLM 服务系统提供了重要参考。

## 许可证

本项目采用 Apache License 2.0 许可证。