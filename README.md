# PyTorch 自定义 CUDA 卷积算子实验

本项目是智能计算系统 Lab6 算子优化实验。项目以 CIFAR-10 / ResNet-18 推理为应用场景，通过 PyTorch C++/CUDA Extension 接入自定义 Conv2d 算子，并对比多种卷积实现的正确性与吞吐量。

当前包含四类算子：

- `conv2d_baseline_fp32`：FP32 直接卷积基线。
- `conv2d_optim_fp32`：FP32 implicit-GEMM 优化卷积。
- `conv2d_optim_fp16`：FP16 implicit-GEMM 优化卷积。
- `conv2d_optim_wmma`：基于 WMMA / Tensor Core 的 FP16 前向卷积。

## 目录结构

```text
Lab6_experiment/
├── cpp/          # PyTorch C++ extension 与 pybind11 绑定
├── cuda/         # CUDA kernel 实现
├── include/      # 卷积参数结构体与类型声明
├── modules/      # Python Conv2d 封装与 ResNet-18 模型
├── pytorch/      # setup.py、模型权重与数据目录
├── docs/         # 实验报告、benchmark 数据和绘图脚本
├── tests/        # 轻量回归测试
├── build_ext.py  # JIT 构建辅助脚本
├── inference.py  # 推理、准确率和吞吐量测试入口
└── setup.sh      # 统一构建脚本
```

## 环境要求

- Python 3.7+
- PyTorch，需带 CUDA 支持
- CUDA Toolkit
- NVIDIA GPU

注意：本项目的自定义算子只支持 CUDA tensor，没有 CPU fallback。若当前环境安装的是 CPU 版 PyTorch，可以运行部分静态测试和 `inference.py --help`，但不能编译或执行 CUDA 算子。

## 构建算子

在 `Lab6_experiment` 项目根目录运行：

```bash
sh setup.sh
```

等价命令为：

```bash
python ./pytorch/setup.py build_ext --inplace
```

该命令会一次性构建四个扩展模块：baseline FP32、optim FP32、optim FP16 和 optim WMMA。构建完成后，`modules/` 中的 Python 封装即可直接 import 对应扩展。

## 运行推理与 benchmark

默认运行 FP32 优化模型：

```bash
python inference.py --model fp32
```

可选模型：

```text
baseline  FP32 直接卷积基线
fp32      FP32 implicit-GEMM 优化卷积
fp16      FP16 implicit-GEMM 优化卷积
wmma      WMMA / Tensor Core 前向卷积
```

常用命令：

```bash
python inference.py --model fp16 --run-times 10 --accuracy-samples 50
python inference.py --model wmma --run-times 10 --batch-sizes 8,16,32,64,128,256,512
python inference.py --model fp32 --plot --save-plots-dir docs/figures/runtime
```

重要参数：

- `--model`：选择算子实现，取值为 `baseline`、`fp32`、`fp16`、`wmma`。
- `--batch-sizes`：吞吐量测试的 batch size 列表，例如 `8,16,32,64,128,256,512`。
- `--run-times`：每个 batch size 重复测试次数，默认 `10`。
- `--accuracy-samples`：准确率测试样本数，默认 `50`；设为 `0` 或负数时使用完整测试集。
- `--plot`：显示预测图和吞吐量图。
- `--save-plots-dir`：将图保存到指定目录，适合无图形界面环境。
- `--no-progress`：关闭 tqdm 进度条。

## 测试

项目提供了不依赖 CUDA 编译的轻量回归测试，用于检查构建入口、推理 CLI、WMMA backward 行为和 FP16 autograd 封装：

```bash
python -m pytest tests/test_project_regressions.py
```

在完整 CUDA 环境中，建议在构建成功后再运行一次实际推理命令，确认扩展模块、模型权重和 CIFAR-10 数据路径均可用。

## WMMA 使用说明

`conv2d_optim_wmma` 当前定位为推理前向算子。其 backward 路径没有实现真实梯度，现在会直接报错，而不是返回静默的零梯度。

如果需要训练，请使用 `baseline`、`fp32` 或 `fp16` 算子；如果只做推理和性能对比，可以使用 `wmma`。
