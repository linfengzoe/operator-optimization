# Lab6 算子优化大作业任务记录

## 2026-05-22 14:53:00

### 任务描述
阅读 Lab6 实验要求 PDF，完成基本要求和进阶要求的代码实现。

### 实验内容
**基本要求（必做）：**
1. 实现 `cuda/conv2d_optim_kernel_fp32.cu` 的 `implgemm` 前向核函数及 `conv2d_cuda_forward` 配置
   - 采用 Implicit GEMM 算法，每个 Block 处理 128×128 的 [K, Oh×Ow] 输出 tile
   - 256 线程/Block，8 warps，内层循环遍历 C×R×S（每次 8 个）
   - 使用 Shared Memory 优化：smemweight[8×132]（K方向）和 smeminput[8×128]（OhOw方向）
   - Grid 配置：blockx=(Oh×Ow+127)/128, blocky=(K+127)/128, blockz=N

2. 实现 `cuda/conv2d_optim_kernel_fp16.cu` 的 FP16 版本
   - 将所有浮点运算替换为 half intrinsics：
     - `0.0f` → `__float2half(0.0f)`
     - `a * b` → `__hmul(a, b)`
     - `a += b` → `a = __hadd(a, b)`

**进阶要求（选做）：**
3. 修复 `test_mma.cu` 中 `wmma_gemm` 的 PLACEHOLDER 地址计算
   - `addr_a = a + a_row * K + ki`（行主序 A 矩阵的 tile 起始地址）
   - `addr_b = b + b_col * K + ki`（列主序 B 矩阵 [N,K] 存储，WMMA col_major 接口）

### 执行结果
- 代码实现完成，需要在 Linux/GPU 环境下编译验证
- 编译命令（FP32/FP16 算子）：`python ./pytorch/setup.py install`
- 验证推理：`python inference.py`（精度需 ≥ 90%，吞吐量需优于 baseline）
- 进阶编译：`nvcc -arch=sm_89 -o test test_mma.cu`

### 修改文件列表
- `cuda/conv2d_optim_kernel_fp32.cu`：实现 implgemm 前向核函数
- `cuda/conv2d_optim_kernel_fp16.cu`：实现 FP16 版本 implgemm 前向核函数
- `test_mma.cu`：修复 wmma_gemm PLACEHOLDER 地址计算

---

## 2026-05-22 14:55:00

### 任务描述
撰写 Lab6 实验大报告（图文并茂）

### 执行结果
- 创建完整实验报告：`docs/Lab6_Report.md`
- 报告包含以下章节：
  1. 实验背景与目标（优化路线图）
  2. 背景知识（ResNet-18 架构图、残差块结构图、Implicit GEMM 原理、FP16/Tensor Core 介绍）
  3. 实验环境与项目结构（文件树、三层架构设计图）
  4. 基本要求实现（Baseline 分析、FP32/FP16 Implicit GEMM 详细设计、线程布局图、Shared Memory 布局图、核心代码）
  5. 进阶要求实现（Tensor Core WMMA 地址推导、性能对比原理）
  6. 实验结果与分析（精度/吞吐量对比表、趋势图）
  7. 总结与展望（优化技术对比表）
  8. 附录（编译运行指南）
- 使用 matplotlib 生成 **10 张真实图片**，保存在 `docs/figures/`
- 图片生成脚本：`docs/gen_figures.py`
- 图片列表：fig1_resnet18、fig2_residual_block、fig3_implicit_gemm、fig4_tile_layout、fig5_shared_memory、fig6_fp16_fp32、fig7_wmma、fig8_arch、fig9_throughput、fig10_tensorcore

---

## 2026-05-22 15:25:00

### 任务描述
编译 CUDA 算子并采集真实基准测试数据，用真实数据替换报告中的预估图表

### 执行结果
- 实测环境：NVIDIA GeForce RTX 4060 Laptop GPU，CUDA 12.9，PyTorch 2.5.1
- 编译脚本：`build_ext.py`（JIT 编译，需通过 `vcvarsall.bat x64` + `-X utf8` 启动）
- 基准测试脚本：`benchmark_real.py`（结果保存至 `docs/benchmark_results.json`）

### 真实测量结果
**精度（前50张 CIFAR-10 测试图）：**
- Baseline FP32：48/50 = 96.00% ✅
- Optim FP32：48/50 = 96.00% ✅
- Optim FP16：48/50 = 96.00% ✅（无精度损失）

**吞吐量（img/s，BatchSize=256）：**
- Baseline FP32：669.9
- Optim FP32：1737.9（加速 2.59×）
- Optim FP16：2307.9（加速 3.45×）

### 新增/更新图表
- `fig9_throughput.png`：用真实数据替换（含每点数值标注）
- `fig11_speedup_accuracy.png`：加速比柱状图 + 精度对比（新增）
- `fig12_bar_throughput.png`：三算子吞吐量柱状对比（新增）

---

## 2026-05-22 15:38:00

### 任务描述
代码 Review：检查乱码、死代码、内存泄漏；编译运行 test_mma.cu 进阶测试；用真实 Tensor Core 数据更新 fig10

### 执行结果
**代码修复：**
- `test_mma.cu`：删除 `#define PLACEHOLDER 0`（死代码）
- `test_mma.cu`：删除 `h_c_ref` 未使用数组的分配与释放（内存泄漏）
- `test_mma.cu`：所有中文字符串改为英文（解决 GBK 编码乱码）

**Tensor Core 实测（nvcc -arch=sm_89 -o test_mma.exe test_mma.cu）：**
| 矩阵尺寸 | CUDA Core | Tensor Core | 加速比 | 验证 |
|---|---|---|---|---|
| 128×128 | 0.102ms | 0.246ms | 0.4× | PASS |
| 256×256 | 0.642ms | 0.023ms | 27.9× | PASS |
| 512×512 | 4.091ms | 0.137ms | 29.9× | PASS |
| 1024×1024 | 32.411ms | 0.935ms | 34.7× | PASS |
| 2048×2048 | 258.708ms | 7.393ms | 35.0× | PASS |

**更新图表：**
- `fig10_tensorcore.png`：用真实 Tensor Core 实测数据重新生成

---

## 2026-05-22 15:49:00

### 任务描述
完成全部三个进阶要求：实现 WMMA Tensor Core ImplicitGEMM 卷积算子，集成到 ResNet-18 推理流程

### 执行结果
**新建文件：**
- `cuda/conv2d_optim_kernel_wmma.cu`：Tensor Core ImplicitGEMM 前向核函数
  - 4 Warp/Block（2×2），每 Block 处理 32×32 tile，CRS 步长 16
  - float 累加器（精度稳定），结果写回时转 half
  - Bug 修复：输出写回循环从 4 次改为 8 次（128×8=1024=32×32）
- `cpp/conv2d_optim_wmma.cpp`：C++ pybind11 接口
- `modules/conv_layer_optim_wmma.py`：PyTorch 层封装
- `modules/resnet_18_optim_wmma.py`：完整 ResNet-18 with WMMA

**WMMA 实测结果（RTX 4060 Laptop GPU）：**
- 精度：48/50 = **96.00%** ✅（≥90% 要求）
- 吞吐量优于 Baseline（BS=8: 621 vs 246 img/s，加速 **2.52×**）✅

**四种算子汇总（BatchSize=256）：**
| 算子 | 精度 | 吞吐量 | 加速比 |
|---|---|---|---|
| Baseline FP32 | 96% | 670 | 1× |
| Optim FP32 | 96% | 1738 | 2.59× |
| Optim FP16 | 96% | 2308 | **3.45×** |
| Optim WMMA | 96% | 1283 | 1.92× |

**三个进阶要求全部完成 ✅**

**新增图表：**
- `fig13_all_operators.png`：四种算子推理性能综合对比（折线图+加速比柱状图）

---

## 2026-05-22 16:00:00

### 任务描述
优化实验报告图表至发表级质量（nature-figure skill，Python 后端）

### 执行结果
**nature-figure 规范应用：**
- `plt.rcParams['font.family'] = 'sans-serif'` + `svg.fonttype = none`
- 去除顶部/右侧坐标轴框（`axes.spines.right/top = False`）
- `legend.frameon = False`，无边框图例
- NMI 语义色板：Baseline=neutral `#767676`，FP32=blue_secondary `#3775BA`，FP16=blue_main `#0F4D92`（hero），WMMA=teal `#42949E`
- 面板标签 a/b，数据来源注释，GPU 环境标注

**重新生成图表（fig9/10/11/12/13）：**
- `fig9`：Hero 折线 + 峰值加速比柱状（双子图 a/b）
- `fig10`：Tensor Core 时间对比（对数坐标）+ 加速比（双子图 a/b）
- `fig11`：精度柱状 + 各 BatchSize 加速比折线（双子图 a/b）
- `fig12`：四种算子吞吐量分组柱状，含峰值标注
- `fig13`：四种算子综合对比折线 + 选定 BS 加速比（双子图 a/b）

**报告文本修复：**
- 修复错别字：`![吴吞吐量柱状图]` → `![吞吐量柱状图（四种算子）]`
- 吞吐量表格新增 WMMA 列（完整 7×BatchSize 数据）
- 所有算子精度标注更新为 96.00%

---

## 2026-05-22 16:08:00

### 任务描述
优化图 1–8（架构/示意图）至发表级质量，使用英文标题（nature-figure skill，Python 后端）

### 执行结果
**新建脚本：** `docs/gen_arch_figures.py`

**重新生成 fig1–fig8（英文标题，NMI 色板，清洁坐标轴）：**
- fig1：ResNet-18 Architecture（渐变蓝-绿-红语义配色，尺寸标注）
- fig2：Residual Block Structure（主路径蓝色，跨层连接红色）
- fig3：Implicit GEMM as Matrix Multiplication（Blues/Oranges/Greens）
- fig4：Thread Block Tile Assignment Strategy（蓝色系 tile 网格）
- fig5：Shared Memory Data Layout（smemweight=Blues，smeminput=Oranges）
- fig6：FP32 vs FP16 Bit Structure & Performance（位结构 + 性能柱状）
- fig7：Tensor Core WMMA API Invocation Flow（流程图 + 循环箭头）
- fig8：Three-Layer Operator Architecture Design（层次框图）

**规范应用：**
- 无顶部/右侧坐标轴，`legend.frameon=False`，Arial 字体
- NMI 色板：Baseline=neutral，FP32=blue_secondary，FP16=blue_main，WMMA=teal，架构层=语义色

---

## 2026-05-22 16:13:00

### 任务描述
删除项目中的中间/废弃无关文件

### 执行结果
**删除文件：**
- `docs/gen_figures.py`：最初图表生成脚本，已被 `gen_arch_figures.py` + `gen_pub_figures.py` 完全替代
- `docs/gen_real_figures.py`：中间版本，已被 `gen_pub_figures.py` 替代

**保留文件（当前有效）：**
- `docs/gen_arch_figures.py`：Fig 1–8 发表级架构/示意图生成脚本
- `docs/gen_pub_figures.py`：Fig 9–13 发表级数据图表生成脚本
- `build_ext.py`：Windows 下 JIT 编译 CUDA 算子的构建脚本

---

## 2026-05-22 15:57:00

### 任务描述
修复 WMMA 算子精度 Bug（精度仅 6%），调试并验证修复后效果

### Bug 根因分析
- **症状**：WMMA 算子精度 6%（随机猜测水平），吞吐量正常
- **根因**：输出写回循环 `for (int i = 0; i < 4; ++i)` 每线程写 4 个值
  - 128 threads × 4 = 512，但 smem_o 为 32×32 = 1024 个 float 值
  - 只写回了前半部分（kl ∈ [0,16)），后 16 行 K 值全部丢失
- **修复**：将循环改为 `for (int i = 0; i < 8; ++i)`，128×8 = 1024 ✅

### 修改文件
- `cuda/conv2d_optim_kernel_wmma.cu`：Line 119，循环变量 `i < 4` → `i < 8`，`tx * 4 + i` → `tx * 8 + i`

### 验证结果
- 修复后精度：48/50 = **96.00%** ✅
- 吞吐量（BS=8）：**621 img/s**（Baseline 246，加速 2.52×）✅

---

## 2026-05-22 16:15:00

### 任务描述
使用 nature-writing / nature-polishing / paper-audit 三项技能全面优化实验报告

### Audit 发现的问题及修复

| 问题 | 严重性 | 修复 |
|:--|:--|:--|
| 缺少摘要（Abstract） | Major | 新增完整摘要段落 |
| 图片仅有 alt text，无规范图注 | Major | 为全部 13 张图添加 Nature 格式图注（图号 + 说明文字） |
| 完成情况总览在 code block 中 | Moderate | 改写为学术散文段落（基本/进阶分段） |
| 引用已删除的 gen_figures.py | Moderate | 更新为正确脚本路径 |
| 实测环境散落注释中 | Minor | 新增 §6.0 实测环境规范表格 |
| 目录缺少新增小节 | Minor | 同步更新目录结构 |

### 新增/修改内容
- **摘要**：含四项核心数据（精度 96%、FP16 最高 3.45×、WMMA 小 batch 2.52×、Tensor Core 35×）
- **§6.0 实测环境表**：GPU / 架构 / CUDA / PyTorch / 数据集 / 评测方法完整说明
- **13 张图的规范图注**（"图 X | 标题。说明文字" 格式，含子图 (a)(b) 标注）
- **§7.1 总结**：code block → 两段结构化学术散文（基本要求/进阶要求各一段）
- **目录**：新增 §6.0、更新 §5.2 标题
