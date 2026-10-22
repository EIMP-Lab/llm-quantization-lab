# LLM Quantization Lab

大模型量化的 CPU 入门实训。用一个小型手写数字分类器理解 RTN、GPTQ 的误差补偿与 AWQ 的激活感知缩放，并比较位宽、校准数据、输出误差和模型文件大小。

实验使用 Python 和 NumPy，无需 GPU，也不需要下载大语言模型权重。仓库包含数据集与训练好的基线模型。

## 开始实验

需要 Python 3.10 或以上版本。克隆仓库、安装依赖并运行：

```bash
git clone https://github.com/EIMP-Lab/llm-quantization-lab.git
cd llm-quantization-lab
python -m pip install -r requirements.txt
python lab.py
```

在浏览器打开 `results/report.html`，查看基线预测、准确率、输出误差和文件大小。结果表同时保存为 `results/results.csv`。

习惯使用 Notebook 的同学，可以打开 `CPU量化实训.ipynb`。终端脚本能够完成全部实验，Notebook 是另一种操作方式。

## 任务 A：实现对称量化

打开 `student_task.py`，补全 `symmetric_quantize()` 中的三处 TODO：

1. 计算每个输出通道的量化尺度。
2. 缩放、取整、截断，得到整数编码。
3. 反量化，得到浮点近似。

```bash
python student_task.py
```

未填写时脚本会提示完成 TODO。填写正确后，脚本会检查全零通道、数值范围及反量化结果，并打印课堂梯度下降与两通道补偿例题的结果。

思考：为什么既需要 `round`，又需要 `clip`？三轮梯度下降是否等于精确收敛？

## 任务 B：比较位宽

```bash
python lab.py --bits 8 4 2 --out results_bits
```

打开 `results_bits/report.html`，使用同一个 FP32 基线和测试集比较：

- 同一方法的 8、4、2 bit 结果。
- 同一位宽下 RTN、GPTQ-edu、AWQ-inspired 的结果。
- 分类准确率、logit MSE、实际模型文件字节数。

思考：权重 MSE 更小，输出 MSE 是否一定更小？4 bit 文件为什么不一定恰好是 FP32 文件的八分之一？

## 任务 C：比较校准样本数

```bash
python lab.py --bits 4 --calibration 16 --out cal16
python lab.py --bits 4 --calibration 128 --out cal128
```

比较 `cal16/results.csv` 与 `cal128/results.csv`。保持基线、位宽、测试集不变，只改变校准样本数。

RTN 不使用校准数据，两次运行的结果应一致。分析 GPTQ-edu 与 AWQ-inspired 的指标变化，并解释为什么更多校准样本不保证所有测试指标都改善。参数选择使用验证集，测试集用于最终报告。

## 实验报告

按照 `实验报告模板.txt` 整理：

- 完成后的 `student_task.py` 及运行检查结果。
- `results_bits/results.csv`、`cal16/results.csv`、`cal128/results.csv`。
- 至少三条由对照数据支持的结论。
- 小模型实验能够说明什么，以及不能据此推断什么。

## 模型、数据与量化约定

模型结构为 `64 → 32 → 10`，隐藏层使用 ReLU，输入是 8×8 灰度手写数字图像。训练使用交叉熵与 SGD。

数据使用 UCI Optical Recognition of Handwritten Digits 的官方训练与测试划分。官方训练集内部再分为训练 2672、校准 572、验证 579 个样本；官方测试集保留 1797 个样本。随机种子为 42。数据来源与 CC BY 4.0 许可见 `数据来源与许可.txt`。

实验只量化两个线性层的权重，偏置保持 FP32。权重矩阵形状为 `[输入通道, 输出通道]`，每个输出通道一个尺度，零点为 0。对称窄范围分别是 2 bit 的 −1…1、4 bit 的 −7…7、8 bit 的 −127…127；`np.rint` 在半整数处取最近的偶数。

| 方法 | 实验实现 |
| --- | --- |
| RTN | 使用每个输出通道的最大绝对值确定尺度，取最近量化值 |
| GPTQ-edu | 基于校准输入的二次误差目标，逐步量化并补偿剩余输入通道 |
| AWQ-inspired | 搜索正对角缩放，以层输出重构误差选择缩放参数 |

GPTQ-edu 省略完整算法的分组、列重排和块优化；AWQ-inspired 没有实现完整 clipping 和模型映射。它们用于理解核心思想，不是生产级 LLM 量化实现。两层量化使用 FP32 基线收集的校准激活。

## 指标与实验范围

- `accuracy`：独立测试集分类准确率。
- `accuracy_drop_pp`：相比 FP32 的准确率下降，单位为百分点。
- `weight_mse`：有效浮点权重的均方误差。
- `test_logit_mse`：量化模型与基线输出 logits 的均方误差。
- `packed_payload_bytes`：权重编码、尺度、偏置等数值数据的字节数。
- `actual_file_bytes`：包含文件头的实际文件大小；FP32 使用 `baseline.npz` 的大小。

整数权重真实打包到 `.qbin` 文件，推理前解包并反量化，再使用 NumPy 浮点矩阵乘。本实验评价数值质量和文件大小，不衡量低比特算子加速，也不能由此推断真实 LLM 的生成质量。

## 可选：运行数值检查

```bash
python -m unittest discover -s tests -v
```

检查覆盖课堂手算、误差补偿最优点、低位宽打包回读、零输入与病态输入、基线保护、模型预测回读。

可选训练命令为 `python lab.py --retrain`，会更新工作区中的 `baseline.npz`。课堂位宽和校准对照使用仓库提供的固定基线。

## 阅读材料

- [GPTQ 论文](https://arxiv.org/abs/2210.17323) · [作者实现](https://github.com/IST-DASLab/gptq)
- [AWQ 论文](https://arxiv.org/abs/2306.00978) · [作者实现](https://github.com/mit-han-lab/llm-awq)
- [SmoothQuant 论文](https://arxiv.org/abs/2211.10438) · [作者实现](https://github.com/mit-han-lab/smoothquant)
