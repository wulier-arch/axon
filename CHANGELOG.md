# 更新日志

本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)（SemVer）。

## [未发布]

### 计划中

见 [README 路线图](../README.md#路线图)：层、优化器、损失函数、训练循环、注意力、分词器、模型序列化、浏览器 demo、benchmark。

## [0.1.0] - 2026-10-02

首个可用的核心版本：自动微分 + 张量运算 + 卷积，全部通过梯度校验。

### 新增

- **Tensor 类**：`Float64Array` 存储，支持任意维；`variable()` 创建叶子，`tensor()` 创建常数
- **自动微分**：反向模式，拓扑排序后逆序回传；`backward` `collectLeaves` `zeroGrad`
- **梯度校验工具**：`checkGradient`（同时用一阶/二阶中心差分取优）、`numericalGradient`、`highOrderGradient`
- **逐元素算子**：`add` `sub` `mul` `div` `scale` `relu` `tanh` `sigmoid` `exp` `log`
- **矩阵运算**：`matmul`（2D 与批次 3D）、`softmax`（数值稳定）、`transpose`
- **形状算子**：`sum` `mean` `reshape` `gatherRows` `concat` `argmaxLast`
- **卷积与池化**：`conv2d`（im2col + GEMM）、`maxPool2d`、`avgPool2d`
- **测试套件**：34 个用例，覆盖所有算子的梯度正确性与形状校验

### 修复

开发过程中修正的三个关键缺陷，记录在此以免重蹈覆辙：

- **梯度链断链**：判断是否建图时误用 `requiresGrad`，导致组合运算（如 `sub(a,b)`）结果正确但梯度丢失。改用 `isGraphNode()`，区分「叶子变量」与「图中的中间节点」。
- **中间节点梯度被丢弃**：`accumulateGrad` 开头过滤 `requiresGrad`，导致中间张量的梯度无处存放。
- **matmul 权重梯度错误**：反向时只取了每行的首个元素 `out.grad[i*n]`，遗漏了其余输出列的贡献。正确公式为 `dL/db[p][j] = Σ_i out.grad[i][j]·a[i][p]`。2D 与 3D 两处均已修正。

### 已知限制

- 尚无层、优化器、训练循环，当前只能手动搭建计算图
- `concat` 仅支持 `axis=0`
- 无 GPU 加速，纯 CPU 实现
- 卷积输入为单样本（`[C,H,W]`），尚不支持批次维度

[未发布]: https://github.com/wulier-arch/axon/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/wulier-arch/axon/releases/tag/v0.1.0