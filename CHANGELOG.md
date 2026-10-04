# 更新日志

本项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)（SemVer）。

## [未发布]

### 计划中

见 [README 路线图](README.md#路线图)：BPE 分词器、BatchNorm、模型序列化（JSON）、
`conv2d` 批次维度。

层与 `Sequential`、优化器、损失函数、训练循环、基准测试已于 v0.2.0 发布，
浏览器端 demo 已于 v0.2.1 发布；`LayerNorm`、`Embedding`、`MultiHeadAttention`
与 `TransformerBlock` 已于 v0.3.0 发布。

## [0.3.0] - 2026-10-02

版本目标：补齐 Transformer 的两个前置层，为后续注意力实现铺路。

### 新增

- **`LayerNorm`**：沿特征维标准化，逐行独立计算，不依赖 batch 内其他样本
- **`Embedding`**：整数 id → 稠密向量查表
- **`MultiHeadAttention`**：含可选因果掩码。反向不手写，而是由
  `matmul`/`transpose`/`reshape`/`softmax`/`add` 组合而成，梯度经 tape 自动串联
- **`TransformerBlock`**：pre-LN（默认）与 post-LN 两种排布，可选因果掩码，
  FFN 默认用 GELU，`dFF` 默认为 `dModel` 的 4 倍
- **`gelu`**：tanh 近似版激活，Transformer 前馈网络常用。
  与 ReLU 的关键差别在负半轴不硬截断，而是平滑衰减到 0，保留了负区信息
- **`transpose` 推广为通用 N 维轴置换**（默认行为与原二维实现逐字一致）。
  多头注意力需要把 `[L, h, d_h]` 的头维提到最前再换回，二维版本做不到

`TransformerBlock` 与 `MultiHeadAttention` 一样不手写反向——残差与子层都由
已单独校验过的算子组合而成，梯度经 tape 自动串联。

### pre-LN 与 post-LN

默认采用 pre-LN（先归一化再进子层）：

    x ← x + Attention(LayerNorm(x))
    x ← x + FFN(LayerNorm(x))

残差路径上始终有一条恒等通路，梯度不必穿过归一化层才能回传，深层网络更稳。
原论文的 post-LN 通过 `normFirst: false` 保留。

### 如何判断「梯度写错」还是「差分不够准」

注意力链深约 10 个算子，一次有限差分校验的相对误差会到 1e-4 量级，
单看一个数字无法区分「解析梯度错了」与「中央差分不够准」。

判据是**残差是否随 eps 线性收敛**：

| eps | 实测 maxRelError | 比值 |
| --- | --- | --- |
| 1e-5 | 2.38e-4 | — |
| 1e-6 | 2.38e-5 | **10.01** |
| 1e-7 | 3.12e-5 | 0.76（舍入误差抬头） |

误差与 eps 成正比，正是中央差分截断误差的特征（O(h²) 的分子除以 2h 后为 O(h)）；
真梯度写错则会留下一个不随 eps 消失的下限。已把该判据固化为回归用例：
向 `scale` 的反向注入 1% 误差后，比值立刻变成 **1.00**，判据准确报警。

`TransformerBlock` 链更深（约 20 个算子），截断误差比注意力层又高一个量级，
但收敛比值同样是 10.0；向 `LayerNorm` 反向注入 0.5% 误差后比值变为 **1.03**。

比较须取截断主导的区间。注意力层取 1e-5 → 1e-6，block 取 1e-4 → 1e-5。
再往下降到 1e-7，舍入误差（∝1/h）抬头、误差不再下降，拿那一段比会得出错误结论。

### 实现过程中被梯度校验抓到的错误

`LayerNorm` 的反向最初写成把 γ_k 乘到整个括号上：

```js
dx_k = (γ_k/s) · [ dy_k − mean(dy) − x̂_k·mean(dy⊙x̂) ]   // ❌
```

推导后正确形式是 γ_k 只作用于 dy_k 这一项：

```js
dx_k = (1/s) · [ γ_k·dy_k − mean(dy⊙γ) − x̂_k·mean(dy⊙γ⊙x̂) ]   // ✅
```

两处差异叠加使相对误差达 1.16。关闭 γ 时测试反而通过——这正是静默错误的
典型形态：局部测试全绿，只有真正对照数值梯度才暴露。已把该形态补成回归用例。

### 验证

- 测试由 80 增至 **95**，Node 18/20/22/24 与本地 24 全部通过
- `gelu` 梯度 `maxRelError = 0`（绝对误差 3.6e-13）；`gelu(0)=0`、
  `gelu(1)≈0.8413`、`gelu(-2)≈-0.0455`，负半轴平滑衰减而非硬截断
- `LayerNorm` 梯度回归保护：把 γ_k 乘回整个括号，校验立即报 `maxRelError=6.09e-1`
- 收敛判据回归保护：向 `scale` 反向注入 1% 误差，比值 10.01 → **1.00**；
  向 `LayerNorm` 反向注入 0.5% 误差，比值 10.0 → **1.03**，均准确报警
- 残差通路：把 block 子层权重全置零，输出与输入逐位相同（漏写残差相加会立刻暴露）
- 因果性：改动后 2 个 token 的输入，前 2 个位置输出逐位相同
- 端到端：`Embedding → 两层 TransformerBlock → Linear`，交叉熵从 ln(3)≈1.0986 降到 0.2 以下

## [0.2.1] - 2026-10-02

版本目标：补齐演示物料，并打通 npm 首发。

### 新增

- **在线 Demo**：<https://wulier-arch.github.io/axon/demo/>，无打包步骤，浏览器直接 `import` `src/` 源码运行
- **训练动画 GIF**：`docs/demo.gif`（900×472，30 帧），用 axon 自身训练螺旋分类并逐帧记录，展示决策边界从粗糙到完全分离的全过程

### 修复

- **README 动画在 npm 页面裂图**：`docs/` 不在 `package.json` 的 `files` 中（npm 包仅 16 个源文件），相对路径 `docs/demo.gif` 在 GitHub 上正常，但 npm 按仓库根解析会指向不存在的文件。改用 `raw.githubusercontent.com` 绝对地址，两端均可渲染。

### 验证

- Node 18 / 20 / 22 / 24 各 60/60 测试通过；CI 矩阵（18/20/22）与 Pages 部署均为绿色
- `npm pack` 产物 16 文件 / 24.1 kB，`dependencies` 与 `devDependencies` 均为空
- 从 `v0.2.1` tag 全新克隆跑测试 60/60；从打包产物全新安装后跑 README 示例 14/14（耗时 0.2s），其中线性回归学到 `w = 2.511`（真值 2.5）、`checkGradient` 最大相对误差 0

## [0.2.0] - 2026-10-02

版本目标：从「算子库」升级为「能真正训练的框架」。

### 新增

- **层**：`Linear`（He 初始化，`std = sqrt(2/fan_in)`，避免深层网络方差逐层翻倍而发散）、`Dropout`、`Sequential`
- **损失函数**：`crossEntropy`（融合实现，log-sum-exp 数值稳定）、`mse`、`binaryCrossEntropy`、`accuracy`
- **优化器**：`SGD`、`Momentum`、`Adam`、`AdamW`（权重衰减与梯度解耦）、`RMSProp`
- **学习率调度**：`Scheduler`（constant / cosine / step / exponential）
- **训练循环** `Trainer`：batch 切分与打乱、epoch 循环、早停、训练历史记录
- **合成数据集**：XOR、螺旋、高斯簇、线性回归，全部程序生成无需下载
- **基准测试** `npm run benchmark`：吞吐与训练能力量化
- **测试套件**扩充至 57 个用例，其中 23 个为端到端训练验证

### 修复

- **`add` 广播规则不完整**：缺少 `[m,n] + [n]` 按行广播，导致全连接层加偏置直接报错。改用完整的 numpy 右对齐规则。
- **优化器缓冲区越界**：`SGD`/`Momentum` 按 `params[0].size` 给所有参数分配动量缓冲区，但权重 `[in,out]` 与偏置 `[out]` 元素数不同，偏置缓冲区过短导致越界写入。`Float64Array` 对越界写入静默忽略，训练会悄悄学错且不报错。改为逐参数按 `p.size` 分配。
- **`mse` 梯度为 NaN**：`sum` 默认沿最后一维求和，对 `[N,1]` 的回归目标得到 `[N]` 而非标量。损失值正常但梯度全坏，属静默错误。
- **`Trainer` 未切片回归目标**：lossFn 收到整份数据而非 batch 切片，形状不匹配。

### 验证

| 数据集 | 准确率 | 损失 | 耗时 |
| --- | --- | --- | --- |
| XOR (4 样本) | 100.0% | 0.0002 | 6ms |
| 螺旋 (300 样本) | 100.0% | 0.0248 | 223ms |
| 高斯簇 (400 样本) | 100.0% | 0.0004 | 9ms |

线性回归学到 `w = 2.511`（真值 2.5）、`b = -1.2115`（真值 -1.2）。

性能（Node v24 · arm64 实测）：`matmul` 512×512 达 2.36 GFLOP/s。

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

[未发布]: https://github.com/wulier-arch/axon/compare/v0.3.0...HEAD
[0.3.0]: https://github.com/wulier-arch/axon/compare/v0.2.1...v0.3.0
[0.2.1]: https://github.com/wulier-arch/axon/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/wulier-arch/axon/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/wulier-arch/axon/releases/tag/v0.1.0