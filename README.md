# axon

> 零依赖的神经网络框架。从张量运算到自动微分，全部手写，无任何第三方依赖。
> 同一份代码在浏览器和 Node.js 里都能跑。

[![License: MIT](https://img.shields.io/badge/license-MIT-a48aff.svg)](LICENSE)
[![CI](https://github.com/wulier-arch/axon/actions/workflows/ci.yml/badge.svg)](https://github.com/wulier-arch/axon/actions/workflows/ci.yml)
[![tests](https://img.shields.io/badge/tests-34%20passed-4ec9a0.svg)](test/)
[![dependencies](https://img.shields.io/badge/dependencies-0-ff6b6b.svg)](package.json)
[![node](https://img.shields.io/badge/node-%3E%3D18-5cb85c.svg)](package.json)

**🚧 早期开发中**：张量、自动微分、卷积已就绪并通过梯度校验；层、优化器、训练循环尚未提交。

---

## 为什么做这个

现有的 JS 深度学习库基本是 TensorFlow.js / ONNX 的封装。你没法在不装依赖的前提下读懂它们的反向传播，也没法在浏览器里断点进去看梯度是怎么流回去的。

axon 想做的是另一件事：**把这件事摊开给你看**。

- **零依赖** —— 不装任何东西，克隆下来就能跑
- **梯度可验证** —— 每个算子都自带有限差分对照，梯度错了立刻暴露，不是"跑起来没报错就算对"
- **双端同构** —— 浏览器里能训练，Node 里能训练，同一份代码

## 快速开始

无需安装：

```js
import { Tensor, add, mul, backward } from "axon";

const x = Tensor.variable([2, 3], [2]);       // 需要梯度的叶子
const y = mul(x, x).sum();                     // 前向
backward(y);                                  // 反向

console.log(x.grad);  // Float64Array [ 4, 6 ]  ← d(x²)/dx = 2x
```

验证梯度是否正确：

```js
import { checkGradient } from "axon";

const loss = () => mul(x, x).sum();
backward(loss());

const { passed, report } = checkGradient(loss, [x]);
console.log(report);      // maxAbsError=... maxRelError=...
console.log(passed);      // true
```

## 已实现

| 类别 | 算子 |
| --- | --- |
| 逐元素 | `add` `sub` `mul` `div` `scale` `relu` `tanh` `sigmoid` `exp` `log` |
| 矩阵 | `matmul`（2D 与批次 3D）、`softmax`、`transpose` |
| 形状 | `sum` `mean` `reshape` `gatherRows` `concat` `argmaxLast` |
| 卷积 | `conv2d`（im2col + GEMM）、`maxPool2d`、`avgPool2d` |
| 自动微分 | `backward`、`collectLeaves`、`zeroGrad` |
| 验证 | `checkGradient`、`numericalGradient`、`highOrderGradient` |

## 设计要点

### 反向模式自动微分

每个算子在前向时把自己的输入快照（`_prev`）和反向闭包（`_backward`）挂到输出张量上。`backward()` 做拓扑排序后逆序执行，保证用到某个张量梯度时它已经算好。

关键区分是这个概念：

```js
isGraphNode()   // 我是叶子变量，或我是某个算子的产物
```

中间张量（如 `mul` 的输出）`requiresGrad` 是 `false`——它不是叶子——但它**确实在图里**。如果用 `requiresGrad` 判断要不要建图，组合运算的梯度链就会断掉。这个坑 axon 踩过，也修好了。

### 梯度校验内建

`checkGradient` 用有限差分对照解析梯度，同时算一阶和二阶中心差分并逐元素取误差更小者。这样既能抓住解析梯度的错误，也能排除"数值梯度本身不够准"造成的误判。

```js
// 手写新算子后，这样验证你的梯度公式
const r = checkGradient(() => myOp(a, b).sum(), [a, b]);
if (!r.passed) throw new Error(r.report);
```

### 性能取向

`Float64Array` 存储（数值稳定，梯度检查可靠）；`matmul` 与 `conv2d` 走 i-k-j 循环顺序（对行主序缓存友好，比朴素 i-j-k 快数倍）；`conv2d` 用 im2col + GEMM 复用矩阵乘。

## 开发

```bash
git clone https://github.com/wulier-arch/axon.git
cd axon
npm test
```

无需 `npm install`——本项目零依赖。

## 路线图

- [ ] 层：Linear / Conv2D / BatchNorm / Dropout / Embedding
- [ ] 优化器：SGD / Momentum / Adam / AdamW
- [ ] 损失函数：MSE / CrossEntropy
- [ ] 训练循环：batch、early stopping、学习率调度
- [ ] 注意力：MultiHeadAttention / Transformer Block
- [ ] BPE 分词器
- [ ] 模型序列化（JSON）
- [ ] 浏览器端可视化 demo
- [ ] benchmark 与性能基线

## 参与贡献

欢迎提 Issue 和 PR，见 [CONTRIBUTING.md](CONTRIBUTING.md)。参与即表示同意 [行为准则](CODE_OF_CONDUCT.md)。

**如果你实现了新算子**，请务必附上梯度校验用例——这是本项目唯一的质量底线。

## 许可

[MIT](LICENSE) © 2026 wulier-arch