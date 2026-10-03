# 贡献指南

感谢你愿意为 axon 出力。这是一个从零实现的神经网络框架，欢迎所有层次的贡献。

## 项目原则

这三条决定了这个项目的方向，提 PR 前请先认同：

1. **零依赖**：不引入任何运行时或开发期依赖。用 Node 内置的 `node:test` 而不是 Jest，用 `node --test` 而不是 vitest。测试只用 `Float64Array` 和标准库。
2. **梯度必须可验证**：任何新算子都要附梯度校验用例，用框架自带的 `checkGradient` 对照有限差分。这是本项目的质量底线，不是可选项。
3. **双端同构**：代码不能依赖 Node 专有 API，否则浏览器里就跑不了。

## 环境

```bash
git clone https://github.com/wulier-arch/axon.git
cd axon

# 无需 npm install
npm test
```

需要 Node 18 以上。

## 实现新算子

以一个逐元素算子为例，结构大致如下：

```js
export function myOp(a, b) {
  const out = new Tensor(new Float64Array(a.size), a.shape);
  for (let i = 0; i < out.size; i++) out.data[i] /* 计算 */;

  // 判据用 isGraphNode() 而非 requiresGrad，
  // 否则组合运算的梯度链会断
  if (track(a, b)) {
    out._prev = [a, b];
    out._op = "myOp";
    out._backward = () => {
      // 上游梯度在 out.grad 里，把它按链式法则分发给各输入
      if (a.isGraphNode()) a.accumulateGrad(/* d out / d a */);
      if (b.isGraphNode()) b.accumulateGrad(/* d out / d b */);
    };
  }
  return out;
}
```

三个容易踩的坑：

| 坑 | 后果 | 正确做法 |
| --- | --- | --- |
| 用 `requiresGrad` 判断是否建图 | 组合运算梯度丢失 | 用 `isGraphNode()` |
| 在 `_backward` 里过滤 `requiresGrad` | 中间节点梯度存不下 | 无条件写回，由算子自己决定 |
| 累加时漏掉某个维度 | 梯度"部分正确"，极难发现 | 每个元素都用数值梯度对照 |

## 必须通过的自检

```bash
npm test
```

测试里包含梯度校验，任何算子的梯度公式写错都会被 `checkGradient` 抓到。

## 提交规范

使用 [Conventional Commits](https://www.conventionalcommits.org/zh-hans/)：

```
feat: 添加 LayerNorm 层
fix: 修正 maxPool2d 梯度在重叠窗口下的累加
docs: 补充 conv2d 的形状约束说明
test: 为 softmax 增加数值稳定性用例
perf: matmul 改用分块提升缓存命中率
```

## 代码风格

- 2 空格缩进，LF 换行，文件末尾保留换行（见 `.editorconfig`）
- 注释解释**为什么这么做**，而不是复述代码在做什么
- 涉及数值稳定性的地方（指数衰减、除零、溢出）必须写明原因

## 行为准则

参与即表示同意 [行为准则](CODE_OF_CONDUCT.md)。

## 许可

提交贡献即表示你同意你的作品按 [MIT 协议](LICENSE) 授权。