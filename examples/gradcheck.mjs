/**
 * 梯度校验演示 —— 用有限差分验证自动微分算出的梯度。
 *
 * 运行：node examples/gradcheck.mjs
 *
 * 输出里的 maxAbsError 是解析梯度与数值梯度的最大绝对误差。
 * 双精度、h=1e-5 下，量级 1e-7 属于正常；如果差到 1e-3 以上，
 * 基本可以确定是某个算子的反向写错了，而不是数值精度问题。
 */
import {
  Tensor, matmul, add, tanh, backward, checkGradient,
} from "../src/index.js";

// 2 → 3 → 2 的小网络。权重刻意控制在 ±0.4 以内：
// tanh 一旦饱和，导数接近 0，相对误差会被放大成噪声。
const W1 = Tensor.variable([0.2, -0.3, 0.1, 0.4, -0.2, 0.3], [2, 3]);
const b1 = Tensor.variable([0.05, -0.1, 0.02], [1, 3]);
const W2 = Tensor.variable([0.3, -0.2, 0.1, 0.25, -0.35, 0.15], [3, 2]);
const b2 = Tensor.variable([0.1, -0.05], [1, 2]);

const x = Tensor.tensor([0.5, -0.2, -0.1, 0.3, 0.4, 0.1], [3, 2]);

const lossFn = () =>
  add(matmul(tanh(add(matmul(x, W1), b1)), W2), b2).sum();

const inputs = [W1, b1, W2, b2];

// 必须先清空梯度：上次反向的累加结果还留在 .grad 里，
// 而数值梯度是按当前参数算的，两者混在一起必然对不上。
for (const t of inputs) t.zeroGrad();

const loss = lossFn();
backward(loss);

const result = checkGradient(lossFn, inputs);

console.log("loss =", loss.data[0].toFixed(6), "  params =",
  inputs.reduce((sum, t) => sum + t.size, 0));
console.log(result.report);
console.log("passed =", result.passed);

if (!result.passed) process.exitCode = 1;
