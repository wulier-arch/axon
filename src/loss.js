/**
 * 损失函数：返回标量 Tensor，直接参与反向传播。
 *
 * 设计原则：内部一律用 logits 计算，不做 softmax 融合。
 * 融合版本（logSoftmax + NLL）数值更稳且更快，但会让梯度校验更难写对，
 * 这里优先保证正确性，稳定性靠 log-sum-exp 技巧保证。
 */

import { Tensor } from "./tensor/tensor.js";
import { sum, mul, sub, log, softmax, scale } from "./tensor/ops.js";

/**
 * 均方误差：(a-b)² 的全局均值。
 *
 * 注意用 sum(a, 0) 而非默认的 sum(a)：默认 dim=-1 表示沿最后一维求和，
 * 对 [N,1] 的回归目标会得到 [N] 而非标量，语义就错了。
 */
export function mse(pred, target) {
  const diff = sub(pred, target);
  const squared = mul(diff, diff);
  return scale(sum(squared, 0), 1 / (pred.size || 1));
}

/**
 * 交叉熵（softmax + NLL 融合版）。
 *
 * log-sum-exp 技巧：
 *   log Σ exp(z) = max(z) + log Σ exp(z - max(z))
 * 直接算 exp(z) 在 z 较大时会溢出为 Infinity，先减去最大值可完全避免。
 */
export function crossEntropy(logits, labels) {
  const [rows, classes] = logits.shape;
  const rowStride = classes;

  const out = new Tensor([0], [1]);
  let total = 0;

  for (let r = 0; r < rows; r++) {
    const off = r * rowStride;
    let max = -Infinity;
    for (let c = 0; c < classes; c++) max = Math.max(max, logits.data[off + c]);

    let sumExp = 0;
    for (let c = 0; c < classes; c++) sumExp += Math.exp(logits.data[off + c] - max);
    const logSumExp = max + Math.log(sumExp);

    const label = labels instanceof Array ? labels[r] : labels.data[r];
    total += logSumExp - logits.data[off + label];
  }

  out.data[0] = total / rows;

  if (logits.isGraphNode()) {
    out._prev = [logits, labels, classes];
    out._op = "cross_entropy";
    out._backward = () => {
      // d(loss)/d(logits) = (softmax - onehot) / rows
      const probs = softmax(logits);
      const g = new Float64Array(logits.size);
      for (let r = 0; r < rows; r++) {
        const off = r * rowStride;
        const label = labels instanceof Array ? labels[r] : labels.data[r];
        for (let c = 0; c < classes; c++) {
          const p = probs.data[off + c];
          const isTarget = c === label ? 1 : 0;
          g[off + c] = (p - isTarget) / rows;
        }
      }
      logits.accumulateGrad(g);
    };
  }
  return out;
}

/**
 * 二元交叉熵：-(y·log(p) + (1-y)·log(1-p))
 *
 * 概率被 clamp 到 [EPS, 1-EPS]：log(0) 会得到 -Infinity，
 * 一次越界就能让整个训练的损失变成 NaN。
 */
const EPS = 1e-7;

function clampProb(t) {
  const out = new Tensor(new Float64Array(t.size), t.shape);
  for (let i = 0; i < t.size; i++) {
    out.data[i] = Math.min(1 - EPS, Math.max(EPS, t.data[i]));
  }
  if (t.isGraphNode()) {
    // clamp 区间内导数为 1；越界点梯度取 0
    const mask = new Float64Array(t.size);
    for (let i = 0; i < t.size; i++) {
      mask[i] = t.data[i] > EPS && t.data[i] < 1 - EPS ? 1 : 0;
    }
    out._prev = [t, mask];
    out._op = "clamp";
    out._backward = () => {
      const g = new Float64Array(t.size);
      for (let i = 0; i < t.size; i++) g[i] = out.grad[i] * mask[i];
      t.accumulateGrad(g);
    };
  }
  return out;
}

export function binaryCrossEntropy(pred, target) {
  const p = clampProb(pred);
  // 1 - p
  const oneMinusP = new Tensor(new Float64Array(p.size), p.shape);
  for (let i = 0; i < p.size; i++) oneMinusP.data[i] = 1 - p.data[i];

  const total = sub(mul(target, log(p)), mul(sub(Tensor.tensor([1], [1]), target), log(oneMinusP)));
  return scale(sum(total), -1 / pred.size);
}

/** 预测正确率，用于训练过程的可读指标 */
export function accuracy(logits, labels) {
  const probs = softmax(logits);
  const [rows, classes] = logits.shape;
  let correct = 0;
  for (let r = 0; r < rows; r++) {
    let best = 0;
    for (let c = 1; c < classes; c++) {
      if (probs.data[r * classes + c] > probs.data[r * classes + best]) best = c;
    }
    const label = labels instanceof Array ? labels[r] : labels.data[r];
    if (best === label) correct++;
  }
  return correct / rows;
}