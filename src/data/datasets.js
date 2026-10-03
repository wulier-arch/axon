/**
 * 合成数据集：无需下载即可验证框架是否真的能训练。
 * 覆盖线性可分、非线性可分、螺旋三类难度递进的场景。
 */

import { Tensor } from "../tensor/tensor.js";
import { makeRng } from "../nn/layers.js";

/** 两个高斯簇，线性可分，逻辑回归就能分开 */
export function makeBlobs({ samples = 200, features = 2, classes = 2, scale = 2, seed = 7 } = {}) {
  const rng = makeRng(seed);
  const x = new Float64Array(samples * features);
  const y = new Array(samples);

  const centers = [];
  for (let c = 0; c < classes; c++) {
    const center = [];
    for (let f = 0; f < features; f++) center.push(rng() * scale * 2 - scale);
    centers.push(center);
  }

  for (let i = 0; i < samples; i++) {
    const label = Math.floor(rng() * classes);
    y[i] = label;
    for (let f = 0; f < features; f++) {
      // 高斯噪声：Box-Muller 的简化版，用两次均匀分布近似
      const n = (rng() + rng() + rng() + rng() - 2) * 0.7;
      x[i * features + f] = centers[label][f] + n * scale * 0.35;
    }
  }

  return { x: Tensor.tensor(x, [samples, features]), y, classes };
}

/**
 * 螺旋数据集：两层同心螺旋交错，线性模型完全无法拟合，
 * 必须靠非线性层（ReLU）+ 足够的隐藏层数才能分开。
 * 这是检验框架是否"真的会学"的经典场景。
 */
export function makeSpiral({ samples = 300, noise = 0.12, seed = 42 } = {}) {
  const rng = makeRng(seed);
  const perClass = Math.floor(samples / 2);
  const x = new Float64Array(samples * 2);
  const y = new Array(samples);

  for (let c = 0; c < 2; c++) {
    for (let i = 0; i < perClass; i++) {
      const idx = c * perClass + i;
      const t = (i / perClass) * 3 * Math.PI;
      const r = 0.4 + (i / perClass) * 1.6;
      // 两类的角度偏移 π，使其交错
      const offset = c * Math.PI;
      x[idx * 2] = r * Math.sin(t + offset) + (rng() - 0.5) * noise;
      x[idx * 2 + 1] = r * Math.cos(t + offset) + (rng() - 0.5) * noise;
      y[idx] = c;
    }
  }
  // 洗牌，避免按类连续排列导致 batch 有偏
  const order = Array.from({ length: samples }, (_, i) => i);
  for (let i = samples - 1; i > 0; i--) {
    const j = Math.floor(rng() * (i + 1));
    [order[i], order[j]] = [order[j], order[i]];
  }

  const xs = new Float64Array(samples * 2);
  const ys = new Array(samples);
  for (let i = 0; i < samples; i++) {
    xs[i * 2] = x[order[i] * 2];
    xs[i * 2 + 1] = x[order[i] * 2 + 1];
    ys[i] = y[order[i]];
  }

  return { x: Tensor.tensor(xs, [samples, 2]), y: ys, classes: 2 };
}

/** 异或：2 输入 4 样本，单层网络学不会，经典反例 */
export function makeXor() {
  return {
    x: Tensor.tensor([0, 0, 0, 1, 1, 0, 1, 1], [4, 2]),
    y: [0, 1, 1, 0],
    classes: 2,
  };
}

/** 线性回归：y = wx + b */
export function makeLinearRegression({ samples = 100, seed = 5 } = {}) {
  const rng = makeRng(seed);
  const x = new Float64Array(samples);
  const y = new Float64Array(samples);
  const w = 2.5, b = -1.2;
  for (let i = 0; i < samples; i++) {
    x[i] = rng() * 2;
    y[i] = w * x[i] + b + (rng() - 0.5) * 0.1;
  }
  return {
    x: Tensor.tensor(x, [samples, 1]),
    yTensor: Tensor.tensor(y, [samples, 1]),
    trueW: w,
    trueB: b,
  };
}