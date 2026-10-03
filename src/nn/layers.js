/**
 * 全连接层：y = xW + b
 *
 * 初始化用 He（Kaiming）初始化：权重服从 N(0, sqrt(2/fan_in))。
 * 理由：ReLU 在正半轴导数为 1，若用方差为 1 的初始化，每过一层方差就翻倍，
 * 深网络会直接发散。He 初始化让方差逐层保持稳定。
 */

import { Tensor } from "../tensor/tensor.js";
import { add, matmul, relu, tanh, sigmoid, softmax } from "../tensor/ops.js";

/** 可复现的伪随机数发生器，保证初始化可重复 */
export function makeRng(seed = 42) {
  let s = seed >>> 0 || 1;
  return function rng() {
    // xorshift32：周期足够长，统计性质够用
    s ^= s << 13; s >>>= 0;
    s ^= s >> 17;
    s ^= s << 5; s >>>= 0;
    return s / 4294967296;
  };
}

/** Box-Muller 变换，把均匀分布转成标准正态 */
function gaussian(rng) {
  let u = 0, v = 0;
  while (u === 0) u = rng();
  while (v === 0) v = rng();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
}

export class Linear {
  /**
   * @param {number} inFeatures 输入维度
   * @param {number} outFeatures 输出维度
   * @param {object} opts bias / activation / seed / initScale
   */
  constructor(inFeatures, outFeatures, opts = {}) {
    this.inFeatures = inFeatures;
    this.outFeatures = outFeatures;
    this.useBias = opts.bias !== false;
    this.activation = opts.activation || null;
    this.name = opts.name || "linear";

    const rng = makeRng(opts.seed ?? 1234);
    // He 初始化：std = sqrt(2 / fan_in)
    const std = opts.initScale ?? Math.sqrt(2 / inFeatures);
    const w = new Float64Array(inFeatures * outFeatures);
    for (let i = 0; i < w.length; i++) w[i] = gaussian(rng) * std;

    this.weight = Tensor.variable(w, [inFeatures, outFeatures]);
    this.bias = this.useBias
      ? Tensor.variable(new Float64Array(outFeatures), [outFeatures])
      : null;
  }

  /** 参数列表，供优化器遍历 */
  parameters() {
    return this.bias ? [this.weight, this.bias] : [this.weight];
  }

  /** 接受 [batch, inFeatures] */
  forward(x) {
    let out = matmul(x, this.weight);
    if (this.bias) out = add(out, this.bias);
    if (this.activation === "relu") out = relu(out);
    else if (this.activation === "tanh") out = tanh(out);
    else if (this.activation === "sigmoid") out = sigmoid(out);
    return out;
  }

  toJSON() {
    return {
      type: "Linear",
      inFeatures: this.inFeatures,
      outFeatures: this.outFeatures,
      activation: this.activation,
      weight: Array.from(this.weight.data),
      bias: this.bias ? Array.from(this.bias.data) : null,
    };
  }
}

/**
 * Dropout：训练时按概率置零，推理时整体除以 keep 概率做期望补偿。
 * 只在前向丢弃、反向按掩码放行，比"加噪声"的做法梯度更干净。
 */
export class Dropout {
  constructor(p = 0.5, seed = 999) {
    this.p = p;
    this.rng = makeRng(seed);
    this.name = "dropout";
  }

  forward(x) {
    if (this.p <= 0) return x;
    const keep = 1 - this.p;
    const mask = new Float64Array(x.size);
    const out = new Tensor(new Float64Array(x.size), x.shape);
    for (let i = 0; i < x.size; i++) {
      const on = this.rng() < keep;
      mask[i] = on ? 1 / keep : 0;   // 反向缩放期望
      out.data[i] = on ? x.data[i] : 0;
    }
    if (x.isGraphNode()) {
      out._prev = [x, mask];
      out._op = "dropout";
      out._backward = () => {
        const g = new Float64Array(x.size);
        for (let i = 0; i < x.size; i++) g[i] = out.grad[i] * mask[i];
        x.accumulateGrad(g);
      };
    }
    return out;
  }
}

/** 各激活函数的前向，便于 Sequential 统一调用 */
export const activations = { relu, tanh, sigmoid, softmax };