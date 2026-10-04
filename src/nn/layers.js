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

/**
 * LayerNorm：沿特征维标准化。
 *
 *   μ = mean(x)       s = sqrt(var(x) + eps)       x̂ = (x - μ) / s
 *   y = γ ⊙ x̂ + β
 *
 * 逐行独立计算，不依赖 batch 内其他样本——这与 BatchNorm 是本质区别，
 * 也是它在序列任务（每个样本长度可变）里更常用的原因。
 *
 * 反向最容易写错的是 dx：x̂ 同时依赖整行的 μ 与 s，而它们又都依赖 x 的每一个分量。
 * 推导时 ∂(x_i − μ)/∂x_k = δ_ik − 1/N（漏掉 −1/N 那一项就会得到错误结果），
 * 顺着算下去：
 *
 *   ∂x̂_i/∂x_k = (1/s)·[ δ_ik − 1/N − x̂_i·x̂_k/N ]
 *   dx_k = (γ_k/s)·[ dy_k − mean(dy) − x̂_k·mean(dy ⊙ x̂) ]
 *
 * 三项耦合都不能少。只写 dx_k = dy_k/s 时 loss 仍能下降，但梯度是错的，
 * 正是本项目要抓的那类静默错误。第三项的符号尤其容易写反。
 *
 * γ 初始化为 1、β 为 0：训练初期整层等价于恒等映射，不会破坏已学到的
 * 表征（若 γ 随机初始化，深层网络初期就会被放大或压缩）。
 */
export class LayerNorm {
  /**
   * @param {number} features 特征维，必须等于输入的列数
   * @param {object} opts eps / gamma / beta
   */
  constructor(features, opts = {}) {
    this.features = features;
    this.eps = opts.eps ?? 1e-5;
    this.name = "layernorm";
    this.useGamma = opts.gamma !== false;
    this.useBeta = opts.beta !== false;

    this.gamma = this.useGamma
      ? Tensor.variable(new Float64Array(features).fill(1), [features])
      : null;
    this.beta = this.useBeta
      ? Tensor.variable(new Float64Array(features), [features])
      : null;
  }

  parameters() {
    const params = [];
    if (this.gamma) params.push(this.gamma);
    if (this.beta) params.push(this.beta);
    return params;
  }

  /** 接受 [batch, features]，逐行标准化 */
  forward(x) {
    const [rows, cols] = x.shape;
    if (cols !== this.features) {
      throw new Error(`LayerNorm: 特征维不匹配，期望 ${this.features}，实际 ${cols}`);
    }

    const n = cols;
    const out = new Tensor(new Float64Array(rows * n), [rows, n]);
    // 反向要用：每行的 x̂ 与 s
    const xhat = new Float64Array(rows * n);
    const stds = new Float64Array(rows);

    for (let r = 0; r < rows; r++) {
      const off = r * n;
      let mean = 0;
      for (let k = 0; k < n; k++) mean += x.data[off + k];
      mean /= n;

      let varsum = 0;
      for (let k = 0; k < n; k++) {
        const d = x.data[off + k] - mean;
        varsum += d * d;
      }
      const s = Math.sqrt(varsum / n + this.eps);
      stds[r] = s;

      for (let k = 0; k < n; k++) {
        const xh = (x.data[off + k] - mean) / s;
        xhat[off + k] = xh;
        out.data[off + k] = (this.gamma ? this.gamma.data[k] : 1) * xh
          + (this.beta ? this.beta.data[k] : 0);
      }
    }

    const gamma = this.gamma, beta = this.beta;
    if (x.isGraphNode()) {
      out._prev = [x, xhat, stds];
      out._op = "layernorm";
      out._backward = () => {
        const g = new Float64Array(x.size);
        if (gamma) {
          const gg = new Float64Array(n);
          for (let r = 0; r < rows; r++) {
            const off = r * n;
            for (let k = 0; k < n; k++) gg[k] += out.grad[off + k] * xhat[off + k];
          }
          gamma.accumulateGrad(gg);
        }
        if (beta) {
          const gb = new Float64Array(n);
          for (let r = 0; r < rows; r++) {
            const off = r * n;
            for (let k = 0; k < n; k++) gb[k] += out.grad[off + k];
          }
          beta.accumulateGrad(gb);
        }
        for (let r = 0; r < rows; r++) {
          const off = r * n;
          // 注意 γ 的下标是 i（输出维）而非 k：y_i = γ_i·x̂_i + β_i，
          // 求导时 dy·x̂ 这两项均值都要带 γ_i 加权，漏掉就会得到错误梯度。
          let dyMean = 0, dyXhatMean = 0;
          for (let k = 0; k < n; k++) {
            const w = out.grad[off + k] * (gamma ? gamma.data[k] : 1);
            dyMean += w;
            dyXhatMean += w * xhat[off + k];
          }
          dyMean /= n;
          dyXhatMean /= n;
          const invS = 1 / stds[r];
          for (let k = 0; k < n; k++) {
            // γ_k 只作用于 dy_k 这一项，不能乘到整个括号上：
            //   dx_k = (1/s)·[ γ_k·dy_k − mean(dy⊙γ) − x̂_k·mean(dy⊙γ⊙x̂) ]
            // 后两项已是全行耦合的量，γ 已计入其中。
            const gk = gamma ? gamma.data[k] : 1;
            g[off + k] = invS
              * (gk * out.grad[off + k] - dyMean - xhat[off + k] * dyXhatMean);
          }
        }
        x.accumulateGrad(g);
      };
    }
    return out;
  }

  toJSON() {
    return {
      type: "LayerNorm",
      features: this.features,
      eps: this.eps,
      gamma: this.gamma ? Array.from(this.gamma.data) : null,
      beta: this.beta ? Array.from(this.beta.data) : null,
    };
  }
}

/**
 * Embedding：把整数 id 映射成稠密向量，本质是查表。
 *
 * 输入是 id 序列而非张量，因此手写 _backward 而不复用通用算子。
 *
 * 反向的关键是**梯度必须按词累加**：同一个词表条目会在一句话里出现多次，
 * 每次都贡献一份梯度。若写成 weight.grad[id] = g 覆盖而非累加，后出现的
 * 词会抹掉先前的贡献——loss 照样下降，但学出来的嵌入是错的。
 */
export class Embedding {
  /**
   * @param {number} vocabSize 词表大小
   * @param {number} features 嵌入维度
   */
  constructor(vocabSize, features, opts = {}) {
    this.vocabSize = vocabSize;
    this.features = features;
    this.name = "embedding";

    const rng = makeRng(opts.seed ?? 7);
    // 标准正态初始化：嵌入应在同一量级上，否则下游层的尺度会被词表大小带偏
    const data = new Float64Array(vocabSize * features);
    for (let i = 0; i < data.length; i++) {
      let u = 0, v = 0;
      while (u === 0) u = rng();
      while (v === 0) v = rng();
      data[i] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
    }
    this.weight = Tensor.variable(data, [vocabSize, features]);
  }

  parameters() {
    return [this.weight];
  }

  /**
   * @param {number[]|Int32Array} ids 词 id 序列
   * @returns {Tensor} 形状 [ids.length, features]
   */
  forward(ids) {
    const w = this.weight;
    for (const id of ids) {
      if (!Number.isInteger(id) || id < 0 || id >= this.vocabSize) {
        throw new Error(`Embedding: 词 id 越界 [0,${this.vocabSize})，收到 ${id}`);
      }
    }

    const n = ids.length;
    const f = this.features;
    const out = new Tensor(new Float64Array(n * f), [n, f]);
    for (let i = 0; i < n; i++) {
      const src = ids[i] * f;
      const dst = i * f;
      for (let j = 0; j < f; j++) out.data[dst + j] = w.data[src + j];
    }

    if (w.isGraphNode()) {
      const idsCopy = Int32Array.from(ids);
      out._prev = [w, idsCopy];
      out._op = "embedding";
      out._backward = () => {
        // 先在临时缓冲里累加，最后一次性写回：重复 id 必须求和而非覆盖
        const g = new Float64Array(w.size);
        for (let i = 0; i < n; i++) {
          const row = idsCopy[i] * f;
          const off = i * f;
          for (let j = 0; j < f; j++) g[row + j] += out.grad[off + j];
        }
        w.accumulateGrad(g);
      };
    }
    return out;
  }

  toJSON() {
    return {
      type: "Embedding",
      vocabSize: this.vocabSize,
      features: this.features,
      weight: Array.from(this.weight.data),
    };
  }
}
/** 各激活函数的前向，便于 Sequential 统一调用 */
export const activations = { relu, tanh, sigmoid, softmax };