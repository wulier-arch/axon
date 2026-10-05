/**
 * 全连接层：y = xW + b
 *
 * 初始化用 He（Kaiming）初始化：权重服从 N(0, sqrt(2/fan_in))。
 * 理由：ReLU 在正半轴导数为 1，若用方差为 1 的初始化，每过一层方差就翻倍，
 * 深网络会直接发散。He 初始化让方差逐层保持稳定。
 */

import { Tensor } from "../tensor/tensor.js";
import {
  add, matmul, relu, tanh, sigmoid, softmax, reshape, transpose, scale, gelu,
} from "../tensor/ops.js";

/** 可复现的伪随机数发生器，保证初始化可重复 */
export function makeRng(seed = 42) {
  let s = seed >>> 0 || 1;
  const rng = function rng() {
    // xorshift32：周期足够长，统计性质够用
    s ^= s << 13; s >>>= 0;
    s ^= s >> 17;
    s ^= s << 5; s >>>= 0;
    return s / 4294967296;
  };
  rng.getState = () => s;
  rng.setState = (state) => {
    s = Number(state) >>> 0 || 1;
  };
  return rng;
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
      name: this.name,
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
    this.seed = seed;
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

  toJSON() {
    return {
      type: "Dropout",
      p: this.p,
      rngState: this.rng.getState(),
      name: this.name,
    };
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
      name: this.name,
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
      name: this.name,
    };
  }
}
/**
 * MultiHeadAttention：把序列的每个位置同时关注所有位置。
 *
 *   Q = X·W_q    K = X·W_k    V = X·W_v              [L, d]
 *   每个头：scores = Q·Kᵀ / √d_h → softmax → ·V      [L, L]
 *   拼接各头 → [L, d] → 再过 W_o 投影
 *
 * 关于 √d_h 那个除数：Q·Kᵀ 的每一项是 d_h 个乘积之和，方差随 d_h 线性增长。
 * 不除会让 logits 随维度变大，softmax 饱和成 one-hot，梯度趋近于 0——
 * 表现为「能跑但学不动」，且不报任何错。
 *
 * 实现上刻意不手写反向，而是把 matmul / transpose / reshape / softmax / add
 * 组合起来，梯度由 tape 自动串起来。这些算子每个都单独做过有限差分校验，
 * 组合后的正确性只需再校验一次整层的梯度即可。
 *
 * 头维拆分：d = h·d_h 按连续块切分，即第 i 个头占特征区间 [i·d_h, (i+1)·d_h)。
 *
 * @param {number} dModel 模型维度，必须能被 heads 整除
 * @param {number} heads  头数
 * @param {object} opts   causal：是否启用因果掩码（自回归解码时用）
 */
export class MultiHeadAttention {
  constructor(dModel, heads, opts = {}) {
    if (dModel % heads !== 0) {
      throw new Error(`MultiHeadAttention: dModel ${dModel} 不能被头数 ${heads} 整除`);
    }
    this.dModel = dModel;
    this.heads = heads;
    this.headDim = dModel / heads;
    this.scale = 1 / Math.sqrt(this.headDim);
    this.causal = opts.causal ?? false;
    this.name = "mha";

    const rng = makeRng(opts.seed ?? 21);
    // 注意力投影常用 1/√d 初始化：Q·Kᵀ 的量级才不会随维度放大
    const std = opts.initScale ?? 1 / Math.sqrt(dModel);
    const mk = () => {
      const data = new Float64Array(dModel * dModel);
      for (let i = 0; i < data.length; i++) {
        let u = 0, v = 0;
        while (u === 0) u = rng();
        while (v === 0) v = rng();
        data[i] = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v) * std;
      }
      return Tensor.variable(data, [dModel, dModel]);
    };
    this.wq = mk();
    this.wk = mk();
    this.wv = mk();
    this.wo = mk();
  }

  parameters() {
    return [this.wq, this.wk, this.wv, this.wo];
  }

  /**
   * @param {Tensor} x 形状 [seqLen, dModel]
   * @returns {Tensor} 形状 [seqLen, dModel]
   */
  forward(x) {
    if (x.shape[1] !== this.dModel) {
      throw new Error(
        `MultiHeadAttention: 输入维度 ${x.shape[1]}，期望 ${this.dModel}`
      );
    }
    const L = x.shape[0];
    const h = this.heads;
    const dh = this.headDim;

    const splitHeads = (t) => transpose(reshape(t, [L, h, dh]), [1, 0, 2]);

    const q = splitHeads(matmul(x, this.wq));   // [h, L, dh]
    const k = splitHeads(matmul(x, this.wk));
    const v = splitHeads(matmul(x, this.wv));

    // [h, L, dh] @ [h, dh, L] -> [h, L, L]
    const scores = matmul(q, transpose(k, [0, 2, 1]));
    let logits = scale(scores, this.scale);

    if (this.causal) {
      // 用大负数而非 -Infinity：避免全行被屏蔽时 exp 出现 NaN。
      // 位置 i 只允许关注 j ≤ i。
      const m = new Float64Array(L * L).fill(0);
      for (let i = 0; i < L; i++) {
        for (let j = i + 1; j < L; j++) m[i * L + j] = -1e9;
      }
      logits = add(logits, Tensor.tensor(m, [1, L, L]));
    }

    const attn = softmax(logits);              // [h, L, L]
    const ctx = matmul(attn, v);               // [h, L, dh]

    // [h, L, dh] -> [L, h, dh] -> [L, d]
    const merged = reshape(transpose(ctx, [1, 0, 2]), [L, this.dModel]);
    return matmul(merged, this.wo);
  }

  toJSON() {
    return {
      type: "MultiHeadAttention",
      dModel: this.dModel,
      heads: this.heads,
      causal: this.causal,
      wq: Array.from(this.wq.data),
      wk: Array.from(this.wk.data),
      wv: Array.from(this.wv.data),
      wo: Array.from(this.wo.data),
      name: this.name,
    };
  }
}

/**
 * TransformerBlock：现代 Transformer 的标准单元。
 *
 *   x ← x + Attention(LayerNorm(x))
 *   x ← x + FFN(LayerNorm(x))
 *
 * 默认是 **pre-LN**（先归一化再进子层），这是 2018 年之后的通行做法。
 * 与原论文的 post-LN（子层之后归一化）相比，pre-LN 在深层网络里更稳：
 * 残差路径上始终有一条恒等通路，梯度不必穿过归一化层才能回传，
 * 训练时不易发散、也不需要精细的 warmup 调参。
 *
 * FFN 是先扩再缩的两层前馈，dFF 通常取 4·dModel：
 *
 *   FFN(x) = W₂·gelu(W₁·x + b₁) + b₂
 *
 * 它逐位置作用、参数共享，参数量占整个 block 的大头；先扩张再收缩
 * 的结构让中间表示有更大的容量，而最终输出维度不变，可以直接残差相加。
 *
 * 和 MultiHeadAttention 一样，这里不手写反向——残差与子层都由
 * add / LayerNorm / Linear 等已校验过的算子组合而成。
 */
export class TransformerBlock {
  /**
   * @param {number} dModel 模型维度，须能被 heads 整除
   * @param {number} heads  注意力头数
   * @param {object} opts   dFF / causal / normFirst / seed / ffnActivation
   */
  constructor(dModel, heads, opts = {}) {
    this.dModel = dModel;
    this.name = "transformer";
    this.normFirst = opts.normFirst !== false;
    this.dFF = opts.dFF ?? dModel * 4;

    this.norm1 = new LayerNorm(dModel);
    this.attn = new MultiHeadAttention(dModel, heads, {
      causal: opts.causal ?? false,
      seed: opts.seed ?? 31,
    });
    this.norm2 = new LayerNorm(dModel);

    const act = opts.ffnActivation ?? "gelu";
    this.ff1 = new Linear(dModel, this.dFF, { seed: (opts.seed ?? 31) + 1 });
    this.ff2 = new Linear(this.dFF, dModel, { seed: (opts.seed ?? 31) + 2 });
    this.activation = act;
    this.actFn = act === "relu" ? relu : act === "gelu" ? gelu : tanh;
  }

  parameters() {
    return [
      ...this.norm1.parameters(),
      ...this.attn.parameters(),
      ...this.norm2.parameters(),
      ...this.ff1.parameters(),
      ...this.ff2.parameters(),
    ];
  }

  countParams() {
    return this.parameters().reduce((sum, t) => sum + t.size, 0);
  }

  /** 接受 [seqLen, dModel]，返回同形状 */
  forward(x) {
    if (x.shape[1] !== this.dModel) {
      throw new Error(`TransformerBlock: 输入维度 ${x.shape[1]}，期望 ${this.dModel}`);
    }
    let h = x;
    if (this.normFirst) {
      h = add(x, this.attn.forward(this.norm1.forward(x)));
      h = add(h, this._ffn(this.norm2.forward(h)));
    } else {
      h = this.norm1.forward(add(x, this.attn.forward(x)));
      h = this.norm2.forward(add(h, this._ffn(h)));
    }
    return h;
  }

  _ffn(x) {
    return this.ff2.forward(this.actFn(this.ff1.forward(x)));
  }

  toJSON() {
    return {
      type: "TransformerBlock",
      dModel: this.dModel,
      heads: this.attn.heads,
      dFF: this.dFF,
      normFirst: this.normFirst,
      activation: this.activation,
      causal: this.attn.causal,
      norm1: this.norm1.toJSON(),
      attn: this.attn.toJSON(),
      norm2: this.norm2.toJSON(),
      ff1: this.ff1.toJSON(),
      ff2: this.ff2.toJSON(),
      name: this.name,
    };
  }
}

/** 各激活函数的前向，便于 Sequential 统一调用 */
export const activations = { relu, tanh, sigmoid, softmax };
