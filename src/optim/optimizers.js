/**
 * 优化器。
 *
 * 所有优化器共享同一个接口：
 *   step(params, grads)  用 grads 更新 params 缓冲区（原地修改）
 *
 * 一阶动量按元素维护，二阶动量同理；t 是步数计数。
 */

export class SGD {
  constructor({ lr = 0.01, momentum = 0 } = {}) {
    this.lr = lr;
    this.momentum = momentum;
    this.velocity = null;
  }

  step(params, grads) {
    // 动量缓冲区必须逐参数按各自尺寸分配：
    // 权重 [in,out] 与偏置 [out] 的元素数不同，
    // 统一按 params[0].size 分配会让偏置缓冲区越界或过短，
    // 越界写入在 Float64Array 上是静默的，训练会悄悄学错。
    if (!this.velocity) {
      this.velocity = params.map((p) => new Float64Array(p.size));
    }
    for (let i = 0; i < params.length; i++) {
      const p = params[i], g = grads[i];
      if (!g || this.velocity[i].length !== p.size) continue;
      for (let j = 0; j < p.size; j++) {
        this.velocity[i][j] = this.momentum * this.velocity[i][j] + g[j];
        p.data[j] -= this.lr * this.velocity[i][j];
      }
    }
  }
}

export class Momentum {
  constructor({ lr = 0.01, beta = 0.9 } = {}) {
    this.lr = lr;
    this.beta = beta;
    this.v = null;
  }

  step(params, grads) {
    // 同 SGD：动量缓冲区需按各参数自身尺寸分配
    if (!this.v) this.v = params.map((p) => new Float64Array(p.size));
    for (let i = 0; i < params.length; i++) {
      const p = params[i], g = grads[i];
      if (!g || this.v[i].length !== p.size) continue;
      for (let j = 0; j < p.size; j++) {
        this.v[i][j] = this.beta * this.v[i][j] + g[j];
        p.data[j] -= this.lr * this.v[i][j];
      }
    }
  }
}

/**
 * Adam：m 用一阶矩估计，v 用二阶矩估计，再做偏差校正。
 *
 * 偏差校正不可省：m、v 初始为 0，前几步会被严重低估，
 * 导致最初的学习率远小于设定值，训练起步很慢。
 */
export class Adam {
  constructor({ lr = 0.001, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weightDecay = 0 } = {}) {
    this.lr = lr;
    this.beta1 = beta1;
    this.beta2 = beta2;
    this.eps = eps;
    this.weightDecay = weightDecay;
    this.m = null;
    this.v = null;
    this.t = 0;
  }

  step(params, grads) {
    if (!this.m) {
      this.m = params.map((p) => new Float64Array(p.size));
      this.v = params.map((p) => new Float64Array(p.size));
    }
    this.t++;

    const bc1 = 1 - Math.pow(this.beta1, this.t);
    const bc2 = 1 - Math.pow(this.beta2, this.t);

    for (let i = 0; i < params.length; i++) {
      const p = params[i], g = grads[i];
      if (!g) continue;
      for (let j = 0; j < p.size; j++) {
        let grad = g[j];
        // L2 正则（不含偏置项）
        if (this.weightDecay > 0) grad += this.weightDecay * p.data[j];

        this.m[i][j] = this.beta1 * this.m[i][j] + (1 - this.beta1) * grad;
        this.v[i][j] = this.beta2 * this.v[i][j] + (1 - this.beta2) * grad * grad;

        const mHat = this.m[i][j] / bc1;
        const vHat = this.v[i][j] / bc2;
        p.data[j] -= this.lr * mHat / (Math.sqrt(vHat) + this.eps);
      }
    }
  }
}

/**
 * AdamW：权重衰减与梯度解耦。
 *
 * 与 Adam+L2 的区别：L2 把 wd*p 加进梯度，会随梯度大小被自适应缩放一起改变；
 * 解耦后直接 p -= lr*wd*p，衰减强度与梯度历史无关。
 */
export class AdamW {
  constructor({ lr = 0.001, beta1 = 0.9, beta2 = 0.999, eps = 1e-8, weightDecay = 0.01 } = {}) {
    this.adam = new Adam({ lr, beta1, beta2, eps });
    this.weightDecay = weightDecay;
  }

  step(params, grads) {
    if (!this.adam.m) {
      this.adam.m = params.map((p) => new Float64Array(p.size));
      this.adam.v = params.map((p) => new Float64Array(p.size));
    }
    this.adam.t++;
    const { beta1, beta2, eps, lr } = this.adam;
    const bc1 = 1 - Math.pow(beta1, this.adam.t);
    const bc2 = 1 - Math.pow(beta2, this.adam.t);

    for (let i = 0; i < params.length; i++) {
      const p = params[i], g = grads[i];
      if (!g) continue;
      for (let j = 0; j < p.size; j++) {
        this.adam.m[i][j] = beta1 * this.adam.m[i][j] + (1 - beta1) * g[j];
        this.adam.v[i][j] = beta2 * this.adam.v[i][j] + (1 - beta2) * g[j] * g[j];
        const mHat = this.adam.m[i][j] / bc1;
        const vHat = this.adam.v[i][j] / bc2;
        p.data[j] -= lr * mHat / (Math.sqrt(vHat) + eps);
        // 解耦衰减，直接作用在权重上
        p.data[j] -= lr * this.weightDecay * p.data[j];
      }
    }
  }
}

/** RMSProp：用梯度平方的滑动平均归一化学习率 */
export class RMSProp {
  constructor({ lr = 0.001, beta = 0.9, eps = 1e-8 } = {}) {
    this.lr = lr;
    this.beta = beta;
    this.eps = eps;
    this.s = null;
  }

  step(params, grads) {
    if (!this.s) this.s = params.map((p) => new Float64Array(p.size));
    for (let i = 0; i < params.length; i++) {
      const p = params[i], g = grads[i];
      if (!g) continue;
      for (let j = 0; j < p.size; j++) {
        this.s[i][j] = this.beta * this.s[i][j] + (1 - this.beta) * g[j] * g[j];
        p.data[j] -= this.lr * g[j] / (Math.sqrt(this.s[i][j]) + this.eps);
      }
    }
  }
}

/**
 * 学习率调度器。
 * 传 baseLr 返回当前步应使用的学习率。
 */
export class Scheduler {
  /**
   * @param {object} base 学习率
   * @param {'constant'|'cosine'|'step'|'exponential'} strategy
   */
  constructor(base, strategy = "constant", opts = {}) {
    this.base = base;
    this.strategy = strategy;
    this.stepSize = opts.stepSize ?? 20;
    this.gamma = opts.gamma ?? 0.5;
    this.totalSteps = opts.totalSteps ?? 100;
  }

  at(step) {
    switch (this.strategy) {
      case "cosine": {
        // 余弦退火：从 base 平滑降到 0，训练末期细化收敛
        const t = Math.min(step / Math.max(1, this.totalSteps), 1);
        return this.base * 0.5 * (1 + Math.cos(Math.PI * t));
      }
      case "step":
        return this.base * Math.pow(this.gamma, Math.floor(step / this.stepSize));
      case "exponential":
        return this.base * Math.pow(this.gamma, step);
      default:
        return this.base;
    }
  }
}