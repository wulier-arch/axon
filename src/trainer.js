/**
 * 训练循环。
 *
 * 每一步：清零梯度 → 前向 → 算损失 → 反向 → 优化器更新。
 * 注意梯度必须在每步清零，否则会累积成越来越大的步长直至发散。
 */

import { backward } from "./autodiff.js";

export class Trainer {
  /**
   * @param {object} opts
   * @param {import('./nn/sequential.js').Sequential} opts.model
   * @param {object} opts.optimizer
   * @param {(logits, labels) => import('./tensor/tensor.js').Tensor} opts.lossFn
   * @param {(logits, labels) => number} [opts.metricFn]
   * @param {number} [opts.epochs]
   * @param {number} [opts.batchSize]
   * @param {object} [opts.scheduler] 学习率调度器，提供 at(step)
   * @param {(info) => void} [opts.onEpochEnd]
   */
  constructor(opts) {
    this.model = opts.model;
    this.optimizer = opts.optimizer;
    this.lossFn = opts.lossFn;
    this.metricFn = opts.metricFn || null;
    this.epochs = opts.epochs ?? 50;
    this.batchSize = opts.batchSize ?? 32;
    this.shuffle = opts.shuffle ?? true;
    this.scheduler = opts.scheduler || null;
    this.onEpochEnd = opts.onEpochEnd || null;
    this.onBatchEnd = opts.onBatchEnd || null;
    this.globalStep = 0;
    this.history = [];
  }

  /** 把数据按 batch 切分；batchSize 为 null 时整批一次 */
  *batches(x, y, batchSize, shuffle, rng) {
    const n = x.shape[0];
    let order = Array.from({ length: n }, (_, i) => i);
    if (shuffle) {
      for (let i = n - 1; i > 0; i--) {
        const j = Math.floor(rng() * (i + 1));
        [order[i], order[j]] = [order[j], order[i]];
      }
    }
    const bs = batchSize ?? n;
    for (let start = 0; start < n; start += bs) {
      const idx = order.slice(start, start + bs);
      if (idx.length === 0) continue;
      yield { xIdx: idx, size: idx.length };
    }
  }

  /** 从 [rows, cols] 张量里按行索引取子集 */
  static selectRows(t, indices) {
    const [, cols] = t.shape;
    const out = new (t.constructor)(new Float64Array(indices.length * cols), [indices.length, cols]);
    for (let i = 0; i < indices.length; i++) {
      for (let j = 0; j < cols; j++) {
        out.data[i * cols + j] = t.data[indices[i] * cols + j];
      }
    }
    return out;
  }

  /**
   * 跑一轮，返回平均损失与平均指标。
   * 整个 epoch 的所有 batch 损失取平均，而不是只看最后一个 batch。
   */
  runEpoch(x, y, labels, rng) {
    let totalLoss = 0;
    let totalMetric = 0;
    let seen = 0;

    for (const { xIdx, size } of this.batches(x, y, this.batchSize, this.shuffle, rng)) {
      const xb = Trainer.selectRows(x, xIdx);

      // 标签既可能是类索引数组，也可能是回归目标张量，两种都要按 batch 切片
      let yb;
      if (Array.isArray(labels)) {
        yb = xIdx.map((i) => labels[i]);
      } else if (labels && typeof labels === "object" && labels.shape) {
        yb = Trainer.selectRows(labels, xIdx);
      } else {
        yb = labels;
      }

      // 每步都要清零，否则梯度会跨步累积
      for (const p of this.model.parameters()) p.zeroGrad();

      const logits = this.model.forward(xb);
      const loss = this.lossFn(logits, yb);
      backward(loss);

      if (this.scheduler) {
        this.optimizer.lr = this.scheduler.at(this.globalStep);
      }
      const params = this.model.parameters();
      this.optimizer.step(params, params.map((p) => p.grad));

      totalLoss += loss.data[0] * size;
      if (this.metricFn) totalMetric += this.metricFn(logits, yb) * size;
      seen += size;
      this.globalStep++;

      if (this.onBatchEnd) {
        this.onBatchEnd({ loss: loss.data[0], metric: this.metricFn ? this.metricFn(logits, yb) : null, step: this.globalStep });
      }
    }

    return { loss: totalLoss / seen, metric: seen ? totalMetric / seen : 0 };
  }

  /** 完整训练，返回每个 epoch 的历史记录 */
  fit(x, y, labels, { evalX = null, evalY = null } = {}) {
    const rng = (() => {
      let s = 12345;
      return () => {
        s ^= s << 13; s >>>= 0;
        s ^= s >> 17;
        s ^= s << 5; s >>>= 0;
        return s / 4294967296;
      };
    })();

    for (let epoch = 0; epoch < this.epochs; epoch++) {
      const train = this.runEpoch(x, y, labels, rng);
      const record = { epoch, loss: train.loss, metric: train.metric };

      if (evalX && evalY) {
        const logits = this.model.forward(evalX);
        record.evalLoss = this.lossFn(logits, evalY).data[0];
        record.evalMetric = this.metricFn ? this.metricFn(logits, evalY) : null;
      }

      this.history.push(record);
      if (this.onEpochEnd) this.onEpochEnd(record);
    }
    return this.history;
  }

  /** 早停：验证指标连续 patience 轮不再提升就停 */
  fitWithEarlyStopping(x, y, labels, evalX, evalY, { patience = 10 } = {}) {
    let best = -Infinity;
    let bestEpoch = 0;
    let since = 0;

    this.onEpochEnd = (record) => {
      const score = record.evalMetric ?? record.metric;
      if (score > best + 1e-6) {
        best = score;
        bestEpoch = record.epoch;
        since = 0;
      } else {
        since++;
      }
      if (since >= patience) {
        this.epochs = record.epoch + 1;   // 下一轮起停止
      }
    };

    const history = this.fit(x, y, labels, { evalX, evalY });
    return { history, bestEpoch, bestScore: best };
  }
}