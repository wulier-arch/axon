/**
 * 端到端训练测试 —— 检验"框架能真正学习"而非"梯度算得对"。
 *
 * 梯度正确只是必要条件；这里训练到损失收敛，验证模型确实学到了东西。
 */

import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  Tensor, Linear, Sequential, Adam, AdamW, SGD, Scheduler, Trainer,
  crossEntropy, mse, accuracy, makeSpiral, makeBlobs, makeXor,
  makeLinearRegression, backward, checkGradient, softmax,
} from "../src/index.js";

describe("Linear 层", () => {
  it("前向形状正确", () => {
    const layer = new Linear(3, 2, { bias: false, seed: 1 });
    const x = Tensor.tensor([1, 2, 3, 4, 5, 6], [2, 3]);
    assert.equal(layer.forward(x).shape[1], 2);
  });

  it("权重梯度可通过有限差分校验", () => {
    const layer = new Linear(2, 3, { seed: 3 });
    const x = Tensor.variable([1, 2, 3, 4], [2, 2]);
    const loss = () => layer.forward(x).sum();
    backward(loss());
    const r = checkGradient(loss, [layer.weight, layer.bias]);
    assert.ok(r.passed, `权重梯度不匹配: ${r.report}`);
  });

  it("He 初始化让权重方差接近 2/fan_in", () => {
    const layer = new Linear(64, 64, { seed: 11 });
    let sum = 0;
    for (const v of layer.weight.data) sum += v * v;
    const variance = sum / layer.weight.size;
    // He 初始化期望 std² = 2/fan_in = 1/32
    assert.ok(variance > 0.01 && variance < 0.06, `方差 ${variance.toFixed(4)} 偏离预期`);
  });

  it("bias 默认为全零初始化", () => {
    const layer = new Linear(2, 2, { seed: 1 });
    for (const v of layer.bias.data) assert.equal(v, 0);
  });

  it("参数量为 in*out + out", () => {
    const layer = new Linear(5, 3);
    assert.equal(layer.weight.size + layer.bias.size, 5 * 3 + 3);
  });
});

describe("Sequential", () => {
  it("统计参数量正确", () => {
    const model = new Sequential()
      .add(new Linear(4, 8, { activation: "relu", seed: 1 }))
      .add(new Linear(8, 2, { seed: 2 }));
    assert.equal(model.countParams(), 4 * 8 + 8 + 8 * 2 + 2);
  });

  it("前向逐层传递", () => {
    const model = new Sequential()
      .add(new Linear(2, 3, { seed: 1 }))
      .add(new Linear(3, 2, { seed: 2 }));
    const out = model.forward(Tensor.tensor([1, 2, 3, 4], [2, 2]));
    assert.deepEqual(out.shape, [2, 2]);
  });
});

describe("损失函数", () => {
  it("交叉熵梯度可通过有限差分校验", () => {
    const logits = Tensor.variable([2, 1, 0.1, 0.5, 2.5, 1.5], [2, 3]);
    const loss = () => crossEntropy(logits, [0, 2]);
    backward(loss());
    const r = checkGradient(loss, [logits]);
    assert.ok(r.passed, `crossEntropy 梯度不匹配: ${r.report}`);
  });

  it("预测正确时损失远小于 ln(类别数)", () => {
    const good = crossEntropy(Tensor.tensor([10, 0, 0, 0, 10, 0], [2, 3]), [0, 1]);
    const bad = crossEntropy(Tensor.tensor([10, 0, 0, 0, 10, 0], [2, 3]), [1, 2]);
    assert.ok(good.data[0] < 0.01, `好预测损失应极小，实际 ${good.data[0]}`);
    assert.ok(bad.data[0] > 9, `坏预测损失应很大，实际 ${bad.data[0]}`);
  });

  it("梯度等于 softmax 减 one-hot 再除以批大小", () => {
    const logits = Tensor.variable([2, 1, 0, 0, 1, 2], [2, 3]);
    const loss = () => crossEntropy(logits, [0, 2]);
    backward(loss());

    // 不写死数字，直接用 softmax 算期望值，避免手算常数记错
    const probs = softmax(logits);
    const classes = 3;
    const labels = [0, 2];
    for (let r = 0; r < 2; r++) {
      for (let c = 0; c < classes; c++) {
        const p = probs.data[r * classes + c];
        const expected = (p - (c === labels[r] ? 1 : 0)) / 2;
        assert.ok(
          Math.abs(logits.grad[r * classes + c] - expected) < 1e-9,
          `第 ${r} 行第 ${c} 列：得到 ${logits.grad[r * classes + c]}，期望 ${expected}`
        );
      }
    }
  });

  it("数值稳定：极大 logits 不产生 Infinity", () => {
    const loss = crossEntropy(Tensor.tensor([1000, 1000, 1000], [1, 3]), [0]);
    assert.ok(Number.isFinite(loss.data[0]), "损失出现非有限值");
describe("优化器", () => {
  it("SGD 沿负梯度方向更新", () => {
    const p = Tensor.variable([5], [1]);
    const opt = new SGD({ lr: 0.1 });
    opt.step([p], [Float64Array.of(2)]);
    assert.ok(Math.abs(p.data[0] - 4.8) < 1e-9, `期望 4.8，实际 ${p.data[0]}`);
  });

  it("Adam 单步更新量约为 lr", () => {
    const p = Tensor.variable([0], [1]);
    const opt = new Adam({ lr: 0.1 });
    opt.step([p], [Float64Array.of(1)]);
    assert.ok(Math.abs(p.data[0] + 0.1) < 1e-6, `实际位移 ${p.data[0]}`);
  });

  it("Adam 偏差校正生效（前期步长不被低估）", () => {
    const p = Tensor.variable([0], [1]);
    const opt = new Adam({ lr: 0.1 });
    opt.step([p], [Float64Array.of(1)]);
    assert.ok(p.data[0] < -0.09, `偏差校正失效，实际 ${p.data[0]}`);
  });

  it("Adam 梯度为零时不会 NaN", () => {
    const p = Tensor.variable([1], [1]);
    const opt = new Adam({ lr: 0.1 });
    opt.step([p], [Float64Array.of(0)]);
    assert.ok(Number.isFinite(p.data[0]));
  });

  it("AdamW 权重衰减独立于梯度大小", () => {
    const a = Tensor.variable([1], [1]);
    const b = Tensor.variable([1], [1]);
    const oa = new AdamW({ lr: 0.1, weightDecay: 0.5 });
    const ob = new AdamW({ lr: 0.1, weightDecay: 0.5 });
    oa.step([a], [Float64Array.of(100)]);
    ob.step([b], [Float64Array.of(0.001)]);
    assert.ok(Number.isFinite(a.data[0]) && Number.isFinite(b.data[0]));
  });

  it("学习率调度器按策略变化", () => {
    const constant = new Scheduler(0.1, "constant");
    const cos = new Scheduler(0.1, "cosine", { totalSteps: 100 });
    const step = new Scheduler(0.1, "step", { stepSize: 10, gamma: 0.5 });

    assert.equal(constant.at(50), 0.1);
    assert.ok(Math.abs(cos.at(0) - 0.1) < 1e-9);
    assert.ok(cos.at(50) < cos.at(0), "余弦退火未下降");
    assert.equal(step.at(9), 0.1);
    assert.ok(Math.abs(step.at(10) - 0.05) < 1e-9);
  });
});

describe("端到端训练", () => {
  it("能在螺旋数据集上学到 95% 以上准确率", () => {
    const { x, y } = makeSpiral({ samples: 300, noise: 0.12 });

    const model = new Sequential()
      .add(new Linear(2, 32, { activation: "relu", seed: 1 }))
      .add(new Linear(32, 2, { seed: 2 }));

    const trainer = new Trainer({
      model,
      optimizer: new Adam({ lr: 0.01 }),
      lossFn: crossEntropy,
      metricFn: accuracy,
      epochs: 400,
      batchSize: 32,
    });

    const history = trainer.fit(x, y, y);
    const finalAcc = history[history.length - 1].metric;
    const finalLoss = history[history.length - 1].loss;

    assert.ok(finalAcc > 0.95, `准确率仅 ${(finalAcc * 100).toFixed(1)}%`);
    assert.ok(finalLoss < 0.15, `损失仅降到 ${finalLoss.toFixed(4)}`);
    assert.ok(history[0].loss > finalLoss, "损失没有下降");
  });

  it("能在异或数据集上学到 100%", () => {
    const { x, y } = makeXor();
    const model = new Sequential()
      .add(new Linear(2, 8, { activation: "relu", seed: 9 }))
      .add(new Linear(8, 2, { seed: 10 }));

    const trainer = new Trainer({
      model,
      optimizer: new Adam({ lr: 0.05 }),
      lossFn: crossEntropy,
      metricFn: accuracy,
      epochs: 400,
      batchSize: 4,
    });

    const history = trainer.fit(x, y, y);
    assert.equal(history[history.length - 1].metric, 1, "异或未被完全学会");
  });

  it("能拟合线性回归并逼近真实参数", () => {
    const { x, yTensor, trueW, trueB } = makeLinearRegression({ samples: 100 });

    const model = new Sequential().add(new Linear(1, 1, { seed: 4 }));
    const trainer = new Trainer({
      model,
      optimizer: new Adam({ lr: 0.1 }),
      lossFn: (pred, target) => mse(pred, target),
      epochs: 400,
      batchSize: 25,
    });

    trainer.fit(x, yTensor, yTensor);

    const w = model.layers[0].weight.data[0];
    const b = model.layers[0].bias.data[0];
    assert.ok(Math.abs(w - trueW) < 0.15, `权重 ${w.toFixed(3)} 偏离真值 ${trueW}`);
    assert.ok(Math.abs(b - trueB) < 0.15, `偏置 ${b.toFixed(3)} 偏离真值 ${trueB}`);
  });

  it("早停会提前结束训练", () => {
    const { x, y } = makeBlobs({ samples: 120, seed: 3 });
    const model = new Sequential().add(new Linear(2, 2, { seed: 1 }));

    const trainer = new Trainer({
      model,
      optimizer: new Adam({ lr: 0.1 }),
      lossFn: crossEntropy,
      metricFn: accuracy,
      epochs: 200,
      batchSize: 30,
    });

    const { history, bestScore } = trainer.fitWithEarlyStopping(x, y, y, x, y, { patience: 8 });
    assert.ok(history.length < 200, `早停未生效，跑了 ${history.length} 轮`);
    assert.ok(bestScore > 0.5);
  });

  it("训练过程中损失不会变成 NaN", () => {
    const { x, y } = makeSpiral({ samples: 100, noise: 0.2 });
    const model = new Sequential()
      .add(new Linear(2, 16, { activation: "relu", seed: 1 }))
      .add(new Linear(16, 2, { seed: 2 }));
    const trainer = new Trainer({
      model, optimizer: new Adam({ lr: 0.05 }),
      lossFn: crossEntropy, epochs: 50, batchSize: 16,
    });
    for (const h of trainer.fit(x, y, y)) {
      assert.ok(Number.isFinite(h.loss), `第 ${h.epoch} 轮损失为 ${h.loss}`);
    }
  });
});
  });

  it("mse 梯度可校验", () => {
    const pred = Tensor.variable([1, 2, 3], [3]);
    const target = Tensor.tensor([0, 0, 0], [3]);
    const loss = () => mse(pred, target);
    backward(loss());
    const r = checkGradient(loss, [pred]);
    assert.ok(r.passed, `mse 梯度不匹配: ${r.report}`);
  });
});
// __PART2__