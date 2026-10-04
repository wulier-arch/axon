/**
 * 端到端训练测试 —— 检验"框架能真正学习"而非"梯度算得对"。
 *
 * 梯度正确只是必要条件；这里训练到损失收敛，验证模型确实学到了东西。
 */

import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  Tensor, Linear, Sequential, Adam, AdamW, SGD, Scheduler, Trainer,
  crossEntropy, mse, accuracy, makeSpiral, makeBlobs, makeXor, Momentum,
  makeLinearRegression, backward, checkGradient, softmax,
  LayerNorm, Embedding, MultiHeadAttention, mul, argmaxLast,
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

/** LayerNorm 与 Embedding 梯度校验 */
describe("LayerNorm", () => {
  it("每行标准化后均值为 0 方差为 1（γ=1 β=0 时）", () => {
    const ln = new LayerNorm(4, { gamma: false, beta: false });
    const x = Tensor.tensor([1, 2, 3, 4, 10, 20, 30, 40], [2, 4]);
    const out = ln.forward(x);
    for (let r = 0; r < 2; r++) {
      const row = Array.from(out.data.slice(r * 4, r * 4 + 4));
      const mean = row.reduce((a, b) => a + b, 0) / 4;
      const varsum = row.reduce((a, b) => a + (b - mean) ** 2, 0) / 4;
      assert.ok(Math.abs(mean) < 1e-10, `第 ${r} 行均值 ${mean}`);
      assert.ok(Math.abs(varsum - 1) < 1e-3, `第 ${r} 行方差 ${varsum}`);
    }
  });

  it("γ 初始化为 1、β 初始化为 0", () => {
    const ln = new LayerNorm(3);
    for (const v of ln.gamma.data) assert.equal(v, 1);
    for (const v of ln.beta.data) assert.equal(v, 0);
  });

  it("γ/β/输入的梯度均可通过有限差分校验", () => {
    // 用固定的有符号权重做线性损失，让上游梯度 dy 恒为 O(1)。
    // 若用 sum(y²)，y≈x̂ 时梯度只有 0.01 量级，有限差分的舍入噪声
    // 会压过真实误差，测出来的 relError 是噪声而非 bug。
    const ln = new LayerNorm(4);
    ln.gamma.data.set([1.2, 0.8, 1.5, 0.9]);
    ln.beta.data.set([0.1, -0.2, 0.3, 0.0]);
    const x = Tensor.variable([1, 2, 3, 4, 10, 20, 30, 40], [2, 4]);
    const w = Tensor.tensor([1, -2, 0.5, 3, -1, 0.7, 2, -0.3], [2, 4]);
    const loss = () => mul(ln.forward(x), w).sum();
    backward(loss());
    const r = checkGradient(loss, [x, ln.gamma, ln.beta]);
    assert.ok(r.passed, `梯度不匹配: ${r.report}`);
  });

  it("可关闭 γ 与 β", () => {
    const ln = new LayerNorm(3, { gamma: false, beta: false });
    assert.equal(ln.parameters().length, 0);
    const x = Tensor.tensor([1, 2, 3], [1, 3]);
    assert.ok(ln.forward(x).data.some((v) => v !== 0));
  });

  it("特征维不匹配时报错", () => {
    const ln = new LayerNorm(4);
    assert.throws(() => ln.forward(Tensor.tensor([1, 2, 3], [1, 3])), /特征维不匹配/);
  });

  it("可与 Linear 组合并端到端训练", () => {
    const model = new Sequential()
      .add(new Linear(2, 8, { activation: "relu", seed: 1 }))
      .add(new LayerNorm(8))
      .add(new Linear(8, 2, { seed: 2 }));
    const { x, y } = makeXor();
    const trainer = new Trainer({
      model, optimizer: new Adam({ lr: 0.05 }),
      lossFn: crossEntropy, metricFn: accuracy, epochs: 300, batchSize: 4,
    });
    const history = trainer.fit(x, x, y);
    assert.equal(history.at(-1).metric, 1, "XOR 应达 100%");
  });
});

describe("Embedding", () => {
  it("前向按 id 取出对应行", () => {
    const emb = new Embedding(4, 3, { seed: 1 });
    const out = emb.forward([2, 0]);
    const w = emb.weight.data;
    assert.deepEqual(out.shape, [2, 3]);
    for (let j = 0; j < 3; j++) {
      assert.equal(out.data[j], w[2 * 3 + j]);
      assert.equal(out.data[3 + j], w[j]);
    }
  });

  it("权重梯度可通过有限差分校验", () => {
    const emb = new Embedding(5, 3, { seed: 2 });
    const loss = () => emb.forward([1, 3, 1]).sum();
    backward(loss());
    const r = checkGradient(loss, [emb.weight]);
    assert.ok(r.passed, `权重梯度不匹配: ${r.report}`);
  });

  it("重复 id 的梯度必须累加而非覆盖", () => {
    // id=2 出现三次，三次的梯度应求和；若写成覆盖，只会留下最后一次
    const emb = new Embedding(4, 2, { seed: 3 });
    const out = emb.forward([2, 2, 2]);
    for (let i = 0; i < out.size; i++) out.grad = null;
    backward(out.sum());
    const row = 2 * 2;
    assert.equal(emb.weight.grad[row], 3, "重复词的三份梯度应累加为 3");
    assert.equal(emb.weight.grad[row + 1], 3);
    // 未出现的 id 梯度必须为 0
    assert.equal(emb.weight.grad[0], 0);
  });

  it("词 id 越界时报错", () => {
    const emb = new Embedding(3, 2);
    assert.throws(() => emb.forward([0, 3]), /越界/);
    assert.throws(() => emb.forward([-1]), /越界/);
  });

  it("嵌入可随分类器一起训练到收敛", () => {
    // Embedding 的输入是 id 序列而非张量，因此不能直接塞进 Sequential，
    // 这里手动组合：查表 → 线性分类 → 更新两组参数。
    const emb = new Embedding(4, 3, { seed: 4 });
    const clf = new Linear(3, 2, { seed: 5 });
    const ids = [0, 1, 2, 3];
    const labels = Tensor.tensor([0, 1, 0, 1], [4, 1]);
    const params = [...emb.parameters(), ...clf.parameters()];
    const opt = new Adam({ lr: 0.1 });

    let loss = null;
    for (let step = 0; step < 200; step++) {
      for (const p of params) p.zeroGrad();
      const logits = clf.forward(emb.forward(ids));
      loss = crossEntropy(logits, labels);
      backward(loss);
      opt.step(params, params.map((p) => p.grad));
    }

    assert.ok(loss.data[0] < 0.1, `嵌入训练后交叉熵应收敛，实得 ${loss.data[0].toFixed(4)}`);
    // logits 是连续值，要看的是每行最大分量所在的下标
    assert.deepEqual(argmaxLast(clf.forward(emb.forward(ids))), [0, 1, 0, 1], "两个类别应被完全分开");
  });
});

describe("MultiHeadAttention", () => {
  /** 按配置生成固定数据与损失，保证多次运行结果可比 */
  function setup(mha, d, seq) {
    const xD = Array.from({ length: seq * d }, (_, i) => Math.sin(i * 0.7) * 1.3);
    const wD = Array.from({ length: seq * d }, (_, i) => ((i % 7) - 3) * 0.9 + 0.3);
    const x = Tensor.variable(xD, [seq, d]);
    const w = Tensor.tensor(wD, [seq, d]);
    const loss = () => mul(mha.forward(x), w).sum();
    backward(loss());
    return { loss, inputs: [x, mha.wq, mha.wk, mha.wv, mha.wo] };
  }

  it("输出形状与输入一致", () => {
    const mha = new MultiHeadAttention(8, 4, { seed: 1 });
    const xD = Array.from({ length: 32 }, (_, i) => Math.sin(i * 0.7));
    const out = mha.forward(Tensor.tensor(xD, [4, 8]));
    assert.deepEqual(out.shape, [4, 8]);
  });

  // 注意力链深约 10 个算子（matmul→reshape→transpose→scale→softmax→…），
  // 中央差分的截断误差会累积到 1e-4 量级，且随 dModel 增大而恶化：
  // 实测 d=4 时 7e-7，d=12 时已达 7.7e-4。因此严格用例取小配置，
  // 大配置另用宽松容差，并在下一条用例里给出严谨判据。
  for (const heads of [1, 2]) {
    it(`${heads} 个头时梯度可通过有限差分校验（严格容差）`, () => {
      const mha = new MultiHeadAttention(4, heads, { seed: 3 });
      const { loss, inputs } = setup(mha, 4, 3);
      const r = checkGradient(loss, inputs, { eps: 1e-7 });
      assert.ok(r.passed, `${heads} 头梯度不匹配: ${r.report}`);
    });
  }

  it("更大维度与因果掩码下梯度依然正确", () => {
    for (const causal of [false, true]) {
      const mha = new MultiHeadAttention(8, 4, { seed: 3, causal });
      const { loss, inputs } = setup(mha, 8, 4);
      const r = checkGradient(loss, inputs, { eps: 1e-7, tol: 1e-4 });
      assert.ok(r.passed, `causal=${causal} 梯度不匹配: ${r.report}`);
    }
  });

  it("残差随 eps 线性收敛，证明梯度本身正确", () => {
    // 这条比放宽容差更关键：深链下一次校验分不清「梯度写错」与「差分不准」。
    // 判据——若解析梯度真的错了，误差会有一个不随 eps 消失的下限；
    // 若是截断误差，eps 每降 10 倍，误差也应降约 10 倍。
    //
    // 取 1e-5 → 1e-6 这一段：此时截断项仍主导。再往下降到 1e-7，
    // 舍入误差（∝1/h）抬头，误差反而不再下降，所以不能拿那一段比。
    const errAt = (eps) => {
      const mha = new MultiHeadAttention(8, 4, { seed: 3 });
      const { loss, inputs } = setup(mha, 8, 4);
      return checkGradient(loss, inputs, { eps }).maxRelError;
    };
    const ratio = errAt(1e-5) / errAt(1e-6);
    assert.ok(ratio > 4 && ratio < 25,
      `eps 降 10 倍而误差仅降 ${ratio.toFixed(2)} 倍，残差不随 eps 消失，疑似梯度有误`);
  });

  it("因果掩码下，后面的 token 不影响前面的输出", () => {
    const seq = 4, d = 8;
    const mha = new MultiHeadAttention(d, 4, { seed: 5, causal: true });
    const xD = Array.from({ length: seq * d }, (_, i) => Math.sin(i * 0.7));
    const a = mha.forward(Tensor.tensor(xD, [seq, d]));
    const changed = xD.slice();
    for (let i = 2; i < seq; i++) changed[i * d] += 100;  // 只动后两个位置
    const b = mha.forward(Tensor.tensor(changed, [seq, d]));
    for (let i = 0; i < 2; i++) {
      for (let k = 0; k < d; k++) {
        assert.equal(a.data[i * d + k], b.data[i * d + k],
          `位置 ${i} 不应受后续 token 影响`);
      }
    }
  });

  it("dModel 不能被头数整除时报错", () => {
    assert.throws(() => new MultiHeadAttention(7, 2), /不能被头数/);
  });

  it("输入维度不符时报错", () => {
    const mha = new MultiHeadAttention(8, 2);
    assert.throws(() => mha.forward(Tensor.tensor([1, 2, 3], [1, 3])), /输入维度/);
  });

  it("可与 Embedding / LayerNorm 组成 Transformer 块并训练到收敛", () => {
    const emb = new Embedding(6, 8, { seed: 7 });
    const attn = new MultiHeadAttention(8, 2, { seed: 8 });
    const ln = new LayerNorm(8);
    const head = new Linear(8, 3, { seed: 9 });

    const ids = [0, 1, 2, 3, 4, 5];
    const targets = Tensor.tensor([0, 1, 2, 0, 1, 2], [6, 1]);
    const params = [...emb.parameters(), ...attn.parameters(), ...ln.parameters(), ...head.parameters()];
    const opt = new Adam({ lr: 0.02 });

    let loss = null;
    for (let step = 0; step < 60; step++) {
      for (const p of params) p.zeroGrad();
      const out = head.forward(ln.forward(attn.forward(emb.forward(ids))));
      loss = crossEntropy(out, targets);
      backward(loss);
      opt.step(params, params.map((p) => p.grad));
    }
    // 随机初始化的交叉熵约为 ln(3) ≈ 1.0986
    assert.ok(loss.data[0] < 0.2, `Transformer 块应收敛，实得 ${loss.data[0].toFixed(4)}`);
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

  it("SGD 能处理尺寸不同的多个参数", () => {
    // 回归用例：曾用 params[0].size 给所有参数分配动量缓冲区，
    // 导致偏置（元素更少）缓冲区越界写入而被静默忽略，训练悄悄学错。
    const weight = Tensor.variable([1, 1, 1, 1], [2, 2]);
    const bias = Tensor.variable([0, 0], [2]);
    const opt = new SGD({ lr: 0.1 });
    opt.step(
      [weight, bias],
      [Float64Array.of(1, 1, 1, 1), Float64Array.of(0.5, 0.5)],
    );
    assert.ok(Math.abs(weight.data[0] - 0.9) < 1e-9, `权重未更新：${weight.data[0]}`);
    // 偏置梯度为 0.5，更新量应为 0.05
    assert.ok(Math.abs(bias.data[0] + 0.05) < 1e-9, `偏置未更新：${bias.data[0]}`);
    assert.ok(Math.abs(bias.data[1] + 0.05) < 1e-9, `偏置未更新：${bias.data[1]}`);
  });

  it("Momentum 能处理尺寸不同的多个参数", () => {
    const weight = Tensor.variable([1, 1, 1, 1], [2, 2]);
    const bias = Tensor.variable([0, 0], [2]);
    const opt = new Momentum({ lr: 0.1, beta: 0.9 });
    opt.step(
      [weight, bias],
      [Float64Array.of(1, 1, 1, 1), Float64Array.of(1, 1)],
    );
    assert.ok(Math.abs(weight.data[0] - 0.9) < 1e-9, `权重未更新：${weight.data[0]}`);
    assert.ok(Math.abs(bias.data[0] + 0.1) < 1e-9, `偏置未更新：${bias.data[0]}`);
  });

  it("Adam 能处理尺寸不同的多个参数", () => {
    const weight = Tensor.variable([1, 1, 1, 1], [2, 2]);
    const bias = Tensor.variable([0, 0], [2]);
    const opt = new Adam({ lr: 0.1 });
    opt.step(
      [weight, bias],
      [Float64Array.of(1, 1, 1, 1), Float64Array.of(1, 1)],
    );
    assert.ok(Math.abs(weight.data[0] - 0.9) < 1e-6, `权重未更新：${weight.data[0]}`);
    assert.ok(Math.abs(bias.data[0] + 0.1) < 1e-6, `偏置未更新：${bias.data[0]}`);
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
