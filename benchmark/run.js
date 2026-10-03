/**
 * axon 基准测试 —— 量化性能与训练能力。
 *
 * 这不是"跑分炫耀"，而是给出可复现的数据：
 *   - 吞吐：每秒能处理多少次运算
 *   - 训练能力：在标准数据集上能达到什么准确率
 *
 * 运行：npm run benchmark
 */

import {
  Tensor, Linear, Sequential, Adam, Trainer,
  crossEntropy, accuracy, mse, makeSpiral, makeBlobs, makeXor,
  matmul, conv2d,
} from "../src/index.js";

const RULE = "─".repeat(64);

/**
 * 按显示宽度对齐：中文/全角字符在终端占两列，
 * 而 String.padEnd 按字符个数补齐，中文表格必然错位。
 */
function width(s) {
  let w = 0;
  for (const ch of String(s)) {
    const code = ch.codePointAt(0);
    const wide =
      (code >= 0x1100 && code <= 0x115f) ||
      (code >= 0x2e80 && code <= 0xa4cf) ||
      (code >= 0xac00 && code <= 0xd7a3) ||
      (code >= 0xf900 && code <= 0xfaff) ||
      (code >= 0xfe30 && code <= 0xfe6f) ||
      (code >= 0xff00 && code <= 0xff60) ||
      (code >= 0xffe0 && code <= 0xffe6);
    w += wide ? 2 : 1;
  }
  return w;
}

const pad = (s, n) => String(s) + " ".repeat(Math.max(0, n - width(s)));
const padL = (s, n) => " ".repeat(Math.max(0, n - width(s))) + String(s);

/** 预热后取多次运行的中位数，避免首次 JIT 干扰 */
function measure(warmup, runs, fn) {
  for (let i = 0; i < warmup; i++) fn();
  const times = [];
  for (let i = 0; i < runs; i++) {
    const t0 = performance.now();
    fn();
    times.push(performance.now() - t0);
  }
  times.sort((a, b) => a - b);
  return times[Math.floor(times.length / 2)];
}

function row(label, value, unit = "") {
  // 每列之间留两个空格；unit 为空时也不产生悬空空白
  let line = `  ${pad(label, 30)}  ${padL(value, 12)}`;
  if (unit) line += `  ${unit}`;
  console.log(line);
}

console.log(`\n${RULE}`);
console.log(" axon 基准测试");
console.log(` 环境：Node ${process.version} · ${process.arch}`);
console.log(`${RULE}\n`);

console.log("【矩阵乘法】");
console.log(`  ${pad("尺寸", 30)}  ${padL("耗时", 12)}  吞吐`);
for (const n of [128, 256, 512, 1024]) {
  const a = Tensor.tensor(new Float64Array(n * n).fill(0.5), [n, n]);
  const b = Tensor.tensor(new Float64Array(n * n).fill(0.5), [n, n]);
  const ms = measure(1, 3, () => matmul(a, b));
  const gflops = (2 * n * n * n) / (ms / 1000) / 1e9;
  row(`${n}×${n}`, ms.toFixed(1) + "ms", gflops.toFixed(2) + " GFLOP/s");
}

console.log("\n【卷积】");
console.log(`  ${pad("配置", 30)} ${padL("耗时", 12)}`);
for (const [c, k, size] of [[16, 32, 32], [32, 64, 32], [64, 64, 28]]) {
  const x = Tensor.tensor(new Float64Array(c * size * size).fill(0.1), [c, size, size]);
  const w = Tensor.tensor(new Float64Array(k * c * 9).fill(0.05), [k, c, 3, 3]);
  const ms = measure(1, 3, () => conv2d(x, w, null, 1, 1));
  row(`Conv ${c}→${k} @ ${size}×${size}`, ms.toFixed(2) + "ms");
}

console.log("\n【训练吞吐 · 单步前向+反向】");
console.log(`  ${pad("模型", 30)} ${padL("单步耗时", 12)}`);
{
  const x = Tensor.tensor(new Float64Array(32 * 784).fill(0.1), [32, 784]);
  const y = new Array(32).fill(3);
  const model = new Sequential()
    .add(new Linear(784, 128, { activation: "relu", seed: 1 }))
    .add(new Linear(128, 10, { seed: 2 }));
  const trainer = new Trainer({
    model, optimizer: new Adam({ lr: 0.001 }),
    lossFn: crossEntropy, epochs: 1, batchSize: 32, shuffle: false,
  });
  const ms = measure(1, 5, () => trainer.fit(x, y, y));
  row("MLP 784→128→10", ms.toFixed(2) + "ms");
}

console.log("\n【训练能力】");
console.log(`  ${pad("数据集", 24)}${padL("准确率", 11)}${padL("损失", 11)}${padL("耗时", 12)}`);

function evaluate(name, { buildModel, optimizer, lossFn, metricFn, epochs, batchSize }) {
  const model = buildModel();
  const trainer = new Trainer({ model, optimizer, lossFn, metricFn, epochs, batchSize });
  const t0 = performance.now();
  const history = trainer.fit(...datasetOf(name), { });
  const last = history[history.length - 1];
  const elapsed = performance.now() - t0;
  console.log(
    `  ${pad(name, 24)}${padL((last.metric * 100).toFixed(1) + "%", 11)}` +
    `${padL(last.loss.toFixed(4), 11)}${padL(elapsed.toFixed(0) + "ms", 12)}`
  );
}

// 各数据集预先生成，避免把准备时间算进训练耗时
const xor = makeXor();
const spiral = makeSpiral({ samples: 300, noise: 0.12 });
const blobs = makeBlobs({ samples: 400, features: 2, seed: 21 });

function datasetOf(name) {
  if (name.startsWith("XOR")) return [xor.x, xor.y, xor.y];
  if (name.startsWith("螺旋")) return [spiral.x, spiral.y, spiral.y];
  return [blobs.x, blobs.y, blobs.y];
}

evaluate("XOR (4 样本)", {
  buildModel: () => new Sequential()
    .add(new Linear(2, 8, { activation: "relu", seed: 9 }))
    .add(new Linear(8, 2, { seed: 10 })),
  optimizer: new Adam({ lr: 0.05 }),
  lossFn: crossEntropy, metricFn: accuracy, epochs: 400, batchSize: 4,
});

evaluate("螺旋 (300 样本)", {
  buildModel: () => new Sequential()
    .add(new Linear(2, 32, { activation: "relu", seed: 1 }))
    .add(new Linear(32, 2, { seed: 2 })),
  optimizer: new Adam({ lr: 0.01 }),
  lossFn: crossEntropy, metricFn: accuracy, epochs: 400, batchSize: 32,
});

evaluate("高斯簇 (400 样本)", {
  buildModel: () => new Sequential().add(new Linear(2, 2, { seed: 5 })),
  optimizer: new Adam({ lr: 0.05 }),
  lossFn: crossEntropy, metricFn: accuracy, epochs: 100, batchSize: 32,
});

console.log(`\n${RULE}`);
console.log("说明：纯 CPU 实现，未使用 SIMD / 多线程 / WASM。");
console.log("      性能上限受此限制，优势在于零依赖、可读性与可验证。");
console.log(`${RULE}\n`);