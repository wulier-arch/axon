/**
 * 螺旋分类可视化 demo。
 *
 * 直接从 ../src 引入框架源码，没有任何打包步骤——
 * 这本身就是 axon 设计目标的验证：纯 ESM、零依赖、浏览器可跑。
 */

import {
  Linear, Sequential, Adam, crossEntropy, accuracy,
  makeSpiral, softmax, backward, Tensor,
} from "../src/index.js";

const canvas = document.querySelector("#view");
const ctx = canvas.getContext("2d");
const el = {
  epoch: document.querySelector("#epoch"),
  loss: document.querySelector("#loss"),
  acc: document.querySelector("#acc"),
  speed: document.querySelector("#speed"),
  params: document.querySelector("#params"),
  bar: document.querySelector("#bar"),
  start: document.querySelector("#start"),
  pause: document.querySelector("#pause"),
  reset: document.querySelector("#reset"),
};

const SAMPLES = 300;
const TOTAL_EPOCHS = 220;
const GRID = 48;              // 决策边界的采样密度

const COLORS = [
  [255, 107, 139],           // 类别 A：粉
  [74, 214, 200],            // 类别 B：青
];

/* ---------- 数据与模型 ---------- */

const { x: dataX, y: labels } = makeSpiral({ samples: SAMPLES, noise: 0.12 });

function buildModel() {
  return new Sequential()
    .add(new Linear(2, 32, { activation: "relu", seed: 1 }))
    .add(new Linear(32, 2, { seed: 2 }));
}

const model = buildModel();
el.params.textContent = String(model.countParams());

const optimizer = new Adam({ lr: 0.02 });
let params = model.parameters();

/** 跑一轮全量训练，返回损失 */
function trainStep() {
  for (const p of params) p.zeroGrad();

  const logits = model.forward(dataX);
  const loss = crossEntropy(logits, labels);
  backward(loss);

  optimizer.step(params, params.map((p) => p.grad));
  return loss.data[0];
}

/** 评估当前准确率与预测标签 */
function evaluate() {
  const logits = model.forward(dataX);
  const probs = softmax(logits);
  const predicted = new Array(SAMPLES);
  for (let i = 0; i < SAMPLES; i++) {
    predicted[i] = probs.data[i * 2] > probs.data[i * 2 + 1] ? 0 : 1;
  }
  return { loss: crossEntropy(logits, labels).data[0], acc: accuracy(logits, labels), predicted };
}

/* ---------- 可视化 ---------- */

// 预先算好网格坐标，避免每帧重复分配
const gridPoints = [];
for (let gy = 0; gy < GRID; gy++) {
  for (let gx = 0; gx < GRID; gx++) {
    gridPoints.push([
      -1.6 + (gx / (GRID - 1)) * 3.2,
      -1.6 + (gy / (GRID - 1)) * 3.2,
    ]);
  }
}
const gridData = Float64Array.from(gridPoints.flat());
// 必须用 Tensor.tensor 构造：手写普通对象会缺少 ndim/size 等访问器，
// 导致 matmul 走到「仅支持 2D 或 3D，得到 undefinedD」分支。
const gridInput = Tensor.tensor(gridData, [gridPoints.length, 2]);

let predictedLabels = new Array(SAMPLES).fill(0);

function draw(probs) {
  const W = canvas.width, H = canvas.height;
  const cell = W / GRID;

  // 1) 决策区域：颜色深浅反映对该类的置信度
  for (let i = 0; i < gridPoints.length; i++) {
    const p = probs.data[i * 2];        // 类别 A 的概率
    const c = COLORS[0];
    const alpha = 0.08 + Math.abs(p - 0.5) * 0.16;
    ctx.fillStyle = `rgba(${c[0]},${c[1]},${c[2]},${alpha})`;
    ctx.fillRect((i % GRID) * cell, Math.floor(i / GRID) * cell, cell + 1, cell + 1);
  }

  // 2) 数据点
  const toPx = (v) => ((v + 1.6) / 3.2) * W;
  for (let i = 0; i < SAMPLES; i++) {
    const px = toPx(dataX.data[i * 2]);
    const py = H - toPx(dataX.data[i * 2 + 1]);
    const correct = predictedLabels[i] === labels[i];

    ctx.beginPath();
    ctx.arc(px, py, 4, 0, Math.PI * 2);
    ctx.fillStyle = correct ? "rgba(255,255,255,0.55)" : "#ff3b3b";
    ctx.fill();

    ctx.beginPath();
    ctx.arc(px, py, 4, 0, Math.PI * 2);
    ctx.strokeStyle = `rgb(${COLORS[labels[i]].join(",")})`;
    ctx.lineWidth = 2;
    ctx.stroke();
  }
}

/* ---------- 训练循环 ---------- */

let epoch = 0;
let running = false;
let frameId = 0;
let lastFrameTime = 0;

function updateStats(loss, acc, speed) {
  el.epoch.textContent = String(epoch);
  el.loss.textContent = loss.toFixed(4);
  el.acc.textContent = (acc * 100).toFixed(1) + "%";
  el.speed.textContent = speed;
  el.bar.style.width = ((epoch / TOTAL_EPOCHS) * 100).toFixed(1) + "%";
}

function loop(timestamp) {
  if (!running) return;

  if (!lastFrameTime) lastFrameTime = timestamp;
  const dt = timestamp - lastFrameTime;
  lastFrameTime = timestamp;

  // 每帧步数随帧时长自适应：慢机器上不至于动画卡死
  const steps = dt > 40 ? 1 : 3;
  let loss = 0;
  for (let s = 0; s < steps; s++) {
    if (epoch >= TOTAL_EPOCHS) { finish(); return; }
    epoch++;
    loss = trainStep();
  }

  const result = evaluate();
  predictedLabels = result.predicted;
  render();
  updateStats(result.loss, result.acc, steps + " 步/帧");

  frameId = requestAnimationFrame(loop);
}

function finish() {
  running = false;
  cancelAnimationFrame(frameId);
  const result = evaluate();
  predictedLabels = result.predicted;
  render();
  updateStats(result.loss, result.acc, "完成");
  el.pause.disabled = true;
  el.start.textContent = "已完成";
  el.start.disabled = true;
}

function reset() {
  running = false;
  cancelAnimationFrame(frameId);
  epoch = 0;
  lastFrameTime = 0;

  // 重建模型以恢复初始权重
  const fresh = buildModel();
  model.layers = fresh.layers;
  params = model.parameters();

  el.loss.textContent = "—";
  el.acc.textContent = "—";
  el.speed.textContent = "—";
  el.bar.style.width = "0%";
  el.start.textContent = "开始训练";
  el.start.disabled = false;
  el.pause.disabled = true;

  const result = evaluate();
  predictedLabels = result.predicted;
  updateStats(0, 0, "待开始");
  render();
}

/* ---------- 事件绑定 ---------- */

el.start.addEventListener("click", () => {
  if (running) return;
  if (epoch >= TOTAL_EPOCHS) reset();
  running = true;
  lastFrameTime = 0;
  el.start.disabled = true;
  el.pause.disabled = false;
  frameId = requestAnimationFrame(loop);
});

el.pause.addEventListener("click", () => {
  if (!running) return;
  running = false;
  cancelAnimationFrame(frameId);
  el.start.disabled = false;
  el.pause.disabled = true;
  el.speed.textContent = "已暂停";
});

el.reset.addEventListener("click", reset);

reset();
function render() {
  draw(softmax(model.forward(gridInput)));
}
