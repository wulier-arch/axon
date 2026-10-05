import {
  Adam,
  Linear,
  Sequential,
  Tensor,
  accuracy,
  backward,
  crossEntropy,
  makeXor,
  softmax,
} from "../../src/index.js";

const COLORS = ["#ff6b8b", "#4ad6c8"];
const GRID_SIZE = 48;
const INPUT_MIN = -0.24;
const INPUT_MAX = 1.24;

export function createXorRuntime({ epochs = 400, hidden = 8, learningRate = 0.05 } = {}) {
  const { x, y } = makeXor();
  const model = new Sequential()
    .add(new Linear(2, hidden, { activation: "relu", seed: 9 }))
    .add(new Linear(hidden, 2, { seed: 10 }));

  return {
    x,
    labels: y,
    model,
    optimizer: new Adam({ lr: learningRate }),
    parameters: model.parameters(),
    epochs,
    epoch: 0,
    loss: null,
  };
}

export function trainXorEpoch(runtime) {
  for (const parameter of runtime.parameters) parameter.zeroGrad();

  const logits = runtime.model.forward(runtime.x);
  const loss = crossEntropy(logits, runtime.labels);
  backward(loss);
  runtime.optimizer.step(
    runtime.parameters,
    runtime.parameters.map((parameter) => parameter.grad),
  );

  runtime.epoch += 1;
  runtime.loss = loss.data[0];
  return runtime.loss;
}

export function evaluateXor(runtime) {
  const logits = runtime.model.forward(runtime.x);
  const probabilities = softmax(logits);
  const predictions = [];

  for (let row = 0; row < runtime.labels.length; row++) {
    predictions.push(probabilities.data[row * 2] > probabilities.data[row * 2 + 1] ? 0 : 1);
  }

  return {
    loss: crossEntropy(logits, runtime.labels).data[0],
    accuracy: accuracy(logits, runtime.labels),
    predictions,
  };
}

export function trainXor(runtime, steps = runtime.epochs) {
  for (let step = 0; step < steps; step++) trainXorEpoch(runtime);
  return evaluateXor(runtime);
}

function makeGrid() {
  const points = [];
  for (let row = 0; row < GRID_SIZE; row++) {
    for (let column = 0; column < GRID_SIZE; column++) {
      points.push([
        INPUT_MIN + (column / (GRID_SIZE - 1)) * (INPUT_MAX - INPUT_MIN),
        INPUT_MIN + (row / (GRID_SIZE - 1)) * (INPUT_MAX - INPUT_MIN),
      ]);
    }
  }
  return Tensor.tensor(Float64Array.from(points.flat()), [points.length, 2]);
}

function blend(hex, alpha) {
  const value = Number.parseInt(hex.slice(1), 16);
  const red = (value >> 16) & 255;
  const green = (value >> 8) & 255;
  const blue = value & 255;
  return `rgba(${red}, ${green}, ${blue}, ${alpha})`;
}

function drawXor(canvas, runtime, grid, predictions) {
  const context = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const padding = 46;
  const plotWidth = width - padding * 2;
  const plotHeight = height - padding * 2;
  const cellWidth = plotWidth / GRID_SIZE;
  const cellHeight = plotHeight / GRID_SIZE;

  const probabilities = softmax(runtime.model.forward(grid));
  context.fillStyle = "#0d1017";
  context.fillRect(0, 0, width, height);

  for (let index = 0; index < probabilities.shape[0]; index++) {
    const classZeroProbability = probabilities.data[index * 2];
    const color = classZeroProbability >= 0.5 ? COLORS[0] : COLORS[1];
    const confidence = Math.abs(classZeroProbability - 0.5) * 2;
    context.fillStyle = blend(color, 0.05 + confidence * 0.18);
    context.fillRect(
      padding + (index % GRID_SIZE) * cellWidth,
      padding + Math.floor(index / GRID_SIZE) * cellHeight,
      cellWidth + 1,
      cellHeight + 1,
    );
  }

  context.strokeStyle = "#2b3342";
  context.lineWidth = 1;
  context.strokeRect(padding, padding, plotWidth, plotHeight);

  const toX = (value) => padding + ((value - INPUT_MIN) / (INPUT_MAX - INPUT_MIN)) * plotWidth;
  const toY = (value) => padding + plotHeight -
    ((value - INPUT_MIN) / (INPUT_MAX - INPUT_MIN)) * plotHeight;

  for (let row = 0; row < runtime.labels.length; row++) {
    const x = toX(runtime.x.data[row * 2]);
    const y = toY(runtime.x.data[row * 2 + 1]);
    const correct = predictions[row] === runtime.labels[row];
    const color = COLORS[runtime.labels[row]];

    context.beginPath();
    context.arc(x, y, correct ? 9 : 11, 0, Math.PI * 2);
    context.fillStyle = correct ? color : "#ff4d4d";
    context.fill();
    context.lineWidth = 3;
    context.strokeStyle = "#0d1017";
    context.stroke();

    context.beginPath();
    context.arc(x, y, 4, 0, Math.PI * 2);
    context.fillStyle = "#f7f8fb";
    context.fill();
  }
}

function mountXorExample() {
  const canvas = document.querySelector("#view");
  if (!canvas) return;

  const elements = {
    epoch: document.querySelector("#epoch"),
    loss: document.querySelector("#loss"),
    accuracy: document.querySelector("#accuracy"),
    speed: document.querySelector("#speed"),
    parameters: document.querySelector("#parameters"),
    progress: document.querySelector("#progress"),
    start: document.querySelector("#start"),
    pause: document.querySelector("#pause"),
    reset: document.querySelector("#reset"),
  };

  const grid = makeGrid();
  let runtime = createXorRuntime();
  let running = false;
  let frameId = 0;
  let lastTimestamp = 0;

  function update(result, speed) {
    elements.epoch.textContent = `${runtime.epoch} / ${runtime.epochs}`;
    elements.loss.textContent = result.loss.toFixed(4);
    elements.accuracy.textContent = `${(result.accuracy * 100).toFixed(1)}%`;
    elements.speed.textContent = speed;
    elements.progress.style.width = `${Math.min(100, (runtime.epoch / runtime.epochs) * 100)}%`;
  }

  function render(speed) {
    const result = evaluateXor(runtime);
    drawXor(canvas, runtime, grid, result.predictions);
    update(result, speed);
  }

  function finish() {
    running = false;
    cancelAnimationFrame(frameId);
    render("Complete");
    elements.start.disabled = true;
    elements.start.textContent = "Complete";
    elements.pause.disabled = true;
  }

  function loop(timestamp) {
    if (!running) return;

    if (!lastTimestamp) lastTimestamp = timestamp;
    const elapsed = timestamp - lastTimestamp;
    lastTimestamp = timestamp;
    const steps = elapsed > 40 ? 3 : 8;

    for (let step = 0; step < steps; step++) {
      if (runtime.epoch >= runtime.epochs) {
        finish();
        return;
      }
      trainXorEpoch(runtime);
    }

    render(`${steps} steps/frame`);
    frameId = requestAnimationFrame(loop);
  }

  function reset() {
    running = false;
    cancelAnimationFrame(frameId);
    runtime = createXorRuntime();
    lastTimestamp = 0;

    elements.parameters.textContent = String(runtime.model.countParams());
    elements.loss.textContent = "-";
    elements.accuracy.textContent = "-";
    elements.speed.textContent = "-";
    elements.start.disabled = false;
    elements.start.textContent = "Start training";
    elements.pause.disabled = true;
    render("Ready");
  }

  elements.start.addEventListener("click", () => {
    if (running) return;
    if (runtime.epoch >= runtime.epochs) reset();
    running = true;
    lastTimestamp = 0;
    elements.start.disabled = true;
    elements.pause.disabled = false;
    frameId = requestAnimationFrame(loop);
  });

  elements.pause.addEventListener("click", () => {
    if (!running) return;
    running = false;
    cancelAnimationFrame(frameId);
    elements.start.disabled = false;
    elements.pause.disabled = true;
    elements.speed.textContent = "Paused";
  });

  elements.reset.addEventListener("click", reset);
  reset();
}

if (typeof document !== "undefined") mountXorExample();
