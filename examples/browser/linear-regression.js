import {
  Adam,
  Linear,
  Sequential,
  Trainer,
  makeLinearRegression,
  mse,
} from "../../src/index.js";

export function createRegressionRuntime({
  epochs = 400,
  learningRate = 0.1,
  batchSize = 25,
  seed = 12345,
} = {}) {
  const data = makeLinearRegression({ samples: 100 });
  const model = new Sequential().add(new Linear(1, 1, { seed: 4 }));
  const trainer = new Trainer({
    model,
    optimizer: new Adam({ lr: learningRate }),
    lossFn: mse,
    epochs,
    batchSize,
  });

  let randomState = seed >>> 0;
  const rng = () => {
    randomState ^= randomState << 13;
    randomState >>>= 0;
    randomState ^= randomState >> 17;
    randomState ^= randomState << 5;
    randomState >>>= 0;
    return randomState / 4294967296;
  };

  return {
    data,
    model,
    trainer,
    rng,
    epochs,
    epoch: 0,
    loss: null,
  };
}

export function trainRegressionEpoch(runtime) {
  const record = runtime.trainer.runEpoch(
    runtime.data.x,
    runtime.data.yTensor,
    runtime.data.yTensor,
    runtime.rng,
  );

  runtime.epoch += 1;
  runtime.loss = record.loss;
  return record;
}

export function evaluateRegression(runtime) {
  const layer = runtime.model.layers[0];
  const prediction = runtime.model.forward(runtime.data.x);
  return {
    loss: mse(prediction, runtime.data.yTensor).data[0],
    weight: layer.weight.data[0],
    bias: layer.bias.data[0],
    trueWeight: runtime.data.trueW,
    trueBias: runtime.data.trueB,
    predictions: prediction,
  };
}

export function trainRegression(runtime, steps = runtime.epochs) {
  for (let step = 0; step < steps; step++) trainRegressionEpoch(runtime);
  return evaluateRegression(runtime);
}

function getPlotBounds(data) {
  let minimumX = Infinity;
  let maximumX = -Infinity;
  let minimumY = Infinity;
  let maximumY = -Infinity;

  for (let index = 0; index < data.x.data.length; index++) {
    const x = data.x.data[index];
    const y = data.yTensor.data[index];
    minimumX = Math.min(minimumX, x);
    maximumX = Math.max(maximumX, x);
    minimumY = Math.min(minimumY, y);
    maximumY = Math.max(maximumY, y);
  }

  return {
    minimumX: Math.min(0, minimumX),
    maximumX: Math.max(2, maximumX),
    minimumY: minimumY - 0.25,
    maximumY: maximumY + 0.25,
  };
}

function drawRegression(canvas, runtime, result) {
  const context = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const padding = { left: 58, right: 28, top: 28, bottom: 48 };
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;
  const bounds = getPlotBounds(runtime.data);

  const toX = (value) =>
    padding.left + ((value - bounds.minimumX) / (bounds.maximumX - bounds.minimumX)) * plotWidth;
  const toY = (value) =>
    padding.top + plotHeight -
    ((value - bounds.minimumY) / (bounds.maximumY - bounds.minimumY)) * plotHeight;

  context.fillStyle = "#0d1017";
  context.fillRect(0, 0, width, height);

  context.strokeStyle = "#252c39";
  context.lineWidth = 1;
  context.font = "12px ui-sans-serif, sans-serif";
  context.fillStyle = "#7f8798";
  context.textAlign = "center";
  context.textBaseline = "top";

  for (let step = 0; step <= 4; step++) {
    const value = bounds.minimumX + (step / 4) * (bounds.maximumX - bounds.minimumX);
    const x = toX(value);
    context.beginPath();
    context.moveTo(x, padding.top);
    context.lineTo(x, padding.top + plotHeight);
    context.stroke();
    context.fillText(value.toFixed(1), x, padding.top + plotHeight + 10);
  }

  context.textAlign = "right";
  context.textBaseline = "middle";
  for (let step = 0; step <= 4; step++) {
    const value = bounds.minimumY + (step / 4) * (bounds.maximumY - bounds.minimumY);
    const y = toY(value);
    context.beginPath();
    context.moveTo(padding.left, y);
    context.lineTo(padding.left + plotWidth, y);
    context.stroke();
    context.fillText(value.toFixed(1), padding.left - 10, y);
  }

  const targetStartY = runtime.data.trueW * bounds.minimumX + runtime.data.trueB;
  const targetEndY = runtime.data.trueW * bounds.maximumX + runtime.data.trueB;
  context.beginPath();
  context.setLineDash([7, 6]);
  context.moveTo(toX(bounds.minimumX), toY(targetStartY));
  context.lineTo(toX(bounds.maximumX), toY(targetEndY));
  context.strokeStyle = "#e7b85b";
  context.lineWidth = 2;
  context.stroke();
  context.setLineDash([]);

  for (let index = 0; index < runtime.data.x.data.length; index++) {
    const x = runtime.data.x.data[index];
    const y = runtime.data.yTensor.data[index];
    const predictedY = result.predictions.data[index];
    const pixelX = toX(x);
    const pixelY = toY(y);

    context.beginPath();
    context.moveTo(pixelX, pixelY);
    context.lineTo(pixelX, toY(predictedY));
    context.strokeStyle = "rgba(119, 167, 255, 0.16)";
    context.lineWidth = 1;
    context.stroke();

    context.beginPath();
    context.arc(pixelX, pixelY, 4.5, 0, Math.PI * 2);
    context.fillStyle = "#77a7ff";
    context.fill();
    context.strokeStyle = "#0d1017";
    context.lineWidth = 2;
    context.stroke();
  }

  context.beginPath();
  context.moveTo(toX(bounds.minimumX), toY(result.weight * bounds.minimumX + result.bias));
  context.lineTo(toX(bounds.maximumX), toY(result.weight * bounds.maximumX + result.bias));
  context.strokeStyle = "#4ad6c8";
  context.lineWidth = 3;
  context.stroke();
}

function mountRegressionExample() {
  const canvas = document.querySelector("#view");
  if (!canvas) return;

  const elements = {
    epoch: document.querySelector("#epoch"),
    loss: document.querySelector("#loss"),
    weight: document.querySelector("#weight"),
    bias: document.querySelector("#bias"),
    target: document.querySelector("#target"),
    speed: document.querySelector("#speed"),
    progress: document.querySelector("#progress"),
    start: document.querySelector("#start"),
    pause: document.querySelector("#pause"),
    reset: document.querySelector("#reset"),
  };

  let runtime = createRegressionRuntime();
  let running = false;
  let frameId = 0;
  let lastTimestamp = 0;

  function update(result, speed) {
    elements.epoch.textContent = `${runtime.epoch} / ${runtime.epochs}`;
    elements.loss.textContent = result.loss.toFixed(4);
    elements.weight.textContent = result.weight.toFixed(3);
    elements.bias.textContent = result.bias.toFixed(3);
    elements.target.textContent = `${result.trueWeight.toFixed(3)}, ${result.trueBias.toFixed(3)}`;
    elements.speed.textContent = speed;
    elements.progress.style.width = `${Math.min(100, (runtime.epoch / runtime.epochs) * 100)}%`;
  }

  function render(speed) {
    const result = evaluateRegression(runtime);
    drawRegression(canvas, runtime, result);
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
    const steps = elapsed > 40 ? 2 : 6;

    for (let step = 0; step < steps; step++) {
      if (runtime.epoch >= runtime.epochs) {
        finish();
        return;
      }
      trainRegressionEpoch(runtime);
    }

    render(`${steps} epochs/frame`);
    frameId = requestAnimationFrame(loop);
  }

  function reset() {
    running = false;
    cancelAnimationFrame(frameId);
    runtime = createRegressionRuntime();
    lastTimestamp = 0;

    elements.loss.textContent = "-";
    elements.weight.textContent = "-";
    elements.bias.textContent = "-";
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

if (typeof document !== "undefined") mountRegressionExample();
