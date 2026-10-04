import {
  Adam,
  Linear,
  Sequential,
  Trainer,
  accuracy,
  crossEntropy,
  makeSpiral,
} from "../src/index.js";

const { x, y } = makeSpiral({ samples: 300 });

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

trainer.fit(x, y, y);

const final = trainer.history.at(-1);
console.log("final epoch:", final.epoch);
console.log("loss:", final.loss.toFixed(4));
console.log("accuracy:", `${(final.metric * 100).toFixed(1)}%`);

