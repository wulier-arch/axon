# Getting started

This guide gets you from zero to a trained model and a verified gradient.

## Install

```bash
npm install github:wulier-arch/axon
```

The registry package is not published yet, so install directly from GitHub.
The package has no runtime dependencies. The same source also runs directly in modern browsers.

## Train a model

```js
import {
  Adam,
  Linear,
  Sequential,
  Trainer,
  accuracy,
  crossEntropy,
  makeSpiral,
} from "axon-net";

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
console.log(trainer.history.at(-1));
```

The spiral dataset is not linearly separable, so this small network has to learn a nonlinear decision boundary.

## Understand autograd

Leaf tensors are created with `Tensor.variable`. Operators build a graph, and `backward` propagates gradients back through it.

```js
import { Tensor, backward, mul } from "axon-net";

const x = Tensor.variable([2, 3], [2]);
backward(mul(x, x).sum());

console.log(Array.from(x.grad)); // [4, 6]
```

## Verify a gradient

`checkGradient` compares the analytical backward pass with central finite differences.

```js
import { Tensor, checkGradient, mul } from "axon-net";

const x = Tensor.variable([2, 3], [2]);
const report = checkGradient(() => mul(x, x).sum(), [x]);

if (!report.passed) throw new Error(report.report);
```

For deep expressions, compare errors at two shrinking epsilon values. The error should fall roughly linearly while truncation dominates; a gradient bug leaves a larger residual.

## Run the repository examples

```bash
git clone https://github.com/wulier-arch/axon.git
cd axon
node examples/autograd.mjs
node examples/train-spiral.mjs
```

## Run the browser examples

The browser examples import the framework source directly. They need no install step,
bundler, or backend:

- [XOR classifier](https://wulier-arch.github.io/axon/examples/browser/xor.html)
- [Linear regression](https://wulier-arch.github.io/axon/examples/browser/linear-regression.html)

From a local clone, serve the repository root and open either page:

```bash
python3 -m http.server 8000
```

## Next steps

- Open the browser demo: https://wulier-arch.github.io/axon/demo/
- Try the browser examples: https://wulier-arch.github.io/axon/examples/browser/xor.html
- Read the [English API reference](api.md)
- Pick a task from the pinned roadmap issue: https://github.com/wulier-arch/axon/issues/1
