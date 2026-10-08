# API reference

This reference covers every public export in axon. `loadModel` is not yet in a
tagged release. All APIs are available from the package root:

```js
import {
  Tensor,
  Linear,
  Sequential,
  Adam,
  Trainer,
  crossEntropy,
} from "axon-net";
```

The package is dependency-free and the same modules run in Node.js and modern
browsers. Tensor values are stored in `Float64Array` buffers.

## Contents

- [Core tensor API](#core-tensor-api)
- [Tensor operations](#tensor-operations)
- [Shape, indexing, and convolution](#shape-indexing-and-convolution)
- [Autograd](#autograd)
- [Gradient checking](#gradient-checking)
- [Neural-network layers](#neural-network-layers)
- [Losses and metrics](#losses-and-metrics)
- [Optimizers](#optimizers)
- [Training](#training)
- [Synthetic datasets](#synthetic-datasets)

## Core tensor API

### `Tensor`

```js
new Tensor(data, shape)
Tensor.tensor(data, shape)
Tensor.variable(data, shape)
```

`Tensor` stores an N-dimensional value and the graph state needed for
reverse-mode automatic differentiation.

- `data`: a `Float64Array`, or any iterable of numbers that can be converted to
  one.
- `shape`: an array of non-negative dimensions. The product of the shape must
  equal the number of values.
- `Tensor.tensor(data, shape)` creates a constant tensor.
- `Tensor.variable(data, shape)` creates a leaf tensor with
  `requiresGrad = true`.

Useful properties and methods:

- `shape`: the tensor dimensions.
- `data`: the backing `Float64Array`.
- `grad`: the accumulated gradient, or `null` before the first backward pass.
- `requiresGrad`: whether this is a trainable leaf variable.
- `size`: the number of scalar elements.
- `ndim`: the number of dimensions.
- `sum()`: returns the sum of all elements as a scalar tensor.
- `isGraphNode()`: reports whether the tensor is a graph leaf or an operation
  result.
- `accumulateGrad(values)`: adds values to the existing gradient. Most user code
  does not need to call this directly.
- `parameterGrad()`: returns `grad` for a leaf variable, otherwise `null`.
- `zeroGrad()`: clears the tensor gradient.
- `clone()`: returns a new tensor with copied data and shape. The clone is not a
  graph variable.
- `toJSON()`: returns `{ shape, data }` with a plain JavaScript array for data.

```js
import { Tensor } from "axon-net";

const x = Tensor.variable([1, 2, 3], [3]);

console.log(x.size);       // 3
console.log(x.ndim);       // 1
console.log(x.isGraphNode()); // true
console.log(x.toJSON());   // { shape: [3], data: [1, 2, 3] }
```

### `numElements`

```js
numElements(shape) -> number
```

Returns the product of all dimensions. An empty shape represents a scalar and
returns `1`.

```js
numElements([2, 3, 4]); // 24
numElements([]);        // 1
```

### `swap`

```js
swap(shape) -> number[]
```

Swaps the two dimensions of a 2D shape. This is a shape helper, not a tensor
transpose.

```js
swap([3, 5]); // [5, 3]
```

## Tensor operations

Every operation below returns a new `Tensor`. If any input is part of a graph,
the result records the operation needed for backpropagation.

### `add`

```js
add(a, b) -> Tensor
```

Elementwise addition with NumPy-style, right-aligned broadcasting. Examples
include `[2, 3] + [3]`, `[2, 3] + [1]`, and `[2, 3] + [2, 3]`. Incompatible
shapes throw an error.

### `sub`

```js
sub(a, b) -> Tensor
```

Elementwise subtraction. It is implemented as `add(a, scale(b, -1))`, so it
supports the same broadcasting rules as `add`.

### `mul`

```js
mul(a, b) -> Tensor
```

Elementwise multiplication. The inputs must have the same number of elements
and are paired by flat element order. The result uses `a.shape`.

### `div`

```js
div(a, b) -> Tensor
```

Elementwise division by flat element order. The inputs are expected to have the
same number of elements. Broadcasting is not implemented. Division by zero
follows normal JavaScript number semantics.

### `scale`

```js
scale(a, factor) -> Tensor
```

Multiplies every element by a numeric factor while preserving shape and graph
history.

### `matmul`

```js
matmul(a, b) -> Tensor
```

Matrix multiplication for two supported layouts:

- 2D: `[m, k] @ [k, n] -> [m, n]`
- 3D: `[batch, m, k] @ [batch, k, n] -> [batch, m, n]`

The inner dimensions must match. Other ranks throw an error.

### `relu`, `tanh`, `sigmoid`, `exp`, `log`, and `gelu`

```js
relu(a) -> Tensor
tanh(a) -> Tensor
sigmoid(a) -> Tensor
exp(a) -> Tensor
log(a) -> Tensor
gelu(a) -> Tensor
```

These functions apply one scalar function to every element and preserve the
input shape. `gelu` uses the common tanh approximation. `log` follows
`Math.log`, so non-positive inputs produce JavaScript's normal `-Infinity` or
`NaN` values.

### `softmax`

```js
softmax(a) -> Tensor
```

Computes a numerically stable softmax independently along the last dimension.
The output has the same shape as the input and each last-axis slice sums to `1`.

```js
import { Tensor, add, mul, matmul, softmax } from "axon-net";

const x = Tensor.tensor([1, 2, 3, 4], [2, 2]);
const weights = Tensor.tensor([1, -1, 0.5, 2], [2, 2]);

const scores = add(matmul(x, weights), Tensor.tensor([0.1, -0.2], [2]));
const probabilities = softmax(scores);

console.log(probabilities.shape); // [2, 2]
```

### `sum`

```js
sum(a, dim = -1) -> Tensor
```

Sums along one dimension and removes that dimension from the result. Negative
dimensions count from the end.

For a 1D tensor, `sum` always returns a scalar tensor with shape `[1]`. The
`dim` argument does not change that behavior.

```js
sum(Tensor.tensor([1, 2, 3], [3])).data[0];          // 6
sum(Tensor.tensor([1, 2, 3, 4], [2, 2]), 0).shape;   // [2]
sum(Tensor.tensor([1, 2, 3, 4], [2, 2]), 1).shape;   // [2]
```

### `mean`

```js
mean(a, dim = -1) -> Tensor
```

Averages along a dimension. For 1D input it returns a scalar tensor. For other
ranks it follows the same dimension rules and output shape as `sum`.

### `reshape`

```js
reshape(a, shape) -> Tensor
```

Returns a tensor with a new shape and a copied data buffer. The number of
elements must remain the same. Inferred dimensions such as `-1` are not
supported.

### `transpose`

```js
transpose(a, axes) -> Tensor
```

Permutes tensor axes. With no `axes`, all axes are reversed, which is equivalent
to a matrix transpose for 2D input. A custom `axes` array must be a permutation
of `0` through `a.ndim - 1`.

```js
transpose(Tensor.tensor([1, 2, 3, 4, 5, 6], [2, 3])).shape;       // [3, 2]
transpose(Tensor.tensor([1, 2, 3, 4, 5, 6], [2, 3]), [0, 1]).shape; // [2, 3]
```

## Shape, indexing, and convolution

### `gatherRows`

```js
gatherRows(a, indices) -> Tensor
```

Selects rows from a 2D tensor. `indices` is an array of zero-based row numbers.
The result has shape `[indices.length, a.shape[1]]`, and repeated indices
accumulate gradient on the source rows.

```js
const table = Tensor.tensor([1, 2, 3, 4], [2, 2]);
const rows = gatherRows(table, [1, 1, 0]);
console.log(rows.toJSON());
// { shape: [3, 2], data: [3, 4, 3, 4, 1, 2] }
```

### `concat`

```js
concat(list, axis = 0) -> Tensor
```

Concatenates tensors along the first axis. All trailing dimensions must match.
Only `axis = 0` is supported.

```js
const a = Tensor.tensor([1, 2], [1, 2]);
const b = Tensor.tensor([3, 4, 5, 6], [2, 2]);
concat([a, b]).shape; // [3, 2]
```

### `argmaxLast`

```js
argmaxLast(a) -> number[]
```

Returns the index of the largest value in each last-axis slice as a plain
JavaScript array. On ties, the first occurrence wins. There is no gradient for
an argmax.

### `conv2d`

```js
conv2d(x, weight, bias = null, stride = 1, padding = 0) -> Tensor
```

2D convolution for a single image or a batch, with channels-last output:

- `x`: `[C, H, W]` or `[N, C, H, W]`
- `weight`: `[outChannels, C, kernelHeight, kernelWidth]`
- `bias`: optional `[outChannels]`
- `stride`: a number or `[strideHeight, strideWidth]`
- `padding`: a number or `[paddingHeight, paddingWidth]`
- output: `[outHeight, outWidth, outChannels]` or
  `[N, outHeight, outWidth, outChannels]`

The implementation uses im2col followed by matrix multiplication. Batched input
shares the same weight and bias across all `N` images. Gradients are accumulated
across the batch for the weight and bias.

```js
import { Tensor, conv2d } from "axon-net";

const image = Tensor.tensor(
  [1, 2, 3, 4, 5, 6, 7, 8, 9],
  [1, 3, 3]
);
const kernel = Tensor.tensor([1, 0, 0, -1], [1, 1, 2, 2]);
const output = conv2d(image, kernel);

console.log(output.shape); // [2, 2, 1]
```

### `maxPool2d` and `avgPool2d`

```js
maxPool2d(x, kernel = 2, stride = null) -> Tensor
avgPool2d(x, kernel = 2, stride = null) -> Tensor
```

Pool one `[C, H, W]` input into `[outHeight, outWidth, C]`. `kernel` is a
number, `stride` defaults to the kernel size, and only complete windows are
computed. Max pooling sends gradient to the selected maximum element; average
pooling divides gradient evenly over the window.

## Autograd

### `backward`

```js
backward(root) -> void
```

Runs reverse-mode automatic differentiation from a scalar graph root. If the
root has no gradient yet, it is seeded with `1`. Gradients are accumulated on
leaf variables and intermediate graph nodes.

Call `backward` once per optimization step and clear old gradients before the
next forward pass.

```js
import { Tensor, backward, mul } from "axon-net";

const x = Tensor.variable([2, 3], [2]);
backward(mul(x, x).sum());

console.log(Array.from(x.grad)); // [4, 6]
```

### `collectLeaves`

```js
collectLeaves(root) -> Tensor[]
```

Traverses a graph and returns all reachable leaf tensors with
`requiresGrad = true`.

### `zeroGrad`

```js
zeroGrad(root) -> void
```

Clears gradients on every tensor reachable from `root`. A common pattern is
`zeroGrad(loss)` before the next forward pass, or
`for (const p of model.parameters()) p.zeroGrad()`.

## Gradient checking

Gradient checking evaluates a scalar loss at perturbed input values. The input
data is restored after each helper returns.

### `numericalGradient`

```js
numericalGradient(lossFn, inputs, eps = 1e-5) -> Float64Array[]
```

Computes a first-order central-difference gradient for every input tensor. The
returned array is aligned with `inputs`.

### `highOrderGradient`

```js
highOrderGradient(lossFn, inputs, h = 7e-4) -> Float64Array[]
```

Computes a fourth-order central-difference approximation. It is useful when a
small first-order difference looks suspicious because of roundoff.

### `checkGradient`

```js
checkGradient(lossFn, inputs, { eps = 1e-5, tol = 1e-5 } = {}) -> {
  maxAbsError,
  maxRelError,
  passed,
  report
}
```

Compares the gradients already stored on `inputs` with both numerical
approximations. Run the analytical backward pass first:

```js
import { Tensor, backward, checkGradient, div, log, mul, sum } from "axon-net";

const x = Tensor.variable([0.5, 1.5], [2]);
backward(sum(mul(x, log(x))));

const result = checkGradient(() => sum(mul(x, log(x))), [x]);
console.log(result.report, result.passed);
```

`checkGradient` is intended for small tensors. Its cost grows with the number
of input elements because the loss function is evaluated repeatedly for each
one.

## Neural-network layers

Layer classes expose compatible methods where applicable:

- `forward(input) -> Tensor`
- `parameters() -> Tensor[]`
- `toJSON()`, where serialization is implemented

### `makeRng`

```js
makeRng(seed = 42) -> () => number
```

Creates a deterministic xorshift32 random-number function that returns values
in `[0, 1)`. The same seed produces the same sequence. Seed `0` is normalized
to `1`.

### `Linear`

```js
new Linear(inFeatures, outFeatures, opts = {})
```

Implements `y = xW + b` for `[batch, inFeatures]` input. The weight has shape
`[inFeatures, outFeatures]`. Initialization is Gaussian with standard deviation
`sqrt(2 / inFeatures)` by default.

Options:

- `bias`: set to `false` to omit the bias tensor. Default: `true`.
- `activation`: `"relu"`, `"tanh"`, or `"sigmoid"`. Default: no activation.
- `seed`: initialization seed. Default: `1234`.
- `initScale`: explicit initialization standard deviation.
- `name`: layer name used by `Sequential.summary()`.

Methods: `forward(x)`, `parameters()`, and `toJSON()`.

### `Dropout`

```js
new Dropout(p = 0.5, seed = 999)
```

Masks elements independently when `p > 0`; `p = 0` returns the input tensor
unchanged. Backward uses the inverse keep probability. v0.3.1 has no train/eval
mode switch, so call this layer only during training or omit it when you need
deterministic evaluation.

Methods: `forward(x)` and `toJSON()`.

### `LayerNorm`

```js
new LayerNorm(features, opts = {})
```

Normalizes each row of a `[rows, features]` input over the feature dimension and
applies optional elementwise scale and shift.

Options:

- `eps`: variance floor. Default: `1e-5`.
- `gamma`: set to `false` to disable the learned scale.
- `beta`: set to `false` to disable the learned shift.

Methods: `forward(x)`, `parameters()`, and `toJSON()`.

### `Embedding`

```js
new Embedding(vocabSize, features, opts = {})
```

Maps a sequence of integer token IDs to dense vectors. `forward(ids)` returns a
tensor with shape `[ids.length, features]`. IDs must be integers in
`[0, vocabSize)`. Repeated IDs accumulate gradient on the same embedding row.

Options:

- `seed`: initialization seed. Default: `7`.

Methods: `forward(ids)`, `parameters()`, and `toJSON()`.

### `MultiHeadAttention`

```js
new MultiHeadAttention(dModel, heads, opts = {})
```

Applies self-attention to a `[sequenceLength, dModel]` tensor and returns the
same shape. `dModel` must be divisible by `heads`.

Options:

- `causal`: mask future positions when `true`. Default: `false`.
- `seed`: initialization seed. Default: `21`.
- `initScale`: explicit projection initialization standard deviation.
  Default: `1 / sqrt(dModel)`.

Methods: `forward(x)`, `parameters()`, and `toJSON()`.

### `TransformerBlock`

```js
new TransformerBlock(dModel, heads, opts = {})
```

Combines multi-head self-attention, residual connections, layer normalization,
and a two-layer position-wise feed-forward network. Input and output are
`[sequenceLength, dModel]`.

Options:

- `dFF`: feed-forward hidden size. Default: `4 * dModel`.
- `causal`: use causal attention. Default: `false`.
- `normFirst`: use pre-layer normalization when `true`. Default: `true`.
- `seed`: initialization seed. Default: `31`.
- `ffnActivation`: `"gelu"`, `"relu"`, or `"tanh"`. Default: `"gelu"`.

Methods: `forward(x)`, `parameters()`, `countParams()`, and `toJSON()`.

### `Sequential`

```js
new Sequential(layers = [])
```

Runs layers in order and exposes a single parameter list for the optimizer.

Methods:

- `add(layer)`: appends a layer and returns the same `Sequential` instance.
- `forward(x)`: runs every layer in order.
- `parameters()`: collects parameters from every layer that implements
  `parameters()`.
- `countParams()`: returns the total number of parameter elements.
- `summary()`: returns a compact layer and parameter summary.
- `toJSON()`: serializes layers that implement `toJSON()`.

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

### `loadModel`

```js
loadModel(serialized) -> Sequential
```

Rebuilds a model from a plain object produced by `Sequential.toJSON()`, or from
the equivalent JSON string. The reconstructed model has the same layers, the
same weights, and, for `Dropout` layers, the same random-number state, so
training continues from the exact point where it was serialized.

```js
import { loadModel, makeSpiral, Sequential } from "axon-net";

const { x, y } = makeSpiral({ samples: 120 });
const model = new Sequential()
  .add(new Linear(2, 16, { activation: "relu", seed: 1 }))
  .add(new Linear(16, 2, { seed: 2 }));

// ... train the model ...

const snapshot = JSON.stringify(model.toJSON());
const restored = loadModel(snapshot);

console.log(restored.summary());
console.log(restored.countParams());
```

The payload is validated before any layer is built: the format marker
(`"axon-model"`), the version, and the shape of every layer entry must be
consistent, and every weight must be a finite number of the exact expected
length. A mismatch throws with a message naming the offending path. Loading
never calls `eval` and only constructs the known layer types, so untrusted JSON
cannot introduce arbitrary code or unexpected prototypes.

Round-tripping works for `Linear`, `Dropout`, `LayerNorm`, `Embedding`,
`MultiHeadAttention`, `TransformerBlock`, and nested `Sequential` models.
`TransformerBlock.toJSON()` stores `norm1`, `attn`, `norm2`, `ff1`, and `ff2`
recursively, so no weights are dropped. `Sequential.toJSON()` throws if a layer
does not implement `toJSON()` rather than silently omitting it.

### `activations`

```js
activations.relu(a)
activations.tanh(a)
activations.sigmoid(a)
activations.softmax(a)
```

An object containing four activation functions for code that selects an
activation at runtime. It does not include `gelu`.

## Losses and metrics

Loss functions return a scalar tensor that can be passed directly to
`backward`.

### `mse`

```js
mse(prediction, target) -> Tensor
```

Computes the global mean of `(prediction - target)^2`. The two tensors must have
compatible broadcast shapes.

### `crossEntropy`

```js
crossEntropy(logits, labels) -> Tensor
```

Computes mean softmax cross-entropy. `logits` has shape `[rows, classes]`.
`labels` can be a JavaScript array or a tensor, with one class index per row.
The implementation uses a log-sum-exp calculation for numerical stability.

### `binaryCrossEntropy`

```js
binaryCrossEntropy(prediction, target) -> Tensor
```

Computes mean binary cross-entropy for probabilities and targets with the same
number of elements. Prediction values are clamped to
`[1e-7, 1 - 1e-7]` before the logarithm.

### `accuracy`

```js
accuracy(logits, labels) -> number
```

Returns the fraction of rows whose largest logit matches the label. `labels`
can be a JavaScript array or a tensor. This is a metric and does not return a
graph node.

## Optimizers

All optimizers expose:

```js
optimizer.step(params, grads) -> void
```

`params` is an array of tensors and `grads` is an array with the corresponding
gradient buffers. Updates mutate `tensor.data` in place. Missing gradients are
skipped. Optimizer state assumes that the parameter list keeps the same shape
and order after the first call.

### `SGD`

```js
new SGD({ lr = 0.01, momentum = 0 } = {})
```

Stochastic gradient descent with optional velocity accumulation.

### `Momentum`

```js
new Momentum({ lr = 0.01, beta = 0.9 } = {})
```

Accumulates a velocity using `v = beta * v + grad` and applies `lr * v`.

### `Adam`

```js
new Adam({
  lr = 0.001,
  beta1 = 0.9,
  beta2 = 0.999,
  eps = 1e-8,
  weightDecay = 0,
} = {})
```

Adam with first- and second-moment estimates and bias correction.
`weightDecay > 0` adds L2 regularization to the gradient.

### `AdamW`

```js
new AdamW({
  lr = 0.001,
  beta1 = 0.9,
  beta2 = 0.999,
  eps = 1e-8,
  weightDecay = 0.01,
} = {})
```

Adam with decoupled weight decay. The decay is applied directly to parameter
values rather than being folded into the adaptive gradient.

### `RMSProp`

```js
new RMSProp({ lr = 0.001, beta = 0.9, eps = 1e-8 } = {})
```

RMSProp using a moving average of squared gradients to normalize each update.

### `Scheduler`

```js
new Scheduler(base, strategy = "constant", opts = {})
scheduler.at(step) -> number
```

Returns the learning rate for a zero-based optimizer step. Assign the result to
`optimizer.lr` when needed.

Strategies:

- `"constant"`: always returns `base`.
- `"cosine"`: cosine decay from `base` to `0` over `opts.totalSteps`.
- `"step"`: multiplies `base` by `opts.gamma` every `opts.stepSize` steps.
- `"exponential"`: multiplies `base` by `opts.gamma ^ step`.

Defaults are `stepSize = 20`, `gamma = 0.5`, and `totalSteps = 100`.

```js
import { Adam, Scheduler, Sequential, Trainer } from "axon-net";

const optimizer = new Adam({ lr: 0.01 });
const scheduler = new Scheduler(0.01, "cosine", { totalSteps: 1000 });

// Pass `scheduler` to Trainer, or call scheduler.at(step) and assign
// the result to optimizer.lr in a custom loop.
```

## Training

### `Trainer`

```js
new Trainer({
  model,
  optimizer,
  lossFn,
  metricFn,
  epochs = 50,
  batchSize = 32,
  shuffle = true,
  scheduler,
  onEpochEnd,
  onBatchEnd,
})
```

Coordinates mini-batch training with the forward pass, loss, backward pass, and
optimizer update.

Required options:

- `model`: an object with `forward(x)` and `parameters()`.
- `optimizer`: an object with `step(params, grads)`.
- `lossFn`: a function `(logits, labels) -> Tensor`.

Optional options:

- `metricFn`: `(logits, labels) -> number`.
- `epochs`, `batchSize`, and `shuffle`: training-loop controls.
- `scheduler`: an object with `at(step) -> learningRate`.
- `onEpochEnd(record)`: called after each epoch.
- `onBatchEnd(record)`: called after each batch.

Methods:

- `fit(x, y, labels, { evalX = null, evalY = null } = {}) -> history[]`.
  `x` and `y` are the training features and targets. `labels` is either a
  label array or a target tensor. When both evaluation tensors are supplied,
  each history record also contains `evalLoss` and `evalMetric`.
- `runEpoch(x, y, labels, rng) -> { loss, metric }`: runs one epoch.
- `batches(x, y, batchSize, shuffle, rng)`: generator used by `runEpoch`.
- `fitWithEarlyStopping(...) -> { history, bestEpoch, bestScore }`.

`fitWithEarlyStopping` expects a higher-is-better metric. It uses `evalMetric`
when available and falls back to the training `metric`.

```js
const history = trainer.fit(x, y, y, { evalX: x, evalY: y });
const last = history.at(-1);

console.log(last.epoch, last.loss, last.metric);
```

## Synthetic datasets

Dataset functions return tensors and labels that can be passed directly to
`Trainer`.

### `makeBlobs`

```js
makeBlobs({
  samples = 200,
  features = 2,
  classes = 2,
  scale = 2,
  seed = 7,
} = {}) -> { x, y, classes }
```

Generates samples around randomly placed Gaussian cluster centers. `x` has
shape `[samples, features]`; `y` is an array of class indices.

### `makeSpiral`

```js
makeSpiral({ samples = 300, noise = 0.12, seed = 42 } = {}) -> {
  x,
  y,
  classes
}
```

Generates the classic two-class interleaved spiral. `x` has shape
`[samples, 2]`, `y` contains `0` or `1`, and `classes` is `2`. Use an even
sample count so both classes have the same number of points.

### `makeXor`

```js
makeXor() -> { x, y, classes }
```

Returns the four-row XOR dataset: `x` has shape `[4, 2]`, `y` is
`[0, 1, 1, 0]`, and `classes` is `2`.

### `makeLinearRegression`

```js
makeLinearRegression({ samples = 100, seed = 5 } = {}) -> {
  x,
  yTensor,
  trueW,
  trueB
}
```

Generates noisy samples from `y = 2.5x - 1.2`. Features have shape
`[samples, 1]`, targets have shape `[samples, 1]`, and `trueW` and `trueB`
contain the source coefficients.
