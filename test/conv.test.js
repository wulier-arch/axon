import assert from "node:assert/strict";
import { describe, it } from "node:test";
import { Tensor, backward, checkGradient, conv2d, mul } from "../src/index.js";

function assertGradientClose(lossFn, inputs, label, tol = 1e-5) {
  for (const tensor of inputs) tensor.zeroGrad();
  backward(lossFn());
  const result = checkGradient(lossFn, inputs, { tol });
  assert.ok(result.passed, `${label}: ${result.report}`);
}

describe("conv2d batch support", () => {
  it("keeps the single-image result unchanged", () => {
    const image = Tensor.tensor(
      [1, 2, 3, 4, 5, 6, 7, 8, 9],
      [1, 3, 3]
    );
    const kernel = Tensor.tensor([1, 0, 0, -1], [1, 1, 2, 2]);
    const out = conv2d(image, kernel);

    assert.deepEqual(out.shape, [2, 2, 1]);
    assert.deepEqual(Array.from(out.data), [-4, -4, -4, -4]);
  });

  it("processes a batch as independent images", () => {
    const images = Tensor.tensor([
      1, 2, 3, 4, 5, 6, 7, 8, 9,
      9, 8, 7, 6, 5, 4, 3, 2, 1,
    ], [2, 1, 3, 3]);
    const kernel = Tensor.tensor([1, 0, 0, -1], [1, 1, 2, 2]);
    const bias = Tensor.tensor([0.5], [1]);
    const out = conv2d(images, kernel, bias);

    const first = conv2d(
      Tensor.tensor([1, 2, 3, 4, 5, 6, 7, 8, 9], [1, 3, 3]),
      kernel,
      bias
    );
    const second = conv2d(
      Tensor.tensor([9, 8, 7, 6, 5, 4, 3, 2, 1], [1, 3, 3]),
      kernel,
      bias
    );

    assert.deepEqual(out.shape, [2, 2, 2, 1]);
    assert.deepEqual(Array.from(out.data), [
      ...first.data,
      ...second.data,
    ]);
  });

  it("applies stride and padding to every image in the batch", () => {
    const images = Tensor.tensor([
      1, 2, 3, 4,
      5, 6, 7, 8,
      9, 10, 11, 12,
      13, 14, 15, 16,

      16, 15, 14, 13,
      12, 11, 10, 9,
      8, 7, 6, 5,
      4, 3, 2, 1,
    ], [2, 1, 4, 4]);
    const kernel = Tensor.tensor([1, 0, 0, -1], [1, 1, 2, 2]);
    const out = conv2d(images, kernel, null, 2, 1);

    for (let n = 0; n < 2; n++) {
      const sample = Tensor.tensor(
        Array.from(images.data.slice(n * 16, (n + 1) * 16)),
        [1, 4, 4]
      );
      const expected = conv2d(sample, kernel, null, 2, 1);
      assert.deepEqual(
        Array.from(out.data.slice(n * expected.size, (n + 1) * expected.size)),
        Array.from(expected.data)
      );
    }
    assert.deepEqual(out.shape, [2, 3, 3, 1]);
  });

  it("passes finite-difference checks for input, weight, and bias", () => {
    const x = Tensor.variable([
      0.2, -0.4, 0.6, 0.1,
      -0.3, 0.5, -0.2, 0.7,
      0.4, -0.1, 0.3, -0.5,

      -0.6, 0.2, 0.1, 0.4,
      0.3, -0.7, 0.5, -0.2,
      -0.1, 0.6, -0.4, 0.2,
    ], [2, 1, 3, 4]);
    const w = Tensor.variable([
      0.5, -0.3, 0.2, 0.4,
      -0.2, 0.6, -0.5, 0.1,
    ], [2, 1, 2, 2]);
    const b = Tensor.variable([0.25, -0.15], [2]);
    const upstream = Tensor.tensor([
      1, -0.5,
      0.25, 0.75,
      -0.3, 0.4,
      0.6, -0.2,
      0.1, 0.3,
      0.8, -0.1,

      -0.1, 0.8,
      0.5, -0.6,
      -0.4, 0.3,
      0.2, 0.7,
      -0.7, 0.9,
      0.4, -0.2,
    ], [2, 2, 3, 2]);

    const loss = () => mul(conv2d(x, w, b, 2, 1), upstream).sum();
    assertGradientClose(loss, [x, w, b], "conv2d batch");
  });

  it("builds a graph when only bias requires gradients", () => {
    const x = Tensor.tensor([
      1, 2, 3, 4,
      5, 6, 7, 8,
      9, 10, 11, 12,
    ], [2, 1, 2, 3]);
    const w = Tensor.tensor([1, 1, 1, 1], [1, 1, 2, 2]);
    const b = Tensor.variable([0.5], [1]);

    backward(conv2d(x, w, b).sum());

    assert.deepEqual(Array.from(b.grad), [4]);
  });

  it("rejects invalid input, channel, and bias shapes", () => {
    const image = Tensor.tensor([1, 2, 3, 4], [1, 2, 2]);
    const kernel = Tensor.tensor([1, 1, 1, 1], [1, 1, 2, 2]);

    assert.throws(
      () => conv2d(Tensor.tensor([1], [1, 1, 1, 1, 1]), kernel),
      /x 为/
    );
    assert.throws(
      () => conv2d(image, Tensor.tensor([1, 1], [1, 2, 1, 1])),
      /通道不匹配/
    );
    assert.throws(
      () => conv2d(image, kernel, Tensor.tensor([1, 2], [2])),
      /bias 应为/
    );
  });
});
