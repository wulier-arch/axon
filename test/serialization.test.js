import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  Dropout,
  Embedding,
  LayerNorm,
  Linear,
  MultiHeadAttention,
  Sequential,
  Tensor,
  TransformerBlock,
  loadModel,
} from "../src/index.js";

function assertNumbersEqual(actual, expected, message) {
  assert.equal(actual.length, expected.length, `${message}: 长度不一致`);
  for (let i = 0; i < actual.length; i++) {
    assert.equal(actual[i], expected[i], `${message}: 第 ${i} 项不一致`);
  }
}

function assertTensorEqual(actual, expected, message) {
  assert.deepEqual(actual.shape, expected.shape, `${message}: 形状不一致`);
  assertNumbersEqual(actual.data, expected.data, message);
}

function roundTrip(model) {
  return loadModel(JSON.stringify(model));
}

function sampleMatrix() {
  return Tensor.tensor([
    0.5, -1.0, 1.5, 0.25,
    -0.75, 0.8, -0.4, 1.2,
  ], [2, 4]);
}

describe("模型序列化", () => {
  it("Linear 往返后参数与输出完全一致", () => {
    const layer = new Linear(3, 4, {
      activation: "tanh",
      bias: true,
      name: "projection",
    });
    layer.weight.data.set([
      0.1, -0.2, 0.3, 0.4,
      -0.5, 0.6, -0.7, 0.8,
      0.9, -1.0, 1.1, -1.2,
    ]);
    layer.bias.data.set([0.01, -0.02, 0.03, -0.04]);
    const model = new Sequential([layer]);
    const x = Tensor.tensor([1, 2, 3, -4, 5, -6], [2, 3]);

    const loaded = roundTrip(model);

    assert.deepEqual(loaded.toJSON(), model.toJSON());
    assertTensorEqual(loaded.forward(x), model.forward(x), "Linear 输出");
  });

  it("LayerNorm 往返后保留 eps、缩放、偏置与输出", () => {
    const layer = new LayerNorm(4, { eps: 1e-4, name: "norm" });
    layer.gamma.data.set([1.1, 0.9, 1.3, 0.7]);
    layer.beta.data.set([0.1, -0.2, 0.3, -0.4]);
    const model = new Sequential([layer]);
    const x = sampleMatrix();

    const loaded = roundTrip(model);

    assert.deepEqual(loaded.toJSON(), model.toJSON());
    assertTensorEqual(loaded.forward(x), model.forward(x), "LayerNorm 输出");
  });

  it("Dropout 往返后从相同 RNG 状态继续采样", () => {
    const layer = new Dropout(0.4, 123456);
    const model = new Sequential([layer]);
    const x = Tensor.tensor([1, 2, 3, 4, 5, 6, 7, 8], [2, 4]);

    layer.forward(x);
    const loaded = roundTrip(model);
    const loadedLayer = loaded.layers[0];

    assert.equal(loadedLayer.rng.getState(), layer.rng.getState());
    assertTensorEqual(
      loadedLayer.forward(x),
      layer.forward(x),
      "Dropout 续跑输出"
    );
    assert.equal(loadedLayer.rng.getState(), layer.rng.getState());
  });

  it("Embedding 往返后完整保留查表权重", () => {
    const layer = new Embedding(5, 3, { seed: 17 });
    layer.weight.data.set([
      0.1, 0.2, 0.3,
      -0.4, 0.5, -0.6,
      0.7, -0.8, 0.9,
      1.0, -1.1, 1.2,
      -1.3, 1.4, -1.5,
    ]);
    const model = new Sequential([layer]);
    const ids = [4, 1, 1, 0];

    const loaded = roundTrip(model);
    const loadedLayer = loaded.layers[0];

    assert.deepEqual(loadedLayer.toJSON(), layer.toJSON());
    assertTensorEqual(
      loadedLayer.forward(ids),
      layer.forward(ids),
      "Embedding 输出"
    );
  });

  it("MultiHeadAttention 往返后四个投影矩阵与输出一致", () => {
    const layer = new MultiHeadAttention(4, 2, { causal: true, seed: 23 });
    const model = new Sequential([layer]);
    const x = sampleMatrix();

    const loaded = roundTrip(model);
    const loadedLayer = loaded.layers[0];

    assert.deepEqual(loadedLayer.toJSON(), layer.toJSON());
    assertTensorEqual(
      loadedLayer.forward(x),
      layer.forward(x),
      "MultiHeadAttention 输出"
    );
  });

  it("TransformerBlock 递归恢复全部子层权重与配置", () => {
    const layer = new TransformerBlock(4, 2, {
      dFF: 6,
      causal: true,
      normFirst: false,
      ffnActivation: "relu",
      seed: 31,
    });
    layer.norm1.gamma.data.set([1.1, 0.9, 1.3, 0.7]);
    layer.norm2.beta.data.set([0.2, -0.1, 0.4, -0.3]);
    const model = new Sequential([layer]);
    const x = sampleMatrix();

    const loaded = roundTrip(model);
    const loadedLayer = loaded.layers[0];

    assert.ok(loadedLayer instanceof TransformerBlock);
    assert.deepEqual(loadedLayer.toJSON(), layer.toJSON());
    assert.equal(loadedLayer.parameters().length, layer.parameters().length);
    for (let i = 0; i < layer.parameters().length; i++) {
      assertNumbersEqual(
        loadedLayer.parameters()[i].data,
        layer.parameters()[i].data,
        `TransformerBlock 参数 ${i}`
      );
    }
    assertTensorEqual(loadedLayer.forward(x), layer.forward(x), "TransformerBlock 输出");
  });

  it("嵌套 Sequential 可从 JSON 字符串完整恢复", () => {
    const nested = new Sequential([
      new Linear(3, 2, { activation: "relu", name: "nested-hidden" }),
    ]);
    const model = new Sequential([
      new Linear(2, 3, { name: "input-projection" }),
      nested,
    ]);
    const x = Tensor.tensor([0.2, -0.4, 1.1, 0.5], [2, 2]);

    const loaded = roundTrip(model);

    assert.ok(loaded.layers[1] instanceof Sequential);
    assert.deepEqual(loaded.toJSON(), model.toJSON());
    assertTensorEqual(loaded.forward(x), model.forward(x), "嵌套模型输出");
  });

  it("拒绝错误格式、未知层、缺失字段、非法权重与激活", () => {
    const valid = new Sequential([
      new Linear(2, 2, { activation: "relu" }),
    ]).toJSON();

    assert.throws(
      () => loadModel({ ...valid, format: "other-format" }),
      /format must be "axon-model"/
    );
    assert.throws(
      () => loadModel({ ...valid, version: 2 }),
      /version is unsupported/
    );
    assert.throws(
      () => loadModel({ ...valid, layers: [{ type: "Mystery" }] }),
      /unknown layer type "Mystery"/
    );
    assert.throws(
      () => loadModel({ ...valid, layers: [{ type: "Linear" }] }),
      /missing required field "inFeatures"/
    );
    assert.throws(
      () => loadModel({
        ...valid,
        layers: [{ ...valid.layers[0], activation: "softmax" }],
      }),
      /activation has unsupported value "softmax"/
    );
    assert.throws(
      () => loadModel({
        ...valid,
        layers: [{ ...valid.layers[0], weight: [1] }],
      }),
      /weight has length 1; expected 4/
    );
    assert.throws(
      () => loadModel({
        ...valid,
        layers: [{ ...valid.layers[0], weight: [1, 2, NaN, 4] }],
      }),
      /weight\[2\] must be a finite number/
    );
    assert.throws(() => loadModel("not-json"), /input is not valid JSON/);
  });

  it("载荷里的 __proto__ 字段不会污染原型", () => {
    const payload = JSON.parse(
      '{"format":"axon-model","version":1,"type":"Sequential","layers":[' +
        '{"type":"Linear","inFeatures":1,"outFeatures":1,"activation":null,' +
        '"bias":null,"weight":[1],' +
        '"__proto__":{"polluted":true,"type":"Mystery"}}]}'
    );

    const loaded = loadModel(payload);

    assert.ok(loaded instanceof Sequential);
    assert.equal({}.polluted, undefined);
    assert.equal(Object.prototype.polluted, undefined);
    assert.ok(loaded.layers[0] instanceof Linear);
  });

  it("Sequential 遇到无法序列化的层时明确报错", () => {
    const model = new Sequential([{ name: "custom" }]);

    assert.throws(
      () => model.toJSON(),
      /layer 0 \(custom\) does not implement toJSON\(\)/
    );
  });
});
