/**
 * Model serialization and loading.
 *
 * The public payload is deliberately explicit: a format marker, a version,
 * and the complete recursive layer tree. Loading never evaluates code and
 * validates every field before constructing tensors.
 */

import { Tensor } from "../tensor/tensor.js";
import {
  Dropout,
  Embedding,
  LayerNorm,
  Linear,
  MultiHeadAttention,
  TransformerBlock,
} from "./layers.js";
import { Sequential } from "./sequential.js";

const MODEL_FORMAT = "axon-model";
const MODEL_VERSION = 1;
const LINEAR_ACTIVATIONS = new Set(["relu", "tanh", "sigmoid"]);
const FFN_ACTIVATIONS = new Set(["gelu", "relu", "tanh"]);

function fail(path, message) {
  throw new TypeError(`axon loadModel: ${path} ${message}`);
}

function requireObject(value, path) {
  if (value === null || typeof value !== "object" || Array.isArray(value)) {
    fail(path, "must be an object");
  }
  return value;
}

function requirePositiveInteger(value, path) {
  if (!Number.isSafeInteger(value) || value <= 0) {
    fail(path, "must be a positive integer");
  }
  return value;
}

function requireBoolean(value, path) {
  if (typeof value !== "boolean") fail(path, "must be a boolean");
  return value;
}

function requireFiniteNumber(value, path) {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    fail(path, "must be a finite number");
  }
  return value;
}

function requireArrayOfNumbers(value, length, path) {
  if (!Array.isArray(value)) fail(path, "must be an array");
  if (value.length !== length) {
    fail(path, `has length ${value.length}; expected ${length}`);
  }
  for (let i = 0; i < value.length; i++) {
    if (typeof value[i] !== "number" || !Number.isFinite(value[i])) {
      fail(`${path}[${i}]`, "must be a finite number");
    }
  }
  return Float64Array.from(value);
}

function requireOwn(object, key, path) {
  if (!Object.prototype.hasOwnProperty.call(object, key)) {
    fail(path, `is missing required field "${key}"`);
  }
  return object[key];
}

function applyName(layer, json, path) {
  if (json.name === undefined) return layer;
  if (typeof json.name !== "string" || json.name.length === 0) {
    fail(`${path}.name`, "must be a non-empty string when present");
  }
  layer.name = json.name;
  return layer;
}

function loadLinear(json, path) {
  const inFeatures = requirePositiveInteger(
    requireOwn(json, "inFeatures", path),
    `${path}.inFeatures`
  );
  const outFeatures = requirePositiveInteger(
    requireOwn(json, "outFeatures", path),
    `${path}.outFeatures`
  );
  const activation = json.activation ?? null;
  if (activation !== null && !LINEAR_ACTIVATIONS.has(activation)) {
    fail(`${path}.activation`, `has unsupported value "${activation}"`);
  }

  const biasJson = requireOwn(json, "bias", path);
  if (biasJson !== null && !Array.isArray(biasJson)) {
    fail(`${path}.bias`, "must be an array or null");
  }

  const layer = new Linear(inFeatures, outFeatures, {
    bias: biasJson !== null,
    activation,
  });
  applyName(layer, json, path);

  layer.weight = Tensor.variable(
    requireArrayOfNumbers(
      requireOwn(json, "weight", path),
      inFeatures * outFeatures,
      `${path}.weight`
    ),
    [inFeatures, outFeatures]
  );
  if (biasJson !== null) {
    layer.bias = Tensor.variable(
      requireArrayOfNumbers(biasJson, outFeatures, `${path}.bias`),
      [outFeatures]
    );
  } else {
    layer.bias = null;
  }
  return layer;
}

function loadDropout(json, path) {
  const p = requireFiniteNumber(requireOwn(json, "p", path), `${path}.p`);
  if (p < 0 || p > 1) fail(`${path}.p`, "must be between 0 and 1");

  const layer = new Dropout(p);
  applyName(layer, json, path);
  const rngState = requirePositiveInteger(
    requireOwn(json, "rngState", path),
    `${path}.rngState`
  );
  if (rngState > 0xffffffff) {
    fail(`${path}.rngState`, "must fit in an unsigned 32-bit integer");
  }
  layer.rng.setState(rngState);
  return layer;
}

function loadLayerNorm(json, path) {
  const features = requirePositiveInteger(
    requireOwn(json, "features", path),
    `${path}.features`
  );
  const eps = requireFiniteNumber(requireOwn(json, "eps", path), `${path}.eps`);
  if (eps <= 0) fail(`${path}.eps`, "must be greater than zero");

  const gammaJson = requireOwn(json, "gamma", path);
  const betaJson = requireOwn(json, "beta", path);
  if (gammaJson !== null && !Array.isArray(gammaJson)) {
    fail(`${path}.gamma`, "must be an array or null");
  }
  if (betaJson !== null && !Array.isArray(betaJson)) {
    fail(`${path}.beta`, "must be an array or null");
  }

  const layer = new LayerNorm(features, {
    eps,
    gamma: gammaJson !== null,
    beta: betaJson !== null,
  });
  applyName(layer, json, path);

  layer.gamma = gammaJson === null
    ? null
    : Tensor.variable(
      requireArrayOfNumbers(gammaJson, features, `${path}.gamma`),
      [features]
    );
  layer.beta = betaJson === null
    ? null
    : Tensor.variable(
      requireArrayOfNumbers(betaJson, features, `${path}.beta`),
      [features]
    );
  return layer;
}

function loadEmbedding(json, path) {
  const vocabSize = requirePositiveInteger(
    requireOwn(json, "vocabSize", path),
    `${path}.vocabSize`
  );
  const features = requirePositiveInteger(
    requireOwn(json, "features", path),
    `${path}.features`
  );

  const layer = new Embedding(vocabSize, features);
  applyName(layer, json, path);
  layer.weight = Tensor.variable(
    requireArrayOfNumbers(
      requireOwn(json, "weight", path),
      vocabSize * features,
      `${path}.weight`
    ),
    [vocabSize, features]
  );
  return layer;
}

function loadMultiHeadAttention(json, path) {
  const dModel = requirePositiveInteger(
    requireOwn(json, "dModel", path),
    `${path}.dModel`
  );
  const heads = requirePositiveInteger(
    requireOwn(json, "heads", path),
    `${path}.heads`
  );
  const causal = requireBoolean(
    requireOwn(json, "causal", path),
    `${path}.causal`
  );

  const layer = new MultiHeadAttention(dModel, heads, { causal });
  applyName(layer, json, path);
  const weightLength = dModel * dModel;
  layer.wq = Tensor.variable(
    requireArrayOfNumbers(requireOwn(json, "wq", path), weightLength, `${path}.wq`),
    [dModel, dModel]
  );
  layer.wk = Tensor.variable(
    requireArrayOfNumbers(requireOwn(json, "wk", path), weightLength, `${path}.wk`),
    [dModel, dModel]
  );
  layer.wv = Tensor.variable(
    requireArrayOfNumbers(requireOwn(json, "wv", path), weightLength, `${path}.wv`),
    [dModel, dModel]
  );
  layer.wo = Tensor.variable(
    requireArrayOfNumbers(requireOwn(json, "wo", path), weightLength, `${path}.wo`),
    [dModel, dModel]
  );
  return layer;
}

function loadTransformerBlock(json, path) {
  const dModel = requirePositiveInteger(
    requireOwn(json, "dModel", path),
    `${path}.dModel`
  );
  const heads = requirePositiveInteger(
    requireOwn(json, "heads", path),
    `${path}.heads`
  );
  const dFF = requirePositiveInteger(
    requireOwn(json, "dFF", path),
    `${path}.dFF`
  );
  const normFirst = requireBoolean(
    requireOwn(json, "normFirst", path),
    `${path}.normFirst`
  );
  const causal = requireBoolean(
    requireOwn(json, "causal", path),
    `${path}.causal`
  );
  const activation = requireOwn(json, "activation", path);
  if (!FFN_ACTIVATIONS.has(activation)) {
    fail(`${path}.activation`, `has unsupported value "${activation}"`);
  }

  const norm1 = loadLayer(requireOwn(json, "norm1", path), `${path}.norm1`);
  const attn = loadLayer(requireOwn(json, "attn", path), `${path}.attn`);
  const norm2 = loadLayer(requireOwn(json, "norm2", path), `${path}.norm2`);
  const ff1 = loadLayer(requireOwn(json, "ff1", path), `${path}.ff1`);
  const ff2 = loadLayer(requireOwn(json, "ff2", path), `${path}.ff2`);

  if (!(norm1 instanceof LayerNorm) || norm1.features !== dModel) {
    fail(`${path}.norm1`, `must be a LayerNorm with features ${dModel}`);
  }
  if (!(norm2 instanceof LayerNorm) || norm2.features !== dModel) {
    fail(`${path}.norm2`, `must be a LayerNorm with features ${dModel}`);
  }
  if (
    !(attn instanceof MultiHeadAttention)
    || attn.dModel !== dModel
    || attn.heads !== heads
    || attn.causal !== causal
  ) {
    fail(
      `${path}.attn`,
      `must be causal=${causal} MultiHeadAttention(dModel=${dModel}, heads=${heads})`
    );
  }
  if (
    !(ff1 instanceof Linear)
    || ff1.inFeatures !== dModel
    || ff1.outFeatures !== dFF
  ) {
    fail(
      `${path}.ff1`,
      `must be Linear(${dModel}, ${dFF})`
    );
  }
  if (
    !(ff2 instanceof Linear)
    || ff2.inFeatures !== dFF
    || ff2.outFeatures !== dModel
  ) {
    fail(
      `${path}.ff2`,
      `must be Linear(${dFF}, ${dModel})`
    );
  }

  const block = new TransformerBlock(dModel, heads, {
    dFF,
    causal,
    normFirst,
    ffnActivation: activation,
  });
  block.norm1 = norm1;
  block.attn = attn;
  block.norm2 = norm2;
  block.ff1 = ff1;
  block.ff2 = ff2;
  block.activation = activation;
  applyName(block, json, path);
  return block;
}

function loadSequential(json, path = "") {
  const prefix = path ? `${path}.` : "";
  if (json.format !== MODEL_FORMAT) {
    fail(`${prefix}format`, `must be "${MODEL_FORMAT}"`);
  }
  if (json.version !== MODEL_VERSION) {
    fail(
      `${prefix}version`,
      `is unsupported (${JSON.stringify(json.version)}); expected ${MODEL_VERSION}`
    );
  }
  if (json.type !== "Sequential") {
    fail(`${prefix}type`, 'must be "Sequential"');
  }
  if (!Array.isArray(json.layers)) {
    fail(`${prefix}layers`, "must be an array");
  }

  const layers = Array.from(json.layers, (layer, index) =>
    loadLayer(layer, `${prefix}layers[${index}]`)
  );
  return new Sequential(layers);
}

function loadLayer(value, path) {
  const json = requireObject(value, path);
  if (typeof json.type !== "string") {
    fail(`${path}.type`, "must be a string");
  }

  switch (json.type) {
    case "Linear":
      return loadLinear(json, path);
    case "Dropout":
      return loadDropout(json, path);
    case "LayerNorm":
      return loadLayerNorm(json, path);
    case "Embedding":
      return loadEmbedding(json, path);
    case "MultiHeadAttention":
      return loadMultiHeadAttention(json, path);
    case "TransformerBlock":
      return loadTransformerBlock(json, path);
    case "Sequential":
      return loadSequential(json, path);
    default:
      fail(`${path}.type`, `has unknown layer type "${json.type}"`);
  }
}

/**
 * Load a Sequential model from the JSON payload produced by model.toJSON()
 * or from its JSON string representation.
 *
 * @param {object|string} serialized
 * @returns {Sequential}
 */
export function loadModel(serialized) {
  let payload = serialized;
  if (typeof payload === "string") {
    try {
      payload = JSON.parse(payload);
    } catch {
      fail("input", "is not valid JSON");
    }
  }

  payload = requireObject(payload, "input");
  return loadSequential(payload);
}
