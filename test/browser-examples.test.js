import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { describe, it } from "node:test";
import { fileURLToPath } from "node:url";
import {
  createXorRuntime,
  trainXor,
} from "../examples/browser/xor.js";
import {
  createRegressionRuntime,
  trainRegression,
} from "../examples/browser/linear-regression.js";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "..");

describe("浏览器示例", () => {
  it("XOR 页面使用的训练逻辑可以完全学会四组样本", () => {
    const runtime = createXorRuntime({ epochs: 400 });
    const result = trainXor(runtime);

    assert.equal(result.accuracy, 1, "XOR 示例应达到 100% 准确率");
    assert.ok(result.loss < 0.1, `XOR 损失应低于 0.1，实得 ${result.loss.toFixed(4)}`);
  });

  it("线性回归页面使用的训练逻辑可以逼近真实参数", () => {
    const runtime = createRegressionRuntime({ epochs: 400 });
    const result = trainRegression(runtime);

    assert.ok(
      Math.abs(result.weight - result.trueWeight) < 0.15,
      `学习到的权重 ${result.weight.toFixed(3)} 应接近 ${result.trueWeight}`,
    );
    assert.ok(
      Math.abs(result.bias - result.trueBias) < 0.15,
      `学习到的偏置 ${result.bias.toFixed(3)} 应接近 ${result.trueBias}`,
    );
  });

  it("两个页面都直接加载仓库源码且不依赖 CDN", () => {
    const pages = ["xor.html", "linear-regression.html"];

    for (const page of pages) {
      const html = readFileSync(resolve(ROOT, "examples/browser", page), "utf-8");
      assert.match(html, /<script type="module" src="\.\/[^"]+\.js"><\/script>/);
      assert.match(html, /<link rel="stylesheet" href="\.\/styles\.css">/);
      assert.doesNotMatch(html, /https?:\/\/[^"']+\.(?:js|css)/);
    }
  });
});
