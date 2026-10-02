/**
 * 梯度正确性测试 —— 框架可信度的基础。
 * 每个算子都用有限差分法验证解析梯度，容差 1e-5 相对误差。
 */
import assert from "node:assert/strict";
import { describe, it } from "node:test";
import {
  Tensor, add, mul, scale, sub, div, matmul, relu, tanh, sigmoid,
  softmax, sum, mean, reshape, transpose, gatherRows, concat,
  backward, checkGradient,
} from "../src/index.js";

/** 构造可复现的伪随机数，避免测试间相互影响 */
function seeded(seed = 42) {
  let s = seed >>> 0;
  return () => {
    s = (s * 1664525 + 1013904223) >>> 0;
    return s / 4294967296 - 0.5;
  };
}

/**
 * 断言解析梯度与数值梯度一致。
 *
 * 必须先清空梯度：上一次 backward 的累加结果还留在 .grad 里，
 * 而数值梯度是按当前参数算的，两者混在一起必然对不上。
 */
function assertGradientClose(lossFn, inputs, label, tol = 1e-5) {
  for (const t of inputs) t.zeroGrad();
  backward(lossFn());
  const r = checkGradient(lossFn, inputs, { tol });
  assert.ok(
    r.passed,
    `${label} 梯度不匹配: ${r.report} ` +
    `(梯度是否已回传: ${inputs.every((t) => t.grad !== null)})`
  );
}

describe("Tensor 基础", () => {
  it("形状与元素数一致", () => {
    const t = Tensor.tensor([1, 2, 3, 4, 5, 6], [2, 3]);
    assert.equal(t.size, 6);
    assert.equal(t.ndim, 2);
  });

  it("元素数不符时立即报错", () => {
    assert.throws(() => Tensor.tensor([1, 2, 3], [2, 2]), /形状不匹配/);
  });

  it("sum 求所有元素和", () => {
    const t = Tensor.tensor([1, 2, 3], [3]);
    assert.equal(t.sum().data[0], 6);
  });

  it("梯度累加而非覆盖", () => {
    const t = Tensor.variable([2], [1]);
    // 同一个叶子被用到两次，梯度应相加
    backward(mul(t, t).sum());
    assert.equal(t.grad[0], 4);
  });
});

describe("逐元素算子梯度", () => {
  it("add", () => {
    const a = Tensor.variable([1, 2, 3], [3]);
    const b = Tensor.variable([4, 5, 6], [3]);
    backward(add(a, b).sum());
    assertGradientClose(() => add(a, b).sum(), [a, b], "add");
  });

  it("add 支持标量广播", () => {
    const a = Tensor.variable([1, 2, 3, 4], [2, 2]);
    const s = Tensor.tensor([10], [1]);
    const out = add(a, s);
    assert.deepEqual(out.shape, [2, 2]);
    assert.deepEqual(Array.from(out.data), [11, 12, 13, 14]);
    backward(out.sum());
    assertGradientClose(() => add(a, s).sum(), [a], "add 广播");
  });

  it("sub", () => {
    const a = Tensor.variable([1, 2, 3], [3]);
    const b = Tensor.variable([4, 5, 6], [3]);
    backward(sub(a, b).sum());
    assertGradientClose(() => sub(a, b).sum(), [a, b], "sub");
  });

  it("mul", () => {
    const a = Tensor.variable([1, 2, 3], [3]);
    const b = Tensor.variable([4, 5, 6], [3]);
    backward(mul(a, b).sum());
    assertGradientClose(() => mul(a, b).sum(), [a, b], "mul");
  });

  it("div", () => {
    const a = Tensor.variable([1, 2, 3], [3]);
    const b = Tensor.variable([4, 5, 7], [3]);
    backward(div(a, b).sum());
    assertGradientClose(() => div(a, b).sum(), [a, b], "div");
  });

  it("scale", () => {
    const a = Tensor.variable([1, 2, 3], [3]);
    backward(scale(a, 3.5).sum());
    assertGradientClose(() => scale(a, 3.5).sum(), [a], "scale");
  });

  it("mul 形状不匹配时报错", () => {
    assert.throws(() => mul(Tensor.tensor([1, 2], [2]), Tensor.tensor([1], [1])), /形状必须一致/);
  });
});

describe("激活函数梯度", () => {
  for (const [name, fn] of [["relu", relu], ["tanh", tanh], ["sigmoid", sigmoid]]) {
    it(name, () => {
      const x = Tensor.variable([0.5, -0.3, 1.2, -2.0], [2, 2]);
      backward(fn(x).sum());
      assertGradientClose(() => fn(x).sum(), [x], name);
    });
  }

  it("relu 在负半轴梯度为 0", () => {
    const x = Tensor.variable([-1, 2], [2]);
    backward(relu(x).sum());
    assert.equal(x.grad[0], 0);
    assert.equal(x.grad[1], 1);
  });

  it("tanh 输出落在 (-1,1)", () => {
    const x = Tensor.tensor([-10, 0, 10], [3]);
    for (const v of tanh(x).data) assert.ok(v > -1 && v < 1);
  });
});

describe("softmax", () => {
  it("每行和为 1", () => {
    const x = Tensor.tensor([1, 2, 3, 1, 1, 1], [2, 3]);
    const out = softmax(x);
    assert.ok(Math.abs(out.data[0] + out.data[1] + out.data[2] - 1) < 1e-12);
    assert.ok(Math.abs(out.data[3] + out.data[4] + out.data[5] - 1) < 1e-12);
  });

  it("数值稳定：极大输入不溢出为 NaN", () => {
    const x = Tensor.tensor([1000, 1001, 1002], [1, 3]);
    for (const v of softmax(x).data) assert.ok(Number.isFinite(v));
  });

  it("梯度校验", () => {
    const x = Tensor.variable([1, 2, 3, 0.5, -1, 2], [2, 3]);
    const loss = () => mul(softmax(x), Tensor.tensor([0.3, 0.7, 1.0, 1.0, 1.0, 1.0], [2, 3])).sum();
    backward(loss());
    assertGradientClose(loss, [x], "softmax");
  });
describe("matmul 梯度", () => {
  it("2D 前向结果正确", () => {
    const a = Tensor.tensor([1, 2, 3, 4], [2, 2]);
    const b = Tensor.tensor([1, 0, 0, 1], [2, 2]);
    const out = matmul(a, b);
    assert.deepEqual(out.shape, [2, 2]);
    assert.deepEqual(Array.from(out.data), [1, 2, 3, 4]);
  });

  it("2D 梯度校验", () => {
    // 用确定性的有理数，避免伪随机数经 matmul 后量级放大、
    // 让有限差分的舍入误差变得可见
    const a = Tensor.variable([1, 2, 3, 0.5, -1, 2], [2, 3]);
    const b = Tensor.variable([0.5, -1, 2, 3, 1, -2], [3, 2]);
    const loss = () => mul(matmul(a, b), Tensor.tensor([1, 2, 3, 4], [2, 2])).sum();
    assertGradientClose(loss, [a, b], "matmul 2D");
  });

  it("批次 3D 梯度校验", () => {
    const r = seeded(11);
    const a = Tensor.variable(new Float64Array(12).map(() => r()), [2, 2, 3]);
    const b = Tensor.variable(new Float64Array(12).map(() => r()), [2, 3, 2]);
    const loss = () => matmul(a, b).sum();
    backward(loss());
    assertGradientClose(loss, [a, b], "matmul 3D");
  });

  it("内维不匹配时报错", () => {
    // a 是 [1,3]，b 是 [2,1]，内维 3≠2
    assert.throws(
      () => matmul(Tensor.tensor([1, 2, 3], [1, 3]), Tensor.tensor([1, 2], [2, 1])),
      /内维不匹配/
    );
  });

  it("批次维度必须相等", () => {
    assert.throws(
      () => matmul(Tensor.tensor([1], [1, 1, 1]), Tensor.tensor([1], [2, 1, 1])),
      /形状不匹配/
    );
  });
});

describe("形状算子梯度", () => {
  it("reshape 不改变元素顺序", () => {
    const x = Tensor.variable([1, 2, 3, 4, 5, 6], [2, 3]);
    const out = reshape(x, [3, 2]);
    assert.deepEqual(Array.from(out.data), [1, 2, 3, 4, 5, 6]);
    backward(out.sum());
    assertGradientClose(() => reshape(x, [3, 2]).sum(), [x], "reshape");
  });

  it("transpose 梯度校验", () => {
    const x = Tensor.variable([1, 2, 3, 4, 5, 6], [2, 3]);
    const loss = () => mul(transpose(x), Tensor.tensor([1, 1, 1, 2, 2, 2], [3, 2])).sum();
    backward(loss());
    assertGradientClose(loss, [x], "transpose");
  });

  it("sum 沿维度求和", () => {
    const x = Tensor.tensor([1, 2, 3, 4, 5, 6], [2, 3]);
    const out = sum(x, 1);
    assert.deepEqual(out.shape, [2]);
    assert.deepEqual(Array.from(out.data), [6, 15]);
  });

  it("sum 全局求和返回标量", () => {
    const x = Tensor.tensor([1, 2, 3], [3]);
    assert.equal(sum(x).data[0], 6);
  });

  it("sum 指定维度梯度校验", () => {
    const r = seeded(3);
    const x = Tensor.variable(new Float64Array(6).map(() => r()), [2, 3]);
    const loss = () => scale(sum(x, 1), 2.5).sum();
    backward(loss());
    assertGradientClose(loss, [x], "sum_dim");
  });

  it("mean 等于 sum 除以维度长度", () => {
    const x = Tensor.tensor([1, 2, 3, 4, 5, 6], [2, 3]);
    const s = sum(x, 1).data;
    const m = mean(x, 1).data;
    assert.equal(m[0], s[0] / 3);
    assert.equal(m[1], s[1] / 3);
  });

  it("gatherRows 梯度只回传到被选中行", () => {
    const x = Tensor.variable([1, 2, 3, 4, 5, 6], [3, 2]);
    backward(gatherRows(x, [0, 2]).sum());
    assert.deepEqual(Array.from(x.grad), [1, 1, 0, 0, 1, 1]);
    assertGradientClose(() => gatherRows(x, [0, 2]).sum(), [x], "gatherRows");
  });

  it("gatherRows 索引越界时报错", () => {
    assert.throws(() => gatherRows(Tensor.tensor([1], [1, 1]), [5]), /索引越界/);
  });

  it("concat 沿首维拼接且梯度正确回传", () => {
    const a = Tensor.variable([1, 2, 3, 4], [2, 2]);
    const b = Tensor.variable([5, 6, 7, 8], [2, 2]);
    const out = concat([a, b], 0);
    assert.equal(out.shape[0], 4);
    backward(out.sum());
    assertGradientClose(() => concat([a, b], 0).sum(), [a, b], "concat");
  });

  it("concat 形状不兼容时报错", () => {
    assert.throws(
      () => concat([Tensor.tensor([1, 2], [1, 2]), Tensor.tensor([1, 2, 3], [1, 3])], 0),
      /形状需一致/
    );
  });
});
});