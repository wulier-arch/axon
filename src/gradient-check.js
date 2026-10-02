/**
 * 有限差分数值梯度 —— 验证自动微分正确性的黄金标准。
 *
 * 原理：对每个元素做中心差分 (f(x+h) - f(x-h)) / 2h。
 * h 取 eps^(1/3) 量级最平衡：太小会被浮点噪声淹没，太大截断误差占主导。
 * 用中心差分而非前向差分，误差是 O(h^2) 而非 O(h)。
 */

const DEFAULT_EPS = 1e-5;

/**
 * 计算 lossFn 在 inputs 各点上的数值梯度。
 * @param {() => Tensor} lossFn 返回标量张量的函数，内部自行建图
 * @param {Tensor[]} inputs 需要求梯度的叶子张量
 * @param {number} eps 扰动步长
 * @returns {Float64Array[]} 每个输入对应的数值梯度
 */
export function numericalGradient(lossFn, inputs, eps = DEFAULT_EPS) {
  return inputs.map((input) => {
    const grad = new Float64Array(input.size);
    const snapshot = Float64Array.from(input.data);

    for (let i = 0; i < input.size; i++) {
      // 中心差分：+eps 与 -eps 各算一次
      input.data[i] = snapshot[i] + eps;
      const lossPlus = lossFn().data[0];

      input.data[i] = snapshot[i] - eps;
      const lossMinus = lossFn().data[0];

      grad[i] = (lossPlus - lossMinus) / (2 * eps);
    }

    input.data.set(snapshot);
    return grad;
  });
}

/**
 * 对比解析梯度与数值梯度。
 * @returns {{ maxAbsError:number, maxRelError:number, passed:boolean, report:string }}
 */
export function checkGradient(lossFn, inputs, { eps = DEFAULT_EPS, tol = 1e-5 } = {}) {
  const analytic = inputs.map((t) => (t.grad ? Float64Array.from(t.grad) : new Float64Array(t.size)));
  const numeric = numericalGradient(lossFn, inputs, eps);

  let maxAbsError = 0;
  let maxRelError = 0;
  let worstIndex = -1;

  for (let k = 0; k < inputs.length; k++) {
    for (let i = 0; i < inputs[k].size; i++) {
      const a = analytic[k][i];
      const n = numeric[k][i];
      const abs = Math.abs(a - n);
      // 相对误差用 max(|a|,|n|,1e-8) 做分母，避免真值接近 0 时比值爆炸
      const scale = Math.max(Math.abs(a), Math.abs(n), 1e-8);
      const rel = abs / scale;
      if (abs > maxAbsError) { maxAbsError = abs; }
      if (rel > maxRelError && abs > 1e-10) { maxRelError = rel; worstIndex = i; }
    }
  }

  const passed = maxRelError < tol || maxAbsError < 1e-8;
  return {
    maxAbsError,
    maxRelError,
    passed,
    report:
      `maxAbsError=${maxAbsError.toExponential(3)} ` +
      `maxRelError=${maxRelError.toExponential(3)}` +
      (worstIndex >= 0 ? ` @index=${worstIndex}` : ""),
  };
}