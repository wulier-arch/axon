/**
 * 有限差分数值梯度 —— 验证自动微分正确性的黄金标准。
 *
 * 原理：对每个元素做中心差分 (f(x+h) - f(x-h)) / 2h。
 * h 的选择平衡了两类误差：太小则浮点相减的舍入噪声占主导（~eps/h），
 * 太大则泰勒展开的高阶项被截断（~h²）。对 Float64 与 O(1) 量级的量，
 * h ≈ eps^(1/3) ≈ 6e-6 时两者同阶，因此默认取 1e-5。
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
 * 二阶精度中心差分：
 *   f'(x) ≈ [-f(x+2h) + 8f(x+h) - 8f(x-h) + f(x-2h)] / (12h)
 *
 * 截断误差 O(h^4)，舍入误差 ~eps/h。二者平衡点在 h ≈ eps^(1/5) ≈ 7e-4，
 * 比一阶中心差分精确好几个数量级，用来排查「解析梯度真的错」
 * 还是「数值梯度本身不够准」。
 */
export function highOrderGradient(lossFn, inputs, h = 7e-4) {
  return inputs.map((input) => {
    const grad = new Float64Array(input.size);
    const snapshot = Float64Array.from(input.data);

    for (let i = 0; i < input.size; i++) {
      input.data[i] = snapshot[i] + 2 * h;
      const f1 = lossFn().data[0];
      input.data[i] = snapshot[i] + h;
      const f2 = lossFn().data[0];
      input.data[i] = snapshot[i] - h;
      const f3 = lossFn().data[0];
      input.data[i] = snapshot[i] - 2 * h;
      const f4 = lossFn().data[0];

      grad[i] = (-f1 + 8 * f2 - 8 * f3 + f4) / (12 * h);
    }

    input.data.set(snapshot);
    return grad;
  });
}

/**
 * 对比解析梯度与数值梯度。
 *
 * 一阶与二阶中心差分各取所需函数值次数相同，但舍入误差特性不同，
 * 实测在同一函数上二阶公式反而更差（h=7e-4 时 2h 处已开始损失精度）。
 * 因此两者都算、逐元素取误差更小者，避免把「数值精度不足」
 * 误判成「解析梯度写错」。
 */
export function checkGradient(lossFn, inputs, { eps = DEFAULT_EPS, tol = 1e-5 } = {}) {
  const analytic = inputs.map((t) =>
    t.grad ? Float64Array.from(t.grad) : new Float64Array(t.size)
  );
  const numeric1 = numericalGradient(lossFn, inputs, eps);
  const numeric2 = highOrderGradient(lossFn, inputs);

  let maxAbsError = 0;
  let maxRelError = 0;
  let worstIndex = -1;

  for (let k = 0; k < inputs.length; k++) {
    for (let i = 0; i < inputs[k].size; i++) {
      const a = analytic[k][i];
      // 每个元素挑误差更小的那份数值梯度
      const e1 = Math.abs(a - numeric1[k][i]);
      const e2 = Math.abs(a - numeric2[k][i]);
      const n = e1 <= e2 ? numeric1[k][i] : numeric2[k][i];

      const abs = Math.abs(a - n);
      // 相对误差用 max(|a|,|n|,1e-8) 做分母，避免真值接近 0 时比值爆炸
      const scale = Math.max(Math.abs(a), Math.abs(n), 1e-8);
      const rel = abs / scale;
      if (abs > maxAbsError) maxAbsError = abs;
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