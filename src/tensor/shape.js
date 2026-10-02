/** 形状变换与采样算子（含各自的反向传播规则）。 */

import { Tensor, numElements } from "./tensor.js";

import { track } from "./ops.js";

/** 按行索引取出 [k,n] 的若干行，常用于交叉熵取真值 */
export function gatherRows(a, indices) {
  const [m, n] = a.shape;
  if (Math.max(...indices) >= m || Math.min(...indices) < 0) {
    throw new Error(`gatherRows: 索引越界 [0,${m})`);
  }
  const out = new Tensor(new Float64Array(indices.length * n), [indices.length, n]);
  for (let i = 0; i < indices.length; i++) {
    for (let j = 0; j < n; j++) out.data[i * n + j] = a.data[indices[i] * n + j];
  }
  if (a.requiresGrad) {
    out._prev = [a, indices];
    out._op = "gatherRows";
    out._backward = () => {
      const g = new Float64Array(a.size);
      for (let i = 0; i < indices.length; i++) {
        for (let j = 0; j < n; j++) g[indices[i] * n + j] += out.grad[i * n + j];
      }
      a.accumulateGrad(g);
    };
  }
  return out;
}

/** 沿 axis 拼接；仅支持 axis 为 0（沿特征维堆叠） */
export function concat(list, axis = 0) {
  if (list.length === 0) throw new Error("concat: 列表为空");
  if (axis !== 0) throw new Error(`concat: 暂仅支持 axis=0，得到 ${axis}`);

  const inner = list[0].shape.slice(1).reduce((x, y) => x * y, 1);
  for (const t of list) {
    const ti = t.shape.slice(1).reduce((x, y) => x * y, 1);
    if (ti !== inner) throw new Error(`concat: 除首维外形状需一致，[${t.shape}]`);
  }

  const rows = list.reduce((s, t) => s + t.shape[0], 0);
  const outShape = [rows, ...list[0].shape.slice(1)];
  const out = new Tensor(new Float64Array(rows * inner), outShape);

  let base = 0;
  for (const t of list) {
    const seg = t.shape[0];
    out.data.set(t.data.subarray(0, seg * inner), base * inner);
    base += seg;
  }

  if (list.some((t) => t.requiresGrad)) {
    out._prev = [list];
    out._op = "concat";
    out._backward = () => {
      let b2 = 0;
      for (const t of list) {
        if (t.requiresGrad) {
          const seg = t.shape[0];
          t.accumulateGrad(out.grad.slice(b2 * inner, (b2 + seg) * inner));
        }
        b2 += t.shape[0];
      }
    };
  }
  return out;
}

/** 最后一维的 argmax，返回下标数组 */
export function argmaxLast(a) {
  const last = a.shape[a.shape.length - 1];
  const rows = a.size / last;
  const out = new Array(rows);
  for (let r = 0; r < rows; r++) {
    let best = 0;
    for (let j = 1; j < last; j++) if (a.data[r * last + j] > a.data[r * last + best]) best = j;
    out[r] = best;
  }
  return out;
}