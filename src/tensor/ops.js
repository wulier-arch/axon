import { Tensor, numElements } from "./tensor.js";

/**
 * 判断这次运算是否需要建计算图。
 *
 * 除了叶子变量（requiresGrad），已经建过图的中间张量也要算进来。
 * 否则 sub(a,b) 这类由中间节点参与的组合运算会丢失反向链。
 */
export function track(...tensors) {
  return tensors.some((t) => t.isGraphNode());
}

/**
 * 广播索引映射：把输出的一维下标换算成输入的一维下标。
 * 采用 numpy 式右对齐规则，逐维判断是「整维复制」还是「一一对应」。
 */
function broadcastIndex(outShape, inShape, outFlat, inShapeLen) {
  const offset = outShape.length - inShapeLen;
  let inFlat = 0;
  let stride = 1;
  for (let d = inShapeLen - 1; d >= 0; d--) {
    const dim = inShape[d];
    const outDimIndex = Math.floor(outFlat / stride) % outShape[d + offset];
    if (dim !== 1) inFlat += outDimIndex * stride;
    stride *= dim;
  }
  return inFlat;
}

/** 加法，支持标量广播与同形状 */
/**
 * 计算两个形状的广播结果，不兼容时返回 null。
 * numpy 规则：从最后一维开始逐维比较，要么相等，要么其中一方为 1。
 */
function resolveBroadcast(shapeA, shapeB) {
  const rank = Math.max(shapeA.length, shapeB.length);
  const out = new Array(rank);

  for (let i = 0; i < rank; i++) {
    // 缺失的维度视为 1
    const a = shapeA[shapeA.length - rank + i] ?? 1;
    const b = shapeB[shapeB.length - rank + i] ?? 1;
    if (a === b) out[i] = a;
    else if (a === 1) out[i] = b;
    else if (b === 1) out[i] = a;
    else return null;
  }
  return out;
}

/**
 * 广播加法。
 *
 * 支持三类情形（按 numpy 右对齐规则）：
 *   1. 标量广播：[n] + [1]
 *   2. 按行广播：[m,n] + [n]  ← 全连接层加偏置的常规用法
 *   3. 同形状   ：[m,n] + [m,n]
 */
export function add(a, b) {
  const outShape = resolveBroadcast(a.shape, b.shape);
  if (!outShape) {
    throw new Error(`add: 形状不兼容 [${a.shape}] + [${b.shape}]`);
  }

  const out = new Tensor(new Float64Array(numElements(outShape)), outShape);
  const outSize = out.size;
  const aScalar = a.size === 1 && outSize !== 1;
  const bScalar = b.size === 1 && outSize !== 1;

  for (let i = 0; i < outSize; i++) {
    const ai = aScalar ? 0 : (a.size === outSize ? i : broadcastIndex(outShape, a.shape, i, a.shape.length));
    const bi = bScalar ? 0 : (b.size === outSize ? i : broadcastIndex(outShape, b.shape, i, b.shape.length));
    out.data[i] = a.data[ai] + b.data[bi];
  }

  if (track(a, b)) {
    out._prev = [a, b];
    out._op = "add";
    out._backward = () => {
      if (a.isGraphNode()) {
        const ga = new Float64Array(a.size);
        for (let i = 0; i < outSize; i++) {
          const ai = aScalar ? 0 : (a.size === outSize ? i : broadcastIndex(outShape, a.shape, i, a.shape.length));
          ga[ai] += out.grad[i];
        }
        a.accumulateGrad(ga);
      }
      if (b.isGraphNode()) {
        const gb = new Float64Array(b.size);
        for (let i = 0; i < outSize; i++) {
          const bi = bScalar ? 0 : (b.size === outSize ? i : broadcastIndex(outShape, b.shape, i, b.shape.length));
          gb[bi] += out.grad[i];
        }
        b.accumulateGrad(gb);
      }
    };
  }
  return out;
}

/** 逐元素乘法，形状须一致 */
export function mul(a, b) {
  if (a.size !== b.size) {
    throw new Error(`mul: 形状必须一致，[${a.shape}] * [${b.shape}]`);
  }
  const out = new Tensor(new Float64Array(a.size), a.shape);
  for (let i = 0; i < out.size; i++) out.data[i] = a.data[i] * b.data[i];

  if (track(a, b)) {
    out._prev = [a, b];
    out._op = "mul";
    out._backward = () => {
      if (a.isGraphNode()) {
        const g = new Float64Array(a.size);
        for (let i = 0; i < g.length; i++) g[i] = out.grad[i] * b.data[i];
        a.accumulateGrad(g);
      }
      if (b.isGraphNode()) {
        const g = new Float64Array(b.size);
        for (let i = 0; i < g.length; i++) g[i] = out.grad[i] * a.data[i];
        b.accumulateGrad(g);
      }
    };
  }
  return out;
}

/** 乘以常数，用于学习率缩放 */
export function scale(a, factor) {
  const out = new Tensor(new Float64Array(a.size), a.shape);
  for (let i = 0; i < out.size; i++) out.data[i] = a.data[i] * factor;
  if (a.isGraphNode()) {
    out._prev = [a];
    out._op = "scale";
    out._backward = () => {
      const g = new Float64Array(a.size);
      for (let i = 0; i < g.length; i++) g[i] = out.grad[i] * factor;
      a.accumulateGrad(g);
    };
  }
  return out;
}
/**
 * 矩阵乘：2D [m,k]@[k,n]->[m,n]，以及批次 3D [b,m,k]@[b,k,n]->[b,m,n]。
 * 前向采用 i-k-j 循环顺序，对行主序缓存友好，比朴素 i-j-k 快数倍。
 */
export function matmul(a, b) {
  if (a.ndim === 2 && b.ndim === 2) return matmul2d(a, b);
  if (a.ndim === 3 && b.ndim === 3) return matmul3d(a, b);
  throw new Error(`matmul: 仅支持 2D 或 3D，得到 ${a.ndim}D`);
}

function matmul2d(a, b) {
  const [m, k] = a.shape;
  const [k2, n] = b.shape;
  if (k !== k2) throw new Error(`matmul: 内维不匹配 [${a.shape}] @ [${b.shape}]`);

  const out = new Tensor(new Float64Array(m * n), [m, n]);
  for (let i = 0; i < m; i++) {
    const rowO = i * n, rowA = i * k;
    for (let p = 0; p < k; p++) {
      const av = a.data[rowA + p];
      if (av === 0) continue;
      for (let j = 0; j < n; j++) out.data[rowO + j] += av * b.data[p * n + j];
    }
  }

  if (track(a, b)) {
    out._prev = [a, b];
    out._op = "matmul";
    out._backward = () => {
      if (a.isGraphNode()) {
        const ga = new Float64Array(m * k);
        for (let i = 0; i < m; i++) {
          const rowO = i * n, rowA = i * k;
          for (let p = 0; p < k; p++) {
            let s = 0;
            for (let j = 0; j < n; j++) s += out.grad[rowO + j] * b.data[p * n + j];
            ga[rowA + p] = s;
          }
        }
        a.accumulateGrad(ga);
      }
      if (b.isGraphNode()) {
        const gb = new Float64Array(k * n);
        // dL/db[p][j] = Σ_i out.grad[i][j] · a[i][p]
        // 必须对所有输出列 j 累加：只取 j=0 会丢掉其余列的贡献
        for (let p = 0; p < k; p++) {
          const rowB = p * n;
          for (let i = 0; i < m; i++) {
            const av = a.data[i * k + p];
            if (av === 0) continue;
            const rowO = i * n;
            for (let j = 0; j < n; j++) gb[rowB + j] += out.grad[rowO + j] * av;
          }
        }
        b.accumulateGrad(gb);
      }
    };
  }
  return out;
}

function matmul3d(a, b) {
  const [B, m, k] = a.shape;
  const [B2, k2, n] = b.shape;
  if (B !== B2 || k !== k2) {
    throw new Error(`matmul(batch): 形状不匹配 [${a.shape}] @ [${b.shape}]`);
  }

  const out = new Tensor(new Float64Array(B * m * n), [B, m, n]);
  for (let bi = 0; bi < B; bi++) {
    const aOff = bi * m * k, bOff = bi * k * n, oOff = bi * m * n;
    for (let i = 0; i < m; i++) {
      for (let p = 0; p < k; p++) {
        const av = a.data[aOff + i * k + p];
        if (av === 0) continue;
        for (let j = 0; j < n; j++) {
          out.data[oOff + i * n + j] += av * b.data[bOff + p * n + j];
        }
      }
    }
  }

  if (track(a, b)) {
    out._prev = [a, b];
    out._op = "matmul_batch";
    out._backward = () => {
      if (a.isGraphNode()) {
        const ga = new Float64Array(B * m * k);
        for (let bi = 0; bi < B; bi++) {
          const aOff = bi * m * k, bOff = bi * k * n, oOff = bi * m * n;
          for (let i = 0; i < m; i++) {
            for (let p = 0; p < k; p++) {
              let s = 0;
              for (let j = 0; j < n; j++) s += out.grad[oOff + i * n + j] * b.data[bOff + p * n + j];
              ga[aOff + i * k + p] = s;
            }
          }
        }
        a.accumulateGrad(ga);
      }
      if (b.isGraphNode()) {
        const gb = new Float64Array(B * k * n);
        for (let bi = 0; bi < B; bi++) {
          const aOff = bi * m * k, bOff = bi * k * n, oOff = bi * m * n;
          for (let p = 0; p < k; p++) {
            for (let i = 0; i < m; i++) {
              const av = a.data[aOff + i * k + p];
              if (av === 0) continue;
              const rowO = oOff + i * n;
              for (let j = 0; j < n; j++) gb[bOff + p * n + j] += out.grad[rowO + j] * av;
            }
          }
        }
        b.accumulateGrad(gb);
      }
    };
  }
  return out;
}

/** 逐元素减法 */
export function sub(a, b) {
  return add(a, scale(b, -1));
}
/** 逐元素函数，附带导数；用于 relu / tanh / sigmoid 等激活 */
export function unary(a, fn, dfn, name) {
  const out = new Tensor(new Float64Array(a.size), a.shape);
  for (let i = 0; i < a.size; i++) out.data[i] = fn(a.data[i]);
  if (a.isGraphNode()) {
    out._prev = [a];
    out._op = name;
    out._backward = () => {
      const g = new Float64Array(a.size);
      for (let i = 0; i < a.size; i++) g[i] = out.grad[i] * dfn(a.data[i]);
      a.accumulateGrad(g);
    };
  }
  return out;
}

export const relu = (a) => unary(a, (x) => (x > 0 ? x : 0), (x) => (x > 0 ? 1 : 0), "relu");
export const tanh = (a) => unary(a, Math.tanh, (x) => 1 - Math.tanh(x) ** 2, "tanh");
export const sigmoid = (a) =>
  unary(a, (x) => 1 / (1 + Math.exp(-x)),
    (x) => { const s = 1 / (1 + Math.exp(-x)); return s * (1 - s); }, "sigmoid");
export const exp = (a) => unary(a, Math.exp, Math.exp, "exp");
export const log = (a) => unary(a, Math.log, (x) => 1 / x, "log");

/**
 * GELU（tanh 近似版），Transformer 前馈网络常用的激活。
 *
 *   gelu(x) ≈ 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³)))
 *
 * 精确形式是 x·Φ(x)，需要 erf；tanh 近似把 erf 换成 tanh，
 * 精度差在 1e-3 量级，却更省算力，且求导干净——这是 GPT-2 起
 * 各家实现都在用的版本。
 *
 * 求导时别漏掉三次项的导数：
 *   d/dx gelu = 0.5·(1 + tanh(u)) + 0.5·x·(1 − tanh²(u))·u′
 *   其中 u′ = √(2/π)·(1 + 3·0.044715·x²)
 */
const GELU_C = Math.sqrt(2 / Math.PI);
const GELU_A = 0.044715;
export const gelu = (a) =>
  unary(
    a,
    (x) => 0.5 * x * (1 + Math.tanh(GELU_C * (x + GELU_A * x * x * x))),
    (x) => {
      const u = GELU_C * (x + GELU_A * x * x * x);
      const t = Math.tanh(u);
      const du = GELU_C * (1 + 3 * GELU_A * x * x);
      return 0.5 * (1 + t) + 0.5 * x * (1 - t * t) * du;
    },
    "gelu"
  );

/** 沿最后一维做 softmax（数值稳定版：先减最大值） */
export function softmax(a) {
  if (a.ndim < 1) throw new Error("softmax: 至少需要一维");
  const last = a.shape[a.shape.length - 1];
  const out = new Tensor(new Float64Array(a.size), a.shape);
  const rows = a.size / last;

  for (let r = 0; r < rows; r++) {
    const off = r * last;
    let max = -Infinity;
    for (let j = 0; j < last; j++) max = Math.max(max, a.data[off + j]);
    let sum = 0;
    for (let j = 0; j < last; j++) {
      out.data[off + j] = Math.exp(a.data[off + j] - max);
      sum += out.data[off + j];
    }
    for (let j = 0; j < last; j++) out.data[off + j] /= sum;
  }

  if (a.isGraphNode()) {
    out._prev = [a];
    out._op = "softmax";
    out._backward = () => {
      const g = new Float64Array(a.size);
      for (let r = 0; r < rows; r++) {
        const off = r * last;
        let dot = 0;
        for (let j = 0; j < last; j++) dot += out.grad[off + j] * out.data[off + j];
        for (let j = 0; j < last; j++) {
          g[off + j] = out.data[off + j] * (out.grad[off + j] - dot);
        }
      }
      a.accumulateGrad(g);
    };
  }
  return out;
}
/** 沿指定维度求和；1D 输入时对全部元素求和并返回标量 */
export function sum(a, dim = -1) {
  if (a.ndim === 1) {
    let s = 0;
    for (let i = 0; i < a.size; i++) s += a.data[i];
    const out = new Tensor([s], [1]);
    if (a.isGraphNode()) {
      out._prev = [a];
      out._op = "sum_all";
      out._backward = () => {
        const g = new Float64Array(a.size);
        g.fill(out.grad[0]);
        a.accumulateGrad(g);
      };
    }
    return out;
  }

  const d = dim < 0 ? a.ndim + dim : dim;
  if (d < 0 || d >= a.ndim) throw new Error(`sum: 维度越界 dim=${dim} ndim=${a.ndim}`);

  const outShape = a.shape.slice();
  outShape.splice(d, 1);
  const out = new Tensor(new Float64Array(numElements(outShape)), outShape);

  const inner = a.shape.slice(d + 1).reduce((x, y) => x * y, 1);
  const dimSize = a.shape[d];
  const outer = a.size / (inner * dimSize);

  for (let o = 0; o < outer; o++) {
    for (let i = 0; i < inner; i++) {
      let s = 0;
      for (let k = 0; k < dimSize; k++) s += a.data[o * dimSize * inner + k * inner + i];
      out.data[o * inner + i] = s;
    }
  }

  if (a.isGraphNode()) {
    out._prev = [a, dim];
    out._op = "sum_dim";
    out._backward = () => {
      const g = new Float64Array(a.size);
      for (let o = 0; o < outer; o++) {
        for (let i = 0; i < inner; i++) {
          const gv = out.grad[o * inner + i];
          for (let k = 0; k < dimSize; k++) g[o * dimSize * inner + k * inner + i] = gv;
        }
      }
      a.accumulateGrad(g);
    };
  }
  return out;
}

/** 沿指定维度求均值 */
export function mean(a, dim = -1) {
  const d = a.ndim === 1 ? 0 : (dim < 0 ? a.ndim + dim : dim);
  const n = a.shape[d];
  return scale(sum(a, dim), 1 / n);
}

/** 展平到指定形状，元素总数须一致 */
export function reshape(a, shape) {
  if (numElements(shape) !== a.size) {
    throw new Error(`reshape: 元素数不匹配 ${a.size} -> [${shape}]`);
  }
  const out = new Tensor(Float64Array.from(a.data), shape);
  if (a.isGraphNode()) {
    out._prev = [a];
    out._op = "reshape";
    out._backward = () => a.accumulateGrad(out.grad);
  }
  return out;
}

/** 交换二维张量的两个轴 */
/**
 * 轴置换。默认反转全部轴（等价于二维转置）；传 axes 可指定任意置换。
 *
 * 多头注意力需要把 [L, h, dh] 的头维提到最前或换回序列维，
 * 二维版本做不到，所以这里推广成通用 N 维置换。
 *
 * 反向就是再置换一次：out = transpose(a, axes) 则
 * grad_a = transpose(grad_out, inverse)，其中 inverse[axes[j]] = j。
 *
 * @param {Tensor} a
 * @param {number[]} [axes] 长度须等于 a.ndim，且各元素为 0..ndim-1 的排列
 */
export function transpose(a, axes) {
  const ndim = a.ndim;
  if (!axes) {
    axes = Array.from({ length: ndim }, (_, i) => ndim - 1 - i);
  }
  if (axes.length !== ndim) {
    throw new Error(`transpose: axes 长度 ${axes.length} 与维度数 ${ndim} 不符`);
  }
  const seen = new Set(axes);
  if (seen.size !== ndim || axes.some((x) => !Number.isInteger(x) || x < 0 || x >= ndim)) {
    throw new Error(`transpose: axes 必须是 0..${ndim - 1} 的排列，得到 [${axes}]`);
  }

  const oldShape = a.shape;
  const newShape = axes.map((ax) => oldShape[ax]);

  // 行主序下的步长，用下标 <-> 坐标的互换
  const stridesOf = (shape) => {
    const s = new Array(shape.length);
    let acc = 1;
    for (let i = shape.length - 1; i >= 0; i--) { s[i] = acc; acc *= shape[i]; }
    return s;
  };
  const oldStrides = stridesOf(oldShape);
  const newStrides = stridesOf(newShape);

  const out = new Tensor(new Float64Array(a.size), newShape);
  for (let o = 0; o < out.size; o++) {
    let rem = o;
    let idx = 0;
    for (let d = 0; d < ndim; d++) {
      const c = Math.floor(rem / newStrides[d]);
      rem -= c * newStrides[d];
      idx += c * oldStrides[axes[d]];
    }
    out.data[o] = a.data[idx];
  }

  if (a.isGraphNode()) {
    const inverse = new Array(ndim);
    for (let j = 0; j < ndim; j++) inverse[axes[j]] = j;
    out._prev = [a];
    out._op = "transpose";
    out._backward = () => {
      const g = new Float64Array(a.size);
      for (let o = 0; o < out.size; o++) {
        let rem = o;
        let idx = 0;
        for (let d = 0; d < ndim; d++) {
          const c = Math.floor(rem / newStrides[d]);
          rem -= c * newStrides[d];
          idx += c * oldStrides[axes[d]];
        }
        g[idx] += out.grad[o];
      }
      a.accumulateGrad(g);
    };
  }
  return out;
}

/** 逐元素除法 */
export function div(a, b) {
  const out = new Tensor(new Float64Array(a.size), a.shape);
  for (let i = 0; i < out.size; i++) out.data[i] = a.data[i] / b.data[i];
  if (track(a, b)) {
    out._prev = [a, b];
    out._op = "div";
    out._backward = () => {
      if (a.isGraphNode()) {
        const g = new Float64Array(a.size);
        for (let i = 0; i < g.length; i++) g[i] = out.grad[i] / b.data[i];
        a.accumulateGrad(g);
      }
      if (b.isGraphNode()) {
        const g = new Float64Array(b.size);
        for (let i = 0; i < g.length; i++) {
          g[i] = (-out.grad[i] * a.data[i]) / (b.data[i] * b.data[i]);
        }
        b.accumulateGrad(g);
      }
    };
  }
  return out;
}