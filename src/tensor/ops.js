import { Tensor, numElements } from "./tensor.js";

/**
 * 判断这次运算是否需要建计算图。
 *
 * 除了叶子变量（requiresGrad），已经建过图的中间张量也要算进来。
 * 否则 sub(a,b) 这类由中间节点参与的组合运算会丢失反向链，
 * 表现为「结果对，但某个输入的梯度是 null」。
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
export function add(a, b) {
  const outShape = a.shape.length === 1 && a.shape[0] === 1 && b.ndim > 0
    ? b.shape
    : (b.shape.length === 1 && b.shape[0] === 1 && a.ndim > 0 ? a.shape
      : (a.size === b.size ? a.shape : null));
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
      if (a.requiresGrad) {
        const ga = new Float64Array(a.size);
        for (let i = 0; i < outSize; i++) {
          const ai = aScalar ? 0 : (a.size === outSize ? i : broadcastIndex(outShape, a.shape, i, a.shape.length));
          ga[ai] += out.grad[i];
        }
        a.accumulateGrad(ga);
      }
      if (b.requiresGrad) {
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
      if (a.requiresGrad) {
        const g = new Float64Array(a.size);
        for (let i = 0; i < g.length; i++) g[i] = out.grad[i] * b.data[i];
        a.accumulateGrad(g);
      }
      if (b.requiresGrad) {
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
  if (a.requiresGrad) {
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
      if (a.requiresGrad) {
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
      if (b.requiresGrad) {
        const gb = new Float64Array(k * n);
        for (let p = 0; p < k; p++) {
          const rowB = p * n;
          for (let i = 0; i < m; i++) {
            const g = out.grad[i * n];
            if (g === 0) continue;
            for (let j = 0; j < n; j++) gb[rowB + j] += g * a.data[i * k + p];
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
      if (a.requiresGrad) {
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
      if (b.requiresGrad) {
        const gb = new Float64Array(B * k * n);
        for (let bi = 0; bi < B; bi++) {
          const aOff = bi * m * k, bOff = bi * k * n, oOff = bi * m * n;
          for (let p = 0; p < k; p++) {
            for (let i = 0; i < m; i++) {
              const g = out.grad[oOff + i * n];
              if (g === 0) continue;
              for (let j = 0; j < n; j++) gb[bOff + p * n + j] += g * a.data[aOff + i * k + p];
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
  if (a.requiresGrad) {
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

  if (a.requiresGrad) {
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
    if (a.requiresGrad) {
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

  if (a.requiresGrad) {
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
  if (a.requiresGrad) {
    out._prev = [a];
    out._op = "reshape";
    out._backward = () => a.accumulateGrad(out.grad);
  }
  return out;
}

/** 交换二维张量的两个轴 */
export function transpose(a) {
  if (a.ndim !== 2) throw new Error(`transpose: 仅支持 2D，得到 ${a.ndim}D`);
  const [m, n] = a.shape;
  const out = new Tensor(new Float64Array(a.size), [n, m]);
  for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) out.data[j * m + i] = a.data[i * n + j];

  if (a.requiresGrad) {
    out._prev = [a];
    out._op = "transpose";
    out._backward = () => {
      const g = new Float64Array(a.size);
      for (let i = 0; i < m; i++) for (let j = 0; j < n; j++) g[i * n + j] = out.grad[j * m + i];
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
      if (a.requiresGrad) {
        const g = new Float64Array(a.size);
        for (let i = 0; i < g.length; i++) g[i] = out.grad[i] / b.data[i];
        a.accumulateGrad(g);
      }
      if (b.requiresGrad) {
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