/**
 * 二维卷积与池化。
 *
 * 输入布局为 [C, H, W]（NCHW 单样本），权重为 [OC, C, KH, KW]，
 * 输出布局为 [outH, outW, OC]（channels-last，便于直接接全连接层）。
 */
import { Tensor } from "./tensor.js";

/**
 * im2col + GEMM 实现卷积。
 * 直接写六重循环虽然直观但缓存命中率很差；先把输入铺成
 * [outH*outW, C*KH*KW] 矩阵，再复用 GEMM，梯度也更规整。
 */
export function conv2d(x, w, b = null, stride = 1, padding = 0) {
  if (x.ndim !== 3 || w.ndim !== 4) {
    throw new Error(`conv2d: 需要 x 为 [C,H,W]、w 为 [OC,C,KH,KW]，得到 x=${x.shape} w=${w.shape}`);
  }
  const [C, H, W] = x.shape;
  const [OC, , KH, KW] = w.shape;
  if (C !== w.shape[1]) throw new Error(`conv2d: 通道不匹配 x=${C} w=${w.shape[1]}`);

  const [sH, sW] = Array.isArray(stride) ? stride : [stride, stride];
  const [padH, padW] = Array.isArray(padding) ? padding : [padding, padding];

  const outH = Math.floor((H + 2 * padH - KH) / sH) + 1;
  const outW = Math.floor((W + 2 * padW - KW) / sW) + 1;
  if (outH <= 0 || outW <= 0) {
    throw new Error(`conv2d: 输出尺寸非法 H=${H} W=${W} KH=${KH} KW=${KW}`);
  }

  const patchSize = C * KH * KW;
  const outSpatial = outH * outW;

  // im2col：[outH*outW, C*KH*KW]，越界位置填 0（padding）
  const col = new Float64Array(outSpatial * patchSize);
  for (let oh = 0; oh < outH; oh++) {
    for (let ow = 0; ow < outW; ow++) {
      const rowBase = (oh * outW + ow) * patchSize;
      for (let c = 0; c < C; c++) {
        for (let kh = 0; kh < KH; kh++) {
          const ih = oh * sH + kh - padH;
          if (ih < 0 || ih >= H) continue;
          for (let kw = 0; kw < KW; kw++) {
            const iw = ow * sW + kw - padW;
            if (iw < 0 || iw >= W) continue;
            col[rowBase + c * KH * KW + kh * KW + kw] = x.data[c * H * W + ih * W + iw];
          }
        }
      }
    }
  }

  // GEMM: [OC, patchSize] × [patchSize, outSpatial]
  const outFlat = new Float64Array(OC * outSpatial);
  for (let oc = 0; oc < OC; oc++) {
    const wRow = oc * patchSize;
    for (let p = 0; p < patchSize; p++) {
      const wv = w.data[wRow + p];
      if (wv === 0) continue;
      for (let s = 0; s < outSpatial; s++) {
        outFlat[oc * outSpatial + s] += wv * col[s * patchSize + p];
      }
    }
  }

  const out = new Tensor(new Float64Array(outSpatial * OC), [outH, outW, OC]);
  for (let s = 0; s < outSpatial; s++) {
    for (let oc = 0; oc < OC; oc++) out.data[s * OC + oc] = outFlat[oc * outSpatial + s];
  }
  if (b) {
    for (let s = 0; s < outSpatial; s++) {
      for (let oc = 0; oc < OC; oc++) out.data[s * OC + oc] += b.data[oc];
    }
  }

  if (x.requiresGrad || w.requiresGrad) {
    out._prev = [x, w, b, sH, sW, padH, padW, { outH, outW, col, patchSize }];
    out._op = "conv2d";
    out._backward = () => {
      const meta = out._prev[7];

      if (w.isGraphNode()) {
        const dw = new Float64Array(w.size);
        for (let oc = 0; oc < OC; oc++) {
          for (let p = 0; p < meta.patchSize; p++) {
            let acc = 0;
            for (let q = 0; q < outSpatial; q++) {
              acc += out.grad[oc * outSpatial + q] * meta.col[q * meta.patchSize + p];
            }
            dw[oc * meta.patchSize + p] = acc;
          }
        }
        w.accumulateGrad(dw);
      }

      if (b && b.requiresGrad) {
        const db = new Float64Array(OC);
        for (let q = 0; q < outSpatial; q++) {
          for (let oc = 0; oc < OC; oc++) db[oc] += out.grad[oc * outSpatial + q];
        }
        b.accumulateGrad(db);
      }

      if (x.isGraphNode()) {
        const dx = new Float64Array(x.size);
        for (let q = 0; q < outSpatial; q++) {
          const oh = (q / meta.outW) | 0, ow = q % meta.outW;
          for (let c = 0; c < C; c++) {
            for (let kh = 0; kh < KH; kh++) {
              const ih = oh * sH + kh - padH;
              if (ih < 0 || ih >= H) continue;
              for (let kw = 0; kw < KW; kw++) {
                const iw = ow * sW + kw - padW;
                if (iw < 0 || iw >= W) continue;
                const pIdx = c * KH * KW + kh * KW + kw;
                let acc = 0;
                for (let oc = 0; oc < OC; oc++) {
                  acc += out.grad[oc * outSpatial + q] * w.data[oc * meta.patchSize + pIdx];
                }
                dx[c * H * W + ih * W + iw] += acc;
              }
            }
          }
        }
        x.accumulateGrad(dx);
      }
    };
  }
  return out;
}
/** 2×2 最大池化，stride 默认与 kernel 相同；返回 [outH, outW, C] */
export function maxPool2d(x, kernel = 2, stride = null) {
  const [C, H, W] = x.shape;
  const s = stride ?? kernel;
  const outH = Math.floor((H - kernel) / s) + 1;
  const outW = Math.floor((W - kernel) / s) + 1;
  const out = new Tensor(new Float64Array(outH * outW * C), [outH, outW, C]);

  // 记录每个输出对应的输入最大值位置，反向时只把梯度送回那一个点
  const argmaxPos = new Int32Array(out.size);

  for (let oh = 0; oh < outH; oh++) {
    for (let ow = 0; ow < outW; ow++) {
      for (let c = 0; c < C; c++) {
        let best = -Infinity, bestPos = 0;
        for (let i = 0; i < kernel; i++) {
          for (let j = 0; j < kernel; j++) {
            const idx = c * H * W + (oh * s + i) * W + (ow * s + j);
            if (x.data[idx] > best) { best = x.data[idx]; bestPos = idx; }
          }
        }
        const o = (oh * outW + ow) * C + c;
        out.data[o] = best;
        argmaxPos[o] = bestPos;
      }
    }
  }

  if (x.isGraphNode()) {
    out._prev = [x, argmaxPos];
    out._op = "maxPool2d";
    out._backward = () => {
      const dx = new Float64Array(x.size);
      for (let o = 0; o < out.size; o++) dx[argmaxPos[o]] += out.grad[o];
      x.accumulateGrad(dx);
    };
  }
  return out;
}

/** 2×2 平均池化，梯度在窗口内均匀分配 */
export function avgPool2d(x, kernel = 2, stride = null) {
  const [C, H, W] = x.shape;
  const s = stride ?? kernel;
  const outH = Math.floor((H - kernel) / s) + 1;
  const outW = Math.floor((W - kernel) / s) + 1;
  const out = new Tensor(new Float64Array(outH * outW * C), [outH, outW, C]);

  for (let oh = 0; oh < outH; oh++) {
    for (let ow = 0; ow < outW; ow++) {
      for (let c = 0; c < C; c++) {
        let acc = 0;
        for (let i = 0; i < kernel; i++) {
          for (let j = 0; j < kernel; j++) acc += x.data[c * H * W + (oh * s + i) * W + (ow * s + j)];
        }
        out.data[(oh * outW + ow) * C + c] = acc / (kernel * kernel);
      }
    }
  }

  if (x.isGraphNode()) {
    out._prev = [x, kernel, s];
    out._op = "avgPool2d";
    out._backward = () => {
      const dx = new Float64Array(x.size);
      const share = 1 / (kernel * kernel);
      for (let o = 0; o < out.size; o++) {
        const c = o % C;
        const sIdx = (o / C) | 0;
        const oh = (sIdx / outW) | 0, ow = sIdx % outW;
        for (let i = 0; i < kernel; i++) {
          for (let j = 0; j < kernel; j++) {
            dx[c * H * W + (oh * s + i) * W + (ow * s + j)] += out.grad[o] * share;
          }
        }
      }
      x.accumulateGrad(dx);
    };
  }
  return out;
}