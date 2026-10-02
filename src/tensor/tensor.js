/**
 * axon/src/tensor/tensor.js — 标量到任意维张量 + 反向模式自动微分
 *
 * 设计取舍：
 *  - 统一用 Float64Array 存数据：数值稳定，梯度检查更可靠
 *  - 标量一律表示成 shape=[1]，不为标量单独开一套类型，减少分支
 *  - 每个算子保留 prev（输入快照）与 backward 闭包，构成反向传播的 tape
 */

/** 张量的数据缓冲都用 Float64Array，梯度检查需要足够的精度 */
export class Tensor {
  /**
   * @param {Float64Array|number[]} data
   * @param {number[]} shape
   */
  constructor(data, shape) {
    this.data = data instanceof Float64Array ? data : Float64Array.from(data);
    this.shape = shape;
    this.grad = null;
    this.requiresGrad = false;

    /** 反向传播用：指向产生本张量的算子，null 表示叶子 */
    this._backward = null;
    /** 反向传播用：算子的输入快照 */
    this._prev = null;
    /** 反向传播用：算子名，用于调试与报错 */
    this._op = "";

    if (this.data.length !== numElements(shape)) {
      throw new Error(
        `张量形状不匹配：shape=[${shape}] 需要 ${numElements(shape)} 个元素，实际 ${this.data.length}`
      );
    }
  }

  /** 创建不需要梯度的张量 */
  static tensor(data, shape) {
    return new Tensor(data, shape);
  }

  /** 创建需要梯分的叶子张量 */
  static variable(data, shape) {
    const t = new Tensor(data, shape);
    t.requiresGrad = true;
    return t;
  }

  get size() {
    return numElements(this.shape);
  }

  get ndim() {
    return this.shape.length;
  }

  /**
   * 把上游梯度存进本张量。
   *
   * 中间张量的 requiresGrad 是 false（它只是算子产物），但梯度必须照常存下，
   * 否则下游算子读不到 .grad。因此这里不按 requiresGrad 过滤，
   * 而是由各算子自己决定要不要调用。
   */
  accumulateGrad(gradData) {
    if (!this.grad) {
      this.grad = Float64Array.from(gradData);
    } else {
      for (let i = 0; i < this.data.length; i++) this.grad[i] += gradData[i];
    }
  }

  /** 叶子变量的梯度：常数张量没有梯度可言 */
  parameterGrad() {
    return this.requiresGrad ? this.grad : null;
  }

  /** 清空梯度，训练循环每轮调用 */
  zeroGrad() {
    this.grad = null;
  }

  /**
   * 全部元素之和，标量。
   *
   * 判据用 isGraphNode()：中间张量（如 mul 的输出）虽然 requiresGrad 为 false，
   * 但它已经建了图、带着 _prev，若按 requiresGrad 判断就会漏掉反向链。
   */
  sum() {
    let s = 0;
    for (let i = 0; i < this.data.length; i++) s += this.data[i];
    const out = new Tensor([s], [1]);
    if (this.isGraphNode()) {
      out._prev = [this];
      out._op = "sum";
      out._backward = () => {
        // d(sum)/dx_i = 1，上游梯度原样广播到每个元素
        const g = new Float64Array(this.size);
        g.fill(out.grad[0]);
        this.accumulateGrad(g);
      };
    }
    return out;
  }

  /**
   * 是否属于可求导的计算图：自己是叶子变量，或由某个算子产出。
   * 用于区分「常数张量」与「图中的中间节点」。
   */
  isGraphNode() {
    return this.requiresGrad || this._backward !== null || this._prev !== null;
  }

  /** 深拷贝，不共享底层缓冲 */
  clone() {
    return new Tensor(Float64Array.from(this.data), this.shape.slice());
  }

  /** 人类可读的形式 */
  toString() {
    return `Tensor(shape=[${this.shape}], requiresGrad=${this.requiresGrad})`;
  }

  /** 导出为普通 JS 数组，便于序列化与测试 */
  toJSON() {
    return { shape: this.shape.slice(), data: Array.from(this.data) };
  }
}

/** 形状的元素总数 */
export function numElements(shape) {
  let n = 1;
  for (let s of shape) n *= s;
  return n;
}

/** 把 2D 形状 [a,b] 转置成 [b,a] */
export function swap(shape) {
  return [shape[1], shape[0]];
}