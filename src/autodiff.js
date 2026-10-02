import { Tensor } from "./tensor/tensor.js";

/**
 * 反向传播主循环。
 *
 * 每个中间张量持有 _backward 闭包与 _prev 输入快照，
 * 这里做拓扑排序后按逆序执行，保证用到某个张量梯度时它已算好。
 * 对重复使用的子图用 seen 集合去重，只计算一次。
 */
export function backward(root) {
  if (!root.isGraphNode()) {
    throw new Error("backward: 目标张量既未建图也未开启 requiresGrad");
  }
  if (!root.grad) {
    // 约定：标量损失反向时上游梯度为 1
    root.grad = Float64Array.of(1);
  }

  const topo = [];
  const seen = new Set();

  function build(t) {
    if (seen.has(t)) return;
    seen.add(t);
    // 先依赖后当前，保证上游先于下游完成
    const prev = t._prev || [];
    for (const item of prev) {
      if (item instanceof Tensor) build(item);
    }
    topo.push(t);
  }
  build(root);

  for (let i = topo.length - 1; i >= 0; i--) {
    const t = topo[i];
    if (t._backward && t.grad) t._backward();
  }
}

/**
 * 反向传播主循环（修正版）。
 *
 * 关键点：算子的 _backward 闭包会把上游梯度写进下游张量的 .grad，
 * 但下游张量自己的 .grad 必须先存在。因此这里在执行每个节点的
 * _backward 之前，根据它在图中的角色注入梯度：
 *
 *   - 根节点：损失对自身的导数，约定为 1
 *   - 中间节点：由上游算子在上一层回传时写入
 *   - 叶子变量：只读取，不参与计算
 *
 * 上一版依赖「每个中间节点恰好被上游写过一次」，一旦某个节点
 * 被跳过（例如被多个分支共享且某一分支未贡献梯度），其 .grad 就是
 * null，后续 _backward 便无法执行。这里显式注入，消除这个隐式前提。
 */
export function backward2(root) {
  if (!root.isGraphNode()) {
    throw new Error("backward: 目标张量既未建图也未开启 requiresGrad");
  }

  const topo = [];
  const seen = new Set();

  function build(t) {
    if (seen.has(t)) return;
    seen.add(t);
    for (const item of t._prev || []) {
      if (item instanceof Tensor) build(item);
    }
    topo.push(t);
  }
  build(root);

  // 根节点上游梯度为 1
  root.grad = Float64Array.of(1);

  for (let i = topo.length - 1; i >= 0; i--) {
    const t = topo[i];
    // 中间节点若还没收到上游梯度，说明它对最终损失无贡献，
    // 按「零梯度」处理，避免 _backward 读到 null
    if (!t.grad && t._backward) {
      t.grad = new Float64Array(t.size);
    }
    if (t._backward && t.grad) t._backward();
  }
}

/** 收集子图里所有需要梯度的叶子张量 */
export function collectLeaves(root) {
  const leaves = [];
  const seen = new Set();
  const stack = [root];
  while (stack.length) {
    const t = stack.pop();
    if (seen.has(t)) continue;
    seen.add(t);
    const prev = t._prev || [];
    for (const item of prev) {
      if (item instanceof Tensor) stack.push(item);
    }
    if (!t._prev && t.requiresGrad) leaves.push(t);
  }
  return leaves;
}

/** 清空子图内所有张量的梯度 */
export function zeroGrad(root) {
  const seen = new Set();
  const stack = [root];
  while (stack.length) {
    const t = stack.pop();
    if (seen.has(t)) continue;
    seen.add(t);
    for (const item of t._prev || []) {
      if (item instanceof Tensor) stack.push(item);
    }
    t.zeroGrad();
  }
}