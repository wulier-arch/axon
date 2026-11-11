/**
 * 单隐层宽度扫描 —— 同一任务只改宽度，看准确率怎么变。
 *
 * 起因：文档里「宽度不够就学不会」只是断言，这里给出可复现的数字。
 * 除了 width，学习率 / 轮数 / 批大小 / 随机种子全部固定，
 * 所以任何差异都只能归到宽度这一个变量上。
 *
 * 运行：npm run benchmark:width
 */

import {
  Adam, Linear, Sequential, Trainer, accuracy, crossEntropy, makeSpiral,
} from "../src/index.js";

const WIDTHS = [2, 4, 8, 16, 32];
const EPOCHS = 400;
const BATCH = 32;
const LR = 0.01;

const pct = (v) => `${(v * 100).toFixed(1)}%`;

console.log("单隐层宽度扫描｜螺旋分类 300 样本｜Adam lr=0.01｜400 轮｜batch 32");
console.log("");
console.log("  宽度   最终准确率   最终损失   耗时      首次到 100% 的轮次");

for (const width of WIDTHS) {
  const {x, y} = makeSpiral({samples: 300});
  const model = new Sequential()
    .add(new Linear(2, width, {activation: "relu", seed: 1}))
    .add(new Linear(width, 2, {seed: 2}));
  const trainer = new Trainer({
    model,
    optimizer: new Adam({lr: LR}),
    lossFn: crossEntropy,
    metricFn: accuracy,
    epochs: EPOCHS,
    batchSize: BATCH,
  });

  const started = performance.now();
  trainer.fit(x, y, y);
  const ms = performance.now() - started;

  const final = trainer.history.at(-1);
  const hit = trainer.history.find((h) => h.metric >= 0.999);

  console.log(
    `  ${String(width).padStart(4)}   ${pct(final.metric).padStart(10)}   ${final.loss
      .toFixed(4)
      .padStart(8)}   ${`${Math.round(ms)}ms`.padStart(7)}   ${hit ? `第 ${hit.epoch} 轮` : "未达到"}`,
  );
}

console.log("");
console.log("说明：同一份源码在浏览器里也能跑（见 demo/），这里用 Node.js 计时。");
