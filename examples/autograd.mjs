import {
  Tensor,
  backward,
  checkGradient,
  mul,
} from "../src/index.js";

const x = Tensor.variable([2, 3], [2]);
const loss = mul(x, x).sum();

backward(loss);
console.log("gradient:", Array.from(x.grad));

const report = checkGradient(() => mul(x, x).sum(), [x]);
console.log("gradient check passed:", report.passed);
console.log(report.report);

