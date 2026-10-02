export {
  add, mul, scale, sub, div, matmul, relu, tanh, sigmoid, exp, log, softmax,
  sum, mean, reshape, transpose,
} from "./tensor/ops.js";

export { gatherRows, concat, argmaxLast } from "./tensor/shape.js";

export { conv2d, maxPool2d, avgPool2d } from "./tensor/conv.js";

export { Tensor, numElements, swap } from "./tensor/tensor.js";
export { backward, backward2, collectLeaves, zeroGrad } from "./autodiff.js";
export { numericalGradient, checkGradient } from "./gradient-check.js";