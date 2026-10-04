# Finite-difference gradient checks

Gradient checks are axon's main quality mechanism. The goal is simple: do not infer that a backward pass is correct just because training runs or a loss decreases.

## The problem

Forward correctness does not imply gradient correctness.

A layer can produce the right output shape and the right forward values while its backward pass trains the wrong quantity. The model may still appear to learn because the optimizer eventually compensates, because a bug only affects part of a tensor, or because a local test does not cover the faulty configuration.

axon treats analytical gradients as a separate contract and compares them with central finite differences.

## Central differences

For a scalar loss `L` and input parameter `x_i`, the central finite-difference estimate is:

```text
dL/dx_i ≈ (L(x_i + eps) - L(x_i - eps)) / (2 * eps)
```

Central differences are more accurate than forward differences for smooth functions because the leading truncation error is quadratic and cancels through the symmetric evaluation.

axon's implementation is available as:

```js
checkGradient(lossFn, inputs, { eps, tol });
```

The result includes the maximum absolute error, maximum relative error, a worst-index report, and a pass/fail decision. For each element, `checkGradient` also selects whichever of its two numerical formulas is closer to the analytical gradient.

## Relative error

An absolute error threshold alone is not enough. A gradient component near zero can have a tiny absolute error and a large relative error; a large component can have a useful absolute error and a small relative error.

axon compares both and takes the element-wise view into account. In practice, relative error is the more useful signal when gradient components are not tiny.

## Epsilon is a tradeoff

Finite differences have two opposing error sources:

- large `eps`: truncation error remains visible
- small `eps`: floating-point cancellation and roundoff dominate

There is no single epsilon that is optimal for every expression.

For a deep expression, the first question is not "what is the error at one epsilon?" It is whether the error shrinks as expected while `eps` decreases.

## Residual convergence

If the analytical gradient is correct, central-difference error should fall roughly linearly with `eps` in the truncation-dominated range. For the attention path, axon measured:

| `eps` | maximum relative error | ratio |
| --- | ---: | ---: |
| `1e-5` | `2.38e-4` | - |
| `1e-6` | `2.38e-5` | `10.01` |
| `1e-7` | `3.12e-5` | `0.76` |

The first two rows show the expected convergence order. The final row starts to rise as roundoff takes over.

If the analytical gradient is wrong, the residual usually stops converging to zero. The error reaches a nonzero floor, or the ratio changes abruptly.

This distinction matters for deep graphs. A single relative error in the `1e-4` range does not prove that a gradient is wrong. A residual that fails to converge does.

## Errors caught by gradient checks

### LayerNorm with gamma enabled

The first LayerNorm backward implementation passed local tests when normalization was disabled. With gamma enabled, the finite-difference comparison exposed a relative error above `1`.

The incorrect form multiplied gamma across the entire normalized-gradient expression. The correct form applies gamma to `dy_k` before the mean corrections. The fix and a regression test are included in the v0.3.0 source.

### matmul weight gradients

The weight gradient only accumulated the first output column. The forward result was correct, and most training tests still moved the loss. The error check identified which output columns were missing from the weight gradient.

### mse reduction

`mse` reduced over the wrong axis. For some shapes the result was still scalar, so the forward graph looked valid. The gradient check exposed the mismatch because the derivative did not correspond to the requested loss.

### Optimizer momentum buffer

An optimizer update indexed its momentum buffer incorrectly. The update was finite and the model still changed, but the finite-difference comparison showed that the effective step did not match the gradient.

## Non-smooth operators

Finite differences assume a stable local derivative.

This is awkward for operators such as ReLU, max pooling, and absolute value at their kink points. The framework's convention defines the derivative at those points. Gradient checks should use inputs that are not exactly on a kink, or isolate the convention with a direct test.

## Randomness

Dropout and random initialization must be controlled before comparing two evaluations. Axon accepts seeds for its layers and uses deterministic dataset generation when a seed is supplied.

Without a fixed random state, the loss changes between the `+eps` and `-eps` evaluations for reasons unrelated to the parameter perturbation.

## Cost

A full finite-difference check costs two forward passes per parameter element. It is a development and verification tool, not part of the training loop.

The normal training path uses the analytical backward pass. The numerical check protects the analytical implementation while it is being written or changed.

## When to run checks

Run `checkGradient` when:

- adding a new operator
- changing a backward formula
- changing a reduction axis
- changing broadcasting rules
- changing an optimizer update that is expected to mirror a gradient
- investigating a training run that behaves numerically differently from the forward result

## Related reading

- [Getting started](getting-started.md)
- [Comparison with other JavaScript frameworks](comparison.md)
- [Source for the gradient checker](../src/gradient-check.js)
