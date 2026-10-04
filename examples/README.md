# Examples

All examples run directly from the repository with Node.js 18 or newer. No install step is required.

```bash
node examples/autograd.mjs
node examples/train-spiral.mjs
```

## autograd.mjs

Demonstrates `Tensor.variable`, `backward`, and `checkGradient`.

## train-spiral.mjs

Builds a two-layer classifier, trains it on the generated spiral dataset, and prints the final loss and accuracy.

