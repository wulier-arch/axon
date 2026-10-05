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

## Browser examples

The browser examples load `../../src/index.js` directly. There is no package install,
bundler, CDN, or backend process involved.

- [XOR classifier](https://wulier-arch.github.io/axon/examples/browser/xor.html)
- [Linear regression](https://wulier-arch.github.io/axon/examples/browser/linear-regression.html)

To run them from a local clone, serve the repository root over HTTP:

```bash
python3 -m http.server 8000
```

Then open:

- http://localhost:8000/examples/browser/xor.html
- http://localhost:8000/examples/browser/linear-regression.html
