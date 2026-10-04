# Comparison

This page describes intended use and implementation tradeoffs. It is not a performance ranking.

| Project | Main strength | Best fit | Tradeoff |
| --- | --- | --- | --- |
| axon | Readable, zero-dependency training stack | Learning autograd, training systems, gradients, and small models | Slow scalar JavaScript; no WASM, SIMD, or operator fusion |
| TensorFlow.js | Production ecosystem and hardware acceleration | Real applications that need WebGL, WebGPU, native acceleration, pretrained models, and broad tooling | The full training path is difficult to inspect end to end |
| Brain.js | Simple neural-network API | Quick feed-forward or recurrent network experiments | Not a general tensor and autograd framework |
| ml5.js | Friendly high-level API | Beginners and creative-coding projects | A high-level layer rather than a from-scratch learning implementation |

## Why axon exists

axon is not trying to replace TensorFlow.js. It targets a different question:

> Can the full path from tensor operations to backpropagation stay small enough to read, debug, and verify?

That leads to different choices:

- `Float64Array` instead of a fused GPU tensor runtime
- scalar JavaScript loops instead of WASM or SIMD kernels
- compositional autograd instead of hand-written backward functions in high-level layers
- finite-difference checks for every operator
- browser and Node.js support without a build step

## Choose TensorFlow.js when

- throughput or GPU acceleration is a requirement
- you need pretrained models or a large production ecosystem
- you are deploying a model rather than studying its implementation

## Choose Brain.js when

- a compact neural-network API is enough
- you do not need tensors, convolution, custom optimizers, or transformer components
- the internals of autograd are not part of the goal

## Choose ml5.js when

- the project should stay at a friendly, high-level API
- the goal is creative work or a first machine-learning experience
- lower-level training mechanics are not important

## Choose axon when

- you want to read every operator and training-loop step
- you care about finite-difference gradient verification
- you are learning how autograd, convolution, attention, and optimizers work
- you want one dependency-free source tree that runs in a browser and Node.js

## Reproduce the numbers

axon's benchmark command is included in the repository:

```bash
npm run benchmark
```

Numbers are machine-dependent and should be reproduced locally before drawing conclusions.

