# axon roadmap

North star: make axon a mainstream educational neural-network framework for JavaScript, with a reputation for readable implementations and numerically verified gradients.

The pinned [v0.4.0 roadmap issue](https://github.com/wulier-arch/axon/issues/1) tracks work in progress.

## Stage 1: foundation

Status: complete through v0.3.1.

- Zero-dependency runtime, browser and Node.js support
- Reverse-mode autograd, tensors, convolution, layers, losses, optimizers, and training loops
- `LayerNorm`, `Embedding`, `MultiHeadAttention`, `TransformerBlock`, and `GELU`
- JSON model serialization (`Sequential.toJSON()`) and loading (`loadModel`)
- Finite-difference gradient checks for every operator
- Browser demo, project site, npm-ready package metadata, CI, and reproducible benchmarks
- Community standards: contributing guide, code of conduct, security policy, issue templates, and pull request template
- `v0.3.0` and `v0.3.1` GitHub Releases and social preview

## Stage 2: adoption

Status: in progress.

Completion criteria:

- 25 or more authentic GitHub Stars
- 5 or more external Issues or Discussions
- 3 or more external pull requests
- English API reference covering every public export (complete)
- Browser examples for XOR and regression (complete)
- Browser attention example (open)
- At least one reproducible comparison with TensorFlow.js and Brain.js
- A tagged release with no failing CI checks

## Stage 3: mainstream

Completion criteria:

- 300 or more GitHub Stars
- 10 or more external contributors
- One independent article, talk, course, or production use case
- Inclusion in a relevant curated list such as `awesome-machine-learning`
- At least one detailed tutorial or conference-style walkthrough
- Stable documentation for every public export
- Regular releases with a documented compatibility policy

## Stage 4: ecosystem

- Stable v1.0 API
- JSON model serialization and loading (basic round-trip complete; schema
  stability and migration guarantees land with the v1.0 API)
- Additional optimized educational examples
- Integrations with browser tooling, notebooks, or visualization libraries
- Contributor guide for adding a new operator, layer, optimizer, and gradient check

## Non-goals

- Replacing TensorFlow.js for production-scale training
- Chasing raw throughput at the cost of readability
- Adding dependencies when the implementation can remain small and understandable
- Buying or exchanging Stars, artificial engagement, or unverified usage claims
