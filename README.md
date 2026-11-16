# Phylo2Vec BME

Balanced minimum evolution (BME) tree search in Phylo2Vec coordinates. For an N-leaf tree encoded as a
Phylo2Vec vector `v`, the BME change of every single-coordinate move is computed in O(N²) (the "discrete
gradient") and used for tree search on CPU and GPU, scaling to N = 10⁷ with minibatch condensation.

Work in progress.

## Setup

Needs the system `cc`, `g++`, CMake ≥ 3.24 and the CUDA toolkit (`nvcc` on `PATH`). pixi provides the rest:

```bash
pixi install            # package environment: cxxopts, GoogleTest
```

## Build and test

CMake builds everything into `build/` (presets in `CMakePresets.json`; CUDA is used when `nvcc` is found, for the
build machine's GPUs):

```bash
pixi run build                   # build
pixi run test                    # ctest
pixi run test-asan               # CPU tests under AddressSanitizer + UBSan (build-asan/)
```
