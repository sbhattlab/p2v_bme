# Phylo2Vec BME

Balanced minimum evolution (BME) tree search in Phylo2Vec coordinates. For an N-leaf tree encoded as a
Phylo2Vec vector `v`, the BME change of every single-coordinate move is computed in O(N²) (the "discrete
gradient") and used for tree search on CPU and GPU, scaling to N = 10⁷ with minibatch condensation.

Work in progress. `p2v-bme` is the command-line tool: aligned DNA FASTA in, unrooted Newick out, distances under
JC69, K80, F81, F84, TN93 or LogDet, search on the GPU or the CPU (`--accelerator`; `--precision double` on the CPU).
GTR+Γ distances and bootstrap support come next.

## Layout

| Path | Contents |
|---|---|
| `src/` | The package. Public API `inference.hpp` (`infer()` in `inference.cpp`) and `io.hpp`; the search `search.cpp`, start trees `start_tree.cpp`, branch lengths `branch_lengths.cpp`; distances `substitution_models.cpp` (JC69 to TN93, LogDet) and `eigen_dist.cpp` (eigenvalue GTR+Γ estimators); the device interface `device.hpp`, the GPU device `device_gpu.cu` (all kernels) and the CPU device `device_cpu.cpp` (OpenMP); settings `config.hpp`, random draws `random.hpp`; FASTA, .maple and Newick I/O `io.cpp`; the run report `report.cpp`; command line `cli.cpp`; simulation `simulations.cpp`. Also the C reference library (`reference/bme_reference.c`) |
| `tests/` | GoogleTest unit tests and C/CUDA checks, run by ctest |
| `data/`, `build/` | Inputs and results, build outputs (git-ignored) |

## Setup

Needs the system `cc`, `g++`, CMake ≥ 3.24 and the CUDA toolkit (`nvcc` on `PATH`). pixi provides the rest:

```bash
pixi install            # package environment: cxxopts, GoogleTest
```

## Build and test

CMake builds everything into `build/` (presets in `CMakePresets.json`; CUDA is used when `nvcc` is found, for the
build machine's GPUs):

```bash
pixi run build                   # p2v-bme, libraries, tests
pixi run test                    # ctest; the tests labelled gpu need a GPU
ctest --test-dir build -LE gpu   # CPU tests only
pixi run test-asan               # CPU tests under AddressSanitizer + UBSan (build-asan/)
```

For another GPU, configure by hand inside `pixi shell`, e.g. for H100:
`cmake --preset default -DCMAKE_CUDA_ARCHITECTURES=90`. Without CUDA (`cmake --preset cpu`, into `build-cpu/`) only the
CPU libraries and tests build.

On a shared machine, pick a free GPU first (`nvidia-smi`, then `CUDA_VISIBLE_DEVICES=<id>`).

## Usage

```bash
build/p2v-bme -s aln.fa -m JC69                       # writes aln.fa.nwk and aln.fa.json
build/p2v-bme -s aln.fa -m JC69 --prefix run -t random --seed 3 -v
build/p2v-bme -s aln.fa -m JC69 -t tree.nwk --max-rounds 0   # score a given tree (BME length, branch lengths)
build/p2v-bme --help        # core options; --help-all for all
```

By default the search starts from a sample-and-place tree and runs up to 8 rounds of minibatch moves. It stops
early once a round gains at most a 10⁻⁶ fraction of the total gain, or changes nothing. BME branch lengths are
then estimated. The tree is written unrooted to `<prefix>.nwk` (negative lengths as 0 unless `--no-neg-as-zero`);
`<prefix>.json` is the run report (settings, timings, results). Progress is printed on stderr with `-v`.
