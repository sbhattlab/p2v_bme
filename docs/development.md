# Development

## Recommended setup

Besides the build prerequisites ([README](../README.md#prerequisites)):

- an NVIDIA GPU with the CUDA toolkit, to build and test the GPU code (CI has no GPU). One GPU is enough: the multi-GPU
  tests run several devices on it.
- several CPU cores: the tests run 8 at a time, and the CPU code and the GTR+GC fit are multithreaded (OpenMP).

## Environments and tasks

[pixi](https://pixi.sh) manages two environments (`pixi.toml`); the C and C++ compilers and the CUDA toolkit come from
the system.

| Environment | What for                                                                                                   | Tasks                                                                         |
| ----------- | ---------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------------------------- |
| `default`   | building and testing p2v-bme: CMake, cxxopts, GoogleTest, Google Benchmark, Python, pytest, pre-commit, ty | `build`, `test`, `test-asan`, `test-python`, `make_test_fasta`                |
| `paper`     | the paper's scripts and plots: NumPy, pandas, Matplotlib, FastME, FastTree, VeryFastTree                   | `check-paper` and one task per script ([paper/README.md](../paper/README.md)) |

| Task                            | Runs                                                                                                                                  |
| ------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------- |
| `pixi run build`                | `cmake --preset default` and `cmake --build --preset default -j 8`: `build/p2v-bme`, the tests and the paper's tools                  |
| `pixi run test`                 | `build`, then `ctest --preset default` (8 tests at a time)                                                                            |
| `pixi run test-asan`            | the CPU tests under AddressSanitizer and UndefinedBehaviorSanitizer, in `build-asan/`                                                 |
| `pixi run test-python`          | `build`, then the Python tests (`tests/python`, pytest); extra arguments go to pytest, e.g. `-m "not gpu"`                            |
| `pixi run make_test_fasta`      | `build`, then `build/make_test_fasta`: a simulated alignment and its true tree ([examples](../examples/README.md#how-they-were-made)) |
| `pixi run -e paper check-paper` | `build`, then every paper script and plot on tiny inputs, on the CPU                                                                  |

## CMake

Four configure presets (`CMakePresets.json`), each with a build preset of the same name, and a test preset for all but
`release`:

| Preset    | Folder           | Build                                                                                                          |
| --------- | ---------------- | -------------------------------------------------------------------------------------------------------------- |
| `default` | `build/`         | the package, the tests and the paper's tools; the GPU code if CUDA is found                                    |
| `ci`      | `build-ci/`      | CPU only, warnings as errors, as in CI                                                                         |
| `asan`    | `build-asan/`    | CPU only, with AddressSanitizer and UndefinedBehaviorSanitizer                                                 |
| `release` | `build-release/` | p2v-bme alone, as the prebuilt binary: GPU code for compute capability 7.5 to 12.0 and PTX, runtimes linked in |

```bash
pixi run cmake --preset ci
pixi run cmake --build --preset ci -j 8
pixi run ctest --preset ci
```

Options, set at configure time (`-D<option>=<value>`); later builds in the same folder keep them:

| Option                     | Default                 | Meaning                                                                  |
| -------------------------- | ----------------------- | ------------------------------------------------------------------------ |
| `P2VBME_CUDA`              | `ON` if CUDA is found   | build the GPU code                                                       |
| `CMAKE_CUDA_ARCHITECTURES` | `native`                | the GPUs to compile for: those of the build machine, or e.g. `90` (H100) |
| `P2VBME_BUILD_PAPER_TOOLS` | `ON` in the presets     | build the paper's tools (`paper/tools`) into `build/`                    |
| `BUILD_TESTING`            | `ON`                    | build the tests                                                          |
| `P2VBME_STATIC_RUNTIME`    | `OFF` (`release`: `ON`) | link the CUDA runtime, libstdc++, libgcc and libgomp into p2v-bme        |

The version is set in `project()` of `CMakeLists.txt`; `pixi.toml` and `CITATION.cff` repeat it, so a release changes
all three.

## Tests

- **C++:** GoogleTest, one file per source file (`tests/test_<source>.cpp` for `src/<source>.cpp`), run by ctest. The
  GPU tests have the label `gpu` and run one at a time; `ctest --preset default -LE gpu` runs the others, `-L gpu` only
  them.
- **Command line:** `tests/python` (pytest): the options, the outputs and the examples (`examples/`). ctest runs them as
  `cli` (CPU) and, in CUDA builds, `cli_gpu` (the `gpu` marker); `pixi run test-python` runs them directly.
- **Inputs:** the tests simulate their alignments; `pixi run make_test_fasta` (`tests/make_test_fasta.cpp`) writes one
  with its true tree ([examples/README.md](../examples/README.md#how-they-were-made)).

## Formatting and checks

[pre-commit](https://pre-commit.com) runs the hooks of `.pre-commit-config.yaml` on each commit: trailing whitespace,
isort and black (Python), clang-format (C, C++, CUDA; style in `.clang-format`), prettier (Markdown and JSON), and ty
(the Python types); before each push it runs the CPU tests of the command line. To install the hooks:

```bash
pixi run pre-commit install
```

## Continuous integration

On each pull request and push to main:

- pre-commit.ci runs the hooks, except ty and pytest, and pushes its fixes to the pull request;
- GitHub Actions (`.github/workflows`), on CPU runners: `ci-cpp.yml` builds the `ci` preset and runs the C++ tests;
  `ci-python.yml` runs ty, the command-line tests (`-m "not gpu"`) and `check-paper`.
- `cd-binary.yml` builds the prebuilt binary (the `release` preset, in NVIDIA's CUDA 12.8 image on Rocky Linux 8) and
  runs the examples with it on a runner without CUDA: on each push to main as a workflow artifact, and on each published
  release as files attached to the release.

The runners have no GPU: run the GPU tests locally (`pixi run test`) before a pull request that touches the GPU code.
