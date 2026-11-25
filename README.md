# VecME

VecME infers phylogenetic trees from aligned DNA sequences by balanced minimum evolution (BME), for up to millions of
sequences. Trees are encoded as Phylo2Vec integer vectors: every single-integer rearrangement of a window of the tree is
scored in one pass, and a multi-scale search over windows of condensed subtrees, from local ones to the whole tree, runs
these passes in minibatches on GPUs or CPU cores. The standard bootstrap gives each branch of the tree its transfer
(TBE) and exact supports.

The command-line program is `vecme`: an aligned FASTA file in, an unrooted Newick tree and a JSON report of the run out.

## Installation

### Prebuilt binary

Each [release](https://github.com/sbhattlab/vecme/releases) has a Linux binary with its SHA-256 checksum. The latest:

```bash
curl -fsSLO https://github.com/sbhattlab/vecme/releases/latest/download/vecme-linux-x86_64.tar.gz
tar xzf vecme-linux-x86_64.tar.gz
vecme-linux-x86_64/vecme --version
```

| Needs  |                                                                                                                                                 |
| ------ | ----------------------------------------------------------------------------------------------------------------------------------------------- |
| System | Linux x86-64 with glibc 2.28 or newer (e.g. RHEL 8, Ubuntu 20.04)                                                                               |
| GPU    | NVIDIA, compute capability 7.5 to 12.0 (Turing to Blackwell), driver 525 or newer; newer GPUs run its PTX, with a driver for CUDA 12.8 or newer |
| CPU    | nothing more: without a usable GPU, vecme says why and runs on the CPU                                                                          |

Otherwise, build it from source.

### From source

#### Prerequisites

- Linux on x86-64 (other systems: issue [#14](https://github.com/sbhattlab/vecme/issues/14))
- a C and C++17 compiler with OpenMP (e.g. GCC)
- [pixi](https://pixi.sh), which installs the other build dependencies (CMake included):

  ```bash
  curl -fsSL https://pixi.sh/install.sh | bash
  ```

- optional, for the GPU code: the CUDA toolkit 12 or newer and an NVIDIA GPU; without them, vecme runs on the CPU

#### Build

```bash
git clone https://github.com/sbhattlab/vecme.git
cd vecme
pixi run build    # build/vecme, with the GPU code if CUDA is found
pixi run test     # optional: the tests
```

`pixi run build` configures and builds the `default` CMake preset (`CMakePresets.json`). The GPU code is compiled for
the GPUs of the build machine; for another card, set its architecture once, e.g. for an H100:

```bash
pixi run cmake --preset default -DCMAKE_CUDA_ARCHITECTURES=90
pixi run build
```

To put `vecme` on your `PATH`, install it, e.g. into `~/.local/bin`:

```bash
pixi run cmake --install build --prefix ~/.local
```

The other presets and options are in [docs/development.md](docs/development.md).

## Quick start

The repository has two small simulated alignments with their true trees ([examples/README.md](examples/README.md)). From
the repository root:

```bash
build/vecme -s examples/jc69_1k.fa -m JC69 -p jc69_1k_gpu -a gpu    # on the GPU
build/vecme -s examples/jc69_1k.fa -m JC69 -p jc69_1k_cpu -a cpu    # on the CPU
```

Each writes `<prefix>.nwk`, the tree, `<prefix>.json`, the report of the run, and `<prefix>.log`, what it printed; if
they exist, it stops (`--overwrite` replaces them). Without `-a`, vecme uses the GPU if it finds one, else the CPU. With
10 bootstrap replicates:

```bash
build/vecme -s examples/jc69_1k.fa -m JC69 -p jc69_1k_b10 -b 10
```

This also writes `jc69_1k_b10.support.nwk`, the tree with the supports of its branches, and `jc69_1k_b10.boottrees`, the
replicates' trees.

## Usage

```text
vecme -s ALIGNMENT -m MODEL [options]
```

| Option                 | Meaning                                                                                                                              |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `-s, --alignment FILE` | aligned DNA sequences: FASTA, or `.maple`                                                                                            |
| `-m, --model MODEL`    | the distances' model: `JC69`, `K80`, `F81`, `F84`, `TN93`, `LogDet`, `GTR` (rates fitted) or `GTR+GC` (rates and gamma shape fitted) |
| `-p, --prefix STR`     | the outputs' prefix (default: the alignment's path)                                                                                  |
| `-a, --accelerator`    | `cpu`, `gpu` or `auto` (default)                                                                                                     |
| `-d, --devices`        | GPUs (`2`: the first two; `0,2`) or CPU threads                                                                                      |
| `-b, --bootstrap B`    | B bootstrap replicates, and the supports of the tree's branches                                                                      |
| `--seed N`             | the seed of the run (default 1)                                                                                                      |
| `-q, --quiet`          | no progress on stdout (it still goes to `<prefix>.log`)                                                                              |

`vecme --help` lists these options and `vecme --help-all` all of them. The [documentation](docs/README.md) describes
each option ([docs/usage.md](docs/usage.md)), the models ([docs/models.md](docs/models.md)) and the outputs
([docs/outputs.md](docs/outputs.md)).

## Reproducing the paper

The paper's analyses are scripts run through pixi tasks: [paper/README.md](paper/README.md) lists them in the order they
run, and [data/README.md](data/README.md) the data they download and write.

## Citation

If you use VecME, please cite:

```bibtex
@article{bhatt2026vecme,
  title   = {VecME: phylogenetic inference with uncertainty for millions of sequences},
  author  = {Bhatt, Samir and Scheidwasser, Neil and Penn, Matthew J. and Andersen, Frederik M. and
             Ferguson, Neil M. and Donnelly, Christl A. and Kraemer, Moritz U. G. and Suchard, Marc A. and
             Aarestrup, Frank M. and Khurana, Mark P. and Zhu, Harrison B. H. and Duch{\^e}ne, David A.},
  year    = {2026},
  note    = {Under review}
}
```

To cite the software itself, see [CITATION.cff](CITATION.cff) (GitHub's "Cite this repository").

## License

[LGPL-3.0](LICENSE)
