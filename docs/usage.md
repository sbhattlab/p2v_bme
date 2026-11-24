# Usage

```text
p2v-bme -s ALIGNMENT -m MODEL [options]       infer a tree
p2v-bme -t TREE --supports FILE [-p PREFIX]   supports of TREE's branches over the trees in FILE
```

The options are grouped as in `p2v-bme --help-all`; the files they write are in [outputs.md](outputs.md).

## Input

`-s` takes an aligned DNA FASTA file, or a MAPLE file if its name ends in `.maple`. A, C, G and T (or U), in either
case, are bases; anything else (gaps, `N`, other IUPAC codes) is missing data. Sequence names (the first word of each
header) must be unique. Gzipped files are not read yet ([#12](https://github.com/sbhattlab/p2v_bme/issues/12)):
decompress them first.

## Core options

| Option                      | Default        | Meaning                                                                                                                                                 |
| --------------------------- | -------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `-s, --alignment FILE`      | required       | the alignment                                                                                                                                           |
| `-m, --model MODEL`         | required       | `JC69`, `K80`, `F81`, `F84`, `TN93`, `LogDet` or `GTR+GC` ([models.md](models.md))                                                                      |
| `-p, --prefix STR`          | alignment path | output prefix                                                                                                                                           |
| `-a, --accelerator`         | `auto`         | `cpu`, `gpu`, or `auto` (the GPU if one is found, else the CPU)                                                                                         |
| `-d, --devices`             | `auto`         | GPU: `auto` (the first), `N` (the first N) or a list (`0,2`; `1,` for GPU 1 only); CPU: `auto` (all cores but one, or `OMP_NUM_THREADS`) or `N` threads |
| `-b, --bootstrap B\|A-B`    | `0`            | `B` bootstrap replicates after the tree; `A-B`: replicates A to B only, without the tree                                                                |
| `--seed N`                  | `1`            | seed of the start tree, the GTR+GC fit and the bootstrap                                                                                                |
| `-q, --quiet`               |                | no progress on stdout (it still goes to `<prefix>.log`)                                                                                                 |
| `-v, --version`             |                | print the version                                                                                                                                       |
| `-h, --help` / `--help-all` |                | print the core / all options                                                                                                                            |

The results do not depend on the number of threads or GPUs.

## Substitution model options

| Option              | Default  | Meaning                                                                               |
| ------------------- | -------- | ------------------------------------------------------------------------------------- |
| `--gtr-fit-pairs N` | `100000` | GTR+GC: sequence pairs used to fit the rates and alpha ([models.md](models.md#gtrgc)) |

## Search options

| Option                   | Default  | Meaning                                                                                    |
| ------------------------ | -------- | ------------------------------------------------------------------------------------------ |
| `--max-rounds N`         | `8`      | search rounds; `0` scores the start tree                                                   |
| `--tol TOL`              | `1e-6`   | stop when a round gains at most TOL x the total gain; `0`: only when a round gains nothing |
| `--time-limit S`         | `0`      | no new round after S seconds (`0`: no limit)                                               |
| `--representative-cap N` | `64`     | representative sequences per clade in a window (1-256); larger is more exact and slower    |
| `--precision`            | `single` | `single` or `double` (CPU only)                                                            |

## Start tree options

| Option            | Default  | Meaning                                                                                                |
| ----------------- | -------- | ------------------------------------------------------------------------------------------------------ |
| `-t, --tree TREE` | `sample` | `sample` (sample-and-place), `random` (Yule-Harding), or a binary Newick tree on the alignment's names |
| `--sample-size N` | `0`      | sample size of sample-and-place; `0`: min(100,000, N / 10)                                             |

Sample-and-place builds and searches a tree on a random sample of the sequences, then attaches every other sequence next
to its nearest sampled sequence (p-distance). Alignments of fewer than 16 sequences start from a random tree.

## Branch lengths and output options

| Option                | Default | Meaning                                              |
| --------------------- | ------- | ---------------------------------------------------- |
| `--no-branch-lengths` |         | no BME branch lengths                                |
| `--no-neg-as-zero`    |         | keep negative branch lengths (default: written as 0) |
| `--report-full`       |         | add the run's details to the report                  |

## Bootstrap options

| Option            | Default | Meaning                                                                                       |
| ----------------- | ------- | --------------------------------------------------------------------------------------------- |
| `--supports FILE` |         | with `-t TREE` only: supports of TREE's branches over the trees in FILE (one Newick per line) |

`-b B` runs B nonparametric bootstrap replicates, each searched like the tree, without branch lengths and with the model
parameters fitted once on the alignment. The tree's branches get TBE and Felsenstein supports
([outputs.md](outputs.md#supports-prefixsupportnwk)). Replicate r has the same seed in `-b B` and `-b A-B`, so a
bootstrap can be split into jobs (below).

## Input options

| Option       | Default | Meaning                                                                                                   |
| ------------ | ------- | --------------------------------------------------------------------------------------------------------- |
| `--no-dedup` |         | keep all identical sequences (default: search two per group, reattach the rest with zero-length branches) |

## Examples

### Devices

```bash
p2v-bme -s aln.fa -m JC69 -a gpu            # the first GPU
p2v-bme -s aln.fa -m JC69 -a gpu -d 2       # the first two GPUs
p2v-bme -s aln.fa -m JC69 -a gpu -d 0,2     # GPUs 0 and 2
p2v-bme -s aln.fa -m JC69 -a gpu -d 1,      # GPU 1 only
p2v-bme -s aln.fa -m JC69 -a cpu -d 16      # 16 CPU threads
```

GPU numbers count the GPUs visible to CUDA (`CUDA_VISIBLE_DEVICES`).

### A bootstrap in several jobs

`-b 100` as four jobs, then the supports over their 100 trees:

```bash
p2v-bme -s aln.fa -m GTR+GC -p tree             # tree.nwk
p2v-bme -s aln.fa -m GTR+GC -b 1-25 -p boot1    # boot1.boottrees
p2v-bme -s aln.fa -m GTR+GC -b 26-50 -p boot2
p2v-bme -s aln.fa -m GTR+GC -b 51-75 -p boot3
p2v-bme -s aln.fa -m GTR+GC -b 76-100 -p boot4
cat boot1.boottrees boot2.boottrees boot3.boottrees boot4.boottrees > boot.nwk
p2v-bme -t tree.nwk --supports boot.nwk -p tree_b100    # tree_b100.support.nwk
```

Each job needs the same options and `--seed` as the tree.

### Supports from repeat searches

```bash
p2v-bme -s aln.fa -m JC69 -p seed1
p2v-bme -s aln.fa -m JC69 --seed 2 -p seed2
p2v-bme -s aln.fa -m JC69 --seed 3 -p seed3
cat seed2.nwk seed3.nwk > repeats.nwk
p2v-bme -t seed1.nwk --supports repeats.nwk -p seed1_repeats    # seed1_repeats.support.nwk
```

`--supports` takes any trees on the same leaves, e.g. a true tree ([examples](../examples/README.md#running-p2v-bme)).
