# Examples

Two simulated alignments with their true trees, to try p2v-bme and to see what it recovers.

| Files                            | Sequences x sites | Model          | Median p-distance |
| -------------------------------- | ----------------- | -------------- | ----------------- |
| `jc69_1k.fa`, `jc69_1k_true.nwk` | 1,000 x 1,500     | JC69           | 0.59              |
| `gtr_1k.fa`, `gtr_1k_true.nwk`   | 1,000 x 5,000     | GTR+GC (below) | 0.38              |

`gtr_1k`: rates AC, AG, AT, CG, CT, GT = 1, 4, 1, 1, 4, 1; base frequencies A, C, G, T = 0.30, 0.20, 0.27, 0.23; gamma
shape alpha = 0.5. Median p-distance: over 4,000 random pairs of sequences, on the sites where both have a base.

## How they were made

With `make_test_fasta` (`tests/make_test_fasta.cpp`, built with p2v-bme), from the repository root:

```bash
pixi run make_test_fasta examples/jc69_1k 1000 1500 1
pixi run make_test_fasta examples/gtr_1k 1000 5000 2 --rates 1,4,1,1,4,1 --freqs 0.30,0.20,0.27,0.23 --alpha 0.5
```

- Tree (seed: the last positional argument): pairs of subtrees joined uniformly at random (Yule-Harding topologies),
  each branch length uniform in [0.01, 0.1] substitutions per site; written unrooted to `*_true.nwk`.
- Sites (seed + 1): simulated down the tree, with the rate matrix normalised to one substitution per unit time and each
  site's rate drawn from Gamma(alpha, 1 / alpha) (`gtr_1k`; `jc69_1k` has no rate variation).
- Sequences t0 ... t999. For the command-line tests, t3 has an `N` at site 6 and t4 gaps at sites 11-14.
- The tool also writes `*_dups.fa` and `*.maple`, inputs of the command-line tests; they are not kept here.

## Running p2v-bme

```bash
p2v-bme -s examples/jc69_1k.fa -m JC69 -p jc69_1k
p2v-bme -s examples/gtr_1k.fa -m GTR+GC -p gtr_1k
p2v-bme -s examples/jc69_1k.fa -m JC69 -b 10 -p jc69_1k_b10
```

Without `-p`, the outputs go next to the alignment (`examples/jc69_1k.fa.nwk`, ...).

To compare a tree with the true one, give it to `--supports` as the only replicate: each branch of the true tree gets
100 (found) or 0 (missing), so the report's `mean_exact` is 100 minus the normalised Robinson-Foulds (RF) distance, in
%:

```bash
p2v-bme -t examples/jc69_1k_true.nwk --supports jc69_1k.nwk -p jc69_1k_vs_true
```

## Results

p2v-bme 0.1.0 (67365ef), default options (`--seed 1`).

|                                              | `jc69_1k`     | `gtr_1k`                                 |
| -------------------------------------------- | ------------- | ---------------------------------------- |
| RF to the true tree                          | 0             | 0.10% (1 of 997 branches)                |
| BME length                                   | 110.186       | 115.031                                  |
| Fitted alpha (true 0.5)                      |               | 0.416                                    |
| Fitted rates (true 0.5, 2, 0.5, 0.5, 2, 0.5) |               | 0.417, 2.045, 0.464, 0.505, 2.087, 0.482 |
| `-b 10`: mean support, exact / TBE           | 99.65 / 99.86 | 99.70 / 99.84                            |
| Time, CPU                                    | 0.5 s         | 1.7 s                                    |
| Time, GPU                                    | 0.4 s         | 1.4 s                                    |
| Time, CPU, `-b 10`                           | 8 s           | 10 s                                     |

- The fitted rates are scaled to mean 1, so the true ones are 1, 4, 1, 1, 4, 1 scaled to 0.5, 2, 0.5, 0.5, 2, 0.5.
- `--seed` 1 to 5 give the same RF (0 and 0.10%), alpha 0.414-0.416 and BME lengths 110.185-110.189 and 115.03-115.34.
  The BME length of a tree this size is computed from up to 64 representatives per clade, so it differs slightly between
  runs even when the tree is the same; for `gtr_1k`, the fitted distances also change with the seed.
- The CPU and GPU runs give the same topologies.
- Times: wall time of the command, median of 3 runs; CPU: AMD Ryzen Threadripper PRO 5995WX, 127 threads (the default),
  shared with other jobs; GPU: one NVIDIA RTX A6000. On 4 threads (`-d 4`): 0.6 s and 10 s, of which 9 s is the GTR+GC
  fit.
