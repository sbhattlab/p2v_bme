# Models

`-m` sets the model of the pairwise distances. Names are case-insensitive.

| Model    | Alias | Reference                                                                                  | Parameters                                 |
| -------- | ----- | ------------------------------------------------------------------------------------------ | ------------------------------------------ |
| `JC69`   | `JC`  | [Jukes & Cantor 1969](https://doi.org/10.1016/B978-1-4832-3211-9.50009-7)                  |                                            |
| `K80`    | `K2P` | [Kimura 1980](https://doi.org/10.1007/BF01731581)                                          |                                            |
| `F81`    |       | [Felsenstein 1981](https://doi.org/10.1007/BF01734359)                                     | base frequencies                           |
| `F84`    |       | [Felsenstein & Churchill 1996](https://doi.org/10.1093/oxfordjournals.molbev.a025575)      | base frequencies                           |
| `TN93`   | `TN`  | [Tamura & Nei 1993](https://doi.org/10.1093/oxfordjournals.molbev.a040023)                 | base frequencies                           |
| `LogDet` |       | [Lockhart et al. 1994](https://doi.org/10.1093/oxfordjournals.molbev.a040136) (paralinear) |                                            |
| `GTR+GC` |       | [Penn et al. 2025](https://doi.org/10.1007/s11538-024-01403-z)                             | exchangeabilities and gamma shape (fitted) |

- A pair's distance uses the sites where both sequences have a base; gaps and ambiguity codes are skipped.
- Pairs without a shared site get distance 2, and distances are capped at 10 (also where a formula saturates).
- Base frequencies (F81, F84, TN93) are counted over the whole alignment, and the formulas are those of
  [ape](https://github.com/emmanuelparadis/ape)'s `dist.dna`; F84 and TN93 fall back to F81 where the base frequencies
  leave them undefined.
- The GPU computes distances in single precision; `--precision double` runs on the CPU.

## GTR+GC

`+GC` is a continuous gamma, as in [IQ-TREE 3](https://github.com/iqtree/iqtree3)'s AliSim, not IQ-TREE's discrete `+G`,
which p2v-bme refuses; plain `GTR` is not available. The distance is the eigenvalue estimator of Penn et al. (2025).

The six exchangeabilities (mean 1) and the shape alpha are fitted to the alignment before the search, without a tree, by
maximum composite likelihood over `--gtr-fit-pairs` random sequence pairs (default 100,000, drawn with `--seed`; all
pairs if there are fewer). Pairs sharing fewer than 30 sites are skipped. Alpha lies in [0.05, 148] (the upper end means
no rate variation), and a small penalty on the rate variance only matters when the pairs cannot identify alpha. The fit
runs on the CPU in double precision; its values are printed and stored in the report (`substitution_model`), and the
bootstrap replicates reuse them.

p2v-bme warns when the fit does not converge, or when no pair qualifies (the distances then use equal rates and alpha =
1). In single precision, it also warns when alpha is outside [0.05, 1000] or the exchangeabilities span more than
100-fold, the range where single-precision distances were checked against double precision; `--precision double` avoids
the issue.
