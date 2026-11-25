# Reproducing the paper

The paper's analyses are Python scripts in this folder, run from the repository root through pixi tasks of the `paper`
environment, which also has the comparison programs (FastME, FastTree, VeryFastTree). Each script writes into `data/`
([data/README.md](../data/README.md)); the figures are scripts in `plots/`.

```bash
pixi run build              # build/vecme and the paper's tools (paper/tools: bme_scores, eval_cmaple_rf, ...)
pixi install -e paper
pixi run -e paper check-paper
```

`check-paper` runs every script and plot on tiny simulated inputs, on the CPU, in about a minute (CI runs it). The full
runs take hours to days on GPUs. Every script skips the runs whose outputs exist (`--overwrite` reruns them), so it can
be stopped and started again, and its options split the work (`--sets`, `--sizes`, `--replicates`, `--steps`, ...). Each
task passes its arguments to the script, and each script's `--help` lists its options, inputs and outputs:

```bash
pixi run -e paper run_benchmarks --help
```

The scripts that run vecme use the GPU if one is found (`--accelerator`, `--devices`, as vecme's `-a` and `-d`).

The tasks, in the order they run:

1. [Download the datasets](#download-the-datasets): `download_data`
2. [Simulate the datasets](#simulate-the-datasets): `simulate_data`
3. [Benchmarks](#benchmarks): `run_benchmarks`
4. [Comparison programs](#comparison-programs): `run_baselines`
5. [FastME comparison](#fastme-comparison): `run_fastme`
6. [Bootstrap calibration](#bootstrap-calibration): `run_calibration`
7. [GTR+GC fit recovery](#gtrgc-fit-recovery): `run_gtr_fit_recovery`
8. [SILVA 144](#silva-144): `run_silva`
9. [SARS-CoV-2 genomes](#sars-cov-2-genomes): `run_viridian`
10. [Tables](#tables): `make_tables`
11. [Figures](#figures): `plots/`

## Download the datasets

- **Command:** `pixi run -e paper download_data`
- **Requires:** `curl`; about 25 GB of disk (SILVA's alignment: 20 GB)
- **Inputs:** the deposits' URLs (RNASim, SILVA 144, CMAPLE, Viridian)
- **Outputs:** `data/datasets/<source>/`, with `download_<source>.json`

Fetches each file unless a copy with the deposit's size and MD5 exists, and extracts from the zips the files the scripts
read. `--sources` picks the datasets; `--sources silva144-arb` adds SILVA's ARB database (7.2 GB), needed only by
`run_silva`'s gtdb step; `--dry-run` prints what would be fetched.

## Simulate the datasets

- **Command:** `pixi run -e paper simulate_data`
- **Requires:** `pixi run build` (`sim_jc69_fast`); `download_data --sources rnasim`
- **Inputs:** `data/datasets/rnasim/aln_dna.fa.xz`, `true_topo.tree`
- **Outputs:** `data/datasets/<set>/<n>/rep<r>.fa`, `rep<r>.true.nwk`, `rep<r>.random_start.nwk`; the same files in
  `data/datasets/fastme/<set>/<n>/`

The simulated sets of the scaling, comparison, sequence-length, representative, FastME and calibration tables, with the
paper's generators and seeds: RNASim subsets (`rnasim`), JC69 sequences on their trees (`jc10k`, `jc1k`, `jc100k`,
`jc1000k`) and on Yule trees (`yule10k`), from 1,000 to 10^7 sequences (`--sets`, `--sizes`, `--replicates`). The FastME
comparison's sets (`fastme`) are subsets, as the paper's: `jc10k` and `rnasim` of the 10^4-sequence sets, `jc1m` and
`rna1m` of the 10^6-sequence sets.

## Benchmarks

- **Command:** `pixi run -e paper run_benchmarks`
- **Requires:** `pixi run build` (`vecme`, `bme_scores`, `eval_cmaple_rf`); `simulate_data`;
  `download_data --sources cmaple` (pandemic sets)
- **Inputs:** `data/datasets/<set>/<n>/rep<r>.fa`, `rep<r>.true.nwk`; `data/datasets/cmaple/simulated/`
- **Outputs:** `data/results/<experiment>/<set>/<n>/rep<r>_seed<s>.{nwk,json}` and their scores against the true tree

vecme (`-m JC69`) on the simulated sets, scored against their true trees. `--experiments`: `scaling`, `random_start`,
`seqlen`, `representatives`, `round_limit`, `pandemic` (default: all), each with the paper's sets, sizes, replicates and
seeds.

## Comparison programs

- **Command:** `pixi run -e paper run_baselines --run FAMILY ...`, or `--source SOURCE` to import recorded runs
- **Requires:** `pixi run build` (`eval_cmaple_rf`); for `--run`: `simulate_data`, `download_data --sources cmaple`, and
  for `--run dipper` the DIPPER program (`--dipper`; not in the pixi environments)
- **Inputs:** for `--run`, the simulated sets and `data/datasets/cmaple/simulated/`; for `--source`, the recorded runs
  (a git repository, with `--ref`, or a folder)
- **Outputs:** `data/results/baselines/{default,default_100k,default_yule,random_start,dipper,maple}/`; with `--source`,
  `PROVENANCE.json` (commit, and each imported file's recorded path and checksum)

FastTree, VeryFastTree, FastME, DIPPER, MAPLE and CMAPLE, each tree scored against the true tree. `--run` takes the
families `default`, `random_start`, `dipper` and `maple`. The programs do not depend on vecme, so recorded runs can be
imported unchanged instead (`--source`), in the same layout. One of `--run` and `--source` is needed.

## FastME comparison

- **Command:** `pixi run -e paper run_fastme`
- **Requires:** `pixi run build` (`vecme`, `bme_scores`); `simulate_data --sets fastme`
- **Inputs:** `data/datasets/fastme/<set>/<n>/rep<r>.fa`, `rep<r>.true.nwk`, `rep<r>.random_start.nwk`
- **Outputs:** `data/results/fastme/<set>/<n>/rep<r>_*`, `rep<r>.case.json`; `data/results/fastme/cases.jsonl`

vecme and FastME on the same JC69 distance matrix, from their own and from the same random start trees, every tree
rescored exactly.

## Bootstrap calibration

- **Command:** `pixi run -e paper run_calibration`
- **Requires:** `pixi run build`; `simulate_data` (`rnasim` at 10^5 and 10^6, `yule10k` at 10^6, replicate 0);
  `download_data --sources cmaple`
- **Inputs:** `data/datasets/<set>/<n>/rep0.fa`, `rep0.true.nwk`; `data/datasets/cmaple/simulated/` (the 100K set and
  CMAPLE's and MAPLE's trees)
- **Outputs:** `data/results/calibration/<set>/<n>/` (trees, supports, per-branch tables); the summaries in
  `data/results/calibration/`

The supports of a reference tree's branches from 64 bootstrap replicates and from 8 repeat searches, against the true
tree. `--steps ref boot seed score`; `--replicates A-B` spreads the bootstrap over jobs.

## GTR+GC fit recovery

- **Command:** `pixi run -e paper run_gtr_fit_recovery`
- **Requires:** `pixi run build` (`vecme`)
- **Inputs:** none (alignments simulated with the paper's settings and seeds)
- **Outputs:** `data/results/gtr_fit/runs/<alignment>.{nwk,json}`, `data/results/gtr_fit/recovery.tsv`

vecme's GTR+GC fit (`-m GTR+GC --max-rounds 0`, on the CPU) on alignments of 1,000 sequences simulated under GTR with
gamma rates, compared with the true shape and exchangeabilities.

## SILVA 144

- **Command:** `pixi run -e paper run_silva`
- **Requires:** `pixi run build` (`vecme`, `reattach_left_out`); `download_data --sources silva144`; for the gtdb step,
  SILVA's ARB database dumped as text with ARB's tools (`--arb-dump`; ARB is not in the pixi environments)
- **Inputs:** `data/datasets/silva144/SILVA_144_SSURef_tax_silva_full_align_trunc.fasta.gz`
- **Outputs:** `data/results/silva/`

`--steps`, in order: `distinct` (the columns and sequences searched), `tree` (`-m GTR+GC`), `replicates` (the bootstrap;
`--replicates A-B` spreads it over jobs), `supports`, `records`, `leaveout`, `gtdb`.

## SARS-CoV-2 genomes

- **Command:** `pixi run -e paper run_viridian`
- **Requires:** `pixi run build`; `download_data --sources viridian`
- **Inputs:** `data/datasets/viridian/12733488/extracted/*.maple`, `data/datasets/viridian/15546527/*.jsonl.zip`,
  `data/datasets/viridian/alias_key.json` (the Pango alias key, for the records step)
- **Outputs:** `data/results/viridian/`

`--steps`, in order: `maple` (MAPLE's tree and each genome's lineage and date), `tree` (`-m JC69`), `replicates` (the
bootstrap; `--replicates A-B`), `repeats` (repeat searches), `supports`, `records`.

## Tables

- **Command:** `pixi run -e paper make_tables`
- **Requires:** the `run_*` scripts' results
- **Inputs:** `data/results/`
- **Outputs:** `data/tables/tableS<N>_<name>.md` (one Markdown table per supplementary table) and
  `data/tables/<name>.tsv` (the figures' data)

Every table of the paper (`--tables`; default: all). A table whose results are missing is skipped, with a note.

## Figures

- **Command:** `pixi run -e paper <script>`, e.g. `pixi run -e paper fig2a_scaling_time`; `pixi run -e paper fig2`,
  `fig3` and `fig4` draw a figure with its supplementary figures (Fig. 2 with Figs S1-S4, Fig. 3 with Fig. S5)
- **Requires:** `make_tables`
- **Inputs:** `data/tables/<name>.tsv`
- **Outputs:** `img/fig2/` (Fig. 2 with Figs S1-S4), `img/fig3/` (Fig. 3 with Fig. S5) and `img/fig4/` (Fig. 4):
  `<figure>.pdf`, `<figure>.svg`, `<figure>.png`

| Script                            | Figure                                                         | Reads                                       |
| --------------------------------- | -------------------------------------------------------------- | ------------------------------------------- |
| `fig2a_scaling_time`              | 2a: time against the number of sequences                       | `scaling.tsv`, `random_tree.tsv`            |
| `fig2b_scaling_rf_error`          | 2b: RF error against the number of sequences                   | `scaling.tsv`, `random_tree.tsv`            |
| `fig3a_pandemic_time`             | 3a: time of VecME, MAPLE and CMAPLE, pandemic sets             | `maple.tsv`                                 |
| `fig3b_pandemic_rf_error`         | 3b: RF error of the three programs by branch length, pandemic  | `maple.tsv`                                 |
| `fig3c_sarscov2_resolution`       | 3c: resolution of the SARS-CoV-2 tree by sampling month        | `viridian_resolution.tsv`                   |
| `fig4a_silva_clades`              | 4a: the SILVA 144 tree reduced to the domains and major clades | `silva_clades.tsv`, `silva_clade_paths.tsv` |
| `fig4b_silva_support`             | 4b: SILVA 144 branches by support class and split size         | `silva_support.tsv`                         |
| `figS1_3_scaling`                 | S1-S3: time, RF error, host and GPU memory of every method     | `scaling_methods.tsv`                       |
| `figS4_seqlen_effect`             | S4: RF error on 1,000 and 10,000 sites                         | `scaling_methods.tsv`                       |
| `figS5a_sarscov2_variants`        | S5a: supports of the named SARS-CoV-2 variants                 | `viridian_variants.tsv`                     |
| `figS5b_sarscov2_shared_branches` | S5b: branches shared by our tree and MAPLE's                   | `viridian_substitutions.tsv`                |

`figS1_3_scaling` writes one figure per dataset (S1: RNASim, S2: JC10k, S3: random-tree JC), e.g.
`img/fig2/figS2_scaling.pdf`.
