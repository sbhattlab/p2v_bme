# Outputs

| File                   | Written by                      | Contents                      |
| ---------------------- | ------------------------------- | ----------------------------- |
| `<prefix>.nwk`         | runs with `-s`, except `-b A-B` | the tree                      |
| `<prefix>.json`        | every run                       | the run report                |
| `<prefix>.support.nwk` | `-b B`, `--supports`            | the tree with branch supports |
| `<prefix>.boottrees`   | `-b`                            | the bootstrap trees           |
| `<prefix>.log`         | every run                       | what the run prints           |

The default prefix is the alignment's path (with `--supports`, the tree's). Existing files are overwritten. Progress
goes to stdout, warnings and errors to stderr; the exit status is 0 on success, 1 on an error and 2 on a usage error.

## Tree (`<prefix>.nwk`)

Unrooted binary Newick (three subtrees at the top level), with BME branch lengths in substitutions per site; negative
lengths are written as 0 unless `--no-neg-as-zero`. Identical sequences set aside before the search are reattached to
their group with zero-length branches.

## Supports (`<prefix>.support.nwk`)

The same tree with each internal branch labelled `TBE/exact`, in %, e.g. `(t2:0.01,t3:0.03)98.5/87:0.05`: the transfer
bootstrap expectation ([Lemoine et al. 2018](https://doi.org/10.1038/s41586-018-0043-0)) and Felsenstein's bootstrap
proportion ([Felsenstein 1985](https://doi.org/10.1111/j.1558-5646.1985.tb00420.x)). Branches to reattached identical
sequences have no label, nor, with `--supports`, branches of TREE outside the leaves of the trees in FILE.

## Bootstrap trees (`<prefix>.boottrees`)

One Newick tree per replicate and line, without branch lengths, on the searched sequences (identical copies left out);
the input of `--supports`.

## Report (`<prefix>.json`)

| Field                             | Contents                                                                                                                                            |
| --------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| `version`, `command_line`, `date` | the program's version, the arguments, the end time (UTC)                                                                                            |
| `substitution_model`              | `name`; for GTR+GC also `rates` (AC, AG, AT, CG, CT, GT), `alpha` and `fit` (pairs used, iterations, convergence, log-likelihood, base frequencies) |
| `bme_length`                      | the tree length, the sum of the BME branch lengths (below); `null` without branch lengths                                                           |
| `rounds_run`, `stop_reason`       | search rounds, and why the search stopped: `converged`, `max_rounds`, `time_limit` or `none`                                                        |
| `bootstrap`                       | with `-b`: the replicates and their trees file; with `-b B`, the mean TBE and exact supports                                                        |
| `time`, `warnings`                | wall time (s), warnings                                                                                                                             |
| `details`                         | with `--report-full`: input size, settings as used, devices, time per stage, peak memory, gain per round                                            |

With `-b A-B`, the search fields add up over the replicates. With `--supports`, the report has `supports` instead: the
files, the numbers of trees and leaves, and the mean TBE and exact supports.

## BME length

`bme_length` is exact for trees of up to 64 sequences. Larger trees are scored from at most 64 representative sequences
per clade, so it is an estimate that depends on the tree's rooting. On `examples/jc69_1k`, five runs that found the same
topology reported 110.1847 to 110.1893 (4e-5 relative), against an exact length of 111.487 (1.2% higher): small
differences between reported lengths do not rank trees.

## Log (`<prefix>.log`)

Progress, warnings and errors as printed, in order; the progress also with `-q`. The log is opened first, so an
unwritable prefix fails at once. A failed run leaves only its log, which ends with the error; the other outputs are
written when the run ends (`.boottrees`: from the first replicate).
