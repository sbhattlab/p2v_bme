"""Benchmarks of p2v-bme on simulated alignments (tables scaling, random_tree, baseline_10k, seqlen, maple,
representative_sensitivity; figures scaling-rnasim, scaling-jc10k, scaling-random-tree, seqlen-effect,
method_benchmarks; the round-limit runs of the supplement).

Inputs:  data/datasets/<set>/<n>/rep<r>.fa, rep<r>.true.nwk                     (simulate_data.py)
         data/datasets/cmaple/simulated/aln_<size>_<scale>.maple, tree_<size>_<scale>.nwk   (download_data.py)
Options: --experiments, --sets, --sizes, --replicates, --caps, --round-limits, --accelerator, --devices,
         --exact-max, --score-cap, --datasets-dir, --cmaple-dir, --results-dir, --p2vbme, --bme-scores,
         --eval-cmaple-rf, --overwrite
Flow:    each experiment x set x size x run of the paper's design (dataset replicate r, seed s; see runs_of)
           |
           v
         p2v-bme -s rep<r>.fa -m JC69 --no-dedup --seed s --report-full [+ the experiment's option]
           |      (skipped if done; --no-dedup: every sequence searched, as in the paper)
           |
           v
         RF to the true tree: p2v-bme -t <run>.nwk --supports rep<r>.true.nwk;
         10^6 and 10^7 expanded sequences: also the backbone splits; pandemic sets: eval_cmaple_rf
           |
           v
         seqlen (up to --exact-max sequences): BME lengths of our and the true trees, exact by bme_scores;
         representatives: BME lengths by bme_scores --cap, the paper's evaluator (cap: the number of sequences up to
         --exact-max, i.e. exact, else --score-cap); RF between the tree of each cap and the cap-64 tree
Outputs: data/results/<experiment>/<set>/<n>/<run>.{nwk,json}, <run>.vs_true.{support.nwk,json}
           <run>: rep<r>_seed<s>, + _cap<P> (representatives), + _rounds<k> (round_limit)
         <run>.true_vs.{support.nwk,json}, <run>.backbone.json        expanded alignments (10^6, 10^7)
         <run>.eval.json                                               pandemic sets (eval_cmaple_rf)
         <run>.score.json, rep<r>.true.score.json                      tree lengths (seqlen, representatives)
         <run>.vs_cap64 / .vs_seed2.{support.nwk,json}                 representatives: RF to the cap-64 tree
         data/results/seqlen/diagnose_10k_sequences.jsonl              lengths of our and the true trees
         data/results/representatives/eval/<set>_<n>_rep<r>_P<P>[_seed2].json   lengths and RF to the cap-64 tree
         The seqlen runs on 10^4 sites are the scaling runs of jc10k.

    pixi run -e paper python -m paper.run_benchmarks --experiments scaling --sets rnasim --sizes 1000 5000
"""

from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter
from json import dumps, loads
from pathlib import Path
from subprocess import run
from typing import NamedTuple

from paper._config import (
    BME_SCORES,
    DEFAULT_DATASETS_DIR,
    DEFAULT_RESULTS_DIR,
    EVAL_CMAPLE_RF,
    P2VBME,
)
from paper.utils import (
    branch_table,
    compare_to_truth,
    read_json,
    run_p2vbme,
    supports,
    write_json,
)

MODEL = "JC69"  # the distances of every simulated set
SIZES = (1_000, 5_000, 10_000, 50_000, 100_000, 1_000_000, 10_000_000)
N_REPLICATES = (
    5  # datasets per size, or seeds on one dataset for the expanded alignments
)
EXPANDED_SETS = (
    "rnasim",
    "jc10k",
)  # from 10^6 sequences: clades grown from the 10^5-sequence alignment
EXPANDED_FROM = 1_000_000
BACKBONE_N = 100_000  # sequences of the alignment that the expanded ones grow from
CAPS = (16, 32, 64, 128, 256)
REFERENCE_CAP = 64  # the default cap, which the others are compared with
SECOND_SAMPLE = (
    10  # the second cap-64 run of the sweep: seed s + 10 (the paper's replicate + 11)
)
ROUND_LIMITS = (1, 2)
PANDEMIC_SCALES = (
    "0.00005",
    "0.0001",
    "0.00015",
)  # mean branch lengths, as in the CMAPLE file names
PANDEMIC_SIZES = (100_000, 200_000, 500_000)
PANDEMIC_SEED = 20261105
EXACT_MAX = 10_000  # sequences up to which trees are scored exactly
SCORE_CAP = 1024  # representatives per clade of the paper's evaluator above EXACT_MAX

# Sets and sizes of each experiment in the paper
DESIGN = {
    "scaling": {"rnasim": SIZES, "jc10k": SIZES, "jc1k": (10_000,), "yule10k": SIZES},
    "random_start": {"rnasim": (10_000,), "jc10k": (10_000,)},
    "seqlen": {
        "jc10k": (10_000,),
        "jc100k": (10_000, 100_000),
        "jc1000k": (10_000, 100_000),
    },
    "representatives": {s: (10_000, 100_000, 1_000_000) for s in ("rnasim", "jc10k")},
    "round_limit": {
        s: (100_000, 1_000_000, 10_000_000) for s in ("rnasim", "jc10k", "yule10k")
    },
    "pandemic": {f"cmaple_{scale}": PANDEMIC_SIZES for scale in PANDEMIC_SCALES},
}


class Run(NamedTuple):
    """One p2v-bme run: experiment, set, size, dataset replicate, seed, name, extra options."""

    experiment: str
    set: str
    n: int
    rep: int
    seed: int
    name: str
    options: tuple[str, ...] = ()


def paper_seeds(set_name: str, n: int) -> list[tuple[int, int]]:
    """Gives the paper's runs of one dataset size: (set, n) -> [(dataset replicate, seed)].

    The expanded alignments exist once: five seeds on replicate 0; otherwise five replicates, seed replicate + 1.
    """
    if set_name in EXPANDED_SETS and n >= EXPANDED_FROM:
        return [(0, s) for s in range(1, N_REPLICATES + 1)]
    return [(r, r + 1) for r in range(N_REPLICATES)]


def runs_of(
    experiment: str, set_name: str, n: int, caps: list[int], round_limits: list[int]
) -> list[Run]:
    """Lists the paper's runs of one experiment, set and size: (experiment, set, n) -> runs."""
    if experiment == "pandemic":
        return [
            Run(experiment, set_name, n, 0, PANDEMIC_SEED, f"rep0_seed{PANDEMIC_SEED}")
        ]
    pairs = paper_seeds(set_name, n)
    # One run (replicate 0, seed 1) for the sweep at 10^6 and the round limits at 10^7
    if (experiment == "representatives" and n >= EXPANDED_FROM) or (
        experiment == "round_limit" and n > EXPANDED_FROM
    ):
        pairs = [(0, 1)]
    runs = []
    for r, s in pairs:
        if experiment == "representatives":
            for p in caps:
                runs.append(
                    Run(
                        experiment,
                        set_name,
                        n,
                        r,
                        s,
                        f"rep{r}_seed{s}_cap{p}",
                        ("--representative-cap", str(p)),
                    )
                )
            if REFERENCE_CAP in caps:
                s2, p = s + SECOND_SAMPLE, REFERENCE_CAP
                runs.append(
                    Run(
                        experiment,
                        set_name,
                        n,
                        r,
                        s2,
                        f"rep{r}_seed{s2}_cap{p}",
                        ("--representative-cap", str(p)),
                    )
                )
        elif experiment == "round_limit":
            for k in round_limits:
                runs.append(
                    Run(
                        experiment,
                        set_name,
                        n,
                        r,
                        s,
                        f"rep{r}_seed{s}_rounds{k}",
                        ("--max-rounds", str(k)),
                    )
                )
        elif experiment == "random_start":
            runs.append(
                Run(experiment, set_name, n, r, s, f"rep{r}_seed{s}", ("-t", "random"))
            )
        else:
            runs.append(Run(experiment, set_name, n, r, s, f"rep{r}_seed{s}"))
    return runs


def design(args: Namespace) -> list[Run]:
    """Lists the runs the options select: args -> runs, in the order of the experiments, sets and sizes."""
    runs = []
    for experiment in args.experiments:
        for set_name, sizes in DESIGN[experiment].items():
            if args.sets and set_name not in args.sets:
                continue
            for n in args.sizes or sizes:
                for r in runs_of(experiment, set_name, n, args.caps, args.round_limits):
                    if args.replicates is None or r.rep in args.replicates:
                        runs.append(r)
    return runs


def dataset_files(args: Namespace, r: Run) -> tuple[Path, Path]:
    """Finds a run's alignment and true tree: run -> (alignment, true tree)."""
    if r.experiment == "pandemic":
        scale, size = r.set.removeprefix("cmaple_"), f"{r.n // 1000}K"
        return (
            args.cmaple_dir / f"aln_{size}_{scale}.maple",
            args.cmaple_dir / f"tree_{size}_{scale}.nwk",
        )
    folder = args.datasets_dir / r.set / str(r.n)
    return folder / f"rep{r.rep}.fa", folder / f"rep{r.rep}.true.nwk"


def prefix_of(args: Namespace, r: Run) -> Path:
    """Gives the prefix of a run's outputs: run -> data/results/<experiment>/<set>/<n>/<name>.

    The seqlen runs on 10^4 sites are the scaling runs of jc10k.
    """
    experiment = (
        "scaling" if r.experiment == "seqlen" and r.set == "jc10k" else r.experiment
    )
    return args.results_dir / experiment / r.set / str(r.n) / r.name


def infer(args: Namespace, r: Run) -> dict:
    """Runs p2v-bme on a run's alignment unless its report exists: run -> the report."""
    alignment, _ = dataset_files(args, r)
    if not alignment.exists():
        raise FileNotFoundError(
            f"{alignment} is missing: run simulate_data.py or download_data.py first"
        )
    # Without dedup, as the paper's runs: every sequence is searched, also the third and later identical copies
    options = [
        "-s",
        str(alignment),
        "-m",
        MODEL,
        "--no-dedup",
        "--seed",
        str(r.seed),
        "--report-full",
    ]
    options += ["-a", args.accelerator, "-d", args.devices, *r.options]
    return run_p2vbme(args.p2vbme, options, prefix_of(args, r), args.overwrite)


def backbone_errors(args: Namespace, prefix: Path, true_tree: Path, n: int) -> dict:
    """Counts the backbone splits of the true tree missing from a tree: (run prefix, true tree, n) -> record.

    The 10^6- and 10^7-sequence alignments grow each sequence of a 10^5-sequence alignment into a clade of n / 10^5:
    the true splits with more than n / 10^5 leaves on each side are the 10^5-sequence tree's.
    Writes <prefix>.true_vs.{support.nwk,json} (the true tree's branches over the run's tree) and <prefix>.backbone.json.
    """
    out = Path(f"{prefix}.backbone.json")
    if out.exists() and not args.overwrite:
        return read_json(out)
    compared = Path(f"{prefix}.true_vs")
    supports(args.p2vbme, true_tree, Path(f"{prefix}.nwk"), compared, args.overwrite)
    _, branches = branch_table(Path(f"{compared}.support.nwk"))
    backbone = branches["p"] > n // BACKBONE_N
    missing = int((branches["exact"][backbone] < 100.0).sum())
    total = int(backbone.sum())
    record = {
        "backbone_rf": missing,
        "backbone_max": total,
        "backbone_rf_pct": 100.0 * missing / max(total, 1),
    }
    write_json(out, record)
    return record


def eval_pandemic(args: Namespace, prefix: Path, true_tree: Path, sites: int) -> dict:
    """Scores a pandemic-set tree with eval_cmaple_rf unless done: (run prefix, true tree) -> its record.

    All-splits RF and RF on the true branches of >= 0.5 and >= 1 expected substitution, as in the paper.
    """
    out = Path(f"{prefix}.eval.json")
    if args.overwrite or not out.exists():
        result = run(
            [args.eval_cmaple_rf, str(true_tree), f"{prefix}.nwk", str(sites)],
            check=True,
            capture_output=True,
            text=True,
        )
        out.write_text(result.stdout)
    return loads(out.read_text())


def compare(args: Namespace, r: Run, report: dict) -> None:
    """Compares a run's tree with the true tree: run -> its comparison records."""
    _, true_tree = dataset_files(args, r)
    prefix = prefix_of(args, r)
    if not true_tree.exists():
        print(f"no true tree {true_tree}: {prefix} not compared")
        return
    if r.experiment == "pandemic":
        eval_pandemic(args, prefix, true_tree, report["details"]["input"]["sites"])
        return
    compare_to_truth(args.p2vbme, Path(f"{prefix}.nwk"), true_tree, args.overwrite)
    if r.set in EXPANDED_SETS and r.n >= EXPANDED_FROM:
        backbone_errors(args, prefix, true_tree, r.n)


def score(
    args: Namespace,
    alignment: Path,
    trees: list[Path],
    outputs: list[Path],
    cap: int | None = None,
) -> list[float]:
    """Computes the BME lengths of trees on one alignment unless done: (alignment, trees, outputs) -> lengths.

    bme_scores: exact from the full distance matrix, or with cap, the paper's evaluator with at most cap
    representatives per clade (exact when cap >= the number of sequences).
    Each output is a JSON object with "bme_length".
    """
    todo = [(t, o) for t, o in zip(trees, outputs) if args.overwrite or not o.exists()]
    if todo:
        command = [
            args.bme_scores,
            "-s",
            str(alignment),
            "-m",
            MODEL,
            *(["--cap", str(cap)] if cap else []),
            *(str(t) for t, _ in todo),
        ]
        lines = run(
            command, check=True, capture_output=True, text=True
        ).stdout.splitlines()
        assert len(lines) == len(
            todo
        ), f"bme_scores gave {len(lines)} lengths for {len(todo)} trees"
        for (tree, out), line in zip(todo, lines):
            record = loads(line)
            assert record["tree"] == str(tree), (record["tree"], tree)
            write_json(
                out,
                {
                    "tree": str(tree),
                    "bme_length": record["bme_length"],
                    "scorer": "bme_scores",
                    "cap": cap,
                },
            )
    return [read_json(o)["bme_length"] for o in outputs]


def rf_between(args: Namespace, prefix: Path, other: Path, label: str) -> float:
    """Computes the RF distance between two runs' trees: (run prefix, other run's prefix) -> RF in %.

    Writes <prefix>.<label>.{support.nwk,json}; for binary trees on the same leaves, RF = 100 - mean exact support.
    """
    compared = Path(f"{prefix}.{label}")
    report = supports(
        args.p2vbme,
        Path(f"{prefix}.nwk"),
        Path(f"{other}.nwk"),
        compared,
        args.overwrite,
    )
    return 100.0 - report["supports"]["mean_exact"]


def representative_records(args: Namespace, runs: list[Run]) -> None:
    """Scores the sweep of the cap on representatives: runs -> eval/<set>_<n>_rep<r>_P<P>[_seed2].json.

    Per tree, as the paper's records: its BME length (key L_cap<cap>, cap = n when exact) and the RF distance to the
    cap-64 tree of the same seed (key rf_vs_<set>_<n>_rep<r>_P64); for the cap-64 tree, the RF distance to the
    second sample's cap-64 tree (key rf_vs_..._P64_seed2).
    """
    sweep = [r for r in runs if r.experiment == "representatives"]
    for set_name, n, rep in sorted({(r.set, r.n, r.rep) for r in sweep}):
        group = [r for r in sweep if (r.set, r.n, r.rep) == (set_name, n, rep)]
        prefixes = {r.name: prefix_of(args, r) for r in group}
        trees = [Path(f"{p}.nwk") for p in prefixes.values()]
        if not all(t.exists() for t in trees):
            print(f"representatives {set_name} {n} rep{rep}: trees missing, not scored")
            continue
        alignment, _ = dataset_files(args, group[0])
        score_cap = n if n <= args.exact_max else args.score_cap
        lengths = score(
            args,
            alignment,
            trees,
            [Path(f"{p}.score.json") for p in prefixes.values()],
            score_cap,
        )
        cap_key = f"L_cap{score_cap}"
        base = f"{set_name}_{n}_rep{rep}_P{REFERENCE_CAP}"
        seed = min(
            r.seed for r in group
        )  # the first sample; the second is seed + SECOND_SAMPLE
        reference = prefixes.get(f"rep{rep}_seed{seed}_cap{REFERENCE_CAP}")
        second = prefixes.get(f"rep{rep}_seed{seed + SECOND_SAMPLE}_cap{REFERENCE_CAP}")
        for r, length in zip(group, lengths):
            cap, prefix = int(r.options[-1]), prefixes[r.name]
            record = {cap_key: length}
            name = f"{set_name}_{n}_rep{rep}_P{cap}"
            if r.seed != seed:
                name += "_seed2"
            elif cap != REFERENCE_CAP and reference is not None:
                record[f"rf_vs_{base}"] = rf_between(
                    args, prefix, reference, "vs_cap64"
                )
            elif cap == REFERENCE_CAP and second is not None:
                record[f"rf_vs_{base}_seed2"] = rf_between(
                    args, prefix, second, "vs_seed2"
                )
            write_json(
                args.results_dir / "representatives" / "eval" / f"{name}.json", record
            )


def seqlen_records(args: Namespace, runs: list[Run]) -> None:
    """Scores the seqlen trees and their true trees exactly: runs -> diagnose_10k_sequences.jsonl.

    Per dataset of at most --exact-max sequences (the paper: 10^4): the BME lengths of our tree and of the true tree
    on the same alignment, their difference in ppm and our tree's RF error.
    """
    lines = []
    for r in runs:
        if r.experiment != "seqlen" or r.n > args.exact_max:
            continue
        alignment, true_tree = dataset_files(args, r)
        prefix = prefix_of(args, r)
        tree = Path(f"{prefix}.nwk")
        if not (tree.exists() and true_tree.exists()):
            print(f"seqlen {r.set} rep{r.rep}: trees missing, not scored")
            continue
        true_score = (
            args.results_dir
            / "seqlen"
            / r.set
            / str(r.n)
            / f"rep{r.rep}.true.score.json"
        )
        ours, truth = score(
            args,
            alignment,
            [tree, true_tree],
            [Path(f"{prefix}.score.json"), true_score],
        )
        rf = compare_to_truth(args.p2vbme, tree, true_tree, args.overwrite)["rf_pct"]
        record = {"tag": r.set, "rep": r.rep, "L_true": truth, "L_ours": ours}
        record.update(ours_minus_true_ppm=1e6 * (ours - truth) / truth, rf_ours_pct=rf)
        lines.append(dumps(record))
    if lines:
        out = args.results_dir / "seqlen" / "diagnose_10k_sequences.jsonl"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = ArgumentParser(
        description=__doc__, formatter_class=RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--experiments", nargs="+", choices=list(DESIGN), default=list(DESIGN)
    )
    parser.add_argument(
        "--sets", nargs="+", help="only these sets (default: each experiment's)"
    )
    parser.add_argument(
        "--sizes", nargs="+", type=int, help="these sizes instead of each set's"
    )
    parser.add_argument(
        "--replicates",
        nargs="+",
        type=int,
        help="only these dataset replicates (default: all)",
    )
    parser.add_argument(
        "--caps",
        nargs="+",
        type=int,
        default=list(CAPS),
        help="representatives: caps to run",
    )
    parser.add_argument(
        "--round-limits",
        nargs="+",
        type=int,
        default=list(ROUND_LIMITS),
        help="round_limit: rounds",
    )
    parser.add_argument(
        "--accelerator", default="auto", help="p2v-bme -a (default: auto)"
    )
    parser.add_argument("--devices", default="auto", help="p2v-bme -d (default: auto)")
    parser.add_argument(
        "--exact-max",
        type=int,
        default=EXACT_MAX,
        help="score exactly up to N sequences",
    )
    parser.add_argument(
        "--score-cap",
        type=int,
        default=SCORE_CAP,
        help=f"representatives per clade when scoring above --exact-max sequences (default: {SCORE_CAP})",
    )
    parser.add_argument("--datasets-dir", type=Path, default=Path(DEFAULT_DATASETS_DIR))
    parser.add_argument(
        "--cmaple-dir",
        type=Path,
        default=Path(DEFAULT_DATASETS_DIR) / "cmaple" / "simulated",
    )
    parser.add_argument("--results-dir", type=Path, default=Path(DEFAULT_RESULTS_DIR))
    parser.add_argument("--p2vbme", default=P2VBME)
    parser.add_argument("--bme-scores", default=BME_SCORES)
    parser.add_argument("--eval-cmaple-rf", default=EVAL_CMAPLE_RF)
    parser.add_argument(
        "--overwrite", action="store_true", help="run again what exists"
    )
    args = parser.parse_args()

    runs = design(args)
    for i, r in enumerate(runs, 1):
        print(f"[{i}/{len(runs)}] {r.experiment} {r.set} {r.n} {r.name}", flush=True)
        compare(args, r, infer(args, r))
    if "representatives" in args.experiments:
        representative_records(args, runs)
    if "seqlen" in args.experiments:
        seqlen_records(args, runs)


if __name__ == "__main__":
    main()
