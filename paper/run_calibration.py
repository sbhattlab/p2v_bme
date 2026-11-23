"""Bootstrap calibration: the supports of a reference tree's branches against the true tree (table bootstrap; the
supplement's calibration on independent lineages).

Inputs:  data/datasets/<set>/<n>/rep0.fa, rep0.true.nwk                         (simulate_data.py)
         data/datasets/cmaple/simulated/aln_100K_0.0001.maple, tree_100K_0.0001.nwk and the CMAPLE and MAPLE trees
         aln_100K_0.0001.maple.treefile, aln_100K_0.0001_wo_ref_tree.tree     (download_data.py)
Options: --sets, --steps, --replicates, --bootstrap, --repeats, --accelerator, --devices, --datasets-dir, --cmaple-dir,
         --results-dir, --p2vbme, --overwrite
Flow:    each calibration set (RNASim 10^5 and 10^6, random tree 10^6, pandemic 10^5; random tree 10^7 on request)
           |
           v
         ref      p2v-bme -s <alignment> -m JC69 --no-dedup --seed 1 --report-full
         boot     p2v-bme ... --seed 1 -b r-r                               r = 1..64, one run each (--replicates)
         seed     p2v-bme ... --seed s --no-branch-lengths                  s = 2..9: the search's noise alone
           |      (each run skipped if its report exists; --no-dedup: every tree on every sequence, as in the paper)
           v
         score    p2v-bme -t ref.nwk --supports {boot.nwk, seed.nwk, the true tree; pandemic: CMAPLE's, MAPLE's tree}
           |      (once every tree of the set exists; multifurcations of the other trees resolved as the paper did)
           v
         per-branch tables (columns of the paper's support_calibration) and their summaries
Outputs: data/results/calibration/<set>/<n>/ref.{nwk,json}, boot<r>.{boottrees,json}, seed<s>.{nwk,json}
         .../boot.nwk, seed.nwk                                   the replicate trees, one per line
         .../ref.{boot,seed,vs_true,vs_cmaple,vs_maple}.{support.nwk,json}
         .../<tag>_{boot,seed}_branches.tsv     node size p branch_length exact tbe delta_true correct true_transfer
                                                (+ the same against the CMAPLE and MAPLE trees for the pandemic set)
         data/results/calibration/rnasim_bootstrap_calibration_summary.json, maple_bootstrap_calibration_summary.json,
         calibration_<n>.json (random trees): the paper's summaries
         <tag>: rnasim_100000, rnasim_1000000, yule10k_1000000, yule10k_10000000, maple_00001_100000

    pixi run -e paper python -m paper.run_calibration --sets rnasim_100000 --steps boot --replicates 1-16
"""

from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter
from json import dumps
from pathlib import Path
from re import findall

from numpy import (
    argsort,
    array_equal,
    column_stack,
    empty,
    inf,
    isnan,
    loadtxt,
    ndarray,
)
from numpy import repeat as np_repeat
from numpy import rint, savetxt, unique

from paper._config import DEFAULT_DATASETS_DIR, DEFAULT_RESULTS_DIR, P2VBME
from paper.utils import (
    branch_table,
    compare_to_truth,
    read_json,
    run_p2vbme,
    supports,
)

MODEL = "JC69"
REFERENCE_SEED = (
    1  # the reference tree's seed, also the seed of every bootstrap replicate
)
N_BOOTSTRAP = 64
N_REPEATS = 8  # repeat searches with seeds 2..9 on the unchanged alignment
PANDEMIC = "cmaple_0.0001"  # 100,000 sequences, 30,000 sites, mean branch length 1e-4

# tag -> (set, n), as the paper's runs name them; other sizes of a simulated set as <set>_<n> (calibration())
CALIBRATIONS = {
    "rnasim_100000": ("rnasim", 100_000),
    "rnasim_1000000": ("rnasim", 1_000_000),
    "yule10k_1000000": ("yule10k", 1_000_000),
    "yule10k_10000000": ("yule10k", 10_000_000),
    "maple_00001_100000": (PANDEMIC, 100_000),
}
SIMULATED_SETS = ("rnasim", "jc10k", "yule10k")
PAPER_SETS = (
    "rnasim_100000",
    "rnasim_1000000",
    "yule10k_1000000",
    "maple_00001_100000",
)
STEPS = ("ref", "boot", "seed", "score")

# Summaries, as the paper's records
SIZE_BINS = [
    ("2", 2, 2),
    ("3–10", 3, 10),
    ("11–100", 11, 100),
    ("101–1k", 101, 1000),
    ("1k–10k", 1001, 10000),
    (">10k", 10001, 2**31),
]
# Pandemic branches by their estimated length (30,000 sites: one substitution = 3.3e-5)
REGIMES = {
    "zero_snp": ("Zero-mutation / near-zero (bl < 1.5e-5, ~0 SNPs)", -inf, 1.5e-5),
    "one_snp": ("Single-mutation (~1 SNP, 1.5e-5 <= bl < 5e-5)", 1.5e-5, 5e-5),
    "ge2_snp": (">=2 mutations (bl >= 5e-5)", 5e-5, inf),
    "all_mutated": ("All mutated branches (>=1 SNP, bl >= 1.5e-5)", 1.5e-5, inf),
}
RANDOM_TREE_SIZE_BINS = (
    (2, 10),
    (11, 100),
    (101, 1000),
    (1001, 10000),
    (10001, 100000),
)
RELIABILITY_EDGES = (0.0, 0.1, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0)
HIGH_SUPPORT = 0.95

COLUMNS = [
    "node",
    "size",
    "p",
    "branch_length",
    "exact",
    "tbe",
    "delta_true",
    "correct",
    "true_transfer",
]
FORMATS = ["%d", "%d", "%d", "%.7g", "%.6f", "%.6f", "%d", "%d", "%.6f"]


def calibration(tag: str) -> tuple[str, int]:
    """Reads a calibration set's tag: tag -> (set, n); a paper tag, or <set>_<n> of a simulated set (e.g. a tiny test)."""
    if tag in CALIBRATIONS:
        return CALIBRATIONS[tag]
    set_name, _, n = tag.rpartition("_")
    if set_name not in SIMULATED_SETS or not n.isdigit():
        raise ValueError(
            f"unknown calibration set {tag}: {', '.join(CALIBRATIONS)} or <set>_<n>"
        )
    return set_name, int(n)


def files_of(args: Namespace, tag: str) -> dict[str, Path]:
    """Finds a calibration set's inputs: tag -> {alignment, true, and for the pandemic set cmaple, maple}."""
    set_name, n = calibration(tag)
    if set_name == PANDEMIC:
        stem = args.cmaple_dir / f"aln_{n // 1000}K_{set_name.removeprefix('cmaple_')}"
        return {
            "alignment": Path(f"{stem}.maple"),
            "true": args.cmaple_dir
            / f"tree_{n // 1000}K_{set_name.removeprefix('cmaple_')}.nwk",
            "cmaple": Path(f"{stem}.maple.treefile"),
            "maple": Path(f"{stem}_wo_ref_tree.tree"),
        }
    folder = args.datasets_dir / set_name / str(n)
    return {"alignment": folder / "rep0.fa", "true": folder / "rep0.true.nwk"}


def folder_of(args: Namespace, tag: str) -> Path:
    """Gives a calibration set's results folder: tag -> data/results/calibration/<set>/<n>."""
    set_name, n = calibration(tag)
    return args.results_dir / "calibration" / set_name / str(n)


def run_names(args: Namespace) -> dict[str, list[tuple[str, list[str]]]]:
    """Lists the runs of a calibration set by step: args -> {step: [(name, p2v-bme options)]}."""
    seed = ["--seed", str(REFERENCE_SEED)]
    return {
        "ref": [("ref", [*seed, "--report-full"])],
        "boot": [(f"boot{r:03d}", [*seed, "-b", f"{r}-{r}"]) for r in args.replicates],
        "seed": [
            (f"seed{s}", ["--seed", str(s), "--no-branch-lengths"])
            for s in range(2, args.repeats + 2)
        ],
    }


def infer(args: Namespace, tag: str) -> None:
    """Runs the selected steps' trees of one calibration set unless their reports exist: tag -> trees."""
    alignment = files_of(args, tag)["alignment"]
    if not alignment.exists():
        raise FileNotFoundError(
            f"{alignment} is missing: run simulate_data.py or download_data.py first"
        )
    # Without dedup, as the paper's runs: the replicate trees then hold every sequence, as the reference does
    common = [
        "-s",
        str(alignment),
        "-m",
        MODEL,
        "--no-dedup",
        "-a",
        args.accelerator,
        "-d",
        args.devices,
    ]
    for step, runs in run_names(args).items():
        if step not in args.steps:
            continue
        for name, options in runs:
            print(f"{tag} {name}", flush=True)
            run_p2vbme(
                args.p2vbme,
                [*common, *options],
                folder_of(args, tag) / name,
                args.overwrite,
            )


def binary_tree(tree: Path, out: Path) -> Path:
    """Resolves the multifurcations of a Newick tree as the paper's packing did: (tree, out) -> a binary tree.

    A node with children c1..ck (k > 2; the root: k > 3) becomes (((c1, c2), c3) ..., ck), the new branches of length 0.
    Returns tree itself when it is binary; otherwise writes out (unless it exists) and returns it.
    """
    if out.exists():
        return out
    tokens = findall(r"\(|\)|,|;|[^(),;]+", tree.read_text())
    opens: list[int] = []  # output position of each open "("
    commas: list[list[int]] = []  # output positions of the commas between its children
    changed = False
    for i, t in enumerate(tokens):
        if t == "(":
            opens.append(i)
            commas.append([])
        elif t == "," and commas:
            commas[-1].append(i)
        elif t == ")":
            start, between = opens.pop(), commas.pop()
            k = len(between) + 1
            if k > (3 if not opens else 2):
                tokens[start] = "(" * (k - 1)
                for j in between[1:]:
                    tokens[j] = "):0,"
                changed = True
    if not changed:
        return tree
    out.write_text("".join(tokens).strip() + "\n")
    return out


def gather(trees: list[Path], out: Path, overwrite: bool) -> None:
    """Writes trees into one file, one per line, unless it exists: (trees, out) -> out."""
    if overwrite or not out.exists():
        out.write_text("".join(t.read_text().strip() + "\n" for t in trees))


def compared_to(
    sup: dict[str, ndarray], against: Path, name: str
) -> dict[str, ndarray]:
    """Compares the reference's branches with another tree: (supports' branches, file against it, name) -> columns.

    delta_<name>: the leaves to move to obtain a split of the other tree; <name>_correct: 1 if it has the split;
    <name>_transfer: the transfer index to it (the paper's columns; "true" gives delta_true, correct, true_transfer).
    """
    _, other = branch_table(against)
    assert array_equal(
        sup["size"], other["size"]
    ), f"{against} is not on the reference tree"
    assert not isnan(
        other["exact"]
    ).any(), f"{against}: a branch of the reference is not on the other tree's leaves"
    transfer = other["tbe"] / 100.0
    correct = "correct" if name == "true" else f"{name}_correct"
    return {
        f"delta_{name}": rint((1.0 - transfer) * (sup["p"] - 1)).astype(int),
        correct: (other["exact"] == 100.0).astype(int),
        f"{name}_transfer": transfer,
    }


def score(args: Namespace, tag: str) -> None:
    """Scores the reference's branches once every tree exists: tag -> <tag>_{boot,seed}_branches.tsv."""
    folder, files = folder_of(args, tag), files_of(args, tag)
    names = run_names(args)
    boot_trees = [
        folder / f"boot{r:03d}.boottrees" for r in range(1, args.bootstrap + 1)
    ]
    seed_trees = [folder / f"{name}.nwk" for name, _ in names["seed"]]
    reference = folder / "ref.nwk"
    missing = [t for t in [reference, *boot_trees, *seed_trees] if not t.exists()]
    if missing:
        print(
            f"{tag}: {len(missing)} trees missing (first {missing[0]}); scoring is left to the run that finds them all"
        )
        return
    gather(boot_trees, folder / "boot.nwk", args.overwrite)
    gather(seed_trees, folder / "seed.nwk", args.overwrite)
    for kind in ("boot", "seed"):
        supports(
            args.p2vbme,
            reference,
            folder / f"{kind}.nwk",
            folder / f"ref.{kind}",
            args.overwrite,
        )
    true_tree = binary_tree(files["true"], folder / "true.binary.nwk")
    compare_to_truth(args.p2vbme, reference, true_tree, args.overwrite)
    others = [k for k in ("cmaple", "maple") if k in files]
    for k in others:
        other = binary_tree(files[k], folder / f"{k}.binary.nwk")
        supports(args.p2vbme, reference, other, folder / f"ref.vs_{k}", args.overwrite)
    for kind in ("boot", "seed"):
        out = folder / f"{tag}_{kind}_branches.tsv"
        if out.exists() and not args.overwrite:
            continue
        support_nwk = folder / f"ref.{kind}.support.nwk"
        _, sup = branch_table(support_nwk)
        columns = {
            "node": sup["node"],
            "size": sup["size"],
            "p": sup["p"],
            "branch_length": sup["length"],
        }
        columns.update(exact=sup["exact"] / 100.0, tbe=sup["tbe"] / 100.0)
        columns.update(compared_to(sup, folder / "ref.vs_true.support.nwk", "true"))
        header, formats = list(COLUMNS), list(FORMATS)
        for k in others:
            columns.update(compared_to(sup, folder / f"ref.vs_{k}.support.nwk", k))
            header += [f"delta_{k}", f"{k}_correct", f"{k}_transfer"]
            formats += ["%d", "%d", "%.6f"]
        assert not isnan(
            columns["exact"]
        ).any(), f"{support_nwk}: a branch without support"
        table = column_stack([columns[c] for c in header])
        savetxt(
            out,
            table,
            fmt=formats,
            delimiter="\t",
            header="\t".join(header),
            comments="",
        )
        print(f"{tag}: wrote {out}")


def read_branches(path: Path) -> dict[str, ndarray]:
    """Reads a per-branch table: path -> its columns."""
    header = path.open().readline().strip().split("\t")
    data = loadtxt(path, delimiter="\t", skiprows=1, ndmin=2)
    return {name: data[:, i] for i, name in enumerate(header)}


def pct(x: ndarray) -> float:
    """Gives the mean of x in percent: x -> 100 mean(x) (nan if empty)."""
    return float(100.0 * x.mean()) if x.size else float("nan")


def by_size(d: dict[str, ndarray]) -> dict:
    """Averages the supports per clade-size bin of the smaller side p: branches -> bins."""
    out: dict[str, list] = dict(
        labels=[],
        counts=[],
        mean_tbe=[],
        mean_exact=[],
        mean_true_transfer=[],
        mean_correct=[],
    )
    for label, lo, hi in SIZE_BINS:
        m = (d["p"] >= lo) & (d["p"] <= hi)
        out["labels"].append(label)
        out["counts"].append(int(m.sum()))
        for key, col in (
            ("mean_tbe", "tbe"),
            ("mean_exact", "exact"),
            ("mean_true_transfer", "true_transfer"),
            ("mean_correct", "correct"),
        ):
            out[key].append(pct(d[col][m]))
    return out


def auc(score: ndarray, positive: ndarray) -> float:
    """Computes the area under the ROC curve of a score for a true/false outcome: (score, outcome) -> AUC.

    By rank sum, tied scores sharing their mean rank.
    """
    order = argsort(score, kind="stable")
    _, first, counts = unique(score[order], return_index=True, return_counts=True)
    ranks = empty(score.size)
    ranks[order] = np_repeat(first + (counts + 1) / 2.0, counts)
    n1 = int(positive.sum())
    return float(
        (ranks[positive].sum() - n1 * (n1 + 1) / 2) / (n1 * (positive.size - n1))
    )


def version_of(args: Namespace, tag: str) -> str:
    """Gives the p2v-bme version of a calibration's reference run: tag -> version."""
    return read_json(folder_of(args, tag) / "ref.json")["version"]


def count_trees(path: Path) -> int:
    """Counts the trees of a file with one per line: path -> count."""
    return len(path.read_text().split())


def rnasim_summary(args: Namespace) -> None:
    """Summarises the RNASim calibrations: tables -> rnasim_bootstrap_calibration_summary.json."""
    summary: dict = {}
    for tag, key in (("rnasim_100000", "N100k"), ("rnasim_1000000", "N1M")):
        for kind in ("boot", "seed"):
            path = folder_of(args, tag) / f"{tag}_{kind}_branches.tsv"
            if not path.exists():
                continue
            d = read_branches(path)
            entry: dict = dict(
                n_branches=int(d["p"].size),
                mean_tbe_pct=pct(d["tbe"]),
                mean_exact_pct=pct(d["exact"]),
            )
            if kind == "boot":
                entry.update(
                    mean_true_transfer_pct=pct(d["true_transfer"]),
                    mean_correct_pct=pct(d["correct"]),
                )
            entry["bins"] = by_size(d)
            summary[f"{key}_{kind}"] = entry
            summary["version"] = version_of(args, tag)
    if summary:
        write_summary(args, "rnasim_bootstrap_calibration_summary.json", summary)


def maple_summary(args: Namespace) -> None:
    """Summarises the pandemic calibration by branch length and size: tables -> maple_bootstrap_..._summary.json."""
    tag = "maple_00001_100000"
    folder = folder_of(args, tag)
    paths = [folder / f"{tag}_{kind}_branches.tsv" for kind in ("boot", "seed")]
    if not all(p.exists() for p in paths):
        return
    boot, seed = (read_branches(p) for p in paths)
    assert array_equal(boot["node"], seed["node"])

    def stats(m: ndarray) -> dict:
        return dict(
            branches=int(m.sum()),
            share_pct=pct(m),
            true_exact_pct=pct(boot["correct"][m]),
            true_tbe_pct=pct(boot["true_transfer"][m]),
            cmaple_exact_pct=pct(boot["cmaple_correct"][m]),
            cmaple_tbe_pct=pct(boot["cmaple_transfer"][m]),
            maple_exact_pct=pct(boot["maple_correct"][m]),
            boot_fbp_pct=pct(boot["exact"][m]),
            boot_tbe_pct=pct(boot["tbe"][m]),
            seed_exact_pct=pct(seed["exact"][m]),
            seed_tbe_pct=pct(seed["tbe"][m]),
        )

    bl, wrong = boot["branch_length"], boot["correct"] == 0
    regimes = {
        k: dict(label=label, **stats((bl >= lo) & (bl < hi)))
        for k, (label, lo, hi) in REGIMES.items()
    }
    sizes = [dict(bin="All splits", **stats(bl == bl))]
    sizes += [
        dict(bin=label, **stats((boot["p"] >= lo) & (boot["p"] <= hi)))
        for label, lo, hi in SIZE_BINS
    ]
    report = read_json(folder / "ref.json")
    summary = dict(
        dataset="MAPLE simulated pandemic (aln_100K_0.0001)",
        N=int(bl.size + 3),
        L=report["details"]["input"]["sites"],
        B_boot=count_trees(folder / "boot.nwk"),
        S_seed=count_trees(folder / "seed.nwk"),
        total_branches=int(bl.size),
        wrong_branches_total=int(wrong.sum()),
        wrong_branches_zero_snp=int((wrong & (bl < REGIMES["zero_snp"][2])).sum()),
        regimes=regimes,
        by_size=sizes,
        view="branches of our reference tree, classified by their estimated length",
        version=report["version"],
    )
    write_summary(args, "maple_bootstrap_calibration_summary.json", summary)


def random_tree_summary(args: Namespace, tag: str) -> None:
    """Summarises a random-tree calibration: tables -> calibration_<n>.json."""
    _, n = calibration(tag)
    folder = folder_of(args, tag)
    paths = [folder / f"{tag}_{kind}_branches.tsv" for kind in ("boot", "seed")]
    if not all(p.exists() for p in paths):
        return
    boot, seed = (read_branches(p) for p in paths)
    assert array_equal(boot["node"], seed["node"])
    ok, exact, p, again = boot["correct"] == 1, boot["exact"], boot["p"], seed["exact"]
    high = exact >= HIGH_SUPPORT
    sizes = []
    for lo, hi in (*RANDOM_TREE_SIZE_BINS, (RANDOM_TREE_SIZE_BINS[-1][1] + 1, n)):
        m = (p >= lo) & (p <= hi)
        if m.any():
            sizes.append(
                dict(
                    size_from=lo,
                    size_to=hi,
                    branches=int(m.sum()),
                    correct_pct=pct(ok[m]),
                    support95_share_pct=pct(high[m]),
                    support95_wrong_pct=pct(~ok[m & high]),
                )
            )
    reliability = []
    for lo, hi in zip(RELIABILITY_EDGES, RELIABILITY_EDGES[1:]):
        m = (exact <= hi) & ((exact > lo) | (lo == 0.0))
        reliability.append(
            dict(
                support_above=lo,
                support_to=hi,
                branches=int(m.sum()),
                mean_support_pct=pct(exact[m]),
                correct_pct=pct(ok[m]),
            )
        )
    summary = dict(
        dataset=f"first random-tree dataset of {n:,} leaves (yule10k rep0, {n})",
        n_leaves=n,
        branches=int(ok.size),
        bootstrap_trees=count_trees(folder / "boot.nwk"),
        repeat_searches=count_trees(folder / "seed.nwk"),
        wrong_branches=int((~ok).sum()),
        wrong_pct=pct(~ok),
        auc_exact=auc(exact, ok),
        auc_tbe=auc(boot["tbe"], ok),
        mean_exact_pct=pct(exact),
        mean_tbe_pct=pct(boot["tbe"]),
        mean_true_transfer_pct=pct(boot["true_transfer"]),
        support95=dict(
            share_pct=pct(high),
            wrong_pct=pct(~ok[high]),
            wrong_branches=int((~ok & high).sum()),
        ),
        by_size=sizes,
        reliability=reliability,
        repeat=dict(
            mean_exact_pct=pct(again),
            mean_tbe_pct=pct(seed["tbe"]),
            wrong_in_all_pct=pct(again[~ok] == 1),
            correct_in_all_pct=pct(again[ok] == 1),
        ),
        version=version_of(args, tag),
    )
    write_summary(args, f"calibration_{n}.json", summary)


def write_summary(args: Namespace, name: str, summary: dict) -> None:
    """Writes a summary next to the calibration folders: (name, summary) -> data/results/calibration/<name>."""
    out = args.results_dir / "calibration" / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(dumps(summary, indent=2) + "\n")
    print(f"wrote {out}")


def parse_range(text: str) -> list[int]:
    """Parses replicates as A-B or A: text -> the list A..B."""
    first, _, last = text.partition("-")
    return list(range(int(first), int(last or first) + 1))


def main() -> None:
    parser = ArgumentParser(
        description=__doc__, formatter_class=RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--sets",
        nargs="+",
        default=list(PAPER_SETS),
        help=f"{', '.join(CALIBRATIONS)} or <set>_<n> of {', '.join(SIMULATED_SETS)}",
    )
    parser.add_argument("--steps", nargs="+", choices=STEPS, default=list(STEPS))
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=N_BOOTSTRAP,
        help="bootstrap replicates of a set (B)",
    )
    parser.add_argument(
        "--replicates",
        type=parse_range,
        help="the bootstrap replicates to run here, A-B (default: 1-B)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=N_REPEATS,
        help="repeat searches, seeds 2..repeats + 1",
    )
    parser.add_argument(
        "--accelerator", default="auto", help="p2v-bme -a (default: auto)"
    )
    parser.add_argument("--devices", default="auto", help="p2v-bme -d (default: auto)")
    parser.add_argument("--datasets-dir", type=Path, default=Path(DEFAULT_DATASETS_DIR))
    parser.add_argument(
        "--cmaple-dir",
        type=Path,
        default=Path(DEFAULT_DATASETS_DIR) / "cmaple" / "simulated",
    )
    parser.add_argument("--results-dir", type=Path, default=Path(DEFAULT_RESULTS_DIR))
    parser.add_argument("--p2vbme", default=P2VBME)
    parser.add_argument(
        "--overwrite", action="store_true", help="run again what exists"
    )
    args = parser.parse_args()
    if args.replicates is None:
        args.replicates = list(range(1, args.bootstrap + 1))

    for tag in args.sets:
        infer(args, tag)
        if "score" in args.steps:
            score(args, tag)
    if "score" in args.steps:
        rnasim_summary(args)
        maple_summary(args)
        for tag in ("yule10k_1000000", "yule10k_10000000"):
            random_tree_summary(args, tag)


if __name__ == "__main__":
    main()
