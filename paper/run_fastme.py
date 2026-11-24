"""FastME comparison: p2v-bme and FastME on the same distance matrix, every tree rescored exactly (table fastme).

Both programs minimise the BME length of the same matrix (p2v-bme's JC69 distances, written by bme_scores).
Every output tree is rescored in double precision from the full matrix by bme_scores, which shares no code with the
search. Three arms per case:
  sample  each program's own start: p2v-bme sample-and-place (seed rep + 1); FastME BIONJ, NNI and SPR, rerun from
          its own output until the exact length stops falling (its time: the first run alone)
  rand    the same random start tree for both (the set's, rooted as the paper's); FastME rerun to convergence from it
          (its time: all runs)
  polish  p2v-bme started from FastME's converged tree of the sample arm
and an SPR check of our trees (sample, rand): FastME's SPR from our tree until the length stops falling.
FastME rereads a start tree's branch lengths as its length and keeps the start when it finds no shorter tree, so every
start is given uniform lengths that add up to its exact BME length (else a better tree can be thrown away).

Inputs:  data/datasets/fastme/<set>/<n>/rep<r>.fa, rep<r>.true.nwk, rep<r>.random_start.nwk
                                                                     (simulate_data.py --sets fastme)
         sets jc10k, rnasim (1,000-10,000 sequences, 5 replicates: subsets of the 10^4-sequence sets); jc1m, rna1m
         (random subsets of the 10^6-sequence alignments: 100 replicates at 1,000 and 2,000, 20 at 5,000)
Options: --sets, --sizes, --replicates, --arms, --accelerator, --devices, --keep-matrix, --overwrite, paths
Flow:    each set x size x replicate
           |
           v
         bme_scores --phylip: the JC69 matrix as p2v-bme computes it
           |
           v
         p2v-bme (sample, rand, polish) and FastME (BIONJ start, the random start) to convergence, FastME's SPR from
         our trees                                              (each step skipped if its output exists)
           |
           v
         bme_scores: exact lengths; p2v-bme --supports: RF to the true tree and between the programs' trees
Outputs: data/results/fastme/<set>/<n>/rep<r>_*.{nwk,json}       trees, p2v-bme reports, FastME runs
         data/results/fastme/<set>/<n>/rep<r>.case.json            the case's record
         data/results/fastme/cases.jsonl                           every case found (make_tables.py)

    pixi run -e paper python -m paper.run_fastme --sets jc10k rnasim --sizes 1000 2000
"""

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from json import dumps, loads
from pathlib import Path
from re import findall
from subprocess import DEVNULL, run
from time import perf_counter

from paper._config import (
    BME_SCORES,
    DEFAULT_DATASETS_DIR,
    DEFAULT_RESULTS_DIR,
    FASTME,
    P2VBME,
)
from paper.utils import (
    read_json,
    run_p2vbme,
    supports,
    with_edge_lengths,
    write_json,
)

# Replicates per set and size, as the paper's comparison
CASES = {
    "jc10k": {1000: 5, 2000: 5, 5000: 5, 10000: 5},
    "rnasim": {1000: 5, 2000: 5, 5000: 5, 10000: 5},
    "jc1m": {1000: 100, 2000: 100, 5000: 20},
    "rna1m": {1000: 100, 2000: 100, 5000: 20},
}
RAND_MAX_N = 5000  # the random-start arm only up to this size (FastME from a random tree takes hours above)
ARMS = ("sample", "rand", "polish")
MODEL = "JC69"
FASTME_NNI_SPR = ["-n", "B", "-s"]  # BME NNI and SPR, the better of both
FASTME_SPR = ["-s"]
REL_TOL = (
    1e-12  # a rerun that shortens the tree by less than this (relative) ends the reruns
)


def exact_lengths(tool: str, fasta: Path, trees: list[Path]) -> list[float]:
    """Scores trees exactly in double precision: (alignment, trees) -> their BME lengths."""
    out = run(
        [tool, "-s", str(fasta), "-m", MODEL, *map(str, trees)],
        check=True,
        capture_output=True,
        text=True,
    )
    return [loads(line)["bme_length"] for line in out.stdout.splitlines()]


def run_fastme(fastme: str, phylip: Path, args: list[str], out: Path) -> float:
    """Runs FastME once on one thread: (matrix, options, output tree) -> wall seconds; its log goes to <out>.info."""
    t0 = perf_counter()
    cmd = [
        fastme,
        "-i",
        str(phylip),
        *args,
        "-T",
        "1",
        "-o",
        str(out),
        "-I",
        str(out.with_suffix(".info")),
    ]
    run(cmd, check=True, stdout=DEVNULL, stderr=DEVNULL)
    return perf_counter() - t0


def fastme_to_convergence(
    fastme: str,
    tool: str,
    fasta: Path,
    phylip: Path,
    start: Path | None,
    moves: list[str],
    prefix: Path,
    overwrite: bool,
) -> dict:
    """Reruns FastME from its own output until the exact length stops falling: (start, moves) -> run record.

    start None: BIONJ (FastME's default start), else that tree.
    Record (also <prefix>.json): passes [{tree, wall_s, L}], tree (the shortest), L, wall_s (all passes).
    """
    record_path = Path(f"{prefix}.json")
    if record_path.exists() and not overwrite:
        return read_json(record_path)
    passes: list[dict] = []
    current, current_length = start, None
    if start is not None:
        current_length = exact_lengths(tool, fasta, [start])[0]
    while True:
        out = Path(f"{prefix}_pass{len(passes)}.nwk")
        if current is None:
            args = ["-m", "I", *moves]
        else:
            assert current_length is not None
            # uniform lengths that add up to the start's BME length: FastME takes their sum as its length
            n_leaves = len(findall(r"[(,]([^(),;]+)", Path(current).read_text()))
            edge = current_length / (2 * n_leaves - 3)
            uniform = with_edge_lengths(
                current, edge, Path(f"{prefix}_pass{len(passes)}.start.nwk")
            )
            args = ["-u", str(uniform), *moves]
        wall = run_fastme(fastme, phylip, args, out)
        length = exact_lengths(tool, fasta, [out])[0]
        passes.append({"tree": str(out), "wall_s": wall, "L": length})
        if current_length is not None and length >= current_length * (1 - REL_TOL):
            break
        current, current_length = out, length
    best = min(passes, key=lambda p: p["L"])
    record = {
        "start": str(start) if start else "BIONJ",
        "moves": moves,
        "passes": passes,
        "tree": best["tree"],
        "L": best["L"],
        "wall_s": sum(p["wall_s"] for p in passes),
    }
    write_json(record_path, record)
    return record


def rf_pct(
    p2vbme: str, tree: Path, other: Path, prefix: Path, overwrite: bool
) -> float:
    """Computes the normalised RF distance of two binary trees in %: (tree, other) -> 100 - mean exact support."""
    return (
        100.0
        - supports(p2vbme, tree, other, prefix, overwrite)["supports"]["mean_exact"]
    )


def solver_entry(report: dict) -> dict:
    """Takes the run's time and devices from a p2v-bme report: report -> the case's solver entry."""
    details = report["details"]
    devices = details["devices"]
    gpus = [d for d in devices if d.startswith("GPU")]
    return {
        "wall_s": details["time"]["start_tree"] + details["time"]["search"],
        "bl_s": details["time"]["branch_lengths"],
        "n_gpus": len(gpus),
        "gpu_name": gpus[0].split(": ", 1)[1] if gpus else "",
        "devices": devices,
        "rounds_run": report["rounds_run"],
        "solver_len": report["bme_length"],
    }


def score_case(
    fasta: Path, true_tree: Path, folder: Path, name: str, arms: list[str], args
) -> dict:
    """Runs and scores one case: (alignment, true tree) -> its record (cases.jsonl's fields)."""
    seed = int(name.removeprefix("rep")) + 1
    phylip = folder / f"{name}.phy"
    if not phylip.exists():
        run(
            [args.bme_scores, "-s", str(fasta), "-m", MODEL, "--phylip", str(phylip)],
            check=True,
        )
    p2v = [
        *args.p2vbme_options,
        "-s",
        str(fasta),
        "-m",
        MODEL,
        "--seed",
        str(seed),
        "--report-full",
        "--no-dedup",
    ]
    ow = args.overwrite

    def ours(arm: str, start: Path | None) -> tuple[Path, dict]:
        """Runs p2v-bme for one arm: (arm, start tree) -> (tree, report)."""
        prefix = folder / f"{name}_seed{seed}_{arm}"
        report = run_p2vbme(
            args.p2vbme, [*p2v, *(["-t", str(start)] if start else [])], prefix, ow
        )
        return Path(f"{prefix}.nwk"), report

    def fastme(start: Path | None, moves: list[str], label: str) -> dict:
        """Runs FastME to convergence: (start, moves) -> run record."""
        return fastme_to_convergence(
            args.fastme,
            args.bme_scores,
            fasta,
            phylip,
            start,
            moves,
            folder / f"{name}_{label}",
            ow,
        )

    def rf(tree: Path, other: Path, label: str) -> float:
        """RF between two of the case's trees in %."""
        return rf_pct(
            args.p2vbme, tree, other, folder / f"{Path(tree).stem}.vs_{label}", ow
        )

    record: dict = {
        "tag": fasta.parent.parent.name,
        "n": int(fasta.parent.name),
        "rep": int(name[3:]),
    }
    record["evaluator"] = "exact double precision (paper/tools/bme_scores.cpp)"
    converged = fastme(None, FASTME_NNI_SPR, "fastme")
    first = converged["passes"][0]
    bionj = folder / f"{name}_bionj.nwk"
    if ow or not bionj.exists():
        run_fastme(args.fastme, phylip, ["-m", "I"], bionj)
    # FastME's first run returned its BIONJ start unchanged
    record["kept_start"] = (
        first["L"] == exact_lengths(args.bme_scores, fasta, [bionj])[0]
    )
    starts = {"sample": None, "polish": Path(converged["tree"])}
    references = {"sample": converged, "polish": converged}
    times = {"sample": first["wall_s"], "polish": first["wall_s"]}
    if "rand" in arms:
        # the set's random start, rooted as the paper gave it to both programs (p2v-bme's search depends on the rooting)
        starts["rand"] = fasta.with_name(f"{name}.random_start.nwk")
        references["rand"] = fastme(starts["rand"], FASTME_NNI_SPR, "fastme_rand")
        times["rand"] = references["rand"]["wall_s"]
    for arm in arms:
        tree, report = ours(arm, starts[arm])
        theirs = Path(references[arm]["tree"])
        length = exact_lengths(args.bme_scores, fasta, [tree])[0]
        entry = {
            "L": length,
            "rf_truth": rf(tree, true_tree, "true"),
            "L_fastme": references[arm]["L"],
            "rf_truth_fastme": rf(theirs, true_tree, "true"),
            "rf_vs_fastme": rf(tree, theirs, "fastme"),
            "solver": solver_entry(report),
            "fastme_wall_s": times[arm],
            "rel_diff": (length - references[arm]["L"]) / references[arm]["L"],
        }
        if arm != "polish":
            climb = fastme(tree, FASTME_SPR, f"seed{seed}_{arm}_spr")
            entry["spr"] = {
                "L_start": length,
                "L_end": climb["L"],
                "gain": length - climb["L"],
                "rel_gain": (length - climb["L"]) / length,
                "iterations": len(climb["passes"]),
                "wall_s": climb["wall_s"],
                "rf_moved_pct": rf(tree, Path(climb["tree"]), "spr"),
            }
        record[arm] = entry
    if not args.keep_matrix:
        phylip.unlink()
    return record


def main() -> None:
    parser = ArgumentParser(
        description=__doc__, formatter_class=RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--sets",
        nargs="+",
        choices=list(CASES),
        default=list(CASES),
        help="datasets (default: all four)",
    )
    parser.add_argument(
        "--sizes", nargs="+", type=int, help="sequences (default: each set's sizes)"
    )
    parser.add_argument(
        "--replicates",
        nargs="+",
        type=int,
        help="only these replicates (default: as the paper, per set and size)",
    )
    parser.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    parser.add_argument(
        "--accelerator", default="auto", help="p2v-bme -a (cpu, gpu, auto)"
    )
    parser.add_argument("--devices", default="auto", help="p2v-bme -d")
    parser.add_argument(
        "--keep-matrix",
        action="store_true",
        help="keep the PHYLIP matrices (N^2 values each)",
    )
    parser.add_argument(
        "--overwrite", action="store_true", help="rerun steps whose outputs exist"
    )
    parser.add_argument("--datasets-dir", type=Path, default=Path(DEFAULT_DATASETS_DIR))
    parser.add_argument("--results-dir", type=Path, default=Path(DEFAULT_RESULTS_DIR))
    parser.add_argument("--p2vbme", default=P2VBME)
    parser.add_argument("--bme-scores", default=BME_SCORES)
    parser.add_argument("--fastme", default=FASTME)
    args = parser.parse_args()
    args.p2vbme_options = ["-a", args.accelerator, "-d", args.devices]

    out_dir = args.results_dir / "fastme"
    for tag in args.sets:
        for n in args.sizes or CASES[tag]:
            replicates = (
                args.replicates
                if args.replicates is not None
                else range(CASES[tag].get(n, 0))
            )
            for r in replicates:
                fasta = args.datasets_dir / "fastme" / tag / str(n) / f"rep{r}.fa"
                folder = out_dir / tag / str(n)
                case = folder / f"rep{r}.case.json"
                if case.exists() and not args.overwrite:
                    continue
                if not fasta.exists():
                    raise FileNotFoundError(
                        f"{fasta} is missing: run simulate_data.py --sets fastme first"
                    )
                folder.mkdir(parents=True, exist_ok=True)
                arms = [a for a in args.arms if a != "rand" or n <= RAND_MAX_N]
                true_tree = fasta.with_name(f"rep{r}.true.nwk")
                write_json(
                    case, score_case(fasta, true_tree, folder, f"rep{r}", arms, args)
                )
                print(f"run_fastme: {tag} {n} rep{r} done", flush=True)
    cases = sorted(out_dir.glob("*/*/rep*.case.json"))
    records = [read_json(c) for c in cases]
    records.sort(key=lambda c: (c["tag"], c["n"], c["rep"]))
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "cases.jsonl").write_text("".join(dumps(c) + "\n" for c in records))
    print(f"run_fastme: {len(records)} cases in {out_dir / 'cases.jsonl'}")


if __name__ == "__main__":
    main()
