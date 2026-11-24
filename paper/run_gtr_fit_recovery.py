"""Recovery of the GTR+GC parameters by p2v-bme's fit on simulated alignments (table gtr_fit).

Alignments of 1,000 sequences are simulated on a random ultrametric tree under GTR with unequal base
frequencies and gamma rates across sites, at two divergence levels (median p-distance) and three
lengths, with the paper's simulation code and seeds,
so each alignment is the one the paper's fit saw.
p2v-bme fits each one (-m GTR+GC, no search) and the fitted shape and exchangeabilities are
compared with the truth.

Inputs:  none (simulated)
Options: --sequences, --replicates, --alphas, --sites, --divergence, --pairs, --threads,
         --keep-fasta, --overwrite, paths
Flow:    each alpha x sites x divergence x replicate
           |
           v
         simulate the alignment (the paper's code and seeds) -> FASTA
           |
           v
         p2v-bme -m GTR+GC --max-rounds 0 --no-branch-lengths -t random -a cpu
           |      (skipped if its report exists)
           v
         alpha_hat, RMS error of the log exchangeabilities
Outputs: data/results/gtr_fit/runs/<alignment>.{nwk,json}    p2v-bme's reports
         data/results/gtr_fit/recovery.tsv                   one row per alignment (make_tables.py)

    pixi run -e paper python -m paper.run_gtr_fit_recovery
"""

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from os import environ
from pathlib import Path
from subprocess import run

from numpy import (
    array,
    cumsum,
    diag,
    einsum,
    empty,
    exp,
    fill_diagonal,
    frombuffer,
    full,
    int64,
    log,
    mean,
    median,
    ndarray,
    power,
    sqrt,
    triu_indices,
    uint8,
    zeros,
)
from numpy.linalg import eigh
from numpy.random import Generator, default_rng

from paper._config import DEFAULT_RESULTS_DIR, P2VBME
from paper.utils import read_json

RATES = array([1.0, 4.0, 1.0, 1.0, 4.0, 1.0])  # true exchangeabilities AC, AG, AT, CG, CT, GT
PI = array([0.30, 0.20, 0.27, 0.23])  # true base frequencies A, C, G, T
SEED = 20261115  # the paper's: alignment seed = SEED + 1000 rep + 100 alpha + sites + 1000 p
PAIRS_FOR_SCALE = 4000  # leaf pairs whose median p-distance sets the tree's scale
COLUMNS = (
    "alpha_true\tsites\ttarget_p\treplicate\talpha_hat\trate_rms_log_error\tAC\tAG\tAT\tCG\tCT\tGT\t"
    "iterations\tfit_seconds\tconverged\tpairs"
)

# ---- The paper's simulation, unchanged -----------------------------------------------------------


def gtr_eigen(rates: ndarray, pi: ndarray) -> tuple[ndarray, ndarray, ndarray]:
    """Decomposes the GTR rate matrix scaled to one substitution per unit time:
    (rates, pi) -> (lam, vr, vl)."""
    s = zeros((4, 4))
    s[triu_indices(4, 1)] = rates
    s = s + s.T
    q = s * pi[None, :]
    fill_diagonal(q, 0.0)
    fill_diagonal(q, -q.sum(1))
    q /= -(pi * diag(q)).sum()
    root = sqrt(pi)
    lam, u = eigh(root[:, None] * q / root[None, :])
    return lam, u / root[:, None], u.T * root[None, :]


def yule_tree(n: int, rng: Generator) -> tuple[ndarray, ndarray]:
    """Draws a random ultrametric tree on n leaves:
    n -> (parent of every node, node heights); root 2n - 2."""
    parent = full(2 * n - 1, -1, dtype=int64)
    height = zeros(2 * n - 1)
    live = list(range(n))
    t = 0.0
    for node in range(n, 2 * n - 1):
        t += rng.exponential(1.0 / len(live))
        i, j = rng.choice(len(live), size=2, replace=False)
        a, b = live[i], live[j]
        parent[a] = parent[b] = node
        height[node] = t
        live = [x for k, x in enumerate(live) if k not in (i, j)] + [node]
    return parent, height


def pair_distances(
    parent: ndarray, height: ndarray, n: int, n_pairs: int, rng: Generator
) -> ndarray:
    """Measures random leaf pairs: (tree, n_pairs)
    -> twice the height of each pair's common ancestor."""
    out = empty(n_pairs)
    for k in range(n_pairs):
        a, b = rng.choice(n, size=2, replace=False)
        while a != b:
            if height[a] <= height[b]:
                a = parent[a]
            else:
                b = parent[b]
        out[k] = 2.0 * height[a]
    return out


def expected_p(
    d: ndarray, lam: ndarray, vr: ndarray, vl: ndarray, pi: ndarray, alpha: float
) -> ndarray:
    """Computes the expected share of differing sites under GTR+Gamma: tree distances -> p."""
    m = power(1.0 - lam[None, :] * d[:, None] / alpha, -alpha)
    same = einsum("i,ik,pk,ki->p", pi, vr, m, vl)
    return 1.0 - same


def simulate(n: int, sites: int, alpha: float, target_p: float, rng: Generator) -> ndarray:
    """Simulates an alignment on a Yule tree scaled to a median p-distance:
    -> n x sites codes 0..3."""
    lam, vr, vl = gtr_eigen(RATES, PI)
    parent, height = yule_tree(n, rng)
    d = pair_distances(parent, height, n, PAIRS_FOR_SCALE, rng)
    lo, hi = 1e-6, 1e6
    for _ in range(80):
        mid = sqrt(lo * hi)
        if median(expected_p(mid * d, lam, vr, vl, PI, alpha)) < target_p:
            lo = mid
        else:
            hi = mid
    scale = sqrt(lo * hi)
    rate = rng.gamma(alpha, 1.0 / alpha, size=sites)
    state = empty((2 * n - 1, sites), dtype=uint8)
    state[2 * n - 2] = rng.choice(4, size=sites, p=PI)
    for node in range(2 * n - 3, -1, -1):
        t = scale * (height[parent[node]] - height[node])
        decay = exp(lam[None, :] * t * rate[:, None])
        cum = cumsum((vr[state[parent[node]]] * decay) @ vl, axis=1)
        u = rng.random(sites) * cum[:, 3]
        state[node] = (u[:, None] > cum[:, :3]).sum(1)
    return state[:n]


def write_fasta(seqs: ndarray, path: Path) -> None:
    """Writes the coded alignment as FASTA, sequences t0, t1, ...: (codes, path) -> file."""
    letters = frombuffer(b"ACGT", dtype=uint8)
    with open(path, "wb") as handle:
        for k, row in enumerate(seqs):
            handle.write(b">t%d\n" % k + letters[row].tobytes() + b"\n")


# ---- p2v-bme's fit -------------------------------------------------------------------------------


def fit(p2vbme: str, fasta: Path, prefix: Path, pairs: int, threads: int) -> dict:
    """Fits GTR+GC with p2v-bme, no search: (alignment, prefix) -> its report <prefix>.json."""
    cmd = [
        p2vbme,
        "-s",
        str(fasta),
        "-m",
        "GTR+GC",
        "-p",
        str(prefix),
        "-a",
        "cpu",
        "-q",
        "-t",
        "random",
    ]
    cmd += [
        "--max-rounds",
        "0",
        "--no-branch-lengths",
        "--report-full",
        "--gtr-fit-pairs",
        str(pairs),
    ]
    run(cmd, check=True, env=dict(environ, OMP_NUM_THREADS=str(threads)))
    return read_json(Path(f"{prefix}.json"))


def main() -> None:
    parser = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("--sequences", type=int, default=1000)
    parser.add_argument("--replicates", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.5, 2.0])
    parser.add_argument("--sites", type=int, nargs="+", default=[1000, 5000, 50000])
    parser.add_argument(
        "--divergence",
        type=float,
        nargs="+",
        default=[0.10, 0.45],
        help="median p-distances",
    )
    parser.add_argument("--pairs", type=int, default=100000, help="p2v-bme --gtr-fit-pairs")
    parser.add_argument(
        "--threads", type=int, default=16, help="OMP_NUM_THREADS of each p2v-bme run"
    )
    parser.add_argument(
        "--keep-fasta",
        action="store_true",
        help="keep the simulated alignments next to the reports",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="fit again alignments whose report exists",
    )
    parser.add_argument("--results-dir", type=Path, default=Path(DEFAULT_RESULTS_DIR))
    parser.add_argument("--p2vbme", default=P2VBME)
    args = parser.parse_args()

    out_dir = args.results_dir / "gtr_fit"
    runs = out_dir / "runs"
    runs.mkdir(parents=True, exist_ok=True)
    true_rates = RATES * 6.0 / RATES.sum()
    lines = [COLUMNS]
    for alpha in args.alphas:
        for sites in args.sites:
            for p in args.divergence:
                for rep in args.replicates:
                    name = f"gtr_a{alpha:g}_L{sites}_p{p:.2f}_rep{rep}"
                    prefix = runs / name
                    report_path = Path(f"{prefix}.json")
                    if args.overwrite or not report_path.exists():
                        seed = SEED + 1000 * rep + int(100 * alpha) + sites + int(1000 * p)
                        fasta = runs / f"{name}.fa"
                        write_fasta(
                            simulate(args.sequences, sites, alpha, p, default_rng(seed)),
                            fasta,
                        )
                        fit(args.p2vbme, fasta, prefix, args.pairs, args.threads)
                        if not args.keep_fasta:
                            fasta.unlink()
                    report = read_json(report_path)
                    model, seconds = (
                        report["substitution_model"],
                        report["details"]["time"]["gtr_fit"],
                    )
                    rates = array(model["rates"])
                    err = float(sqrt(mean(log(rates / true_rates) ** 2)))
                    lines.append(
                        f"{alpha:g}\t{sites}\t{p:.2f}\t{rep}\t{model['alpha']:.6f}\t{err:.6f}\t"
                        + "\t".join(f"{x:.5f}" for x in rates)
                        + f"\t{model['fit']['iterations']}\t{seconds:.3f}\t{int(model['fit']['converged'])}"
                        + f"\t{model['fit']['pairs_used']}"
                    )
                    print(
                        f"{name}: alpha_hat {model['alpha']:.4f}, rate error {err:.4f}",
                        flush=True,
                    )
    (out_dir / "recovery.tsv").write_text("\n".join(lines) + "\n")
    print(f"run_gtr_fit_recovery: {len(lines) - 1} alignments in {out_dir / 'recovery.tsv'}")


if __name__ == "__main__":
    main()
