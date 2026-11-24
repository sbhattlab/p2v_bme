"""Runs every paper script and plot on tiny simulated inputs, on the CPU, as CI does
(a check, not a result).

Inputs:  build/p2v-bme, build/make_test_fasta and the paper's tools (cmake --preset default)
Options: --work-dir, --keep
Flow:    a 300-sequence alignment and its tree from make_test_fasta stand in for RNASim's
           |
           v
         download_data --dry-run;
         simulate_data (60 sequences; FastME sets of 30, also from a 600-sequence expansion);
         run_benchmarks, run_calibration, run_fastme, run_gtr_fit_recovery,
         run_baselines --run default on them;
         run_silva's steps up to the supports on the alignment with five sequences cut short
         (left out by --min-sites);
         --help of run_viridian, whose input (Viridian) has no tiny stand-in here
           |
           v
         make_tables, then each plot whose data the tiny runs make; --help of the others
Outputs: tmp/paper_check/ (removed at the end unless --keep);
         a failing step stops the check with its error

    pixi run -e paper check-paper
"""

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from os import environ
from pathlib import Path
from shutil import copyfile, rmtree
from subprocess import run
from sys import executable

from paper._config import DEFAULT_BUILD_DIR

SIZE = "60"  # sequences of the tiny sets
FASTME_SIZE = "30"  # tiny FastME sets: drawn from the tiny sets (as from 10^4) and their tenfold expansion (10^6)
CUT, CUT_TO = 5, 100  # the SILVA stand-in: sequences cut short, and their sites left
THREADS = "2"
# plots drawn from the tiny runs' tables; the others need downloaded or larger data (only --help)
PLOTS_DRAWN = (
    "fig2a_scaling_time",
    "fig2b_scaling_rf_error",
    "figS1b_scaling_rf_error",
    "figS1c_scaling_host_memory",
    "figS1d_scaling_gpu_memory",
)


def step(*args: str) -> None:
    """Runs one entry point with the paper environment's Python:
    (module, options) -> None; stops on failure."""
    print("check_entry_points:", " ".join(args), flush=True)
    run([executable, "-m", *args], check=True)


def main() -> None:
    parser = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("--work-dir", type=Path, default=Path("tmp/paper_check"))
    parser.add_argument("--keep", action="store_true", help="keep the work folder")
    args = parser.parse_args()
    environ["OMP_NUM_THREADS"] = THREADS
    work = args.work_dir
    data, results, tables = work / "datasets", work / "results", work / "tables"
    rmtree(work, ignore_errors=True)
    (data / "rnasim").mkdir(parents=True)

    # The stand-in for RNASim: an alignment and its true tree
    run(
        [f"{DEFAULT_BUILD_DIR}/make_test_fasta", work / "source", "300", "600", "7"],
        check=True,
    )
    copyfile(work / "source.fa", data / "rnasim" / "aln_dna.fa")
    copyfile(work / "source_true.nwk", data / "rnasim" / "true_topo.tree")

    cpu = ["--accelerator", "cpu", "--devices", THREADS]
    paths = ["--datasets-dir", str(data), "--results-dir", str(results)]
    step("paper.download_data", "--dry-run", "--datasets-dir", str(data))
    step("paper.simulate_data", "--sets", "rnasim", "jc10k", "yule10k", "--sizes", SIZE,
         "--replicates", "0", "1", "--datasets-dir", str(data))  # fmt: skip
    step("paper.simulate_data", "--sets", "fastme", "--sizes", FASTME_SIZE, "--fastme-base", SIZE,
         "--expansion-base", SIZE, "--replicates", "0", "--datasets-dir", str(data))  # fmt: skip
    step("paper.run_benchmarks", "--experiments", "scaling", "random_start", "representatives",
         "round_limit", "--sets", "rnasim", "jc10k", "yule10k", "--sizes", SIZE, "--replicates", "0", "1",
         "--caps", "16", "64", "--exact-max", "50", "--score-cap", "16", *cpu, *paths)  # fmt: skip
    step("paper.run_calibration", "--sets", f"rnasim_{SIZE}", f"yule10k_{SIZE}", "--bootstrap", "4",
         "--repeats", "2", *cpu, *paths)  # fmt: skip
    step(
        "paper.run_fastme",
        "--sets",
        "rnasim",
        "--sizes",
        FASTME_SIZE,
        "--replicates",
        "0",
        *cpu,
        *paths,
    )
    step("paper.run_gtr_fit_recovery", "--sequences", "40", "--replicates", "0", "--alphas", "0.5",
         "--sites", "1000", "--divergence", "0.1", "--threads", THREADS, "--results-dir", str(results))  # fmt: skip
    step("paper.run_baselines", "--run", "default", "--sets", "yule10k", "--sizes", SIZE, "--replicates",
         "0", "--threads", THREADS, *paths)  # fmt: skip
    # The stand-in for SILVA: the alignment with its first sequences cut short
    with open(work / "silva.fa", "w") as fo:
        for i, record in enumerate((work / "source.fa").read_text().split(">")[1:]):
            name, seq = record.split("\n", 1)
            seq = seq.replace("\n", "")
            if i < CUT:
                seq = seq[:CUT_TO] + "-" * (len(seq) - CUT_TO)
            fo.write(f">{name}\n{seq}\n")
    step("paper.run_silva", "--steps", "distinct", "tree", "replicates", "supports", "--alignment",
         str(work / "silva.fa"), "--bootstrap", "2", "--max-sites", "500", "--min-sites", "300", *cpu,
         "--results-dir", str(results))  # fmt: skip
    step("paper.run_viridian", "--help")
    step(
        "paper.make_tables",
        "--results-dir",
        str(results),
        "--tables-dir",
        str(tables),
    )

    img = ["--tables-dir", str(tables), "--img-dir", str(work / "img")]
    for script in sorted(Path("plots").glob("fig*.py")):
        if script.stem in PLOTS_DRAWN:
            step(f"plots.{script.stem}", *img)
        else:
            step(f"plots.{script.stem}", "--help")
    if not args.keep:
        rmtree(work)
    print("check_entry_points: every entry point ran")


if __name__ == "__main__":
    main()
