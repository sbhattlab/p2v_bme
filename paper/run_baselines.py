"""Comparison programs: FastTree, VeryFastTree, FastME, DIPPER, MAPLE and CMAPLE
(tables comparators, baseline_10k, maple; the comparator series of the scaling figures).

They do not depend on p2v-bme, so their recorded runs can be reused:
--source copies the paper's records unchanged into data/results/baselines/, with their provenance.
--run shows how they were made and makes them again,
in the same layout, family by family (one of the two is needed):
  default       FastTree and VeryFastTree, each from its own start tree:
                -nt -fastest -nosupport, -nocat on the JC69 sets (no rate variation simulated),
                -noml for the ME rows; 64 threads
  random_start  the same programs and FastME (BME SPR, its own JC69 distances)
                from one random start tree
  dipper        DIPPER on one GPU: -i m -o t -d 2 --force-binary; 12 threads
  maple         MAPLE and CMAPLE: their published trees and logs (CMAPLE benchmark sets)
                scored against the truth
Every tree is scored against the true tree by eval_cmaple_rf (true splits missed, all splits).

Inputs:  import: the paper's recorded runs, from a git repository (--source, --ref)
         or a folder (--source)
         --run: data/datasets/<set>/<n>/rep<r>.fa, rep<r>.true.nwk (simulate_data.py);
         MAPLE and CMAPLE: data/datasets/cmaple/simulated/ (download_data.py)
Options: --run FAMILY..., --sets, --sizes, --replicates, --threads, --gpu, --dipper, --source,
         --ref, --overwrite, paths
Flow:    import: each recorded file -> its folder of data/results/baselines/ (RECORDS)
                 (skipped if it exists)
         --run: each family x set x size x replicate
           |
           v
         the program (wall time, CPU time and peak memory from the OS)
           |      (skipped if its record exists)
           v
         eval_cmaple_rf against the true tree -> one JSON record,
         the program's version, host and threads
Outputs: data/results/baselines/
           default/runs.jsonl                   default pipelines, PASTA-tree sets <= 50,000
           default_100k/<tag>_rep<r>_100000_<method>.json
           default_yule/yule10k_rep<r>_<n>_<method>.json
           random_start/<tag>_N<n>_rep<r>.json  random start
           dipper/runs.jsonl                    DIPPER
           maple/cmaple_benchmark_summary.json  MAPLE, CMAPLE ("simulated": cmaple, maple)
           PROVENANCE.json                      import: commit, each file's recorded path and sha256

    pixi run -e paper python -m paper.run_baselines --source <repository of the records> --ref main
    pixi run -e paper python -m paper.run_baselines --run default --sets rnasim --sizes 1000 --replicates 0
"""

import hashlib
import json
import os
import re
import shutil
import socket
import subprocess
import time

from argparse import ArgumentParser, Namespace, RawDescriptionHelpFormatter
from contextlib import nullcontext
from datetime import datetime, timezone
from fnmatch import fnmatch
from pathlib import Path

from paper._config import (
    DEFAULT_DATASETS_DIR,
    DEFAULT_RESULTS_DIR,
    EVAL_CMAPLE_RF,
    FASTME,
    FASTTREE,
    P2VBME,
    VERYFASTTREE,
)
from paper.utils import read_json, run_p2vbme, with_edge_lengths, write_json

# The recorded files the tables and figures read (paths below the source's RECORDS_ROOT),
# and the folder of data/results/baselines/ each goes to (the same as --run writes)
RECORDS = (
    ("pasta_default/runs.jsonl", "default"),
    ("pasta_default/README.md", "default"),
    ("ml_default_100k/*.json", "default_100k"),
    ("ml_default_100k/*.time", "default_100k"),
    ("ml_default_100k/README.md", "default_100k"),
    ("yule_default/*.json", "default_yule"),
    ("yule_default/*.time", "default_yule"),
    ("dipper_h200/runs.jsonl", "dipper"),
    ("dipper_h200/README.md", "dipper"),
    ("cpu_rand_*.json", "random_start"),
    ("cmaple_benchmark_summary.json", "maple"),
)
RECORDS_ROOT = "results_hx2"
RANDOM_START_PREFIX = "cpu_rand_"  # recorded random-start files' prefix, dropped

SIZES = (1000, 5000, 10000, 50000, 100000)
# Sets, sizes and replicates of each family, as recorded
GRIDS = {
    "default": {"rnasim": SIZES, "jc10k": SIZES, "jc1k": (10000,), "yule10k": SIZES},
    "random_start": {"rnasim": SIZES, "jc10k": SIZES, "jc1k": (10000,)},
    "dipper": {
        "rnasim": (*SIZES, 1000000),
        "jc10k": (*SIZES, 1000000),
        "jc1k": (10000,),
        "yule10k": (*SIZES, 1000000),
    },
}
REPLICATES = 5  # up to 100,000 sequences; one above
THREADS = 64  # CPU programs, as recorded
DIPPER_THREADS = 12  # DIPPER: the CPU threads of every p2v-bme run of the paper
FAMILIES = ("default", "random_start", "dipper", "maple")
DEFAULT_METHODS = ("FastTree-ML", "VeryFastTree-ML", "FastTree-ME", "VeryFastTree-ME")
JC69_SETS_PREFIXES = ("jc", "yule")  # simulated under JC69: -nocat
RANDOM_START_EDGE = 0.05  # branch length of every edge of the random start tree, as recorded
DIPPER_FLAGS = ["-i", "m", "-o", "t", "-d", "2", "--force-binary"]
# MAPLE and CMAPLE: the CMAPLE benchmark's simulated sets (sequences, mean branch length),
# 30,000 sites
MAPLE_SIZES = ("100K", "200K", "500K")
MAPLE_BRANCH_LENGTHS = ("0.00005", "0.0001", "0.00015")
MAPLE_SITES = 30000


# ---- Import of the recorded results --------------------------------------------------------------


def recorded_files(source: Path, ref: str | None) -> tuple[list[str], str]:
    """Lists the recorded files below RECORDS_ROOT:
    (repository or folder, ref) -> (paths, commit or "")."""
    if ref is not None:
        commit = subprocess.run(
            ["git", "-C", str(source), "rev-parse", ref],
            check=True,
            capture_output=True,
            text=True,
        )
        out = subprocess.run(
            [
                "git",
                "-C",
                str(source),
                "ls-tree",
                "-r",
                "--name-only",
                ref,
                "--",
                RECORDS_ROOT,
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        paths = [p.removeprefix(f"{RECORDS_ROOT}/") for p in out.stdout.split()]
        return paths, commit.stdout.strip()
    root = source / RECORDS_ROOT
    return [str(p.relative_to(root)) for p in root.rglob("*") if p.is_file()], ""


def read_recorded(source: Path, ref: str | None, path: str) -> bytes:
    """Reads one recorded file: (source, ref, path below RECORDS_ROOT) -> its bytes."""
    if ref is not None:
        return subprocess.run(
            ["git", "-C", str(source), "show", f"{ref}:{RECORDS_ROOT}/{path}"],
            check=True,
            capture_output=True,
        ).stdout
    return (source / RECORDS_ROOT / path).read_bytes()


def target_of(path: str) -> str | None:
    """Places a recorded file in our layout:
    path below RECORDS_ROOT -> path below baselines/ (None: not read)."""
    for pattern, folder in RECORDS:
        if fnmatch(path, pattern):
            return f"{folder}/{Path(path).name.removeprefix(RANDOM_START_PREFIX)}"
    return None


def import_records(source: Path, ref: str | None, out: Path, overwrite: bool) -> None:
    """Copies the recorded comparator results unchanged into our layout:
    (source, ref) -> out/, out/PROVENANCE.json.

    PROVENANCE.json keeps the source's commit and each file's recorded path and sha256.
    """
    paths, commit = recorded_files(source, ref)
    files = {}
    for path in sorted(paths):
        name = target_of(path)
        if name is None:
            continue
        target = out / name
        if overwrite or not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(read_recorded(source, ref, path))
        files[name] = {
            "recorded": f"{RECORDS_ROOT}/{path}",
            "sha256": hashlib.sha256(target.read_bytes()).hexdigest(),
        }
    write_json(
        out / "PROVENANCE.json",
        {
            "ref": ref,
            "commit": commit,
            "imported": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "files": files,
        },
    )
    print(f"run_baselines: {len(files)} recorded files in {out}")


# ---- Runs ----------------------------------------------------------------------------------------


def timed(cmd: list[str], env: dict | None = None, stdout: Path | None = None) -> dict:
    """Runs a program and measures it:
    command -> {time, cpu_time_sec, peak_ram_mb} (wall s, CPU s, MiB)."""
    t0 = time.perf_counter()
    with open(stdout, "w", encoding="utf-8") if stdout else nullcontext(subprocess.DEVNULL) as sink:
        # wait4 reaps the process to read its resource use, so no with block
        proc = subprocess.Popen(  # pylint: disable=consider-using-with
            cmd, env=env, stdout=sink, stderr=subprocess.DEVNULL
        )
        _, status, usage = os.wait4(proc.pid, 0)
    wall = time.perf_counter() - t0
    if status != 0:
        raise RuntimeError(f"failed with status {status}: {' '.join(cmd)}")
    return {
        "time": wall,
        "cpu_time_sec": usage.ru_utime + usage.ru_stime,
        "peak_ram_mb": usage.ru_maxrss / 1024.0,
    }


def version(program: str) -> str:
    """Asks a program for its version: program -> its name and version, as it prints them."""
    for flag in ("-V", "-h", "-expert", "--version"):
        # A flag the program does not know fails: the next one is tried
        out = subprocess.run(
            [program, flag], capture_output=True, text=True, stdin=subprocess.DEVNULL, check=False
        )
        if m := re.search(
            r"[A-Za-z]+ \d+(\.\d+)+( (Double|Single) precision)?",
            out.stdout + out.stderr,
        ):
            return m.group(0)
    return "unknown"


def score(eval_rf: str, true_tree: Path, tree: Path, sites: int) -> dict:
    """Scores a tree against the true tree with eval_cmaple_rf:
    (truth, tree, sites) -> its JSON record."""
    out = subprocess.run(
        [eval_rf, str(true_tree), str(tree), str(sites)],
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    return json.loads(out.strip().splitlines()[-1])


def missed(scores: dict) -> dict:
    """Reduces eval_cmaple_rf's record to the true splits missed: record -> {rf, rf_pct}."""
    return {
        "rf": scores["n_true_splits"] - scores["common_all"],
        "rf_pct": scores["rf_all_pct"],
    }


def alignment_sites(fasta: Path) -> int:
    """Counts the sites of an aligned FASTA file from its first sequence: path -> sites."""
    sites = 0
    with open(fasta, encoding="utf-8") as handle:
        handle.readline()
        for line in handle:
            if line.startswith(">"):
                break
            sites += len(line.strip())
    return sites


def default_record(out: Path, tag: str, n: int, rep: int, method: str) -> Path:
    """Locates a default-pipeline record as recorded: (set, size, replicate, method) -> path."""
    name = f"{tag}_rep{rep}_{n}_{method}.json"
    if tag.startswith("yule"):
        return out / "default_yule" / name
    if n >= 100000:
        return out / "default_100k" / name
    return out / "default" / "runs" / name


def run_default(args, out: Path, tag: str, n: int, rep: int) -> None:
    """Runs FastTree and VeryFastTree in their default pipelines on one alignment:
    -> one record per method."""
    fasta = args.datasets_dir / tag / str(n) / f"rep{rep}.fa"
    true_tree = fasta.with_name(f"rep{rep}.true.nwk")
    for method in DEFAULT_METHODS:
        record_path = default_record(out, tag, n, rep, method)
        if record_path.exists() and not args.overwrite:
            continue
        flags = ["-nt", "-fastest", "-nosupport"]
        flags += ["-nocat"] if tag.startswith(JC69_SETS_PREFIXES) else []
        flags += ["-noml"] if method.endswith("-ME") else []
        tree = record_path.with_suffix(".nwk")
        tree.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, OMP_NUM_THREADS=str(args.threads))
        if method.startswith("FastTree"):
            binary, measured = args.fasttree, timed([args.fasttree, *flags, str(fasta)], env, tree)
        else:
            binary = args.veryfasttree
            cmd = [
                binary,
                *flags,
                "-threads",
                str(args.threads),
                "-out",
                str(tree),
                str(fasta),
            ]
            measured = timed(cmd, env)
        record = {
            "tag": tag,
            "N": n,
            "rep": rep,
            "method": method,
            "init_mode": "program default",
            **measured,
        }
        record.update(
            cpu_cores=args.threads,
            flags=" ".join(flags),
            binary=shutil.which(binary) or binary,
        )
        record.update(version=version(binary), host=socket.gethostname())
        record.update(missed(score(args.eval_cmaple_rf, true_tree, tree, alignment_sites(fasta))))
        record["rf_reference"] = "true tree; paper/tools/eval_cmaple_rf.cpp (all splits)"
        write_json(record_path, record)


def write_phylip_alignment(fasta: Path, out: Path) -> None:
    """Writes an aligned FASTA file as sequential PHYLIP for FastME: (fasta, out) -> file."""
    names, seqs = [], []
    with open(fasta, encoding="utf-8") as handle:
        for line in handle:
            if line.startswith(">"):
                names.append(line[1:].split()[0])
                seqs.append([])
            else:
                seqs[-1].append(line.strip())
    with open(out, "w", encoding="utf-8") as handle:
        handle.write(f" {len(names)} {len(''.join(seqs[0]))}\n")
        for name, parts in zip(names, seqs):
            handle.write(f"{name} {''.join(parts)}\n")


def run_random_start(args, out: Path, tag: str, n: int, rep: int) -> None:
    """Runs the programs from one random start tree on one alignment:
    -> random_start/<tag>_N<n>_rep<r>.json."""
    record_path = out / "random_start" / f"{tag}_N{n}_rep{rep}.json"
    if record_path.exists() and not args.overwrite:
        return
    fasta = args.datasets_dir / tag / str(n) / f"rep{rep}.fa"
    true_tree = fasta.with_name(f"rep{rep}.true.nwk")
    work = out / "random_start" / f"{tag}_rep{rep}_{n}"
    work.mkdir(parents=True, exist_ok=True)
    sites = alignment_sites(fasta)
    # The random start: p2v-bme's random tree (no search), every edge 0.05 long
    start_args = [
        "-s",
        str(fasta),
        "-m",
        "JC69",
        "-t",
        "random",
        "--max-rounds",
        "0",
        "--no-branch-lengths",
    ]
    run_p2vbme(
        args.p2vbme,
        [*start_args, "--seed", str(rep + 1), "-a", "cpu"],
        work / "start",
        args.overwrite,
    )
    start = with_edge_lengths(work / "start.nwk", RANDOM_START_EDGE, work / "start_lengths.nwk")
    record = {
        "tag": tag,
        "N": n,
        "L": sites,
        "rep": rep,
        "init_mode": "uniform_random",
        "cpu_cores": args.threads,
    }
    record.update(host=socket.gethostname(), max_splits=n - 3)
    methods: dict[str, dict] = {}
    versions: dict[str, str] = {}
    methods["Random-Init"] = {
        **missed(score(args.eval_cmaple_rf, true_tree, start, sites)),
        "time": 0.0,
        "peak_ram_mb": 0.0,
        "cpu_cores": 1,
        "cpu_time_sec": 0.0,
    }
    env = dict(os.environ, OMP_NUM_THREADS=str(args.threads))
    jc = ["-nocat"] if tag.startswith(JC69_SETS_PREFIXES) else []
    vft = [args.veryfasttree, "-nt", "-fastest"]
    tail = ["-intree", str(start), "-threads", str(args.threads), "-quiet"]
    ft = [args.fasttree, "-nt", "-fastest"]
    phylip = work / "aln.phy"
    # Each program up to the sizes it was recorded at
    runs = {
        "VFT-ME": (
            True,
            [
                *vft,
                "-noml",
                "-nosupport",
                *tail,
                "-out",
                str(work / "VFT-ME.nwk"),
                str(fasta),
            ],
            False,
        ),
        "VFT-ML": (
            n <= 10000,
            [
                *vft,
                *jc,
                "-nosupport",
                *tail,
                "-out",
                str(work / "VFT-ML.nwk"),
                str(fasta),
            ],
            False,
        ),
        "FastTree-ME": (
            n <= 50000 or (tag == "rnasim" and n <= 100000),
            [*ft, "-noml", "-nosupport", "-intree", str(start), "-quiet", str(fasta)],
            True,
        ),
        "FastTree-ML": (
            n <= 10000,
            [*ft, *jc, "-nosupport", "-intree", str(start), "-quiet", str(fasta)],
            True,
        ),
        "FastME-SPR": (
            n <= 5000,
            [
                args.fastme,
                "-i",
                str(phylip),
                "-dJ",
                "-u",
                str(start),
                "-s",
                "-T",
                str(args.threads),
            ],
            False,
        ),
    }
    for method, (wanted, cmd, to_stdout) in runs.items():
        if not wanted:
            continue
        tree = work / f"{method}.nwk"
        if method == "FastME-SPR":
            write_phylip_alignment(fasta, phylip)
            cmd = [*cmd, "-o", str(tree)]
        measured = timed(cmd, env, tree if to_stdout else None)
        methods[method] = {
            **missed(score(args.eval_cmaple_rf, true_tree, tree, sites)),
            **measured,
            "cpu_cores": args.threads,
        }
        versions[method] = version(cmd[0])
    phylip.unlink(missing_ok=True)
    record.update(methods=methods, versions=versions)
    write_json(record_path, record)


def run_dipper(args, out: Path, tag: str, n: int, rep: int) -> None:
    """Runs DIPPER on one GPU on one alignment: -> dipper/runs/dipper_<tag>_N<n>_rep<r>.json."""
    record_path = out / "dipper" / "runs" / f"dipper_{tag}_N{n}_rep{rep}.json"
    if record_path.exists() and not args.overwrite:
        return
    if not args.dipper:
        raise SystemExit(
            "run_baselines: --run dipper needs --dipper (DIPPER is not in the paper environment)"
        )
    fasta = args.datasets_dir / tag / str(n) / f"rep{rep}.fa"
    true_tree = fasta.with_name(f"rep{rep}.true.nwk")
    tree = record_path.with_suffix(".nwk")
    tree.parent.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, OMP_NUM_THREADS=str(args.threads))
    measured = timed([args.dipper, *DIPPER_FLAGS, "-I", str(fasta), "-O", str(tree)], env)
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader", "-i", args.gpu],
        capture_output=True,
        text=True,
        check=False,  # without nvidia-smi, the record keeps an empty GPU name
    ).stdout.strip()
    method = {
        **measured,
        **missed(score(args.eval_cmaple_rf, true_tree, tree, alignment_sites(fasta))),
    }
    method["cpu_cores"] = args.threads
    record = {
        "tag": tag,
        "N": n,
        "rep": rep,
        "cpu_cores": args.threads,
        "host": socket.gethostname(),
        "gpu_name": gpu,
    }
    record.update(max_splits=n - 3, flags=" ".join(DIPPER_FLAGS), version=version(args.dipper))
    record["methods"] = {"DIPPER-GPU": method}
    write_json(record_path, record)


def log_time_memory(log: Path) -> tuple[float | None, float | None]:
    """Reads a program's run time and peak memory from its log:
    path -> (seconds, MiB), None where absent."""
    if not log.exists():
        return None, None
    text = log.read_text(errors="replace")
    seconds = None
    if m := re.search(r"Runtime:\s*([0-9.]+)s", text):
        seconds = float(m.group(1))
    elif m := re.search(r"Elapsed \(wall clock\) time.*?:\s*(?:(\d+):)?(\d+):([0-9.]+)", text):
        seconds = int(m.group(1) or 0) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    elif m := re.search(
        r"Time spent in total updating the topology and branch lengths:\s*([0-9.]+)",
        text,
    ):
        seconds = float(m.group(1))
    ram = re.search(r"Maximum resident set size \(kbytes\):\s*(\d+)", text)
    return seconds, int(ram.group(1)) / 1024.0 if ram else None


def run_maple(args, out: Path) -> None:
    """Scores MAPLE's and CMAPLE's published trees of the CMAPLE benchmark:
    -> maple/cmaple_benchmark_summary.json."""
    record_path = out / "maple" / "cmaple_benchmark_summary.json"
    if record_path.exists() and not args.overwrite:
        return
    folder = args.datasets_dir / "cmaple" / "simulated"
    cases = []
    for bl in MAPLE_BRANCH_LENGTHS:
        for size in MAPLE_SIZES:
            true_tree = folder / f"tree_{size}_{bl}.nwk"
            case = {
                "size": size,
                "n_taxa": int(size[:-1]) * 1000,
                "branch_length": float(bl),
            }
            for key, tree, log in (
                (
                    "cmaple",
                    f"aln_{size}_{bl}.maple.treefile",
                    f"CMaple_{size}_{bl}.log",
                ),
                (
                    "maple",
                    f"aln_{size}_{bl}_wo_ref_tree.tree",
                    f"Maple_{size}_{bl}.log",
                ),
            ):
                seconds, ram = log_time_memory(folder / log)
                case[key] = {
                    "time_s": seconds,
                    "ram_mb": ram,
                    **score(args.eval_cmaple_rf, true_tree, folder / tree, MAPLE_SITES),
                }
            cases.append(case)
    write_json(record_path, {"simulated": cases})


def collect(folder: Path, out: Path) -> None:
    """Gathers one-record files into one JSONL file, as recorded:
    (folder of records, out) -> out."""
    records = [read_json(p) for p in sorted(folder.glob("*.json"))]
    if records:
        out.write_text("".join(json.dumps(r) + "\n" for r in records))


def main() -> None:
    parser = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument(
        "--run",
        nargs="+",
        choices=FAMILIES,
        help="run these families instead of importing",
    )
    parser.add_argument(
        "--source",
        type=Path,
        help="import: the paper's repository (with --ref) or a folder",
    )
    parser.add_argument("--ref", help="import: git ref of --source (default: --source is a folder)")
    parser.add_argument("--sets", nargs="+", help="--run: datasets (default: each family's)")
    parser.add_argument(
        "--sizes", nargs="+", type=int, help="--run: sequences (default: each family's)"
    )
    parser.add_argument(
        "--replicates",
        nargs="+",
        type=int,
        help="--run: only these replicates (default: 0-4, only 0 from 10^6)",
    )
    parser.add_argument("--threads", type=int, help="--run: CPU threads (default: 64, DIPPER 12)")
    parser.add_argument("--gpu", default="0", help="--run dipper: GPU index (CUDA_VISIBLE_DEVICES)")
    parser.add_argument("--dipper", help="--run dipper: the DIPPER program")
    parser.add_argument("--overwrite", action="store_true", help="import or run again what exists")
    parser.add_argument("--datasets-dir", type=Path, default=Path(DEFAULT_DATASETS_DIR))
    parser.add_argument("--results-dir", type=Path, default=Path(DEFAULT_RESULTS_DIR))
    parser.add_argument("--p2vbme", default=P2VBME)
    parser.add_argument("--eval-cmaple-rf", default=EVAL_CMAPLE_RF)
    parser.add_argument("--fasttree", default=FASTTREE)
    parser.add_argument("--veryfasttree", default=VERYFASTTREE)
    parser.add_argument("--fastme", default=FASTME)
    args = parser.parse_args()

    out = args.results_dir / "baselines"
    if not args.run:
        if args.source is None:
            parser.error("give --source (and --ref) to import the recorded results, or --run")
        import_records(args.source, args.ref, out, args.overwrite)
        return
    steps = {
        "default": run_default,
        "random_start": run_random_start,
        "dipper": run_dipper,
    }
    for family in args.run:
        if family == "maple":
            run_maple(args, out)
            continue
        threads = args.threads or (DIPPER_THREADS if family == "dipper" else THREADS)
        family_args = Namespace(**{**vars(args), "threads": threads})
        for tag in args.sets or GRIDS[family]:
            for n in args.sizes or GRIDS[family][tag]:
                replicates = (
                    args.replicates
                    if args.replicates is not None
                    else range(REPLICATES if n <= 100000 else 1)
                )
                for rep in replicates:
                    steps[family](family_args, out, tag, n, rep)
                    print(f"run_baselines: {family} {tag} {n} rep{rep} done", flush=True)
    collect(out / "default" / "runs", out / "default" / "runs.jsonl")
    collect(out / "dipper" / "runs", out / "dipper" / "runs.jsonl")


if __name__ == "__main__":
    main()
