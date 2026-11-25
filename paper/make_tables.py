"""Every table of the paper and the data behind its figures,
from the raw results (Tables S1-S15; Figs 2, 3, S1-S5).

Inputs:  data/results/                                    (the run_*.py scripts)
           scaling/, random_start/, seqlen/,              our runs:
           representatives/, pandemic/                    <set>/<n>/rep<r>_seed<s>[_cap<P>].json
                                                          (vecme report) with .vs_true.json,
                                                          .backbone.json, .vs_cap64.json
                                                          (vecme --supports), .score.json,
                                                          .eval.json
           calibration/, fastme/, gtr_fit/, silva/,       records scored from our trees,
           viridian/,                                     in the paper's file formats
           seqlen/diagnose_10k_sequences.jsonl
           baselines/                                     the comparison programs' recorded results
                                                          (run_baselines.py)
Options: --results-dir, --tables-dir, --tables (default: all), --overwrite
Flow:    each table
           |
           v
         its results -> a table                           (a table whose results are missing is
           |                                              skipped, with a note)
           v
         data/tables/tableS<N>_<name>.md (df.to_markdown), and the data a figure reads as a .tsv
Outputs: data/tables/tableS<N>_<name>.md, numbered as in the supplement:
         S1 representative_sensitivity, S2 gtr_fit, S3 fastme, S4 scaling, S5 maple, S6 bootstrap,
         S7 comparators, S8 baseline_10k, S9 seqlen, S10 random_tree, S11 viridian_support,
         S12 viridian_variants, S13 viridian_thresholds, S14 silva_support, S15 silva_gtdb;
         data/tables/<data>.tsv, the figures' data: scaling, random_tree, maple, silva_support,
         scaling_methods, silva_clades, silva_clade_paths, viridian_variants,
         viridian_substitutions, viridian_resolution

    pixi run -e paper python -m paper.make_tables
    pixi run -e paper python -m paper.make_tables --tables scaling maple --overwrite
"""

# pylint: disable=redefined-outer-name

import json
import re
import sys

from argparse import ArgumentParser, RawDescriptionHelpFormatter
from pathlib import Path

import numpy as np
import pandas as pd

from paper._config import DEFAULT_RESULTS_DIR, DEFAULT_TABLES_DIR
from paper.utils import read_json

SIZES_EXPANDED = (
    1_000_000  # from here on, the PASTA-tree datasets expand the 100,000-leaf alignment (backbone)
)
BACKBONE_SIZE = 100_000
DATASETS = {"rnasim": "RNASim", "jc10k": "JC"}
TIMING = "initialisation+refinement+branch estimation; excludes input conversion/loading and diagnostic RF evaluation"
SD = "sample SD (ddof=1); blank for one measurement"

GPU_MODELS = {
    "NVIDIA H200": "H200",
    "NVIDIA H100 80GB HBM3": "H100",
    "NVIDIA RTX PRO 6000 Blackwell Server Edition": "RTX PRO 6000",
}
# Comparison programs: method name in their records -> short key (the plots use the same keys)
ALIAS = {
    "DIPPER-GPU": "DIPPER",
    "VFT-ML": "VFT-ML",
    "VeryFastTree-ML": "VFT-ML",
    "FastTree-ML": "FastTree-ML",
    "VFT-ME": "VFT-ME",
    "VeryFastTree-ME": "VFT-ME",
    "FastTree-ME": "FastTree-ME",
}

# RNASim calibration: bootstrap replicates and repeat searches per reference tree
# (run_calibration.py)
CALIBRATION_BOOTSTRAPS = 64
CALIBRATION_SEEDS = 8


# GTDB scores: the ranks of the table and the trees scored
RANKS = ["domain", "phylum", "class", "order", "family", "genus", "species"]
GTDB_TREES = {
    "full": r"\method, within the full tree",
    "ours": r"\method, these sequences alone",
    "ft_me": "FastTree, distance stage only",
    "ft_ml": "FastTree, with likelihood",
}

# SARS-CoV-2 tables
VIRIDIAN_THRESHOLDS = (
    "0.7",
    "0.75",
    "0.8",
    "0.9",
)  # rows of the threshold table, as written in the record
VIRIDIAN_PEAKS = (
    ("2021-06", "June 2021"),
    ("2022-01", "January 2022"),
)  # months whose median the table quotes
VIRIDIAN_BINS = {
    (0.5, 1.5): "1",
    (1.5, 2.5): "2",
    (2.5, 3.5): "3",
    (3.5, 5.5): "4–5",
    (5.5, None): "≥ 6",
}

# What a table function returns: file name -> its table (.tsv) or text (.md)
Outputs = dict[str, pd.DataFrame | list[dict] | str]

RUN = re.compile(r"rep(\d+)_seed(\d+)(?:_cap(\d+))?")


class MissingResults(Exception):
    """A table's results are not there (yet)."""


# ---- Reading results ----


def one_file(folder: Path, pattern: str) -> Path:
    """Finds the one file of a folder matching a pattern: (folder, glob) -> path."""
    paths = sorted(folder.glob(pattern))
    if len(paths) != 1:
        raise MissingResults(f"{folder}/{pattern} ({len(paths)} files)")
    return paths[0]


def short_gpu(name: str) -> str:
    """Shortens a GPU name: 'NVIDIA H200' -> 'H200'."""
    return GPU_MODELS.get(name, name.removeprefix("NVIDIA "))


def threads_of(report: Path, details: dict) -> int | str:
    """Reads the CPU threads of one of our runs:
    (report, its details) -> the count ('' if not recorded).

    CPU runs: their workers ("CPU: 16 threads" in the report);
    GPU runs: the host threads that run_vecme recorded in <prefix>.host.json.
    """
    workers = [d for d in details["devices"] if d.startswith("CPU: ")]
    if workers:
        return int(workers[0].split()[1])
    host = report.with_name(f"{report.stem}.host.json")
    return json.loads(host.read_text())["cpu_threads"] if host.exists() else ""


def rf_from(path: Path) -> float | str:
    """Reads the RF error of a vecme --supports comparison:
    report -> 100 - mean exact support ('' if none).

    For binary trees on the same leaves, RF / (2 (n - 3)) = 1 - (shared splits) / (n - 3).
    """
    return 100.0 - json.loads(path.read_text())["supports"]["mean_exact"] if path.exists() else ""


def sidecar(report: Path, name: str) -> Path:
    """Names a file written next to one of our reports:
    (rep0_seed1.json, "score") -> rep0_seed1.score.json."""
    return report.with_name(f"{report.stem}.{name}.json")


# Experiments already warned about reports without gpu_memory_peak_mib
# (our_runs reads an experiment once per table)
warned_no_peak: set[str] = set()


def our_runs(results: Path, experiment: str) -> pd.DataFrame:
    """Reads our runs of one experiment:
    (results dir, experiment) -> one row per vecme report, in run order, blanks as NaN."""
    rows: list[dict] = []
    no_peak = 0  # GPU reports without gpu_memory_peak_mib
    for path in (results / experiment).glob("*/*/*.json"):
        m = RUN.fullmatch(path.stem)
        if m is None:  # .vs_true.json, .score.json, ...: their stems hold a dot
            continue
        report = json.loads(path.read_text())
        details = report["details"]
        t = details["time"]
        gpus = [d.split(": ", 1)[1] for d in details["devices"] if d.startswith("GPU")]
        # Reports from before gpu_memory_peak_mib measured the whole card instead: no value for them
        no_peak += bool(gpus) and "gpu_memory_peak_mib" not in details["memory"]
        score = sidecar(path, "score")
        evaluation = sidecar(path, "eval")
        backbone = sidecar(path, "backbone")
        rows.append(
            {
                "set": path.parent.parent.name,
                "n": int(path.parent.name),
                "rep": int(m[1]),
                "seed": int(m[2]),
                "cap": int(m[3]) if m[3] else details["settings"].get("representative_cap", ""),
                "sites": details["input"]["sites"],
                "time_s": t["start_tree"] + t["search"] + t["branch_lengths"],
                "topology_time_s": t["start_tree"] + t["search"],
                "branch_time_s": t["branch_lengths"],
                "host_memory_mb": details["memory"]["host_ram_mib"],
                "gpu_memory_peak_mib": (
                    details["memory"].get("gpu_memory_peak_mib", "") if gpus else ""
                ),
                "gpu_count": len(gpus),
                "gpu_model": short_gpu(gpus[0]) if gpus else "",
                "gpu_models": frozenset(gpus),
                "cpu_threads": threads_of(path, details),
                "start_tree": details["settings"].get("start_tree", ""),
                "bme_length": report["bme_length"],
                "rf_pct": rf_from(sidecar(path, "vs_true")),
                "backbone_rf_pct": (
                    json.loads(backbone.read_text())["backbone_rf_pct"] if backbone.exists() else ""
                ),
                "rf_vs_cap64_pct": rf_from(sidecar(path, "vs_cap64")),
                "scored_length": (
                    json.loads(score.read_text())["bme_length"] if score.exists() else ""
                ),
                "evaluation": json.loads(evaluation.read_text()) if evaluation.exists() else {},
                "source_path": str(path.relative_to(results)),
            }
        )
    rows.sort(
        key=lambda r: (
            r["set"],
            r["n"],
            r["rep"],
            r["seed"],
            r["cap"] if r["cap"] != "" else 0,
        )
    )
    if not rows:
        raise MissingResults(f"{results / experiment}/<set>/<n>/rep<r>_seed<s>.json")
    if no_peak and experiment not in warned_no_peak:
        warned_no_peak.add(experiment)
        print(
            f"make_tables: {experiment}: {no_peak} GPU reports have no gpu_memory_peak_mib"
            " (written by an earlier vecme); their GPU memory is left blank",
            file=sys.stderr,
        )
    return pd.DataFrame(rows).replace("", np.nan).infer_objects()


def one(values: set, what: str):
    """Returns the one value of a set:
    (values, what they are) -> value; mixed values are an error."""
    if len(values) != 1:
        raise SystemExit(f"mixed {what} in the run records: {sorted(map(str, values))}")
    return next(iter(values))


def one_of(what: str):
    """Makes an aggregator for values that must agree:
    what they are -> (values -> the one value)."""
    return lambda values: one(set(values), what)


def dipper_runs(results: Path) -> list[dict]:
    """Reads DIPPER's runs (one GPU each): results -> records with tag, N, methods."""
    return pd.read_json(
        results / "baselines" / "dipper" / "runs.jsonl", lines=True, precise_float=True
    ).to_dict("records")


def default_runs(results: Path) -> list[dict]:
    """Reads FastTree and VeryFastTree in their default pipelines on the PASTA-tree datasets:
    -> one record per run.

    Runs started from our tree (method names ending in -fromP2V) are left out.
    """
    path = results / "baselines" / "default" / "runs.jsonl"
    records = pd.read_json(path, lines=True, precise_float=True).to_dict("records")
    for path in sorted((results / "baselines" / "default_100k").glob("*.json")):
        records.append(json.loads(path.read_text()))
    return [r for r in records if r["method"] in ALIAS]


def optional(read, results: Path) -> list[dict]:
    """Reads comparison runs that a table can do without:
    (reader, results) -> its records, or none."""
    try:
        return read(results)
    except (MissingResults, FileNotFoundError) as missing:
        print(f"make_tables: no comparison runs: {missing}", file=sys.stderr)
        return []


# ---- Formatting ----


def rounded(table: pd.DataFrame, decimals: dict[str, int]) -> pd.DataFrame:
    """Rounds columns as the paper shows them:
    (table, {column: decimals}) -> the table, whole numbers where 0 decimals, no -0."""
    out = table.copy()
    for column, d in decimals.items():
        values = pd.to_numeric(out[column]).astype("Float64").round(d) + 0.0
        out[column] = values.astype("Int64") if d == 0 else values
    return out


def method_grid(
    means: pd.DataFrame, methods: list[str], names: list[str], sizes: list
) -> pd.DataFrame:
    """Lays out each method's mean RF error and time by size:
    (means by method and n, method keys, their names, sizes) -> one row per size."""
    rf = means["rf_pct"].unstack("method").reindex(index=sizes, columns=methods)
    time = means["time_s"].unstack("method").reindex(index=sizes, columns=methods)
    rf = rf.set_axis([f"{name} RF (%)" for name in names], axis=1)
    time = time.set_axis([f"{name} time (s)" for name in names], axis=1)
    table = pd.concat([rf, time], axis=1).rename_axis("Leaves").reset_index()
    return rounded(table, dict.fromkeys(rf.columns, 2) | dict.fromkeys(time.columns, 3))


def scaling_rows(results: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aggregates the sample-and-place runs on the PASTA-tree datasets:
    results -> (per size, per run)."""
    runs = our_runs(results, "scaling")
    runs = runs[runs["set"].isin(list(DATASETS))]
    if runs.empty:
        raise MissingResults(f"{results}/scaling/{{{','.join(DATASETS)}}}")
    runs = runs.assign(order=runs["set"].map(list(DATASETS).index))
    runs = runs.sort_values(["order", "n"], kind="stable")
    per_run = pd.DataFrame(
        {
            "dataset": runs["set"].map(DATASETS),
            "n": runs["n"],
            "initialisation": "sample",
            "replicate": runs.groupby(["set", "n"]).cumcount(),
            "alignment_columns": runs["sites"],
            "time_s": runs["time_s"],
            "topology_time_s": runs["topology_time_s"],
            "branch_time_s": runs["branch_time_s"],
            "rf_pct": runs["rf_pct"],
            # at 100,000 leaves the backbone is the whole tree
            "backbone_rf_pct": runs["rf_pct"].where(
                runs["n"] == BACKBONE_SIZE, runs["backbone_rf_pct"]
            ),
            "gpu_count": runs["gpu_count"],
            "gpu_model": runs["gpu_model"],
            "cpu_threads": runs["cpu_threads"],
            "recorded_init_mode": runs["start_tree"],
            "host_memory_mb": runs["host_memory_mb"],
            "gpu_memory_peak_mib": runs["gpu_memory_peak_mib"],
            "source_path": runs["source_path"],
            "source_key": "vecme report",
            "timing_scope": TIMING,
        }
    )
    aggregate = per_run.groupby(["dataset", "n"], sort=False).agg(
        n_replicates=("replicate", "size"),
        time_s_mean=("time_s", "mean"),
        time_s_sd=("time_s", "std"),
        rf_pct_mean=("rf_pct", "mean"),
        rf_pct_sd=("rf_pct", "std"),
        backbone_rf_pct_mean=("backbone_rf_pct", "mean"),
        backbone_rf_pct_sd=("backbone_rf_pct", "std"),
        gpu_count=("gpu_count", one_of("GPU counts")),
        gpu_model=("gpu_model", one_of("GPU models")),
        cpu_threads=("cpu_threads", one_of("CPU thread counts")),
        source_paths=("source_path", "|".join),
    )
    aggregate = aggregate.reset_index()
    aggregate.insert(2, "initialisation", "sample")
    # the larger datasets expand replicate 0: its error on the whole tree
    first = runs[(runs["n"] == BACKBONE_SIZE) & (runs["rep"] == 0)].groupby("set")["rf_pct"].first()
    at_base = aggregate["n"] == BACKBONE_SIZE
    aggregate.loc[at_base, "backbone_rf_pct_mean"] = aggregate.loc[at_base, "dataset"].map(
        {DATASETS[k]: v for k, v in first.items()}
    )
    aggregate.loc[at_base, "backbone_rf_pct_sd"] = pd.NA
    aggregate = aggregate.assign(
        source_key="vecme report",
        timing_scope=TIMING,
        large_scale_design=aggregate["n"].map(
            lambda n: "100k-backbone microclade expansion" if n >= SIZES_EXPANDED else ""
        ),
        sd_definition=SD,
    )
    return aggregate, per_run


def table_scaling(results: Path) -> Outputs:
    """Builds Table S4 (scaling) and Fig. 2a,b's data:
    results -> {tableS4_scaling.md, scaling.tsv}."""
    scale, _ = scaling_rows(results)
    table = pd.DataFrame(
        {
            "Data": scale["dataset"],
            "Leaves": scale["n"],
            "Replicates": scale["n_replicates"],
            "Time (s)": scale["time_s_mean"],
            "Time SD": scale["time_s_sd"],
            "RF (%)": scale["rf_pct_mean"],
            "RF SD": scale["rf_pct_sd"],
            "Backbone RF (%)": scale["backbone_rf_pct_mean"],
            "GPUs": scale["gpu_count"],
        }
    )
    decimals = {"Time (s)": 3, "Time SD": 3, "RF (%)": 2, "RF SD": 2, "Backbone RF (%)": 2}
    return {"tableS4_scaling.md": rounded(table, decimals), "scaling.tsv": scale}


def yule_runs(results: Path) -> pd.DataFrame:
    """Reads our runs on the random-tree (Yule) JC alignments: results -> rows of run_frame."""
    runs = our_runs(results, "scaling")
    runs = runs[runs["set"] == "yule10k"]
    if runs.empty:
        raise MissingResults(f"{results}/scaling/yule10k")
    return runs


def table_random_tree(results: Path) -> Outputs:
    """Builds Table S10 (random_tree) and Fig. 2's third series:
    results -> {tableS10_random_tree.md, random_tree.tsv}."""
    ours = yule_runs(results)
    rows = ours.groupby("n", as_index=False).agg(
        n_replicates=("rep", "size"),
        time_s_mean=("time_s", "mean"),
        time_s_sd=("time_s", "std"),
        rf_pct_mean=("rf_pct", "mean"),
        rf_pct_sd=("rf_pct", "std"),
        gpu_count=("gpu_count", one_of("GPU counts")),
        gpu_model=("gpu_model", one_of("GPU models")),
        cpu_threads=("cpu_threads", one_of("CPU thread counts")),
        source_paths=("source_path", "|".join),
    )
    rows.insert(0, "dataset", "JC random tree")
    rows.insert(2, "initialisation", "sample")
    rows = rows.assign(source_key="vecme report", timing_scope=TIMING, sd_definition=SD)

    methods = ["P2V", "DIPPER", "FastTree-ME", "VFT-ME", "FastTree-ML", "VFT-ML"]
    head = [
        "VecME",
        "DIPPER",
        "FastTree ME",
        "VeryFastTree ME",
        "FastTree ML",
        "VeryFastTree ML",
    ]
    cpu = [
        json.loads(p.read_text()) for p in (results / "baselines" / "default_yule").glob("*.json")
    ]
    dipper = [d for d in optional(dipper_runs, results) if d["tag"] == "yule10k"]
    others = [(ALIAS[d["method"]], d["N"], d["time"], d["rf_pct"]) for d in cpu]
    others += [
        ("DIPPER", d["N"], *(d["methods"]["DIPPER-GPU"][k] for k in ("time", "rf_pct")))
        for d in dipper
    ]
    columns = ["method", "n", "time_s", "rf_pct"]
    runs = pd.concat([ours.assign(method="P2V")[columns], pd.DataFrame(others, columns=columns)])
    means = runs.groupby(["method", "n"]).mean()
    table = method_grid(means, methods, head, sorted(set(ours["n"])))
    return {"tableS10_random_tree.md": table, "random_tree.tsv": rows}


def table_maple(results: Path) -> Outputs:
    """Builds Table S5 (maple) and Fig. 3a,b's data: results -> {tableS5_maple.md, maple.tsv}."""
    ours = our_runs(results, "pandemic").to_dict("records")
    record = read_json(results / "baselines" / "maple" / "cmaple_benchmark_summary.json")
    rows: list[dict] = []
    for index, case in enumerate(record["simulated"]):
        n, length = case["n_taxa"], case["branch_length"]
        mine = [
            r
            for r in ours
            if r["n"] == n and abs(float(r["set"].removeprefix("cmaple_")) - length) < 1e-12
        ]
        if len(mine) != 1:
            raise MissingResults(f"{results}/pandemic/cmaple_{length}/{n}: {len(mine)} runs")
        r = mine[0]
        rows.append(
            {
                "n": n,
                "mean_branch_length": length,
                "method": "VecME",
                "time_s": r["time_s"],
                "rf_pct": r["evaluation"]["rf_all_pct"],
                "resolved_rf_pct": r["evaluation"]["rf_res10_pct"],
                "gpu_count": r["gpu_count"],
                "cpu_threads": r["cpu_threads"],
                "host_memory_mb": r["host_memory_mb"],
                "gpu_model": r["gpu_model"],
                "representative_cap": r["cap"],
                "exact_check_eta": "",
                "source_path": r["source_path"],
                "source_key": "vecme report; .eval.json",
                "timing_scope": TIMING,
                "hardware_note": "",
                "resolved_metric": "fraction of true splits missing among branches >=1 expected substitution",
            }
        )
        for name, key in (("CMAPLE", "cmaple"), ("MAPLE", "maple")):
            m = case[key]
            rows.append(
                {
                    "n": n,
                    "mean_branch_length": length,
                    "method": name,
                    "time_s": m["time_s"],
                    "rf_pct": m["rf_all_pct"],
                    "resolved_rf_pct": m["rf_res10_pct"],
                    "gpu_count": 0,
                    "cpu_threads": "",
                    "host_memory_mb": m["ram_mb"],
                    "gpu_model": "",
                    "representative_cap": "",
                    "exact_check_eta": "",
                    "source_path": "baselines/maple/cmaple_benchmark_summary.json",
                    "source_key": f"simulated[{index}].{key}",
                    "timing_scope": (
                        "archived internal Runtime"
                        if name == "CMAPLE"
                        else "archived process wall time"
                    ),
                    "hardware_note": "archived CPU benchmark; CPU model/thread allocation not established locally",
                    "resolved_metric": "fraction of true splits missing among branches >=1 expected substitution",
                }
            )
    table = pd.DataFrame(
        [
            {
                "Leaves": p["n"],
                "Mean branch length": p["mean_branch_length"],
                "VecME time (s)": p["time_s"],
                "CMAPLE time (s)": c["time_s"],
                "MAPLE time (s)": m["time_s"],
                "VecME RF (%)": p["rf_pct"],
                "CMAPLE RF (%)": c["rf_pct"],
                "MAPLE RF (%)": m["rf_pct"],
            }
            for p, c, m in zip(rows[0::3], rows[1::3], rows[2::3])
        ]
    )
    decimals = {column: 1 if "time" in column else 2 for column in table.columns[2:]}
    return {"tableS5_maple.md": rounded(table, decimals), "maple.tsv": rows}


def table_bootstrap(results: Path) -> Outputs:
    """Builds Table S6 (bootstrap): results -> {tableS6_bootstrap.md}."""
    folder = results / "calibration"
    data = read_json(folder / "rnasim_bootstrap_calibration_summary.json")
    y = read_json(folder / "calibration_1000000.json")
    pandemic = read_json(folder / "maple_bootstrap_calibration_summary.json")
    rows: list[dict] = []
    sizes = [(100000, "N100k"), (1000000, "N1M")]
    for n, prefix in sizes:
        for kind in ("boot", "seed"):
            m = data[f"{prefix}_{kind}"]
            rows.append(
                {
                    "dataset": "RNASim",
                    "n": n,
                    "replicate_type": (
                        "column bootstrap" if kind == "boot" else "fixed-alignment seeds"
                    ),
                    "n_replicates": CALIBRATION_BOOTSTRAPS if kind == "boot" else CALIBRATION_SEEDS,
                    "n_branches": m["n_branches"],
                    "mean_tbe_pct": m["mean_tbe_pct"],
                    "mean_exact_support_pct": m["mean_exact_pct"],
                    "mean_truth_transfer_pct": m.get("mean_true_transfer_pct", ""),
                    "mean_exact_truth_pct": m.get("mean_correct_pct", ""),
                    "source_path": "calibration/rnasim_bootstrap_calibration_summary.json",
                    "source_key": f"{prefix}_{kind}",
                    "replicate_count_source": "paper/run_calibration.py",
                    "interpretation": "transfer agreement is not probability of an entire split being correct",
                }
            )
    for kind, count in (("boot", y["bootstrap_trees"]), ("seed", y["repeat_searches"])):
        is_boot = kind == "boot"
        m = y if is_boot else y["repeat"]
        rows.append(
            {
                "dataset": "Random tree",
                "n": y["n_leaves"],
                "replicate_type": "column bootstrap" if is_boot else "fixed-alignment seeds",
                "n_replicates": count,
                "n_branches": y["branches"],
                "mean_tbe_pct": m["mean_tbe_pct"],
                "mean_exact_support_pct": m["mean_exact_pct"],
                "mean_truth_transfer_pct": y["mean_true_transfer_pct"] if is_boot else "",
                "mean_exact_truth_pct": 100.0 - y["wrong_pct"] if is_boot else "",
                "source_path": "calibration/calibration_1000000.json",
                "source_key": "top level" if is_boot else "repeat",
                "replicate_count_source": "paper/run_calibration.py",
                "interpretation": "independent lineages: no sequences added by expansion",
            }
        )
    all_row = pandemic["by_size"][0]
    mutated, zero = pandemic["regimes"]["all_mutated"], pandemic["regimes"]["zero_snp"]
    for label, entry, key_name in (
        ("Pandemic (all splits)", all_row, "by_size[0]"),
        (
            f"Pandemic (≥ 1 SNP, {mutated['share_pct']:.1f}%)",
            mutated,
            "regimes.all_mutated",
        ),
        (
            f"Pandemic (≈ 0 SNPs, {zero['share_pct']:.1f}%)",
            zero,
            "regimes.zero_snp",
        ),
    ):
        for kind, count in (("boot", pandemic["B_boot"]), ("seed", pandemic["S_seed"])):
            is_boot = kind == "boot"
            rows.append(
                {
                    "dataset": label,
                    "n": pandemic["N"],
                    "replicate_type": "column bootstrap" if is_boot else "fixed-alignment seeds",
                    "n_replicates": count,
                    "n_branches": entry["branches"],
                    "mean_tbe_pct": entry["boot_tbe_pct"] if is_boot else entry["seed_tbe_pct"],
                    "mean_exact_support_pct": (
                        entry["boot_fbp_pct"] if is_boot else entry["seed_exact_pct"]
                    ),
                    "mean_truth_transfer_pct": entry["true_tbe_pct"] if is_boot else "",
                    "mean_exact_truth_pct": entry["true_exact_pct"] if is_boot else "",
                    "source_path": "calibration/maple_bootstrap_calibration_summary.json",
                    "source_key": f"{key_name}.{kind}",
                    "replicate_count_source": "paper/run_calibration.py",
                    "interpretation": "distinguishes mutated splits from zero-mutation pandemic polytomies",
                }
            )
    frame = pd.DataFrame(rows).replace("", pd.NA)
    table = pd.DataFrame(
        {
            "Data": frame["dataset"],
            "Leaves": frame["n"],
            "Replicates": frame["replicate_type"],
            "R": frame["n_replicates"],
            "TBE (%)": frame["mean_tbe_pct"],
            "Exact support (%)": frame["mean_exact_support_pct"],
            "Truth transfer (%)": frame["mean_truth_transfer_pct"],
            "Exact truth (%)": frame["mean_exact_truth_pct"],
        }
    )
    return {"tableS6_bootstrap.md": rounded(table, dict.fromkeys(table.columns[4:], 2))}


def table_baseline_10k(results: Path) -> Outputs:
    """Builds Table S8 (baseline_10k): results -> {tableS8_baseline_10k.md}."""
    ours = {"sample": our_runs(results, "scaling"), "random": our_runs(results, "random_start")}
    dipper = dipper_runs(results)
    cases = [
        ("VecME sample", "sample", "sample"),
        ("VecME random", "random", "supplied random tree"),
        ("DIPPER", "DIPPER-GPU", "de novo"),
        ("FastTree ME", "FastTree-ME", "supplied random tree"),
        ("VeryFastTree ME", "VFT-ME", "supplied random tree"),
        ("FastTree ML", "FastTree-ML", "supplied random tree"),
        ("VeryFastTree ML", "VFT-ML", "supplied random tree"),
    ]
    runs: list[tuple] = []  # (dataset, method, time, RF, CPU threads, GPUs, source) per run
    for tag, label in DATASETS.items():
        sources = {
            "DIPPER-GPU": [
                (f"baselines/dipper/runs.jsonl[{tag},N=10000,rep={d['rep']}]", d)
                for d in dipper
                if d["tag"] == tag and d["N"] == 10000
            ],
            "": [
                (str(p.relative_to(results)), json.loads(p.read_text()))
                for p in sorted(
                    (results / "baselines" / "random_start").glob(f"{tag}_N10000_rep*.json")
                )
            ],
        }
        for method, key, _ in cases:
            found = len(runs)
            if key in ours:
                mine = ours[key][(ours[key]["set"] == tag) & (ours[key]["n"] == 10000)]
                columns = ["time_s", "rf_pct", "cpu_threads", "gpu_count", "source_path"]
                runs += [(label, method, *r) for r in mine[columns].itertuples(index=False)]
            for source, data in sources.get(key, sources[""]) if key not in ours else []:
                if key in data["methods"]:
                    m = data["methods"][key]
                    threads = m.get("cpu_cores", data.get("cpu_cores", pd.NA))
                    gpu_count = m.get("n_gpus", 1 if key == "DIPPER-GPU" else 0)
                    runs.append((label, method, m["time"], m["rf_pct"], threads, gpu_count, source))
            if len(runs) == found:
                raise MissingResults(f"baseline_10k: no {method} runs on {tag} at 10,000 leaves")
    frame = pd.DataFrame(
        runs,
        columns=["dataset", "method", "time", "rf_pct", "cpu_threads", "gpu_count", "source_path"],
    )
    rows = frame.groupby(["dataset", "method"], sort=False, as_index=False).agg(
        n_replicates=("time", "size"),
        time_s_mean=("time", "mean"),
        time_s_sd=("time", "std"),
        rf_pct_mean=("rf_pct", "mean"),
        rf_pct_sd=("rf_pct", "std"),
        gpu_count=("gpu_count", one_of("GPU counts")),
        cpu_threads=("cpu_threads", one_of("CPU thread counts")),
        source_paths=("source_path", "|".join),
    )
    keys = {method: key for method, key, _ in cases}
    rows.insert(1, "n", 10000)
    rows.insert(3, "initialisation", rows["method"].map({m: i for m, _, i in cases}))
    rows = rows.assign(
        source_key=rows["method"].map(
            lambda m: "vecme report" if keys[m] in ours else f"methods.{keys[m]}"
        ),
        timing_scope=rows["method"].map(
            lambda m: TIMING if keys[m] in ours else "archived comparator process wall time"
        ),
        interpretation="CPU ME/ML and random VecME are random-start search stress tests; "
        "hardware differs",
        sd_definition="sample SD (ddof=1)",
    )
    table = pd.DataFrame(
        {
            "Data": rows["dataset"],
            "Method": rows["method"],
            "Time (s)": rows["time_s_mean"],
            "Time SD": rows["time_s_sd"],
            "RF (%)": rows["rf_pct_mean"],
            "RF SD": rows["rf_pct_sd"],
            "CPU threads": rows["cpu_threads"],
            "GPUs": rows["gpu_count"],
        }
    )
    decimals = {"Time (s)": 3, "Time SD": 3, "RF (%)": 2, "RF SD": 2}
    return {"tableS8_baseline_10k.md": rounded(table, decimals)}


def table_comparators(results: Path) -> Outputs:
    """Builds Table S7 (comparators): results -> {tableS7_comparators.md}."""
    sizes_max = BACKBONE_SIZE
    ours = our_runs(results, "scaling")
    ours = ours[ours["set"].isin(list(DATASETS)) & (ours["n"] <= sizes_max)]
    dipper = [d for d in dipper_runs(results) if d["tag"] in DATASETS and d["N"] <= sizes_max]
    cpu = [r for r in default_runs(results) if r["tag"] in DATASETS and r["N"] <= sizes_max]
    methods = ["P2V", "DIPPER", "FastTree-ME", "VFT-ME", "FastTree-ML", "VFT-ML"]
    head = [
        "VecME",
        "DIPPER",
        "FastTree ME",
        "VeryFastTree ME",
        "FastTree ML",
        "VeryFastTree ML",
    ]
    gpu = [d["methods"]["DIPPER-GPU"] for d in dipper]
    others = [(d["tag"], "DIPPER", d["N"], m["time"], m["rf_pct"]) for d, m in zip(dipper, gpu)]
    others += [(r["tag"], ALIAS[r["method"]], r["N"], r["time"], r["rf_pct"]) for r in cpu]
    columns = ["set", "method", "n", "time_s", "rf_pct"]
    runs = pd.concat([ours.assign(method="P2V")[columns], pd.DataFrame(others, columns=columns)])
    means = runs.groupby(["set", "method", "n"]).mean()
    tables = []
    for tag, label in DATASETS.items():
        sizes = sorted(set(ours.loc[ours["set"] == tag, "n"]))
        grid = method_grid(means.loc[tag], methods, head, sizes)
        grid.insert(0, "Data", label)
        tables.append(grid)
    return {"tableS7_comparators.md": pd.concat(tables)}


def table_scaling_methods(results: Path) -> Outputs:
    """Collects every method's runs on the simulated sets (Figs 2c,d and S1-S3):
    results -> {scaling_methods.tsv}.

    One row per run: dataset tag, method key, size, time, RF error, host and GPU memory.
    """
    columns = [
        "dataset",
        "method",
        "n",
        "time_s",
        "rf_pct",
        "host_memory_mb",
        "gpu_memory_peak_mib",
    ]
    ours = our_runs(results, "scaling").rename(columns={"set": "dataset", "source_path": "source"})
    ours = ours.assign(method="P2V")[[*columns, "source"]]
    others = []
    for d in optional(dipper_runs, results):
        m = d["methods"]["DIPPER-GPU"]
        if m.get("rf", -1) >= 0 and "rf_pct" in m:
            row = (d["tag"], "DIPPER", d["N"], m["time"], m["rf_pct"], m["peak_ram_mb"])
            others.append((*row, m.get("gpu_vram_mb", pd.NA), "baselines/dipper/runs.jsonl"))
    cpu = optional(default_runs, results)
    for path in sorted((results / "baselines" / "default_yule").glob("*.json")):
        cpu.append(dict(json.loads(path.read_text()), tag="yule10k"))
    for r in cpu:
        row = (r["tag"], ALIAS[r["method"]], r["N"], r["time"], r["rf_pct"], r["peak_ram_mb"])
        others.append((*row, pd.NA, "baselines"))
    others = pd.DataFrame(others, columns=[*columns, "source"])
    return {"scaling_methods.tsv": pd.concat([ours, others])}


def table_silva_support(results: Path) -> Outputs:
    """Builds Table S14 (silva_support) and Fig. 4b's data:
    results -> {tableS14_silva_support.md, silva_support.tsv}."""
    path = one_file(results / "silva", "silva144_dedup_tbe_B*_quantiles.json")
    data = read_json(path)
    rows: list[dict] = []
    for index, b in enumerate(data["bins"]):
        rows.append(
            {
                "split_size_bin": b["bin"],
                "p_min": b["p_min"],
                "p_max": b["p_max"],
                "n_branches": b["n_branches"],
                "median_tbe_pct": b["median_tbe_pct"],
                "q25_tbe_pct": b["q25_tbe_pct"],
                "q75_tbe_pct": b["q75_tbe_pct"],
                "mean_tbe_pct": b["mean_tbe_pct"],
                "frac_ge70_pct": b["frac_ge70_pct"],
                "frac_ge80_pct": b["frac_ge80_pct"],
                "frac_ge90_pct": b["frac_ge90_pct"],
                "n_ge70": b["n_ge70"],
                "n_ge80": b["n_ge80"],
                "n_ge90": b["n_ge90"],
                "mean_pos_branch_length": b["mean_pos_branch_length"],
                "k_unique_leaves": data["K_unique_leaves"],
                "n_expanded_leaves": data["N_expanded_leaves"],
                "b_replicates": data["B_replicates"],
                "source_path": str(path.relative_to(results)),
                "source_key": f"bins[{index}]",
            }
        )
    frame = pd.DataFrame(rows)
    table = pd.DataFrame(
        {
            "Split size p": frame["split_size_bin"],
            "Branches": frame["n_branches"],
            "Median TBE (%)": frame["median_tbe_pct"],
            "Q25 TBE (%)": frame["q25_tbe_pct"],
            "Q75 TBE (%)": frame["q75_tbe_pct"],
            "Mean TBE (%)": frame["mean_tbe_pct"],
            "Share TBE ≥ 70% (%)": frame["frac_ge70_pct"],
            "Share TBE ≥ 80% (%)": frame["frac_ge80_pct"],
        }
    )
    table = rounded(table, dict.fromkeys(table.columns[2:], 2))
    return {"tableS14_silva_support.md": table, "silva_support.tsv": rows}


def table_silva_clades(results: Path) -> Outputs:
    """Collects Fig. 4a's data: results
    -> {silva_clades.tsv (one row per clade), silva_clade_paths.tsv}.

    A clade's path is its best node and its ancestors up to the root,
    each with the branch above it (length, TBE and leaves below).
    """
    bio = read_json(one_file(results / "silva", "silva144_real_biology_B*.json"))
    clades, paths = [], []
    for section, rank in (
        ("domains", "domain"),
        ("phyla", "phylum"),
        ("classes", "class"),
    ):
        for e in bio[section]:
            clades.append(
                {
                    "name": e["name"],
                    "rank": rank,
                    "unique_leaves": e["unique_leaves"],
                    "best_node": e["best_node"],
                    "tbe_pct": e["tbe_pct"],
                    "f1": e["f1"],
                    "purity_pct": e["purity_pct"],
                    "recall_pct": e["recall_pct"],
                }
            )
            nodes = e["ancestor_nodes"]
            for step, node in enumerate(nodes):
                paths.append(
                    {
                        "name": e["name"],
                        "step": step,
                        "node": node,
                        "parent": nodes[step + 1] if step + 1 < len(nodes) else "",
                        "branch_length": e["ancestor_bl"][step],
                        "tbe_pct": e["ancestor_tbe"][step],
                        "size": e["ancestor_sz"][step],
                    }
                )
    return {"silva_clades.tsv": clades, "silva_clade_paths.tsv": paths}


def table_silva_gtdb(results: Path) -> Outputs:
    """Builds Table S15 (silva_gtdb): results -> {tableS15_silva_gtdb.md}."""
    record = read_json(results / "silva" / "gtdb_labelled_scores.json")
    out = record["trees"]
    ranks = [rank.capitalize() for rank in RANKS[1:6]]
    table = pd.DataFrame(
        [
            {"Tree": name}
            | {rank: out[tag][key]["mean_f1"] for rank, key in zip(ranks, RANKS[1:6])}
            | {"Consistent branches (%)": out[tag]["branches_consistent_pct"]}
            for tag, name in GTDB_TREES.items()
            if tag in out
        ]
    )
    decimals = dict.fromkeys(ranks, 2) | {"Consistent branches (%)": 0}
    return {"tableS15_silva_gtdb.md": rounded(table, decimals)}


def table_seqlen(results: Path) -> Outputs:
    """Builds Table S9 (seqlen): results -> {tableS9_seqlen.md}.

    RF columns from our runs (10,000 sites: the scaling runs on JC10k);
    length from the exact rescoring.
    """
    sites = {"jc10k": 10_000, "jc100k": 100_000, "jc1000k": 1_000_000}
    exact = pd.read_json(
        results / "seqlen" / "diagnose_10k_sequences.jsonl", lines=True, precise_float=True
    )
    seqlen, scaling = our_runs(results, "seqlen"), our_runs(results, "scaling")
    runs = pd.concat([seqlen, scaling[scaling["set"] == "jc10k"]])
    rows = []
    for tag, label in sites.items():
        small = runs.loc[(runs["set"] == tag) & (runs["n"] == 10000), "rf_pct"]
        big = seqlen.loc[(seqlen["set"] == tag) & (seqlen["n"] == 100000), "rf_pct"]
        ppm = exact.loc[exact["tag"] == tag, "ours_minus_true_ppm"]
        if small.empty or ppm.empty:
            raise MissingResults(f"seqlen: {tag} at 10,000 sequences")
        rows.append(
            {
                "Sites": label,
                "RF (%), 10,000 sequences": small.mean(),
                "Length minus true (ppm)": ppm.mean(),
                "Shorter than truth": f"{(ppm < 0).sum()}/{len(ppm)}",
                "RF (%), 100,000 sequences": big.mean(),
                "Runs, 100,000 sequences": len(big),
            }
        )
    decimals = {"RF (%), 10,000 sequences": 2, "Length minus true (ppm)": 0}
    decimals["RF (%), 100,000 sequences"] = 2
    return {"tableS9_seqlen.md": rounded(pd.DataFrame(rows), decimals)}


def table_representatives(results: Path) -> Outputs:
    """Builds Table S1 (representative_sensitivity):
    results -> {tableS1_representative_sensitivity.md}.

    Per dataset, size and cap P: RF to the true tree,
    the scored BME length minus that of the P = 64 tree of the same replicate,
    RF to that tree, and the search time (starting tree and rounds, no branch lengths).
    """
    runs = our_runs(results, "representatives")
    # the reference of a replicate: the last of its P = 64 runs in run order (the second seed)
    reference = runs.loc[runs["cap"] == 64, ["set", "n", "rep", "scored_length"]]
    reference = reference.drop_duplicates(["set", "n", "rep"], keep="last")
    runs = runs.merge(reference, on=["set", "n", "rep"], how="left", suffixes=("", "_ref"))
    runs["delta"] = runs["scored_length"] - runs["scored_length_ref"]

    groups = []
    for tag, label in DATASETS.items():
        group = (
            runs[runs["set"] == tag]
            .groupby(["n", "cap"], as_index=False)
            .agg(
                rf=("rf_pct", "mean"),
                rf_sd=("rf_pct", "std"),
                delta=("delta", "mean"),
                delta_sd=("delta", "std"),
                rf_vs=("rf_vs_cap64_pct", "mean"),
                rf_vs_sd=("rf_vs_cap64_pct", "std"),
                time=("topology_time_s", "mean"),
                time_sd=("topology_time_s", "std"),
            )
        )
        groups.append(group.assign(dataset=label))
    table = pd.concat(groups)
    table.loc[table["cap"] == 64, ["delta", "delta_sd", "rf_vs", "rf_vs_sd"]] = (
        pd.NA
    )  # the reference
    names = {"dataset": "Dataset", "n": "N", "cap": "P", "rf": "RF to truth (%)", "rf_sd": "RF SD"}
    names |= {"delta": "ΔL vs P = 64", "delta_sd": "ΔL SD", "rf_vs": "RF vs P = 64 (%)"}
    names |= {"rf_vs_sd": "RF vs P = 64 SD", "time": "Time (s)", "time_sd": "Time SD"}
    table = table.rename(columns=names)[list(names.values())]
    decimals = dict.fromkeys(list(names.values())[3:9], 3) | {"Time (s)": 1, "Time SD": 1}
    return {"tableS1_representative_sensitivity.md": rounded(table, decimals)}


def table_fastme(results: Path) -> Outputs:
    """Builds Table S3 (fastme): results -> {tableS3_fastme.md}."""
    rows = pd.read_json(results / "fastme" / "cases.jsonl", lines=True, precise_float=True).to_dict(
        "records"
    )
    names = {
        "jc10k": "JC",
        "jc1m": "JC subsets",
        "rnasim": "RNASim",
        "rna1m": "RNASim subsets",
    }
    order = list(names)
    ppm = 1e6

    cases = []
    for rank, (arm, start) in enumerate((("sample", "own"), ("rand", "same random"))):
        for r in rows:
            x = r.get(arm)
            if not isinstance(x, dict):
                continue
            cases.append(
                {
                    "Start tree": start,
                    "Dataset": names[r["tag"]],
                    "N": r["n"],
                    "rank": (rank, order.index(r["tag"]), r["n"]),
                    "length": ppm * x["rel_diff"],
                    "shorter": 100 * (x["L"] <= x["L_fastme"] * (1 + 1e-12)),
                    "rf_fastme": x["rf_truth_fastme"],
                    "rf_ours": x["rf_truth"],
                    "spr": ppm * max(0.0, x["spr"]["rel_gain"]),
                    "time_fastme": x["fastme_wall_s"],
                    "time_ours": x["solver"]["wall_s"],
                }
            )
    cases = pd.DataFrame(cases).sort_values("rank", kind="stable")
    table = cases.groupby(["Start tree", "Dataset", "N"], sort=False, as_index=False).agg(
        **{
            "Replicates": ("length", "size"),
            "Length diff. (ppm)": ("length", "mean"),
            "Length SD": ("length", "std"),
            "Ours ≤ FastME (%)": ("shorter", "mean"),
            "RF FastME (%)": ("rf_fastme", "mean"),
            "RF FastME SD": ("rf_fastme", "std"),
            "RF ours (%)": ("rf_ours", "mean"),
            "RF ours SD": ("rf_ours", "std"),
            "SPR gain (ppm)": ("spr", "mean"),
            "SPR gain SD": ("spr", "std"),
            "Time FastME (s)": ("time_fastme", "mean"),
            "Time FastME SD": ("time_fastme", "std"),
            "Time ours (s)": ("time_ours", "mean"),
            "Time ours SD": ("time_ours", "std"),
        }
    )
    decimals = {"Length diff. (ppm)": 1, "Length SD": 1, "Ours ≤ FastME (%)": 0}
    decimals |= dict.fromkeys(table.columns[7:13], 2)
    decimals |= {"Time FastME (s)": 0, "Time FastME SD": 0, "Time ours (s)": 2, "Time ours SD": 2}
    return {"tableS3_fastme.md": rounded(table, decimals)}


def table_gtr_fit(results: Path) -> Outputs:
    """Builds Table S2 (gtr_fit): results -> {tableS2_gtr_fit.md}."""
    rows = pd.read_csv(results / "gtr_fit" / "recovery.tsv", sep="\t")

    fits = rows.groupby(["alpha_true", "sites", "target_p"], as_index=False).agg(
        median=("alpha_hat", "median"),
        low=("alpha_hat", "min"),
        high=("alpha_hat", "max"),
        error=("rate_rms_log_error", "mean"),
    )
    names = {
        "alpha_true": "True α",
        "sites": "Sites",
        "target_p": "Median p",
        "median": "α̂ median",
    }
    names |= {"low": "α̂ min", "high": "α̂ max", "error": "Rate error"}
    table = rounded(fits.rename(columns=names), {"Sites": 0, "Median p": 2, "Rate error": 3})
    for column in ("α̂ median", "α̂ min", "α̂ max"):  # above 100: no rate variation detected
        values = table[column].round(2)
        table[column] = values.astype(object).where(values < 100, ">100")
    return {"tableS2_gtr_fit.md": table}


def table_viridian(results: Path) -> Outputs:
    """Builds Tables S11-S13 (viridian_*) and the data of Figs 3c, S5:
    results -> their .md and .tsv."""
    s = read_json(results / "viridian" / "summary.json")
    # read as text: the record goes on unchanged as the figures' data
    path = results / "viridian" / "resolution_by_month.tsv"
    months = pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)
    out: Outputs = {}

    variants = pd.DataFrame(sorted(s["variants"], key=lambda v: -v["genomes"]))
    table = pd.DataFrame(
        {
            "Variant": variants["name"] + " (" + variants["lineage"] + ")",
            "Genomes": variants["genomes"],
            "Misplaced, ours": variants["ours_misplaced"],
            "Misplaced, MAPLE": variants["maple_misplaced"],
            "Differ": variants["ours_to_maple"],
            "Substitutions, ours": variants["ours_sub"],
            "Substitutions, MAPLE": variants["maple_sub"],
            "Exact (%)": 100 * variants["ours_exact"],
            "Transfer (%)": 100 * variants["ours_tbe"],
            "Repeat (%)": 100 * variants["ours_repeat"],
        }
    )
    decimals = dict.fromkeys(table.columns[1:], 0) | {"Transfer (%)": 2}
    out["tableS12_viridian_variants.md"] = rounded(table, decimals)
    out["viridian_variants.tsv"] = [
        {
            k: v[k]
            for k in (
                "name",
                "lineage",
                "genomes",
                "ours_misplaced",
                "maple_misplaced",
                "ours_to_maple",
            )
        }
        | {k: v[k] for k in ("ours_sub", "maple_sub", "ours_exact", "ours_tbe", "ours_repeat")}
        for v in sorted(s["variants"], key=lambda v: -v["genomes"])
    ]

    ours = {(r["from"], r["to"]): r for r in s["ours"]["by_substitutions"]}
    maple = {(r["from"], r["to"]): r for r in s["maple"]["by_substitutions"]}
    if set(ours) != set(VIRIDIAN_BINS) or set(maple) != set(VIRIDIAN_BINS):
        raise SystemExit(
            "the substitution bins of the SARS-CoV-2 summary are not the ones the table labels"
        )
    lines = [(label, ours[key], maple[key]) for key, label in VIRIDIAN_BINS.items()]
    all_ours = s["ours"]["real_means"] | {"branches": s["ours"]["real"]}
    lines.append(("All", all_ours, s["maple"]["real_means"] | {"branches": s["maple"]["real"]}))
    table = pd.DataFrame(
        [
            {
                "Substitutions": label,
                "Ours branches": o["branches"],
                "Ours shared (%)": o["in_maple_pct"],
                "Ours exact (%)": o["exact_pct"],
                "Ours transfer (%)": o["tbe_pct"],
                "Ours repeat (%)": o["repeat_pct"],
                "MAPLE branches": m["branches"],
                "MAPLE shared (%)": m["in_ours_pct"],
                "MAPLE exact (%)": m["exact_pct"],
                "MAPLE transfer (%)": m["tbe_pct"],
            }
            for label, o, m in lines
        ]
    )
    decimals = dict.fromkeys(table.columns[1:], 0) | {"Ours shared (%)": 1, "MAPLE shared (%)": 1}
    out["tableS11_viridian_support.md"] = rounded(table, decimals)
    out["viridian_substitutions.tsv"] = [
        {
            "substitutions_from": key[0],
            "substitutions_to": "" if key[1] is None else key[1],
            "ours_branches": ours[key]["branches"],
            "ours_in_maple_pct": ours[key]["in_maple_pct"],
            "maple_branches": maple[key]["branches"],
            "maple_in_ours_pct": maple[key]["in_ours_pct"],
        }
        for key in VIRIDIAN_BINS
    ]

    lines = []
    for threshold in VIRIDIAN_THRESHOLDS:
        rows_at = months[months["threshold"] == threshold]
        med = dict(zip(rows_at["month"], rows_at["window_median"].astype(float)))
        if not med or any(month not in med for month, _ in VIRIDIAN_PEAKS):
            raise SystemExit(
                f"resolution_by_month.tsv has no complete rows for the threshold {threshold}"
            )
        line = {"Support threshold (%)": 100 * float(threshold)}
        line["Months with median < 100"] = sum(v < 100 for v in med.values())
        lines.append(
            line | {f"Median count, {label}": med[month] for month, label in VIRIDIAN_PEAKS}
        )
    table = pd.DataFrame(lines)
    out["tableS13_viridian_thresholds.md"] = rounded(table, dict.fromkeys(table.columns, 0))
    out["viridian_resolution.tsv"] = months
    return out


TABLES = {
    "scaling": table_scaling,
    "random_tree": table_random_tree,
    "maple": table_maple,
    "bootstrap": table_bootstrap,
    "baseline_10k": table_baseline_10k,
    "comparators": table_comparators,
    "scaling_methods": table_scaling_methods,
    "silva_support": table_silva_support,
    "silva_clades": table_silva_clades,
    "silva_gtdb": table_silva_gtdb,
    "seqlen": table_seqlen,
    "representative_sensitivity": table_representatives,
    "fastme": table_fastme,
    "gtr_fit": table_gtr_fit,
    "viridian": table_viridian,
}


def main() -> None:
    parser = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=Path(DEFAULT_RESULTS_DIR))
    parser.add_argument("--tables-dir", type=Path, default=Path(DEFAULT_TABLES_DIR))
    parser.add_argument(
        "--tables",
        nargs="+",
        choices=list(TABLES),
        default=list(TABLES),
        metavar="TABLE",
        help=f"tables to write (default: all): {', '.join(TABLES)}",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace tables and data files that exist",
    )
    args = parser.parse_args()

    args.tables_dir.mkdir(parents=True, exist_ok=True)
    skipped = []
    for name in args.tables:
        try:
            outputs = TABLES[name](args.results_dir)
        except (MissingResults, FileNotFoundError) as missing:
            skipped.append(name)
            print(f"make_tables: {name}: skipped, missing {missing}", file=sys.stderr)
            continue
        written = []
        for file, content in outputs.items():
            path = args.tables_dir / file
            if path.exists() and not args.overwrite:
                continue
            if file.endswith(".md"):
                table = pd.DataFrame(content).astype(object).fillna("")
                path.write_text(table.to_markdown(index=False, floatfmt="") + "\n")
            else:
                pd.DataFrame(content).to_csv(path, sep="\t", index=False, lineterminator="\n")
            written.append(file)
        kept = len(outputs) - len(written)
        print(
            f"make_tables: {name}: wrote {', '.join(written) or 'nothing'}"
            + (f" ({kept} kept, see --overwrite)" if kept else "")
        )
    if skipped:
        print(
            f"make_tables: skipped for missing results: {', '.join(skipped)}",
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
