"""
End-to-end tests of the p2v-bme command.
Each test runs the binary and checks its exit code, what it prints and the files it writes (<prefix>.nwk,
<prefix>.json): help and version, errors, and runs on simulated alignments.
The library has its own tests (gtest); these check what only the command does.

ctest runs them in two parts, cli (-m "not gpu") and cli_gpu (-m gpu), with the paths of the binaries in P2VBME_BIN
and P2VBME_MAKE_TEST_FASTA.
By hand, after a build (the binaries default to build/):
    pytest tests/python/test_cli.py -m "not gpu"
"""

import json
import os
import re
import subprocess

from pathlib import Path

import pytest

BUILD = Path(__file__).resolve().parents[2] / "build"
P2V_BME = os.environ.get("P2VBME_BIN", str(BUILD / "p2v-bme"))
MAKE_TEST_FASTA = os.environ.get(
    "P2VBME_MAKE_TEST_FASTA", str(BUILD / "make_test_fasta")
)

# Top-level keys of the report, in order (--report-full adds "details")
SUMMARY_KEYS = [
    "version",
    "command_line",
    "date",
    "substitution_model",
    "bme_length",
    "rounds_run",
    "stop_reason",
    "time",
    "warnings",
]
# Sequence names of input.fa, and of input_dups.fa (three copies of t1 and two of t2 added)
NAMES = sorted(f"t{i}" for i in range(300))
NAMES_DUPS = sorted(
    NAMES + ["t1_copy1", "t1_copy2", "t1_copy3", "t2_copy1", "t2_copy2"]
)


@pytest.fixture(scope="session")
def data(tmp_path_factory):
    """
    The simulated inputs, made once per session.
    input.fa, input.maple, input_dups.fa and input_true.nwk: 300 sequences x 600 sites.
    small.fa: 10 sequences x 300 sites.
    """
    d = tmp_path_factory.mktemp("data")
    subprocess.run([MAKE_TEST_FASTA, d / "input", "300", "600", "7"], check=True)
    subprocess.run([MAKE_TEST_FASTA, d / "small", "10", "300", "3"], check=True)
    return d


def run(*args):
    """Runs p2v-bme with these arguments; returns the finished process (returncode, stdout, stderr)."""
    return subprocess.run(
        [P2V_BME, *map(str, args)], capture_output=True, text=True, check=False
    )


def leaf_names(path):
    """The leaf names of a Newick tree, sorted."""
    return sorted(re.findall(r"[(,]([^(),:;]+)", Path(path).read_text()))


def report(prefix):
    """The run report <prefix>.json."""
    return json.loads(Path(f"{prefix}.json").read_text())


@pytest.mark.parametrize(
    "accelerator", ["cpu", pytest.param("auto", marks=pytest.mark.gpu)]
)
def test_default_run(data, tmp_path, accelerator):
    """Progress on stdout, nothing on stderr, a tree of every sequence and the summary report."""
    p = run(
        "-s", data / "input.fa", "-m", "JC69", "-a", accelerator, "-p", tmp_path / "out"
    )
    assert p.returncode == 0, p.stderr
    # auto: the GPU when one is visible (falling back to the CPU would print a warning)
    assert p.stderr == ""
    lines = p.stdout.splitlines()
    assert lines[0].startswith("p2v-bme: read 300 sequences x 600 sites")
    assert any(line.startswith("[DIAG]") for line in lines)  # progress of the search
    assert lines[-1].startswith("p2v-bme: sample start,") and "BME length" in lines[-1]
    assert leaf_names(tmp_path / "out.nwk") == NAMES
    r = report(tmp_path / "out")
    assert list(r) == SUMMARY_KEYS
    assert r["substitution_model"] == {"name": "JC69"}
    assert r["bme_length"] > 0 and r["rounds_run"] >= 1 and r["warnings"] == []


def test_quiet_prints_nothing(data, tmp_path):
    p = run(
        "-s", data / "input.fa", "-m", "JC69", "-a", "cpu", "-p", tmp_path / "out", "-q"
    )
    assert (p.returncode, p.stdout, p.stderr) == (0, "", "")
    assert leaf_names(tmp_path / "out.nwk") == NAMES
    assert report(tmp_path / "out")["bme_length"] > 0


def test_quiet_keeps_warnings(data, tmp_path):
    """Fewer than 16 sequences: the random start, with a warning on stderr (also with -q) and in the report."""
    p = run(
        "-s", data / "small.fa", "-m", "JC69", "-a", "cpu", "-p", tmp_path / "out", "-q"
    )
    warning = (
        "fewer than 16 sequences, using the random start instead of sample-and-place"
    )
    assert (p.returncode, p.stdout, p.stderr) == (
        0,
        "",
        f"p2v-bme: warning: {warning}\n",
    )
    assert report(tmp_path / "out")["warnings"] == [warning]


def test_options_reach_the_run(data, tmp_path):
    """Options at non-default values are used: the full report lists them among the settings used."""
    p = run(
        *(
            "-s",
            data / "input.fa",
            "-m",
            "TN93",
            "-p",
            tmp_path / "out",
            "--report-full",
        ),
        *(
            "--precision",
            "double",
            "--devices",
            "2",
            "--seed",
            "5",
            "--sample-size",
            "40",
        ),
        *("--max-rounds", "3", "--tol", "0", "--time-limit", "100"),
        *("--gtr-fit-pairs", "1000"),
        *("--profile-cap", "8", "--bl-profile-cap", "8", "--no-neg-as-zero"),
    )
    assert p.returncode == 0, p.stderr
    # double precision without --accelerator: the CPU, with a warning
    assert p.stderr == "p2v-bme: warning: double precision: running on the CPU\n"
    r = report(tmp_path / "out")
    expected = {
        "start_tree": "sample",
        "sample_size": 40,
        "seed": 5,
        "max_rounds": 3,
        "tol": 0,
        "time_limit": 100,
        "profile_cap": 8,
        "branch_lengths": True,
        "bl_profile_cap": 8,
        "neg_as_zero": False,
        "accelerator": "cpu",
        "precision": "double",
        "gtr_fit_pairs": 1000,
    }
    settings = r["details"]["settings"]
    assert {k: settings[k] for k in expected} == expected
    assert r["details"]["devices"] == ["CPU: 2 threads"]
    assert (
        r["substitution_model"]["name"] == "TN93"
        and len(r["substitution_model"]["base_frequencies"]) == 4
    )
    assert 1 <= r["rounds_run"] <= 3


def test_gtr_gc_fit(data, tmp_path):
    """GTR+GC: the rates and alpha are fitted on every pair of the 300 sequences (fewer than 100,000), shown on stdout
    and in the report; on these JC69 data, equal rates and alpha at its upper end (no rate variation).
    """
    p = run(
        "-s", data / "input.fa", "-m", "GTR+GC", "-a", "cpu", "-p", tmp_path / "out"
    )
    assert (p.returncode, p.stderr) == (0, "")
    assert any(
        line.startswith("p2v-bme: GTR+GC fit on 44850 pairs in ")
        for line in p.stdout.splitlines()
    )
    m = report(tmp_path / "out")["substitution_model"]
    assert list(m) == ["name", "rates", "alpha", "fit"]
    assert m["name"] == "GTR+GC"
    assert len(m["rates"]) == 6 and sum(m["rates"]) == pytest.approx(6.0)
    assert all(abs(r - 1.0) < 0.1 for r in m["rates"]) and m["alpha"] > 100
    fit = m["fit"]
    assert (fit["pairs_used"], fit["converged"]) == (44850, True)
    assert fit["iterations"] > 0 and fit["log_likelihood"] < 0
    assert len(fit["base_frequencies"]) == 4


def test_random_start_without_branch_lengths(data, tmp_path):
    p = run(
        *(
            "-s",
            data / "input.fa",
            "-m",
            "JC69",
            "-a",
            "cpu",
            "-p",
            tmp_path / "out",
            "--report-full",
        ),
        *("-t", "random", "--no-branch-lengths"),
    )
    assert p.returncode == 0, p.stderr
    assert p.stdout.splitlines()[-1].startswith("p2v-bme: random start,")
    assert leaf_names(tmp_path / "out.nwk") == NAMES
    r = report(tmp_path / "out")
    assert r["bme_length"] is None
    settings = r["details"]["settings"]
    assert (
        settings["start_tree"],
        settings["sample_size"],
        settings["branch_lengths"],
    ) == ("random", 0, False)


def test_given_tree_with_zero_rounds(data, tmp_path):
    """A Newick start tree and --max-rounds 0: no search; the report scores the given tree."""
    p = run(
        *(
            "-s",
            data / "input.fa",
            "-m",
            "JC69",
            "-a",
            "cpu",
            "-p",
            tmp_path / "out",
            "--report-full",
        ),
        *("-t", data / "input_true.nwk", "--max-rounds", "0"),
    )
    assert p.returncode == 0, p.stderr
    assert leaf_names(tmp_path / "out.nwk") == NAMES
    r = report(tmp_path / "out")
    assert (r["rounds_run"], r["stop_reason"], r["details"]["round_gains"]) == (
        0,
        "none",
        [],
    )
    assert r["bme_length"] > 0


@pytest.mark.parametrize("dedup", [True, False])
def test_duplicates_are_put_back(data, tmp_path, dedup):
    """
    Identical sequences: with dedup, two of each group are searched and the other copies are set aside, then put
    back in the tree.
    Either way, every name is in the tree.
    """
    no_dedup = [] if dedup else ["--no-dedup"]
    p = run(
        "-s",
        data / "input_dups.fa",
        "-m",
        "JC69",
        "-a",
        "cpu",
        "-p",
        tmp_path / "out",
        "--report-full",
        *no_dedup,
    )
    assert p.returncode == 0, p.stderr
    # t1's group of 4 keeps 2, t2's group of 3 keeps 2
    assert ("p2v-bme: 3 identical sequences set aside" in p.stdout) == dedup
    assert leaf_names(tmp_path / "out.nwk") == NAMES_DUPS
    assert report(tmp_path / "out")["details"]["input"]["identical_set_aside"] == (
        3 if dedup else 0
    )


def test_maple_gives_the_fasta_tree(data, tmp_path):
    """The same alignment as .maple and as FASTA gives the same tree."""
    for ext in ("fa", "maple"):
        p = run(
            "-s",
            data / f"input.{ext}",
            "-m",
            "JC69",
            "-a",
            "cpu",
            "-p",
            tmp_path / ext,
            "-q",
        )
        assert p.returncode == 0, p.stderr
    assert (tmp_path / "fa.nwk").read_text() == (tmp_path / "maple.nwk").read_text()


def test_help():
    p = run("--help")
    assert (p.returncode, p.stderr) == (0, "")
    assert "Usage:\n  p2v-bme -s ALIGNMENT -m MODEL [options]\n" in p.stdout
    assert p.stdout.endswith("All options: --help-all\n")


def test_help_all():
    p = run("--help-all")
    assert (p.returncode, p.stderr) == (0, "")
    headings = [
        line.strip() for line in p.stdout.splitlines() if line.endswith("options:")
    ]
    assert headings == [
        "Substitution model options:",
        "Search options:",
        "Start tree options:",
        "Branch lengths and output options:",
        "Input options:",
    ]
    assert "--max-rounds" in p.stdout


@pytest.mark.parametrize("flag", ["--version", "-v"])
def test_version(flag):
    """The version, then the git revision of a development build in brackets."""
    p = run(flag)
    assert (p.returncode, p.stderr) == (0, "")
    assert re.fullmatch(r"p2v-bme 0\.1\.0( \(.+\))?\n", p.stdout), p.stdout


# The errors of the command: its arguments ({data}, {tmp}: the input and output folders), the exit code and the line
# on stderr.
# A missing required option is a usage error (exit code 2); every other error exits with 1.
INPUT = ["-s", "{data}/input.fa", "-p", "{tmp}/out"]
ERRORS = {
    "missing -m": (
        [*INPUT],
        2,
        "p2v-bme: -s ALIGNMENT and -m MODEL are required (see --help)",
    ),
    "extra argument": (
        [*INPUT, "-m", "JC69", "extra"],
        1,
        "p2v-bme: error: unexpected argument 'extra' (see --help)",
    ),
    "unknown model": (
        [*INPUT, "-m", "XYZ"],
        1,
        "p2v-bme: error: unknown value for model: 'XYZ' (JC69, K80, F81, F84, TN93, LogDet or GTR+GC)",
    ),
    "GTR": (
        [*INPUT, "-m", "GTR"],
        1,
        "p2v-bme: error: model 'GTR' (without gamma) is not available; use GTR+GC",
    ),
    "GTR+G": (
        [*INPUT, "-m", "GTR+G"],
        1,
        "p2v-bme: error: model 'GTR+G': continuous gamma is written +GC (e.g. GTR+GC); +G is not used",
    ),
    "unknown accelerator": (
        [*INPUT, "-m", "JC69", "-a", "tpu"],
        1,
        "p2v-bme: error: unknown value for accelerator: 'tpu' (cpu, gpu or auto)",
    ),
    "unknown precision": (
        [*INPUT, "-m", "JC69", "--precision", "quad"],
        1,
        "p2v-bme: error: unknown value for precision: 'quad' (single or double)",
    ),
    # the two GPU errors come before any GPU is used, so they need none
    "GPU listed twice": (
        [*INPUT, "-m", "JC69", "-a", "gpu", "-d", "0,0"],
        1,
        "p2v-bme: error: GPU 0 is listed twice",
    ),
    "GPU in double precision": (
        [*INPUT, "-m", "JC69", "-a", "gpu", "--precision", "double"],
        1,
        "p2v-bme: error: the GPU computes in single precision; double precision runs on the CPU (--accelerator cpu)",
    ),
    # checked by the GTR+GC fit and by infer(), after the read: -q and the CPU keep stdout and stderr to the error
    "no GTR+GC fit pairs": (
        [*INPUT, "-m", "GTR+GC", "-a", "cpu", "-q", "--gtr-fit-pairs", "0"],
        1,
        "p2v-bme: error: GTR+GC fit: the number of pairs must be >= 1",
    ),
    "negative time limit": (
        [*INPUT, "-m", "JC69", "-a", "cpu", "-q", "--time-limit", "-5"],
        1,
        "p2v-bme: error: time limit must be >= 0 (0 = none)",
    ),
    "missing input": (
        ["-s", "{tmp}/missing.fa", "-p", "{tmp}/out", "-m", "JC69", "-a", "cpu"],
        1,
        "p2v-bme: error: cannot open {tmp}/missing.fa",
    ),
    "unwritable output": (
        [
            "-s",
            "{data}/input.fa",
            "-p",
            "/nonexistent_dir/x",
            "-m",
            "JC69",
            "-a",
            "cpu",
        ],
        1,
        "p2v-bme: error: cannot write /nonexistent_dir/x.nwk",
    ),
}


@pytest.mark.parametrize("args, returncode, message", ERRORS.values(), ids=list(ERRORS))
def test_errors(data, tmp_path, args, returncode, message):
    """An error: nothing on stdout, one line on stderr and a non-zero exit code."""
    p = run(*(a.format(data=data, tmp=tmp_path) for a in args))
    assert (p.returncode, p.stdout, p.stderr) == (
        returncode,
        "",
        message.format(data=data, tmp=tmp_path) + "\n",
    )
