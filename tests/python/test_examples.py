"""
End-to-end runs on the example alignments (examples/README.md): p2v-bme on the CPU recovers each true tree and, on
gtr_1k, the GTR+GC parameters, within loose bounds.

CI runs them with the other tests in tests/python (pixi run test-python -m "not gpu"); ctest does not.
By hand, after a build (the binary defaults to build/p2v-bme):
    pytest tests/python/test_examples.py
"""

import json
import math
import os
import subprocess

from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "examples"
P2V_BME = os.environ.get("P2VBME_BIN", str(ROOT / "build" / "p2v-bme"))

# Loose bounds; the README's runs gave 100 and 99.9, alpha 0.416 and a rate error of 0.084 (--seed 1 to 5: the same
# mean_exact, alpha 0.414-0.416, rate error 0.084-0.086)
MIN_MEAN_EXACT = 99.0  # % of the true tree's branches found (100 - normalised RF)
MAX_ALPHA_RELATIVE_ERROR = 0.25
MAX_RATE_RMS_LOG_ERROR = 0.15
# gtr_1k's true parameters (examples/README.md)
TRUE_ALPHA = 0.5
TRUE_RATES = [1.0, 4.0, 1.0, 1.0, 4.0, 1.0]  # AC, AG, AT, CG, CT, GT


def run(*args):
    """Runs p2v-bme on the CPU with these arguments; raises if it fails."""
    subprocess.run([P2V_BME, *map(str, args), "-a", "cpu", "-q"], check=True)


def report(prefix):
    """The run report <prefix>.json."""
    return json.loads(Path(f"{prefix}.json").read_text())


@pytest.mark.parametrize("name, model", [("jc69_1k", "JC69"), ("gtr_1k", "GTR+GC")])
def test_example(tmp_path, name, model):
    """
    The inferred tree has at least MIN_MEAN_EXACT % of the true tree's branches; on gtr_1k, the fitted alpha and
    rates are near the true ones.
    """
    run("-s", EXAMPLES / f"{name}.fa", "-m", model, "-p", tmp_path / name)
    run(
        *("-t", EXAMPLES / f"{name}_true.nwk"),
        *("--supports", tmp_path / f"{name}.nwk"),
        *("-p", tmp_path / "vs_true"),
    )
    assert report(tmp_path / "vs_true")["supports"]["mean_exact"] >= MIN_MEAN_EXACT
    if model == "GTR+GC":
        fit = report(tmp_path / name)["substitution_model"]
        assert abs(fit["alpha"] / TRUE_ALPHA - 1) <= MAX_ALPHA_RELATIVE_ERROR
        # p2v-bme reports the rates scaled to mean 1
        mean_rate = sum(TRUE_RATES) / len(TRUE_RATES)
        log_errors = [
            math.log(r * mean_rate / t) for r, t in zip(fit["rates"], TRUE_RATES)
        ]
        assert (
            math.sqrt(sum(e * e for e in log_errors) / len(log_errors))
            <= MAX_RATE_RMS_LOG_ERROR
        )
