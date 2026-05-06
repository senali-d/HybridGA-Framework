"""
Smoke tests: every variant in build_variants() must run a tiny budget on
F1 without raising, return a non-negative error, and produce a final
population that's in-bounds and the right size.

These tests are cheap (a few seconds total) and catch shape/contract
regressions in any operator before the multi-day full protocol runs.
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ga.core import run_sga  # noqa: E402
from opfunu.cec_based.cec2022 import F12022, F62022  # noqa: E402
from run_experiments import build_variants  # noqa: E402


VARIANTS = build_variants()
DIM = 10  # CEC 2022 supports {2, 10, 20}
POP = 12
MAX_FES = 200


@pytest.mark.parametrize("variant", VARIANTS, ids=[v.name for v in VARIANTS])
def test_variant_runs_clean(variant):
    problem = F12022(ndim=DIM)
    result = run_sga(problem, variant, pop_size=POP, max_fes=MAX_FES, seed=0)

    assert result["error"] >= 0.0, "error must be non-negative"
    assert result["fes_used"] <= MAX_FES, "FE budget must not be exceeded"
    assert len(result["best_solution"]) == DIM
    sol = np.asarray(result["best_solution"], dtype=float)
    assert np.all(sol >= np.asarray(problem.lb) - 1e-9)
    assert np.all(sol <= np.asarray(problem.ub) + 1e-9)


def test_seed_reproducibility():
    """Two runs with the same seed must give identical errors."""
    variant = next(v for v in VARIANTS if v.name == "SGA+chaotic_perturb_mut")
    r1 = run_sga(F12022(ndim=DIM), variant, pop_size=POP, max_fes=MAX_FES, seed=7)
    r2 = run_sga(F12022(ndim=DIM), variant, pop_size=POP, max_fes=MAX_FES, seed=7)
    assert r1["error"] == r2["error"]


def test_islands_with_diversity():
    """Regression check: islands path must apply diversity_fn (was skipped)."""
    from ga.core import FeatureConfig
    from ga import features as feat

    cfg = FeatureConfig(
        name="islands+immigrant",
        n_islands=3,
        migration_interval=4,
        diversity_fn=feat.make_random_immigrant(fraction=0.2),
    )
    r = run_sga(F62022(ndim=DIM), cfg, pop_size=15, max_fes=300, seed=0)
    assert r["error"] >= 0.0
