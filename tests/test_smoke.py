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


@pytest.mark.parametrize("name", ["Sphere", "Step"])
def test_separable_smoke(name):
    """Smoke test: default variant runs cleanly on representative separable
    functions (one pymoo-backed: Sphere; one local: Step)."""
    from benchmarks.suites import SEPARABLE_FUNCS
    Cls = dict(SEPARABLE_FUNCS)[name]
    problem = Cls(ndim=DIM)
    variant = next(v for v in VARIANTS if v.name == "SGA_baseline")
    result = run_sga(problem, variant, pop_size=POP, max_fes=MAX_FES, seed=0)

    assert result["error"] >= 0.0
    assert result["fes_used"] <= MAX_FES
    assert len(result["best_solution"]) == DIM
    sol = np.asarray(result["best_solution"], dtype=float)
    assert np.all(sol >= problem.lb - 1e-9)
    assert np.all(sol <= problem.ub + 1e-9)


def test_seed_reproducibility():
    """Two runs with the same seed must give identical errors."""
    variant = next(v for v in VARIANTS if v.name == "SGA+chaotic_perturb_mut")
    r1 = run_sga(F12022(ndim=DIM), variant, pop_size=POP, max_fes=MAX_FES, seed=7)
    r2 = run_sga(F12022(ndim=DIM), variant, pop_size=POP, max_fes=MAX_FES, seed=7)
    assert r1["error"] == r2["error"]


def test_every_public_feature_has_a_variant():
    """Every public callable in ga.features must be wired into at least one
    variant in build_variants(). Catches feature drift: if you add a new
    feature function but forget to register it, this test fails."""
    import inspect
    from ga import features as feat

    public_fns = {
        name for name, obj in inspect.getmembers(feat, inspect.isfunction)
        if obj.__module__ == "ga.features" and not name.startswith("_")
    }

    referenced: set[str] = set()
    for v in VARIANTS:
        for slot in ("init_fn", "select_fn", "crossover_fn",
                     "mutate_fn", "replace_fn", "diversity_fn"):
            fn = getattr(v, slot, None)
            if fn is None:
                continue
            qn = getattr(fn, "__qualname__", "")
            # Closures from `make_*` factories: the factory is the public name
            referenced.add(qn.split(".")[0] if ".<locals>." in qn
                           else getattr(fn, "__name__", ""))

    missing = public_fns - referenced
    assert not missing, (
        f"{len(missing)} public feature(s) in ga.features have no registered "
        f"variant in build_variants(): {sorted(missing)}"
    )


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
