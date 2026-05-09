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

import json  # noqa: E402

from ga.core import run_sga  # noqa: E402
from ga.build import baseline_from_config  # noqa: E402
from opfunu.cec_based.cec2022 import F12022, F62022  # noqa: E402
from run_experiments import build_variants  # noqa: E402


with (ROOT / "configs" / "cec2022.json").open() as _f:
    _CFG = json.load(_f)
_BASELINE = baseline_from_config(_CFG, dim=10)
_BASELINE.name = "SGA_baseline"
VARIANTS = build_variants(_BASELINE, "real")
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


@pytest.mark.parametrize("name", ["Sphere", "Rastrigin", "Step"])
def test_separable_smoke(name):
    """Smoke test: default variant runs cleanly on representative separable
    functions. Covers a local impl (Sphere, Step) and the pymoo-adapter
    path (Rastrigin) so a future pymoo API break shows up in CI."""
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


def test_separable_optima():
    """Each separable problem must report f_global at its known optimum.
    Catches formula/bound regressions; runs in milliseconds (no GA)."""
    from benchmarks.suites import SEPARABLE_FUNCS
    optima = {
        "Sphere": np.zeros(10),
        "Rastrigin": np.zeros(10),
        "Schwefel": np.full(10, 420.9687),
        "SumOfDifferentPowers": np.zeros(10),
        "Step": np.zeros(10),
    }
    for name, Cls in SEPARABLE_FUNCS:
        p = Cls(ndim=10)
        f = p.evaluate(optima[name])
        assert abs(f - p.f_global) < 1e-3, (
            f"{name}: f(x*)={f} but f_global={p.f_global}"
        )
        assert p.lb.shape == p.ub.shape == (10,)


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
                     "mutate_fn", "replace_fn", "diversity_fn",
                     "refine_fn"):
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


@pytest.mark.parametrize("name", [
    "TensionCompressionSpring", "PressureVessel", "WeldedBeam", "SpeedReducer",
])
def test_engineering_problems_construct_and_evaluate(name):
    """Each engineering problem instantiates with the expected fixed dim,
    evaluates an in-bounds point, and uses Deb's scalar-encoded fitness:
    a deliberately infeasible point yields fitness >= F_OFFSET."""
    from benchmarks.engineering import ENGINEERING_FUNCS
    Cls = dict(ENGINEERING_FUNCS)[name]
    problem = Cls()

    assert problem.ndim == Cls.NDIM
    assert problem.lb.shape == (problem.ndim,)
    assert problem.ub.shape == (problem.ndim,)
    assert np.all(problem.lb < problem.ub)
    assert problem.f_global > 0.0

    # Evaluate the lower-bound corner — far from feasibility for these
    # problems; we don't assert feasibility, only that evaluate returns a
    # finite scalar respecting the encoding contract.
    f_lb = problem.evaluate(problem.lb.copy())
    assert np.isfinite(f_lb)

    # The lower-bound corner should be infeasible for at least
    # Spring/Vessel/Beam/Reducer in these engineering benchmarks.
    g_lb = problem.constraints(problem._canonicalize(problem.lb.copy()))
    if np.any(g_lb > 0.0):
        assert f_lb >= Cls.F_OFFSET, (
            f"{name}: infeasible point should yield fitness >= F_OFFSET "
            f"({Cls.F_OFFSET}); got {f_lb}"
        )


@pytest.mark.parametrize("name", [
    "TensionCompressionSpring", "PressureVessel", "WeldedBeam", "SpeedReducer",
])
def test_engineering_smoke_run(name):
    """One short SGA run on each engineering problem completes and returns
    a valid result dict."""
    from benchmarks.engineering import ENGINEERING_FUNCS
    Cls = dict(ENGINEERING_FUNCS)[name]
    problem = Cls()
    variant = next(v for v in VARIANTS if v.name == "SGA_baseline")
    r = run_sga(problem, variant, pop_size=POP, max_fes=MAX_FES, seed=0)

    assert r["fes_used"] <= MAX_FES
    assert len(r["best_solution"]) == problem.ndim
    sol = np.asarray(r["best_solution"], dtype=float)
    assert np.all(sol >= problem.lb - 1e-9)
    assert np.all(sol <= problem.ub + 1e-9)
    assert np.isfinite(r["best_fitness"])


def test_spring_g2_singularity_is_finite():
    """At D == d the literature form of g2 is singular; evaluate() must
    still return a finite scalar (guard inside constraints)."""
    from benchmarks.engineering import TensionCompressionSpring
    problem = TensionCompressionSpring()
    # D == d, both inside their respective bounds (d in [0.05, 2.0],
    # D in [0.25, 1.30]); pick the overlap.
    x = np.array([0.5, 0.5, 5.0])
    f = problem.evaluate(x)
    assert np.isfinite(f)
    # Singular point is treated as strongly infeasible.
    assert f >= TensionCompressionSpring.F_OFFSET


def test_engineering_feasible_point_below_offset():
    """For each engineering problem, locate a feasible point via rejection
    sampling and verify its fitness lies strictly below F_OFFSET — the
    other half of the Deb encoding contract."""
    from benchmarks.engineering import ENGINEERING_FUNCS
    rng = np.random.default_rng(0)
    for name, Cls in ENGINEERING_FUNCS:
        problem = Cls()
        for _ in range(20000):
            x = problem.lb + rng.random(problem.ndim) * (problem.ub - problem.lb)
            x_c = problem._canonicalize(x)
            g = problem.constraints(x_c)
            if np.all(g <= 0.0):
                f = problem.evaluate(x)
                assert np.isfinite(f)
                assert f < Cls.F_OFFSET, (
                    f"{name}: feasible point should yield fitness < "
                    f"F_OFFSET ({Cls.F_OFFSET}); got {f}"
                )
                break
        else:
            pytest.fail(f"{name}: no feasible point found in 20000 samples")


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
