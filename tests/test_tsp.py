"""Tests for the TSP problem suite (TSPLIB EUC_2D)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


DATA_DIR = ROOT / "data" / "tsplib"
INSTANCES = ["eil51", "berlin52", "kroA100", "ch150", "kroA200"]
HAVE_DATA = all((DATA_DIR / f"{n}.tsp").exists() for n in INSTANCES)
needs_data = pytest.mark.skipif(
    not HAVE_DATA,
    reason="TSPLIB data missing; run `python scripts/fetch_tsplib.py`",
)


def test_euc2d_distance_matches_known_pair():
    """Compute distance between two specific cities by hand and verify
    the parser/distance pipeline."""
    if not HAVE_DATA:
        pytest.skip("TSPLIB data missing")
    from benchmarks.tsp import _parse_euc2d_tsp, _euc2d_distance_matrix
    n, coords = _parse_euc2d_tsp(DATA_DIR / "berlin52.tsp")
    assert n == 52
    d = _euc2d_distance_matrix(coords)
    # Symmetric, zero diagonal.
    assert np.array_equal(d, d.T)
    assert np.all(np.diag(d) == 0)
    # Distance from city 0 to city 1, computed directly with TSPLIB's
    # nint(x) = floor(x + 0.5) rounding (not Python's round-half-to-even).
    expected = int(np.floor(float(np.linalg.norm(coords[0] - coords[1])) + 0.5))
    assert d[0, 1] == expected


@needs_data
@pytest.mark.parametrize("name", INSTANCES)
def test_tsp_instance_basics(name):
    from benchmarks.tsp import TSPLIB_FUNCS, _TSPLIB_OPTIMA
    Cls = dict(TSPLIB_FUNCS)[name]
    p = Cls()
    assert p.ndim > 0
    # f_global matches the known-optimum table.
    assert p.f_global == _TSPLIB_OPTIMA[name]
    # A random permutation evaluates to a finite tour length above optimum.
    rng = np.random.default_rng(0)
    perm = list(rng.permutation(p.ndim))
    tour = p.evaluate(perm)
    assert np.isfinite(tour)
    assert tour > p.f_global


@needs_data
def test_tsp_baseline_smoke_run():
    """Tiny SGA run on eil51 must complete without raising and produce
    a tour shorter than the random initial best."""
    from ga.build import baseline_from_config
    from ga.core import run_sga
    from benchmarks.tsp import TSPLIB_FUNCS

    cfg = json.loads((ROOT / "configs" / "tsp.json").read_text())
    Cls = dict(TSPLIB_FUNCS)["eil51"]
    p = Cls()
    baseline = baseline_from_config(cfg, dim=p.ndim)
    baseline.name = "SGA_baseline"
    r = run_sga(p, baseline, pop_size=30, max_fes=2000, seed=0,
                record_history=False)
    assert r["fes_used"] <= 2000
    assert len(r["best_solution"]) == p.ndim
    # Permutation invariant must hold.
    assert sorted(int(c) for c in r["best_solution"]) == list(range(p.ndim))
    # Should beat at least one random tour.
    rng = np.random.default_rng(99)
    rand_tour = p.evaluate(list(rng.permutation(p.ndim)))
    assert r["best_fitness"] < rand_tour


_NEW_PERM_VARIANTS = [
    "SGA+restart", "SGA+abc_scout", "SGA+cuckoo_discovery",
    "SGA+net_suppress", "SGA+immune_memory", "SGA+pher_minmax",
    "SGA+DE_mut", "SGA+SHADE_mut", "SGA+gsa_full_mut",
    "SGA+bfo_chemotaxis", "SGA+krill_herd_mut", "SGA+sos_mut",
    "SGA+bbo_migration_cx",
    "SGA+greedy_replace", "SGA+archive_replace", "SGA+metropolis_replace",
]


@needs_data
@pytest.mark.parametrize("name", _NEW_PERM_VARIANTS)
def test_perm_variant_preserves_permutation(name):
    """Each variant exposed to TSP must keep every individual a valid
    permutation of 0..n-1 across a short run."""
    from ga.build import baseline_from_config, variant_from_baseline
    from ga.core import run_sga
    from ga.variants import variants_for_encoding
    from benchmarks.tsp import TSPLIB_FUNCS

    cfg = json.loads((ROOT / "configs" / "tsp.json").read_text())
    Cls = dict(TSPLIB_FUNCS)["eil51"]
    p = Cls()
    baseline = baseline_from_config(cfg, dim=p.ndim)
    baseline.name = "SGA_baseline"
    perm_specs = {s.name: s for s in variants_for_encoding("permutation")}
    spec = perm_specs[name]
    var = variant_from_baseline(baseline, spec)

    r = run_sga(p, var, pop_size=20, max_fes=2000, seed=1,
                record_history=False)
    assert r["fes_used"] <= 2000
    assert len(r["best_solution"]) == p.ndim
    assert sorted(int(c) for c in r["best_solution"]) == list(range(p.ndim))
    assert np.isfinite(r["best_fitness"])


def test_tsp_problem_missing_data_raises():
    from benchmarks.tsp import TSPProblem
    with pytest.raises(FileNotFoundError, match="fetch_tsplib"):
        TSPProblem(name="does_not_exist", tsplib_dir="/tmp/no_such_dir_xyz")
