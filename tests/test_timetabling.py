"""Tests for the UCTP problem suite (Socha 2002 instances)."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


DATA_DIR = ROOT / "data" / "socha"
INSTANCES = (
    [f"easy{i:02d}" for i in range(1, 6)]
    + [f"medium{i:02d}" for i in range(1, 6)]
    + [f"hard{i:02d}" for i in range(1, 6)]
)
HAVE_DATA = all((DATA_DIR / f"{n}.tim").exists() for n in INSTANCES)
needs_data = pytest.mark.skipif(
    not HAVE_DATA,
    reason="Socha data missing; run `python scripts/fetch_socha.py`",
)


@needs_data
def test_socha_parser_basics():
    from benchmarks.timetabling import _parse_socha_instance
    d = _parse_socha_instance(DATA_DIR / "easy01.tim")
    assert d["n_events"] > 0
    assert d["n_rooms"] > 0
    assert d["n_features"] > 0
    assert d["n_students"] > 0
    assert d["room_sizes"].shape == (d["n_rooms"],)
    assert d["student_event"].shape == (d["n_students"], d["n_events"])
    assert d["room_features"].shape == (d["n_rooms"], d["n_features"])
    assert d["event_features"].shape == (d["n_events"], d["n_features"])


@needs_data
@pytest.mark.parametrize("name", INSTANCES)
def test_uctp_instance_basics(name):
    from benchmarks.timetabling import SOCHA_FUNCS, N_TIMESLOTS
    Cls = dict(SOCHA_FUNCS)[name]
    p = Cls()
    assert p.ndim > 0
    assert p.f_global == 0.0
    assert float(p.lb[0]) == 0.0
    assert float(p.ub[0]) == N_TIMESLOTS - 1
    # A random assignment evaluates to a finite, non-negative fitness.
    rng = np.random.default_rng(0)
    x = rng.integers(0, N_TIMESLOTS, size=p.ndim).tolist()
    f = p.evaluate(x)
    assert np.isfinite(f)
    assert f >= 0.0


@needs_data
def test_uctp_baseline_smoke_run():
    """Tiny SGA run on easy01 completes without raising, respects the FE
    budget, every gene stays in [0, 44], best_fitness is finite."""
    from ga.build import baseline_from_config
    from ga.core import run_sga
    from benchmarks.timetabling import SOCHA_FUNCS, N_TIMESLOTS

    cfg = json.loads((ROOT / "configs" / "timetabling.json").read_text())
    Cls = dict(SOCHA_FUNCS)["easy01"]
    p = Cls()
    baseline = baseline_from_config(cfg, dim=p.ndim)
    baseline.name = "SGA_baseline"
    r = run_sga(p, baseline, pop_size=20, max_fes=2000, seed=0,
                record_history=False)
    assert r["fes_used"] <= 2000
    assert len(r["best_solution"]) == p.ndim
    for g in r["best_solution"]:
        assert 0 <= int(g) <= N_TIMESLOTS - 1
    assert np.isfinite(r["best_fitness"])
    assert r["best_fitness"] >= 0.0
    # Should beat at least one random assignment.
    rng = np.random.default_rng(99)
    rand_x = rng.integers(0, N_TIMESLOTS, size=p.ndim).tolist()
    assert r["best_fitness"] <= p.evaluate(rand_x)


def test_uctp_problem_missing_data_raises():
    from benchmarks.timetabling import UCTPProblem
    with pytest.raises(FileNotFoundError, match="fetch_socha"):
        UCTPProblem(name="does_not_exist", socha_dir="/tmp/no_such_dir_xyz")
