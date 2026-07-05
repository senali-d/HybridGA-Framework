"""Tests for the UCTP problem suite (Socha 2002 + TTC 2002 instances)
and the Yang & Jat 2011 SSGA loop."""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


SOCHA_DIR = ROOT / "data" / "socha"
TTC_DIR = ROOT / "data" / "ttc2002"
INSTANCES = (
    [f"easy{i:02d}" for i in range(1, 6)]
    + [f"medium{i:02d}" for i in range(1, 6)]
    + [f"hard{i:02d}" for i in range(1, 6)]
)
HAVE_SOCHA = all((SOCHA_DIR / f"{n}.tim").exists() for n in INSTANCES)
HAVE_TTC = (TTC_DIR / "comp01.tim").exists()
needs_socha = pytest.mark.skipif(
    not HAVE_SOCHA,
    reason="Socha data missing; run `python scripts/fetch_socha.py`",
)
needs_ttc = pytest.mark.skipif(
    not HAVE_TTC,
    reason="TTC 2002 data missing; run `python scripts/fetch_ttc2002.py`",
)


@needs_socha
def test_socha_parser_basics():
    from benchmarks.uctp import _parse_socha_instance
    d = _parse_socha_instance(SOCHA_DIR / "easy01.tim")
    assert d["n_events"] > 0
    assert d["n_rooms"] > 0
    assert d["n_features"] > 0
    assert d["n_students"] > 0
    assert d["room_sizes"].shape == (d["n_rooms"],)
    assert d["student_event"].shape == (d["n_students"], d["n_events"])
    assert d["room_features"].shape == (d["n_rooms"], d["n_features"])
    assert d["event_features"].shape == (d["n_events"], d["n_features"])


@needs_socha
@pytest.mark.parametrize("name", INSTANCES)
def test_uctp_instance_basics(name):
    from benchmarks.uctp import SOCHA_FUNCS, N_TIMESLOTS
    Cls = dict(SOCHA_FUNCS)[name]
    p = Cls()
    assert p.ndim > 0
    assert p.f_global == 0.0
    assert float(p.lb[0]) == 0.0
    assert float(p.ub[0]) == N_TIMESLOTS - 1
    rng = np.random.default_rng(0)
    x = rng.integers(0, N_TIMESLOTS, size=p.ndim).tolist()
    f = p.evaluate(x)
    assert np.isfinite(f)
    assert f >= 0.0


@needs_socha
def test_assign_rooms_returns_valid_assignment():
    """assign_rooms produces room indices that are feasible (size + features)
    for every placed event and -1 for unplaced events. No room is double-
    booked within a slot."""
    from benchmarks.uctp import SOCHA_FUNCS, N_TIMESLOTS
    Cls = dict(SOCHA_FUNCS)["easy01"]
    p = Cls()
    rng = np.random.default_rng(7)
    ts = rng.integers(0, N_TIMESLOTS, size=p.ndim)
    rooms = p.assign_rooms(ts)
    assert rooms.shape == (p.ndim,)
    used_per_slot = {}
    for e, r in enumerate(rooms):
        r = int(r)
        if r == -1:
            continue
        # feasibility (size + features)
        assert p.room_ok[e, r]
        # uniqueness within slot
        slot = int(ts[e])
        used_per_slot.setdefault(slot, set())
        assert r not in used_per_slot[slot], (
            f"room {r} double-booked in slot {slot}"
        )
        used_per_slot[slot].add(r)


@needs_socha
def test_score_consistency_with_evaluate():
    """evaluate(x) == score(x, assign_rooms(x)) in fitness."""
    from benchmarks.uctp import SOCHA_FUNCS, N_TIMESLOTS
    Cls = dict(SOCHA_FUNCS)["easy01"]
    p = Cls()
    rng = np.random.default_rng(11)
    for _ in range(5):
        ts = rng.integers(0, N_TIMESLOTS, size=p.ndim).tolist()
        f_eval = p.evaluate(ts)
        f_full, hcv, scv, rooms = p.evaluate_full(ts)
        assert f_eval == f_full
        assert f_full == pytest.approx(p._hcv_weight * hcv + scv)


@needs_socha
def test_uctp_baseline_smoke_run():
    """Tiny SSGA run on easy01 completes, respects FE budget, every gene
    stays in [0, 44], best_fitness is finite, beats a random assignment."""
    from ga.build import baseline_from_config
    from ga.core import run_trial
    from benchmarks.uctp import SOCHA_FUNCS, N_TIMESLOTS

    cfg = json.loads((ROOT / "configs" / "uctp.json").read_text())
    # Trim s_max so the smoke test stays under a few seconds.
    cfg = dict(cfg)
    cfg["refine"] = {"kind": "ls1", "s_max": 3}
    Cls = dict(SOCHA_FUNCS)["easy01"]
    p = Cls()
    baseline = baseline_from_config(cfg, dim=p.ndim)
    baseline.name = "SSGA"
    r = run_trial(p, baseline, pop_size=10, max_fes=80, seed=0,
                  record_history=False)
    assert r["fes_used"] <= 80
    assert len(r["best_solution"]) == p.ndim
    for g in r["best_solution"]:
        assert 0 <= int(g) <= N_TIMESLOTS - 1
    assert np.isfinite(r["best_fitness"])
    assert r["best_fitness"] >= 0.0
    # Should beat at least one random assignment.
    rng = np.random.default_rng(99)
    rand_x = rng.integers(0, N_TIMESLOTS, size=p.ndim).tolist()
    assert r["best_fitness"] <= p.evaluate(rand_x)


@needs_ttc
def test_ttc2002_parser_loads_comp01():
    from benchmarks.uctp import TTC2002_FUNCS
    Cls = dict(TTC2002_FUNCS)["comp01"]
    p = Cls()
    assert p.ndim > 0
    assert p.n_rooms > 0
    assert p.n_features > 0
    assert p.n_students > 0


def test_uctp_problem_missing_data_raises():
    from benchmarks.uctp import UCTPProblem
    with pytest.raises(FileNotFoundError, match="fetch_"):
        UCTPProblem(name="does_not_exist", data_dir="/tmp/no_such_dir_xyz")


UCTP_NATIVE_VARIANTS = [
    "SGA+DE_mut", "SGA+SHADE_mut", "SGA+restart", "SGA+pher_minmax",
    "SGA+immune_memory", "SGA+net_suppress", "SGA+cuckoo_discovery",
    "SGA+abc_scout", "SGA+bfo_chemotaxis", "SGA+krill_herd_mut",
    "SGA+sos_mut", "SGA+bbo_migration_cx", "SGA+gsa_full_mut",
]


def test_uctp_registry_has_all_13_native_variants():
    """Hard invariant: all 13 UCTP-native names appear in the DISCRETE-
    matching set, each exactly once. ANY-tagged variants (greedy_replace,
    archive_replace, metropolis_replace) also match `discrete` and are
    allowed alongside the native 13."""
    from ga.variants import VARIANTS, variants_for_encoding
    names = [s.name for s in variants_for_encoding("discrete")]
    for nm in UCTP_NATIVE_VARIANTS:
        assert nm in names, f"missing DISCRETE variant: {nm}"
    native = [s for s in VARIANTS
              if s.encodings == ("discrete",) and s.name in UCTP_NATIVE_VARIANTS]
    seen = [s.name for s in native]
    assert sorted(seen) == sorted(UCTP_NATIVE_VARIANTS), (
        f"DISCRETE-only registry mismatch: {set(seen) ^ set(UCTP_NATIVE_VARIANTS)}"
    )


@needs_socha
@pytest.mark.parametrize("variant_name", [
    # All variants that override the replace slot (DISCRETE-native + ANY-tagged).
    "SGA+abc_scout", "SGA+greedy_replace", "SGA+archive_replace",
    "SGA+metropolis_replace",
])
def test_uctp_replace_variants_preserve_pop_size(variant_name):
    """SSGA contract: replace_fn must keep len(population) constant. A bug
    in generational_elitism previously collapsed population to 2 for
    archive_replace / abc_scout on UCTP (replace called with offspring=[child])."""
    from ga.build import baseline_from_config, variant_from_baseline
    from ga.core import run_trial
    from ga.variants import VARIANTS
    from benchmarks.uctp import SOCHA_FUNCS

    cfg = dict(json.loads((ROOT / "configs" / "uctp.json").read_text()))
    cfg["refine"] = {"kind": "ls1", "s_max": 1}
    Cls = dict(SOCHA_FUNCS)["easy01"]
    p = Cls()
    baseline = baseline_from_config(cfg, dim=p.ndim)
    baseline.name = "SSGA"
    specs = {s.name: s for s in VARIANTS}
    assert variant_name in specs, f"{variant_name} missing from registry"
    v = variant_from_baseline(baseline, specs[variant_name])

    pop_sizes = []
    real_replace = v.replace_fn
    def _spy(parents, offspring, ctx, _f=real_replace):
        out = _f(parents, offspring, ctx)
        pop_sizes.append(len(out))
        return out
    v.replace_fn = _spy
    pop_size = 12
    run_trial(p, v, pop_size=pop_size, max_fes=60, seed=0, record_history=False)
    assert pop_sizes, "replace_fn was never called"
    assert all(s == pop_size for s in pop_sizes), (
        f"{variant_name} broke pop size: observed {pop_sizes[:6]}, expected all {pop_size}"
    )


@needs_socha
@pytest.mark.parametrize("variant_name", UCTP_NATIVE_VARIANTS)
def test_uctp_variant_smoke(variant_name):
    """Every UCTP-native variant runs on easy01 for a few generations and
    produces a valid timeslot vector. Catches NameErrors, registry mistags,
    encoding/return-type bugs in the adapters."""
    from ga.build import baseline_from_config, variant_from_baseline
    from ga.core import run_trial
    from ga.variants import variants_for_encoding
    from benchmarks.uctp import SOCHA_FUNCS, N_TIMESLOTS

    cfg = dict(json.loads((ROOT / "configs" / "uctp.json").read_text()))
    cfg["refine"] = {"kind": "ls1", "s_max": 2}

    Cls = dict(SOCHA_FUNCS)["easy01"]
    p = Cls()
    baseline = baseline_from_config(cfg, dim=p.ndim)
    baseline.name = "SSGA"

    specs = {s.name: s for s in variants_for_encoding("discrete")}
    assert variant_name in specs, (
        f"{variant_name} missing from DISCRETE variants_for_encoding"
    )
    v = variant_from_baseline(baseline, specs[variant_name])
    r = run_trial(p, v, pop_size=8, max_fes=120, seed=0, record_history=False)

    assert r["fes_used"] <= 120
    assert len(r["best_solution"]) == p.ndim
    for g in r["best_solution"]:
        assert 0 <= int(g) <= N_TIMESLOTS - 1
    assert np.isfinite(r["best_fitness"])
    assert r["best_fitness"] >= 0.0
