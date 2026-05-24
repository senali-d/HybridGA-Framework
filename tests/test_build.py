"""Tests for the config-driven baseline + variant inheritance contract."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ga.build import (
    Spec, baseline_from_config, variant_from_baseline, variant_diff,
    _OVERRIDABLE_SCALARS,
)


def _load(suite):
    with (ROOT / "configs" / f"{suite}.json").open() as f:
        return json.load(f)


def test_engineering_baseline_matches_config():
    cfg = _load("engineering")
    b = baseline_from_config(cfg, dim=4)
    assert b.cx_prob == 0.9
    assert b.elitism_k == 1
    # Selection closure name encodes k=2 from the engineering JSON.
    assert "k2" in b.select_fn.__name__
    # Mutation closure carries eta=20 and indpb=1/D=0.25 for D=4.
    assert "poly_eta20" in b.mutate_fn.__name__
    assert "0.25" in b.mutate_fn.__name__


def test_cec2022_baseline_uses_k3_tournament():
    cfg = _load("cec2022")
    b = baseline_from_config(cfg, dim=10)
    assert "k3" in b.select_fn.__name__
    assert b.elitism_k == 1


def test_variant_inherits_everything_unless_declared():
    cfg = _load("engineering")
    b = baseline_from_config(cfg, dim=4)
    # Variant declares only mutation. Everything else must be identical.
    spec = Spec(name="SGA+test_mut", mutate=lambda ind, ctx: list(ind))
    v = variant_from_baseline(b, spec)
    assert v.name == "SGA+test_mut"
    assert v.select_fn   is b.select_fn      # inherited
    assert v.crossover_fn is b.crossover_fn  # inherited
    assert v.replace_fn  is b.replace_fn     # inherited
    assert v.mutate_fn   is not b.mutate_fn  # changed
    assert v.cx_prob == b.cx_prob
    assert v.elitism_k == b.elitism_k


def test_variant_overrides_only_change_declared_keys():
    cfg = _load("engineering")
    b = baseline_from_config(cfg, dim=4)
    spec = Spec(name="SGA+test_pop", overrides={"pop_size": 30})
    v = variant_from_baseline(b, spec)
    assert getattr(v, "pop_size") == 30
    # Operator slots untouched.
    assert v.select_fn   is b.select_fn
    assert v.crossover_fn is b.crossover_fn
    assert v.mutate_fn   is b.mutate_fn


def test_disallowed_override_raises():
    cfg = _load("engineering")
    b = baseline_from_config(cfg, dim=4)
    spec = Spec(name="SGA+bad", overrides={"name": "tampered"})
    with pytest.raises(ValueError, match="not allowed"):
        variant_from_baseline(b, spec)


def test_wrapper_receives_baseline_slot():
    cfg = _load("engineering")
    b = baseline_from_config(cfg, dim=4)
    seen = {}
    def wrap(base):
        seen["base"] = base
        return lambda *a, **k: base(*a, **k)
    spec = Spec(name="SGA+wrap_select", select_wrapper=wrap)
    v = variant_from_baseline(b, spec)
    assert seen["base"] is b.select_fn
    assert v.select_fn is not b.select_fn


def test_slot_and_wrapper_conflict_raises():
    cfg = _load("engineering")
    b = baseline_from_config(cfg, dim=4)
    spec = Spec(
        name="SGA+conflict",
        select=lambda p, n, c: list(p),
        select_wrapper=lambda base: base,
    )
    with pytest.raises(ValueError, match="provide only one"):
        variant_from_baseline(b, spec)


def test_overridable_scalars_set_documented():
    # Sanity check that the contract surface is small and known.
    assert "pop_size" in _OVERRIDABLE_SCALARS
    assert "elitism_k" in _OVERRIDABLE_SCALARS
    assert "name" not in _OVERRIDABLE_SCALARS


def test_tsp_baseline_uses_permutation_operators():
    cfg = _load("tsp")
    b = baseline_from_config(cfg, dim=51)  # eil51
    assert "k2" in b.select_fn.__name__
    # Baseline crossover is EAX (Edge Assembly Crossover); make_crossover
    # wires {"kind": "eax"} to ga.features.make_eax().
    assert b.crossover_fn.__name__ == "eax_single"
    assert "swap" in b.mutate_fn.__name__
    assert b.cx_prob == 0.9
    assert b.elitism_k == 1
    assert b.bound_handling == "passthrough"
    assert b.init_fn is not None
    assert b.init_fn.__name__ == "random_permutation"


def test_timetabling_baseline_constructs():
    """UCTP baseline: Yang & Jat 2011 SSGA — discrete-integer init,
    slot-uniform crossover, N1/N2/N3 mutation, LS1 refinement,
    steady-state loop, passthrough bounds."""
    cfg = _load("timetabling")
    b = baseline_from_config(cfg, dim=100)
    assert b.init_fn.__name__ == "random_int_vector"
    assert b.crossover_fn.__name__ == "ssga_uniform_slot"
    assert b.mutate_fn.__name__ == "mutate_ssga_n1n2n3"
    assert b.refine_fn is not None
    assert b.loop_kind == "steady_state"
    assert b.bound_handling == "passthrough"
    assert b.cx_prob == 0.8
    assert b.mut_prob == 0.5


def test_variant_diff_reports_replacement_and_wrapper():
    cfg = _load("engineering")
    b = baseline_from_config(cfg, dim=4)
    spec = Spec(
        name="SGA+test",
        mutate=lambda ind, ctx: list(ind),
        select_wrapper=lambda base: base,
        overrides={"pop_size": 30},
    )
    v = variant_from_baseline(b, spec)
    diff = variant_diff(b, v, spec)
    assert diff["name"] == "SGA+test"
    assert "mutate" in diff["slots"]
    assert "select (wrapped)" in diff["slots"]
    assert diff["overrides"] == {"pop_size": 30}


def test_runner_persists_audit_files(tmp_path):
    """Runner-level smoke: a tiny --no-stats --no-plots --no-history run
    must produce suite_config.json and variants_diff.json next to the
    runs.parquet archive."""
    import subprocess
    repo = ROOT
    out = subprocess.run(
        [sys.executable, "scripts/run_experiments.py",
         "--suite", "cec2022",
         "--variants", "Levy_mut",
         "--n-runs", "1",
         "--out-root", str(tmp_path),
         "--tag", "audit_smoke",
         "--no-plots", "--no-stats", "--no-history"],
        cwd=repo, capture_output=True, text=True,
    )
    assert out.returncode == 0, out.stderr
    exp_dir = tmp_path / "audit_smoke"
    assert (exp_dir / "suite_config.json").exists()
    assert (exp_dir / "variants_diff.json").exists()
    cfg = json.loads((exp_dir / "suite_config.json").read_text())
    assert cfg["suite"] == "cec2022"
    diffs = json.loads((exp_dir / "variants_diff.json").read_text())
    names = [d["name"] for d in diffs]
    assert "SGA_baseline" in names and "SGA+Levy_mut" in names
