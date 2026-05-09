"""Tests for the config-driven baseline + variant inheritance contract."""

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from ga.build import (
    Spec, baseline_from_config, variant_from_baseline, _OVERRIDABLE_SCALARS,
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
