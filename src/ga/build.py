"""
Construct SGA baselines and variants from a per-suite JSON config.

The config is the single source of truth for the SGA spec of a problem
family (CEC 2022, engineering, TSP, timetabling, ...). A variant is a
*minimal declaration*: it names the operator slot(s) it changes, and
optionally lists explicit per-variant overrides for scalar knobs. Anything
not declared is inherited bit-identical from the suite's baseline.

This makes it impossible for a "mutation-only" variant to silently diverge
on (say) selection or population size — those would have to appear in the
variant's `Spec.overrides` dict where they are visible to reviewers.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from ga.core import (
    FeatureConfig,
    make_select, make_crossover, make_mutation, make_replace,
)


# Scalar knobs on FeatureConfig that a variant may override via
# Spec.overrides. Anything outside this set is inherited from the baseline.
_OVERRIDABLE_SCALARS = frozenset({
    "cx_prob", "mut_prob", "elitism_k",
    "n_islands", "migration_interval", "archive_size", "bound_handling",
    # Pseudo-knobs that the run protocol reads directly (not on FeatureConfig
    # itself but still legitimate per-variant overrides):
    "pop_size", "max_fes", "n_runs",
})


@dataclass
class Spec:
    """Declarative variant specification.

    A variant changes exactly the operator slots passed here; everything
    else is inherited from the suite baseline. Use `overrides` to express
    intentional scalar divergence (e.g. memetic variants needing a smaller
    population to fund local search). All other knobs MUST stay equal to
    the baseline.

    Slot fields accept either a *replacement* closure (e.g. `mutate=`) or a
    *wrapper* (e.g. `select_wrapper=lambda base: feat.population_aware(base)`).
    A wrapper takes the baseline's current closure and returns a new one;
    this lets a variant compose with the suite's chosen operator (e.g. keep
    the engineering suite's binary tournament while adding a population-
    aware filter on top). At most one of {slot, slot_wrapper} per slot.
    """
    name: str
    encodings: tuple = ("real",)
    # Replacement slots:
    init: Optional[Callable] = None
    select: Optional[Callable] = None
    crossover: Optional[Callable] = None
    mutate: Optional[Callable] = None
    replace: Optional[Callable] = None
    diversity: Optional[Callable] = None
    refine: Optional[Callable] = None
    # Wrapper slots: each takes the baseline's slot and returns a new closure.
    init_wrapper: Optional[Callable] = None
    select_wrapper: Optional[Callable] = None
    crossover_wrapper: Optional[Callable] = None
    mutate_wrapper: Optional[Callable] = None
    replace_wrapper: Optional[Callable] = None
    overrides: dict = field(default_factory=dict)


def _resolve_max_fes(max_fes_field, dim):
    if isinstance(max_fes_field, dict):
        table = {int(k): int(v) for k, v in max_fes_field.items()}
        if dim not in table:
            raise ValueError(
                f"max_fes table has no entry for dim={dim}; "
                f"available dims: {sorted(table)}"
            )
        return table[dim]
    return int(max_fes_field)


def baseline_from_config(suite_cfg: dict, dim: int) -> FeatureConfig:
    """Build the suite's canonical SGA baseline from a parsed JSON config.

    Returned object carries:
      - operator closures wired from the JSON's `selection`/`crossover`/
        `mutation`/`elitism` blocks via the factories in `ga.core`.
      - scalars: `cx_prob` from crossover.pc, `mut_prob` always 1.0
        (per-individual gate; intensity lives in mutation.pm_per_gene),
        `elitism_k` from elitism.k.
    """
    sel_spec = suite_cfg.get("selection", {"kind": "tournament", "k": 3})
    cx_spec  = suite_cfg.get("crossover", {"kind": "sbx", "eta": 20, "pc": 0.9})
    mut_spec = suite_cfg.get("mutation",  {"kind": "poly", "eta": 20})
    elt_spec = suite_cfg.get("elitism",   {"k": 1})

    cfg = FeatureConfig(
        name=f"SGA_baseline_{suite_cfg.get('suite', 'unknown')}",
        select_fn=make_select(sel_spec),
        crossover_fn=make_crossover(cx_spec),
        mutate_fn=make_mutation(mut_spec, dim=dim),
        replace_fn=make_replace(elt_spec),
        cx_prob=float(cx_spec.get("pc", 0.9)),
        # Per-individual mutation gate. Operator semantics (per-gene poly
        # rate, per-position swap rate, whole-vector inversion rate) live
        # inside the mutation closure built by `make_mutation`. The gate
        # therefore defaults to 1.0 and only diverges if the JSON sets
        # `mutation.pm_per_individual` explicitly.
        mut_prob=float(mut_spec.get("pm_per_individual", 1.0)),
        elitism_k=int(elt_spec.get("k", 1)),
    )
    return cfg


def variant_from_baseline(baseline: FeatureConfig, spec: Spec) -> FeatureConfig:
    """Clone `baseline`, swap declared operator slots, apply explicit
    overrides. Every parameter not touched stays bit-identical."""
    # Shallow copy is safe: every FeatureConfig field is either a function
    # reference (closures are not mutated post-construction) or an
    # immutable scalar (int / float / str / bool). Adding a mutable field
    # later (list/dict) will require switching to copy.deepcopy here.
    v = copy.copy(baseline)
    v.name = spec.name

    def _resolve(slot_attr, repl, wrap, slot_label):
        if repl is not None and wrap is not None:
            raise ValueError(
                f"variant {spec.name!r}: provide only one of "
                f"{slot_label} / {slot_label}_wrapper")
        if repl is not None:
            setattr(v, slot_attr, repl)
        elif wrap is not None:
            base_slot = getattr(baseline, slot_attr)
            setattr(v, slot_attr, wrap(base_slot))

    _resolve("init_fn",      spec.init,      spec.init_wrapper,      "init")
    _resolve("select_fn",    spec.select,    spec.select_wrapper,    "select")
    _resolve("crossover_fn", spec.crossover, spec.crossover_wrapper, "crossover")
    _resolve("mutate_fn",    spec.mutate,    spec.mutate_wrapper,    "mutate")
    _resolve("replace_fn",   spec.replace,   spec.replace_wrapper,   "replace")
    if spec.diversity is not None: v.diversity_fn = spec.diversity
    if spec.refine    is not None: v.refine_fn    = spec.refine

    for key, value in spec.overrides.items():
        if key not in _OVERRIDABLE_SCALARS:
            raise ValueError(
                f"variant {spec.name!r}: override {key!r} not allowed; "
                f"allowed keys: {sorted(_OVERRIDABLE_SCALARS)}"
            )
        # pop_size / max_fes / n_runs are read by the run protocol, not
        # by FeatureConfig; we attach them as attributes so the runner can
        # pick them up without changing FeatureConfig's signature.
        setattr(v, key, value)
    return v


def variant_diff(baseline: FeatureConfig, variant: FeatureConfig,
                 spec: Spec) -> dict:
    """Return a small dict describing exactly what differs from baseline.
    Used for run-log printouts and per-experiment audit trails."""
    changed_slots = []
    for slot in ("init", "select", "crossover", "mutate",
                 "replace", "diversity", "refine"):
        if getattr(spec, slot) is not None:
            changed_slots.append(slot)
    for slot in ("init", "select", "crossover", "mutate", "replace"):
        if getattr(spec, f"{slot}_wrapper") is not None:
            changed_slots.append(f"{slot} (wrapped)")
    return {
        "name": variant.name,
        "slots": changed_slots,
        "overrides": dict(spec.overrides),
        "encodings": list(spec.encodings),
    }
