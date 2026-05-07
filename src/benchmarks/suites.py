"""
Benchmark suites and the multi-variant run protocol.

Two suites live here:

  1. CEC 2022 (F1-F12) — rotated/shifted, mostly non-separable.
  2. Separable    (Sphere, Rastrigin, Schwefel, Sum of Different Powers, Step)
                  — canonical separable functions for control comparisons.

Both expose the same instance interface (`ndim`, `lb`, `ub`,
`evaluate(x) -> float`, `f_global`) so `run_sga` consumes them unchanged.

Process-pool parallelism over (variant, function, seed) tuples uses the
`spawn` start method (safe on macOS, where `fork` is unsafe with NumPy/
Accelerate). Variants contain closures from `make_*` factories which the
standard pickler cannot serialize; we use `cloudpickle` to encode them in
the parent and decode inside each worker.
"""

import time
import cloudpickle
import numpy as np

from opfunu.cec_based.cec2022 import (
    F12022, F22022, F32022, F42022, F52022, F62022,
    F72022, F82022, F92022, F102022, F112022, F122022,
)
from pymoo.problems.single import Rastrigin as _PymooRastrigin
from pymoo.problems.single import Schwefel as _PymooSchwefel

from ga.core import run_sga
from benchmarks.engineering import ENGINEERING_FUNCS


# ============================================================================
# CEC 2022 SUITE
# ============================================================================

CEC2022_FUNCS = [
    ("F1",  F12022), ("F2",  F22022), ("F3",  F32022), ("F4",  F42022),
    ("F5",  F52022), ("F6",  F62022), ("F7",  F72022), ("F8",  F82022),
    ("F9",  F92022), ("F10", F102022), ("F11", F112022), ("F12", F122022),
]

# Statically encoded from opfunu's CEC2022 sources: F1-F5 ship shift/rotation
# data for D ∈ {2, 10, 20}; the hybrid (F6-F8) and composition (F9-F12) suites
# only ship D ∈ {10, 20}. We mirror the values here because some F6+ classes
# fail in __init__ (loading missing data files) before `dim_supported` is set,
# which makes runtime probing unreliable.
_CEC2022_DIM_SUPPORTED = {
    "F1": (2, 10, 20), "F2": (2, 10, 20), "F3": (2, 10, 20),
    "F4": (2, 10, 20), "F5": (2, 10, 20),
    "F6": (10, 20), "F7": (10, 20), "F8": (10, 20),
    "F9": (10, 20), "F10": (10, 20), "F11": (10, 20), "F12": (10, 20),
}


# ============================================================================
# SEPARABLE SUITE
# ----------------------------------------------------------------------------
# Rastrigin and Schwefel reuse pymoo's canonical implementations. Sphere,
# SumOfDifferentPowers, and Step are implemented locally because pymoo either
# ships a shifted variant (Sphere: `Σ(x_i-0.5)^2` on [0,1]) or doesn't ship
# the function at all.
# ============================================================================

class _SeparableProblem:
    """Common base. Subclasses set bounds via class attrs and implement
    `_eval(x)`. Exposes the (ndim, lb, ub, evaluate, f_global) interface."""

    _LB = None  # scalar lower bound
    _UB = None  # scalar upper bound
    f_global = 0.0

    def __init__(self, ndim):
        if ndim < 1:
            raise ValueError(f"ndim must be >= 1, got {ndim}")
        self.ndim = int(ndim)
        self.lb = np.full(self.ndim, self._LB, dtype=float)
        self.ub = np.full(self.ndim, self._UB, dtype=float)

    def evaluate(self, x):
        x = np.asarray(x, dtype=float)
        return float(self._eval(x))

    def _eval(self, x):  # pragma: no cover - abstract
        raise NotImplementedError


class Sphere(_SeparableProblem):
    """f(x) = Σ x_i^2. Optimum f*=0 at x*=0."""
    _LB, _UB = -5.12, 5.12

    def _eval(self, x):
        return np.sum(x * x)


class SumOfDifferentPowers(_SeparableProblem):
    """f(x) = Σ |x_i|^(i+1). Optimum f*=0 at x*=0."""
    _LB, _UB = -1.0, 1.0

    def _eval(self, x):
        powers = np.arange(2, self.ndim + 2, dtype=float)
        return np.sum(np.abs(x) ** powers)


class Step(_SeparableProblem):
    """De Jong F3: f(x) = Σ floor(x_i + 0.5)^2. Optimum f*=0 at x*=0."""
    _LB, _UB = -100.0, 100.0

    def _eval(self, x):
        return np.sum(np.floor(x + 0.5) ** 2)


class _PymooAdapter(_SeparableProblem):
    """Wraps a pymoo single-objective Problem. Subclasses set `_PYMOO_CLS`."""

    _PYMOO_CLS = None

    def __init__(self, ndim):
        super().__init__(ndim)
        self._inner = self._PYMOO_CLS(n_var=self.ndim)
        # Sync bounds from pymoo (authoritative for that function's canonical form).
        self.lb = np.asarray(self._inner.xl, dtype=float)
        self.ub = np.asarray(self._inner.xu, dtype=float)

    def _eval(self, x):
        out = {}
        self._inner._evaluate(x[None, :], out)
        return out["F"][0]


class Rastrigin(_PymooAdapter):
    """f(x) = 10n + Σ[x_i^2 - 10 cos(2π x_i)] (pymoo). Optimum f*=0 at x*=0."""
    _PYMOO_CLS = _PymooRastrigin
    _LB, _UB = -5.12, 5.12  # used only if pymoo bounds were unavailable


class Schwefel(_PymooAdapter):
    """Schwefel 2.26 (pymoo): f(x) = 418.9829 n - Σ x_i sin(√|x_i|).
    Optimum f*=0 at x*≈420.9687·1."""
    _PYMOO_CLS = _PymooSchwefel
    _LB, _UB = -500.0, 500.0


SEPARABLE_FUNCS = [
    ("Sphere", Sphere),
    ("Rastrigin", Rastrigin),
    ("Schwefel", Schwefel),
    ("SumOfDifferentPowers", SumOfDifferentPowers),
    ("Step", Step),
]


# ============================================================================
# RUN PROTOCOL (shared dispatch + per-suite wrappers)
# ============================================================================

_SUITES = {
    "cec2022": dict(CEC2022_FUNCS),
    "separable": dict(SEPARABLE_FUNCS),
    "engineering": dict(ENGINEERING_FUNCS),
}


def _worker_run_one(payload):
    """Worker entry point. `payload` is fully picklable (bytes + primitives).

    Variants are cloudpickled in the parent because their `make_*` closures
    aren't picklable by the default pickler. We deserialize once per worker
    job; for spawn-based pools each future is a fresh job, so there is no
    cross-job caching of decoded variants.
    """
    variant_blob, suite_name, func_name, seed, kw = payload
    variant = cloudpickle.loads(variant_blob)
    FCls = _SUITES[suite_name][func_name]
    problem = FCls(ndim=kw["dim"])
    t_run = time.time()
    r = run_sga(
        problem=problem,
        config=variant,
        pop_size=kw["pop_size"],
        max_fes=kw["max_fes"],
        seed=seed,
        record_history=kw["record_history"],
    )
    return variant.name, func_name, seed, r, time.time() - t_run


def _run_protocol(suite_name, funcs_registry, dim_supported_map,
                  variants, dim, n_runs, max_fes, pop_size,
                  recorder, record_history, workers, functions):
    """Shared dispatch core. `dim_supported_map` may be None to skip the
    dim-whitelist check (used by the separable suite which supports any
    positive int)."""
    by_name = dict(funcs_registry)
    if functions is None:
        funcs = list(funcs_registry)
    else:
        wanted = [f.strip() for f in functions if f.strip()]
        unknown = [f for f in wanted if f not in by_name]
        if unknown:
            raise ValueError(
                f"Unknown {suite_name} function(s): {unknown}. "
                f"Valid: {[fn for fn, _ in funcs_registry]}"
            )
        funcs = [(fn, by_name[fn]) for fn in wanted]

    if dim_supported_map is not None:
        unsupported = [
            (fn, dim_supported_map[fn]) for fn, _ in funcs
            if dim not in dim_supported_map[fn]
        ]
        if unsupported:
            raise ValueError(
                f"dim={dim} is not supported by: "
                + ", ".join(f"{fn} (supports {sup})" for fn, sup in unsupported)
            )

    total_jobs = len(variants) * len(funcs) * n_runs
    t0 = time.time()

    err_by_seed: dict = {
        v.name: {fn: {} for fn, _ in funcs} for v in variants
    }

    if workers > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed

        try:
            mp_ctx = mp.get_context("spawn")
        except ValueError:
            print("WARNING: spawn start method unavailable; falling back to "
                  "sequential execution.")
            workers = 1

    if workers > 1:
        variant_blobs = {v.name: cloudpickle.dumps(v) for v in variants}
        kw = {
            "dim": dim, "pop_size": pop_size, "max_fes": max_fes,
            "record_history": record_history,
        }
        all_payloads = [
            (variant_blobs[v.name], suite_name, fn, seed, kw)
            for fn, _ in funcs
            for v in variants
            for seed in range(n_runs)
        ]
        print(f"  Dispatching {len(all_payloads)} jobs across {workers} "
              f"workers (spawn)...")
        done = 0
        with ProcessPoolExecutor(max_workers=workers, mp_context=mp_ctx) as ex:
            futures = [ex.submit(_worker_run_one, p) for p in all_payloads]
            for fut in as_completed(futures):
                vname, fn, seed, r, runtime_s = fut.result()
                err_by_seed[vname][fn][seed] = r["error"]
                if recorder is not None:
                    recorder.add_run(vname, fn, seed, r, runtime_s)
                done += 1
                if done % 50 == 0 or done == total_jobs:
                    elapsed = time.time() - t0
                    rate = done / max(elapsed, 1e-9)
                    eta = (total_jobs - done) / max(rate, 1e-9)
                    print(f"  [{done}/{total_jobs}] "
                          f"({elapsed:.0f}s elapsed, ~{eta:.0f}s left)")
    else:
        job = 0
        for func_name, FCls in funcs:
            problem = FCls(ndim=dim)
            for variant in variants:
                for seed in range(n_runs):
                    job += 1
                    t_run = time.time()
                    r = run_sga(
                        problem=problem, config=variant, pop_size=pop_size,
                        max_fes=max_fes, seed=seed,
                        record_history=record_history,
                    )
                    runtime_s = time.time() - t_run
                    err_by_seed[variant.name][func_name][seed] = r["error"]
                    if recorder is not None:
                        recorder.add_run(variant.name, func_name, seed, r,
                                         runtime_s)
                errs_now = list(err_by_seed[variant.name][func_name].values())
                elapsed = time.time() - t0
                print(f"  [{job}/{total_jobs}] {func_name} x {variant.name}: "
                      f"mean err = {np.mean(errs_now):.3e}  "
                      f"({elapsed:.0f}s elapsed)")

    results = {
        v.name: {
            fn: [err_by_seed[v.name][fn][s] for s in sorted(err_by_seed[v.name][fn])]
            for fn, _ in funcs
        }
        for v in variants
    }
    return results


def run_protocol(variants, dim=10, n_runs=30, max_fes=None, pop_size=50,
                 recorder=None, record_history=True, workers=1,
                 functions=None):
    """Returns dict[variant_name][func_name] -> list of error values across
    runs (sorted by seed). If `recorder` is provided, every run's full
    result is also pushed to it via `recorder.add_run(...)`.

    Parameters
    ----------
    workers : int
        1 = sequential (default). N>1 = process pool with N workers using
        the `spawn` start method. Each `(variant, function, seed)` tuple is
        an independent job. `run_sga` is pure and seed-deterministic, so
        results are identical regardless of `workers`.
    """
    return _run_protocol(
        suite_name="cec2022",
        funcs_registry=CEC2022_FUNCS,
        dim_supported_map=_CEC2022_DIM_SUPPORTED,
        variants=variants, dim=dim, n_runs=n_runs, max_fes=max_fes,
        pop_size=pop_size, recorder=recorder, record_history=record_history,
        workers=workers, functions=functions,
    )


def run_separable_protocol(variants, dim=10, n_runs=30, max_fes=None,
                           pop_size=50, recorder=None, record_history=True,
                           workers=1, functions=None):
    """Same contract as `run_protocol` but for the separable suite. Any
    positive `dim` is accepted (no dimension whitelist)."""
    return _run_protocol(
        suite_name="separable",
        funcs_registry=SEPARABLE_FUNCS,
        dim_supported_map=None,
        variants=variants, dim=dim, n_runs=n_runs, max_fes=max_fes,
        pop_size=pop_size, recorder=recorder, record_history=record_history,
        workers=workers, functions=functions,
    )


def run_engineering_protocol(variants, dim=None, n_runs=30, max_fes=None,
                             pop_size=50, recorder=None, record_history=True,
                             workers=1, functions=None):
    """Same contract as `run_protocol` but for the engineering suite.

    Each engineering problem has a fixed dimensionality (3-7); the `dim`
    argument is passed through to the worker but ignored by the problem
    constructors. Constraints are handled via Deb's feasibility rules
    encoded into the scalar fitness — see `benchmarks/engineering.py`.
    """
    return _run_protocol(
        suite_name="engineering",
        funcs_registry=ENGINEERING_FUNCS,
        dim_supported_map=None,
        variants=variants, dim=(dim if dim is not None else 0),
        n_runs=n_runs, max_fes=max_fes,
        pop_size=pop_size, recorder=recorder, record_history=record_history,
        workers=workers, functions=functions,
    )
