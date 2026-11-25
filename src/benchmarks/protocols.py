"""
Multi-variant run protocol for every benchmark suite.

Process-pool parallelism over (variant, function, seed) tuples uses the
`spawn` start method (safe on macOS, where `fork` is unsafe with NumPy/
Accelerate). Variants contain closures from `make_*` factories which the
standard pickler cannot serialize; we use `cloudpickle` to encode them in
the parent and decode inside each worker.
"""

import time
import cloudpickle
import numpy as np

from ga.core import run_sga
from benchmarks.cec2022_functions import CEC2022_FUNCS, CEC2022_DIM_SUPPORTED
from benchmarks.cec2020_realworld import ENGINEERING_FUNCS
from benchmarks.tsp import TSPLIB_FUNCS


_SUITES = {
    "cec2022": dict(CEC2022_FUNCS),
    "cec2020_realworld": dict(ENGINEERING_FUNCS),
    "tsp": dict(TSPLIB_FUNCS),
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
    dim-whitelist check (used by suites with fixed per-instance dims)."""
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
    """CEC 2022 protocol. Returns dict[variant_name][func_name] -> list of
    error values across runs (sorted by seed). If `recorder` is provided,
    every run's full result is pushed to it via `recorder.add_run(...)`.

    workers : int — 1 = sequential (default). N>1 = process pool with N
    workers using the `spawn` start method. Each (variant, function, seed)
    tuple is an independent job; `run_sga` is pure and seed-deterministic,
    so results are identical regardless of `workers`.
    """
    return _run_protocol(
        suite_name="cec2022",
        funcs_registry=CEC2022_FUNCS,
        dim_supported_map=CEC2022_DIM_SUPPORTED,
        variants=variants, dim=dim, n_runs=n_runs, max_fes=max_fes,
        pop_size=pop_size, recorder=recorder, record_history=record_history,
        workers=workers, functions=functions,
    )


def run_tsp_protocol(variants, dim=None, n_runs=30, max_fes=None,
                     pop_size=100, recorder=None, record_history=True,
                     workers=1, functions=None):
    """TSPLIB EUC_2D instances. Each instance has a fixed city count baked
    into its `.tsp` file, so `dim` is accepted for interface compatibility
    but ignored by the constructors."""
    return _run_protocol(
        suite_name="tsp",
        funcs_registry=TSPLIB_FUNCS,
        dim_supported_map=None,
        variants=variants, dim=(dim if dim is not None else 0),
        n_runs=n_runs, max_fes=max_fes,
        pop_size=pop_size, recorder=recorder, record_history=record_history,
        workers=workers, functions=functions,
    )


def run_cec2020_realworld_protocol(variants, dim=None, n_runs=30,
                                   max_fes=None, pop_size=50, recorder=None,
                                   record_history=True, workers=1,
                                   functions=None):
    """CEC 2020 real-world (5 mechanical engineering problems). Each problem
    has a fixed dimensionality (3-7); `dim` is passed through but ignored
    by the problem constructors. Constraints are handled via Deb's
    feasibility rules encoded into the scalar fitness — see
    `benchmarks/cec2020_realworld.py`.
    """
    return _run_protocol(
        suite_name="cec2020_realworld",
        funcs_registry=ENGINEERING_FUNCS,
        dim_supported_map=None,
        variants=variants, dim=(dim if dim is not None else 0),
        n_runs=n_runs, max_fes=max_fes,
        pop_size=pop_size, recorder=recorder, record_history=record_history,
        workers=workers, functions=functions,
    )
