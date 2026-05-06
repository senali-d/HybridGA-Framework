"""
CEC 2022 suite wrappers and the multi-variant run protocol.

Optional process-pool parallelism over (variant, function, seed) tuples.
Workers use the `fork` start method so closures inside variants (returned by
`make_*` factories) don't need to be picklable — they're inherited via
copy-on-write memory. Spawn-based parallelism would require all features
to be top-level callables.
"""

import time
import numpy as np

from opfunu.cec_based.cec2022 import (
    F12022, F22022, F32022, F42022, F52022, F62022,
    F72022, F82022, F92022, F102022, F112022, F122022,
)

from ga.core import run_sga


CEC2022_FUNCS = [
    ("F1",  F12022), ("F2",  F22022), ("F3",  F32022), ("F4",  F42022),
    ("F5",  F52022), ("F6",  F62022), ("F7",  F72022), ("F8",  F82022),
    ("F9",  F92022), ("F10", F102022), ("F11", F112022), ("F12", F122022),
]
_FUNCS_BY_NAME = dict(CEC2022_FUNCS)


# Module-level state populated in the parent before forking workers; children
# inherit it without pickling. Keep this top-level so children see the same
# binding after a fork-based ProcessPoolExecutor spawns them.
_WORKER_VARIANTS: dict = {}
_WORKER_KW: dict = {}


def _worker_run_one(args):
    """Worker entry point. `args` is small and fully picklable."""
    vname, func_name, seed = args
    variant = _WORKER_VARIANTS[vname]
    FCls = _FUNCS_BY_NAME[func_name]
    problem = FCls(ndim=_WORKER_KW["dim"])
    t_run = time.time()
    r = run_sga(
        problem=problem,
        config=variant,
        pop_size=_WORKER_KW["pop_size"],
        max_fes=_WORKER_KW["max_fes"],
        seed=seed,
        record_history=_WORKER_KW["record_history"],
    )
    return vname, func_name, seed, r, time.time() - t_run


def run_protocol(variants, dim=10, n_runs=30, max_fes=None, pop_size=50,
                 recorder=None, record_history=True, workers=1):
    """Returns dict[variant_name][func_name] -> list of error values across
    runs (sorted by seed). If `recorder` is provided, every run's full
    result is also pushed to it via `recorder.add_run(...)`.

    Parameters
    ----------
    workers : int
        1 = sequential (default). N>1 = process pool with N workers using
        the `fork` start method. Each `(variant, function, seed)` tuple is
        an independent job. `run_sga` is pure and seed-deterministic, so
        results are identical regardless of `workers`.
    """
    total_jobs = len(variants) * len(CEC2022_FUNCS) * n_runs
    t0 = time.time()

    # Per-(variant, function): seed -> error, so we can reassemble in seed
    # order regardless of completion order under parallelism.
    err_by_seed: dict = {
        v.name: {fn: {} for fn, _ in CEC2022_FUNCS} for v in variants
    }

    if workers > 1:
        import multiprocessing as mp
        from concurrent.futures import ProcessPoolExecutor, as_completed

        # Populate inherited state for forked workers.
        global _WORKER_VARIANTS, _WORKER_KW
        _WORKER_VARIANTS = {v.name: v for v in variants}
        _WORKER_KW = {
            "dim": dim, "pop_size": pop_size, "max_fes": max_fes,
            "record_history": record_history,
        }

        try:
            ctx = mp.get_context("fork")
        except ValueError:
            print("WARNING: fork start method unavailable; falling back to "
                  "sequential execution.")
            workers = 1

    if workers > 1:
        all_args = [
            (v.name, fn, seed)
            for fn, _ in CEC2022_FUNCS
            for v in variants
            for seed in range(n_runs)
        ]
        print(f"  Dispatching {len(all_args)} jobs across {workers} workers "
              f"(fork)...")
        done = 0
        with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as ex:
            futures = [ex.submit(_worker_run_one, a) for a in all_args]
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
        # Sequential path with rich per-(function, variant) progress lines.
        job = 0
        for func_name, FCls in CEC2022_FUNCS:
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

    # Materialize results dict in deterministic seed order.
    results = {
        v.name: {
            fn: [err_by_seed[v.name][fn][s] for s in sorted(err_by_seed[v.name][fn])]
            for fn, _ in CEC2022_FUNCS
        }
        for v in variants
    }
    return results
