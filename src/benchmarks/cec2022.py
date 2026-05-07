"""
CEC 2022 suite wrappers and the multi-variant run protocol.

Optional process-pool parallelism over (variant, function, seed) tuples
using the `spawn` start method (safe on macOS, where `fork` is unsafe with
NumPy/Accelerate). Variants contain closures from `make_*` factories which
the standard pickler cannot serialize; we use `cloudpickle` to encode them
in the parent and decode inside each worker.
"""

import time
import cloudpickle
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

# Statically encoded from opfunu's CEC2022 sources: F1-F5 ship shift/rotation
# data for D ∈ {2, 10, 20}; the hybrid (F6-F8) and composition (F9-F12) suites
# only ship D ∈ {10, 20}. We mirror the values here because some F6+ classes
# fail in __init__ (loading missing data files) before `dim_supported` is set,
# which makes runtime probing unreliable.
_DIM_SUPPORTED = {
    "F1": (2, 10, 20), "F2": (2, 10, 20), "F3": (2, 10, 20),
    "F4": (2, 10, 20), "F5": (2, 10, 20),
    "F6": (10, 20), "F7": (10, 20), "F8": (10, 20),
    "F9": (10, 20), "F10": (10, 20), "F11": (10, 20), "F12": (10, 20),
}


def _worker_run_one(payload):
    """Worker entry point. `payload` is fully picklable (bytes + primitives).

    Variants are cloudpickled in the parent because their `make_*` closures
    aren't picklable by the default pickler. We deserialize once per worker
    job; for spawn-based pools each future is a fresh job, so there is no
    cross-job caching of decoded variants.
    """
    variant_blob, func_name, seed, kw = payload
    variant = cloudpickle.loads(variant_blob)
    FCls = _FUNCS_BY_NAME[func_name]
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
        the `fork` start method. Each `(variant, function, seed)` tuple is
        an independent job. `run_sga` is pure and seed-deterministic, so
        results are identical regardless of `workers`.
    """
    if functions is None:
        funcs = list(CEC2022_FUNCS)
    else:
        wanted = [f.strip() for f in functions if f.strip()]
        unknown = [f for f in wanted if f not in _FUNCS_BY_NAME]
        if unknown:
            raise ValueError(
                f"Unknown CEC2022 function(s): {unknown}. "
                f"Valid: {[fn for fn, _ in CEC2022_FUNCS]}"
            )
        funcs = [(fn, _FUNCS_BY_NAME[fn]) for fn in wanted]

    # Validate dim against each function's supported dims up front so workers
    # don't fail mid-dispatch.
    unsupported = [
        (fn, _DIM_SUPPORTED[fn]) for fn, _ in funcs
        if dim not in _DIM_SUPPORTED[fn]
    ]
    if unsupported:
        raise ValueError(
            f"dim={dim} is not supported by: "
            + ", ".join(f"{fn} (supports {sup})" for fn, sup in unsupported)
        )

    total_jobs = len(variants) * len(funcs) * n_runs
    t0 = time.time()

    # Per-(variant, function): seed -> error, so we can reassemble in seed
    # order regardless of completion order under parallelism.
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
        # Cloudpickle each variant once; pass the blob to workers. This
        # bypasses the default pickler's inability to serialize the closures
        # returned by make_* factories.
        variant_blobs = {v.name: cloudpickle.dumps(v) for v in variants}
        kw = {
            "dim": dim, "pop_size": pop_size, "max_fes": max_fes,
            "record_history": record_history,
        }
        all_payloads = [
            (variant_blobs[v.name], fn, seed, kw)
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
        # Sequential path with rich per-(function, variant) progress lines.
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

    # Materialize results dict in deterministic seed order.
    results = {
        v.name: {
            fn: [err_by_seed[v.name][fn][s] for s in sorted(err_by_seed[v.name][fn])]
            for fn, _ in funcs
        }
        for v in variants
    }
    return results
