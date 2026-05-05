"""
CEC 2022 suite wrappers and the multi-variant run protocol.
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


def run_protocol(variants, dim=10, n_runs=30, max_fes=None, pop_size=50):
    """Returns dict[variant_name][func_name] -> list of error values across runs."""
    results = {v.name: {} for v in variants}
    total_jobs = len(variants) * len(CEC2022_FUNCS) * n_runs
    job = 0
    t0 = time.time()

    for func_name, FCls in CEC2022_FUNCS:
        problem = FCls(ndim=dim)
        for variant in variants:
            errors = []
            for seed in range(n_runs):
                job += 1
                r = run_sga(
                    problem=problem,
                    config=variant,
                    pop_size=pop_size,
                    max_fes=max_fes,
                    seed=seed,
                    record_history=False,
                )
                errors.append(r["error"])
            results[variant.name][func_name] = errors
            elapsed = time.time() - t0
            print(f"  [{job}/{total_jobs}] {func_name} x {variant.name}: "
                  f"mean err = {np.mean(errors):.3e}  "
                  f"({elapsed:.0f}s elapsed)")

    return results
