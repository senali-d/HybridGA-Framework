"""
run_experiments.py
------------------
Run multiple SGA variants on the CEC 2022 suite and produce a comparison table.

USAGE:
    python run_experiments.py               # runs the demo (small budget)
    python run_experiments.py --full        # runs the full CEC 2022 protocol

The demo settings are deliberately small so you can verify the pipeline runs
end-to-end in ~1 minute. The full protocol is what you'd use for the thesis.
"""

import argparse
import json
import time
import numpy as np
from scipy import stats

from opfunu.cec_based.cec2022 import (
    F12022, F22022, F32022, F42022, F52022, F62022,
    F72022, F82022, F92022, F102022, F112022, F122022,
)

from sga_core import FeatureConfig, run_sga, default_select
import features as feat


# ============================================================
# Define the variants to compare
# ============================================================
def build_variants():
    """One baseline + a Tier-1 set of feature-injected variants.
    Add or remove entries here to expand the study."""
    variants = []

    # Baseline: pure SGA (default operators)
    variants.append(FeatureConfig(name="SGA_baseline"))

    # F03 — Opposition-based init
    variants.append(FeatureConfig(
        name="SGA+OBL_init",
        init_fn=feat.init_chaotic,  # using chaotic since strict OBL needs eval inside init
    ))

    # F02 — Chaotic-map init
    variants.append(FeatureConfig(
        name="SGA+chaotic_init",
        init_fn=feat.init_chaotic,
    ))

    # F13 — Lévy mutation
    variants.append(FeatureConfig(
        name="SGA+Levy_mut",
        mutate_fn=feat.mutate_levy,
    ))

    # F17 — DE-style differential mutation
    variants.append(FeatureConfig(
        name="SGA+DE_mut",
        select_fn=feat.with_population_aware_select(default_select),
        mutate_fn=feat.make_de_mutation(F=0.5),
    ))

    # F36 — Greedy one-to-one replacement
    variants.append(FeatureConfig(
        name="SGA+greedy_replace",
        replace_fn=feat.replace_greedy,
    ))

    # F40 — Linearly decreasing mutation rate
    variants.append(FeatureConfig(
        name="SGA+adaptive_mut",
        mutate_fn=feat.make_adaptive_mutate(base_indpb=0.1),
    ))

    # F39 — Stagnation restart
    variants.append(FeatureConfig(
        name="SGA+restart",
        diversity_fn=feat.make_diversity_restart(
            stagnation_threshold=15, restart_fraction=0.2),
    ))

    return variants


# ============================================================
# Run protocol
# ============================================================
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
                    record_history=False,  # save memory in big runs
                )
                errors.append(r["error"])
            results[variant.name][func_name] = errors
            elapsed = time.time() - t0
            print(f"  [{job}/{total_jobs}] {func_name} x {variant.name}: "
                  f"mean err = {np.mean(errors):.3e}  "
                  f"({elapsed:.0f}s elapsed)")

    return results


# ============================================================
# Statistical analysis (Wilcoxon + Friedman)
# ============================================================
def summarize(results, baseline_name="SGA_baseline"):
    """Print summary stats and Wilcoxon comparisons vs baseline."""
    print("\n" + "=" * 78)
    print("SUMMARY (mean error over runs)")
    print("=" * 78)
    func_names = [fn for fn, _ in CEC2022_FUNCS]
    variant_names = list(results.keys())

    # Header
    header = f"{'Func':<6}" + "".join(f"{v[:14]:>16}" for v in variant_names)
    print(header)
    for fn in func_names:
        row = f"{fn:<6}"
        for v in variant_names:
            row += f"{np.mean(results[v][fn]):>16.3e}"
        print(row)

    # Wilcoxon vs baseline
    if baseline_name in results:
        print("\n" + "=" * 78)
        print(f"WILCOXON SIGNED-RANK vs {baseline_name}")
        print("(+ = variant better, - = variant worse, = means no significant diff at p<0.05)")
        print("=" * 78)
        for v in variant_names:
            if v == baseline_name:
                continue
            wins = losses = ties = 0
            for fn in func_names:
                a = np.array(results[baseline_name][fn])
                b = np.array(results[v][fn])
                if np.allclose(a, b):
                    ties += 1
                    continue
                try:
                    stat, p = stats.wilcoxon(a, b)
                    if p < 0.05:
                        if np.median(b) < np.median(a):
                            wins += 1
                        else:
                            losses += 1
                    else:
                        ties += 1
                except ValueError:
                    ties += 1
            print(f"  {v:<28} +{wins}  ={ties}  -{losses}  (out of {len(func_names)})")

    # Friedman across all functions
    print("\n" + "=" * 78)
    print("FRIEDMAN RANK (lower = better)")
    print("=" * 78)
    # Build a matrix: rows = functions, cols = variants. Cell = mean error.
    matrix = np.array([[np.mean(results[v][fn]) for v in variant_names]
                        for fn in func_names])
    # Per-row ranking, then mean rank per column
    ranks = np.array([stats.rankdata(row) for row in matrix])
    mean_ranks = ranks.mean(axis=0)
    try:
        chi2, p = stats.friedmanchisquare(*[matrix[:, j] for j in range(matrix.shape[1])])
        print(f"  Friedman chi2 = {chi2:.3f}, p = {p:.4f}")
    except Exception as e:
        print(f"  Friedman test failed: {e}")
    sorted_pairs = sorted(zip(variant_names, mean_ranks), key=lambda x: x[1])
    for name, rank in sorted_pairs:
        print(f"  {name:<28} rank = {rank:.2f}")


# ============================================================
# Main
# ============================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="Run full CEC 2022 protocol (slow). Default: demo.")
    parser.add_argument("--dim", type=int, default=10)
    args = parser.parse_args()

    if args.full:
        n_runs = 30
        max_fes = 200_000 if args.dim <= 10 else 1_000_000
        pop_size = 50
        print(f"Running FULL protocol: D={args.dim}, MaxFES={max_fes}, runs={n_runs}")
    else:
        n_runs = 5
        max_fes = 5_000
        pop_size = 30
        print(f"Running DEMO protocol: D={args.dim}, MaxFES={max_fes}, runs={n_runs}")
        print("(Use --full for the real CEC 2022 budget.)")

    variants = build_variants()
    print(f"Variants to test: {[v.name for v in variants]}\n")

    results = run_protocol(variants, dim=args.dim, n_runs=n_runs,
                            max_fes=max_fes, pop_size=pop_size)

    summarize(results, baseline_name="SGA_baseline")

    # Persist raw data so you can do more analysis later
    out = {v: {f: list(map(float, errs)) for f, errs in funcs.items()}
            for v, funcs in results.items()}
    with open("results.json", "w") as f:
        json.dump(out, f, indent=2)
    print("\nRaw results saved to results.json")


if __name__ == "__main__":
    main()
