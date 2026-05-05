"""
Statistical summary: mean-error table, Wilcoxon vs baseline, Friedman ranks.
"""

import numpy as np
from scipy import stats


def summarize(results, baseline_name="SGA_baseline", func_order=None):
    """Print summary stats and Wilcoxon comparisons vs baseline.

    `func_order` controls the row order; defaults to the keys of the first
    variant in `results`.
    """
    print("\n" + "=" * 78)
    print("SUMMARY (mean error over runs)")
    print("=" * 78)
    variant_names = list(results.keys())
    func_names = func_order or list(results[variant_names[0]].keys())

    header = f"{'Func':<6}" + "".join(f"{v[:14]:>16}" for v in variant_names)
    print(header)
    for fn in func_names:
        row = f"{fn:<6}"
        for v in variant_names:
            row += f"{np.mean(results[v][fn]):>16.3e}"
        print(row)

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

    print("\n" + "=" * 78)
    print("FRIEDMAN RANK (lower = better)")
    print("=" * 78)
    matrix = np.array([[np.mean(results[v][fn]) for v in variant_names]
                        for fn in func_names])
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
