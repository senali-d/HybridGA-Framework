"""
Statistical evaluation for GA-variant comparisons.

Console-friendly summary:
    summarize(results, baseline_name)

Thesis-grade tables (writes CSVs/text into <out_dir>/summary/):
    write_summary(results, out_dir, baseline_name)

Underlying primitives (also usable standalone):
    mean_error_table(results)
    wilcoxon_table(results, baseline_name)
    cliffs_delta_table(results, baseline_name)
    friedman_with_ranks(results)
    holm_posthoc(results)
    nemenyi_posthoc(results)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# ============================================================
# Console summary (kept for backward compatibility)
# ============================================================

def summarize(results, baseline_name="SGA_baseline", func_order=None):
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
        print("(+ = variant better, - = variant worse, = = no significant diff at p<0.05)")
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
                    _, p = stats.wilcoxon(a, b)
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
    for name, rank in sorted(zip(variant_names, mean_ranks), key=lambda x: x[1]):
        print(f"  {name:<28} rank = {rank:.2f}")


# ============================================================
# Tabular primitives
# ============================================================

def mean_error_table(results) -> pd.DataFrame:
    variant_names = list(results.keys())
    func_names = list(results[variant_names[0]].keys())
    rows = []
    for fn in func_names:
        row = {"function": fn}
        for v in variant_names:
            arr = np.asarray(results[v][fn])
            row[f"{v}__mean"] = float(arr.mean())
            row[f"{v}__std"] = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def wilcoxon_table(results, baseline_name) -> pd.DataFrame:
    variant_names = list(results.keys())
    func_names = list(results[variant_names[0]].keys())
    rows = []
    for v in variant_names:
        if v == baseline_name:
            continue
        for fn in func_names:
            a = np.asarray(results[baseline_name][fn])
            b = np.asarray(results[v][fn])
            if np.allclose(a, b):
                stat_w, p = np.nan, 1.0
                sign = "="
            else:
                try:
                    stat_w, p = stats.wilcoxon(a, b)
                except ValueError:
                    stat_w, p = np.nan, 1.0
                if p < 0.05:
                    sign = "+" if np.median(b) < np.median(a) else "-"
                else:
                    sign = "="
            rows.append({
                "variant": v, "function": fn,
                "W": float(stat_w) if not np.isnan(stat_w) else None,
                "p_value": float(p), "sign_vs_baseline": sign,
                "median_baseline": float(np.median(a)),
                "median_variant": float(np.median(b)),
            })
    return pd.DataFrame(rows)


def _cliffs_delta(a: np.ndarray, b: np.ndarray) -> float:
    """Cliff's delta: P(a>b) - P(a<b). Negative => b smaller (variant better
    when smaller-is-better)."""
    a = np.asarray(a)
    b = np.asarray(b)
    gt = np.sum(a[:, None] > b[None, :])
    lt = np.sum(a[:, None] < b[None, :])
    n = a.size * b.size
    if n == 0:
        return 0.0
    return float((gt - lt) / n)


def _cliffs_magnitude(d: float) -> str:
    ad = abs(d)
    if ad < 0.147:
        return "negligible"
    if ad < 0.33:
        return "small"
    if ad < 0.474:
        return "medium"
    return "large"


def cliffs_delta_table(results, baseline_name) -> pd.DataFrame:
    variant_names = list(results.keys())
    func_names = list(results[variant_names[0]].keys())
    rows = []
    for v in variant_names:
        if v == baseline_name:
            continue
        for fn in func_names:
            a = np.asarray(results[baseline_name][fn])
            b = np.asarray(results[v][fn])
            d = _cliffs_delta(a, b)
            rows.append({
                "variant": v, "function": fn,
                "cliffs_delta": d, "magnitude": _cliffs_magnitude(d),
            })
    return pd.DataFrame(rows)


def friedman_with_ranks(results):
    variant_names = list(results.keys())
    func_names = list(results[variant_names[0]].keys())
    matrix = np.array([[np.mean(results[v][fn]) for v in variant_names]
                       for fn in func_names])
    per_row_ranks = np.array([stats.rankdata(row) for row in matrix])
    mean_ranks = per_row_ranks.mean(axis=0)
    try:
        chi2, p = stats.friedmanchisquare(
            *[matrix[:, j] for j in range(matrix.shape[1])])
    except Exception:
        chi2, p = float("nan"), float("nan")
    ranks_df = pd.DataFrame({"variant": variant_names,
                             "mean_rank": mean_ranks}
                            ).sort_values("mean_rank").reset_index(drop=True)
    return float(chi2), float(p), ranks_df, matrix, variant_names, func_names


def holm_posthoc(results) -> pd.DataFrame:
    """Pairwise Wilcoxon between every variant pair, p-values Holm-adjusted.

    Aggregates across functions: for each (vi, vj) pair we compare per-function
    mean errors using a paired Wilcoxon over the function set.
    """
    from statsmodels.stats.multitest import multipletests

    variant_names = list(results.keys())
    func_names = list(results[variant_names[0]].keys())
    means = {v: np.array([np.mean(results[v][fn]) for fn in func_names])
             for v in variant_names}

    pairs = []
    raw_p = []
    for i, vi in enumerate(variant_names):
        for vj in variant_names[i + 1:]:
            a, b = means[vi], means[vj]
            if np.allclose(a, b):
                p = 1.0
            else:
                try:
                    _, p = stats.wilcoxon(a, b)
                except ValueError:
                    p = 1.0
            pairs.append((vi, vj))
            raw_p.append(p)
    if not raw_p:
        return pd.DataFrame(columns=["variant_a", "variant_b", "p_raw", "p_holm", "reject_h0"])
    reject, p_adj, _, _ = multipletests(raw_p, alpha=0.05, method="holm")
    return pd.DataFrame({
        "variant_a": [a for a, _ in pairs],
        "variant_b": [b for _, b in pairs],
        "p_raw": raw_p,
        "p_holm": p_adj,
        "reject_h0": reject,
    })


def nemenyi_posthoc(results) -> pd.DataFrame:
    """Nemenyi post-hoc on the Friedman matrix (functions × variants of mean errors)."""
    import scikit_posthocs as sp
    variant_names = list(results.keys())
    func_names = list(results[variant_names[0]].keys())
    matrix = np.array([[np.mean(results[v][fn]) for v in variant_names]
                       for fn in func_names])
    df = pd.DataFrame(matrix, columns=variant_names, index=func_names)
    pmat = sp.posthoc_nemenyi_friedman(df.values)
    pmat.index = variant_names
    pmat.columns = variant_names
    return pmat


# ============================================================
# Disk writer
# ============================================================

def success_rate_table(results, epsilon: float = 1e-2) -> pd.DataFrame:
    """Fraction of runs whose final error <= epsilon, per (variant, function)."""
    variant_names = list(results.keys())
    func_names = list(results[variant_names[0]].keys())
    rows = []
    for fn in func_names:
        row = {"function": fn, "epsilon": epsilon}
        for v in variant_names:
            arr = np.asarray(results[v][fn])
            row[v] = float(np.mean(arr <= epsilon)) if arr.size else 0.0
        rows.append(row)
    return pd.DataFrame(rows)


def behavioral_fingerprint(history_df, runs_df) -> pd.DataFrame:
    """One row per variant: time-averaged behavioral metrics that summarize
    *how* the variant searches, independent of the specific function."""
    if history_df is None or history_df.empty:
        return pd.DataFrame()
    agg = (history_df.groupby("variant")
           .agg(time_avg_xpl_pct=("xpl_pct", "mean"),
                time_avg_xpt_pct=("xpt_pct", "mean"),
                mean_hussain_div=("hussain_div", "mean"),
                mean_step_size=("mean_step_size", "mean"),
                mean_mut_success=("mut_success_rate", "mean"),
                mean_cx_success=("cx_success_rate", "mean"),
                mean_selection_pressure=("selection_pressure", "mean"),
                mean_stagnation=("stagnation_count", "mean"))
           .reset_index())
    err = (runs_df.groupby("variant")["error"].mean()
           .rename("mean_final_error").reset_index())
    return agg.merge(err, on="variant", how="left")


def write_summary(results, out_dir, baseline_name="SGA_baseline",
                  history_df=None, runs_df=None,
                  success_epsilon: float = 1e-2) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mean_error_table(results).to_csv(out_dir / "mean_error_table.csv", index=False)
    wilcoxon_table(results, baseline_name).to_csv(
        out_dir / "wilcoxon_vs_baseline.csv", index=False)
    cliffs_delta_table(results, baseline_name).to_csv(
        out_dir / "cliffs_delta.csv", index=False)

    chi2, p, ranks_df, _, _, _ = friedman_with_ranks(results)
    ranks_df.to_csv(out_dir / "friedman_ranks.csv", index=False)
    with open(out_dir / "friedman.txt", "w") as f:
        f.write(f"Friedman chi2 = {chi2:.6f}\n")
        f.write(f"p-value       = {p:.6g}\n\n")
        f.write("Mean ranks (lower = better):\n")
        for _, row in ranks_df.iterrows():
            f.write(f"  {row['variant']:<28} {row['mean_rank']:.3f}\n")

    try:
        holm_posthoc(results).to_csv(out_dir / "holm_posthoc.csv", index=False)
    except Exception as e:
        (out_dir / "holm_posthoc.txt").write_text(f"Holm post-hoc failed: {e}\n")

    try:
        nemenyi_posthoc(results).to_csv(out_dir / "nemenyi_posthoc.csv")
    except Exception as e:
        (out_dir / "nemenyi_posthoc.txt").write_text(f"Nemenyi post-hoc failed: {e}\n")

    success_rate_table(results, epsilon=success_epsilon).to_csv(
        out_dir / f"success_rate_eps{success_epsilon:g}.csv", index=False)

    if history_df is not None and runs_df is not None:
        fp = behavioral_fingerprint(history_df, runs_df)
        if not fp.empty:
            fp.to_csv(out_dir / "behavioral_fingerprint.csv", index=False)

    print(f"Wrote summary tables to {out_dir}")
