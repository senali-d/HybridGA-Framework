#!/usr/bin/env python3
"""
Suite-level Friedman + Holm + Nemenyi post-hoc completion.

Recomputes the post-hoc statistics that the original per-run analysis skipped
(statsmodels / scikit_posthocs were not installed for some cells) by POOLING
each suite's per-function archives into a single results[variant][function]
dict and calling the framework's own stats functions. Also renders a Nemenyi
critical-difference diagram per suite with Title-case variant labels.

No GA runs are executed. Reads results/<archive>/runs.parquet (read-only);
writes only into ./results/<suite>/.

Run with the framework venv:
    <framework>/venv/bin/python run_posthoc.py
"""
from __future__ import annotations
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJ = Path("/Users/senalidilumika/Documents/MCS/Research/Thesis-HybridGA")
FRAMEWORK = Path("/Users/senalidilumika/Documents/MCS/Research/Repos/HybridGA-Framework")
RES = PROJ / "results"
HERE = Path(__file__).resolve().parent
OUT = HERE / "results"

sys.path.insert(0, str(FRAMEWORK / "src"))
from analysis.stats import write_summary, friedman_with_ranks, holm_posthoc  # noqa: E402

BASELINE = "SGA_baseline"

# Title-case display names, matching the thesis prose and figures.
NAME = {
    "SGA_baseline": "SGA", "SGA+DE_mut": "SGA-DE", "SGA+SHADE_mut": "SGA-SHADE",
    "SGA+abc_scout": "SGA-ABC", "SGA+archive_replace": "SGA-Archive",
    "SGA+bbo_migration_cx": "SGA-BBO", "SGA+bfo_chemotaxis": "SGA-BFO",
    "SGA+cuckoo_discovery": "SGA-Cuckoo", "SGA+greedy_replace": "SGA-Greedy",
    "SGA+gsa_full_mut": "SGA-GSA", "SGA+immune_memory": "SGA-Immune",
    "SGA+krill_herd_mut": "SGA-Krill", "SGA+metropolis_replace": "SGA-Metropolis",
    "SGA+net_suppress": "SGA-NetSup", "SGA+pher_minmax": "SGA-Pher",
    "SGA+restart": "SGA-Restart", "SGA+sos_mut": "SGA-SOS",
}

TSP_LABEL = {
    "tsp_berlin52_all": "berlin52", "tsp_eil51_all": "eil51", "tsp_st70_all": "st70",
    "tsp_perm_kroA100_all16": "kroA100", "tsp_perm_ch150_all16": "ch150",
    "tsp_perm_kroA200_all16": "kroA200", "tsp_perm_pcb442_all16": "pcb442",
}

SUITES = {
    "cec2022_D10_validation": [f"cec2022_F{i}_subset10" for i in range(1, 13)],
    "cec2022_D20": [f"cec2022_real_D20_F{i}_all16" for i in range(1, 13)],
    "cec2020_engineering": [f"cec2020_F{i}_subset10" for i in range(1, 6)],
    "tsp_all7": list(TSP_LABEL.keys()),
    "tsp_large4": ["tsp_perm_kroA100_all16", "tsp_perm_ch150_all16",
                   "tsp_perm_kroA200_all16", "tsp_perm_pcb442_all16"],
}


def label_of(archive: str) -> str:
    """Distinct per-archive function key so pooling does not collide."""
    if archive in TSP_LABEL:
        return TSP_LABEL[archive]
    m = re.search(r"F(\d+)", archive)
    return f"F{m.group(1)}" if m else archive


def runs_df_to_results_dict(runs_df: pd.DataFrame) -> dict:
    results: dict = {}
    for (variant, function), grp in runs_df.groupby(["variant", "function"]):
        results.setdefault(variant, {})[function] = grp.sort_values("seed")["error"].tolist()
    return results


def complete_results(results: dict) -> dict:
    """Ensure every variant has every function. A missing (variant, function)
    cell is a feasibility failure (e.g. Pressure Vessel returned no feasible
    solution); fill it with a strictly-worst finite value so that variant ranks
    last on that problem -- consistent with the thesis counting such a failure
    as a loss. Keeps Friedman/Holm/Nemenyi well defined."""
    all_fns = sorted({fn for d in results.values() for fn in d})
    for fn in all_fns:
        present = [results[u][fn] for u in results if fn in results[u]]
        worst = max(np.mean(a) for a in present)
        n = max(len(a) for a in present)
        fill = worst * 1.01 + 1.0
        for v, d in results.items():
            d.setdefault(fn, [fill] * n)
    return results


def load_pool(archives: list[str]) -> pd.DataFrame:
    frames = []
    for a in archives:
        df = pd.read_parquet(RES / a / "runs.parquet").copy()
        df["function"] = label_of(a)           # distinct key per archive
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def holm_vs_baseline(results: dict) -> pd.DataFrame:
    """Convenience view: each variant's Holm-adjusted pairwise p vs the baseline,
    with the win/lose direction by suite-mean error (lower = better)."""
    holm = holm_posthoc(results)
    fns = list(results[BASELINE].keys())
    mean_err = {v: float(np.mean([np.mean(results[v][f]) for f in fns])) for v in results}
    rows = []
    for _, r in holm.iterrows():
        if BASELINE not in (r.variant_a, r.variant_b):
            continue
        other = r.variant_b if r.variant_a == BASELINE else r.variant_a
        better = mean_err[other] < mean_err[BASELINE]
        rows.append({
            "variant": NAME.get(other, other),
            "p_raw": r.p_raw, "p_holm": r.p_holm,
            "reject_h0": bool(r.reject_h0),
            "direction": ("variant_better" if better else "baseline_better"),
        })
    return pd.DataFrame(rows).sort_values("p_holm").reset_index(drop=True)


def cd_diagram(results: dict, title: str, save_path: Path) -> None:
    import scikit_posthocs as sp
    res2 = {NAME.get(v, v): d for v, d in results.items()}
    _, _, ranks_df, _, vnames, fnames = friedman_with_ranks(res2)
    matrix = np.array([[np.mean(res2[v][fn]) for v in vnames] for fn in fnames])
    pmat = sp.posthoc_nemenyi_friedman(matrix)
    pmat.index = vnames
    pmat.columns = vnames
    avg_rank = ranks_df.set_index("variant")["mean_rank"].reindex(vnames)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    sp.critical_difference_diagram(avg_rank, pmat, ax=ax)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    report = ["# Post-hoc completion — report\n",
              "Suite-level Friedman + Holm + Nemenyi recomputed on existing "
              "`runs.parquet` (no GA runs). Lower Friedman rank = better.\n"]
    summary_rows = []
    for suite, archives in SUITES.items():
        runs_df = load_pool(archives)
        results = complete_results(runs_df_to_results_dict(runs_df))
        out_suite = OUT / suite
        (out_suite).mkdir(parents=True, exist_ok=True)

        write_summary(results, out_suite / "summary",
                      baseline_name=BASELINE, runs_df=runs_df)
        hvb = holm_vs_baseline(results)
        hvb.to_csv(out_suite / "holm_vs_baseline.csv", index=False)
        cd_diagram(results, "Critical Difference (Nemenyi, α=0.05)",
                   out_suite / "cd_diagram.png")

        chi2, p, ranks_df, _, vnames, fnames = friedman_with_ranks(results)
        full_holm = holm_posthoc(results)
        n_sig = int(hvb["reject_h0"].sum())
        n_sig_total = int(full_holm["reject_h0"].sum())
        best = NAME.get(ranks_df.iloc[0]["variant"], ranks_df.iloc[0]["variant"])
        summary_rows.append({
            "suite": suite, "n_problems": len(fnames), "n_variants": len(vnames),
            "friedman_chi2": round(chi2, 3), "friedman_p": p,
            "holm_sig_vs_baseline": n_sig,
            "holm_sig_pairs_total": n_sig_total, "holm_pairs": len(full_holm),
            "best_variant": best, "best_rank": round(float(ranks_df.iloc[0]["mean_rank"]), 2),
        })
        report.append(
            f"## {suite}\n"
            f"- functions/instances: {len(fnames)} ({', '.join(fnames)})\n"
            f"- variants: {len(vnames)}\n"
            f"- Friedman: chi2 = {chi2:.3f}, p = {p:.3e}\n"
            f"- best mean rank: {best} ({ranks_df.iloc[0]['mean_rank']:.2f})\n"
            f"- Holm-significant pairs: {n_sig_total}/{len(full_holm)} total, "
            f"{n_sig} vs baseline\n"
        )
        print(f"[done] {suite}: chi2={chi2:.2f} p={p:.2e}  "
              f"Holm-sig {n_sig_total}/{len(full_holm)} (vs baseline {n_sig})")
    pd.DataFrame(summary_rows).to_csv(OUT / "suite_summary.csv", index=False)

    # Validation note
    report.append(
        "## Validation\n"
        "- `cec2022_D20` Friedman chi2 is expected to be ~95 (the thesis states "
        "\"chi2 ~ 95 at D=20, k=17, N=12, p<1e-9\"). See the value above.\n"
        "- `cec2022_D10_validation` rank order should match the thesis's reported "
        "CEC 2022 D=10 Friedman ranking (Table 4.x).\n"
    )
    (OUT / "REPORT.md").write_text("\n".join(report))
    print(f"\nWrote outputs + REPORT.md to {OUT}")


if __name__ == "__main__":
    main()
