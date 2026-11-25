"""
Plot generation for thesis-grade analysis.

All plotters consume the parquet artifacts written by ExperimentRecorder
(`runs.parquet`, `history.parquet`) so figures can be regenerated standalone.

Public entry point:
    generate_all(exp_dir)            # produce every figure
"""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .stats import (
    friedman_with_ranks,
    cliffs_delta_table,
    wilcoxon_table,
    behavioral_fingerprint,
)


sns.set_theme(context="paper", style="whitegrid")


# ---------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------

def _ensure(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


_TRAILING_INT_RE = re.compile(r"(\d+)$")


def _func_sort_key(s: str) -> tuple:
    # CEC-style "F1", "F12" sort numerically; TSPLIB names like "eil51",
    # "kroA100" sort by trailing integer then prefix. Falls back to (1, 0, name)
    # for anything without a trailing integer so it still sorts deterministically.
    m = _TRAILING_INT_RE.search(s)
    if m:
        return (0, int(m.group(1)), s)
    return (1, 0, s)


def _function_order(df: pd.DataFrame) -> list[str]:
    return sorted(df["function"].unique(), key=_func_sort_key)


def _variants(df: pd.DataFrame) -> list[str]:
    return sorted(df["variant"].unique())


def _results_dict(runs_df: pd.DataFrame) -> dict:
    out: dict[str, dict[str, list[float]]] = {}
    for v, vdf in runs_df.groupby("variant"):
        out[v] = {}
        for fn, fdf in vdf.groupby("function"):
            out[v][fn] = fdf.sort_values("seed")["error"].tolist()
    return out


def _interp_to_grid(g: pd.DataFrame, fes_grid: np.ndarray, ycol: str) -> np.ndarray:
    """Linear-interpolate one run's series onto a shared FE grid (forward fill)."""
    g = g.sort_values("fes")
    fes = g["fes"].to_numpy()
    y = g[ycol].to_numpy()
    if len(fes) == 0:
        return np.full_like(fes_grid, np.nan, dtype=float)
    # best_fitness is monotone non-increasing already; for diversity it isn't
    return np.interp(fes_grid, fes, y, left=y[0], right=y[-1])


# ---------------------------------------------------------------
# Convergence
# ---------------------------------------------------------------

def plot_convergence(history_df: pd.DataFrame, out_dir: Path,
                     ycol: str = "best_fitness") -> None:
    out_combined = _ensure(out_dir / "convergence")
    out_individual = _ensure(out_dir / "convergence" / "per_variant")

    funcs = _function_order(history_df)
    variants = _variants(history_df)
    palette = dict(zip(variants, sns.color_palette("tab10", n_colors=len(variants))))

    for fn in funcs:
        sub = history_df[history_df["function"] == fn]
        if sub.empty:
            continue
        max_fes = int(sub["fes"].max())
        fes_grid = np.linspace(1, max_fes, 200)

        # combined plot: mean ± std band per variant
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for v in variants:
            vsub = sub[sub["variant"] == v]
            if vsub.empty:
                continue
            curves = np.stack([
                _interp_to_grid(g, fes_grid, ycol) for _, g in vsub.groupby("seed")
            ])
            mean = curves.mean(axis=0)
            std = curves.std(axis=0)
            ax.plot(fes_grid, mean, label=v, color=palette[v], lw=1.4)
            ax.fill_between(fes_grid, mean - std, mean + std,
                            color=palette[v], alpha=0.15)
        ax.set_yscale("symlog", linthresh=1e-8)
        ax.set_xlabel("Function evaluations")
        ax.set_ylabel("Best fitness (so far)")
        ax.set_title(f"Convergence — {fn}")
        ax.legend(fontsize=7, loc="best", ncol=2)
        fig.tight_layout()
        fig.savefig(out_combined / f"{fn}_all.png", dpi=150)
        plt.close(fig)

        # per-variant plots: every seed light, mean bold
        for v in variants:
            vsub = sub[sub["variant"] == v]
            if vsub.empty:
                continue
            fig, ax = plt.subplots(figsize=(6, 4))
            curves = []
            for _, g in vsub.groupby("seed"):
                y = _interp_to_grid(g, fes_grid, ycol)
                ax.plot(fes_grid, y, color="0.7", lw=0.7, alpha=0.7)
                curves.append(y)
            curves = np.stack(curves)
            ax.plot(fes_grid, curves.mean(axis=0),
                    color=palette[v], lw=1.8, label="mean")
            ax.set_yscale("symlog", linthresh=1e-8)
            ax.set_xlabel("Function evaluations")
            ax.set_ylabel("Best fitness")
            ax.set_title(f"{fn} — {v}")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(out_individual / f"{fn}_{v}.png", dpi=150)
            plt.close(fig)


# ---------------------------------------------------------------
# Boxplots
# ---------------------------------------------------------------

def plot_boxplots(runs_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir = _ensure(out_dir / "boxplots")
    funcs = _function_order(runs_df)
    variants = _variants(runs_df)

    for fn in funcs:
        sub = runs_df[runs_df["function"] == fn]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.boxplot(data=sub, x="variant", y="error", order=variants, ax=ax)
        ax.set_yscale("symlog", linthresh=1e-8)
        ax.set_title(f"Final error — {fn}")
        ax.tick_params(axis="x", rotation=30)
        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment("right")
        fig.tight_layout()
        fig.savefig(out_dir / f"{fn}_box.png", dpi=150)
        plt.close(fig)

    # 3x4 grid of all 12 functions
    n = len(funcs)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows),
                             sharey=False)
    for i, fn in enumerate(funcs):
        ax = axes.flat[i]
        sub = runs_df[runs_df["function"] == fn]
        sns.boxplot(data=sub, x="variant", y="error", order=variants, ax=ax)
        ax.set_yscale("symlog", linthresh=1e-8)
        ax.set_title(fn)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=45, labelsize=7)
        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment("right")
    for j in range(n, rows * cols):
        axes.flat[j].axis("off")
    fig.suptitle("Final error distributions across CEC 2022 suite")
    fig.tight_layout()
    fig.savefig(out_dir / "all_functions_grid.png", dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------
# Diversity
# ---------------------------------------------------------------

def plot_diversity(history_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir = _ensure(out_dir / "diversity")
    funcs = _function_order(history_df)
    variants = _variants(history_df)
    palette = dict(zip(variants, sns.color_palette("tab10", n_colors=len(variants))))

    for fn in funcs:
        sub = history_df[history_df["function"] == fn]
        if sub.empty:
            continue
        max_fes = int(sub["fes"].max())
        fes_grid = np.linspace(1, max_fes, 200)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for v in variants:
            vsub = sub[sub["variant"] == v]
            if vsub.empty:
                continue
            curves = np.stack([
                _interp_to_grid(g, fes_grid, "diversity") for _, g in vsub.groupby("seed")
            ])
            ax.plot(fes_grid, curves.mean(axis=0), label=v,
                    color=palette[v], lw=1.4)
        ax.set_xlabel("Function evaluations")
        ax.set_ylabel("Population diversity (mean per-dim std)")
        ax.set_title(f"Diversity — {fn}")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(out_dir / f"{fn}_all.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------
# Rank / CD diagram
# ---------------------------------------------------------------

def plot_friedman_bars(runs_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir = _ensure(out_dir / "ranks")
    results = _results_dict(runs_df)
    _, _, ranks_df, *_ = friedman_with_ranks(results)
    fig, ax = plt.subplots(figsize=(7, 4))
    sns.barplot(data=ranks_df, x="variant", y="mean_rank", ax=ax,
                color=sns.color_palette("tab10")[0])
    ax.set_title("Friedman mean ranks (lower is better)")
    ax.tick_params(axis="x", rotation=30)
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")
    fig.tight_layout()
    fig.savefig(out_dir / "friedman_ranks.png", dpi=150)
    plt.close(fig)


def plot_cd_diagram(runs_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir = _ensure(out_dir / "ranks")
    try:
        import scikit_posthocs as sp
    except Exception as e:
        (out_dir / "cd_diagram.txt").write_text(f"scikit-posthocs unavailable: {e}\n")
        return

    results = _results_dict(runs_df)
    _, _, ranks_df, _, variant_names, func_names = friedman_with_ranks(results)
    matrix = np.array([[np.mean(results[v][fn]) for v in variant_names]
                       for fn in func_names])
    df = pd.DataFrame(matrix, columns=variant_names, index=func_names)
    avg_rank = ranks_df.set_index("variant")["mean_rank"]
    avg_rank = avg_rank.reindex(variant_names)
    try:
        pmat = sp.posthoc_nemenyi_friedman(df.values)
        pmat.index = variant_names
        pmat.columns = variant_names
        fig, ax = plt.subplots(figsize=(8, 3.5))
        sp.critical_difference_diagram(avg_rank, pmat, ax=ax)
        ax.set_title("Critical Difference (Nemenyi, α=0.05)")
        fig.tight_layout()
        fig.savefig(out_dir / "cd_diagram.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        (out_dir / "cd_diagram.txt").write_text(f"CD diagram failed: {e}\n")


# ---------------------------------------------------------------
# Heatmaps (Wilcoxon sign + Cliff's delta)
# ---------------------------------------------------------------

def plot_heatmaps(runs_df: pd.DataFrame, out_dir: Path,
                  baseline_name: str = "SGA_baseline") -> None:
    out_dir = _ensure(out_dir / "heatmaps")
    results = _results_dict(runs_df)

    if baseline_name in results:
        wdf = wilcoxon_table(results, baseline_name)
        sign_map = {"+": 1, "=": 0, "-": -1}
        pivot = wdf.assign(s=wdf["sign_vs_baseline"].map(sign_map)) \
                   .pivot(index="variant", columns="function", values="s")
        # preserve numeric F-order
        pivot = pivot.reindex(columns=sorted(pivot.columns,
                                             key=_func_sort_key))
        fig, ax = plt.subplots(figsize=(max(6, 0.5 * pivot.shape[1] + 4),
                                        0.5 * pivot.shape[0] + 2))
        sns.heatmap(pivot, cmap="RdYlGn", center=0, vmin=-1, vmax=1,
                    cbar_kws={"label": "Wilcoxon sign vs baseline"},
                    annot=True, fmt="d", ax=ax)
        ax.set_title(f"Wilcoxon sign vs {baseline_name} (+1 better, −1 worse)")
        fig.tight_layout()
        fig.savefig(out_dir / "wilcoxon_heatmap.png", dpi=150)
        plt.close(fig)

        cdf = cliffs_delta_table(results, baseline_name)
        pivot = cdf.pivot(index="variant", columns="function", values="cliffs_delta")
        pivot = pivot.reindex(columns=sorted(pivot.columns,
                                             key=_func_sort_key))
        fig, ax = plt.subplots(figsize=(max(6, 0.5 * pivot.shape[1] + 4),
                                        0.5 * pivot.shape[0] + 2))
        sns.heatmap(pivot, cmap="RdYlGn", center=0, vmin=-1, vmax=1,
                    annot=True, fmt=".2f",
                    cbar_kws={"label": "Cliff's δ vs baseline (negative = better)"},
                    ax=ax)
        ax.set_title(f"Cliff's delta vs {baseline_name}")
        fig.tight_layout()
        fig.savefig(out_dir / "cliffs_delta_heatmap.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------
# ECDF / performance profile
# ---------------------------------------------------------------

def plot_ecdf_performance(history_df: pd.DataFrame, out_dir: Path,
                          target_precision: float = 1e-2) -> None:
    """For each variant, fraction of runs reaching best_fitness<=tau by FE budget t.
    Produces a pooled-across-functions plot AND one plot per function."""
    out_dir = _ensure(out_dir / "ecdf")
    variants = _variants(history_df)
    funcs = _function_order(history_df)
    palette = dict(zip(variants, sns.color_palette("tab10", n_colors=len(variants))))

    def _ecdf_for(sub_df, ax, title):
        max_fes = int(sub_df["fes"].max())
        fes_grid = np.linspace(1, max_fes, 200)
        for v in variants:
            vsub = sub_df[sub_df["variant"] == v]
            if vsub.empty:
                continue
            hits = []
            for (_k), g in vsub.groupby(["function", "seed"]):
                curve = _interp_to_grid(g, fes_grid, "best_fitness")
                hits.append(curve <= target_precision)
            if not hits:
                continue
            ecdf = np.mean(np.stack(hits), axis=0)
            ax.plot(fes_grid, ecdf, label=v, color=palette[v], lw=1.4)
        ax.set_xlabel("Function evaluations")
        ax.set_ylabel(f"Fraction with best ≤ {target_precision:g}")
        ax.set_title(title)
        ax.set_ylim(-0.02, 1.02)
        ax.legend(fontsize=7, ncol=2)

    # Pooled across all functions
    fig, ax = plt.subplots(figsize=(7, 4.5))
    _ecdf_for(history_df, ax,
              "Performance profile (all functions pooled)")
    fig.tight_layout()
    fig.savefig(out_dir / "ecdf_all_functions.png", dpi=150)
    plt.close(fig)

    # Per-function
    for fn in funcs:
        sub = history_df[history_df["function"] == fn]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(7, 4.5))
        _ecdf_for(sub, ax, f"Performance profile — {fn}")
        fig.tight_layout()
        fig.savefig(out_dir / f"{fn}_ecdf.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------
# Stagnation timeline
# ---------------------------------------------------------------

def plot_stagnation(history_df: pd.DataFrame, out_dir: Path) -> None:
    """Distribution of generations between successive best-so-far improvements."""
    out_dir = _ensure(out_dir / "stagnation")
    rows = []
    for (v, fn, seed), g in history_df.groupby(["variant", "function", "seed"]):
        g = g.sort_values("gen")
        bf = g["best_fitness"].to_numpy()
        gens = g["gen"].to_numpy()
        improve_idx = np.where(np.diff(bf) < 0)[0] + 1
        if len(improve_idx) < 2:
            continue
        gaps = np.diff(gens[improve_idx])
        for gap in gaps:
            rows.append({"variant": v, "function": fn, "gap": int(gap)})
    if not rows:
        return
    df = pd.DataFrame(rows)
    variants = _variants(history_df)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.boxplot(data=df, x="variant", y="gap", ax=ax, order=variants)
    ax.set_yscale("log")
    ax.set_ylabel("Generations between improvements")
    ax.set_title("Stagnation gap distribution (all functions pooled)")
    ax.tick_params(axis="x", rotation=30)
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")
    fig.tight_layout()
    fig.savefig(out_dir / "stagnation_box_all.png", dpi=150)
    plt.close(fig)

    for fn in df["function"].unique():
        sub = df[df["function"] == fn]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.boxplot(data=sub, x="variant", y="gap", ax=ax, order=variants)
        ax.set_yscale("log")
        ax.set_ylabel("Generations between improvements")
        ax.set_title(f"Stagnation gap — {fn}")
        ax.tick_params(axis="x", rotation=30)
        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment("right")
        fig.tight_layout()
        fig.savefig(out_dir / f"{fn}_box.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------
# Exploration ↔ Exploitation phase plot
# ---------------------------------------------------------------

def plot_phase(history_df: pd.DataFrame, out_dir: Path) -> None:
    """Scatter of (diversity, |Δ best_fitness|) per generation, colored by gen.
    Pooled-across-functions panel AND per-function panels."""
    out_dir = _ensure(out_dir / "phase")
    variants = _variants(history_df)
    funcs = _function_order(history_df)
    n = len(variants)
    cols = 3
    rows = (n + cols - 1) // cols

    def _draw_panels(sub_df, title, save_path):
        fig, axes = plt.subplots(rows, cols, figsize=(4.5 * cols, 3.5 * rows),
                                 squeeze=False)
        for i, v in enumerate(variants):
            ax = axes.flat[i]
            vsub = sub_df[sub_df["variant"] == v].copy()
            if vsub.empty:
                ax.axis("off"); continue
            vsub = vsub.sort_values(["function", "seed", "gen"])
            vsub["delta_best"] = (vsub.groupby(["function", "seed"])["best_fitness"]
                                      .diff().fillna(0).abs())
            agg = vsub.groupby("gen", as_index=False).agg(
                diversity=("diversity", "mean"),
                delta=("delta_best", "mean"),
            )
            sc = ax.scatter(agg["diversity"], agg["delta"] + 1e-12,
                            c=agg["gen"], cmap="viridis", s=10)
            ax.set_yscale("log")
            ax.set_xlabel("Diversity")
            ax.set_ylabel("|Δ best| (log)")
            ax.set_title(v, fontsize=9)
            plt.colorbar(sc, ax=ax, label="Gen")
        for j in range(n, rows * cols):
            axes.flat[j].axis("off")
        fig.suptitle(title)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

    _draw_panels(history_df,
                 "Exploration ↔ Exploitation phase (all functions pooled)",
                 out_dir / "phase_per_variant.png")

    for fn in funcs:
        sub = history_df[history_df["function"] == fn]
        if sub.empty:
            continue
        _draw_panels(sub,
                     f"Exploration ↔ Exploitation phase — {fn}",
                     out_dir / f"{fn}_phase.png")


# ---------------------------------------------------------------
# Exploration / Exploitation (XPL%) curves
# ---------------------------------------------------------------

def plot_xpl_xpt(history_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir = _ensure(out_dir / "xpl_xpt")
    if "xpl_pct" not in history_df.columns:
        return
    funcs = _function_order(history_df)
    variants = _variants(history_df)
    palette = dict(zip(variants, sns.color_palette("tab10", n_colors=len(variants))))

    for fn in funcs:
        sub = history_df[history_df["function"] == fn]
        if sub.empty:
            continue
        max_fes = int(sub["fes"].max())
        fes_grid = np.linspace(1, max_fes, 200)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for v in variants:
            vsub = sub[sub["variant"] == v]
            if vsub.empty:
                continue
            curves = np.stack([
                _interp_to_grid(g, fes_grid, "xpl_pct") for _, g in vsub.groupby("seed")
            ])
            ax.plot(fes_grid, curves.mean(axis=0), label=v,
                    color=palette[v], lw=1.4)
        ax.set_xlabel("Function evaluations")
        ax.set_ylabel("XPL% (exploration)")
        ax.set_ylim(0, 105)
        ax.set_title(f"Exploration vs Exploitation — {fn}")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(out_dir / f"{fn}_xpl.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------
# Step size distribution
# ---------------------------------------------------------------

def plot_step_size(history_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir = _ensure(out_dir / "step_size")
    if "mean_step_size" not in history_df.columns:
        return
    funcs = _function_order(history_df)
    variants = _variants(history_df)
    palette = dict(zip(variants, sns.color_palette("tab10", n_colors=len(variants))))

    # Pooled view across all functions
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.boxplot(data=history_df, x="variant", y="mean_step_size",
                order=variants, ax=ax)
    ax.set_yscale("symlog", linthresh=1e-3)
    ax.set_title("Mean parent→child step size (all functions/seeds pooled)")
    ax.tick_params(axis="x", rotation=30)
    for tick in ax.get_xticklabels():
        tick.set_horizontalalignment("right")
    fig.tight_layout()
    fig.savefig(out_dir / "step_size_box_all.png", dpi=150)
    plt.close(fig)

    # Per-function distribution boxplot
    for fn in funcs:
        sub = history_df[history_df["function"] == fn]
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.boxplot(data=sub, x="variant", y="mean_step_size",
                    order=variants, ax=ax)
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_title(f"Step size distribution — {fn}")
        ax.tick_params(axis="x", rotation=30)
        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment("right")
        fig.tight_layout()
        fig.savefig(out_dir / f"{fn}_box.png", dpi=150)
        plt.close(fig)

    # Per-function step-size vs FEs curve (mean across seeds)
    for fn in funcs:
        sub = history_df[history_df["function"] == fn]
        if sub.empty:
            continue
        max_fes = int(sub["fes"].max())
        fes_grid = np.linspace(1, max_fes, 200)
        fig, ax = plt.subplots(figsize=(7, 4.5))
        for v in variants:
            vsub = sub[sub["variant"] == v]
            if vsub.empty:
                continue
            curves = np.stack([
                _interp_to_grid(g, fes_grid, "mean_step_size")
                for _, g in vsub.groupby("seed")
            ])
            ax.plot(fes_grid, curves.mean(axis=0), label=v,
                    color=palette[v], lw=1.4)
        ax.set_yscale("symlog", linthresh=1e-3)
        ax.set_xlabel("Function evaluations")
        ax.set_ylabel("Mean step size")
        ax.set_title(f"Step size vs FEs — {fn}")
        ax.legend(fontsize=7, ncol=2)
        fig.tight_layout()
        fig.savefig(out_dir / f"{fn}_curve.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------
# Operator success rates
# ---------------------------------------------------------------

def plot_operator_success(history_df: pd.DataFrame, out_dir: Path) -> None:
    out_dir = _ensure(out_dir / "op_success")
    if "mut_success_rate" not in history_df.columns:
        return
    variants = _variants(history_df)
    funcs = _function_order(history_df)
    palette = dict(zip(variants, sns.color_palette("tab10", n_colors=len(variants))))

    # Pooled across all functions
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, col, title in zip(axes, ["mut_success_rate", "cx_success_rate"],
                              ["Mutation success rate", "Crossover success rate"]):
        sns.boxplot(data=history_df, x="variant", y=col, order=variants, ax=ax)
        ax.set_title(f"{title} (all functions pooled)")
        ax.set_ylabel("Fraction of children improving")
        ax.tick_params(axis="x", rotation=30)
        for tick in ax.get_xticklabels():
            tick.set_horizontalalignment("right")
    fig.tight_layout()
    fig.savefig(out_dir / "operator_success_box_all.png", dpi=150)
    plt.close(fig)

    # Per-function: mutation + crossover side by side
    for fn in funcs:
        sub = history_df[history_df["function"] == fn]
        if sub.empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
        for ax, col, title in zip(axes, ["mut_success_rate", "cx_success_rate"],
                                  ["Mutation success", "Crossover success"]):
            sns.boxplot(data=sub, x="variant", y=col, order=variants, ax=ax)
            ax.set_title(f"{title} — {fn}")
            ax.set_ylabel("Fraction improving")
            ax.tick_params(axis="x", rotation=30)
            for tick in ax.get_xticklabels():
                tick.set_horizontalalignment("right")
        fig.tight_layout()
        fig.savefig(out_dir / f"{fn}_box.png", dpi=150)
        plt.close(fig)

    # Per-function curve: mutation success vs FEs (one line per variant)
    for fn in funcs:
        sub = history_df[history_df["function"] == fn]
        if sub.empty:
            continue
        max_fes = int(sub["fes"].max())
        fes_grid = np.linspace(1, max_fes, 200)
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
        for ax, col, title in zip(axes, ["mut_success_rate", "cx_success_rate"],
                                  ["Mutation success rate", "Crossover success rate"]):
            for v in variants:
                vsub = sub[sub["variant"] == v]
                if vsub.empty:
                    continue
                curves = np.stack([
                    _interp_to_grid(g, fes_grid, col)
                    for _, g in vsub.groupby("seed")
                ])
                ax.plot(fes_grid, curves.mean(axis=0), label=v,
                        color=palette[v], lw=1.3)
            ax.set_title(f"{title} — {fn}")
            ax.set_xlabel("Function evaluations")
            ax.set_ylim(-0.02, 1.02)
        axes[0].set_ylabel("Fraction improving")
        axes[1].legend(fontsize=7, ncol=2, loc="best")
        fig.tight_layout()
        fig.savefig(out_dir / f"{fn}_curve.png", dpi=150)
        plt.close(fig)


# ---------------------------------------------------------------
# Behavioral fingerprint (clustering)
# ---------------------------------------------------------------

def plot_fingerprint(history_df: pd.DataFrame, runs_df: pd.DataFrame,
                     out_dir: Path) -> None:
    out_dir = _ensure(out_dir / "fingerprint")
    fp = behavioral_fingerprint(history_df, runs_df)
    if fp.empty or len(fp) < 2:
        return
    feat_cols = [c for c in fp.columns if c != "variant"]
    X = fp[feat_cols].to_numpy()
    # z-score normalize each column
    mu = X.mean(axis=0)
    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Xn = (X - mu) / sd
    fp_norm = pd.DataFrame(Xn, columns=feat_cols, index=fp["variant"])

    # Heatmap of normalized fingerprints
    fig, ax = plt.subplots(figsize=(max(8, 0.7 * len(feat_cols) + 4),
                                    0.5 * len(fp) + 2))
    sns.heatmap(fp_norm, cmap="RdBu_r", center=0, annot=True, fmt=".2f",
                cbar_kws={"label": "z-score (per metric)"}, ax=ax)
    ax.set_title("Behavioral fingerprint (z-scored per metric)")
    fig.tight_layout()
    fig.savefig(out_dir / "fingerprint_heatmap.png", dpi=150)
    plt.close(fig)

    # Hierarchical clustering dendrogram
    try:
        from scipy.cluster.hierarchy import linkage, dendrogram
        Z = linkage(Xn, method="average", metric="euclidean")
        fig, ax = plt.subplots(figsize=(8, 4.5))
        dendrogram(Z, labels=fp["variant"].tolist(), leaf_rotation=30,
                   leaf_font_size=9, ax=ax)
        ax.set_title("Behavioral fingerprint clustering (Ward-avg over z-scored metrics)")
        ax.set_ylabel("Distance")
        fig.tight_layout()
        fig.savefig(out_dir / "fingerprint_dendrogram.png", dpi=150)
        plt.close(fig)
    except Exception as e:
        (out_dir / "fingerprint_dendrogram.txt").write_text(f"Dendrogram failed: {e}\n")


# ---------------------------------------------------------------
# Master entry point
# ---------------------------------------------------------------

def generate_all(exp_dir, baseline_name: str = "SGA_baseline") -> None:
    exp_dir = Path(exp_dir)
    runs_path = exp_dir / "runs.parquet"
    hist_path = exp_dir / "history.parquet"
    if not runs_path.exists():
        raise FileNotFoundError(f"{runs_path} missing — run experiments first")

    runs_df = pd.read_parquet(runs_path)
    plots_dir = _ensure(exp_dir / "plots")

    plot_boxplots(runs_df, plots_dir)
    plot_friedman_bars(runs_df, plots_dir)
    plot_cd_diagram(runs_df, plots_dir)
    plot_heatmaps(runs_df, plots_dir, baseline_name=baseline_name)

    if hist_path.exists():
        hist_df = pd.read_parquet(hist_path)
        plot_convergence(hist_df, plots_dir)
        plot_diversity(hist_df, plots_dir)
        plot_ecdf_performance(hist_df, plots_dir)
        plot_stagnation(hist_df, plots_dir)
        plot_phase(hist_df, plots_dir)
        plot_xpl_xpt(hist_df, plots_dir)
        plot_step_size(hist_df, plots_dir)
        plot_operator_success(hist_df, plots_dir)
        plot_fingerprint(hist_df, runs_df, plots_dir)
    else:
        print("[plots] history.parquet missing - skipping convergence/diversity/etc.")

    print(f"Wrote plots to {plots_dir}")
