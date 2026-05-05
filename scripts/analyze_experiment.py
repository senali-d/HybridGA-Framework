"""
Re-run statistical summaries and plot generation against an existing
experiment archive (created by run_experiments.py). Lets you tweak plots
or stats without re-running the GA.

USAGE:
    python scripts/analyze_experiment.py results/exp_2026-05-05_14-30-12_D10_demo/
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pandas as pd

from analysis.stats import write_summary
from analysis import plots as plot_mod


def runs_to_results(runs_df: pd.DataFrame) -> dict:
    out: dict[str, dict[str, list[float]]] = {}
    for v, vdf in runs_df.groupby("variant"):
        out[v] = {}
        for fn, fdf in vdf.groupby("function"):
            out[v][fn] = fdf.sort_values("seed")["error"].tolist()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("exp_dir", type=str, help="Path to an exp_<...> directory.")
    ap.add_argument("--baseline", type=str, default="SGA_baseline")
    ap.add_argument("--no-plots", action="store_true")
    ap.add_argument("--no-stats", action="store_true")
    args = ap.parse_args()

    exp_dir = Path(args.exp_dir)
    runs_df = pd.read_parquet(exp_dir / "runs.parquet")
    results = runs_to_results(runs_df)

    if not args.no_stats:
        hist_path = exp_dir / "history.parquet"
        hist_df = pd.read_parquet(hist_path) if hist_path.exists() else None
        write_summary(results, exp_dir / "summary", baseline_name=args.baseline,
                      history_df=hist_df, runs_df=runs_df)

    if not args.no_plots:
        plot_mod.generate_all(exp_dir, baseline_name=args.baseline)


if __name__ == "__main__":
    main()
