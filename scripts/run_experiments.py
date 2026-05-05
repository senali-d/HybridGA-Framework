"""
CLI entry point: build variants, run the CEC 2022 protocol, persist
per-batch results, statistical summaries, and plots.

USAGE:
    python scripts/run_experiments.py             # demo (small budget)
    python scripts/run_experiments.py --full      # full CEC 2022 protocol
    python scripts/run_experiments.py --no-plots  # skip plot generation
    python scripts/run_experiments.py --tag pilot1
"""

import argparse
import sys
from pathlib import Path

# Make src/ importable when running directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ga.core import FeatureConfig, default_select
from ga import features as feat
from benchmarks.cec2022 import run_protocol
from analysis.stats import summarize, write_summary
from analysis import plots as plot_mod
from persistence import ExperimentRecorder


def build_variants():
    """One baseline + a Tier-1 set of feature-injected variants."""
    variants = []

    variants.append(FeatureConfig(name="SGA_baseline"))

    variants.append(FeatureConfig(
        name="SGA+OBL_init",
        init_fn=feat.init_opposition_strict,
    ))

    variants.append(FeatureConfig(
        name="SGA+chaotic_init",
        init_fn=feat.init_chaotic,
    ))

    variants.append(FeatureConfig(
        name="SGA+Levy_mut",
        mutate_fn=feat.mutate_levy,
    ))

    variants.append(FeatureConfig(
        name="SGA+DE_mut",
        select_fn=feat.with_population_aware_select(default_select),
        mutate_fn=feat.make_de_mutation(F=0.5),
    ))

    variants.append(FeatureConfig(
        name="SGA+greedy_replace",
        replace_fn=feat.replace_greedy,
    ))

    variants.append(FeatureConfig(
        name="SGA+adaptive_mut",
        mutate_fn=feat.make_adaptive_mutate(base_indpb=0.1),
    ))

    variants.append(FeatureConfig(
        name="SGA+restart",
        diversity_fn=feat.make_diversity_restart(
            stagnation_threshold=15, restart_fraction=0.2),
    ))

    return variants


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="Run full CEC 2022 protocol (slow). Default: demo.")
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--out-root", type=str, default="results",
                        help="Root directory for per-batch experiment archives.")
    parser.add_argument("--tag", type=str, default=None,
                        help="Override timestamped folder name.")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-stats", action="store_true")
    parser.add_argument("--no-history", action="store_true",
                        help="Skip per-generation history recording (saves disk).")
    parser.add_argument("--baseline", type=str, default="SGA_baseline")
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

    recorder = ExperimentRecorder(
        out_root=args.out_root,
        args={**vars(args), "n_runs": n_runs, "max_fes": max_fes,
              "pop_size": pop_size},
        variants=[v.name for v in variants],
        tag=args.tag,
    )

    results = run_protocol(
        variants, dim=args.dim, n_runs=n_runs,
        max_fes=max_fes, pop_size=pop_size,
        recorder=recorder, record_history=not args.no_history,
    )

    summarize(results, baseline_name=args.baseline)
    recorder.flush()

    if not args.no_stats:
        import pandas as _pd
        _runs_df = _pd.read_parquet(recorder.exp_dir / "runs.parquet")
        _hist_path = recorder.exp_dir / "history.parquet"
        _hist_df = _pd.read_parquet(_hist_path) if _hist_path.exists() else None
        write_summary(results, recorder.exp_dir / "summary",
                      baseline_name=args.baseline,
                      history_df=_hist_df, runs_df=_runs_df)

    if not args.no_plots:
        try:
            plot_mod.generate_all(recorder.exp_dir, baseline_name=args.baseline)
        except Exception as e:
            print(f"[plots] generation failed: {e}")

    print(f"\nExperiment archive: {recorder.exp_dir}")


if __name__ == "__main__":
    main()
