"""
CLI entry point: build variants, run the CEC 2022 protocol, print summary,
save raw results to results/results.json.

USAGE:
    python scripts/run_experiments.py             # demo (small budget)
    python scripts/run_experiments.py --full      # full CEC 2022 protocol
"""

import argparse
import json
from pathlib import Path

from ga.core import FeatureConfig, default_select
from ga import features as feat
from benchmarks.cec2022 import run_protocol
from analysis.stats import summarize


def build_variants():
    """One baseline + a Tier-1 set of feature-injected variants."""
    variants = []

    variants.append(FeatureConfig(name="SGA_baseline"))

    variants.append(FeatureConfig(
        name="SGA+OBL_init",
        init_fn=feat.init_chaotic,
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
    parser.add_argument("--out", type=str, default="results/results.json")
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

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {v: {f: list(map(float, errs)) for f, errs in funcs.items()}
                    for v, funcs in results.items()}
    with open(out_path, "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nRaw results saved to {out_path}")


if __name__ == "__main__":
    main()
