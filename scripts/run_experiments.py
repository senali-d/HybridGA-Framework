"""
CLI entry point: build variants, run the CEC 2022 protocol, persist
per-batch results, statistical summaries, and plots.

USAGE:
    python scripts/run_experiments.py             # demo (small budget)
    python scripts/run_experiments.py --full      # full CEC 2022 protocol
    python scripts/run_experiments.py --no-plots  # skip plot generation
    python scripts/run_experiments.py --tag pilot1
    python scripts/run_experiments.py --variants pher,levy   # subset only
    python scripts/run_experiments.py --pop-size 40 --max-gen 200 --n-runs 10
    python scripts/run_experiments.py --dim 20 --max-fes 50000 --workers 8
"""

import argparse
import sys
from pathlib import Path

# Make src/ importable when running directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ga.core import FeatureConfig, default_select
from ga import features as feat
from benchmarks.suites import run_protocol, run_separable_protocol
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
        name="SGA+GWO_select",
        select_fn=feat.select_leader_hierarchy,
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

    variants.append(FeatureConfig(
        name="SGA+CLONALG_hyper",
        mutate_fn=feat.make_inverse_hypermutate(rho=5.0, sigma_max=0.2),
    ))

    variants.append(FeatureConfig(
        name="SGA+clonal_expand", select_fn=feat.select_clonal_expansion,
    ))

    variants.append(FeatureConfig(
        name="SGA+pher_evap",
        mutate_fn=feat.make_pheromone_evaporation(rho=0.1),
    ))

    variants.append(FeatureConfig(
        name="SGA+pher_deposit",
        mutate_fn=feat.make_pheromone_deposit(rho=0.1, Q=1.0),
    ))

    variants.append(FeatureConfig(
        name="SGA+pher_heuristic",
        mutate_fn=feat.make_pheromone_heuristic(rho=0.1, alpha=1.0, beta=2.0),
    ))

    variants.append(FeatureConfig(
        name="SGA+pher_minmax",
        mutate_fn=feat.make_pheromone_minmax(rho=0.1, tau_min=0.1, tau_max=5.0),
    ))

    variants.append(FeatureConfig(
        name="SGA+pher_eas",
        mutate_fn=feat.make_pheromone_eas(rho=0.1, Q=1.0, e=5.0),
    ))

    # ---------- Tier 2 (remaining) ----------
    variants.append(FeatureConfig(
        name="SGA+chaotic_tent_init", init_fn=feat.init_chaotic_tent,
    ))
    variants.append(FeatureConfig(
        name="SGA+chaotic_sin_init", init_fn=feat.init_chaotic_sinusoidal,
    ))
    variants.append(FeatureConfig(
        name="SGA+encircling_mut", mutate_fn=feat.mutate_encircling_prey,
    ))
    variants.append(FeatureConfig(
        name="SGA+archive_replace",
        replace_fn=feat.make_archive_replace(fraction=0.1),
    ))
    variants.append(FeatureConfig(
        name="SGA+islands", n_islands=4, migration_interval=10,
    ))

    variants.append(FeatureConfig(
        name="SGA+aging",
        diversity_fn=feat.make_aging_replace(tau=20),
    ))

    variants.append(FeatureConfig(
        name="SGA+receptor_edit",
        diversity_fn=feat.make_receptor_editing(fraction=0.1),
    ))

    variants.append(FeatureConfig(
        name="SGA+affinity_gauss",
        mutate_fn=feat.make_affinity_gaussian(sigma_max=0.2, rho=2.0),
    ))

    variants.append(FeatureConfig(
        name="SGA+pher_asrank",
        mutate_fn=feat.make_pheromone_asrank(rho=0.1, w=6),
    ))

    variants.append(FeatureConfig(
        name="SGA+pher_edge_cx",
        crossover_fn=feat.make_pheromone_edge_crossover(rho=0.1),
    ))

    variants.append(FeatureConfig(
        name="SGA+pher_weighted_cx",
        crossover_fn=feat.make_pheromone_weighted_crossover(rho=0.1),
    ))

    variants.append(FeatureConfig(
        name="SGA+pher_candlist",
        select_fn=feat.make_candidate_list_select(m=5, tournsize=2),
    ))

    # ---------- Tier 3 ----------
    variants.append(FeatureConfig(
        name="SGA+spiral_woa_mut", mutate_fn=feat.mutate_spiral_woa,
    ))
    variants.append(FeatureConfig(
        name="SGA+sine_cosine_mut", mutate_fn=feat.mutate_sine_cosine,
    ))
    variants.append(FeatureConfig(
        name="SGA+poi_select", select_fn=feat.select_probability_of_improvement,
    ))
    variants.append(FeatureConfig(
        name="SGA+pollination_mut", mutate_fn=feat.make_pollination_switch(),
    ))
    variants.append(FeatureConfig(
        name="SGA+firefly_mut", mutate_fn=feat.make_firefly_attraction(),
    ))
    variants.append(FeatureConfig(
        name="SGA+gauss_cauchy_mut", mutate_fn=feat.make_gauss_cauchy_mix(),
    ))
    variants.append(FeatureConfig(
        name="SGA+t_gated_mut",
        mutate_fn=feat.make_t_ratio_gated([
            (0.5, feat.mutate_levy),
            (1.0, feat.mutate_encircling_prey),
        ]),
    ))
    variants.append(FeatureConfig(
        name="SGA+periodic_obl",
        diversity_fn=feat.make_periodic_obl(period=10, fraction=0.3),
    ))

    variants.append(FeatureConfig(
        name="SGA+neg_select",
        mutate_fn=feat.make_negative_selection(threshold=0.05, push=0.1),
    ))

    variants.append(FeatureConfig(
        name="SGA+immune_memory",
        diversity_fn=feat.make_immune_memory_pool(k_inject=2, dist_thresh_frac=0.05),
    ))

    variants.append(FeatureConfig(
        name="SGA+adaptive_pop",
        replace_fn=feat.make_adaptive_resize(min_n=20, max_n=80, target_div_frac=0.05),
    ))

    variants.append(FeatureConfig(
        name="SGA+multiparent_cx",
        crossover_fn=feat.crossover_multiparent,
    ))

    variants.append(FeatureConfig(
        name="SGA+net_suppress",
        diversity_fn=feat.make_network_suppression(threshold_frac=0.02),
    ))

    variants.append(FeatureConfig(
        name="SGA+cauchy_elite",
        mutate_fn=feat.make_cauchy_hypermutate_elite(elite_frac=0.2, scale=0.1),
    ))

    variants.append(FeatureConfig(
        name="SGA+pher_guided_mut",
        mutate_fn=feat.make_pheromone_guided_mutation(
            rho=0.1, alpha=1.0, beta=2.0),
    ))

    # ---------- Tier 4 ----------
    # Initialization
    variants.append(FeatureConfig(
        name="SGA+lhs_init", init_fn=feat.init_latin_hypercube,
    ))
    variants.append(FeatureConfig(
        name="SGA+sobol_init", init_fn=feat.init_sobol,
    ))
    variants.append(FeatureConfig(
        name="SGA+halton_init", init_fn=feat.init_halton,
    ))
    variants.append(FeatureConfig(
        name="SGA+levy_init", init_fn=feat.init_levy_distributed,
    ))
    # Selection
    variants.append(FeatureConfig(
        name="SGA+adaptive_tourney", select_fn=feat.select_adaptive_tournament,
    ))
    variants.append(FeatureConfig(
        name="SGA+archive_mating", select_fn=feat.select_best_archive_mating,
    ))
    variants.append(FeatureConfig(
        name="SGA+immigrant",
        diversity_fn=feat.make_random_immigrant(fraction=0.1),
    ))
    variants.append(FeatureConfig(
        name="SGA+crowding_select", select_fn=feat.select_distance_crowding,
    ))
    variants.append(FeatureConfig(
        name="SGA+social_rank_select", select_fn=feat.select_social_rank_weighted,
    ))
    # Variation
    variants.append(FeatureConfig(
        name="SGA+mfo_spiral_mut", mutate_fn=feat.mutate_mfo_spiral,
    ))
    variants.append(FeatureConfig(
        name="SGA+bubble_net_mut", mutate_fn=feat.mutate_bubble_net,
    ))
    variants.append(FeatureConfig(
        name="SGA+hho_besiege_mut", mutate_fn=feat.mutate_hho_besiege,
    ))
    variants.append(FeatureConfig(
        name="SGA+brownian_mut", mutate_fn=feat.make_brownian(),
    ))
    variants.append(FeatureConfig(
        name="SGA+eddy_fad_mut", mutate_fn=feat.make_eddy_fad(p_jump=0.2),
    ))
    variants.append(FeatureConfig(
        name="SGA+bat_mut", mutate_fn=feat.mutate_bat_echolocation,
    ))
    variants.append(FeatureConfig(
        name="SGA+gsa_mut", mutate_fn=feat.mutate_gsa_attraction,
    ))
    variants.append(FeatureConfig(
        name="SGA+eo_pool_mut", mutate_fn=feat.mutate_equilibrium_pool,
    ))
    variants.append(FeatureConfig(
        name="SGA+sma_mut", mutate_fn=feat.mutate_sma_oscillation,
    ))
    variants.append(FeatureConfig(
        name="SGA+aquila_mut", mutate_fn=feat.mutate_aquila_4mode,
    ))
    variants.append(FeatureConfig(
        name="SGA+rsa_mut", mutate_fn=feat.mutate_rsa_walk_swim,
    ))
    variants.append(FeatureConfig(
        name="SGA+noa_cache_mut", mutate_fn=feat.mutate_noa_cache_recover,
    ))
    variants.append(FeatureConfig(
        name="SGA+salp_chain_mut", mutate_fn=feat.mutate_salp_chain,
    ))
    variants.append(FeatureConfig(
        name="SGA+levy_xover_hybrid_mut",
        mutate_fn=feat.mutate_levy_crossover_hybrid,
    ))
    variants.append(FeatureConfig(
        name="SGA+oppositional_mut",
        mutate_fn=feat.make_oppositional_mutate(),
    ))
    # Replacement
    variants.append(FeatureConfig(
        name="SGA+crowding_replace", replace_fn=feat.replace_crowding,
    ))
    # Parameter control
    variants.append(FeatureConfig(
        name="SGA+nlin_decay_mut",
        mutate_fn=feat.make_nonlinear_decay_mutate(base_indpb=0.1, mode="cosine"),
    ))
    variants.append(FeatureConfig(
        name="SGA+SHADE_mut", mutate_fn=feat.make_shade_de(),
    ))
    variants.append(FeatureConfig(
        name="SGA+hho_phase_mut", mutate_fn=feat.make_hho_phase_switch(),
    ))
    variants.append(FeatureConfig(
        name="SGA+rank_step_mut", mutate_fn=feat.mutate_rank_step,
    ))
    # Diversity / global search
    variants.append(FeatureConfig(
        name="SGA+chaotic_perturb_mut",
        mutate_fn=feat.make_chaotic_perturbation(scale=0.05),
    ))
    variants.append(FeatureConfig(
        name="SGA+fad_jump",
        diversity_fn=feat.make_fad_jump(period=10, fraction=0.2),
    ))
    variants.append(FeatureConfig(
        name="SGA+qpso_mut", mutate_fn=feat.mutate_quantum_qpso,
    ))

    variants.append(FeatureConfig(
        name="SGA+danger_signal",
        mutate_fn=feat.make_danger_signal(stag_threshold=10, boost=5.0),
    ))

    variants.append(FeatureConfig(
        name="SGA+pso_velocity_mut",
        mutate_fn=feat.make_pso_velocity_mutate(w=0.7, c1=1.5, c2=1.5),
    ))

    variants.append(FeatureConfig(
        name="SGA+abc_employed_mut",
        mutate_fn=feat.make_abc_mutate(phase="employed"),
    ))

    variants.append(FeatureConfig(
        name="SGA+abc_onlooker_mut",
        mutate_fn=feat.make_abc_mutate(phase="onlooker"),
    ))
    
    variants.append(FeatureConfig(
        name="SGA+abc_mixed_mut",
        mutate_fn=feat.make_abc_mutate(phase="mixed"),
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
    parser.add_argument("--variants", type=str, default=None,
                        help="Comma-separated substrings; only variants whose "
                             "name contains any of these are run. Baseline is "
                             "always included. Example: --variants pher,levy")
    parser.add_argument("--exact", action="store_true",
                        help="Treat --variants entries as exact names "
                             "(matching either v.name or 'SGA+'+entry) "
                             "instead of substrings. Useful when one needle "
                             "would otherwise match a longer variant name "
                             "(e.g. DE_mut vs SHADE_mut).")
    parser.add_argument("--workers", type=int, default=1,
                        help="Process-pool workers over (variant, function, "
                             "seed) tuples. Default 1 (sequential). Uses fork "
                             "start method, so closures inside variants are "
                             "fine. Results are seed-deterministic.")
    parser.add_argument("--pop-size", type=int, default=None,
                        help="Population size. Default: 30 (demo) / 50 (full).")
    parser.add_argument("--max-fes", type=int, default=None,
                        help="FE budget per run. Default: 5000 (demo); for "
                             "--full, 200k at D<=10 or 1M at D>10.")
    parser.add_argument("--max-gen", type=int, default=None,
                        help="Cap by generations instead of FEs. Sets "
                             "max_fes = max_gen * pop_size. Overrides --max-fes.")
    parser.add_argument("--suite", type=str, default="cec2022",
                        choices=["cec2022", "separable"],
                        help="Benchmark suite. 'cec2022' (default) or "
                             "'separable' (Sphere/Rastrigin/Schwefel/"
                             "SumOfDifferentPowers/Step, any dim).")
    parser.add_argument("--functions", type=str, default=None,
                        help="Comma-separated function names to run "
                             "(e.g. F1,F3,F7 for cec2022; Sphere,Step for "
                             "separable). Default: all in the suite.")
    parser.add_argument("--n-runs", type=int, default=None,
                        help="Independent seeds per (variant, function). "
                             "Default: 5 (demo) / 30 (full).")
    args = parser.parse_args()

    # Mode defaults; explicit flags override.
    if args.full:
        n_runs = 30
        max_fes = 200_000 if args.dim <= 10 else 1_000_000
        pop_size = 50
        mode = "FULL"
    else:
        n_runs = 5
        max_fes = 5_000
        pop_size = 30
        mode = "DEMO"

    if args.pop_size is not None:
        pop_size = args.pop_size
    if args.max_fes is not None:
        max_fes = args.max_fes
    if args.max_gen is not None:
        max_fes = args.max_gen * pop_size
    if args.n_runs is not None:
        n_runs = args.n_runs

    print(f"Running {mode} protocol: D={args.dim}, pop={pop_size}, "
          f"MaxFES={max_fes} (~{max_fes // pop_size} gens), runs={n_runs}")
    if mode == "DEMO" and not any([args.pop_size, args.max_fes,
                                    args.max_gen, args.n_runs]):
        print("(Use --full for the real CEC 2022 budget, or override with "
              "--pop-size / --max-fes / --max-gen / --n-runs.)")

    variants = build_variants()
    if args.variants:
        needles = [s.strip() for s in args.variants.split(",") if s.strip()]
        if args.exact:
            wanted = set(needles) | {f"SGA+{n}" for n in needles}
            variants = [
                v for v in variants
                if v.name == args.baseline or v.name in wanted
            ]
        else:
            variants = [
                v for v in variants
                if v.name == args.baseline or any(n in v.name for n in needles)
            ]
        if not any(v.name == args.baseline for v in variants):
            print(f"WARNING: baseline {args.baseline!r} not in filtered set; "
                  "Wilcoxon vs baseline will be skipped.")
    print(f"Variants to test ({len(variants)}): {[v.name for v in variants]}\n")

    recorder = ExperimentRecorder(
        out_root=args.out_root,
        args={**vars(args), "n_runs": n_runs, "max_fes": max_fes,
              "pop_size": pop_size},
        variants=[v.name for v in variants],
        tag=args.tag,
    )

    funcs_filter = (
        [s.strip() for s in args.functions.split(",") if s.strip()]
        if args.functions else None
    )

    protocol = run_protocol if args.suite == "cec2022" else run_separable_protocol
    results = protocol(
        variants, dim=args.dim, n_runs=n_runs,
        max_fes=max_fes, pop_size=pop_size,
        recorder=recorder, record_history=not args.no_history,
        workers=args.workers, functions=funcs_filter,
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
