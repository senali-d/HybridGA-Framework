"""
Variant registry: each entry is a `Spec` declaring exactly which operator
slot(s) the variant changes (or wraps) relative to the suite baseline.

Anything not declared here is inherited bit-identical from the suite
baseline built by `ga.build.baseline_from_config`. Per-variant scalar
divergence (e.g. memetic SGA needing a smaller population) must appear in
`Spec.overrides` where it is reviewable.

Encodings are tagged so the runner can filter variants that don't apply
to a given suite (e.g. real-coded mutation has no meaning on TSP).
"""

from ga import features as feat
from ga.build import Spec
from ga.core import default_select


# ---- Encoding constants for tagging ------------------------------------
REAL = ("real",)
PERM = ("permutation",)
ANY  = ("real", "permutation", "discrete")


VARIANTS = [
    # --- Tier 1 ------------------------------------------------------
    Spec("SGA+OBL_init",          init=feat.init_opposition_strict),
    Spec("SGA+GWO_select",        select=feat.select_leader_hierarchy),
    Spec("SGA+chaotic_init",      init=feat.init_chaotic),
    Spec("SGA+Levy_mut",          mutate=feat.mutate_levy),
    # DE_mut composes a population-aware filter on top of the suite's
    # selection (binary tournament for engineering, k=3 for cec2022, ...).
    Spec("SGA+DE_mut",
         select_wrapper=feat.with_population_aware_select,
         mutate=feat.make_de_mutation(F=0.5)),
    Spec("SGA+greedy_replace",    replace=feat.replace_greedy),
    Spec("SGA+adaptive_mut",      mutate=feat.make_adaptive_mutate(base_indpb=0.1)),
    Spec("SGA+restart",
         diversity=feat.make_diversity_restart(stagnation_threshold=15,
                                               restart_fraction=0.2)),
    Spec("SGA+CLONALG_hyper",
         mutate=feat.make_inverse_hypermutate(rho=5.0, sigma_max=0.2)),
    Spec("SGA+clonal_expand",     select=feat.select_clonal_expansion),
    Spec("SGA+pher_evap",         mutate=feat.make_pheromone_evaporation(rho=0.1)),
    Spec("SGA+pher_deposit",      mutate=feat.make_pheromone_deposit(rho=0.1, Q=1.0)),
    Spec("SGA+pher_heuristic",
         mutate=feat.make_pheromone_heuristic(rho=0.1, alpha=1.0, beta=2.0)),
    Spec("SGA+pher_minmax",
         mutate=feat.make_pheromone_minmax(rho=0.1, tau_min=0.1, tau_max=5.0)),
    Spec("SGA+pher_eas",
         mutate=feat.make_pheromone_eas(rho=0.1, Q=1.0, e=5.0)),

    # --- Tier 2 ------------------------------------------------------
    Spec("SGA+chaotic_tent_init", init=feat.init_chaotic_tent),
    Spec("SGA+chaotic_sin_init",  init=feat.init_chaotic_sinusoidal),
    Spec("SGA+encircling_mut",    mutate=feat.mutate_encircling_prey),
    Spec("SGA+archive_replace",   replace=feat.make_archive_replace(fraction=0.1)),
    Spec("SGA+islands",           overrides={"n_islands": 4, "migration_interval": 10}),
    Spec("SGA+aging",             diversity=feat.make_aging_replace(tau=20)),
    Spec("SGA+receptor_edit",     diversity=feat.make_receptor_editing(fraction=0.1)),
    Spec("SGA+affinity_gauss",
         mutate=feat.make_affinity_gaussian(sigma_max=0.2, rho=2.0)),
    Spec("SGA+pher_asrank",       mutate=feat.make_pheromone_asrank(rho=0.1, w=6)),
    Spec("SGA+pher_edge_cx",      crossover=feat.make_pheromone_edge_crossover(rho=0.1)),
    Spec("SGA+pher_weighted_cx",
         crossover=feat.make_pheromone_weighted_crossover(rho=0.1)),
    Spec("SGA+pher_candlist",     select=feat.make_candidate_list_select(m=5, tournsize=2)),

    # --- Tier 3 ------------------------------------------------------
    Spec("SGA+spiral_woa_mut",    mutate=feat.mutate_spiral_woa),
    Spec("SGA+sine_cosine_mut",   mutate=feat.mutate_sine_cosine),
    Spec("SGA+poi_select",        select=feat.select_probability_of_improvement),
    Spec("SGA+pollination_mut",   mutate=feat.make_pollination_switch()),
    Spec("SGA+firefly_mut",       mutate=feat.make_firefly_attraction()),
    Spec("SGA+gauss_cauchy_mut",  mutate=feat.make_gauss_cauchy_mix()),
    Spec("SGA+fast_ga_mut",
         mutate=feat.make_fast_ga_mutate(beta=1.5, dist="levy", scale=0.01)),
    Spec("SGA+fast_ga_cauchy_mut",
         mutate=feat.make_fast_ga_mutate(beta=1.5, dist="cauchy", scale=0.01)),
    Spec("SGA+fast_ga_anneal_mut",
         mutate=feat.make_fast_ga_mutate(beta=1.5, dist="levy",
                                          scale=0.1, scale_end=0.001)),
    Spec("SGA+t_gated_mut",
         mutate=feat.make_t_ratio_gated([
             (0.5, feat.mutate_levy),
             (1.0, feat.mutate_encircling_prey),
         ])),
    Spec("SGA+periodic_obl",
         diversity=feat.make_periodic_obl(period=10, fraction=0.3)),
    Spec("SGA+neg_select",
         mutate=feat.make_negative_selection(threshold=0.05, push=0.1)),
    Spec("SGA+immune_memory",
         diversity=feat.make_immune_memory_pool(k_inject=2,
                                                dist_thresh_frac=0.05)),
    Spec("SGA+adaptive_pop",
         replace=feat.make_adaptive_resize(min_n=20, max_n=80,
                                            target_div_frac=0.05)),
    Spec("SGA+multiparent_cx",    crossover=feat.crossover_multiparent),
    Spec("SGA+net_suppress",
         diversity=feat.make_network_suppression(threshold_frac=0.02)),
    Spec("SGA+cauchy_elite",
         mutate=feat.make_cauchy_hypermutate_elite(elite_frac=0.2, scale=0.1)),
    Spec("SGA+pher_guided_mut",
         mutate=feat.make_pheromone_guided_mutation(rho=0.1, alpha=1.0, beta=2.0)),

    # --- Tier 4 ------------------------------------------------------
    Spec("SGA+lhs_init",          init=feat.init_latin_hypercube),
    Spec("SGA+sobol_init",        init=feat.init_sobol),
    Spec("SGA+halton_init",       init=feat.init_halton),
    Spec("SGA+levy_init",         init=feat.init_levy_distributed),
    Spec("SGA+adaptive_tourney",  select=feat.select_adaptive_tournament),
    Spec("SGA+archive_mating",    select=feat.select_best_archive_mating),
    Spec("SGA+immigrant",         diversity=feat.make_random_immigrant(fraction=0.1)),
    Spec("SGA+crowding_select",   select=feat.select_distance_crowding),
    Spec("SGA+social_rank_select", select=feat.select_social_rank_weighted),
    Spec("SGA+mfo_spiral_mut",    mutate=feat.mutate_mfo_spiral),
    Spec("SGA+bubble_net_mut",    mutate=feat.mutate_bubble_net),
    Spec("SGA+hho_besiege_mut",   mutate=feat.mutate_hho_besiege),
    Spec("SGA+brownian_mut",      mutate=feat.make_brownian()),
    Spec("SGA+eddy_fad_mut",      mutate=feat.make_eddy_fad(p_jump=0.2)),
    Spec("SGA+bat_mut",           mutate=feat.mutate_bat_echolocation),
    Spec("SGA+gsa_mut",           mutate=feat.mutate_gsa_attraction),
    Spec("SGA+eo_pool_mut",       mutate=feat.mutate_equilibrium_pool),
    Spec("SGA+sma_mut",           mutate=feat.mutate_sma_oscillation),
    Spec("SGA+aquila_mut",        mutate=feat.mutate_aquila_4mode),
    Spec("SGA+rsa_mut",           mutate=feat.mutate_rsa_walk_swim),
    Spec("SGA+noa_cache_mut",     mutate=feat.mutate_noa_cache_recover),
    Spec("SGA+salp_chain_mut",    mutate=feat.mutate_salp_chain),
    Spec("SGA+levy_xover_hybrid_mut", mutate=feat.mutate_levy_crossover_hybrid),
    Spec("SGA+oppositional_mut",  mutate=feat.make_oppositional_mutate()),
    Spec("SGA+crowding_replace",  replace=feat.replace_crowding),
    Spec("SGA+nlin_decay_mut",
         mutate=feat.make_nonlinear_decay_mutate(base_indpb=0.1, mode="cosine")),
    Spec("SGA+SHADE_mut",         mutate=feat.make_shade_de()),
    Spec("SGA+hho_phase_mut",     mutate=feat.make_hho_phase_switch()),
    Spec("SGA+rank_step_mut",     mutate=feat.mutate_rank_step),
    Spec("SGA+chaotic_perturb_mut",
         mutate=feat.make_chaotic_perturbation(scale=0.05)),
    Spec("SGA+fad_jump",          diversity=feat.make_fad_jump(period=10, fraction=0.2)),
    Spec("SGA+qpso_mut",          mutate=feat.mutate_quantum_qpso),
    Spec("SGA+danger_signal",
         mutate=feat.make_danger_signal(stag_threshold=10, boost=5.0)),
    Spec("SGA+pso_velocity_mut",
         mutate=feat.make_pso_velocity_mutate(w=0.7, c1=1.5, c2=1.5)),
    Spec("SGA+abc_employed_mut",  mutate=feat.make_abc_mutate(phase="employed")),
    Spec("SGA+abc_onlooker_mut",  mutate=feat.make_abc_mutate(phase="onlooker")),
    Spec("SGA+abc_mixed_mut",     mutate=feat.make_abc_mutate(phase="mixed")),
    Spec("SGA+pso_refine",
         refine=feat.make_sga_pso_refine(top_k=5, n_iters=3, w=0.7, c1=1.5,
                                          c2=1.5, every_n_gens=5)),
    Spec("SGA+metropolis_replace",
         replace=feat.make_metropolis_replace(T0=1.0, T_end=1e-3,
                                               schedule="exp", elitism_k=1)),
    Spec("SGA+cuckoo_discovery",
         diversity=feat.make_cuckoo_discovery(pa=0.25)),
    Spec("SGA+abc_scout",         replace=feat.make_abc_scout()),
    Spec("SGA+bfo_chemotaxis",    mutate=feat.make_bfo_chemotaxis(step_size_frac=0.05)),
    Spec("SGA+krill_herd_mut",    mutate=feat.make_krill_herd()),
    Spec("SGA+sos_mut",           mutate=feat.mutate_sos),
    Spec("SGA+bbo_migration_cx",  crossover=feat.crossover_bbo),
    Spec("SGA+gsa_full_mut",      mutate=feat.make_gsa_full()),

    # --- Permutation / TSP --------------------------------------------
    # GACO: a *full* GA+ACO hybrid for combinatorial encodings. Replaces the
    # suite's OX crossover with Ant-Colony-System edge-recombination
    # construction (pheromone tau + heuristic eta, pseudo-random-proportional
    # rule, k-NN candidate lists, online local pheromone update), and adds a
    # per-generation refine that does 2-opt local search on the elite (charged
    # against MaxFES) plus a global pheromone update. Mutation (swap/inversion)
    # is inherited from the suite baseline. Three deposit schemes are wired as
    # separate variants: MAX-MIN (default), rank-based, and elitist Ant System.
    Spec("SGA+GACO_hybrid",
         crossover=feat.make_aco_recombination(alpha=1.0, beta=3.0, q0=0.3,
                                               rho=0.1, cand_k=16,
                                               local_update=True, xi=0.1),
         refine=feat.make_aco_refine(rho=0.1, two_opt_top_k=3,
                                     deposit_scheme="mmas", cand_k=16),
         encodings=PERM),
    Spec("SGA+GACO_asrank",
         crossover=feat.make_aco_recombination(alpha=1.0, beta=3.0, q0=0.3,
                                               rho=0.1, cand_k=16,
                                               local_update=True, xi=0.1),
         refine=feat.make_aco_refine(rho=0.1, two_opt_top_k=3,
                                     deposit_scheme="asrank", asrank_w=6,
                                     cand_k=16),
         encodings=PERM),
    Spec("SGA+GACO_eas",
         crossover=feat.make_aco_recombination(alpha=1.0, beta=3.0, q0=0.3,
                                               rho=0.1, cand_k=16,
                                               local_update=True, xi=0.1),
         refine=feat.make_aco_refine(rho=0.1, two_opt_top_k=3,
                                     deposit_scheme="eas", cand_k=16),
         encodings=PERM),
    # The TSP suite baseline now *is* EAX (Edge Assembly Crossover), so an
    # explicit "+EAX" variant would just duplicate it. Keep an OX variant so
    # the pre-EAX crossover stays reachable as an ablation point.
    Spec("SGA+OX_crossover", crossover=feat.crossover_ox, encodings=PERM),
    # EAX + ACO, *additive*: keeps the suite's EAX crossover untouched and
    # layers on (a) ACO-guided segment-rebuild mutation and (b) the pheromone
    # + 2-opt refine. Nothing is replaced except the (free) mutate slot and
    # the (empty) refine slot — so it gets EAX's recombination *and* the full
    # GACO machinery (edge-pheromone learning + memetic 2-opt).
    Spec("SGA+EAX_ACO",
         mutate=feat.make_aco_guided_mutation(seg_frac=0.2, seg_prob=0.5,
                                              alpha=1.0, beta=3.0, q0=0.3,
                                              rho=0.1, cand_k=16),
         refine=feat.make_aco_refine(rho=0.1, two_opt_top_k=3,
                                     deposit_scheme="mmas", cand_k=16),
         encodings=PERM),
]


def variants_for_encoding(encoding: str):
    """Return only variants whose `encodings` tuple contains `encoding`."""
    return [s for s in VARIANTS if encoding in s.encodings]
