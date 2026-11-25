"""
Variant registry: each entry is a `Spec` declaring exactly which operator
slot(s) the variant changes (or wraps) relative to the suite baseline.

Anything not declared here is inherited bit-identical from the suite
baseline built by `ga.build.baseline_from_config`. Per-variant scalar
divergence (e.g. a smaller population for one variant) must appear in
`Spec.overrides` where it is reviewable.

Encodings are tagged so the runner can filter variants that don't apply
to a given suite (e.g. real-coded mutation has no meaning on TSP).
"""

from ga import features as feat
from ga.build import Spec


# ---- Encoding constants for tagging ------------------------------------
REAL = ("real",)
PERM = ("permutation",)
ANY = ("real", "permutation")


VARIANTS = [
    # ================================================================
    # REAL-CODED (cec2022, cec2020_realworld)
    # ================================================================

    Spec("SGA+DE_mut",            mutate=feat.make_de_mutation(F=0.5)),
    Spec("SGA+SHADE_mut",         mutate=feat.make_shade_de()),
    # greedy_replace, archive_replace, metropolis_replace operate purely on
    # fitness comparisons + identity-preserving list() cloning - they don't
    # touch gene values, so they're safe on permutation encodings too.
    Spec("SGA+greedy_replace",    replace=feat.replace_greedy, encodings=ANY),
    Spec("SGA+archive_replace",   replace=feat.make_archive_replace(fraction=0.1),
         encodings=ANY),
    Spec("SGA+metropolis_replace",
         replace=feat.make_metropolis_replace(T0=1.0, T_end=1e-3, elitism_k=1),
         encodings=ANY),
    Spec("SGA+restart",
         diversity=feat.make_diversity_restart(stagnation_threshold=15,
                                               restart_fraction=0.2)),
    Spec("SGA+pher_minmax",
         mutate=feat.make_pheromone_minmax(rho=0.1, tau_min=0.1, tau_max=5.0)),
    Spec("SGA+immune_memory",
         diversity=feat.make_immune_memory_pool(k_inject=2,
                                                dist_thresh_frac=0.05)),
    Spec("SGA+net_suppress",
         diversity=feat.make_network_suppression(threshold_frac=0.02)),
    Spec("SGA+cuckoo_discovery",
         diversity=feat.make_cuckoo_discovery(pa=0.25)),
    Spec("SGA+abc_scout",         replace=feat.make_abc_scout()),
    Spec("SGA+bfo_chemotaxis",    mutate=feat.make_bfo_chemotaxis(step_size_frac=0.05)),
    Spec("SGA+krill_herd_mut",    mutate=feat.make_krill_herd()),
    Spec("SGA+sos_mut",           mutate=feat.mutate_sos),
    Spec("SGA+bbo_migration_cx",  crossover=feat.crossover_bbo),
    Spec("SGA+gsa_full_mut",      mutate=feat.make_gsa_full()),

    # ================================================================
    # PERMUTATION (TSP) - edge-aware operators (thesis sec:permutation). Each
    # preserves its named feature's metaphor — DE differential, SHADE
    # current-to-pbest, GSA gravity, BFO swim/tumble, krill food/herd,
    # SOS three phases, BBO migration - but rewrites the update in terms
    # of edges, swap sequences, and 2-opt moves.
    # ================================================================

    Spec("SGA+DE_mut",
         mutate=feat.make_de_perm(F=0.5),
         encodings=PERM),
    Spec("SGA+SHADE_mut",
         mutate=feat.make_shade_perm(),
         encodings=PERM),
    Spec("SGA+restart",
         diversity=feat.make_diversity_restart_perm(stagnation_threshold=15,
                                                    restart_fraction=0.2),
         encodings=PERM),
    # pher_minmax on TSP -> MMAS edge-pheromone (this *is* the permutation-
    # native form of pheromone min-max bounds; reuses the ACO machinery).
    Spec("SGA+pher_minmax",
         crossover=feat.make_aco_recombination(alpha=1.0, beta=3.0, q0=0.3,
                                               rho=0.1, cand_k=16,
                                               local_update=True, xi=0.1),
         refine=feat.make_aco_refine(rho=0.1, two_opt_top_k=3, cand_k=16),
         encodings=PERM),
    Spec("SGA+immune_memory",
         diversity=feat.make_immune_memory_pool_perm(k_inject=2,
                                                     dist_thresh_frac=0.05),
         encodings=PERM),
    Spec("SGA+net_suppress",
         diversity=feat.make_network_suppression_perm(threshold_frac=0.02),
         encodings=PERM),
    Spec("SGA+cuckoo_discovery",
         diversity=feat.make_cuckoo_discovery_perm(pa=0.25),
         encodings=PERM),
    Spec("SGA+abc_scout",
         replace=feat.make_abc_scout_perm(),
         encodings=PERM),
    Spec("SGA+bfo_chemotaxis",
         mutate=feat.make_bfo_chemotaxis_perm(N_s=4),
         encodings=PERM),
    Spec("SGA+krill_herd_mut",
         mutate=feat.make_krill_herd_perm(),
         encodings=PERM),
    Spec("SGA+sos_mut",
         mutate=feat.make_sos_perm(),
         encodings=PERM),
    Spec("SGA+bbo_migration_cx",
         crossover=feat.crossover_bbo_perm,
         encodings=PERM),
    Spec("SGA+gsa_full_mut",
         mutate=feat.make_gsa_full_perm(),
         encodings=PERM),
]


def variants_for_encoding(encoding: str):
    """Return only variants whose `encodings` tuple contains `encoding`."""
    return [s for s in VARIANTS if encoding in s.encodings]
