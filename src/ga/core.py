"""
sga_core.py
-----------
A clean, parameterized Simple Genetic Algorithm built on DEAP primitives.
Each pipeline stage (init, select, crossover, mutate, replace) is a
*plug point* that can be swapped via a `FeatureConfig`.

This is the backbone for the CEC 2022 study on the impact of nature-inspired
algorithm features on the SGA's behavior.
"""

import random
from collections import deque
import numpy as np
from deap import base, creator, tools

# -------- DEAP type setup (do this once at module load) --------
# Minimization: weights = -1.0 because CEC 2022 reports error = f - f*,
# and we want lower error.
if not hasattr(creator, "FitnessMin"):
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
if not hasattr(creator, "Individual"):
    creator.create("Individual", list, fitness=creator.FitnessMin)


class FeatureConfig:
    """
    Holds the swappable functions for one experimental variant.
    Pass any subset of these to override the SGA defaults.

    All functions receive (and may use) a context dict that carries
    generation index, max generations, bounds, rng, etc.

    Mutation rate convention (Mühlenbein / DEAP eaSimple):
        mut_prob = 1.0 (per-individual gate) is paired with indpb = 1/dim
        (per-gene rate, set inside `default_mutate`). Expected genes mutated
        per offspring = 1.0. Tune `indpb` to change mutation intensity; do
        NOT also tune `mut_prob` — two multiplicative knobs invite
        accidental under-mutation.
    """
    def __init__(
        self,
        name="SGA_baseline",
        init_fn=None,           # (n, dim, lb, ub, ctx) -> list of vectors
        select_fn=None,         # (population, k, ctx) -> list of selected
        crossover_fn=None,      # (parent1, parent2, ctx) -> (child1, child2)
        mutate_fn=None,         # (individual, ctx) -> mutated individual
        replace_fn=None,        # (parents, offspring, ctx) -> new population
        diversity_fn=None,      # (population, ctx) -> population (optional injection)
        refine_fn=None,         # (population, ctx) -> population (post-replacement local refinement,
                                # e.g. nested PSO sub-loop on top-k for SGA-style hybrid)
        cx_prob=0.9,
        mut_prob=1.0,
        elitism_k=1,
        n_islands=1,            # F46: number of sub-populations (1 = single panmictic pop)
        migration_interval=10,  # F46: migrate every G generations
        archive_size=5,         # F37/F31/F09: capacity of historical-best archive
        bound_handling="clip",  # how the core treats out-of-bounds offspring:
                                # "clip" | "reflect" | "reinit"
    ):
        self.name = name
        self.init_fn = init_fn
        self.select_fn = select_fn
        self.crossover_fn = crossover_fn
        self.mutate_fn = mutate_fn
        self.replace_fn = replace_fn
        self.diversity_fn = diversity_fn
        self.refine_fn = refine_fn
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob
        self.elitism_k = elitism_k
        self.n_islands = n_islands
        self.migration_interval = migration_interval
        self.archive_size = archive_size
        if bound_handling not in ("clip", "reflect", "reinit", "passthrough"):
            raise ValueError(
                f"bound_handling must be 'clip', 'reflect', 'reinit', or "
                f"'passthrough'; got {bound_handling!r}"
            )
        self.bound_handling = bound_handling


# ===================== DEFAULT (BASELINE) OPERATORS =====================

def default_init(n, dim, lb, ub, ctx):
    """Uniform random initialization — the SGA baseline."""
    rng = ctx["rng"]
    return [rng.uniform(lb, ub).tolist() for _ in range(n)]


def default_select(population, k, ctx):
    """Tournament selection of size 3 — DEAP's tools.selTournament."""
    return tools.selTournament(population, k, tournsize=3)


def default_crossover(p1, p2, ctx):
    """Simulated Binary Crossover (SBX) — standard for real-coded GA."""
    c1, c2 = tools.cxSimulatedBinaryBounded(
        list(p1), list(p2), eta=20.0,
        low=ctx["lb"].tolist(), up=ctx["ub"].tolist(),
    )
    return c1, c2


def default_mutate(ind, ctx):
    """Polynomial bounded mutation — standard real-coded GA mutation."""
    mutated, = tools.mutPolynomialBounded(
        list(ind), eta=20.0,
        low=ctx["lb"].tolist(), up=ctx["ub"].tolist(),
        indpb=1.0 / ctx["dim"],
    )
    return mutated


def generational_elitism(parents, offspring, k):
    """Generational replacement with k-elitism: keep best k parents,
    fill the rest with the best offspring. Reusable from feature variants."""
    parents_sorted = sorted(parents, key=lambda x: x.fitness.values[0])
    elites = parents_sorted[:k]
    offspring_sorted = sorted(offspring, key=lambda x: x.fitness.values[0])
    return elites + offspring_sorted[: len(parents) - k]


def default_replace(parents, offspring, ctx):
    """Generational replacement with k-elitism (default operator)."""
    return generational_elitism(parents, offspring, ctx["elitism_k"])


# ===================== OPERATOR FACTORIES (config-driven) =====================
#
# Each factory accepts a `spec` dict (taken straight from a suite JSON) and
# returns a closure with the SGA core's expected signature. The `kind` field
# selects the implementation; remaining fields are operator parameters.
#
# Adding a new operator:
#   1. Implement the closure inline (or import from `ga/features.py`).
#   2. Add a branch under the relevant `make_*` factory.
#   3. Reference its `kind` from a suite JSON. No other code changes needed.

def _resolve_pm_per_gene(spec_value, dim):
    """`pm_per_gene` may be the literal string '1/D', a float, or absent
    (defaults to 1/D)."""
    if spec_value is None or spec_value == "1/D":
        return 1.0 / dim
    return float(spec_value)


def make_init(spec):
    """Build an initialization closure from a spec.

    Defaults to uniform real-valued init when spec is None or
    `kind == "uniform"`; `kind == "random_permutation"` produces random
    permutations of {0..dim-1} for combinatorial encodings (TSP).
    """
    if spec is None:
        return None  # let the SGA core fall back to default_init
    kind = spec.get("kind", "uniform")
    if kind == "uniform":
        return None
    if kind == "random_permutation":
        def _init(n, dim, lb, ub, ctx):
            rng = ctx["rng"]
            return [list(rng.permutation(dim)) for _ in range(n)]
        _init.__name__ = "random_permutation"
        return _init
    raise ValueError(f"unknown init kind: {kind!r}")


def make_select(spec):
    """Build a selection closure from a spec like {'kind':'tournament','k':2}."""
    kind = spec.get("kind", "tournament")
    if kind == "tournament":
        k = int(spec.get("k", 3))
        def _select(population, n, ctx):
            return tools.selTournament(population, n, tournsize=k)
        _select.__name__ = f"tournament_k{k}"
        return _select
    raise ValueError(f"unknown selection kind: {kind!r}")


def make_crossover(spec):
    """Build a crossover closure from a spec like
    {'kind':'sbx','eta':20,'pc':0.9}.  `pc` is read by the core loop via
    FeatureConfig.cx_prob, not the closure itself."""
    kind = spec.get("kind", "sbx")
    if kind == "sbx":
        eta = float(spec.get("eta", 20.0))
        def _cx(p1, p2, ctx):
            c1, c2 = tools.cxSimulatedBinaryBounded(
                list(p1), list(p2), eta=eta,
                low=ctx["lb"].tolist(), up=ctx["ub"].tolist(),
            )
            return c1, c2
        _cx.__name__ = f"sbx_eta{int(eta)}"
        return _cx
    if kind == "ox":
        # Order Crossover for permutation encodings (TSP). Operates on
        # Python lists treated as permutations of {0..len-1}.
        def _cx(p1, p2, ctx):
            c1, c2 = tools.cxOrdered(list(p1), list(p2))
            return c1, c2
        _cx.__name__ = "ox"
        return _cx
    raise ValueError(f"unknown crossover kind: {kind!r}")


def make_mutation(spec, dim):
    """Build a mutation closure. `dim` is needed to resolve `1/D` rates."""
    kind = spec.get("kind", "poly")
    if kind == "poly":
        eta = float(spec.get("eta", 20.0))
        indpb = _resolve_pm_per_gene(spec.get("pm_per_gene"), dim)
        def _mut(ind, ctx):
            mutated, = tools.mutPolynomialBounded(
                list(ind), eta=eta,
                low=ctx["lb"].tolist(), up=ctx["ub"].tolist(),
                indpb=indpb,
            )
            return mutated
        _mut.__name__ = f"poly_eta{int(eta)}_indpb{indpb:.4g}"
        return _mut
    if kind in ("swap", "inversion"):
        # Permutation mutation. `pm` is the per-position swap probability
        # (kind="swap") or the whole-vector inversion probability
        # (kind="inversion"). An optional `alt` names a second permutation
        # operator; when set, each call picks `kind` or `alt` with equal
        # probability, so a single spec can mix swap and inversion moves.
        pm = float(spec.get("pm", 1.0 / dim))
        alt = spec.get("alt")
        _PERM_OPS = ("swap", "inversion")
        if alt is not None and alt not in _PERM_OPS:
            raise ValueError(f"unknown alt mutation kind: {alt!r}")

        def _apply_swap(seq, rng):
            for i in range(len(seq)):
                if rng.random() < pm:
                    j = int(rng.integers(0, len(seq)))
                    seq[i], seq[j] = seq[j], seq[i]
            return seq

        def _apply_inversion(seq, rng):
            if rng.random() < pm:
                i, j = sorted(rng.integers(0, len(seq), size=2).tolist())
                seq[i:j + 1] = seq[i:j + 1][::-1]
            return seq

        _OP_FNS = {"swap": _apply_swap, "inversion": _apply_inversion}
        primary_fn = _OP_FNS[kind]
        alt_fn = _OP_FNS[alt] if alt is not None else None

        def _mut(ind, ctx):
            seq = list(ind)
            rng = ctx["rng"]
            op = primary_fn
            if alt_fn is not None and rng.random() < 0.5:
                op = alt_fn
            return op(seq, rng)

        _mut.__name__ = (
            f"{kind}_or_{alt}_pm{pm:.4g}" if alt is not None
            else f"{kind}_pm{pm:.4g}"
        )
        return _mut
    raise ValueError(f"unknown mutation kind: {kind!r}")


def make_replace(spec):
    """Build a replacement closure from a spec like {'k':1}.

    Note: the SGA core always populates `ctx['elitism_k']` from
    `FeatureConfig.elitism_k`, so the spec's `k` is effectively the
    *initial* value carried by the baseline FeatureConfig. A variant that
    overrides `elitism_k` will be honored automatically because the
    closure reads from `ctx`.
    """
    k_default = int(spec.get("k", 1))
    def _replace(parents, offspring, ctx):
        return generational_elitism(parents, offspring, int(ctx["elitism_k"]))
    _replace.__name__ = f"generational_elitism_k{k_default}"
    return _replace


# ===================== EVOLUTION LOOP =====================

def clip_to_bounds(vec, lb, ub):
    """Hard clip — kept for backward compatibility and operator-internal use."""
    return np.clip(vec, lb, ub).tolist()


def apply_bounds(vec, lb, ub, mode="clip", rng=None):
    """Bound handler used by the core loop on init/crossover/mutation outputs.

    "clip"    — np.clip onto [lb, ub] (default; matches historical behavior).
    "reflect" — fold out-of-bounds back into [lb, ub] (commonly used in
                CEC-suite literature; reduces boundary-mass bias).
    "reinit"  — replace out-of-bounds genes with uniform samples in [lb, ub].
    """
    arr = np.asarray(vec, dtype=float)
    if mode == "clip":
        return np.clip(arr, lb, ub).tolist()
    if mode == "reflect":
        # Fold repeatedly in case |out-of-bound| > range; cheap closed form.
        span = ub - lb
        # Map to a 2*span-periodic triangular wave centered on [lb, ub].
        t = np.mod(arr - lb, 2.0 * span)
        folded = np.where(t > span, 2.0 * span - t, t)
        out = lb + folded
        # Numerical safety in case of degenerate spans.
        return np.clip(out, lb, ub).tolist()
    if mode == "reinit":
        oob = (arr < lb) | (arr > ub)
        if not np.any(oob):
            return arr.tolist()
        rng = rng if rng is not None else np.random.default_rng()
        fresh = rng.uniform(lb, ub)
        out = np.where(oob, fresh, arr)
        return out.tolist()
    if mode == "passthrough":
        # No-op for non-real encodings (permutations, discrete assignments)
        # where lb/ub have no meaningful clip/reflect interpretation.
        # Returns a list (possibly of ints) so downstream code never sees
        # numpy scalar types.
        if isinstance(vec, np.ndarray):
            return vec.tolist()
        return list(vec)
    raise ValueError(f"unknown bound_handling mode: {mode!r}")


def make_individual(vec):
    ind = creator.Individual(vec)
    ind.age = 0  # A3 aging: lifespan counter, incremented per generation
    return ind


def evaluate_population(pop, evaluate_fn, eval_counter):
    """Evaluate every individual whose fitness is invalid; track FE count."""
    n_evals = 0
    for ind in pop:
        if not ind.fitness.valid:
            f = evaluate_fn(np.asarray(ind))
            ind.fitness.values = (float(f),)
            n_evals += 1
    eval_counter[0] += n_evals
    return n_evals


def _hussain_diversity(positions: np.ndarray) -> float:
    """Hussain et al. diversity:  (1/N) Σᵢ (1/D) Σⱼ |median_j - x_{ij}|.
    Robust population-spread measure (median-centered, scale-aware)."""
    if positions.size == 0:
        return 0.0
    median = np.median(positions, axis=0)
    return float(np.mean(np.mean(np.abs(positions - median), axis=1)))


def run_sga(
    problem,
    config: FeatureConfig,
    pop_size=50,
    max_fes=None,
    seed=0,
    record_history=True,
):
    """
    Run one independent trial of an SGA variant on a given Opfunu problem.

    Returns:
        result: dict with best_fitness, best_solution, fes_used, history
    """
    # F46: dispatch to island co-evolution when configured
    if getattr(config, "n_islands", 1) > 1:
        return run_sga_islands(
            problem, config, pop_size=pop_size, max_fes=max_fes,
            seed=seed, record_history=record_history,
        )
    rng = np.random.default_rng(seed)
    random.seed(seed)  # DEAP's selection helpers use Python's `random`

    dim = problem.ndim
    lb = np.asarray(problem.lb, dtype=float)
    ub = np.asarray(problem.ub, dtype=float)

    # Default MaxFES per CEC 2022 rules (only D=10 and D=20 are specified).
    if max_fes is None:
        _CEC2022_MAXFES = {10: 200_000, 20: 1_000_000}
        if dim not in _CEC2022_MAXFES:
            raise ValueError(
                f"CEC2022 MaxFES is defined only for dim 10 and 20; "
                f"got dim={dim}. Pass max_fes explicitly to override."
            )
        max_fes = _CEC2022_MAXFES[dim]

    # Build the context dict every operator can read
    ctx = {
        "rng": rng, "lb": lb, "ub": ub, "dim": dim,
        "pop_size": pop_size, "max_fes": max_fes,
        "elitism_k": config.elitism_k,
        "gen": 0, "max_gen_estimate": max_fes // pop_size,
        "best_so_far": None,
        "stagnation_counter": 0,
        "last_best_fitness": np.inf,
        # Backbone extensions for nature-inspired features:
        "_population": [],                              # current pop, refreshed each gen
        "archive": deque(maxlen=config.archive_size),   # historical-best archive (F09/F31/F37)
        "shade_memory": {                               # F42 SHADE-style adaptive F/CR
            "F":  deque([0.5] * 5, maxlen=10),
            "CR": deque([0.5] * 5, maxlen=10),
        },
        "t_ratio": 0.0,                                 # gen / max_gen_estimate (F40/41/44)
        "hho_energy": 1.0,                              # |E| escape energy (F20/F43)
        "config": config,                               # so operators can read e.g. mut_prob
        "bound_handling": getattr(config, "bound_handling", "clip"),
        # Exposed for refine_fn / sub-loop operators that need to evaluate
        # candidates inside a single GA generation while respecting MaxFES.
        "evaluate_fn": problem.evaluate,
        "eval_counter": None,  # filled in below once eval_counter is constructed
        # The problem object itself, so structure-aware operators (e.g. the
        # GACO hybrid, which needs the TSP distance matrix for pheromone /
        # heuristic / 2-opt) can reach instance data the abstract
        # `(lb, ub, dim, evaluate)` interface does not expose.
        "problem": problem,
    }

    # Pick operators (use defaults if not overridden)
    init_fn = config.init_fn or default_init
    select_fn = config.select_fn or default_select
    crossover_fn = config.crossover_fn or default_crossover
    mutate_fn = config.mutate_fn or default_mutate
    replace_fn = config.replace_fn or default_replace

    eval_counter = [0]
    evaluate_fn = problem.evaluate
    ctx["eval_counter"] = eval_counter

    # ---- Initialization ----
    init_vectors = init_fn(pop_size, dim, lb, ub, ctx)
    population = [make_individual(apply_bounds(v, lb, ub, mode=ctx["bound_handling"], rng=rng)) for v in init_vectors]
    evaluate_population(population, evaluate_fn, eval_counter)

    history = []
    best = min(population, key=lambda x: x.fitness.values[0])
    ctx["best_so_far"] = best
    div_max_so_far = 0.0  # for XPL%

    # ---- Generational loop ----
    gen = 0
    while eval_counter[0] < max_fes:
        gen += 1
        ctx["gen"] = gen
        ctx["t_ratio"] = min(1.0, gen / max(ctx["max_gen_estimate"], 1))
        # HHO escape energy: E = 2*E0*(1 - t/T), E0 ~ U(-1,1)
        ctx["hho_energy"] = 2.0 * (rng.random() * 2.0 - 1.0) * (1.0 - ctx["t_ratio"])
        ctx["_population"] = list(population)
        # A3 aging: increment lifespan counter on every survivor
        for _ind in population:
            _ind.age = getattr(_ind, "age", 0) + 1

        # ---- Selection (with selection-pressure tracking) ----
        pre_select_fits = np.array([ind.fitness.values[0] for ind in population])
        raw_parents = select_fn(population, len(population), ctx)
        parents = []
        for p in raw_parents:
            new_p = creator.Individual(list(p))
            try:
                if p.fitness.valid:
                    new_p.fitness.values = p.fitness.values
            except Exception:
                pass
            parents.append(new_p)
        post_select_fits = np.array(
            [p.fitness.values[0] if p.fitness.valid else float("inf")
             for p in parents]
        )
        # Selection intensity (Mühlenbein & Schlierkamp-Voosen 1993):
        #   I = |mean(selected) - mean(population)| / std(population)
        # Dimensionless, comparable across functions with different fitness
        # scales. For tournament k=3 the expected value is ~0.85.
        # Stored under the legacy column name `selection_pressure` to keep
        # the parquet schema stable across existing archives.
        pre_std = float(np.std(pre_select_fits))
        if pre_std > 1e-30 and np.all(np.isfinite(post_select_fits)):
            selection_pressure = float(
                abs(np.mean(post_select_fits) - np.mean(pre_select_fits))
                / pre_std
            )
        else:
            selection_pressure = 0.0

        # ---- Crossover (track child→parent mapping for step-size + cx success) ----
        offspring = []
        cx_parent_idx = []   # index into parents[] for each child's primary parent
        cx_was_applied = []  # was crossover actually applied to this child?
        for i in range(0, len(parents) - 1, 2):
            p1, p2 = parents[i], parents[i + 1]
            if rng.random() < config.cx_prob:
                c1, c2 = crossover_fn(p1, p2, ctx)
                cx_was_applied.extend([True, True])
            else:
                c1, c2 = list(p1), list(p2)
                cx_was_applied.extend([False, False])
            offspring.append(make_individual(apply_bounds(c1, lb, ub, mode=ctx["bound_handling"], rng=rng)))
            offspring.append(make_individual(apply_bounds(c2, lb, ub, mode=ctx["bound_handling"], rng=rng)))
            cx_parent_idx.extend([i, i + 1])

        # ---- Mutation (track which were mutated and pre-mutation vector) ----
        mut_was_applied = [False] * len(offspring)
        pre_mut_vecs = [list(o) for o in offspring]
        for i, ind in enumerate(offspring):
            if rng.random() < config.mut_prob:
                m = mutate_fn(ind, ctx)
                offspring[i] = make_individual(apply_bounds(m, lb, ub, mode=ctx["bound_handling"], rng=rng))
                mut_was_applied[i] = True

        # ---- Evaluate offspring (respect FE budget) ----
        for ind in offspring:
            if eval_counter[0] >= max_fes:
                break
            if not ind.fitness.valid:
                f = evaluate_fn(np.asarray(ind))
                ind.fitness.values = (float(f),)
                eval_counter[0] += 1

        evaluated_mask = [o.fitness.valid for o in offspring]

        # ---- Operator success rates & step size (over evaluated offspring) ----
        cx_attempts = cx_successes = 0
        mut_attempts = mut_successes = 0
        step_sizes = []
        for i, child in enumerate(offspring):
            if not evaluated_mask[i]:
                continue
            if cx_parent_idx[i] < len(parents):
                parent = parents[cx_parent_idx[i]]
                # step size = ||child - primary parent||₂
                step_sizes.append(float(np.linalg.norm(
                    np.asarray(child, dtype=float) - np.asarray(parent, dtype=float))))
                if cx_was_applied[i] and parent.fitness.valid:
                    cx_attempts += 1
                    if child.fitness.values[0] < parent.fitness.values[0]:
                        cx_successes += 1
            if mut_was_applied[i]:
                parent = parents[cx_parent_idx[i]] if cx_parent_idx[i] < len(parents) else None
                if parent is not None and parent.fitness.valid:
                    mut_attempts += 1
                    if child.fitness.values[0] < parent.fitness.values[0]:
                        mut_successes += 1

        cx_success_rate = (cx_successes / cx_attempts) if cx_attempts else 0.0
        mut_success_rate = (mut_successes / mut_attempts) if mut_attempts else 0.0
        mean_step_size = float(np.mean(step_sizes)) if step_sizes else 0.0

        # Drop unevaluated trailing offspring (budget exhausted)
        offspring = [o for o in offspring if o.fitness.valid]

        if not offspring:
            break

        # ---- Replacement ----
        population = replace_fn(population, offspring, ctx)
        # Evaluate any individuals introduced by replace_fn (e.g. A8 resize)
        for ind in population:
            if not ind.fitness.valid and eval_counter[0] < max_fes:
                f = evaluate_fn(np.asarray(ind))
                ind.fitness.values = (float(f),)
                eval_counter[0] += 1
        population = [ind for ind in population if ind.fitness.valid]
        if not population:
            break

        # ---- Optional diversity injection ----
        if config.diversity_fn is not None:
            population = config.diversity_fn(population, ctx)
            for ind in population:
                if not ind.fitness.valid and eval_counter[0] < max_fes:
                    f = evaluate_fn(np.asarray(ind))
                    ind.fitness.values = (float(f),)
                    eval_counter[0] += 1
            # Drop any leftover unevaluated members (budget exhausted)
            population = [ind for ind in population if ind.fitness.valid]
            if not population:
                break

        # ---- Optional local refinement (e.g. SGA-style nested PSO) ----
        if config.refine_fn is not None:
            population = config.refine_fn(population, ctx)
            for ind in population:
                if not ind.fitness.valid and eval_counter[0] < max_fes:
                    f = evaluate_fn(np.asarray(ind))
                    ind.fitness.values = (float(f),)
                    eval_counter[0] += 1
            population = [ind for ind in population if ind.fitness.valid]
            if not population:
                break

        # ---- Track best & history ----
        cur_best = min(population, key=lambda x: x.fitness.values[0])
        if cur_best.fitness.values[0] < best.fitness.values[0]:
            best = creator.Individual(list(cur_best))
            best.fitness.values = cur_best.fitness.values
            ctx["best_so_far"] = best
            ctx["stagnation_counter"] = 0
            # Push the new best into the historical archive (F09/F31/F37)
            archived = creator.Individual(list(best))
            archived.fitness.values = best.fitness.values
            ctx["archive"].append(archived)
        else:
            ctx["stagnation_counter"] += 1

        if record_history:
            fitnesses = np.array([ind.fitness.values[0] for ind in population])
            positions = np.array([list(ind) for ind in population])
            diversity_std = float(np.mean(np.std(positions, axis=0)))
            hussain_div = _hussain_diversity(positions)
            div_max_so_far = max(div_max_so_far, hussain_div)
            xpl_pct = 100.0 * hussain_div / div_max_so_far if div_max_so_far > 0 else 0.0
            history.append({
                "gen": gen,
                "fes": eval_counter[0],
                "best_fitness": float(best.fitness.values[0]),
                "mean_fitness": float(fitnesses.mean()),
                "diversity": diversity_std,
                "hussain_div": hussain_div,
                "xpl_pct": xpl_pct,
                "xpt_pct": 100.0 - xpl_pct,
                "stagnation_count": int(ctx["stagnation_counter"]),
                "mut_success_rate": mut_success_rate,
                "cx_success_rate": cx_success_rate,
                "mean_step_size": mean_step_size,
                "selection_pressure": selection_pressure if np.isfinite(selection_pressure) else 0.0,
            })

    error = float(best.fitness.values[0]) - float(problem.f_global)
    if error < 1e-8:
        error = 0.0

    return {
        "variant": config.name,
        "seed": seed,
        "best_fitness": float(best.fitness.values[0]),
        "error": error,
        "best_solution": list(best),
        "fes_used": eval_counter[0],
        "history": history,
    }


# ===================== F46: ISLAND CO-EVOLUTION =====================

def run_sga_islands(problem, config, pop_size=50, max_fes=None, seed=0,
                    record_history=True):
    """K isolated sub-populations evolve in parallel; best individuals
    migrate every `config.migration_interval` generations.

    To stay close to the single-pop loop without duplicating its full
    instrumentation, each island holds its own panmictic SGA state and
    advances one generation at a time. Selection/crossover/mutation never
    cross island boundaries between migrations.
    """
    rng = np.random.default_rng(seed)
    random.seed(seed)

    dim = problem.ndim
    lb = np.asarray(problem.lb, dtype=float)
    ub = np.asarray(problem.ub, dtype=float)
    if max_fes is None:
        _CEC2022_MAXFES = {10: 200_000, 20: 1_000_000}
        if dim not in _CEC2022_MAXFES:
            raise ValueError(
                f"CEC2022 MaxFES is defined only for dim 10 and 20; "
                f"got dim={dim}. Pass max_fes explicitly to override."
            )
        max_fes = _CEC2022_MAXFES[dim]

    K = max(1, config.n_islands)
    island_pop_sizes = [pop_size // K] * K
    for i in range(pop_size - sum(island_pop_sizes)):
        island_pop_sizes[i] += 1

    init_fn = config.init_fn or default_init
    select_fn = config.select_fn or default_select
    crossover_fn = config.crossover_fn or default_crossover
    mutate_fn = config.mutate_fn or default_mutate
    replace_fn = config.replace_fn or default_replace
    evaluate_fn = problem.evaluate
    eval_counter = [0]

    # Per-island state
    islands = []
    for k in range(K):
        ctx = {
            "rng": rng, "lb": lb, "ub": ub, "dim": dim,
            "pop_size": island_pop_sizes[k], "max_fes": max_fes,
            "elitism_k": config.elitism_k,
            "gen": 0,
            "max_gen_estimate": max_fes // max(pop_size, 1),
            "best_so_far": None,
            "stagnation_counter": 0,
            "last_best_fitness": np.inf,
            "_population": [],
            "archive": deque(maxlen=config.archive_size),
            "shade_memory": {
                "F":  deque([0.5] * 5, maxlen=10),
                "CR": deque([0.5] * 5, maxlen=10),
            },
            "t_ratio": 0.0,
            "hho_energy": 1.0,
            "config": config,
            "bound_handling": getattr(config, "bound_handling", "clip"),
            "island_id": k,
            "evaluate_fn": problem.evaluate,
            "eval_counter": eval_counter,
            "problem": problem,
        }
        init_vectors = init_fn(island_pop_sizes[k], dim, lb, ub, ctx)
        pop = [make_individual(apply_bounds(v, lb, ub, mode=ctx["bound_handling"], rng=rng)) for v in init_vectors]
        evaluate_population(pop, evaluate_fn, eval_counter)
        islands.append({"pop": pop, "ctx": ctx})

    # Global best across all islands
    all_inds = [ind for isl in islands for ind in isl["pop"]]
    best = creator.Individual(list(min(all_inds, key=lambda x: x.fitness.values[0])))
    best.fitness.values = min(all_inds, key=lambda x: x.fitness.values[0]).fitness.values

    history = []
    gen = 0
    while eval_counter[0] < max_fes:
        gen += 1
        for isl in islands:
            ctx = isl["ctx"]
            pop = isl["pop"]
            ctx["gen"] = gen
            ctx["t_ratio"] = min(1.0, gen / max(ctx["max_gen_estimate"], 1))
            ctx["hho_energy"] = 2.0 * (rng.random() * 2 - 1) * (1.0 - ctx["t_ratio"])
            ctx["_population"] = list(pop)
            for _ind in pop:
                _ind.age = getattr(_ind, "age", 0) + 1

            raw_parents = select_fn(pop, len(pop), ctx)
            parents = []
            for src in raw_parents:
                new_p = creator.Individual(list(src))
                if src.fitness.valid:
                    new_p.fitness.values = src.fitness.values
                parents.append(new_p)

            offspring = []
            for i in range(0, len(parents) - 1, 2):
                p1, p2 = parents[i], parents[i + 1]
                if rng.random() < config.cx_prob:
                    c1, c2 = crossover_fn(p1, p2, ctx)
                else:
                    c1, c2 = list(p1), list(p2)
                offspring.append(make_individual(apply_bounds(c1, lb, ub, mode=ctx["bound_handling"], rng=rng)))
                offspring.append(make_individual(apply_bounds(c2, lb, ub, mode=ctx["bound_handling"], rng=rng)))

            for i, ind in enumerate(offspring):
                if rng.random() < config.mut_prob:
                    m = mutate_fn(ind, ctx)
                    offspring[i] = make_individual(apply_bounds(m, lb, ub, mode=ctx["bound_handling"], rng=rng))

            for ind in offspring:
                if eval_counter[0] >= max_fes:
                    break
                if not ind.fitness.valid:
                    f = evaluate_fn(np.asarray(ind))
                    ind.fitness.values = (float(f),)
                    eval_counter[0] += 1
            offspring = [o for o in offspring if o.fitness.valid]
            if not offspring:
                continue

            isl["pop"] = replace_fn(pop, offspring, ctx)
            for ind in isl["pop"]:
                if not ind.fitness.valid and eval_counter[0] < max_fes:
                    f = evaluate_fn(np.asarray(ind))
                    ind.fitness.values = (float(f),)
                    eval_counter[0] += 1
            isl["pop"] = [ind for ind in isl["pop"] if ind.fitness.valid]
            if not isl["pop"]:
                continue

            if config.diversity_fn is not None:
                isl["pop"] = config.diversity_fn(isl["pop"], ctx)
                for ind in isl["pop"]:
                    if not ind.fitness.valid and eval_counter[0] < max_fes:
                        f = evaluate_fn(np.asarray(ind))
                        ind.fitness.values = (float(f),)
                        eval_counter[0] += 1
                isl["pop"] = [ind for ind in isl["pop"] if ind.fitness.valid]
                if not isl["pop"]:
                    continue

            if config.refine_fn is not None:
                isl["pop"] = config.refine_fn(isl["pop"], ctx)
                for ind in isl["pop"]:
                    if not ind.fitness.valid and eval_counter[0] < max_fes:
                        f = evaluate_fn(np.asarray(ind))
                        ind.fitness.values = (float(f),)
                        eval_counter[0] += 1
                isl["pop"] = [ind for ind in isl["pop"] if ind.fitness.valid]
                if not isl["pop"]:
                    continue

            cur_best = min(isl["pop"], key=lambda x: x.fitness.values[0])
            if cur_best.fitness.values[0] < best.fitness.values[0]:
                best = creator.Individual(list(cur_best))
                best.fitness.values = cur_best.fitness.values
                ctx["stagnation_counter"] = 0
                archived = creator.Individual(list(best))
                archived.fitness.values = best.fitness.values
                ctx["archive"].append(archived)
            else:
                ctx["stagnation_counter"] += 1

        # Migration: send each island's best to the next island, replace its worst
        if K > 1 and gen % max(1, config.migration_interval) == 0:
            best_per_island = [
                min(isl["pop"], key=lambda x: x.fitness.values[0])
                for isl in islands
            ]
            for k in range(K):
                src = best_per_island[k]
                dst_isl = islands[(k + 1) % K]["pop"]
                worst_idx = max(range(len(dst_isl)),
                                key=lambda i: dst_isl[i].fitness.values[0])
                migrant = creator.Individual(list(src))
                migrant.fitness.values = src.fitness.values
                dst_isl[worst_idx] = migrant

        if record_history:
            all_inds = [ind for isl in islands for ind in isl["pop"]]
            fitnesses = np.array([ind.fitness.values[0] for ind in all_inds])
            positions = np.array([list(ind) for ind in all_inds])
            diversity_std = float(np.mean(np.std(positions, axis=0))) if len(positions) else 0.0
            hussain_div = _hussain_diversity(positions) if len(positions) else 0.0
            history.append({
                "gen": gen,
                "fes": eval_counter[0],
                "best_fitness": float(best.fitness.values[0]),
                "mean_fitness": float(fitnesses.mean()) if len(fitnesses) else float("inf"),
                "diversity": diversity_std,
                "hussain_div": hussain_div,
                "xpl_pct": 0.0, "xpt_pct": 0.0,
                "stagnation_count": 0,
                "mut_success_rate": 0.0, "cx_success_rate": 0.0,
                "mean_step_size": 0.0, "selection_pressure": 0.0,
            })

    error = float(best.fitness.values[0]) - float(problem.f_global)
    if error < 1e-8:
        error = 0.0
    return {
        "variant": config.name,
        "seed": seed,
        "best_fitness": float(best.fitness.values[0]),
        "error": error,
        "best_solution": list(best),
        "fes_used": eval_counter[0],
        "history": history,
    }
