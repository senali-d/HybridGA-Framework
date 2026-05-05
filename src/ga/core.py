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
        cx_prob=0.9,
        mut_prob=0.1,
        elitism_k=1,
    ):
        self.name = name
        self.init_fn = init_fn
        self.select_fn = select_fn
        self.crossover_fn = crossover_fn
        self.mutate_fn = mutate_fn
        self.replace_fn = replace_fn
        self.diversity_fn = diversity_fn
        self.cx_prob = cx_prob
        self.mut_prob = mut_prob
        self.elitism_k = elitism_k


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


def default_replace(parents, offspring, ctx):
    """
    Generational replacement with k-elitism.
    Keep best k from parents, fill rest from offspring.
    """
    k = ctx["elitism_k"]
    parents_sorted = sorted(parents, key=lambda x: x.fitness.values[0])
    elites = parents_sorted[:k]
    offspring_sorted = sorted(offspring, key=lambda x: x.fitness.values[0])
    new_pop = elites + offspring_sorted[: len(parents) - k]
    return new_pop


# ===================== EVOLUTION LOOP =====================

def clip_to_bounds(vec, lb, ub):
    return np.clip(vec, lb, ub).tolist()


def make_individual(vec):
    ind = creator.Individual(vec)
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
    rng = np.random.default_rng(seed)
    random.seed(seed)  # DEAP's selection helpers use Python's `random`

    dim = problem.ndim
    lb = np.asarray(problem.lb, dtype=float)
    ub = np.asarray(problem.ub, dtype=float)

    # Default MaxFES per CEC 2022 rules
    if max_fes is None:
        max_fes = 200_000 if dim <= 10 else 1_000_000

    # Build the context dict every operator can read
    ctx = {
        "rng": rng, "lb": lb, "ub": ub, "dim": dim,
        "pop_size": pop_size, "max_fes": max_fes,
        "elitism_k": config.elitism_k,
        "gen": 0, "max_gen_estimate": max_fes // pop_size,
        "best_so_far": None,
        "stagnation_counter": 0,
        "last_best_fitness": np.inf,
    }

    # Pick operators (use defaults if not overridden)
    init_fn = config.init_fn or default_init
    select_fn = config.select_fn or default_select
    crossover_fn = config.crossover_fn or default_crossover
    mutate_fn = config.mutate_fn or default_mutate
    replace_fn = config.replace_fn or default_replace

    eval_counter = [0]
    evaluate_fn = problem.evaluate

    # ---- Initialization ----
    init_vectors = init_fn(pop_size, dim, lb, ub, ctx)
    population = [make_individual(clip_to_bounds(v, lb, ub)) for v in init_vectors]
    evaluate_population(population, evaluate_fn, eval_counter)

    history = []
    best = min(population, key=lambda x: x.fitness.values[0])
    ctx["best_so_far"] = best

    # ---- Generational loop ----
    gen = 0
    while eval_counter[0] < max_fes:
        gen += 1
        ctx["gen"] = gen

        # ---- Selection ----
        parents = select_fn(population, len(population), ctx)
        parents = [creator.Individual(list(p)) for p in parents]
        for p, src in zip(parents, select_fn(population, len(population), ctx)):
            p.fitness.values = src.fitness.values

        # ---- Crossover ----
        offspring = []
        for i in range(0, len(parents) - 1, 2):
            p1, p2 = parents[i], parents[i + 1]
            if rng.random() < config.cx_prob:
                c1, c2 = crossover_fn(p1, p2, ctx)
            else:
                c1, c2 = list(p1), list(p2)
            offspring.append(make_individual(clip_to_bounds(c1, lb, ub)))
            offspring.append(make_individual(clip_to_bounds(c2, lb, ub)))

        # ---- Mutation ----
        for i, ind in enumerate(offspring):
            if rng.random() < config.mut_prob:
                m = mutate_fn(ind, ctx)
                offspring[i] = make_individual(clip_to_bounds(m, lb, ub))

        # ---- Evaluate offspring (respect FE budget) ----
        for ind in offspring:
            if eval_counter[0] >= max_fes:
                break
            if not ind.fitness.valid:
                f = evaluate_fn(np.asarray(ind))
                ind.fitness.values = (float(f),)
                eval_counter[0] += 1

        # Drop unevaluated trailing offspring (budget exhausted)
        offspring = [o for o in offspring if o.fitness.valid]

        if not offspring:
            break

        # ---- Replacement ----
        population = replace_fn(population, offspring, ctx)

        # ---- Optional diversity injection (e.g., random immigrants) ----
        if config.diversity_fn is not None:
            population = config.diversity_fn(population, ctx)
            for ind in population:
                if not ind.fitness.valid and eval_counter[0] < max_fes:
                    f = evaluate_fn(np.asarray(ind))
                    ind.fitness.values = (float(f),)
                    eval_counter[0] += 1

        # ---- Track best & history ----
        cur_best = min(population, key=lambda x: x.fitness.values[0])
        if cur_best.fitness.values[0] < best.fitness.values[0]:
            best = creator.Individual(list(cur_best))
            best.fitness.values = cur_best.fitness.values
            ctx["best_so_far"] = best
            ctx["stagnation_counter"] = 0
        else:
            ctx["stagnation_counter"] += 1

        if record_history:
            fitnesses = np.array([ind.fitness.values[0] for ind in population])
            positions = np.array([list(ind) for ind in population])
            diversity = float(np.mean(np.std(positions, axis=0)))
            history.append({
                "gen": gen,
                "fes": eval_counter[0],
                "best_fitness": float(best.fitness.values[0]),
                "mean_fitness": float(fitnesses.mean()),
                "diversity": diversity,
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
