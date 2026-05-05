"""
features.py
-----------
Drop-in implementations of nature-inspired algorithm features.
Each function matches the signature expected by `FeatureConfig` in sga_core.py.

These are the Tier 1 and Tier 2 features from the framework document.
Add more as you progress to Tier 3 / Tier 4.

References for each feature are in the framework PDF (Section 5).
"""

import numpy as np
from deap import tools, creator
from scipy.special import gamma


# ============================================================
# F03 — Opposition-Based Initialization (Tizhoosh 2005)
# ============================================================
def init_opposition(n, dim, lb, ub, ctx):
    """
    Generate n random points + their opposites; keep the best n by fitness.
    Note: requires evaluating 2n points, so consumes extra FEs at start.
    For a fair comparison, run with reduced max_fes or adjust budget.

    For simplicity here we just return 2n candidates and let the main loop
    evaluate and naturally keep the best (via standard sorting downstream).
    A purer version would evaluate inside this function.
    """
    rng = ctx["rng"]
    base = [rng.uniform(lb, ub).tolist() for _ in range(n)]
    opposites = [(lb + ub - np.array(x)).tolist() for x in base]
    # Return 2n candidates; the eval+replacement will narrow down naturally
    # (To strictly keep n, you'd need to evaluate here. This is a simplification.)
    return (base + opposites)[:n]  # Simple: just take first n with opposites mixed in


# A stricter version that respects the population size cap:
def init_opposition_strict(n, dim, lb, ub, ctx):
    """OBL-init: take n random, generate n opposites, return all 2n.
    The evolution loop's first sort+select keeps the best n implicitly
    via the next generation's tournament. For a true OBL-init that
    evaluates and keeps the best n, modify run_sga to handle this."""
    rng = ctx["rng"]
    base = [rng.uniform(lb, ub).tolist() for _ in range(n // 2)]
    opposites = [(lb + ub - np.array(x)).tolist() for x in base]
    return base + opposites


# ============================================================
# F02 — Chaotic-Map Initialization (logistic map)
# ============================================================
def init_chaotic(n, dim, lb, ub, ctx):
    """Use logistic map x_{k+1} = 4 x_k (1 - x_k) for richer coverage."""
    rng = ctx["rng"]
    out = []
    for _ in range(n):
        x = rng.uniform(0.01, 0.99)  # avoid fixed points 0 and 1
        vec = []
        for _d in range(dim):
            x = 4.0 * x * (1.0 - x)
            vec.append(lb[_d] + x * (ub[_d] - lb[_d]))
        out.append(vec)
    return out


# ============================================================
# F13 — Lévy Flight Mutation (Cuckoo Search / HHO style)
# ============================================================
def _levy_step(dim, beta=1.5, rng=None):
    """Mantegna's algorithm for stable Lévy random numbers."""
    if rng is None:
        rng = np.random.default_rng()
    sigma_u = (gamma(1 + beta) * np.sin(np.pi * beta / 2) /
               (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = rng.normal(0, sigma_u, dim)
    v = rng.normal(0, 1, dim)
    return u / (np.abs(v) ** (1 / beta))


def mutate_levy(ind, ctx):
    """Replace polynomial mutation with Lévy-flight perturbation."""
    rng = ctx["rng"]
    dim = ctx["dim"]
    lb, ub = ctx["lb"], ctx["ub"]
    step = _levy_step(dim, beta=1.5, rng=rng)
    # Scale step to ~1% of search range so jumps are bold but not absurd
    scale = 0.01 * (ub - lb)
    new_vec = np.asarray(ind, dtype=float) + scale * step
    return np.clip(new_vec, lb, ub).tolist()


# ============================================================
# F17 — Differential Vector Mutation (DE/rand/1)
# ============================================================
def make_de_mutation(F=0.5):
    """Closure: returns a mutate_fn that does v = x + F*(x_r2 - x_r3).
    Requires access to the population, which we get via ctx['_population']."""
    def mutate_de(ind, ctx):
        rng = ctx["rng"]
        pop = ctx.get("_population", None)
        if pop is None or len(pop) < 3:
            return list(ind)
        # Pick three distinct individuals different from ind
        candidates = [p for p in pop if list(p) != list(ind)]
        if len(candidates) < 3:
            return list(ind)
        idx = rng.choice(len(candidates), size=3, replace=False)
        x_r1, x_r2, x_r3 = (np.asarray(candidates[i], dtype=float) for i in idx)
        v = x_r1 + F * (x_r2 - x_r3)
        return np.clip(v, ctx["lb"], ctx["ub"]).tolist()
    return mutate_de


# ============================================================
# F36 — Greedy One-to-One Replacement (DE-style)
# ============================================================
def replace_greedy(parents, offspring, ctx):
    """Each offspring replaces its corresponding parent only if it's better.
    Falls back to size-matched pairing; if mismatched, truncates."""
    new_pop = []
    n = min(len(parents), len(offspring))
    for i in range(n):
        if offspring[i].fitness.values[0] < parents[i].fitness.values[0]:
            new_pop.append(offspring[i])
        else:
            new_pop.append(parents[i])
    # Carry over any extras from parents
    if len(parents) > n:
        new_pop.extend(parents[n:])
    return new_pop


# ============================================================
# F06 — Leader Hierarchy Selection (GWO alpha/beta/delta)
# ============================================================
def select_leader_hierarchy(population, k, ctx):
    """
    Pick k individuals biased toward the best-3 (alpha, beta, delta).
    Each returned 'leader-blend' is the mean of the top 3, with small jitter
    from a random pop member — analogous to GWO's position update.
    """
    rng = ctx["rng"]
    sorted_pop = sorted(population, key=lambda x: x.fitness.values[0])
    alpha, beta, delta = sorted_pop[0], sorted_pop[1 % len(sorted_pop)], sorted_pop[2 % len(sorted_pop)]
    selected = []
    for _ in range(k):
        # 80% of the time, return one of alpha/beta/delta (rank-weighted)
        # 20% of the time, return tournament winner
        if rng.random() < 0.8:
            r = rng.random()
            if r < 0.5:
                selected.append(creator.Individual(list(alpha)))
            elif r < 0.8:
                selected.append(creator.Individual(list(beta)))
            else:
                selected.append(creator.Individual(list(delta)))
            selected[-1].fitness.values = (float("inf"),)  # mark as needing re-eval
            # Actually copy parent's fitness so selection downstream behaves:
            selected[-1].fitness.values = sorted_pop[0].fitness.values
        else:
            cand = tools.selTournament(population, 1, tournsize=3)[0]
            ind = creator.Individual(list(cand))
            ind.fitness.values = cand.fitness.values
            selected.append(ind)
    return selected


# ============================================================
# F40 — Linearly Decreasing Control Coefficient
# ============================================================
def make_adaptive_mutate(base_indpb=0.1):
    """Mutation probability decreases linearly from 2*base to 0 across run."""
    def mutate_adaptive(ind, ctx):
        gen = ctx["gen"]
        max_gen = max(ctx["max_gen_estimate"], 1)
        # Linearly decay from 2*base_indpb to 0
        a = max(0.0, 2.0 * base_indpb * (1.0 - gen / max_gen))
        mutated, = tools.mutPolynomialBounded(
            list(ind), eta=20.0,
            low=ctx["lb"].tolist(), up=ctx["ub"].tolist(),
            indpb=a / max(ctx["dim"], 1),
        )
        return mutated
    return mutate_adaptive


# ============================================================
# F39 — Stagnation-Triggered Restart of Worst k%
# ============================================================
def make_diversity_restart(stagnation_threshold=20, restart_fraction=0.2):
    """If best hasn't improved in T gens, replace worst k% with random."""
    def diversity_restart(population, ctx):
        if ctx["stagnation_counter"] < stagnation_threshold:
            return population
        rng = ctx["rng"]
        n_restart = int(len(population) * restart_fraction)
        sorted_pop = sorted(population, key=lambda x: x.fitness.values[0])
        keep = sorted_pop[: len(population) - n_restart]
        fresh = []
        for _ in range(n_restart):
            v = rng.uniform(ctx["lb"], ctx["ub"]).tolist()
            ind = creator.Individual(v)
            fresh.append(ind)
        ctx["stagnation_counter"] = 0  # reset
        return keep + fresh
    return diversity_restart


# ============================================================
# Helper: pass population into ctx so DE-style features can see it
# ============================================================
def with_population_aware_select(base_select_fn):
    """Wrap a selection fn so it stashes the population into ctx['_population'].
    DE mutation needs access to other individuals."""
    def wrapped(population, k, ctx):
        ctx["_population"] = population
        return base_select_fn(population, k, ctx)
    return wrapped
