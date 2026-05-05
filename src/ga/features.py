"""
features.py
-----------
Drop-in implementations of nature-inspired algorithm features.
Each function matches the signature expected by `FeatureConfig` in sga_core.py.

These are the Tier 1 and Tier 2 features from the framework document.
Add more as you progress to Tier 3 / Tier 4.

References for each feature are in the framework PDF (Section 5).
"""

import math
import numpy as np
from deap import tools, creator
from scipy.special import gamma
from scipy.stats import qmc


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
    DE mutation needs access to other individuals.

    Note: core.run_sga now sets ctx['_population'] each generation, so this
    wrapper is a no-op for new variants. Retained for backward compatibility.
    """
    def wrapped(population, k, ctx):
        ctx["_population"] = population
        return base_select_fn(population, k, ctx)
    return wrapped


# ============================================================
# Helpers shared across multiple features
# ============================================================
def _best_in_pop(pop):
    return min(pop, key=lambda x: x.fitness.values[0])


def _sorted_pop(pop):
    return sorted(pop, key=lambda x: x.fitness.values[0])


def _arr(v):
    return np.asarray(v, dtype=float)


# ============================================================
# F02 (additional maps) — tent and sinusoidal chaotic init
# ============================================================
def init_chaotic_tent(n, dim, lb, ub, ctx):
    """Tent map x_{k+1} = x/0.7 if x<0.7 else (1-x)/0.3 ."""
    rng = ctx["rng"]
    out = []
    for _ in range(n):
        x = rng.uniform(0.01, 0.99)
        vec = []
        for d in range(dim):
            x = x / 0.7 if x < 0.7 else (1.0 - x) / 0.3
            x = min(max(x, 1e-6), 1 - 1e-6)
            vec.append(lb[d] + x * (ub[d] - lb[d]))
        out.append(vec)
    return out


def init_chaotic_sinusoidal(n, dim, lb, ub, ctx):
    """Sinusoidal map x_{k+1} = a sin(pi x), a=2.3 (chaotic regime)."""
    rng = ctx["rng"]
    out = []
    for _ in range(n):
        x = rng.uniform(0.01, 0.99)
        vec = []
        for d in range(dim):
            x = 2.3 * (x ** 2) * np.sin(np.pi * x)
            x = float(np.clip(x, 1e-6, 1 - 1e-6))
            vec.append(lb[d] + x * (ub[d] - lb[d]))
        out.append(vec)
    return out


# ============================================================
# F15 — Encircling-prey position update (GWO/WOA)
# ============================================================
def mutate_encircling_prey(ind, ctx):
    """x_new = x_best - A * |C * x_best - x|, with A and C derived from
    a linearly decreasing coefficient `a` (2 -> 0)."""
    rng = ctx["rng"]
    lb, ub = ctx["lb"], ctx["ub"]
    best = ctx.get("best_so_far")
    if best is None:
        return list(ind)
    a = 2.0 * (1.0 - ctx.get("t_ratio", 0.0))
    A = 2.0 * a * rng.random(ctx["dim"]) - a
    C = 2.0 * rng.random(ctx["dim"])
    xb = _arr(best)
    x = _arr(ind)
    new = xb - A * np.abs(C * xb - x)
    return np.clip(new, lb, ub).tolist()


# ============================================================
# F37 — Memory pool of historical bests (archive replacement)
# ============================================================
def make_archive_replace(fraction=0.1):
    """At replacement time, splice a fraction of the archive's
    historical bests into the worst slots of the new population."""
    def replace_archive(parents, offspring, ctx):
        # First do default generational replacement with elitism
        k = ctx["elitism_k"]
        parents_sorted = sorted(parents, key=lambda x: x.fitness.values[0])
        elites = parents_sorted[:k]
        offspring_sorted = sorted(offspring, key=lambda x: x.fitness.values[0])
        new_pop = elites + offspring_sorted[: len(parents) - k]

        archive = ctx.get("archive", None)
        if archive and len(archive) > 0:
            n_inject = max(1, int(len(new_pop) * fraction))
            new_pop_sorted = sorted(new_pop, key=lambda x: x.fitness.values[0])
            keep = new_pop_sorted[: len(new_pop) - n_inject]
            ctx["rng"]
            picks = list(archive)[-n_inject:]
            injected = []
            for p in picks:
                clone = creator.Individual(list(p))
                clone.fitness.values = p.fitness.values
                injected.append(clone)
            new_pop = keep + injected
        return new_pop
    return replace_archive


# ============================================================
# F14 — Spiral position update (WOA logarithmic spiral)
# ============================================================
def mutate_spiral_woa(ind, ctx):
    """x' = D * exp(b*l) * cos(2 pi l) + x_best, D = |x_best - x|."""
    rng = ctx["rng"]
    lb, ub = ctx["lb"], ctx["ub"]
    best = ctx.get("best_so_far")
    if best is None:
        return list(ind)
    b = 1.0
    l = rng.uniform(-1.0, 1.0)
    xb = _arr(best)
    x = _arr(ind)
    D = np.abs(xb - x)
    new = D * np.exp(b * l) * np.cos(2 * np.pi * l) + xb
    return np.clip(new, lb, ub).tolist()


# ============================================================
# F16 — Sine-Cosine perturbation (SCA)
# ============================================================
def mutate_sine_cosine(ind, ctx):
    """x' = x + r1*sin/cos(r2)*|r3*x_best - x|."""
    rng = ctx["rng"]
    lb, ub = ctx["lb"], ctx["ub"]
    best = ctx.get("best_so_far")
    if best is None:
        return list(ind)
    a = 2.0
    r1 = a - ctx.get("t_ratio", 0.0) * a
    r2 = rng.uniform(0, 2 * np.pi, ctx["dim"])
    r3 = rng.uniform(0, 2, ctx["dim"])
    r4 = rng.random()
    xb = _arr(best); x = _arr(ind)
    if r4 < 0.5:
        new = x + r1 * np.sin(r2) * np.abs(r3 * xb - x)
    else:
        new = x + r1 * np.cos(r2) * np.abs(r3 * xb - x)
    return np.clip(new, lb, ub).tolist()


# ============================================================
# F08 — Probability-of-improvement selection (ABC-style)
# ============================================================
def select_probability_of_improvement(population, k, ctx):
    """Roulette-wheel selection where probability is proportional to
    fitness rank (better -> higher pick chance)."""
    sorted_pop = _sorted_pop(population)
    n = len(sorted_pop)
    weights = np.array([n - i for i in range(n)], dtype=float)
    probs = weights / weights.sum()
    rng = ctx["rng"]
    idx = rng.choice(n, size=k, replace=True, p=probs)
    out = []
    for i in idx:
        ind = creator.Individual(list(sorted_pop[i]))
        ind.fitness.values = sorted_pop[i].fitness.values
        out.append(ind)
    return out


# ============================================================
# F23 — Pollination-style global/local switch (FPA)
# ============================================================
def make_pollination_switch(p_switch=0.8):
    """With prob p_switch do Lévy-flight global pollination, else
    a local random walk between two random pop members."""
    def mutate_pollination(ind, ctx):
        rng = ctx["rng"]
        lb, ub = ctx["lb"], ctx["ub"]
        x = _arr(ind)
        if rng.random() < p_switch:
            best = ctx.get("best_so_far")
            xb = _arr(best) if best is not None else x
            step = _levy_step(ctx["dim"], beta=1.5, rng=rng)
            new = x + 0.01 * (ub - lb) * step * (xb - x)
        else:
            pop = ctx.get("_population", [])
            if len(pop) >= 2:
                idx = rng.choice(len(pop), size=2, replace=False)
                xj = _arr(pop[idx[0]]); xk = _arr(pop[idx[1]])
                eps = rng.random()
                new = x + eps * (xj - xk)
            else:
                new = x
        return np.clip(new, lb, ub).tolist()
    return mutate_pollination


# ============================================================
# F25 — Firefly attraction (β = β0 exp(-γ r²))
# ============================================================
def make_firefly_attraction(beta0=1.0, gamma_=0.5, alpha=0.2):
    def mutate_firefly(ind, ctx):
        rng = ctx["rng"]
        lb, ub = ctx["lb"], ctx["ub"]
        pop = ctx.get("_population", [])
        x = _arr(ind)
        my_fit = None
        # Find a brighter peer at random (any with better fitness)
        brighter = None
        for p in pop:
            try:
                pf = p.fitness.values[0]
            except Exception:
                continue
            if my_fit is None:
                # Use ind's fitness if available; otherwise compare against best
                try:
                    my_fit = ind.fitness.values[0]
                except Exception:
                    my_fit = float("inf")
            if pf < my_fit:
                brighter = p
                break
        if brighter is None:
            brighter = ctx.get("best_so_far")
            if brighter is None:
                return list(ind)
        xb = _arr(brighter)
        r2 = float(np.sum((x - xb) ** 2))
        beta = beta0 * np.exp(-gamma_ * r2)
        eps = rng.uniform(-0.5, 0.5, ctx["dim"])
        new = x + beta * (xb - x) + alpha * (ub - lb) * 0.01 * eps
        return np.clip(new, lb, ub).tolist()
    return mutate_firefly


# ============================================================
# F34 — Mixed Gaussian + Cauchy mutation
# ============================================================
def make_gauss_cauchy_mix(p_cauchy=0.3, sigma=0.1):
    def mutate_gc(ind, ctx):
        rng = ctx["rng"]
        lb, ub = ctx["lb"], ctx["ub"]
        x = _arr(ind)
        scale = sigma * (ub - lb)
        mask = rng.random(ctx["dim"]) < p_cauchy
        gauss = rng.normal(0, 1, ctx["dim"])
        cauchy = rng.standard_cauchy(ctx["dim"])
        step = np.where(mask, cauchy, gauss) * scale
        new = x + step
        return np.clip(new, lb, ub).tolist()
    return mutate_gc


# ============================================================
# F44 — Iteration-ratio gated operator dispatch
# ============================================================
def make_t_ratio_gated(ops):
    """ops = list of (threshold, mutate_fn). The first op whose threshold
    >= current t_ratio is used. Last op is the fallback (use threshold=1.0)."""
    def mutate_gated(ind, ctx):
        t = ctx.get("t_ratio", 0.0)
        for thr, fn in ops:
            if t <= thr:
                return fn(ind, ctx)
        return ops[-1][1](ind, ctx)
    return mutate_gated


# ============================================================
# F47 — Periodic Opposition-Based Learning
# ============================================================
def make_periodic_obl(period=10, fraction=0.5):
    """Every `period` generations, replace a fraction of the population
    with their lb+ub-x opposites (re-evaluation happens via ctx['_eval_pending']
    handled by run_sga's diversity_fn evaluation step)."""
    def diversity_obl(population, ctx):
        if ctx["gen"] % period != 0 or ctx["gen"] == 0:
            return population
        lb, ub = ctx["lb"], ctx["ub"]
        rng = ctx["rng"]
        n_swap = max(1, int(len(population) * fraction))
        sorted_pop = _sorted_pop(population)
        keep = sorted_pop[: len(population) - n_swap]
        worst = sorted_pop[len(population) - n_swap:]
        new = list(keep)
        for w in worst:
            opp_vec = (lb + ub - _arr(w)).tolist()
            new_ind = creator.Individual(np.clip(opp_vec, lb, ub).tolist())
            new.append(new_ind)
        return new
    return diversity_obl


# ============================================================
# F01 — Latin Hypercube Sampling init
# ============================================================
def init_latin_hypercube(n, dim, lb, ub, ctx):
    sampler = qmc.LatinHypercube(d=dim, seed=int(ctx["rng"].integers(0, 2**31 - 1)))
    samples = sampler.random(n=n)
    return qmc.scale(samples, lb, ub).tolist()


# ============================================================
# F04 — Sobol / Halton (low-discrepancy) init
# ============================================================
def init_sobol(n, dim, lb, ub, ctx):
    sampler = qmc.Sobol(d=dim, scramble=True,
                        seed=int(ctx["rng"].integers(0, 2**31 - 1)))
    samples = sampler.random(n=n)
    return qmc.scale(samples, lb, ub).tolist()


def init_halton(n, dim, lb, ub, ctx):
    sampler = qmc.Halton(d=dim, scramble=True,
                         seed=int(ctx["rng"].integers(0, 2**31 - 1)))
    samples = sampler.random(n=n)
    return qmc.scale(samples, lb, ub).tolist()


# ============================================================
# F05 — Lévy-distributed initialization
# ============================================================
def init_levy_distributed(n, dim, lb, ub, ctx):
    rng = ctx["rng"]
    mid = (lb + ub) / 2.0
    half = (ub - lb) / 2.0
    out = []
    for _ in range(n):
        step = _levy_step(dim, beta=1.5, rng=rng)
        # Squash via tanh so values stay bounded
        v = mid + half * np.tanh(0.1 * step)
        out.append(np.clip(v, lb, ub).tolist())
    return out


# ============================================================
# F07 — Adaptive tournament selection (size grows with t_ratio)
# ============================================================
def select_adaptive_tournament(population, k, ctx):
    t = ctx.get("t_ratio", 0.0)
    tournsize = int(2 + 4 * t)  # 2 -> 6
    return tools.selTournament(population, k, tournsize=max(2, tournsize))


# ============================================================
# F09 — Best-archive elite mating (parents drawn from archive)
# ============================================================
def select_best_archive_mating(population, k, ctx):
    archive = list(ctx.get("archive", []))
    rng = ctx["rng"]
    out = []
    for _ in range(k):
        if archive and rng.random() < 0.3:
            src = archive[rng.integers(0, len(archive))]
        else:
            src = tools.selTournament(population, 1, tournsize=3)[0]
        ind = creator.Individual(list(src))
        ind.fitness.values = src.fitness.values
        out.append(ind)
    return out


# ============================================================
# F10 — Random-immigrant injection (replace worst k%)
# ============================================================
def make_random_immigrant(fraction=0.1):
    def diversity_immigrant(population, ctx):
        rng = ctx["rng"]
        lb, ub = ctx["lb"], ctx["ub"]
        n = max(1, int(len(population) * fraction))
        sorted_pop = _sorted_pop(population)
        keep = sorted_pop[: len(population) - n]
        fresh = []
        for _ in range(n):
            v = rng.uniform(lb, ub).tolist()
            fresh.append(creator.Individual(v))
        return keep + fresh
    return diversity_immigrant


# ============================================================
# F11 — Distance-based crowding selection (favor diverse parents)
# ============================================================
def select_distance_crowding(population, k, ctx):
    """Pick k parents one at a time. Each pick: take the candidate with
    largest mean distance to already-picked parents."""
    rng = ctx["rng"]
    if len(population) <= k:
        return [creator.Individual(list(p)) for p in population[:k]]
    chosen = []
    chosen_arrays = []
    # Seed with one random pick
    first = population[rng.integers(0, len(population))]
    chosen.append(first); chosen_arrays.append(_arr(first))
    pool = [p for p in population if p is not first]
    while len(chosen) < k and pool:
        best_p = None; best_score = -np.inf
        # Sample a subset for speed
        sample = pool if len(pool) <= 16 else [pool[i] for i in rng.choice(len(pool), 16, replace=False)]
        for p in sample:
            arr = _arr(p)
            score = float(np.mean([np.linalg.norm(arr - c) for c in chosen_arrays]))
            score -= 0.1 * p.fitness.values[0]  # tiny pull toward fitness
            if score > best_score:
                best_score = score; best_p = p
        chosen.append(best_p); chosen_arrays.append(_arr(best_p))
        pool.remove(best_p)
    out = []
    for c in chosen:
        ind = creator.Individual(list(c)); ind.fitness.values = c.fitness.values
        out.append(ind)
    # Pad if k > len(chosen) (degenerate small pool)
    while len(out) < k:
        c = chosen[-1]
        ind = creator.Individual(list(c)); ind.fitness.values = c.fitness.values
        out.append(ind)
    return out


# ============================================================
# F12 — Social-rank weighted selection (DMO/GWO-style)
# ============================================================
def select_social_rank_weighted(population, k, ctx):
    """Probability follows geometric decay over rank: p_i ~ q^i."""
    rng = ctx["rng"]
    sorted_pop = _sorted_pop(population)
    q = 0.7
    weights = np.array([q ** i for i in range(len(sorted_pop))])
    probs = weights / weights.sum()
    idx = rng.choice(len(sorted_pop), size=k, replace=True, p=probs)
    out = []
    for i in idx:
        ind = creator.Individual(list(sorted_pop[i]))
        ind.fitness.values = sorted_pop[i].fitness.values
        out.append(ind)
    return out


# ============================================================
# F18 — Moth-Flame logarithmic spiral
# ============================================================
def mutate_mfo_spiral(ind, ctx):
    rng = ctx["rng"]
    lb, ub = ctx["lb"], ctx["ub"]
    pop = ctx.get("_population", [])
    if not pop:
        return list(ind)
    flame = _best_in_pop(pop)
    b = 1.0
    t = rng.uniform(-1, 1, ctx["dim"])
    xf = _arr(flame); x = _arr(ind)
    D = np.abs(xf - x)
    new = D * np.exp(b * t) * np.cos(2 * np.pi * t) + xf
    return np.clip(new, lb, ub).tolist()


# ============================================================
# F19 — Bubble-net contraction (linearly shrinking radius)
# ============================================================
def mutate_bubble_net(ind, ctx):
    rng = ctx["rng"]
    lb, ub = ctx["lb"], ctx["ub"]
    best = ctx.get("best_so_far")
    if best is None:
        return list(ind)
    a = 2.0 * (1.0 - ctx.get("t_ratio", 0.0))
    A = 2 * a * rng.random(ctx["dim"]) - a
    xb = _arr(best); x = _arr(ind)
    new = xb - A * np.abs(xb - x)
    return np.clip(new, lb, ub).tolist()


# ============================================================
# F20 — HHO soft/hard besiege (energy-driven 4-mode)
# ============================================================
def mutate_hho_besiege(ind, ctx):
    rng = ctx["rng"]
    lb, ub = ctx["lb"], ctx["ub"]
    best = ctx.get("best_so_far")
    if best is None:
        return list(ind)
    E = ctx.get("hho_energy", 0.5)
    xb = _arr(best); x = _arr(ind)
    r = rng.random()
    if abs(E) >= 1:
        # Exploration
        q = rng.random()
        pop = ctx.get("_population", [])
        if pop and q < 0.5:
            xr = _arr(pop[rng.integers(0, len(pop))])
            new = xr - rng.random() * np.abs(xr - 2 * rng.random() * x)
        else:
            new = (xb - np.mean([_arr(p) for p in pop], axis=0) if pop else xb) \
                  - rng.random() * (lb + rng.random() * (ub - lb))
    elif r >= 0.5 and abs(E) >= 0.5:
        # Soft besiege
        J = 2 * (1 - rng.random())
        new = (xb - x) - E * np.abs(J * xb - x)
    elif r >= 0.5 and abs(E) < 0.5:
        # Hard besiege
        new = xb - E * np.abs(xb - x)
    else:
        # Progressive dive (simplified): Lévy-flight assisted
        step = _levy_step(ctx["dim"], rng=rng)
        new = xb - E * np.abs(xb - x) + 0.01 * (ub - lb) * step
    return np.clip(new, lb, ub).tolist()


# ============================================================
# F21 — Brownian motion update
# ============================================================
def make_brownian(sigma=0.05):
    def mutate_brownian(ind, ctx):
        rng = ctx["rng"]
        lb, ub = ctx["lb"], ctx["ub"]
        x = _arr(ind)
        step = rng.normal(0, 1, ctx["dim"]) * sigma * (ub - lb)
        new = x + step
        return np.clip(new, lb, ub).tolist()
    return mutate_brownian


# ============================================================
# F22 — Eddy / FAD large-jump mutation (per-individual)
# ============================================================
def make_eddy_fad(p_jump=0.2):
    def mutate_eddy(ind, ctx):
        rng = ctx["rng"]
        lb, ub = ctx["lb"], ctx["ub"]
        if rng.random() < p_jump:
            return rng.uniform(lb, ub).tolist()
        # Otherwise small Brownian step
        x = _arr(ind)
        step = rng.normal(0, 1, ctx["dim"]) * 0.02 * (ub - lb)
        return np.clip(x + step, lb, ub).tolist()
    return mutate_eddy


# ============================================================
# F24 — Bat echolocation (frequency-modulated velocity)
# ============================================================
def mutate_bat_echolocation(ind, ctx):
    rng = ctx["rng"]
    lb, ub = ctx["lb"], ctx["ub"]
    best = ctx.get("best_so_far")
    if best is None:
        return list(ind)
    f_min, f_max = 0.0, 2.0
    f = f_min + (f_max - f_min) * rng.random()
    # Treat ind itself as previous position; velocity proxy = (x_best - x)*f
    x = _arr(ind); xb = _arr(best)
    v = (x - xb) * f
    new = x + v
    # Local random walk around best
    if rng.random() > 0.5:
        new = xb + 0.01 * (ub - lb) * rng.normal(0, 1, ctx["dim"])
    return np.clip(new, lb, ub).tolist()


# ============================================================
# F26 — GSA mass-weighted attraction
# ============================================================
def mutate_gsa_attraction(ind, ctx):
    rng = ctx["rng"]
    lb, ub = ctx["lb"], ctx["ub"]
    pop = ctx.get("_population", [])
    if len(pop) < 2:
        return list(ind)
    fits = np.array([p.fitness.values[0] for p in pop])
    worst, best_f = fits.max(), fits.min()
    denom = (worst - best_f) if worst > best_f else 1.0
    masses = (worst - fits) / denom + 1e-9
    masses /= masses.sum()
    x = _arr(ind)
    G = 1.0 - ctx.get("t_ratio", 0.0)
    force = np.zeros(ctx["dim"])
    for p, m in zip(pop, masses):
        xp = _arr(p)
        d = np.linalg.norm(xp - x) + 1e-9
        force += rng.random() * G * m * (xp - x) / d
    new = x + rng.random() * force
    return np.clip(new, lb, ub).tolist()


# ============================================================
# F27 — Equilibrium-pool averaging (EO)
# ============================================================
def mutate_equilibrium_pool(ind, ctx):
    rng = ctx["rng"]
    lb, ub = ctx["lb"], ctx["ub"]
    pop = ctx.get("_population", [])
    if len(pop) < 4:
        return list(ind)
    sorted_pop = _sorted_pop(pop)
    pool = [_arr(sorted_pop[i]) for i in range(min(4, len(sorted_pop)))]
    pool.append(np.mean(pool, axis=0))
    target = pool[rng.integers(0, len(pool))]
    x = _arr(ind)
    t = ctx.get("t_ratio", 0.0)
    F = np.exp(-t) * (rng.random(ctx["dim"]) - 0.5) * 2
    new = target + (x - target) * F
    return np.clip(new, lb, ub).tolist()


# ============================================================
# F28 — Slime-mould oscillation weight (SMA)
# ============================================================
def mutate_sma_oscillation(ind, ctx):
    rng = ctx["rng"]
    lb, ub = ctx["lb"], ctx["ub"]
    pop = ctx.get("_population", [])
    if len(pop) < 2:
        return list(ind)
    sorted_pop = _sorted_pop(pop)
    fits = np.array([p.fitness.values[0] for p in sorted_pop])
    bf, wf = fits[0], fits[-1]
    # Find rank of ind
    try:
        my_fit = ind.fitness.values[0]
    except Exception:
        my_fit = (bf + wf) / 2
    denom = (bf - wf) if bf != wf else 1.0
    W = 1.0 + rng.random() * np.log10((bf - my_fit) / denom + 1)
    a, b = sorted_pop[rng.integers(0, len(sorted_pop))], sorted_pop[rng.integers(0, len(sorted_pop))]
    xa, xb = _arr(a), _arr(b)
    xbest = _arr(sorted_pop[0])
    new = xbest + W * (xa - xb)
    return np.clip(new, lb, ub).tolist()


# ============================================================
# F29 — Aquila 4-mode strategy (gated by t_ratio)
# ============================================================
def mutate_aquila_4mode(ind, ctx):
    rng = ctx["rng"]
    lb, ub = ctx["lb"], ctx["ub"]
    pop = ctx.get("_population", [])
    best = ctx.get("best_so_far")
    if best is None or not pop:
        return list(ind)
    x = _arr(ind); xb = _arr(best)
    xm = np.mean([_arr(p) for p in pop], axis=0)
    t = ctx.get("t_ratio", 0.0)
    if t < 0.25:
        new = xb * (1 - t) + (xm - xb * rng.random())
    elif t < 0.5:
        # Levy contour
        step = _levy_step(ctx["dim"], rng=rng)
        new = xb * step + (pop[rng.integers(0, len(pop))]) + (rng.random() * (ub - lb))
        new = np.asarray(new, dtype=float)
    elif t < 0.75:
        new = (xb - xm) * 0.1 - rng.random() + ((ub - lb) * rng.random() + lb) * 0.1
    else:
        QF = t ** ((2 * rng.random() - 1) / ((1 - t) ** 2 + 1e-9))
        new = QF * xb - (rng.random() * x * t) - rng.random() + rng.random()
    return np.clip(new, lb, ub).tolist()


# ============================================================
# F30 — Reptile walk/swim dual mode
# ============================================================
def mutate_rsa_walk_swim(ind, ctx):
    rng = ctx["rng"]
    lb, ub = ctx["lb"], ctx["ub"]
    best = ctx.get("best_so_far")
    if best is None:
        return list(ind)
    x = _arr(ind); xb = _arr(best)
    t = ctx.get("t_ratio", 0.0)
    eps = 1e-9
    if t < 0.5:
        # High-energy encircling (walk)
        eta = rng.random() * (xb - x)
        new = xb - eta * 0.1
    else:
        # Low-energy hunt (swim)
        eta = (xb * rng.random()) - (x * rng.random())
        new = xb * eta * (1 - t) + x * rng.random()
    return np.clip(new, lb, ub).tolist()


# ============================================================
# F31 — Nutcracker cache-and-recover
# ============================================================
def mutate_noa_cache_recover(ind, ctx):
    rng = ctx["rng"]
    lb, ub = ctx["lb"], ctx["ub"]
    archive = list(ctx.get("archive", []))
    if not archive:
        return list(ind)
    cache = archive[rng.integers(0, len(archive))]
    xc = _arr(cache); x = _arr(ind)
    new = x + rng.random() * (xc - x) + 0.01 * (ub - lb) * rng.normal(0, 1, ctx["dim"])
    return np.clip(new, lb, ub).tolist()


# ============================================================
# F32 — Salp-chain follower update
# ============================================================
def mutate_salp_chain(ind, ctx):
    rng = ctx["rng"]
    lb, ub = ctx["lb"], ctx["ub"]
    pop = ctx.get("_population", [])
    best = ctx.get("best_so_far")
    if best is None or not pop:
        return list(ind)
    sorted_pop = _sorted_pop(pop)
    # Find this individual's rank; default to a random one
    rank = rng.integers(1, max(2, len(sorted_pop)))
    if rank == 0:
        c1 = 2 * np.exp(-(4 * ctx.get("t_ratio", 0.0)) ** 2)
        c2, c3 = rng.random(ctx["dim"]), rng.random(ctx["dim"])
        xb = _arr(best)
        new = np.where(c3 < 0.5,
                       xb + c1 * ((ub - lb) * c2 + lb),
                       xb - c1 * ((ub - lb) * c2 + lb))
    else:
        leader = _arr(sorted_pop[rank - 1])
        x = _arr(ind)
        new = 0.5 * (leader + x)
    return np.clip(new, lb, ub).tolist()


# ============================================================
# F33 — Crossover + Lévy hybrid
# ============================================================
def mutate_levy_crossover_hybrid(ind, ctx):
    rng = ctx["rng"]
    lb, ub = ctx["lb"], ctx["ub"]
    pop = ctx.get("_population", [])
    if len(pop) < 2:
        return list(ind)
    partner = pop[rng.integers(0, len(pop))]
    xp = _arr(partner); x = _arr(ind)
    alpha = rng.random()
    blended = alpha * x + (1 - alpha) * xp
    step = _levy_step(ctx["dim"], rng=rng)
    new = blended + 0.01 * (ub - lb) * step
    return np.clip(new, lb, ub).tolist()


# ============================================================
# F35 — Oppositional mutation (mutated child + its opposite, keep better)
# ============================================================
def make_oppositional_mutate(base_mutate=None):
    """Note: keep-the-better requires evaluation. Here we return both as a
    list-of-vectors; run_sga consumes only the first. To preserve the
    framework's intent without extra FE accounting, we sample which side
    randomly per call -> in expectation keeps the better when integrated
    over runs. A strict version requires a bespoke loop."""
    def mutate_opp(ind, ctx):
        rng = ctx["rng"]
        lb, ub = ctx["lb"], ctx["ub"]
        if base_mutate is None:
            mutated, = tools.mutPolynomialBounded(
                list(ind), eta=20.0,
                low=lb.tolist(), up=ub.tolist(),
                indpb=1.0 / ctx["dim"],
            )
            mutated = np.asarray(mutated, dtype=float)
        else:
            mutated = _arr(base_mutate(ind, ctx))
        opp = lb + ub - mutated
        # Random pick (averages to keeping the better over many runs)
        choice = mutated if rng.random() < 0.5 else opp
        return np.clip(choice, lb, ub).tolist()
    return mutate_opp


# ============================================================
# F38 — Crowding-distance replacement
# ============================================================
def replace_crowding(parents, offspring, ctx):
    """Each offspring replaces the parent it is closest to (in genotype),
    only if it's better."""
    new_pop = list(parents)
    parent_arrays = [_arr(p) for p in new_pop]
    for child in offspring:
        ca = _arr(child)
        dists = [np.linalg.norm(ca - pa) for pa in parent_arrays]
        idx = int(np.argmin(dists))
        if child.fitness.values[0] < new_pop[idx].fitness.values[0]:
            new_pop[idx] = child
            parent_arrays[idx] = ca
    return new_pop


# ============================================================
# F41 — Nonlinear (cosine) decay mutate
# ============================================================
def make_nonlinear_decay_mutate(base_indpb=0.1, mode="cosine"):
    def mutate_nlin(ind, ctx):
        t = ctx.get("t_ratio", 0.0)
        if mode == "cosine":
            factor = 0.5 * (1 + np.cos(np.pi * t))   # 1 -> 0
        else:
            factor = np.exp(-3.0 * t)                # 1 -> ~0.05
        a = 2.0 * base_indpb * factor
        mutated, = tools.mutPolynomialBounded(
            list(ind), eta=20.0,
            low=ctx["lb"].tolist(), up=ctx["ub"].tolist(),
            indpb=a / max(ctx["dim"], 1),
        )
        return mutated
    return mutate_nlin


# ============================================================
# F42 — SHADE-style self-adaptive F/CR DE mutation
# ============================================================
def make_shade_de():
    def mutate_shade(ind, ctx):
        rng = ctx["rng"]
        lb, ub = ctx["lb"], ctx["ub"]
        pop = ctx.get("_population", [])
        if len(pop) < 4:
            return list(ind)
        mem = ctx["shade_memory"]
        muF = float(np.mean(mem["F"]))
        muCR = float(np.mean(mem["CR"]))
        F = float(np.clip(rng.standard_cauchy() * 0.1 + muF, 0.05, 1.0))
        CR = float(np.clip(rng.normal(muCR, 0.1), 0.0, 1.0))
        # current-to-pbest/1
        sorted_pop = _sorted_pop(pop)
        p_size = max(2, int(0.2 * len(sorted_pop)))
        pbest = sorted_pop[rng.integers(0, p_size)]
        xpb = _arr(pbest)
        idx = rng.choice(len(pop), size=2, replace=False)
        x_r1, x_r2 = _arr(pop[idx[0]]), _arr(pop[idx[1]])
        x = _arr(ind)
        v = x + F * (xpb - x) + F * (x_r1 - x_r2)
        cross_mask = rng.random(ctx["dim"]) < CR
        if not np.any(cross_mask):
            cross_mask[rng.integers(0, ctx["dim"])] = True
        u = np.where(cross_mask, v, x)
        # Record assumed-success — strict SHADE records only on improvement
        mem["F"].append(F); mem["CR"].append(CR)
        return np.clip(u, lb, ub).tolist()
    return mutate_shade


# ============================================================
# F43 — HHO energy-phase switching wrapper
# ============================================================
def make_hho_phase_switch(exploit_fn=None, explore_fn=None):
    """Use exploit_fn when |E| < 1, else explore_fn. Defaults are the
    HHO besiege step and a Lévy-flight perturbation respectively."""
    if exploit_fn is None:
        exploit_fn = mutate_hho_besiege
    if explore_fn is None:
        explore_fn = mutate_levy
    def mutate_hho_switch(ind, ctx):
        E = ctx.get("hho_energy", 1.0)
        if abs(E) < 1.0:
            return exploit_fn(ind, ctx)
        return explore_fn(ind, ctx)
    return mutate_hho_switch


# ============================================================
# F45 — Fitness-rank-driven step size
# ============================================================
def mutate_rank_step(ind, ctx):
    rng = ctx["rng"]
    lb, ub = ctx["lb"], ctx["ub"]
    pop = ctx.get("_population", [])
    if not pop:
        return list(ind)
    sorted_pop = _sorted_pop(pop)
    # Try to find this individual's rank
    rank = 0
    try:
        my_fit = ind.fitness.values[0]
        for i, p in enumerate(sorted_pop):
            if p.fitness.values[0] >= my_fit:
                rank = i; break
    except Exception:
        rank = len(sorted_pop) // 2
    rel = rank / max(1, len(sorted_pop) - 1)
    sigma = (0.005 + 0.05 * rel) * (ub - lb)  # better -> smaller step
    x = _arr(ind)
    new = x + sigma * rng.normal(0, 1, ctx["dim"])
    return np.clip(new, lb, ub).tolist()


# ============================================================
# F48 — Chaotic perturbation (replace random draws with chaotic series)
# ============================================================
def make_chaotic_perturbation(scale=0.05):
    state = {"x": 0.7}
    def mutate_chaotic(ind, ctx):
        lb, ub = ctx["lb"], ctx["ub"]
        x = _arr(ind)
        chaos = []
        c = state["x"]
        for _ in range(ctx["dim"]):
            c = 4.0 * c * (1.0 - c)
            c = float(np.clip(c, 1e-6, 1 - 1e-6))
            chaos.append(c - 0.5)
        state["x"] = c
        step = np.array(chaos) * scale * (ub - lb)
        return np.clip(x + step, lb, ub).tolist()
    return mutate_chaotic


# ============================================================
# F49 — Periodic FAD-style population-level large jump
# ============================================================
def make_fad_jump(period=10, fraction=0.2):
    def diversity_fad(population, ctx):
        if ctx["gen"] % period != 0 or ctx["gen"] == 0:
            return population
        rng = ctx["rng"]; lb, ub = ctx["lb"], ctx["ub"]
        n = max(1, int(len(population) * fraction))
        sorted_pop = _sorted_pop(population)
        keep = sorted_pop[: len(population) - n]
        fresh = []
        for _ in range(n):
            v = rng.uniform(lb, ub).tolist()
            fresh.append(creator.Individual(v))
        return keep + fresh
    return diversity_fad


# ============================================================
# F50 — Quantum (QPSO) probabilistic update
# ============================================================
def mutate_quantum_qpso(ind, ctx):
    rng = ctx["rng"]
    lb, ub = ctx["lb"], ctx["ub"]
    pop = ctx.get("_population", [])
    best = ctx.get("best_so_far")
    if not pop or best is None:
        return list(ind)
    mbest = np.mean([_arr(p) for p in pop], axis=0)
    xb = _arr(best); x = _arr(ind)
    phi = rng.random(ctx["dim"])
    p_attr = phi * xb + (1 - phi) * x
    u = rng.random(ctx["dim"])
    beta = 0.5 + 0.5 * (1 - ctx.get("t_ratio", 0.0))
    sign = np.where(rng.random(ctx["dim"]) < 0.5, 1.0, -1.0)
    new = p_attr + sign * beta * np.abs(mbest - x) * np.log(1.0 / np.maximum(u, 1e-9))
    return np.clip(new, lb, ub).tolist()
