"""
features.py
-----------
Drop-in implementations of nature-inspired algorithm features.
Each function matches the signature expected by `FeatureConfig` in sga_core.py.

Three families, all numbered in a single F-series:
  - F01–F50: nature-inspired (chaotic, Lévy/DE/SHADE, GWO/WOA/HHO/SCA/FPA,
    firefly/GSA/EO/SMA/Aquila/RSA/NOA/salp/MFO/bat/QPSO, etc.)
  - F51–F62: artificial immune system (CLONALG, opt-IA, opt-aiNet,
    negative selection, danger theory, HFIA)
  - F63–F72: ant-colony / pheromone (binned per-dim pheromone matrix;
    AS / EAS / MMAS / ASrank / heuristic / candidate-list)

References for each feature are in the framework PDF (Section 5).
"""

import math
import numpy as np
from deap import tools, creator
from scipy.special import gamma
from scipy.stats import qmc

from ga.core import generational_elitism


# ============================================================
# F03 — Opposition-Based Initialization (Tizhoosh 2005)
# ============================================================
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
                src = alpha
            elif r < 0.8:
                src = beta
            else:
                src = delta
            ind = creator.Individual(list(src))
            ind.fitness.values = src.fitness.values
            selected.append(ind)
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
        new_pop = generational_elitism(parents, offspring, ctx["elitism_k"])

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
    # Normalize fitness across the population so the fitness term is on the
    # same scale as the distance term regardless of CEC function magnitude.
    fits = np.array([p.fitness.values[0] for p in population], dtype=float)
    f_min, f_max = float(fits.min()), float(fits.max())
    f_span = (f_max - f_min) if (f_max > f_min) else 1.0
    rng_norm = float(np.linalg.norm(np.asarray(ctx["ub"]) - np.asarray(ctx["lb"])))
    fit_pull = 0.1 * rng_norm  # ≤ 10% of max possible distance
    chosen = []
    chosen_arrays = []
    # Seed with one random pick
    first_idx = int(rng.integers(0, len(population)))
    first = population[first_idx]
    chosen.append(first); chosen_arrays.append(_arr(first))
    pool = [p for p in population if p is not first]
    while len(chosen) < k and pool:
        best_p = None; best_score = -np.inf
        # Sample a subset for speed
        sample = pool if len(pool) <= 16 else [pool[i] for i in rng.choice(len(pool), 16, replace=False)]
        for p in sample:
            arr = _arr(p)
            score = float(np.mean([np.linalg.norm(arr - c) for c in chosen_arrays]))
            f_norm = (p.fitness.values[0] - f_min) / f_span  # 0=best, 1=worst
            score -= fit_pull * f_norm
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
    """Approximation of OBL-mutate without extra FEs: produce a mutated
    candidate and its opposite, then return one chosen uniformly at random.
    A strict version (evaluate both, keep better) would require a bespoke
    loop and consume an extra FE per call."""
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
    def mutate_chaotic(ind, ctx):
        lb, ub = ctx["lb"], ctx["ub"]
        x = _arr(ind)
        state = ctx.setdefault("_chaotic_perturb_state", {"x": 0.7})
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
def _norm_fitness_rank(pop, ind):
    """Return rank of `ind` within `pop` normalised to [0,1] (0 = best)."""
    sorted_pop = _sorted_pop(pop)
    n = len(sorted_pop)
    try:
        my_fit = ind.fitness.values[0]
    except Exception:
        return 0.5
    rank = 1.0
    for i, p in enumerate(sorted_pop):
        if p.fitness.values[0] >= my_fit:
            rank = i / max(1, n - 1)
            break
    assert 0.0 <= rank <= 1.0, f"_norm_fitness_rank produced {rank}"
    return rank


# ============================================================
# F51 — Inversely Proportional Hypermutation (CLONALG)
# ============================================================
def make_inverse_hypermutate(rho=5.0, sigma_max=0.2):
    """Step size shrinks for fitter individuals: sigma ~ sigma_max*exp(-rho*r),
    where r is the normalised rank (0=best, 1=worst)."""
    def mutate_inverse(ind, ctx):
        rng = ctx["rng"]
        lb, ub = ctx["lb"], ctx["ub"]
        pop = ctx.get("_population", [])
        r = _norm_fitness_rank(pop, ind) if pop else 0.5
        # Better individuals (small r) should mutate LESS, so we use exp(-rho*(1-r))
        # Per CLONALG the convention is 'better antibody -> smaller step size'.
        sigma = sigma_max * np.exp(-rho * (1.0 - r))
        x = _arr(ind)
        step = rng.normal(0, 1, ctx["dim"]) * sigma * (ub - lb)
        return np.clip(x + step, lb, ub).tolist()
    return mutate_inverse


# ============================================================
# F52 — Clonal Expansion (better individuals get more clones)
# ============================================================
def select_clonal_expansion(population, k, ctx):
    """Top half gets 2x copies, next quarter gets 1x, bottom quarter discarded.
    Truncate or repeat to exactly k individuals."""
    sorted_pop = _sorted_pop(population)
    n = len(sorted_pop)
    top_half = sorted_pop[: n // 2]
    next_quarter = sorted_pop[n // 2: 3 * n // 4]
    pool = []
    for ind in top_half:
        pool.append(ind); pool.append(ind)
    pool.extend(next_quarter)
    if len(pool) < k:
        # pad with cycling top
        i = 0
        while len(pool) < k:
            pool.append(top_half[i % max(1, len(top_half))])
            i += 1
    pool = pool[:k]
    out = []
    for c in pool:
        ind = creator.Individual(list(c))
        ind.fitness.values = c.fitness.values
        if hasattr(c, "age"):
            ind.age = c.age
        out.append(ind)
    return out


# ============================================================
# F53 — Aging / Lifespan removal (opt-IA)
# ============================================================
def make_aging_replace(tau=20):
    """Drop any individual whose .age >= tau and refill with random."""
    def diversity_aging(population, ctx):
        rng = ctx["rng"]
        lb, ub = ctx["lb"], ctx["ub"]
        survivors = [ind for ind in population if getattr(ind, "age", 0) < tau]
        n_drop = len(population) - len(survivors)
        if n_drop == 0:
            return population
        fresh = []
        for _ in range(n_drop):
            v = rng.uniform(lb, ub).tolist()
            fresh.append(creator.Individual(v))
        return survivors + fresh
    return diversity_aging


# ============================================================
# F54 — Receptor Editing (every-gen replace worst k%)
# ============================================================
def make_receptor_editing(fraction=0.1):
    """Differs from F39 (stagnation-triggered) by firing every generation."""
    def diversity_receptor(population, ctx):
        rng = ctx["rng"]
        lb, ub = ctx["lb"], ctx["ub"]
        n_drop = max(1, int(len(population) * fraction))
        sorted_pop = _sorted_pop(population)
        keep = sorted_pop[: len(population) - n_drop]
        fresh = []
        for _ in range(n_drop):
            v = rng.uniform(lb, ub).tolist()
            fresh.append(creator.Individual(v))
        return keep + fresh
    return diversity_receptor


# ============================================================
# F55 — Negative Selection (push away from known optima)
# ============================================================
def make_negative_selection(threshold=0.05, push=0.1):
    """If the mutated point is closer than `threshold * range` to any
    archived best, displace it radially away by `push * range`."""
    def mutate_neg_select(ind, ctx):
        rng = ctx["rng"]
        lb, ub = ctx["lb"], ctx["ub"]
        # base mutation (polynomial)
        base, = tools.mutPolynomialBounded(
            list(ind), eta=20.0,
            low=lb.tolist(), up=ub.tolist(),
            indpb=1.0 / ctx["dim"],
        )
        x = _arr(base)
        archive = list(ctx.get("archive", []))
        if not archive:
            return base
        rng_span = ub - lb
        thr = threshold * np.linalg.norm(rng_span)
        for tabu in archive:
            ta = _arr(tabu)
            d_vec = x - ta
            d = float(np.linalg.norm(d_vec))
            if d < thr and d > 1e-12:
                x = x + (push * rng_span * d_vec / d)
            elif d <= 1e-12:
                x = x + 0.1 * rng_span * rng.normal(0, 1, ctx["dim"])
        return np.clip(x, lb, ub).tolist()
    return mutate_neg_select


# ============================================================
# F56 — Affinity-Based Gaussian Mutation (opt-aiNet)
# ============================================================
def make_affinity_gaussian(sigma_max=0.2, rho=2.0):
    """opt-aiNet affinity: best individuals mutate the LEAST.
    sigma = sigma_max * exp(-rho * (1 - r)), where r is rank-normalized
    in [0,1] with 0 = best. So r=0 -> sigma_max*exp(-rho) (smallest step),
    r=1 -> sigma_max (largest step). Matches F51 CLONALG convention."""
    def mutate_affinity(ind, ctx):
        rng = ctx["rng"]
        lb, ub = ctx["lb"], ctx["ub"]
        pop = ctx.get("_population", [])
        r = _norm_fitness_rank(pop, ind) if pop else 0.5
        sigma = sigma_max * np.exp(-rho * (1.0 - r))
        x = _arr(ind)
        step = rng.normal(0, 1, ctx["dim"]) * sigma * (ub - lb)
        return np.clip(x + step, lb, ub).tolist()
    return mutate_affinity


# ============================================================
# F57 — Immune Memory Pool (diversity-aware archive injection)
# ============================================================
def make_immune_memory_pool(k_inject=2, dist_thresh_frac=0.05):
    """Maintain a separate diverse archive in ctx['immune_memory'] and
    inject k_inject members each generation into the worst slots.
    Membership rule: a candidate joins if its distance to all current
    members exceeds dist_thresh_frac * range_norm and pool < capacity.
    """
    def diversity_immune(population, ctx):
        lb, ub = ctx["lb"], ctx["ub"]
        memory = ctx.setdefault("immune_memory", [])
        # Update memory with current best if it's "novel enough"
        rng_norm = float(np.linalg.norm(ub - lb))
        thr = dist_thresh_frac * rng_norm
        cur_best = _best_in_pop(population)
        ca = _arr(cur_best)
        if not memory:
            mem_clone = creator.Individual(list(cur_best))
            mem_clone.fitness.values = cur_best.fitness.values
            memory.append(mem_clone)
        else:
            min_d = min(np.linalg.norm(ca - _arr(m)) for m in memory)
            if min_d > thr and len(memory) < 20:
                mem_clone = creator.Individual(list(cur_best))
                mem_clone.fitness.values = cur_best.fitness.values
                memory.append(mem_clone)
        # Inject from memory into worst slots
        if not memory:
            return population
        sorted_pop = _sorted_pop(population)
        n_inject = min(k_inject, len(memory), len(population) // 2)
        keep = sorted_pop[: len(population) - n_inject]
        injected = []
        for m in memory[-n_inject:]:
            clone = creator.Individual(list(m))
            clone.fitness.values = m.fitness.values
            injected.append(clone)
        return keep + injected
    return diversity_immune


# ============================================================
# F58 — Adaptive Population Resizing
# ============================================================
def make_adaptive_resize(min_n=20, max_n=100, target_div_frac=0.05):
    """If Hussain diversity < target -> grow (add random); if > 2*target -> shrink (drop worst).
    Pop size is clamped to [min_n, max_n]."""
    def replace_adaptive(parents, offspring, ctx):
        new_pop = generational_elitism(parents, offspring, ctx["elitism_k"])
        # diversity measure
        positions = np.array([list(p) for p in new_pop], dtype=float)
        if positions.size == 0:
            return new_pop
        med = np.median(positions, axis=0)
        div = float(np.mean(np.mean(np.abs(positions - med), axis=1)))
        rng_norm = float(np.linalg.norm(ctx["ub"] - ctx["lb"]))
        target = target_div_frac * rng_norm
        rng = ctx["rng"]
        lb, ub = ctx["lb"], ctx["ub"]
        if div < target and len(new_pop) < max_n:
            n_add = min(max(1, len(new_pop) // 10), max_n - len(new_pop))
            for _ in range(n_add):
                v = rng.uniform(lb, ub).tolist()
                new_pop.append(creator.Individual(v))
        elif div > 2 * target and len(new_pop) > min_n:
            n_drop = min(max(1, len(new_pop) // 10), len(new_pop) - min_n)
            new_pop = sorted(new_pop, key=lambda x: x.fitness.values[0])[: len(new_pop) - n_drop]
        return new_pop
    return replace_adaptive


# ============================================================
# F59 — Combinatorial Recombination (multi-parent crossover)
# ============================================================
def crossover_multiparent(p1, p2, ctx):
    """Blend p1, p2, and one random extra from population per dim coordinate."""
    rng = ctx["rng"]
    lb, ub = ctx["lb"], ctx["ub"]
    pop = ctx.get("_population", [])
    if pop:
        p3 = pop[rng.integers(0, len(pop))]
    else:
        p3 = p1
    a, b, c = _arr(p1), _arr(p2), _arr(p3)
    mask1 = rng.random(ctx["dim"])
    mask2 = rng.random(ctx["dim"])
    mask2 = mask2 / (mask1 + mask2 + 1e-9)
    mask1 = 1.0 - mask2
    # Convex combination per dim across three parents
    w1 = rng.random(ctx["dim"]); w2 = rng.random(ctx["dim"]); w3 = rng.random(ctx["dim"])
    s = w1 + w2 + w3
    w1, w2, w3 = w1 / s, w2 / s, w3 / s
    c1 = w1 * a + w2 * b + w3 * c
    c2 = w3 * a + w1 * b + w2 * c
    return (np.clip(c1, lb, ub).tolist(), np.clip(c2, lb, ub).tolist())


# ============================================================
# F60 — Immune Network Suppression (remove too-similar members)
# ============================================================
def make_network_suppression(threshold_frac=0.02):
    """If two individuals are within threshold (Euclidean), drop the worse
    and refill with a random replacement."""
    def diversity_suppress(population, ctx):
        lb, ub = ctx["lb"], ctx["ub"]
        rng = ctx["rng"]
        rng_norm = float(np.linalg.norm(ub - lb))
        thr = threshold_frac * rng_norm
        pop = list(population)
        arrays = [_arr(p) for p in pop]
        keep_mask = [True] * len(pop)
        for i in range(len(pop)):
            if not keep_mask[i]:
                continue
            for j in range(i + 1, len(pop)):
                if not keep_mask[j]:
                    continue
                if np.linalg.norm(arrays[i] - arrays[j]) < thr:
                    # Drop the worse
                    if pop[i].fitness.values[0] <= pop[j].fitness.values[0]:
                        keep_mask[j] = False
                    else:
                        keep_mask[i] = False; break
        survivors = [pop[i] for i in range(len(pop)) if keep_mask[i]]
        n_replace = len(pop) - len(survivors)
        for _ in range(n_replace):
            v = rng.uniform(lb, ub).tolist()
            survivors.append(creator.Individual(v))
        return survivors
    return diversity_suppress


# ============================================================
# F61 — Danger Theory Signaling (boost step on stagnation)
# ============================================================
def make_danger_signal(stag_threshold=10, boost=5.0):
    """When stagnation_counter exceeds threshold, multiply mutation step
    size by `boost` to escape the basin."""
    def mutate_danger(ind, ctx):
        rng = ctx["rng"]
        lb, ub = ctx["lb"], ctx["ub"]
        x = _arr(ind)
        scale = 0.05
        if ctx.get("stagnation_counter", 0) >= stag_threshold:
            scale *= boost
        step = rng.normal(0, 1, ctx["dim"]) * scale * (ub - lb)
        return np.clip(x + step, lb, ub).tolist()
    return mutate_danger


# ============================================================
# F62 — Cauchy Hypermutation on elites (HFIA)
# ============================================================
def make_cauchy_hypermutate_elite(elite_frac=0.2, scale=0.1):
    """Top elite_frac of the population gets a heavy-tailed Cauchy step;
    everyone else falls back to standard polynomial mutation."""
    def mutate_cauchy_elite(ind, ctx):
        rng = ctx["rng"]
        lb, ub = ctx["lb"], ctx["ub"]
        pop = ctx.get("_population", [])
        r = _norm_fitness_rank(pop, ind) if pop else 0.5
        if r <= elite_frac:
            x = _arr(ind)
            step = rng.standard_cauchy(ctx["dim"]) * scale * (ub - lb)
            return np.clip(x + step, lb, ub).tolist()
        else:
            mutated, = tools.mutPolynomialBounded(
                list(ind), eta=20.0,
                low=lb.tolist(), up=ub.tolist(),
                indpb=1.0 / ctx["dim"],
            )
            return mutated
    return mutate_cauchy_elite


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


# ============================================================
# ANT COLONY / PHEROMONE FEATURES (F63–F72) — continuation of F-series
# ============================================================
# Continuous-domain adaptation: per-dimension binned pheromone matrix.
# A (dim, K) array `tau` lives in ctx["pheromone"]; each row is a dimension
# and each column is a discretized bin of [lb_d, ub_d]. All ACO features
# share this matrix via ctx.setdefault, so they compose if multiple are
# active, but each is also self-contained: it lazy-inits the matrix and
# runs evaporation + deposit once per generation (gated by a sentinel).

def _pheromone_init(ctx, K=20, tau0=1.0, tau_min=0.01, tau_max=10.0):
    if "pheromone" in ctx:
        return ctx["pheromone"]
    state = {
        "tau": np.full((ctx["dim"], K), tau0, dtype=float),
        "K": K,
        "tau_min": tau_min,
        "tau_max": tau_max,
    }
    ctx["pheromone"] = state
    ctx["_pheromone_last_gen"] = -1
    return state


def _bin_index_scalar(x, lb_d, ub_d, K):
    span = (ub_d - lb_d) if (ub_d - lb_d) > 0 else 1.0
    frac = (float(x) - float(lb_d)) / float(span)
    return int(np.clip(int(frac * K), 0, K - 1))


def _bin_centers(lb, ub, K):
    edges = np.linspace(0.0, 1.0, K + 1)
    mids = 0.5 * (edges[:-1] + edges[1:])
    lb_a = np.asarray(lb, dtype=float)
    ub_a = np.asarray(ub, dtype=float)
    return lb_a[:, None] + mids[None, :] * (ub_a - lb_a)[:, None]


def _deposit_individual(tau, ind_vec, amount, lb, ub, K):
    lb_a = np.asarray(lb, dtype=float)
    ub_a = np.asarray(ub, dtype=float)
    span = ub_a - lb_a
    span = np.where(span <= 0, 1.0, span)
    fracs = (np.asarray(ind_vec, dtype=float) - lb_a) / span
    bins = np.clip((fracs * K).astype(int), 0, K - 1)
    for d, b in enumerate(bins):
        tau[d, int(b)] += amount


def _sample_dim_value(tau_row, lb_d, ub_d, K, rng,
                      alpha=1.0, eta_row=None, beta=0.0):
    w = np.asarray(tau_row, dtype=float) ** alpha
    if eta_row is not None and beta != 0.0:
        w = w * (np.asarray(eta_row, dtype=float) ** beta)
    w_sum = float(w.sum())
    if not np.isfinite(w_sum) or w_sum <= 0:
        b = int(rng.integers(0, K))
    else:
        p = w / w_sum
        b = int(rng.choice(K, p=p))
    span = (float(ub_d) - float(lb_d)) / K
    return float(lb_d) + (b + float(rng.random())) * span


def _eta_from_best(ctx, K):
    """Per-bin eta heuristic: 1 / (1 + |bin_center − x_best_d|).
    Returns (dim, K) array, or None if best_so_far isn't set."""
    best = ctx.get("best_so_far")
    if best is None:
        return None
    centers = _bin_centers(ctx["lb"], ctx["ub"], K)
    xb = np.asarray(list(best), dtype=float)[:, None]
    return 1.0 / (1.0 + np.abs(centers - xb))


def _maybe_evaporate_and_deposit(ctx, rho, deposit_strategy,
                                 Q=1.0, e=5.0, w=6, clip=False):
    """Run evaporate + deposit at most once per generation."""
    if ctx.get("_pheromone_last_gen") == ctx["gen"]:
        return
    pher = ctx["pheromone"]
    tau, K = pher["tau"], pher["K"]
    lb, ub = ctx["lb"], ctx["ub"]
    tau *= (1.0 - rho)
    pop = ctx.get("_population", [])

    def _fit(p):
        try:
            return float(p.fitness.values[0])
        except Exception:
            return None

    if deposit_strategy in ("as", "eas") and pop:
        for p in pop:
            f = _fit(p)
            if f is None:
                continue
            _deposit_individual(tau, p, Q / (1.0 + max(0.0, f)), lb, ub, K)
        if deposit_strategy == "eas":
            best = ctx.get("best_so_far")
            if best is not None:
                fb = _fit(best)
                if fb is not None:
                    _deposit_individual(tau, best,
                                        e * Q / (1.0 + max(0.0, fb)),
                                        lb, ub, K)
    elif deposit_strategy == "asrank" and pop:
        sorted_pop = _sorted_pop(pop)
        top = sorted_pop[: max(1, w - 1)]
        for r, p in enumerate(top):
            f = _fit(p)
            if f is None:
                continue
            weight = (w - 1 - r) * Q / (1.0 + max(0.0, f))
            _deposit_individual(tau, p, weight, lb, ub, K)
        best = ctx.get("best_so_far")
        if best is not None:
            fb = _fit(best)
            if fb is not None:
                _deposit_individual(tau, best,
                                    w * Q / (1.0 + max(0.0, fb)),
                                    lb, ub, K)

    if clip:
        np.clip(tau, pher["tau_min"], pher["tau_max"], out=tau)
    ctx["_pheromone_last_gen"] = ctx["gen"]


# ============================================================
# F63 — Pheromone Evaporation
# ============================================================
def make_pheromone_evaporation(rho=0.1, K=20, alpha=1.0):
    """ACO mutation under standard AS deposit; the studied knob is rho."""
    def mutate_pher_evap(ind, ctx):
        rng = ctx["rng"]
        lb, ub = ctx["lb"], ctx["ub"]
        pher = _pheromone_init(ctx, K=K)
        _maybe_evaporate_and_deposit(ctx, rho, "as", Q=1.0)
        tau = pher["tau"]
        d = int(rng.integers(0, ctx["dim"]))
        new = list(ind)
        new[d] = _sample_dim_value(tau[d], lb[d], ub[d], K, rng, alpha=alpha)
        return new
    return mutate_pher_evap


# ============================================================
# F64 — Pheromone Deposit by Tour Quality
# ============================================================
def make_pheromone_deposit(rho=0.1, Q=1.0, K=20, alpha=1.0):
    """AS quality-weighted deposit Δτ ∝ Q/(1+f); knob is Q."""
    def mutate_pher_deposit(ind, ctx):
        rng = ctx["rng"]
        lb, ub = ctx["lb"], ctx["ub"]
        pher = _pheromone_init(ctx, K=K)
        _maybe_evaporate_and_deposit(ctx, rho, "as", Q=Q)
        tau = pher["tau"]
        d = int(rng.integers(0, ctx["dim"]))
        new = list(ind)
        new[d] = _sample_dim_value(tau[d], lb[d], ub[d], K, rng, alpha=alpha)
        return new
    return mutate_pher_deposit


# ============================================================
# F65 — Heuristic Information η = 1/d
# ============================================================
def make_pheromone_heuristic(rho=0.1, alpha=1.0, beta=2.0, K=20):
    """Sample bins proportional to τ^α · η^β where η = 1/(1+|center−x_best|).
    Falls back to η ≡ 1 until best_so_far is established."""
    def mutate_pher_heur(ind, ctx):
        rng = ctx["rng"]
        lb, ub = ctx["lb"], ctx["ub"]
        pher = _pheromone_init(ctx, K=K)
        _maybe_evaporate_and_deposit(ctx, rho, "as", Q=1.0)
        tau = pher["tau"]
        eta = _eta_from_best(ctx, K)
        d = int(rng.integers(0, ctx["dim"]))
        new = list(ind)
        eta_row = eta[d] if eta is not None else None
        new[d] = _sample_dim_value(tau[d], lb[d], ub[d], K, rng,
                                   alpha=alpha, eta_row=eta_row, beta=beta)
        return new
    return mutate_pher_heur


# ============================================================
# F66 — MAX-MIN Pheromone Bounds (MMAS)
# ============================================================
def make_pheromone_minmax(rho=0.1, tau_min=0.1, tau_max=5.0, K=20, alpha=1.0):
    """MMAS: τ is clipped to [tau_min, tau_max] every generation."""
    def mutate_pher_mmas(ind, ctx):
        rng = ctx["rng"]
        lb, ub = ctx["lb"], ctx["ub"]
        pher = _pheromone_init(ctx, K=K, tau_min=tau_min, tau_max=tau_max)
        _maybe_evaporate_and_deposit(ctx, rho, "as", Q=1.0, clip=True)
        tau = pher["tau"]
        d = int(rng.integers(0, ctx["dim"]))
        new = list(ind)
        new[d] = _sample_dim_value(tau[d], lb[d], ub[d], K, rng, alpha=alpha)
        return new
    return mutate_pher_mmas


# ============================================================
# F67 — Elitist (EAS) Reinforcement
# ============================================================
def make_pheromone_eas(rho=0.1, Q=1.0, e=5.0, K=20, alpha=1.0):
    """Standard AS deposit + extra deposit of weight e on global best."""
    def mutate_pher_eas(ind, ctx):
        rng = ctx["rng"]
        lb, ub = ctx["lb"], ctx["ub"]
        pher = _pheromone_init(ctx, K=K)
        _maybe_evaporate_and_deposit(ctx, rho, "eas", Q=Q, e=e)
        tau = pher["tau"]
        d = int(rng.integers(0, ctx["dim"]))
        new = list(ind)
        new[d] = _sample_dim_value(tau[d], lb[d], ub[d], K, rng, alpha=alpha)
        return new
    return mutate_pher_eas


# ============================================================
# F68 — Rank-Based Update (ASrank)
# ============================================================
def make_pheromone_asrank(rho=0.1, w=6, Q=1.0, K=20, alpha=1.0):
    """Top w-1 deposit with linear weights; global best deposits with weight w."""
    def mutate_pher_asrank(ind, ctx):
        rng = ctx["rng"]
        lb, ub = ctx["lb"], ctx["ub"]
        pher = _pheromone_init(ctx, K=K)
        _maybe_evaporate_and_deposit(ctx, rho, "asrank", w=w, Q=Q)
        tau = pher["tau"]
        d = int(rng.integers(0, ctx["dim"]))
        new = list(ind)
        new[d] = _sample_dim_value(tau[d], lb[d], ub[d], K, rng, alpha=alpha)
        return new
    return mutate_pher_asrank


# ============================================================
# F69 — Pheromone-Based Edge Recombination (crossover)
# ============================================================
def make_pheromone_edge_crossover(rho=0.1, K=20, alpha=1.0, deposit=True):
    """Per-dim discrete pick: child inherits p1[d] or p2[d] proportional to
    τ at each parent's bin. Set deposit=False to skip evaporate+deposit
    (when paired with a deposit-bearing mutate that owns the scheme)."""
    def crossover_pher_edge(p1, p2, ctx):
        rng = ctx["rng"]
        lb, ub = ctx["lb"], ctx["ub"]
        pher = _pheromone_init(ctx, K=K)
        if deposit:
            _maybe_evaporate_and_deposit(ctx, rho, "as", Q=1.0)
        tau = pher["tau"]
        a = np.asarray(list(p1), dtype=float)
        b = np.asarray(list(p2), dtype=float)
        c1 = a.copy()
        c2 = b.copy()
        for d in range(ctx["dim"]):
            ba = _bin_index_scalar(a[d], lb[d], ub[d], K)
            bb = _bin_index_scalar(b[d], lb[d], ub[d], K)
            ta = tau[d, ba] ** alpha
            tb = tau[d, bb] ** alpha
            tot = ta + tb
            pa = 0.5 if (tot <= 0 or not np.isfinite(tot)) else ta / tot
            if rng.random() < pa:
                c1[d] = a[d]; c2[d] = b[d]
            else:
                c1[d] = b[d]; c2[d] = a[d]
        return (np.clip(c1, lb, ub).tolist(),
                np.clip(c2, lb, ub).tolist())
    return crossover_pher_edge


# ============================================================
# F70 — Pheromone-Weighted Crossover (continuous blend)
# ============================================================
def make_pheromone_weighted_crossover(rho=0.1, K=20, alpha=1.0, deposit=True):
    """Per-dim blend child = w·p1 + (1−w)·p2 with w τ-proportional.
    Set deposit=False to skip evaporate+deposit (when paired with a
    deposit-bearing mutate that owns the scheme)."""
    def crossover_pher_weighted(p1, p2, ctx):
        lb, ub = ctx["lb"], ctx["ub"]
        pher = _pheromone_init(ctx, K=K)
        if deposit:
            _maybe_evaporate_and_deposit(ctx, rho, "as", Q=1.0)
        tau = pher["tau"]
        a = np.asarray(list(p1), dtype=float)
        b = np.asarray(list(p2), dtype=float)
        c1 = np.empty_like(a)
        c2 = np.empty_like(a)
        for d in range(ctx["dim"]):
            ba = _bin_index_scalar(a[d], lb[d], ub[d], K)
            bb = _bin_index_scalar(b[d], lb[d], ub[d], K)
            ta = tau[d, ba] ** alpha
            tb = tau[d, bb] ** alpha
            tot = ta + tb
            wt = 0.5 if (tot <= 0 or not np.isfinite(tot)) else ta / tot
            c1[d] = wt * a[d] + (1.0 - wt) * b[d]
            c2[d] = (1.0 - wt) * a[d] + wt * b[d]
        return (np.clip(c1, lb, ub).tolist(),
                np.clip(c2, lb, ub).tolist())
    return crossover_pher_weighted


# ============================================================
# F71 — Candidate List (Nearest Neighbor) selection
# ============================================================
def make_candidate_list_select(m=5, tournsize=2):
    """For each parent slot, run a tournament restricted to the m nearest
    Euclidean neighbors of a random anchor in the population."""
    def select_candlist(population, k, ctx):
        rng = ctx["rng"]
        n = len(population)
        if n <= m:
            return tools.selTournament(population, k, tournsize=tournsize)
        positions = np.array([list(p) for p in population], dtype=float)
        out = []
        for _ in range(k):
            anchor_idx = int(rng.integers(0, n))
            d = np.linalg.norm(positions - positions[anchor_idx], axis=1)
            nn_idx = np.argsort(d)[:m]
            cand = [population[int(i)] for i in nn_idx]
            winner = tools.selTournament(cand, 1,
                                         tournsize=min(tournsize, len(cand)))[0]
            ind = creator.Individual(list(winner))
            ind.fitness.values = winner.fitness.values
            out.append(ind)
        return out
    return select_candlist


# ============================================================
# F72 — Pheromone-Guided Mutation
# ============================================================
def make_pheromone_guided_mutation(rho=0.1, alpha=1.0, beta=2.0, K=20,
                                   indpb=None):
    """For each dim, with probability indpb (default 1/dim), resample by
    drawing a bin proportional to τ^α · η^β then uniform inside it."""
    def mutate_pher_guided(ind, ctx):
        rng = ctx["rng"]
        lb, ub = ctx["lb"], ctx["ub"]
        pher = _pheromone_init(ctx, K=K)
        _maybe_evaporate_and_deposit(ctx, rho, "as", Q=1.0)
        tau = pher["tau"]
        eta = _eta_from_best(ctx, K)
        p = (1.0 / max(1, ctx["dim"])) if indpb is None else indpb
        new = list(ind)
        for d in range(ctx["dim"]):
            if rng.random() < p:
                eta_row = eta[d] if eta is not None else None
                new[d] = _sample_dim_value(
                    tau[d], lb[d], ub[d], K, rng,
                    alpha=alpha, eta_row=eta_row, beta=beta,
                )
        return new
    return mutate_pher_guided


# ============================================================
# F73 — PSO velocity-guided mutation (gbest-only)
# ============================================================
def make_pso_velocity_mutate(w=0.7, c1=1.5, c2=1.5, vmax_frac=0.2):
    """Social-only PSO step as a mutation operator:
        v ← w·v + (c1+c2)·r·(gbest − x),  x ← x + v
    Velocity is keyed by the rounded vector so it survives across
    generations when the same point recurs; otherwise initialized to zero.
    """
    def mutate_pso(ind, ctx):
        rng = ctx["rng"]
        lb, ub = ctx["lb"], ctx["ub"]
        x = _arr(ind)
        gbest = ctx.get("best_so_far")
        if gbest is None:
            return list(ind)
        g = _arr(gbest)
        vel_store = ctx.setdefault("_pso_vel", {})
        key = tuple(np.round(x, 8).tolist())
        v = vel_store.get(key, np.zeros_like(x))
        r1 = rng.random(x.shape)
        r2 = rng.random(x.shape)
        v = w * v + c1 * r1 * (g - x) + c2 * r2 * (g - x)
        vmax = vmax_frac * (ub - lb)
        v = np.clip(v, -vmax, vmax)
        new_x = x + v
        new_x = np.clip(new_x, lb, ub)
        vel_store[tuple(np.round(new_x, 8).tolist())] = v
        return new_x.tolist()
    return mutate_pso


# ============================================================
# F74 — ABC fitness-guided mutation (employed / onlooker / mixed)
# ============================================================
def make_abc_mutate(phase="employed", n_dims=1):
    """ABC-style perturbation:  v_d = x_d + φ · (x_d − x_{k,d})
    on `n_dims` randomly chosen dimensions.
        - employed: neighbor k drawn uniformly at random.
        - onlooker: k drawn with probability proportional to fitness rank
                    (better individuals more likely).
        - mixed:    50/50 alternation per call (canonical ABC pattern).
    """
    if phase not in ("employed", "onlooker", "mixed"):
        raise ValueError(f"phase must be employed/onlooker/mixed, got {phase}")

    def mutate_abc(ind, ctx):
        rng = ctx["rng"]
        pop = ctx.get("_population", [])
        if len(pop) < 2:
            return list(ind)
        x = _arr(ind)
        dim = ctx["dim"]

        mode = phase
        if mode == "mixed":
            mode = "employed" if rng.random() < 0.5 else "onlooker"

        if mode == "employed":
            k = int(rng.integers(0, len(pop)))
        else:
            fits = np.array(
                [p.fitness.values[0] if p.fitness.valid else np.inf for p in pop],
                dtype=float,
            )
            order = np.argsort(fits)        # best first (minimization)
            ranks = np.empty_like(order)
            ranks[order] = np.arange(len(pop))
            weights = (len(pop) - ranks).astype(float)
            s = weights.sum()
            if s <= 0 or not np.isfinite(s):
                k = int(rng.integers(0, len(pop)))
            else:
                k = int(rng.choice(len(pop), p=weights / s))

        xk = _arr(pop[k])
        n = max(1, min(n_dims, dim))
        dims = rng.choice(dim, size=n, replace=False)
        phi = rng.uniform(-1.0, 1.0, size=n)
        new = x.copy()
        new[dims] = x[dims] + phi * (x[dims] - xk[dims])
        return np.clip(new, ctx["lb"], ctx["ub"]).tolist()
    return mutate_abc


# ============================================================
# F75 — SGA-style nested PSO refinement on top-k elites
# ============================================================
def make_sga_pso_refine(top_k=5, n_iters=3, w=0.7, c1=1.5, c2=1.5,
                       vmax_frac=0.2, every_n_gens=1, stagnation_threshold=0):
    """Swarming Genetic Algorithm: after GA replacement, run a small PSO
    sub-swarm on the top-k individuals for `n_iters` velocity steps.

    Each particle keeps its personal best (pbest) across generations via
    `ctx["_sga_pso_state"]`; the global best (gbest) is the GA's
    `best_so_far`. FE cost per generation: ~ top_k * n_iters, charged
    against `ctx["eval_counter"]` and capped by `ctx["max_fes"]`.

    Gating:
        every_n_gens: run only every Nth generation (1 = every gen).
        stagnation_threshold: skip unless ctx['stagnation_counter'] >= T.

    Plug as `refine_fn` in FeatureConfig — runs once per generation after
    replacement.
    """
    def refine(population, ctx):
        evaluate_fn = ctx.get("evaluate_fn")
        eval_counter = ctx.get("eval_counter")
        if evaluate_fn is None or eval_counter is None:
            return population  # core not configured to support sub-loop
        max_fes = ctx["max_fes"]
        if eval_counter[0] >= max_fes:
            return population
        if every_n_gens > 1 and (ctx.get("gen", 0) % every_n_gens) != 0:
            return population
        if ctx.get("stagnation_counter", 0) < stagnation_threshold:
            return population

        rng = ctx["rng"]
        lb, ub = ctx["lb"], ctx["ub"]
        vmax = vmax_frac * (ub - lb)

        # Sort population (best first); pick top-k as particles.
        order = sorted(range(len(population)),
                       key=lambda i: population[i].fitness.values[0])
        k = min(top_k, len(order))
        idxs = order[:k]

        # Persistent per-slot PSO state (velocity + pbest position/fitness).
        # Keyed by rank slot rather than identity so it survives replacement.
        state = ctx.setdefault("_sga_pso_state", {})
        gbest = ctx.get("best_so_far")
        if gbest is None:
            return population
        g = _arr(gbest)

        for it in range(n_iters):
            for slot, idx in enumerate(idxs):
                if eval_counter[0] >= max_fes:
                    return population
                ind = population[idx]
                x = _arr(ind)
                fx = ind.fitness.values[0]
                s = state.get(slot)
                if s is None:
                    v = np.zeros_like(x)
                    pbest_x = x.copy()
                    pbest_f = fx
                else:
                    v, pbest_x, pbest_f = s
                    if v.shape != x.shape:
                        v = np.zeros_like(x)
                    if pbest_x.shape != x.shape:
                        pbest_x = x.copy()
                        pbest_f = fx

                r1 = rng.random(x.shape)
                r2 = rng.random(x.shape)
                v = w * v + c1 * r1 * (pbest_x - x) + c2 * r2 * (g - x)
                v = np.clip(v, -vmax, vmax)
                new_x = np.clip(x + v, lb, ub)

                f_new = float(evaluate_fn(new_x))
                eval_counter[0] += 1

                # Build a fresh Individual (avoid mutating shared fitness state).
                new_ind = creator.Individual(new_x.tolist())
                new_ind.age = getattr(ind, "age", 0)
                new_ind.fitness.values = (f_new,)
                population[idx] = new_ind

                if f_new < pbest_f:
                    pbest_x = new_x.copy()
                    pbest_f = f_new
                if f_new < gbest.fitness.values[0]:
                    g = new_x.copy()  # update local view of gbest

                state[slot] = (v, pbest_x, pbest_f)
        return population
    return refine


# ============================================================
# F76 — Metropolis replacement with cooling schedule (SA-style)
# ============================================================
def make_metropolis_replace(T0=1.0, T_end=1e-3, schedule="exp", elitism_k=1):
    """One-to-one replacement that *always* accepts a better offspring and
    accepts a worse one with probability exp(-Δf / T(t)). Temperature T(t)
    cools from T0 to T_end as ctx['t_ratio'] goes 0 → 1, mirroring SA's
    explore-then-exploit transition. Combats premature convergence by
    letting early generations climb out of basins.

    schedule:
      - "exp":    T = T0 * (T_end/T0) ** t_ratio   (geometric cooling)
      - "linear": T = T0 + (T_end - T0) * t_ratio
      - "cosine": T = T_end + 0.5*(T0-T_end)*(1+cos(pi*t_ratio))

    Δf is normalized by max(|f_parent|, 1e-12) so the schedule is roughly
    scale-invariant across benchmarks. Top-`elitism_k` parents are always
    preserved unconditionally to guarantee monotone best-so-far.
    """
    if schedule not in ("exp", "linear", "cosine"):
        raise ValueError(f"schedule must be exp/linear/cosine, got {schedule}")
    if T0 <= 0 or T_end <= 0:
        raise ValueError("T0 and T_end must be positive")

    def _temperature(t):
        t = min(max(t, 0.0), 1.0)
        if schedule == "exp":
            return T0 * (T_end / T0) ** t
        if schedule == "linear":
            return T0 + (T_end - T0) * t
        # cosine
        return T_end + 0.5 * (T0 - T_end) * (1.0 + math.cos(math.pi * t))

    def replace_metropolis(parents, offspring, ctx):
        rng = ctx["rng"]
        T = _temperature(ctx.get("t_ratio", 0.0))

        # Preserve top-k elites unconditionally.
        k = max(0, min(elitism_k, len(parents)))
        parents_sorted = sorted(parents, key=lambda x: x.fitness.values[0])
        elites = parents_sorted[:k]
        elite_ids = {id(p) for p in elites}

        new_pop = list(elites)
        n = min(len(parents), len(offspring))
        for i in range(n):
            p, o = parents[i], offspring[i]
            if id(p) in elite_ids:
                continue  # already carried over as elite
            f_p = p.fitness.values[0]
            f_o = o.fitness.values[0]
            if f_o <= f_p:
                new_pop.append(o)
            else:
                delta = (f_o - f_p) / max(abs(f_p), 1e-12)
                accept_prob = math.exp(-delta / T) if T > 0 else 0.0
                new_pop.append(o if rng.random() < accept_prob else p)

        # Carry over any unmatched extras (non-elite parents past index n)
        for j in range(n, len(parents)):
            if id(parents[j]) not in elite_ids:
                new_pop.append(parents[j])
        return new_pop[: len(parents)]

    return replace_metropolis


# ============================================================
# F77 — Fast GA mutation (power-law α/n rate + heavy-tailed step)
# Doerr, Le, Makhmara, Nguyen (2017) — "Fast Genetic Algorithms"
# ============================================================
def _sample_power_law_alpha(beta, n, rng):
    """Draw alpha in {1, 2, ..., n/2} with P(alpha=k) ~ k^{-beta}.
    Normalized over the truncated support, per Doerr et al."""
    upper = max(1, n // 2)
    ks = np.arange(1, upper + 1, dtype=float)
    weights = ks ** (-beta)
    weights /= weights.sum()
    return int(rng.choice(ks, p=weights))


def make_fast_ga_mutate(beta=1.5, dist="cauchy", scale=0.1, scale_end=None):
    """Fast-GA mutation: per call, draw alpha ~ power-law(beta) over {1..n/2},
    set per-coordinate mutation rate p = alpha / n, and perturb each selected
    coordinate by a heavy-tailed step (Cauchy or Lévy).

    Args:
        beta:      power-law exponent (Doerr et al. recommend 1.5).
        dist:      "cauchy" or "levy" — distribution for the step magnitude.
        scale:     initial step size as a fraction of (ub - lb).
        scale_end: if not None, linearly anneal scale -> scale_end using
                   ctx['t_ratio'] (0 at start, 1 at end of run).
    """
    dist = dist.lower()
    if dist not in ("cauchy", "levy"):
        raise ValueError(f"dist must be 'cauchy' or 'levy', got {dist!r}")

    def mutate_fast_ga(ind, ctx):
        rng = ctx["rng"]
        dim = ctx["dim"]
        lb, ub = ctx["lb"], ctx["ub"]
        x = _arr(ind)

        alpha = _sample_power_law_alpha(beta, dim, rng)
        p = alpha / dim
        mask = rng.random(dim) < p
        if not mask.any():
            mask[rng.integers(0, dim)] = True

        if dist == "cauchy":
            step = rng.standard_cauchy(dim)
        else:
            step = _levy_step(dim, beta=1.5, rng=rng)

        if scale_end is not None:
            t = float(ctx.get("t_ratio", 0.0))
            cur_scale = scale + (scale_end - scale) * t
        else:
            cur_scale = scale

        delta = np.where(mask, step, 0.0) * cur_scale * (ub - lb)
        return np.clip(x + delta, lb, ub).tolist()

    return mutate_fast_ga


# ============================================================
# F78 — Cuckoo Search egg discovery (Yang & Deb 2009)
# ============================================================
def make_cuckoo_discovery(pa=0.25):
    """Per generation, discover-and-replace a fraction of nests.

    For each individual, with probability `pa` build a "discovered" replacement
    `x' = x + r * (x_perm1 - x_perm2)` where the two permutations are random
    other nests (Yang & Deb 2009's biased-walk discovery rule). The new vector
    is bound-handled and evaluated by the core's post-diversity eval loop.
    Pairs with Lévy mutation (F13) for the full CS algorithm; standalone here
    so the ablation isolates the discovery mechanism.
    """
    def diversity_cuckoo(population, ctx):
        rng = ctx["rng"]
        lb, ub = ctx["lb"], ctx["ub"]
        n = len(population)
        if n < 3:
            return population
        for i in range(n):
            if rng.random() >= pa:
                continue
            j, k = rng.choice(n, size=2, replace=False)
            x = _arr(population[i])
            xj = _arr(population[j])
            xk = _arr(population[k])
            r = rng.random()
            new_x = x + r * (xj - xk)
            new_x = np.clip(new_x, lb, ub)
            new_ind = creator.Individual(new_x.tolist())
            new_ind.age = 0
            population[i] = new_ind
        return population
    return diversity_cuckoo


# ============================================================
# F79 — ABC explicit scout phase (Karaboga & Basturk 2007)
# ============================================================
def make_abc_scout(limit=None):
    """Trial-counter abandonment with scout re-initialization.

    Wraps generational k-elitism. Each individual carries `ind.trial`,
    incremented whenever it survives a generation without being improved on
    by its slot's offspring; reset to 0 on improvement. Once `trial > limit`
    the individual is abandoned and replaced by a uniform random scout (with
    invalid fitness — the core's post-replace eval loop charges the FE).

    `limit` defaults to `pop_size * dim` per Karaboga's canonical heuristic.
    """
    def replace_abc(parents, offspring, ctx):
        rng = ctx["rng"]
        lb, ub = ctx["lb"], ctx["ub"]
        dim = ctx["dim"]
        pop_size = ctx["pop_size"]
        lim = limit if limit is not None else max(1, pop_size * dim)

        # Pair offspring against parents by sorted-fitness slot for trial
        # tracking. Use generational k-elitism for the actual survival
        # decision (consistent with the default replacer).
        survivors = generational_elitism(parents, offspring, int(ctx["elitism_k"]))

        parents_by_f = sorted(parents, key=lambda x: x.fitness.values[0])
        offspring_by_f = sorted(offspring, key=lambda x: x.fitness.values[0])
        # Slot-wise improvement check (best-vs-best, 2nd-vs-2nd, ...) is the
        # standard ABC analogue here.
        m = min(len(parents_by_f), len(offspring_by_f))
        improved_ids = set()
        unimproved_ids = set()
        for i in range(m):
            p = parents_by_f[i]
            o = offspring_by_f[i]
            if o.fitness.values[0] < p.fitness.values[0]:
                improved_ids.add(id(p))
            else:
                unimproved_ids.add(id(p))

        # Update trial counters on whatever survived.
        for ind in survivors:
            t = getattr(ind, "trial", 0)
            if id(ind) in improved_ids:
                ind.trial = 0
            elif id(ind) in unimproved_ids:
                ind.trial = t + 1
            else:
                # Newly produced offspring carried into survivors via elitism
                # of offspring: fresh slot.
                ind.trial = 0

        # Scout phase: abandon any survivor whose trial exceeded the limit.
        for i, ind in enumerate(survivors):
            if getattr(ind, "trial", 0) > lim:
                scout_vec = rng.uniform(lb, ub).tolist()
                scout = creator.Individual(scout_vec)
                scout.age = 0
                scout.trial = 0
                # Leave fitness invalid; core's post-replace eval will score it.
                survivors[i] = scout
        return survivors
    return replace_abc


# ============================================================
# F80 — Bacterial Foraging chemotaxis (Passino 2002)
# ============================================================
def make_bfo_chemotaxis(step_size_frac=0.05):
    """Swim-and-tumble per-individual mutation.

    Each bacterium keeps a persistent unit direction `ind.bfo_dir` and the
    fitness it observed last time it was perturbed (`ind.bfo_prev_fitness`).
    If the most-recent step improved fitness, *swim* — take another step in
    the same direction. Otherwise *tumble* — sample a new uniform random
    unit direction first, then step. First call (no state) tumbles.

    Step magnitude = `step_size_frac * (ub - lb)`.
    """
    def mutate_bfo(ind, ctx):
        rng = ctx["rng"]
        lb, ub = ctx["lb"], ctx["ub"]
        dim = ctx["dim"]
        x = _arr(ind)
        cur_f = ind.fitness.values[0] if ind.fitness.valid else np.inf

        prev_dir = getattr(ind, "bfo_dir", None)
        prev_f = getattr(ind, "bfo_prev_fitness", np.inf)

        # Tumble if no history or fitness worsened/equal (Passino's rule).
        if prev_dir is None or cur_f >= prev_f:
            d = rng.standard_normal(dim)
            norm = np.linalg.norm(d)
            d = d / norm if norm > 1e-12 else np.ones(dim) / np.sqrt(dim)
        else:
            d = np.asarray(prev_dir, dtype=float)
            if d.shape != x.shape:
                d = rng.standard_normal(dim)
                d /= max(np.linalg.norm(d), 1e-12)

        step = step_size_frac * (ub - lb) * d
        new_x = np.clip(x + step, lb, ub)
        # Stash state on the *new* individual via ctx side-channel: the core
        # rebuilds the Individual after bound-handling, so we attach attrs to
        # the list we return — but the core wraps with make_individual which
        # only sets age. Instead, store keyed by id(ind) on ctx and migrate
        # at next call: pragmatic since BFO state is robust to slot churn.
        # Simpler: write directly to ind so future generations carrying the
        # same Individual (elitism path) benefit; new offspring will tumble
        # on their first call, which is correct.
        ind.bfo_dir = d
        ind.bfo_prev_fitness = cur_f
        return new_x.tolist()
    return mutate_bfo


# ============================================================
# F81 — Krill Herd movement (Gandomi & Alavi 2012)
# ============================================================
def make_krill_herd(Nmax=0.01, Vf=0.02, Dmax=0.005, n_neighbors=5,
                    inertia_n=0.5, inertia_f=0.5):
    """Three additive motion sources: neighbor-induced, foraging, diffusion.

    Per call:
      N_i = Nmax * α_local + ω_n * N_prev   (neighbor attraction by fitness)
      F_i = Vf   * (β_food + β_best) + ω_f * F_prev   (toward food + pbest)
      D_i = Dmax * (1 - t_ratio) * random_unit            (diffusion)
      x' = x + Δt * (N_i + F_i + D_i)

    Δt is scaled to half the bounding-box diagonal per dim (Gandomi & Alavi).
    Persistent state (N_prev, F_prev, pbest) lives on `ind` as attrs.
    """
    def mutate_kh(ind, ctx):
        rng = ctx["rng"]
        lb, ub = ctx["lb"], ctx["ub"]
        dim = ctx["dim"]
        pop = ctx.get("_population", [])
        if len(pop) < 2:
            return list(ind)
        x = _arr(ind)
        cur_f = ind.fitness.values[0] if ind.fitness.valid else np.inf

        fits = np.array(
            [p.fitness.values[0] if p.fitness.valid else np.inf for p in pop],
            dtype=float,
        )
        worst, best_f = fits.max(), fits.min()
        f_span = (worst - best_f) if worst > best_f + 1e-30 else 1.0

        # Neighbor-induced motion (local attraction by fitness gradient).
        dists = np.array(
            [np.linalg.norm(_arr(p) - x) for p in pop], dtype=float
        )
        ds = np.mean(dists) / 5.0 + 1e-12
        order = np.argsort(dists)
        K = min(n_neighbors, len(pop))
        nbr_idx = order[:K]
        alpha_local = np.zeros(dim)
        for j in nbr_idx:
            xj = _arr(pop[j])
            khat = (xj - x) / (dists[j] + ds)
            fhat = (cur_f - fits[j]) / f_span
            if not np.isfinite(fhat):
                continue
            alpha_local += fhat * khat

        N_prev = getattr(ind, "kh_N", np.zeros(dim))
        if N_prev.shape != x.shape:
            N_prev = np.zeros(dim)
        N = Nmax * alpha_local + inertia_n * N_prev

        # Foraging motion: toward food (inv-fitness-weighted centroid) and
        # personal best.
        inv_w = 1.0 / (fits - best_f + 1.0)
        positions = np.array([_arr(p) for p in pop])
        food = (positions * inv_w[:, None]).sum(axis=0) / inv_w.sum()
        food_fit = float(np.mean(fits))  # cheap proxy
        beta_food = ((cur_f - food_fit) / f_span) * (food - x) / (
            np.linalg.norm(food - x) + 1e-12)

        pbest_x = np.asarray(getattr(ind, "kh_pbest_x", x), dtype=float)
        pbest_f = float(getattr(ind, "kh_pbest_f", cur_f))
        if pbest_x.shape != x.shape:
            pbest_x = x.copy(); pbest_f = cur_f
        beta_best = ((cur_f - pbest_f) / f_span) * (pbest_x - x) / (
            np.linalg.norm(pbest_x - x) + 1e-12)

        F_prev = getattr(ind, "kh_F", np.zeros(dim))
        if F_prev.shape != x.shape:
            F_prev = np.zeros(dim)
        F = Vf * (beta_food + beta_best) + inertia_f * F_prev

        # Random diffusion, decaying with time.
        t = float(ctx.get("t_ratio", 0.0))
        rand_dir = rng.uniform(-1.0, 1.0, dim)
        D = Dmax * (1.0 - t) * rand_dir

        dt = 0.5 * float(np.mean(ub - lb))
        new_x = np.clip(x + dt * (N + F + D), lb, ub)

        # Persist state for next call.
        ind.kh_N = N
        ind.kh_F = F
        if cur_f < pbest_f:
            ind.kh_pbest_x = x.copy()
            ind.kh_pbest_f = cur_f
        else:
            ind.kh_pbest_x = pbest_x
            ind.kh_pbest_f = pbest_f
        return new_x.tolist()
    return mutate_kh


# ============================================================
# F82 — Symbiotic Organisms Search (Cheng & Prayogo 2014)
# ============================================================
def mutate_sos(ind, ctx):
    """Per call, pick one of {mutualism, commensalism, parasitism} uniformly.

    Parameter-free (like TLBO). Parasitism greedily compares and may charge
    one extra FE against ctx['eval_counter']; skipped if the budget is spent.
    """
    rng = ctx["rng"]
    lb, ub = ctx["lb"], ctx["ub"]
    dim = ctx["dim"]
    pop = ctx.get("_population", [])
    if len(pop) < 2:
        return list(ind)
    x = _arr(ind)
    best = ctx.get("best_so_far")
    x_best = _arr(best) if best is not None else x.copy()

    # Pick a partner ≠ self by identity where possible.
    j = int(rng.integers(0, len(pop)))
    xj = _arr(pop[j])

    phase = int(rng.integers(0, 3))
    if phase == 0:
        # Mutualism: mutual vector, benefit factors ∈ {1,2}.
        mv = 0.5 * (x + xj)
        BF1 = 1 + int(rng.integers(0, 2))
        r = rng.random(dim)
        new_x = x + r * (x_best - mv * BF1)
    elif phase == 1:
        # Commensalism: x_i helped by x_j; x_j unchanged.
        r = rng.uniform(-1.0, 1.0, dim)
        new_x = x + r * (x_best - xj)
    else:
        # Parasitism: build parasite by replacing random dims of x with
        # uniform random values; greedy keep if better.
        evaluate_fn = ctx.get("evaluate_fn")
        eval_counter = ctx.get("eval_counter")
        max_fes = ctx.get("max_fes", np.inf)
        if evaluate_fn is None or eval_counter is None or eval_counter[0] >= max_fes:
            return list(x)
        parasite = x.copy()
        n_swap = max(1, int(rng.integers(1, dim + 1) // 2))
        dims = rng.choice(dim, size=n_swap, replace=False)
        parasite[dims] = rng.uniform(lb[dims], ub[dims])
        parasite = np.clip(parasite, lb, ub)
        f_par = float(evaluate_fn(parasite))
        eval_counter[0] += 1
        cur_f = ind.fitness.values[0] if ind.fitness.valid else np.inf
        if f_par < cur_f:
            return parasite.tolist()
        return list(x)

    return np.clip(new_x, lb, ub).tolist()


# ============================================================
# F83 — Biogeography-Based Optimization migration (Simon 2008)
# ============================================================
def crossover_bbo(p1, p2, ctx):
    """Rate-controlled per-gene migration.

    Each parent's immigration rate λ and emigration rate μ are functions of
    its rank in the current `_population`:
        λ = 1 - rank/N      μ = rank/N        (sinusoidal-family, linear form)
    For each gene of each child: with prob λ_self, replace with the same gene
    from a population member chosen by roulette on μ. Otherwise keep self.
    Children are migrants (no blended pair). Rates are per-individual, not
    per-pair — the BBO distinction emphasized in the user's request.
    """
    rng = ctx["rng"]
    pop = ctx.get("_population", [])
    if len(pop) < 2:
        return list(p1), list(p2)

    fits = np.array(
        [p.fitness.values[0] if p.fitness.valid else np.inf for p in pop],
        dtype=float,
    )
    N = len(pop)
    order = np.argsort(fits)                    # best first
    ranks = np.empty(N, dtype=float)
    ranks[order] = np.arange(N)
    # Simon's convention: good HSI (rank=0, best) → high μ (emigrates
    # features), low λ (accepts few incoming). Poor → opposite.
    lam = ranks / max(N - 1, 1)                 # best → 0, worst → 1
    mu = 1.0 - ranks / max(N - 1, 1)            # best → 1, worst → 0
    mu_sum = mu.sum()
    if mu_sum <= 0 or not np.isfinite(mu_sum):
        return list(p1), list(p2)
    mu_probs = mu / mu_sum
    positions = np.array([_arr(p) for p in pop])

    def _migrate(child_vec, child_lambda):
        out = np.asarray(child_vec, dtype=float).copy()
        dim = out.shape[0]
        for d in range(dim):
            if rng.random() < child_lambda:
                src = int(rng.choice(N, p=mu_probs))
                out[d] = positions[src, d]
        return out.tolist()

    # Find each parent's rank within the population (by closest match);
    # parents passed in are clones of population members, so id() comparison
    # is unreliable across the parent-clone step in core.py — use a value-
    # based nearest lookup, which is exact for clones produced by the core.
    def _rank_of(parent_vec):
        pv = np.asarray(parent_vec, dtype=float)
        diffs = np.linalg.norm(positions - pv, axis=1)
        return int(ranks[int(np.argmin(diffs))])

    r1 = _rank_of(p1)
    r2 = _rank_of(p2)
    lam1 = r1 / max(N - 1, 1)
    lam2 = r2 / max(N - 1, 1)
    c1 = _migrate(list(p1), lam1)
    c2 = _migrate(list(p2), lam2)
    return c1, c2


# ============================================================
# F84 — GSA with full G(t) decay + K-best subset (Rashedi 2009)
# ============================================================
def make_gsa_full(G0=100.0, alpha=20.0, kbest_final_frac=0.02):
    """Full GSA attraction with the canonical pieces F26 simplified:
      - Gravitational constant decays as G(t) = G0 * exp(-alpha * t_ratio).
      - Attraction sums only over the K-best subset, with K shrinking from N
        down to max(2, kbest_final_frac * N) over the run.
    Mass calculation otherwise unchanged from F26.
    """
    def mutate_gsa(ind, ctx):
        rng = ctx["rng"]
        lb, ub = ctx["lb"], ctx["ub"]
        dim = ctx["dim"]
        pop = ctx.get("_population", [])
        N = len(pop)
        if N < 2:
            return list(ind)

        fits = np.array(
            [p.fitness.values[0] if p.fitness.valid else np.inf for p in pop],
            dtype=float,
        )
        worst, best_f = fits.max(), fits.min()
        denom = (worst - best_f) if worst > best_f + 1e-30 else 1.0
        masses = (worst - fits) / denom + 1e-9
        masses /= masses.sum()

        t = float(ctx.get("t_ratio", 0.0))
        G = G0 * math.exp(-alpha * t)

        # K-best subset: shrink linearly from N down to floor(kbest_final_frac*N).
        K_final = max(2, int(math.ceil(kbest_final_frac * N)))
        K = max(K_final, int(round(N - (N - K_final) * t)))
        K = min(K, N)
        kbest_idx = np.argsort(fits)[:K]

        x = _arr(ind)
        force = np.zeros(dim)
        for j in kbest_idx:
            xj = _arr(pop[j])
            d = np.linalg.norm(xj - x) + 1e-9
            force += rng.random() * G * masses[j] * (xj - x) / d

        # Treat force as acceleration; symplectic-style step with random
        # velocity persistence on ind.
        v_prev = np.asarray(getattr(ind, "gsa_v", np.zeros(dim)), dtype=float)
        if v_prev.shape != x.shape:
            v_prev = np.zeros(dim)
        v_new = rng.random() * v_prev + force
        new_x = np.clip(x + v_new, lb, ub)
        ind.gsa_v = v_new
        return new_x.tolist()
    return mutate_gsa


# ============================================================================
# F85 — GACO: Genetic Algorithm + Ant Colony Optimization hybrid (TSP)
# ============================================================================
#
# A *full* hybrid for permutation/TSP encodings. Unlike F63–F72 (which are a
# binned per-dimension pheromone EDA for real-coded search), this maintains a
# genuine edge-pheromone matrix tau[i][j] over cities and couples it with the
# GA in three places:
#
#   1. crossover  -> `make_aco_recombination`
#      ACS-style probabilistic tour construction biased by tau (learned) and
#      eta = 1/dist (heuristic), with the candidate set restricted to the two
#      parents' tour edges (an edge-recombination flavour). Pseudo-random-
#      proportional rule: with prob q0 exploit argmax tau^alpha * eta^beta,
#      otherwise sample roulette-wise. Falls back to all unvisited cities when
#      the parent-edge candidate set is empty.
#
#   2. refine     -> `make_aco_refine`  (runs every generation, post-replacement)
#      a. 2-opt local search on the top-k individuals (memetic step), using
#         O(n^2) vectorised delta evaluation against the distance matrix. Each
#         *accepted* improving move is charged 1 FE against MaxFES, so the
#         hybrid's local search competes with the GA for the same budget.
#      b. MAX-MIN-style pheromone update: evaporate tau *= (1 - rho); the
#         iteration-best and global-best tours each deposit 1/length on their
#         edges; clamp tau to [tau_max/(2n), tau_max] with
#         tau_max = 1/(rho * best_length_so_far).
#
# The mutation slot is intentionally left to the suite baseline (swap/inversion)
# so the variant's divergence from baseline stays minimal and reviewable.
#
# Both operators share state via ctx["_aco"], lazily built from
# ctx["problem"].dist on first use. They therefore only make sense on a TSP
# problem object exposing an integer `dist` matrix; using them elsewhere raises.


def _aco_problem_dist(ctx):
    problem = ctx.get("problem")
    D = getattr(problem, "dist", None)
    if D is None:
        raise ValueError(
            "GACO operators require a TSP-style problem exposing a `.dist` "
            "matrix on ctx['problem']; got "
            f"{type(problem).__name__ if problem is not None else None}."
        )
    return np.asarray(D, dtype=float)


def _nn_tour_length(D):
    """Greedy nearest-neighbour tour length from city 0 (a crude upper bound
    used only to seed pheromone scaling)."""
    n = D.shape[0]
    visited = np.zeros(n, dtype=bool)
    cur = 0
    visited[0] = True
    total = 0.0
    for _ in range(n - 1):
        row = D[cur].copy()
        row[visited] = np.inf
        nxt = int(np.argmin(row))
        total += D[cur, nxt]
        visited[nxt] = True
        cur = nxt
    total += D[cur, 0]
    return float(max(total, 1e-9))


def _aco_state(ctx, rho, cand_k=16):
    st = ctx.get("_aco")
    if st is not None:
        return st
    D = _aco_problem_dist(ctx)
    n = D.shape[0]
    with np.errstate(divide="ignore"):
        eta = 1.0 / D
    eta[~np.isfinite(eta)] = 0.0
    np.fill_diagonal(eta, 0.0)
    nn_len = _nn_tour_length(D)
    tau_max = 1.0 / (rho * nn_len)
    tau = np.full((n, n), tau_max, dtype=float)
    np.fill_diagonal(tau, 0.0)
    iu, ju = np.triu_indices(n, k=1)
    keep = ~((iu == 0) & (ju == n - 1))       # drop the whole-tour reversal
    # k-nearest-neighbour candidate lists (ACS speed-up): for each city, the
    # `k` closest other cities, in ascending distance order.
    k = int(max(1, min(cand_k, n - 1)))
    cand = np.argsort(D, axis=1)[:, 1:k + 1].astype(np.int64)
    st = {
        "D": D, "eta": eta, "n": n,
        "tau": tau, "rho": float(rho),
        "best_len": nn_len,          # best tour length seen so far (for MMAS bounds)
        "tau_max": tau_max,
        "tau_min": tau_max / (2.0 * n),
        "tau0": 1.0 / (n * nn_len),  # ACS local-update target (small)
        "triu_i": iu[keep], "triu_j": ju[keep],
        "cand": cand, "cand_k": k,
    }
    ctx["_aco"] = st
    return st


def _as_perm(ind, n):
    """Coerce an individual to a clean Python-int list of length n; if it is
    not a valid permutation of 0..n-1, repair it (keep first occurrences, then
    append the missing cities in order)."""
    seq = [int(x) for x in ind]
    if len(seq) == n:
        seen = set(seq)
        if len(seen) == n and min(seen) == 0 and max(seen) == n - 1:
            return seq
    out, used = [], set()
    for c in seq:
        if 0 <= c < n and c not in used:
            out.append(c); used.add(c)
    for c in range(n):
        if c not in used:
            out.append(c)
    return out


def _tour_neighbours(perm, n):
    """For each city, the set of cities adjacent to it in the cyclic tour."""
    nb = [set() for _ in range(n)]
    L = len(perm)
    for i, c in enumerate(perm):
        nb[c].add(perm[(i - 1) % L])
        nb[c].add(perm[(i + 1) % L])
    return nb


def make_aco_recombination(alpha=1.0, beta=3.0, q0=0.3, rho=0.1,
                           cand_k=16, local_update=True, xi=0.1):
    """ACS-style edge-recombination crossover guided by pheromone + heuristic.

    Builds two child tours. Each starts from a random city and repeatedly
    appends the next city chosen from the union of (a) the current city's tour
    neighbours in p1 and p2 and (b) its k-nearest-neighbour candidate list,
    restricted to unvisited cities; if that set is empty, every unvisited city
    is a candidate. The choice uses Ant Colony System's pseudo-random-
    proportional rule on weights tau[cur][j]^alpha * eta[cur][j]^beta: with
    probability q0 exploit the argmax, otherwise sample roulette-wise.

    When `local_update` is set, ACS's online step-by-step pheromone update is
    applied to each chosen edge: tau_ij <- (1-xi)*tau_ij + xi*tau0 (clamped to
    >= tau_min). This drains pheromone from edges already used during this
    generation's matings, nudging later constructions toward other edges.
    """
    def crossover_aco(p1, p2, ctx):
        rng = ctx["rng"]
        st = _aco_state(ctx, rho, cand_k=cand_k)
        n = st["n"]
        tau, eta, cand_lists = st["tau"], st["eta"], st["cand"]
        tau0 = st["tau0"]
        a = _as_perm(p1, n)
        b = _as_perm(p2, n)
        nb_a = _tour_neighbours(a, n)
        nb_b = _tour_neighbours(b, n)

        def construct(start):
            visited = np.zeros(n, dtype=bool)
            tour = [start]
            visited[start] = True
            cur = start
            all_cities = np.arange(n)
            for _ in range(n - 1):
                pool = nb_a[cur] | nb_b[cur]
                pool.update(int(c) for c in cand_lists[cur])
                cand = [c for c in pool if not visited[c]]
                if not cand:
                    cand = all_cities[~visited].tolist()
                cand = np.asarray(cand, dtype=int)
                w = (tau[cur, cand] ** alpha) * (eta[cur, cand] ** beta)
                tot = float(w.sum())
                if not np.isfinite(tot) or tot <= 0.0:
                    nxt = int(cand[rng.integers(0, len(cand))])
                elif rng.random() < q0:
                    nxt = int(cand[int(np.argmax(w))])
                else:
                    nxt = int(cand[rng.choice(len(cand), p=w / tot)])
                if local_update:
                    nv = max((1.0 - xi) * tau[cur, nxt] + xi * tau0,
                             st["tau_min"])
                    tau[cur, nxt] = nv
                    tau[nxt, cur] = nv
                tour.append(nxt)
                visited[nxt] = True
                cur = nxt
            return tour

        c1 = construct(int(a[0]))
        c2 = construct(int(rng.integers(0, n)))
        return c1, c2

    crossover_aco.__name__ = (
        f"aco_recomb_a{alpha:g}_b{beta:g}_q{q0:g}"
        f"_k{cand_k}{'_lu' if local_update else ''}"
    )
    return crossover_aco


def _two_opt_best_move(tour, D, iu, ju):
    """Return (i, j, delta) of the best (most-negative-delta) 2-opt move that
    reverses tour[i+1 .. j], or None if no improving move exists. `iu, ju` are
    the precomputed upper-triangle position-pair indices (whole-tour reversal
    already excluded). Fully vectorised: O(n^2) time and memory.
    """
    t = np.asarray(tour, dtype=np.int64)
    tn = np.roll(t, -1)                       # successor of each position
    e = D[t, tn]                              # length of edge leaving position i
    a, b = t[iu], tn[iu]                      # edge (a, b) at position iu
    c, d = t[ju], tn[ju]                      # edge (c, d) at position ju
    g = D[a, c] + D[b, d] - e[iu] - e[ju]
    k = int(np.argmin(g))
    if g[k] >= -1e-9:
        return None
    return int(iu[k]), int(ju[k]), float(g[k])


def make_aco_refine(rho=0.1, two_opt_top_k=3, two_opt_max_moves=None,
                    deposit_q=1.0, deposit_scheme="mmas",
                    asrank_w=6, eas_e=None, cand_k=16):
    """Per-generation refine: 2-opt local search on the elite, then a global
    pheromone update. Shares ctx['_aco'] with the crossover.

    `two_opt_max_moves` caps accepted 2-opt moves per individual per
    generation (default: n // 8, min 5). Each accepted move costs 1 FE.

    `deposit_scheme` selects how the (evaporated) pheromone matrix is
    reinforced:
      - "mmas"   : iteration-best and global-best tours each deposit 1/length
                   (MAX-MIN Ant System).
      - "asrank" : the (w-1) best tours in the population deposit
                   (w-r)/length_r for ranks r = 1..w-1; the global-best tour
                   deposits w/length_gb (rank-based Ant System).
      - "eas"    : every population member deposits 1/length (classic Ant
                   System), and the global-best tour additionally deposits
                   e/length_gb (Elitist Ant System; default e = n).
    After deposition the matrix is clamped to [tau_max/(2n), tau_max] with
    tau_max = 1/(rho * best_length_so_far), keeping every scheme bounded.
    """
    scheme = str(deposit_scheme).lower()
    if scheme not in ("mmas", "asrank", "eas"):
        raise ValueError(f"unknown deposit_scheme: {deposit_scheme!r}")

    def refine_aco(population, ctx):
        st = _aco_state(ctx, rho, cand_k=cand_k)
        n = st["n"]
        D = st["D"]
        eval_counter = ctx.get("eval_counter")
        max_fes = ctx.get("max_fes", float("inf"))
        cap = two_opt_max_moves if two_opt_max_moves is not None else max(5, n // 8)
        iu, ju = st["triu_i"], st["triu_j"]

        # ---- 2-opt on the top-k individuals (memetic step) ----
        order = sorted(range(len(population)),
                       key=lambda i: population[i].fitness.values[0])
        for idx in order[:max(0, two_opt_top_k)]:
            if eval_counter is not None and eval_counter[0] >= max_fes:
                break
            ind = population[idx]
            tour = _as_perm(ind, n)
            cur_len = float(ind.fitness.values[0])
            moves = 0
            while moves < cap:
                if eval_counter is not None and eval_counter[0] >= max_fes:
                    break
                mv = _two_opt_best_move(tour, D, iu, ju)
                if mv is None:
                    break
                i, j, delta = mv
                tour[i + 1:j + 1] = tour[i + 1:j + 1][::-1]
                cur_len += delta
                moves += 1
                if eval_counter is not None:
                    eval_counter[0] += 1
            if moves:
                ind[:] = tour
                ind.fitness.values = (cur_len,)

        # ---- global pheromone update: evaporate, then scheme-specific deposit ----
        tau = st["tau"]
        tau *= (1.0 - st["rho"])

        def _len(perm):
            p = np.asarray(perm)
            return float(D[p, np.roll(p, -1)].sum())

        def deposit(perm_like, amount):
            perm = _as_perm(perm_like, n)
            p = np.asarray(perm)
            q = np.roll(p, -1)
            tau[p, q] += amount
            tau[q, p] += amount

        ranked = sorted(population, key=lambda x: x.fitness.values[0])
        iter_best = ranked[0]
        best_so_far = ctx.get("best_so_far")
        gb = best_so_far if best_so_far is not None else iter_best
        len_gb = _len(_as_perm(gb, n))
        st["best_len"] = min(st["best_len"], len_gb, _len(_as_perm(iter_best, n)))

        if scheme == "mmas":
            li = _len(_as_perm(iter_best, n))
            if li > 0:
                deposit(iter_best, deposit_q / li)
            if len_gb > 0:
                deposit(gb, deposit_q / len_gb)
        elif scheme == "asrank":
            w = int(max(2, asrank_w))
            for r, ind in enumerate(ranked[:w - 1], start=1):
                length = _len(_as_perm(ind, n))
                if length > 0:
                    deposit(ind, deposit_q * (w - r) / length)
            if len_gb > 0:
                deposit(gb, deposit_q * w / len_gb)
        else:  # "eas"
            for ind in population:
                length = _len(_as_perm(ind, n))
                if length > 0:
                    deposit(ind, deposit_q / length)
            e = float(eas_e) if eas_e is not None else float(n)
            if len_gb > 0:
                deposit(gb, e * deposit_q / len_gb)

        # MMAS-style bounds, refreshed from the best length seen so far.
        st["tau_max"] = 1.0 / (st["rho"] * max(st["best_len"], 1e-9))
        st["tau_min"] = st["tau_max"] / (2.0 * n)
        st["tau0"] = 1.0 / (n * max(st["best_len"], 1e-9))
        np.clip(tau, st["tau_min"], st["tau_max"], out=tau)
        np.fill_diagonal(tau, 0.0)
        return population

    refine_aco.__name__ = f"aco_refine_2opt{two_opt_top_k}_{scheme}_rho{rho:g}"
    return refine_aco


# ============================================================================
# F86 — EAX: Edge Assembly Crossover for permutation/TSP (Nagata & Kobayashi)
# ============================================================================
#
# Single-AB-cycle ("E-set type: single") variant. Given two parent tours A and
# B:
#   1. Form the multigraph G_AB = (A-edges) ∪ (B-edges); every city has
#      A-degree 2 and B-degree 2.
#   2. Decompose G_AB into AB-cycles: closed walks that strictly alternate an
#      A-edge then a B-edge. (Always possible — the multigraph is AB-balanced.)
#   3. Pick one non-trivial AB-cycle at random as the E-set.
#   4. Intermediate child edge set = (A-edges \ {A-edges in the E-set})
#      ∪ {B-edges in the E-set}. This stays 2-regular, so it is a disjoint
#      union of one or more sub-tours.
#   5. If it is a single tour, that is the child. Otherwise, repeatedly merge
#      the smallest sub-tour into another one with the cheapest 2-edge
#      reconnection (uses the distance matrix when available), until one tour
#      remains.
# Falls back to copying the parents if a degenerate E-set would create a
# doubled edge (rare; happens only with certain shared-edge configurations).
#
# Mutation/selection/etc. are inherited from the suite baseline; the EAX
# variant simply swaps OX -> EAX in the crossover slot.


def _perm_adj_lists(perm):
    """city -> [prev_in_tour, next_in_tour] for a cyclic permutation."""
    n = len(perm)
    adj = {}
    for i, c in enumerate(perm):
        adj[c] = [perm[(i - 1) % n], perm[(i + 1) % n]]
    return adj


def _ab_cycles(A, B, n):
    """Decompose the A∪B multigraph into AB-cycles. Returns a list of cycles,
    each a list of edges [('A'|'B', u, v), ...] alternating A,B,A,B,..."""
    # Mutable adjacency multilists for A-edges and B-edges.
    aA = {c: list(v) for c, v in _perm_adj_lists(A).items()}
    aB = {c: list(v) for c, v in _perm_adj_lists(B).items()}

    def _take(adj, u, v=None):
        """Remove one edge incident to u (to v if given) from `adj`; return
        the other endpoint."""
        lst = adj[u]
        if v is None:
            v = lst[0]
        lst.remove(v)
        adj[v].remove(u)
        return v

    cycles = []
    # Process A-edges until none remain.
    remaining_A = sum(len(v) for v in aA.values()) // 2
    while remaining_A > 0:
        # Find a start vertex with an unused A-edge.
        s = next(c for c in range(n) if aA[c])
        edges = []
        cur = s
        while True:
            x = _take(aA, cur)           # A-edge cur -> x
            edges.append(("A", cur, x))
            y = _take(aB, x)             # B-edge x -> y
            edges.append(("B", x, y))
            cur = y
            if cur == s:
                break
        remaining_A -= len(edges) // 2
        cycles.append(edges)
    return cycles


def _build_child_adj(A, eset_edges, n):
    """Apply an E-set to A's edge set: drop the E-set's A-edges, add its
    B-edges. Returns the child adjacency dict, or None if it is not 2-regular
    (degenerate / doubled edge)."""
    adj = {c: list(v) for c, v in _perm_adj_lists(A).items()}
    for kind, u, v in eset_edges:
        if kind == "A":
            if v in adj[u]:
                adj[u].remove(v)
            if u in adj[v]:
                adj[v].remove(u)
        else:  # "B"
            adj[u].append(v)
            adj[v].append(u)
    for c in range(n):
        nb = adj[c]
        if len(nb) != 2 or nb[0] == nb[1]:
            return None
    return adj


def _subtours_from_adj(adj, n):
    """Trace the 2-regular graph `adj` into a list of cycles (each a list of
    cities in tour order)."""
    seen = [False] * n
    cycles = []
    for s in range(n):
        if seen[s]:
            continue
        cyc = [s]
        seen[s] = True
        prev, cur = None, s
        while True:
            a, b = adj[cur]
            nxt = b if a == prev else a
            if nxt == s:
                break
            cyc.append(nxt)
            seen[nxt] = True
            prev, cur = cur, nxt
        cycles.append(cyc)
    return cycles


def _merge_subtours(subtours, D, rng):
    """Greedily merge a list of disjoint sub-tours (each a city-ordered cycle)
    into one Hamiltonian tour. At each step the smallest sub-tour is fused into
    another with the cheapest valid 2-edge reconnection."""
    # Represent each sub-tour as its edge list (unordered pairs) for easy
    # reconnection bookkeeping; rebuild order at the end.
    tours = [list(t) for t in subtours]

    def _edges(t):
        return [(t[i], t[(i + 1) % len(t)]) for i in range(len(t))]

    while len(tours) > 1:
        tours.sort(key=len)
        small = tours[0]
        s_edges = _edges(small)
        best = None  # (delta, idx_other, e_small, e_other, pairing)
        for oi in range(1, len(tours)):
            other = tours[oi]
            o_edges = _edges(other)
            for (a, b) in s_edges:
                for (c, d) in o_edges:
                    if D is not None:
                        d1 = D[a, c] + D[b, d] - D[a, b] - D[c, d]
                        d2 = D[a, d] + D[b, c] - D[a, b] - D[c, d]
                    else:
                        d1, d2 = 0.0, 1.0
                    if d1 <= d2:
                        cand = (float(d1), oi, (a, b), (c, d), 1)
                    else:
                        cand = (float(d2), oi, (a, b), (c, d), 2)
                    if best is None or cand[0] < best[0]:
                        best = cand
            if D is None:
                break  # arbitrary merge; first other is fine
        _, oi, (a, b), (c, d), pairing = best
        other = tours[oi]
        # New edge set for the merged tour.
        new_edges = set()
        for e in s_edges:
            if {e[0], e[1]} != {a, b}:
                new_edges.add(frozenset(e))
        for e in _edges(other):
            if {e[0], e[1]} != {c, d}:
                new_edges.add(frozenset(e))
        if pairing == 1:
            new_edges.add(frozenset((a, c)))
            new_edges.add(frozenset((b, d)))
        else:
            new_edges.add(frozenset((a, d)))
            new_edges.add(frozenset((b, c)))
        # Trace the merged cycle.
        adj = {}
        for e in new_edges:
            u, v = tuple(e)
            adj.setdefault(u, []).append(v)
            adj.setdefault(v, []).append(u)
        start = next(iter(adj))
        merged = [start]
        prev, cur = None, start
        while True:
            x, y = adj[cur]
            nxt = y if x == prev else x
            if nxt == start:
                break
            merged.append(nxt)
            prev, cur = cur, nxt
        tours = [t for k, t in enumerate(tours) if k not in (0, oi)] + [merged]
    return tours[0]


def make_eax():
    """Edge Assembly Crossover (single-AB-cycle variant). Produces two
    children; each is built from an independently chosen random AB-cycle."""
    def crossover_eax(p1, p2, ctx):
        rng = ctx["rng"]
        problem = ctx.get("problem")
        n = int(getattr(problem, "ndim", 0)) or len(p1)
        D = getattr(problem, "dist", None)
        if D is not None:
            D = np.asarray(D, dtype=float)
        A = _as_perm(p1, n)
        B = _as_perm(p2, n)
        if n < 4 or A == B:
            return list(A), list(B)
        try:
            cycles = _ab_cycles(A, B, n)
        except Exception:
            return list(A), list(B)
        # Non-trivial cycles only (length-2 cycles are shared edges -> no-op).
        nz = [c for c in cycles if len(c) > 2]
        if not nz:
            return list(A), list(B)

        def one_child():
            for c in rng.permutation(len(nz)):
                try:
                    adj = _build_child_adj(A, nz[int(c)], n)
                    if adj is None:
                        continue
                    subs = _subtours_from_adj(adj, n)
                    tour = (subs[0] if len(subs) == 1
                            else _merge_subtours(subs, D, rng))
                except Exception:
                    continue
                if len(tour) == n and len(set(tour)) == n:
                    return [int(x) for x in tour]
            return list(A)

        return one_child(), one_child()

    crossover_eax.__name__ = "eax_single"
    return crossover_eax


# ============================================================================
# F87 — OX: Order Crossover for permutation/TSP (kept as an ablation point)
# ============================================================================
def crossover_ox(p1, p2, ctx):
    """Plain Order Crossover (DEAP's cxOrdered). Exposed as a feature so the
    old OX baseline stays reachable as the `SGA+OX_crossover` variant now that
    the suite baseline uses EAX."""
    c1, c2 = tools.cxOrdered(list(p1), list(p2))
    return c1, c2


# ============================================================================
# F88 — ACO-guided mutation for permutation/TSP (pairs with make_aco_refine)
# ============================================================================
def make_aco_guided_mutation(seg_frac=0.2, seg_prob=0.5, alpha=1.0, beta=3.0,
                             q0=0.3, rho=0.1, cand_k=16, swap_pm=0.02):
    """ACO-guided mutation for TSP — an *additive* enhancement that leaves the
    crossover slot alone.

    With probability `seg_prob`: pick a random contiguous segment of about
    `seg_frac`*n cities and re-order the cities inside it by Ant Colony
    System's pseudo-random-proportional rule on the shared pheromone matrix
    tau and heuristic eta = 1/dist (starting the rebuild from the city just
    before the segment; with probability q0 exploit the argmax, else sample
    roulette-wise). Otherwise fall back to a baseline-style perturbation:
    whole-tour inversion (prob 1/2) or per-position swap with rate `swap_pm`.

    Shares ctx['_aco'] with `make_aco_refine`, so once the refine starts
    depositing pheromone the segment rebuilds are steered by it; in the very
    first generation tau is uniform, so the rebuild is purely greedy-heuristic.
    Only reads tau (the refine owns evaporation/deposit/bounds).
    """
    def mutate_aco_guided(ind, ctx):
        rng = ctx["rng"]
        st = _aco_state(ctx, rho, cand_k=cand_k)
        n = st["n"]
        tour = _as_perm(ind, n)
        if n >= 5 and rng.random() < seg_prob:
            tau, eta = st["tau"], st["eta"]
            L = int(max(2, min(n - 1, round(seg_frac * n))))
            i = int(rng.integers(0, n))                 # segment start position
            seg_pos = [(i + t) % n for t in range(L)]
            anchor = tour[(i - 1) % n]                  # city before the segment
            remaining = set(tour[p] for p in seg_pos)
            new_order = []
            cur = anchor
            while remaining:
                cand = np.fromiter(remaining, dtype=int, count=len(remaining))
                w = (tau[cur, cand] ** alpha) * (eta[cur, cand] ** beta)
                tot = float(w.sum())
                if not np.isfinite(tot) or tot <= 0.0:
                    nxt = int(cand[rng.integers(0, len(cand))])
                elif rng.random() < q0:
                    nxt = int(cand[int(np.argmax(w))])
                else:
                    nxt = int(cand[rng.choice(len(cand), p=w / tot)])
                new_order.append(nxt)
                remaining.discard(nxt)
                cur = nxt
            for p, c in zip(seg_pos, new_order):
                tour[p] = c
            return tour
        # --- baseline-style perturbation fallback ---
        if rng.random() < 0.5:
            a, b = sorted(rng.integers(0, n, size=2).tolist())
            tour[a:b + 1] = tour[a:b + 1][::-1]
        else:
            for p in range(n):
                if rng.random() < swap_pm:
                    qpos = int(rng.integers(0, n))
                    tour[p], tour[qpos] = tour[qpos], tour[p]
        return tour

    mutate_aco_guided.__name__ = f"aco_guided_mut_seg{seg_frac:g}_p{seg_prob:g}"
    return mutate_aco_guided


# ============================================================================
# F86 — Permutation-aware re-init / diversity factories (TSP)
# ============================================================================
#
# Direct perm-aware factories that mirror their real-coded counterparts but
# (a) re-initialise with random permutations instead of uniform draws in
# [lb, ub], and (b) measure individual-to-individual distance with edge-set
# Jaccard rather than Euclidean. Used for the five re-init / distance-
# diversity operators (restart, abc_scout, cuckoo_discovery, net_suppress,
# immune_memory). The seven gene-arithmetic features (DE, SHADE, GSA, BFO,
# krill, SOS, BBO) have edge-aware TSP-native rewrites in F87–F93 below.


def _random_perm(n, rng):
    """Random permutation of 0..n-1 as a plain Python-int list."""
    p = np.arange(n)
    rng.shuffle(p)
    return p.tolist()


def _edges(perm):
    """Undirected edge set of a cyclic TSP tour, with unordered pairs."""
    n = len(perm)
    s = set()
    for i in range(n):
        a = int(perm[i]); b = int(perm[(i + 1) % n])
        s.add((a, b) if a < b else (b, a))
    return s


def _edge_jaccard_dist(p, q):
    """1 - |edges(p) ∩ edges(q)| / |edges(p) ∪ edges(q)|. Returns in [0, 1]."""
    ep, eq = _edges(p), _edges(q)
    union = len(ep | eq)
    if union == 0:
        return 0.0
    return 1.0 - len(ep & eq) / union


def make_diversity_restart_perm(stagnation_threshold=20, restart_fraction=0.2):
    """Perm analogue of make_diversity_restart: replace worst k% with random
    permutations once stagnation_counter crosses the threshold."""
    def diversity_restart_perm(population, ctx):
        if ctx["stagnation_counter"] < stagnation_threshold:
            return population
        rng = ctx["rng"]
        n_restart = int(len(population) * restart_fraction)
        if n_restart <= 0:
            return population
        sorted_pop = sorted(population, key=lambda x: x.fitness.values[0])
        keep = sorted_pop[: len(population) - n_restart]
        fresh = []
        for _ in range(n_restart):
            v = _random_perm(int(ctx["dim"]), rng)
            fresh.append(creator.Individual(v))
        ctx["stagnation_counter"] = 0
        return keep + fresh
    return diversity_restart_perm


def make_abc_scout_perm(limit=None):
    """Perm analogue of make_abc_scout: trial-counter scout abandonment, but
    abandoned slots are reinitialised with a random permutation."""
    def replace_abc_perm(parents, offspring, ctx):
        rng = ctx["rng"]
        dim = int(ctx["dim"])
        pop_size = ctx["pop_size"]
        lim = limit if limit is not None else max(1, pop_size * dim)

        survivors = generational_elitism(parents, offspring, int(ctx["elitism_k"]))

        parents_by_f = sorted(parents, key=lambda x: x.fitness.values[0])
        offspring_by_f = sorted(offspring, key=lambda x: x.fitness.values[0])
        m = min(len(parents_by_f), len(offspring_by_f))
        improved_ids = set()
        unimproved_ids = set()
        for i in range(m):
            p = parents_by_f[i]
            o = offspring_by_f[i]
            if o.fitness.values[0] < p.fitness.values[0]:
                improved_ids.add(id(p))
            else:
                unimproved_ids.add(id(p))

        for ind in survivors:
            t = getattr(ind, "trial", 0)
            if id(ind) in improved_ids:
                ind.trial = 0
            elif id(ind) in unimproved_ids:
                ind.trial = t + 1
            else:
                ind.trial = 0

        for i, ind in enumerate(survivors):
            if getattr(ind, "trial", 0) > lim:
                scout = creator.Individual(_random_perm(dim, rng))
                scout.age = 0
                scout.trial = 0
                survivors[i] = scout
        return survivors
    return replace_abc_perm


def make_cuckoo_discovery_perm(pa=0.25):
    """Perm analogue of make_cuckoo_discovery: with probability pa, replace
    a nest with a random permutation (the perm-native form of biased-walk
    discovery, since on a tour there is no meaningful (xj - xk) vector)."""
    def diversity_cuckoo_perm(population, ctx):
        rng = ctx["rng"]
        dim = int(ctx["dim"])
        for i in range(len(population)):
            if rng.random() >= pa:
                continue
            new_ind = creator.Individual(_random_perm(dim, rng))
            new_ind.age = 0
            population[i] = new_ind
        return population
    return diversity_cuckoo_perm


def make_network_suppression_perm(threshold_frac=0.02):
    """Perm analogue of make_network_suppression. Distance is edge-set
    Jaccard; threshold_frac is the *minimum* fraction of disagreeing edges
    below which two tours are considered too similar (worse is dropped)."""
    def diversity_suppress_perm(population, ctx):
        rng = ctx["rng"]
        dim = int(ctx["dim"])
        thr = float(threshold_frac)
        pop = list(population)
        # Pre-compute edge sets once per generation.
        perms = [_as_perm(p, dim) for p in pop]
        edge_sets = [_edges(p) for p in perms]
        keep_mask = [True] * len(pop)
        for i in range(len(pop)):
            if not keep_mask[i]:
                continue
            for j in range(i + 1, len(pop)):
                if not keep_mask[j]:
                    continue
                ep, eq = edge_sets[i], edge_sets[j]
                union = len(ep | eq)
                d = 1.0 - len(ep & eq) / union if union else 0.0
                if d < thr:
                    worse = i if pop[i].fitness.values[0] > pop[j].fitness.values[0] else j
                    keep_mask[worse] = False
        out = [pop[i] for i in range(len(pop)) if keep_mask[i]]
        n_drop = len(pop) - len(out)
        for _ in range(n_drop):
            out.append(creator.Individual(_random_perm(dim, rng)))
        return out
    return diversity_suppress_perm


def make_immune_memory_pool_perm(k_inject=2, dist_thresh_frac=0.05):
    """Perm analogue of make_immune_memory_pool: novelty is measured by
    edge-set Jaccard distance to existing memory members."""
    def diversity_immune_perm(population, ctx):
        dim = int(ctx["dim"])
        memory = ctx.setdefault("immune_memory_perm_pool", [])
        thr = float(dist_thresh_frac)
        cur_best = _best_in_pop(population)
        cur_perm = _as_perm(cur_best, dim)
        if not memory:
            mc = creator.Individual(list(cur_best))
            mc.fitness.values = cur_best.fitness.values
            memory.append(mc)
        else:
            min_d = min(_edge_jaccard_dist(cur_perm, _as_perm(m, dim)) for m in memory)
            if min_d > thr and len(memory) < 20:
                mc = creator.Individual(list(cur_best))
                mc.fitness.values = cur_best.fitness.values
                memory.append(mc)
        if not memory:
            return population
        sorted_pop = _sorted_pop(population)
        n_inject = min(k_inject, len(memory), len(population) // 2)
        if n_inject <= 0:
            return population
        keep = sorted_pop[: len(population) - n_inject]
        injected = []
        for m in memory[-n_inject:]:
            clone = creator.Individual(list(m))
            clone.fitness.values = m.fitness.values
            injected.append(clone)
        return keep + injected
    return diversity_immune_perm


# ============================================================================
# F87–F93 — TSP-native (edge-aware) versions of the real-coded variants
# ============================================================================
#
# Each operator below keeps the *metaphor* of the real-coded feature it is
# named after, but the mathematical update is rewritten in the language of
# tours and edges so the operator actually exploits TSP problem structure
# (city adjacency, distance matrix, candidate lists, edge sets) rather than
# coerced floats.
#
# Common pattern:
#   - Inputs are valid permutations (run through _as_perm for safety).
#   - "Difference between two tours" is an edge-set delta (F87/F88/F91/F92/F93).
#   - "Gravity / attraction" becomes mass-weighted edge voting (F89).
#   - "Step in a direction" becomes a 2-opt move, optionally restricted to
#     k-nearest-neighbour candidate lists (F90).
#   - Operator parameters (F, CR, G0, alpha, Vf, ...) carry their original
#     meanings whenever the analogue is exact, so SHADE's adaptive F/CR
#     memory in ctx["shade_memory"] still drives F88 unchanged.


def _two_opt_delta(perm, D, i, j):
    """Distance delta for the 2-opt move that reverses perm[i:j+1]. The
    affected edges are (perm[i-1], perm[i]) and (perm[j], perm[j+1 mod n])."""
    n = len(perm)
    a = perm[i - 1]; b = perm[i]
    c = perm[j];     d = perm[(j + 1) % n]
    return D[a, c] + D[b, d] - D[a, b] - D[c, d]


def _ox_insert_segment(target, segment, insert_pos):
    """OX-style insertion: drop every city in `segment` from a copy of
    `target`, then re-insert `segment` contiguously at `insert_pos`.
    Returned permutation has the same length as `target` and is valid."""
    seg_set = set(int(c) for c in segment)
    rest = [int(c) for c in target if int(c) not in seg_set]
    insert_pos = max(0, min(insert_pos, len(rest)))
    return rest[:insert_pos] + [int(c) for c in segment] + rest[insert_pos:]


# ----------------------------------------------------------------------------
# Edge-based DE helpers
# ----------------------------------------------------------------------------
# Apply an edge (a, b) to a tour via a single 2-opt reversal. If `a` and `b`
# sit at positions ia < ib in the tour, reversing the segment perm[ia+1:ib+1]
# places `perm[ia]` adjacent to `perm[ib]` — i.e. introduces edge (a, b).
# Returns True if the edge was inserted, False if the move is degenerate
# (positions adjacent or coincident).

def _insert_edge_2opt(perm, a, b):
    try:
        ia = perm.index(int(a)); ib = perm.index(int(b))
    except ValueError:
        return False
    lo, hi = (ia, ib) if ia < ib else (ib, ia)
    if hi - lo < 2:
        return False
    perm[lo + 1:hi + 1] = perm[lo + 1:hi + 1][::-1]
    return True


def _apply_edge_set(perm, edges, k, rng, cr=None, D=None, only_improving=False):
    """Apply up to k edges from `edges` (an iterable of unordered (a,b)
    pairs) to `perm` via 2-opt insertion. Each edge is independently gated
    by Bernoulli(cr) when cr is provided. When `only_improving` is set and
    D is available, the move is accepted only if it reduces tour length.
    Edges are tried in randomised order so the operator is not biased by
    the iteration order of Python sets."""
    if k <= 0:
        return perm
    pool = list(edges)
    if not pool:
        return perm
    rng.shuffle(pool)
    applied = 0
    for (a, b) in pool:
        if cr is not None and rng.random() >= cr:
            continue
        if only_improving and D is not None:
            try:
                ia = perm.index(int(a)); ib = perm.index(int(b))
            except ValueError:
                continue
            lo, hi = (ia, ib) if ia < ib else (ib, ia)
            if hi - lo < 2:
                continue
            # Delta for the edge-insertion 2-opt (reverses perm[lo+1:hi+1]):
            # removes edges (perm[lo], perm[lo+1]) and (perm[hi], perm[hi+1])
            # and adds edges (perm[lo], perm[hi]) and (perm[lo+1], perm[hi+1]).
            n = len(perm)
            removed = D[perm[lo], perm[lo + 1]] + D[perm[hi], perm[(hi + 1) % n]]
            added = D[perm[lo], perm[hi]] + D[perm[lo + 1], perm[(hi + 1) % n]]
            if added - removed >= 0:
                continue
            perm[lo + 1:hi + 1] = perm[lo + 1:hi + 1][::-1]
            applied += 1
        else:
            if _insert_edge_2opt(perm, a, b):
                applied += 1
        if applied >= k:
            break
    return perm


# ----------------------------------------------------------------------------
# F87 — Permutation DE/rand/1 (edge-based)
# ----------------------------------------------------------------------------
# DE/rand/1 says v = r1 + F * (r2 - r3). On a tour, the "direction" (r2 -
# r3) is the symmetric edge difference: edges that exist in r2 but not in
# r3 (the direction r3 -> r2 in edge space). The mutation copies r1 and
# pushes it along that direction by injecting round(F * |donor_edges|) of
# those edges via 2-opt reversals. Each 2-opt changes exactly two edges,
# so F scales how many TSP-meaningful edges are inherited from the
# differential.

def make_de_perm(F=0.5):
    def mutate_de_perm(ind, ctx):
        rng = ctx["rng"]
        n = int(ctx["dim"])
        pop = ctx.get("_population", []) or []
        ind_id = id(ind)
        pool_idx = [i for i, p in enumerate(pop) if id(p) != ind_id]
        if len(pool_idx) < 3:
            return _as_perm(ind, n)
        picks = rng.choice(pool_idx, size=3, replace=False)
        r1 = _as_perm(pop[int(picks[0])], n)
        r2 = _as_perm(pop[int(picks[1])], n)
        r3 = _as_perm(pop[int(picks[2])], n)
        donor_edges = _edges(r2) - _edges(r3)
        out = list(r1)
        k = max(1, int(round(F * len(donor_edges))))
        # Discrete DE on TSP needs an acceptance filter: with uncorrelated
        # r2/r3 the symmetric edge difference is mostly random edges, and
        # injecting F * |donor| of them into r1 without a filter is closer
        # to randomisation than to "differential exploration". An improving-
        # only 2-opt acceptance (Onwubolu & Davendra 2009 style) preserves
        # the DE metaphor — direction = donor edges, scale = F — while
        # restricting acceptance to TSP-meaningful moves.
        D = None
        try:
            D = _aco_problem_dist(ctx)
        except ValueError:
            D = None
        return _apply_edge_set(out, donor_edges, k, rng,
                               D=D, only_improving=D is not None)
    mutate_de_perm.__name__ = f"de_perm_edge_F{F:g}"
    return mutate_de_perm


# ----------------------------------------------------------------------------
# F88 — Permutation SHADE (current-to-pbest/1, edge-based)
# ----------------------------------------------------------------------------
# SHADE: u = x + F*(pbest - x) + F*(r1 - r2), with F ~ Cauchy(muF, 0.1) and
# CR ~ Normal(muCR, 0.1) drawn from a per-population memory. Permutation
# form:
#   * Target edges = E(pbest) \ E(cur)            (the "to pbest" direction)
#   * Differential edges = E(r1) \ E(r2)          (the random direction)
# Apply up to F*|target| target edges then F*|differential| differential
# edges to a copy of `cur`. Each candidate edge is independently gated by
# Bernoulli(CR), and acceptance is restricted to improving 2-opt moves —
# the analogue of "u_j = v_j if rand<CR" combined with the standard SHADE
# practice of recording F/CR only on improvement. Improving-only gives the
# mutation a SHADE-flavoured trial-and-keep semantics without needing to
# call the evaluator inside the mutate slot.

def make_shade_perm():
    def mutate_shade_perm(ind, ctx):
        rng = ctx["rng"]
        n = int(ctx["dim"])
        pop = ctx.get("_population", []) or []
        if len(pop) < 4:
            return _as_perm(ind, n)
        mem = ctx["shade_memory"]
        muF = float(np.mean(mem["F"]))
        muCR = float(np.mean(mem["CR"]))
        F = float(np.clip(rng.standard_cauchy() * 0.1 + muF, 0.05, 1.0))
        CR = float(np.clip(rng.normal(muCR, 0.1), 0.0, 1.0))

        sorted_pop = _sorted_pop(pop)
        p_size = max(2, int(0.2 * len(sorted_pop)))
        pbest = _as_perm(sorted_pop[int(rng.integers(0, p_size))], n)
        idx = rng.choice(len(pop), size=2, replace=False)
        r1 = _as_perm(pop[int(idx[0])], n)
        r2 = _as_perm(pop[int(idx[1])], n)
        cur = _as_perm(ind, n)
        D = None
        try:
            D = _aco_problem_dist(ctx)
        except ValueError:
            D = None

        target = _edges(pbest) - _edges(cur)
        diff = _edges(r1) - _edges(r2)
        k1 = max(1, int(round(F * len(target))))
        k2 = max(1, int(round(F * len(diff))))
        out = list(cur)
        out = _apply_edge_set(out, target, k1, rng, cr=CR,
                              D=D, only_improving=D is not None)
        out = _apply_edge_set(out, diff, k2, rng, cr=CR,
                              D=D, only_improving=D is not None)

        mem["F"].append(F); mem["CR"].append(CR)
        return out
    return mutate_shade_perm


# ----------------------------------------------------------------------------
# F89 — Permutation GSA (mass-weighted edge voting + biased 2-opt)
# ----------------------------------------------------------------------------
# GSA: lighter masses are pulled toward heavier (fitter) ones via Newtonian
# gravity, with the active-set Kbest shrinking over time (G(t) = G0 *
# exp(-alpha * t/T)). Permutation form: each kbest tour casts a mass-
# weighted vote for each of its edges; the current tour then performs
# G(t)-many 2-opt moves where the partner `j` is sampled proportionally to
# the edge vote with the current city. This makes the operator pull the
# tour toward the *edge structure* of the high-mass cluster — the perm
# analogue of being pulled in a high-gravity direction.

def make_gsa_full_perm(G0=100.0, alpha=20.0, kbest_final_frac=0.02):
    def mutate_gsa_perm(ind, ctx):
        rng = ctx["rng"]
        n = int(ctx["dim"])
        pop = ctx.get("_population", []) or []
        if len(pop) < 4:
            return _as_perm(ind, n)
        D = _aco_problem_dist(ctx)
        t = float(ctx.get("t_ratio", 0.0))
        # Kbest shrinks from |pop| to kbest_final_frac * |pop| over the run
        k = max(2, int(len(pop) * (1.0 - t * (1.0 - kbest_final_frac))))
        sorted_pop = _sorted_pop(pop)[:k]
        fits = np.array([p.fitness.values[0] for p in sorted_pop], dtype=float)
        span = max(float(fits.max() - fits.min()), 1e-12)
        masses = (fits.max() - fits) / span  # heaviest = best
        if masses.sum() <= 0:
            return _as_perm(ind, n)

        edge_vote = np.zeros((n, n), dtype=float)
        for m_i, p in zip(masses, sorted_pop):
            perm = _as_perm(p, n)
            for ii in range(n):
                a, b = perm[ii], perm[(ii + 1) % n]
                edge_vote[a, b] += float(m_i)
                edge_vote[b, a] += float(m_i)

        cur = _as_perm(ind, n)
        G_t = G0 * float(np.exp(-alpha * t / max(1.0, ctx.get("max_gen_estimate", 1))))
        n_moves = max(1, min(int(G_t), max(1, n // 4)))
        for _ in range(n_moves):
            i = int(rng.integers(1, n - 1))
            cur_arr = np.asarray(cur)
            votes = edge_vote[cur_arr[i], cur_arr] + 1e-9
            # Disallow no-op picks (immediate neighbours produce length-<2 segments)
            votes[max(0, i - 1):min(n, i + 2)] = 0.0
            tot = float(votes.sum())
            if tot <= 0:
                continue
            j = int(rng.choice(n, p=votes / tot))
            lo, hi = (i, j) if i < j else (j, i)
            if hi - lo >= 2:
                # Accept only improving 2-opt moves (gravity = bias toward
                # lower-energy configurations).
                if _two_opt_delta(cur, D, lo, hi) < 0:
                    cur[lo:hi + 1] = cur[lo:hi + 1][::-1]
        return cur
    return mutate_gsa_perm


# ----------------------------------------------------------------------------
# F90 — Permutation BFO chemotaxis (2-opt swim & tumble)
# ----------------------------------------------------------------------------
# BFO: a tumble is a single step in a random direction; a swim is to keep
# stepping in the same direction while fitness improves, up to N_s times.
# Permutation form: a tumble is a 2-opt move where the partner `j` is the
# city's k-NN candidate-list neighbour (the perm analogue of a small-
# magnitude direction). If the tumble improves the tour, swim by trying
# more candidate-list 2-opt moves anchored at the same position while they
# keep improving.

def make_bfo_chemotaxis_perm(N_s=4):
    def mutate_bfo_perm(ind, ctx):
        rng = ctx["rng"]
        n = int(ctx["dim"])
        st = _aco_state(ctx, rho=0.1, cand_k=16)
        D = st["D"]
        cand = st["cand"]
        cur = _as_perm(ind, n)

        def _candidate_2opt(anchor):
            choices = [int(c) for c in cand[cur[anchor]] if int(c) != cur[anchor]]
            if not choices:
                return None
            partner = int(rng.choice(choices))
            try:
                jpos = cur.index(partner)
            except ValueError:
                return None
            lo, hi = (anchor, jpos) if anchor < jpos else (jpos, anchor)
            if hi - lo < 2:
                return None
            return lo, hi

        # Tumble
        anchor = int(rng.integers(1, n))
        move = _candidate_2opt(anchor)
        if move is None:
            return cur
        lo, hi = move
        if _two_opt_delta(cur, D, lo, hi) < 0:
            cur[lo:hi + 1] = cur[lo:hi + 1][::-1]
            # Swim while improving
            for _ in range(N_s):
                move = _candidate_2opt(lo)
                if move is None:
                    break
                lo2, hi2 = move
                if _two_opt_delta(cur, D, lo2, hi2) >= 0:
                    break
                cur[lo2:hi2 + 1] = cur[lo2:hi2 + 1][::-1]
        return cur
    return mutate_bfo_perm


# ----------------------------------------------------------------------------
# F91 — Permutation Krill Herd (food + herd attraction + random drift)
# ----------------------------------------------------------------------------
# Krill Herd: position update has three components — induced motion from
# the food location (best so far), neighbour-density motion from the herd,
# and physical diffusion (random drift). Permutation form:
#   * Food attraction (Vf): adopt one edge from best-so-far that the
#     current tour lacks, via 2-opt repair.
#   * Herd cohesion (Nmax): adopt one edge from the nearest-fitness
#     neighbour in the population.
#   * Diffusion (Dmax): a random 2-opt move.
# The defaults reflect the original Vf >> Nmax > Dmax ranking.

def make_krill_herd_perm(Vf=0.6, Nmax=0.35, Dmax=0.2):
    def mutate_krill_perm(ind, ctx):
        rng = ctx["rng"]
        n = int(ctx["dim"])
        pop = ctx.get("_population", []) or []
        bs = ctx.get("best_so_far")
        cur = _as_perm(ind, n)
        cur_edges = _edges(cur)

        def _adopt_edge_from(donor_perm):
            diff = list(_edges(donor_perm) - _edges(cur))
            if not diff:
                return False
            a, b = diff[int(rng.integers(0, len(diff)))]
            try:
                ia = cur.index(int(a)); ib = cur.index(int(b))
            except ValueError:
                return False
            lo, hi = sorted((ia, ib))
            if hi - lo >= 2:
                cur[lo + 1:hi + 1] = cur[lo + 1:hi + 1][::-1]
                return True
            return False

        if bs is not None and rng.random() < Vf:
            _adopt_edge_from(_as_perm(bs, n))
        if pop and rng.random() < Nmax:
            others = [p for p in pop if id(p) != id(ind)]
            if others:
                try:
                    my_f = ind.fitness.values[0]
                    neigh = min(others, key=lambda p: abs(p.fitness.values[0] - my_f))
                except (AttributeError, IndexError):
                    neigh = others[int(rng.integers(0, len(others)))]
                _adopt_edge_from(_as_perm(neigh, n))
        if rng.random() < Dmax:
            i, j = sorted(rng.integers(0, n, size=2).tolist())
            if j - i >= 2:
                cur[i:j + 1] = cur[i:j + 1][::-1]
        return cur
    return mutate_krill_perm


# ----------------------------------------------------------------------------
# F92 — Permutation SOS (mutualism / commensalism / parasitism)
# ----------------------------------------------------------------------------
# Symbiotic Organisms Search has three explicit interaction phases:
#   * Mutualism (both benefit): OX-style segment exchange with a partner —
#     the partner donates a contiguous segment, the current organism
#     integrates it.
#   * Commensalism (one benefits, one unchanged): the current organism
#     adopts a single beneficial edge from the partner.
#   * Parasitism (random sabotage): a random segment reversal.
# Phase is chosen uniformly per call; the three together approximate the
# original triple-step SOS generation.

def make_sos_perm():
    def mutate_sos_perm(ind, ctx):
        rng = ctx["rng"]
        n = int(ctx["dim"])
        pop = ctx.get("_population", []) or []
        cur = _as_perm(ind, n)
        if len(pop) < 2:
            # Parasitism only
            a, b = sorted(rng.integers(0, n, size=2).tolist())
            if b - a >= 2:
                cur[a:b + 1] = cur[a:b + 1][::-1]
            return cur

        phase = int(rng.integers(0, 3))
        if phase == 0:
            j = int(rng.integers(0, len(pop)))
            partner = _as_perm(pop[j], n)
            a, b = sorted(rng.integers(0, n, size=2).tolist())
            return _ox_insert_segment(cur, partner[a:b + 1], a)
        if phase == 1:
            j = int(rng.integers(0, len(pop)))
            partner = _as_perm(pop[j], n)
            diff = list(_edges(partner) - _edges(cur))
            if not diff:
                return cur
            a, b = diff[int(rng.integers(0, len(diff)))]
            try:
                ia = cur.index(int(a)); ib = cur.index(int(b))
            except ValueError:
                return cur
            lo, hi = sorted((ia, ib))
            if hi - lo >= 2:
                cur[lo + 1:hi + 1] = cur[lo + 1:hi + 1][::-1]
            return cur
        # Parasitism
        a, b = sorted(rng.integers(0, n, size=2).tolist())
        if b - a >= 2:
            cur[a:b + 1] = cur[a:b + 1][::-1]
        return cur
    return mutate_sos_perm


# ----------------------------------------------------------------------------
# F93 — Permutation BBO migration crossover
# ----------------------------------------------------------------------------
# BBO: each habitat has emigration rate mu (high for fit habitats) and
# immigration rate lambda (high for unfit habitats); features migrate from
# high-mu to high-lambda. On permutations, a "feature" is a contiguous
# edge segment. The fitter parent acts as donor and contributes a random
# segment; the recipient OX-integrates it. Two children are produced with
# donor/recipient roles swapped, mirroring the original crossover contract.

def crossover_bbo_perm(p1, p2, ctx):
    rng = ctx["rng"]
    n = int(ctx["dim"])
    a = _as_perm(p1, n); b = _as_perm(p2, n)
    try:
        f1 = p1.fitness.values[0]; f2 = p2.fitness.values[0]
    except (AttributeError, IndexError):
        f1 = f2 = 0.0
    if f1 <= f2:
        donor, recipient = a, b
    else:
        donor, recipient = b, a

    def _migrate(don, rec):
        i, j = sorted(rng.integers(0, n, size=2).tolist())
        return _ox_insert_segment(rec, don[i:j + 1], i)

    return _migrate(donor, recipient), _migrate(recipient, donor)


# ============================================================
# Yang & Jat 2011 SSGA operators (UCTP suite)
# ------------------------------------------------------------
# Steady-state GA building blocks for the University Course Timetabling
# Problem. These pair with `run_ssga` in `ga.core` and the `UCTPProblem`
# evaluator in `benchmarks.uctp` (which exposes `evaluate_full`
# and `assign_rooms` for delta-style updates).
#
#   replace_worst        — worst-of-pop replacement (Algorithm 1, line 9)
#   mutate_ssga_n1n2n3   — pick one random neighborhood (N1/N2/N3) and
#                          make ONE random move (paper §IV-A mutation)
#   make_refine_ls1      — LS1 (Algorithm 2): two-phase local search
#                          across N1+N2 with first-improvement acceptance
# ============================================================

def replace_worst(parents, offspring, ctx):
    """Steady-state replacement: drop the `len(offspring)` worst members
    of the parent population and append the offspring. Used by Yang & Jat
    2011 SSGA (Algorithm 1, line 9). Keeps population size constant.
    """
    pool = list(parents)
    pool.sort(key=lambda ind: ind.fitness.values[0])
    keep = pool[: max(0, len(parents) - len(offspring))]
    return keep + list(offspring)


_UCTP_N_TIMESLOTS = 45  # matches benchmarks.uctp.N_TIMESLOTS


def mutate_ssga_n1n2n3(ind, ctx):
    """Yang & Jat 2011 mutation: randomly pick one of {N1, N2, N3} and
    apply ONE move from that neighborhood.

      N1: move one event to a different timeslot.
      N2: swap the timeslots of two events.
      N3: 3-cycle permute three events' timeslots (one of the two
          non-identity permutations).

    Operates on a discrete integer vector (timeslots in [0, 44]).
    """
    rng = ctx["rng"]
    seq = [int(v) for v in ind]
    n = len(seq)
    if n < 2:
        return seq
    # ub may be a numpy array (per-gene); for UCTP every gene shares the
    # same domain so reading position 0 is sufficient.
    ub = ctx.get("ub")
    if ub is None or len(ub) == 0:
        n_slots = _UCTP_N_TIMESLOTS
    else:
        n_slots = int(ub[0]) + 1

    pick = int(rng.integers(0, 3))
    if pick == 0:
        # N1: move one event to a different slot
        e = int(rng.integers(0, n))
        new_slot = int(rng.integers(0, n_slots))
        while new_slot == seq[e] and n_slots > 1:
            new_slot = int(rng.integers(0, n_slots))
        _n1_move(seq, e, new_slot)
    elif pick == 1 or n < 3:
        # N2: swap two events' slots
        i, j = rng.choice(n, size=2, replace=False).tolist()
        _n2_swap(seq, i, j)
    else:
        # N3: 3-cycle permutation (one of the two cyclic permutations of
        # three distinct events' slots, other than the identity).
        i, j, k = rng.choice(n, size=3, replace=False).tolist()
        _n3_cycle(seq, i, j, k, forward=rng.random() < 0.5)
    return seq


# ============================================================
# F94–F106 — UCTP-native (timeslot-vector) versions of the real-coded variants
# ------------------------------------------------------------
# Each operator keeps the metaphor of the named real-coded feature, but the
# update is rewritten in the language of timeslot assignments (discrete int
# vector of length n_events, each gene in [0..N_TIMESLOTS-1]).
#
# Common pattern:
#   - Adapters compose the Yang & Jat primitives _n1_move / _n2_swap / _n3_cycle
#     so the move-set is commensurable with the SSGA baseline.
#   - "Difference between two solutions" is the set of disagreeing genes
#     (Hamming-positive positions).
#   - "Step toward donor" is an N1 move setting x[i] = donor[i].
#   - Distance metric for net_suppress / immune_memory is Hamming on integer
#     vectors, threshold scaled to chromosome length.
#   - LS1 refine is owned by the SSGA loop; these adapters never touch it.
# ============================================================


def _uctp_n_slots(ctx):
    ub = ctx.get("ub")
    if ub is None or len(ub) == 0:
        return _UCTP_N_TIMESLOTS
    return int(ub[0]) + 1


def _as_slot_vec(ind, n):
    return [int(v) for v in ind][:n]


def _n1_move(seq, e, new_slot):
    seq[e] = int(new_slot)
    return seq


def _n2_swap(seq, i, j):
    seq[i], seq[j] = seq[j], seq[i]
    return seq


def _n3_cycle(seq, i, j, k, forward=True):
    if forward:
        seq[i], seq[j], seq[k] = seq[j], seq[k], seq[i]
    else:
        seq[i], seq[j], seq[k] = seq[k], seq[i], seq[j]
    return seq


def _hamming_int(a, b):
    return sum(1 for x, y in zip(a, b) if int(x) != int(y))


def _disagreeing_positions(a, b):
    return [i for i in range(min(len(a), len(b))) if int(a[i]) != int(b[i])]


def _sorted_pop_uctp(pop):
    return sorted(pop, key=lambda x: x.fitness.values[0])


def _random_int_vec(n, n_slots, rng):
    return rng.integers(0, n_slots, size=n).astype(int).tolist()


def _ind_fitness(ind, default=float("inf")):
    try:
        return float(ind.fitness.values[0])
    except (AttributeError, IndexError):
        return default


# ------------------------------------------------------------
# F94 — UCTP DE/rand/1
# ------------------------------------------------------------
def make_de_uctp(F=0.5):
    def mutate_de_uctp(ind, ctx):
        rng = ctx["rng"]
        n = int(ctx["dim"])
        pop = ctx.get("_population", []) or []
        pool = [p for p in pop if id(p) != id(ind)]
        if len(pool) < 3:
            return mutate_ssga_n1n2n3(ind, ctx)
        idx = rng.choice(len(pool), size=3, replace=False)
        r1 = _as_slot_vec(pool[int(idx[0])], n)
        r2 = _as_slot_vec(pool[int(idx[1])], n)
        r3 = _as_slot_vec(pool[int(idx[2])], n)
        diff_pos = _disagreeing_positions(r2, r3)
        if not diff_pos:
            return mutate_ssga_n1n2n3(ind, ctx)
        out = list(r1)
        for i in diff_pos:
            if rng.random() < F:
                out[i] = int(r2[i])
        return out
    mutate_de_uctp.__name__ = f"de_uctp_F{F:g}"
    return mutate_de_uctp


# ------------------------------------------------------------
# F95 — UCTP SHADE (current-to-pbest/1 with adaptive F/CR)
# ------------------------------------------------------------
def make_shade_uctp():
    def mutate_shade_uctp(ind, ctx):
        rng = ctx["rng"]
        n = int(ctx["dim"])
        pop = ctx.get("_population", []) or []
        if len(pop) < 4:
            return mutate_ssga_n1n2n3(ind, ctx)
        mem = ctx["shade_memory"]
        muF = float(np.mean(mem["F"]))
        muCR = float(np.mean(mem["CR"]))
        F = float(np.clip(rng.standard_cauchy() * 0.1 + muF, 0.05, 1.0))
        CR = float(np.clip(rng.normal(muCR, 0.1), 0.0, 1.0))
        sorted_pop = _sorted_pop_uctp(pop)
        p_size = max(2, int(0.2 * len(sorted_pop)))
        pbest = _as_slot_vec(sorted_pop[int(rng.integers(0, p_size))], n)
        idx = rng.choice(len(pop), size=2, replace=False)
        r1 = _as_slot_vec(pop[int(idx[0])], n)
        r2 = _as_slot_vec(pop[int(idx[1])], n)
        cur = _as_slot_vec(ind, n)
        target = _disagreeing_positions(pbest, cur)
        diff = _disagreeing_positions(r1, r2)
        rng.shuffle(target); rng.shuffle(diff)
        out = list(cur)
        if target:
            k1 = max(1, int(round(F * len(target))))
            for i in target[:k1]:
                if rng.random() < CR:
                    out[i] = int(pbest[i])
        if diff:
            k2 = max(1, int(round(F * len(diff))))
            for i in diff[:k2]:
                if rng.random() < CR:
                    out[i] = int(r1[i])
        mem["F"].append(F); mem["CR"].append(CR)
        return out
    return mutate_shade_uctp


# ------------------------------------------------------------
# F96 — UCTP diversity restart
# ------------------------------------------------------------
def make_diversity_restart_uctp(stagnation_threshold=15, restart_fraction=0.2):
    def diversity_restart_uctp(population, ctx):
        if ctx["stagnation_counter"] < stagnation_threshold:
            return population
        rng = ctx["rng"]
        n = int(ctx["dim"])
        n_slots = _uctp_n_slots(ctx)
        n_restart = int(len(population) * restart_fraction)
        if n_restart <= 0:
            return population
        sorted_pop = _sorted_pop_uctp(population)
        keep = sorted_pop[: len(population) - n_restart]
        fresh = [creator.Individual(_random_int_vec(n, n_slots, rng))
                 for _ in range(n_restart)]
        ctx["stagnation_counter"] = 0
        return keep + fresh
    return diversity_restart_uctp


# ------------------------------------------------------------
# F97 — UCTP MMAS τ_min/τ_max pheromone
# ------------------------------------------------------------
def make_pheromone_minmax_uctp(rho=0.1, tau_min=0.1, tau_max=5.0):
    # Deposit is a *constant* scaled to the τ dynamic range — UCTP fitness
    # is HCV-dominated (≥10^6 when infeasible), so the REAL pher_minmax's
    # `1/(1+f)` deposit would collapse to ~10^-7 and the pheromone would
    # never accumulate. With deposit = (τ_max-τ_min)·rho the best slots
    # converge to ~τ_max while uninformative slots decay to τ_min,
    # producing meaningful slot-sampling bias regardless of fitness scale.
    deposit = max((tau_max - tau_min) * rho, 1e-6)
    def mutate_pher_minmax_uctp(ind, ctx):
        rng = ctx["rng"]
        n = int(ctx["dim"])
        n_slots = _uctp_n_slots(ctx)
        tau = ctx.get("pher_uctp_tau")
        if tau is None or tau.shape != (n, n_slots):
            tau = np.full((n, n_slots), tau_max, dtype=float)
            ctx["pher_uctp_tau"] = tau
            ctx["pher_uctp_last_gen"] = -1
        gen = int(ctx.get("gen", 0))
        if ctx.get("pher_uctp_last_gen", -1) != gen:
            tau *= (1.0 - rho)
            bs = ctx.get("best_so_far")
            if bs is not None:
                bs_vec = _as_slot_vec(bs, n)
                for e in range(n):
                    tau[e, bs_vec[e]] += deposit
            np.clip(tau, tau_min, tau_max, out=tau)
            ctx["pher_uctp_last_gen"] = gen
        out = _as_slot_vec(ind, n)
        e = int(rng.integers(0, n))
        row = tau[e]
        s = float(row.sum())
        if s <= 0:
            new_slot = int(rng.integers(0, n_slots))
        else:
            new_slot = int(rng.choice(n_slots, p=row / s))
        out[e] = new_slot
        return out
    return mutate_pher_minmax_uctp


# ------------------------------------------------------------
# F98 — UCTP immune memory pool (Hamming novelty)
# ------------------------------------------------------------
def make_immune_memory_pool_uctp(k_inject=2, dist_thresh_frac=0.05):
    def diversity_immune_uctp(population, ctx):
        n = int(ctx["dim"])
        memory = ctx.setdefault("immune_memory_uctp_pool", [])
        thr = float(dist_thresh_frac) * n
        cur_best = _best_in_pop(population)
        cur_vec = _as_slot_vec(cur_best, n)
        if not memory:
            mc = creator.Individual(list(cur_best))
            mc.fitness.values = cur_best.fitness.values
            memory.append(mc)
        else:
            min_d = min(_hamming_int(cur_vec, _as_slot_vec(m, n)) for m in memory)
            if min_d > thr and len(memory) < 20:
                mc = creator.Individual(list(cur_best))
                mc.fitness.values = cur_best.fitness.values
                memory.append(mc)
        sorted_pop = _sorted_pop_uctp(population)
        n_inject = min(k_inject, len(memory), len(population) // 2)
        if n_inject <= 0:
            return population
        keep = sorted_pop[: len(population) - n_inject]
        injected = []
        for m in memory[-n_inject:]:
            clone = creator.Individual(list(m))
            clone.fitness.values = m.fitness.values
            injected.append(clone)
        return keep + injected
    return diversity_immune_uctp


# ------------------------------------------------------------
# F99 — UCTP network suppression (Hamming similarity)
# ------------------------------------------------------------
def make_network_suppression_uctp(threshold_frac=0.02):
    def diversity_suppress_uctp(population, ctx):
        rng = ctx["rng"]
        n = int(ctx["dim"])
        n_slots = _uctp_n_slots(ctx)
        thr = max(1.0, float(threshold_frac) * n)
        pop = list(population)
        vecs = [_as_slot_vec(p, n) for p in pop]
        keep_mask = [True] * len(pop)
        for i in range(len(pop)):
            if not keep_mask[i]:
                continue
            for j in range(i + 1, len(pop)):
                if not keep_mask[j]:
                    continue
                if _hamming_int(vecs[i], vecs[j]) < thr:
                    worse = i if pop[i].fitness.values[0] > pop[j].fitness.values[0] else j
                    keep_mask[worse] = False
        out = [pop[i] for i in range(len(pop)) if keep_mask[i]]
        for _ in range(len(pop) - len(out)):
            out.append(creator.Individual(_random_int_vec(n, n_slots, rng)))
        return out
    return diversity_suppress_uctp


# ------------------------------------------------------------
# F100 — UCTP cuckoo discovery
# ------------------------------------------------------------
def make_cuckoo_discovery_uctp(pa=0.25):
    def diversity_cuckoo_uctp(population, ctx):
        rng = ctx["rng"]
        n = int(ctx["dim"])
        n_slots = _uctp_n_slots(ctx)
        for i in range(len(population)):
            if rng.random() < pa:
                new_ind = creator.Individual(_random_int_vec(n, n_slots, rng))
                new_ind.age = 0
                population[i] = new_ind
        return population
    return diversity_cuckoo_uctp


# ------------------------------------------------------------
# F101 — UCTP ABC scout
# ------------------------------------------------------------
def make_abc_scout_uctp(limit=None):
    def replace_abc_uctp(parents, offspring, ctx):
        rng = ctx["rng"]
        n = int(ctx["dim"])
        n_slots = _uctp_n_slots(ctx)
        pop_size = ctx["pop_size"]
        # Default limit scaled to n (events), not n*pop_size — SSGA produces
        # one child per gen so pop_size*n would essentially never trigger.
        lim = limit if limit is not None else max(1, n)
        survivors = generational_elitism(parents, offspring, int(ctx["elitism_k"]))

        parents_by_f = sorted(parents, key=lambda x: x.fitness.values[0])
        offspring_by_f = sorted(offspring, key=lambda x: x.fitness.values[0])
        m = min(len(parents_by_f), len(offspring_by_f))
        improved_ids, unimproved_ids = set(), set()
        for i in range(m):
            p = parents_by_f[i]; o = offspring_by_f[i]
            if o.fitness.values[0] < p.fitness.values[0]:
                improved_ids.add(id(p))
            else:
                unimproved_ids.add(id(p))
        for ind in survivors:
            t = getattr(ind, "trial", 0)
            if id(ind) in improved_ids:
                ind.trial = 0
            elif id(ind) in unimproved_ids:
                ind.trial = t + 1
            else:
                ind.trial = 0

        for i, ind in enumerate(survivors):
            if getattr(ind, "trial", 0) > lim:
                scout = creator.Individual(_random_int_vec(n, n_slots, rng))
                scout.age = 0
                scout.trial = 0
                survivors[i] = scout
        return survivors
    return replace_abc_uctp


# ------------------------------------------------------------
# F102 — UCTP BFO chemotaxis (tumble + event-local swim)
# ------------------------------------------------------------
def make_bfo_chemotaxis_uctp(N_s=4):
    def mutate_bfo_uctp(ind, ctx):
        rng = ctx["rng"]
        n = int(ctx["dim"])
        n_slots = _uctp_n_slots(ctx)
        out = _as_slot_vec(ind, n)
        if n < 2:
            return out
        e = int(rng.integers(0, n))
        _n1_move(out, e, int(rng.integers(0, n_slots)))
        # Event-local swim: a few more N1 moves on nearby event indices.
        step = max(2, n // 8)
        for _ in range(N_s):
            e2 = (e + int(rng.integers(1, step))) % n
            _n1_move(out, e2, int(rng.integers(0, n_slots)))
        return out
    return mutate_bfo_uctp


# ------------------------------------------------------------
# F103 — UCTP Krill herd (food + herd + drift)
# ------------------------------------------------------------
def make_krill_herd_uctp(Vf=0.6, Nmax=0.35, Dmax=0.2):
    def mutate_krill_uctp(ind, ctx):
        rng = ctx["rng"]
        n = int(ctx["dim"])
        n_slots = _uctp_n_slots(ctx)
        pop = ctx.get("_population", []) or []
        bs = ctx.get("best_so_far")
        cur = _as_slot_vec(ind, n)

        def _adopt_from(donor_vec):
            diff = _disagreeing_positions(cur, donor_vec)
            if not diff:
                return
            i = int(rng.choice(diff))
            cur[i] = int(donor_vec[i])

        if bs is not None and rng.random() < Vf:
            _adopt_from(_as_slot_vec(bs, n))
        if pop and rng.random() < Nmax:
            others = [p for p in pop if id(p) != id(ind)]
            if others:
                my_f = _ind_fitness(ind)
                if math.isfinite(my_f):
                    neigh = min(others, key=lambda p: abs(_ind_fitness(p) - my_f))
                else:
                    neigh = others[int(rng.integers(0, len(others)))]
                _adopt_from(_as_slot_vec(neigh, n))
        if rng.random() < Dmax:
            e = int(rng.integers(0, n))
            cur[e] = int(rng.integers(0, n_slots))
        return cur
    return mutate_krill_uctp


# ------------------------------------------------------------
# F104 — UCTP SOS (mutualism / commensalism / parasitism)
# ------------------------------------------------------------
def make_sos_uctp():
    def mutate_sos_uctp(ind, ctx):
        rng = ctx["rng"]
        n = int(ctx["dim"])
        pop = ctx.get("_population", []) or []
        cur = _as_slot_vec(ind, n)
        if n < 3:
            return cur

        def _parasitism():
            i, j, k = rng.choice(n, size=3, replace=False).tolist()
            _n3_cycle(cur, i, j, k, forward=rng.random() < 0.5)
            return cur

        if len(pop) < 2:
            return _parasitism()

        phase = int(rng.integers(0, 3))
        if phase == 0:
            partner = _as_slot_vec(pop[int(rng.integers(0, len(pop)))], n)
            a, b = sorted(rng.integers(0, n, size=2).tolist())
            for i in range(a, b + 1):
                cur[i] = int(partner[i])
            return cur
        if phase == 1:
            partner = _as_slot_vec(pop[int(rng.integers(0, len(pop)))], n)
            diff = _disagreeing_positions(cur, partner)
            if not diff:
                return cur
            i = int(rng.choice(diff))
            cur[i] = int(partner[i])
            return cur
        return _parasitism()
    return mutate_sos_uctp


# ------------------------------------------------------------
# F105 — UCTP BBO migration crossover
# ------------------------------------------------------------
def crossover_bbo_uctp(p1, p2, ctx):
    rng = ctx["rng"]
    n = int(ctx["dim"])
    a = _as_slot_vec(p1, n)
    b = _as_slot_vec(p2, n)
    f1 = _ind_fitness(p1, 0.0)
    f2 = _ind_fitness(p2, 0.0)
    donor, recipient = (a, b) if f1 <= f2 else (b, a)

    def _migrate(don, rec):
        i, j = sorted(rng.integers(0, n, size=2).tolist())
        child = list(rec)
        for k in range(i, j + 1):
            child[k] = int(don[k])
        return child

    return _migrate(donor, recipient), _migrate(recipient, donor)


# ------------------------------------------------------------
# F106 — UCTP GSA (mass-weighted slot voting)
# ------------------------------------------------------------
def make_gsa_full_uctp(G0=100.0, alpha=20.0, kbest_final_frac=0.02):
    def mutate_gsa_uctp(ind, ctx):
        rng = ctx["rng"]
        n = int(ctx["dim"])
        n_slots = _uctp_n_slots(ctx)
        pop = ctx.get("_population", []) or []
        if len(pop) < 4:
            return mutate_ssga_n1n2n3(ind, ctx)
        t = float(ctx.get("t_ratio", 0.0))
        k = max(2, int(len(pop) * (1.0 - t * (1.0 - kbest_final_frac))))
        sorted_pop = _sorted_pop_uctp(pop)[:k]
        fits = np.array([p.fitness.values[0] for p in sorted_pop], dtype=float)
        span = max(float(fits.max() - fits.min()), 1e-12)
        masses = (fits.max() - fits) / span
        if masses.sum() <= 0:
            return mutate_ssga_n1n2n3(ind, ctx)
        vote = np.zeros((n, n_slots), dtype=float)
        for m_i, p in zip(masses, sorted_pop):
            vec = _as_slot_vec(p, n)
            for e in range(n):
                vote[e, vec[e]] += float(m_i)
        G_t = G0 * float(np.exp(-alpha * t / max(1.0, ctx.get("max_gen_estimate", 1))))
        n_moves = max(1, min(int(G_t), max(1, n // 4)))
        out = _as_slot_vec(ind, n)
        for _ in range(n_moves):
            e = int(rng.integers(0, n))
            row = vote[e] + 1e-9
            probs = row / row.sum()
            out[e] = int(rng.choice(n_slots, p=probs))
        return out
    return mutate_gsa_uctp


def _events_in_hcv(problem, x, rooms):
    """Return event indices contributing to current HCV: unplaced events
    (room == -1) plus events whose slot has any student-conflict involving
    one of their students."""
    n = problem.n_events
    n_slots = _UCTP_N_TIMESLOTS
    bad = [e for e in range(n) if int(rooms[e]) == -1]
    ts = np.asarray(x, dtype=np.int64)
    onehot = np.zeros((n, n_slots), dtype=np.int8)
    onehot[np.arange(n), ts] = 1
    attend_slot = problem._student_event_i @ onehot  # (n_students, n_slots)
    conflict_mask = attend_slot > 1
    if not bool(conflict_mask.any()):
        return bad
    se = problem.student_event
    in_bad = set(bad)
    for e in range(n):
        if e in in_bad:
            continue
        t = int(ts[e])
        if bool(np.any(se[:, e] & conflict_mask[:, t])):
            bad.append(e)
    return bad


def _ls1_apply(problem, x, rooms, hcv, scv, s_max, rng,
               max_trials=None, per_event_n1=8, per_event_n2=8):
    """In-place LS1 (Yang & Jat 2011 Algorithm 2) on a single individual.

    Two phases:
      1. Reduce HCV: only the subset of events currently contributing to
         HCV are visited. For each, try `per_event_n1` random N1 moves
         (move to another slot) then `per_event_n2` N2 swaps, accepting
         the first move that strictly reduces total HCV.
      2. If feasible (hcv==0), reduce SCV across all events with the same
         per-event trial caps, accepting moves that reduce SCV without
         reintroducing HCV.

    `s_max` caps accepted moves per phase. `max_trials` caps the total
    number of trial-move scorings across both phases (defensive bound for
    huge instances; defaults to 50 * s_max).

    N3 (3-cycle) is not enumerated here — small benefit, big cost. The
    mutation operator (`mutate_ssga_n1n2n3`) still samples N3 so the
    family stays reachable through the outer GA loop.
    """
    n = problem.n_events
    n_slots = _UCTP_N_TIMESLOTS
    if max_trials is None:
        max_trials = 50 * s_max

    state = {"rooms": rooms, "hcv": hcv, "scv": scv, "trials": 0}

    def _try_n1(e, target_predicate):
        if state["trials"] >= max_trials:
            return False
        old_slot = int(x[e])
        k = min(per_event_n1, n_slots)
        choices = rng.choice(n_slots, size=k, replace=False).tolist()
        for new_slot in choices:
            ns = int(new_slot)
            if ns == old_slot:
                continue
            state["trials"] += 1
            x[e] = ns
            new_rooms = problem.assign_rooms(
                x, slots=[old_slot, ns], rooms=state["rooms"],
            )
            _f, nh, nss = problem.score(x, new_rooms)
            if target_predicate(nh, nss):
                state["rooms"], state["hcv"], state["scv"] = new_rooms, nh, nss
                return True
            x[e] = old_slot
            if state["trials"] >= max_trials:
                return False
        return False

    def _try_n2(e, target_predicate):
        if state["trials"] >= max_trials or n < 2:
            return False
        k = min(per_event_n2, n)
        partners = rng.choice(n, size=k, replace=False).tolist()
        for f in partners:
            f = int(f)
            if f == e:
                continue
            slot_e, slot_f = int(x[e]), int(x[f])
            if slot_e == slot_f:
                continue
            state["trials"] += 1
            x[e], x[f] = slot_f, slot_e
            new_rooms = problem.assign_rooms(
                x, slots=[slot_e, slot_f], rooms=state["rooms"],
            )
            _f, nh, nss = problem.score(x, new_rooms)
            if target_predicate(nh, nss):
                state["rooms"], state["hcv"], state["scv"] = new_rooms, nh, nss
                return True
            x[e], x[f] = slot_e, slot_f
            if state["trials"] >= max_trials:
                return False
        return False

    # ---- Phase 1: HCV repair ----
    steps = 0
    hcv_events = _events_in_hcv(problem, x, state["rooms"])
    rng.shuffle(hcv_events)
    for e in hcv_events:
        if state["hcv"] == 0 or steps >= s_max or state["trials"] >= max_trials:
            break
        cur = state["hcv"]
        pred = lambda nh, ns, _h=cur: nh < _h
        if _try_n1(e, pred) or _try_n2(e, pred):
            steps += 1

    # ---- Phase 2: SCV reduction (only if feasible) ----
    if state["hcv"] == 0 and state["trials"] < max_trials:
        steps = 0
        order = list(range(n))
        rng.shuffle(order)
        for e in order:
            if steps >= s_max or state["trials"] >= max_trials:
                break
            cur = state["scv"]
            pred = lambda nh, ns, _s=cur: (nh == 0 and ns < _s)
            if _try_n1(e, pred) or _try_n2(e, pred):
                steps += 1

    fitness = float(problem._hcv_weight * state["hcv"] + state["scv"])
    return x, state["rooms"], state["hcv"], state["scv"], fitness


def make_refine_ls1(s_max=20):
    """Return a population-level refine_fn that applies LS1 to every
    individual. Used by the generational SGA backbone for variants that
    want LS1 as an end-of-generation refinement; `run_ssga` invokes
    `_ls1_apply` directly for the SSGA-style per-child schedule.
    """
    def refine_ls1(population, ctx):
        problem = ctx["problem"]
        rng = ctx["rng"]
        out = []
        for ind in population:
            x = [int(v) for v in ind]
            _f, hcv, scv, rooms = problem.evaluate_full(x)
            x, rooms, hcv, scv, fitness = _ls1_apply(
                problem, x, rooms, hcv, scv, s_max, rng,
            )
            new_ind = creator.Individual(x)
            new_ind.fitness.values = (float(fitness),)
            new_ind.age = getattr(ind, "age", 0)
            out.append(new_ind)
        return out
    refine_ls1.__name__ = f"refine_ls1_smax{s_max}"
    return refine_ls1
