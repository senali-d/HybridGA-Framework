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
def make_pheromone_edge_crossover(rho=0.1, K=20, alpha=1.0):
    """Per-dim discrete pick: child inherits p1[d] or p2[d] proportional to
    τ at each parent's bin."""
    def crossover_pher_edge(p1, p2, ctx):
        rng = ctx["rng"]
        lb, ub = ctx["lb"], ctx["ub"]
        pher = _pheromone_init(ctx, K=K)
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
def make_pheromone_weighted_crossover(rho=0.1, K=20, alpha=1.0):
    """Per-dim blend child = w·p1 + (1−w)·p2 with w τ-proportional."""
    def crossover_pher_weighted(p1, p2, ctx):
        lb, ub = ctx["lb"], ctx["ub"]
        pher = _pheromone_init(ctx, K=K)
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
