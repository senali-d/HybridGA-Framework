# SGA + Nature-Inspired Features on CEC 2022

A clean experimental pipeline to study how features extracted from recent
nature-inspired algorithms affect the behavior of the Simple Genetic
Algorithm. Built on **DEAP** (GA primitives) + **Opfunu** (CEC 2022 functions)
+ **SciPy** (statistical tests).

## Project layout

```
HybridGA-Framework/
├── src/
│   ├── ga/              # SGA backbone + nature-inspired feature plug-ins
│   │   ├── core.py
│   │   └── features.py
│   ├── benchmarks/      # CEC 2022 wrappers + multi-variant protocol runner
│   │   └── cec2022.py
│   └── analysis/        # Statistical summaries (Wilcoxon, Friedman, plotting)
│       └── stats.py
├── scripts/
│   └── run_experiments.py   # CLI entry point
├── results/             # Raw JSON outputs (gitignored)
├── pyproject.toml
└── README.md
```

## Quick start

```bash
# 1. Create and activate a virtual environment (first time only)
python -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows PowerShell

# 2. Install the project + all dependencies
pip install -e .

# 3. Every new shell, just reactivate:
#    source venv/bin/activate

# Demo: small budget, ~1 minute, sanity-check the pipeline
python scripts/run_experiments.py

# Full CEC 2022 protocol: 12 functions, 30 runs, MaxFES per spec
python scripts/run_experiments.py --full --dim 10
python scripts/run_experiments.py --full --dim 20
```

## Smoke tests

Cheap (~2 s) parametrized tests that run **every variant** in
`build_variants()` for one tiny trial (D=10, pop=12, MaxFES=200) on F1.
They assert no exceptions, in-bounds solutions, non-negative error, FE
budget respected, plus seed reproducibility, islands+diversity, and
*coverage*: every public feature in `ga.features` must be wired into at
least one variant.

```bash
pip install pytest                     # one-time

# Run all smoke tests
pytest tests/ -q

# Run a single feature's smoke test
pytest tests/test_smoke.py -k "Levy_mut" -q

# Verbose, show every variant pass/fail
pytest tests/ -v
```

Run these before kicking off any long protocol run — a 2-second test catches
operator-shape regressions that would otherwise blow up partway through a
multi-day full run.

## How feature injection works

Every variant is a `FeatureConfig` object that overrides one or more of the
SGA's plug-points. The default operators stay in place for everything not
overridden, so you can isolate the effect of a single feature.

```python
from ga.core import FeatureConfig, run_sga
from ga import features as feat
from opfunu.cec_based.cec2022 import F12022

# Define a variant that injects Lévy-flight mutation
variant = FeatureConfig(
    name="SGA+Levy_mut",
    mutate_fn=feat.mutate_levy,
)

problem = F12022(ndim=10)
result = run_sga(problem, variant, pop_size=50, max_fes=200_000, seed=0)
print(f"Best fitness: {result['best_fitness']}")
print(f"Error vs F*: {result['error']}")
```

## What's implemented (Tier 1 + Tier 2 from the framework)

| Feature ID | Name | Plug-point |
|---|---|---|
| F02 | Chaotic-map init (logistic) | init_fn |
| F03 | Opposition-based init | init_fn |
| F06 | GWO leader-hierarchy selection | select_fn |
| F13 | Lévy-flight mutation | mutate_fn |
| F17 | DE/rand/1 differential mutation | mutate_fn |
| F36 | Greedy one-to-one replacement | replace_fn |
| F39 | Stagnation-triggered restart | diversity_fn |
| F40 | Linearly decreasing mutation rate | mutate_fn |

Add Tier 3 features by writing more functions in `src/ga/features.py` matching
the same signatures, then registering them in `build_variants()` inside
`scripts/run_experiments.py`.

## CEC 2022 protocol compliance

- **Suite:** 12 functions (F1–F12), search range [-100, 100]^D
- **Dimensions:** 10 and 20
- **MaxFES:** 200,000 (D=10) / 1,000,000 (D=20)
- **Independent runs:** 30 per (variant, function)
- **Reported metric:** error = f(x_best) − F*; values < 1e-8 set to 0

## Statistical analysis

`summarize()` produces:

1. **Mean error table** across all (variant, function) pairs
2. **Wilcoxon signed-rank test** of each variant vs baseline (per function),
   reported as wins / ties / losses at α = 0.05
3. **Friedman test** across the full suite with mean ranks per variant
4. **Holm post-hoc** pairwise Wilcoxon (FWER-controlled) and **Nemenyi**
   post-hoc on the Friedman matrix
5. **Cliff's delta** effect sizes vs baseline, with magnitude labels

## Compute notes

- **Demo run** (5 runs, MaxFES=5000, ~75 variants, 12 functions): ~3 minutes
- **Full run** at D=10 (30 runs, MaxFES=200k, ~75 variants, 12 functions):
  roughly 1–2 hours per variant on a single core; with the full variant set
  this is multi-day on one core. Use `--workers N` to parallelize across
  `(variant, function, seed)` tuples on N processes (fork-based, so feature
  closures are fine; results are seed-deterministic).
- **Full run at D=20** is roughly 5× heavier per variant.

```bash
python scripts/run_experiments.py --full --dim 10 --workers 8
python scripts/run_experiments.py --variants pher,levy --workers 4
```

## Behavioral metrics (in `history` dict)

When `record_history=True`, each generation logs:
- `gen`, `fes`: generation index and FE count
- `best_fitness`, `mean_fitness`: convergence indicators
- `diversity`: mean per-dimension std of population (population spread)

Use these to plot exploration/exploitation behavior per variant.
For the official Hussain et al. (2019) exploration/exploitation ratio,
add per-dimension diversity tracking and compute Div(t)/Div_max as the
exploration percentage.

## Caveats and known simplifications

1. **Opposition-based init** is simplified — strict OBL evaluates 2n points
   and keeps the best n, costing extra FEs at start. Implement that if you
   want to follow Tizhoosh (2005) exactly.
2. **DE mutation** uses DE/rand/1; for stronger results try DE/best/2 or
   self-adaptive SHADE-style F and CR.
3. **Population size and elitism** are held fixed across variants for
   fair comparison. Don't tune per-variant — that's how papers get accused
   of cherry-picking.

## References

See the framework PDF for the full reference list (37 entries covering the
20 algorithms, CEC 2022, methodology, and the metaphor critique literature).
