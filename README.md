# HybridGA-Framework

A clean experimental pipeline for studying how features extracted from
nature-inspired metaheuristics affect the behavior of the Simple Genetic
Algorithm (SGA). Built on **DEAP** (GA primitives) + **Opfunu** (CEC 2022) +
**SciPy** (statistical tests).

The framework evaluates **17 named experiments** (1 baseline + 16 feature
variants) across **three problem suites**:

| Suite (`--suite`) | Problem class | Count | Encoding | Notes |
|---|---|---|---|---|
| `cec2022` | CEC 2022 single-objective black-box | 12 (F1–F12) | real | D ∈ {2, 10, 20}; competition protocol |
| `cec2020_realworld` | CEC 2020 Real-World constrained design | 5 | real | Fixed D (3–7); Deb feasibility-rule scalar fitness |
| `tsp` | Travelling Salesman (TSPLIB EUC_2D) | 6 instances | permutation | EAX crossover baseline |

## The 17 experiments

The baseline **`SGA_baseline`** is built from each suite's JSON config (its
default operators differ by encoding). On top of it, **16 variants** are
defined, each overriding exactly one operator slot:

| Variant | Slot | Idea |
|---|---|---|
| `SGA+DE_mut` | mutation | DE/rand/1 differential mutation |
| `SGA+SHADE_mut` | mutation | Success-history adaptive DE (SHADE) |
| `SGA+greedy_replace` | replacement | Greedy one-to-one parent/child compare |
| `SGA+archive_replace` | replacement | External elite archive injection |
| `SGA+metropolis_replace` | replacement | Simulated-annealing acceptance |
| `SGA+restart` | diversity | Stagnation-triggered population restart |
| `SGA+pher_minmax` | mutation / crossover | MMAS pheromone bounds (τ_min/τ_max) |
| `SGA+immune_memory` | diversity | Memory-pool injection (AIS) |
| `SGA+net_suppress` | diversity | Network-suppression diversity (AIS) |
| `SGA+cuckoo_discovery` | diversity | Cuckoo abandon-and-rebuild |
| `SGA+abc_scout` | replacement | ABC scout-bee re-init |
| `SGA+bfo_chemotaxis` | mutation | BFO swim/tumble chemotaxis |
| `SGA+krill_herd_mut` | mutation | Krill food + herd + drift |
| `SGA+sos_mut` | mutation | SOS mutualism/commensalism/parasitism |
| `SGA+bbo_migration_cx` | crossover | BBO habitat migration |
| `SGA+gsa_full_mut` | mutation | GSA mass-weighted gravitational attraction |

Each variant has up to two implementations (`make_*`, `make_*_perm`) so it
can run on every suite. The runner selects the right one automatically by
encoding.

## Project layout

```
HybridGA-Framework/
├── src/
│   ├── ga/
│   │   ├── core.py         # SGA + SSGA loops, default operators, FeatureConfig
│   │   ├── features.py     # All 16 variants' implementations (real / perm)
│   │   ├── variants.py     # Spec registry — wires features into named variants
│   │   └── build.py        # baseline_from_config + variant_from_baseline
│   ├── benchmarks/
│   │   ├── cec2022_functions.py    # 12 CEC 2022 functions (F1–F12)
│   │   ├── cec2020_realworld.py    # 5 CEC 2020-RW engineering problems
│   │   ├── tsp.py                  # 6 TSPLIB EUC_2D instances
│   │   └── protocols.py            # Multi-variant runner + per-suite wrappers
│   ├── analysis/           # stats.py (Wilcoxon/Friedman/Holm/Nemenyi) + plots.py
│   └── persistence/        # recorder.py (JSON + parquet archives)
├── scripts/
│   ├── run_experiments.py          # Primary CLI entry point
│   ├── fetch_tsplib.py             # One-time: download TSPLIB .tsp files
│   ├── analyze_experiment.py       # Post-hoc stats on a single archive
│   └── merge_and_analyze.py        # Merge & analyze multiple archives
├── configs/
│   ├── cec2022.json
│   ├── cec2020_realworld.json
│   └── tsp.json
├── tests/                  # pytest smoke + suite-specific checks
├── results/                # Per-run archives (gitignored)
└── pyproject.toml
```

## Quick start

```bash
# 1. Create and activate a virtual environment (first time only)
python -m venv venv
source venv/bin/activate          # macOS / Linux
# venv\Scripts\activate           # Windows PowerShell

# 2. Install the project + all dependencies
pip install -e .

# 3. Run smoke tests (~30 s) before any long protocol run
pytest tests/ -q
```

### One-time data fetches (TSP)

The TSPLIB `.tsp` files are not bundled. Run the fetcher once after install:

```bash
python scripts/fetch_tsplib.py     # -> data/tsplib/
```

Re-running is safe — existing files are skipped unless you pass `--force`.

## Running experiments

The CLI follows the same pattern for every suite:

```bash
python scripts/run_experiments.py --suite <suite> [flags]
```

### Full sweep: all 16 variants + baseline, every suite

The same 16 variant names are valid on every suite — the runner picks the
encoding-appropriate implementation from `configs/<suite>.json` (`encoding`
field). `SGA_baseline` is always included automatically; do not list it in
`--variants`. Each command writes to its own `results/<tag>/` folder and
does not touch any other run.

```bash
# Define the variant list once, reuse across suites
ALL16="DE_mut,SHADE_mut,greedy_replace,archive_replace,metropolis_replace,\
restart,pher_minmax,immune_memory,net_suppress,cuckoo_discovery,abc_scout,\
bfo_chemotaxis,krill_herd_mut,sos_mut,bbo_migration_cx,gsa_full_mut"

# CEC 2022 — real-coded, 12 functions × 30 runs (D=10, MaxFES=200k from protocol)
python scripts/run_experiments.py --suite cec2022 --full --exact \
    --dim 10 --workers 12 --tag cec2022_D10_all16 --variants "$ALL16"

# CEC 2020 Real-World — 5 engineering problems × 30 runs (fixed dim per problem)
python scripts/run_experiments.py --suite cec2020_realworld --full --exact \
    --workers 8 --tag cec2020rw_all16 --variants "$ALL16"

# TSP — 6 TSPLIB instances × 30 runs × 500k FEs each
python scripts/run_experiments.py --suite tsp --full --exact \
    --workers 12 --max-fes 500000 --tag tsp_all16 --variants "$ALL16"
```

Per-suite quick variations are documented below.

### Full sweep: one function per batch (isolated archives)

Sometimes you want a **separate archive per function/instance** — for
incremental commits, per-function plots, or to resume part-way through.
The loops below reuse `$ALL16` from the previous subsection.

```bash
# CEC 2022 — one tag per function (F1..F12)
for fn in F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12; do
  python scripts/run_experiments.py --suite cec2022 --full --exact \
      --dim 10 --workers 12 \
      --functions "$fn" --tag "cec2022_D10_${fn}" \
      --variants "$ALL16"
done

# CEC 2020 Real-World — one tag per engineering problem
for fn in TensionCompressionSpring PressureVessel WeldedBeam \
          ThreeBarTruss SpeedReducer; do
  python scripts/run_experiments.py --suite cec2020_realworld --full --exact \
      --workers 8 \
      --functions "$fn" --tag "cec2020rw_${fn}" \
      --variants "$ALL16"
done

# TSP — one tag per TSPLIB instance
for fn in eil51 berlin52 st70 kroA100 ch150 kroA200; do
  python scripts/run_experiments.py --suite tsp --full --exact \
      --workers 12 --max-fes 500000 \
      --functions "$fn" --tag "tsp_${fn}_all16" \
      --variants "$ALL16"
done
```

Each iteration writes only to `results/<tag>/`; previous folders are
untouched. If a tag already exists it is wiped before the new run, so
delete or rename a tag only when you want to replace its contents.

### CEC 2022 (real-coded, default)

```bash
# Demo: small budget, ~3 min on one core
python scripts/run_experiments.py

# Full competition protocol: 12 functions × 30 runs × MaxFES per CEC 2022 spec
python scripts/run_experiments.py --full --dim 10                # MaxFES 200k
python scripts/run_experiments.py --full --dim 20 --workers 8    # MaxFES 1M

# Subset of functions / variants
python scripts/run_experiments.py --functions F1,F3,F7
python scripts/run_experiments.py --variants DE_mut,SHADE_mut --exact
```

### CEC 2020 Real-World (engineering)

Each problem has fixed dimensionality (3–7), so `--dim` is ignored.
Constraints are encoded into scalar fitness via Deb's feasibility rules
(every operator works unchanged).

```bash
python scripts/run_experiments.py --suite cec2020_realworld
python scripts/run_experiments.py --suite cec2020_realworld --full
python scripts/run_experiments.py --suite cec2020_realworld --functions WeldedBeam
```

| Problem | NDIM | Notes |
|---|---|---|
| TensionCompressionSpring | 3 | RC-15 |
| PressureVessel           | 4 | RC-17; discrete thicknesses snap to 0.0625″ |
| WeldedBeam               | 4 | RC-18 |
| ThreeBarTruss            | 2 | RC-19 |
| SpeedReducer             | 7 | RC-20; integer teeth count |

A solution is feasible iff `best_fitness < F_OFFSET` (per-problem constant
exposed on the class).

### TSP (TSPLIB EUC_2D)

Permutation encoding. Baseline uses **EAX crossover** + swap mutation.
Each instance has a fixed city count baked into its `.tsp` file, so
`--dim` is ignored.

```bash
# Smoke
python scripts/run_experiments.py --suite tsp

# Full protocol: 6 instances × 30 runs × 50k FEs each
python scripts/run_experiments.py --full --suite tsp --workers 8

# Subset of instances
python scripts/run_experiments.py --suite tsp --functions eil51,kroA100
```

| Instance | Cities | Known optimum |
|---|---|---|
| eil51    |  51 | 426    |
| berlin52 |  52 | 7,542  |
| st70     |  70 | 675    |
| kroA100  | 100 | 21,282 |
| ch150    | 150 | 6,528  |
| kroA200  | 200 | 29,368 |

### Common flags

| Flag | Default (demo / full) | Notes |
|---|---|---|
| `--suite` | `cec2022` | `cec2022`, `cec2020_realworld`, `tsp` |
| `--dim` | 10 | CEC 2022 supports {2, 10, 20}; ignored elsewhere |
| `--pop-size` | 30 / 50 | Population size |
| `--max-fes` | 5,000 / per-suite config | FE budget per run |
| `--max-gen` | — | Cap by generations; sets `max_fes = max_gen × pop_size` |
| `--n-runs` | 5 / 30 | Independent seeds per (variant, function) |
| `--variants` | (all) | Comma-separated substring filter, e.g. `DE_mut,SHADE_mut`. Add `--exact` to match the full name |
| `--functions` | (all in suite) | Comma-separated function/instance names |
| `--workers` | 1 | Process-pool size; results are seed-deterministic |
| `--no-history` / `--no-plots` / `--no-stats` | off | Skip outputs |
| `--tag` | timestamp | Override archive folder name |

Every run produces a per-batch archive under `results/<tag>/` containing
`runs.parquet`, `history.parquet`, `suite_config.json`, `variants_diff.json`,
and a `summary/` directory with the statistical analysis output.

## Feature injection

Every variant is just a `FeatureConfig` with overridden plug-points; the
default operators stay in place for slots you don't touch.

```python
from ga.core import FeatureConfig, run_sga
from ga import features as feat
from benchmarks.cec2022_functions import CEC2022_FUNCS

variant = FeatureConfig(
    name="SGA+DE_mut",
    mutate_fn=feat.make_de_mutation(F=0.5),
    select_wrapper=feat.with_population_aware_select,
)

problem = dict(CEC2022_FUNCS)["F1"](ndim=10)
result = run_sga(problem, variant, pop_size=50, max_fes=200_000, seed=0)
print(f"Best fitness: {result['best_fitness']}, error: {result['error']}")
```

## Statistical analysis

After each run, `summarize()` produces:

1. **Mean error table** across all (variant, function) pairs
2. **Wilcoxon signed-rank** of each variant vs baseline (per function), as
   wins / ties / losses at α = 0.05
3. **Friedman test** across the suite with mean ranks per variant
4. **Holm post-hoc** pairwise Wilcoxon (FWER-controlled) and **Nemenyi**
5. **Cliff's delta** effect sizes vs baseline, with magnitude labels

For comparing multiple separate runs (e.g. baseline first, then a variant
batch), merge their archives post-hoc:

```bash
python scripts/merge_and_analyze.py \
    --archives results/exp_*_baseline results/exp_*_variants \
    --baseline SGA_baseline \
    --out results/merged_analysis
```

Merged archives must share `--dim`, `--max-fes`, `--pop-size`, and the seed
set (Wilcoxon is paired by seed index).

## Smoke tests

```bash
pytest tests/ -q                       # all tests (~30 s)
pytest tests/test_smoke.py -v          # variant-by-variant smoke
pytest tests/test_tsp.py -q            # TSP-specific
```

The smoke suite runs every variant in the registry on a tiny budget (D=10,
pop=12, MaxFES=200) on F1 to catch operator-shape regressions before any
multi-day run.

## Compute notes

- **Demo** (`--n-runs 5`, `--max-fes 5000`, 17 experiments × 12 functions):
  ~3 min on one core.
- **Full CEC 2022 at D=10** (30 runs, MaxFES=200k, 17 experiments × 12
  functions): ~1–2 h per variant on one core. Use `--workers N` to
  parallelize across `(variant, function, seed)` tuples.
- **Full at D=20** is ~5× heavier per variant.

## CEC 2022 protocol compliance

- **Suite:** 12 functions (F1–F12), search range [-100, 100]ᴰ
- **Dimensions:** 10 and 20
- **MaxFES:** 200,000 (D=10) / 1,000,000 (D=20)
- **Independent runs:** 30 per (variant, function)
- **Reported metric:** error = f(x_best) − F*; values < 1e-8 set to 0
