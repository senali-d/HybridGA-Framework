"""
CLI entry point: build variants, run the CEC 2022 protocol, persist
per-batch results, statistical summaries, and plots.

USAGE:
    python scripts/run_experiments.py             # demo (small budget)
    python scripts/run_experiments.py --full      # full CEC 2022 protocol
    python scripts/run_experiments.py --no-plots  # skip plot generation
    python scripts/run_experiments.py --tag pilot1
    python scripts/run_experiments.py --variants pher,levy   # subset only
    python scripts/run_experiments.py --pop-size 40 --max-gen 200 --n-runs 10
    python scripts/run_experiments.py --dim 20 --max-fes 50000 --workers 8
"""

import argparse
import json
import multiprocessing as mp
import sys
from pathlib import Path

# Use forkserver on macOS/Linux: fresh-process safety of `spawn` without
# re-importing every dependency per worker, and lets workers share the
# read-only distance/cost matrices loaded once in the helper.
try:
    mp.set_start_method("forkserver", force=True)
except (RuntimeError, ValueError):
    pass

# Make src/ importable when running directly
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ga.core import FeatureConfig, default_select
from ga.build import baseline_from_config, variant_from_baseline, variant_diff
from ga.variants import VARIANTS, variants_for_encoding


CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"


def load_suite_config(suite):
    """Load the per-suite JSON config (configs/<suite>.json)."""
    path = CONFIG_DIR / f"{suite}.json"
    if not path.exists():
        return None
    with path.open() as f:
        return json.load(f)


def apply_cli_overrides(suite_cfg, args):
    """Mutate the loaded suite config in-place from CLI flags before
    constructing baselines. Keeps the runner's `--pop-size`/`--max-fes`/
    `--n-runs` flags as universal escape hatches that propagate to every
    variant (since variants inherit from baseline)."""
    if args.pop_size is not None:
        suite_cfg["pop_size"] = int(args.pop_size)
    if args.n_runs is not None:
        suite_cfg["n_runs"] = int(args.n_runs)
    if args.max_fes is not None:
        suite_cfg["max_fes"] = int(args.max_fes)
    if args.max_gen is not None:
        # Convert max_gen → max_fes using current pop_size.
        suite_cfg["max_fes"] = int(args.max_gen) * int(suite_cfg["pop_size"])
    return suite_cfg


def resolve_max_fes(suite_cfg, dim):
    mf = suite_cfg.get("max_fes")
    if mf is None:
        return None
    if isinstance(mf, dict):
        table = {int(k): int(v) for k, v in mf.items()}
        if dim not in table:
            raise ValueError(
                f"max_fes table has no entry for dim={dim}; "
                f"available dims: {sorted(table)}"
            )
        return table[dim]
    return int(mf)
from ga import features as feat
from benchmarks.protocols import (
    run_protocol, run_cec2020_realworld_protocol,
    run_tsp_protocol, run_uctp_protocol,
)
from analysis.stats import summarize, write_summary
from analysis import plots as plot_mod
from persistence import ExperimentRecorder


def build_variants(baseline, encoding):
    """Baseline + every variant in the registry that applies to this encoding."""
    variants = [baseline]
    for spec in variants_for_encoding(encoding):
        variants.append(variant_from_baseline(baseline, spec))
    return variants


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--full", action="store_true",
                        help="Run full CEC 2022 protocol (slow). Default: demo.")
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--out-root", type=str, default="results",
                        help="Root directory for per-batch experiment archives.")
    parser.add_argument("--tag", type=str, default=None,
                        help="Override timestamped folder name.")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-stats", action="store_true")
    parser.add_argument("--no-history", action="store_true",
                        help="Skip per-generation history recording (saves disk).")
    parser.add_argument("--baseline", type=str, default="SGA_baseline")
    parser.add_argument("--variants", type=str, default=None,
                        help="Comma-separated substrings; only variants whose "
                             "name contains any of these are run. Baseline is "
                             "always included. Example: --variants pher,levy")
    parser.add_argument("--exact", action="store_true",
                        help="Treat --variants entries as exact names "
                             "(matching either v.name or 'SGA+'+entry) "
                             "instead of substrings. Useful when one needle "
                             "would otherwise match a longer variant name "
                             "(e.g. DE_mut vs SHADE_mut).")
    parser.add_argument("--workers", type=int, default=1,
                        help="Process-pool workers over (variant, function, "
                             "seed) tuples. Default 1 (sequential). Uses fork "
                             "start method, so closures inside variants are "
                             "fine. Results are seed-deterministic.")
    parser.add_argument("--pop-size", type=int, default=None,
                        help="Population size. Default: 30 (demo) / 50 (full).")
    parser.add_argument("--max-fes", type=int, default=None,
                        help="FE budget per run. Default: 5000 (demo); for "
                             "--full, 200k at D<=10 or 1M at D>10.")
    parser.add_argument("--max-gen", type=int, default=None,
                        help="Cap by generations instead of FEs. Sets "
                             "max_fes = max_gen * pop_size. Overrides --max-fes.")
    parser.add_argument("--suite", type=str, default="cec2022",
                        choices=["cec2022", "cec2020_realworld",
                                 "tsp", "uctp"],
                        help="Benchmark suite. 'cec2022' (12 CEC 2022 "
                             "functions, default), 'cec2020_realworld' "
                             "(5 mechanical-engineering problems from "
                             "CEC 2020-RW; fixed dim, --dim ignored), "
                             "'tsp' (6 TSPLIB instances), or "
                             "'uctp' (30 timetabling instances).")
    parser.add_argument("--functions", type=str, default=None,
                        help="Comma-separated function names to run "
                             "(e.g. F1,F3,F7 for cec2022; WeldedBeam for "
                             "cec2020_realworld; eil51 for tsp; "
                             "socha_small1 for uctp). "
                             "Default: all in the suite.")
    parser.add_argument("--n-runs", type=int, default=None,
                        help="Independent seeds per (variant, function). "
                             "Default: 5 (demo) / 30 (full).")
    args = parser.parse_args()

    # Always load the suite config; it is the single source of truth for
    # the SGA spec. Demo mode (--no --full) overlays small budgets after.
    suite_cfg = load_suite_config(args.suite)
    if suite_cfg is None:
        raise FileNotFoundError(
            f"configs/{args.suite}.json not found. Each suite must have "
            f"a config file under configs/."
        )

    if not args.full:
        # Demo overlays — keep existing operator spec from JSON, just shrink
        # pop/FE/runs so the smoke run is fast.
        suite_cfg["pop_size"] = 30
        suite_cfg["max_fes"] = 5_000
        suite_cfg["n_runs"] = 5
        mode = "DEMO"
    else:
        mode = "FULL"

    suite_cfg = apply_cli_overrides(suite_cfg, args)

    pop_size = int(suite_cfg["pop_size"])
    n_runs = int(suite_cfg["n_runs"])
    max_fes = resolve_max_fes(suite_cfg, args.dim)
    if max_fes is None:
        raise NotImplementedError(
            f"suite {args.suite!r}: configs/{args.suite}.json has no "
            f"`max_fes` field, and the runner only supports FE-budget "
            f"stopping today. Either add `max_fes` to the config or "
            f"pass --max-fes / --max-gen on the CLI. "
            f"(See the config's `_stopping_note` for the intended "
            f"stopping criterion.)"
        )

    print(f"Running {mode} protocol: D={args.dim}, pop={pop_size}, "
          f"MaxFES={max_fes} (~{max_fes // pop_size} gens), runs={n_runs}")
    if mode == "DEMO" and not any([args.pop_size, args.max_fes,
                                    args.max_gen, args.n_runs]):
        print("(Use --full for the real CEC 2022 budget, or override with "
              "--pop-size / --max-fes / --max-gen / --n-runs.)")

    # Build the suite baseline from the resolved JSON, then derive every
    # registry variant on top of it. Each variant inherits all unmodified
    # operator/scalar params from the baseline.
    encoding = suite_cfg.get("encoding", "real")
    baseline = baseline_from_config(suite_cfg, dim=args.dim)
    baseline.name = args.baseline  # match --baseline string for downstream stats
    variants = build_variants(baseline, encoding)

    if args.variants:
        needles = [s.strip() for s in args.variants.split(",") if s.strip()]
        if args.exact:
            wanted = set(needles) | {f"SGA+{n}" for n in needles}
            variants = [
                v for v in variants
                if v.name == args.baseline or v.name in wanted
            ]
        else:
            variants = [
                v for v in variants
                if v.name == args.baseline or any(n in v.name for n in needles)
            ]
        if not any(v.name == args.baseline for v in variants):
            print(f"WARNING: baseline {args.baseline!r} not in filtered set; "
                  "Wilcoxon vs baseline will be skipped.")
    print(f"Variants to test ({len(variants)}): {[v.name for v in variants]}\n")

    recorder = ExperimentRecorder(
        out_root=args.out_root,
        args={**vars(args), "n_runs": n_runs, "max_fes": max_fes,
              "pop_size": pop_size},
        variants=[v.name for v in variants],
        tag=args.tag,
    )
    # Audit trail: resolved suite config + per-variant diff vs baseline.
    (recorder.exp_dir / "suite_config.json").write_text(
        json.dumps(suite_cfg, indent=2, default=str)
    )
    name_to_spec = {s.name: s for s in VARIANTS}
    diffs = []
    for v in variants:
        if v.name == args.baseline:
            diffs.append({"name": v.name, "slots": [], "overrides": {},
                          "encodings": [encoding]})
        elif v.name in name_to_spec:
            diffs.append(variant_diff(baseline, v, name_to_spec[v.name]))
    (recorder.exp_dir / "variants_diff.json").write_text(
        json.dumps(diffs, indent=2)
    )

    funcs_filter = (
        [s.strip() for s in args.functions.split(",") if s.strip()]
        if args.functions else None
    )

    protocols = {
        "cec2022": run_protocol,
        "cec2020_realworld": run_cec2020_realworld_protocol,
        "tsp": run_tsp_protocol,
        "uctp": run_uctp_protocol,
    }
    if args.suite not in protocols:
        raise NotImplementedError(
            f"suite {args.suite!r}: config exists at "
            f"configs/{args.suite}.json but the problem evaluators and "
            f"`run_{args.suite}_protocol` are not yet wired into "
            f"src/benchmarks/. (See the config's `_note` field.)"
        )
    protocol = protocols[args.suite]
    results = protocol(
        variants, dim=args.dim, n_runs=n_runs,
        max_fes=max_fes, pop_size=pop_size,
        recorder=recorder, record_history=not args.no_history,
        workers=args.workers, functions=funcs_filter,
    )

    summarize(results, baseline_name=args.baseline)
    recorder.flush()

    if not args.no_stats:
        import pandas as _pd
        _runs_df = _pd.read_parquet(recorder.exp_dir / "runs.parquet")
        _hist_path = recorder.exp_dir / "history.parquet"
        _hist_df = _pd.read_parquet(_hist_path) if _hist_path.exists() else None
        write_summary(results, recorder.exp_dir / "summary",
                      baseline_name=args.baseline,
                      history_df=_hist_df, runs_df=_runs_df)

    if not args.no_plots:
        try:
            plot_mod.generate_all(recorder.exp_dir, baseline_name=args.baseline)
        except Exception as e:
            print(f"[plots] generation failed: {e}")

    print(f"\nExperiment archive: {recorder.exp_dir}")


if __name__ == "__main__":
    main()
