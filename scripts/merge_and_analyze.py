"""
Merge runs.parquet from multiple experiment archives and compute
comparative statistics (Wilcoxon vs baseline, Friedman ranks, Holm,
Nemenyi, Cliff's delta, success rates, mean error tables).

USAGE:
    python scripts/merge_and_analyze.py \
        --archives results/exp_*_baseline_v2 results/exp_*_pher_batch \
        --baseline SGA_baseline \
        --out results/merged_analysis_2026-05-06

Notes:
- Same (variant, function, seed) appearing in multiple archives is
  de-duplicated (last occurrence wins).
- Pass --archives with shell globs OR explicit paths.
- The output folder will contain the same summary files that
  write_summary() produces for a normal experiment run.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from analysis.stats import summarize, write_summary


_CONFIG_KEYS_MUST_MATCH = ("dim", "max_fes", "pop_size", "n_runs", "full")


def validate_archive_configs(archives: list[Path], strict: bool = True) -> None:
    """Refuse to merge archives whose protocol parameters disagree.

    Wilcoxon is paired by seed index; mismatched dim / FE budget / pop size
    / n_runs invalidate the comparison. Mutation parameters (mut_prob,
    indpb) are FeatureConfig defaults, not CLI args, so they aren't in
    config.json - we surface git_sha mismatches as a soft warning since
    a code-level convention change (e.g., mut_prob 0.1 -> 1.0) won't
    show up otherwise.
    """
    configs = {}
    for arc in archives:
        cfg_path = arc / "config.json"
        if not cfg_path.exists():
            print(f"[warn] {arc.name}: no config.json - skipping validation")
            continue
        with open(cfg_path) as f:
            configs[arc.name] = json.load(f)
    if len(configs) < 2:
        return

    ref_name, ref_cfg = next(iter(configs.items()))
    ref_args = ref_cfg.get("args", {})
    mismatches: list[str] = []
    for name, cfg in configs.items():
        if name == ref_name:
            continue
        args = cfg.get("args", {})
        for key in _CONFIG_KEYS_MUST_MATCH:
            if args.get(key) != ref_args.get(key):
                mismatches.append(
                    f"  {name}.{key}={args.get(key)!r} vs "
                    f"{ref_name}.{key}={ref_args.get(key)!r}"
                )
    if mismatches:
        msg = (
            "Archive protocol parameters disagree:\n"
            + "\n".join(mismatches)
            + "\nWilcoxon pairing by seed requires identical protocol. "
            "Pass --no-validate to override."
        )
        if strict:
            raise SystemExit(msg)
        print(f"WARNING: {msg}")

    shas = {name: cfg.get("git_sha") for name, cfg in configs.items()}
    unique_shas = {s for s in shas.values() if s}
    if len(unique_shas) > 1:
        print("WARNING: archives produced from different git commits - "
              "code-level changes (e.g., mutation defaults) may differ:")
        for name, sha in shas.items():
            print(f"  {name}: {sha}")


def load_merged_runs(archives: list[Path]) -> pd.DataFrame:
    frames = []
    for arc in archives:
        p = arc / "runs.parquet"
        if not p.exists():
            print(f"[skip] {p} not found")
            continue
        df = pd.read_parquet(p)
        df["_source"] = arc.name
        frames.append(df)
        print(f"[load] {arc.name}: {len(df)} rows, "
              f"variants={sorted(df['variant'].unique())}")
    if not frames:
        raise SystemExit("No runs.parquet files found in given archives.")
    merged = pd.concat(frames, ignore_index=True)
    # De-dup: same (variant, function, seed) → keep last
    before = len(merged)
    merged = merged.drop_duplicates(
        subset=["variant", "function", "seed"], keep="last"
    )
    if len(merged) < before:
        print(f"[dedup] dropped {before - len(merged)} duplicate rows")
    return merged


def load_merged_history(archives: list[Path]) -> pd.DataFrame | None:
    frames = []
    for arc in archives:
        p = arc / "history.parquet"
        if p.exists():
            frames.append(pd.read_parquet(p))
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=["variant", "function", "seed", "gen"], keep="last"
    )


def runs_df_to_results_dict(runs_df: pd.DataFrame) -> dict:
    results: dict = {}
    for (variant, function), grp in runs_df.groupby(["variant", "function"]):
        grp_sorted = grp.sort_values("seed")
        results.setdefault(variant, {})[function] = grp_sorted["error"].tolist()
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--archives", nargs="+", required=True,
                        help="Experiment archive directories to merge.")
    parser.add_argument("--baseline", default="SGA_baseline",
                        help="Variant name used as Wilcoxon reference.")
    parser.add_argument("--out", required=True,
                        help="Output directory for merged summary.")
    parser.add_argument("--no-validate", action="store_true",
                        help="Downgrade protocol-mismatch errors to warnings.")
    args = parser.parse_args()

    archives = [Path(a) for a in args.archives]
    validate_archive_configs(archives, strict=not args.no_validate)
    runs_df = load_merged_runs(archives)
    hist_df = load_merged_history(archives)

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    runs_df.to_parquet(out / "runs.parquet", index=False)
    if hist_df is not None:
        hist_df.to_parquet(out / "history.parquet", index=False)

    results = runs_df_to_results_dict(runs_df)

    print(f"\n[merged] variants: {list(results.keys())}")
    print(f"[merged] functions: "
          f"{sorted(next(iter(results.values())).keys())}")
    print(f"[merged] baseline for Wilcoxon: {args.baseline}\n")

    if args.baseline not in results:
        print(f"WARNING: baseline {args.baseline!r} not in merged set; "
              "Wilcoxon comparisons will be skipped.")

    summarize(results, baseline_name=args.baseline)
    write_summary(results, out / "summary",
                  baseline_name=args.baseline,
                  history_df=hist_df, runs_df=runs_df)

    print(f"\nMerged analysis written to: {out}")


if __name__ == "__main__":
    main()
