"""
Three-phase feature-ablation orchestrator.

Goal: from the 87 GA features registered in run_experiments.build_variants(),
identify the subset that most improves CEC2022 (D=10) results, then assemble
those into one composite FeatureConfig the user can transplant to other suites.

USAGE:
    python scripts/feature_ablation.py --phase all --tag screen_v1 --workers 8
    python scripts/feature_ablation.py --phase A    --tag screen_v1 --workers 8
    python scripts/feature_ablation.py --phase B    --tag screen_v1 --workers 8
    python scripts/feature_ablation.py --phase C    --tag screen_v1 --workers 8
    python scripts/feature_ablation.py --phase report --tag screen_v1
    python scripts/feature_ablation.py --phase all --tag smoke --workers 4 --smoke

Phases:
  A — wide screen: all 87 + baseline on 5 representative CEC2022 funcs,
      30k FEs / pop=30 / 5 runs. Output: top 30 by Friedman rank.
  B — confirmation: top 30 + baseline on 6 funcs (different mix),
      100k FEs / pop=50 / 10 runs. Output: top K (default 7) by Friedman
      rank, filtered to wilcoxon_wins >= 1.
  C — composite validation: composite (top-K merged, last-write-wins per
      plug-point with highest-ranked winning) + baseline + top-2 individuals,
      all 12 CEC2022 funcs, 200k FEs / pop=50 / 15 runs.
  report — re-render REPORT.md from existing phase parquets.

State persisted under results/<tag>/:
  phaseA/, phaseB/, phaseC/   full ExperimentRecorder dumps
  phaseA_top30.{json,pkl}     ranked names + cloudpickled FeatureConfigs
  phaseB_topK.{json,pkl}
  composite_spec.json         which feature won each plug-point + dropped log
  composite.pkl               cloudpickled composite FeatureConfig
  manifest.json               run config (FEs, seeds, function lists, decision rule)
  REPORT.md                   human-readable summary
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cloudpickle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ga.core import FeatureConfig
from benchmarks.suites import run_protocol
from analysis.stats import (
    friedman_with_ranks,
    wilcoxon_table,
    mean_error_table,
    write_summary,
)
from persistence import ExperimentRecorder

from run_experiments import build_variants  # 87 + baseline


# --------------------------------------------------------------------------
# Phase configuration
# --------------------------------------------------------------------------

PHASE_A_FUNCS = ["F1", "F4", "F6", "F9", "F11"]
PHASE_B_FUNCS = ["F1", "F3", "F4", "F5", "F7", "F10"]
PHASE_C_FUNCS = [f"F{i}" for i in range(1, 13)]

PHASE_A_PARAMS = dict(pop_size=30, max_fes=30_000, n_runs=5)
PHASE_B_PARAMS = dict(pop_size=50, max_fes=100_000, n_runs=10)
PHASE_C_PARAMS = dict(pop_size=50, max_fes=200_000, n_runs=15)

SMOKE_A_PARAMS = dict(pop_size=20, max_fes=2_000, n_runs=2)
SMOKE_B_PARAMS = dict(pop_size=20, max_fes=2_000, n_runs=2)
SMOKE_C_PARAMS = dict(pop_size=20, max_fes=2_000, n_runs=2)

K_PHASE_A_TO_B = 30
K_PHASE_B_TO_C = 7
SLOW_VARIANT_THRESHOLD_S = 30.0  # median per-run runtime in Phase A
BASELINE_NAME = "SGA_baseline"

PLUG_POINT_SLOTS = (
    "init_fn",
    "select_fn",
    "crossover_fn",
    "mutate_fn",
    "replace_fn",
    "diversity_fn",
    "refine_fn",
)
SCALAR_SLOTS = (
    "cx_prob",
    "mut_prob",
    "elitism_k",
    "n_islands",
    "migration_interval",
    "archive_size",
    "bound_handling",
)


# --------------------------------------------------------------------------
# Result helpers
# --------------------------------------------------------------------------

def runs_parquet_to_results(runs_df: pd.DataFrame) -> dict:
    """Reconstruct results[variant][func] = [errors sorted by seed] from a
    runs.parquet (the format ExperimentRecorder writes)."""
    out: dict = {}
    for v, g_v in runs_df.groupby("variant"):
        out[v] = {}
        for fn, g_fn in g_v.groupby("function"):
            out[v][fn] = (
                g_fn.sort_values("seed")["error"].astype(float).tolist()
            )
    return out


def _sanitize_results(results: dict) -> tuple[dict, list[str]]:
    """Replace NaN seed-errors (from crashed worker jobs — see
    `_worker_run_one` in benchmarks/suites.py) with +inf so they sort to
    "worst" deterministically in Friedman/Wilcoxon rather than silently
    corrupting other variants' ranks. Variants whose every (function, seed)
    crashed are dropped and their names returned for the caller to log."""
    cleaned: dict = {}
    dropped: list[str] = []
    for v, per_fn in results.items():
        any_finite = False
        new_per_fn = {}
        for fn, errs in per_fn.items():
            arr = np.asarray(errs, dtype=float)
            if np.isfinite(arr).any():
                any_finite = True
            arr = np.where(np.isfinite(arr), arr, np.inf)
            new_per_fn[fn] = arr.tolist()
        if any_finite:
            cleaned[v] = new_per_fn
        else:
            dropped.append(v)
    return cleaned, dropped


def compute_ranking(results: dict, baseline_name: str,
                    runs_df: pd.DataFrame | None = None) -> pd.DataFrame:
    """Return a DataFrame with one row per variant, sorted best -> worst.

    Columns: variant, friedman_rank, wilcoxon_wins, wilcoxon_losses,
             median_runtime_s, mean_error_geomean.
    Sort: friedman_rank ASC, then wilcoxon_wins DESC, then runtime ASC.
    """
    results, dropped = _sanitize_results(results)
    for v in dropped:
        print(f"  [WARN] {v}: all runs crashed; dropped from ranking.")
    if not results:
        return pd.DataFrame(columns=[
            "variant", "friedman_rank", "wilcoxon_wins", "wilcoxon_losses",
            "median_runtime_s", "mean_error_geomean",
        ])

    _, _, ranks_df, _, variants, _ = friedman_with_ranks(results)
    ranks_map = dict(zip(ranks_df["variant"], ranks_df["mean_rank"]))

    if baseline_name in results:
        wt = wilcoxon_table(results, baseline_name)
        wins = (wt[wt["sign_vs_baseline"] == "+"]
                .groupby("variant").size().to_dict())
        losses = (wt[wt["sign_vs_baseline"] == "-"]
                  .groupby("variant").size().to_dict())
    else:
        wins, losses = {}, {}

    if runs_df is not None and "runtime_s" in runs_df.columns:
        rt = runs_df.groupby("variant")["runtime_s"].median().to_dict()
    else:
        rt = {}

    # Intersect function sets across variants so ragged result dicts (e.g.
    # a variant that crashed on every seed of one function and was dropped
    # there) don't produce mismatched per-function arrays.
    func_sets = [set(results[v].keys()) for v in variants]
    func_names = sorted(set.intersection(*func_sets)) if func_sets else []

    rows = []
    for v in variants:
        # Floor at 1e-8 (CEC convention: error below this counts as a solve).
        # Without it, a single near-zero mean drags geomean to ~0 and inverts
        # the column's ordering vs. friedman_rank — see SGA+qpso_mut in
        # phaseA_top30.json (err_gmean=0.000 yet rank 43.6).
        per_func_means = np.array(
            [max(np.mean(results[v][fn]), 1e-8) for fn in func_names]
        )
        if per_func_means.size and np.isfinite(per_func_means).all():
            geomean = float(np.exp(np.log(per_func_means).mean()))
        else:
            geomean = float("inf")
        rows.append({
            "variant": v,
            "friedman_rank": float(ranks_map.get(v, np.nan)),
            "wilcoxon_wins": int(wins.get(v, 0)),
            "wilcoxon_losses": int(losses.get(v, 0)),
            "median_runtime_s": float(rt.get(v, np.nan)),
            "mean_error_geomean": geomean,
        })
    df = pd.DataFrame(rows).sort_values(
        by=["friedman_rank", "wilcoxon_wins", "median_runtime_s"],
        ascending=[True, False, True],
    ).reset_index(drop=True)
    return df


# --------------------------------------------------------------------------
# Composite construction
# --------------------------------------------------------------------------

def build_composite(top_k_configs_best_first: list[FeatureConfig],
                    composite_name: str = "SGA+composite") -> tuple[FeatureConfig, dict]:
    """Merge top-K FeatureConfigs into one composite.

    Convention (matches the user's "last-write-wins" choice with rank-aware
    intent): we iterate WORST -> BEST so the BEST-ranked feature's overrides
    are written last and survive. Equivalent: highest-ranked feature wins
    each plug-point conflict.

    Scalars are only overwritten when they differ from the FeatureConfig
    default (so a feature that didn't intentionally set cx_prob=0.85 won't
    silently bash a default someone else cared about).
    """
    default = FeatureConfig()
    out = FeatureConfig(name=composite_name)
    dropped = []
    plug_provenance = {}
    scalar_provenance = {}

    # Track who currently owns each slot so we can log drops as we overwrite.
    current_owner_slot: dict[str, str] = {}

    for cfg in reversed(top_k_configs_best_first):
        for slot in PLUG_POINT_SLOTS:
            v = getattr(cfg, slot)
            if v is None:
                continue
            prev = current_owner_slot.get(slot)
            if prev is not None:
                # We are overwriting a previously-set slot. Because we go
                # worst -> best, the current write is the better-ranked one
                # and should win; the previous (worse) write is dropped.
                dropped.append({
                    "slot": slot,
                    "dropped_from": prev,
                    "kept_from": cfg.name,
                })
            setattr(out, slot, v)
            current_owner_slot[slot] = cfg.name
            plug_provenance[slot] = cfg.name

        for slot in SCALAR_SLOTS:
            v = getattr(cfg, slot)
            if v != getattr(default, slot):
                setattr(out, slot, v)
                scalar_provenance[slot] = {"from": cfg.name, "value": v}

    spec = {
        "name": out.name,
        "ranked_features_best_first": [c.name for c in top_k_configs_best_first],
        "plug_points": {
            slot: {"from": plug_provenance.get(slot)} if plug_provenance.get(slot)
            else {"from": None, "uses_default": True}
            for slot in PLUG_POINT_SLOTS
        },
        "scalar_overrides": scalar_provenance,
        "dropped_overrides": dropped,
        "merge_rule": (
            "Iterate top-K worst->best so the highest-ranked feature's "
            "non-None plug-point survives. Scalars only overwrite when "
            "they differ from FeatureConfig() defaults."
        ),
    }
    return out, spec


# --------------------------------------------------------------------------
# Phase runners
# --------------------------------------------------------------------------

def _run_phase(phase_name: str, variants, dim, params, functions,
               workers, tag_root: Path):
    phase_dir = tag_root / f"phase{phase_name}"
    phase_dir.mkdir(parents=True, exist_ok=True)

    # Resume support: if runs.parquet already exists for this phase, skip.
    runs_parquet = phase_dir / "runs.parquet"
    if runs_parquet.exists():
        print(f"[Phase {phase_name}] {runs_parquet} already exists - "
              "skipping (delete to re-run).")
        runs_df = pd.read_parquet(runs_parquet)
        results = runs_parquet_to_results(runs_df)
        return results, runs_df, phase_dir

    print(f"\n{'=' * 78}\n[Phase {phase_name}] dim=D{dim}, "
          f"variants={len(variants)}, funcs={len(functions)}, "
          f"runs={params['n_runs']}, FEs={params['max_fes']}, "
          f"pop={params['pop_size']}, workers={workers}\n{'=' * 78}")

    args = {
        "phase": phase_name,
        "dim": dim,
        "n_runs": params["n_runs"],
        "max_fes": params["max_fes"],
        "pop_size": params["pop_size"],
        "functions": functions,
        "workers": workers,
        "full": False,
        "tag": phase_dir.name,
    }
    recorder = ExperimentRecorder(
        out_root=phase_dir.parent,
        args=args,
        variants=[v.name for v in variants],
        tag=phase_dir.name,
    )

    t0 = time.time()
    results = run_protocol(
        variants=variants,
        dim=dim,
        n_runs=params["n_runs"],
        max_fes=params["max_fes"],
        pop_size=params["pop_size"],
        recorder=recorder,
        record_history=False,  # screening doesn't need per-gen history
        workers=workers,
        functions=functions,
    )
    elapsed = time.time() - t0
    recorder.flush()
    print(f"[Phase {phase_name}] done in {elapsed/60:.1f} min")

    runs_df = pd.read_parquet(runs_parquet)
    # Always write summary CSVs so the user can explore.
    write_summary(results, phase_dir / "summary",
                  baseline_name=BASELINE_NAME, runs_df=runs_df)
    return results, runs_df, phase_dir


def phase_a(workers: int, dim: int, tag_root: Path, smoke: bool):
    variants = build_variants()
    if smoke:
        # 8 variants is enough to exercise the pipeline + composite.
        keep = {BASELINE_NAME, "SGA+OBL_init", "SGA+sobol_init",
                "SGA+Levy_mut", "SGA+DE_mut", "SGA+SHADE_mut",
                "SGA+greedy_replace", "SGA+restart"}
        variants = [v for v in variants if v.name in keep]
        funcs = ["F1", "F4"]
        params = SMOKE_A_PARAMS
    else:
        funcs = PHASE_A_FUNCS
        params = PHASE_A_PARAMS
    results, runs_df, phase_dir = _run_phase(
        "A", variants, dim, params, funcs, workers, tag_root,
    )

    # Rank and pick top K_PHASE_A_TO_B (baseline always retained).
    ranking = compute_ranking(results, BASELINE_NAME, runs_df=runs_df)
    ranking.to_csv(phase_dir / "ranking.csv", index=False)
    print("\n[Phase A] Top 15 by Friedman rank:")
    print(ranking.head(15).to_string(index=False))

    k = min(K_PHASE_A_TO_B, len(ranking) - 1) if not smoke else min(5, len(ranking) - 1)
    chosen_names = []
    for v in ranking["variant"]:
        if v == BASELINE_NAME:
            continue
        if len(chosen_names) >= k:
            break
        chosen_names.append(v)

    # Slow-variant filter.
    rt_map = dict(zip(ranking["variant"], ranking["median_runtime_s"]))
    if not smoke:
        kept = []
        for n in chosen_names:
            if rt_map.get(n, 0.0) > SLOW_VARIANT_THRESHOLD_S:
                print(f"  (drop slow variant {n}: median runtime "
                      f"{rt_map[n]:.1f}s > {SLOW_VARIANT_THRESHOLD_S}s)")
            else:
                kept.append(n)
        chosen_names = kept

    name_to_cfg = {v.name: v for v in build_variants()}
    chosen_cfgs = [name_to_cfg[n] for n in chosen_names]

    out_json = tag_root / "phaseA_top30.json"
    out_pkl = tag_root / "phaseA_top30.pkl"
    out_json.write_text(json.dumps({
        "selected": chosen_names,
        "ranking_full": ranking.to_dict(orient="records"),
        "k": k,
        "smoke": smoke,
    }, indent=2, default=float))
    with open(out_pkl, "wb") as f:
        cloudpickle.dump(chosen_cfgs, f)
    print(f"\n[Phase A] Wrote {len(chosen_names)} top variants -> {out_json}")
    return chosen_cfgs


def phase_b(workers: int, dim: int, tag_root: Path, smoke: bool):
    pkl = tag_root / "phaseA_top30.pkl"
    if not pkl.exists():
        raise SystemExit(
            f"Phase B requires Phase A output at {pkl}. Run --phase A first.")
    with open(pkl, "rb") as f:
        top_a: list[FeatureConfig] = cloudpickle.load(f)

    name_to_cfg = {v.name: v for v in build_variants()}
    baseline = name_to_cfg[BASELINE_NAME]
    variants = [baseline] + top_a

    if smoke:
        funcs = ["F1", "F4"]
        params = SMOKE_B_PARAMS
    else:
        funcs = PHASE_B_FUNCS
        params = PHASE_B_PARAMS

    results, runs_df, phase_dir = _run_phase(
        "B", variants, dim, params, funcs, workers, tag_root,
    )
    ranking = compute_ranking(results, BASELINE_NAME, runs_df=runs_df)
    ranking.to_csv(phase_dir / "ranking.csv", index=False)
    print("\n[Phase B] Full ranking:")
    print(ranking.to_string(index=False))

    # Filter: must beat baseline on >= 1 function.
    survivors = ranking[
        (ranking["variant"] != BASELINE_NAME) & (ranking["wilcoxon_wins"] >= 1)
    ]
    if len(survivors) < 3:
        print(f"\n[Phase B] WARNING: only {len(survivors)} variants beat "
              "baseline on >=1 function. Composite may not be warranted.")
        # Take all survivors anyway (or fall back to top 3 by rank).
        if len(survivors) == 0:
            survivors = ranking[ranking["variant"] != BASELINE_NAME].head(3)
            print("[Phase B] Falling back to top 3 by Friedman rank.")

    k = min(K_PHASE_B_TO_C, len(survivors)) if not smoke else min(3, len(survivors))
    top_b_names = survivors["variant"].head(k).tolist()
    top_b_cfgs = [name_to_cfg[n] for n in top_b_names]

    out_json = tag_root / "phaseB_topK.json"
    out_pkl = tag_root / "phaseB_topK.pkl"
    out_json.write_text(json.dumps({
        "selected": top_b_names,
        "ranking_full": ranking.to_dict(orient="records"),
        "k": k,
        "smoke": smoke,
    }, indent=2, default=float))
    with open(out_pkl, "wb") as f:
        cloudpickle.dump(top_b_cfgs, f)
    print(f"\n[Phase B] Wrote top {len(top_b_names)} -> {out_json}")

    # Build composite right here so Phase C can pick it up.
    composite, spec = build_composite(
        top_b_cfgs,
        composite_name=f"SGA+composite_top{len(top_b_cfgs)}",
    )
    spec_path = tag_root / "composite_spec.json"
    spec_path.write_text(json.dumps(spec, indent=2, default=float))
    with open(tag_root / "composite.pkl", "wb") as f:
        cloudpickle.dump(composite, f)
    print(f"[Phase B] Built composite: {composite.name}")
    print(f"          plug-points: {spec['plug_points']}")
    print(f"          dropped:     {spec['dropped_overrides']}")

    return top_b_cfgs, composite


def phase_c(workers: int, dim: int, tag_root: Path, smoke: bool):
    composite_pkl = tag_root / "composite.pkl"
    topb_pkl = tag_root / "phaseB_topK.pkl"
    if not composite_pkl.exists() or not topb_pkl.exists():
        raise SystemExit(
            f"Phase C requires composite + Phase B winners. "
            f"Run --phase B first.")
    with open(composite_pkl, "rb") as f:
        composite: FeatureConfig = cloudpickle.load(f)
    with open(topb_pkl, "rb") as f:
        top_b: list[FeatureConfig] = cloudpickle.load(f)

    name_to_cfg = {v.name: v for v in build_variants()}
    baseline = name_to_cfg[BASELINE_NAME]
    # baseline + composite + top-2 individuals (for "is composite > best single?")
    individuals = top_b[:2]
    variants = [baseline, composite] + individuals

    if smoke:
        funcs = ["F1", "F4"]
        params = SMOKE_C_PARAMS
    else:
        funcs = PHASE_C_FUNCS
        params = PHASE_C_PARAMS

    results, runs_df, phase_dir = _run_phase(
        "C", variants, dim, params, funcs, workers, tag_root,
    )
    ranking = compute_ranking(results, BASELINE_NAME, runs_df=runs_df)
    ranking.to_csv(phase_dir / "ranking.csv", index=False)
    print("\n[Phase C] Validation ranking:")
    print(ranking.to_string(index=False))
    return results, ranking


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------

def write_manifest(tag_root: Path, dim: int, workers: int, smoke: bool):
    manifest = {
        "tag": tag_root.name,
        "dim": dim,
        "workers": workers,
        "smoke": smoke,
        "phases": {
            "A": {"functions": PHASE_A_FUNCS,
                  **(SMOKE_A_PARAMS if smoke else PHASE_A_PARAMS)},
            "B": {"functions": PHASE_B_FUNCS,
                  **(SMOKE_B_PARAMS if smoke else PHASE_B_PARAMS),
                  "k_select": K_PHASE_A_TO_B,
                  "slow_variant_threshold_s": SLOW_VARIANT_THRESHOLD_S},
            "C": {"functions": PHASE_C_FUNCS,
                  **(SMOKE_C_PARAMS if smoke else PHASE_C_PARAMS),
                  "k_select": K_PHASE_B_TO_C,
                  "wilcoxon_wins_filter": ">=1"},
        },
        "decision_rule": (
            "rank ascending by friedman_rank; tie-break by wilcoxon_wins "
            "descending then runtime ascending."
        ),
        "merge_rule": (
            "composite plug-points: highest-ranked feature wins (iterate "
            "worst->best, last write survives). Scalars overwrite only if "
            "they differ from FeatureConfig() defaults."
        ),
        "baseline_name": BASELINE_NAME,
    }
    (tag_root / "manifest.json").write_text(json.dumps(manifest, indent=2))


def render_report(tag_root: Path):
    """Emit REPORT.md from whatever phase outputs exist."""
    lines = ["# Feature ablation report", ""]
    lines.append(f"**Tag:** `{tag_root.name}`  ")
    lines.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}  ")
    manifest_p = tag_root / "manifest.json"
    if manifest_p.exists():
        manifest = json.loads(manifest_p.read_text())
        lines.append(f"**Dim:** D={manifest['dim']}  ")
        lines.append(f"**Workers:** {manifest['workers']}  ")
        if manifest.get("smoke"):
            lines.append("**Mode:** SMOKE TEST (tiny budget)  ")
    lines.append("")

    def _phase_section(letter: str, title: str):
        phase_dir = tag_root / f"phase{letter}"
        ranking_csv = phase_dir / "ranking.csv"
        if not ranking_csv.exists():
            return
        ranking = pd.read_csv(ranking_csv)
        lines.append(f"## Phase {letter}: {title}")
        lines.append("")
        head = ranking.head(15) if letter != "C" else ranking
        # Markdown table.
        cols = ["variant", "friedman_rank", "wilcoxon_wins",
                "wilcoxon_losses", "median_runtime_s", "mean_error_geomean"]
        cols = [c for c in cols if c in head.columns]
        lines.append("| " + " | ".join(cols) + " |")
        lines.append("|" + "|".join(["---"] * len(cols)) + "|")
        for _, row in head.iterrows():
            cells = []
            for c in cols:
                v = row[c]
                if isinstance(v, float):
                    if c == "mean_error_geomean":
                        cells.append(f"{v:.3e}")
                    else:
                        cells.append(f"{v:.2f}")
                else:
                    cells.append(str(v))
            lines.append("| " + " | ".join(cells) + " |")
        lines.append("")

    _phase_section("A", "Wide screen (87 features + baseline)")
    _phase_section("B", "Confirmation (top 30 + baseline)")

    # Composite spec.
    spec_p = tag_root / "composite_spec.json"
    if spec_p.exists():
        spec = json.loads(spec_p.read_text())
        lines.append("## Composite construction")
        lines.append("")
        lines.append(f"**Composite name:** `{spec['name']}`")
        lines.append("")
        lines.append("**Top-K winners (best -> worst):**")
        for i, n in enumerate(spec["ranked_features_best_first"], 1):
            lines.append(f"  {i}. `{n}`")
        lines.append("")
        lines.append("**Plug-point assignments (highest-ranked wins):**")
        lines.append("")
        lines.append("| slot | from |")
        lines.append("|---|---|")
        for slot, info in spec["plug_points"].items():
            origin = info.get("from") or "*(default)*"
            lines.append(f"| `{slot}` | {origin} |")
        lines.append("")
        if spec.get("scalar_overrides"):
            lines.append("**Non-default scalars:**")
            lines.append("")
            for slot, info in spec["scalar_overrides"].items():
                lines.append(f"- `{slot}` = `{info['value']}` (from `{info['from']}`)")
            lines.append("")
        if spec.get("dropped_overrides"):
            lines.append("**Dropped overrides (lower-ranked feature lost the slot):**")
            lines.append("")
            for d in spec["dropped_overrides"]:
                lines.append(
                    f"- slot `{d['slot']}`: dropped `{d['dropped_from']}`, "
                    f"kept `{d['kept_from']}`"
                )
            lines.append("")
        lines.append(f"**Merge rule:** {spec['merge_rule']}")
        lines.append("")

    _phase_section("C", "Composite validation (all 12 CEC2022 funcs, full budget)")

    # Recommendation.
    cranking = tag_root / "phaseC" / "ranking.csv"
    if cranking.exists():
        r = pd.read_csv(cranking)
        if not r.empty:
            best = r.iloc[0]
            lines.append("## Transplant recommendation")
            lines.append("")
            lines.append(
                f"On the full CEC2022 (D=10) validation, the top-ranked "
                f"variant is **`{best['variant']}`** "
                f"(Friedman rank {best['friedman_rank']:.2f}, "
                f"{best['wilcoxon_wins']} wins / "
                f"{best['wilcoxon_losses']} losses vs baseline)."
            )
            lines.append("")
            if "composite" in best["variant"]:
                lines.append(
                    "The composite outperforms the best single feature on "
                    "this validation. Recommend transplanting the "
                    "composite (see `composite_spec.json`) to the "
                    "engineering / separable / TSP / timetabling suites."
                )
            else:
                lines.append(
                    "The composite did NOT beat the best single feature. "
                    "Recommend transplanting the single feature instead, "
                    "or running greedy forward selection (a different "
                    "composition strategy) before committing."
                )
            lines.append("")

    out = tag_root / "REPORT.md"
    out.write_text("\n".join(lines) + "\n")
    print(f"Wrote {out}")


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", required=True,
                        help="Run directory under results/.")
    parser.add_argument("--phase", choices=["A", "B", "C", "all", "report"],
                        default="all")
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--out-root", type=str, default="results")
    parser.add_argument("--smoke", action="store_true",
                        help="Run with tiny budget on a few variants for "
                             "pipeline validation (~minutes, not hours).")
    args = parser.parse_args()

    tag_root = Path(args.out_root) / args.tag
    tag_root.mkdir(parents=True, exist_ok=True)
    write_manifest(tag_root, args.dim, args.workers, args.smoke)

    if args.phase in ("A", "all"):
        phase_a(args.workers, args.dim, tag_root, args.smoke)
    if args.phase in ("B", "all"):
        phase_b(args.workers, args.dim, tag_root, args.smoke)
    if args.phase in ("C", "all"):
        phase_c(args.workers, args.dim, tag_root, args.smoke)

    render_report(tag_root)
    print(f"\nDone. See {tag_root / 'REPORT.md'}")


if __name__ == "__main__":
    main()
