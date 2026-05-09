"""Re-run Phase C top variants with per-generation history recording.

Loads the composite from results/<tag>/composite.pkl alongside the named
top variants from build_variants(), runs the CEC2022 protocol with
record_history=True, then renders convergence/diversity/etc plots.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cloudpickle

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from benchmarks.suites import run_protocol
from persistence import ExperimentRecorder
from analysis import plots as plot_mod

from run_experiments import build_variants


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source-tag", default="screen_v1",
                    help="Tag under results/ to load composite.pkl from.")
    ap.add_argument("--out-tag", default="screen_v1_curves",
                    help="Output tag under results/.")
    ap.add_argument("--dim", type=int, default=10)
    ap.add_argument("--n-runs", type=int, default=5)
    ap.add_argument("--max-fes", type=int, default=200_000)
    ap.add_argument("--pop-size", type=int, default=50)
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    composite_pkl = ROOT / "results" / args.source_tag / "composite.pkl"
    if not composite_pkl.exists():
        raise SystemExit(f"Missing {composite_pkl}")
    with open(composite_pkl, "rb") as f:
        composite = cloudpickle.load(f)

    name_to_cfg = {v.name: v for v in build_variants()}
    wanted_named = ["SGA_baseline", "SGA+crowding_replace",
                    "SGA+fast_ga_cauchy_anneal"]
    variants = [name_to_cfg[n] for n in wanted_named] + [composite]

    funcs = [f"F{i}" for i in range(1, 13)]
    out_root = ROOT / "results"
    recorder = ExperimentRecorder(
        out_root=out_root,
        args={
            "dim": args.dim, "n_runs": args.n_runs,
            "max_fes": args.max_fes, "pop_size": args.pop_size,
            "functions": funcs, "workers": args.workers,
            "source_tag": args.source_tag,
        },
        variants=[v.name for v in variants],
        tag=args.out_tag,
    )

    print(f"Variants: {[v.name for v in variants]}")
    print(f"Funcs: {funcs}")
    print(f"D={args.dim} pop={args.pop_size} FEs={args.max_fes} "
          f"runs={args.n_runs} workers={args.workers}")

    t0 = time.time()
    run_protocol(
        variants=variants, dim=args.dim,
        n_runs=args.n_runs, max_fes=args.max_fes,
        pop_size=args.pop_size, recorder=recorder,
        record_history=True, workers=args.workers, functions=funcs,
    )
    recorder.flush()
    print(f"Done in {(time.time() - t0) / 60:.1f} min")

    print("Generating plots...")
    plot_mod.generate_all(recorder.exp_dir, baseline_name="SGA_baseline")
    print(f"Output: {recorder.exp_dir}")


if __name__ == "__main__":
    main()
