"""
Per-batch experiment archival.

Each invocation of run_experiments.py creates a single timestamped directory
under <out_root>/exp_<timestamp>_D<dim>_<mode>/ holding:
  - config.json     : args, variant names, git SHA, library versions
  - runs.parquet    : tidy long-form per-run final metrics
  - history.parquet : per-generation history for every run
  - summary/        : statistical CSVs (filled by analysis.stats)
  - plots/          : figures (filled by analysis.plots)
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd


def _git_sha() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def _lib_versions() -> dict[str, str]:
    versions: dict[str, str] = {"python": sys.version.split()[0]}
    for mod in ("numpy", "scipy", "pandas", "matplotlib", "seaborn",
                "deap", "opfunu", "statsmodels", "scikit_posthocs"):
        try:
            m = __import__(mod)
            versions[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            versions[mod] = "not-installed"
    return versions


@dataclass
class ExperimentRecorder:
    out_root: Path
    args: dict[str, Any]
    variants: list[str]
    tag: str | None = None
    runs_buf: list[dict] = field(default_factory=list)
    hist_buf: list[dict] = field(default_factory=list)
    exp_dir: Path = field(init=False)

    def __post_init__(self):
        self.out_root = Path(self.out_root)
        self.out_root.mkdir(parents=True, exist_ok=True)
        if self.tag:
            name = self.tag
        else:
            mode = "full" if self.args.get("full") else "demo"
            ts = time.strftime("%Y-%m-%d_%H-%M-%S")
            name = f"exp_{ts}_D{self.args.get('dim', '?')}_{mode}"
        self.exp_dir = self.out_root / name
        self.exp_dir.mkdir(parents=True, exist_ok=True)
        (self.exp_dir / "summary").mkdir(exist_ok=True)
        (self.exp_dir / "plots").mkdir(exist_ok=True)
        self._write_config()

    def _write_config(self) -> None:
        cfg = {
            "args": self.args,
            "variants": self.variants,
            "git_sha": _git_sha(),
            "platform": platform.platform(),
            "library_versions": _lib_versions(),
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        with open(self.exp_dir / "config.json", "w") as f:
            json.dump(cfg, f, indent=2)

    def add_run(self, variant: str, function: str, seed: int,
                result: dict, runtime_s: float) -> None:
        self.runs_buf.append({
            "variant": variant,
            "function": function,
            "seed": seed,
            "error": float(result["error"]),
            "best_fitness": float(result["best_fitness"]),
            "fes_used": int(result["fes_used"]),
            "runtime_s": float(runtime_s),
        })
        for h in result.get("history", []) or []:
            self.hist_buf.append({
                "variant": variant,
                "function": function,
                "seed": seed,
                "gen": int(h["gen"]),
                "fes": int(h["fes"]),
                "best_fitness": float(h["best_fitness"]),
                "mean_fitness": float(h["mean_fitness"]),
                "diversity": float(h["diversity"]),
                "hussain_div": float(h.get("hussain_div", 0.0)),
                "xpl_pct": float(h.get("xpl_pct", 0.0)),
                "xpt_pct": float(h.get("xpt_pct", 0.0)),
                "stagnation_count": int(h.get("stagnation_count", 0)),
                "mut_success_rate": float(h.get("mut_success_rate", 0.0)),
                "cx_success_rate": float(h.get("cx_success_rate", 0.0)),
                "mean_step_size": float(h.get("mean_step_size", 0.0)),
                "selection_pressure": float(h.get("selection_pressure", 0.0)),
            })

    def flush(self) -> None:
        runs_df = pd.DataFrame(self.runs_buf)
        runs_df.to_parquet(self.exp_dir / "runs.parquet", index=False)
        if self.hist_buf:
            hist_df = pd.DataFrame(self.hist_buf)
            hist_df.to_parquet(self.exp_dir / "history.parquet", index=False)
        print(f"\nArchived {len(self.runs_buf)} runs to {self.exp_dir}")
