"""
Travelling Salesman Problem suite backed by TSPLIB EUC_2D instances.

Each problem exposes the same `(ndim, lb, ub, evaluate, f_global)`
interface as the real-coded benchmarks, so the SGA core consumes them
unchanged. `evaluate(perm)` accepts a permutation of `{0..n-1}` and
returns the closed-tour length using TSPLIB's nearest-integer Euclidean
distance convention (`int(round(d))`).

Instance files (`.tsp`) are not bundled. Run `scripts/fetch_tsplib.py`
once to populate `data/tsplib/` from a public TSPLIB mirror, or set the
`TSPLIB_DIR` environment variable to point at an existing copy.

Bound handling: combinatorial encoding has no meaningful clip/reflect
operation, so suites using TSP set `bound_handling: "passthrough"` in
their JSON config.
"""

import os
from pathlib import Path

import numpy as np


# Known optima (closed-tour length under TSPLIB EUC_2D rounding).
# Source: TSPLIB-95 "Best known solutions for symmetric TSPs" page.
_TSPLIB_OPTIMA = {
    "st70":     675,
    "eil51":    426,
    "berlin52": 7542,
    "kroA100":  21282,
    "ch150":    6528,
    "kroA200":  29368,
}


def _default_tsplib_dir():
    env = os.environ.get("TSPLIB_DIR")
    if env:
        return Path(env)
    # Repo-root/data/tsplib by default.
    return Path(__file__).resolve().parents[2] / "data" / "tsplib"


def _parse_euc2d_tsp(path: Path):
    """Minimal TSPLIB EUC_2D parser. Returns (n, coords) where coords is
    an (n, 2) float array. Raises if the file is not EUC_2D."""
    text = path.read_text()
    edge_type = None
    n = None
    in_coords = False
    coords = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        if ":" in line and not in_coords:
            key, _, val = line.partition(":")
            key = key.strip().upper()
            val = val.strip()
            if key == "DIMENSION":
                n = int(val)
            elif key == "EDGE_WEIGHT_TYPE":
                edge_type = val.upper()
        if line.upper() == "NODE_COORD_SECTION":
            in_coords = True
            continue
        if line.upper() in ("EOF", "DISPLAY_DATA_SECTION"):
            break
        if in_coords:
            parts = line.split()
            # Format: "<id> <x> <y>"
            if len(parts) >= 3:
                coords.append((float(parts[1]), float(parts[2])))
    if edge_type != "EUC_2D":
        raise ValueError(
            f"{path.name}: only EDGE_WEIGHT_TYPE EUC_2D is supported; "
            f"found {edge_type!r}"
        )
    if n is None or len(coords) != n:
        raise ValueError(
            f"{path.name}: dimension mismatch (header says {n}, parsed "
            f"{len(coords)} coords)"
        )
    return n, np.asarray(coords, dtype=float)


def _euc2d_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """TSPLIB EUC_2D distance: nint(sqrt((x_i-x_j)^2 + (y_i-y_j)^2)).

    Per the TSPLIB-95 spec, ``nint(x) = floor(x + 0.5)`` — round half *up*,
    not Python/NumPy's round-half-to-even. Using the wrong rule shifts a
    handful of edge weights by one unit and makes optimal-tour lengths
    disagree with the published optima, so compute it explicitly.
    """
    diff = coords[:, None, :] - coords[None, :, :]
    d = np.sqrt(np.sum(diff * diff, axis=-1))
    return np.floor(d + 0.5).astype(np.int64)


class TSPProblem:
    """A single TSPLIB EUC_2D instance.

    Attributes match the real-coded benchmark interface so the SGA core
    consumes it unchanged. `lb`/`ub` are dummy permutation bounds
    (0..n-1) for downstream tooling; the GA uses passthrough bound
    handling and never clips against them.
    """

    INSTANCE_NAME = None  # set on subclasses or via from_tsplib_name

    def __init__(self, name=None, tsplib_dir=None):
        name = name or self.INSTANCE_NAME
        if name is None:
            raise ValueError(
                "TSPProblem needs an instance name (e.g. 'eil51')."
            )
        tsplib_dir = Path(tsplib_dir) if tsplib_dir else _default_tsplib_dir()
        path = tsplib_dir / f"{name}.tsp"
        if not path.exists():
            raise FileNotFoundError(
                f"TSPLIB instance not found: {path}\n"
                f"Run `python scripts/fetch_tsplib.py` to download the "
                f"five baseline instances, or set TSPLIB_DIR to an "
                f"existing TSPLIB copy."
            )
        n, coords = _parse_euc2d_tsp(path)
        self.name = name
        self.coords = coords
        self.dist = _euc2d_distance_matrix(coords)
        self.ndim = n
        self.lb = np.zeros(n, dtype=float)
        self.ub = np.full(n, n - 1, dtype=float)
        self.f_global = float(_TSPLIB_OPTIMA.get(name, 0.0))

    def evaluate(self, perm) -> float:
        """Closed-tour length under TSPLIB EUC_2D rounding.

        `perm` is any iterable of int city indices (0..n-1). Duplicate or
        out-of-range entries are not validated here for speed; the SGA
        baseline uses a permutation init + permutation operators (OX,
        swap, inversion) that preserve the permutation invariant.
        """
        idx = np.asarray(perm, dtype=np.int64)
        nxt = np.roll(idx, -1)
        return float(self.dist[idx, nxt].sum())


def _make_instance(name):
    cls = type(
        f"TSP_{name}",
        (TSPProblem,),
        {"INSTANCE_NAME": name, "__doc__": f"TSPLIB instance {name}."},
    )
    # Re-bind __init__ so positional `ndim` (passed by the shared run
    # protocol) is accepted and ignored — TSP instances have a fixed dim
    # baked into the .tsp file, mirroring the engineering benchmarks.
    def _init(self, ndim=None, tsplib_dir=None):
        super(cls, self).__init__(name=name, tsplib_dir=tsplib_dir)
    cls.__init__ = _init
    return cls


# Registry: order matches the spec in configs/tsp.json (plus st70).
TSPLIB_FUNCS = [
    ("st70",     _make_instance("st70")),
    ("eil51",    _make_instance("eil51")),
    ("berlin52", _make_instance("berlin52")),
    ("kroA100",  _make_instance("kroA100")),
    ("ch150",    _make_instance("ch150")),
    ("kroA200",  _make_instance("kroA200")),
]
