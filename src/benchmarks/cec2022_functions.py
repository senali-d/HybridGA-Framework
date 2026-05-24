"""
CEC 2022 single-objective benchmark suite (F1-F12).

Twelve rotated/shifted, mostly non-separable functions, sourced from the
`opfunu` package. Each class exposes the standard problem interface
(`ndim`, `lb`, `ub`, `evaluate(x) -> float`, `f_global`) so `run_sga` can
consume them unchanged.
"""

from opfunu.cec_based.cec2022 import (
    F12022, F22022, F32022, F42022, F52022, F62022,
    F72022, F82022, F92022, F102022, F112022, F122022,
)


CEC2022_FUNCS = [
    ("F1",  F12022), ("F2",  F22022), ("F3",  F32022), ("F4",  F42022),
    ("F5",  F52022), ("F6",  F62022), ("F7",  F72022), ("F8",  F82022),
    ("F9",  F92022), ("F10", F102022), ("F11", F112022), ("F12", F122022),
]

# Statically encoded from opfunu's CEC2022 sources: F1-F5 ship shift/rotation
# data for D ∈ {2, 10, 20}; the hybrid (F6-F8) and composition (F9-F12) suites
# only ship D ∈ {10, 20}. We mirror the values here because some F6+ classes
# fail in __init__ (loading missing data files) before `dim_supported` is set,
# which makes runtime probing unreliable.
CEC2022_DIM_SUPPORTED = {
    "F1": (2, 10, 20), "F2": (2, 10, 20), "F3": (2, 10, 20),
    "F4": (2, 10, 20), "F5": (2, 10, 20),
    "F6": (10, 20), "F7": (10, 20), "F8": (10, 20),
    "F9": (10, 20), "F10": (10, 20), "F11": (10, 20), "F12": (10, 20),
}
