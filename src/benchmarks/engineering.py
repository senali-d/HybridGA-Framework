"""
Mechanical-engineering design problems from the CEC 2020 Real-World
Constrained Optimization suite (Kumar, Wu, Ali, Mallipeddi, Suganthan, Das
2020), ported verbatim from the reference MATLAB implementation in
P-N-Suganthan/2020-RW-Constrained-Optimisation (Software/CEC2020/cec20_func.m).

    - TensionCompressionSpring  (prob_k=17, RC-spring case 1)   3 vars, 4 g
    - PressureVessel            (prob_k=18)                      4 vars, 4 g
    - WeldedBeam                (prob_k=19)                      4 vars, 5 g
    - ThreeBarTruss             (prob_k=20)                      2 vars, 3 g
    - SpeedReducer              (prob_k=15)                      7 vars, 11 g

I-Beam and the 5-segment Cantilever are NOT part of CEC2020-RW and are
omitted from this suite. Bounds, objective scaling, and constraint forms
match the MATLAB source exactly (including its non-textbook quirks, e.g.,
WB uses `Pc ∝ sqrt(t² b⁶/30)` and `delta = 6PL³/(E t² b)`, SR uses 7.477
not 7.4777, and SR's x5 lower bound is 7.3).

Constraint handling
-------------------
Deb's feasibility rules encoded into a *scalar* fitness:

    feasible   →  evaluate(x) = f(x)             (always < F_OFFSET)
    infeasible →  evaluate(x) = F_OFFSET + Σ max(0, g_i(x))

F_OFFSET is chosen larger than any plausible feasible f for each problem so
the scalar ordering is lex-equivalent to Deb's rule. No changes to ga/core
or feature operators are required. Discrete encoding for PressureVessel
(thicknesses are round(x)·0.0625) is applied inside `_canonicalize(x)`.
"""

import numpy as np


class EngineeringProblem:
    """Base class. Subclasses set NDIM/LB/UB/F_GLOBAL/F_OFFSET and implement
    `_objective(x)` and `constraints(x)`. Optional `_canonicalize(x)` snaps
    discrete variables before objective and constraint evaluation."""

    NDIM = None
    LB = None
    UB = None
    F_GLOBAL = 0.0
    F_OFFSET = 1.0e6

    def __init__(self, ndim=None):
        self.ndim = int(self.NDIM)
        self.lb = np.asarray(self.LB, dtype=float)
        self.ub = np.asarray(self.UB, dtype=float)
        self.f_global = float(self.F_GLOBAL)

    def _canonicalize(self, x):
        return np.asarray(x, dtype=float)

    def _objective(self, x):  # pragma: no cover - abstract
        raise NotImplementedError

    def constraints(self, x):  # pragma: no cover - abstract
        raise NotImplementedError

    def evaluate(self, x):
        x_c = self._canonicalize(np.asarray(x, dtype=float))
        f = float(self._objective(x_c))
        g = np.asarray(self.constraints(x_c), dtype=float)
        v = float(np.sum(np.maximum(g, 0.0)))
        if v == 0.0:
            return f
        return float(self.F_OFFSET) + v


# ============================================================================
# CEC2020-RW prob_k = 17 : Tension/Compression Spring (case 1)
# Variables x = [d (wire dia.), D (mean coil dia.), N (active coils)]
# ============================================================================

class TensionCompressionSpring(EngineeringProblem):
    NDIM = 3
    LB = [0.05, 0.25, 2.00]
    UB = [2.00, 1.30, 15.0]
    F_GLOBAL = 0.012665232788
    F_OFFSET = 100.0

    def _objective(self, x):
        d, D, N = x
        return d * d * D * (N + 2.0)

    def constraints(self, x):
        d, D, N = x
        g1 = 1.0 - (D ** 3 * N) / (71785.0 * d ** 4)

        denom = 12566.0 * (D * d ** 3 - d ** 4)
        if abs(denom) <= 1.0e-12:
            g2 = 1.0e6
        else:
            g2 = (4.0 * D * D - d * D) / denom + 1.0 / (5108.0 * d * d) - 1.0

        g3 = 1.0 - 140.45 * d / (D * D * N)
        g4 = (d + D) / 1.5 - 1.0
        return np.array([g1, g2, g3, g4])


# ============================================================================
# CEC2020-RW prob_k = 18 : Pressure Vessel
# Variables: x = [n_Ts, n_Th, R, L] with n_Ts, n_Th ∈ [0.51, 99.49] rounded
# to integers and multiplied by 0.0625 to get Ts, Th in inches.
# ============================================================================

class PressureVessel(EngineeringProblem):
    NDIM = 4
    LB = [0.51, 0.51, 10.0, 10.0]
    UB = [99.49, 99.49, 200.0, 200.0]
    F_GLOBAL = 6059.714335048436
    # Upper bound for feasible f: evaluate objective at the (snapped)
    # corner Ts = Th = 99·0.0625 = 6.1875, R = L = 200, then add margin.
    F_OFFSET = 1.01 * (
        0.6224 * (99 * 0.0625) * 200.0 * 200.0
        + 1.7781 * (99 * 0.0625) * 200.0 * 200.0
        + 3.1661 * (99 * 0.0625) ** 2 * 200.0
        + 19.84 * (99 * 0.0625) ** 2 * 200.0
    )

    def _canonicalize(self, x):
        x = np.array(x, dtype=float)
        x[0] = 0.0625 * round(x[0])
        x[1] = 0.0625 * round(x[1])
        return x

    def _objective(self, x):
        Ts, Th, R, L = x
        return (0.6224 * Ts * R * L
                + 1.7781 * Th * R * R
                + 3.1661 * Ts * Ts * L
                + 19.84 * Ts * Ts * R)

    def constraints(self, x):
        Ts, Th, R, L = x
        g1 = -Ts + 0.0193 * R
        g2 = -Th + 0.00954 * R
        g3 = -np.pi * R * R * L - (4.0 / 3.0) * np.pi * R ** 3 + 1_296_000.0
        g4 = L - 240.0
        return np.array([g1, g2, g3, g4])


# ============================================================================
# CEC2020-RW prob_k = 19 : Welded Beam
# Variables: x = [h, l, t, b]   (CEC variable order x1..x4)
# Note: CEC2020 uses the non-textbook constants Pc ∝ sqrt(t²·b⁶/30) and
# delta = 6PL³/(E·t²·b). J term is (l²/4 + (h+t)²/4). Only 5 inequality
# constraints (no welding-cost or h≥0.125 limit).
# ============================================================================

class WeldedBeam(EngineeringProblem):
    NDIM = 4
    LB = [0.125, 0.1, 0.1, 0.1]
    UB = [2.0, 10.0, 10.0, 2.0]
    F_GLOBAL = 1.724852308597366
    F_OFFSET = 1.0e4

    _P = 6000.0
    _L = 14.0
    _E = 30.0e6
    _G = 12.0e6
    _TAU_MAX = 13_600.0
    _SIGMA_MAX = 30_000.0
    _DELTA_MAX = 0.25

    def _objective(self, x):
        h, l, t, b = x
        return 1.10471 * h * h * l + 0.04811 * t * b * (14.0 + l)

    def constraints(self, x):
        h, l, t, b = x
        P, L, E, G = self._P, self._L, self._E, self._G

        Pc = (4.013 * E * np.sqrt(t * t * b ** 6 / 30.0) / (L * L)
              * (1.0 - t / (2.0 * L) * np.sqrt(E / (4.0 * G))))
        sigma = 6.0 * P * L / (b * t * t)
        delta = 6.0 * P * L ** 3 / (E * t * t * b)

        J = 2.0 * (np.sqrt(2.0) * h * l
                   * (l * l / 4.0 + (h + t) ** 2 / 4.0))
        R = np.sqrt(l * l / 4.0 + (h + t) ** 2 / 4.0)
        M = P * (L + l / 2.0)
        tau_pp = M * R / J
        tau_p = P / (np.sqrt(2.0) * h * l)
        tau = np.sqrt(tau_p * tau_p
                      + 2.0 * tau_p * tau_pp * l / (2.0 * R)
                      + tau_pp * tau_pp)

        g1 = tau - self._TAU_MAX
        g2 = sigma - self._SIGMA_MAX
        g3 = h - b
        g4 = delta - self._DELTA_MAX
        g5 = P - Pc
        return np.array([g1, g2, g3, g4, g5])


# ============================================================================
# CEC2020-RW prob_k = 20 : Three-Bar Truss
# Variables: x = [A1, A2] ∈ [0,1]².  L=100, P=2, σ_max=2 baked into formula.
# ============================================================================

class ThreeBarTruss(EngineeringProblem):
    NDIM = 2
    LB = [0.0, 0.0]
    UB = [1.0, 1.0]
    F_GLOBAL = 263.8958433764628
    F_OFFSET = 1.0e3

    def _objective(self, x):
        A1, A2 = x
        return (2.0 * np.sqrt(2.0) * A1 + A2) * 100.0

    def constraints(self, x):
        A1, A2 = x
        denom = np.sqrt(2.0) * A1 * A1 + 2.0 * A1 * A2
        if abs(denom) <= 1e-12:
            return np.array([1.0e6, 1.0e6, 1.0e6])
        g1 = (np.sqrt(2.0) * A1 + A2) / denom * 2.0 - 2.0
        g2 = A2 / denom * 2.0 - 2.0
        g3 = 1.0 / (np.sqrt(2.0) * A2 + A1) * 2.0 - 2.0
        return np.array([g1, g2, g3])


# ============================================================================
# CEC2020-RW prob_k = 15 : Weight Minimization of a Speed Reducer
# Variables: x = [b, m, z, l1, l2, d1, d2]
# Note: x5 (l2) lower bound is 7.3 in CEC2020 (not the 7.8 in some other
# sources). Teeth count z is *not* snapped to integer in the CEC code.
# Objective uses 7.477 (CEC) not 7.4777.
# ============================================================================

class SpeedReducer(EngineeringProblem):
    NDIM = 7
    LB = [2.6, 0.7, 17.0, 7.3, 7.3, 2.9, 5.0]
    UB = [3.6, 0.8, 28.0, 8.3, 8.3, 3.9, 5.5]
    F_GLOBAL = 2994.42410581
    F_OFFSET = 1.0e6

    def _objective(self, x):
        b, m, z, l1, l2, d1, d2 = x
        return (0.7854 * b * m * m
                * (3.3333 * z * z + 14.9334 * z - 43.0934)
                - 1.508 * b * (d1 * d1 + d2 * d2)
                + 7.477 * (d1 ** 3 + d2 ** 3)
                + 0.7854 * (l1 * d1 * d1 + l2 * d2 * d2))

    def constraints(self, x):
        b, m, z, l1, l2, d1, d2 = x
        g1 = -b * m * m * z + 27.0
        g2 = -b * m * m * z * z + 397.5
        g3 = -m * d1 ** 4 * z * l1 ** (-3) + 1.93
        g4 = -m * d2 ** 4 * z / l2 ** 3 + 1.93
        g5 = 10.0 * d1 ** (-3) * np.sqrt(16.91e6
                                         + (745.0 * l1 / (m * z)) ** 2) - 1100.0
        g6 = 10.0 * d2 ** (-3) * np.sqrt(157.5e6
                                         + (745.0 * l2 / (m * z)) ** 2) - 850.0
        g7 = m * z - 40.0
        g8 = -b / m + 5.0
        g9 = b / m - 12.0
        g10 = 1.5 * d1 - l1 + 1.9
        g11 = 1.1 * d2 - l2 + 1.9
        return np.array([g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11])


ENGINEERING_FUNCS = [
    ("TensionCompressionSpring", TensionCompressionSpring),
    ("PressureVessel",           PressureVessel),
    ("WeldedBeam",               WeldedBeam),
    ("ThreeBarTruss",            ThreeBarTruss),
    ("SpeedReducer",             SpeedReducer),
]
