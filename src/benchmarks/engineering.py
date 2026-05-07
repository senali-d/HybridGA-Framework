"""
Classic constrained mechanical-engineering design problems used widely as
metaheuristic benchmarks (Coello, Arora, Deb references):

    - TensionCompressionSpring   (3 vars, 4 inequality constraints)
    - PressureVessel             (4 vars, 4 inequality constraints, mixed-integer)
    - WeldedBeam                 (4 vars, 7 inequality constraints)
    - SpeedReducer               (7 vars, 11 inequality constraints, mixed-integer)

Constraint handling
-------------------
Deb's feasibility rules, encoded into a *scalar* fitness so that every
existing selection / replacement / mutation operator in the framework
compares solutions correctly with no awareness of constraints:

    feasible   →  evaluate(x) = f(x)             (always < F_OFFSET)
    infeasible →  evaluate(x) = F_OFFSET + Σ max(0, g_i(x))

Because `F_OFFSET` is chosen larger than any plausible feasible f for the
problem, the resulting scalar ordering is *lex-equivalent* to Deb's pair
comparison rule (any feasible beats any infeasible; feasibles ranked by f;
infeasibles ranked by total constraint violation). No changes to
`ga/core.py` or any feature operator are required.

Discrete variables (Pressure Vessel thicknesses snap to 0.0625" multiples;
Speed Reducer teeth count is integer) are handled inside `evaluate()` via
`_canonicalize(x)`. The GA continues to see continuous box-bounded vars.

Convention: `best_fitness < F_OFFSET`  iff  the best solution is feasible.
"""

import numpy as np


class EngineeringProblem:
    """Base class. Subclasses set NDIM/LB/UB/F_GLOBAL/F_OFFSET as class
    attrs and implement `_objective(x)` and `constraints(x)`. Optional
    `_canonicalize(x)` snaps discrete variables before objective and
    constraint evaluation."""

    NDIM = None
    LB = None        # array-like, shape (NDIM,)
    UB = None        # array-like, shape (NDIM,)
    F_GLOBAL = 0.0   # known literature optimum (feasible)
    F_OFFSET = 1.0e6  # > any plausible feasible f

    def __init__(self, ndim=None):
        # The shared run-protocol passes ndim=kw["dim"]; we ignore the
        # incoming value because engineering problems have a fixed
        # dimensionality. Stored as a public attr so the GA core sees it.
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
# Tension / Compression Spring
# ----------------------------------------------------------------------------
# Variables: x = [d (wire diameter), D (mean coil diameter), N (active coils)]
# Minimize:  f = (N + 2) * D * d^2
# Subject to four inequality constraints (Belegundu 1982; Arora 1989).
# ============================================================================

class TensionCompressionSpring(EngineeringProblem):
    NDIM = 3
    LB = [0.05, 0.25, 2.0]
    UB = [2.00, 1.30, 15.0]
    F_GLOBAL = 0.012665232788
    F_OFFSET = 100.0

    def _objective(self, x):
        d, D, N = x
        return (N + 2.0) * D * d * d

    def constraints(self, x):
        d, D, N = x
        g1 = 1.0 - (D ** 3 * N) / (71785.0 * d ** 4)

        denom = 12566.0 * (D * d ** 3 - d ** 4)
        if abs(denom) <= 1.0e-12:
            # The literature form of g2 is singular when D == d.
            # Treat singular / near-singular points as strongly infeasible
            # with a large but finite violation so evaluate() stays finite.
            g2 = 1.0e12
        else:
            g2 = (4.0 * D * D - d * D) / denom + 1.0 / (5108.0 * d * d) - 1.0

        g3 = 1.0 - (140.45 * d) / (D * D * N)
        g4 = (D + d) / 1.5 - 1.0
        return np.array([g1, g2, g3, g4])


# ============================================================================
# Pressure Vessel
# ----------------------------------------------------------------------------
# Variables: x = [Ts (shell thickness), Th (head thickness), R (radius), L (length)]
# Ts, Th must be positive integer multiples of 0.0625 inches.
# Minimize:  f = 0.6224 Ts R L + 1.7781 Th R^2
#               + 3.1661 Ts^2 L + 19.84 Ts^2 R
# Subject to four inequality constraints (Kannan & Kramer 1994).
# ============================================================================

class PressureVessel(EngineeringProblem):
    NDIM = 4
    LB = [0.0625, 0.0625, 10.0, 10.0]
    UB = [99.0,   99.0,   200.0, 200.0]
    F_GLOBAL = 6059.714335048436
    F_OFFSET = 1.0e7

    def _canonicalize(self, x):
        x = np.array(x, dtype=float)
        # Ts (x[0]) and Th (x[1]) snap to nearest 0.0625" multiple, >= 0.0625
        x[0] = max(round(x[0] / 0.0625) * 0.0625, 0.0625)
        x[1] = max(round(x[1] / 0.0625) * 0.0625, 0.0625)
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
# Welded Beam
# ----------------------------------------------------------------------------
# Variables: x = [h (weld thickness), l (weld length),
#                 t (beam height),    b (beam width)]
# Minimize:  f = 1.10471 h^2 l + 0.04811 t b (14.0 + l)
# Subject to seven inequality constraints (Coello 2000; Ragsdell & Phillips 1976).
# ============================================================================

class WeldedBeam(EngineeringProblem):
    NDIM = 4
    LB = [0.1, 0.1, 0.1, 0.1]
    UB = [2.0, 10.0, 10.0, 2.0]
    F_GLOBAL = 1.724852308597366
    F_OFFSET = 1.0e4

    # Material / loading constants
    _P = 6000.0      # load (lb)
    _L = 14.0        # overhang (in)
    _E = 30.0e6      # Young's modulus (psi)
    _G = 12.0e6      # shear modulus (psi)
    _TAU_MAX = 13_600.0
    _SIGMA_MAX = 30_000.0
    _DELTA_MAX = 0.25

    def _objective(self, x):
        h, l, t, b = x
        return 1.10471 * h * h * l + 0.04811 * t * b * (self._L + l)

    def constraints(self, x):
        h, l, t, b = x
        P, L, E, G = self._P, self._L, self._E, self._G

        M = P * (L + l / 2.0)
        R = np.sqrt(l * l / 4.0 + ((h + t) / 2.0) ** 2)
        J = 2.0 * (np.sqrt(2.0) * h * l
                   * (l * l / 12.0 + ((h + t) / 2.0) ** 2))

        tau_p = P / (np.sqrt(2.0) * h * l)
        tau_pp = M * R / J
        tau = np.sqrt(tau_p * tau_p
                      + 2.0 * tau_p * tau_pp * l / (2.0 * R)
                      + tau_pp * tau_pp)

        sigma = 6.0 * P * L / (b * t * t)
        delta = 4.0 * P * L ** 3 / (E * b * t ** 3)
        Pc = ((4.013 * E * np.sqrt(t * t * b ** 6 / 36.0)) / (L * L)) \
             * (1.0 - (t / (2.0 * L)) * np.sqrt(E / (4.0 * G)))

        g1 = tau - self._TAU_MAX
        g2 = sigma - self._SIGMA_MAX
        g3 = h - b
        g4 = 0.10471 * h * h + 0.04811 * t * b * (L + l) - 5.0
        g5 = 0.125 - h
        g6 = delta - self._DELTA_MAX
        g7 = P - Pc
        return np.array([g1, g2, g3, g4, g5, g6, g7])


# ============================================================================
# Speed Reducer
# ----------------------------------------------------------------------------
# Variables: x = [b (face width), m (module), z (teeth, INTEGER),
#                 l1 (shaft 1 length), l2 (shaft 2 length),
#                 d1 (shaft 1 dia.),   d2 (shaft 2 dia.)]
# Minimize:  classical 7-var weight expression (Golinski 1973).
# Subject to 11 inequality constraints.
# ============================================================================

class SpeedReducer(EngineeringProblem):
    NDIM = 7
    LB = [2.6, 0.7, 17.0, 7.3, 7.8, 2.9, 5.0]
    UB = [3.6, 0.8, 28.0, 8.3, 8.3, 3.9, 5.5]
    F_GLOBAL = 2994.42410581
    F_OFFSET = 1.0e6

    def _canonicalize(self, x):
        x = np.array(x, dtype=float)
        # Teeth count z (x[2]) is integer.
        x[2] = float(int(round(x[2])))
        return x

    def _objective(self, x):
        b, m, z, l1, l2, d1, d2 = x
        return (0.7854 * b * m * m
                * (3.3333 * z * z + 14.9334 * z - 43.0934)
                - 1.508 * b * (d1 * d1 + d2 * d2)
                + 7.4777 * (d1 ** 3 + d2 ** 3)
                + 0.7854 * (l1 * d1 * d1 + l2 * d2 * d2))

    def constraints(self, x):
        b, m, z, l1, l2, d1, d2 = x
        g1 = 27.0 / (b * m * m * z) - 1.0
        g2 = 397.5 / (b * m * m * z * z) - 1.0
        g3 = 1.93 * l1 ** 3 / (m * z * d1 ** 4) - 1.0
        g4 = 1.93 * l2 ** 3 / (m * z * d2 ** 4) - 1.0
        A1 = (745.0 * l1 / (m * z)) ** 2 + 16.9e6
        g5 = np.sqrt(A1) / (110.0 * d1 ** 3) - 1.0
        A2 = (745.0 * l2 / (m * z)) ** 2 + 157.5e6
        g6 = np.sqrt(A2) / (85.0 * d2 ** 3) - 1.0
        g7 = m * z / 40.0 - 1.0
        g8 = 5.0 * m / b - 1.0
        g9 = b / (12.0 * m) - 1.0
        g10 = (1.5 * d1 + 1.9) / l1 - 1.0
        g11 = (1.1 * d2 + 1.9) / l2 - 1.0
        return np.array([g1, g2, g3, g4, g5, g6, g7, g8, g9, g10, g11])


ENGINEERING_FUNCS = [
    ("TensionCompressionSpring", TensionCompressionSpring),
    ("PressureVessel",           PressureVessel),
    ("WeldedBeam",               WeldedBeam),
    ("SpeedReducer",             SpeedReducer),
]
