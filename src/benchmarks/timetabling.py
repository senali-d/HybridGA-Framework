"""
University Course Timetabling Problem (UCTP) suite.

Two instance sets are exposed:

  - `SOCHA_FUNCS`     — Socha 2002 instances (15 in `data/socha/`):
                        easy01-05, medium01-05, hard01-05.
  - `TTC2002_FUNCS`   — TTC 2002 competition instances (20 in
                        `data/ttc2002/`): comp01..comp20.

Both use the same `.tim` file format (same parser) and the same evaluation
machinery. The combined registry `UCTP_FUNCS` merges them in
configs/timetabling.json's `instances` order.

Each instance assigns `n` events to 45 timeslots (9 per day x 5 days) and
`m` rooms, subject to four hard constraints and three soft constraints
from Rossi-Doria et al. (2002):

  Hard constraints (HCV):
    H1. No student attends two events in the same timeslot.
    H2. The room is large enough for the event's attending students.
    H3. The room satisfies every feature the event requires.
    H4. Only one event is in a given (room, timeslot) at a time.

  Soft constraints (SCV):
    S1. A student has a class in the last timeslot of a day.
    S2. A student has more than two classes in a row on a day.
    S3. A student has a single class on a day.

Encoding: an integer vector `x[0..n-1]` where `x[i] in {0..44}` is the
timeslot of event `i`. Room assignment is recovered by a per-timeslot
max-cardinality bipartite matching against rooms whose size+features fit
(Rossi-Doria et al. 2003, Yang & Jat 2011). Events that cannot be placed
contribute one HCV each, covering H2/H3/H4 simultaneously.

Fitness (minimization): `f(x) = C * #hcv + #scv` where `C` is set so any
single HCV outweighs all possible SCVs (see `self._hcv_weight`).
`f_global = 0` (a feasible timetable with no soft penalty).

Instance file format (.tim, identical for Socha and TTC 2002):
  line 1:   n_events n_rooms n_features n_students
  next:     n_rooms ints     (room sizes)
  next:     n_students * n_events 0/1 ints, row-major (student-event)
  next:     n_rooms * n_features 0/1 ints, row-major (room-features)
  next:     n_events * n_features 0/1 ints, row-major (event-features)

Instance files are not bundled. Run:
  - `scripts/fetch_socha.py`    to populate `data/socha/`
  - `scripts/fetch_ttc2002.py`  to populate `data/ttc2002/`
or set `SOCHA_DIR` / `TTC2002_DIR` to point at existing copies.

Bound handling: discrete-integer encoding has no meaningful clip / reflect
operation, so the suite uses `bound_handling: "passthrough"`.
"""

import os
from pathlib import Path

import numpy as np


N_TIMESLOTS = 45
SLOTS_PER_DAY = 9
N_DAYS = 5


def _default_socha_dir():
    env = os.environ.get("SOCHA_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "data" / "socha"


def _default_ttc2002_dir():
    env = os.environ.get("TTC2002_DIR")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "data" / "ttc2002"


def _parse_socha_instance(path: Path):
    """Parse a .tim instance into raw arrays. Returns a dict.

    The same parser works for Socha 2002 and TTC 2002 — file format is
    identical (only the header sizes differ).
    """
    nums = np.fromstring(path.read_text(), sep=" ", dtype=np.int64)
    if nums.size < 4:
        raise ValueError(f"{path.name}: file too short")
    n_events, n_rooms, n_features, n_students = (int(x) for x in nums[:4])
    cursor = 4

    def take(count):
        nonlocal cursor
        end = cursor + count
        if end > nums.size:
            raise ValueError(
                f"{path.name}: truncated (need {count} more ints at "
                f"cursor {cursor}, file has {nums.size})"
            )
        chunk = nums[cursor:end]
        cursor = end
        return chunk

    room_sizes = take(n_rooms).astype(np.int64)
    student_event = take(n_students * n_events).reshape(n_students, n_events).astype(bool)
    room_features = take(n_rooms * n_features).reshape(n_rooms, n_features).astype(bool)
    event_features = take(n_events * n_features).reshape(n_events, n_features).astype(bool)

    return {
        "n_events": n_events,
        "n_rooms": n_rooms,
        "n_features": n_features,
        "n_students": n_students,
        "room_sizes": room_sizes,
        "student_event": student_event,
        "room_features": room_features,
        "event_features": event_features,
    }


# ---- bipartite max-cardinality matching (Hopcroft-Karp would overkill
# the problem sizes here; the simple augmenting-path routine matches
# Rossi-Doria's network-flow result for graphs with <=20 left nodes / 20
# right nodes and is exact). ---------------------------------------------

def _max_match(adj, n_left, n_right):
    """Return (match_left, n_matched) where match_left[u] is the matched
    right-node or -1. `adj[u]` is the iterable of feasible right-nodes
    for left-node u.
    """
    match_left = [-1] * n_left
    match_right = [-1] * n_right

    def _augment(u, seen):
        for v in adj[u]:
            if seen[v]:
                continue
            seen[v] = True
            if match_right[v] == -1 or _augment(match_right[v], seen):
                match_left[u] = v
                match_right[v] = u
                return True
        return False

    n_matched = 0
    for u in range(n_left):
        seen = [False] * n_right
        if _augment(u, seen):
            n_matched += 1
    return match_left, n_matched


class UCTPProblem:
    """A single UCTP instance (Socha or TTC 2002).

    Exposes the same `(ndim, lb, ub, evaluate, f_global)` interface as the
    real-coded and TSP benchmarks so the SGA core consumes it unchanged.
    Additional API for paper-faithful steady-state GAs:

      - `assign_rooms(timeslots, slots=None)` — max-cardinality matching
        per slot. `slots=None` matches every slot; passing a subset reuses
        an existing `rooms` array and only rematches the named slots.
      - `score(timeslots, rooms)` — vectorized fitness from a timeslot
        vector and a precomputed room vector. Returns `(fitness, hcv, scv)`.
      - `evaluate_full(x)` — `(fitness, hcv, scv, rooms)`; does not go
        through the FE-counted path.

    `lb` / `ub` are timeslot bounds (0..44) for every gene; the GA uses
    passthrough bound handling and never clips against them.
    """

    INSTANCE_NAME = None  # set on subclasses
    DATA_DIR = None       # set on subclasses ("socha" or "ttc2002" key)

    def __init__(self, name=None, data_dir=None, socha_dir=None):
        # Back-compat: callers may still pass `socha_dir=...`.
        if socha_dir is not None and data_dir is None:
            data_dir = socha_dir
        name = name or self.INSTANCE_NAME
        if name is None:
            raise ValueError("UCTPProblem needs an instance name (e.g. 'easy01').")

        if data_dir is None:
            data_dir = (
                _default_ttc2002_dir() if self.DATA_DIR == "ttc2002"
                else _default_socha_dir()
            )
        data_dir = Path(data_dir)
        path = data_dir / f"{name}.tim"
        if not path.exists():
            hint = (
                "scripts/fetch_ttc2002.py" if self.DATA_DIR == "ttc2002"
                else "scripts/fetch_socha.py"
            )
            raise FileNotFoundError(
                f"UCTP instance not found: {path}\n"
                f"Run `python {hint}` to download the bundled instances, "
                f"or point SOCHA_DIR / TTC2002_DIR at an existing copy."
            )
        data = _parse_socha_instance(path)
        self.name = name
        self.n_events = data["n_events"]
        self.n_rooms = data["n_rooms"]
        self.n_features = data["n_features"]
        self.n_students = data["n_students"]
        self.room_sizes = data["room_sizes"]
        self.student_event = data["student_event"]
        self.room_features = data["room_features"]
        self.event_features = data["event_features"]

        # Derived: event size, per-event feasible-room mask (size + features).
        self.event_size = self.student_event.sum(axis=0).astype(np.int64)
        size_ok = self.room_sizes[None, :] >= self.event_size[:, None]
        # features_required subset features_present <=> required AND NOT present == 0
        feat_missing = (
            self.event_features[:, None, :]
            & ~self.room_features[None, :, :]
        ).any(axis=-1)
        self.room_ok = size_ok & ~feat_missing  # (n_events, n_rooms)

        # Per-event list of feasible rooms (no ordering needed for max-card
        # matching — the augmenting path finds the optimum regardless).
        self._feasible_rooms = [
            [int(r) for r in range(self.n_rooms) if self.room_ok[e, r]]
            for e in range(self.n_events)
        ]

        # ndim / bounds / optimum
        self.ndim = int(self.n_events)
        self.lb = np.zeros(self.n_events, dtype=float)
        self.ub = np.full(self.n_events, N_TIMESLOTS - 1, dtype=float)
        self.f_global = 0.0

        # HCV weight: any single hard violation must outweigh the largest
        # possible SCV total. Paper §V sets C = 10^6 explicitly; we use the
        # same scale-free constant for full paper fidelity.
        self._hcv_weight = 1_000_000

        # Cast student_event to int8 once for fast matrix products in score().
        self._student_event_i = self.student_event.astype(np.int8)

    # ---- room assignment (max-cardinality matching) ------------------

    def assign_rooms(self, timeslots, slots=None, rooms=None):
        """Return a length-`n_events` array of room indices (or -1 for
        unplaced events). If `slots` is a list/iterable of slot indices
        and `rooms` is a starting room vector, only rematch those slots
        (much cheaper than a full rematch — used by LS1 trial moves).
        """
        ts = np.asarray(timeslots, dtype=np.int64)
        if rooms is None or slots is None:
            # Full rematch on all 45 slots. Build events_by_slot via a
            # vectorized scatter rather than a Python loop.
            rooms_out = np.full(self.n_events, -1, dtype=np.int64)
            events_by_slot = [[] for _ in range(N_TIMESLOTS)]
            for e in range(self.n_events):
                events_by_slot[int(ts[e])].append(e)
            for slot in range(N_TIMESLOTS):
                events = events_by_slot[slot]
                if not events:
                    continue
                adj = [self._feasible_rooms[e] for e in events]
                match_left, _ = _max_match(adj, len(events), self.n_rooms)
                for i, e in enumerate(events):
                    rooms_out[e] = match_left[i]
            return rooms_out

        # Partial rematch: only build events_by_slot for the requested
        # slots. Iterate the timeslots array once to bucket events into
        # those slots; events in other slots keep their existing rooms.
        slot_set = set(int(s) for s in slots)
        rooms_out = np.asarray(rooms, dtype=np.int64).copy()
        buckets = {s: [] for s in slot_set}
        for e in range(self.n_events):
            t = int(ts[e])
            if t in slot_set:
                buckets[t].append(e)
                rooms_out[e] = -1
        for slot, events in buckets.items():
            if not events:
                continue
            adj = [self._feasible_rooms[e] for e in events]
            match_left, _ = _max_match(adj, len(events), self.n_rooms)
            for i, e in enumerate(events):
                rooms_out[e] = match_left[i]
        return rooms_out

    # ---- vectorized scoring -----------------------------------------

    def score(self, timeslots, rooms):
        """Compute `(fitness, hcv, scv)` from a timeslot vector and a
        room vector. Does NOT go through the FE-counted path."""
        ts = np.asarray(timeslots, dtype=np.int64)
        rm = np.asarray(rooms, dtype=np.int64)
        # H2/H3/H4 (and unplaced events from matching): one HCV per event
        # with room == -1.
        hcv = int(np.sum(rm < 0))

        # H1: per student, count slots with attendance > 1.
        # attend_slot[s, t] = sum over events e with ts[e]==t of student_event[s, e]
        # Built via one-hot multiply: attend_slot = student_event @ onehot(ts)
        onehot = np.zeros((self.n_events, N_TIMESLOTS), dtype=np.int8)
        onehot[np.arange(self.n_events), ts] = 1
        attend_slot = self._student_event_i @ onehot  # (n_students, N_TIMESLOTS)
        hcv += int(np.sum(np.maximum(attend_slot - 1, 0)))

        # ---- soft constraints ----
        scv = 0
        # S1: classes in the last slot of each day.
        last_slots = np.arange(SLOTS_PER_DAY - 1, N_TIMESLOTS, SLOTS_PER_DAY)
        # Per (student, last-slot) cell counts how many classes the
        # student has in that slot (typically 0/1 — but >1 is already an
        # H1 violation; we still penalize the soft side per attendance).
        scv += int(attend_slot[:, last_slots].sum())

        # Reshape to (n_students, n_days, slots_per_day) booleans.
        day_attend = (attend_slot > 0).reshape(
            self.n_students, N_DAYS, SLOTS_PER_DAY
        )
        # S2: >2 consecutive attended slots in a day. Each run of length
        # L >= 3 contributes L-2; equivalent to counting positions where
        # current AND prev1 AND prev2 are all attended.
        if SLOTS_PER_DAY >= 3:
            triple = (
                day_attend[:, :, 2:]
                & day_attend[:, :, 1:-1]
                & day_attend[:, :, :-2]
            )
            scv += int(triple.sum())

        # S3: exactly one class on a day.
        day_count = day_attend.sum(axis=2)
        scv += int((day_count == 1).sum())

        fitness = float(self._hcv_weight * hcv + scv)
        return fitness, hcv, scv

    # ---- evaluation -------------------------------------------------

    def evaluate(self, assignment) -> float:
        """Return `C * #hcv + #scv` for the given event->timeslot vector.

        FE-counted entry point used by the GA core (`problem.evaluate`).
        Internally calls `assign_rooms` + `score` and discards the
        intermediates.
        """
        x = np.asarray(assignment, dtype=np.int64)
        if x.shape != (self.n_events,):
            raise ValueError(
                f"assignment shape {x.shape}, expected ({self.n_events},)"
            )
        # Defensive: clip into [0, 44].
        x = np.clip(x, 0, N_TIMESLOTS - 1)
        rooms = self.assign_rooms(x)
        f, _, _ = self.score(x, rooms)
        return f

    def evaluate_full(self, assignment):
        """Same as `evaluate` but returns `(fitness, hcv, scv, rooms)`.
        Intended for paper-faithful local search loops (LS1/LS2 in
        Yang & Jat 2011) that need the per-constraint counts and the
        explicit room assignment for delta-style updates. Not FE-counted
        on its own — the caller is responsible for accounting.
        """
        x = np.asarray(assignment, dtype=np.int64)
        x = np.clip(x, 0, N_TIMESLOTS - 1)
        rooms = self.assign_rooms(x)
        f, hcv, scv = self.score(x, rooms)
        return f, hcv, scv, rooms


# ---- instance factories ------------------------------------------------

def _make_instance(name, data_dir_key="socha"):
    cls = type(
        f"UCTP_{name}",
        (UCTPProblem,),
        {
            "INSTANCE_NAME": name,
            "DATA_DIR": data_dir_key,
            "__doc__": f"UCTP instance {name} ({data_dir_key}).",
        },
    )

    # Re-bind __init__ so the shared run protocol's positional `ndim`
    # argument is accepted and ignored — UCTP instances have a fixed
    # event count baked into their .tim file.
    def _init(self, ndim=None, data_dir=None, socha_dir=None):
        super(cls, self).__init__(
            name=name, data_dir=data_dir, socha_dir=socha_dir,
        )
    cls.__init__ = _init
    return cls


# Registry: 15 Socha instances (5 easy + 5 medium + 5 hard).
SOCHA_FUNCS = [
    (n, _make_instance(n, data_dir_key="socha"))
    for n in (
        [f"easy{i:02d}" for i in range(1, 6)]
        + [f"medium{i:02d}" for i in range(1, 6)]
        + [f"hard{i:02d}" for i in range(1, 6)]
    )
]

# TTC 2002 competition instances (comp01..comp20). The paper compares
# EGSGA against the TTC entrants on this exact 20-instance test set
# (Table VII).
TTC2002_FUNCS = [
    (f"comp{i:02d}", _make_instance(f"comp{i:02d}", data_dir_key="ttc2002"))
    for i in range(1, 21)
]

# Merged registry exposed to the suite dispatcher. Same-named entries
# would shadow each other; Socha and TTC 2002 use disjoint name spaces
# (easy*/medium*/hard* vs comp*) so the merge is safe.
UCTP_FUNCS = SOCHA_FUNCS + TTC2002_FUNCS
