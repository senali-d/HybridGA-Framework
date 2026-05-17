"""
University Course Timetabling Problem (UCTP) suite — Socha 2002 instances
from the IRIDIA "Metaheuristics Network" benchmark set.

Each instance assigns `n` events to 45 timeslots (9 per day × 5 days) and
`m` rooms, subject to four hard constraints and three soft constraints
from Rossi-Doria et al. (2002):

  Hard constraints (HCV):
    H1. No student attends two events in the same timeslot.
    H2. The room is large enough for the event's attending students.
    H3. The room satisfies every feature the event requires.
    H4. Only one event is in a given (room, timeslot) at a time.

  Soft constraints (SCV):
    S1. A student has a class in the last timeslot of a day (slot 8/17/...).
    S2. A student has more than two classes in a row on a day.
    S3. A student has a single class on a day.

Encoding: an integer vector `x[0..n-1]` where `x[i] ∈ {0..44}` is the
timeslot of event `i`. Room assignment is recovered inside `evaluate`
by a greedy per-timeslot matching that picks, for each event in that
slot, the smallest feasible (size + features) room not already used in
the slot. Events that cannot be placed contribute one HCV each.

Fitness (minimization): `f(x) = C * #hcv + #scv` where `C` is set so
any single HCV outweighs all possible SCVs (see `_HCV_WEIGHT`).
`f_global = 0` (a feasible timetable with no soft penalty).

Instance file format (Socha .tim):
  line 1:   n_events n_rooms n_features n_students
  next:     n_rooms ints     (room sizes)
  next:     n_students * n_events 0/1 ints, row-major (student-event)
  next:     n_rooms * n_features 0/1 ints, row-major (room-features)
  next:     n_events * n_features 0/1 ints, row-major (event-features)

Instance files are not bundled. Run `scripts/fetch_socha.py` once to
populate `data/socha/`, or set `SOCHA_DIR` to point at an existing copy.

Bound handling: discrete-integer encoding has no meaningful clip /
reflect operation, so the suite uses `bound_handling: "passthrough"`.
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


def _parse_socha_instance(path: Path):
    """Parse a .tim instance into raw arrays. Returns a dict."""
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


class UCTPProblem:
    """A single Socha-2002 UCTP instance.

    Exposes the same `(ndim, lb, ub, evaluate, f_global)` interface as the
    real-coded and TSP benchmarks so the SGA core consumes it unchanged.
    `lb` / `ub` are timeslot bounds (0..44) for every gene; the GA uses
    passthrough bound handling and never clips against them.
    """

    INSTANCE_NAME = None  # set on subclasses

    def __init__(self, name=None, socha_dir=None):
        name = name or self.INSTANCE_NAME
        if name is None:
            raise ValueError("UCTPProblem needs an instance name (e.g. 'easy01').")
        socha_dir = Path(socha_dir) if socha_dir else _default_socha_dir()
        path = socha_dir / f"{name}.tim"
        if not path.exists():
            raise FileNotFoundError(
                f"Socha instance not found: {path}\n"
                f"Run `python scripts/fetch_socha.py` to download the "
                f"15 baseline instances, or set SOCHA_DIR to an existing copy."
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
        # room_ok[event, room]: room large enough AND has every required feature.
        size_ok = self.room_sizes[None, :] >= self.event_size[:, None]
        # features_required ⊆ features_present <=> required AND NOT present == 0
        feat_missing = (
            self.event_features[:, None, :]
            & ~self.room_features[None, :, :]
        ).any(axis=-1)
        self.room_ok = size_ok & ~feat_missing  # (n_events, n_rooms)

        # Per-event list of feasible rooms sorted by ascending capacity —
        # the greedy matcher tries the tightest fit first to free large
        # rooms for harder events.
        order = np.argsort(self.room_sizes, kind="stable")
        self._feasible_rooms = [
            [int(r) for r in order if self.room_ok[e, r]]
            for e in range(self.n_events)
        ]

        # ndim / bounds / optimum
        self.ndim = int(self.n_events)
        self.lb = np.zeros(self.n_events, dtype=float)
        self.ub = np.full(self.n_events, N_TIMESLOTS - 1, dtype=float)
        self.f_global = 0.0

        # HCV weight: any single hard violation must outweigh the largest
        # possible SCV total. Worst-case SCV is bounded by
        # n_students * (S1 + S2 + S3) <= n_students * 3 * N_DAYS per day-
        # related count; we use a comfortable upper bound.
        self._hcv_weight = int(self.n_students * 3 * N_DAYS + self.n_events + 1)

    # ---- evaluation --------------------------------------------------

    def evaluate(self, assignment) -> float:
        """Return `C * #hcv + #scv` for the given event→timeslot vector.

        `assignment` is any iterable of ints (length `n_events`) with
        values in `[0, 44]`. Out-of-range values are clipped here so
        unconstrained operators never crash this path.
        """
        x = np.asarray(assignment, dtype=np.int64)
        if x.shape != (self.n_events,):
            raise ValueError(
                f"assignment shape {x.shape}, expected ({self.n_events},)"
            )
        # Defensive: clip into [0, 44]. Discrete operators should already
        # produce in-range values, but evaluate has to be total.
        x = np.clip(x, 0, N_TIMESLOTS - 1)

        hcv = 0
        # Group events by timeslot.
        events_in_slot = [[] for _ in range(N_TIMESLOTS)]
        for e in range(self.n_events):
            events_in_slot[int(x[e])].append(e)

        # Per-slot greedy room matching. An event with no feasible room
        # left contributes one HCV (covers H2/H3/H4 simultaneously).
        event_placed = np.zeros(self.n_events, dtype=bool)
        for slot, events in enumerate(events_in_slot):
            if not events:
                continue
            used_rooms = set()
            # Order events by fewest feasible rooms first (hardest first)
            # so the greedy match doesn't waste a tight room on an event
            # that has many alternatives.
            events_sorted = sorted(
                events, key=lambda e: len(self._feasible_rooms[e])
            )
            for e in events_sorted:
                placed = False
                for r in self._feasible_rooms[e]:
                    if r not in used_rooms:
                        used_rooms.add(r)
                        event_placed[e] = True
                        placed = True
                        break
                if not placed:
                    hcv += 1

        # H1: per student, count timeslots in which they attend >1 event.
        # Vectorize over students by computing student_event @ slot_count.
        # slot_count[s, t] = # events attended by student s in slot t.
        slot_of = x  # (n_events,)
        # one-hot event → slot, multiply by attendance
        # Equivalent to: for each student, count events in each slot.
        # Build (n_students, n_timeslots) attendance-per-slot matrix.
        attend_slot = np.zeros((self.n_students, N_TIMESLOTS), dtype=np.int64)
        # np.add.at: scatter-add along the slot axis weighted by attendance
        # (broadcast student_event columns into slot bins).
        for e in range(self.n_events):
            attend_slot[:, slot_of[e]] += self.student_event[:, e].astype(np.int64)
        hcv += int(np.sum(np.maximum(attend_slot - 1, 0)))

        # ---- soft constraints ----
        scv = 0
        # S1: classes in the last slot of a day. Last-slot indices: 8, 17, 26, 35, 44.
        last_slots = np.arange(SLOTS_PER_DAY - 1, N_TIMESLOTS, SLOTS_PER_DAY)
        scv += int(attend_slot[:, last_slots].sum())

        # Reshape attendance to (n_students, n_days, slots_per_day) for S2/S3.
        day_attend = (attend_slot > 0).reshape(
            self.n_students, N_DAYS, SLOTS_PER_DAY
        )
        # S2: >2 consecutive attended slots in a day. Each "run" of length
        # L ≥ 3 contributes L-2 to the count (one penalty per slot beyond
        # the second consecutive). Compute via run-length scan.
        # For each (student, day), walk slots and count runs.
        # Vectorized: count positions where prev2 and prev1 and current all attended.
        if SLOTS_PER_DAY >= 3:
            triple = (
                day_attend[:, :, 2:]
                & day_attend[:, :, 1:-1]
                & day_attend[:, :, :-2]
            )
            scv += int(triple.sum())

        # S3: exactly one class on a day.
        day_count = day_attend.sum(axis=2)  # (n_students, n_days)
        scv += int((day_count == 1).sum())

        return float(self._hcv_weight * hcv + scv)


def _make_instance(name):
    cls = type(
        f"UCTP_{name}",
        (UCTPProblem,),
        {"INSTANCE_NAME": name, "__doc__": f"Socha UCTP instance {name}."},
    )

    # Re-bind __init__ so the shared run protocol's positional `ndim`
    # argument is accepted and ignored — UCTP instances have a fixed
    # event count baked into their .tim file.
    def _init(self, ndim=None, socha_dir=None):
        super(cls, self).__init__(name=name, socha_dir=socha_dir)
    cls.__init__ = _init
    return cls


# Registry: order matches configs/timetabling.json (5 easy, 5 medium, 5 hard).
SOCHA_FUNCS = [
    (n, _make_instance(n))
    for n in (
        [f"easy{i:02d}" for i in range(1, 6)]
        + [f"medium{i:02d}" for i in range(1, 6)]
        + [f"hard{i:02d}" for i in range(1, 6)]
    )
]
