"""Physical macro-actions. Only this module handles realized hidden worlds.

Time is abstract graph time, not wall-clock time. Public snapshots are immutable
and contain neither scenario IDs nor random seeds nor hidden door states.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
import numpy as np

TAKE_SAFE = 6
REGIMES = {"static_clean": (0.0, 0.0), "changing_clean": (0.2, 0.0),
           "changing_missed": (0.2, 0.25)}
N_TASKS = 6


def keyed_seed(*parts: object) -> int:
    payload = json.dumps(parts, separators=(",", ":"), ensure_ascii=True).encode()
    return int.from_bytes(hashlib.sha256(payload).digest()[:16], "big")


@dataclass(frozen=True)
class Layout:
    a: tuple[float, float, float]
    b: tuple[float, float, float]
    v: tuple[float, float, float]
    sensing: float
    safe: float
    budget: float

    def __post_init__(self):
        if any(len(x) != 3 for x in (self.a, self.b, self.v)):
            raise ValueError("Exactly three doors are required")
        values = (*self.a, *self.b, *self.v, self.sensing, self.safe, self.budget)
        if not all(np.isfinite(x) and x > 0 for x in values):
            raise ValueError("All travel, sensing and budget values must be positive")

    def return_cost(self, position: int) -> float:
        # S=0, U_i=1+i, V_i=4+i, G=7. No action starts at G.
        if position == 0:
            return 0.0
        if 1 <= position <= 3:
            return self.a[position - 1]
        if 4 <= position <= 6:
            return self.v[position - 4]
        raise ValueError("Cannot act from terminal goal")


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    namespace: str
    regime: str
    layout: Layout
    worlds: tuple[tuple[bool, bool, bool], ...]
    flip: float
    miss: float

    def __post_init__(self):
        if len(self.worlds) != N_TASKS or any(len(w) != 3 for w in self.worlds):
            raise ValueError("A scenario must contain six three-door worlds")
        if not (0 <= self.flip <= 1 and 0 <= self.miss <= 1):
            raise ValueError("Invalid probability")


def make_scenario(namespace: str, index: int, regime: str) -> Scenario:
    if namespace.split("/")[0] not in {"train", "val", "test", "pilot"}:
        raise ValueError("Use train, val, test or pilot seed namespaces")
    if index < 0:
        raise ValueError("Scenario index must be nonnegative")
    flip, miss = REGIMES[regime]
    identity = f"{namespace}/{regime}/{index:06d}"
    layout_rng = np.random.default_rng(keyed_seed(identity, "layout"))
    world_rng = np.random.default_rng(keyed_seed(identity, "world"))
    safe = float(layout_rng.uniform(20, 30))
    layout = Layout(tuple(layout_rng.uniform(6, 14, 3)),
                    tuple(layout_rng.uniform(1, 4, 3)),
                    tuple(layout_rng.uniform(0.5, 2.5, 3)),
                    float(layout_rng.uniform(0.5, 2)), safe,
                    safe * float(layout_rng.uniform(1.1, 1.6)))
    doors = world_rng.random(3) < 0.5
    worlds = [tuple(bool(x) for x in doors)]
    for _ in range(N_TASKS - 1):
        doors = np.logical_xor(doors, world_rng.random(3) < flip)
        worlds.append(tuple(bool(x) for x in doors))
    return Scenario(identity, namespace, regime, layout, tuple(worlds), flip, miss)


def sensing_miss(scenario: Scenario, task: int, door: int, attempt: int) -> bool:
    """An action-count-independent draw, indexed only by this door's attempt."""
    rng = np.random.default_rng(keyed_seed(scenario.scenario_id, "sensing", task, door, attempt))
    return bool(rng.random() < scenario.miss)


@dataclass(frozen=True)
class Snapshot:
    layout: Layout
    flip: float
    miss: float
    task: int = 0
    position: int = 0
    remaining: float = 0.0
    belief: tuple[float, float, float] = (0.5, 0.5, 0.5)
    last: tuple[int, int, int] = (-1, -1, -1)  # unknown=-1, blocked=0, open=1
    age: tuple[int, int, int] = (-1, -1, -1)  # boundary count, unknown=-1
    observed: tuple[bool, bool, bool] = (False, False, False)
    attempted: tuple[bool, bool, bool] = (False, False, False)


def initial_snapshot(layout: Layout, flip: float, miss: float) -> Snapshot:
    return Snapshot(layout, flip, miss, remaining=layout.budget)


def tuple_set(values: tuple, index: int, value) -> tuple:
    return values[:index] + (value,) + values[index + 1:]


def observe(snapshot: Snapshot, door: int, opened: bool) -> Snapshot:
    return replace(snapshot,
                   belief=tuple_set(snapshot.belief, door, float(opened)),
                   last=tuple_set(snapshot.last, door, int(opened)),
                   age=tuple_set(snapshot.age, door, 0),
                   observed=tuple_set(snapshot.observed, door, True))


@dataclass(frozen=True)
class PhysicalResult:
    snapshot: Snapshot
    dt: float
    success: bool
    failure: bool
    blocked_attempt: int
    evidence: tuple[tuple[int, bool], ...]

    @property
    def ended(self) -> bool:
        return self.success or self.failure


def physical_step(snapshot: Snapshot, action: int, world: tuple[bool, ...],
                  missed: bool = False) -> PhysicalResult:
    """Execute one macro-action without a task-boundary reset.

    The planner supplies a hypothetical world; the environment supplies the
    actual world. Evidence is emitted only at completed physical events.
    Exact equality completes an event before the timeout check.
    """
    if action not in range(7):
        raise ValueError("Action must be in [0, 6]")
    if snapshot.remaining <= 0:
        raise ValueError("Cannot act after timeout")
    s = snapshot
    start_remaining = float(s.remaining)
    completed_duration = 0.0
    evidence: list[tuple[int, bool]] = []
    blocked = 0
    success = False
    if 3 <= action < 6:
        s = replace(s, attempted=tuple_set(s.attempted, action - 3, True))

    def spend(duration: float, destination: int | None = None) -> bool:
        nonlocal s, completed_duration
        duration = float(duration)
        event_time = completed_duration + duration
        # At most three positive segments occur in a macro-action. Use ONE
        # four-ULP equality convention for cumulative and remaining time alike.
        # This covers independent decimal literals (6.2 + 1.1 versus 7.3) as
        # well as subtraction roundoff. Shortfalls beyond this bound fail.
        tolerance = 4 * math.ulp(max(start_remaining, event_time, duration))
        if event_time - start_remaining > tolerance or duration - s.remaining > tolerance:
            s = replace(s, remaining=0.0)
            return False
        completed_duration = event_time
        remaining = start_remaining - completed_duration
        if abs(remaining) <= tolerance:
            remaining = 0.0
        s = replace(s, remaining=remaining,
                    position=s.position if destination is None else destination)
        return True

    def result() -> PhysicalResult:
        # A tolerated equality must never charge more than the available time.
        consumed = min(start_remaining, max(0.0, start_remaining - s.remaining))
        return PhysicalResult(s, consumed, success,
                              not success and s.remaining <= 0, blocked, tuple(evidence))

    if not spend(s.layout.return_cost(s.position), 0):
        return result()
    if action == TAKE_SAFE:
        success = spend(s.layout.safe, 7)
    elif action < 3:
        door = action
        if spend(s.layout.a[door], 1 + door):
            opened = bool(world[door])
            s = observe(s, door, opened)
            evidence.append((door, opened))
            if opened:
                success = spend(s.layout.b[door], 7)
            else:
                blocked = 1
    else:
        door = action - 3
        if spend(s.layout.v[door], 4 + door) and spend(s.layout.sensing):
            if not missed:
                opened = bool(world[door])
                s = observe(s, door, opened)
                evidence.append((door, opened))
    return result()


def next_task(snapshot: Snapshot) -> Snapshot:
    p = snapshot.flip
    return replace(snapshot, task=snapshot.task + 1, position=0,
                   remaining=snapshot.layout.budget,
                   belief=tuple(p + (1 - 2 * p) * q for q in snapshot.belief),
                   age=tuple(a + 1 if a >= 0 else -1 for a in snapshot.age),
                   observed=(False,) * 3, attempted=(False,) * 3)


class Environment:
    """One six-task sequence. Policies receive snapshot, never this object."""
    def __init__(self, scenario: Scenario):
        self._scenario = scenario
        self._snapshot = initial_snapshot(scenario.layout, scenario.flip, scenario.miss)
        self._attempts = [0, 0, 0]
        self._done = False
        self._task_time = 0.0
        self._task_inspections = 0
        self._task_blocked = 0

    @property
    def snapshot(self) -> Snapshot:
        return self._snapshot

    def step(self, action: int) -> tuple[Snapshot, float, bool, dict]:
        if self._done:
            raise RuntimeError("Sequence ended; construct a fresh environment")
        s = self._snapshot
        missed = False
        if 3 <= action < 6:
            door = action - 3
            missed = sensing_miss(self._scenario, s.task, door, self._attempts[door])
            self._attempts[door] += 1
            self._task_inspections += 1
        event = physical_step(s, action, self._scenario.worlds[s.task], missed)
        self._task_time += event.dt
        self._task_blocked += event.blocked_attempt
        reward = -event.dt + 100 * event.success - 100 * event.failure
        info = {"dt": event.dt, "evidence": event.evidence, "task_ended": event.ended,
                "success": event.success, "failure": event.failure,
                "blocked_attempt": event.blocked_attempt, "task_record": None}
        self._snapshot = event.snapshot
        if event.ended:
            info["task_record"] = {
                "task": s.task, "success": int(event.success),
                "failure": int(event.failure), "time": self._task_time,
                "inspections": self._task_inspections, "blocked_attempts": self._task_blocked,
                "reward": -self._task_time + (100 if event.success else -100),
                "penalized_cost": self._task_time + 200 * event.failure}
            self._done = s.task == N_TASKS - 1
            if not self._done:
                self._snapshot = next_task(event.snapshot)
                self._attempts = [0, 0, 0]
                self._task_time = 0.0
                self._task_inspections = self._task_blocked = 0
        return self._snapshot, reward, self._done, info
