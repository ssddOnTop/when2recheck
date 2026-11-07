"""Fair policies accept only immutable public snapshots."""
from __future__ import annotations
from dataclasses import replace
from itertools import product
import math
import numpy as np
from .environment import Snapshot, TAKE_SAFE, physical_step


def reactive(snapshot: Snapshot) -> int:
    l = snapshot.layout
    back = l.return_cost(snapshot.position)
    candidates = [(back + l.a[i] + l.b[i], i) for i in range(3) if snapshot.last[i] != 0]
    candidates.append((back + l.safe, TAKE_SAFE))
    best = min(cost for cost, _ in candidates)
    # Route costs add at most three terms; decimal-equivalent routes tie.
    tolerance = 4 * math.ulp(max(1.0, abs(best)))
    return min(action for cost, action in candidates if cost - best <= tolerance)


def reset(snapshot: Snapshot) -> int:
    # Only this task's permitted observations survive the policy's memory reset.
    last = tuple(x if seen else -1 for x, seen in zip(snapshot.last, snapshot.observed))
    return reactive(replace(snapshot, last=last))


def ttl(snapshot: Snapshot, threshold: int) -> int:
    if threshold not in (1, 2, 3, 6):
        raise ValueError("Frozen TTL candidates are 1, 2, 3, 6")
    route = reactive(snapshot)
    if route < 3 and snapshot.last[route] != -1 and snapshot.age[route] >= threshold:
        if not snapshot.attempted[route]:
            return route + 3
    return route


def world_weights(beliefs: tuple[float, ...]) -> list[tuple[tuple[bool, ...], float]]:
    if len(beliefs) != 3 or any(not 0 <= q <= 1 for q in beliefs):
        raise ValueError("Expected three probabilities")
    return [(world, float(np.prod([q if state else 1 - q for q, state in zip(beliefs, world)])))
            for world in product((False, True), repeat=3)]


def continuation_cost(snapshot: Snapshot, world: tuple[bool, ...]) -> float:
    """Reactive continuation receives only evidence from completed physical events."""
    cost = 0.0
    # At most three newly blocked doors then the safe route. No inspection here.
    for _ in range(4):
        event = physical_step(snapshot, reactive(snapshot), world)
        cost += event.dt
        if event.ended:
            return cost + 200 * event.failure
        snapshot = event.snapshot
    raise AssertionError("Reactive continuation exceeded three doors and safe fallback")


def voi_costs(snapshot: Snapshot) -> np.ndarray:
    scores = np.zeros(7, dtype=np.float64)
    for world, probability in world_weights(snapshot.belief):
        if probability == 0:
            continue
        for action in range(7):
            branches = ((False, 1 - snapshot.miss), (True, snapshot.miss)) if 3 <= action < 6 else ((False, 1.0),)
            for missed, weight in branches:
                if weight == 0:
                    continue
                first = physical_step(snapshot, action, world, missed)
                cost = first.dt + 200 * first.failure
                if not first.ended:
                    cost += continuation_cost(first.snapshot, world)
                scores[action] += probability * weight * cost
    return scores


def voi(snapshot: Snapshot) -> int:
    costs = voi_costs(snapshot)
    # A score sums at most 8 worlds x 2 sensing branches. Group values within
    # 16 ULPs of the minimum to avoid treating summation noise as an advantage.
    # This is about 4.6e-13 at cost 200, not an experiment-scale cost margin.
    best = float(np.min(costs))
    tolerance = 16 * math.ulp(max(1.0, abs(best)))
    tied = [a for a in range(7) if costs[a] - best <= tolerance]
    return min(tied, key=lambda a: (int(3 <= a < 6), a))


def choose(name: str, snapshot: Snapshot, threshold: int | None = None) -> int:
    if name == "reactive":
        return reactive(snapshot)
    if name == "reset":
        return reset(snapshot)
    if name == "voi":
        return voi(snapshot)
    if name == "ttl" and threshold is not None:
        return ttl(snapshot, threshold)
    raise ValueError(f"Unknown or unconfigured policy: {name}")
