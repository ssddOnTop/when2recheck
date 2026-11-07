"""One-task physical adapter, public comparators, and isolated scenario streams.

The effective feature contrast is cost/structural information. Boundary age and
 task index never vary in this one-task study. Hidden worlds, family labels,
seed identities and the reachable-door permutation are not policy inputs.
"""
from __future__ import annotations
from dataclasses import asdict, dataclass, replace
from fractions import Fraction
from itertools import product
import math
from pathlib import Path
import numpy as np

from stale_nav.environment import Layout, Snapshot, initial_snapshot, physical_step, keyed_seed
from stale_nav.features import FEATURE_NAMES, FEATURE_SCALES, MASKED_COLUMNS, MASK_INDICES
from stale_nav.policies import reactive, voi, world_weights
from . import control as ctl

STRATA = tuple((family, scale, miss) for family in ("direct", "inspection", "safe")
               for scale in ("4/5", "6/5") for miss in ("0", "1/4"))
NAMESPACES = {"train": ctl.PREFIX + "/train", "validation": ctl.PREFIX + "/validation",
              "heldout": ctl.PREFIX + "/heldout-sealed", "pilot": ctl.PREFIX + "/pilot"}
BASELINES = ("ALWAYS_SAFE", "DIRECT", "INSPECT_REACTIVE", "INSPECT_BEST_MISS", "reactive", "voi")


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    namespace: str
    stratum: str
    layout: Layout
    world: tuple[bool, bool, bool]
    miss: float

    def __post_init__(self):
        if len(self.world) != 3 or not 0 <= self.miss <= 1:
            raise ValueError("Invalid one-task scenario")


def _rng(*key):
    return np.random.Generator(np.random.PCG64(keyed_seed(*key)))


def _sample(namespace: str, master: int, seed: int | str, stratum: int, index: int,
            grant: ctl.Grant | None = None, split: str | None = None,
            release: dict | None = None) -> Scenario:
    # The sampling primitive itself also rejects reserved namespaces without a
    # matching capability; callers cannot accidentally bypass Factory's gate.
    if not namespace.startswith(ctl.UNIT_NAMESPACE + "/"):
        if grant is None or split not in NAMESPACES or NAMESPACES[split] != namespace:
            raise ctl.GateError("Reserved RNG namespace requires an execution grant")
        grant.check("generate_" + split)
        if split == "heldout":
            if release is None:
                raise ctl.GateError("Held-out RNG remains sealed without a verified release")
            ctl.match_bindings(release, grant.bindings)
            if ctl.sha(Path(release["release_path"])) != release["release_sha256"]:
                raise ctl.GateError("Held-out release changed during generation")
    if stratum not in range(12) or not isinstance(index, int) or index < 0:
        raise ValueError("Invalid stratum or episode index")
    family, scale_text, miss_text = STRATA[stratum]
    key = (namespace, master, seed, family, scale_text, miss_text, index)
    scale = float(Fraction(scale_text))
    bounds = ctl.plan()["prospective_distribution"]["families"][family]
    layout_rng = _rng(*key, "layout")
    parameters = {name: float(layout_rng.uniform(float(Fraction(bounds[name][0])), float(Fraction(bounds[name][1])))) * scale
                  for name in ("a", "b", "v", "sensing", "safe")}
    reachable = int(_rng(*key, "permutation").integers(0, 3))
    def edges(name):
        return tuple(parameters[name] if i == reachable else 80 * scale for i in range(3))
    layout = Layout(edges("a"), edges("b"), edges("v"), parameters["sensing"], parameters["safe"], 60 * scale)
    world = tuple(bool(x) for x in (_rng(*key, "world").random(3) < .5))
    sid = f"{namespace}/seed-{seed}/{family}/scale-{scale_text.replace('/', '_')}/miss-{miss_text.replace('/', '_')}/{index:08d}"
    return Scenario(sid, namespace, f"{family}|{scale_text}|{miss_text}", layout, world, float(Fraction(miss_text)))


def unit_scenario(index: int, stratum: int = 0, label: str = "acceptance") -> Scenario:
    """Ungated unit-only draws. Never uses prospective master seeds/namespaces."""
    if not label or "/" in label or ".." in label:
        raise ValueError("Unit label must be one nonempty path component")
    return _sample(ctl.UNIT_NAMESPACE + "/" + label, 719, "unit", stratum, index)


class ScenarioFactory:
    def __init__(self, split: str, seed: int | None, review: Path | None, audit_dir: Path,
                 release: Path | None = None):
        if split not in NAMESPACES:
            raise ctl.GateError("Unknown prospective split")
        self.grant = ctl.authorize(review, "generate_" + split)
        self.split, self.seed = split, seed
        seeds = ctl.plan()["prospective_seeds_and_seal"]
        if split in ("train", "pilot"):
            if seed not in ctl.config()["training_seeds"] or (split == "pilot" and seed != 1000):
                raise ctl.GateError("Seed is not in the declared prospective split")
            self.master = seeds["training_master_seed"]
        else:
            if seed is not None:
                raise ctl.GateError("Paired evaluation splits do not accept model-specific seeds")
            self.master = seeds["validation_master_seed"] if split == "validation" else seeds["heldout_reservation"]["master_seed"]
        self.release = ctl.heldout_release(release, self.grant) if split == "heldout" else None
        self.audit_dir = ctl.checked_output(audit_dir)
        self.started = False
        self.count = 0

    def generate(self, index: int, stratum: int) -> Scenario:
        self.grant.check("generate_" + self.split)
        if self.split in ("validation", "heldout"):
            limit = 64 if self.split == "validation" else 128
            if index not in range(limit):
                raise ctl.GateError("Evaluation index is outside the frozen split")
        if not self.started:
            # Receipt is created before the first random draw. A failed heldout
            # generation keeps its receipt; it needs a separately reviewed repair.
            self.audit_dir.mkdir(parents=True, exist_ok=True)
            metadata = {"split": self.split, "namespace": NAMESPACES[self.split],
                        "first_requested_index": index, "first_requested_stratum": stratum,
                        "review_sha256": self.grant.review_sha256,
                        "implementation_sha256": self.grant.bindings["implementation_sha256"],
                        "release": self.release}
            if self.split == "heldout":
                receipts = ctl.EVIDENCE / "heldout-receipts"
                receipts.mkdir(parents=True, exist_ok=True)
                receipt = receipts / (self.grant.bindings["reservation_sha256"] + ".json")
                with receipt.open("x") as f:
                    import json
                    json.dump({**metadata, "dataset_directory": str(self.audit_dir.resolve())}, f, sort_keys=True)
            with (self.audit_dir / "generation-start.json").open("x") as f:
                import json
                json.dump(metadata, f, sort_keys=True)
            self.started = True
        scenario = _sample(NAMESPACES[self.split], self.master,
                           self.seed if self.seed is not None else "evaluation", stratum, index,
                           self.grant, self.split, self.release)
        self.count += 1
        return scenario


def sensing_miss(scenario: Scenario, door: int, attempt: int) -> bool:
    return bool(_rng(scenario.scenario_id, "sensing", 0, door, attempt).random() < scenario.miss)


class OneTaskEnv:
    """One task, one termination. No original six-task reset or belief advance."""
    def __init__(self, scenario: Scenario, grant: ctl.Grant | None = None, release: dict | None = None):
        if not scenario.namespace.startswith(ctl.UNIT_NAMESPACE + "/"):
            split = next((k for k, v in NAMESPACES.items() if scenario.namespace == v), None)
            if grant is None or split is None:
                raise ctl.GateError("Prospective environments require an execution grant")
            grant.check(("evaluate_" + split) if split in ("validation", "heldout") else split)
            if split == "heldout":
                if release is None:
                    raise ctl.GateError("Held-out environment remains sealed")
                ctl.match_bindings(release, grant.bindings)
                if ctl.sha(Path(release["release_path"])) != release["release_sha256"]:
                    raise ctl.GateError("Held-out environment release changed")
        self._scenario = scenario
        self._snapshot = initial_snapshot(scenario.layout, 0.0, scenario.miss)
        self._done = False
        self._attempts = [0, 0, 0]
        self.time = 0.0
        self.inspections = self.blocked = self.decisions = 0

    @property
    def snapshot(self):
        return self._snapshot

    def step(self, action: int):
        if self._done:
            raise RuntimeError("One-task episode already ended; create a fresh environment")
        if action not in range(7):
            raise ValueError("Seven fixed legal actions")
        missed = False
        if 3 <= action < 6:
            door = action - 3
            missed = sensing_miss(self._scenario, door, self._attempts[door])
            self._attempts[door] += 1
            self.inspections += 1
        event = physical_step(self._snapshot, action, self._scenario.world, missed)
        self._snapshot, self._done = event.snapshot, event.ended
        self.time += event.dt
        self.blocked += event.blocked_attempt
        self.decisions += 1
        reward = -event.dt + 100 * event.success - 100 * event.failure
        info = {"dt": event.dt, "evidence": event.evidence, "success": bool(event.success),
                "failure": bool(event.failure), "blocked_attempt": event.blocked_attempt}
        if self._done:
            info["episode"] = {"time": self.time, "failure": int(event.failure), "success": int(event.success),
                               "reward": 100 - self.time - 200 * event.failure,
                               "penalized_cost": self.time + 200 * event.failure,
                               "inspections": self.inspections, "blocked_attempts": self.blocked,
                               "decisions": self.decisions}
        return self._snapshot, float(reward), self._done, info


def feature_spec():
    scales = dict(FEATURE_SCALES)  # Never mutate the original normalizer.
    scales.update(remaining_budget=72., initial_task_budget=72., task_index=1., sensing_cost=2.64, safe_cost=31.2)
    for i in range(3):
        scales[f"boundary_age_{i}"] = 1.
        for prefix in ("approach_cost", "crossing_cost", "viewpoint_cost"):
            scales[f"{prefix}_{i}"] = 96.
    return {"names": list(FEATURE_NAMES), "scales": scales, "masked_columns": list(MASKED_COLUMNS),
            "interpretation": ctl.config()["ablation_interpretation"]}


# Private immutable tuple, independent from the original module's ndarray.
SCALES = tuple(feature_spec()["scales"][name] for name in FEATURE_NAMES)


def encode(s: Snapshot, variant: str = "full") -> np.ndarray:
    if variant not in ("full", "masked"):
        raise ValueError("Unknown variant")
    if s.task != 0 or any(age not in (-1, 0) for age in s.age):
        raise ValueError("Follow-up features require one-task invariant age/index")
    values = [float(s.position == i) for i in range(8)] + [s.remaining, s.task, s.flip, s.miss]
    values += list(s.belief) + [0 if x == -1 else 2 * x - 1 for x in s.last]
    values += [float(x != -1) for x in s.last] + [max(0, a) for a in s.age]
    values += list(s.observed) + list(s.attempted)
    values += list(s.layout.a) + list(s.layout.b) + list(s.layout.v)
    values += [s.layout.sensing, s.layout.safe, s.layout.budget]
    array = np.asarray(values, dtype=np.float32) / np.asarray(SCALES, dtype=np.float32)
    if variant == "masked":
        array[list(MASK_INDICES)] = 0
    return array


def public_target(s: Snapshot) -> int:
    costs = [s.layout.a[i] + s.layout.b[i] for i in range(3)]
    best = min(costs)
    tolerance = 4 * math.ulp(max(1., abs(best)))
    return min(i for i, cost in enumerate(costs) if cost - best <= tolerance)


def navigation_cost(s: Snapshot, action: int) -> float:
    """Public-belief expectation. Reuse physical events for equality/budget rules."""
    terms = []
    for world, weight in world_weights(s.belief):
        event = physical_step(s, action, world)
        cost = event.dt + 200 * event.failure
        if not event.ended:
            # Navigation can continue only after a physically observed blockage.
            rest = physical_step(event.snapshot, 6, world)
            cost += rest.dt + 200 * rest.failure
        terms.append(weight * cost)
    return math.fsum(terms)


def choose(name: str, snapshot: Snapshot) -> int:
    if name == "ALWAYS_SAFE":
        return 6
    if name == "reactive":
        return reactive(snapshot)
    if name == "voi":
        return voi(snapshot)
    target = public_target(snapshot)
    if name == "DIRECT":
        return 6 if snapshot.last[target] == 0 else target
    if name not in ("INSPECT_REACTIVE", "INSPECT_BEST_MISS"):
        raise ValueError("Unknown public policy")
    if not snapshot.attempted[target]:
        return target + 3
    if snapshot.last[target] == 0:
        return 6
    if snapshot.last[target] == 1 or name == "INSPECT_REACTIVE":
        return target
    direct, safe = navigation_cost(snapshot, target), navigation_cost(snapshot, 6)
    tolerance = 16 * math.ulp(max(1., min(direct, safe)))
    return target if direct - safe <= tolerance else 6


def public_record(snapshot: Snapshot):
    return asdict(snapshot)


def scenario_record(s: Scenario):
    return asdict(s)


def scenario_from_record(row: dict):
    layout = row["layout"]
    return Scenario(row["scenario_id"], row["namespace"], row["stratum"],
                    Layout(tuple(layout["a"]), tuple(layout["b"]), tuple(layout["v"]),
                           layout["sensing"], layout["safe"], layout["budget"]),
                    tuple(row["world"]), row["miss"])
