"""Existing graph physics/features adapted to the generic numerical ports."""
import numpy as np

from ..core.ports import EnvironmentSpec, StepResult
from ..environment import Environment, REGIMES, make_scenario
from ..features import FEATURE_NAMES, encode


def sequence_record(task_rows: list[dict], identity: dict) -> dict:
    if len(task_rows) != 6 or [r["task"] for r in task_rows] != list(range(6)):
        raise ValueError("Expected exactly six ordered tasks")
    result = dict(identity)
    for key in ("success", "failure", "time", "inspections", "blocked_attempts", "reward", "penalized_cost"):
        result[key] = sum(r[key] for r in task_rows)
    result["success_fraction"] = result["success"] / 6
    result["time_per_task"] = result["time"] / 6
    result["inspections_per_task"] = result["inspections"] / 6
    if not np.isclose(result["reward"], 600 - result["time"] - 200 * result["failure"]):
        raise AssertionError("Raw reward identity violated")
    return result


class GraphEnvironment:
    def __init__(self, factory):
        self.factory = factory
        self.env = None
        self.scenario = None
        self.last_action = None
        self.closed = False

    def reset(self):
        self.scenario = self.factory.next_scenario()
        self.env = Environment(self.scenario)
        self.last_action = None
        return encode(self.env.snapshot, self.factory.variant)

    def step(self, action):
        if type(action) is not int or action not in range(self.factory.spec.action_count):
            raise ValueError("Graph action out of bounds")
        task_index = self.env.snapshot.task
        snapshot, reward, done, info = self.env.step(action)
        self.last_action = action
        # Scenario identifiers are output metadata only, never encoded features.
        public = {**info, "scenario_id": self.scenario.scenario_id,
                  "namespace": self.factory.namespace, "regime": self.scenario.regime,
                  "task": task_index}
        return StepResult(encode(snapshot, self.factory.variant), float(reward), bool(done), False, public)

    def scene(self):
        """Use only public Snapshot evidence; never inspect Scenario.worlds."""
        s = self.env.snapshot
        nodes = [{"id": "S", "x": 0.0, "y": 1.0, "label": "Start"},
                 {"id": "G", "x": 4.0, "y": 1.0, "label": "Goal"}]
        edges = []
        for i in range(3):
            state = "unknown" if s.last[i] < 0 else "open" if s.last[i] else "blocked"
            nodes.extend([
                {"id": f"U{i}", "x": 2.0, "y": float(i), "label": f"Door {i}", "state": state},
                {"id": f"V{i}", "x": 1.0, "y": float(i) - 0.2, "label": f"View {i}"}])
            edges.extend([
                {"source": "S", "target": f"U{i}", "label": f"travel {s.layout.a[i]:.1f}",
                 "active": self.last_action == i},
                {"source": f"U{i}", "target": "G", "label": f"cross {s.layout.b[i]:.1f}",
                 "active": self.last_action == i},
                {"source": "S", "target": f"V{i}", "label": f"inspect {s.layout.v[i] + s.layout.sensing:.1f}",
                 "active": self.last_action == i + 3}])
        edges.append({"source": "S", "target": "G", "label": f"safe {s.layout.safe:.1f}",
                      "active": self.last_action == 6})
        position_ids = ("S", "U0", "U1", "U2", "V0", "V1", "V2", "G")
        return {"title": "Public graph state (macro-action)", "nodes": nodes, "edges": edges,
                "agent": position_ids[s.position], "details": {"task": int(s.task),
                    "remaining": float(s.remaining), "beliefs": [float(x) for x in s.belief],
                    "evidence_age": [int(x) for x in s.age]}}

    def close(self):
        self.closed = True


class GraphFactory:
    spec = EnvironmentSpec(len(FEATURE_NAMES), 7)

    def __init__(self, namespace: str, variant: str = "full"):
        if variant not in ("full", "masked"):
            raise ValueError("Graph variant must be full or masked")
        self.namespace = namespace
        self.variant = variant
        self.ordinal = 0
        self.slots = []

    def next_scenario(self):
        ordinal = self.ordinal
        self.ordinal += 1
        regimes = tuple(REGIMES)
        return make_scenario(self.namespace, ordinal, regimes[ordinal % len(regimes)])

    def create(self, env_slot):
        env = GraphEnvironment(self)
        self.slots.append(env)
        return env

    def partial_sequences(self):
        return sum(bool(slot.env.snapshot.task > 0 or
                        slot.env.snapshot.remaining < slot.env.snapshot.layout.budget)
                   for slot in self.slots if slot.env is not None)


def factory(config: dict, *, seed: int, variant: str):
    """Trusted plugin factory(config, *, seed, variant) -> EnvironmentFactory.

    Development graph runs use a pilot-only namespace; final-study construction
    remains exclusively in the reviewed legacy train orchestration.
    """
    if config:
        raise ValueError("Development graph backend accepts no backend config keys")
    return GraphFactory(f"pilot/development/seed-{seed}", variant)
