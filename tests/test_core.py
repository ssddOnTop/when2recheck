"""Architecture and numerical boundary contracts; no experiment outputs."""
import ast
import json
import random
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from stale_nav.adapters.runtime import seed_everything
from stale_nav.core import (ActorCritic, CompositeObserver, EnvironmentSpec, NullObserver,
                            PPOConfig, StepResult, gae, learn)


class Clock:
    def __init__(self):
        self.value = 0

    def now(self):
        self.value += 1
        return float(self.value)


class FakeEnvironment:
    def __init__(self):
        self.closed = False
        self.resets = self.scene_calls = 0
        self.info = {"public": [1, 2]}
        self.actions = []

    def reset(self):
        self.resets += 1
        self.t = 0
        return np.zeros(5, dtype=np.float32)

    def step(self, action):
        assert type(action) is int and 0 <= action < 3
        self.actions.append(action)
        self.t += 1
        obs = np.array([self.t, action, random.random(), np.random.random(), 1], dtype=np.float32)
        return StepResult(obs, float(action) / 2, self.t == 2 and self.resets % 2 == 0,
                          self.t == 2 and self.resets % 2 == 1, self.info)

    def scene(self):
        self.scene_calls += 1
        # Rendering must not alter the learning random stream.
        random.random()
        np.random.random(3)
        torch.rand(3)
        return {"nodes": [{"id": "x", "x": 0.0, "y": 1.0}], "edges": [], "agent": "x"}

    def close(self):
        self.closed = True


class Factory:
    spec = EnvironmentSpec(5, 3)

    def __init__(self, env_type=FakeEnvironment):
        self.slots = []
        self.env_type = env_type

    def create(self, env_slot):
        env = self.env_type()
        self.slots.append(env)
        return env


class Recorder:
    wants_scene = False

    def __init__(self):
        self.events = []

    def on_event(self, event):
        json.dumps(event, allow_nan=False)
        self.events.append(event)


def settings(**overrides):
    return replace(PPOConfig(num_envs=2, rollout_steps=4, epochs=1, minibatch_size=4,
                             hidden_sizes=(8,)), **overrides)


def test_core_has_only_numerical_and_protocol_dependencies():
    import stale_nav.core as core
    root = Path(core.__file__).parent
    allowed = {"dataclasses", "typing", "math", "random", "numpy", "torch"}
    local = {p.stem for p in root.glob("*.py")}
    forbidden_calls = {"open", "print", "input", "save", "load", "read_text", "write_text",
                       "mkdir", "seed", "manual_seed", "set_num_threads", "use_deterministic_algorithms",
                       "socket", "perf_counter", "time", "sleep"}
    for path in root.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                assert all(n.name.split(".")[0] in allowed for n in node.names), path
            if isinstance(node, ast.ImportFrom):
                assert node.level <= 1, path
                assert node.module.split(".")[0] in (local if node.level else allowed), path
            if isinstance(node, ast.Call):
                name = node.func.id if isinstance(node.func, ast.Name) else getattr(node.func, "attr", "")
                assert name not in forbidden_calls, (path, name)
    with pytest.raises(TypeError):
        ActorCritic()
    model = ActorCritic(5, 3)
    assert model(torch.zeros(2, 5))[0].shape == (2, 3)


def test_gae_time_limit_bootstrap_and_no_reset_recurrence():
    rewards = np.array([[1.], [2.], [100.]])
    values = np.array([[.5], [.7], [9.]])
    terminals = np.array([[0.], [0.], [1.]])
    truncations = np.array([[0.], [1.], [0.]])
    next_values = np.array([[.7], [5.], [777.]])
    advantages, targets = gae(rewards, values, terminals, truncations, next_values, .9, .8)
    np.testing.assert_allclose(advantages[:, 0], [5.306, 5.8, 91.])
    np.testing.assert_allclose(targets[:, 0], [5.806, 6.5, 100.])
    # Both flags: termination takes precedence, with no bootstrap.
    terminals[1] = 1
    advantages, _ = gae(rewards, values, terminals, truncations, next_values, .9, .8)
    assert advantages[1, 0] == pytest.approx(1.3)
    # Ordinary final rollout cut bootstraps but has no future recurrence.
    advantages, _ = gae(rewards[:1], values[:1], np.zeros((1, 1)), np.zeros((1, 1)), next_values[:1], .9, .8)
    assert advantages[0, 0] == pytest.approx(1.13)


def test_learner_uses_final_observation_not_reset_for_truncation(monkeypatch):
    import stale_nav.core.learner as learner
    class Limit:
        def reset(self):
            return [100., 0.]
        def step(self, action):
            return StepResult([7., 0.], 1., truncated=True)
        def close(self):
            pass
    class LimitFactory:
        spec = EnvironmentSpec(2, 2)
        def create(self, slot):
            return Limit()
    captured = []
    real_gae = learner.gae
    def capture(*args):
        captured.append(args[4].copy())
        return real_gae(*args)
    monkeypatch.setattr(learner, "gae", capture)
    seed_everything(0)
    model = ActorCritic(2, 2, ())
    with torch.no_grad():
        model.critic[0].weight[:] = torch.tensor([[1., 0.]])
        model.critic[0].bias.zero_()
    learn(LimitFactory(), settings(num_envs=1), 3, clock=Clock(), model=model)
    np.testing.assert_array_equal(captured[0], [[7.], [7.], [7.]])


def test_non_graph_training_events_partial_rollout_and_cleanup():
    seed_everything(3)
    factory, observer = Factory(), Recorder()
    result = learn(factory, settings(), 10, clock=Clock(), observer=observer)
    assert result.metrics["env_steps"] == 10
    assert result.metrics["updates"] == 2
    assert observer.events[0]["type"] == "started"
    assert observer.events[-1]["type"] == "finished"
    transitions = [e for e in observer.events if e["type"] == "transition"]
    assert [e["step"] for e in transitions] == list(range(1, 11))
    assert all(len(e["observation"]) == 5 and "scene" not in e for e in transitions)
    assert any(e["truncated"] for e in transitions) and any(e["terminated"] for e in transitions)
    assert all(e.closed and e.resets == 3 and not e.scene_calls for e in factory.slots)
    assert result.model.spec == EnvironmentSpec(5, 3)


def test_observers_cannot_change_training_rng_info_or_observations():
    class Noisy:
        @property
        def wants_scene(self):
            torch.rand(5)
            random.random()
            return True
        def on_event(self, event):
            torch.rand(11)
            np.random.random(9)
            random.random()
            if event["type"] == "transition":
                event["observation"][0] = 999
                event["info"]["public"].append(999)
                event["scene"]["nodes"][0]["x"] = 999
    def run(observer):
        seed_everything(12)
        factory = Factory()
        result = learn(factory, settings(), 16, clock=Clock(), observer=observer)
        return factory, result.model.state_dict(), torch.get_rng_state()
    ordinary, expected, expected_rng = run(None)
    captured = Recorder()
    composite = CompositeObserver(Noisy(), captured)
    visual, actual, actual_rng = run(composite)
    assert composite.wants_scene and not CompositeObserver(NullObserver()).wants_scene
    assert all(torch.equal(expected[k], actual[k]) for k in expected)
    assert torch.equal(expected_rng, actual_rng)
    assert [s.actions for s in ordinary.slots] == [s.actions for s in visual.slots]
    assert all(s.info == {"public": [1, 2]} and s.scene_calls == 8 for s in visual.slots)
    for event in captured.events:
        if event["type"] == "transition":
            assert event["info"]["public"] == [1, 2]
            assert event["observation"][0] < 999
            assert event["scene"]["nodes"][0]["x"] == 0.


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 1e100])
def test_nonfinite_or_overflow_reward_rejected_and_closed(bad):
    class Bad(FakeEnvironment):
        def step(self, action):
            return StepResult(np.zeros(5), bad)
    factory = Factory(Bad)
    with pytest.raises(ValueError, match="reward"):
        learn(factory, settings(), 8, clock=Clock())
    assert all(e.closed for e in factory.slots)


def test_bad_observation_and_action_bounds_close_early():
    class Bad(FakeEnvironment):
        def reset(self):
            return [1, 2]
    factory = Factory(Bad)
    with pytest.raises(ValueError, match="observation"):
        learn(factory, settings(), 8, clock=Clock())
    assert factory.slots[0].closed
    class BadModel(ActorCritic):
        def sample(self, x):
            return torch.full((len(x),), 3), torch.zeros(len(x)), torch.zeros(len(x))
    factory = Factory()
    with pytest.raises(ValueError, match="action bounds"):
        learn(factory, settings(), 8, clock=Clock(), model=BadModel(5, 3))
    assert all(e.closed and not e.actions for e in factory.slots)


def test_creation_and_observer_errors_close_created_slots():
    class BadFactory(Factory):
        def create(self, slot):
            if slot == 1:
                raise RuntimeError("creation failed")
            return super().create(slot)
    factory = BadFactory()
    with pytest.raises(RuntimeError, match="creation failed"):
        learn(factory, settings(), 8, clock=Clock())
    assert factory.slots[0].closed
    class BadObserver(Recorder):
        def on_event(self, event):
            raise RuntimeError("observer failed")
    factory = Factory()
    with pytest.raises(RuntimeError, match="observer failed"):
        learn(factory, settings(), 8, clock=Clock(), observer=BadObserver())
    assert all(e.closed for e in factory.slots)


def test_environment_cleanup_attempts_all_slots_even_when_close_raises():
    class CloseError(FakeEnvironment):
        def close(self):
            self.closed = True
            raise RuntimeError("close failure")
    factory = Factory(CloseError)
    with pytest.raises(ExceptionGroup, match="Environment cleanup failed") as caught:
        learn(factory, settings(), 8, clock=Clock())
    assert len(caught.value.exceptions) == 2
    assert all(s.closed for s in factory.slots)


@pytest.mark.parametrize("dimensions", [(0, 3), (5, -1), (True, 3), (2.5, 2)])
def test_invalid_dimensions(dimensions):
    with pytest.raises(ValueError, match="dimensions"):
        EnvironmentSpec(*dimensions)
