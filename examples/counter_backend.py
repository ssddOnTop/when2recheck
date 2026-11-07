"""Non-graph plugin: python -m stale_nav.cli learn --backend
examples.counter_backend:factory --backend-config '{"time_limit":5}' --steps 16
--output /tmp/counter-development (run from the repository root).

Five observations, three actions, true termination and independent time limits.
A future simulator can implement these same ports without any core changes.
"""
import numpy as np
from stale_nav.core.ports import EnvironmentSpec, StepResult


class Counter:
    def __init__(self, time_limit):
        self.time_limit = time_limit
        self.closed = False

    def observation(self):
        return np.array([self.position / 3, 1.0, self.elapsed / self.time_limit,
                         self.last_action / 2, 1.0], dtype=np.float32)

    def reset(self):
        self.position = self.elapsed = self.last_action = 0
        self.ended = False
        return self.observation()

    def step(self, action):
        if self.ended or action not in range(3):
            raise ValueError("Reset ended episodes; actions must be 0, 1 or 2")
        self.elapsed += 1
        self.last_action = action
        self.position = max(-3, self.position + action - 1)
        terminated = self.position >= 3
        truncated = self.elapsed >= self.time_limit and not terminated
        self.ended = terminated or truncated
        return StepResult(self.observation(), 1.0 if terminated else -0.01,
                          terminated, truncated, {"elapsed": self.elapsed})

    def close(self):
        self.closed = True


class CounterFactory:
    spec = EnvironmentSpec(5, 3)

    def __init__(self, time_limit):
        self.time_limit = time_limit

    def create(self, env_slot):
        return Counter(self.time_limit)


def factory(config: dict, *, seed: int, variant: str):
    """Return independent non-autoresetting slots; seed/variant are plugin inputs."""
    if set(config) - {"time_limit"}:
        raise ValueError("Only time_limit is supported")
    time_limit = config.get("time_limit", 5)
    if type(time_limit) is not int or time_limit <= 0:
        raise ValueError("time_limit must be a positive integer")
    return CounterFactory(time_limit)
