"""Backend-neutral boundaries. Plugins own data, lifecycle and public metadata.

A factory advertises one fixed spec and creates independent, non-autoresetting
slots. reset() returns a finite flat float observation; step() returns the FINAL
observation, even on a time limit. Only the learner calls reset after either end
flag. scene() is optional and must return only public JSON-compatible data.
Observers are output-only; they never receive environments, models or tensors.
"""
from dataclasses import dataclass, field
from typing import Protocol


@dataclass(frozen=True)
class EnvironmentSpec:
    observation_size: int
    action_count: int

    def __post_init__(self):
        for value in (self.observation_size, self.action_count):
            if type(value) is not int or value <= 0:
                raise ValueError("Observation and action dimensions must be positive integers")


@dataclass(frozen=True)
class StepResult:
    observation: object
    reward: float
    terminated: bool = False
    truncated: bool = False
    info: dict = field(default_factory=dict)


class Environment(Protocol):
    def reset(self) -> object: ...
    def step(self, action: int) -> StepResult: ...
    def close(self) -> None: ...


class SceneProvider(Protocol):
    def scene(self) -> dict:
        """Return a fresh public-only scene, never a hidden world or live snapshot."""
        ...


class EnvironmentFactory(Protocol):
    spec: EnvironmentSpec

    def create(self, env_slot: int) -> Environment: ...


class Clock(Protocol):
    def now(self) -> float: ...


class Observer(Protocol):
    wants_scene: bool

    def on_event(self, event: dict) -> None: ...


class NullObserver:
    wants_scene = False

    def on_event(self, event: dict) -> None:
        pass


class RunStore(Observer, Protocol):
    """Application-owned persistence; the core itself never reads/writes storage.

    start must refuse existing outputs, including incomplete runs. save receives
    the completed LearnResult only after learning; close must release resources.
    """
    def start(self, metadata: dict) -> None: ...
    def save(self, result: object, metadata: dict) -> dict: ...
    def close(self) -> None: ...
