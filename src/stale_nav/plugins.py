"""Trusted, in-process backend plugins; loading a plugin executes Python code.

A module:factory callable receives (config: dict, *, seed: int, variant: str) and
returns an EnvironmentFactory with .spec and .create(env_slot). See core.ports
for reset/final-observation/time-limit semantics. Config is backend-specific JSON.
Named installed plugins may use the 'stale_nav.backends' entry-point group.
Add a plugin without editing the learner, runner or CLI; graph is just an alias.
"""
from importlib import import_module
from importlib.metadata import entry_points

from .core.ports import EnvironmentSpec

BUILTINS = {"graph": "stale_nav.adapters.graph:factory"}


def load_backend(backend: str, config: dict | None = None, *, seed=0, variant="full"):
    if config is not None and not isinstance(config, dict):
        raise ValueError("Backend configuration must be a JSON object")
    target = BUILTINS.get(backend, backend)
    if ":" in target:
        module, name = target.split(":", 1)
        if not module or not name or ":" in name:
            raise ValueError("Backend must be module:factory")
        constructor = getattr(import_module(module), name)
    else:
        matches = list(entry_points(group="stale_nav.backends", name=target))
        if len(matches) != 1:
            raise ValueError(f"Unknown or ambiguous backend {backend!r}; use trusted module:factory")
        constructor = matches[0].load()
    if not callable(constructor):
        raise TypeError("Backend plugin factory must be callable")
    factory = constructor(dict(config or {}), seed=seed, variant=variant)
    if not callable(getattr(factory, "create", None)):
        raise TypeError("Backend factory must expose create(env_slot)")
    spec = getattr(factory, "spec", None)
    if spec is None:
        raise TypeError("Backend factory must expose an EnvironmentSpec")
    EnvironmentSpec(spec.observation_size, spec.action_count)
    return factory
