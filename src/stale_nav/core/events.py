"""Detached, JSON-compatible telemetry, never a learning input."""
import math
import random
import numpy as np


def public_copy(value):
    """Copy primitives recursively; reject tensors and arbitrary live objects."""
    if value is None or type(value) in (str, bool, int):
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise ValueError("Event numbers must be finite")
        return value
    if type(value) in (list, tuple):
        return [public_copy(v) for v in value]
    if type(value) is dict and all(type(k) is str for k in value):
        return {k: public_copy(v) for k, v in value.items()}
    raise TypeError("Events accept only public JSON primitives, not live objects")


class CompositeObserver:
    def __init__(self, *children):
        self.children = tuple(children)

    @property
    def wants_scene(self):
        return any(child.wants_scene for child in self.children)

    def on_event(self, event: dict):
        for child in self.children:
            child.on_event(public_copy(event))


class EventSink:
    """Restore RNG (never seed it) and detach every observer payload.

    No callback return value participates in learning. Plugins are trusted code,
    not a security sandbox; observers receive no references to learning state.
    """
    def __init__(self, observer, compute):
        self.observer = observer
        self.compute = compute

    def _call(self, callback):
        state, python_state = np.random.get_state(), random.getstate()
        try:
            with self.compute.preserve_rng():
                return callback()
        finally:
            np.random.set_state(state)
            random.setstate(python_state)

    @property
    def wants_scene(self):
        return bool(self._call(lambda: self.observer.wants_scene))

    def emit(self, event):
        payload = public_copy(event)
        self._call(lambda: self.observer.on_event(payload))
