"""Concrete process clock and deterministic process configuration."""
import os
import random
import time

# Configure before any CUDA context is created, including legacy evaluation.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch


class SystemClock:
    def now(self) -> float:
        return time.perf_counter()


def seed_everything(seed: int):
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        raise ValueError("Deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


class IsolatedObserver:
    """Protect process RNG for external observer callbacks and lifecycle hooks."""
    def __init__(self, observer):
        self.observer = observer

    def _call(self, callback):
        state, numpy_state = random.getstate(), np.random.get_state()
        devices = list(range(torch.cuda.device_count())) if torch.cuda.is_initialized() else []
        try:
            with torch.random.fork_rng(devices=devices):
                return callback()
        finally:
            random.setstate(state)
            np.random.set_state(numpy_state)

    @property
    def wants_scene(self):
        return self._call(lambda: self.observer.wants_scene)

    def on_event(self, event):
        self._call(lambda: self.observer.on_event(event))

    def close(self):
        close = getattr(self.observer, "close", None)
        if close is not None:
            self._call(close)
