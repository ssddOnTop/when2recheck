"""Composition root: concrete runtime, trusted plugins, storage and optional UI."""
from dataclasses import asdict
from pathlib import Path

from .adapters.runtime import IsolatedObserver, SystemClock, seed_everything
from .adapters.compute import compute_metadata, create_compute
from .core import CompositeObserver, PPOConfig, learn


def browser_observer_factory(output: Path, port: int = 0):
    """Construct lazily, after the output store has claimed its new directory."""
    def create():
        from .adapters.visualization import BrowserObserver
        observer = BrowserObserver(output=output, port=port)
        print(f"Visualization: {observer.url}", flush=True)
        return observer
    return create


def run_learning(factory, config: PPOConfig, steps: int, *, store, seed: int,
                 device=None, compute=None, clock=None, observer=None, observer_factory=None,
                 metadata=None):
    """Injectable application runner; no mandatory filesystem or wall clock in tests.

    Store is the RunStore protocol. An optional observer is owned and closed here
    on success/error. Replay closes BEFORE store.save seals the artifact manifest.
    observer_factory avoids starting a UI when output claiming fails.
    """
    observed = IsolatedObserver(observer) if observer is not None else None
    try:
        if observer is not None and observer_factory is not None:
            raise ValueError("Supply observer or observer_factory, not both")
        if compute is not None and device is not None:
            raise ValueError("Supply compute or device, not both")
        compute = compute if compute is not None else create_compute(device if device is not None else "cpu")
        metadata = {**(metadata or {}), **compute_metadata(compute)}
        store.start(metadata)
        if observer_factory is not None:
            observer = observer_factory()
        observed = IsolatedObserver(observer) if observer is not None else None
        composite = CompositeObserver(IsolatedObserver(store), *([observed] if observed is not None else []))
        seed_everything(seed)
        try:
            result = learn(factory, config, steps, clock=clock if clock is not None else SystemClock(),
                           observer=composite, compute=compute)
        finally:
            if observed is not None:
                closing, observed = observed, None
                closing.close()
        # Closing the UI above writes its replay; the manifest must include it.
        return store.save(result, metadata)
    finally:
        try:
            if observed is not None:
                observed.close()
        finally:
            store.close()


def learn_development(*, backend: str, backend_config: dict, steps: int, seed: int,
                      variant: str, output: Path, device=None, compute=None, visualize=False,
                      visualization_port=0, clock=None, store=None, observer=None):
    """Generic development command; not an approved frozen final-study run."""
    from .adapters.persistence import DevelopmentStore
    from .plugins import load_backend
    if compute is not None and device is not None:
        raise ValueError("Supply compute or device, not both")
    compute = compute if compute is not None else create_compute(device if device is not None else "cpu")
    seed_everything(seed)
    factory = load_backend(backend, backend_config, seed=seed, variant=variant)
    config = PPOConfig()
    metadata = {"backend": backend, "backend_config": backend_config,
                "spec": {"observation_size": factory.spec.observation_size,
                         "action_count": factory.spec.action_count},
                "ppo": asdict(config), "variant": variant, "training_seed": seed,
                "requested_steps": steps, **compute_metadata(compute),
                "kind": "development_learn", "approved_final_study": False}
    result = run_learning(factory, config, steps, store=store if store is not None else DevelopmentStore(output),
                          seed=seed, compute=compute, clock=clock, observer=observer, metadata=metadata,
                          observer_factory=browser_observer_factory(output, visualization_port) if visualize else None)
    if visualize:
        print(f"Visualization replay: {output / 'visualization.html'}", flush=True)
    return result
