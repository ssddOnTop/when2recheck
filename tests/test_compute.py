"""Compute injection, explicit selection, early validation and CUDA integration."""
import ast
from contextlib import contextmanager
import json
from pathlib import Path
import random

import numpy as np
import pytest
import torch

from stale_nav.adapters.compute import CPUBackend, CUDABackend, create_compute
from stale_nav.adapters.persistence import DevelopmentStore
from stale_nav.adapters.runtime import IsolatedObserver, seed_everything
from stale_nav.application import learn_development, run_learning
from stale_nav.cli import main, parser
from stale_nav.core import learn
from stale_nav.experiments import load_config
from stale_nav.provenance import ROOT, json_hash, source_manifest
from test_core import Clock, Factory, Recorder, settings
from test_plugins import MemoryStore


def arguments(command, output):
    result = [command, "--output", str(output)]
    if command == "train":
        result += ["--review", str(output.parent / "review.json")]
    elif command == "evaluate":
        result += ["--split", "pilot"]
    return result


@pytest.mark.parametrize("command", ["learn", "smoke", "train", "evaluate"])
@pytest.mark.parametrize("flags", [
    ["--cpu", "--gpu"], ["--gpu", "--cpu"],
    ["--cpu", "--device", "cpu"], ["--device", "cpu", "--cpu"],
    ["--gpu", "--device", "cuda"], ["--device", "cuda:0", "--gpu"],
])
def test_flags_are_mutually_exclusive(command, flags, tmp_path):
    with pytest.raises(SystemExit) as exc:
        parser().parse_args(arguments(command, tmp_path / "out") + flags)
    assert exc.value.code == 2
    assert not (tmp_path / "out").exists()


@pytest.mark.parametrize("command", ["learn", "smoke", "train", "evaluate"])
def test_flags_select_requested_device(command, tmp_path):
    base = arguments(command, tmp_path / "out")
    # None distinguishes absent selection from an explicit CPU override.
    assert parser().parse_args(base).device is None
    for flags, device in [(["--cpu"], "cpu"), (["--gpu"], "cuda"),
                          (["--device", "cuda:1"], "cuda:1")]:
        assert parser().parse_args(base + flags).device == device


def test_generic_default_is_cpu(tmp_path, monkeypatch):
    import stale_nav.application as application
    calls = []
    monkeypatch.setattr(application, "learn_development", lambda **kw: calls.append(kw) or {})
    main(["learn", "--output", str(tmp_path / "out")])
    assert calls[0]["device"] == "cpu"


@pytest.mark.parametrize("command", ["smoke", "train", "evaluate"])
@pytest.mark.parametrize("flags,expected", [([], "cuda"), (["--cpu"], "cpu"),
    (["--gpu"], "cuda"), (["--device", "cuda:1"], "cuda:1")])
def test_study_override_and_absent_flag_preserve_config(command, flags, expected, tmp_path, monkeypatch):
    import stale_nav.ppo as ppo
    import stale_nav.experiments as experiments
    config = json.loads((ROOT / "configs/frozen.json").read_text())
    config["device"] = "cuda"
    path = tmp_path / "config.json"
    original = json.dumps(config)
    path.write_text(original)
    calls = []
    def capture(effective, *args, **kwargs):
        calls.append(effective)
        return {}
    monkeypatch.setattr(ppo, "train", capture)
    monkeypatch.setattr(experiments, "evaluate", capture)
    main(arguments(command, tmp_path / "out") + ["--config", str(path)] + flags)
    assert calls[0]["device"] == expected
    assert path.read_text() == original
    assert not (tmp_path / "out").exists()


def test_override_must_match_hash_bound_review_before_training(tmp_path, monkeypatch):
    import stale_nav.application as application
    config = json.loads((ROOT / "configs/frozen.json").read_text())
    config["device"] = "cuda"
    path = tmp_path / "config.json"
    original = json.dumps(config)
    path.write_text(original)
    review = {"verdict": "PASS_FOR_TRAINING", "source_sha256": source_manifest()["source_sha256"],
              "config_sha256": json_hash(config), "remaining_blockers": []}
    review_path = tmp_path / "review.json"
    review_path.write_text(json.dumps(review))
    calls = []
    monkeypatch.setattr(application, "run_learning", lambda *a, **kw: calls.append(kw) or {})
    args = arguments("train", tmp_path / "out") + ["--config", str(path), "--cpu"]
    with pytest.raises(ValueError, match="final config"):
        main(args)
    assert not calls and not (tmp_path / "out").exists()
    effective = {**config, "device": "cpu"}
    review["config_sha256"] = json_hash(effective)
    review_path.write_text(json.dumps(review))
    main(args)
    assert len(calls) == 1 and isinstance(calls[0]["compute"], CPUBackend)
    assert calls[0]["store"].config == effective
    assert path.read_text() == original


@pytest.mark.parametrize("command", ["learn", "smoke", "train", "evaluate"])
@pytest.mark.parametrize("flags,message", [(["--gpu"], "CUDA GPU requested but unavailable"),
    (["--device", "not-a-device"], "Invalid compute device"),
    (["--device", "mps"], "Unsupported compute device")])
def test_invalid_device_fails_before_output_and_ui(command, flags, message, tmp_path, monkeypatch):
    import stale_nav.application as application
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    def forbidden(*args, **kwargs):
        pytest.fail("UI must not start for an invalid compute selection")
    monkeypatch.setattr(application, "browser_observer_factory", forbidden)
    output = tmp_path / "out"
    args = arguments(command, output) + flags
    if command != "learn":
        args += ["--config", str(ROOT / "configs/frozen.json")]
    if command != "evaluate":
        args += ["--visualize"]
    with pytest.raises(ValueError, match=message):
        main(args)
    assert not output.exists()


def test_direct_runner_validates_before_store_or_ui(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    store = MemoryStore()
    with pytest.raises(ValueError, match="unavailable"):
        run_learning(Factory(), settings(), 8, store=store, seed=0, device="cuda",
                     observer_factory=lambda: pytest.fail("UI started"))
    assert not hasattr(store, "metadata") and store.saved is None


@pytest.mark.parametrize("device", ["cuda:255", "cuda:256", "cuda:257", "cuda:9999"])
def test_wrapped_cuda_indices_rejected_before_availability(device, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("index must be checked before availability"))
    with pytest.raises(ValueError, match="Invalid compute device"):
        create_compute(device)


def test_owned_observer_closed_on_compute_validation_failure(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    closed = []
    class OwnedObserver(Recorder):
        def close(self):
            closed.append(True)
    store = MemoryStore()
    with pytest.raises(ValueError, match="unavailable"):
        run_learning(Factory(), settings(), 8, store=store, seed=0, device="cuda",
                     observer=OwnedObserver())
    assert closed == [True] and store.closed
    assert not hasattr(store, "metadata") and store.saved is None


def test_cpu_backend_never_probes_gpu_and_learning_works_without_cuda(monkeypatch):
    # Let PyTorch register its lazy optimizer machinery before replacing API
    # objects; Dynamo's registration indexes functions by identity.
    torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))])
    def forbidden(*args, **kwargs):
        pytest.fail("CPU backend must not query or synchronize CUDA hardware")
    for name in ("is_available", "device_count", "current_device", "synchronize", "get_rng_state"):
        monkeypatch.setattr(torch.cuda, name, forbidden)
    compute = create_compute("cpu")
    with compute.preserve_rng():
        model = compute.place_model(torch.nn.Linear(5, 3))
        values = compute.tensor(np.zeros(5, dtype=np.float32))
        assert compute.to_numpy(model(values)).shape == (3,)
        assert compute.randperm(4).device.type == "cpu"
        compute.synchronize()
    # CUDA-enabled PyTorch 2.8's Adam itself probes availability even for CPU
    # parameters. Check our adapter separately above, then prove the full CPU
    # learner works when CUDA is unavailable without forbidding framework code.
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    result = learn(Factory(), settings(), 8, compute=compute, clock=Clock())
    assert result.metrics["compute_device"] == "cpu"
    assert next(result.model.parameters()).device == torch.device("cpu")


def test_cuda_index_and_initialization_validated(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(torch.cuda, "current_device", lambda: 1)
    allocations, synchronized = [], []
    monkeypatch.setattr(torch, "empty", lambda *a, **kw: allocations.append(kw["device"]))
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: synchronized.append(device))
    compute = create_compute("cuda")
    assert isinstance(compute, CUDABackend) and compute.device == torch.device("cuda:1")
    assert allocations == synchronized == [torch.device("cuda:1")]
    assert create_compute("cuda:0").device == torch.device("cuda:0")
    with pytest.raises(ValueError, match="index 2.*invalid"):
        create_compute("cuda:2")
    for invalid in ("cuda:-1", "cuda:255", "cuda:256", "cuda:257", "cuda:999999999"):
        with pytest.raises(ValueError, match="Invalid compute device"):
            create_compute(invalid)
    def broken(*args, **kwargs):
        raise RuntimeError("driver allocation failure")
    monkeypatch.setattr(torch, "empty", broken)
    with pytest.raises(ValueError, match="could not be initialized.*driver allocation failure"):
        create_compute("cuda:0")


class TrackingCompute:
    """Independent structural implementation; no inheritance or device dispatch."""
    device = torch.device("cpu")
    name = "test-compute"

    def __init__(self):
        self.calls = []

    def place_model(self, model):
        self.calls.append("place_model")
        return model.to(self.device)

    def tensor(self, data, *, dtype=None):
        self.calls.append("tensor")
        return torch.as_tensor(data, dtype=dtype, device=self.device)

    def to_numpy(self, tensor):
        self.calls.append("to_numpy")
        return tensor.detach().cpu().numpy()

    def randperm(self, size):
        self.calls.append("randperm")
        return torch.randperm(size, device=self.device)

    def synchronize(self):
        self.calls.append("synchronize")

    @contextmanager
    def preserve_rng(self):
        self.calls.append("rng-enter")
        with torch.random.fork_rng(devices=[]):
            yield
        self.calls.append("rng-exit")


def test_injected_compute_owns_placement_conversion_rng_and_timing(monkeypatch):
    import stale_nav.application as application
    monkeypatch.setattr(application, "create_compute", lambda *a: pytest.fail("injection ignored"))
    compute = TrackingCompute()
    class SynchronizedClock(Clock):
        def now(self):
            assert compute.calls[-1] == "synchronize"
            compute.calls.append("clock")
            return super().now()
    store = MemoryStore()
    result = run_learning(Factory(), settings(), 10, compute=compute, clock=SynchronizedClock(),
                          store=store, seed=12, metadata={"compute_device": "false-claim"})
    assert result["compute_backend"] == "TrackingCompute"
    assert result["compute_backend_name"] == "test-compute"
    assert store.metadata["compute_device"] == "cpu"
    assert store.metadata["compute_backend"] == "TrackingCompute"
    assert compute.calls.count("place_model") == 1
    assert compute.calls.count("randperm") == 2
    assert compute.calls.count("tensor") > 10
    assert compute.calls.count("to_numpy") >= 15
    assert compute.calls.count("rng-enter") == compute.calls.count("rng-exit") > 10
    assert compute.calls.count("synchronize") == 5  # start, two updates, finish clock, completion
    assert compute.calls[-1] == "synchronize"


def test_core_contains_no_hardware_policy_or_concrete_adapters():
    import stale_nav.core as core
    for path in Path(core.__file__).parent.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                assert node.attr not in {"cuda", "mps", "cpu", "fork_rng", "is_available"}, path
                if isinstance(node.value, ast.Name) and node.value.id == "torch":
                    assert node.attr not in {"as_tensor", "randperm", "empty", "zeros", "ones"}, path
            if isinstance(node, ast.ImportFrom):
                assert "adapters" not in (node.module or ""), path
            if isinstance(node, ast.Constant) and isinstance(node.value, str):
                assert node.value not in {"cuda", "cpu", "mps"}, path


def test_development_effective_metadata_and_weights(tmp_path):
    output = tmp_path / "cpu"
    summary = learn_development(backend="graph", backend_config={}, steps=4, seed=0,
                                variant="full", output=output)
    checkpoint = torch.load(output / "checkpoint.pt", weights_only=True)
    provenance = json.loads((output / "provenance.json").read_text())
    for data in (summary, checkpoint, provenance):
        assert data["device"] == data["compute_device"] == "cpu"
        assert data["compute_backend"] == "CPUBackend" and data["compute_backend_name"] == "cpu"
    assert all(value.device.type == "cpu" for value in checkpoint["model_state"].values())


def test_indexed_config_is_accepted_without_probing_hardware(tmp_path, monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("config parsing probed hardware"))
    config = json.loads((ROOT / "configs/frozen.json").read_text())
    config["device"] = "cuda:1"
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    assert load_config(path)["device"] == "cuda:1"


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware unavailable")
@pytest.mark.parametrize("backend", ["graph", "examples.counter_backend:factory"])
def test_cuda_training_checkpoint_device_and_observer_parity(tmp_path, monkeypatch, backend):
    from stale_nav.plugins import load_backend
    monkeypatch.syspath_prepend(str(ROOT))
    compute = create_compute("cuda:0")
    class NoisyObserver(Recorder):
        @property
        def wants_scene(self):
            torch.rand(7, device=compute.device)
            return True
        def on_event(self, event):
            torch.rand(11, device=compute.device)
            torch.rand(11)
            random.random()
            np.random.random(11)
        def close(self):
            # The outer adapter also isolates CUDA state during lifecycle hooks.
            torch.rand(13, device=compute.device)
    def run(name, observer):
        seed_everything(14)
        factory = load_backend(backend, {}, seed=14, variant="full")
        output = tmp_path / name
        result = run_learning(factory, settings(), 16, compute=compute, clock=Clock(),
                              store=DevelopmentStore(output), seed=14, observer=observer)
        checkpoint = torch.load(output / "checkpoint.pt", weights_only=True)
        assert result["compute_device"] == str(compute.device)
        assert checkpoint["compute_backend"] == "CUDABackend"
        assert all(t.device == compute.device for t in checkpoint["model_state"].values())
        assert checkpoint["model_state"]["actor.0.weight"].shape[1] == factory.spec.observation_size
        if backend != "graph":
            assert (factory.spec.observation_size, factory.spec.action_count) == (5, 3)
        assert all(s["exp_avg"].device == compute.device
                   for s in checkpoint["optimizer_state"]["state"].values())
        events = [json.loads(line) for line in (output / "events.jsonl").read_text().splitlines()]
        return checkpoint, events, torch.cuda.get_rng_state(compute.device)
    plain, plain_events, plain_rng = run("plain", None)
    observed, observed_events, observed_rng = run("observed", NoisyObserver())
    assert all(torch.equal(value, observed["model_state"][key]) for key, value in plain["model_state"].items())
    assert plain_events == observed_events
    assert torch.equal(plain_rng, observed_rng)


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA hardware unavailable")
def test_outer_observer_preserves_all_initialized_cuda_rngs():
    for index in range(torch.cuda.device_count()):
        torch.rand(1, device=f"cuda:{index}")
    states = torch.cuda.get_rng_state_all()
    observer = IsolatedObserver(Recorder())
    def noisy():
        for index in range(torch.cuda.device_count()):
            torch.rand(7, device=f"cuda:{index}")
    observer._call(noisy)
    assert all(torch.equal(a, b) for a, b in zip(states, torch.cuda.get_rng_state_all()))
