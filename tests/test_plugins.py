"""Trusted backend loading and injectable development/application integration."""
import json
import sys
import types

import pytest
import torch

from stale_nav.application import learn_development, run_learning
from stale_nav.cli import main, parser
from stale_nav.core import EnvironmentSpec
from stale_nav.plugins import load_backend
from stale_nav.ppo import load_policy
from stale_nav.provenance import sha256
from test_core import Clock, Factory, Recorder, settings


class MemoryStore(Recorder):
    def __init__(self):
        super().__init__()
        self.closed = False
        self.saved = None

    def start(self, metadata):
        self.metadata = metadata

    def save(self, result, metadata):
        self.saved = result
        return result.metrics

    def close(self):
        self.closed = True


def test_application_storage_and_clock_injected_without_files():
    store, factory = MemoryStore(), Factory()
    result = run_learning(factory, settings(), 8, store=store, seed=0, clock=Clock(),
                          metadata={"test": "in-memory"})
    assert result["env_steps"] == 8 and result["wall_seconds"] == 2
    assert store.closed and store.metadata == {"test": "in-memory"}
    assert store.saved.model.spec == EnvironmentSpec(5, 3)


def test_application_closes_observer_before_store_save_and_on_failure():
    order = []
    class Observer(Recorder):
        def close(self):
            order.append("observer-close")
    class Store(MemoryStore):
        def save(self, result, metadata):
            order.append("save")
            return super().save(result, metadata)
        def close(self):
            order.append("store-close")
            super().close()
    run_learning(Factory(), settings(), 8, store=Store(), seed=0, clock=Clock(), observer=Observer())
    assert order == ["observer-close", "save", "store-close"]
    order.clear()
    class FailingObserver(Observer):
        def on_event(self, event):
            raise RuntimeError("observer failure")
    factory, store = Factory(), Store()
    with pytest.raises(RuntimeError, match="observer failure"):
        run_learning(factory, settings(), 8, store=store, seed=0, clock=Clock(), observer=FailingObserver())
    assert order == ["observer-close", "store-close"]
    assert all(s.closed for s in factory.slots)
    assert store.saved is None


def test_trusted_module_factory_and_entrypoint(monkeypatch):
    module = types.ModuleType("fake_backend_plugin")
    calls = []
    def build(config, *, seed, variant):
        calls.append((config, seed, variant))
        return Factory()
    module.build = build
    monkeypatch.setitem(sys.modules, module.__name__, module)
    backend = load_backend("fake_backend_plugin:build", {"option": True}, seed=2, variant="custom")
    assert backend.spec == EnvironmentSpec(5, 3)
    assert calls == [({"option": True}, 2, "custom")]
    class Entry:
        def load(self):
            return build
    import stale_nav.plugins as plugins
    monkeypatch.setattr(plugins, "entry_points", lambda **kwargs: [Entry()] if kwargs["name"] == "custom" else [])
    assert load_backend("custom").spec == EnvironmentSpec(5, 3)
    with pytest.raises(ValueError, match="Unknown"):
        load_backend("missing")
    with pytest.raises(ValueError, match="JSON object"):
        load_backend("graph", [])
    module.bad = lambda config, **kwargs: object()
    with pytest.raises(TypeError, match="create"):
        load_backend("fake_backend_plugin:bad")


def test_development_non_graph_artifacts_and_refuse_partial_overwrite(tmp_path, monkeypatch):
    # Import the example without relying on installation of the examples folder.
    monkeypatch.syspath_prepend(str(tmp_path))
    module = types.ModuleType("fake_backend_plugin")
    module.factory = lambda config, **kwargs: Factory()
    monkeypatch.setitem(sys.modules, module.__name__, module)
    output = tmp_path / "development"
    result = learn_development(backend="fake_backend_plugin:factory", backend_config={"mode": "test"},
                               steps=5, seed=2, variant="custom", device="cpu", output=output, clock=Clock())
    assert result["kind"] == "development_learn" and not result["approved_final_study"]
    assert result["spec"] == {"observation_size": 5, "action_count": 3}
    assert result["backend_config"] == {"mode": "test"}
    checkpoint = torch.load(output / "checkpoint.pt", weights_only=True)
    assert checkpoint["pilot"] and checkpoint["model_state"]["actor.0.weight"].shape[1] == 5
    with pytest.raises(ValueError, match="Development"):
        load_policy(output / "checkpoint.pt")
    artifacts = json.loads((output / "artifacts.json").read_text())
    assert all(sha256(output / name) == digest for name, digest in artifacts.items())
    before = {p.name: p.read_bytes() for p in output.iterdir()}
    with pytest.raises(FileExistsError):
        learn_development(backend="fake_backend_plugin:factory", backend_config={}, steps=1, seed=0,
                          variant="full", device="cpu", output=output)
    assert {p.name: p.read_bytes() for p in output.iterdir()} == before
    incomplete = tmp_path / "incomplete"
    incomplete.mkdir()
    (incomplete / "events.jsonl").write_text("partial\n")
    with pytest.raises(FileExistsError):
        learn_development(backend="graph", backend_config={}, steps=1, seed=0,
                          variant="full", device="cpu", output=incomplete)
    assert (incomplete / "events.jsonl").read_text() == "partial\n"


def test_visualization_lazy_factory_replay_before_manifest(tmp_path, monkeypatch, capsys):
    # Exercise the agreed UI interface without editing/testing the actual renderer.
    made = []
    module = types.ModuleType("stale_nav.adapters.visualization")
    class Browser(Recorder):
        wants_scene = True
        url = "http://127.0.0.1:12345/"
        def __init__(self, output, port):
            super().__init__()
            assert output.is_dir()
            self.output = output
            made.append(self)
        def close(self):
            (self.output / "visualization.html").write_text("self-contained replay")
            (self.output / "visualization.json").write_text(json.dumps(self.events))
    module.BrowserObserver = Browser
    monkeypatch.setitem(sys.modules, module.__name__, module)
    output = tmp_path / "visual"
    learn_development(backend="graph", backend_config={}, steps=4, seed=0, variant="full",
                      device="cpu", output=output, visualize=True, clock=Clock())
    printed = capsys.readouterr().out
    assert Browser.url in printed and "visualization.html" in printed
    artifacts = json.loads((output / "artifacts.json").read_text())
    assert artifacts["visualization.html"] == sha256(output / "visualization.html")
    assert artifacts["visualization.json"] == sha256(output / "visualization.json")
    assert all("scene" in e for e in made[0].events if e["type"] == "transition")
    with pytest.raises(FileExistsError):
        learn_development(backend="graph", backend_config={}, steps=4, seed=0, variant="full",
                          device="cpu", output=output, visualize=True)
    assert len(made) == 1
    learn_development(backend="graph", backend_config={}, steps=1, seed=0, variant="full",
                      device="cpu", output=tmp_path / "no-visual", clock=Clock())
    assert len(made) == 1


def test_cli_development_parser_and_json_config(tmp_path, monkeypatch, capsys):
    args = parser().parse_args(["learn", "--output", str(tmp_path / "out")])
    assert args.backend == "graph" and not args.visualize and args.visualization_port == 0
    for command in ("smoke", "train"):
        arguments = [command, "--output", str(tmp_path / command), "--visualize"]
        if command == "train":
            arguments += ["--review", str(tmp_path / "review.json")]
        assert parser().parse_args(arguments).visualize
    import stale_nav.application as application
    calls = []
    monkeypatch.setattr(application, "learn_development", lambda **kwargs: calls.append(kwargs) or {"ok": True})
    main(["learn", "--backend", "custom:factory", "--backend-config", '{"dimension":5}',
          "--steps", "3", "--output", str(tmp_path / "out")])
    assert calls[0]["backend_config"] == {"dimension": 5} and calls[0]["steps"] == 3
    config = tmp_path / "backend.json"
    config.write_text('{"limit":7}')
    main(["learn", "--backend-config", str(config), "--output", str(tmp_path / "second")])
    assert calls[1]["backend_config"] == {"limit": 7}
    assert '"ok": true' in capsys.readouterr().out


@pytest.mark.parametrize("backend,config,has_scene", [
    ("graph", {}, True),
    ("examples.counter_backend:factory", {"time_limit": 2}, False),
])
def test_actual_visual_adapter_generic_pipeline(tmp_path, monkeypatch, capsys, backend, config, has_scene):
    from stale_nav.provenance import ROOT
    monkeypatch.syspath_prepend(str(ROOT))
    output = tmp_path / "actual-visual"
    result = learn_development(backend=backend, backend_config=config, steps=4, seed=0,
                               variant="full", device="cpu", output=output, visualize=True, clock=Clock())
    recording = json.loads((output / "visualization.json").read_text())
    assert recording["complete"] and result["env_steps"] == 4
    transitions = [e for e in recording["events"] if e["type"] == "transition"]
    assert len(transitions) == 4 and all(("scene" in e) == has_scene for e in transitions)
    assert all(len(e["observation"]) == result["spec"]["observation_size"] for e in transitions)
    artifacts = json.loads((output / "artifacts.json").read_text())
    assert artifacts["visualization.html"] == sha256(output / "visualization.html")
    assert artifacts["visualization.json"] == sha256(output / "visualization.json")
    assert "http://127.0.0.1:" in capsys.readouterr().out


def test_cli_actual_non_graph_plugin(tmp_path, monkeypatch, capsys):
    from stale_nav.provenance import ROOT
    monkeypatch.syspath_prepend(str(ROOT))
    output = tmp_path / "cli-example"
    main(["learn", "--backend", "examples.counter_backend:factory", "--backend-config",
          '{"time_limit":2}', "--steps", "3", "--seed", "6", "--output", str(output)])
    result = json.loads(capsys.readouterr().out)
    assert result["kind"] == "development_learn" and result["env_steps"] == 3
    assert result["spec"] == {"observation_size": 5, "action_count": 3}


def test_example_plugin_without_graph_dependencies(monkeypatch):
    from stale_nav.provenance import ROOT
    monkeypatch.syspath_prepend(str(ROOT))
    factory = load_backend("examples.counter_backend:factory", {"time_limit": 2}, seed=0, variant="full")
    assert factory.spec == EnvironmentSpec(5, 3)
    env = factory.create(0)
    assert len(env.reset()) == 5
    first = env.step(1)
    last = env.step(1)
    assert not first.truncated and last.truncated and not last.terminated
    assert last.observation[2] == 1.0  # final observation, not reset state
    env.close()
