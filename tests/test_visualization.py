import json
import random
import re
import socket
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest
import torch

from stale_nav.adapters.visualization import BrowserObserver


def transition(step, **extra):
    return {"type": "transition", "step": step, "env_slot": 0, "action": 1,
            "reward": 1.5, "terminated": False, "truncated": False,
            "observation": [0.1, 0.8], **extra}


def test_loopback_live_and_self_contained_replay(tmp_path):
    output = tmp_path / "run"
    with BrowserObserver(output) as observer:
        assert not output.exists()  # Never preempt the application's exclusive run creation.
        assert observer._server.server_address[0] == "127.0.0.1"
        observer.on_event({"type": "started", "step": 0, "spec": {"observation_size": 2}})
        observer.on_event(transition(1))
        with urlopen(observer.url, timeout=2) as response:
            html = response.read().decode()
            assert "const live = true;" in html
            assert response.headers["Cache-Control"] == "no-store"
        with urlopen(observer.url + "events", timeout=2) as response:
            live = json.load(response)
        assert live["events"][-1]["reward"] == 1.5
        assert not live["complete"]
        with pytest.raises(HTTPError) as exc:
            urlopen(Request(observer.url + "events", headers={"Host": "untrusted.example"}), timeout=2)
        assert exc.value.code == 403
        for path in ["README.md", "../README.md", "events?path=/etc/passwd"]:
            with pytest.raises(HTTPError) as exc:
                urlopen(observer.url + path, timeout=2)
            assert exc.value.code == 404
    recording = json.loads((output / "visualization.json").read_text())
    assert recording["complete"]
    html = (output / "visualization.html").read_text()
    assert "const live = false;" in html
    embedded = re.search(r'<script id="recording" type="application/json">(.*?)</script>', html, re.S)
    assert json.loads(embedded.group(1)) == recording
    assert "<script src=" not in html
    assert not observer._thread.is_alive()
    with socket.socket() as probe:
        assert probe.connect_ex(observer._server.server_address) != 0
    observer.close()  # Idempotent; does not replace the recording.


def test_sampling_is_per_selected_env_not_global_step_modulo(tmp_path):
    with BrowserObserver(tmp_path, sample_every=2, max_events=6) as observer:
        observer.on_event({"type": "started", "step": 0})
        for n in range(5):
            observer.on_event(transition(n * 16 + 1))
            observer.on_event(transition(n * 16 + 2, env_slot=1))
        observer.on_event(transition(81, terminated=True))
        result = observer.snapshot()
        assert [e["step"] for e in result["events"] if e["type"] == "transition"] == [1, 33, 65, 81]
        assert result["sampled_out"] == 2
        assert result["evicted"] == 0
        observer.on_event({"type": "update", "step": 96, "metrics": {"entropy": 0.1}})
        observer.on_event({"type": "finished", "step": 96})
        assert len(observer.snapshot()["events"]) == 6
        assert observer.snapshot()["evicted"] == 1
        assert observer.snapshot()["metadata"]["type"] == "started"


def test_episode_return_includes_sampled_steps_and_resets_at_time_limit(tmp_path):
    with BrowserObserver(tmp_path, sample_every=10) as observer:
        observer.on_event(transition(1, reward=2.0))
        observer.on_event(transition(2, reward=3.0))  # Not retained as a frame.
        observer.on_event(transition(3, reward=-1.0, truncated=True))
        observer.on_event(transition(4, reward=8.0, terminated=True))
        ends = [e for e in observer.snapshot()["events"] if "episode_return" in e]
        assert [(e["episode_return"], e["episode_length"]) for e in ends] == [(4.0, 3), (8.0, 1)]


def test_events_cannot_be_mutated_or_inject_script(tmp_path):
    with BrowserObserver(tmp_path) as observer:
        event = transition(1, scene={"title": "</script><script>alert('bad')</script>",
                                     "nodes": []})
        observer.on_event(event)
        event["scene"]["title"] = "changed"
        exposed = observer.snapshot()
        exposed["events"].clear()
        assert len(observer.snapshot()["events"]) == 1
        with pytest.raises(ValueError):
            observer.on_event(transition(2, reward=float("nan")))
    html = (tmp_path / "visualization.html").read_text()
    assert "<script>alert('bad')</script>" not in html
    assert "\\u003c/script>" in html
    assert "innerHTML" not in html  # Public/plugin labels always use textContent/canvas text.


def test_visualization_does_not_consume_learning_random_streams(tmp_path):
    random.seed(12)
    np.random.seed(12)
    torch.manual_seed(12)
    py_before, np_before, torch_before = random.getstate(), np.random.get_state(), torch.get_rng_state()
    with BrowserObserver(tmp_path) as observer:
        observer.on_event(transition(1))
        observer.snapshot()
    assert random.getstate() == py_before
    assert np.random.get_state()[0] == np_before[0]
    assert np.array_equal(np.random.get_state()[1], np_before[1])
    assert np.random.get_state()[2:] == np_before[2:]
    assert torch.equal(torch.get_rng_state(), torch_before)


@pytest.mark.parametrize("kwargs", [{"port": -1}, {"port": 65536}, {"sample_every": 0},
                                    {"max_events": 1}, {"env_slot": -1}])
def test_invalid_options_do_not_start_server(tmp_path, kwargs):
    with pytest.raises(ValueError):
        BrowserObserver(tmp_path, **kwargs)


def test_existing_recordings_not_overwritten(tmp_path):
    target = tmp_path / "visualization.json"
    target.write_text("original")
    with pytest.raises(FileExistsError):
        BrowserObserver(tmp_path)
    assert target.read_text() == "original"


def test_cleanup_on_training_error_preserves_partial_replay(tmp_path):
    with pytest.raises(RuntimeError):
        with BrowserObserver(tmp_path) as observer:
            observer.on_event(transition(1))
            raise RuntimeError("training stopped")
    assert not observer._thread.is_alive()
    assert json.loads((tmp_path / "visualization.json").read_text())["events"]
