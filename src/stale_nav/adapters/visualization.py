"""Optional loopback telemetry server and self-contained, bounded offline replay.

This adapter consumes public events; it never reads environments, datasets, model
weights, or random generators. Construct it only when visualization is requested.
Call close() before sealing the run's artifact manifest. No browser is launched.
"""
from __future__ import annotations

from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
from threading import Lock, Thread
from typing import Any


class BrowserObserver:
    wants_scene = True

    def __init__(self, output: Path, port: int = 0, sample_every: int = 1,
                 max_events: int = 2000, env_slot: int = 0):
        if not 0 <= port <= 65535:
            raise ValueError("Visualization port must be between 0 and 65535")
        if sample_every < 1 or max_events < 2 or env_slot < 0:
            raise ValueError("Invalid visualization sampling, capacity, or environment slot")
        self.output = Path(output)
        for name in ("visualization.html", "visualization.json"):
            if (self.output / name).exists():
                raise FileExistsError(self.output / name)
        self.sample_every, self.env_slot = sample_every, env_slot
        self._events: deque[dict[str, Any]] = deque(maxlen=max_events)
        self._lock = Lock()
        self._seen = self._sampled = self._evicted = 0
        self._episode_return = 0.0
        self._episode_length = 0
        self._closed = False
        self._metadata: dict[str, Any] = {}
        self._template = Path(__file__).with_name("dashboard.html").read_text(encoding="utf-8")
        observer = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                # Reject arbitrary Host headers (including DNS-rebinding hosts).
                if self.headers.get("Host") not in {
                    f"127.0.0.1:{self.server.server_port}",
                    f"localhost:{self.server.server_port}",
                }:
                    self.send_error(403)
                    return
                # Fixed resources only: never expose the run directory or arbitrary files.
                if self.path == "/":
                    content = observer._page(live=True).encode("utf-8")
                    content_type = "text/html; charset=utf-8"
                elif self.path == "/events":
                    content = json.dumps(observer.snapshot(), allow_nan=False).encode("utf-8")
                    content_type = "application/json"
                else:
                    self.send_error(404)
                    return
                self.send_response(200)
                self.send_header("Content-Type", content_type)
                self.send_header("Content-Length", str(len(content)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Content-Security-Policy", "default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; img-src data:; frame-ancestors 'none'")
                self.end_headers()
                try:
                    self.wfile.write(content)
                except (BrokenPipeError, ConnectionResetError):
                    pass

            def log_message(self, format, *args):
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self._server.daemon_threads = True
        self.url = f"http://127.0.0.1:{self._server.server_port}/"
        self._thread = Thread(target=self._server.serve_forever,
                              kwargs={"poll_interval": 0.05}, daemon=True,
                              name="rl-visualization")
        self._thread.start()

    def on_event(self, event: dict) -> None:
        # Deep-copy through JSON to prevent later mutation by callers; disallow NaN.
        copy = json.loads(json.dumps(event, allow_nan=False))
        with self._lock:
            if self._closed:
                return
            if copy.get("type") == "started":
                self._metadata = copy
            if copy.get("type") == "transition":
                if copy.get("env_slot", 0) != self.env_slot:
                    return
                self._seen += 1
                self._episode_return += float(copy.get("reward", 0.0))
                self._episode_length += 1
                if copy.get("terminated") or copy.get("truncated"):
                    copy["episode_return"] = self._episode_return
                    copy["episode_length"] = self._episode_length
                    self._episode_return, self._episode_length = 0.0, 0
                if ((self._seen - 1) % self.sample_every and
                        not copy.get("terminated") and not copy.get("truncated")):
                    self._sampled += 1
                    return
            if len(self._events) == self._events.maxlen:
                self._evicted += 1
            self._events.append(copy)

    def snapshot(self) -> dict:
        with self._lock:
            payload = {"schema_version": 1, "metadata": self._metadata,
                       "events": list(self._events), "complete": self._closed,
                       "env_slot": self.env_slot, "sample_every": self.sample_every,
                       "sampled_out": self._sampled, "evicted": self._evicted}
            return json.loads(json.dumps(payload, allow_nan=False))

    def _page(self, live: bool) -> str:
        # Escape '<' so arbitrary labels cannot terminate the inert JSON script tag.
        recording = json.dumps(self.snapshot(), allow_nan=False).replace("<", "\\u003c")
        return self._template.replace("__LIVE_MODE__", "true" if live else "false").replace("__RECORDING_JSON__", recording)

    def close(self) -> None:
        with self._lock:
            if self._closed:
                return
            self._closed = True
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=2)
        self.output.mkdir(parents=True, exist_ok=True)
        # Exclusive writes preserve any prior recording rather than silently replacing it.
        with (self.output / "visualization.json").open("x", encoding="utf-8") as stream:
            json.dump(self.snapshot(), stream, allow_nan=False, indent=2)
            stream.write("\n")
        with (self.output / "visualization.html").open("x", encoding="utf-8") as stream:
            stream.write(self._page(live=False))

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
