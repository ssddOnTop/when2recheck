"""Content-addressed inputs, source manifests, exact commands and runtime versions."""
from __future__ import annotations
import csv
import hashlib
import importlib.metadata
import io
import json
import tarfile
import platform
from pathlib import Path
import shlex
import subprocess
import sys
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[2]
OWNED_SCRIPTS = {"sync_owned.sh", "remote_python.sh", "run_final.py"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def json_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def source_manifest(root: Path = ROOT) -> dict:
    files = []
    for folder in ("src", "tests", "configs", "scripts"):
        for p in (root / folder).rglob("*"):
            if p.is_file() and "__pycache__" not in p.parts and p.suffix not in {".pyc", ".pyo"}:
                if folder != "scripts" or p.name in OWNED_SCRIPTS:
                    files.append(p)
    files += [root / name for name in ("pyproject.toml", "README.md") if (root / name).exists()]
    manifest = {p.relative_to(root).as_posix(): sha256(p) for p in sorted(files)}
    return {"files": manifest, "source_sha256": json_hash(manifest)}


def write_json(path: Path, value):
    path.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_archive(manifest: dict, root: Path = ROOT) -> dict:
    """Immutable deterministic tar of listed source plus the frozen protocol only."""
    entries = dict(manifest["files"])
    protocol = root / "planning/submission-protocol.json"
    if protocol.exists():
        entries["planning/submission-protocol.json"] = sha256(protocol)
    key = json_hash(entries)
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w", format=tarfile.USTAR_FORMAT) as tar:
        for name, digest in sorted(entries.items()):
            data = (root / name).read_bytes()
            if hashlib.sha256(data).hexdigest() != digest:
                raise ValueError("Source changed while snapshot was being created")
            entry = tarfile.TarInfo(name)
            entry.size = len(data)
            entry.mode = 0o644
            entry.mtime = 0
            tar.addfile(entry, io.BytesIO(data))
    data = buffer.getvalue()
    digest = hashlib.sha256(data).hexdigest()
    directory = root.parent / "local-materials/source-archives"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{key}.tar"
    try:
        with path.open("xb") as f:
            f.write(data)
        path.chmod(0o444)
    except FileExistsError:
        if sha256(path) != digest:
            raise ValueError("Existing content-addressed source archive is corrupt")
    return {"path": str(path.resolve()), "sha256": digest, "manifest_sha256": key, "files": entries}


def runtime_versions() -> dict:
    versions = {"python": sys.version, "platform": platform.platform()}
    for package in ("numpy", "torch", "matplotlib", "gymnasium", "pytest", "scipy"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not installed"
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
                                capture_output=True, text=True, timeout=10, check=False)
        versions["gpu_hardware"] = result.stdout.strip() or result.stderr.strip()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        versions["gpu_hardware"] = "unavailable"
    return versions


def begin_run(output: Path, config: dict, kind: str, inputs: list[Path] = ()) -> dict:
    # Refuse reuse, including partially completed outputs. No silent overwrites.
    output.mkdir(parents=True, exist_ok=False)
    manifest = source_manifest()
    snapshot = source_archive(manifest)
    protocol = ROOT / "planning/submission-protocol.json"
    audit = ROOT.parent / "local-materials/audits/protocol-review-v2.json"
    paths = list(inputs) + [p for p in (protocol, audit) if p.exists()]
    provenance = {
        **manifest, "kind": kind, "graph_only": True, "source_snapshot": snapshot,
        "config_sha256": json_hash(config), "config": config,
        "input_sha256": {str(p.resolve()): sha256(p) for p in paths},
        "command": shlex.join([sys.executable, *sys.argv]),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": runtime_versions(),
    }
    write_json(output / "source_snapshot.json", snapshot)
    write_json(output / "provenance.json", provenance)
    write_json(output / "config.json", config)
    return provenance


def finish_run(output: Path, summary: dict):
    write_json(output / "summary.json", summary)
    artifacts = {p.name: sha256(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != "artifacts.json"}
    write_json(output / "artifacts.json", artifacts)


def verify_artifacts(output: Path):
    snapshot = json.loads((output / "source_snapshot.json").read_text())
    if sha256(Path(snapshot["path"])) != snapshot["sha256"]:
        raise ValueError("Historical source archive hash mismatch")
    artifacts = json.loads((output / "artifacts.json").read_text())
    for name, digest in artifacts.items():
        if sha256(output / name) != digest:
            raise ValueError(f"Artifact hash mismatch: {output / name}")
