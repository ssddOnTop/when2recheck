#!/usr/bin/env python3
"""Verify saved experiment evidence without importing or executing training code.

Uses Python's standard library. Source archives are inspected, never extracted.
Hashes establish internal consistency, not independent scientific correctness.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import sys
import tarfile


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def canonical_digest(value: object) -> str:
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def read_json(path: Path):
    def reject_constant(value):
        raise ValueError(f"Non-finite JSON constant: {value}")
    return json.loads(path.read_text(), parse_constant=reject_constant)


def relative_name(name: str) -> PurePosixPath:
    result = PurePosixPath(name)
    require(bool(name) and not result.is_absolute() and ".." not in result.parts,
            f"Unsafe manifest path: {name!r}")
    require(result.as_posix() == name and name != ".", f"Noncanonical manifest path: {name!r}")
    return result


def relocated_path(saved: str, root: Path) -> Path:
    """Accept this project root or relocate its historical Mac/Linux absolute path."""
    path = Path(saved)
    if path.is_absolute() and path.is_relative_to(root):
        candidate = path
    else:
        parts = path.parts
        markers = [i for i in range(len(parts) - 1) if parts[i:i + 2] == ("academics", "comp597")]
        require(path.is_absolute() and len(markers) == 1, f"Cannot relocate project input: {saved}")
        candidate = root.joinpath(*parts[markers[0] + 2:])
    resolved = candidate.resolve()
    require(resolved.is_relative_to(root), f"Input escapes project: {saved}")
    return resolved


def verify_run(directory: Path, root: Path) -> dict:
    directory = directory.resolve()
    require(directory.is_relative_to(root), f"Run is outside project: {directory}")
    index = directory / "artifacts.json"
    require(not index.is_symlink(), "Completion manifest symlink refused")
    require(index.is_file(), "Completion manifest must be a regular file")
    artifacts = read_json(index)
    required = {"provenance.json", "config.json", "source_snapshot.json", "summary.json"}
    require(required <= set(artifacts), "Completion manifest lacks required metadata")
    actual = {p.name for p in directory.iterdir() if p.is_file() and p.name != "artifacts.json"}
    require(actual == set(artifacts), "Artifact manifest does not exactly cover saved files")
    for name, expected in artifacts.items():
        path = relative_name(name)
        require(len(path.parts) == 1, "Run artifacts must be immediate files")
        file = directory / name
        require(not file.is_symlink(), f"Artifact symlink refused: {name}")
        require(digest(file) == expected, f"Artifact hash mismatch: {name}")

    provenance = read_json(directory / "provenance.json")
    config = read_json(directory / "config.json")
    snapshot = read_json(directory / "source_snapshot.json")
    require(provenance["graph_only"] is True, "Graph-only provenance marker missing")
    require(canonical_digest(config) == canonical_digest(provenance["config"]),
            "Saved configuration differs from provenance")
    require(canonical_digest(config) == provenance["config_sha256"], "Configuration hash mismatch")
    require(canonical_digest(snapshot) == canonical_digest(provenance["source_snapshot"]),
            "Source snapshot metadata mismatch")
    require(canonical_digest(provenance["files"]) == provenance["source_sha256"], "Source manifest hash mismatch")
    require(canonical_digest(snapshot["files"]) == snapshot["manifest_sha256"], "Archive manifest hash mismatch")
    for name, expected in provenance["files"].items():
        require(snapshot["files"].get(name) == expected, f"Source file absent or changed in snapshot: {name}")

    archive = relocated_path(snapshot["path"], root)
    require(digest(archive) == snapshot["sha256"], "Source archive byte hash mismatch")
    seen = set()
    with tarfile.open(archive, "r:*") as tar:
        for member in tar:
            relative_name(member.name)
            require(member.isfile() and not member.issym() and not member.islnk(), "Archive contains nonregular entry")
            require(member.name not in seen, "Archive contains duplicate entry")
            require(member.name in snapshot["files"], "Archive contains unexpected entry")
            seen.add(member.name)
            stream = tar.extractfile(member)
            require(stream is not None, "Archive member cannot be read")
            with stream:
                hasher = hashlib.sha256()
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    hasher.update(chunk)
            require(hasher.hexdigest() == snapshot["files"][member.name], f"Archived source mismatch: {member.name}")
    require(seen == set(snapshot["files"]), "Archive has missing source members")
    for saved, expected in provenance["input_sha256"].items():
        require(digest(relocated_path(saved, root)) == expected, f"Saved input hash mismatch: {saved}")
    return {"run": str(directory), "passed": True, "kind": provenance["kind"],
            "artifact_count": len(artifacts), "archived_source_files": len(seen),
            "source_sha256": provenance["source_sha256"], "config_sha256": provenance["config_sha256"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="*", type=Path, help="Completed run directories")
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--discover", type=Path, help="Check every provenance.json beneath this run folder")
    args = parser.parse_args()
    root = args.project_root.resolve()
    runs = list(args.runs)
    if args.discover:
        discovery_root = args.discover.resolve()
        require(discovery_root.is_relative_to(root), "Discovery folder is outside project")
        runs.extend(p.parent for p in sorted(discovery_root.rglob("provenance.json")))
    require(bool(runs), "No run directories selected")
    results = []
    for directory in dict.fromkeys(p.resolve() for p in runs):
        try:
            results.append(verify_run(directory, root))
        except (OSError, ValueError, KeyError, TypeError, tarfile.TarError) as error:
            results.append({"run": str(directory), "passed": False,
                            "error": f"{type(error).__name__}: {error}"})
    passed = all(row["passed"] for row in results)
    print(json.dumps({"passed": passed, "runs": results,
                      "scope": "Independent byte/configuration/source consistency. Not a scientific validity or authorship proof."}, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except (OSError, ValueError) as error:
        print(f"Verification failed: {error}", file=sys.stderr)
        sys.exit(1)
