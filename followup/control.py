"""Fail-closed review gates and content-addressed provenance.

A design PASS does not authorize any prospective generation or execution.
Approval files are human-reviewed capabilities, not cryptographic identities.
Only manifest/verification and unit-design helpers are ungated. Every public
prospective entry point requires an exact implementation approval. Held-out
release additionally binds all ten final checkpoints and frozen validation.
"""
from __future__ import annotations
from dataclasses import dataclass
import hashlib
import importlib.metadata
import io
import json
from pathlib import Path
import platform
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
PLAN = ROOT / "planning/followup-protocol.json"
CONFIG = ROOT / "followup/execution_config.json"
DESIGN_REVIEW = ROOT.parent / "local-materials/audits/followup-design-review.json"
EVIDENCE = ROOT.parent / "local-materials/followup/implementation"
STUDY = "followup-safe-balance-v1"
PREFIX = "comp597/stale-nav/" + STUDY
UNIT_NAMESPACE = PREFIX + "/unit-design"
OPERATIONS = frozenset({"generate_train", "generate_validation", "generate_pilot", "generate_heldout",
                        "pilot", "train", "evaluate_validation", "evaluate_heldout",
                        "seal_validation", "aggregate"})
BINDING_KEYS = ("implementation_sha256", "protocol_sha256", "reservation_sha256",
                "config_sha256", "design_review_sha256", "original_source_sha256")


class GateError(ValueError):
    pass


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text())


def write_json(path: Path, value):
    path.write_text(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


def plan():
    return read_json(PLAN)


def config():
    c, p = read_json(CONFIG), plan()
    hp = p["planned_training_not_authorized"]
    if c["study_id"] != STUDY or c["horizon_tasks"] != 1:
        raise GateError("Wrong study or horizon")
    if any(value != hp[key] for key, value in c["ppo"].items()):
        raise GateError("PPO configuration differs from approved protocol")
    if c["training_seeds"] != p["prospective_seeds_and_seal"]["independent_training_seeds"] or c["variants"] != ["full", "masked"]:
        raise GateError("Matched seeds/variants differ from approved protocol")
    if c["pilot_steps"] != 4096 or c["pilot_seed"] != 1000 or c["pilot_variant"] != "full":
        raise GateError("Only the predeclared feasibility pilot is permitted")
    return c


def manifest():
    # Enumerate all follow-up files, including tests and approved analysis. New
    # execution files cannot silently escape the review by missing an allowlist.
    files = {p.relative_to(ROOT).as_posix(): sha(p) for p in sorted((ROOT / "followup").rglob("*"))
             if p.is_file() and "__pycache__" not in p.parts and ".pytest_cache" not in p.parts}
    files["planning/followup-protocol.json"] = sha(PLAN)
    p, review = plan(), read_json(DESIGN_REVIEW)
    if review.get("verdict") != "PASS_FOR_FOLLOWUP_IMPLEMENTATION" or review.get("remaining_blockers"):
        raise GateError("Independent design review is not accepted")
    for name, expected in review["new_files_sha256"].items():
        if sha(ROOT / name) != expected:
            raise GateError(f"Approved design/analytical file changed: {name}")
    # Import only the frozen read-only provenance helper, never the old trainer.
    from stale_nav.provenance import source_manifest
    original_manifest = source_manifest()
    original = original_manifest["source_sha256"]
    if original != review["original_source_manifest_sha256"]:
        raise GateError("Audited original source changed")
    # Archive the exact frozen dependencies as read-only input copies too. A
    # future package can verify historical execution without the live checkout.
    files.update(original_manifest["files"])
    for stem in ("protocol", "environment", "policies"):
        name = p["frozen_inputs"][stem + "_path"]
        value = p["frozen_inputs"][stem + "_sha256"]
        if sha(ROOT / name) != value:
            raise GateError("Frozen mechanics input changed: " + name)
        files[name] = value
    reservation = canonical(p["prospective_seeds_and_seal"]["heldout_reservation"])
    if reservation != review["heldout_reservation_countersignature"]["accepted_specification_sha256"]:
        raise GateError("Held-out reservation differs from countersigned specification")
    config()
    return {"study_id": STUDY, "implementation_sha256": canonical(files), "files": files,
            "protocol_sha256": sha(PLAN), "reservation_sha256": reservation,
            "config_sha256": sha(CONFIG), "design_review_sha256": sha(DESIGN_REVIEW),
            "original_source_sha256": original,
            "interpretation": config()["ablation_interpretation"],
            "execution_status": "BLOCKED_UNTIL_INDEPENDENT_IMPLEMENTATION_PASS"}


def match_bindings(record: dict, expected: dict):
    for key in BINDING_KEYS:
        if record.get(key) != expected[key]:
            raise GateError(f"Approval/artifact binding mismatch: {key}")
    if record.get("study_id") != STUDY:
        raise GateError("Wrong study ID")


@dataclass(frozen=True)
class Grant:
    review: Path
    review_sha256: str
    bindings: dict
    permissions: frozenset[str]

    def check(self, operation: str):
        if operation not in self.permissions or sha(self.review) != self.review_sha256:
            raise GateError("Permission absent or review changed during execution")
        record = read_json(self.review)
        if record.get("unit_fixture"):
            raise GateError("Unit review fixtures never grant prospective execution")
        review_permissions(record, self.bindings, operation)


def review_permissions(record: dict, expected: dict, operation: str) -> frozenset:
    """Pure schema/binding validation, not a capability or execution approval."""
    if operation not in OPERATIONS:
        raise GateError("Unknown execution operation")
    if record.get("verdict") != "PASS_FOR_FOLLOWUP_EXECUTION" or record.get("remaining_blockers", None) != []:
        raise GateError("Design approval is insufficient; require PASS_FOR_FOLLOWUP_EXECUTION with no blockers")
    match_bindings(record, expected)
    permissions = frozenset(record.get("permissions", []))
    if not permissions <= OPERATIONS or operation not in permissions:
        raise GateError("Requested operation not independently approved")
    return permissions


def authorize(review: Path | None, operation: str) -> Grant:
    if review is None:
        raise GateError("Prospective execution refused: independent implementation approval is required")
    record = read_json(review)
    if record.get("unit_fixture"):
        raise GateError("Unit review fixtures never grant prospective execution")
    expected = manifest()
    permissions = review_permissions(record, expected, operation)
    return Grant(review.resolve(), sha(review), expected, permissions)


def checked_output(path: Path) -> Path:
    path = path.resolve()
    base = EVIDENCE.resolve()
    if path == base or not path.is_relative_to(base):
        raise GateError("Execution writes must stay below ../local-materials/followup/implementation")
    return path


def source_archive(output: Path, bindings: dict):
    archive = output / "source.tar"
    with tarfile.open(archive, "x", format=tarfile.USTAR_FORMAT) as tar:
        for name, expected in sorted(bindings["files"].items()):
            data = (ROOT / name).read_bytes()
            if hashlib.sha256(data).hexdigest() != expected:
                raise GateError("Source changed during snapshot")
            member = tarfile.TarInfo(name)
            member.size, member.mode, member.mtime = len(data), 0o444, 0
            tar.addfile(member, io.BytesIO(data))
    archive.chmod(0o444)


def begin(output: Path, grant: Grant, operation: str, inputs=()):
    grant.check(operation)
    output = checked_output(output)
    output.mkdir(parents=True, exist_ok=False)
    source_archive(output, grant.bindings)
    versions = {"python": sys.version, "platform": platform.platform()}
    for name in ("numpy", "torch", "pytest"):
        versions[name] = importlib.metadata.version(name)
    write_json(output / "provenance.json", {
        **{k: grant.bindings[k] for k in ("study_id", *BINDING_KEYS)},
        "kind": operation, "review_sha256": grant.review_sha256,
        "source_files": grant.bindings["files"], "runtime": versions,
        "command_argv": [sys.executable, *sys.argv],
        "input_sha256": {str(Path(p).resolve()): sha(Path(p)) for p in (grant.review, DESIGN_REVIEW, *inputs)},
        "config": config(), "interpretation": config()["ablation_interpretation"]})
    return output


def finish(output: Path, summary: dict, grant: Grant):
    if manifest()["implementation_sha256"] != grant.bindings["implementation_sha256"]:
        raise GateError("Source changed during execution; no completed artifact may be published")
    write_json(output / "summary.json", summary)
    files = {p.name: sha(p) for p in sorted(output.iterdir()) if p.is_file() and p.name != "artifacts.json"}
    write_json(output / "artifacts.json", files)


def verify(output: Path, expected: dict | None = None):
    files = read_json(output / "artifacts.json")
    actual = {p.name for p in output.iterdir() if p.is_file() and p.name != "artifacts.json"}
    if actual != set(files):
        raise GateError("Artifact coverage mismatch")
    for name, value in files.items():
        if Path(name).name != name or sha(output / name) != value:
            raise GateError("Artifact hash mismatch")
    provenance = read_json(output / "provenance.json")
    if expected is not None:
        match_bindings(provenance, expected)
        for name, value in provenance.get("input_sha256", {}).items():
            if not Path(name).is_file() or sha(Path(name)) != value:
                raise GateError("Saved execution input changed or is unavailable: " + name)
    with tarfile.open(output / "source.tar") as archive:
        members = archive.getmembers()
        wanted = provenance["source_files"]
        if len(members) != len(wanted) or {m.name for m in members} != set(wanted):
            raise GateError("Source archive member coverage mismatch")
        for m in members:
            if not m.isfile() or hashlib.sha256(archive.extractfile(m).read()).hexdigest() != wanted[m.name]:
                raise GateError("Historical source archive mismatch")
    if canonical(provenance["source_files"]) != provenance["implementation_sha256"]:
        raise GateError("Source manifest digest mismatch")
    return provenance


def expected_models():
    c = config()
    return {(v, s) for v in c["variants"] for s in c["training_seeds"]}


def validate_model_metadata(records: list[dict], bindings: dict):
    found = set()
    devices = set()
    for ck in records:
        match_bindings(ck, bindings)
        pair = (ck.get("variant"), ck.get("training_seed"))
        if pair in found or pair not in expected_models():
            raise GateError("Duplicate or unexpected matched model")
        found.add(pair)
        if ck.get("device") not in config()["allowed_devices"] or ck.get("config") != config():
            raise GateError("Checkpoint device/config metadata mismatch")
        from .domain import feature_spec
        if ck.get("features") != feature_spec():
            raise GateError("Checkpoint feature specification mismatch")
        devices.add(ck.get("device"))
        if ck.get("pilot") is not False or ck.get("unit_fixture", False):
            raise GateError("Only genuine final models can release held-out data")
        if ck.get("env_steps") != config()["ppo"]["total_env_steps_per_run"] or ck.get("horizon_tasks") != 1:
            raise GateError("Final budget or one-task horizon mismatch")
    if found != expected_models() or len(devices) != 1:
        raise GateError("Require all ten matched final models on one device")


def checkpoint_records(paths: list[Path], bindings: dict):
    import torch
    records = []
    for path in paths:
        verify(path.parent, bindings)
        records.append(torch.load(path, map_location="cpu", weights_only=True))
    validate_model_metadata(records, bindings)
    return records


def heldout_release(path: Path | None, grant: Grant):
    if path is None:
        raise GateError("Held-out data remains sealed without independent release")
    release = read_json(path)
    if release.get("unit_fixture"):
        raise GateError("Unit release fixtures never unseal prospective data")
    if release.get("verdict") != "PASS_FOR_FOLLOWUP_HELDOUT_RELEASE" or release.get("remaining_blockers", None) != []:
        raise GateError("Held-out release is not approved")
    match_bindings(release, grant.bindings)
    items = release.get("checkpoints", [])
    paths = [Path(item["path"]) for item in items]
    for item, checkpoint in zip(items, paths):
        if sha(checkpoint) != item["sha256"]:
            raise GateError("Released checkpoint hash mismatch")
    checkpoint_records(paths, grant.bindings)
    decision = release["validation_decisions"]
    decision_path = Path(decision["path"])
    if sha(decision_path) != decision["sha256"]:
        raise GateError("Validation decision hash mismatch")
    verify(decision_path.parent, grant.bindings)
    frozen = read_json(decision_path)
    if frozen.get("frozen") is not True or frozen.get("split") != "validation":
        raise GateError("Validation decisions must be frozen before held-out generation")
    released_hashes = sorted(item["sha256"] for item in items)
    if frozen.get("checkpoint_sha256") != released_hashes:
        raise GateError("Validation decisions do not bind these ten final models")
    return {**{k: grant.bindings[k] for k in ("study_id", *BINDING_KEYS)},
            "release_path": str(path.resolve()), "release_sha256": sha(path),
            "checkpoint_sha256": released_hashes, "validation_sha256": sha(decision_path)}
