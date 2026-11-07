"""Standard-library regression tests for the independent evidence verifier."""
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tarfile
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("verify_package", Path(__file__).with_name("verify_package.py"))
verifier = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verifier)


def save(path, value):
    path.write_text(json.dumps(value))


def seal(run):
    save(run / "artifacts.json", {p.name: verifier.digest(p) for p in run.iterdir()
                                 if p.is_file() and p.name != "artifacts.json"})


class EvidenceVerifierTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="comp597-verifier-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name).resolve()
        self.run = self.root / "runs" / "test"
        self.run.mkdir(parents=True)
        source = b"print('source fixture, not a training result')\n"
        files = {"src/example.py": hashlib.sha256(source).hexdigest()}
        self.archive = self.root / "source.tar"
        with tarfile.open(self.archive, "w") as tar:
            member = tarfile.TarInfo("src/example.py")
            member.size = len(source)
            tar.addfile(member, io.BytesIO(source))
        self.config = {"ppo": {"gamma": 1.0}}
        snapshot = {"files": files, "path": str(self.archive),
                    "manifest_sha256": verifier.canonical_digest(files),
                    "sha256": verifier.digest(self.archive)}
        self.provenance = {"files": files, "source_sha256": verifier.canonical_digest(files),
                           "source_snapshot": snapshot, "config": self.config,
                           "config_sha256": verifier.canonical_digest(self.config),
                           "input_sha256": {}, "graph_only": True, "kind": "test_fixture"}
        save(self.run / "provenance.json", self.provenance)
        save(self.run / "config.json", self.config)
        save(self.run / "source_snapshot.json", snapshot)
        save(self.run / "summary.json", {"fixture": True})
        seal(self.run)

    def check_rejected(self):
        with self.assertRaises((ValueError, FileNotFoundError)):
            verifier.verify_run(self.run, self.root)

    def test_intact_evidence_passes(self):
        self.assertTrue(verifier.verify_run(self.run, self.root)["passed"])

    def test_changed_artifact_fails(self):
        save(self.run / "summary.json", {"fixture": False})
        self.check_rejected()

    def test_unlisted_artifact_fails(self):
        (self.run / "unexpected.csv").write_text("not in manifest")
        self.check_rejected()

    def test_missing_metadata_fails(self):
        (self.run / "summary.json").unlink()
        seal(self.run)
        self.check_rejected()

    def test_resealed_config_mismatch_fails(self):
        save(self.run / "config.json", {"ppo": {"gamma": 0.5}})
        seal(self.run)
        self.check_rejected()

    def test_config_numeric_boolean_mismatch_fails(self):
        self.provenance["config"] = {"ppo": {"gamma": True}}
        save(self.run / "provenance.json", self.provenance)
        seal(self.run)
        self.check_rejected()

    def test_config_integer_float_mismatch_fails(self):
        self.provenance["config"] = {"ppo": {"gamma": 1}}
        save(self.run / "provenance.json", self.provenance)
        seal(self.run)
        self.check_rejected()

    def test_config_key_order_is_not_significant(self):
        config = {"ppo": {"gamma": 1.0, "seed": 597}}
        self.provenance["config"] = {"ppo": {"seed": 597, "gamma": 1.0}}
        self.provenance["config_sha256"] = verifier.canonical_digest(config)
        save(self.run / "config.json", config)
        save(self.run / "provenance.json", self.provenance)
        seal(self.run)
        self.assertTrue(verifier.verify_run(self.run, self.root)["passed"])

    def test_completion_index_symlink_rejected_before_read(self):
        from unittest.mock import patch
        index = self.run / "artifacts.json"
        target = self.root / "index-target.json"
        target.write_bytes(index.read_bytes())
        index.unlink()
        index.symlink_to(target)
        with patch.object(verifier, "read_json") as reader:
            with self.assertRaisesRegex(ValueError, "Completion manifest symlink"):
                verifier.verify_run(self.run, self.root)
            reader.assert_not_called()

    def test_outside_completion_index_rejected_before_read(self):
        from unittest.mock import patch
        with tempfile.TemporaryDirectory(prefix="comp597-outside-fixture-") as outside:
            index = self.run / "artifacts.json"
            target = Path(outside) / "index.json"
            target.write_bytes(index.read_bytes())
            index.unlink()
            index.symlink_to(target)
            with patch.object(verifier, "read_json") as reader:
                with self.assertRaisesRegex(ValueError, "Completion manifest symlink"):
                    verifier.verify_run(self.run, self.root)
                reader.assert_not_called()

    def test_nonregular_completion_index_fails(self):
        index = self.run / "artifacts.json"
        index.unlink()
        index.mkdir()
        self.check_rejected()

    def test_modified_archive_fails(self):
        with self.archive.open("ab") as stream:
            stream.write(b"tampered")
        self.check_rejected()

    def test_member_hash_checked_independently(self):
        with tarfile.open(self.archive, "w") as tar:
            member = tarfile.TarInfo("src/example.py")
            member.size = 3
            tar.addfile(member, io.BytesIO(b"bad"))
        snapshot = self.provenance["source_snapshot"]
        snapshot["sha256"] = verifier.digest(self.archive)
        save(self.run / "source_snapshot.json", snapshot)
        save(self.run / "provenance.json", self.provenance)
        seal(self.run)
        self.check_rejected()

    def test_symlink_artifact_fails(self):
        (self.run / "summary.json").unlink()
        target = self.root / "summary-target.json"
        save(target, {"fixture": True})
        (self.run / "summary.json").symlink_to(target)
        seal(self.run)
        self.check_rejected()

    def test_input_hash_checked(self):
        target = self.root / "input.json"
        save(target, {"fixture": True})
        self.provenance["input_sha256"][str(target)] = "0" * 64
        save(self.run / "provenance.json", self.provenance)
        seal(self.run)
        self.check_rejected()

    def test_relocation_preserves_project_suffix(self):
        saved = "/home/ssdd/academics/comp597/local-materials/source.tar"
        self.assertEqual(verifier.relocated_path(saved, self.root), self.root / "local-materials/source.tar")

    def test_relocation_and_manifest_escape_rejected(self):
        for name in ("../outside", "/etc/passwd", "./src/file", "src/../file"):
            with self.assertRaises(ValueError):
                verifier.relative_name(name)
        with self.assertRaises(ValueError):
            verifier.relocated_path("/home/ssdd/academics/comp597/../../outside", self.root)


if __name__ == "__main__":
    unittest.main(verbosity=2)
