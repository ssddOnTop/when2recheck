"""Prospective design checks only. No RNG, dataset generator, training or test run.

Exact rational formulas are checked against separately stored hand calculations
and deterministic hypothetical branches through the frozen physical kernel.
The new study's reserved held-out seed is hashed but never used to draw worlds.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
from fractions import Fraction
from itertools import product
import hashlib
import io
import json
import math
from pathlib import Path
import sys
import tarfile

ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "planning/followup-protocol.json"
NEW_FILES = ("planning/followup-protocol.json", "followup/analytical_validation.py",
             "followup/test_analytical_validation.py")
POLICIES = ("DIRECT", "INSPECT_REACTIVE", "INSPECT_BEST_MISS", "ALWAYS_SAFE")

# Read-only reuse of the accepted physical semantics, never its RNG or trainer.
sys.path.insert(0, str(ROOT / "src"))
from stale_nav.environment import Layout, Snapshot, initial_snapshot, physical_step
from stale_nav.policies import voi_costs
from stale_nav.provenance import source_manifest


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_digest(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load_plan(path: Path = PLAN_PATH) -> dict:
    return json.loads(path.read_text())


@dataclass(frozen=True)
class Case:
    a: Fraction
    b: Fraction
    v: Fraction
    sensing: Fraction
    safe: Fraction
    budget: Fraction
    q: Fraction = Fraction(1, 2)
    miss: Fraction = Fraction(0)
    other_cost: Fraction = Fraction(80)

    @classmethod
    def from_dict(cls, values: dict):
        return cls(**{k: Fraction(v) for k, v in values.items()})

    @property
    def overhead(self) -> Fraction:
        return 2 * self.v + self.sensing

    def scaled(self, scale: Fraction) -> Case:
        return replace(self, **{name: getattr(self, name) * scale for name in
                                ("a", "b", "v", "sensing", "safe", "budget", "other_cost")})

    def snapshot(self) -> Snapshot:
        other = float(self.other_cost)
        layout = Layout((float(self.a), other, other), (float(self.b), other, other),
                        (float(self.v), other, other), float(self.sensing),
                        float(self.safe), float(self.budget))
        return replace(initial_snapshot(layout, 0.0, float(self.miss)),
                       belief=(float(self.q), .5, .5))


def path_result(length: Fraction, budget: Fraction) -> tuple[Fraction, Fraction]:
    failure = Fraction(int(length > budget))
    return min(length, budget) + 200 * failure, failure


def average(branches) -> tuple[Fraction, Fraction]:
    if sum(weight for weight, _ in branches) != 1:
        raise AssertionError("Branch weights must sum to exactly one")
    return tuple(sum(weight * outcome[i] for weight, outcome in branches) for i in (0, 1))


def exact_expectations(c: Case) -> dict[str, tuple[Fraction, Fraction]]:
    """Goal-path formulas, independent of the event simulator and policy code."""
    f = lambda length: path_result(length, c.budget)
    h, q, m = c.overhead, c.q, c.miss
    direct = average([(q, f(c.a + c.b)), (1 - q, f(2 * c.a + c.safe))])
    inspected_open = f(h + c.a + c.b)
    inspected_closed = f(h + c.safe)
    missed_direct = average([(q, inspected_open), (1 - q, f(h + 2 * c.a + c.safe))])
    # This choice is an expectation over public q, not a choice per hidden world.
    missed_best = min((missed_direct, inspected_closed), key=lambda result: result[0])
    reactive = average([(q, inspected_open), ((1 - q) * (1 - m), inspected_closed),
                        ((1 - q) * m, f(h + 2 * c.a + c.safe))])
    best_miss = average([((1 - m) * q, inspected_open),
                        ((1 - m) * (1 - q), inspected_closed), (m, missed_best)])
    return {"DIRECT": direct, "INSPECT_REACTIVE": reactive,
            "INSPECT_BEST_MISS": best_miss, "ALWAYS_SAFE": f(c.safe)}


def continuation_action(snapshot: Snapshot, policy: str) -> int:
    """Only public inputs. The hypothetical world is not an argument."""
    if policy == "ALWAYS_SAFE" or snapshot.last[0] == 0:
        return 6
    if snapshot.last[0] == 1 or policy in ("DIRECT", "INSPECT_REACTIVE"):
        return 0
    if policy != "INSPECT_BEST_MISS":
        raise ValueError(policy)
    # No evidence after the first inspection: compare navigation expectations
    # using the current budget and required return from the physical viewpoint.
    back = snapshot.layout.return_cost(snapshot.position)
    a, b, safe = snapshot.layout.a[0], snapshot.layout.b[0], snapshot.layout.safe
    q = snapshot.belief[0]
    def penalized(length):
        return min(length, snapshot.remaining) + 200 * (length > snapshot.remaining)
    direct = q * penalized(back + a + b) + (1 - q) * penalized(back + 2 * a + safe)
    safe_cost = penalized(back + safe)
    return 0 if direct <= safe_cost else 6


def simulate_branch(c: Case, policy: str, world: tuple[bool, ...], missed: bool):
    """Finite hypothetical branch, not a generated scenario or episode dataset."""
    if policy not in POLICIES:
        raise ValueError(policy)
    snapshot = c.snapshot()
    action = 6 if policy == "ALWAYS_SAFE" else (3 if policy.startswith("INSPECT") else 0)
    elapsed = 0.0
    trace = []
    for step in range(3):
        result = physical_step(snapshot, action, world, missed if step == 0 else False)
        elapsed += result.dt
        trace.append({"action": action, "position_before": snapshot.position,
                      "position_after": result.snapshot.position, "dt": result.dt,
                      "evidence": result.evidence})
        if result.ended:
            return elapsed + 200 * result.failure, int(result.failure), trace
        snapshot = result.snapshot
        action = continuation_action(snapshot, policy)
    raise AssertionError("Comparator exceeded inspect, direct, safe sequence")


def enumeration(c: Case, policy: str) -> tuple[float, float]:
    cost = failure = mass = 0.0
    for world in product((False, True), repeat=3):
        world_weight = (c.q if world[0] else 1 - c.q) / 4
        for missed, miss_weight in ((False, 1 - c.miss), (True, c.miss)):
            weight = float(world_weight * miss_weight)
            if not weight:
                continue
            branch_cost, branch_failure, _ = simulate_branch(c, policy, world, missed)
            cost += weight * branch_cost
            failure += weight * branch_failure
            mass += weight
    if not math.isclose(mass, 1.0, abs_tol=1e-12):
        raise AssertionError("World/sensing weights do not sum to one")
    return cost, failure


def case_checks(plan: dict) -> list[dict]:
    rows = []
    for entry in plan["hand_cases"]:
        c = Case.from_dict(entry["parameters"])
        results = exact_expectations(c)
        reported = {}
        for name in POLICIES:
            exact_cost, exact_failure = results[name]
            assert exact_cost == Fraction(entry["hand_expected_penalized_cost"][name]), (entry["id"], name)
            assert exact_failure == Fraction(entry["hand_expected_failure_probability"][name]), (entry["id"], name)
            actual_cost, actual_failure = enumeration(c, name)
            assert math.isclose(actual_cost, float(exact_cost), rel_tol=0, abs_tol=1e-10), (entry["id"], name, actual_cost, exact_cost)
            assert math.isclose(actual_failure, float(exact_failure), rel_tol=0, abs_tol=1e-12)
            reported[name] = {"exact_penalized_cost": str(exact_cost),
                              "exact_failure_probability": str(exact_failure),
                              "exact_expected_raw_reward": str(100 - exact_cost),
                              "physical_kernel_expected_cost": actual_cost}
        # Frozen VOI's continuation is explicitly the reactive one, not best-miss.
        scores = voi_costs(c.snapshot())
        for action, name in ((0, "DIRECT"), (3, "INSPECT_REACTIVE"), (6, "ALWAYS_SAFE")):
            assert math.isclose(scores[action], float(results[name][0]), rel_tol=0, abs_tol=1e-10)
        for action in (1, 2, 4, 5):
            assert math.isclose(scores[action], float(c.budget + 200), rel_tol=0, abs_tol=1e-10)
        rows.append({"id": entry["id"], "role": "analytical_design_case_only", "policies": reported})
    return rows


def box_certificates(plan: dict) -> list[dict]:
    """Exact endpoint proof for multi-affine no-failure cost differences."""
    distribution = plan["prospective_distribution"]
    rows = []
    for family, bounds in distribution["families"].items():
        keys = tuple(bounds)
        margins, slacks = [], []
        count = 0
        for scale_text in distribution["cost_scales"]:
            scale = Fraction(scale_text)
            for miss_text in distribution["miss_probabilities"]:
                for values in product(*(bounds[key] for key in keys)):
                    c = Case.from_dict({**dict(zip(keys, values)), "budget": distribution["base_budget"],
                                        "other_cost": distribution["base_dominated_cost"], "miss": miss_text}).scaled(scale)
                    assert c.a + c.b < c.safe <= c.budget < c.other_cost
                    # Includes the longest miss -> blocked -> safe physical path.
                    slack = c.budget - (c.overhead + 2 * c.a + c.safe)
                    assert slack > 0
                    costs = exact_expectations(c)
                    assert all(failure == 0 for _, failure in costs.values())
                    direct = costs["DIRECT"][0]
                    perfect_lower = c.overhead + c.q * (c.a + c.b) + (1 - c.q) * c.safe
                    if family == "direct":
                        margin = min(c.safe, perfect_lower) - direct
                    elif family == "inspection":
                        margin = min(direct, c.safe) - costs["INSPECT_REACTIVE"][0]
                    elif family == "safe":
                        margin = min(direct, perfect_lower) - c.safe
                    else:
                        raise ValueError(family)
                    assert margin > 0
                    margins.append(margin)
                    slacks.append(slack)
                    count += 1
        minimum = min(margins)
        assert minimum == Fraction(distribution["required_minimum_margin_by_family_at_lower_scale"][family])
        rows.append({"family": family, "endpoint_checks": count,
                     "exact_minimum_strict_first_action_margin": str(minimum),
                     "exact_minimum_longest_path_budget_slack": str(min(slacks)),
                     "scope": "Multi-affine primary box certificate, not dataset performance"})
    return rows


def protocol_checks(plan: dict) -> dict:
    assert plan["status"] == "DRAFT_REQUIRES_INDEPENDENT_REVIEW"
    assert plan["authorization_now"] == {
        "analytical_checks_only": True, "dataset_generation": False, "training": False,
        "heldout_evaluation": False, "original_package_writes": False}
    inputs = plan["frozen_inputs"]
    for stem in ("protocol", "environment", "policies"):
        assert digest(ROOT / inputs[f"{stem}_path"]) == inputs[f"{stem}_sha256"]
    review = ROOT / plan["motivation"]["review_path"]
    assert digest(review) == plan["motivation"]["review_sha256"]
    original = source_manifest()
    assert original["source_sha256"] == inputs["original_source_manifest_sha256"]
    seeds = plan["prospective_seeds_and_seal"]
    assert canonical_digest(seeds["heldout_reservation"]) == seeds["heldout_reservation_sha256"]
    namespaces = [seeds[k] for k in ("design_namespace", "training_namespace", "validation_namespace")]
    namespaces.append(seeds["heldout_reservation"]["namespace"])
    assert len(set(namespaces)) == 4
    assert all(n.startswith("comp597/stale-nav/followup-safe-balance-v1/") for n in namespaces)
    assert len(set(seeds["independent_training_seeds"])) == 5
    assert not set(seeds["independent_training_seeds"]) & set(range(5))
    assert plan["fair_horizons"]["primary_tasks_per_episode"] == 1
    train = plan["planned_training_not_authorized"]
    assert train["gamma"] == 1 and train["gae_lambda"] == .95 and train["reward_scale"] == .01
    assert train["total_final_training_steps"] == 2 * 5 * train["total_env_steps_per_run"]
    assert train["total_env_steps_per_run"] % (train["num_envs"] * train["rollout_steps"]) == 0
    assert plan["planned_evaluation_not_authorized"]["total_independent_heldout_episodes"] == 3 * 2 * 2 * 128
    return {"original_source_manifest_sha256": original["source_sha256"],
            "heldout_reservation_sha256": seeds["heldout_reservation_sha256"],
            "heldout_generated": False, "training_performed": False,
            "draft_status": plan["status"]}


def protected_check(baseline: Path) -> dict:
    files = json.loads(baseline.read_text())
    changed = [name for name, expected in files.items() if not (ROOT / name).is_file() or digest(ROOT / name) != expected]
    if changed:
        raise AssertionError(f"Protected original files changed: {changed}")
    return {"baseline_sha256": digest(baseline), "protected_file_count": len(files), "changed": []}


def run(output: Path, plan_path: Path, protected_baseline: Path):
    allowed = (ROOT.parent / "local-materials/followup").resolve()
    output = output.resolve()
    if output == allowed or not output.is_relative_to(allowed):
        raise ValueError("Evidence must go into a NEW subdirectory of ../local-materials/followup")
    output.mkdir(parents=True, exist_ok=False)
    plan = load_plan(plan_path)
    checks = protocol_checks(plan)
    before = protected_check(protected_baseline)
    report = {"kind": "prospective_analytical_validation_only", "checks": checks,
              "hand_cases": case_checks(plan), "interval_certificates": box_certificates(plan),
              "protected_before": before, "protected_after": protected_check(protected_baseline),
              "dataset_generation_performed": False, "heldout_evaluation_performed": False,
              "original_submission_modified": False}
    files = {name: digest(ROOT / name) for name in NEW_FILES}
    archive = output / "followup-source.tar"
    with tarfile.open(archive, "x", format=tarfile.USTAR_FORMAT) as tar:
        for name in sorted(files):
            data = (ROOT / name).read_bytes()
            item = tarfile.TarInfo(name)
            item.size, item.mode, item.mtime = len(data), 0o444, 0
            tar.addfile(item, io.BytesIO(data))
    archive.chmod(0o444)
    provenance = {"new_files_sha256": files, "followup_manifest_sha256": canonical_digest(files),
                  "original_inputs": plan["frozen_inputs"], "audit_sha256": plan["motivation"]["review_sha256"],
                  "command_argv": [sys.executable, *sys.argv], "python": sys.version,
                  "archive_sha256": digest(archive), "no_new_dependencies": True}
    for name, content in (("analytical-results.json", report), ("provenance.json", provenance)):
        (output / name).write_text(json.dumps(content, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(output), "design_cases": len(report["hand_cases"]),
                      "certificates": report["interval_certificates"], **checks,
                      "followup_manifest_sha256": provenance["followup_manifest_sha256"]}, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, default=PLAN_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--protected-baseline", type=Path, required=True)
    args = parser.parse_args()
    run(args.output, args.plan, args.protected_baseline)


if __name__ == "__main__":
    main()
