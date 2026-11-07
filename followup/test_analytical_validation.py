"""Isolated prospective analytical tests. Never collect the original test suite."""
from dataclasses import replace
from fractions import Fraction as F
from itertools import product
from pathlib import Path
import copy
import inspect
import json
import math
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analytical_validation as av


@pytest.fixture
def plan():
    return av.load_plan()


def design_case(plan, name):
    entry = next(c for c in plan["hand_cases"] if c["id"] == name)
    return av.Case.from_dict(entry["parameters"])


@pytest.mark.parametrize("name,expected", [
    ("direct_clean", (19, 23, 23, 25)),
    ("inspection_clean", (28, 21, 21, 25)),
    ("safe_clean", (29, 23, 23, 20)),
    ("direct_missed", (19, 24, 24, 25)),
    ("inspection_missed", (28, F(47, 2), F(91, 4), 25)),
    ("safe_missed", (29, 26, F(95, 4), 20)),
    ("tight_inspection_clean", (F(241, 2), 21, 21, 25)),
    ("tight_inspection_missed", (F(241, 2), F(185, 4), F(91, 4), 25)),
    ("inspection_exact_return_budget", (F(239, 2), 21, 21, 25)),
    ("inspection_return_timeout", (119, F(241, 2), F(241, 2), 25)),
])
def test_independent_literal_hand_costs(plan, name, expected):
    case = design_case(plan, name)
    results = av.exact_expectations(case)
    assert tuple(results[p][0] for p in av.POLICIES) == tuple(map(F, expected))
    for policy in av.POLICIES:
        actual = av.enumeration(case, policy)
        assert actual == pytest.approx(tuple(map(float, results[policy])), abs=1e-10)


def test_stored_derivations_and_all_seven_voi_actions(plan):
    assert len(av.case_checks(plan)) == 10


def test_three_robust_first_action_certificates(plan):
    rows = av.box_certificates(plan)
    assert {r["family"]: r["exact_minimum_strict_first_action_margin"] for r in rows} == {
        "direct": "64/25", "inspection": "2/5", "safe": "34/25"}
    assert all(r["endpoint_checks"] == 128 for r in rows)
    assert all(F(r["exact_minimum_longest_path_budget_slack"]) > 0 for r in rows)


@pytest.mark.parametrize("name", ["direct_clean", "inspection_clean", "safe_clean", "direct_missed",
                                  "inspection_missed", "safe_missed"])
@pytest.mark.parametrize("scale", [F(4, 5), F(6, 5)])
def test_lower_upper_cost_regimes_preserve_primary_rankings(plan, name, scale):
    original = design_case(plan, name)
    c = original.scaled(scale)
    before, after = av.exact_expectations(original), av.exact_expectations(c)
    for policy in av.POLICIES:
        assert after[policy] == (before[policy][0] * scale, 0)
        assert av.enumeration(c, policy) == pytest.approx((float(after[policy][0]), 0), abs=1e-10)


@pytest.mark.parametrize("scale", [F(4, 5), F(6, 5)])
def test_time_scaling_does_not_scale_failure_payout(plan, scale):
    original = design_case(plan, "tight_inspection_missed")
    before, after = av.exact_expectations(original), av.exact_expectations(original.scaled(scale))
    for policy in av.POLICIES:
        cost, failure = before[policy]
        assert after[policy] == ((cost - 200 * failure) * scale + 200 * failure, failure)
        assert av.enumeration(original.scaled(scale), policy) == pytest.approx(tuple(map(float, after[policy])), abs=1e-10)


@pytest.mark.parametrize("world,expected_actions,expected_dt", [
    ((True, False, False), [3, 0], [2, 12]),
    ((False, False, False), [3, 6], [2, 26]),
])
def test_inspection_must_physically_return_from_viewpoint(plan, world, expected_actions, expected_dt):
    c = design_case(plan, "inspection_clean")
    cost, failure, trace = av.simulate_branch(c, "INSPECT_REACTIVE", world, False)
    assert [r["action"] for r in trace] == expected_actions
    assert [r["dt"] for r in trace] == expected_dt
    assert trace[0]["position_after"] == 4
    assert trace[1]["position_before"] == 4
    assert cost == sum(expected_dt) and failure == 0


def test_miss_preserves_unknown_and_blocked_path_pays_both_returns(plan):
    c = design_case(plan, "inspection_missed")
    snapshot = c.snapshot()
    event = av.physical_step(snapshot, 3, (False, True, True), True)
    assert event.evidence == ()
    assert event.snapshot.last == snapshot.last
    assert event.snapshot.belief == snapshot.belief
    assert av.continuation_action(event.snapshot, "INSPECT_BEST_MISS") == 6
    assert av.continuation_action(event.snapshot, "INSPECT_REACTIVE") == 0
    cost, failure, trace = av.simulate_branch(c, "INSPECT_REACTIVE", (False, True, True), True)
    assert [r["action"] for r in trace] == [3, 0, 6]
    assert [r["dt"] for r in trace] == [2, 11, 35]
    assert (cost, failure) == (48, 0)


def test_return_budget_negative_control_rejects_teleport(plan):
    bad = design_case(plan, "inspection_return_timeout")
    good = design_case(plan, "inspection_exact_return_budget")
    cost, failed, trace = av.simulate_branch(bad, "INSPECT_REACTIVE", (False,) * 3, False)
    assert cost == 227 and failed == 1
    assert trace[0]["evidence"] == ((0, False),)
    assert sum(t["dt"] for t in trace) == 27
    assert av.simulate_branch(good, "INSPECT_REACTIVE", (False,) * 3, False)[:2] == (28, 0)


def test_partial_sensing_never_reveals_unpaid_evidence(plan):
    c = design_case(plan, "inspection_clean")
    event = av.physical_step(replace(c.snapshot(), remaining=1.5), 3, (True,) * 3)
    assert event.failure and event.evidence == () and event.dt == 1.5


@pytest.mark.parametrize("position,expected", [(0, 25), (1, 35), (4, 26)])
def test_always_safe_is_public_and_pays_return(plan, position, expected):
    c = design_case(plan, "inspection_clean")
    snapshot = replace(c.snapshot(), position=position)
    assert av.continuation_action(snapshot, "ALWAYS_SAFE") == 6
    a = av.physical_step(snapshot, 6, (False,) * 3)
    b = av.physical_step(snapshot, 6, (True,) * 3)
    assert a == b and a.dt == expected and a.success and a.evidence == ()


def test_irrelevant_hidden_states_cannot_change_policy_trace(plan):
    c = design_case(plan, "inspection_missed")
    assert tuple(inspect.signature(av.continuation_action).parameters) == ("snapshot", "policy")
    for policy in av.POLICIES:
        for opened, missed in product((False, True), repeat=2):
            reference = av.simulate_branch(c, policy, (opened, False, False), missed)
            for other in product((False, True), repeat=2):
                assert av.simulate_branch(c, policy, (opened, *other), missed) == reference


def test_no_scenario_generator_or_random_draw_is_used(plan, monkeypatch):
    import numpy as np
    import stale_nav.environment as environment
    def forbidden(*args, **kwargs):
        raise AssertionError("Dataset generation or random draw is forbidden in this task")
    monkeypatch.setattr(environment, "make_scenario", forbidden)
    monkeypatch.setattr(environment, "Environment", forbidden)
    monkeypatch.setattr(np.random, "default_rng", forbidden)
    av.protocol_checks(plan)
    av.case_checks(plan)
    av.box_certificates(plan)


def test_no_old_result_dataset_read(plan, monkeypatch):
    original = Path.read_bytes
    def guarded(path):
        assert "experiment-results" not in path.parts and "runs" not in path.parts
        return original(path)
    monkeypatch.setattr(Path, "read_bytes", guarded)
    av.protocol_checks(plan)
    av.case_checks(plan)


def test_heldout_reservation_commitment_and_no_execution_authority(plan):
    checked = av.protocol_checks(plan)
    assert checked["draft_status"] == "DRAFT_REQUIRES_INDEPENDENT_REVIEW"
    assert checked["heldout_generated"] is checked["training_performed"] is False
    changed = copy.deepcopy(plan)
    changed["prospective_seeds_and_seal"]["heldout_reservation"]["master_seed"] += 1
    with pytest.raises(AssertionError):
        av.protocol_checks(changed)


def test_inspection_advantage_is_not_claim_of_reactive_optimality(plan):
    c = design_case(plan, "tight_inspection_missed")
    scores = av.exact_expectations(c)
    assert scores["INSPECT_REACTIVE"][0] > scores["ALWAYS_SAFE"][0]
    assert scores["INSPECT_BEST_MISS"][0] < scores["ALWAYS_SAFE"][0]
    assert scores["INSPECT_REACTIVE"][1] == F(1, 8)
    assert scores["INSPECT_BEST_MISS"][1] == 0


def test_outputs_outside_followup_evidence_directory_are_refused(tmp_path):
    # pytest's own temp directory is deliberately inside the allowed evidence
    # tree. Pick a sibling outside that tree; run() must refuse before writing.
    forbidden = av.ROOT.parent / "local-materials/outside-followup-analytical-test-refused"
    assert not forbidden.exists()
    with pytest.raises(ValueError, match="NEW subdirectory"):
        av.run(forbidden, av.PLAN_PATH, tmp_path / "unused")
    assert not forbidden.exists()
