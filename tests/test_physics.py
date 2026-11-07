from dataclasses import replace
import numpy as np
import pytest

from stale_nav.environment import (Environment, Layout, Scenario, initial_snapshot, make_scenario,
                                   next_task, physical_step, sensing_miss)
from stale_nav.features import encode
from stale_nav.policies import reactive, voi_costs


@pytest.fixture
def layout():
    return Layout((10., 12., 14.), (1., 2., 3.), (1., 2., 2.5), 1., 25., 60.)


def scenario(layout, worlds=None, flip=.2, miss=0.):
    return Scenario("pilot/fixture", "pilot", "fixture", layout,
                    worlds or ((True, False, True),) * 6, flip, miss)


@pytest.mark.parametrize("position,back", [(0, 0), (1, 10), (2, 12), (3, 14), (4, 1), (5, 2), (6, 2.5)])
@pytest.mark.parametrize("action", range(7))
def test_forced_hub_return_exactly_once(layout, position, back, action):
    s = replace(initial_snapshot(layout, .2, 0), position=position)
    e = physical_step(s, action, (False, True, False))
    if action < 3:
        cost = layout.a[action] + (layout.b[action] if action == 1 else 0)
        assert e.snapshot.position == (7 if action == 1 else action + 1)
    elif action < 6:
        cost = layout.v[action - 3] + layout.sensing
        assert e.snapshot.position == action + 1
    else:
        cost = layout.safe
        assert e.snapshot.position == 7
    assert e.dt == back + cost
    assert e.snapshot.remaining == 60 - e.dt


@pytest.mark.parametrize("budget,action,evidence,success", [
    (9.9, 0, (), False), (10., 0, ((0, True),), False),
    (10.5, 0, ((0, True),), False), (11., 0, ((0, True),), True),
    (.9, 3, (), False), (1., 3, (), False), (1.9, 3, (), False),
    (2., 3, ((0, True),), False), (24.9, 6, (), False), (25., 6, (), True),
])
def test_exact_equality_and_unpaid_evidence(layout, budget, action, evidence, success):
    s = replace(initial_snapshot(layout, 0, 0), remaining=budget)
    e = physical_step(s, action, (True, False, False))
    assert e.dt == pytest.approx(budget)
    assert e.success == success
    assert e.failure == (not success)
    assert e.evidence == evidence
    assert e.snapshot.last[0] == (1 if evidence else -1)


def test_blocked_exact_arrival_no_crossing(layout):
    e = physical_step(replace(initial_snapshot(layout, 0, 0), remaining=10), 0, (False,) * 3)
    assert e.evidence == ((0, False),)
    assert e.failure and e.blocked_attempt == 1 and e.dt == 10


def test_partial_return_does_not_reveal_target(layout):
    s = replace(initial_snapshot(layout, 0, 0), position=1, remaining=9)
    for action in range(7):
        e = physical_step(s, action, (True,) * 3)
        assert e.dt == 9 and e.failure and e.evidence == ()


def test_missing_evidence_preserves_history(layout):
    s = physical_step(initial_snapshot(layout, .2, 1), 3, (True, False, False)).snapshot
    s = next_task(s)
    before = (s.last, s.belief, s.age, s.observed)
    e = physical_step(s, 3, (False,) * 3, missed=True)
    assert (e.snapshot.last, e.snapshot.belief, e.snapshot.age, e.snapshot.observed) == before
    assert e.snapshot.attempted == (True, False, False)
    assert e.evidence == ()
    assert e.dt == 2


def test_boundaries_and_final_termination(layout):
    env = Environment(scenario(layout, worlds=((True, False, True), (False, True, True)) * 3))
    s, reward, done, info = env.step(0)
    assert not done and info["success"] and reward == 89
    assert s.task == 1 and s.position == 0 and s.remaining == layout.budget
    assert s.belief == pytest.approx((.8, .5, .5))
    assert s.age == (1, -1, -1) and not any(s.observed) and not any(s.attempted)
    e = env.step(3)
    assert e[0].last[0] == 0 and e[0].belief[0] == 0
    assert e[0].task == 1
    # No within-task state change or belief drift: repeated physical checks agree.
    assert env.step(3)[0].belief[0] == 0
    env.step(6)
    for _ in range(3):
        env.step(6)
    s, _, done, _ = env.step(0)  # task five has door zero blocked
    assert not done and s.task == 5
    s, _, done, info = env.step(6)
    assert done and info["success"] and info["task_record"]["time"] == 45


def test_final_does_not_propagate_and_new_episode_clears(layout):
    env = Environment(scenario(layout))
    for _ in range(5):
        s, _, done, _ = env.step(6)
        assert not done
    s, _, done, info = env.step(0)
    assert done and info["success"] and s.task == 5
    assert s.belief[0] == 1 and s.age[0] == 0 and s.observed[0]
    with pytest.raises(RuntimeError):
        env.step(6)
    fresh = Environment(scenario(layout)).snapshot
    assert fresh.last == (-1,) * 3 and fresh.age == (-1,) * 3 and fresh.belief == (.5,) * 3


def test_failure_boundary_keeps_arrival_evidence(layout):
    # Exact arrival, no crossing budget, then propagate once into task one.
    env = Environment(scenario(replace(layout, budget=10)))
    s, reward, done, info = env.step(0)
    assert reward == -110 and not done and info["failure"]
    assert s.belief == pytest.approx((.8, .5, .5))
    assert s.last[0] == 1 and s.age[0] == 1 and s.remaining == 10


@pytest.mark.parametrize("p,q,expected", [(0, 1, 1), (.2, 1, .8), (.2, 0, .2), (.2, .5, .5), (1, .8, .2)])
def test_exact_flip_belief(layout, p, q, expected):
    s = replace(initial_snapshot(layout, p, 0), belief=(q,) * 3)
    assert next_task(s).belief == pytest.approx((expected,) * 3)


def test_hidden_changes_cannot_change_public_inputs_or_candidate_costs(layout):
    env_a = Environment(scenario(layout, ((True,) * 3,) * 6))
    env_b = Environment(scenario(layout, ((False,) * 3,) * 6))
    assert env_a.snapshot == env_b.snapshot
    for variant in ("full", "masked"):
        np.testing.assert_array_equal(encode(env_a.snapshot, variant), encode(env_b.snapshot, variant))
    np.testing.assert_array_equal(voi_costs(env_a.snapshot), voi_costs(env_b.snapshot))
    assert reactive(env_a.snapshot) == reactive(env_b.snapshot)
    # A completed inspection is the first permitted distinction.
    assert env_a.step(3)[0].last != env_b.step(3)[0].last


def test_paired_policies_different_action_counts():
    sc = make_scenario("pilot", 7, "changing_missed")
    a, b = Environment(sc), Environment(sc)
    for task in range(6):
        # A takes one decision; B inspects a target then navigates.
        s_b, _, _, info = b.step(3)
        assert info["evidence"] == (() if sensing_miss(sc, task, 0, 0) else ((0, sc.worlds[task][0]),))
        a.step(6)
        b.step(6)
        assert a.snapshot.task == b.snapshot.task
    assert a._scenario.worlds == b._scenario.worlds
    assert make_scenario("pilot", 7, "changing_missed") == sc


def test_independent_keyed_misses_and_namespaces():
    sc = make_scenario("pilot", 7, "changing_missed")
    draws = [sensing_miss(sc, 2, 1, i) for i in range(100)]
    for i in range(100):
        sensing_miss(sc, 0, 0, i)
    assert draws == [sensing_miss(sc, 2, 1, i) for i in range(100)]
    assert any(draws) and not all(draws)
    scenarios = [make_scenario(ns, 0, "changing_clean") for ns in ("train", "val", "test")]
    assert len({s.scenario_id for s in scenarios}) == 3
    assert len({s.layout for s in scenarios}) == 3


def test_no_incidental_observations(layout):
    s = physical_step(initial_snapshot(layout, 0, 0), 6, (True,) * 3).snapshot
    assert s.last == (-1,) * 3
    s = physical_step(initial_snapshot(layout, 0, 0), 0, (False, True, True)).snapshot
    assert s.last == (0, -1, -1)


def test_generated_layout_bounds_and_static_worlds():
    for index in range(50):
        sc = make_scenario('train/seed-0', index, 'static_clean')
        l = sc.layout
        assert 20 <= l.safe <= 30
        assert all(6 <= x <= 14 for x in l.a)
        assert all(1 <= x <= 4 for x in l.b)
        assert all(.5 <= x <= 2.5 for x in l.v)
        assert .5 <= l.sensing <= 2
        assert 1.1 <= l.budget / l.safe <= 1.6
        assert len(set(sc.worlds)) == 1
        assert l.safe <= l.budget <= 48


@pytest.mark.parametrize('position', [0, 1, 4])
@pytest.mark.parametrize('action', [0, 3])
def test_decimal_exact_event_budget(position, action):
    l = Layout((6.1, 8., 9.), (1.1, 2., 3.), (1.1, 1.5, 2.), 1.2, 25., 40.)
    back = l.return_cost(position)
    total = (back + l.a[0] + l.b[0]) if action == 0 else (back + l.v[0] + l.sensing)
    s = replace(initial_snapshot(l, 0, 0), position=position, remaining=total)
    event = physical_step(s, action, (True, False, False))
    assert event.evidence == ((0, True),)
    assert event.success == (action == 0)
    assert event.failure == (action == 3)
    assert event.snapshot.remaining == 0
    assert event.dt == total


@pytest.mark.parametrize('action', [0, 3])
@pytest.mark.parametrize('shortfall', [1e-8, 1e-10, '64_ulps'])
def test_decimal_genuine_shortfall_never_completes_event(action, shortfall):
    import math
    l = Layout((6.1, 8., 9.), (1.1, 2., 3.), (1.1, 1.5, 2.), 1.2, 25., 40.)
    total = l.a[0] + l.b[0] if action == 0 else l.v[0] + l.sensing
    budget = total - (64 * math.ulp(total) if shortfall == '64_ulps' else shortfall)
    s = replace(initial_snapshot(l, 0, 0), remaining=budget)
    event = physical_step(s, action, (True, False, False))
    assert event.failure and not event.success
    # A completed approach still provides evidence; incomplete sensing does not.
    assert event.evidence == (((0, True),) if action == 0 else ())
    assert event.snapshot.remaining == 0 and event.dt == budget


@pytest.mark.parametrize('action,arrival', [(0, 6.1), (3, 1.1)])
def test_outside_tolerance_before_arrival_has_no_evidence(action, arrival):
    import math
    l = Layout((6.1, 8., 9.), (1.1, 2., 3.), (1.1, 1.5, 2.), 1.2, 25., 40.)
    budget = arrival - 64 * math.ulp(arrival)
    event = physical_step(replace(initial_snapshot(l, 0, 0), remaining=budget), action, (True,) * 3)
    assert event.failure and event.evidence == () and event.dt == budget


# Budgets are independent decimal literals, not expressions built from costs.
LITERAL_EVENT_CASES = [
    (0, 0, 7.3), (1, 0, 13.5), (2, 0, 15.3), (4, 0, 7.9),
    (0, 3, 1.7), (1, 3, 7.9), (2, 3, 9.7), (4, 3, 2.3),
]


@pytest.mark.parametrize('position,action,budget', LITERAL_EVENT_CASES)
@pytest.mark.parametrize('offset_ulps', [-2, 0, 2])
def test_independent_literal_budgets_complete_within_equality_bound(position, action, budget, offset_ulps):
    import math
    l = Layout((6.2, 8., 9.), (1.1, 2., 3.), (.6, 1.5, 2.), 1.1, 25., 40.)
    available = budget + offset_ulps * math.ulp(budget)
    s = replace(initial_snapshot(l, 0, 0), position=position, remaining=available)
    event = physical_step(s, action, (True, False, False))
    assert event.evidence == ((0, True),)
    assert event.success == (action == 0)
    assert event.failure == (action == 3)
    assert event.snapshot.remaining == 0
    assert event.dt == available
    assert 0 <= event.dt <= available


@pytest.mark.parametrize('position,action,budget', LITERAL_EVENT_CASES)
@pytest.mark.parametrize('shortfall', ['64_ulps', 1e-10])
def test_independent_literal_budgets_outside_bound_fail(position, action, budget, shortfall):
    import math
    l = Layout((6.2, 8., 9.), (1.1, 2., 3.), (.6, 1.5, 2.), 1.1, 25., 40.)
    available = budget - (64 * math.ulp(budget) if shortfall == '64_ulps' else shortfall)
    s = replace(initial_snapshot(l, 0, 0), position=position, remaining=available)
    event = physical_step(s, action, (True, False, False))
    assert event.failure and not event.success
    assert event.evidence == (((0, True),) if action == 0 else ())
    assert event.snapshot.remaining == 0 and event.dt == available
    assert 0 <= event.dt <= available


def test_literal_goal_budget_advances_task_as_success():
    l = Layout((6.2, 8., 9.), (1.1, 2., 3.), (.6, 1.5, 2.), 1.1, 25., 7.3)
    env = Environment(scenario(l))
    snapshot, reward, done, info = env.step(0)
    assert info['success'] and not info['failure'] and not done
    assert info['dt'] == 7.3 and reward == 100 - 7.3
    assert snapshot.task == 1 and snapshot.position == 0 and snapshot.remaining == 7.3
    assert snapshot.last[0] == 1 and snapshot.belief[0] == pytest.approx(.8)
