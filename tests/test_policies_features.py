from dataclasses import fields, replace
import json
import numpy as np
import pytest
import torch

from stale_nav.environment import Layout, Snapshot, initial_snapshot, next_task, physical_step
from stale_nav.features import FEATURE_NAMES, MASKED_COLUMNS, MASK_INDICES, encode, feature_spec
from stale_nav.policies import continuation_cost, reactive, reset, ttl, voi, voi_costs, world_weights
from stale_nav.ppo import ActorCritic, seed_everything
from stale_nav.provenance import ROOT


@pytest.fixture
def hand():
    return initial_snapshot(Layout((10, 100, 100), (1, 100, 100), (1, 100, 100), 1, 25, 60), 0, 0)


def test_hand_computable_voi(hand):
    costs = voi_costs(hand)
    assert costs[[0, 6, 3]] == pytest.approx([28, 25, 21])
    assert voi(hand) == 3
    costly = replace(hand, layout=replace(hand.layout, sensing=30))
    assert voi(costly) == 6
    assert voi_costs(costly)[3] > 25


def test_voi_all_missed_and_known_static(hand):
    missed = replace(hand, miss=1)
    # Travel/sense/return wastes three units; then a .5 open route costs 28.
    assert voi_costs(missed)[3] == pytest.approx(31)
    assert voi(missed) == 6
    known = replace(hand, belief=(1, .5, .5), last=(1, -1, -1), age=(0, -1, -1))
    assert voi(known) == 0
    assert voi_costs(known)[[0, 3]] == pytest.approx([11, 14])


def test_voi_weights_and_continuation_no_oracle(hand):
    weights = world_weights((.2, .7, .9))
    assert len(weights) == 8 and sum(w for _, w in weights) == pytest.approx(1)
    assert dict(weights)[(True, False, True)] == pytest.approx(.2 * .3 * .9)
    # An oracle would choose safe for 25. Reactive must first discover closure.
    assert continuation_cost(hand, (False, False, False)) == 45
    assert continuation_cost(hand, (True, False, False)) == 11


def test_voi_budget_failure_and_position(hand):
    tight = replace(hand, remaining=10)
    assert voi_costs(tight)[0] == pytest.approx(210)  # evidence at arrival, no crossing
    at_view = replace(hand, position=4)
    assert voi_costs(at_view)[6] == pytest.approx(26)


def test_navigation_ties_before_inspection(monkeypatch, hand):
    import stale_nav.policies as policies
    monkeypatch.setattr(policies, "voi_costs", lambda s: np.array([30, 30, 30, 21, 22, 23, 21]))
    assert voi(hand) == 6
    monkeypatch.setattr(policies, "voi_costs", lambda s: np.array([21, 21, 30, 21, 22, 23, 21]))
    assert voi(hand) == 0
    equal = replace(hand, layout=replace(hand.layout, a=(10, 10, 10), b=(15, 15, 15)))
    assert reactive(equal) == 0


def test_route_conditioned_ttl_attempt_miss_and_unknown(hand):
    assert ttl(hand, 1) == 0  # unknown age is not stale
    opened = physical_step(hand, 3, (True, False, False)).snapshot
    stale = next_task(opened)
    assert ttl(stale, 1) == 3
    missed = physical_step(stale, 3, (False, False, False), missed=True).snapshot
    assert ttl(missed, 1) == 0
    closed = replace(stale, last=(0, -1, -1))
    assert ttl(closed, 1) == 6  # frozen TTL never rechecks remembered-closed doors
    assert ttl(stale, 2) == 0
    with pytest.raises(ValueError):
        ttl(stale, 0)


def test_reset_ignores_prior_tasks_only(hand):
    closed = physical_step(hand, 0, (False, False, False)).snapshot
    assert reactive(closed) == reset(closed) == 6
    old = next_task(closed)
    assert reactive(old) == 6 and reset(old) == 0


def test_explicit_mask_and_fixed_analytic_scales():
    declared = json.loads((ROOT / "configs/features.json").read_text())
    assert declared == json.loads(json.dumps(feature_spec()))
    s = initial_snapshot(Layout((6, 8, 14), (1, 2, 4), (.5, 1, 2.5), 2, 30, 48), .2, .25)
    s = replace(s, last=(1, 0, -1), age=(5, 2, -1), remaining=24)
    full, masked = encode(s), encode(s, "masked")
    assert len(full) == len(masked) == len(FEATURE_NAMES)
    assert np.all(np.abs(full) <= 1)
    assert np.all(masked[list(MASK_INDICES)] == 0)
    kept = [i for i in range(len(full)) if i not in MASK_INDICES]
    np.testing.assert_array_equal(full[kept], masked[kept])
    assert masked[FEATURE_NAMES.index("remaining_budget")] == .5
    assert masked[FEATURE_NAMES.index("flip_probability")] == pytest.approx(.2)
    assert masked[FEATURE_NAMES.index("miss_probability")] == .25
    assert "initial_task_budget" in MASKED_COLUMNS
    assert not ({"scenario_id", "seed", "worlds", "hidden"} & {f.name for f in fields(Snapshot)})
    # Unknown and known age zero differ even though normalized age itself is zero.
    unknown = encode(replace(s, last=(-1,) * 3, age=(-1,) * 3))
    observed = encode(replace(s, last=(0,) * 3, age=(0,) * 3))
    assert not np.array_equal(unknown, observed)


def test_actor_and_critic_receive_identical_masked_tensor(hand):
    seed_everything(9)
    model = ActorCritic()
    received = []
    hooks = [net.register_forward_pre_hook(lambda module, args: received.append(args[0].clone()))
             for net in (model.actor, model.critic)]
    tensor = torch.tensor(encode(hand, "masked")[None])
    logits, values = model(tensor)
    for hook in hooks:
        hook.remove()
    assert logits.shape == (1, 7) and values.shape == (1,)
    assert len(received) == 2
    assert torch.equal(received[0], received[1]) and torch.equal(received[0], tensor)
    assert torch.all(received[1][:, MASK_INDICES] == 0)


def test_voi_sensing_branch_weights(hand):
    assert voi_costs(replace(hand, miss=.25))[3] == pytest.approx(.75 * 21 + .25 * 31)


def test_enumerated_all_timeout_tie_prefers_navigation(hand):
    s = replace(hand, remaining=.1, belief=(.2, .5, .8), miss=.25)
    costs = voi_costs(s)
    assert costs == pytest.approx(np.full(7, 200.1), abs=1e-12)
    assert voi(s) == 0


def test_real_voi_tie_and_meaningful_small_difference(hand):
    # With v=1 and sensing=5, inspection's expected cost is exactly 25.
    equal = replace(hand, layout=replace(hand.layout, sensing=5.))
    assert voi_costs(equal)[[3, 6]] == pytest.approx([25, 25])
    assert voi(equal) == 6
    cheaper = replace(hand, layout=replace(hand.layout, sensing=5. - 1e-9))
    assert voi_costs(cheaper)[3] < voi_costs(cheaper)[6]
    assert voi(cheaper) == 3


def test_reactive_decimal_route_tie_and_true_difference(hand):
    l = replace(hand.layout, a=(6.2, 6.1, 100), b=(1.0, 1.1, 100))
    s = replace(hand, layout=l)
    assert reactive(s) == 0
    cheaper = replace(s, layout=replace(l, b=(1.0, 1.1 - 1e-9, 100)))
    assert reactive(cheaper) == 1
