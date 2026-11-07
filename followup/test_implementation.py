"""Implementation acceptance using public fixtures/unit-design keys only.

No prospective Factory.generate success, production pilot, train, or evaluation
is called. Synthetic approvals test validation only; they never authorize data
creation. Tiny tensor updates exercise PPO math, not a prospective experiment.
"""
from dataclasses import asdict, replace
import copy
import json
import math
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest
import torch

from followup import control as ctl
from followup import domain as d
from followup import learning as learn
from followup import runner
from stale_nav.environment import Layout, initial_snapshot, physical_step
from stale_nav.features import SCALES as ORIGINAL_SCALES


@pytest.fixture
def bindings():
    return ctl.manifest()


def hand(target=0, opened=True, miss=0., budget=60., a=10., b=1., v=1., sensing=1., safe=25.):
    def edges(value):
        return tuple(value if i == target else 80. for i in range(3))
    return d.Scenario(ctl.UNIT_NAMESPACE + "/public-hand", ctl.UNIT_NAMESPACE + "/public-hand",
                      "inspection|4/5|0", Layout(edges(a), edges(b), edges(v), sensing, safe, budget),
                      tuple(opened if i == target else False for i in range(3)), miss)


def approval(tmp_path, bindings, permissions):
    path = tmp_path / "unit-only-synthetic-review.json"
    ctl.write_json(path, {"unit_fixture": True, "verdict": "PASS_FOR_FOLLOWUP_EXECUTION", "remaining_blockers": [],
                         "study_id": ctl.STUDY, **{k: bindings[k] for k in ctl.BINDING_KEYS},
                         "permissions": permissions})
    return path


@pytest.mark.parametrize("split", ["train", "validation", "heldout", "pilot"])
def test_all_reserved_generators_refuse_without_approval(tmp_path, split):
    output = tmp_path / split
    with pytest.raises(ctl.GateError, match="approval"):
        d.ScenarioFactory(split, 1000 if split in ("train", "pilot") else None, None, output)
    with pytest.raises(ctl.GateError):
        d._sample(d.NAMESPACES[split], 0, "forbidden", 0, 0, split=split)
    assert not output.exists()


@pytest.mark.parametrize("operation", sorted(ctl.OPERATIONS))
def test_design_pass_cannot_open_execution_gate(operation):
    with pytest.raises(ctl.GateError, match="Design approval is insufficient"):
        ctl.authorize(ctl.DESIGN_REVIEW, operation)


@pytest.mark.parametrize("key", ctl.BINDING_KEYS)
def test_approval_binds_every_digest(tmp_path, bindings, key):
    path = approval(tmp_path, bindings, ["generate_train"])
    bad = ctl.read_json(path)
    bad[key] = "wrong"
    ctl.write_json(path, bad)
    with pytest.raises(ctl.GateError, match=key):
        ctl.review_permissions(ctl.read_json(path), bindings, "generate_train")


def test_permission_and_review_mutation_refused(tmp_path, bindings):
    path = approval(tmp_path, bindings, ["generate_train"])
    permissions = ctl.review_permissions(ctl.read_json(path), bindings, "generate_train")
    assert permissions == frozenset({"generate_train"})
    with pytest.raises(ctl.GateError, match="Unit review fixtures"):
        ctl.authorize(path, "generate_train")
    grant = ctl.Grant(path, ctl.sha(path), bindings, permissions)
    with pytest.raises(ctl.GateError, match="Unit review fixtures"):
        grant.check("generate_train")
    with pytest.raises(ctl.GateError, match="not independently approved"):
        ctl.review_permissions(ctl.read_json(path), bindings, "train")
    with path.open("a") as f:
        f.write("\n")
    with pytest.raises(ctl.GateError, match="review changed"):
        grant.check("generate_train")


def test_heldout_is_still_sealed_after_implementation_approval(tmp_path, bindings):
    path = approval(tmp_path, bindings, ["generate_heldout"])
    permissions = ctl.review_permissions(ctl.read_json(path), bindings, "generate_heldout")
    grant = ctl.Grant(path, ctl.sha(path), bindings, permissions)
    with pytest.raises(ctl.GateError, match="sealed"):
        ctl.heldout_release(None, grant)
    with pytest.raises(ctl.GateError):
        d._sample(d.NAMESPACES["heldout"], 2026110713, "evaluation", 0, 0, grant, "heldout")
    release = tmp_path / "unit-only-incomplete-release.json"
    ctl.write_json(release, {"unit_fixture": True, "verdict": "PASS_FOR_FOLLOWUP_HELDOUT_RELEASE", "remaining_blockers": [],
                            "study_id": ctl.STUDY, **{k: bindings[k] for k in ctl.BINDING_KEYS}, "checkpoints": []})
    with pytest.raises(ctl.GateError, match="Unit release fixtures"):
        ctl.heldout_release(release, grant)
    with pytest.raises(ctl.GateError, match="all ten"):
        ctl.validate_model_metadata([], bindings)


def test_all_ten_final_models_and_correct_features_required(bindings):
    # Metadata-only fixtures. No checkpoint, dataset or empirical result is written.
    rows = [{"study_id": ctl.STUDY, **{k: bindings[k] for k in ctl.BINDING_KEYS},
             "variant": variant, "training_seed": seed, "device": "cpu", "config": ctl.config(),
             "features": d.feature_spec(), "pilot": False, "env_steps": 196608, "horizon_tasks": 1}
            for variant, seed in sorted(ctl.expected_models())]
    ctl.validate_model_metadata(rows, bindings)
    for change in ({"pilot": True}, {"env_steps": 98304}, {"horizon_tasks": 6}, {"unit_fixture": True}, {"features": {}}):
        altered = copy.deepcopy(rows)
        altered[0].update(change)
        with pytest.raises(ctl.GateError):
            ctl.validate_model_metadata(altered, bindings)
    with pytest.raises(ctl.GateError, match="all ten"):
        ctl.validate_model_metadata(rows[:-1], bindings)
    with pytest.raises(ctl.GateError, match="Duplicate"):
        ctl.validate_model_metadata(rows + rows[:1], bindings)


@pytest.mark.parametrize("arguments", [
    ["generate", "--split", "train", "--seed", "1000", "--count", "1"],
    ["generate", "--split", "val"], ["generate", "--split", "test"],
    ["pilot"], ["train", "--variant", "full", "--seed", "1000"],
    ["evaluate", "--split", "test", "--dataset", "unused", "--checkpoints", "unused"],
    ["seal-validation", "--evaluation", "unused", "--checkpoints", "unused"],
    ["aggregate", "--evaluation", "unused"],
])
def test_cli_default_refusal_creates_no_outputs(tmp_path, arguments):
    output = tmp_path / "must-not-exist"
    result = subprocess.run([sys.executable, "-m", "followup", *arguments, "--output", str(output)],
                            cwd=ctl.ROOT, capture_output=True, text=True, check=False)
    assert result.returncode == 2 and "independent implementation approval" in result.stderr
    assert not output.exists()


def test_unit_streams_only_and_repeatability():
    ids, targets = set(), set()
    for stratum in range(12):
        for i in range(6):
            a = d.unit_scenario(i, stratum)
            b = d.unit_scenario(i, stratum)
            assert a == b and a.namespace.startswith(ctl.UNIT_NAMESPACE + "/")
            assert all(not a.namespace.startswith(n + "/") for n in d.NAMESPACES.values())
            assert a.layout.budget < min(x for x in a.layout.a if x > a.layout.budget)
            targets.add(d.public_target(d.OneTaskEnv(a).snapshot))
            ids.add(a.scenario_id)
    assert len(ids) == 72 and targets == {0, 1, 2}
    assert d.unit_scenario(0, label="a") != d.unit_scenario(0, label="b")
    with pytest.raises(ValueError):
        d.unit_scenario(0, label="../heldout-sealed")


def test_keyed_missingness_does_not_follow_unrelated_actions():
    sc = replace(d.unit_scenario(3, 1), miss=.25)
    draws = [d.sensing_miss(sc, 1, i) for i in range(80)]
    for i in range(100):
        d.sensing_miss(sc, 2, i)
    assert draws == [d.sensing_miss(sc, 1, i) for i in range(80)]
    assert any(draws) and not all(draws)
    assert draws == [d.sensing_miss(replace(sc, world=tuple(not x for x in sc.world)), 1, i) for i in range(80)]


@pytest.mark.parametrize("target", range(3))
@pytest.mark.parametrize("opened", [False, True])
def test_policy_permutations_and_fresh_one_task_termination(target, opened):
    sc = hand(target, opened)
    initial = d.OneTaskEnv(sc).snapshot
    assert d.public_target(initial) == target
    assert d.choose("DIRECT", initial) == target
    assert d.choose("INSPECT_REACTIVE", initial) == target + 3
    assert d.choose("INSPECT_BEST_MISS", initial) == target + 3
    assert d.choose("ALWAYS_SAFE", initial) == 6
    assert d.choose("voi", initial) == target + 3
    for name in d.BASELINES:
        row = runner.roll_episode(sc, lambda s, name=name: d.choose(name, s))
        assert row["success"] == 1 and row["failure"] == 0
        assert row["reward"] == pytest.approx(100 - row["penalized_cost"])
    env = d.OneTaskEnv(sc)
    s, _, done, _ = env.step(6)
    assert done and s.task == 0
    with pytest.raises(RuntimeError):
        env.step(6)
    fresh = d.OneTaskEnv(sc).snapshot
    assert fresh.last == (-1,) * 3 and fresh.belief == (.5,) * 3 and fresh.age == (-1,) * 3


@pytest.mark.parametrize("target", range(3))
def test_miss_and_best_miss_are_public_and_physical(target):
    env = d.OneTaskEnv(hand(target, opened=False, miss=1.))
    s, reward, done, info = env.step(target + 3)
    assert reward == -2 and not done and info["evidence"] == ()
    assert s.position == 4 + target and s.last == (-1,) * 3 and s.belief == (.5,) * 3
    assert d.choose("INSPECT_BEST_MISS", s) == 6
    assert d.choose("INSPECT_REACTIVE", s) == target
    s, _, done, info = env.step(6)
    assert done and info["episode"]["time"] == 28  # Includes viewpoint return.
    assert info["episode"]["inspections"] == 1 and s.age == (-1,) * 3


def test_budget_equalities_and_unreached_evidence():
    for budget, success in ((7.3, True), (7.3 - 1e-10, False)):
        env = d.OneTaskEnv(hand(a=6.2, b=1.1, budget=budget))
        s, _, done, info = env.step(0)
        assert done and info["success"] == success and s.remaining == 0 and s.task == 0
        assert info["dt"] == budget and s.last[0] == 1
    env = d.OneTaskEnv(hand(budget=.9))
    s, _, done, info = env.step(3)
    assert done and info["evidence"] == () and s.last == (-1,) * 3
    for budget, success in ((28., True), (27., False)):
        env = d.OneTaskEnv(hand(opened=False, budget=budget))
        env.step(3)
        _, _, done, info = env.step(6)
        assert done and info["success"] == success
        assert info["episode"]["time"] == budget


def test_continuation_costs_use_kernel_decimal_equality():
    sc = hand(a=6.2, b=1.1, budget=7.3)
    snapshot = replace(d.OneTaskEnv(sc).snapshot, belief=(1., .5, .5), last=(1, -1, -1))
    assert d.navigation_cost(snapshot, 0) == pytest.approx(7.3)
    assert d.navigation_cost(replace(snapshot, remaining=7.3 - 1e-10), 0) == pytest.approx(207.3 - 1e-10)


def test_no_hidden_inputs_and_no_original_normalizer_changes():
    before = ORIGINAL_SCALES.copy()
    a = d.OneTaskEnv(hand(opened=True)).snapshot
    b = d.OneTaskEnv(hand(opened=False)).snapshot
    assert a == b
    for variant in ("full", "masked"):
        np.testing.assert_array_equal(d.encode(a, variant), d.encode(b, variant))
    for i in range(12):
        s = d.OneTaskEnv(d.unit_scenario(1, i)).snapshot
        for variant in ("full", "masked"):
            x = d.encode(s, variant)
            assert x.shape == (42,) and np.max(np.abs(x)) <= 1
    np.testing.assert_array_equal(before, ORIGINAL_SCALES)
    spec = d.feature_spec()
    assert spec["scales"]["remaining_budget"] == 72
    assert "structural" in spec["interpretation"] and "invariant" in spec["interpretation"]
    assert not {"scenario_id", "world", "stratum", "namespace"} & set(asdict(a))


def test_same_mask_for_actor_critic_and_invariant_age():
    env = d.OneTaskEnv(hand())
    states = [env.snapshot, env.step(3)[0]]
    for state in states:
        full, masked = d.encode(state), d.encode(state, "masked")
        assert all(full[d.FEATURE_NAMES.index(f"boundary_age_{i}")] == 0 for i in range(3))
        assert full[d.FEATURE_NAMES.index("task_index")] == 0
        assert np.all(masked[list(d.MASK_INDICES)] == 0)
        kept = [i for i in range(42) if i not in d.MASK_INDICES]
        np.testing.assert_array_equal(full[kept], masked[kept])
    with learn.execution_context("cpu"):
        model = learn.ActorCritic(1000)
        seen = []
        hooks = [module.register_forward_pre_hook(lambda _, args: seen.append(args[0].clone())) for module in (model.actor, model.critic)]
        try:
            model(torch.tensor(d.encode(states[0], "masked")[None]))
        finally:
            for hook in hooks:
                hook.remove()
        assert torch.equal(seen[0], seen[1])
        assert model.actor[-1].out_features == 7  # No hidden/action legality mask.


def test_local_rng_and_execution_settings_are_restored():
    rng = torch.get_rng_state().clone()
    threads = torch.get_num_threads()
    deterministic = torch.are_deterministic_algorithms_enabled()
    workspace = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    with learn.execution_context("cpu"):
        model = learn.ActorCritic(1000)
        generator = torch.Generator().manual_seed(1000)
        model.sample(torch.ones(4, 42), generator)
        assert torch.are_deterministic_algorithms_enabled()
    assert torch.equal(rng, torch.get_rng_state())
    assert threads == torch.get_num_threads()
    assert deterministic == torch.are_deterministic_algorithms_enabled()
    assert workspace == os.environ.get("CUBLAS_WORKSPACE_CONFIG")


def test_terminal_and_rollout_cut_gae():
    adv, target = learn.gae(np.array([[-.02], [.97]]), np.array([[.1], [.2]]), np.array([[0.], [1.]]), np.array([999.]))
    assert target[:, 0] == pytest.approx([.9115, .97])
    adv, target = learn.gae(np.array([[.9], [-.03]]), np.array([[.2], [.4]]), np.zeros((2, 1)), np.array([.7]))
    assert target[:, 0] == pytest.approx([1.5565, .67])
    adv, _ = learn.gae(np.array([[1.], [100.]]), np.zeros((2, 1)), np.ones((2, 1)), np.array([999.]))
    assert adv[:, 0] == pytest.approx([1, 100])


def test_bounded_fixture_collection_update_and_checkpoint(tmp_path):
    # Eight unit-fixture transitions and one small optimizer update, not a pilot.
    def execute():
        with learn.execution_context("cpu"):
            model = learn.ActorCritic(1000)
            rng = torch.Generator().manual_seed(1000)
            def new_slot():
                return learn.Slot.from_scenario(hand(budget=7.3, a=6.2, b=1.1))
            slots = [new_slot(), new_slot()]
            batch, episodes = learn.collect_rollout(model, slots, new_slot, rng, "masked", 4, .01, "cpu")
            assert batch["dones"].sum() > 0
            assert all(s.env.snapshot.task == 0 for s in slots)
            assert all(row["namespace"].startswith(d.ctl.UNIT_NAMESPACE) for row in episodes)
            hp = {**ctl.config()["ppo"], "epochs": 1, "minibatch_size": 4}
            optimizer = torch.optim.Adam(model.parameters(), lr=hp["learning_rate"], eps=1e-5)
            losses = learn.ppo_update(model, optimizer, batch, hp, rng, "cpu")
            assert all(math.isfinite(v) for v in losses.values())
            return model.state_dict(), batch
    first, batch1 = execute()
    second, batch2 = execute()
    assert all(torch.equal(first[k], second[k]) for k in first)
    np.testing.assert_array_equal(batch1["actions"], batch2["actions"])
    checkpoint = tmp_path / "unit-fixture-weights.pt"
    torch.save({"unit_fixture": True, "state": first}, checkpoint)
    loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
    assert all(torch.equal(first[k], loaded["state"][k]) for k in first)


def test_raw_trace_totals_and_model_pairing_use_only_unit_rows(tmp_path):
    records = []
    log = runner.CSVLog(tmp_path / "unit-traces.csv")
    try:
        for stratum in range(12):
            sc = d.unit_scenario(0, stratum)
            slots = [(name, "baseline") for name in d.BASELINES] + [("ppo_" + v, seed) for v, seed in sorted(ctl.expected_models())]
            for name, seed in slots:
                # Synthetic bookkeeping slots all execute public ALWAYS_SAFE.
                identity = {"policy": name, "training_seed": seed}
                row = runner.roll_episode(sc, lambda s: 6, lambda t: log.row({**identity, **t}))
                records.append({**identity, **row})
    finally:
        log.close()
    runner.validate_episode_rows(records, 1)
    runner.verify_trace_totals(records, tmp_path / "unit-traces.csv")
    with pytest.raises(ctl.GateError):
        runner.validate_episode_rows(records[:-1], 1)
    with pytest.raises(ctl.GateError):
        runner.validate_episode_rows(records + records[:1], 1)
    changed = copy.deepcopy(records)
    changed[0]["time"] += 1
    with pytest.raises(ctl.GateError, match="total mismatch"):
        runner.verify_trace_totals(changed, tmp_path / "unit-traces.csv")


def test_review_manifest_covers_execution_and_approved_files(bindings):
    files = bindings["files"]
    for name in ("control.py", "domain.py", "learning.py", "runner.py", "__main__.py", "execution_config.json", "test_implementation.py", "analytical_validation.py", "test_analytical_validation.py"):
        assert "followup/" + name in files
    assert "planning/followup-protocol.json" in files
    assert "src/stale_nav/environment.py" in files
    assert bindings["original_source_sha256"] == "85dfcc3d077683d49c12a13a670f2f95b2560f227643618343d724cb0ae8cd6b"
    assert ctl.config()["ppo"]["total_env_steps_per_run"] == 196608
    assert ctl.config()["training_seeds"] == [1000, 1001, 1002, 1003, 1004]
    assert ctl.config()["horizon_tasks"] == 1


@pytest.mark.parametrize('split', ['train', 'validation', 'heldout', 'pilot'])
def test_reserved_episode_adapter_cannot_bypass_cli_gate(split):
    sc = replace(hand(), namespace=d.NAMESPACES[split])
    with pytest.raises(ctl.GateError, match='execution grant'):
        d.OneTaskEnv(sc)
    with pytest.raises(ctl.GateError, match='execution grant'):
        runner.roll_episode(sc, lambda snapshot: 6)


def test_feasibility_requires_stop_not_automatic_budget_reduction():
    # Pure schema/arithmetic checks, not a pilot artifact or measured result.
    fixture = {'env_steps': 4096, 'status': 'FEASIBLE', 'wall_seconds': 1., 'estimated_all_final_minutes': 8.}
    runner.validate_feasibility_summary(fixture)
    for update in ({'wall_seconds': 12., 'estimated_all_final_minutes': 96.},
                   {'env_steps': 2048}, {'unit_fixture': True}, {'wall_seconds': float('nan')},
                   {'estimated_all_final_minutes': 1.}):
        with pytest.raises(ctl.GateError):
            runner.validate_feasibility_summary({**fixture, **update})
    assert ctl.config()['ppo']['total_env_steps_per_run'] == 196608


def test_source_snapshot_and_input_provenance_verification(tmp_path, bindings):
    output = tmp_path / 'unit-provenance-fixture'
    output.mkdir()
    source_input = tmp_path / 'unit-input.txt'
    source_input.write_text('public unit fixture\n')
    ctl.source_archive(output, bindings)
    provenance = {'unit_fixture': True, 'kind': 'unit_fixture', 'study_id': ctl.STUDY,
                  **{k: bindings[k] for k in ctl.BINDING_KEYS}, 'source_files': bindings['files'],
                  'input_sha256': {str(source_input): ctl.sha(source_input)}}
    ctl.write_json(output / 'provenance.json', provenance)
    ctl.write_json(output / 'artifacts.json', {p.name: ctl.sha(p) for p in output.iterdir()})
    assert ctl.verify(output, bindings)['unit_fixture'] is True
    source_input.write_text('changed unit fixture\n')
    with pytest.raises(ctl.GateError, match='execution input changed'):
        ctl.verify(output, bindings)
    # Historical source remains independently verifiable without live inputs.
    ctl.verify(output)
    (output / 'unlisted.txt').write_text('unlisted\n')
    with pytest.raises(ctl.GateError, match='coverage'):
        ctl.verify(output)
