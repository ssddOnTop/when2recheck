import csv
import json
from pathlib import Path
import numpy as np
import pytest
import torch

from stale_nav.aggregate import aggregate, read_csv, validate_raw
from stale_nav.environment import Environment, REGIMES, make_scenario
from stale_nav.experiments import (evaluate, load_config, load_ttl, require_review, select_ttl,
                                   validate_checkpoint_set)
from stale_nav.features import encode, feature_spec
from stale_nav.ppo import ActorCritic, gae, load_policy, seed_everything, sequence_record, train
from stale_nav.provenance import ROOT, json_hash, sha256, source_manifest, verify_artifacts, write_json


@pytest.fixture
def config():
    config = load_config(ROOT / "configs/frozen.json")
    config["device"] = "cpu"
    return config


def test_gae_terminal_manual_scaled_time():
    # Travel 2 then success after 3: raw rewards -2,97; no final bootstrap.
    rewards = np.array([[-.02], [.97]], dtype=np.float32)
    values = np.array([[.1], [.2]], dtype=np.float32)
    adv, targets = gae(rewards, values, np.array([[0.], [1.]]), np.array([999.]))
    assert adv[:, 0] == pytest.approx([.08 + .95 * .77, .77])
    assert targets[:, 0] == pytest.approx([.9115, .97])


def test_gae_rollout_cut_and_intermediate_task():
    # Intermediate task success remains live, so the next task value contributes.
    reward = np.array([[.9], [-.03]])
    values = np.array([[.2], [.4]])
    adv, targets = gae(reward, values, np.zeros((2, 1)), np.array([.7]))
    assert adv[:, 0] == pytest.approx([1.1 + .95 * .27, .27])
    assert targets[:, 0] == pytest.approx([1.5565, .67])
    # True termination cuts both bootstrap and the recurrence, even after autoreset.
    adv, _ = gae(np.array([[1.], [100.]]), np.zeros((2, 1)), np.array([[1.], [1.]]), np.array([999.]))
    assert adv[:, 0] == pytest.approx([1, 100])


def test_raw_six_task_reward_and_time_identity():
    env = Environment(make_scenario("pilot", 9, "changing_missed"))
    records, rewards = [], []
    while True:
        _, reward, done, info = env.step(0)  # repeated attempts can produce failure
        rewards.append(reward)
        if info["task_record"]:
            records.append(info["task_record"])
        if done:
            break
    row = sequence_record(records, {})
    assert row["reward"] == pytest.approx(sum(rewards))
    assert row["reward"] == pytest.approx(600 - row["time"] - 200 * row["failure"])
    assert row["penalized_cost"] == pytest.approx(row["time"] + 200 * row["failure"])
    assert row["failure"] > 0


def save_checkpoint(path, config, variant="full", seed=0, pilot=True):
    seed_everything(seed)
    model = ActorCritic(config["ppo"]["hidden_sizes"])
    payload = {"model_state": model.state_dict(), "config": config, "variant": variant,
               "training_seed": seed, "pilot": pilot,
               "env_steps": 4096 if pilot else config["ppo"]["total_env_steps"],
               "source_sha256": source_manifest()["source_sha256"], "config_sha256": json_hash(config),
               "features": feature_spec()}
    torch.save(payload, path)
    return model, payload


def test_checkpoint_reproduces_actor_and_critic(tmp_path, config):
    model, ck = save_checkpoint(tmp_path / "checkpoint.pt", config, variant="masked")
    policy, loaded = load_policy(tmp_path / "checkpoint.pt")
    snapshot = Environment(make_scenario("pilot", 0, "changing_clean")).snapshot
    tensor = torch.tensor(encode(snapshot, "masked")[None])
    other = ActorCritic()
    other.load_state_dict(loaded["model_state"])
    for a, b in zip(model(tensor), other(tensor)):
        assert torch.equal(a, b)
    assert policy(snapshot) == int(model(tensor)[0].argmax())
    assert ck["source_sha256"] == loaded["source_sha256"]


def test_deterministic_seed_actions_and_optimizer():
    def run():
        seed_everything(2)
        model = ActorCritic()
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, eps=1e-5)
        x = torch.ones(16, len(encode(Environment(make_scenario("pilot", 0, "static_clean")).snapshot)))
        actions, lp, values = model.sample(x)
        loss = -lp.mean() + values.square().mean()
        loss.backward()
        optimizer.step()
        return actions, model.state_dict()
    actions_a, a = run()
    actions_b, b = run()
    assert torch.equal(actions_a, actions_b)
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_review_gate_no_training(tmp_path, config):
    with pytest.raises(ValueError, match="independent"):
        require_review(None, config)
    with pytest.raises(ValueError, match="independent"):
        train(config, tmp_path / "must-not-exist", "full", 0, pilot=False)
    assert not (tmp_path / "must-not-exist").exists()
    review = tmp_path / "review.json"
    write_json(review, {"verdict": "PASS_FOR_TRAINING", "source_sha256": "wrong"})
    with pytest.raises(ValueError, match="source"):
        require_review(review, config)
    write_json(review, {"verdict": "PASS_FOR_TRAINING", "source_sha256": source_manifest()["source_sha256"],
                        "config_sha256": json_hash(config), "remaining_blockers": []})
    require_review(review, config)


def test_final_checkpoints_equal_budget_all_seeds_no_pilots(config):
    template = {"source_sha256": source_manifest()["source_sha256"], "config_sha256": json_hash(config),
                "env_steps": 196608, "pilot": False}
    all_models = [{**template, "variant": v, "training_seed": s} for v in ("full", "masked") for s in range(5)]
    validate_checkpoint_set(all_models, config, "test")
    with pytest.raises(ValueError, match="every matched"):
        validate_checkpoint_set(all_models[:-1], config, "test")
    changed = [{**m} for m in all_models]
    changed[0]["env_steps"] = 4096
    with pytest.raises(ValueError, match="budget"):
        validate_checkpoint_set(changed, config, "test")
    changed[0] = {**all_models[0], "pilot": True}
    with pytest.raises(ValueError, match="budget"):
        validate_checkpoint_set(changed, config, "test")


def test_config_changes_and_fallbacks(tmp_path, config):
    path = tmp_path / "config.json"
    altered = json.loads(json.dumps(config))
    altered["ppo"]["gamma"] = .99
    write_json(path, altered)
    with pytest.raises(ValueError, match="differs"):
        load_config(path)
    altered = json.loads(json.dumps(config))
    altered["ppo"]["total_env_steps"] = 98304
    write_json(path, altered)
    with pytest.raises(ValueError, match="reason"):
        load_config(path)
    altered["fallback_reason"] = "Example measured pre-test throughput fallback"
    write_json(path, altered)
    assert load_config(path)["ppo"]["total_env_steps"] == 98304


def test_validation_selection_exact_count_and_ties(tmp_path, config, monkeypatch):
    import stale_nav.experiments as experiments
    calls = []
    real_make = make_scenario

    def make(namespace, index, regime):
        calls.append((namespace, index, regime))
        return real_make(namespace, index, regime)

    def run(sc, policy, identity):
        # Equal costs intentionally check the smaller-threshold tie rule.
        return [], {**identity, "penalized_cost": 42}

    monkeypatch.setattr(experiments, "make_scenario", make)
    monkeypatch.setattr(experiments, "run_sequence", run)
    out = tmp_path / "selection"
    result = select_ttl(config, out)
    assert result["thresholds"] == {r: 1 for r in REGIMES}
    assert len(calls) == 64 * 3 * 4
    assert {ns for ns, _, _ in calls} == {"val"}
    assert {i for _, i, _ in calls} == set(range(64))
    assert load_ttl(out / "ttl_selection.json", config) == result["thresholds"]


def test_real_pilot_evaluation_pairing_and_aggregation(tmp_path, config):
    out = tmp_path / "evaluation"
    checkpoint = tmp_path / "model.pt"
    save_checkpoint(checkpoint, config)
    result = evaluate(config, out, "pilot", [checkpoint], pilot_sequences=2)
    assert result["sequences_per_regime"] == 2 and result["pilot_ttl_unselected"]
    rows = validate_raw(out)
    assert len(rows) == 2 * 3 * 5
    scenario_sets = {}
    for row in rows:
        assert row["namespace"] == "pilot"
        scenario_sets.setdefault(row["policy"], set()).add(row["scenario_id"])
    assert all(ids == scenario_sets["reactive"] for ids in scenario_sets.values())
    agg = tmp_path / "aggregate"
    aggregate(out, [], agg)
    tables = read_csv(agg / "per_seed_summary.csv")
    target = next(r for r in tables if r["policy"] == "voi" and r["regime"] == "static_clean" and r["metric"] == "penalized_cost")
    expected = np.mean([float(r["penalized_cost"]) for r in rows if r["policy"] == "voi" and r["regime"] == "static_clean"])
    assert float(target["mean"]) == pytest.approx(expected) and int(target["n"]) == 2
    verify_artifacts(agg)
    with (out / "sequences.csv").open("a") as f:
        f.write("corruption\n")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_raw(out)


def test_public_policy_trace_deterministic():
    from stale_nav.policies import choose
    sc = make_scenario("pilot", 22, "changing_missed")
    def run():
        env, trace = Environment(sc), []
        done = False
        while not done:
            action = choose("voi", env.snapshot)
            s, reward, done, info = env.step(action)
            trace.append((action, s, reward, done, info))
        return trace
    assert run() == run()


def test_source_archive_is_deterministic_and_historical(tmp_path):
    import tarfile
    from stale_nav.provenance import source_archive
    root = tmp_path / 'project'
    (root / 'src').mkdir(parents=True)
    (root / 'planning').mkdir()
    (root / 'src/a.py').write_text('value = 1\n')
    (root / 'planning/submission-protocol.json').write_text('{"revision":2}\n')
    manifest = source_manifest(root)
    first = source_archive(manifest, root)
    assert source_archive(manifest, root) == first
    (root / 'src/a.py').write_text('value = 2\n')
    assert sha256(Path(first['path'])) == first['sha256']
    with tarfile.open(first['path']) as archive:
        assert set(archive.getnames()) == {'src/a.py', 'planning/submission-protocol.json'}
        assert archive.extractfile('src/a.py').read() == b'value = 1\n'


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA hardware unavailable')
def test_gpu_deterministic_forward_backward():
    import os
    from stale_nav.features import FEATURE_NAMES
    assert os.environ['CUBLAS_WORKSPACE_CONFIG'] in (':4096:8', ':16:8')
    def run():
        seed_everything(8)
        model = ActorCritic().cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)
        x = torch.ones((16, len(FEATURE_NAMES)), device='cuda')
        logits, values = model(x)
        (logits.square().mean() + values.square().mean()).backward()
        optimizer.step()
        torch.cuda.synchronize()
        return model(x)
    first, second = run(), run()
    assert all(torch.equal(a, b) for a, b in zip(first, second))


def test_config_matches_authoritative_frozen_protocol():
    protocol = json.loads((ROOT / 'planning/submission-protocol.json').read_text())['frozen_configuration']
    config = json.loads((ROOT / 'configs/frozen.json').read_text())
    for key in ('ppo', 'regimes', 'evaluation'):
        assert config[key] == protocol[key]
    assert protocol['doors'] == 3 and protocol['tasks_per_sequence'] == 6


def test_default_config_uses_pretest_equal_budget_fallback():
    from stale_nav.cli import parser
    args = parser().parse_args(['manifest'])
    final = load_config(args.config)
    assert final['ppo']['total_env_steps'] == 98304
    assert final['ppo']['training_seeds'] == [0, 1, 2, 3, 4]
    assert final['fallback_reason']


@pytest.mark.parametrize('evaluation,training,allowed', [
    ('pilot_evaluation', 'pilot_train', True),
    ('val_evaluation', 'final_train', True),
    ('test_evaluation', 'final_train', True),
    ('pilot_evaluation', 'final_train', False),
    ('val_evaluation', 'pilot_train', False),
    ('test_evaluation', 'pilot_train', False),
])
def test_learning_curve_split_compatibility(evaluation, training, allowed):
    from stale_nav.aggregate import check_learning_curve_kind
    if allowed:
        check_learning_curve_kind(evaluation, training)
    else:
        with pytest.raises(ValueError, match='mix'):
            check_learning_curve_kind(evaluation, training)
