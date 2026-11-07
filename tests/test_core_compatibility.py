"""Frozen graph numerical equivalence and legacy artifact contracts."""
import csv
import json

import numpy as np
import pytest
import torch
from torch.distributions import Categorical

from stale_nav.adapters.graph import GraphFactory
from stale_nav.adapters.persistence import GraphStudyStore
from stale_nav.adapters.runtime import seed_everything
from stale_nav.core import learn
from stale_nav.environment import Environment, REGIMES, make_scenario
from stale_nav.features import encode, feature_spec
from stale_nav.ppo import ActorCritic, gae, load_policy, train
from test_core import Clock, Recorder, settings


def reference_graph_update(seed, hp):
    """Independent pre-refactor rollout and clipped PPO equations, one update."""
    seed_everything(seed)
    model = ActorCritic(hp.hidden_sizes)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp.learning_rate, eps=1e-5)
    ordinal = 0
    regimes = tuple(REGIMES)
    def make():
        nonlocal ordinal
        sc = make_scenario("pilot/reference", ordinal, regimes[ordinal % len(regimes)])
        ordinal += 1
        return Environment(sc)
    slots = [make() for _ in range(hp.num_envs)]
    shape = (hp.rollout_steps, hp.num_envs)
    observations = np.empty((*shape, GraphFactory.spec.observation_size), dtype=np.float32)
    actions = np.empty(shape, dtype=np.int64)
    lp = np.empty(shape, dtype=np.float32)
    values, rewards, dones = np.empty_like(lp), np.empty_like(lp), np.empty_like(lp)
    for t in range(hp.rollout_steps):
        observations[t] = np.stack([encode(env.snapshot) for env in slots])
        with torch.no_grad():
            a, logp, v = model.sample(torch.tensor(observations[t]))
        actions[t], lp[t], values[t] = a.numpy(), logp.numpy(), v.numpy()
        for e in range(hp.num_envs):
            _, reward, done, _ = slots[e].step(int(actions[t, e]))
            rewards[t, e], dones[t, e] = reward * hp.reward_scale, done
            if done:
                slots[e] = make()
    with torch.no_grad():
        next_values = model(torch.tensor(np.stack([encode(env.snapshot) for env in slots])))[1].numpy()
    # The old recurrence is retained independently rather than calling core GAE.
    advantages = np.zeros_like(values)
    carry = np.zeros_like(next_values)
    for t in reversed(range(hp.rollout_steps)):
        bootstrap = next_values if t == hp.rollout_steps - 1 else values[t + 1]
        alive = 1.0 - dones[t]
        delta = rewards[t] + hp.gamma * bootstrap * alive - values[t]
        carry = delta + hp.gamma * hp.gae_lambda * alive * carry
        advantages[t] = carry
    returns = advantages + values
    obs = torch.tensor(observations.reshape(-1, observations.shape[-1]))
    act, old_lp = torch.tensor(actions.ravel()), torch.tensor(lp.ravel())
    adv, targets = torch.tensor(advantages.ravel()), torch.tensor(returns.ravel())
    adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
    for _ in range(hp.epochs):
        order = torch.randperm(len(act))
        for start in range(0, len(act), hp.minibatch_size):
            idx = order[start:start + hp.minibatch_size]
            logits, predicted = model(obs[idx])
            dist = Categorical(logits=logits)
            ratio = (dist.log_prob(act[idx]) - old_lp[idx]).exp()
            policy_loss = -torch.minimum(ratio * adv[idx], ratio.clamp(
                1 - hp.clip_coefficient, 1 + hp.clip_coefficient) * adv[idx]).mean()
            value_loss = .5 * (predicted - targets[idx]).square().mean()
            loss = policy_loss + hp.value_coefficient * value_loss - hp.entropy_coefficient * dist.entropy().mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), hp.max_grad_norm)
            optimizer.step()
    return model.state_dict(), actions.ravel().tolist(), torch.get_rng_state()


def test_actual_core_matches_pre_refactor_graph_optimizer_bitwise():
    hp = settings(rollout_steps=32, epochs=2)
    expected, actions, rng = reference_graph_update(3, hp)
    seed_everything(3)
    observer = Recorder()
    result = learn(GraphFactory("pilot/reference"), hp, hp.num_envs * hp.rollout_steps,
                   clock=Clock(), observer=observer)
    assert actions == [e["action"] for e in observer.events if e["type"] == "transition"]
    assert all(torch.equal(value, result.model.state_dict()[key]) for key, value in expected.items())
    assert torch.equal(rng, torch.get_rng_state())


def test_legacy_facade_artifact_schema_and_core_integration(tmp_path, monkeypatch):
    import stale_nav.adapters.persistence as persistence
    import stale_nav.application as application
    from stale_nav.provenance import ROOT
    config = json.loads((ROOT / "configs/frozen.json").read_text())
    config["device"] = "cpu"
    config["ppo"].update(num_envs=2, rollout_steps=32, minibatch_size=32, hidden_sizes=[8], epochs=1)
    # Isolate this unit test from source archives and the historical materials.
    def begin(output, config, kind, inputs):
        output.mkdir()
        return {"kind": kind, "source_sha256": "test-only-source"}
    monkeypatch.setattr(persistence, "begin_run", begin)
    real_learn = application.learn
    calls = []
    def capture(*args, **kwargs):
        calls.append(args)
        return real_learn(*args, **kwargs)
    monkeypatch.setattr(application, "learn", capture)
    out = tmp_path / "legacy"
    summary = train(config, out, "masked", 0, pilot=True, clock=Clock())
    assert len(calls) == 1 and calls[0][2] == 4096
    assert summary["kind"] == "pilot_train" and summary["env_steps"] == 4096
    with (out / "steps.csv").open() as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == GraphStudyStore.log_fields
        rows = list(reader)
    assert len(rows) == 4096
    # Frozen reward scaling is rounded exactly once into the float32 rollout.
    assert all(r["scaled_reward"] == str(np.float32(float(r["raw_reward"]) * .01)) for r in rows)
    with (out / "updates.csv").open() as stream:
        reader = csv.DictReader(stream)
        assert reader.fieldnames == ["env_step", "update", "wall_seconds", "steps_per_second",
                                     "policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction"]
    act, checkpoint = load_policy(out / "checkpoint.pt")
    assert checkpoint["checkpoint_rule"] == "final_completed_budget" and checkpoint["pilot"]
    assert checkpoint["features"] == feature_spec()
    assert checkpoint["optimizer_state"]["state"]
    assert 0 <= act(Environment(make_scenario("pilot", 0, "static_clean")).snapshot) < 7
    sequences = json.loads((out / "sequences.json").read_text())
    assert len(sequences) == summary["completed_sequences"]
    assert all(r["reward"] == pytest.approx(600 - r["time"] - 200 * r["failure"]) for r in sequences)
    with pytest.raises(FileExistsError):
        train(config, out, "masked", 0, pilot=True)


def test_graph_scene_never_reads_hidden_worlds():
    factory = GraphFactory("pilot/public")
    env = factory.create(0)
    env.reset()
    class PoisonScenario:
        @property
        def worlds(self):
            raise AssertionError("hidden worlds accessed by visualization")
    env.scenario = PoisonScenario()
    scene = env.scene()
    assert scene["agent"] == "S"
    assert {n["state"] for n in scene["nodes"] if "state" in n} == {"unknown"}
    assert "worlds" not in json.dumps(scene)
    env.close()
