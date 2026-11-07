"""Compatibility facade for frozen graph study APIs.

The production optimizer/rollout algorithm lives in core.learner. This module
only supplies graph defaults, study gates and concrete application adapters.
"""
from __future__ import annotations
from dataclasses import fields
from pathlib import Path
import numpy as np

from .adapters.runtime import seed_everything
import torch
from .adapters.graph import GraphFactory, sequence_record
from .adapters.persistence import GraphStudyStore, load_checkpoint
from .core import ActorCritic as CoreActorCritic, PPOConfig
from .core.learner import gae as core_gae
from .features import FEATURE_NAMES, encode


class ActorCritic(CoreActorCritic):
    """Legacy graph constructor; the actual model requires explicit dimensions."""
    def __init__(self, hidden_sizes=(64, 64)):
        super().__init__(len(FEATURE_NAMES), GraphFactory.spec.action_count, hidden_sizes)


def gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray,
        next_values: np.ndarray, gamma: float = 1.0, lam: float = 0.95):
    """Legacy graph GAE: dones are true sequence terminations, not time limits."""
    if rewards.shape != values.shape or rewards.shape != dones.shape or values.ndim != 2:
        raise ValueError("GAE arrays must have equal [time, env] shape")
    if next_values.shape != values.shape[1:]:
        raise ValueError("Next values must have [env] shape")
    bootstraps = np.concatenate((values[1:], next_values[None]), axis=0)
    return core_gae(rewards, values, dones, np.zeros_like(dones), bootstraps, gamma, lam)


def train(config: dict, output: Path, variant: str, seed: int, pilot: bool,
          approval: Path | None = None, *, observer=None, clock=None,
          visualize=False, visualization_port=0) -> dict:
    if not pilot:
        from .experiments import require_review
        require_review(approval, config)
    if variant not in ("full", "masked") or seed not in config["ppo"]["training_seeds"]:
        raise ValueError("Variant and seed must belong to the frozen experiment")
    hp = config["ppo"]
    steps = 4096 if pilot else hp["total_env_steps"]
    batch_size = hp["num_envs"] * hp["rollout_steps"]
    if steps % batch_size or batch_size % hp["minibatch_size"]:
        raise ValueError("Budgets must contain complete rollouts and minibatches")
    if hp["gamma"] != 1 or hp["gae_lambda"] != 0.95 or hp["reward_scale"] != 0.01:
        raise ValueError("Frozen finite objective: gamma=1, lambda=.95, reward scale=.01")
    from .application import browser_observer_factory, run_learning
    namespace = f"pilot/train/seed-{seed}" if pilot else f"train/seed-{seed}"
    factory = GraphFactory(namespace, variant)
    store = GraphStudyStore(output, config, variant, seed, pilot, approval, factory)
    settings = PPOConfig(**{f.name: hp[f.name] for f in fields(PPOConfig)})
    result = run_learning(factory, settings, steps, store=store, seed=seed,
                          device=config.get("device", "cpu"), clock=clock, observer=observer,
                          observer_factory=browser_observer_factory(output, visualization_port) if visualize else None)
    if visualize:
        print(f"Visualization replay: {output / 'visualization.html'}", flush=True)
    return result


def load_policy(path: Path, device: str = "cpu"):
    checkpoint = load_checkpoint(path, device)
    if checkpoint.get("kind") == "development_learn":
        raise ValueError("Development checkpoints are not frozen graph study policies")
    model = ActorCritic(checkpoint["config"]["ppo"]["hidden_sizes"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    variant = checkpoint["variant"]

    def act(snapshot):
        with torch.no_grad():
            features = torch.as_tensor(encode(snapshot, variant)[None], device=device)
            return int(model(features)[0].argmax(dim=-1).item())

    return act, checkpoint
