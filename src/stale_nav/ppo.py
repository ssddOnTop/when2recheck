"""Original compact PPO implementation, not copied from an upstream codebase.

One shared masked input tensor feeds separate actor and critic MLPs. Rewards
are scaled exactly once before gamma=1, decision-step-lambda GAE. Only complete
six-task sequences set done. Rollout cuts bootstrap the next public state.
"""
from __future__ import annotations
import csv
import os
import random
import time
from pathlib import Path
import numpy as np

# Set before any CUDA context or cuBLAS handle is created, including evaluation.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import torch
from torch import nn
from torch.distributions import Categorical

from .environment import Environment, REGIMES, make_scenario
from .features import FEATURE_NAMES, encode, feature_spec
from .provenance import begin_run, finish_run, json_hash, write_csv, write_json


def seed_everything(seed: int):
    if os.environ.get("CUBLAS_WORKSPACE_CONFIG") not in (":4096:8", ":16:8"):
        raise ValueError("Deterministic CUDA requires CUBLAS_WORKSPACE_CONFIG=:4096:8 or :16:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)


class ActorCritic(nn.Module):
    def __init__(self, hidden_sizes=(64, 64)):
        super().__init__()

        def network(output: int, gain: float):
            layers = []
            previous = len(FEATURE_NAMES)
            for hidden in hidden_sizes:
                layer = nn.Linear(previous, hidden)
                nn.init.orthogonal_(layer.weight, np.sqrt(2))
                nn.init.zeros_(layer.bias)
                layers.extend((layer, nn.Tanh()))
                previous = hidden
            last = nn.Linear(previous, output)
            nn.init.orthogonal_(last.weight, gain)
            nn.init.zeros_(last.bias)
            return nn.Sequential(*layers, last)

        self.actor = network(7, 0.01)
        self.critic = network(1, 1.0)

    def forward(self, features: torch.Tensor):
        return self.actor(features), self.critic(features).squeeze(-1)

    def sample(self, features: torch.Tensor):
        logits, values = self(features)
        distribution = Categorical(logits=logits)
        actions = distribution.sample()
        return actions, distribution.log_prob(actions), values


def gae(rewards: np.ndarray, values: np.ndarray, dones: np.ndarray,
        next_values: np.ndarray, gamma: float = 1.0, lam: float = 0.95):
    """Inputs are [time, env]. dones[t] terminates the transition at t.

    Caller supplies already-scaled rewards. Duration is charged in reward,
    not through an additional time discount or lambda exponent.
    """
    if rewards.shape != values.shape or rewards.shape != dones.shape:
        raise ValueError("GAE arrays must have equal [time, env] shape")
    advantages = np.zeros_like(values)
    carry = np.zeros_like(next_values)
    for t in reversed(range(len(rewards))):
        bootstrap = next_values if t == len(rewards) - 1 else values[t + 1]
        alive = 1.0 - dones[t]
        delta = rewards[t] + gamma * bootstrap * alive - values[t]
        carry = delta + gamma * lam * alive * carry
        advantages[t] = carry
    return advantages, advantages + values


def sequence_record(task_rows: list[dict], identity: dict) -> dict:
    if len(task_rows) != 6 or [r["task"] for r in task_rows] != list(range(6)):
        raise ValueError("Expected exactly six ordered tasks")
    result = dict(identity)
    for key in ("success", "failure", "time", "inspections", "blocked_attempts", "reward", "penalized_cost"):
        result[key] = sum(r[key] for r in task_rows)
    result["success_fraction"] = result["success"] / 6
    result["time_per_task"] = result["time"] / 6
    result["inspections_per_task"] = result["inspections"] / 6
    if not np.isclose(result["reward"], 600 - result["time"] - 200 * result["failure"]):
        raise AssertionError("Raw reward identity violated")
    return result


def train(config: dict, output: Path, variant: str, seed: int, pilot: bool,
          approval: Path | None = None) -> dict:
    if not pilot:
        from .experiments import require_review
        require_review(approval, config)
    if variant not in ("full", "masked") or seed not in config["ppo"]["training_seeds"]:
        raise ValueError("Variant and seed must belong to the frozen experiment")
    hp = config["ppo"]
    steps = 4096 if pilot else hp["total_env_steps"]
    num_envs, horizon = hp["num_envs"], hp["rollout_steps"]
    batch_size = num_envs * horizon
    if steps % batch_size or batch_size % hp["minibatch_size"]:
        raise ValueError("Budgets must contain complete rollouts and minibatches")
    if hp["gamma"] != 1 or hp["gae_lambda"] != 0.95 or hp["reward_scale"] != 0.01:
        raise ValueError("Frozen finite objective: gamma=1, lambda=.95, reward scale=.01")
    provenance = begin_run(output, config, "pilot_train" if pilot else "final_train",
                           [approval] if approval else [])
    seed_everything(seed)
    device = torch.device(config.get("device", "cpu"))
    model = ActorCritic(hp["hidden_sizes"]).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=hp["learning_rate"], eps=1e-5)
    ordinal = 0
    namespace = f"pilot/train/seed-{seed}" if pilot else f"train/seed-{seed}"
    regimes = tuple(REGIMES)

    def new_env():
        nonlocal ordinal
        scenario = make_scenario(namespace, ordinal, regimes[ordinal % len(regimes)])
        ordinal += 1
        return Environment(scenario), scenario

    slots = [new_env() for _ in range(num_envs)]
    task_buffers = [[] for _ in range(num_envs)]
    tasks: list[dict] = []
    sequences: list[dict] = []
    updates: list[dict] = []
    global_step = 0
    started = time.perf_counter()
    log_fields = ["env_step", "update", "rollout_step", "env_slot", "scenario_id", "regime", "task",
                  "action", "dt", "raw_reward", "scaled_reward", "sequence_done", "task_ended"]
    with (output / "steps.csv").open("w", newline="") as f:
        logger = csv.DictWriter(f, fieldnames=log_fields)
        logger.writeheader()
        for update in range(1, steps // batch_size + 1):
            observations = np.empty((horizon, num_envs, len(FEATURE_NAMES)), dtype=np.float32)
            actions = np.empty((horizon, num_envs), dtype=np.int64)
            logprobs = np.empty((horizon, num_envs), dtype=np.float32)
            values = np.empty_like(logprobs)
            rewards = np.empty_like(logprobs)
            dones = np.empty_like(logprobs)
            for t in range(horizon):
                observations[t] = np.stack([encode(env.snapshot, variant) for env, _ in slots])
                with torch.no_grad():
                    a, lp, v = model.sample(torch.as_tensor(observations[t], device=device))
                actions[t], logprobs[t], values[t] = a.cpu().numpy(), lp.cpu().numpy(), v.cpu().numpy()
                for e in range(num_envs):
                    env, scenario = slots[e]
                    task_index = env.snapshot.task
                    _, raw_reward, done, info = env.step(int(actions[t, e]))
                    rewards[t, e] = raw_reward * hp["reward_scale"]
                    dones[t, e] = done
                    global_step += 1
                    logger.writerow(dict(zip(log_fields, [global_step, update, t, e, scenario.scenario_id,
                        scenario.regime, task_index, actions[t, e], info["dt"], raw_reward,
                        rewards[t, e], int(done), int(info["task_ended"])])))
                    identity = {"policy": f"ppo_{variant}", "training_seed": seed,
                                "scenario_id": scenario.scenario_id, "namespace": namespace, "regime": scenario.regime}
                    if info["task_record"] is not None:
                        record = info["task_record"]
                        task_buffers[e].append(record)
                        tasks.append({**identity, **record, "env_step": global_step})
                    if done:
                        sequences.append(sequence_record(task_buffers[e], {**identity, "env_step": global_step}))
                        task_buffers[e] = []
                        slots[e] = new_env()
            with torch.no_grad():
                next_features = np.stack([encode(env.snapshot, variant) for env, _ in slots])
                next_values = model(torch.as_tensor(next_features, device=device))[1].cpu().numpy()
            advantages, returns = gae(rewards, values, dones, next_values, hp["gamma"], hp["gae_lambda"])
            obs = torch.as_tensor(observations.reshape(-1, len(FEATURE_NAMES)), device=device)
            act = torch.as_tensor(actions.ravel(), device=device)
            old_lp = torch.as_tensor(logprobs.ravel(), device=device)
            adv = torch.as_tensor(advantages.ravel(), device=device)
            targets = torch.as_tensor(returns.ravel(), device=device)
            # One batch-wide normalization, identical in both variants.
            adv = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
            losses = []
            for _ in range(hp["epochs"]):
                order = torch.randperm(batch_size, device=device)
                for start in range(0, batch_size, hp["minibatch_size"]):
                    idx = order[start:start + hp["minibatch_size"]]
                    logits, predicted = model(obs[idx])
                    dist = Categorical(logits=logits)
                    lp = dist.log_prob(act[idx])
                    logratio = lp - old_lp[idx]
                    ratio = logratio.exp()
                    unclipped = ratio * adv[idx]
                    clipped = ratio.clamp(1 - hp["clip_coefficient"], 1 + hp["clip_coefficient"]) * adv[idx]
                    policy_loss = -torch.minimum(unclipped, clipped).mean()
                    value_loss = 0.5 * (predicted - targets[idx]).square().mean()
                    entropy = dist.entropy().mean()
                    loss = policy_loss + hp["value_coefficient"] * value_loss - hp["entropy_coefficient"] * entropy
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), hp["max_grad_norm"])
                    optimizer.step()
                    with torch.no_grad():
                        losses.append([policy_loss.item(), value_loss.item(), entropy.item(),
                                       ((ratio - 1) - logratio).mean().item(),
                                       ((ratio - 1).abs() > hp["clip_coefficient"]).float().mean().item()])
            elapsed = time.perf_counter() - started
            means = np.mean(losses, axis=0)
            row = {"env_step": global_step, "update": update, "wall_seconds": elapsed,
                   "steps_per_second": global_step / elapsed,
                   **dict(zip(("policy_loss", "value_loss", "entropy", "approx_kl", "clip_fraction"), means.tolist()))}
            updates.append(row)
            f.flush()
            write_csv(output / "updates.csv", updates)
    wall = time.perf_counter() - started
    checkpoint = {"model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                  "variant": variant, "training_seed": seed, "env_steps": global_step,
                  "pilot": pilot, "config": config, "config_sha256": json_hash(config),
                  "source_sha256": provenance["source_sha256"], "features": feature_spec(),
                  "torch_rng_state": torch.get_rng_state(), "checkpoint_rule": "final_completed_budget"}
    torch.save(checkpoint, output / "checkpoint.pt")
    write_csv(output / "episodes.csv", tasks)
    write_csv(output / "sequences.csv", sequences)
    write_json(output / "episodes.json", tasks)
    write_json(output / "sequences.json", sequences)
    summary = {"kind": provenance["kind"], "variant": variant, "training_seed": seed,
               "env_steps": global_step, "completed_sequences": len(sequences),
               "partial_sequences_excluded": sum(bool(env.snapshot.task > 0 or env.snapshot.remaining < env.snapshot.layout.budget)
                                                 for env, _ in slots),
               "partial_completed_tasks": sum(map(len, task_buffers)),
               "wall_seconds": wall, "steps_per_second": global_step / wall,
               "compute_device": str(device), "source_sha256": provenance["source_sha256"],
               "estimated_ten_final_training_minutes": (10 * hp["total_env_steps"] * wall / global_step / 60)}
    finish_run(output, summary)
    return summary


def load_policy(path: Path, device: str = "cpu"):
    # All checkpoint payloads are tensors and primitive containers.
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    model = ActorCritic(checkpoint["config"]["ppo"]["hidden_sizes"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    variant = checkpoint["variant"]

    def act(snapshot):
        with torch.no_grad():
            features = torch.as_tensor(encode(snapshot, variant)[None], device=device)
            return int(model(features)[0].argmax(dim=-1).item())

    return act, checkpoint
