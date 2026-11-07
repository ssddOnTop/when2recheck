"""Gated prospective commands. No operation here is run by implementation tests.

Use the manifest command to obtain review bindings. The original design PASS is
never execution authority. Empirical reports must call this a one-task active
sensing study and a cost/structural-information ablation, not an age ablation.
"""
from __future__ import annotations
from collections import defaultdict
import csv
import json
from pathlib import Path
import time
import numpy as np

from . import control as ctl
from .domain import (BASELINES, STRATA, NAMESPACES, ScenarioFactory, choose, encode,
                     feature_spec, OneTaskEnv, public_record, scenario_record, scenario_from_record)


class CSVLog:
    def __init__(self, path: Path):
        self.file = path.open("w", newline="")
        self.writer = None

    def row(self, row):
        row = {k: json.dumps(v, sort_keys=True) if isinstance(v, (list, tuple, dict)) else v for k, v in row.items()}
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=list(row))
            self.writer.writeheader()
        self.writer.writerow(row)

    def close(self):
        self.file.close()


def _identity(scenario):
    return {"scenario_id": scenario.scenario_id, "namespace": scenario.namespace, "stratum": scenario.stratum}


def generate_dataset(split: str, review: Path | None, output: Path, seed: int | None = None,
                     count: int | None = None, release: Path | None = None):
    operation = "generate_" + split
    grant = ctl.authorize(review, operation)
    factory = ScenarioFactory(split, seed, review, output, release)
    if split in ("validation", "heldout"):
        if count is not None:
            raise ctl.GateError("Evaluation counts are frozen, not a CLI tuning parameter")
        per_stratum = 64 if split == "validation" else 128
        indices = ((i, s) for s in range(12) for i in range(per_stratum))
        total = per_stratum * 12
    else:
        ceiling = (ctl.config()["pilot_steps"] if split == "pilot" else ctl.config()["ppo"]["total_env_steps_per_run"]) + ctl.config()["ppo"]["num_envs"]
        if count is None or not 1 <= count <= ceiling:
            raise ValueError("Train/pilot generation count must fit the declared per-run step budget plus initial environments")
        total = count
        indices = ((i // 12, i % 12) for i in range(count))
    output = ctl.begin(output, grant, operation, [release] if release else [])
    with (output / "scenarios.jsonl").open("w") as f:
        for index, stratum in indices:
            f.write(json.dumps(scenario_record(factory.generate(index, stratum)), sort_keys=True) + "\n")
    ctl.finish(output, {"split": split, "episodes": total, "seed": seed,
                        "namespace": NAMESPACES[split], "release": factory.release,
                        "unit_of_evaluation": "one_task_episode"}, grant)
    return {"output": str(output), "episodes": total, "split": split}


def validate_feasibility_summary(summary: dict):
    if summary.get("unit_fixture") or summary.get("env_steps") != 4096 or summary.get("status") != "FEASIBLE":
        raise ctl.GateError("Pilot does not permit final training; reviewed amendment may be required")
    wall = summary.get("wall_seconds", 0.)
    if not np.isfinite(wall) or wall <= 0:
        raise ctl.GateError("Invalid measured pilot wall time")
    projected = 10 * ctl.config()["ppo"]["total_env_steps_per_run"] * wall / 4096 / 60
    if not np.isclose(summary.get("estimated_all_final_minutes", float("inf")), projected) or projected > 90:
        raise ctl.GateError("Feasibility rule requires STOP, not automatic budget reduction")


def pilot_feasibility(directory: Path | None, bindings: dict):
    if directory is None:
        raise ctl.GateError("Final training requires the separate approved feasibility pilot")
    provenance = ctl.verify(directory, bindings)
    summary = ctl.read_json(directory / "summary.json")
    if provenance["kind"] != "pilot" or provenance.get("unit_fixture"):
        raise ctl.GateError("Artifact is not a genuine prospective pilot")
    validate_feasibility_summary(summary)


def train(review: Path | None, output: Path, variant: str, seed: int,
          pilot: bool = False, pilot_evidence: Path | None = None, device: str | None = None):
    operation = "pilot" if pilot else "train"
    grant = ctl.authorize(review, operation)
    c, hp = ctl.config(), ctl.config()["ppo"]
    if variant not in c["variants"] or seed not in c["training_seeds"]:
        raise ctl.GateError("Unapproved variant or seed")
    if pilot and (variant != c["pilot_variant"] or seed != c["pilot_seed"]):
        raise ctl.GateError("Only the predeclared full/1000 feasibility pilot is allowed")
    if not pilot:
        pilot_feasibility(pilot_evidence, grant.bindings)
    device = device or c["default_device"]
    if device not in c["allowed_devices"]:
        raise ctl.GateError("Unapproved device")
    factory = ScenarioFactory("pilot" if pilot else "train", seed, review, output)
    steps = c["pilot_steps"] if pilot else hp["total_env_steps_per_run"]
    batch_size = hp["num_envs"] * hp["rollout_steps"]
    if steps % batch_size or batch_size % hp["minibatch_size"]:
        raise ctl.GateError("Frozen budget/minibatch partition mismatch")
    inputs = [pilot_evidence / "artifacts.json"] if pilot_evidence else []
    output = ctl.begin(output, grant, operation, inputs)
    if pilot:
        receipts = ctl.EVIDENCE / "pilot-receipts"
        receipts.mkdir(parents=True, exist_ok=True)
        with (receipts / (grant.bindings["implementation_sha256"] + ".json")).open("x") as f:
            json.dump({"output": str(output), "review_sha256": grant.review_sha256}, f)
    # Import heavy learning dependencies only after gates and feasibility pass.
    import torch
    from .learning import ActorCritic, Slot, collect_rollout, execution_context, ppo_update
    trace, episode_log, updates = CSVLog(output / "steps.csv"), CSVLog(output / "episodes.csv"), CSVLog(output / "updates.csv")
    episodes = []
    ordinal = 0
    def new_slot():
        nonlocal ordinal
        scenario = factory.generate(ordinal // 12, ordinal % 12)
        ordinal += 1
        return Slot.from_scenario(scenario, grant)
    try:
        with execution_context(device):
            model = ActorCritic(seed, hp["hidden_sizes"]).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=hp["learning_rate"], eps=1e-5)
            generator = torch.Generator(device=device).manual_seed(seed)
            slots = [new_slot() for _ in range(hp["num_envs"])]
            start = time.perf_counter()
            for first_step in range(0, steps, batch_size):
                grant.check(operation)
                batch, completed = collect_rollout(model, slots, new_slot, generator, variant,
                    hp["rollout_steps"], hp["reward_scale"], device, first_step, trace.row)
                losses = ppo_update(model, optimizer, batch, hp, generator, device)
                for row in completed:
                    row = {"policy": "ppo_" + variant, "training_seed": seed, **row}
                    episodes.append(row)
                    episode_log.row(row)
                elapsed = time.perf_counter() - start
                updates.row({"env_steps": first_step + batch_size, "wall_seconds": elapsed,
                             "steps_per_second": (first_step + batch_size) / elapsed, **losses})
                trace.file.flush()
                episode_log.file.flush()
                updates.file.flush()
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            metadata = {**{k: grant.bindings[k] for k in ("study_id", *ctl.BINDING_KEYS)},
                        "variant": variant, "training_seed": seed, "env_steps": steps,
                        "pilot": pilot, "horizon_tasks": 1, "device": device,
                        "config": c, "features": feature_spec()}
            torch.save({**metadata, "model_state": model.state_dict(), "optimizer_state": optimizer.state_dict(),
                        "local_generator_state": generator.get_state()}, output / "checkpoint.pt")
            projected = 10 * hp["total_env_steps_per_run"] * elapsed / steps / 60
            summary = {"kind": operation, "variant": variant, "training_seed": seed, "env_steps": steps,
                       "completed_episodes": len(episodes), "wall_seconds": elapsed, "steps_per_second": steps / elapsed,
                       "estimated_all_final_minutes": projected,
                       "status": "STOP_REQUIRES_REVIEWED_AMENDMENT" if pilot and projected > 90 else ("FEASIBLE" if pilot else "COMPLETE"),
                       "device": device, "gpu": torch.cuda.get_device_name(0) if device == "cuda" else "not used",
                       "partial_episodes_excluded": sum(s.env.decisions > 0 for s in slots),
                       "generated_episode_count": factory.count, "horizon_tasks": 1,
                       "interpretation": c["ablation_interpretation"]}
    finally:
        for log in (trace, episode_log, updates):
            log.close()
    ctl.write_json(output / "episodes.json", episodes)
    ctl.finish(output, summary, grant)
    return summary


def load_dataset(directory: Path, grant: ctl.Grant, release: Path | None = None):
    provenance = ctl.verify(directory, grant.bindings)
    summary = ctl.read_json(directory / "summary.json")
    split = summary["split"]
    if split not in ("validation", "heldout") or provenance["kind"] != "generate_" + split:
        raise ctl.GateError("Evaluation requires a complete paired validation or held-out dataset")
    grant.check("evaluate_" + split)
    released = ctl.heldout_release(release, grant) if split == "heldout" else None
    if released and summary["release"] != released:
        raise ctl.GateError("Held-out dataset is not bound to this exact release")
    count = 64 if split == "validation" else 128
    expected = {f"{NAMESPACES[split]}/seed-evaluation/{family}/scale-{scale.replace('/', '_')}/miss-{miss.replace('/', '_')}/{i:08d}"
                for family, scale, miss in STRATA for i in range(count)}
    # No RNG or generator is invoked when reading a frozen paired dataset.
    with (directory / "scenarios.jsonl").open() as f:
        scenarios = [scenario_from_record(json.loads(line)) for line in f]
    actual = [s.scenario_id for s in scenarios]
    if len(actual) != len(expected) or set(actual) != expected or any(s.namespace != NAMESPACES[split] for s in scenarios):
        raise ctl.GateError("Dataset scenario coverage/namespace mismatch")
    if summary["episodes"] != len(expected):
        raise ctl.GateError("Dataset count mismatch")
    return split, scenarios, released


def roll_episode(scenario, actor, emit=None, grant=None, release=None):
    """Unit fixtures are ungated; prospective episodes require their grant."""
    env = OneTaskEnv(scenario, grant, release)
    first_action = None
    while True:
        before = env.snapshot
        action = actor(before)
        if first_action is None:
            first_action = action
        _, reward, done, info = env.step(action)
        if emit:
            emit({**_identity(scenario), "decision": env.decisions, "action": action,
                  "dt": info["dt"], "raw_reward": reward, "done": int(done),
                  "evidence": info["evidence"], "public_before": public_record(before)})
        if done:
            return {**_identity(scenario), "first_action": first_action, **info["episode"]}


def evaluate(review: Path | None, dataset: Path, checkpoints: list[Path], output: Path,
             split: str, release: Path | None = None, device: str | None = None):
    grant = ctl.authorize(review, "evaluate_" + split)
    if split not in ("validation", "heldout"):
        raise ctl.GateError("Only declared evaluation splits are supported")
    records = ctl.checkpoint_records(checkpoints, grant.bindings)
    actual_split, scenarios, released = load_dataset(dataset, grant, release)
    if actual_split != split:
        raise ctl.GateError("Requested split differs from dataset")
    hashes = sorted(ctl.sha(path) for path in checkpoints)
    if released and hashes != released["checkpoint_sha256"]:
        raise ctl.GateError("Evaluation models differ from released models")
    device = device or ctl.config()["default_device"]
    if device not in ctl.config()["allowed_devices"]:
        raise ctl.GateError("Unapproved evaluation device")
    output = ctl.begin(output, grant, "evaluate_" + split,
                       [dataset / "artifacts.json", *checkpoints, *([release] if release else [])])
    import torch
    from .learning import ActorCritic, execution_context
    rows = []
    traces, episodes = CSVLog(output / "steps.csv"), CSVLog(output / "episodes.csv")
    try:
        with execution_context(device):
            policies = [(name, "baseline", lambda s, name=name: choose(name, s)) for name in BASELINES]
            for record in records:
                model = ActorCritic(record["training_seed"], ctl.config()["ppo"]["hidden_sizes"]).to(device)
                model.load_state_dict(record["model_state"])
                model.eval()
                def act(snapshot, model=model, variant=record["variant"]):
                    with torch.no_grad():
                        tensor = torch.as_tensor(encode(snapshot, variant)[None], device=device)
                        return int(model(tensor)[0].argmax(dim=-1).item())
                policies.append(("ppo_" + record["variant"], record["training_seed"], act))
            for scenario in scenarios:
                for name, seed, actor in policies:
                    identity = {"policy": name, "training_seed": seed}
                    row = roll_episode(scenario, actor, lambda trace, identity=identity: traces.row({**identity, **trace}), grant, released)
                    row = {**identity, **row}
                    rows.append(row)
                    episodes.row(row)
    finally:
        traces.close()
        episodes.close()
    ctl.write_json(output / "episodes.json", rows)
    summary = {"split": split, "paired_episodes": len(scenarios), "policy_model_instances": len(BASELINES) + 10,
               "rows": len(rows), "checkpoint_sha256": hashes, "horizon_tasks": 1,
               "interpretation": ctl.config()["ablation_interpretation"], "device": device}
    ctl.finish(output, summary, grant)
    return summary


def seal_validation(review: Path | None, evaluation: Path, checkpoints: list[Path], output: Path):
    grant = ctl.authorize(review, "seal_validation")
    provenance = ctl.verify(evaluation, grant.bindings)
    summary = ctl.read_json(evaluation / "summary.json")
    ctl.checkpoint_records(checkpoints, grant.bindings)
    hashes = sorted(ctl.sha(p) for p in checkpoints)
    if provenance["kind"] != "evaluate_validation" or summary.get("split") != "validation" or summary.get("paired_episodes") != 768:
        raise ctl.GateError("Only complete declared validation evidence can be frozen")
    if summary.get("checkpoint_sha256") != hashes:
        raise ctl.GateError("Validation model set mismatch")
    output = ctl.begin(output, grant, "seal_validation", [evaluation / "artifacts.json", *checkpoints])
    decision = {"frozen": True, "split": "validation", "checkpoint_sha256": hashes,
                "evaluation_artifacts_sha256": ctl.sha(evaluation / "artifacts.json"),
                "decision": "Use the approved fixed configuration and final-budget models. No TTL or hyperparameter selection is implemented. Amendments require a new review before held-out release."}
    ctl.write_json(output / "decisions.json", decision)
    ctl.finish(output, decision, grant)
    return decision


def _stats(values):
    a = np.asarray(values, dtype=float)
    return {"n": len(a), "mean": float(a.mean()),
            "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.,
            "sem": float(a.std(ddof=1) / np.sqrt(len(a))) if len(a) > 1 else None}


def validate_episode_rows(rows, episodes_per_stratum):
    expected_slots = {(name, "baseline") for name in BASELINES} | {(f"ppo_{v}", str(s)) for v, s in ctl.expected_models()}
    expected_strata = {"|".join(s) for s in STRATA}
    coverage, strata = defaultdict(set), defaultdict(set)
    for row in rows:
        key = (row["stratum"], row["scenario_id"])
        slot = (row["policy"], str(row["training_seed"]))
        if row["stratum"] not in expected_strata or slot not in expected_slots or slot in coverage[key]:
            raise ctl.GateError("Unexpected/duplicate episode, stratum or model")
        coverage[key].add(slot)
        strata[row["stratum"]].add(row["scenario_id"])
        if row["failure"] not in (0, 1) or row["success"] != 1 - row["failure"] or row["first_action"] not in range(7):
            raise ctl.GateError("Invalid terminal record")
        if not np.isclose(row["reward"], 100 - row["penalized_cost"]) or not np.isclose(row["penalized_cost"], row["time"] + 200 * row["failure"]):
            raise ctl.GateError("Raw objective identity mismatch")
    if set(strata) != expected_strata or any(len(ids) != episodes_per_stratum for ids in strata.values()):
        raise ctl.GateError("Incomplete predeclared stratum coverage")
    if any(slots != expected_slots for slots in coverage.values()):
        raise ctl.GateError("Missing paired policy/model episode")


def verify_trace_totals(rows, trace_path):
    totals = {}
    with trace_path.open(newline="") as f:
        for trace in csv.DictReader(f):
            key = (trace["policy"], trace["training_seed"], trace["scenario_id"])
            item = totals.setdefault(key, {"time": 0., "reward": 0., "decisions": 0, "done": False, "first_action": int(trace["action"])})
            if item["done"] or int(trace["decision"]) != item["decisions"] + 1:
                raise ctl.GateError("Trace continues after termination or has missing decisions")
            item["time"] += float(trace["dt"])
            item["reward"] += float(trace["raw_reward"])
            item["decisions"] += 1
            item["done"] = bool(int(trace["done"]))
    for row in rows:
        item = totals.pop((row["policy"], str(row["training_seed"]), row["scenario_id"]), None)
        if item is None or not item["done"]:
            raise ctl.GateError("Episode has no complete trace")
        if item["decisions"] != row["decisions"] or item["first_action"] != row["first_action"]:
            raise ctl.GateError("Episode/trace decision mismatch")
        if not np.isclose(item["time"], row["time"]) or not np.isclose(item["reward"], row["reward"]):
            raise ctl.GateError("Episode/trace total mismatch")
    if totals:
        raise ctl.GateError("Unmatched trace episodes")


def aggregate(review: Path | None, evaluation: Path, output: Path):
    grant = ctl.authorize(review, "aggregate")
    p = ctl.verify(evaluation, grant.bindings)
    if p["kind"] not in ("evaluate_validation", "evaluate_heldout"):
        raise ctl.GateError("Not a prospective evaluation artifact")
    rows = ctl.read_json(evaluation / "episodes.json")
    validate_episode_rows(rows, 64 if p["kind"] == "evaluate_validation" else 128)
    verify_trace_totals(rows, evaluation / "steps.csv")
    grouped, paired = defaultdict(list), defaultdict(dict)
    for row in rows:
        if not np.isclose(row["reward"], 100 - row["penalized_cost"]) or not np.isclose(row["penalized_cost"], row["time"] + 200 * row["failure"]):
            raise ctl.GateError("Raw objective identity mismatch")
        grouped[(row["policy"], str(row["training_seed"]), row["stratum"])].append(row)
        paired[(row["policy"], row["stratum"])].setdefault(row["scenario_id"], []).append(row)
    summaries, differences, seed_spread, first_actions = [], [], [], []
    seed_means = defaultdict(list)
    metrics = ("penalized_cost", "time", "failure", "reward", "inspections", "blocked_attempts")
    for (policy, seed, stratum), group in sorted(grouped.items()):
        for action in range(7):
            count = sum(r["first_action"] == action for r in group)
            first_actions.append({"policy": policy, "seed": seed, "stratum": stratum,
                                  "action": action, "count": count, "episodes": len(group), "fraction": count / len(group)})
        for metric in metrics:
            stat = _stats([r[metric] for r in group])
            summaries.append({"policy": policy, "seed": seed, "stratum": stratum, "metric": metric, **stat})
            if seed != "baseline":
                seed_means[(policy, stratum, metric)].append(stat["mean"])
    for (policy, stratum), scenarios in sorted(paired.items()):
        reference = paired[("ALWAYS_SAFE", stratum)]
        if set(scenarios) != set(reference):
            raise ctl.GateError("Unpaired scenario IDs")
        for metric in metrics:
            delta = [np.mean([r[metric] for r in group]) - reference[sid][0][metric] for sid, group in sorted(scenarios.items())]
            stat = _stats(delta)
            differences.append({"policy": policy, "reference": "ALWAYS_SAFE", "stratum": stratum,
                                "metric": metric, "uncertainty_unit": "paired_episode_conditional_on_models", **stat})
    for (policy, stratum, metric), values in sorted(seed_means.items()):
        seed_spread.append({"policy": policy, "stratum": stratum, "metric": metric,
                            "uncertainty_unit": "independent_training_seed", **_stats(values)})
    mixture_groups = defaultdict(list)
    for row in differences:
        mixture_groups[(row["policy"], row["metric"])].append(row)
    mixture = []
    for (policy, metric), group in sorted(mixture_groups.items()):
        if len(group) != 12:
            raise ctl.GateError("Equal-weight mixture requires every predeclared stratum")
        mixture.append({"policy": policy, "metric": metric, "reference": "ALWAYS_SAFE",
                        "mean": sum(r["mean"] for r in group) / 12,
                        "sem": float(np.sqrt(sum(r["sem"] ** 2 for r in group)) / 12),
                        "paired_episodes": sum(r["n"] for r in group),
                        "uncertainty_unit": "stratified_paired_episode_conditional_on_models"})
    output = ctl.begin(output, grant, "aggregate", [evaluation / "artifacts.json"])
    for name, values in (("per_seed", summaries), ("paired_vs_always_safe", differences),
                         ("training_seed_spread", seed_spread), ("first_action_frequencies", first_actions),
                         ("equal_weight_mixture", mixture)):
        ctl.write_json(output / (name + ".json"), values)
        log = CSVLog(output / (name + ".csv"))
        try:
            for row in values:
                log.row(row)
        finally:
            log.close()
    summary = {"rows": len(rows), "unit": "one_task_episode", "interpretation": ctl.config()["ablation_interpretation"],
               "scope": "Graph-only active sensing; no age/stale-memory or PPO superiority claim."}
    ctl.finish(output, summary, grant)
    return summary
