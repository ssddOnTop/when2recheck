"""Split-safe evaluation and validation-only baseline selection."""
from __future__ import annotations
import json
import re
import time
from functools import partial
from pathlib import Path
import numpy as np

from .environment import Environment, REGIMES, make_scenario
from .policies import choose
from .ppo import load_policy, sequence_record
from .provenance import (ROOT, begin_run, finish_run, json_hash, sha256,
                         source_manifest, verify_artifacts, write_csv, write_json)


def load_config(path: Path) -> dict:
    config = json.loads(path.read_text())
    reference = json.loads((ROOT / "configs/frozen.json").read_text())
    # All deviations except predeclared throughput fallbacks are rejected.
    normalized = json.loads(json.dumps(config))
    hp = normalized["ppo"]
    if hp["total_env_steps"] not in (196608, 98304):
        raise ValueError("Unapproved final budget")
    if hp["training_seeds"] not in ([0, 1, 2, 3, 4], [0, 1, 2]):
        raise ValueError("Unapproved seed set")
    reduced = hp["total_env_steps"] != 196608 or len(hp["training_seeds"]) != 5
    if len(hp["training_seeds"]) == 3 and hp["total_env_steps"] != 98304:
        raise ValueError("Three-seed fallback follows the reduced-step fallback")
    if reduced and not normalized.get("fallback_reason"):
        raise ValueError("Record measured throughput and pre-test fallback reason")
    hp["total_env_steps"] = 196608
    hp["training_seeds"] = [0, 1, 2, 3, 4]
    normalized.pop("fallback_reason", None)
    if not re.fullmatch(r"cpu|cuda(?::[0-9]+)?", str(normalized.get("device"))):
        raise ValueError("Device must be cpu, cuda, or cuda:N")
    normalized["device"] = reference["device"]
    if normalized != reference:
        raise ValueError("Configuration differs from frozen protocol")
    return config


def require_review(path: Path | None, config: dict):
    if path is None:
        raise ValueError("Full training is blocked until an independent implementation review is supplied")
    review = json.loads(path.read_text())
    if review.get("verdict") != "PASS_FOR_TRAINING":
        raise ValueError("Review must explicitly pass implementation for training")
    if review.get("source_sha256") != source_manifest()["source_sha256"]:
        raise ValueError("Implementation review does not match current source")
    if review.get("config_sha256") != json_hash(config):
        raise ValueError("Implementation review does not match final config")
    if review.get("remaining_blockers", []):
        raise ValueError("Review lists unresolved blockers")


def run_sequence(scenario, policy, identity: dict):
    env = Environment(scenario)
    tasks = []
    while True:
        action = policy(env.snapshot)
        _, _, done, info = env.step(action)
        if info["task_record"] is not None:
            tasks.append(info["task_record"])
        if done:
            break
    return [{**identity, **task} for task in tasks], sequence_record(tasks, identity)


def select_ttl(config: dict, output: Path) -> dict:
    """Exactly 64 validation sequences per regime; no test inputs accepted."""
    begin_run(output, config, "validation_ttl_selection")
    rows, task_rows = [], []
    selection = {}
    for regime in REGIMES:
        scores = {}
        for k in config["ttl_candidates"]:
            costs = []
            for index in range(64):
                scenario = make_scenario("val", index, regime)
                identity = {"policy": "ttl", "threshold": k, "training_seed": "baseline",
                            "scenario_id": scenario.scenario_id, "namespace": "val", "regime": regime}
                tasks, row = run_sequence(scenario, partial(choose, "ttl", threshold=k), identity)
                rows.append(row)
                task_rows.extend(tasks)
                costs.append(row["penalized_cost"])
            scores[k] = float(np.mean(costs))
        selection[regime] = min(scores, key=lambda k: (scores[k], k))
    write_csv(output / "sequences.csv", rows)
    write_csv(output / "episodes.csv", task_rows)
    write_json(output / "sequences.json", rows)
    write_json(output / "episodes.json", task_rows)
    frozen = {"thresholds": selection, "namespace": "val", "sequences_per_regime": 64,
              "candidates": [1, 2, 3, 6], "objective": "mean_sequence_penalized_cost",
              "tie_rule": "smaller_threshold", "config_sha256": json_hash(config),
              "source_sha256": source_manifest()["source_sha256"],
              "validation_rows_sha256": sha256(output / "sequences.csv")}
    write_json(output / "ttl_selection.json", frozen)
    finish_run(output, frozen)
    return frozen


def load_ttl(path: Path, config: dict) -> dict:
    verify_artifacts(path.parent)
    selection = json.loads(path.read_text())
    if selection.get("namespace") != "val" or selection.get("sequences_per_regime") != 64:
        raise ValueError("TTL must be selected on exactly 64 validation sequences per regime")
    if selection.get("source_sha256") != source_manifest()["source_sha256"]:
        raise ValueError("TTL selection source mismatch")
    if selection.get("config_sha256") != json_hash(config):
        raise ValueError("TTL selection configuration mismatch")
    if selection.get("validation_rows_sha256") != sha256(path.parent / "sequences.csv"):
        raise ValueError("Validation raw data mismatch")
    if set(selection["thresholds"]) != set(REGIMES) or any(k not in (1, 2, 3, 6) for k in selection["thresholds"].values()):
        raise ValueError("Invalid TTL thresholds")
    return selection["thresholds"]


def validate_checkpoint_set(checkpoints: list[dict], config: dict, split: str):
    seen = set()
    for ck in checkpoints:
        key = (ck["variant"], ck["training_seed"])
        if key in seen:
            raise ValueError("Duplicate model seed")
        seen.add(key)
        if split != "pilot":
            if ck["pilot"] or ck["env_steps"] != config["ppo"]["total_env_steps"]:
                raise ValueError("Only final completed-budget checkpoints can enter validation or test")
            if ck["source_sha256"] != source_manifest()["source_sha256"] or ck["config_sha256"] != json_hash(config):
                raise ValueError("Checkpoint source/config mismatch")
    if split == "test":
        expected = {(variant, seed) for variant in ("full", "masked") for seed in config["ppo"]["training_seeds"]}
        if seen != expected:
            raise ValueError("Test requires every matched full/masked final seed, not selected completed seeds")


def evaluate(config: dict, output: Path, split: str, checkpoint_paths: list[Path],
             ttl_path: Path | None = None, pilot_sequences: int = 4) -> dict:
    if split not in ("pilot", "val", "test"):
        raise ValueError("Evaluation split must be pilot, val or test")
    from .adapters.compute import compute_metadata, create_compute
    compute = create_compute(config.get("device", "cpu"))
    count = {"val": 64, "test": 128, "pilot": pilot_sequences}[split]
    if count <= 0:
        raise ValueError("Sequence count must be positive")
    thresholds = {r: 1 for r in REGIMES} if split == "pilot" and ttl_path is None else None
    if thresholds is None:
        if ttl_path is None:
            raise ValueError("Validation/test evaluation requires a frozen validation TTL selection")
        thresholds = load_ttl(ttl_path, config)
    loaded = [load_policy(path, compute=compute) for path in checkpoint_paths]
    validate_checkpoint_set([ck for _, ck in loaded], config, split)
    inputs = checkpoint_paths + ([ttl_path] if ttl_path else [])
    provenance = begin_run(output, config, f"{split}_evaluation", inputs)
    # Baselines execute only Python/NumPy on CPU, even when a GPU was selected.
    if not loaded:
        compute = create_compute("cpu")
    execution = compute_metadata(compute)
    provenance.update({**execution, "requested_device": config.get("device", "cpu")})
    write_json(output / "provenance.json", provenance)
    compute.synchronize()
    start = time.perf_counter()
    tasks, sequences, scenario_rows = [], [], []
    for regime in REGIMES:
        policies = [(name, "baseline", partial(choose, name, threshold=thresholds[regime]))
                    for name in ("reactive", "ttl", "reset", "voi")]
        policies += [(f"ppo_{ck['variant']}", ck["training_seed"], act) for act, ck in loaded]
        for index in range(count):
            scenario = make_scenario(split, index, regime)
            # World hashes are output provenance only, never policy features.
            scenario_rows.append({"scenario_id": scenario.scenario_id, "regime": regime,
                                  "namespace": split, "world_sha256": json_hash(scenario.worlds),
                                  "layout_sha256": json_hash(scenario.layout.__dict__)})
            for name, seed, act in policies:
                identity = {"policy": name, "training_seed": seed, "scenario_id": scenario.scenario_id,
                            "namespace": split, "regime": regime}
                task, row = run_sequence(scenario, act, identity)
                tasks.extend(task)
                sequences.append(row)
    write_csv(output / "episodes.csv", tasks)
    write_csv(output / "sequences.csv", sequences)
    write_json(output / "episodes.json", tasks)
    write_json(output / "sequences.json", sequences)
    write_csv(output / "scenarios.csv", scenario_rows)
    compute.synchronize()
    wall = time.perf_counter() - start
    summary = {**execution, "requested_device": config.get("device", "cpu"),
               "split": split, "sequences_per_regime": count, "regimes": list(REGIMES),
               "model_seed_count": len(loaded), "ttl_thresholds": thresholds,
               "pilot_ttl_unselected": split == "pilot" and ttl_path is None,
               "wall_seconds": wall, "policy_sequences_per_second": len(sequences) / wall,
               "evaluation_action_rule": "PPO deterministic argmax, lowest action index on ties",
               "source_sha256": provenance["source_sha256"],
               "estimated_full_evaluation_minutes": wall * (128 / count) * ((4 + 10) / (4 + len(loaded))) / 60}
    finish_run(output, summary)
    return summary
