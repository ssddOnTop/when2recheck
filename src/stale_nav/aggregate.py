"""Raw-data-derived tables and plot data. No report prose is generated."""
from __future__ import annotations
from collections import defaultdict
import csv
import json
from pathlib import Path
import numpy as np

from .provenance import begin_run, finish_run, verify_artifacts, write_csv, write_json

METRICS = ("success_fraction", "time_per_task", "inspections_per_task", "blocked_attempts", "reward", "penalized_cost")


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def stats(values) -> dict:
    a = np.asarray(values, dtype=float)
    n = len(a)
    mean = float(np.mean(a))
    sd = float(np.std(a, ddof=1)) if n > 1 else 0.0
    sem = sd / np.sqrt(n) if n > 1 else None
    return {"n": n, "mean": mean, "sd": sd, "sem": sem,
            "normal_95_low": mean - 1.96 * sem if sem is not None else None,
            "normal_95_high": mean + 1.96 * sem if sem is not None else None}


def validate_raw(evaluation: Path):
    verify_artifacts(evaluation)
    sequences = read_csv(evaluation / "sequences.csv")
    episodes = read_csv(evaluation / "episodes.csv")
    key_fields = ("policy", "training_seed", "scenario_id")
    grouped = defaultdict(list)
    for task in episodes:
        grouped[tuple(task[k] for k in key_fields)].append(task)
    seen = set()
    for row in sequences:
        key = tuple(row[k] for k in key_fields)
        if key in seen:
            raise ValueError("Duplicate sequence row")
        seen.add(key)
        tasks = grouped.pop(key, [])
        if sorted(int(t["task"]) for t in tasks) != list(range(6)):
            raise ValueError("Evaluation sequence must have exactly six task rows")
        for metric in ("success", "failure", "time", "inspections", "blocked_attempts", "reward", "penalized_cost"):
            total = sum(float(t[metric]) for t in tasks)
            if not np.isclose(float(row[metric]), total, rtol=1e-10, atol=1e-8):
                raise ValueError(f"Sequence/task disagreement: {metric}")
        if not np.isclose(float(row["reward"]), 600 - float(row["time"]) - 200 * float(row["failure"])):
            raise ValueError("Failure-inclusive reward identity mismatch")
        for metric, numerator in (("success_fraction", "success"), ("time_per_task", "time"), ("inspections_per_task", "inspections")):
            if not np.isclose(float(row[metric]), float(row[numerator]) / 6):
                raise ValueError(f"Derived metric mismatch: {metric}")
    if grouped:
        raise ValueError("Unmatched task rows")
    return sequences


def check_learning_curve_kind(evaluation_kind: str, training_kind: str):
    if evaluation_kind not in {"pilot_evaluation", "val_evaluation", "test_evaluation"}:
        raise ValueError("Unknown evaluation kind")
    if training_kind not in {"pilot_train", "final_train"}:
        raise ValueError("Learning curves accept only genuine PPO training logs")
    if (evaluation_kind == "pilot_evaluation") != (training_kind == "pilot_train"):
        raise ValueError("Do not mix pilot and final learning curves")


def aggregate(evaluation: Path, training: list[Path], output: Path) -> dict:
    rows = validate_raw(evaluation)
    evaluation_provenance = json.loads((evaluation / "provenance.json").read_text())
    if evaluation_provenance["kind"] not in {"pilot_evaluation", "val_evaluation", "test_evaluation"}:
        raise ValueError("Expected an evaluation, not a training or TTL-selection directory")
    begin_run(output, evaluation_provenance["config"], "aggregate",
              [evaluation / "artifacts.json", *[p / "artifacts.json" for p in training]])
    by_seed = defaultdict(list)
    paired = defaultdict(dict)
    for row in rows:
        by_seed[(row["policy"], row["training_seed"], row["regime"])].append(row)
        paired[(row["policy"], row["regime"])].setdefault(row["scenario_id"], []).append(row)
    per_seed, across_seeds, differences = [], [], []
    seed_means = defaultdict(list)
    for (policy, seed, regime), group in sorted(by_seed.items()):
        for metric in METRICS:
            summary = stats([float(r[metric]) for r in group])
            per_seed.append({"policy": policy, "training_seed": seed, "regime": regime,
                             "metric": metric, "unit_of_uncertainty": "sequence", **summary})
            if seed != "baseline":
                seed_means[(policy, regime, metric)].append(summary["mean"])
    for (policy, regime, metric), means in sorted(seed_means.items()):
        across_seeds.append({"policy": policy, "regime": regime, "metric": metric,
                             "unit_of_uncertainty": "independent_training_seed", **stats(means)})
    # Average trained seeds within each paired scenario before sequence inference.
    # This is conditional on the available trained models, not seed uncertainty.
    paired_values = []
    for (policy, regime), scenarios in sorted(paired.items()):
        reference = paired[("reactive", regime)]
        if set(scenarios) != set(reference):
            raise ValueError("Policies do not share the same paired scenario IDs")
        seed_sets = {tuple(sorted(r["training_seed"] for r in group)) for group in scenarios.values()}
        if len(seed_sets) != 1:
            raise ValueError("Missing trained seed on some paired scenarios")
        for metric in METRICS:
            delta = []
            for scenario_id, group in sorted(scenarios.items()):
                mean = float(np.mean([float(r[metric]) for r in group]))
                baseline = float(reference[scenario_id][0][metric])
                delta.append(mean - baseline)
                paired_values.append({"policy": policy, "regime": regime, "scenario_id": scenario_id,
                                      "metric": metric, "model_mean": mean, "reactive": baseline,
                                      "paired_difference": mean - baseline, "model_count": len(group)})
            differences.append({"policy": policy, "reference": "reactive", "regime": regime,
                                "metric": metric, "unit_of_uncertainty": "paired_sequence_conditional_on_models",
                                **stats(delta)})
    write_csv(output / "per_seed_summary.csv", per_seed)
    write_csv(output / "training_seed_variation.csv", across_seeds)
    write_csv(output / "paired_differences.csv", differences)
    write_csv(output / "paired_sequence_values.csv", paired_values)
    learning, curves = [], []
    window = 2048  # Frozen before final training: one 16x128 rollout.
    groups = defaultdict(list)
    seen_runs = set()
    for directory in training:
        verify_artifacts(directory)
        provenance = json.loads((directory / "provenance.json").read_text())
        check_learning_curve_kind(evaluation_provenance["kind"], provenance["kind"])
        summary = json.loads((directory / "summary.json").read_text())
        run_key = (summary["variant"], summary["training_seed"])
        if run_key in seen_runs:
            raise ValueError("Duplicate learning-curve run")
        seen_runs.add(run_key)
        if provenance["source_sha256"] != evaluation_provenance["source_sha256"]:
            raise ValueError("Training/evaluation source mismatch")
        bins = defaultdict(list)
        for row in read_csv(directory / "sequences.csv"):
            end = ((int(row["env_step"]) - 1) // window + 1) * window
            bins[end].append(float(row["reward"]))
        for end, rewards in sorted(bins.items()):
            mean = float(np.mean(rewards))
            learning.append({"policy": f"ppo_{summary['variant']}", "training_seed": summary["training_seed"],
                             "window_end_env_steps": end, "completed_sequences": len(rewards), "mean_raw_return": mean})
            groups[(f"ppo_{summary['variant']}", end)].append(mean)
    for (policy, end), means in sorted(groups.items()):
        curves.append({"policy": policy, "window_end_env_steps": end,
                       "unit_of_variation": "training_seed", **stats(means)})
    write_csv(output / "learning_per_seed.csv", learning)
    write_csv(output / "learning_curve.csv", curves)
    if curves:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7, 4))
        for policy in sorted({r["policy"] for r in curves}):
            group = [r for r in curves if r["policy"] == policy]
            x = [r["window_end_env_steps"] for r in group]
            y = np.array([r["mean"] for r in group])
            sd = np.array([r["sd"] for r in group])
            counts = sorted({r["n"] for r in group})
            ax.plot(x, y, label=f"{policy}, seeds per window={counts}")
            if min(counts) > 1:
                ax.fill_between(x, y - sd, y + sd, alpha=.2)
        ax.set(xlabel="Training environment steps", ylabel="Mean raw six-task sequence return",
               title="Graph-only PPO: 2048-step windows, shading is seed SD")
        ax.legend()
        fig.tight_layout()
        fig.savefig(output / "learning_curve.png", dpi=150)
        plt.close(fig)
    result = {"evaluation_rows": len(rows), "per_seed_summary_rows": len(per_seed),
              "paired_summary_rows": len(differences), "training_runs": len(training),
              "learning_window_env_steps": window, "evaluation_source_sha256": evaluation_provenance["source_sha256"],
              "uncertainty": "Normal-approximation sequence intervals conditional on trained models; separate training-seed SD. Never flatten seeds times sequences.",
              "time_metric": "All-task consumed graph time, including budget-exhaustion failures; penalized cost adds 200 per failure."}
    write_json(output / "table_data.json", {"per_seed": per_seed, "seed_variation": across_seeds, "paired": differences})
    finish_run(output, result)
    return result
