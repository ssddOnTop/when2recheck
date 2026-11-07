"""Concrete artifact stores, checkpoint I/O and graph-specific raw CSV schema."""
import csv
import json
from pathlib import Path
import numpy as np
import torch

from ..provenance import (begin_run, finish_run, json_hash, source_manifest,
                          write_csv, write_json)


def load_checkpoint(path: Path, device="cpu"):
    return torch.load(path, map_location=device, weights_only=True)


class DevelopmentStore:
    """Explicitly non-study artifacts. Never resumes or overwrites partial output."""
    wants_scene = False

    def __init__(self, output: Path):
        self.output = Path(output)
        self.stream = None
        self.updates = []

    def start(self, metadata):
        self.output.mkdir(parents=True, exist_ok=False)
        write_json(self.output / "provenance.json", {**source_manifest(), **metadata,
                                                    "kind": "development_learn", "approved_final_study": False})
        self.stream = (self.output / "events.jsonl").open("x")

    def on_event(self, event):
        # Scenes belong to optional replay adapters, not the default event log.
        self.stream.write(json.dumps({k: v for k, v in event.items() if k != "scene"}, allow_nan=False) + "\n")
        if event["type"] == "update":
            self.updates.append(event["metrics"])
            self.stream.flush()
            write_csv(self.output / "updates.csv", self.updates)

    def save(self, result, metadata):
        self.close()
        checkpoint = {**metadata, "kind": "development_learn", "approved_final_study": False,
                      "pilot": True, "env_steps": result.metrics["env_steps"],
                      "model_state": result.model.state_dict(),
                      "optimizer_state": result.optimizer.state_dict()}
        torch.save(checkpoint, self.output / "checkpoint.pt")
        summary = {**metadata, **result.metrics, "kind": "development_learn", "approved_final_study": False}
        finish_run(self.output, summary)
        return summary

    def close(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None


class GraphStudyStore:
    """Preserve legacy steps/tasks/sequences/checkpoint schemas at the boundary."""
    wants_scene = False
    log_fields = ["env_step", "update", "rollout_step", "env_slot", "scenario_id", "regime", "task",
                  "action", "dt", "raw_reward", "scaled_reward", "sequence_done", "task_ended"]

    def __init__(self, output, config, variant, seed, pilot, approval, factory):
        self.output, self.config = Path(output), config
        self.variant, self.seed, self.pilot = variant, seed, pilot
        self.approval, self.factory = approval, factory
        self.stream = None
        self.tasks, self.sequences, self.updates = [], [], []
        self.task_buffers = [[] for _ in range(config["ppo"]["num_envs"])]

    def close(self):
        if self.stream is not None:
            self.stream.close()
            self.stream = None

    def start(self, metadata):
        self.provenance = begin_run(self.output, self.config, "pilot_train" if self.pilot else "final_train",
                                    [self.approval] if self.approval else [])
        self.provenance.update(metadata)
        write_json(self.output / "provenance.json", self.provenance)
        self.stream = (self.output / "steps.csv").open("x", newline="")
        self.logger = csv.DictWriter(self.stream, fieldnames=self.log_fields)
        self.logger.writeheader()

    def on_event(self, event):
        if event["type"] == "update":
            self.updates.append(event["metrics"])
            self.stream.flush()
            write_csv(self.output / "updates.csv", self.updates)
        if event["type"] != "transition":
            return
        info, e, step = event["info"], event["env_slot"], event["step"]
        self.logger.writerow(dict(zip(self.log_fields, [step, event["update"], event["rollout_step"], e,
            info["scenario_id"], info["regime"], info["task"], event["action"], info["dt"],
            event["reward"], np.float32(event["scaled_reward"]), int(event["terminated"]), int(info["task_ended"])])))
        identity = {"policy": f"ppo_{self.variant}", "training_seed": self.seed,
                    "scenario_id": info["scenario_id"], "namespace": info["namespace"], "regime": info["regime"]}
        if info["task_record"] is not None:
            record = info["task_record"]
            self.task_buffers[e].append(record)
            self.tasks.append({**identity, **record, "env_step": step})
        if event["terminated"]:
            from .graph import sequence_record
            self.sequences.append(sequence_record(self.task_buffers[e], {**identity, "env_step": step}))
            self.task_buffers[e] = []

    def save(self, result, metadata):
        self.close()
        from ..features import feature_spec
        metrics = result.metrics
        checkpoint = {**metadata, "model_state": result.model.state_dict(), "optimizer_state": result.optimizer.state_dict(),
                      "variant": self.variant, "training_seed": self.seed, "env_steps": metrics["env_steps"],
                      "pilot": self.pilot, "config": self.config, "config_sha256": json_hash(self.config),
                      "source_sha256": self.provenance["source_sha256"], "features": feature_spec(),
                      "torch_rng_state": torch.get_rng_state(), "checkpoint_rule": "final_completed_budget"}
        torch.save(checkpoint, self.output / "checkpoint.pt")
        write_csv(self.output / "episodes.csv", self.tasks)
        write_csv(self.output / "sequences.csv", self.sequences)
        write_json(self.output / "episodes.json", self.tasks)
        write_json(self.output / "sequences.json", self.sequences)
        summary = {**metadata, "kind": self.provenance["kind"], "variant": self.variant, "training_seed": self.seed,
                   "env_steps": metrics["env_steps"], "completed_sequences": len(self.sequences),
                   "partial_sequences_excluded": self.factory.partial_sequences(),
                   "partial_completed_tasks": sum(map(len, self.task_buffers)),
                   "wall_seconds": metrics["wall_seconds"], "steps_per_second": metrics["steps_per_second"],
                   "compute_device": metrics["compute_device"], "source_sha256": self.provenance["source_sha256"],
                   "estimated_ten_final_training_minutes": (
                       10 * self.config["ppo"]["total_env_steps"] * metrics["wall_seconds"] / metrics["env_steps"] / 60)}
        finish_run(self.output, summary)
        return summary
