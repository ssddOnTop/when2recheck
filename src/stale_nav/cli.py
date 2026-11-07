"""Command-line entry points. Full training requires an independent review file."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from .provenance import ROOT, json_hash, source_manifest


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="stale-nav", description="Graph-only frozen experiment")
    sub = p.add_subparsers(dest="command", required=True)
    for command in ("smoke", "train", "evaluate", "select-ttl", "manifest"):
        s = sub.add_parser(command)
        s.add_argument("--config", type=Path, default=ROOT / "configs/final.json")
        if command != "manifest":
            s.add_argument("--output", type=Path, required=True)
        if command in ("smoke", "train"):
            s.add_argument("--variant", choices=("full", "masked"), default="full")
            s.add_argument("--seed", type=int, default=0)
        if command == "train":
            s.add_argument("--review", type=Path, required=True,
                           help="Independent PASS_FOR_TRAINING review bound to source/config SHA256")
        if command == "evaluate":
            s.add_argument("--split", choices=("pilot", "val", "test"), required=True)
            s.add_argument("--checkpoints", type=Path, nargs="*", default=[])
            s.add_argument("--ttl-selection", type=Path)
            s.add_argument("--pilot-sequences", type=int, default=4)
    s = sub.add_parser("aggregate")
    s.add_argument("--evaluation", type=Path, required=True)
    s.add_argument("--training", type=Path, nargs="*", default=[])
    s.add_argument("--output", type=Path, required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    from .experiments import evaluate, load_config, require_review, select_ttl
    if args.command == "aggregate":
        from .aggregate import aggregate
        result = aggregate(args.evaluation, args.training, args.output)
    else:
        config = load_config(args.config)
        if args.command == "manifest":
            result = {**source_manifest(), "config_sha256": json_hash(config)}
        elif args.command in ("smoke", "train"):
            from .ppo import train
            if args.seed not in config["ppo"]["training_seeds"]:
                raise ValueError("Seed is not in the frozen training seed set")
            review = args.review if args.command == "train" else None
            if args.command == "train":
                require_review(review, config)
            result = train(config, args.output, args.variant, args.seed, args.command == "smoke", review)
        elif args.command == "select-ttl":
            result = select_ttl(config, args.output)
        else:
            result = evaluate(config, args.output, args.split, args.checkpoints,
                              args.ttl_selection, args.pilot_sequences)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
