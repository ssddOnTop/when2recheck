"""Command-line entry points. Full training requires an independent review file."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

from .provenance import ROOT, json_hash, source_manifest


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="stale-nav", description="Frozen graph study and generic development learner")
    sub = p.add_subparsers(dest="command", required=True)
    for command in ("smoke", "train", "evaluate", "select-ttl", "manifest"):
        s = sub.add_parser(command)
        s.add_argument("--config", type=Path, default=ROOT / "configs/final.json")
        if command != "manifest":
            s.add_argument("--output", type=Path, required=True)
        if command in ("smoke", "train"):
            s.add_argument("--variant", choices=("full", "masked"), default="full")
            s.add_argument("--seed", type=int, default=0)
            s.add_argument("--visualize", action="store_true")
            s.add_argument("--visualization-port", type=int, default=0)
        if command == "train":
            s.add_argument("--review", type=Path, required=True,
                           help="Independent PASS_FOR_TRAINING review bound to source/config SHA256")
        if command == "evaluate":
            s.add_argument("--split", choices=("pilot", "val", "test"), required=True)
            s.add_argument("--checkpoints", type=Path, nargs="*", default=[])
            s.add_argument("--ttl-selection", type=Path)
            s.add_argument("--pilot-sequences", type=int, default=4)
    s = sub.add_parser("learn", help="Development only, not approved final-study training")
    s.add_argument("--backend", default="graph", help="graph, trusted module:factory or installed entry point")
    s.add_argument("--backend-config", default="{}", help="Backend JSON object (inline), or path to a JSON file")
    s.add_argument("--steps", type=int, default=4096)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--variant", default="full")
    s.add_argument("--device", default="cpu")
    s.add_argument("--output", type=Path, required=True)
    s.add_argument("--visualize", action="store_true")
    s.add_argument("--visualization-port", type=int, default=0)
    s = sub.add_parser("aggregate")
    s.add_argument("--evaluation", type=Path, required=True)
    s.add_argument("--training", type=Path, nargs="*", default=[])
    s.add_argument("--output", type=Path, required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "learn":
        from .application import learn_development
        text = args.backend_config
        try:
            backend_config = json.loads(text)
        except json.JSONDecodeError:
            backend_config = json.loads(Path(text).read_text())
        if not isinstance(backend_config, dict):
            raise ValueError("Backend configuration must be a JSON object")
        result = learn_development(backend=args.backend, backend_config=backend_config,
                                   steps=args.steps, seed=args.seed, variant=args.variant,
                                   device=args.device, output=args.output, visualize=args.visualize,
                                   visualization_port=args.visualization_port)
        print(json.dumps(result, indent=2, sort_keys=True))
        return
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
            result = train(config, args.output, args.variant, args.seed, args.command == "smoke", review,
                           visualize=args.visualize, visualization_port=args.visualization_port)
        elif args.command == "select-ttl":
            result = select_ttl(config, args.output)
        else:
            result = evaluate(config, args.output, args.split, args.checkpoints,
                              args.ttl_selection, args.pilot_sequences)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
