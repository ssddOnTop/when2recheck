#!/usr/bin/env python3
"""Explicitly authorized final training only. Does not select TTL or inspect test."""
import argparse
from pathlib import Path
from stale_nav.experiments import load_config, require_review
from stale_nav.ppo import train
from stale_nav.provenance import ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/final.json")
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    require_review(args.review, config)
    args.output.mkdir(parents=True, exist_ok=False)
    for seed in config["ppo"]["training_seeds"]:
        for variant in ("full", "masked"):
            output = args.output / f"{variant}-seed-{seed}"
            result = train(config, output, variant, seed, pilot=False, approval=args.review)
            print(result, flush=True)


if __name__ == "__main__":
    main()
