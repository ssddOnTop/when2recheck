"""Isolated, default-refusing prospective study interface.

Run through followup/remote_python.sh. It invokes the parent-owned scoped_run.sh
and redirects Python/library caches below the follow-up implementation directory.
Examples, for a future independently approved execution only:
  python -m followup manifest
  python -m followup pilot --review REVIEW.json --output IMPLEMENTATION/pilot
  python -m followup train --review REVIEW.json --variant full --seed 1000
      --pilot-evidence IMPLEMENTATION/pilot --output IMPLEMENTATION/full-1000
  python -m followup generate --review REVIEW.json --split validation --output DIR
  python -m followup evaluate --review REVIEW.json --split validation
      --dataset DIR --checkpoints TEN_FINAL_CHECKPOINTS --output OUT
  python -m followup seal-validation --review REVIEW.json --evaluation DIR
      --checkpoints TEN_FINAL_CHECKPOINTS --output OUT
  python -m followup generate --review REVIEW.json --split heldout
      --release INDEPENDENT_RELEASE.json --output DIR

All output paths must be below ../local-materials/followup/implementation.
The implementation approval must bind every field printed by manifest and grant
explicit permissions. The design approval does NOT suffice. A separate held-out
release binds ten final checkpoints and frozen validation decisions. A >90-minute
pilot projection stops final training. No automatic reduced-budget fallback.

This is a one-task graph-only active sensing study. The effective feature contrast
is cost/structural information, not age: encoded ages and task index are invariant.
Masked inputs also hide affordable-door identity. No PPO advantage is promised.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from . import control as ctl


def parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = p.add_subparsers(dest="command", required=True)
    manifest = commands.add_parser("manifest", help="Read-only bindings; optional new review manifest file")
    manifest.add_argument("--output", type=Path)
    verify = commands.add_parser("verify", help="Read-only verification of an existing follow-up artifact")
    verify.add_argument("directory", type=Path)
    for name in ("generate", "pilot", "train", "evaluate", "seal-validation", "aggregate"):
        s = commands.add_parser(name)
        s.add_argument("--review", type=Path, help="Independent implementation approval; missing means refusal")
        s.add_argument("--output", type=Path, required=True)
        if name in ("generate", "evaluate"):
            choices = ("train", "validation", "val", "heldout", "test", "pilot") if name == "generate" else ("validation", "val", "heldout", "test")
            s.add_argument("--split", choices=choices, required=True)
            s.add_argument("--release", type=Path)
        if name == "generate":
            s.add_argument("--seed", type=int)
            s.add_argument("--count", type=int)
        if name in ("train", "pilot", "evaluate"):
            s.add_argument("--device", choices=("cpu", "cuda"))
        if name == "train":
            s.add_argument("--variant", choices=("full", "masked"), required=True)
            s.add_argument("--seed", type=int, required=True)
            s.add_argument("--pilot-evidence", type=Path)
        if name in ("evaluate", "seal-validation"):
            s.add_argument("--checkpoints", type=Path, nargs="+", required=True)
        if name == "evaluate":
            s.add_argument("--dataset", type=Path, required=True)
        if name in ("seal-validation", "aggregate"):
            s.add_argument("--evaluation", type=Path, required=True)
    return p


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    try:
        if args.command == "manifest":
            result = ctl.manifest()
            result["required_execution_review"] = {
                "verdict": "PASS_FOR_FOLLOWUP_EXECUTION", "remaining_blockers": [],
                "study_id": ctl.STUDY,
                **{k: result[k] for k in ctl.BINDING_KEYS},
                "permissions": "Reviewer must explicitly select operations; this schema grants nothing."}
            if args.output:
                target = ctl.checked_output(args.output)
                target.parent.mkdir(parents=True, exist_ok=True)
                with target.open("x") as f:
                    json.dump(result, f, indent=2, sort_keys=True)
                    f.write("\n")
        elif args.command == "verify":
            result = ctl.verify(args.directory)
        else:
            from . import runner
            split = {"val": "validation", "test": "heldout"}.get(getattr(args, "split", None), getattr(args, "split", None))
            if args.command == "generate":
                result = runner.generate_dataset(split, args.review, args.output, args.seed, args.count, args.release)
            elif args.command == "pilot":
                result = runner.train(args.review, args.output, "full", 1000, pilot=True, device=args.device)
            elif args.command == "train":
                result = runner.train(args.review, args.output, args.variant, args.seed, pilot_evidence=args.pilot_evidence, device=args.device)
            elif args.command == "evaluate":
                result = runner.evaluate(args.review, args.dataset, args.checkpoints, args.output, split, args.release, args.device)
            elif args.command == "seal-validation":
                result = runner.seal_validation(args.review, args.evaluation, args.checkpoints, args.output)
            else:
                result = runner.aggregate(args.review, args.evaluation, args.output)
        print(json.dumps(result, indent=2, sort_keys=True))
    except ctl.GateError as error:
        p.error(str(error))


if __name__ == "__main__":
    main()
