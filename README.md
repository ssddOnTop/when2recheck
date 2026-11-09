# COMP 597: Learning When to Recheck

PPO navigation: traverse a route, inspect a passage, or take a safe detour. Currently graph-based, with separate environment, compute, and visualization adapters. Unreal Engine integration is not implemented yet.

## Setup

Python 3.11+:

```bash
git clone git@github.com:ssddOnTop/when2recheck.git
cd when2recheck
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -e '.[test]'
```

## Run

```bash
RUN="runs/quickstart-$(date +%Y%m%d-%H%M%S)"
python3 -m stale_nav.cli learn --backend graph --cpu \
  --steps 4096 --output "$RUN"
```

- **GPU:** replace `--cpu` with `--gpu` (NVIDIA CUDA required).
- **Visualization:** add `--visualize`. Open the printed URL during training or `$RUN/visualization.html` afterwards.
- **Feature ablation:** add `--variant masked`.

Checkpoints, metrics, and logs go into `$RUN`. Use a new output directory for each run; `runs/` is ignored by Git.

## Tests and help

```bash
python3 -m pytest -q
python3 -m stale_nav.cli --help
python3 -m stale_nav.cli learn --help
```

**macOS import errors:** prefix Python commands with `env -u DYLD_LIBRARY_PATH` to avoid mixing Homebrew and virtual-environment PyTorch libraries.

`learn` is for development. Final-study commands require current, hash-matched approvals; code or README edits invalidate old approvals.
