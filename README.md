# COMP 597: Learning When to Recheck

PPO navigation with an interchangeable environment backend. The current graph simulator compares route traversal, paid inspection, and safe detours. Unreal Engine integration is not implemented yet.

## Setup

Requires Python 3.11+. Run from the repository root:

```bash
git clone git@github.com:ssddOnTop/comp597_research.git
cd comp597_research
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
python -m pytest -q
```

## Quick run

```bash
RUN="runs/quickstart-$(date +%Y%m%d-%H%M%S)"
python -m stale_nav.cli learn --backend graph --device cpu \
  --steps 4096 --output "$RUN"
```

Use `--device cuda` with CUDA-enabled PyTorch, or `--variant masked` for the graph feature ablation. Each run needs a new output directory. `runs/` is inside the repository and ignored by Git.

The development run saves a checkpoint, `events.jsonl`, optimizer metrics, provenance, and a summary. It does **not** reproduce or replace the reviewed final-study experiment.

### Optional visualization

Add `--visualize` to the command above. Open the printed localhost URL while training runs. Afterwards, open `$RUN/visualization.html` for an offline replay.

The viewer shows public graph state, actions, rewards, episode returns, and PPO losses. It has playback controls and a bounded history for one environment. Hidden door states are never displayed. Visualization is disabled by default and requires no extra dependencies. The server closes when training ends. `--visualization-port` can select a port; the default chooses a free one.

## Architecture and new backends

```text
CLI / application       selects and connects adapters
        |
core/                   PPO model, learning loop, GAE, and abstract ports
        |
adapters/               graph environment, storage, clock, visualization
```

The core has no file, network, console, or concrete clock I/O. Environments, observers, and clocks are injected. Storage implements the output observer and application-owned `RunStore` contract. Graph features and scenario generation stay outside the core.

A backend factory supplies observation/action dimensions and environments with `reset()`, `step()`, and `close()`. `step()` returns a `StepResult` with the final observation, reward, termination, and truncation flags. Optional `scene()` supplies public visualization data. Backends must not auto-reset.

Try the non-graph example without changing the learner:

```bash
python -m stale_nav.cli learn --backend examples.counter_backend:factory \
  --backend-config '{"time_limit":5}' --steps 256 \
  --output "runs/counter-$(date +%Y%m%d-%H%M%S)" --visualize
```

For Unreal, add an adapter implementing these same ports and load it with `--backend your_module:factory`. Put engine transport and observation/action translation there, not in the core. Installed plugins can also use the `stale_nav.backends` entry-point group. Load only trusted plugins.

The current learner supports flat numeric observations and discrete actions. Different sensors or continuous controls must be translated to that contract; arbitrary engine APIs are not automatically compatible.

## Reviewed experiments

The original `smoke`, `train`, `evaluate`, `select-ttl`, and `aggregate` commands remain available. `smoke` and `train` also accept `--visualize`. Final-study training still requires a hash-matched independent review; held-out evaluation requires matched checkpoints and validation-only TTL selection. Development checkpoints are separate.

`report/` contains the historical report. `followup/` remains a separate stopped study. Code or README edits invalidate old approvals for new training, not archived historical evidence.

Use `python -m stale_nav.cli --help` or a subcommand's `--help` for options.
