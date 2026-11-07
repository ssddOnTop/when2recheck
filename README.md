# stale-nav: graph-only navigation experiment

This package implements the approved revision-2 graph experiment. It does not run Unreal Engine or measure Unreal time. The frozen submission protocol takes precedence over the conceptual baseline plan.

## Current state and safety gates

The implementation includes a physical simulator, four classical policies, full and masked PPO, acceptance tests, and raw-data aggregation. Final training has not been authorized. The `train` command and final-run script require an independent implementation review tied to the exact source and configuration hashes. A design approval alone does not open this gate.

The first permitted training run is a separate 4096-step smoke run after tests pass. Smoke outputs are pilots, not final evidence. Test evaluation refuses pilot checkpoints, incomplete budgets, duplicate models, and missing matched training seeds. This repository does not contain final experimental findings. Read each run's `summary.json`, `provenance.json`, and `artifacts.json` for its actual completion state.

The remote acceptance suite passed, including deterministic CUDA forward/backward checks. The initial 4096-step full-policy CUDA pilot, paired pilot evaluation, and raw-data aggregation also completed. On an RTX 5090, that first pilot took 14.5921 seconds, or 280.699 environment steps/second. Its conservative projection was 116.737 minutes for all ten original-budget runs. The first rollout consumed 14.4061 seconds and the second 0.1859 seconds, so CUDA warmup dominates that estimate. These timings are feasibility evidence, not performance rankings.

The prescribed pre-test fallback is now recorded in `configs/final.json`: 98304 steps for both variants, with all five seeds retained. It projects to 58.369 minutes under the original conservative measurement. The CLI defaults to this configuration. `configs/frozen.json` preserves the original 196608-step target. The original pilot and immutable source archive remain intact. No held-out test result informed this decision, and no full training has started.

No command below edits the report or planning protocol. No command submits work, calls Canvas, publishes to GitHub, or creates a Git commit.

## Installation and remote execution

Python 3.11 or newer is required. Dependencies are declared in `pyproject.toml`. The prepared remote environment already has them. An editable install is not required: the wrapper sets `PYTHONPATH` to the local package source on the remote host.

Run these commands from the repository root on the local machine:

```bash
bash scripts/sync_owned.sh
bash scripts/remote_python.sh -m pytest -q
bash scripts/remote_python.sh -m stale_nav.cli manifest
```

`sync_owned.sh` copies only `src/`, `tests/`, `configs/`, the project metadata, this README, and three named experiment scripts. It does not use deletion flags. The parent owns `project_env.sh`, `scoped_run.sh`, and `build_report.sh`; the sync script never sends them.

The Python wrapper uses SSH with BatchMode, strict host-key checking, no agent forwarding, no host-key updates, and a 10-second keepalive. It sources `project_env.sh`, then invokes the parent's offline `scoped_run.sh`. Remote writes stay inside `/home/ssdd/academics/comp597` or `/tmp`. No global install or shell-profile change is needed.

The default configuration uses CUDA. Set `device` to `cpu` in a separate configuration if CUDA is unavailable. All other settings remain subject to frozen-configuration validation. The implementation sets `CUBLAS_WORKSPACE_CONFIG=:4096:8` before CUDA initialization and enables deterministic Torch algorithms. It never disables determinism to hide a failure. Exact reproducibility is expected within the same dependency, hardware, and device environment, not promised across platforms.

## Test, then run the isolated pilot

Use a new output directory for every run. Commands refuse to overwrite existing directories, including incomplete runs.

```bash
bash scripts/remote_python.sh -m pytest -q
bash scripts/remote_python.sh -m stale_nav.cli smoke \
  --variant full --seed 0 \
  --output /home/ssdd/academics/comp597/experiment-results/pilot/full-seed-0
bash scripts/remote_python.sh -m stale_nav.cli evaluate \
  --split pilot --pilot-sequences 4 \
  --checkpoints /home/ssdd/academics/comp597/experiment-results/pilot/full-seed-0/checkpoint.pt \
  --output /home/ssdd/academics/comp597/experiment-results/pilot/evaluation
bash scripts/remote_python.sh -m stale_nav.cli aggregate \
  --evaluation /home/ssdd/academics/comp597/experiment-results/pilot/evaluation \
  --training /home/ssdd/academics/comp597/experiment-results/pilot/full-seed-0 \
  --output /home/ssdd/academics/comp597/experiment-results/pilot/aggregate
```

Pilot evaluation uses a separate `pilot` scenario namespace. Its default TTL threshold is an explicitly unselected value of 1. This value is not a validation result. A one-seed pilot cannot establish PPO seed variation or a full-versus-masked conclusion.

The smoke summary records environment steps per wall-clock second and estimates the time for all ten final training runs. Pilot evaluation estimates the 128-sequence, 14-policy-instance evaluation cost separately. These are estimates, not completion guarantees. Process startup and initial model construction are outside the trainer's throughput timer. Lazy CUDA kernel initialization inside the first rollout remains included. Do not mistake a warmed repeat for an additional independent training seed.

## Full training requires independent review

Do not execute this section until the parent independently accepts the implementation. The reviewer provides a JSON file with these fields:

```json
{
  "verdict": "PASS_FOR_TRAINING",
  "source_sha256": "exact hash printed by the manifest command",
  "config_sha256": "exact hash printed by the manifest command",
  "remaining_blockers": []
}
```

The original final budget is 196608 environment steps per run. The measured-throughput fallback in the default `configs/final.json` uses 98304 steps for both variants. Each variant still uses seeds 0, 1, 2, 3, and 4. Both use 16 environments, 128 decisions per rollout, four optimizer epochs, minibatches of 256, and learning rate 0.0003. PPO uses gamma 1, decision-step GAE lambda 0.95, and reward scale 0.01. Separate actor and critic MLPs each have two 64-unit tanh layers. The implementation is original and does not copy upstream PPO source.

If projected training exceeds 90 minutes, first reduce both variants to 98304 steps before inspecting any final-test result. If that remains excessive, use seeds 0, 1, and 2 for both variants. A fallback configuration must record `fallback_reason`, including measured throughput. It needs a new independent review hash. Do not alter budgets after seeing test rankings. If the last fallback remains infeasible, report incomplete work or pilot evidence without claiming final results.

After approval, the following script runs every matched final seed. It never evaluates the test split:

```bash
bash scripts/remote_python.sh scripts/run_final.py \
  --review /home/ssdd/academics/comp597/local-materials/audits/implementation-review.json \
  --output /home/ssdd/academics/comp597/experiment-results/final
```

Individual authorized runs are also available through `python -m stale_nav.cli train --variant full --seed 0 --review ... --output ...`. Checkpoints are saved only at the final completed budget, not selected by validation or test score. The checkpoint contains model and optimizer state, the variant, seed, feature specification, source/config hashes, and the completed step count. Mid-run resume is not implemented. Preserve incomplete output directories; do not silently replace failed seeds.

## Validation-only TTL selection and paired final evaluation

Select one route-conditioned TTL threshold per regime using exactly 64 validation sequences. The candidates are 1, 2, 3, and 6. Minimize mean sequence penalized cost; use the smaller threshold on a tie. This command has no test-split input:

```bash
bash scripts/remote_python.sh -m stale_nav.cli select-ttl \
  --output /home/ssdd/academics/comp597/experiment-results/validation/ttl
```

Freeze this selection before final evaluation. Final test evaluation uses exactly 128 held-out sequences per regime and requires all ten matched final checkpoints, or all six checkpoints under an approved three-seed fallback. Use explicit checkpoint paths when calling the local wrapper; local wildcards do not expand on the remote machine.

The following block is intended for the remote shell. It still sources the project environment and uses the sandbox runner:

```bash
source /home/ssdd/academics/comp597/research/scripts/project_env.sh
export PYTHONPATH=/home/ssdd/academics/comp597/research/src
ROOT=/home/ssdd/academics/comp597
RUNNER="$ROOT/research/scripts/scoped_run.sh"
PYTHON="$ROOT/.venv/bin/python"
bash "$RUNNER" "$PYTHON" -m stale_nav.cli evaluate \
  --split test \
  --ttl-selection "$ROOT/experiment-results/validation/ttl/ttl_selection.json" \
  --checkpoints "$ROOT"/experiment-results/final/*/checkpoint.pt \
  --output "$ROOT/experiment-results/test"
bash "$RUNNER" "$PYTHON" -m stale_nav.cli aggregate \
  --evaluation "$ROOT/experiment-results/test" \
  --training "$ROOT"/experiment-results/final/* \
  --output "$ROOT/experiment-results/aggregate"
```

PPO evaluation chooses the highest-logit action, with lowest-index ties. Classical and learned policies use identical scenario IDs within each evaluation. Training, validation, test, and pilot IDs have disjoint namespaces. Training regimes cycle equally at sequence creation; a sequence's regime stays fixed. Different policy durations can produce different completed-sequence counts under the same environment-step budget. No hyperparameter is selected from test outcomes.

## Physical and information contract

Each sequence contains six tasks on one cost layout and three hidden doors. A task starts at the hub. The safe route always fits the initial task budget. Shortcuts have an approach segment and a hidden traversability event. Inspection travels to a viewpoint and then pays sensing time. Every action routes through the hub, including repeated actions at the same endpoint. Blocked shortcuts do not charge crossing cost.

The environment charges only consumed time. It reveals door evidence on completed approach arrival, and inspection evidence only after completed arrival and sensing. Exact-budget goal arrival succeeds. Exact-budget door arrival preserves evidence but fails when no crossing budget remains. Interrupted travel or sensing cannot reveal unpaid evidence. There is no incidental visibility.

Floating-point comparisons use one four-unit-in-the-last-place (ULP) equality convention for both cumulative cost and remaining time. The bound uses the largest of the original action budget, cumulative event time, and current segment duration. Thus a literal budget of 7.3 completes a route with costs 6.2 and 1.1, despite their rounded sum of 7.300000000000001. Differences within this bound are treated as equality; single-ULP shortfalls are not distinguishable from decimal rounding. Shortfalls outside the bound fail. Tolerated positive or negative residuals are clamped to zero, and consumed time never exceeds the available budget. Tests cover independent literal TAKE, sensing, and return-route budgets, inside-bound equality, and outside-bound failures.

Door states remain fixed within a task. At each intermediate success or failure, the environment preserves legitimate memory, advances the task, applies one pre-generated world transition, propagates beliefs once, increments known boundary ages, clears task flags, and resets position and budget. The final task terminates without another transition. A fresh episode clears memory.

World trajectories are pre-generated from a stream independent of layout draws. Sensing misses are keyed by scenario, task, door, and that door's attempt number. A policy's unrelated action count cannot shift hidden transitions or misses. Immutable public snapshots contain neither world states nor seed identifiers. Debug hashes in output files are provenance only and never enter policy inputs.

Raw reward is negative consumed time plus 100 for success or minus 100 for failure. A six-task sequence therefore has reward `600 - total_time - 200 * failures`. PPO scales that reward once by 0.01. Intermediate task boundaries do not terminate GAE. Sequence endings disable bootstrap; rollout cuts retain it. Lambda is applied per decision, not raised to elapsed time.

## Policies and feature ablation

- **Reactive:** choose the lowest-cost route not remembered blocked; unknown doors remain eligible. The safe route is always a candidate.
- **Reset:** use reactive navigation after forgetting prior-task state history. Retain current-task observations and known layout.
- **Route-conditioned TTL:** inspect a previously observed door only if reactive routing selected it, its boundary age reaches the threshold, and no inspection was attempted this task. A miss still consumes the one permitted attempt. Remembered-blocked doors are never selected, so this baseline does not proactively discover reopened doors.
- **Single-step physical VOI:** enumerate all eight door worlds using public beliefs. For each of seven first actions, include physical position, time budget, and sensing outcomes. Continue with reactive navigation that observes only physically encountered doors. Do not inspect within the simulated continuation. Replan after the first actual action. Minimize time plus 200 per task failure; prefer navigation on cost ties, then the lowest action index. A score sums at most 16 weighted branches, so values within 16 ULPs of the minimum form a numerical tie. Reactive route comparisons use a four-ULP bound for their three-term travel sums. Meaningful cost differences outside those bounds still determine the choice.
- **PPO full and masked:** use the same seven legal actions, architecture, optimizer, seeds, budgets, and known regime parameters. Both actor and critic receive the same encoded feature vector.

`configs/features.json` publishes all 42 columns and their analytic scales. The explicit-feature mask zeros boundary ages, approach costs, crossing costs, viewpoint costs, sensing cost, safe-route cost, and initial task budget. It retains the vector size, current position, remaining budget, task index, beliefs, observed states, known-state flags, current-task flags, and flip/miss probabilities. Beliefs retain indirect age information. Remaining budget retains indirect cost information. This is not an ablation of all temporal or cost information.

The hand case is checked in tests: expected TAKE cost 28, SAFE cost 25, and physical INSPECT followed by reactive cost 21. Expensive sensing reverses the inspection preference. Additional tests cover missing evidence, exact event boundaries, public-state isolation, paired worlds, deterministic checkpoint behavior, feature masks, GAE, split gates, and raw-table consistency.

## Output files and uncertainty

Each training run writes:

- `steps.csv`: every environment transition, including actual duration, raw/scaled reward, task boundary, sequence termination, scenario ID, and action.
- `updates.csv`: optimizer losses, entropy, approximate KL, clip fraction, wall time, and throughput after each rollout update.
- `episodes.csv` and `.json`: completed task records. An episode here is a task-level raw record; the policy episode is the six-task sequence.
- `sequences.csv` and `.json`: only complete six-task sequences. Partial trailing sequences are excluded and counted in the summary.
- `checkpoint.pt`, `config.json`, `summary.json`, `provenance.json`, and `artifacts.json`.

Evaluation writes paired task/sequence records and scenario layout/world hashes. Time summaries include all tasks, including failures. Penalized cost adds 200 per failure. Successful-task-only time is not used as the primary metric.

Aggregation verifies artifact hashes and recomputes sequence totals from task rows. It writes per-seed summaries, separate training-seed variation, paired sequence differences against reactive, learning-curve CSVs, JSON table data, and a learning-curve PNG when training logs are supplied. It does not write report prose. Learning curves use fixed 2048-step windows of completed raw sequence returns. Shading shows training-seed standard deviation, not 128-sequence uncertainty. A one-seed pilot has no variation band.

The evaluation unit is a six-task sequence, not one of its correlated tasks. Five training seeds are a separate axis from 128 held-out sequences per regime. Paired comparisons first average available models within each scenario, then compare with reactive on that same scenario. Their normal-approximation intervals are conditional on those trained models. Training-seed variation is reported separately; the seed-by-sequence cross-product is never counted as independent observations.

The source hash covers owned implementation, tests, configs, experiment scripts, metadata, and README. It excludes report files, output data, caches, and parent-owned scripts. Every run also records a deterministic, read-only source tar archive under `../local-materials/source-archives/`. The archive contains only listed source files and the frozen protocol. Its content-addressed name and SHA256 digest are recorded in `provenance.json` and `source_snapshot.json`; the artifact manifest covers those records. Verification checks the archive digest, so historical source remains verifiable after later README edits.

Provenance separately hashes available protocol inputs, records the exact command and dependency versions, and records detected GPU hardware. Artifact manifests cover the written raw data and checkpoint. Changing source after a review invalidates the full-training gate; changing source after final training invalidates final evaluation. Final evaluation must therefore finish against the reviewed source before later documentation changes. Historical aggregation remains possible through the recorded archives and manifests.

## Limitations

This is a finite parallel-corridor surrogate, not a building simulator or arbitrary-topology benchmark. Time units are abstract. The independent flip model is known, not fitted. Tasks reset to the origin without simulated return travel. Door changes occur only at task boundaries. Sensing misses are independent and keyed, not a model of realistic camera failure.

VOI computes an exact expectation for a restricted current-task reactive continuation, not an optimal POMDP policy. PPO optimizes a full six-task sequence. This horizon mismatch prevents attributing a performance difference only to better sensing decisions. No claim of Unreal deployment, real-time navigation performance, or general building robustness follows from these experiments.
