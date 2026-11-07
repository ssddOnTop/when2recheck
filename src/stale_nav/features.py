"""Fixed analytic normalization and the identical actor/critic feature mask."""
from __future__ import annotations
import numpy as np
from .environment import Snapshot

# Scales are parameter bounds, never estimated from any data split.
FEATURE_SCALES: dict[str, float] = {
    **{f"position_{p}": 1.0 for p in ("S", "U0", "U1", "U2", "V0", "V1", "V2", "G")},
    "remaining_budget": 48.0, "task_index": 5.0, "flip_probability": 1.0,
    "miss_probability": 1.0,
    **{f"belief_{i}": 1.0 for i in range(3)},
    **{f"last_state_{i}": 1.0 for i in range(3)},
    **{f"known_{i}": 1.0 for i in range(3)},
    **{f"boundary_age_{i}": 5.0 for i in range(3)},
    **{f"observed_this_task_{i}": 1.0 for i in range(3)},
    **{f"inspection_attempted_{i}": 1.0 for i in range(3)},
    **{f"approach_cost_{i}": 14.0 for i in range(3)},
    **{f"crossing_cost_{i}": 4.0 for i in range(3)},
    **{f"viewpoint_cost_{i}": 2.5 for i in range(3)},
    "sensing_cost": 2.0, "safe_cost": 30.0, "initial_task_budget": 48.0,
}
FEATURE_NAMES = tuple(FEATURE_SCALES)
MASKED_COLUMNS = tuple(n for n in FEATURE_NAMES if
                       n.startswith(("boundary_age_", "approach_cost_", "crossing_cost_", "viewpoint_cost_"))
                       or n in {"sensing_cost", "safe_cost", "initial_task_budget"})
MASK_INDICES = tuple(FEATURE_NAMES.index(n) for n in MASKED_COLUMNS)
SCALES = np.array(list(FEATURE_SCALES.values()), dtype=np.float32)


def encode(snapshot: Snapshot, variant: str = "full") -> np.ndarray:
    if variant not in {"full", "masked"}:
        raise ValueError("variant must be full or masked")
    s = snapshot
    values = [float(s.position == p) for p in range(8)]
    values += [s.remaining, s.task, s.flip, s.miss]
    values += list(s.belief)
    values += [0 if x == -1 else 2 * x - 1 for x in s.last]
    values += [float(x != -1) for x in s.last]
    values += [max(0, a) for a in s.age]
    values += list(s.observed) + list(s.attempted)
    values += list(s.layout.a) + list(s.layout.b) + list(s.layout.v)
    values += [s.layout.sensing, s.layout.safe, s.layout.budget]
    vector = np.asarray(values, dtype=np.float32) / SCALES
    if variant == "masked":
        vector[list(MASK_INDICES)] = 0
    return vector


def feature_spec() -> dict:
    return {"names": FEATURE_NAMES, "analytic_scales": FEATURE_SCALES,
            "masked_columns_actor_and_critic": MASKED_COLUMNS,
            "meaning": "Explicit-feature ablation. Beliefs and remaining budget retain indirect age/cost information."}
