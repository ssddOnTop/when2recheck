#!/usr/bin/env bash
# Run only inside the parent-owned sandbox; keep follow-up caches/results local.
set -euo pipefail
payload='set -euo pipefail
base=/home/ssdd/academics/comp597/local-materials/followup/implementation
export PYTHONPATH=/home/ssdd/academics/comp597/research/src:/home/ssdd/academics/comp597/research
export PYTHONPYCACHEPREFIX="$base/cache/pycache"
export PYTHONUSERBASE="$base/local"
export XDG_CACHE_HOME="$base/cache"
export XDG_CONFIG_HOME="$base/config"
export XDG_DATA_HOME="$base/local/share"
export XDG_STATE_HOME="$base/local/state"
export XDG_RUNTIME_DIR="$base/runtime"
export TMPDIR="$base/tmp"
export PIP_CACHE_DIR="$base/cache/pip"
export UV_CACHE_DIR="$base/cache/uv"
export CUDA_CACHE_PATH="$base/cache/cuda"
export TORCH_HOME="$base/cache/torch"
export TORCHINDUCTOR_CACHE_DIR="$base/cache/torchinductor"
export TORCH_EXTENSIONS_DIR="$base/cache/torch-extensions"
export TRITON_CACHE_DIR="$base/cache/triton/cache"
export TRITON_DUMP_DIR="$base/cache/triton/dump"
export TRITON_OVERRIDE_DIR="$base/cache/triton/override"
export NUMBA_CACHE_DIR="$base/cache/numba"
export MPLCONFIGDIR="$base/config/matplotlib"
export HF_HOME="$base/cache/huggingface"
mkdir -p "$XDG_CACHE_HOME" "$XDG_CONFIG_HOME" "$XDG_DATA_HOME" "$XDG_STATE_HOME" "$XDG_RUNTIME_DIR" "$TMPDIR" "$MPLCONFIGDIR"
chmod 700 "$XDG_RUNTIME_DIR"
exec /home/ssdd/academics/comp597/.venv/bin/python "$@"'
printf -v quoted_payload '%q' "$payload"
printf -v arguments ' %q' "$@"
# scoped_run.sh sources project_env.sh before confinement. The inner command
# then overrides cache paths without changing either parent-owned script.
remote="bash /home/ssdd/academics/comp597/research/scripts/scoped_run.sh bash -c $quoted_payload --$arguments"
printf -v quoted_remote '%q' "$remote"
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ForwardAgent=no \
  -o UpdateHostKeys=no -o ServerAliveInterval=10 archssh.ssdd.dev \
  "bash -c $quoted_remote"
