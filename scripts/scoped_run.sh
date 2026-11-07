#!/usr/bin/env bash
# Run offline project commands with ordinary filesystem writes limited to the
# COMP 597 directory and /tmp. GPU and standard character devices stay usable.
set -euo pipefail
script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source "$script_dir/project_env.sh"
if [[ $# -eq 0 ]]; then
  printf 'Usage: bash scripts/scoped_run.sh COMMAND [ARG ...]\n' >&2
  exit 2
fi
if [[ "$(uname -s)" != Linux ]] || ! command -v bwrap >/dev/null; then
  printf 'This runner requires Linux and bubblewrap; it will not run unconfined.\n' >&2
  exit 1
fi
args=(--die-with-parent --new-session --unshare-net
  --ro-bind / / --bind "$COMP597_ROOT" "$COMP597_ROOT" --bind /tmp /tmp
  --dev-bind /dev /dev --remount-ro /dev
  --ro-bind /dev/shm /dev/shm --ro-bind /dev/pts /dev/pts
  --ro-bind /dev/mqueue /dev/mqueue --ro-bind /proc /proc)
# /dev may contain child mounts that its own remount does not cover.
if [[ -d /dev/hugepages ]]; then
  args+=(--ro-bind /dev/hugepages /dev/hugepages)
fi
shopt -s nullglob
for device in /dev/null /dev/zero /dev/random /dev/urandom /dev/nvidia* /dev/nvidia-caps/*; do
  if [[ -c "$device" ]]; then
    args+=(--dev-bind "$device" "$device")
  fi
done
exec bwrap "${args[@]}" --chdir "$COMP597_ROOT/research" "$@"
