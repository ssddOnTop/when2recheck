#!/usr/bin/env bash
# Synchronize implementation files only. Never delete or sync parent-owned work.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
remote='archssh.ssdd.dev:/home/ssdd/academics/comp597/research/'
transport='ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ForwardAgent=no -o UpdateHostKeys=no -o ServerAliveInterval=10'
rsync -az -e "$transport" --exclude '__pycache__' --exclude '*.pyc' \
  "$root/src" "$root/tests" "$root/configs" "$root/pyproject.toml" "$root/README.md" "$remote"
rsync -az -e "$transport" \
  "$root/scripts/sync_owned.sh" "$root/scripts/remote_python.sh" "$root/scripts/run_final.py" \
  "${remote}scripts/"
