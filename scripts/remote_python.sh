#!/usr/bin/env bash
# Offline Python through the parent-owned write-confinement runner.
set -euo pipefail
printf -v arguments ' %q' "$@"
command="set -euo pipefail; source /home/ssdd/academics/comp597/research/scripts/project_env.sh; export PYTHONPATH=/home/ssdd/academics/comp597/research/src; exec bash /home/ssdd/academics/comp597/research/scripts/scoped_run.sh /home/ssdd/academics/comp597/.venv/bin/python${arguments}"
printf -v quoted_command '%q' "$command"
ssh -o BatchMode=yes -o StrictHostKeyChecking=yes -o ForwardAgent=no \
  -o UpdateHostKeys=no -o ServerAliveInterval=10 archssh.ssdd.dev \
  "bash -c $quoted_command"
