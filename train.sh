#!/usr/bin/env bash
# Task-owned DLC launcher only. This script never starts local GPU training.
set -euo pipefail
TASK_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="/root/miniconda3/envs/lerobot_v0.5.1/bin/python"
export PYTHONDONTWRITEBYTECODE=1

case "${1:-}" in
  "")
    if [[ "$#" -ne 0 ]]; then
      echo "Usage: bash train.sh [--status]" >&2
      exit 2
    fi
    exec "$PYTHON" "$TASK_ROOT/cloud/launch.py"
    ;;
  --status)
    if [[ "$#" -ne 1 ]]; then
      echo "Usage: bash train.sh [--status]" >&2
      exit 2
    fi
    exec "$PYTHON" "$TASK_ROOT/cloud/control.py" status
    ;;
  -h|--help)
    echo "Usage: bash train.sh [--status]"
    echo "No arguments: start/resume the task-owned DLC controller; an existing controller is reused."
    echo "--status: query owned DLC jobs. At most two concurrent jobs/two GPUs."
    ;;
  *)
    echo "Usage: bash train.sh [--status]" >&2
    exit 2
    ;;
esac
