#!/usr/bin/env bash
# Wrapper for scheduling the recursive self-improvement cycle (cron/launchd).
# Reads ANTHROPIC_API_KEY from the environment or a .env file next to this
# script's repo root — set one of those up before scheduling this.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

if [ -f .env ]; then
  set -a
  source .env
  set +a
fi

source venv/bin/activate

mkdir -p rsi_data/logs
LOG_FILE="rsi_data/logs/$(date +%Y%m%d_%H%M%S).log"

python cli.py --self-improve --cycles "${RSI_CYCLES:-1}" >> "$LOG_FILE" 2>&1
echo "self-improve cycle finished, see $LOG_FILE"
