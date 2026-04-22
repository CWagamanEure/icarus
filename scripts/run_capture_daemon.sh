#!/usr/bin/env bash
# Supervise the capture script: restart on crash with exponential backoff.
# Writes rotated daily DBs to data/capture/YYYY-MM-DD.sqlite3.
#
# Usage:
#   scripts/run_capture_daemon.sh                       # start (foreground)
#   nohup scripts/run_capture_daemon.sh >capture.log 2>&1 &   # detach
#
# Stop with: pkill -f capture_filter_eval.py  (or Ctrl+C if foreground)

set -uo pipefail

cd "$(dirname "$0")/.."

BACKOFF=5
MAX_BACKOFF=300

while true; do
  START_TS=$(date +%s)
  echo "[$(date -u +%FT%TZ)] starting capture"

  PYTHONPATH=src python scripts/capture_filter_eval.py \
    --asset BTC \
    --kraken-market BTC/USD \
    --basis-anchor-exchange coinbase \
    --basis-min-live-spot-venues 1 \
    --basis-common-price-process-var-per-sec 20 \
    --basis-process-var-per-sec 0.005 \
    --basis-rho-per-second 0.995 \
    --rotate-daily \
    --capture-dir data/capture \
    "$@"
  EXIT_CODE=$?

  NOW=$(date +%s)
  RUN_SECONDS=$(( NOW - START_TS ))
  echo "[$(date -u +%FT%TZ)] capture exited with code $EXIT_CODE after ${RUN_SECONDS}s"

  if [ "$EXIT_CODE" -eq 0 ]; then
    echo "clean exit — stopping daemon loop"
    exit 0
  fi

  # Reset backoff if the last run was "long" (>60s), otherwise escalate.
  if [ "$RUN_SECONDS" -gt 60 ]; then
    BACKOFF=5
  else
    BACKOFF=$(( BACKOFF * 2 ))
    if [ "$BACKOFF" -gt "$MAX_BACKOFF" ]; then
      BACKOFF=$MAX_BACKOFF
    fi
  fi

  echo "restarting in ${BACKOFF}s"
  sleep "$BACKOFF"
done
