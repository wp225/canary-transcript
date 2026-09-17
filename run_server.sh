#!/usr/bin/env bash
# Live demo server (segmentation + transcription + upload), for recording the demo
# video and for local use. No sudo, no systemd: it dies on reboot, restart by hand.
#
#   ./run_server.sh            # localhost only; reach it via VS Code port forwarding
#   HOST=0.0.0.0 ./run_server.sh   # also visible on the lab network (10.33.48.77)
#
# Logs to server.log, pid in server.pid.
set -euo pipefail
cd "$(dirname "$0")"
HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8000}"
if [ -f server.pid ] && kill -0 "$(cat server.pid)" 2>/dev/null; then
  echo "already running (pid $(cat server.pid)) on port $PORT"; exit 0
fi
nohup ./.venv/bin/python -m uvicorn app:app --host "$HOST" --port "$PORT" \
  > server.log 2>&1 &
echo $! > server.pid
echo "started pid $(cat server.pid) on $HOST:$PORT (log: server.log)"
