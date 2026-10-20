#!/bin/bash
# Container entrypoint: start model server, wait for ready, run PyWorker agent.
set -e
mkdir -p /var/log/portal

uvicorn model_server:app --host 127.0.0.1 --port 8000 \
  > /var/log/portal/model.log 2>&1 &

echo "[aivm] model server starting (LTX-2.3 download/load at first boot)..."

for i in $(seq 1 480); do
  if grep -q "Model ready" /var/log/portal/model.log 2>/dev/null; then
    echo "[aivm] model server is ready"
    break
  fi
  if grep -q "Traceback" /var/log/portal/model.log 2>/dev/null; then
    echo "[aivm] model server crashed:"
    tail -50 /var/log/portal/model.log
    exit 1
  fi
  sleep 5
done

exec python worker.py
