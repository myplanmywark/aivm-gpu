#!/bin/bash
# Vast.ai Serverless start-server script.
# Installs deps, starts the model server, then runs the PyWorker agent.
set -e

echo "[aivm] installing requirements..."
pip install -r requirements.txt

echo "[aivm] starting model server (LTX-2.3 loads at startup)..."
mkdir -p /var/log/portal
uvicorn model_server:app --host 127.0.0.1 --port 8000 \
  > /var/log/portal/model.log 2>&1 &

echo "[aivm] waiting for model server to report ready..."
for i in $(seq 1 240); do
  if grep -q "Model ready" /var/log/portal/model.log 2>/dev/null; then
    echo "[aivm] model server is ready"
    break
  fi
  if grep -q "Traceback" /var/log/portal/model.log 2>/dev/null; then
    echo "[aivm] model server crashed:"
    tail -30 /var/log/portal/model.log
    exit 1
  fi
  sleep 5
done

echo "[aivm] starting PyWorker agent..."
python worker.py
