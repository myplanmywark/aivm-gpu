#!/bin/bash
# ============================================================================
# AI VIDEO MAKER — GCP GPU VM setup (Ubuntu/Deep Learning VM par paste karo)
# ============================================================================
# Ye script:
#   1. repo clone karega (aivm-gpu)
#   2. dependencies install karega
#   3. env file banayega (R2 + Redis + worker key)
#   4. systemd service banayega (reboot par bhi auto-start)
# ============================================================================
set -e

REPO="https://github.com/myplanmywark/aivm-gpu"

echo "[1/5] git install..."
apt-get update -qq && apt-get install -y -qq git python3-venv > /dev/null

echo "[2/5] repo clone..."
git clone --depth 1 "$REPO" /opt/aivm 2>/dev/null || (cd /opt/aivm && git pull)
cd /opt/aivm

echo "[3/5] python venv + dependencies..."
python3 -m venv /opt/aivm/venv
/opt/aivm/venv/bin/pip install -q --upgrade pip
/opt/aivm/venv/bin/pip install -q -r requirements.txt

echo "[4/5] env file (APNI VALUES YAHAN EDIT KARNA MAT BHoolNA)..."
cat > /opt/aivm/aivm.env <<'ENVEOF'
R2_ACCOUNT_ID=5651cb24380e29f9d075b213f1cadc65
R2_ACCESS_KEY_ID=d2be56a39ec8a34c5a186a39a9cddddb
R2_SECRET_ACCESS_KEY=a3db2cac93d0a3dd814fcc60f7a78bdab1cd856c045f8f8e674bb5e1755a78cc
R2_BUCKET=ai-video-maker
REDIS_URL=rediss://default:gQAAAAAAAUoaAAIgcDFkM2RiMzZkYTVjOTU0YTFlYWM3NDI2OTY1ZjVkZDgyMA@measured-bat-84506.upstash.io:6379
HF_HOME=/workspace/hf
WORKER_KEY=aivm-secret-2026
ENVEOF

echo "[5/5] systemd service..."
cat > /etc/systemd/system/aivm.service <<'SVCEOF'
[Unit]
Description=AI Video Maker GPU Worker
After=network-online.target

[Service]
User=root
WorkingDirectory=/opt/aivm
EnvironmentFile=/opt/aivm/aivm.env
ExecStart=/opt/aivm/venv/bin/uvicorn model_server:app --host 0.0.0.0 --port 8000
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
SVCEOF

systemctl daemon-reload
systemctl enable aivm
systemctl start aivm

echo ""
echo "=============================================="
echo "DONE! Service chal raha hai (model load ho raha hai)"
echo "Live log dekhne ke liye:  journalctl -u aivm -f"
echo "Model 'ready' hone par VM ka EXTERNAL IP bhejo!"
echo "=============================================="
