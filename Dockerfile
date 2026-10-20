FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY model_server.py worker.py start.sh ./
RUN chmod +x start.sh && mkdir -p /workspace/hf /var/log/portal

ENV HF_HOME=/workspace/hf

CMD ["bash", "start.sh"]
