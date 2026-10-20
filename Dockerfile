FROM python:3.12-slim
ENV PYTHONUNBUFFERED=1 PYTHONUTF8=1 PIP_NO_CACHE_DIR=1 \
    APP_ENV=production BUDGET_DATA_DIR=/data BUDGET_MODEL_DIR=/models \
    BUDGET_CACHE_DIR=/data/cache HF_HOME=/data/cache/huggingface OCR_PYTHON=/opt/ocr/bin/python \
    OMP_NUM_THREADS=2 MKL_NUM_THREADS=2
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 libgl1 libglib2.0-0 fonts-noto-cjk && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements-backend.txt requirements-ocr.txt ./
COPY Receipt_Budget_Book/requirements-tested.txt /tmp/requirements-training.txt
RUN pip install torch==2.10.0 torchvision==0.25.0 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements-backend.txt -r /tmp/requirements-training.txt \
    && python -m venv /opt/ocr && /opt/ocr/bin/pip install -r requirements-ocr.txt
COPY backend ./backend
COPY frontend ./frontend
COPY tools ./tools
COPY Receipt_Budget_Book/*.py ./Receipt_Budget_Book/
RUN useradd --uid 10001 --create-home app && mkdir -p /data/cache /models && chown -R app:app /data /app
USER app
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/health',timeout=3)"
CMD ["python","-m","uvicorn","backend.app:app","--host","0.0.0.0","--port","8000","--workers","1","--proxy-headers","--forwarded-allow-ips","*"]
