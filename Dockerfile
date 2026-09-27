FROM python:3.12-slim-bookworm AS model
WORKDIR /build
RUN pip install --no-cache-dir torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu
RUN pip install --no-cache-dir onnx==1.18.0 onnxruntime==1.22.1 numpy==2.2.6
COPY scripts/export_model.py ./export_model.py
RUN python export_model.py --output /build/models/realesr-general-x4v3.onnx

FROM python:3.12-slim-bookworm AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg ca-certificates \
    && rm -rf /var/lib/apt/lists/* && useradd --create-home --uid 1000 bot
WORKDIR /app
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY --from=model /build/models/realesr-general-x4v3.onnx /app/models/realesr-general-x4v3.onnx
COPY bot ./bot
COPY THIRD_PARTY.md ./
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=10000 WORK_DIR=/tmp/circle-bot
USER 1000:1000
EXPOSE 10000
CMD ["python", "-m", "bot.app"]
