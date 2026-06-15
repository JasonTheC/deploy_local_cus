FROM nvcr.io/nvidia/pytorch:24.01-py3

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ cmake libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
# Downgrade numpy to 1.x for cv2 compatibility (base image ships numpy 2.x + prebuilt cv2)
RUN pip install --no-cache-dir "numpy>=1.21.0,<2"

RUN pip install --no-cache-dir \
    pydicom pynetdicom nibabel Pillow natsort pycryptodome \
    segmentation-models-pytorch \
    cupy-cuda12x \
    "websockets>=12.0"

COPY mobilenet_models/ ./mobilenet_models/
COPY pos_cupy_finalv2.py .
COPY check_gpu.py .
COPY pacs-transfer-server.py .

RUN mkdir -p /app/US_images /app/outputs

EXPOSE 8890 7556

ENV MAX_CONCURRENT_JOBS=2
ENV MAX_QUEUE_SIZE=16
ENV PACS_WS_PORT=7556

CMD ["python", "pacs-transfer-server.py"]
