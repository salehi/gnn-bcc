FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONHASHSEED=0 \
    MPLCONFIGDIR=/tmp/mpl \
    HOME=/tmp \
    OMP_NUM_THREADS=8 \
    MKL_NUM_THREADS=8

WORKDIR /app

# CPU-only torch wheel: ~200MB instead of the ~2.5GB CUDA build.
# 927 nodes train full-batch in seconds on CPU, so a GPU buys nothing here.
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.5.1

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ /app/src/

# /tmp must be writable for matplotlib's font cache when running as an
# arbitrary host UID that has no /etc/passwd entry inside the container.
RUN mkdir -p /tmp/mpl && chmod 1777 /tmp /tmp/mpl

CMD ["python", "-m", "src.train"]
