# F1 Lab -- container image
#
#   docker build -t f1lab .
#   docker run --rm -p 8501:8501 -v f1lab-cache:/data/f1_cache f1lab
#
# The app is a Streamlit multi-page app (src/f1.py). Nothing here needs a
# compiler: pandas, numpy, plotly, matplotlib, streamlit, fastf1 and requests
# all publish wheels for python:3.12-slim, so the build stays small and fast.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app/src \
    F1LAB_CACHE_DIR=/data/f1_cache \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

WORKDIR /app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Only what the app reads at runtime.
COPY src/ ./src/
COPY .streamlit/ ./.streamlit/
COPY README.md ./README.md

# The cache directory is the only state worth keeping between runs: it holds the
# OpenF1 responses, so a restarted container does not re-download a whole season.
# Mount a volume here in production (see DEPLOYMENT.md).
RUN mkdir -p /data/f1_cache /app/logs && chown -R 10001:10001 /data /app

USER 10001:10001

EXPOSE 8501

# python:3.12-slim has no curl, so the probe uses the standard library.
HEALTHCHECK --interval=30s --timeout=5s --start-period=45s --retries=3 \
    CMD python -c "import sys,urllib.request; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8501/_stcore/health', timeout=4).status == 200 else 1)"

CMD ["streamlit", "run", "src/f1.py", "--server.port=8501", "--server.address=0.0.0.0"]
